# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not OPENAI_API_KEY or not NEO4J_URI or not NEO4J_PASSWORD:
    raise RuntimeError("Please set OPENAI_API_KEY / NEO4J_URI / NEO4J_PASSWORD in your .env file.")

# Initialize models and drivers
set_llm_cache(InMemoryCache())
chat = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_pool_size=50)

# Load pre-computed entity embeddings
with open("path/to/your_entity_embeddings.pkl", "rb") as f:
    entity_data = pickle.load(f)

ENTITY_LIST: List[str] = entity_data["entities"]
ENTITY_EMB = np.array(entity_data["embeddings"], dtype=np.float32)
ENTITY_EMB /= (np.linalg.norm(ENTITY_EMB, axis=1, keepdims=True) + 1e-12)
ENTITY_LIST_LOWER = [e.lower() for e in ENTITY_LIST]

# Sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _encode_normalized(texts: List[str], batch_size: int = 64) -> np.ndarray:
    embs = embedder.encode(texts, convert_to_numpy=True, batch_size=batch_size)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms


def _cosine_topk_batch(q_embs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = np.dot(q_embs, ENTITY_EMB.T)
    part = np.argpartition(sims, -k, axis=1)[:, -k:]
    rows = np.arange(sims.shape[0])[:, None]
    sorted_idx = np.argsort(sims[rows, part], axis=1)[:, ::-1]
    return part[rows, sorted_idx], sims


@lru_cache(maxsize=10000)
def extract_entities_by_llm(question: str) -> List[str]:
    template = f"""
    Extract domain-specific entities from the question.
    Rules:
    - Entities include materials, dimensions, processes, or numerical constraints.
    - Preserve the original text form and units.
    - Separate entities with commas, and end with <END>.
    Q: Example question
    A: entity1, entity2<END>
    <CLS>{question}<SEP>
    """
    resp = chat.invoke([
        SystemMessage(content="You are a helpful assistant that extracts domain-specific entities."),
        HumanMessage(content=template)
    ]).content
    match = re.search(r"(.*?)<END>", resp)
    return [e.strip() for e in match.group(1).split(",") if e.strip()] if match else []


@lru_cache(maxsize=10000)
def _match_one(ent: str, top_k=5, sim_threshold=0.70) -> List[str]:
    ent_l = ent.lower()
    subs = [ENTITY_LIST[i] for i, kg in enumerate(ENTITY_LIST_LOWER) if ent_l in kg]
    if subs:
        return subs[:top_k]
    q = _encode_normalized([ent])
    idxs, sims = _cosine_topk_batch(q, top_k)
    return [ENTITY_LIST[i] for i in idxs[0] if sims[0, i] >= sim_threshold]


def match_entities_to_kg(entities: List[str], top_k=5, sim_threshold=0.70) -> List[str]:
    if not entities:
        return []
    matched = []
    for ent in entities:
        matched.extend(_match_one(ent, top_k, sim_threshold))
    seen, out = set(), []
    for m in matched:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def ensure_indexes():
    cyphers = [
        "CREATE INDEX entitylocal_label IF NOT EXISTS FOR (e:EntityLocal) ON (e.label)",
        "CREATE INDEX doc_id IF NOT EXISTS FOR (d:Doc) ON (d.id)"
    ]
    with driver.session() as sess:
        for c in cyphers:
            sess.run(c)


def get_entity_context_bulk(entities: List[str], doc_id: str) -> List[str]:
    if not entities:
        return []
    cypher = """
    UNWIND $ents AS ent
    MATCH (d:Doc {id:$doc_id})-[:HAS_CHILD*0..]->(s)-[:MENTION]->(e:EntityLocal {label:ent})
    RETURN DISTINCT s.name AS title
    """
    with driver.session() as sess:
        return [r["title"] for r in sess.run(cypher, ents=entities, doc_id=doc_id) if r["title"]]


SYSTEM_PROMPT = """You are an expert in technical standard documents.
Answer only based on the given evidence.
Reply with a concise factual answer or 'I don't know' if not found.
"""


@lru_cache(maxsize=10000)
def final_answer(question: str, evidence: str) -> str:
    if not evidence:
        return "I don't know"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Question: " + question),
        AIMessage(content="Evidence:\n" + evidence),
        HumanMessage(content="Now provide ONLY the final answer.")
    ]
    output = chat.invoke(messages).content.strip()
    return output.splitlines()[0] if output else ""


def process_one_item(qa: Dict, doc_id: str) -> Dict:
    qid = qa.get("id")
    question = qa.get("question", "")
    gold = qa.get("answers") or []
    try:
        ents = extract_entities_by_llm(question)
        matched = match_entities_to_kg(ents)
        context_titles = get_entity_context_bulk(matched, doc_id)
        evidence = "\n".join(context_titles)
        pred = final_answer(question, evidence)
        return {"id": qid, "question": question, "gold": gold, "pred": pred}
    except Exception as e:
        return {"id": qid, "question": question, "gold": gold, "error": str(e)}


def run_qa_dataset(input_path: str, output_path: str, doc_id: str, limit: int = None, max_workers: int = 4):
    ensure_indexes()
    with open(input_path, "r") as f:
        qa_items = [json.loads(line) for line in f if line.strip()]
    if limit:
        qa_items = qa_items[:limit]
    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(output_path, "w") as fout:
        futures = [ex.submit(process_one_item, qa, doc_id) for qa in qa_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running QA"):
            fout.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            fout.flush()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # input_path:  QA dataset (.jsonl)
    # output_path: output file (.jsonl)
    # doc_id:      document ID in Neo4j
    input_path = "path/to/your_input.jsonl"     # Replace with your QA dataset path
    output_path = "output/predictions.jsonl"    # Replace with desired output file
    doc_id = "YOUR_DOC_ID"                      # Replace with your document ID in Neo4j
    run_qa_dataset(input_path, output_path, doc_id)
