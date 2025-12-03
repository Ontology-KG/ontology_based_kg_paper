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
    """정규화된 q_embs와 ENTITY_EMB 간 코사인 유사도 top-k 인덱스(부분정렬) 반환"""
    sims = np.dot(q_embs, ENTITY_EMB.T)                        # (B, N)
    part = np.argpartition(sims, -k, axis=1)[:, -k:]           # (B, k)
    rows = np.arange(sims.shape[0])[:, None]
    sorted_idx = np.argsort(sims[rows, part], axis=1)[:, ::-1] # k개 내림차순
    return part[rows, sorted_idx], sims                        # (B,k), (B,N)


@lru_cache(maxsize=10000)
def extract_entities_by_llm(question: str) -> List[str]:
    template = f"""
    Extract domain-specific entities from the question.

    RULES:
    - Entities = terms that could appear as <subj> or <obj> in a knowledge triplet.
    - Keep text as-is (materials, processes, dimensional constraints, units).
    - Keep numeric values with units (e.g., "3 inches", "75 mm").
    - Keep designations (e.g., "W24×162") as entities.
    - Do not normalize away any values.
    - Separate with commas, end with <END>.

    Q: What is the standard specification for the examination of rolled steel plates?
    A: rolled steel plates, Straight-Beam Ultrasonic Examination of Rolled Steel Plates for Special Applications<END>

    Now extract entities:
    <CLS>{question}<SEP>
    """
    resp = chat.invoke([
        SystemMessage(content="You are a helpful assistant that extracts domain-specific entities."),
        HumanMessage(content=template)
    ]).content
    ents: List[str] = []
    m = re.search(r"(.*?)<END>", resp)
    if m:
        ents = [e.strip() for e in m.group(1).split(",") if e.strip()]
    return ents

@lru_cache(maxsize=10000)
def _match_one(ent: str, top_k=5, sim_threshold=0.70) -> List[str]:
    ent_l = ent.lower()
    subs = [ENTITY_LIST[i] for i, kg in enumerate(ENTITY_LIST_LOWER) if ent_l in kg]
    if subs:
        return subs[:top_k]
    q = _encode_normalized([ent])                      # (1,D)
    idxs, sims = _cosine_topk_batch(q, top_k)          # (1,k)
    out = []
    for i in idxs[0]:
        if sims[0, i] >= sim_threshold:
            out.append(ENTITY_LIST[i])
    return out



def match_entities_to_kg(entities: List[str], top_k=5, sim_threshold=0.70) -> List[str]:
    if not entities:
        return []
    direct, todo = [], []
    for ent in entities:
        ent_l = ent.lower()
        hits = [i for i, kg in enumerate(ENTITY_LIST_LOWER) if ent_l in kg]
        if hits:
            direct.extend([ENTITY_LIST[i] for i in hits[:top_k]])
        else:
            todo.append(ent)

    matched = direct[:]
    if todo:
        q_embs = _encode_normalized(todo, batch_size=64)
        idxs, sims = _cosine_topk_batch(q_embs, top_k)
        for r in range(q_embs.shape[0]):
            for i in idxs[r]:
                if sims[r, i] >= sim_threshold:
                    matched.append(ENTITY_LIST[i])
    for ent in entities:
        matched.extend(_match_one(ent, top_k, sim_threshold))

    seen, out = set(), []
    for m in matched:
        if m not in seen:
            seen.add(m); out.append(m)
    return out

def ensure_indexes():
    cyphers = [
        "CREATE INDEX entitylocal_label IF NOT EXISTS FOR (e:EntityLocal) ON (e.label)",
        "CREATE INDEX doc_id IF NOT EXISTS FOR (d:Doc) ON (d.id)"
    ]
    with driver.session() as sess:
        for c in cyphers:
            sess.run(c)

def deduplicate_context_titles(titles: List[str]) -> List[str]:
    seen, out = set(), []
    for t in titles:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for t in triples:
        if t["type"] == "table":
            key = ("table", t.get("condition"), t.get("case"), t.get("consequence"))
        else:
            key = ("text", t.get("subj"), t.get("edge"), t.get("obj"))
        if key not in seen:
            seen.add(key); out.append(t)
    return out

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
#--------------추가
from itertools import combinations

def safe_label(node):
    return node.get("label") if hasattr(node, "get") else str(node)

def find_shortest_path(start_entity: str, end_entity: str, doc_id: str, max_paths=5) -> List[str]:
    with driver.session() as session:
        cypher = """
        MATCH (d:Doc {id:$doc_id})
        MATCH (d)-[:HAS_CHILD*0..]->()-[:MENTION]->(start:EntityLocal {label:$start})
        MATCH (d)-[:HAS_CHILD*0..]->()-[:MENTION]->(end:EntityLocal {label:$end})
        MATCH p = allShortestPaths((start)-[r*..6]-(end))
        WHERE ALL(r IN relationships(p) WHERE NOT type(r) IN ['HAS_CHILD','MENTION'])
        RETURN p LIMIT $max_paths
        """
        result = session.run(cypher, doc_id=doc_id, start=start_entity, end=end_entity, max_paths=max_paths)
        paths = []
        for record in result:
            path = record["p"]
            triples = [f"{safe_label(rel.start_node)}->{rel.type}->{safe_label(rel.end_node)}"
                       for rel in path.relationships]
            if triples:
                paths.append(" | ".join(triples))
        return paths

def find_all_entity_paths(matched_entities: List[str], doc_id: str, max_paths=3) -> List[str]:
    all_paths = []
    pairs = list(combinations(matched_entities, 2))
    for (start, end) in pairs:
        paths = find_shortest_path(start, end, doc_id=doc_id, max_paths=max_paths)
        if paths:
            all_paths.extend(paths)
    return list(set(all_paths))

def explore_neighbors_fast(entities: List[str], doc_id: str,
                           max_hops: int = 2,
                           neighbor_limit: int = 400,
                           max_cases: int = 30) -> List[Dict]:
    if not entities:
        return []
    triples: List[Dict] = []
    cypher_table = """
    UNWIND $ents AS ent
    MATCH (d:Doc {id:$doc_id})-[:HAS_CHILD*0..]->()-[:MENTION]->(cond:EntityLocal)
    WHERE toLower(cond.label) CONTAINS toLower(ent)
    MATCH (cond)<-[:HAS_CONDITION|HAS_CONDITION_AND|HAS_CONDITION_OR]-(c)
          -[r:HAS_CONSEQUENCE|HAS_CONSEQUENCE_AND|HAS_CONSEQUENCE_OR]->(res:EntityLocal)
    RETURN DISTINCT cond.label AS condition, c.label AS case, res.label AS consequence
    LIMIT $max_cases
    """
    cypher_text = f"""
    UNWIND $ents AS ent
    MATCH (d:Doc {{id:$doc_id}})-[:HAS_CHILD*0..]->()-[:MENTION]->(e:EntityLocal {{label:ent}})
    MATCH p=(e)-[r*1..{int(max_hops)}]->(n:EntityLocal)
    WHERE ALL(x IN r WHERE type(x) <> 'HAS_CHILD' AND type(x) <> 'MENTION')
    UNWIND r AS rel
    WITH DISTINCT startNode(rel).label AS subj, toLower(type(rel)) AS rel, endNode(rel).label AS obj
    RETURN subj, rel, obj
    LIMIT $neighbor_limit
    """

    with driver.session() as sess:
        for r in sess.run(cypher_table, ents=entities, doc_id=doc_id, max_cases=max_cases):
            triples.append({"type": "table",
                            "condition": r["condition"],
                            "case": r["case"],
                            "consequence": r["consequence"]})
        for r in sess.run(cypher_text, ents=entities, doc_id=doc_id, neighbor_limit=neighbor_limit):
            triples.append({"type": "text",
                            "subj": r["subj"],
                            "edge": r["rel"],
                            "obj": r["obj"]})
    return deduplicate_triples(triples)
#-----------------------------------------------------------
def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True))

def explore_neighbors_semantic(question: str, entities: List[str], doc_id: str, max_hops=3, top_k=3):
    embed_q = embedder.encode([question], convert_to_numpy=True)[0]
    visited = set()
    best_triples = []

    def triple_to_sentence(subj, rel, obj):
        return f"{subj} {rel.replace('_',' ')} {obj}"

    frontier = entities[:]
    for hop in range(max_hops):
        next_frontier = []
        for ent in frontier:
            with driver.session() as session:
                cypher = """
                MATCH (d:Doc {id:$doc_id})-[:HAS_CHILD*0..]->()-[:MENTION]->(e:EntityLocal {label:$ent})
                MATCH (e)-[r]->(n:EntityLocal)
                RETURN e.label AS subj, type(r) AS rel, n.label AS obj
                """
                results = session.run(cypher, doc_id=doc_id, ent=ent)
                candidates = []
                for r in results:
                    subj, rel, obj = r["subj"], r["rel"].lower(), r["obj"]
                    if (subj, rel, obj) in visited:
                        continue
                    visited.add((subj, rel, obj))
                    sent = triple_to_sentence(subj, rel, obj)
                    sent_emb = embedder.encode([sent], convert_to_numpy=True)[0]
                    sim = cosine_similarity_manual(np.array([embed_q]), np.array([sent_emb]))[0][0]
                    candidates.append((sim, {"type": "text", "subj": subj, "edge": rel, "obj": obj}))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    selected = [c[1] for c in candidates[:top_k]]
                    best_triples.extend(selected)
                    next_frontier.extend([c["obj"] for c in selected])

        frontier = next_frontier

    return best_triples

SYSTEM_PROMPT = """You are an expert in API SPEC documents.
Answer questions based ONLY on the provided context.
Rules:
- Output ONLY the final answer.
- Do NOT add any explanations, reasoning, or polite phrases.
- Do NOT generate multiple-choice options.
- If the answer is not in the context → reply exactly "I don't know".
- If the answer is present → state it directly, using the exact wording from the context if possible.
- For Yes/No questions → answer only "Yes" or "No".
- For formula questions → output the formula exactly.
- For rule/definition questions → copy the requirement or definition verbatim.
- Answer must be a single short phrase or sentence, nothing else.
"""

@lru_cache(maxsize=10000)
def final_answer(question: str, path_evidence: str, neighbor_evidence: str) -> str:
    if not path_evidence and not neighbor_evidence:
        return "I don't know"
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Question: " + question),
        AIMessage(content="Evidence:\n\n" + (path_evidence or "") + "\n\n" + (neighbor_evidence or "")),
        HumanMessage(content="Now provide ONLY the final answer.")
    ]
    output = chat.invoke(messages).content.strip()
    return output.splitlines()[0] if output else ""

def process_one_item(qa: Dict, doc_id: str) -> Dict:
    qid        = qa.get("id")
    question   = qa.get("question", "")
    gold       = qa.get("answers") or qa.get("gold") or []
    qtype      = qa.get("type")
    unans      = qa.get("unanswerable", False)

    try:
        # 1) 엔티티 추출
        ents = extract_entities_by_llm(question)
        if not ents:
            return _pack_out(qid, question, gold, "", qtype, unans, [], [], "", [])

        # 2) 엔티티 매칭
        matched = match_entities_to_kg(ents, top_k=5, sim_threshold=0.70)
        if not matched:
            return _pack_out(qid, question, gold, "", qtype, unans, ents, [], "", [])

        # 3) 컨텍스트 타이틀
        context_titles = deduplicate_context_titles(get_entity_context_bulk(matched, doc_id))

        # 4) 이웃 탐색
        triples = explore_neighbors_fast(matched, doc_id, max_hops=3, neighbor_limit=400, max_cases=30)

        # 4-1) semantic triple 보강
        semantic_triples = explore_neighbors_semantic(question, matched, doc_id, max_hops=3, top_k=3)
        triples.extend(semantic_triples)

        # 4-2) 엔티티 쌍 최단경로 path evidence
        paths = find_all_entity_paths(matched, doc_id, max_paths=3)
        path_evidence = "\n".join(paths)

        # 5) Evidence 문자열
        neighbor_evidence = "\n".join(
            [f"{t.get('condition','')} -> {t.get('consequence','')} (case: {t.get('case','')})"
            if t["type"] == "table"
            else f"{t.get('subj','')} {t.get('edge','')} {t.get('obj','')}"
            for t in triples]
        )

        # 6) 최종 답변
        pred = final_answer(question, path_evidence, neighbor_evidence)

        return _pack_out(qid, question, gold, pred, qtype, unans,ents, matched, path_evidence, triples)

    except Exception as e:
        return _pack_out(
            qid, question, gold, "", qtype, unans,
            [], [], "", "", [], error=str(e)
        )

def _pack_out(qid, question, gold, pred, qtype, unans,
              ents, matched, path_evidence, triples, error: str = None) -> Dict:
    out = {
        "id": qid,
        "question": question,
        "gold": gold,
        "pred": pred,
        "type": qtype,
        "unanswerable": unans,
        "retrieval": {
            "extracted_entities": ents,
            "matched_entities": matched,
            "path_evidence": path_evidence,
            "neighbor_evidence": "\n".join(
                [f"{t.get('condition','')} -> {t.get('consequence','')} (case: {t.get('case','')})"
                 if t["type"] == "table"
                 else f"{t.get('subj','')} {t.get('edge','')} {t.get('obj','')}"
                 for t in triples]
            ),
            "neighbor_triples": triples
        }
    }
    if error:
        out["error"] = error
    return out

def run_qa_dataset(input_path: str, output_path: str, doc_id: str, limit: int = None, max_workers: int = 6):
    ensure_indexes()

    with open(input_path, "r") as f:
        qa_lines = f.readlines()
    if limit:
        qa_lines = qa_lines[:limit]

    qa_items = [json.loads(line) for line in qa_lines if line.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(output_path, "w") as fout:
        futures = [ex.submit(process_one_item, qa, doc_id) for qa in qa_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running QA Dataset (parallel)"):
            out = fut.result()
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    DOC_ID = "API_2W"  # 예: "A578A578M_07", "API_2W", "A6A6M_14" API_2W_remove_pruning API_2W_remove_sameword
    run_qa_dataset(
        input_path="QA/type/API_2W_boolean_v2_half.jsonl",
        output_path="output/api_boolean/API_2W_boolean_new.jsonl",
        doc_id=DOC_ID,
        limit=None,         
        max_workers=6      
    )
