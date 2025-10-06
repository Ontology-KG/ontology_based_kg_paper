# -*- coding: utf-8 -*-
import os, re, json, time, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from numpy.linalg import norm
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER = SentenceTransformer(EMBED_MODEL, device=DEVICE)

def ensure_unit_norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec if n < eps else (vec / n)

def embed_texts(texts: List[str]) -> np.ndarray:
    return EMBEDDER.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]

def embed_long_document_st(text: str, model: SentenceTransformer, max_length=512, stride=256) -> np.ndarray:
    v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

import pickle

def load_index(pkl_path: str) -> Dict[str, Any]:
    """
    pickle: List[{"title": str, "merged_text": str, "embedding": np.ndarray}]
    """
    with open(pkl_path, "rb") as f:
        records: List[Dict[str, Any]] = pickle.load(f)

    titles, texts, embs = [], [], []
    for r in records:
        titles.append(r["title"])
        texts.append(r.get("merged_text", ""))
        v = ensure_unit_norm(np.asarray(r["embedding"], dtype=np.float32))
        embs.append(v)

    matrix = np.stack(embs, axis=0).astype(np.float32)  # (N, D)
    return {"titles": titles, "texts": texts, "matrix": matrix}

def search_index(index: Dict[str, Any], query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float, int]]:
    q_vec = embed_long_document_st(query, EMBEDDER)
    scores = index["matrix"] @ q_vec   # (N,)
    valid_idx = np.where(scores >= threshold)[0]
    if len(valid_idx) == 0:
        return []
    order = np.argsort(scores[valid_idx])[::-1]
    sel = valid_idx[order][:top_k]
    return [(index["titles"][i], float(scores[i]), int(i)) for i in sel]

AURA_HOSTNAME = os.getenv("AURA_HOSTNAME")
AURA_USER     = os.getenv("AURA_USER")
AURA_PASS     = os.getenv("AURA_PASS")
QUERY_API_URL = f"https://{AURA_HOSTNAME}/db/neo4j/query/v2"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

def run_cypher_http(statement: str, params: Optional[dict] = None):
    payload = {"statement": statement}
    if params:
        payload["parameters"] = params
    resp = requests.post(
        QUERY_API_URL,
        headers=HEADERS,
        auth=(AURA_USER, AURA_PASS),  # Basic auth
        data=json.dumps(payload),
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("401 Unauthorized: 사용자/비밀번호 확인 필요")
    if resp.status_code == 403:
        raise RuntimeError("403 Forbidden: 권한/네트워크 정책 확인 필요")
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    fields = data["data"]["fields"]
    values = data["data"]["values"]

    rows = []
    for row in values:
        rows.append([row[i] if isinstance(row, list) else row for i in range(len(fields))])
    return {"keys": fields, "rows": rows}

def run_cypher(statement: str, params: Optional[dict] = None):
    return run_cypher_http(statement, params)

def _parse_data_api_rows(res: Dict[str, Any]) -> List[List[Any]]:
    rows = []
    results = res.get("results") or []
    if results:
        data = results[0].get("data", [])
        for d in data:
            rows.append(d["row"])
    return rows

def _rows_to_dicts(keys: List[str], rows: List[List]) -> List[dict]:
    out = []
    for row in rows:
        d = {}
        for i, k in enumerate(keys):
            d[k] = row[i] if i < len(row) else None
        out.append(d)
    return out

def split_title_path(title: str) -> List[str]:
    parts = re.split(r"\s*(?:->|›|»|:|/|\\)\s*", title)
    parts = [p.strip() for p in parts if p.strip()]
    normed = []
    for p in parts:
        x = p.replace("μ", "µ")
        x = re.sub(r"\s+", " ", x).strip()
        normed.append(x)
    return normed

def fetch_triples_title_path_scope(doc_id: str, title_path: str, limit: int = 3000,
                                   fuzzy: bool = False, depth: int = 3):
    parts = split_title_path(title_path)
    comparator = "CONTAINS" if fuzzy else "="

    common_cypher = f"""
    WITH $parts AS parts
    MATCH (d:Doc {{id: $doc_id}})
    CALL {{
      WITH d, parts
      MATCH p = (d)-[:HAS_CHILD*1..4]->(a)
      WHERE size(nodes(p)) >= 2 AND size(nodes(p)) <= size(parts) + 1
        AND size(nodes(p)) - 1 <= size(parts)
        AND ALL(i IN range(0, size(nodes(p)) - 2) WHERE
                i < size(parts) AND 
                toLower(nodes(p)[i+1].name) {comparator} toLower(parts[i]))
      RETURN p
      ORDER BY length(p) DESC
      LIMIT 1
    }}
    WITH last(nodes(p)) AS anchor, p
    WITH anchor, [n IN nodes(p)[1..] | toLower(trim(n.name))] AS sec_parts
    WITH anchor, apoc.text.join(sec_parts, ' -> ') AS sec_key

    MATCH (anchor)-[:HAS_CHILD*0..{depth}]->(x)
    WHERE x:Section OR x:SubSection OR x:SubSubSection OR x:SubSubSubSection
    WITH sec_key, collect(DISTINCT anchor) + collect(DISTINCT x) AS scope_nodes
    UNWIND scope_nodes AS sn
    MATCH (sn)-[:MENTION]->(e:EntityLocal)
    WITH sec_key, collect(DISTINCT e) AS ents
    """

    cypher_1hop = common_cypher + f"""
    UNWIND ents AS h
    MATCH (h)-[r]->(t:EntityLocal)
    WHERE t IN ents AND r.doc_id = $doc_id
      AND r.page IS NOT NULL AND r.bbox IS NOT NULL
    RETURN coalesce(h.label, h.id) AS h,
           type(r) AS rel,
           coalesce(t.label, t.id) AS t,
           1.0 AS w,
           r.page AS page,
           r.bbox AS bbox,
           sec_key AS source_section
    ORDER BY r.page, r.bbox
    LIMIT {limit // 2}
    """

    cypher_2hop = common_cypher + f"""
    UNWIND ents AS h
    MATCH (h)-[r1]->(m:EntityLocal)-[r2]->(t:EntityLocal)
    WHERE t IN ents AND m IN ents
      AND r1.doc_id = $doc_id AND r2.doc_id = $doc_id
      AND r1.page IS NOT NULL AND r1.bbox IS NOT NULL
      AND r2.page IS NOT NULL AND r2.bbox IS NOT NULL
    RETURN coalesce(h.label, h.id) AS h,
           type(r1) + " -> " + type(r2) AS rel,
           coalesce(t.label, t.id) AS t,
           0.8 AS w,
           r1.page AS page,
           r1.bbox AS bbox,
           sec_key AS source_section
    ORDER BY r1.page, r1.bbox
    LIMIT {limit // 2}
    """

    params = {"doc_id": doc_id, "parts": parts}
    res1 = run_cypher(cypher_1hop, params)
    rows1 = res1.get("rows") or _parse_data_api_rows(res1) or []
    res2 = run_cypher(cypher_2hop, params)
    rows2 = res2.get("rows") or _parse_data_api_rows(res2) or []

    all_rows = rows1 + rows2
    all_rows.sort(key=lambda x: x[3], reverse=True)
    return all_rows[:limit]  # [(h, rel, t, w, page, bbox, source_section)]

def fetch_triples_global_mixed(doc_id: str, limit: int = 3000):
    common_cypher = f"""
    MATCH (d:Doc {{id: $doc_id}})
    MATCH p = (d)-[:HAS_CHILD*0..4]->(x)
    WITH collect(DISTINCT x) AS nodes
    UNWIND nodes AS n
    MATCH (n)-[:MENTION]->(e:EntityLocal)
    WITH collect(DISTINCT e) AS ents
    """

    cypher_1hop = common_cypher + f"""
    UNWIND ents AS h
    MATCH (h)-[r]->(t:EntityLocal)
    WHERE t IN ents AND r.doc_id = $doc_id
      AND r.page IS NOT NULL AND r.bbox IS NOT NULL
    RETURN coalesce(h.label, h.id) AS h,
           type(r) AS rel,
           coalesce(t.label, t.id) AS t,
           1.0 AS w,
           r.page AS page,
           r.bbox AS bbox
    ORDER BY r.page, r.bbox
    LIMIT {limit // 2}
    """

    cypher_2hop = common_cypher + f"""
    UNWIND ents AS h
    MATCH (h)-[r1]->(m:EntityLocal)-[r2]->(t:EntityLocal)
    WHERE t IN ents AND m IN ents
      AND r1.doc_id = $doc_id AND r2.doc_id = $doc_id
      AND r1.page IS NOT NULL AND r1.bbox IS NOT NULL
      AND r2.page IS NOT NULL AND r2.bbox IS NOT NULL
    RETURN coalesce(h.label, h.id) AS h,
           type(r1) + " -> " + type(r2) AS rel,
           coalesce(t.label, t.id) AS t,
           0.8 AS w,
           r1.page AS page,
           r1.bbox AS bbox
    ORDER BY r1.page, r1.bbox
    LIMIT {limit // 2}
    """

    params = {"doc_id": doc_id}
    res1 = run_cypher(cypher_1hop, params)
    rows1 = res1.get("rows") or _parse_data_api_rows(res1) or []
    res2 = run_cypher(cypher_2hop, params)
    rows2 = res2.get("rows") or _parse_data_api_rows(res2) or []

    all_rows = []
    for (h, rel, t, w, page, bbox) in rows1:
        all_rows.append((h, rel, t, float(w), page, bbox))
    for (h, rel, t, w, page, bbox) in rows2:
        all_rows.append((h, rel, t, float(w), page, bbox))
    all_rows.sort(key=lambda x: x[3], reverse=True)
    return all_rows[:limit]  # [(h, rel, t, w, page, bbox)]

def explore_neighbors_guided(
    entities: Optional[List[str]],
    doc_id: str,
    max_cases: int = 20,
    question: Optional[str] = None,
    context_titles: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    return: {type:'table', condition, case, consequence, anchor, text?, score?}
    """
    triples: List[Dict[str, Any]] = []

    cypher_table_all = """
    MATCH (d:Doc {id:$doc_id})-[:HAS_CHILD*0..]->()-[:MENTION]->(cond:EntityLocal)
    MATCH (cond)<-[r1:HAS_CONDITION|HAS_CONDITION_AND|HAS_CONDITION_OR]-(c)
          -[r2:HAS_CONSEQUENCE|HAS_CONSEQUENCE_AND|HAS_CONSEQUENCE_OR]->(res:EntityLocal)
    RETURN cond.label AS condition,
           c.label    AS case,
           res.label  AS consequence,
           r2.anchor_id AS anchor
    """

    cypher_table_with_ent = """
    MATCH (d:Doc {id:$doc_id})-[:HAS_CHILD*0..]->()-[:MENTION]->(cond:EntityLocal)
    WHERE toLower(cond.label) CONTAINS toLower($ent)
    MATCH (cond)<-[r1:HAS_CONDITION|HAS_CONDITION_AND|HAS_CONDITION_OR]-(c)
          -[r2:HAS_CONSEQUENCE|HAS_CONSEQUENCE_AND|HAS_CONSEQUENCE_OR]->(res:EntityLocal)
    RETURN cond.label AS condition,
           c.label    AS case,
           res.label  AS consequence,
           r2.anchor_id AS anchor
    LIMIT $max_cases
    """

    if entities:
        for ent in entities:
            params = {"doc_id": doc_id, "ent": ent, "max_cases": max_cases}
            res = run_cypher(cypher_table_with_ent, params)
            keys, rows = res.get("keys", []), res.get("rows", [])
            for r in _rows_to_dicts(keys, rows):
                if context_titles:
                    anch = (r.get("anchor") or "")
                    if not any(ct in anch for ct in context_titles):
                        continue
                triples.append({
                    "type": "table",
                    "condition": r.get("condition"),
                    "case": r.get("case"),
                    "consequence": r.get("consequence"),
                    "anchor": r.get("anchor"),
                })
        return triples[:max_cases]


    params_all = {"doc_id": doc_id}
    res_all = run_cypher(cypher_table_all, params_all)
    keys_all, rows_all = res_all.get("keys", []), res_all.get("rows", [])
    table_rows = _rows_to_dicts(keys_all, rows_all)

    if context_titles:
        table_rows = [r for r in table_rows if any(ct in (r.get("anchor") or "") for ct in context_titles)]
    if not table_rows:
        return []

    texts = [f"{r.get('condition','')} [CASE] {r.get('case','')} -> {r.get('consequence','')}" for r in table_rows]

    if not question:  
        out = []
        for r, t in zip(table_rows, texts):
            out.append({
                "type": "table",
                "condition": r.get("condition"),
                "case": r.get("case"),
                "consequence": r.get("consequence"),
                "anchor": r.get("anchor"),
                "text": t,
            })
        return out[:max_cases]

    qv = embed_text(question)          # (d,)
    X  = embed_texts(texts)            # (N, d)
    scores = (X @ qv).tolist()

    idxs = list(range(len(table_rows)))
    idxs.sort(key=lambda i: scores[i], reverse=True)

    out = []
    for i in idxs[:max_cases]:
        r = table_rows[i]
        out.append({
            "type": "table",
            "condition": r.get("condition"),
            "case": r.get("case"),
            "consequence": r.get("consequence"),
            "anchor": r.get("anchor"),
            "text": texts[i],
            "score": float(scores[i]),
        })
    return out

def zscore(arr: List[float]) -> List[float]:
    if not arr: return []
    mu = float(np.mean(arr))
    sd = float(np.std(arr)) + 1e-8
    return [(x - mu) / sd for x in arr]

def section_hit_sim(payload: Dict[str, Any], section_sims: Dict[str, float]) -> float:
    sec = (payload.get("source_section") or "").lower().strip()
    return float(section_sims.get(sec, 0.0))

def dedup_by_triple(items):
    seen = {}
    for payload, sc in items:
        key = (payload.get("h"), payload.get("r"), payload.get("t"), (payload.get("source_section") or "_GLOBAL_"))
        if key not in seen or sc > seen[key][1]:
            seen[key] = (payload, sc)
    return list(seen.values())

def mmr_diversify(candidates, k=30, lambda_div=0.7, max_per_section=10, emb_model=None):
    if not candidates: return []
    texts = [c[0].get("text") or f"{c[0].get('h')} [{c[0].get('r')}] {c[0].get('t')}" for c in candidates]
    X = EMBEDDER.encode(texts, convert_to_numpy=True, normalize_embeddings=True) if emb_model is None \
        else emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    sorted_idx = np.argsort([-c[1] for c in candidates])
    selected, selected_idx, per_section = [], [], {}

    def cosine_sim(i, j): return float(np.dot(X[i], X[j]))

    while len(selected) < min(k, len(candidates)):
        best_i, best_val = None, -1e9
        for i in sorted_idx:
            if i in selected_idx: 
                continue
            payload, base = candidates[i]
            sec_key = (payload.get("source_section") or "_GLOBAL_").lower().strip()
            if per_section.get(sec_key, 0) >= max_per_section:
                continue
            if not selected_idx:
                mmr_score = base
            else:
                sim_to_S = max(cosine_sim(i, j) for j in selected_idx)
                mmr_score = lambda_div * base - (1 - lambda_div) * sim_to_S
            if mmr_score > best_val:
                best_val, best_i = mmr_score, i
        if best_i is None:
            break
        selected_idx.append(best_i)
        selected.append(candidates[best_i])
        sec_key = (candidates[best_i][0].get("source_section") or "_GLOBAL_").lower().strip()
        per_section[sec_key] = per_section.get(sec_key, 0) + 1
    return selected

def _onto_compat_stub(h, r, t) -> float:
    return 0.0

def _path_prior_stub(w, page, bbox) -> float:
    score = 0.0
    if w is not None:
        if w >= 1.0: score += 0.2   # 1-hop
        elif w >= 0.8: score += 0.1 # 2-hop
    return score

def fetch_triples_mixed_guided(
    doc_id: str,
    section_title_paths: Optional[List[str] | List[Tuple[str, int]]] = None,
    include_graph: bool = True,
    sec_depth: int = 3,
    fuzzy: bool = False,
    sec_limit_default: int = 3000,
    glb_limit: int = 3000,
    include_table: bool = True,
    entities: Optional[List[str]] = None,
    context_titles: Optional[List[str]] = None,
    question: Optional[str] = None,
) -> List[Dict[str, Any]]:

    triples: List[Dict[str, Any]] = []

    # Section graph scope
    if include_graph and section_title_paths:
        for item in section_title_paths:
            title_path, sec_lim = (item if isinstance(item, tuple) else (item, sec_limit_default))
            rows = fetch_triples_title_path_scope(
                doc_id=doc_id, title_path=title_path, limit=sec_lim, fuzzy=fuzzy, depth=sec_depth
            )
            for (h, rel, t, w, page, bbox, source_section) in rows:
                triples.append({
                    "type": "graph", "source": "section", "hop": "1or2",
                    "h": h, "r": rel, "t": t, "w": float(w),
                    "page": page, "bbox": bbox,
                    "source_section": (source_section or "").lower().strip(),
                    "text": f"{h} [{rel}] {t}",
                })

    # Global graph scope
    if include_graph:
        rows_glb = fetch_triples_global_mixed(doc_id=doc_id, limit=glb_limit)
        for (h, rel, t, w, page, bbox) in rows_glb:
            triples.append({
                "type": "graph", "source": "global", "hop": "1or2",
                "h": h, "r": rel, "t": t, "w": float(w),
                "page": page, "bbox": bbox, "source_section": None,
                "text": f"{h} [{rel}] {t}",
            })

    # Table (+optional semantic via question)
    if include_table:
        print("include_table")
        table_sem = explore_neighbors_guided(
            entities=entities, doc_id=doc_id, max_cases=min(200, glb_limit),
            context_titles=context_titles, question=question
        )
        for r in table_sem or []:
            if r.get("type") == "table":
                triples.append({
                    "type": "table", "source": "table",
                    "condition": r.get("condition"), "case": r.get("case"),
                    "consequence": r.get("consequence"), "w": 0.9,
                    "page": None, "bbox": None, "source_section": None,
                    "text": f"{r.get('condition')} [CASE] {r.get('case')} -> {r.get('consequence')}",
                    "anchor": r.get("anchor"),
                })
            else:
                w = float(r.get("score", 0.7))
                triples.append({
                    "type": "semantic", "source": "semantic",
                    "h": r.get("h"), "r": r.get("rel"), "t": r.get("t"),
                    "w": w, "page": r.get("page"), "bbox": r.get("bbox"),
                    "source_section": r.get("source_section"),
                    "text": f"{r.get('h')} [{r.get('rel')}] {r.get('t')}",
                })

    def _key(d: Dict[str, Any]) -> Tuple:
        return (d.get("type"), d.get("h"), d.get("r"), d.get("t"),
                d.get("condition"), d.get("case"), d.get("consequence"),
                d.get("page"), str(d.get("bbox")), d.get("source_section"), d.get("source"))
    seen, deduped = set(), []
    for d in triples:
        k = _key(d)
        if k in seen: continue
        seen.add(k); deduped.append(d)

    deduped.sort(key=lambda x: x.get("w", 0.0), reverse=True)
    return deduped


@dataclass
class HybridHP:
    K_SEC: int = 30
    K_GLB: int = 30
    K_FINAL: int = 30
    SEC_TRIPLE_CAP: int = 5000
    GLB_TRIPLE_CAP: int = 5000
    MMR_LAMBDA: float = 0.7
    MAX_PER_SECTION: int = 10
    SEC_CONF_LOW: float = 0.35
    BETA_SCALE: float = 0.6

def hybrid_retrieve_topk(
    question: str,
    doc_id: str,
    section_title_paths: List[str] | List[Tuple[str,int]],
    section_sims: Dict[str, float],
    hp: HybridHP = HybridHP(),
    entities: Optional[List[str]] = None,
):
    pool = fetch_triples_mixed_guided(
        doc_id=doc_id,
        section_title_paths=section_title_paths,
        include_graph=True, sec_depth=3, fuzzy=False,
        sec_limit_default=hp.SEC_TRIPLE_CAP, glb_limit=hp.GLB_TRIPLE_CAP,
        include_table=True, entities=entities,
        context_titles=list(section_sims.keys()), question=question
    )

    qv = embed_text(question)
    texts = [p.get("text", "") for p in pool]
    X = embed_texts(texts) if texts else np.zeros((0, qv.shape[0]), dtype=np.float32)
    sims = list((X @ qv)) if len(X) else []
    zs = zscore(sims) if sims else []

    max_sec_sim = max(section_sims.values()) if section_sims else 0.0
    beta = hp.BETA_SCALE * max_sec_sim
    if max_sec_sim < hp.SEC_CONF_LOW:
        beta *= 0.5

    def final_score(payload, base_z):
        sec_bias = beta * section_hit_sim(payload, section_sims) if payload.get("source") == "section" else 0.0
        ont = _onto_compat_stub(payload.get("h"), payload.get("r"), payload.get("t"))
        pathp = _path_prior_stub(payload.get("w"), payload.get("page"), payload.get("bbox"))
        return float(base_z + sec_bias + 0.6*ont + 0.3*pathp)

    scored = [(p, final_score(p, b)) for p, b in zip(pool, zs)]
    scored.sort(key=lambda x: x[1], reverse=True)

    topN = scored[: (hp.K_SEC + hp.K_GLB + 2*hp.K_FINAL)]
    dedup = dedup_by_triple(topN)
    selected = mmr_diversify(dedup, k=hp.K_FINAL, lambda_div=hp.MMR_LAMBDA, max_per_section=hp.MAX_PER_SECTION)
    return selected  # [(payload, score)]


SYSTEM_PROMPT = (
"""
You are an expert in API SPEC documents.
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
)
def format_triples_for_context(
    triples_ranked: List[Tuple[str, str, str, float, float, Any, Any]],
    show_scores: bool = False,
) -> str:
    lines = []
    for (h, r, t, _w, sim, _page, _bbox) in triples_ranked:
        lines.append(f"({h}; {r}; {t})" if not show_scores else f"({h}; {r}; {t})  // sim={sim:.3f}")
    return "\n".join(lines)

def build_prompt(question: str, triples_ranked: List[Tuple[str, str, str, float, float, Any, Any]]) -> str:
    ctx = format_triples_for_context(triples_ranked, show_scores=False)
    return (
        f"""{SYSTEM_PROMPT}
Now answer the following question using the provided related Triplets.

Related Triplets:
{ctx}

Question: {question}
Answer:"""
    )

from openai import OpenAI
def call_openai_chat(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "[No OPENAI_API_KEY]:\n\n" + prompt
    time.sleep(2)
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    return call_openai_chat(prompt, system)

def answer_hybrid(
    doc_id: str,
    title_paths,                # str | list[str] | list[tuple[str,int]]
    question: str,
    section_sim_index=None,
    hp: HybridHP = HybridHP()
) -> Dict[str, Any]:
    if isinstance(title_paths, str):
        title_paths = [title_paths]
    paths_and_limits: List[Tuple[str, int]] = []
    for it in title_paths:
        paths_and_limits.append(it if isinstance(it, tuple) else (it, hp.SEC_TRIPLE_CAP))

    section_sims: Dict[str, float] = {}
    for path, _ in paths_and_limits:
        key = ' -> '.join([p.strip().lower() for p in split_title_path(path)])
        if section_sim_index is not None:
            section_sims[key] = float(section_sim_index.get(key, 0.0))
        else:
            section_sims[key] = float(np.dot(embed_text(path), embed_text(question)))

    selected = hybrid_retrieve_topk(
        question=question,
        doc_id=doc_id,
        section_title_paths=paths_and_limits,
        section_sims=section_sims,
        hp=hp
    )  # [(payload, score)]

    triples_ranked = []
    for payload, sc in selected:
        triples_ranked.append((
            payload.get("h"), payload.get("r"), payload.get("t"),
            payload.get("w"), float(sc), payload.get("page"), payload.get("bbox")
        ))
    prompt = build_prompt(question, triples_ranked)
    pred = call_llm(prompt)

    path_stats = []
    for p, lim in paths_and_limits:
        k = ' -> '.join([x.strip().lower() for x in split_title_path(p)])
        cnt = sum(1 for (pl, _) in selected if (pl.get("source_section") or "") == k)
        path_stats.append({"title_path": p, "limit": lim, "selected": cnt, "sec_sim": section_sims.get(k, 0.0)})

    pred_triplet = [{
        "h": h, "r": r, "t": t,
        "sim": float(sim), "page": page,
        "bbox": list(bbox) if bbox is not None else None,
    } for (h, r, t, w, sim, page, bbox) in triples_ranked]

    return {
        "scope": {"type": "hybrid_sec3hop+global1hop", "doc_id": doc_id},
        "path_stats": path_stats,
        "stats": {
            "sec_topk": hp.K_SEC, "glb_topk": hp.K_GLB, "final_k": hp.K_FINAL,
            "mmr_lambda": hp.MMR_LAMBDA, "max_per_section": hp.MAX_PER_SECTION,
            "beta_scale": hp.BETA_SCALE, "sec_conf_low": hp.SEC_CONF_LOW,
            "triples_selected": len(triples_ranked),
        },
        "prompt_preview": prompt,
        "pred": pred,
        "pred_triplet": pred_triplet,
    }

if __name__ == "__main__":
    DOC_LIST = ["API_2W"]
    for DOC_ID in DOC_LIST:
        index = load_index(f"{DOC_ID}_title_embeddings.pkl")
        qa_dataset = pd.read_json("file_path", lines=True)
        
        # 샘플하겠다!
        qa_dataset = qa_dataset.sample(frac=1).reset_index(drop=True)
        output_path = "output_path"
        hp = HybridHP(K_SEC=30, K_GLB=30, K_FINAL=30, MAX_PER_SECTION=30)
        with open(output_path, "a", encoding="utf-8") as f:  
            for idx, row in qa_dataset.iterrows():
                Q = row['question']
                results = search_index(index, Q, top_k=5, threshold=0.1)
                        
                TITLE_PATH = [i[0] for i in results[:3]]
                TITLE_PATH_QUOTAS = [(TITLE_PATH[0], 20), (TITLE_PATH[1], 10), (TITLE_PATH[2], 5)]
                res = answer_hybrid(DOC_ID, TITLE_PATH, Q, hp=hp)
                
                record = {
                    "id": row["id"],
                    "question": row["question"],
                    "answers": row["answers"],
                    "unanswerable": row["unanswerable"],
                    "page": row["page_no"],
                    "bbox": row["bbox"],
                    "type": row["type"], 
                    "pred_answer": res["pred"],
                    "pred_triplet": res["pred_triplet"], 
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if idx % 30 == 0:
                    if not results:
                        print("No hits above threshold.")
                    else:
                        print("=== TOP-TITLE RESULT ===")
                        for title, score, idx in results:
                            print(f"- {title}\t(sim={score:.4f})")
                    print("=== DOC RESULT ===")
                    print(res["scope"], res["stats"])
                    print("prompt: ")
                    print(res['prompt_preview'])
                    print("---- Pred Answer ----")
                    print(res["pred"])
        print("finish!")