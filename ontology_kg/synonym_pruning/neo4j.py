# -*- coding: utf-8 -*-
import json, math, re, time, requests, hashlib, ast
from typing import Iterable, Dict, Any, List
import os

BATCH_SIZE = 500
# ====================================================
# BBOX
def _parse_bbox(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try: return [float(v) for v in x]
        except: return None
    if isinstance(x, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                arr = parser(x)
                if isinstance(arr, (list, tuple)) and len(arr) == 4:
                    return [float(v) for v in arr]
            except: pass
    return None

def _round_bbox(bbox, ndigits=3):
    if bbox is None: return None
    return [round(float(v), ndigits) for v in bbox]

def _bbox_hash(doc_id, anchor_id, subj, edge, obj, page, bbox):
    key = f"{doc_id}|{anchor_id}|{subj}|{edge}|{obj}|{page}|{bbox}".encode("utf-8")
    return hashlib.blake2b(key, digest_size=12).hexdigest()



def now_epoch() -> int:
    return int(time.time())

def split_title_chain(title: str) -> List[str]:
    if title is None: return []
    parts = [p.strip() for p in str(title).split("->")]
    return [p for p in parts if p]

def to_reltype(verb: str) -> str:
    s = (verb or "").strip()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "RELATED_TO").upper()

def run_cypher(statement: str, params: dict | None = None):
    payload = {"statement": statement, "parameters": params or {}}
    r = requests.post(
        AURA_URI,
        auth=(AURA_USER, AURA_PASS),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120
    )
    r.raise_for_status()
    res = r.json()
    if res.get("errors"):
        raise RuntimeError(res["errors"])
    return res



CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Doc) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT sec_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT subsec_id IF NOT EXISTS FOR (n:SubSection) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT subsub_id IF NOT EXISTS FOR (n:SubSubSection) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT subsubsub_id IF NOT EXISTS FOR (n:SubSubSubSection) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_local_id IF NOT EXISTS FOR (e:EntityLocal) REQUIRE e.id IS UNIQUE",
]

# -------------------- UPSERT_HIERARCHY --------------------
UPSERT_HIERARCHY = """
UNWIND $rows AS row
MERGE (d:Doc {id: row.doc_id})
  ON CREATE SET d.created_at = timestamp()
SET d.name = row.doc_name

WITH row, d
FOREACH (_ IN CASE WHEN row.section IS NULL THEN [] ELSE [1] END |
  MERGE (sec:Section {id: row.sec_id})
    ON CREATE SET sec.created_at = timestamp()
  SET sec.name = row.section
  MERGE (d)-[:HAS_CHILD]->(sec)

  FOREACH (_ IN CASE WHEN row.subsection IS NULL THEN [] ELSE [1] END |
    MERGE (sub:SubSection {id: row.subsec_id})
      ON CREATE SET sub.created_at = timestamp()
    SET sub.name = row.subsection
    MERGE (sec)-[:HAS_CHILD]->(sub)

    FOREACH (_ IN CASE WHEN row.subsubsection IS NULL THEN [] ELSE [1] END |
      MERGE (ssub:SubSubSection {id: row.subsub_id})
        ON CREATE SET ssub.created_at = timestamp()
      SET ssub.name = row.subsubsection
      MERGE (sub)-[:HAS_CHILD]->(ssub)

      FOREACH (_ IN CASE WHEN row.subsubsubsection IS NULL THEN [] ELSE [1] END |
        MERGE (sssub:SubSubSubSection {id: row.subsubsub_id})
          ON CREATE SET sssub.created_at = timestamp()
        SET sssub.name = row.subsubsubsection
        MERGE (ssub)-[:HAS_CHILD]->(sssub)
      )
    )
  )
)
"""

# -------------------- stable ID --------------------
def stable_eid(label: str, scope: str) -> str:
    key = f"{label}||{scope}".encode("utf-8")
    return "E:" + hashlib.blake2b(key, digest_size=12).hexdigest()

# -------------------- df -> rows --------------------
def rows_from_df(df, doc_id: str, doc_name: str) -> Iterable[Dict[str, Any]]:
    for _, r in df.iterrows():
        tokens = split_title_chain(r.get("title", ""))
        level  = len(tokens)

        section          = tokens[0] if level >= 1 else None
        subsection       = tokens[1] if level >= 2 else None
        subsubsection    = tokens[2] if level >= 3 else None
        subsubsubsection = tokens[3] if level >= 4 else None

        sec_id        = f"{doc_id}::{section}" if section else None
        subsec_id     = f"{doc_id}::{section} -> {subsection}" if subsection else None
        subsub_id     = f"{doc_id}::{section} -> {subsection} -> {subsubsection}" if subsubsection else None
        subsubsub_id  = f"{doc_id}::{section} -> {subsection} -> {subsubsection} -> {subsubsubsection}" if subsubsubsection else None

        if subsubsubsection:
            anchor_label, anchor_id = "SubSubSubSection", subsubsub_id
        elif subsubsection:
            anchor_label, anchor_id = "SubSubSection", subsub_id
        elif subsection:
            anchor_label, anchor_id = "SubSection", subsec_id
        elif section:
            anchor_label, anchor_id = "Section", sec_id
        else:
            anchor_label, anchor_id = None, None

        pts = r.get("kg_edges", []) or []
        if isinstance(pts, str):
            try:
                pts = json.loads(pts)
            except Exception:
                try:
                    pts = ast.literal_eval(pts)
                except Exception:
                    pts = []

        base_title_row = {
            "doc_id": doc_id, "doc_name": doc_name,
            "section": section, "subsection": subsection,
            "subsubsection": subsubsection, "subsubsubsection": subsubsubsection,
            "sec_id": sec_id, "subsec_id": subsec_id,
            "subsub_id": subsub_id, "subsubsub_id": subsubsub_id,
            "anchor_label": anchor_label, "anchor_id": anchor_id,
        }

        title_scope = f"{doc_id}||{anchor_id}" if anchor_id else doc_id

        for t in pts:
            subj = (t.get("subj") or "").strip()
            edge = (t.get("edge") or "").strip()
            obj  = (t.get("obj")  or "").strip()
            if not (subj and edge and obj):
                continue

            page_raw = t.get("page", None)
            page = int(page_raw) if page_raw is not None and str(page_raw).strip() != "" else None
            bbox_raw = _parse_bbox(t.get("coordinates"))
            bbox = _round_bbox(bbox_raw, ndigits=3) if bbox_raw is not None else None
            bbox_hash = _bbox_hash(doc_id, anchor_id, subj, edge, obj, page, bbox) if (page is not None and bbox is not None) else None

            lsid = stable_eid(subj, title_scope)
            loid = stable_eid(obj,  title_scope)
            source = doc_id

            yield {
                **base_title_row,
                "subj": subj, "obj": obj, "edge": edge,
                "reltype": to_reltype(edge),
                "source": source,
                "ts": int(t.get("ts", now_epoch())),
                "lsid": lsid, "loid": loid,
                "title_scope": title_scope,
                "page": page,
                "bbox": bbox,
                "bbox_hash": bbox_hash,
            }


# -------------------- insert --------------------
def load_rows(rows: Iterable[Dict[str, Any]]):
    for q in CREATE_CONSTRAINTS:
        run_cypher(q)

    rows = list(rows)
    if not rows:
        print("No rows to load."); return

    # ---------- title-level ----------
    title_rows, seen = [], set()
    for params in rows:
        key = (params["sec_id"], params["subsec_id"], params["subsub_id"], params["subsubsub_id"])
        if key in seen:
            continue
        seen.add(key)
        title_rows.append({
            "doc_id": params["doc_id"], "doc_name": params["doc_name"],
            "section": params["section"], "subsection": params["subsection"],
            "subsubsection": params["subsubsection"], "subsubsubsection": params["subsubsubsection"],
            "sec_id": params["sec_id"], "subsec_id": params["subsec_id"],
            "subsub_id": params["subsub_id"], "subsubsub_id": params["subsubsub_id"],
        })

    for i in range(0, len(title_rows), BATCH_SIZE):
        chunk = title_rows[i:i+BATCH_SIZE]
        run_cypher(UPSERT_HIERARCHY, {"rows": chunk})
        print(f"[Hierarchy] {i//BATCH_SIZE+1}/{math.ceil(len(title_rows)/BATCH_SIZE or 1)}")

    # ---------- entity & relation ----------
    for i, r in enumerate(rows, 1):
        # 1) local entity
        cypher_entities_local = """
        MERGE (s:EntityLocal {id:$lsid})
          ON CREATE SET s.label=$subj, s.source=$source, s.created_at=timestamp()
        SET s.label = coalesce(s.label, $subj)

        MERGE (o:EntityLocal {id:$loid})
          ON CREATE SET o.label=$obj, o.source=$source, o.created_at=timestamp()
        SET o.label = coalesce(o.label, $obj)
        """
        run_cypher(cypher_entities_local, r)

        # 2) MENTION 
        if r["anchor_id"]:
            cypher_mention = """
            MATCH (a {id:$anchor_id})
            MATCH (s:EntityLocal {id:$lsid})
            MATCH (o:EntityLocal {id:$loid})
            MERGE (a)-[:MENTION]->(s)
            MERGE (a)-[:MENTION]->(o)
            """
            run_cypher(cypher_mention, r)

        # 3) relation (doc_id + anchor_id + label + page + bbox_hash)
        reltype = r["reltype"]
        cypher_rel = f"""
        MATCH (s:EntityLocal {{id:$lsid}})
        MATCH (o:EntityLocal {{id:$loid}})
        MERGE (s)-[r:`{reltype}` {{
            doc_id:$doc_id,
            anchor_id:$anchor_id,
            label:$edge,
            page:$page,
            bbox_hash:$bbox_hash
        }}]->(o)
        ON CREATE SET r.source=$source, r.ts=$ts, r.bbox=$bbox
        """
        run_cypher(cypher_rel, r)

        if i % 100 == 0:
            print(f"[Triplets] {i}/{len(rows)}")

    print("Done.")

# -------------------- 
if __name__ == "__main__":
    import pandas as pd
    data = ["A578A578M_07", "API_2W", "A6A6M_14"]
    for DOC_name in ["API_2W"]:
        print(f"{DOC_name} start!")
        df = pd.read_json(f"file_path", lines=True)
        rows = rows_from_df(df, DOC_name, DOC_name)
        load_rows(rows)
