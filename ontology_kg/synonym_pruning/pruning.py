# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import ast
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer


def collect_triples_from_df(df: pd.DataFrame, col: str = "parsed_triplets") -> List[Tuple[str, str, str]]:
    """Collect all (subj, edge, obj) triples from a dataframe column."""
    triples = []
    for row in df[col].dropna():
        for tri in row:
            h, r, t = tri.get("subj", ""), tri.get("edge", ""), tri.get("obj", "")
            if h and r and t:
                triples.append((h, r, t))
    return triples


def normalized(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Normalize entity forms using lemmatization and detect duplicates."""
    triples = collect_triples_from_df(df)
    entities = sorted(list({h for h, _, _ in triples} | {t for _, _, t in triples}))
    lemmatizer = WordNetLemmatizer()
    normalized = defaultdict(list)

    for w in entities:
        w_clean = w.lower().replace("-", " ")
        lemma = " ".join([lemmatizer.lemmatize(tok) for tok in w_clean.split()])
        normalized[lemma].append(w)

    normalized = dict(normalized)
    duplicates = {lemma: words for lemma, words in normalized.items() if len(words) > 1}
    return duplicates


def replace_with_representative(df: pd.DataFrame) -> List[List[Dict[str, str]]]:
    """Replace duplicate entities with their representative lemma form."""
    duplicates = normalized(df)
    replaced = []
    change_counter = Counter()
    total = changed = 0
    triplets = df["parsed_triplets"]

    for row in triplets:
        new_row = []
        for triplet in row:
            new_triplet = triplet.copy()
            for field in ["subj", "obj"]:
                val = new_triplet[field]
                total += 1
                for lemma, words in duplicates.items():
                    if val in words:
                        if val != lemma:
                            changed += 1
                            change_counter[(val, lemma)] += 1
                        new_triplet[field] = lemma
                        break
            new_row.append(new_triplet)
        replaced.append(new_row)

    stats = {
        "total entity count": total,
        "number of replacements": changed,
        "replacement ratio": changed / total if total else 0,
        "replacements per representative": change_counter,
    }
    print(stats)
    return replaced


def parse_triplet(triplet_str: str) -> Optional[Dict[str, str]]:
    """Parse a string-formatted triplet into a dictionary."""
    pattern = r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*)"
    match = re.match(pattern, triplet_str)
    if match:
        subj, obj, edge = match.groups()
        return {"subj": subj.strip(), "obj": obj.strip(), "edge": edge.strip()}
    return None


def extract_triplets(cell_value: str) -> List[Dict[str, str]]:
    """Extract all triplets from a cell containing multiple <triplet> patterns."""
    if not isinstance(cell_value, str) or "<triplet>" not in cell_value:
        return []
    parts = cell_value.split("\n")
    results = []
    for p in parts:
        parsed = parse_triplet(p)
        if parsed:
            results.append(parsed)
    return results


def replace_edge_with_canonical(edge_df: pd.DataFrame, parsed_triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Replace predicate edges with canonical labels based on clustering results."""
    edge_to_canonical = {}

    for _, row in edge_df.iterrows():
        cluster_id = row["cluster_id"]
        canonical_predicate = row["canonical_predicate"]
        paraphrases_str = row["paraphrases"]

        if cluster_id == -1:
            continue

        try:
            paraphrases = ast.literal_eval(paraphrases_str) if isinstance(paraphrases_str, str) else paraphrases_str
            for paraphrase in paraphrases:
                edge_to_canonical[paraphrase] = canonical_predicate
        except Exception:
            continue

    updated_triplets = []
    for triplet in parsed_triplets:
        updated_triplet = triplet.copy()
        current_edge = triplet["edge"]
        if current_edge in edge_to_canonical:
            updated_triplet["edge"] = edge_to_canonical[current_edge]
        updated_triplets.append(updated_triplet)
    return updated_triplets


@dataclass
class PostProcessHP:
    """Hyperparameters for graph post-processing."""
    absolute_percentile: float = 0.0
    relative_percentile: float = 1.0
    remove_self_loops: bool = True
    remove_inverse_edges: bool = True


def _self_loops(G: nx.DiGraph) -> set:
    return {(u, v) for u, v in G.edges if u == v}


def _inverse_edges(G: nx.DiGraph) -> set:
    def w(u, v): return G[u][v].get("weight", 1)
    return {(u, v) for u, v in G.edges if G.has_edge(v, u) and w(v, u) > w(u, v)}


def _absolute_percentile_edges(G: nx.DiGraph, p: float) -> set:
    if p <= 0:
        return set()
    if p >= 1:
        return set(G.edges)
    edges = list(G.edges)
    weights = np.array([G[u][v].get("weight", 1) for u, v in edges], dtype=float)
    k = int(p * len(edges))
    if k <= 0:
        return set()
    idx = np.argpartition(weights, k)[:k]
    return {edges[i] for i in idx}


def _relative_percentile_edges(G: nx.DiGraph, keep: float) -> set:
    assert 0.0 <= keep <= 1.0
    if keep >= 1.0:
        return set()
    drop = set()
    for n in G.nodes:
        outs = list(G.out_edges(n))
        if not outs:
            continue
        ws = np.array([G[u][v].get("weight", 1) for u, v in outs], dtype=float)
        if ws.sum() <= 0:
            keep_i = int(np.argmax(ws))
            for i, e in enumerate(outs):
                if i != keep_i:
                    drop.add(e)
            continue
        sorted_ws = np.sort(ws)[::-1]
        cum = (sorted_ws / sorted_ws.sum()).cumsum()
        cut_i = int(np.argmax(cum > keep))
        cutoff = sorted_ws[cut_i]
        for (e, w) in zip(outs, ws):
            if w <= cutoff:
                drop.add(e)
    return drop


def post_process(G: nx.DiGraph, hp: PostProcessHP) -> Tuple[nx.DiGraph, int]:
    """Remove redundant, low-weight, inverse, or self-loop edges."""
    to_remove = set()
    to_remove |= _absolute_percentile_edges(G, hp.absolute_percentile)
    to_remove |= _relative_percentile_edges(G, hp.relative_percentile)
    if hp.remove_inverse_edges:
        to_remove |= _inverse_edges(G)
    if hp.remove_self_loops:
        to_remove |= _self_loops(G)
    Gp = nx.edge_subgraph(G, set(G.edges) - to_remove).copy()
    return Gp, len(to_remove)


def _add_edge(G: nx.DiGraph, u: str, v: str, rel: str, coords=None, page=None, source=None):
    """Add or update an edge in the graph."""
    if G.has_edge(u, v):
        d = G[u][v]
        d["weight"] = d.get("weight", 0) + 1
        pc = d.setdefault("pred_counts", Counter())
        pc[rel] += 1
        d.setdefault("pages", set())
        if page is not None:
            d["pages"].add(page)
        d.setdefault("coords", [])
        if coords is not None:
            d["coords"].append(coords)
        d.setdefault("sources", set())
        if source is not None:
            d["sources"].add(source)
    else:
        G.add_edge(
            u, v,
            weight=1,
            pred_counts=Counter({rel: 1}),
            edge=rel,
            coordinates=coords,
            page=page,
            pages={page} if page is not None else set(),
            coords=[coords] if coords is not None else [],
            sources={source} if source is not None else set(),
        )


def _finalize_majority_relation(G: nx.DiGraph):
    """Select the most frequent relation label for each edge."""
    for _, _, d in G.edges(data=True):
        pc: Counter = d.get("pred_counts", Counter())
        if pc:
            d["edge"] = pc.most_common(1)[0][0]


def build_graph_for_title_group_from_df(
    df: pd.DataFrame,
    triplet_col: str = "parsed_triplets",
    page_col: str = "page",
    coord_col: str = "coordinates",
    row_id_col: str = "sort_id",
) -> Tuple[nx.DiGraph, int]:
    """Build a graph for all triplets within a single title group."""
    G = nx.DiGraph()
    seen = set()
    dup_removed = 0

    for i, row in df.iterrows():
        triples = row.get(triplet_col, None)
        page = row.get(page_col, None)
        coords = row.get(coord_col, None)
        source = row.get(row_id_col, i)
        if not isinstance(triples, list):
            continue

        for t in triples:
            u = str(t.get("subj", "")).strip()
            v = str(t.get("obj", "")).strip()
            r = str(t.get("edge", "related_to")).strip()
            if not (u and v):
                continue
            key = (u, r, v)
            if key in seen:
                dup_removed += 1
                continue
            seen.add(key)
            _add_edge(G, u, v, r, coords, page, source)

    _finalize_majority_relation(G)
    return G, dup_removed


def graph_to_edge_records(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Convert graph edges into structured records."""
    return [
        {"subj": u, "obj": v, "edge": d.get("edge"), "coordinates": d.get("coordinates"), "page": d.get("page")}
        for u, v, d in G.edges(data=True)
    ]


def build_kgs_by_title_and_annotate_rows(
    df: pd.DataFrame,
    canonical_col: str = "canonical_triplets",
    title_col: str = "title",
    page_col: str = "page",
    coord_col: str = "coordinates",
    row_id_col: str = "sort_id",
    hp: "PostProcessHP" = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build pruned KGs grouped by title and annotate back to each row."""
    if hp is None:
        hp = PostProcessHP()

    pruned_edges_row = defaultdict(list)
    kept_edges_count_row = defaultdict(int)
    summary_rows = []
    dup_removed_total = 0
    removed_total = 0

    for title, g in df.groupby(title_col, dropna=False):
        G, dup_removed = build_graph_for_title_group_from_df(
            g, triplet_col=canonical_col, page_col=page_col, coord_col=coord_col, row_id_col=row_id_col
        )
        Gp, removed = post_process(G, hp)
        dup_removed_total += dup_removed
        removed_total += removed

        title_edge_records = graph_to_edge_records(Gp)
        summary_rows.append(
            {"title": title, "kg_edges": title_edge_records, "removed_duplicates_count": dup_removed, "removed_edges_count": removed}
        )

        for u, v, d in Gp.edges(data=True):
            edge_label = d.get("edge")
            pages = d.get("pages")
            coords_list = d.get("coords")
            sources = d.get("sources", set())
            edge_rec = {"subj": u, "obj": v, "edge": edge_label, "title": title, "pages": pages, "coordinates_list": coords_list}
            for sid in sources:
                pruned_edges_row[sid].append(edge_rec)
                kept_edges_count_row[sid] += 1

    kg_summary = pd.DataFrame(summary_rows)
    df2 = df.copy()
    df2["pruned_edges_row"] = df2[row_id_col].map(lambda rid: pruned_edges_row.get(rid, []))
    df2["kept_edges_count_row"] = df2[row_id_col].map(lambda rid: kept_edges_count_row.get(rid, 0))
    print("Duplicate edges removed:", dup_removed_total)
    print("Pruned edges removed:", removed_total)
    return kg_summary, df2


if __name__ == "__main__":
    # Example usage (replace paths accordingly)
    data_name = "YOUR_DATA_NAME"
    print(data_name)

    edge_df = pd.read_csv(f"path/to/{data_name}_triplet_edge.csv")
    df = pd.read_excel(f"path/to/{data_name}_triplet_total.xlsx")

    df["parsed_triplets"] = df["triplets"].apply(extract_triplets)
    df["canonical_triplets"] = replace_with_representative(df)
    df["canonical_triplets"] = df["canonical_triplets"].apply(
        lambda triplets: replace_edge_with_canonical(edge_df, triplets) if triplets else []
    )

    hp = PostProcessHP(absolute_percentile=0.0, relative_percentile=1.0, remove_self_loops=True, remove_inverse_edges=True)

    kg_summary, df_row_annot = build_kgs_by_title_and_annotate_rows(
        df,
        canonical_col="canonical_triplets",
        title_col="title",
        page_col="page",
        coord_col="coordinates",
        row_id_col="sort_id",
        hp=hp,
    )

    kg_summary.to_json("path/to/output_kg_summary.jsonl", orient="records", lines=True, force_ascii=False)
    df_row_annot[["sort_id", "title", "text", "page", "coordinates", "pruned_edges_row"]].to_csv(
        "path/to/output_row_annotations.csv", index=False
    )
