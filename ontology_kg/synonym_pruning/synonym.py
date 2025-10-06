# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import networkx as nx
from collections import defaultdict, Counter

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import hdbscan

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


def collect_triples_from_df(df: pd.DataFrame, col: str = "parsed_triplets") -> List[Tuple[str, str, str]]:
    triples = []
    for row in df[col].dropna():
        for tri in row:
            h = tri.get("subj", "")
            r = tri.get("edge", "")
            t = tri.get("obj", "")
            if h and r and t:
                triples.append((h, r, t))
    return triples


def build_entity_relation_index(triples: List[Tuple[str, str, str]]):
    entities = sorted(list({h for h, _, _ in triples} | {t for _, _, t in triples}))
    relations = sorted(list({r for _, r, _ in triples}))
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    return entities, relations, ent2id, rel2id


def build_undirected_graph(triples: List[Tuple[str, str, str]], ent2id: Dict[str, int]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(ent2id.values())
    weight = Counter()
    
    for h, _, t in triples:
        u, v = ent2id[h], ent2id[t]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        weight[key] += 1
        
    for (u, v), w in weight.items():
        G.add_edge(u, v, weight=float(w))
    return G


def graph_to_transition_matrix(G: nx.Graph, topk: Optional[int] = None) -> sp.csr_matrix:
    n = G.number_of_nodes()
    rows, cols, data = [], [], []
    
    for u in G.nodes():
        nbrs = list(G[u].items())
        if topk is not None and len(nbrs) > topk:
            nbrs = sorted(nbrs, key=lambda x: x[1].get("weight", 1.0), reverse=True)[:topk]
        
        for v, attr in nbrs:
            rows.append(u)
            cols.append(v)
            data.append(attr.get("weight", 1.0))
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    
    # Row normalize
    row_sums = np.array(A.sum(axis=1)).reshape(-1)
    row_sums[row_sums == 0.0] = 1.0
    invD = sp.diags(1.0 / row_sums)
    T = invD @ A
    return T


def compute_text_embeddings(entities: List[str], 
                           model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    if not SBERT_AVAILABLE:
        raise ImportError("No sentence-transformers")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(entities, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def diffusion_smooth(Z: np.ndarray, T: sp.csr_matrix, 
                    alpha: float = 0.15, K: int = 3, lam: float = 0.3) -> np.ndarray:
    n, d = Z.shape
    accum = alpha * Z.copy()  # k=0 항
    curr = Z.copy()
    
    for k in range(1, K + 1):
        curr = T @ curr  # T^k Z
        accum += (alpha * ((1 - alpha) ** k)) * curr
    
    Z_smoothed = (1 - lam) * Z + lam * accum
    return Z_smoothed


def compute_structural_embeddings(G: nx.Graph, embedding_dim: int = 64) -> np.ndarray:
    n_nodes = G.number_of_nodes()
    if n_nodes <= 1:
        return np.zeros((n_nodes, embedding_dim), dtype=np.float32)
    
    spectral_emb = _compute_spectral_embedding(G, n_nodes, embedding_dim // 2)
    struct_stats = _compute_structural_stats(G, n_nodes)
    combined = np.hstack([spectral_emb, struct_stats]).astype(np.float32)
    combined = normalize(combined, norm="l2")
    return combined


def _compute_spectral_embedding(G: nx.Graph, n_nodes: int, dim: int) -> np.ndarray:
    if n_nodes <= 1:
        return np.zeros((n_nodes, dim), dtype=np.float32)

    rows, cols, data = [], [], []
    for u, v, attr in G.edges(data=True):
        w = float(attr.get("weight", 1.0))
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([w, w])
    
    if not rows:
        return np.zeros((n_nodes, dim), dtype=np.float32)
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)

    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    Dmh = sp.diags(1.0 / np.sqrt(deg))
    L = Dmh @ A @ Dmh
    
    k = max(1, min(dim, n_nodes - 1))
    
    try:
        vals, vecs = eigsh(L, k=k, which="LM")
        X = vecs.astype(np.float32)
    except:
        try:
            Ld = L.toarray()
            vals, vecs = np.linalg.eigh(Ld)
            X = vecs[:, -k:].astype(np.float32) if k > 0 else np.zeros((n_nodes, dim), np.float32)
        except:
            X = np.zeros((n_nodes, k), dtype=np.float32)
    
    if X.shape[1] < dim:
        pad = np.zeros((n_nodes, dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > dim:
        X = X[:, :dim]
    
    return normalize(X, norm="l2")


def _compute_structural_stats(G: nx.Graph, n_nodes: int) -> np.ndarray:
    if n_nodes == 0:
        return np.zeros((0, 6), dtype=np.float32)
    
    feats = np.zeros((n_nodes, 6), dtype=np.float32)
    
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, weight="weight")
    except:
        pr = {u: 0.0 for u in G.nodes()}
    
    try:
        core = nx.core_number(G)
    except:
        core = {u: 0 for u in G.nodes()}
    
    try:
        clust = nx.clustering(G, weight="weight")
    except:
        clust = {u: 0.0 for u in G.nodes()}
    
    try:
        andeg = nx.average_neighbor_degree(G, weight="weight")
    except:
        andeg = {u: 0.0 for u in G.nodes()}
    
    for u in G.nodes():
        feats[u, 0] = float(deg.get(u, 0))
        feats[u, 1] = float(wdeg.get(u, 0.0))
        feats[u, 2] = float(pr.get(u, 0.0))
        feats[u, 3] = float(core.get(u, 0))
        feats[u, 4] = float(clust.get(u, 0.0))
        feats[u, 5] = float(andeg.get(u, 0.0))
    
    feats = StandardScaler().fit_transform(feats)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize(feats, norm="l2")


def fuse_embeddings(text_emb: np.ndarray, struct_emb: np.ndarray, 
                   w_text: float = 0.6, w_struct: float = 0.4) -> np.ndarray:
    def _standardize(X):
        if X is None or X.size == 0:
            return None
        X = StandardScaler().fit_transform(X)
        return normalize(X, norm="l2")
    
    text_norm = _standardize(text_emb)
    struct_norm = _standardize(struct_emb)
    
    if struct_norm is None:
        return text_norm
    
    if text_norm.shape[1] != struct_norm.shape[1]:
        fused = np.hstack([w_text * text_norm, w_struct * struct_norm])
    else:
        fused = w_text * text_norm + w_struct * struct_norm
    
    return normalize(fused, norm="l2")


def run_hdbscan(embeddings: np.ndarray, min_cluster_size: Optional[int] = None,
                cluster_selection_epsilon: float = 0.0, min_samples: Optional[int] = None):
    N = embeddings.shape[0]
    if min_cluster_size is None:
        min_cluster_size = max(3, int(np.log(max(N, 2))))
    
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(embeddings)
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    
    return clusterer, labels, probs


def build_entity_dictionary(entities: List[str], labels: np.ndarray, 
                           probs: np.ndarray, embeddings: np.ndarray,
                           prob_thresh: float = 0.2, include_noise: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    assignments = {}
    cluster_members = defaultdict(list)
    noise_members = []
    
    for i, (lab, p) in enumerate(zip(labels, probs)):
        if lab == -1 or p < prob_thresh:
            assignments[entities[i]] = -1
            if include_noise:
                noise_members.append(i)
        else:
            assignments[entities[i]] = int(lab)
            cluster_members[int(lab)].append(i)
    
    rows = []
    
    for cid, idxs in sorted(cluster_members.items()):
        aliases = [entities[i] for i in idxs]
        canonical = _pick_canonical_entity(entities, embeddings, idxs)
        
        rows.append({
            "cluster_id": cid,
            "canonical_label": canonical,
            "aliases": aliases,
            "size": len(idxs)
        })
    
    if include_noise and noise_members:
        noise_aliases = [entities[i] for i in noise_members]
        noise_canonical = min(noise_aliases, key=len)
        
        rows.append({
            "cluster_id": -1,
            "canonical_label": noise_canonical,
            "aliases": noise_aliases,
            "size": len(noise_aliases)
        })
    
    entity_dict_df = pd.DataFrame(rows).sort_values(
        ["size", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)
    
    return entity_dict_df, assignments


def _pick_canonical_entity(entities: List[str], embeddings: np.ndarray, 
                          member_indices: List[int]) -> str:
    if not member_indices:
        return ""
    
    sub_embeddings = embeddings[member_indices]
    centroid = sub_embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(sub_embeddings, centroid).ravel()
    sorted_pairs = sorted(
        [(i, d, len(entities[i])) for i, d in zip(member_indices, distances)],
        key=lambda x: (x[1], x[2])
    )
    
    return entities[sorted_pairs[0][0]]


def build_entity_lexicon(df: pd.DataFrame, 
                        parsed_col: str = "parsed_triplets",
                        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                        diffusion_alpha: float = 0.15,
                        diffusion_K: int = 3,
                        diffusion_lambda: float = 0.3,
                        graph_topk: Optional[int] = None,
                        struct_embedding_dim: int = 64,
                        w_text: float = 0.6,
                        w_struct: float = 0.4,
                        min_cluster_size: Optional[int] = None,
                        cluster_selection_epsilon: float = 0.0,
                        min_samples: Optional[int] = None,
                        prob_thresh: float = 0.2,
                        include_noise: bool = True):
    # 1. triples
    triples = collect_triples_from_df(df, col=parsed_col)
    if len(triples) == 0:
        raise ValueError(f"'{parsed_col}' error")
    
    # 2. index
    entities, relations, ent2id, rel2id = build_entity_relation_index(triples)
    
    # 3. graph
    G = build_undirected_graph(triples, ent2id)
    T = graph_to_transition_matrix(G, topk=graph_topk)
    
    # 4. text embedding
    text_embeddings = compute_text_embeddings(entities, model_name=text_model)
    
    # 5. diffused embedding
    diffused_embeddings = diffusion_smooth(
        text_embeddings, T, alpha=diffusion_alpha, K=diffusion_K, lam=diffusion_lambda
    )
    
    # 6. struct embedding
    struct_embeddings = compute_structural_embeddings(G, embedding_dim=struct_embedding_dim)
    
    # 7. Fuse embedding 
    fused_embeddings = fuse_embeddings(
        diffused_embeddings, struct_embeddings, w_text=w_text, w_struct=w_struct
    )
    
    # 8. clustering
    clusterer, labels, probs = run_hdbscan(
        fused_embeddings,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples
    )
    
    # 9. entity dictionary
    entity_dict_df, assignments = build_entity_dictionary(
        entities, labels, probs, fused_embeddings, 
        prob_thresh=prob_thresh, include_noise=include_noise
    )
    
    # result
    artifacts = {
        'entities': entities,
        'relations': relations,
        'ent2id': ent2id,
        'rel2id': rel2id,
        'graph': G,
        'transition_matrix': T,
        'text_embeddings': text_embeddings,
        'diffused_embeddings': diffused_embeddings,
        'structural_embeddings': struct_embeddings,
        'fused_embeddings': fused_embeddings,
        'clusterer': clusterer,
        'labels': labels,
        'probabilities': probs
    }
    
    return entity_dict_df, assignments, artifacts


def collect_triples_from_df(df: pd.DataFrame, col: str = "parsed_triplets") -> List[Tuple[str, str, str]]:
    triples = []
    for row in df[col].dropna():
        for tri in row:
            h = tri.get("subj", "")
            r = tri.get("edge", "")
            t = tri.get("obj", "")
            if h and r and t:
                triples.append((h, r, t))
    return triples

def build_relation_index(triples: List[Tuple[str, str, str]]):
    relations = sorted(list({r for _, r, _ in triples}))
    rel2id = {r: i for i, r in enumerate(relations)}
    return relations, rel2id

def pick_cluster_canonical(names: List[str], X: np.ndarray, member_indices: List[int]) -> str:
    if not member_indices:
        return ""
    sub = X[member_indices]
    centroid = sub.mean(axis=0, keepdims=True)
    dists = cosine_distances(sub, centroid).ravel()
    order = sorted([(i, d, len(names[i])) for i, d in zip(member_indices, dists)], key=lambda x: (x[1], x[2]))
    return names[order[0][0]]


def filter_relations_by_patterns(relations: List[str], 
                                excluded_patterns: Optional[List[str]] = None) -> Tuple[List[str], List[str], Dict[str, bool]]:
    if not excluded_patterns:
        return relations, [], {r: False for r in relations}
    
    excluded_relations = []
    filtered_relations = []
    exclusion_map = {}
    
    for relation in relations:
        is_excluded = False
        for pattern in excluded_patterns:
            if pattern.lower() in relation.lower():
                is_excluded = True
                break
        
        if is_excluded:
            excluded_relations.append(relation)
            exclusion_map[relation] = True
        else:
            filtered_relations.append(relation)
            exclusion_map[relation] = False
    
    return filtered_relations, excluded_relations, exclusion_map

def build_filtered_relation_index(triples: List[Tuple[str, str, str]], 
                                 filtered_relations: List[str]) -> Tuple[List[str], Dict[str, int]]:
    filtered_set = set(filtered_relations)
    rel2id = {r: i for i, r in enumerate(filtered_relations)}
    return filtered_relations, rel2id

def compute_relation_text_embeddings(relations: List[str],
                                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                     batch_size: int = 64) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers No")
    model = SentenceTransformer(model_name)
    Z = model.encode(relations, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(Z, dtype=np.float32)



def _compact_type_index(entity_assignments: Dict[str, int]) -> Dict[int, int]:
    pos_types = sorted({cid for cid in entity_assignments.values() if cid is not None and cid >= 0})
    type2local = {cid: i for i, cid in enumerate(pos_types)}
    type2local[-1] = len(type2local)  # UNK
    return type2local

def build_selectional_signatures(triples: List[Tuple[str, str, str]],
                                 rel2id: Dict[str, int],
                                 entity_assignments: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    type2local = _compact_type_index(entity_assignments)
    local2type = {v: k for k, v in type2local.items()}
    R = len(rel2id); T = len(type2local)

    H = np.zeros((R, T), dtype=np.float32)
    Tm = np.zeros((R, T), dtype=np.float32)

    for h, r, t in triples:
        if r not in rel2id: 
            continue
        rid = rel2id[r]
        h_c = entity_assignments.get(h, -1)
        t_c = entity_assignments.get(t, -1)
        h_loc = type2local.get(h_c, type2local[-1])
        t_loc = type2local.get(t_c, type2local[-1])
        H[rid, h_loc] += 1.0
        Tm[rid, t_loc] += 1.0

    # 정규화(L1) + NaN 방지
    H_sum = H.sum(axis=1, keepdims=True); H_sum[H_sum == 0.0] = 1.0
    T_sum = Tm.sum(axis=1, keepdims=True); T_sum[T_sum == 0.0] = 1.0
    H = H / H_sum
    Tm = Tm / T_sum
    return H, Tm, type2local, local2type

def spectral_relation_embedding(H: np.ndarray,
                                Tm: np.ndarray,
                                dim: int = 16) -> np.ndarray:

    R = H.shape[0]
    if R == 0:
        return np.zeros((0, dim), dtype=np.float32)

    Hn = normalize(H, norm="l2")
    Tn = normalize(Tm, norm="l2")
    S = Hn @ Hn.T + Tn @ Tn.T 


    np.fill_diagonal(S, 1.0)
    try:
        vals, vecs = np.linalg.eigh(S)
        X = vecs[:, -min(dim, R-1):] if R > 1 else np.zeros((R, 1))
    except Exception:
        X = np.random.randn(R, min(dim, max(1, R-1))).astype(np.float32)

    X = normalize(np.asarray(X, dtype=np.float32), norm="l2")
    if X.shape[1] < dim:
        pad = np.zeros((R, dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > dim:
        X = X[:, :dim]
    return X

# -------------------------------
# 4) embedding & HDBSCAN
# -------------------------------

def fuse_relation_embeddings(Z_text: np.ndarray,
                             H: np.ndarray,
                             Tm: np.ndarray,
                             Z_struct: Optional[np.ndarray] = None,
                             w_text: float = 0.5,
                             w_sig: float = 0.35,
                             w_struct: float = 0.15,
                             sig_svd_dim: int = 64) -> np.ndarray:
    R = Z_text.shape[0]

    Sig = np.hstack([H, Tm]).astype(np.float32)  # (R, 2T)

    Sig = StandardScaler(with_mean=True, with_std=True).fit_transform(Sig)
    if Sig.shape[1] > sig_svd_dim:
        svd = TruncatedSVD(n_components=sig_svd_dim, random_state=42)
        Sig_red = svd.fit_transform(Sig)
    else:
        Sig_red = Sig
    Sig_red = normalize(Sig_red, norm="l2")

    Zt = StandardScaler(with_mean=True, with_std=True).fit_transform(Z_text)
    Zt = normalize(Zt, norm="l2")
    if Z_struct is None or Z_struct.size == 0:
        Zs = np.zeros((R, 0), dtype=np.float32)
        w_struct = 0.0
    else:
        Zs = StandardScaler(with_mean=True, with_std=True).fit_transform(Z_struct)
        Zs = normalize(Zs, norm="l2")
    Z = np.hstack([w_text * Zt, w_sig * Sig_red, w_struct * Zs]).astype(np.float32)
    Z = normalize(Z, norm="l2")
    return Z

def run_hdbscan_with_retry(X: np.ndarray,
                           min_cluster_size: Optional[int] = None,
                           min_samples: Optional[int] = None,
                           cluster_selection_epsilon: float = 0.0):

    N = X.shape[0]
    if min_cluster_size is None:
        min_cluster_size = max(2, int(np.log(max(N, 2))))
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)

    def _fit(mcs, ms, cse):
        cl = hdbscan.HDBSCAN(metric="euclidean",
                             min_cluster_size=mcs,
                             min_samples=ms,
                             cluster_selection_epsilon=cse,
                             core_dist_n_jobs=1,
                             prediction_data=True)
        labels = cl.fit_predict(X)
        probs = getattr(cl, "probabilities_", np.ones_like(labels, dtype=float))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return cl, labels, probs, n_clusters

    tries = [
        (min_cluster_size,            min_samples,                  cluster_selection_epsilon),
        (max(2, min_cluster_size//2), max(1, min_samples//2),      cluster_selection_epsilon + 0.05),
        (2,                           1,                            cluster_selection_epsilon + 0.10),
    ]
    last = None
    for mcs, ms, cse in tries:
        cl, labels, probs, k = _fit(mcs, ms, cse)
        last = (cl, labels, probs)
        if k > 0:
            return cl, labels, probs
    return last

def find_inverse_candidates(H: np.ndarray, Tm: np.ndarray, relations: List[str],
                            top_k: int = 3, thresh: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
    Hn, Tn = normalize(H, norm="l2"), normalize(Tm, norm="l2")
    sim_HT = np.clip(Hn @ Tn.T, 0.0, 1.0)  # (R,R)
    sim_TH = sim_HT.T
    inv_score = 0.5 * (sim_HT + sim_TH)   # (R,R)

    out = {}
    R = len(relations)
    for i in range(R):
        scores = inv_score[i]
        idx = np.argsort(-scores)
        cand = []
        for j in idx:
            if j == i:
                continue
            s = float(scores[j])
            if s < thresh:
                break
            cand.append((relations[j], s))
            if len(cand) >= top_k:
                break
        out[relations[i]] = cand
    return out

def build_relation_dictionary(relations: List[str],
                              labels: np.ndarray,
                              probs: np.ndarray,
                              X: np.ndarray,
                              triples: List[Tuple[str, str, str]],
                              rel2id: Dict[str, int],
                              entity_assignments: Dict[str, int],
                              local2type: Dict[int, int],
                              excluded_relations: List[str],  
                              exclusion_map: Dict[str, bool], 
                              entity_dict_df: Optional[pd.DataFrame] = None,
                              prob_thresh: float = 0.2,
                              top_k_sig: int = 5,
                              inverse_map: Optional[Dict[str, List[Tuple[str, float]]]] = None,
                              include_noise: bool = False,
                              include_lowprob_in_noise: bool = True,
                              noise_label: str = "NOISE/UNKNOWN",
                              noise_max_members: int = 200
                              ) -> Tuple[pd.DataFrame, Dict[str, int]]:

    type_label_map = {}
    if entity_dict_df is not None and "cluster_id" in entity_dict_df.columns:
        for _, row in entity_dict_df.iterrows():
            type_label_map[int(row["cluster_id"])] = str(row.get("canonical_label", f"T{int(row['cluster_id'])}"))

    assignments = {}
    cluster_members = defaultdict(list)
    noise_indices = []

    for rel in excluded_relations:
        assignments[rel] = -1

    for i, (lab, p) in enumerate(zip(labels, probs)):
        rel_name = relations[i]
        if lab == -1:
            assignments[rel_name] = -1
            if include_noise:
                noise_indices.append(i)
            continue
        if p < prob_thresh:
            assignments[rel_name] = -1
            if include_noise and include_lowprob_in_noise:
                noise_indices.append(i)
            continue
        assignments[rel_name] = int(lab)
        cluster_members[int(lab)].append(i)

    rel_head_counts = defaultdict(Counter)  
    rel_tail_counts = defaultdict(Counter)
    
    all_relations = list(set([r for _, r, _ in triples]))
    all_rel2id = {r: i for i, r in enumerate(all_relations)}
    
    for h, r, t in triples:
        if r in all_rel2id:
            rid = all_rel2id[r]
            h_c = entity_assignments.get(h, -1)
            t_c = entity_assignments.get(t, -1)
            rel_head_counts[rid][h_c] += 1
            rel_tail_counts[rid][t_c] += 1

    def _top_sig(counter: Counter, k: int):
        total = sum(counter.values()) or 1
        items = counter.most_common(k)
        pretty = []
        for t_id, cnt in items:
            if t_id == -1:
                name = "UNK"
            else:
                name = type_label_map.get(t_id, f"T{t_id}")
            pretty.append({"type_id": int(t_id), "label": name, "count": int(cnt), "ratio": round(cnt/total, 4)})
        return pretty

    rows = []

    for cid, idxs in sorted(cluster_members.items(), key=lambda x: x[0]):
        members = [relations[i] for i in idxs]
        canonical = pick_cluster_canonical(relations, X, idxs)

        head_counter = Counter(); tail_counter = Counter()
        for i in idxs:
            rel_name = relations[i]
            if rel_name in all_rel2id:
                all_rid = all_rel2id[rel_name]
                head_counter.update(rel_head_counts[all_rid])
                tail_counter.update(rel_tail_counts[all_rid])

        domain_signature = _top_sig(head_counter, top_k_sig)
        range_signature  = _top_sig(tail_counter, top_k_sig)

        inv_cands = []
        if inverse_map:
            tmp = Counter()
            for i in idxs:
                for rel_j, score in inverse_map.get(relations[i], []):
                    tmp[(rel_j)] += score
            inv_cands = [{"relation": rel, "score": round(float(s), 4)} for rel, s in tmp.most_common(top_k_sig)]

        rows.append({
            "cluster_id": cid,
            "canonical_predicate": canonical,
            "paraphrases": members,
            "size": len(members),
            "domain_signature": domain_signature,
            "range_signature": range_signature,
            "inverse_candidates": inv_cands
        })

    if include_noise:
        all_noise_members = excluded_relations + [relations[i] for i in noise_indices]
        if len(all_noise_members) > 0:
            canonical = noise_label

            head_counter = Counter(); tail_counter = Counter()
            # 제외된 관계들의 시그니처도 포함
            for rel_name in all_noise_members:
                if rel_name in all_rel2id:
                    all_rid = all_rel2id[rel_name]
                    head_counter.update(rel_head_counts[all_rid])
                    tail_counter.update(rel_tail_counts[all_rid])

            domain_signature = _top_sig(head_counter, top_k_sig)
            range_signature  = _top_sig(tail_counter, top_k_sig)

            inv_cands = []
            if inverse_map:
                tmp = Counter()
                for i in noise_indices:
                    for rel_j, score in inverse_map.get(relations[i], []):
                        tmp[(rel_j)] += score
                inv_cands = [{"relation": rel, "score": round(float(s), 4)} for rel, s in tmp.most_common(top_k_sig)]

            members_out = all_noise_members[:noise_max_members]
            if len(all_noise_members) > noise_max_members:
                members_out = members_out + [f"...(+{len(all_noise_members)-noise_max_members} more)"]

            rows.append({
                "cluster_id": -1,
                "canonical_predicate": canonical,
                "paraphrases": members_out,
                "size": len(all_noise_members),
                "domain_signature": domain_signature,
                "range_signature": range_signature,
                "inverse_candidates": inv_cands
            })

    cols = ["cluster_id", "canonical_predicate", "paraphrases", "size",
            "domain_signature", "range_signature", "inverse_candidates"]
    if not rows:
        return pd.DataFrame(columns=cols), assignments

    rel_dict_df = pd.DataFrame(rows)[cols]
    rel_dict_df = rel_dict_df.sort_values(["size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    return rel_dict_df, assignments


# -------------------------------
# 7) main pipeline
# -------------------------------

def build_relation_lexicon_from_df(df: pd.DataFrame,
                                   entity_assignments: Dict[str, int],
                                   entity_dict_df: Optional[pd.DataFrame] = None,
                                   parsed_col: str = "parsed_triplets",
                                   excluded_patterns: Optional[List[str]] = None,  # 👈 NEW
                                   sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                                   w_text: float = 0.5,
                                   w_sig: float = 0.35,
                                   w_struct: float = 0.15,
                                   sig_svd_dim: int = 64,
                                   struct_dim: int = 16,
                                   min_cluster_size: Optional[int] = None,
                                   min_samples: Optional[int] = None,
                                   cluster_selection_epsilon: float = 0.0,
                                   prob_thresh: float = 0.2,
                                   inverse_top_k: int = 3,
                                   inverse_thresh: float = 0.7):
    # 1) triple
    triples = collect_triples_from_df(df, col=parsed_col)
    if len(triples) == 0:
        raise ValueError("parsed_triplets No.")

    # 2) filtering
    all_relations = sorted(list({r for _, r, _ in triples}))
    filtered_relations, excluded_relations, exclusion_map = filter_relations_by_patterns(
        all_relations, excluded_patterns
    )
    
    if len(filtered_relations) == 0:
        assignments = {r: -1 for r in all_relations}
        empty_df = pd.DataFrame(columns=["cluster_id", "canonical_predicate", "paraphrases", "size",
                                        "domain_signature", "range_signature", "inverse_candidates"])
        return empty_df, assignments, {}

    # 3) index
    filtered_relations, rel2id = build_filtered_relation_index(triples, filtered_relations)

    # 4) selectional 
    H, Tm, type2local, local2type = build_selectional_signatures(triples, rel2id, entity_assignments)

    # 5) text embedding
    Z_text = compute_relation_text_embeddings(filtered_relations, model_name=sbert_model)

    # 6) spectral_relation_embedding
    Z_struct = spectral_relation_embedding(H, Tm, dim=struct_dim)

    # 7) fuse
    Z = fuse_relation_embeddings(Z_text, H, Tm, Z_struct,
                                 w_text=w_text, w_sig=w_sig, w_struct=w_struct,
                                 sig_svd_dim=sig_svd_dim)

    # 8) HDBSCAN 
    clusterer, labels, probs = run_hdbscan_with_retry(X=Z,
                                                      min_cluster_size=min_cluster_size,
                                                      min_samples=min_samples,
                                                      cluster_selection_epsilon=cluster_selection_epsilon)

    # 9) inverse 
    inverse_map = find_inverse_candidates(H, Tm, filtered_relations, top_k=inverse_top_k, thresh=inverse_thresh)

    # 10) relation dictionary
    rel_dict_df, rel_assignments = build_relation_dictionary(
        filtered_relations, labels, probs, Z,
        triples, rel2id,
        entity_assignments, local2type,
        excluded_relations, exclusion_map,
        entity_dict_df=entity_dict_df,
        prob_thresh=prob_thresh,
        top_k_sig=5,
        inverse_map=inverse_map,
        include_noise=True,
        include_lowprob_in_noise=True,
        noise_label="NOISE/UNKNOWN",
        noise_max_members=200
    )

    artifacts = dict(
        all_relations=all_relations,
        filtered_relations=filtered_relations,
        excluded_relations=excluded_relations,
        exclusion_map=exclusion_map,
        rel2id=rel2id,
        H=H, T=Tm,
        Z_text=Z_text,
        Z_struct=Z_struct,
        Z_fused=Z,
        labels=labels, probs=probs,
        inverse_map=inverse_map,
        type2local=type2local, local2type=local2type
    )
    return rel_dict_df, rel_assignments, artifacts


def parse_triplet(triplet_str):
    pattern = r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*)"
    match = re.match(pattern, triplet_str)
    if match:
        subj, obj, edge = match.groups()
        return {"subj": subj.strip(), "obj": obj.strip(), "edge": edge.strip()}
    return None

def extract_triplets(cell_value: str):
    if not isinstance(cell_value, str) or "<triplet>" not in cell_value:
        return []
    parts = cell_value.split("\n")
    results = []
    for p in parts:
        parsed = parse_triplet(p)
        if parsed:
            results.append(parsed)
    return results


if __name__ == "__main__":
    # data_name = "A578A578M_07_triplet"
    # data_name = "A6A6M_14_triplet"
    data_name = "API_2W_triplet"

    df_text = pd.read_csv(f"text_path")
    df_table = pd.read_csv(f"table_path")
    df = pd.concat([df_text, df_table])
    df = df.sort_values("sort_id")
    df.reset_index(drop=True, inplace=True)

    df["parsed_triplets"] = df["triplets"].apply(extract_triplets)
    

    entity_dict, assignments, artifacts = build_entity_lexicon(
        df,
        parsed_col="parsed_triplets",
        text_model="sentence-transformers/all-MiniLM-L6-v2",
        diffusion_alpha=0.15, 
        diffusion_K=3, 
        diffusion_lambda=0.3, 
        struct_embedding_dim=64, 
        w_text=0.6, 
        w_struct=0.4, 
        min_cluster_size=3, 
        cluster_selection_epsilon=0.05,
        prob_thresh=0.2,
        include_noise=True  
    )
    entity_dict.to_csv(f"entity_output_path", index=False)
    print("entity finish")
    

    EXCLUDED_PATTERNS = [
    "has_condition",
    "has_condition_AND",
    "has_condition_OR",
    "has_consequence",
    "has_consequence_AND",
    "has_consequence_OR",
    "has_footnote" ,
    "has_case" ,
    "has_note" ,
    "without",
    "divided"
    ""
    ]
    
    print("Patterns to exclude:", EXCLUDED_PATTERNS)
    

    rel_dict, rel_assign, arts = build_relation_lexicon_from_df(
        df,
        entity_assignments=assignments,
        entity_dict_df=entity_dict,
        parsed_col="parsed_triplets",
        excluded_patterns=EXCLUDED_PATTERNS,  
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        w_text=0.65, w_sig=0.25, w_struct=0.1,
        sig_svd_dim=32, struct_dim=8,
        min_cluster_size=2, min_samples=1,
        cluster_selection_epsilon=0.05,
        prob_thresh=0.1,
        inverse_top_k=3, inverse_thresh=0.5
    )

    excluded_rels = [rel for rel, cid in rel_assign.items() if cid == -1]
    print(f"\ncount- {len(excluded_rels)} : cluster_id = -1")


    rel_dict.to_csv(f"output_path", index=False)
    print("edge finish")
    