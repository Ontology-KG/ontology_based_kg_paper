import pandas as pd
import numpy as np
from numpy.linalg import norm
from sentence_transformers import models, SentenceTransformer
import os
import torch
import pickle
import pickle
import numpy as np
import torch

EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL, device="cuda")

def embed_long_document_st(text: str, model, max_length=512, stride=256):
    # chunking
    tokens = model.tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )
    embs = []
    for i in range(len(tokens["input_ids"])):
        chunk_text = model.tokenizer.decode(tokens["input_ids"][i], skip_special_tokens=True)
        emb = model.encode(chunk_text, convert_to_numpy=False, normalize_embeddings=True)
        embs.append(emb)
    return torch.stack(embs).mean(0) 


def ensure_unit_norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    n = np.linalg.norm(vec)
    return vec if n < eps else (vec / n)

def build_title_embeddings(df: pd.DataFrame,
                           title_col: str = "title",
                           text_col: str = "text") -> pd.DataFrame:

    merged = df.groupby(title_col)[text_col].apply(lambda s: " ".join(s.astype(str).tolist())).reset_index()
    merged.rename(columns={text_col: "merged_text"}, inplace=True)

    embeddings = []
    for _, row in merged.iterrows():
        vec_t = embed_long_document_st(row["merged_text"], model)
        vec = vec_t.detach().cpu().numpy().astype(np.float32)
        vec = ensure_unit_norm(vec)  
        embeddings.append(vec)

    merged["embedding"] = embeddings
    return merged  # columns: title, merged_text, embedding


def save_index(merged_df: pd.DataFrame, pkl_path):
    records = []
    for _, row in merged_df.iterrows():
        records.append({
            "title": row["title"],
            "merged_text": row.get("merged_text", ""),
            "embedding": row["embedding"],
        })

    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)

    print(f"[saved] {pkl_path} (rows={len(records)})")

### 
if __name__ == "__main__":
    doc_list = ["A6A6M_14", "API_2W"]
    for doc_name in doc_list:
        a578 = pd.read_excel("file_path")
        merged = build_title_embeddings(a578, title_col="title", text_col="text")

        pkl_path = f"output_path"
        save_index(merged, pkl_path)