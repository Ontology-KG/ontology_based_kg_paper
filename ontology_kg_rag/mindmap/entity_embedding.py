import pandas as pd
import json
import pickle
from sentence_transformers import SentenceTransformer


def split_title_chain(title: str):
    if not title:
        return []
    parts = [p.strip() for p in str(title).split("->")]
    return [p for p in parts if p]

entities = set()

input_path = "path/to/your_kg_file.jsonl"  

if input_path.endswith(".csv"):
    df = pd.read_csv(input_path)
    rows = df.to_dict("records")
elif input_path.endswith(".jsonl"):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
else:
    raise ValueError("Unsupported file format. Please use CSV or JSONL.")

for row in rows:
    titles = split_title_chain(row.get("title", ""))
    for t in titles:
        entities.add(t)
    for edge in row.get("kg_edges", []):
        entities.add(edge.get("subj", ""))
        entities.add(edge.get("obj", ""))

entities = list(set([e.strip() for e in entities if isinstance(e, str) and e.strip()]))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    entities,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True
)

entity_emb_dict = {
    "entities": entities,
    "embeddings": embeddings
}

output_path = "path/to/output/entity_embeddings.pkl"

with open(output_path, "wb") as f:
    pickle.dump(entity_emb_dict, f)


