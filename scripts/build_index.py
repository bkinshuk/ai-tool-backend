# scripts/build_index.py
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import pickle
import os

DATA_CSV = Path("../data/ai_tools_200_with_prompts.csv")  # adjust path if needed
OUT_DIR = Path("../data/index_faiss")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED_MODEL = "all-MiniLM-L6-v2"   # chosen model
EMBED_DIM = 384

print("Loading CSV...")
df = pd.read_csv(DATA_CSV)

# Build list of prompts and mapping to tool index (row index)
prompt_texts = []
prompt_tool_idx = []
prompt_tool_name = []
prompt_source = []  # example_prompt_1/2/3

for i, row in df.iterrows():
    for n in (1, 2, 3):
        col = f"example_prompt_{n}"
        txt = row.get(col, "")
        if pd.isna(txt) or str(txt).strip() == "":
            continue
        prompt_texts.append(str(txt).strip())
        prompt_tool_idx.append(int(i))
        prompt_tool_name.append(row['tool_name'])
        prompt_source.append(col)

print(f"Total example prompts: {len(prompt_texts)}")

# Load embedding model
print("Loading embedding model:", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL)

# Compute embeddings in batches
batch_size = 64
embs = []
for i in tqdm(range(0, len(prompt_texts), batch_size)):
    batch = prompt_texts[i:i+batch_size]
    e = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    embs.append(e)
embeddings = np.vstack(embs).astype('float32')
print("Embeddings shape:", embeddings.shape)

# Build FAISS index (inner product on normalized vectors = cosine similarity)
index = faiss.IndexFlatIP(EMBED_DIM)   # inner product
index.add(embeddings)                  # add vectors
print("FAISS index built. n_vectors:", index.ntotal)

# Save index and metadata (write JSON metadata to avoid pickle/unpickle side effects)
import json

faiss.write_index(index, str(OUT_DIR / "faiss_index.bin"))

# Force everything into plain python built-ins (no numpy/pandas types)
def _to_primitive(x):
    # convert numpy scalars/arrays to plain python
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

meta = {
    "prompt_texts": [str(x) for x in prompt_texts],
    "prompt_tool_idx": [int(x) for x in prompt_tool_idx],
    "prompt_tool_name": [str(x) for x in prompt_tool_name],
    "prompt_source": [str(x) for x in prompt_source],
    # df.to_dict(orient="records") may contain numpy types; convert them
    "tools": [
        {k: _to_primitive(v) for k, v in rec.items()}
        for rec in df.to_dict(orient="records")
    ],
}

# write JSON metadata
with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved FAISS index to", OUT_DIR / "faiss_index.bin")
print("Saved metadata JSON to", OUT_DIR / "metadata.json")

