import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

OUT_DIR = Path("../data/index_faiss")
EMBED_MODEL = "all-MiniLM-L6-v2"

print("STEP 1: before reading index")
index_path = OUT_DIR / "faiss_index.bin"
print("Index path:", index_path)

print("STEP 2: read index")
index = faiss.read_index(str(index_path))
print("STEP 2 DONE: index loaded")

print("STEP 3: load metadata.pkl")
with open(OUT_DIR / "metadata.pkl", "rb") as f:
    meta = pickle.load(f)
print("STEP 3 DONE: metadata loaded")

print("STEP 4: inspect metadata keys:", meta.keys())

print("STEP 5: loading SentenceTransformer")
model = SentenceTransformer(EMBED_MODEL)
print("STEP 5 DONE")

print("STEP 6: testing simple encode()")
emb = model.encode(["hello"], normalize_embeddings=True)
print("STEP 6 DONE: embed OK")

print("STEP 7: final OK marker")
