# debug_recommend.py
import sys, os, platform, glob, traceback

print("=== Basic info ===")
print("Python:", sys.version.replace("\n"," "))
print("Platform:", platform.platform())
print("cwd:", os.getcwd())

def safe_run(fn, name):
    try:
        print(f"\n-- TRY: {name}")
        res = fn()
        print(f"   OK: {name}")
        return res
    except Exception as e:
        print(f"   EXCEPTION in {name}: {e!r}")
        traceback.print_exc()
        return None

# 1) Imports
safe_run(lambda: __import__("numpy"), "import numpy")
safe_run(lambda: __import__("pandas"), "import pandas")
safe_run(lambda: __import__("faiss"), "import faiss")
safe_run(lambda: __import__("torch"), "import torch")
safe_run(lambda: __import__("transformers"), "import transformers")
safe_run(lambda: __import__("sentence_transformers"), "import sentence_transformers")
safe_run(lambda: __import__("tokenizers"), "import tokenizers")

# 2) List candidate index / metadata files in cwd (helps locate file-related crashes)
print("\n=== Files in cwd matching common index/metadata patterns ===")
for pat in ("*.faiss", "*.index", "*.idx", "*.npy", "*.npz", "*.pkl", "*.json", "*.csv"):
    matches = glob.glob(pat)
    if matches:
        print(pat, "->", matches)

# 3) Try loading any faiss index file found
faiss_files = glob.glob("*.faiss") + glob.glob("*.index") + glob.glob("*.idx")
if faiss_files:
    def load_faiss():
        import faiss
        for f in faiss_files:
            print("Attempting faiss.read_index on", f)
            idx = faiss.read_index(f)
            print(" index d:", idx.d if hasattr(idx,'d') else "unknown")
            # try a tiny search (will only run if index loads)
            try:
                import numpy as np
                q = np.zeros((1, idx.d), dtype="float32")
                idx.search(q, 1)
                print("  faiss search OK")
            except Exception as e:
                print("  faiss search exception:", e)
    safe_run(load_faiss, "faiss.read_index on found files")
else:
    print("No faiss index files found in cwd.")

# 4) Try loading numpy .npy/.npz files if present (could cause errors if corrupted)
npy_files = glob.glob("*.npy") + glob.glob("*.npz")
if npy_files:
    def load_npy():
        import numpy as np
        for f in npy_files:
            print("Loading", f)
            a = np.load(f, allow_pickle=True)
            print(" shape:", getattr(a, "shape", "scalar or object"))
    safe_run(load_npy, "load .npy/.npz files")
else:
    print("No .npy/.npz files found in cwd.")

# 5) Try SentenceTransformer model creation and a tiny encode
def test_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    print("About to instantiate SentenceTransformer('all-MiniLM-L6-v2') -- this may download weights if not cached.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model instantiated. Try tiny encode")
    emb = model.encode(["hello"], convert_to_numpy=True)
    print("emb shape:", getattr(emb, "shape", "unknown"))
safe_run(test_sentence_transformer, "SentenceTransformer instantiate + encode")

print("\nIf this script segfaults at any point, note the last printed line above.")
print("If it prints everything OK, the crash is likely in the original script's custom logic after model/index load.")
