# Load model FIRST to avoid macOS Intel segfault
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

import numpy as np
import faiss
from pathlib import Path
import json

OUT_DIR = Path("../data/index_faiss")
EMBED_MODEL = "all-MiniLM-L6-v2"

print("Loading index and metadata...")

index = faiss.read_index(str(OUT_DIR / "faiss_index.bin"))

# load JSON metadata
with open(OUT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

prompt_texts = meta["prompt_texts"]
prompt_tool_idx = meta["prompt_tool_idx"]
prompt_tool_name = meta["prompt_tool_name"]
prompt_source = meta.get("prompt_source", [])
tools = meta.get("tools", [])

# load model
model = SentenceTransformer(EMBED_MODEL)

def recommend_tools(user_prompt, top_k=5, k_prompts=10, agg="max"):
    # embed query
    q = model.encode([user_prompt], normalize_embeddings=True).astype('float32')

    # search top k_prompts example prompts
    D, I = index.search(q, k_prompts)  # D: scores, I: indices
    scores = D[0].tolist()
    indices = I[0].tolist()

    # map prompts → tools
    tool_scores = {}   # tool_index -> list of scores
    tool_best = {}     # tool_index -> (best_score, example_prompt_index)

    for sc, prompt_idx in zip(scores, indices):
        if prompt_idx < 0:
            continue
        tool_idx = prompt_tool_idx[prompt_idx]

        if tool_idx not in tool_scores:
            tool_scores[tool_idx] = []
            tool_best[tool_idx] = (sc, prompt_idx)

        tool_scores[tool_idx].append(sc)

        if sc > tool_best[tool_idx][0]:
            tool_best[tool_idx] = (sc, prompt_idx)

    # aggregate scores
    aggregated = []
    for tool_idx, sc_list in tool_scores.items():
        if agg == "max":
            agg_score = max(sc_list)
        else:
            agg_score = float(sum(sc_list) / len(sc_list))
        best_sc, best_prompt_idx = tool_best[tool_idx]
        aggregated.append((tool_idx, float(agg_score), int(best_prompt_idx)))

    aggregated.sort(key=lambda x: x[1], reverse=True)

    # build results
    results = []
    for tool_idx, score, best_prompt_idx in aggregated[:top_k]:
        tool = tools[tool_idx]  # FIXED: use JSON list, not pandas
        best_prompt = prompt_texts[best_prompt_idx]

        tags = tool.get("tags", "")
        usecase = tool.get("example_use_cases", "")
        rationale = f"Matched keywords and semantics (tags: {tags.split(',')[:3]}); common use case: {usecase}"

        results.append({
            "tool_name": tool.get("tool_name"),
            "url": tool.get("url"),
            "category": tool.get("normalized_category", tool.get("category")),
            "subcategory": tool.get("normalized_subcategory", tool.get("subcategory")),
            "score": round(score, 4),
            "best_example_prompt": best_prompt,
            "rationale": rationale,
        })

    return results


if __name__ == "__main__":
    queries = [
        "Create a 30-second promotional video for a SaaS product highlighting features and CTA",
        "Write 5 social media captions for a new coffee shop launch in a friendly tone",
        "Generate a Python function for mean-variance portfolio optimization with sample data"
    ]

    for q in queries:
        print("="*80)
        print("User prompt:", q)
        recs = recommend_tools(q, top_k=5, k_prompts=12, agg="max")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['tool_name']} (score {r['score']}) - {r['url']}")
            print(f"   Matched prompt: {r['best_example_prompt']}")
            print(f"   Rationale: {r['rationale']}")
            print()
