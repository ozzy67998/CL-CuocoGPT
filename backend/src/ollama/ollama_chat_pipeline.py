#!/usr/bin/env python3
"""
groq_chat_pipeline.py - Ollama HTTP-only client (compatible with Ollama 0.6.x)

This version:
- Does NOT rewrite the user query.
- Uses pipeline.py output as context for the LLM summary.
- Removes emojis for cleaner logs.
"""

import os
import sys
import json
import textwrap
import requests
from typing import Any, Dict
from functools import lru_cache

# -------------------------
# Load .env optionally
# -------------------------
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        print(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path)
except Exception:
    pass

# -------------------------
# Pipeline import
# -------------------------
PIPELINE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../NLP"))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

print("Loading pipeline components...")
try:
    from pipeline import handle_query
    print("Pipeline components loaded successfully")
except Exception as e:
    print(f"Failed to import pipeline: {e}")
    sys.exit(1)

# -------------------------
# Config
# -------------------------
OLLAMA_HOST = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = (os.environ.get("OLLAMA_MODEL") or os.environ.get("ollama_model") or "llama3:8b-instruct-q4_0").strip().strip('"').strip("'")
_TEMP = os.environ.get("OLLAMA_TEMPERATURE") or os.environ.get("ollama_temperature")
_MAX_TOK = os.environ.get("OLLAMA_MAX_TOKENS") or os.environ.get("ollama_max_tokens")

try:
    _TEMP = float(_TEMP) if _TEMP is not None else None
except Exception:
    _TEMP = None
try:
    _MAX_TOK = int(_MAX_TOK) if _MAX_TOK is not None else None
except Exception:
    _MAX_TOK = None

# ---- Vector description similarity (FAISS) ----
_DESC_INDEX_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_index"))
_DESC_META = os.path.join(_DESC_INDEX_DIR, "desc_meta.json")
_DESC_EMB = os.path.join(_DESC_INDEX_DIR, "embeddings.npy")
_DESC_FAISS = os.path.join(_DESC_INDEX_DIR, "faiss.index")

@lru_cache(maxsize=1)
def _load_desc_index():
    try:
        if not (os.path.exists(_DESC_META) and os.path.exists(_DESC_EMB) and os.path.exists(_DESC_FAISS)):
            return None
        import json, numpy as np, faiss
        from sentence_transformers import SentenceTransformer
        with open(_DESC_META, encoding="utf-8") as f:
            meta = json.load(f)
        emb = np.load(_DESC_EMB)
        index = faiss.read_index(_DESC_FAISS)
        model_name = meta.get("meta", {}).get("model") or "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        return {"meta": meta, "emb": emb, "index": index, "model": model}
    except Exception:
        return None

def _similar_description_chunks(query: str, top_k: int = 5):
    data = _load_desc_index()
    if not data:
        return []
    import numpy as np
    model = data["model"]
    index = data["index"]
    meta = data["meta"]
    records = meta.get("records", [])
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, min(top_k, len(records)))
    out = []
    for score, idx in zip(D[0], I[0]):
        if int(idx) < len(records):
            r = records[int(idx)]
            out.append({
                "id": r["id"],
                "name": r["name"],
                "chunk_id": r["chunk_id"],
                "text": r["text"].strip(),
                "score": float(score)
            })
    return out

# -------------------------
# Ollama helpers
# -------------------------
def list_models() -> list:
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "models" in data:
                return [m.get("name") for m in data["models"] if isinstance(m, dict) and "name" in m]
            return []
    except Exception as e:
        print(f"Error listing models: {e}")
    return []

def ensure_model_available(desired_model: str) -> str:
    models = list_models()
    if not models:
        print("No models reported by Ollama server; using requested model anyway")
        return desired_model
    print(f"Ollama server reports models: {models}")
    if desired_model in models:
        return desired_model
    print(f"Requested model '{desired_model}' not available; using '{models[0]}' instead")
    return models[0]

def api_generate(prompt: str, model_name: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    options = {}
    if _TEMP is not None:
        options["temperature"] = _TEMP
    if _MAX_TOK is not None:
        options["num_predict"] = _MAX_TOK
    if options:
        payload["options"] = options

    print(f"POST {url} model={model_name} payload_preview={json.dumps(payload)[:1000]}")
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            txt = data.get("response") or data.get("text") or ""
            if isinstance(txt, list):
                txt = "".join(txt)
            return (txt or "").strip()
        else:
            print(f"Generate error {r.status_code}: {r.text[:400]}")
    except Exception as e:
        print(f"Generate exception: {e}")
    return ""

_current_model = ensure_model_available(DEFAULT_MODEL)

def run_ollama(prompt: str) -> str:
    if not _current_model:
        print("No model selected; returning prompt")
        return prompt
    return api_generate(prompt, _current_model)

# -------------------------
# Prepare brief for LLM summary
# -------------------------
def _brief_result_for_llm(result: Dict[str, Any]) -> str:
    """
    Produce a fully explicit structured summary of the pipeline output.
    This will be used as context for the LLM, avoiding hallucinations.
    """
    lines = []

    intent = result.get("intent")
    if intent:
        lines.append(f"intent={intent}")

    slots = result.get("slots") or {}
    recipe_names = slots.get("recipe_name") or []
    if recipe_names and isinstance(recipe_names[0], (list, tuple)):
        lines.append(f"slot_recipe_name={recipe_names[0][0]}")

    cooking_time = slots.get("cooking_time")
    if isinstance(cooking_time, int):
        lines.append(f"slot_cooking_time={cooking_time}m")

    recipes = result.get("kg_results") or []
    if recipes:
        for r in recipes:
            name = r.get("recipe_name") or r.get("recipe_uri")
            minutes = r.get("minutes")
            ingredients = r.get("ingredients") or []
            steps = r.get("steps") or []
            tags = r.get("tags") or []

            lines.append(f"Recipe: {name}")
            if minutes:
                lines.append(f"Minutes: {minutes}")
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
            if ingredients:
                lines.append("Ingredients: " + ", ".join(ingredients))
            if steps:
                lines.append("Steps:")
                for i, s in enumerate(steps, start=1):
                    lines.append(f"{i:02d}. {s}")

    return "\n".join(lines)

def _summarize_with_llm(user_query: str, result: Dict[str, Any]) -> str:
    brief = _brief_result_for_llm(result)
    prompt = (
        "You are a bilingual PT/EN assistant.\n"
        "ONLY use the following structured recipe output and the user query. "
        "Do NOT invent anything. Do NOT add URLs, extra steps, or external references. "
        "Produce a concise summary in 2-5 lines, in natural language.\n\n"
        f"User query: {user_query}\n\n"
        f"{brief}\n\n"
        "Answer:"
    )
    desc_chunks = result.get("similar_description_chunks") or []
    if desc_chunks:
        desc_lines = "\n".join(f"- {c['text']}" for c in desc_chunks)
        # Prepend instruction block; LLM should start output with these lines integrated.
        prepend = (
            "IMPORTANT: Start your answer by naturally incorporating the following recipe description snippets "
            "(do NOT alter their facts; you may lightly merge phrasing):\n"
            f"{desc_lines}\n\n"
        )
        prompt = prepend + prompt  # prompt variable assumed existing in original code
    return run_ollama(prompt)

# -------------------------
# Main pipeline flow
# -------------------------
def _process_query(user_query: str):
    user_query = user_query.strip()
    if not user_query:
        return
    print(f"\nProcessing query: '{user_query}'")
    # Skip LLM rewrite entirely; just use query directly
    rewritten = user_query
    handle, extract_slots = _load_pipeline()
    result = handle(user_query, intent_top_k=1, sparql_top_k=5)
    # Add top-5 similar description chunks from vector index
    result["similar_description_chunks"] = _similar_description_chunks(user_query, top_k=5)
    print("\nPipeline result summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    summary = _summarize_with_llm(user_query, result)
    if summary:
        print("\nLLM summary:")
        print(summary)
    print("\nQuery processing complete!")

# -------------------------
# CLI / interactive
# -------------------------
def main():
    print("Recipe Knowledge Graph Query System")
    print("=" * 50)
    if len(sys.argv) > 1:
        _process_query(" ".join(sys.argv[1:]))
        return
    if not sys.stdin.isatty():
        print("No query provided.")
        print("Usage: python ollama_chat_pipeline.py \"query\"")
        sys.exit(1)
    print("Interactive mode (Ctrl-D to exit).")
    try:
        while True:
            q = input("query> ").strip()
            if not q:
                break
            _process_query(q)
    except (KeyboardInterrupt, EOFError):
        pass
    print("Goodbye!")

if __name__ == "__main__":
    main()
