import os, csv, json, re
from typing import List, Dict, Any
from functools import lru_cache

DATA_CSV = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data/curated/filtered_pt_en.csv"))
INDEX_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data/vector_index"))
os.makedirs(INDEX_DIR, exist_ok=True)

META_PATH = os.path.join(INDEX_DIR, "desc_meta.json")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

MODEL_NAME = os.environ.get("DESC_EMBED_MODEL", "intfloat/multilingual-e5-base")
_USE_E5 = "e5" in MODEL_NAME.lower()

_BOILERPLATE_PATTERNS = [
    r'\bsubmitted to\s+"?zaar"?\b',
    r'\bposted\b',
    r'\bphoto\b',
    r'\byield\b',
    r'\bservings?\b',
    r'\bprep(?:aration)? time\b',
]
_BP_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.I)

def _is_low_info(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return True
    alpha = sum(ch.isalpha() for ch in t)
    if alpha / max(1, len(t)) < 0.45:
        return True
    if _BP_RE.search(t):
        return True
    if len(re.findall(r'\w+', t)) < 6:
        return True
    return False

def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    out = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = (row.get("id") or "").strip()
            name = (row.get("name") or "").strip()
            desc = (row.get("description") or "").strip()
            if rid and name and desc:
                out.append({"id": rid, "name": name, "description": desc})
    return out

# CHANGED: one chunk per description (full description) if not low-info
def _chunk(desc: str) -> List[str]:
    if _is_low_info(desc):
        return []
    return [desc.strip()]

def build_index(csv_path: str = DATA_CSV):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np, faiss
    except Exception:
        print("Install: pip install sentence-transformers faiss-cpu")
        return
    rows = _read_rows(csv_path)
    if not rows:
        print("No rows; abort.")
        return

    chunks = []
    for r in rows:
        for i, ch in enumerate(_chunk(r["description"])):
            chunks.append({"id": r["id"], "name": r["name"], "chunk_id": f"{r['id']}::c{i}", "text": ch})
    if not chunks:
        print("No valid chunks after filtering; abort.")
        return

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(MODEL_NAME)
    enc_inp = [f"passage: {t}" if _USE_E5 else t for t in texts]
    emb = model.encode(enc_inp, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    import numpy as np
    np.save(EMB_PATH, emb.astype("float32"))
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index, FAISS_PATH)

    meta = {
        "model": MODEL_NAME,
        "use_e5": _USE_E5,
        "source_csv": os.path.abspath(csv_path),
        "n_chunks": len(chunks)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"records": chunks, "meta": meta}, f, ensure_ascii=False)
    print("Index built:", meta)

@lru_cache(maxsize=1)
def _load_meta():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def _load_index():
    meta = _load_meta()
    if not meta:
        return None
    try:
        import numpy as np, faiss
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    emb = np.load(EMB_PATH)
    if not os.path.exists(FAISS_PATH):
        return None
    index = faiss.read_index(FAISS_PATH)
    model_name = (meta.get("meta") or {}).get("model") or MODEL_NAME
    use_e5 = (meta.get("meta") or {}).get("use_e5")
    model = SentenceTransformer(model_name)
    return {"emb": emb, "index": index, "model": model, "records": meta["records"], "use_e5": use_e5}

def similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    data = _load_index()
    if not data:
        return []
    model = data["model"]
    index = data["index"]
    records = data["records"]
    use_e5 = bool(data.get("use_e5"))

    q_text = f"query: {query}" if use_e5 else query
    qv = model.encode([q_text], normalize_embeddings=True).astype("float32")

    # Fetch more than needed to allow dedup by chunk_id, then trim to top_k
    search_k = min(max(top_k * 5, top_k), len(records))
    D, I = index.search(qv, search_k)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    MIN_SCORE = 0.4  # adjust as needed
    top_score = float(D[0][0]) if len(D[0]) else -1.0
    for score, idx in zip(D[0], I[0]):
        i = int(idx)
        if i < 0 or i >= len(records):
            continue
        rec = records[i]
        cid = rec.get("chunk_id")
        if cid in seen:
            continue
        if score < MIN_SCORE:
            continue
        seen.add(cid)
        out.append({
            "id": rec["id"],
            "name": rec["name"],
            "chunk_id": rec["chunk_id"],
            "text": rec["text"].strip(),
            "score": float(score),
        })
        if len(out) >= top_k:
            break
    return out

def best_description_for_names(names: List[str], top_per_name: int = 1) -> Dict[str, List[str]]:
    meta = _load_meta()
    if not meta:
        return {}
    records = meta["records"]
    by = {}
    for r in records:
        by.setdefault(r["name"].lower(), []).append(r["text"])
    out = {}
    for n in names:
        out[n] = by.get(n.lower(), [])[:top_per_name]
    return out

def _augment_with_descriptions(result: Dict[str, Any], user_query: str):
    if not similarity_search:
        return
    slots = result.get("slots") or {}
    recipe_names = slots.get("recipe_name") or []
    flat = []
    for item in recipe_names:
        if isinstance(item, (list, tuple)):
            flat.append(str(item[0]))
        else:
            flat.append(str(item))
    flat = [x for x in flat if x]
    result["descriptions_by_name"] = best_description_for_names(flat, 1) if best_description_for_names else {}
    # Keep only top-3 (dedup happens inside similarity_search)
    result["similar_description_chunks"] = similarity_search(user_query, top_k=3)

if __name__ == "__main__":
    build_index()