import os
import ast
import fasttext
import pandas as pd
import re
from tqdm import tqdm

# Ensure curated directory exists
os.makedirs("data/curated", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/RAW_recipes.csv")

# Safely parse lists from text columns (if exist)
if "tags" in df.columns:
    df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])
else:
    df["tags"] = [[]] * len(df)

if "steps" in df.columns:
    df["steps"] = df["steps"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])
else:
    df["steps"] = [[]] * len(df)

# Load language identification model
LANG_MODEL = fasttext.load_model("lid.176.bin")

COUNTRY_KEYWORDS = [
    # General
    "portuguese", "portugal", "azores", "madeira", "lisbon", "porto",
    "coimbra", "algarve", "braga", "sintra",

    # Foods and ingredients
    "bacalhau", "sardine", "pastel de nata", "nata", "codfish", "chouriço",
    "linguiça", "feijoada", "caldo verde", "piri-piri", "alheira",
    "azeite", "vinho verde", "port wine", "bolinho", "travesseiro",
    "arroz doce", "bifana", "francesinha", "cataplana",

    # Cultural/holiday references
    "fado", "saudade", "português", "lusitan", "carnaval", "lisboa"
]

# Probability threshold to accept fastText language prediction
FT_PROB_THRESHOLD = 0.60
MIN_LANG_DETECT_LEN = 3

# --- Caches for repeated checks ---
ft_cache = {}          # text → (is_pt, prob, diff)
word_cache = {}        # single word → bool
keyword_cache = {}     # text → keyword or None


def ft_prediction(text: str):
    """Return list of (label, prob) for top 2 predictions."""
    if not isinstance(text, str):
        return []
    txt = text.replace("\n", " ").strip()
    if len(txt) < 4:
        return []
    try:
        labels, probs = LANG_MODEL.predict(txt[:400], k=2)
        return list(zip(labels, probs))
    except Exception:
        return []


def ft_is_portuguese_confident(text: str, min_prob: float = 0.7, diff_margin: float = 0.15):
    """Cached confident Portuguese detection."""
    if text in ft_cache:
        return ft_cache[text]

    preds = ft_prediction(text)
    if not preds:
        ft_cache[text] = (False, 0.0, 0.0)
        return ft_cache[text]

    top_label, top_prob = preds[0]
    if top_label != "__label__pt":
        ft_cache[text] = (False, top_prob, 0.0)
        return ft_cache[text]

    # Compare with 2nd best label
    diff = top_prob
    if len(preds) > 1:
        _, second_prob = preds[1]
        diff = top_prob - second_prob
        if diff < diff_margin:
            ft_cache[text] = (False, top_prob, diff)
            return ft_cache[text]

    result = (top_prob >= min_prob, top_prob, diff)
    ft_cache[text] = result
    return result


def is_word_portuguese(word: str, min_prob: float = 0.8) -> bool:
    """Detect if a single word is confidently Portuguese (cached)."""
    if word in word_cache:
        return word_cache[word]
    if not isinstance(word, str):
        word_cache[word] = False
        return False
    if not word.isalpha():  # skip digits or symbols like '24mins'
        word_cache[word] = False
        return False
    if any(x in word.lower() for x in ["mins", "min", "hour", "hrs", "oz", "tbsp", "cup"]):
        word_cache[word] = False
        return False
    if len(word) < 3:
        word_cache[word] = False
        return False

    preds = ft_prediction(word)
    if not preds:
        word_cache[word] = False
        return False
    label, prob = preds[0]
    result = label == "__label__pt" and prob >= min_prob
    word_cache[word] = result
    return result


def contains_country_keyword(text: str):
    """Return first country keyword found (whole word match, cached)."""
    if text in keyword_cache:
        return keyword_cache[text]
    if not isinstance(text, str):
        keyword_cache[text] = None
        return None
    text_low = text.lower()
    for kw in COUNTRY_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text_low):
            keyword_cache[text] = kw
            return kw
    keyword_cache[text] = None
    return None


def contains_portuguese_word(text: str):
    """Return first detected Portuguese word from text, or None."""
    if not isinstance(text, str):
        return None
    words = re.findall(r"\b\w+\b", text.lower())
    for w in words:
        if is_word_portuguese(w):
            return w
    return None


def row_reason(row) -> str:
    """Return reason for marking recipe as Portuguese."""
    # --- 1️⃣ Country keyword check (highest priority) ---
    name = row.get("name", "")
    kw = contains_country_keyword(name)
    if kw:
        return f"Keyword '{kw}' found in name"

    for tag in row.get("tags", []):
        kw = contains_country_keyword(tag)
        if kw:
            return f"Keyword '{kw}' found in tags"

    desc = row.get("description", "")
    kw = contains_country_keyword(desc)
    if kw:
        return f"Keyword '{kw}' found in description"

    for s in row.get("steps", []):
        kw = contains_country_keyword(s)
        if kw:
            return f"Keyword '{kw}' found in steps"

    # --- 2️⃣ If no keyword found, fallback to Portuguese detection ---
    kw = contains_portuguese_word(name)
    if kw:
        return f"Portuguese word '{kw}' found in name"

    is_pt, prob, diff = ft_is_portuguese_confident(name)
    if is_pt:
        return f"Portuguese detected in name: '{name}' (p={prob:.2f}, diff={diff:.2f})"

    for tag in row.get("tags", []):
        if not isinstance(tag, str) or len(tag.strip()) < MIN_LANG_DETECT_LEN:
            continue
        kw = contains_portuguese_word(tag)
        if kw:
            return f"Portuguese word '{kw}' found in tag"
        is_pt, prob, diff = ft_is_portuguese_confident(tag)
        if is_pt:
            return f"Portuguese detected in tag: '{tag}' (p={prob:.2f}, diff={diff:.2f})"

    kw = contains_portuguese_word(desc)
    if kw:
        return f"Portuguese word '{kw}' found in description"
    is_pt, prob, diff = ft_is_portuguese_confident(desc)
    if is_pt:
        preview = desc[:50].replace("\n", " ") + ("..." if len(desc) > 50 else "")
        return f"Portuguese detected in description: '{preview}' (p={prob:.2f}, diff={diff:.2f})"

    for s in row.get("steps", []):
        kw = contains_portuguese_word(s)
        if kw:
            return f"Portuguese word '{kw}' found in step"
        if not isinstance(s, str) or len(s.strip()) < MIN_LANG_DETECT_LEN:
            continue
        is_pt, prob, diff = ft_is_portuguese_confident(s)
        if is_pt:
            preview = s[:50].replace("\n", " ") + ("..." if len(s) > 50 else "")
            return f"Portuguese detected in step: '{preview}' (p={prob:.2f}, diff={diff:.2f})"

    # --- No Portuguese indicators found ---
    return ""


# Apply reason detection
df["reason_flagged"] = [
    row_reason(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing recipes")
]
df["is_portuguese"] = df["reason_flagged"].apply(lambda x: x != "")

# Filter only Portuguese recipes
pt_df = df[df["is_portuguese"]]

# Save to curated folder
output_path = "data/curated/recipes.csv"
pt_df.to_csv(output_path, index=False)

print(len(pt_df), "Portuguese recipes detected")
print(f"Saved curated results to {output_path}")

print(f"Cache sizes → ft_cache: {len(ft_cache)}, word_cache: {len(word_cache)}, keyword_cache: {len(keyword_cache)}")
