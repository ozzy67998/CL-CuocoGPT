import os
import ast
import pandas as pd
from tqdm import tqdm
import re
import langid

os.makedirs("../..data/curated", exist_ok=True)

df = pd.read_csv("../..data/raw/RAW_recipes.csv")

def safe_parse(x):
    if isinstance(x, str) and x.startswith("["):
        return ast.literal_eval(x)
    return []

df["tags"]  = df["tags"].apply(safe_parse)         if "tags"  in df.columns else [[]]*len(df)
df["steps"] = df["steps"].apply(safe_parse)        if "steps" in df.columns else [[]]*len(df)

COUNTRY_KEYWORDS = [
    "portuguese","portugal","azores","madeira","lisbon","porto",
    "coimbra","algarve","braga","sintra","bacalhau","pastel de nata",
    "nata","chouriço","linguiça","feijoada","caldo verde","piri-piri",
    "alheira","azeite","vinho verde","port wine","bolinho","travesseiro",
    "arroz doce","bifana","francesinha","cataplana","fado","saudade",
    "português","lusitan","carnaval","lisboa",
]

KW_REGEX = re.compile("|".join(re.escape(k) for k in COUNTRY_KEYWORDS))

def contains_kw(text: str) -> bool:
    return isinstance(text, str) and KW_REGEX.search(text.lower()) is not None

def is_pt(text: str) -> bool:
    if not isinstance(text, str) or len(text) < 8: return False
    lang, prob = langid.classify(text)
    return lang == "pt" and prob >= 0.8

def classify_row(row):
    name = row.get("name","")
    desc = row.get("description","")
    tags = row.get("tags", [])

    # keyword pass
    # name
    m = contains_kw(name)
    if m: return True, "keyword:name"

    # description
    m = contains_kw(desc)
    if m: return True, "keyword:description"

    # tags
    for t in tags:
        if contains_kw(t):
            return True, f"keyword:tag:{t}"

    # fallback lang
    if is_pt(name): return True, "langid:name"
    if is_pt(desc): return True, "langid:description"

    return False, ""


is_pt_flags = []
triggers = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    is_pt, reason = classify_row(row)
    is_pt_flags.append(is_pt)
    triggers.append(reason)

pt_df = df[df["is_pt"]]
pt_df.to_csv("data/curated/recipes.csv", index=False)

print("Portuguese recipes:", len(pt_df))
