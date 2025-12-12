import pandas as pd
import csv

# === File paths ===
sentences_file = "../../data/curated/recipes_sentences.csv"   # dataset with langid for sentences
tokens_file = "../../data/curated/recipes_tokens.csv"          # dataset with single words
output_file = "../../data/curated/merged_filtered.csv"

# === Function to safely read CSV or TSV with proper quoting ===
def safe_read_csv(path):
    # Try reading with comma first, then tab if columns look weird
    df = pd.read_csv(path, sep=",", header=None, quoting=csv.QUOTE_MINIMAL, engine="python", dtype=str)
    if df.shape[1] < 2:  # if it parsed into too few columns, try tab
        df = pd.read_csv(path, sep="\t", header=None, quoting=csv.QUOTE_MINIMAL, engine="python", dtype=str)
    return df

# === Load datasets ===
df_sentences = safe_read_csv(sentences_file)
df_tokens = safe_read_csv(tokens_file)

# === Add source column (optional but useful) ===
df_sentences["source"] = "sentence"
df_tokens["source"] = "token"

# === Merge datasets ===
merged_df = pd.concat([df_sentences, df_tokens], ignore_index=True)

# === Drop duplicates by 'id' (assuming id is column index 1) ===
merged_df = merged_df.drop_duplicates(subset=[1], keep="first")

# === Save the cleaned dataset ===
# Use quoting=csv.QUOTE_ALL to keep commas inside list fields intact
merged_df.to_csv(output_file, sep=",", index=False, header=False, quoting=csv.QUOTE_ALL)

print(f"✅ Merged dataset saved to '{output_file}' with {len(merged_df)} unique entries.")
