import pandas as pd
import csv

# === File paths ===
cleaned_file = "../../data/curated/merged_filtered_no_false_positives.csv"
original_file = "../../data/raw/RAW_recipes.csv"
output_file = "../../data/curated/merged_filtered_balanced_10k.csv"

# === Load datasets ===
merged_cleaned_df = pd.read_csv(cleaned_file)
original_df = pd.read_csv(original_file)

# === Drop last 3 columns and add origin column ===
merged_cleaned_df = merged_cleaned_df.iloc[:, :-3].copy()
merged_cleaned_df["origin"] = "portuguese curated"

# === Prepare original dataset ===
original_df = original_df.copy()
original_df["origin"] = "original randomized"

# === Remove any overlap (based on 'id') ===
existing_ids = set(merged_cleaned_df["id"].tolist())
remaining_df = original_df[~original_df["id"].isin(existing_ids)]

# === Compute how many random recipes are needed ===
target_size = 10000
num_existing = len(merged_cleaned_df)
num_needed = max(0, target_size - num_existing)

print(f"Existing recipes: {num_existing}")
print(f"Sampling {num_needed} new recipes to reach {target_size} total...")

# === Random sampling ===
if num_needed > 0:
    sampled_df = remaining_df.sample(n=num_needed, random_state=42)
    final_df = pd.concat([merged_cleaned_df, sampled_df], ignore_index=True)
else:
    final_df = merged_cleaned_df.copy()
    print("⚠️ Dataset already has 10,000 or more recipes — no sampling needed.")

# === Save the final dataset ===
final_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"✅ Final dataset saved to: {output_file}")
print(f"Final number of recipes: {len(final_df)}")
print(f"Portuguese curated recipes: {sum(final_df['origin'] == 'portuguese curated')}")
print(f"Original randomized recipes: {sum(final_df['origin'] == 'original randomized')}")
