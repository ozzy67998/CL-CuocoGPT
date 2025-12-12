import pandas as pd
import csv

# File paths
input_file = "../../data/curated/merged_filtered.csv"
input_file_2 = "../../data/curated/false_positives_lang_detected.csv"
output_file = "../../data/curated/merged_filtered_no_false_positives.csv"

# Load both datasets
merged_df = pd.read_csv(input_file)
false_positives_df = pd.read_csv(input_file_2)

# Identify which column(s) can uniquely match rows (use 'id' if available)
if "id" in merged_df.columns and "id" in false_positives_df.columns:
    # Filter out recipes whose IDs are in the false positives list
    filtered_df = merged_df[~merged_df["id"].isin(false_positives_df["id"])]
else:
    # If IDs are missing, fall back to matching by name (less reliable)
    filtered_df = merged_df[~merged_df["name"].isin(false_positives_df["name"])]

# Save the filtered dataset
filtered_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"✅ Filtered dataset saved to: {output_file}")
print(f"Original entries: {len(merged_df)}")
print(f"Removed false positives: {len(merged_df) - len(filtered_df)}")
print(f"Remaining entries: {len(filtered_df)}")
