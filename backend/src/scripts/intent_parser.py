import os

def clean_csv(file_path):
    # Read all lines
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Keep only non-empty lines (strip whitespace)
    cleaned_lines = [line for line in lines if line.strip() != ""]

    # Write back to the same file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned CSV saved: {file_path}")




if __name__ == "__main__":

    INPUT_FILE = "../../data/raw/bert_intent_training.csv"

    clean_csv(INPUT_FILE)
