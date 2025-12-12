import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

INPUT_FILE = "../../data/curated/merged_filtered_balanced_10k.csv"
OUTPUT_PT_EN = "../../data/curated/filtered_pt_en.csv"
OUTPUT_OTHER = "../../data/curated/other_languages.csv"


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def combine_text(row):
    parts = []

    if isinstance(row.get("name"), str):
        parts.append(row["name"])

    if isinstance(row.get("description"), str):
        parts.append(row["description"])

    if isinstance(row.get("steps"), list):
        parts.append(" ".join(row["steps"]))

    return " ".join(parts).strip()


def main():
    df = pd.read_csv(INPUT_FILE)

    # Combine text for better detection
    df["combined_text"] = df.apply(combine_text, axis=1)

    # Detect language
    df["language"] = df["combined_text"].apply(detect_language)

    # Normalize origin column
    df["origin_normalized"] = df["origin"].astype(str).str.lower()

    # KEEP criteria
    keep_mask = (
        df["language"].isin(["pt", "en"]) |
        df["origin_normalized"].str.contains("portuguese")
    )

    df_keep = df[keep_mask]
    df_other = df[~keep_mask]

    df_keep.to_csv(OUTPUT_PT_EN, index=False)
    df_other.to_csv(OUTPUT_OTHER, index=False)

    print("Done!")
    print(f"Kept (PT/EN or Portuguese origin): {len(df_keep)}")
    print(f"Other languages extracted: {len(df_other)}")


if __name__ == "__main__":
    main()
