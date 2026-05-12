from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = PROJECT_ROOT / "protocol_2026-02-21_prof_followup"
OUTPUT_DIR = PROJECT_ROOT / "liwc" / "input"
OUTPUT_FILE = OUTPUT_DIR / "steam_reviews_for_liwc.csv"
CLEAN_OUTPUT_FILE = OUTPUT_DIR / "steam_reviews_for_liwc_clean.csv"
CHUNK_DIR = OUTPUT_DIR / "chunks"
CHUNK_SIZE = 10000

sys.path.insert(0, str(PROTOCOL_DIR))

from analysis_utils import load_negative_reviews, load_positive_reviews  # noqa: E402


def main() -> None:
    neg = load_negative_reviews(collapse_to_two_groups=True)
    pos = load_positive_reviews(collapse_to_two_groups=True)

    if neg.empty and pos.empty:
        raise RuntimeError("No reviews found for LIWC export.")

    df = pd.concat([neg, pos], ignore_index=True)
    df = df[df["review_text"].astype(str).str.len() > 0].copy()
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", df.index + 1)

    columns = [
        "review_id",
        "appid",
        "game_name",
        "source_BA_Group",
        "BA_Group",
        "sentiment",
        "timestamp",
        "review_text",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_df = df[columns].copy()
    export_df.to_csv(OUTPUT_FILE, index=False)

    clean_df = export_df.copy()
    clean_df["review_text"] = (
        clean_df["review_text"]
        .astype(str)
        .map(lambda text: re.sub(r"\s+", " ", text).strip())
    )
    clean_df.to_csv(CLEAN_OUTPUT_FILE, index=False)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old_chunk in CHUNK_DIR.glob("steam_reviews_for_liwc_clean_part_*.csv"):
        old_chunk.unlink()
    for idx, start in enumerate(range(0, len(clean_df), CHUNK_SIZE), start=1):
        chunk_path = CHUNK_DIR / f"steam_reviews_for_liwc_clean_part_{idx:03d}.csv"
        clean_df.iloc[start : start + CHUNK_SIZE].to_csv(chunk_path, index=False)

    print(f"Saved {len(df)} reviews to {OUTPUT_FILE}")
    print(f"Saved clean LIWC CSV to {CLEAN_OUTPUT_FILE}")
    print(f"Saved {len(list(CHUNK_DIR.glob('*.csv')))} chunks to {CHUNK_DIR}")
    print(df.groupby(["BA_Group", "sentiment"]).size().to_string())


if __name__ == "__main__":
    main()
