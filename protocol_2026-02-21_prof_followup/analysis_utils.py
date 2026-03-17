import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from config import (
    CUSTOM_STOPWORDS_FILE,
    DATA_DIR,
    GROUP_FILES,
    OUTPUT_DIR,
    POSITIVE_OUTPUT_FILES,
    STOPWORDS,
)

TOKEN_RE = re.compile(r"[a-z][a-z0-9']{1,}")


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_stopwords():
    combined = set(STOPWORDS)
    if CUSTOM_STOPWORDS_FILE.exists():
        with CUSTOM_STOPWORDS_FILE.open("r", encoding="utf-8") as f:
            for raw_line in f:
                token = raw_line.strip().lower()
                if not token or token.startswith("#"):
                    continue
                combined.add(token)
    return combined


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _safe_read_csv(path, required_columns):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} fehlt Spalten: {missing}")

    return df


def load_negative_reviews() -> pd.DataFrame:
    frames = []
    required_columns = ["appid", "game_name", "review_text", "timestamp"]

    for group_name, path in GROUP_FILES.items():
        df = _safe_read_csv(path, required_columns)
        if df is None:
            continue

        tmp = df[required_columns].copy()
        tmp["BA_Group"] = group_name
        tmp["sentiment"] = "negative"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=required_columns + ["BA_Group", "sentiment"])

    combined = pd.concat(frames, ignore_index=True)
    combined["review_text"] = combined["review_text"].astype(str).fillna("")
    return combined


def load_positive_reviews() -> pd.DataFrame:
    frames = []
    required_columns = ["appid", "game_name", "review_text", "timestamp", "BA_Group"]

    for group_name, path in POSITIVE_OUTPUT_FILES.items():
        df = _safe_read_csv(path, required_columns)
        if df is None:
            continue

        tmp = df[required_columns].copy()
        tmp["BA_Group"] = tmp["BA_Group"].fillna(group_name)
        tmp["sentiment"] = "positive"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=required_columns + ["sentiment"])

    combined = pd.concat(frames, ignore_index=True)
    combined["review_text"] = combined["review_text"].astype(str).fillna("")
    return combined


def tokenize(text: str, min_len: int = 3):
    text_norm = str(text).lower().replace("’", "'")
    stopwords = get_stopwords()

    tokens = []
    for raw_token in TOKEN_RE.findall(text_norm):
        token = raw_token.strip("'").replace("'", "")
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        tokens.append(token)
    return tokens


def term_frequencies(text_series, min_len: int = 3) -> Counter:
    freq = Counter()
    for text in text_series:
        freq.update(tokenize(text, min_len=min_len))
    return freq


def document_token_sets(text_series, min_len: int = 3):
    docs = []
    for text in text_series:
        docs.append(set(tokenize(text, min_len=min_len)))
    return docs
