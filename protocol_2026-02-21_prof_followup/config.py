from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CUSTOM_STOPWORDS_FILE = SCRIPT_DIR / "custom_stopwords.txt"

CUTOFF_TIMESTAMP_2024 = 1704067200  # 2024-01-01 utc

GROUP1_BASE_FILE = PROJECT_ROOT / "BA_Group1_Post2024.csv"
GROUP2_BASE_FILE = PROJECT_ROOT / "BA_Group2_Pre2024.csv"
GROUP2B_BASE_FILE = PROJECT_ROOT / "BA_Group2_BEFORE_2024.csv"
GROUP3_BASE_FILE = PROJECT_ROOT / "BA_Group3_Control.csv"

# latest files, falls da
GROUP2_LATEST_FILE = DATA_DIR / "BA_Group2_Pre2024_latest.csv"
GROUP3_LATEST_FILE = DATA_DIR / "BA_Group3_Control_latest.csv"

GROUP2_ACTIVE_FILE = GROUP2_LATEST_FILE if GROUP2_LATEST_FILE.exists() else GROUP2_BASE_FILE
GROUP3_ACTIVE_FILE = GROUP3_LATEST_FILE if GROUP3_LATEST_FILE.exists() else GROUP3_BASE_FILE

GROUP_FILES = {
    "Group 1 (Native AI)": GROUP1_BASE_FILE,
    "Group 2 (AI Added - Recent)": GROUP2_ACTIVE_FILE,
    "Group 2 (AI Added - Historic)": GROUP2B_BASE_FILE,
    "Group 3 (Control / No AI)": GROUP3_ACTIVE_FILE,
}

# grp2b optional breiter seed
GROUP2_RECENT_FILE = GROUP2_ACTIVE_FILE

POSITIVE_OUTPUT_FILES = {
    "Group 1 (Native AI)": DATA_DIR / "group1_positive_reviews.csv",
    "Group 2 (AI Added - Recent)": DATA_DIR / "group2a_positive_reviews.csv",
    "Group 2 (AI Added - Historic)": DATA_DIR / "group2b_positive_reviews.csv",
    "Group 3 (Control / No AI)": DATA_DIR / "group3_positive_reviews.csv",
}

# grp2b nur pre-2024
GROUP_POSITIVE_TIME_FILTER = {
    "Group 1 (Native AI)": None,
    "Group 2 (AI Added - Recent)": None,
    "Group 2 (AI Added - Historic)": CUTOFF_TIMESTAMP_2024,
    "Group 3 (Control / No AI)": None,
}

# mini stopwords
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "dont",
    "even",
    "for",
    "from",
    "game",
    "games",
    "get",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "im",
    "in",
    "into",
    "is",
    "it",
    "its",
    "ive",
    "just",
    "like",
    "make",
    "me",
    "more",
    "most",
    "much",
    "my",
    "new",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "people",
    "play",
    "played",
    "player",
    "players",
    "playing",
    "really",
    "so",
    "some",
    "still",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}
