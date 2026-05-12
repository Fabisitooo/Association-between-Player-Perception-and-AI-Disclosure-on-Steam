from pathlib import Path
import importlib.util
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "liwc" / "input" / "steam_reviews_for_liwc_clean.csv"
CHUNK_DIR = PROJECT_ROOT / "liwc" / "input" / "chunks"


def main() -> None:
    checks = [
        ("LIWC-22 desktop/CLI", shutil.which("LIWC-22-cli") is not None),
        ("LIWC-22 license server", shutil.which("LIWC-22-license-server") is not None),
        ("Python package liwc", importlib.util.find_spec("liwc") is not None),
        ("Python package pyliwc", importlib.util.find_spec("pyliwc") is not None),
        ("Clean LIWC input CSV", INPUT_FILE.exists()),
        ("Chunked LIWC input CSVs", any(CHUNK_DIR.glob("steam_reviews_for_liwc_clean_part_*.csv"))),
    ]

    for label, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"{status:8} {label}")

    if INPUT_FILE.exists():
        print(f"\nInput: {INPUT_FILE}")
    chunks = sorted(CHUNK_DIR.glob("steam_reviews_for_liwc_clean_part_*.csv"))
    if chunks:
        print(f"Chunks: {len(chunks)} files in {CHUNK_DIR}")


if __name__ == "__main__":
    main()
