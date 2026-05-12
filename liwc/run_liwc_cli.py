from pathlib import Path
import os
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "liwc" / "input" / "steam_reviews_for_liwc_clean.csv"
CHUNK_DIR = PROJECT_ROOT / "liwc" / "input" / "chunks"
OUTPUT_DIR = PROJECT_ROOT / "liwc" / "output"
OUTPUT_FILE = OUTPUT_DIR / "liwc_results.csv"


def run_cli(cli: str, input_file: Path, output_file: Path) -> None:
    cmd = [
        cli,
        "--mode",
        "wc",
        "--input",
        str(input_file),
        "--output",
        str(output_file),
    ]
    text_column = os.environ.get("LIWC_TEXT_COLUMN")
    if text_column:
        cmd.extend(["--text-column", text_column])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    cli = shutil.which("LIWC-22-cli")
    if cli is None:
        raise SystemExit(
            "LIWC-22-cli not found. Install and activate LIWC-22 first, then rerun this script."
        )

    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}. Run liwc/prepare_liwc_input.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = sorted(CHUNK_DIR.glob("steam_reviews_for_liwc_clean_part_*.csv"))
    if chunks:
        for chunk in chunks:
            output_file = OUTPUT_DIR / chunk.name.replace("steam_reviews_for_liwc_clean_part_", "liwc_results_part_")
            run_cli(cli, chunk, output_file)
        print(f"Saved chunked LIWC output files to {OUTPUT_DIR}")
    else:
        run_cli(cli, INPUT_FILE, OUTPUT_FILE)
        print(f"Saved LIWC output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
