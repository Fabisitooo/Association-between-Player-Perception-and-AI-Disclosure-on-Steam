import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from config import (
    DATA_DIR,
    GROUP2_BASE_FILE,
    GROUP2_LATEST_FILE,
    GROUP3_BASE_FILE,
    GROUP3_LATEST_FILE,
)

GROUP2_CANDIDATES_DEFAULT = DATA_DIR / "group2_additional_game_candidates_robust.csv"
GROUP3_CANDIDATES_DEFAULT = DATA_DIR / "group3_additional_game_candidates_robust.csv"

GROUP2_EXTRA_NEGATIVE = DATA_DIR / "group2_extra_negative_reviews_robust.csv"
GROUP3_EXTRA_NEGATIVE = DATA_DIR / "group3_extra_negative_reviews_robust.csv"
SUMMARY_FILE = DATA_DIR / "latest_negative_expansion_summary.csv"

COOKIES = {
    "birthtime": "568022401",
    "lastagecheckage": "1-0-1988",
    "wants_mature_content": "1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
}


def build_session() -> requests.Session:
    session = requests.Session()
    session.cookies.update(COOKIES)
    session.headers.update(HEADERS)
    return session


def load_candidate_games(path: Path):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"appid", "game_name"}
    if not required.issubset(df.columns):
        return []
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df = df.dropna(subset=["appid"]).copy()
    df["appid"] = df["appid"].astype(int)
    return list(df.drop_duplicates(subset=["appid"])[["appid", "game_name"]].itertuples(index=False, name=None))


def fetch_negative_reviews_for_game(
    appid: int,
    game_name: str,
    reviews_per_game: int,
    max_pages: int,
    sleep_min: float,
    sleep_max: float,
    max_page_retries: int,
):
    session = build_session()
    base_url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
    cursor = "*"
    rows = []

    for _ in range(max_pages):
        params = {
            "filter": "all",
            "language": "english",
            "review_type": "negative",
            "num_per_page": 100,
            "purchase_type": "all",
            "cursor": cursor,
        }

        data = None
        for attempt in range(max_page_retries):
            try:
                resp = session.get(base_url, params=params, timeout=15)
                if resp.status_code in {403, 429}:
                    time.sleep(0.8 + attempt * 1.0)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                time.sleep(0.5 + attempt * 0.5)

        if data is None:
            break

        reviews = data.get("reviews", [])
        if not reviews:
            break

        for review in reviews:
            text = str(review.get("review", "") or "")
            if not text:
                continue
            rows.append(
                {
                    "appid": appid,
                    "game_name": game_name,
                    "review_text": text,
                    "voted_up": review.get("voted_up"),
                    "votes_up": review.get("votes_up"),
                    "weighted_vote_score": review.get("weighted_vote_score"),
                    "timestamp": review.get("timestamp_created"),
                }
            )
            if len(rows) >= reviews_per_game:
                return rows

        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(random.uniform(sleep_min, sleep_max))

    return rows


def run_fetch_for_group(
    group_name: str,
    candidates_file: Path,
    extra_output_file: Path,
    reviews_per_game: int,
    max_pages: int,
    workers: int,
    sleep_min: float,
    sleep_max: float,
    max_page_retries: int,
):
    games = load_candidate_games(candidates_file)
    print(f"[{group_name}] candidate games: {len(games)}")
    if not games:
        pd.DataFrame(
            columns=["appid", "game_name", "review_text", "voted_up", "votes_up", "weighted_vote_score", "timestamp"]
        ).to_csv(extra_output_file, index=False)
        return 0, 0

    rows_all = []
    processed_games = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {
            pool.submit(
                fetch_negative_reviews_for_game,
                appid,
                game_name,
                reviews_per_game,
                max_pages,
                sleep_min,
                sleep_max,
                max_page_retries,
            ): (appid, game_name)
            for appid, game_name in games
        }
        for future in as_completed(future_map):
            appid, game_name = future_map[future]
            processed_games += 1
            try:
                rows = future.result()
            except Exception:
                rows = []
            rows_all.extend(rows)
            print(f"  - {processed_games}/{len(games)} {game_name} ({appid}) -> {len(rows)} negative reviews")

    extra_df = pd.DataFrame(rows_all)
    if extra_df.empty:
        extra_df = pd.DataFrame(
            columns=["appid", "game_name", "review_text", "voted_up", "votes_up", "weighted_vote_score", "timestamp"]
        )
    else:
        dedup_cols = [c for c in ["appid", "timestamp", "review_text"] if c in extra_df.columns]
        if dedup_cols:
            extra_df = extra_df.drop_duplicates(subset=dedup_cols)

    extra_output_file.parent.mkdir(parents=True, exist_ok=True)
    extra_df.to_csv(extra_output_file, index=False)
    print(f"[{group_name}] saved: {extra_output_file} | rows={len(extra_df)} | games={extra_df['appid'].nunique() if not extra_df.empty else 0}")
    return int(extra_df["appid"].nunique()) if not extra_df.empty else 0, len(extra_df)


def combine_base_and_extra(base_file: Path, extra_file: Path, latest_file: Path):
    base_df = pd.read_csv(base_file)
    if extra_file.exists():
        extra_df = pd.read_csv(extra_file)
    else:
        extra_df = pd.DataFrame(columns=base_df.columns)

    if not extra_df.empty:
        for col in base_df.columns:
            if col not in extra_df.columns:
                extra_df[col] = None
        extra_df = extra_df[base_df.columns]

    combined = pd.concat([base_df, extra_df], ignore_index=True)
    dedup_cols = [c for c in ["appid", "timestamp", "review_text"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)

    latest_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(latest_file, index=False)
    return combined


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Holt negative Reviews fuer neue Gruppe-2/3-Kandidaten und baut "
            "aktuelle Latest-Dateien fuer die Folgeanalysen."
        )
    )
    parser.add_argument("--group2-candidates", type=str, default=str(GROUP2_CANDIDATES_DEFAULT))
    parser.add_argument("--group3-candidates", type=str, default=str(GROUP3_CANDIDATES_DEFAULT))
    parser.add_argument("--reviews-per-game", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sleep-min", type=float, default=0.2)
    parser.add_argument("--sleep-max", type=float, default=0.7)
    parser.add_argument("--max-page-retries", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    g2_games_extra, g2_reviews_extra = run_fetch_for_group(
        group_name="Group 2 (AI Added - Recent)",
        candidates_file=Path(args.group2_candidates),
        extra_output_file=GROUP2_EXTRA_NEGATIVE,
        reviews_per_game=args.reviews_per_game,
        max_pages=args.max_pages,
        workers=args.workers,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        max_page_retries=args.max_page_retries,
    )
    g3_games_extra, g3_reviews_extra = run_fetch_for_group(
        group_name="Group 3 (Control / No AI)",
        candidates_file=Path(args.group3_candidates),
        extra_output_file=GROUP3_EXTRA_NEGATIVE,
        reviews_per_game=args.reviews_per_game,
        max_pages=args.max_pages,
        workers=args.workers,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        max_page_retries=args.max_page_retries,
    )

    g2_latest = combine_base_and_extra(GROUP2_BASE_FILE, GROUP2_EXTRA_NEGATIVE, GROUP2_LATEST_FILE)
    g3_latest = combine_base_and_extra(GROUP3_BASE_FILE, GROUP3_EXTRA_NEGATIVE, GROUP3_LATEST_FILE)

    summary = pd.DataFrame(
        [
            {
                "group": "Group 2 (AI Added - Recent)",
                "base_games": int(pd.read_csv(GROUP2_BASE_FILE, usecols=["appid"])["appid"].nunique()),
                "extra_games_with_reviews": g2_games_extra,
                "extra_reviews": g2_reviews_extra,
                "latest_games_total": int(g2_latest["appid"].nunique()),
                "latest_reviews_total": int(len(g2_latest)),
                "latest_file": str(GROUP2_LATEST_FILE),
            },
            {
                "group": "Group 3 (Control / No AI)",
                "base_games": int(pd.read_csv(GROUP3_BASE_FILE, usecols=["appid"])["appid"].nunique()),
                "extra_games_with_reviews": g3_games_extra,
                "extra_reviews": g3_reviews_extra,
                "latest_games_total": int(g3_latest["appid"].nunique()),
                "latest_reviews_total": int(len(g3_latest)),
                "latest_file": str(GROUP3_LATEST_FILE),
            },
        ]
    )
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"[done] summary -> {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
