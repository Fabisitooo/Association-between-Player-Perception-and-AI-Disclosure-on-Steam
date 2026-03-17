import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from config import GROUP_FILES, GROUP_POSITIVE_TIME_FILTER, GROUP2_RECENT_FILE, POSITIVE_OUTPUT_FILES

COOKIES = {
    "birthtime": "568022401",
    "lastagecheckage": "1-0-1988",
    "wants_mature_content": "1",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

GROUP_ALIASES = {
    "group1": "Group 1 (Native AI)",
    "group2a": "Group 2 (AI Added - Recent)",
    "group2b": "Group 2 (AI Added - Historic)",
    "group3": "Group 3 (Control / No AI)",
}


def build_session() -> requests.Session:
    session = requests.Session()
    session.cookies.update(COOKIES)
    session.headers.update(HEADERS)
    return session


def load_unique_games(path):
    df = pd.read_csv(path, usecols=["appid", "game_name"])
    df = df.dropna(subset=["appid"])
    df["appid"] = df["appid"].astype(int)
    return list(df.drop_duplicates(subset=["appid"])[["appid", "game_name"]].itertuples(index=False, name=None))


def load_existing_review_counts(output_file):
    if not output_file.exists():
        return {}

    try:
        df = pd.read_csv(output_file, usecols=["appid"])
    except Exception:
        return {}

    if df.empty:
        return {}

    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df = df.dropna(subset=["appid"])
    df["appid"] = df["appid"].astype(int)
    return df.groupby("appid").size().astype(int).to_dict()


def fetch_positive_reviews(
    appid,
    game_name,
    group_name,
    reviews_per_game,
    max_pages,
    timestamp_before,
    sleep_min,
    sleep_max,
):
    session = build_session()
    base_url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
    cursor = "*"
    params = {
        "filter": "all",
        "language": "english",
        "review_type": "positive",
        "num_per_page": 100,
        "purchase_type": "all",
        "cursor": cursor,
    }

    rows = []
    empty_pages = 0

    for _ in range(max_pages):
        params["cursor"] = cursor
        try:
            response = session.get(base_url, params=params, timeout=15)
            if response.status_code == 429:
                time.sleep(random.uniform(2.0, 4.0))
                continue
            response.raise_for_status()
            data = response.json()
        except Exception:
            break

        reviews = data.get("reviews", [])
        if not reviews:
            empty_pages += 1
            if empty_pages >= 2:
                break
        else:
            empty_pages = 0

        for review in reviews:
            ts = int(review.get("timestamp_created", 0))
            if timestamp_before is not None and ts >= timestamp_before:
                continue

            text = review.get("review", "")
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
                    "timestamp": ts,
                    "BA_Group": group_name,
                    "sentiment": "positive",
                }
            )

            if len(rows) >= reviews_per_game:
                return rows

        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(random.uniform(sleep_min, sleep_max))

    return rows


def append_rows(output_file, rows):
    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not output_file.exists()
    df.to_csv(output_file, mode="a", index=False, header=header)


def run_for_group(
    group_name,
    input_file,
    output_file,
    timestamp_before,
    reviews_per_game,
    max_pages,
    workers,
    sleep_min,
    sleep_max,
    max_games_per_group,
):
    if not input_file.exists():
        print(f"[SKIP] {group_name}: Input fehlt ({input_file.name})")
        return

    games = load_unique_games(input_file)
    existing_counts = load_existing_review_counts(output_file)
    pending = []

    for appid, game_name in games:
        already_have = int(existing_counts.get(appid, 0))
        remaining = reviews_per_game - already_have
        if remaining > 0:
            pending.append((appid, game_name, remaining, already_have))

    if max_games_per_group is not None:
        pending = pending[:max_games_per_group]

    print(
        f"\n[{group_name}] Spiele total={len(games)} | offen={len(pending)} "
        f"(Ziel je Spiel={reviews_per_game}, max_pages={max_pages})"
    )
    if not pending:
        return

    collected_reviews = 0
    completed_games = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                fetch_positive_reviews,
                appid,
                game_name,
                group_name,
                remaining,
                max_pages,
                timestamp_before,
                sleep_min,
                sleep_max,
            ): (appid, game_name, remaining, already_have)
            for appid, game_name, remaining, already_have in pending
        }

        for future in as_completed(future_map):
            appid, game_name, remaining, already_have = future_map[future]
            try:
                rows = future.result()
            except Exception as exc:
                print(f"[WARN] {group_name} | {appid} | {game_name}: {exc}")
                continue

            append_rows(output_file, rows)
            completed_games += 1
            collected_reviews += len(rows)
            print(
                f"  - {completed_games}/{len(pending)} | {game_name} ({appid}) "
                f"hatte {already_have}, geholt +{len(rows)} (fehlend geplant: {remaining})"
            )

    print(f"[DONE] {group_name}: {completed_games} Spiele verarbeitet, {collected_reviews} Reviews gesammelt.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sammelt positive Reviews fuer die vorhandenen BA-Gruppen (neues Protokoll)."
    )
    parser.add_argument("--reviews-per-game", type=int, default=80, help="Zielanzahl positive Reviews je Spiel.")
    parser.add_argument("--max-pages", type=int, default=12, help="Maximale API-Seiten pro Spiel.")
    parser.add_argument("--workers", type=int, default=4, help="Parallele Worker.")
    parser.add_argument("--sleep-min", type=float, default=0.25, help="Minimale Pause zwischen Requests.")
    parser.add_argument("--sleep-max", type=float, default=0.75, help="Maximale Pause zwischen Requests.")
    parser.add_argument(
        "--historic-max-pages",
        type=int,
        default=None,
        help="Optionales max-pages Override fuer historische Gruppe (timestamp_before gesetzt).",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help="Gruppen-Auswahl: all oder kommasepariert: group1,group2a,group2b,group3",
    )
    parser.add_argument(
        "--max-games-per-group",
        type=int,
        default=None,
        help="Optionales Limit fuer schnelle Testlaeufe.",
    )
    parser.add_argument(
        "--group2b-seed",
        type=str,
        choices=["historic_subset", "all_group2"],
        default="historic_subset",
        help=(
            "Seed fuer Gruppe 2b: "
            "historic_subset=nutzt BA_Group2_BEFORE_2024.csv, "
            "all_group2=nutzt alle AppIDs aus BA_Group2_Pre2024.csv."
        ),
    )
    return parser.parse_args()


def selected_groups(raw_groups):
    raw_groups = (raw_groups or "all").strip().lower()
    if raw_groups == "all":
        return list(GROUP_FILES.keys())

    resolved = []
    for token in [g.strip() for g in raw_groups.split(",") if g.strip()]:
        if token not in GROUP_ALIASES:
            valid = ",".join(sorted(GROUP_ALIASES.keys()))
            raise ValueError(f"Unbekannte Gruppe '{token}'. Erlaubt: {valid},all")
        resolved.append(GROUP_ALIASES[token])

    # stabile reihenfolge
    return [g for g in GROUP_FILES.keys() if g in set(resolved)]


def main():
    args = parse_args()
    start = time.time()
    groups = selected_groups(args.groups)

    for group_name in groups:
        input_file = GROUP_FILES[group_name]
        if group_name == "Group 2 (AI Added - Historic)" and args.group2b_seed == "all_group2":
            input_file = GROUP2_RECENT_FILE

        output_file = POSITIVE_OUTPUT_FILES[group_name]
        timestamp_before = GROUP_POSITIVE_TIME_FILTER[group_name]
        effective_max_pages = args.max_pages
        if timestamp_before is not None and args.historic_max_pages is not None:
            effective_max_pages = args.historic_max_pages

        run_for_group(
            group_name=group_name,
            input_file=input_file,
            output_file=output_file,
            timestamp_before=timestamp_before,
            reviews_per_game=args.reviews_per_game,
            max_pages=effective_max_pages,
            workers=args.workers,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            max_games_per_group=args.max_games_per_group,
        )

    duration = time.time() - start
    print(f"\nFERTIG in {duration:.1f}s")


if __name__ == "__main__":
    main()
