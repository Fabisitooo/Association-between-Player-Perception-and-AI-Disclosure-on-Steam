import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import DATA_DIR, PROJECT_ROOT

CUTOFF_DATE = pd.Timestamp("2024-01-01")

GROUP2_BASE = PROJECT_ROOT / "BA_Group2_Pre2024.csv"
GROUP3_BASE = PROJECT_ROOT / "BA_Group3_Control.csv"
AI_MASTER = PROJECT_ROOT / "steam_ai_reviews_final.csv"

GROUP2_CANDIDATES = DATA_DIR / "group2_additional_game_candidates_robust.csv"
GROUP3_CANDIDATES = DATA_DIR / "group3_additional_game_candidates_robust.csv"
SUMMARY_FILE = DATA_DIR / "robust_candidate_summary.csv"


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def load_ids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["appid"])
    return set(pd.to_numeric(df["appid"], errors="coerce").dropna().astype(int).tolist())


def fetch_steamspy_page(page: int, timeout: int = 20) -> List[Dict]:
    url = f"https://steamspy.com/api.php?request=all&page={page}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    rows = []
    for appid_str, item in payload.items():
        appid = to_int(appid_str, 0)
        if appid <= 0:
            continue
        positive = to_int(item.get("positive", 0))
        negative = to_int(item.get("negative", 0))
        total = positive + negative
        rows.append(
            {
                "appid": appid,
                "name_from_steamspy": str(item.get("name", "")).strip(),
                "positive_reviews_est": positive,
                "negative_reviews_est": negative,
                "total_reviews_est": total,
            }
        )
    return rows


def fetch_steamcmd_info(appid: int, timeout: int = 8) -> Optional[Dict]:
    url = f"https://api.steamcmd.net/v1/info/{appid}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get(str(appid), {})
    except Exception:
        return None

    common = data.get("common", {})
    if not common:
        return None

    app_type = str(common.get("type", "")).strip().lower()
    if app_type != "game":
        return None

    release_ts = common.get("steam_release_date")
    release_date = pd.NaT
    if release_ts is not None:
        try:
            release_date = pd.to_datetime(int(release_ts), unit="s", utc=True).tz_convert(None)
        except Exception:
            release_date = pd.NaT

    aicontenttype = common.get("aicontenttype", None)
    ai_flag = False
    if aicontenttype is not None:
        ai_flag = str(aicontenttype).strip() not in {"", "0", "none", "None"}

    return {
        "appid": appid,
        "name_from_info": str(common.get("name", "")).strip(),
        "release_date": release_date,
        "aicontenttype": None if aicontenttype is None else str(aicontenttype),
        "ai_flag": ai_flag,
    }


def save_candidates(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # leerfile trotzdem stabil
        pd.DataFrame(
            columns=[
                "appid",
                "game_name",
                "release_date",
                "aicontenttype",
                "total_reviews_est",
                "positive_reviews_est",
                "negative_reviews_est",
                "source",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(rows).drop_duplicates("appid").to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Robuste Kandidaten-Suche fuer Gruppe 2/3 auf Basis steamspy + steamcmd "
            "(mit kuerzeren Timeouts und Zwischenspeicherung)."
        )
    )
    parser.add_argument("--target-group2-games", type=int, default=101)
    parser.add_argument("--target-group3-games", type=int, default=101)
    parser.add_argument("--min-reviews", type=int, default=30)
    parser.add_argument("--max-control-reviews", type=int, default=5000)
    parser.add_argument("--max-steamspy-pages", type=int, default=80)
    parser.add_argument("--max-rows-per-page", type=int, default=180)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--steamcmd-timeout", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--sleep-between-pages", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_g2 = load_ids(GROUP2_BASE)
    existing_g3 = load_ids(GROUP3_BASE)
    ai_ids = load_ids(AI_MASTER)

    needed_g2 = max(0, args.target_group2_games - len(existing_g2))
    needed_g3 = max(0, args.target_group3_games - len(existing_g3))

    print(
        f"[start] base: group2={len(existing_g2)} group3={len(existing_g3)} | "
        f"needed: group2={needed_g2} group3={needed_g3}"
    )

    g2_rows: List[Dict] = []
    g3_rows: List[Dict] = []
    seen = set()
    processed_info = 0

    for page in range(args.max_steamspy_pages):
        if len(g2_rows) >= needed_g2 and len(g3_rows) >= needed_g3:
            break

        page_rows = fetch_steamspy_page(page)
        if not page_rows:
            continue
        page_rows = sorted(page_rows, key=lambda r: r["total_reviews_est"], reverse=True)

        candidates = []
        for row in page_rows:
            if len(candidates) >= args.max_rows_per_page:
                break
            appid = row["appid"]
            if appid in seen:
                continue
            seen.add(appid)

            total_reviews = row["total_reviews_est"]
            if total_reviews < args.min_reviews:
                continue
            candidates.append(row)

        if not candidates:
            continue

        print(f"[page {page}] candidates={len(candidates)}")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {
                ex.submit(fetch_steamcmd_info, row["appid"], args.steamcmd_timeout): row for row in candidates
            }
            for fut in as_completed(futures):
                row = futures[fut]
                processed_info += 1
                if processed_info % args.progress_every == 0:
                    print(
                        f"  [progress] processed={processed_info} g2_found={len(g2_rows)} g3_found={len(g3_rows)}"
                    )

                if len(g2_rows) >= needed_g2 and len(g3_rows) >= needed_g3:
                    break

                info = None
                try:
                    info = fut.result()
                except Exception:
                    info = None
                if not info:
                    continue

                release_date = info.get("release_date", pd.NaT)
                if pd.isna(release_date):
                    continue

                appid = row["appid"]
                game_name = info.get("name_from_info") or row.get("name_from_steamspy") or f"App {appid}"
                merged = {
                    "appid": appid,
                    "game_name": game_name,
                    "release_date": str(release_date.date()),
                    "aicontenttype": info.get("aicontenttype"),
                    "total_reviews_est": row.get("total_reviews_est", 0),
                    "positive_reviews_est": row.get("positive_reviews_est", 0),
                    "negative_reviews_est": row.get("negative_reviews_est", 0),
                    "source": "steamspy+steamcmd",
                }

                # grp2: ai + pre2024
                if (
                    len(g2_rows) < needed_g2
                    and info.get("ai_flag", False)
                    and release_date < CUTOFF_DATE
                    and appid not in existing_g2
                    and all(r["appid"] != appid for r in g2_rows)
                ):
                    g2_rows.append(merged)
                    print(
                        f"  + [group2] {game_name} ({appid}) release={merged['release_date']} "
                        f"reviews≈{merged['total_reviews_est']}"
                    )
                    save_candidates(g2_rows, GROUP2_CANDIDATES)

                # grp3: no ai + post2024 + cap
                if (
                    len(g3_rows) < needed_g3
                    and not info.get("ai_flag", False)
                    and release_date >= CUTOFF_DATE
                    and args.min_reviews <= merged["total_reviews_est"] <= args.max_control_reviews
                    and appid not in existing_g3
                    and appid not in ai_ids
                    and all(r["appid"] != appid for r in g3_rows)
                ):
                    g3_rows.append(merged)
                    print(
                        f"  + [group3] {game_name} ({appid}) release={merged['release_date']} "
                        f"reviews≈{merged['total_reviews_est']}"
                    )
                    save_candidates(g3_rows, GROUP3_CANDIDATES)

        time.sleep(max(0.0, args.sleep_between_pages))

    save_candidates(g2_rows, GROUP2_CANDIDATES)
    save_candidates(g3_rows, GROUP3_CANDIDATES)

    summary = pd.DataFrame(
        [
            {
                "group": "Group 2 (AI Added - Recent)",
                "base_games": len(existing_g2),
                "target_games": args.target_group2_games,
                "needed_games": needed_g2,
                "found_candidates": len(g2_rows),
                "projected_total_games": len(existing_g2) + len(g2_rows),
                "output_file": str(GROUP2_CANDIDATES),
            },
            {
                "group": "Group 3 (Control / No AI)",
                "base_games": len(existing_g3),
                "target_games": args.target_group3_games,
                "needed_games": needed_g3,
                "found_candidates": len(g3_rows),
                "projected_total_games": len(existing_g3) + len(g3_rows),
                "output_file": str(GROUP3_CANDIDATES),
            },
        ]
    )
    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"[done] group2 candidates={len(g2_rows)} -> {GROUP2_CANDIDATES}")
    print(f"[done] group3 candidates={len(g3_rows)} -> {GROUP3_CANDIDATES}")
    print(f"[done] summary -> {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
