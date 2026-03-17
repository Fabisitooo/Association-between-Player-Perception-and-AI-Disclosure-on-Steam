import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, PROJECT_ROOT

GROUP2_BASE = PROJECT_ROOT / "BA_Group2_Pre2024.csv"
GROUP3_BASE = PROJECT_ROOT / "BA_Group3_Control.csv"

GROUP2_CANDIDATES = DATA_DIR / "group2_additional_game_candidates.csv"
GROUP3_CANDIDATES = DATA_DIR / "group3_additional_game_candidates.csv"
FALLBACK_SUMMARY = DATA_DIR / "fallback_candidate_summary.csv"

CUTOFF_DATE = pd.Timestamp("2024-01-01")


def to_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def load_existing_ids(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["appid"])
    return set(pd.to_numeric(df["appid"], errors="coerce").dropna().astype(int).tolist())


def fetch_steamspy_page(page: int):
    url = f"https://steamspy.com/api.php?request=all&page={page}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return []
    rows = []
    for appid_str, payload in data.items():
        appid = to_int(appid_str)
        if appid <= 0:
            continue
        positive = to_int(payload.get("positive", 0))
        negative = to_int(payload.get("negative", 0))
        total_reviews = positive + negative
        rows.append(
            {
                "appid": appid,
                "name": str(payload.get("name", "")).strip(),
                "positive_reviews_est": positive,
                "negative_reviews_est": negative,
                "total_reviews_est": total_reviews,
            }
        )
    return rows


def fetch_steamcmd_info(appid: int):
    url = f"https://api.steamcmd.net/v1/info/{appid}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", {}).get(str(appid), {})
    common = data.get("common", {})
    if not common:
        return None

    app_type = str(common.get("type", "")).strip()
    name = str(common.get("name", "")).strip()
    aicontenttype = common.get("aicontenttype", None)
    release_ts = common.get("steam_release_date", None)

    release_date = pd.NaT
    if release_ts is not None:
        try:
            release_date = pd.to_datetime(int(release_ts), unit="s", utc=True).tz_convert(None)
        except Exception:
            release_date = pd.NaT

    return {
        "appid": appid,
        "name_from_info": name,
        "type": app_type,
        "aicontenttype": None if aicontenttype is None else str(aicontenttype),
        "release_date": release_date,
    }


def is_ai_game(aicontenttype):
    if aicontenttype is None:
        return False
    return str(aicontenttype).strip() not in {"", "0", "none", "None"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fallback-Discovery: sucht zusaetzliche Game-Kandidaten fuer Gruppe 2/3 "
            "ueber steamspy + steamcmd, wenn store.steampowered.com blockiert ist."
        )
    )
    parser.add_argument("--target-group2-games", type=int, default=101)
    parser.add_argument("--target-group3-games", type=int, default=101)
    parser.add_argument("--min-reviews", type=int, default=30)
    parser.add_argument("--max-control-reviews", type=int, default=5000)
    parser.add_argument("--max-steamspy-pages", type=int, default=80)
    parser.add_argument("--max-info-requests", type=int, default=12000)
    parser.add_argument("--sleep-between-info", type=float, default=0.06)
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_g2 = load_existing_ids(GROUP2_BASE)
    existing_g3 = load_existing_ids(GROUP3_BASE)

    g2_needed = max(0, args.target_group2_games - len(existing_g2))
    g3_needed = max(0, args.target_group3_games - len(existing_g3))

    print(
        f"[start] existing: group2={len(existing_g2)}, group3={len(existing_g3)} | "
        f"needed: group2={g2_needed}, group3={g3_needed}"
    )

    g2_candidates = []
    g3_candidates = []
    seen_appids = set()
    info_requests = 0

    for page in range(args.max_steamspy_pages):
        if len(g2_candidates) >= g2_needed and len(g3_candidates) >= g3_needed:
            break
        rows = fetch_steamspy_page(page)
        if not rows:
            break
        print(f"[steamspy] page={page} rows={len(rows)}")

        for row in rows:
            if len(g2_candidates) >= g2_needed and len(g3_candidates) >= g3_needed:
                break
            appid = row["appid"]
            if appid in seen_appids:
                continue
            seen_appids.add(appid)

            total_reviews = row["total_reviews_est"]
            if total_reviews < args.min_reviews:
                continue

            info_requests += 1
            if info_requests > args.max_info_requests:
                print("[warn] max info requests reached, stopping early")
                break

            try:
                info = fetch_steamcmd_info(appid)
            except Exception:
                continue

            if not info:
                continue
            if str(info["type"]).lower() != "game":
                continue

            release_date = info["release_date"]
            if pd.isna(release_date):
                continue

            ai_flag = is_ai_game(info["aicontenttype"])
            merged = {
                "appid": appid,
                "game_name": info["name_from_info"] or row["name"],
                "release_date": str(release_date.date()),
                "aicontenttype": info["aicontenttype"],
                "total_reviews_est": total_reviews,
                "positive_reviews_est": row["positive_reviews_est"],
                "negative_reviews_est": row["negative_reviews_est"],
                "source": "steamspy+steamcmd",
            }

            # grp2: ai + pre2024
            if (
                len(g2_candidates) < g2_needed
                and ai_flag
                and release_date < CUTOFF_DATE
                and appid not in existing_g2
                and all(c["appid"] != appid for c in g2_candidates)
            ):
                g2_candidates.append(merged)
                print(
                    f"  + [group2] {merged['game_name']} ({appid}) release={merged['release_date']} "
                    f"reviews≈{total_reviews} ai={merged['aicontenttype']}"
                )

            # grp3: no ai + post2024 + review cap
            if (
                len(g3_candidates) < g3_needed
                and not ai_flag
                and release_date >= CUTOFF_DATE
                and args.min_reviews <= total_reviews <= args.max_control_reviews
                and appid not in existing_g3
                and all(c["appid"] != appid for c in g3_candidates)
            ):
                g3_candidates.append(merged)
                print(
                    f"  + [group3] {merged['game_name']} ({appid}) release={merged['release_date']} "
                    f"reviews≈{total_reviews}"
                )

            time.sleep(args.sleep_between_info)

    pd.DataFrame(g2_candidates).to_csv(GROUP2_CANDIDATES, index=False)
    pd.DataFrame(g3_candidates).to_csv(GROUP3_CANDIDATES, index=False)

    summary = pd.DataFrame(
        [
            {
                "group": "Group 2 (AI Added - Recent)",
                "existing_games": len(existing_g2),
                "target_games": args.target_group2_games,
                "needed_games": g2_needed,
                "found_candidates": len(g2_candidates),
                "projected_total_games": len(existing_g2) + len(g2_candidates),
                "candidates_file": str(GROUP2_CANDIDATES),
            },
            {
                "group": "Group 3 (Control / No AI)",
                "existing_games": len(existing_g3),
                "target_games": args.target_group3_games,
                "needed_games": g3_needed,
                "found_candidates": len(g3_candidates),
                "projected_total_games": len(existing_g3) + len(g3_candidates),
                "candidates_file": str(GROUP3_CANDIDATES),
            },
        ]
    )
    summary.to_csv(FALLBACK_SUMMARY, index=False)

    print(f"[done] group2 candidates={len(g2_candidates)} -> {GROUP2_CANDIDATES}")
    print(f"[done] group3 candidates={len(g3_candidates)} -> {GROUP3_CANDIDATES}")
    print(f"[done] summary -> {FALLBACK_SUMMARY}")


if __name__ == "__main__":
    main()
