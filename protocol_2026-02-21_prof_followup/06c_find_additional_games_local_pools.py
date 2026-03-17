import argparse
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
CONTROL_META = PROJECT_ROOT / "BA_Metadata_For_ControlGroup.csv"

GROUP2_CANDIDATES_OUT = DATA_DIR / "group2_additional_game_candidates_local.csv"
GROUP3_CANDIDATES_OUT = DATA_DIR / "group3_additional_game_candidates_local.csv"
SUMMARY_OUT = DATA_DIR / "additional_games_summary_local.csv"


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_unique_appids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["appid"])
    return set(pd.to_numeric(df["appid"], errors="coerce").dropna().astype(int).tolist())


def parse_release_any(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    if getattr(dt, "tzinfo", None) is not None:
        return dt.tz_convert(None)
    return dt


def fetch_steamcmd_info(appid: int, timeout: int = 15) -> Optional[Dict]:
    url = f"https://api.steamcmd.net/v1/info/{appid}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json().get("data", {}).get(str(appid), {})
    common = payload.get("common", {})
    if not common:
        return None

    release_ts = common.get("steam_release_date")
    release_date = pd.NaT
    if release_ts is not None:
        try:
            release_date = pd.to_datetime(int(release_ts), unit="s", utc=True).tz_convert(None)
        except Exception:
            release_date = pd.NaT

    aicontenttype = common.get("aicontenttype", None)
    ai_flag = str(aicontenttype).strip() not in {"", "0", "none", "None"} if aicontenttype is not None else False

    return {
        "appid": appid,
        "name_from_info": str(common.get("name", "")).strip(),
        "type": str(common.get("type", "")).strip().lower(),
        "release_date": release_date,
        "aicontenttype": None if aicontenttype is None else str(aicontenttype),
        "ai_flag": ai_flag,
    }


def fetch_steamspy_map(max_pages: int = 80, timeout: int = 30) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for page in range(max_pages):
        url = f"https://steamspy.com/api.php?request=all&page={page}"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        if not isinstance(data, dict) or not data:
            continue

        for appid_str, payload in data.items():
            appid = to_int(appid_str, default=0)
            if appid <= 0:
                continue
            positive = to_int(payload.get("positive", 0))
            negative = to_int(payload.get("negative", 0))
            out[appid] = {
                "appid": appid,
                "name_from_steamspy": str(payload.get("name", "")).strip(),
                "positive_reviews_est": positive,
                "negative_reviews_est": negative,
                "total_reviews_est": positive + negative,
            }
    return out


def build_group2_candidates(
    target_games: int,
    min_reviews: int,
    max_info_requests: int,
    workers: int,
) -> pd.DataFrame:
    base_ids = load_unique_appids(GROUP2_BASE)
    ai_df = pd.read_csv(AI_MASTER, usecols=["appid", "game_name"])
    ai_df["appid"] = pd.to_numeric(ai_df["appid"], errors="coerce")
    ai_df = ai_df.dropna(subset=["appid"]).copy()
    ai_df["appid"] = ai_df["appid"].astype(int)

    counts = (
        ai_df.groupby(["appid", "game_name"], as_index=False)
        .size()
        .rename(columns={"size": "negative_reviews_in_ai_master"})
        .sort_values("negative_reviews_in_ai_master", ascending=False)
    )

    needed = max(0, target_games - len(base_ids))
    if needed <= 0:
        return pd.DataFrame()

    # nur kandidaten mit genug review-basis
    pool = counts[
        (counts["negative_reviews_in_ai_master"] >= min_reviews) & (~counts["appid"].isin(base_ids))
    ].copy()
    pool = pool.head(max_info_requests)

    infos: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(fetch_steamcmd_info, int(row.appid)): int(row.appid) for row in pool.itertuples()}
        for fut in as_completed(futures):
            appid = futures[fut]
            try:
                info = fut.result()
            except Exception:
                continue
            if not info:
                continue
            if info.get("type") != "game":
                continue
            release_date = info.get("release_date", pd.NaT)
            if pd.isna(release_date) or release_date >= CUTOFF_DATE:
                continue
            infos.append(
                {
                    "appid": appid,
                    "release_date": str(release_date.date()),
                    "aicontenttype": info.get("aicontenttype"),
                    "name_from_info": info.get("name_from_info", ""),
                    "ai_flag_steamcmd": info.get("ai_flag", False),
                }
            )

    if not infos:
        return pd.DataFrame()

    info_df = pd.DataFrame(infos)
    merged = pool.merge(info_df, on="appid", how="inner")
    merged["game_name"] = merged["name_from_info"].where(
        merged["name_from_info"].astype(str).str.len() > 0, merged["game_name"]
    )
    merged = merged.sort_values("negative_reviews_in_ai_master", ascending=False).drop_duplicates("appid")
    merged["source"] = "ai_master + steamcmd"

    return merged.head(needed)[
        [
            "appid",
            "game_name",
            "release_date",
            "negative_reviews_in_ai_master",
            "aicontenttype",
            "ai_flag_steamcmd",
            "source",
        ]
    ]


def build_group3_candidates(
    target_games: int,
    min_reviews: int,
    max_control_reviews: int,
    steamspy_pages: int,
) -> pd.DataFrame:
    base_ids = load_unique_appids(GROUP3_BASE)
    ai_ids = load_unique_appids(AI_MASTER)

    meta = pd.read_csv(CONTROL_META, usecols=["appid", "name", "release_date"])
    meta["appid"] = pd.to_numeric(meta["appid"], errors="coerce")
    meta = meta.dropna(subset=["appid"]).copy()
    meta["appid"] = meta["appid"].astype(int)
    meta["release_date_parsed"] = meta["release_date"].apply(parse_release_any)

    needed = max(0, target_games - len(base_ids))
    if needed <= 0:
        return pd.DataFrame()

    steamspy = fetch_steamspy_map(max_pages=steamspy_pages)

    rows = []
    for row in meta.itertuples():
        appid = int(row.appid)
        if appid in base_ids or appid in ai_ids:
            continue
        if pd.isna(row.release_date_parsed) or row.release_date_parsed < CUTOFF_DATE:
            continue
        spy = steamspy.get(appid)
        if not spy:
            continue
        total_reviews = to_int(spy.get("total_reviews_est", 0))
        if total_reviews < min_reviews or total_reviews > max_control_reviews:
            continue

        rows.append(
            {
                "appid": appid,
                "game_name": str(row.name),
                "release_date": str(row.release_date_parsed.date()),
                "total_reviews_est": total_reviews,
                "positive_reviews_est": to_int(spy.get("positive_reviews_est", 0)),
                "negative_reviews_est": to_int(spy.get("negative_reviews_est", 0)),
                "source": "control_metadata + steamspy",
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("total_reviews_est", ascending=False).drop_duplicates("appid")
    return out.head(needed)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Findet zusätzliche Spiele für Gruppe 2/3 aus lokalen Pools "
            "(AI-Master + Control-Metadata) und externen Metadaten-APIs."
        )
    )
    parser.add_argument("--target-group2-games", type=int, default=101)
    parser.add_argument("--target-group3-games", type=int, default=101)
    parser.add_argument("--min-reviews", type=int, default=30)
    parser.add_argument("--max-control-reviews", type=int, default=5000)
    parser.add_argument("--steamspy-pages", type=int, default=80)
    parser.add_argument("--max-group2-info-requests", type=int, default=450)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    g2_base = len(load_unique_appids(GROUP2_BASE))
    g3_base = len(load_unique_appids(GROUP3_BASE))
    print(f"[start] base games: group2={g2_base}, group3={g3_base}")

    g2_candidates = build_group2_candidates(
        target_games=args.target_group2_games,
        min_reviews=args.min_reviews,
        max_info_requests=args.max_group2_info_requests,
        workers=args.workers,
    )
    g3_candidates = build_group3_candidates(
        target_games=args.target_group3_games,
        min_reviews=args.min_reviews,
        max_control_reviews=args.max_control_reviews,
        steamspy_pages=args.steamspy_pages,
    )

    g2_candidates.to_csv(GROUP2_CANDIDATES_OUT, index=False)
    g3_candidates.to_csv(GROUP3_CANDIDATES_OUT, index=False)

    summary = pd.DataFrame(
        [
            {
                "group": "Group 2 (AI Added - Recent)",
                "base_games": g2_base,
                "target_games": args.target_group2_games,
                "found_candidates": int(len(g2_candidates)),
                "projected_games_total": int(g2_base + len(g2_candidates)),
                "output_file": str(GROUP2_CANDIDATES_OUT),
            },
            {
                "group": "Group 3 (Control / No AI)",
                "base_games": g3_base,
                "target_games": args.target_group3_games,
                "found_candidates": int(len(g3_candidates)),
                "projected_games_total": int(g3_base + len(g3_candidates)),
                "output_file": str(GROUP3_CANDIDATES_OUT),
            },
        ]
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"[done] group2 candidates={len(g2_candidates)} -> {GROUP2_CANDIDATES_OUT}")
    print(f"[done] group3 candidates={len(g3_candidates)} -> {GROUP3_CANDIDATES_OUT}")
    print(f"[done] summary -> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
