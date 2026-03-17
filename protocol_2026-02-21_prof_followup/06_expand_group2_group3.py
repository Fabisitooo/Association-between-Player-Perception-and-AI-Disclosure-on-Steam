import argparse
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from analysis_utils import ensure_directories
from config import DATA_DIR, PROJECT_ROOT

CUTOFF_DATE = pd.Timestamp("2024-01-01")

GROUP2_BASE = PROJECT_ROOT / "BA_Group2_Pre2024.csv"
GROUP3_BASE = PROJECT_ROOT / "BA_Group3_Control.csv"
GROUP2_EXTRA = DATA_DIR / "group2_extra_negative_reviews.csv"
GROUP3_EXTRA = DATA_DIR / "group3_extra_negative_reviews.csv"
GROUP2_EXPANDED = DATA_DIR / "BA_Group2_Pre2024_expanded.csv"
GROUP3_EXPANDED = DATA_DIR / "BA_Group3_Control_expanded.csv"
EXPANSION_SUMMARY = DATA_DIR / "expansion_summary_group2_group3.csv"

GROUP2_TERMS = [
    "AI",
    "Generative",
    "Artificial Intelligence",
    "GPT",
    "LLM",
    "Chat",
    "Neural",
    "Machine Learning",
    "Procedural",
    "Story Generator",
    "Simulator",
    "Novel",
    "Indie",
    "RPG",
    "Strategy",
]

FALLBACK_GROUP3_TERMS = [
    "Indie",
    "Simulation",
    "Strategy",
    "RPG",
    "Adventure",
    "Action",
    "Casual",
]

AI_NOTE_KEYWORDS = [
    "ai generated",
    "ai-generated",
    "generative ai",
    "artificial intelligence",
    "ai generated content disclosure",
]

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


def parse_review_count(tooltip_html):
    if not tooltip_html:
        return 0
    match = re.search(r"([\d,]+) user reviews", str(tooltip_html))
    if not match:
        return 0
    return int(match.group(1).replace(",", ""))


def search_candidates(session, term, page):
    url = (
        "https://store.steampowered.com/search/results/"
        f"?query&term={term}&start={page * 50}&count=50&sort_by=Reviews_DESC&infinite=1"
    )
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    soup = BeautifulSoup(data.get("results_html", ""), "html.parser")
    rows = soup.find_all("a", class_="search_result_row")
    out = []
    for row in rows:
        appid = row.get("data-ds-appid")
        title_el = row.find("span", class_="title")
        if not appid or "," in appid or title_el is None:
            continue

        title = title_el.text.strip()
        review_el = row.find("span", class_="search_review_summary")
        review_count = parse_review_count(review_el.get("data-tooltip-html") if review_el else None)
        out.append({"appid": int(appid), "game_name": title, "reviews_count": review_count})
    return out


def parse_release_date(release_raw):
    if not release_raw:
        return pd.NaT
    return pd.to_datetime(release_raw, errors="coerce")


def get_appdetails(session, appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    key = str(appid)
    if key not in data or not data[key].get("success", False):
        return None
    return data[key].get("data", {})


def has_ai_note(notes):
    notes_l = str(notes or "").lower()
    return any(k in notes_l for k in AI_NOTE_KEYWORDS)


def has_store_disclosure(session, appid):
    url = f"https://store.steampowered.com/app/{appid}/?l=english"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 429:
            time.sleep(random.uniform(1.5, 3.0))
            return False
        resp.raise_for_status()
    except Exception:
        return False
    return "ai generated content disclosure" in resp.text.lower()


def fetch_negative_reviews(session, appid, game_name, max_reviews=80, max_pages=8):
    base_url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
    cursor = "*"
    params = {
        "filter": "all",
        "language": "english",
        "review_type": "negative",
        "num_per_page": 100,
        "purchase_type": "all",
        "cursor": cursor,
    }

    rows = []
    for _ in range(max_pages):
        params["cursor"] = cursor
        try:
            resp = session.get(base_url, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(random.uniform(1.0, 2.5))
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        reviews = data.get("reviews", [])
        if not reviews:
            break

        for review in reviews:
            rows.append(
                {
                    "appid": appid,
                    "game_name": game_name,
                    "review_text": review.get("review", ""),
                    "voted_up": review.get("voted_up"),
                    "votes_up": review.get("votes_up"),
                    "weighted_vote_score": review.get("weighted_vote_score"),
                    "timestamp": review.get("timestamp_created"),
                }
            )
            if len(rows) >= max_reviews:
                return rows

        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(random.uniform(0.2, 0.6))

    return rows


def append_rows(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def load_existing_appids(path: Path):
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["appid"])
    except Exception:
        return set()
    return set(pd.to_numeric(df["appid"], errors="coerce").dropna().astype(int).tolist())


def get_group3_terms():
    metadata = PROJECT_ROOT / "BA_Metadata_For_ControlGroup.csv"
    if not metadata.exists():
        return FALLBACK_GROUP3_TERMS
    try:
        df = pd.read_csv(metadata)
    except Exception:
        return FALLBACK_GROUP3_TERMS
    if "genres" not in df.columns:
        return FALLBACK_GROUP3_TERMS

    genres = []
    for genre_str in df["genres"].dropna().astype(str):
        genres.extend([g.strip() for g in genre_str.split("|") if g.strip()])
    if not genres:
        return FALLBACK_GROUP3_TERMS
    top = pd.Series(genres).value_counts().head(12).index.tolist()
    return top or FALLBACK_GROUP3_TERMS


def combine_expanded(base_file: Path, extra_file: Path, out_file: Path):
    base_df = pd.read_csv(base_file) if base_file.exists() else pd.DataFrame()
    extra_df = pd.read_csv(extra_file) if extra_file.exists() else pd.DataFrame()
    if base_df.empty and extra_df.empty:
        return pd.DataFrame()

    combined = pd.concat([base_df, extra_df], ignore_index=True)
    dedup_cols = [c for c in ["appid", "timestamp", "review_text"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)
    combined.to_csv(out_file, index=False)
    return combined


def expand_group2(
    session,
    target_total_games,
    min_reviews,
    max_search_pages,
    reviews_per_game,
):
    base_ids = load_existing_appids(GROUP2_BASE)
    extra_ids = load_existing_appids(GROUP2_EXTRA)
    current_total_ids = set(base_ids) | set(extra_ids)
    target_missing = max(0, target_total_games - len(current_total_ids))

    print(
        f"[Group2] aktuell={len(current_total_ids)} Spiele | Ziel={target_total_games} | fehlend={target_missing}"
    )
    if target_missing <= 0:
        return

    known_ids = set(current_total_ids)
    added_games = 0

    for term in GROUP2_TERMS:
        if len(known_ids) >= target_total_games:
            break
        print(f"[Group2] Suche term='{term}'")
        empty_pages = 0
        for page in range(max_search_pages):
            if len(known_ids) >= target_total_games:
                break

            candidates = search_candidates(session, term, page)
            if not candidates:
                empty_pages += 1
                if empty_pages >= 3:
                    break
                continue
            empty_pages = 0

            for c in candidates:
                if len(known_ids) >= target_total_games:
                    break
                appid = c["appid"]
                if appid in known_ids:
                    continue
                if c["reviews_count"] < min_reviews:
                    continue

                details = get_appdetails(session, appid)
                if not details:
                    continue

                release_raw = details.get("release_date", {}).get("date", "")
                release_date = parse_release_date(release_raw)
                if pd.isna(release_date) or release_date >= CUTOFF_DATE:
                    continue

                notes = details.get("content_descriptors", {}).get("notes", "")
                if not has_ai_note(notes) and not has_store_disclosure(session, appid):
                    continue

                rows = fetch_negative_reviews(
                    session=session,
                    appid=appid,
                    game_name=c["game_name"],
                    max_reviews=reviews_per_game,
                    max_pages=8,
                )
                if not rows:
                    continue

                append_rows(GROUP2_EXTRA, rows)
                known_ids.add(appid)
                added_games += 1
                print(
                    f"  + [Group2] {c['game_name']} ({appid}) | reviews={len(rows)} | total={len(known_ids)}"
                )

                time.sleep(random.uniform(0.2, 0.7))

    print(f"[Group2] neu gefundene Spiele={added_games} | total={len(known_ids)}")


def expand_group3(
    session,
    target_total_games,
    min_reviews,
    max_reviews_cap,
    max_search_pages,
    reviews_per_game,
):
    base_ids = load_existing_appids(GROUP3_BASE)
    extra_ids = load_existing_appids(GROUP3_EXTRA)
    current_total_ids = set(base_ids) | set(extra_ids)
    target_missing = max(0, target_total_games - len(current_total_ids))

    known_ai_ids = load_existing_appids(PROJECT_ROOT / "steam_ai_reviews_final.csv")
    known_group2_ids = load_existing_appids(GROUP2_BASE) | load_existing_appids(GROUP2_EXTRA)

    print(
        f"[Group3] aktuell={len(current_total_ids)} Spiele | Ziel={target_total_games} | fehlend={target_missing}"
    )
    if target_missing <= 0:
        return

    known_ids = set(current_total_ids)
    terms = get_group3_terms()
    added_games = 0

    for term in terms:
        if len(known_ids) >= target_total_games:
            break
        print(f"[Group3] Suche term='{term}'")
        empty_pages = 0
        for page in range(max_search_pages):
            if len(known_ids) >= target_total_games:
                break

            candidates = search_candidates(session, term, page)
            if not candidates:
                empty_pages += 1
                if empty_pages >= 3:
                    break
                continue
            empty_pages = 0

            for c in candidates:
                if len(known_ids) >= target_total_games:
                    break
                appid = c["appid"]
                if appid in known_ids or appid in known_ai_ids or appid in known_group2_ids:
                    continue
                if c["reviews_count"] < min_reviews or c["reviews_count"] > max_reviews_cap:
                    continue

                details = get_appdetails(session, appid)
                if not details:
                    continue

                release_raw = details.get("release_date", {}).get("date", "")
                release_date = parse_release_date(release_raw)
                if pd.isna(release_date) or release_date < CUTOFF_DATE:
                    continue

                notes = details.get("content_descriptors", {}).get("notes", "")
                if has_ai_note(notes):
                    continue
                if has_store_disclosure(session, appid):
                    continue

                rows = fetch_negative_reviews(
                    session=session,
                    appid=appid,
                    game_name=c["game_name"],
                    max_reviews=reviews_per_game,
                    max_pages=8,
                )
                if not rows:
                    continue

                append_rows(GROUP3_EXTRA, rows)
                known_ids.add(appid)
                added_games += 1
                print(
                    f"  + [Group3] {c['game_name']} ({appid}) | reviews={len(rows)} | total={len(known_ids)}"
                )

                time.sleep(random.uniform(0.2, 0.7))

    print(f"[Group3] neu gefundene Spiele={added_games} | total={len(known_ids)}")


def write_summary(group2_combined: pd.DataFrame, group3_combined: pd.DataFrame):
    rows = []
    for group_name, df in [
        ("Group 2 (AI Added - Recent) expanded", group2_combined),
        ("Group 3 (Control / No AI) expanded", group3_combined),
    ]:
        if df is None or df.empty:
            rows.append({"group": group_name, "n_games": 0, "n_reviews": 0})
        else:
            rows.append(
                {
                    "group": group_name,
                    "n_games": int(df["appid"].nunique()),
                    "n_reviews": int(len(df)),
                }
            )
    pd.DataFrame(rows).to_csv(EXPANSION_SUMMARY, index=False)
    print(f"[DONE] summary -> {EXPANSION_SUMMARY}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sucht zusaetzliche Spiele fuer Gruppe 2 und 3, um die Anzahl Spiele zu erhoehen."
    )
    parser.add_argument("--target-group2-games", type=int, default=105)
    parser.add_argument("--target-group3-games", type=int, default=105)
    parser.add_argument("--min-reviews", type=int, default=30)
    parser.add_argument("--max-control-reviews", type=int, default=5000)
    parser.add_argument("--max-search-pages", type=int, default=35)
    parser.add_argument("--reviews-per-game", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    session = build_session()

    start = time.time()
    expand_group2(
        session=session,
        target_total_games=args.target_group2_games,
        min_reviews=args.min_reviews,
        max_search_pages=args.max_search_pages,
        reviews_per_game=args.reviews_per_game,
    )
    expand_group3(
        session=session,
        target_total_games=args.target_group3_games,
        min_reviews=args.min_reviews,
        max_reviews_cap=args.max_control_reviews,
        max_search_pages=args.max_search_pages,
        reviews_per_game=args.reviews_per_game,
    )

    group2_combined = combine_expanded(GROUP2_BASE, GROUP2_EXTRA, GROUP2_EXPANDED)
    group3_combined = combine_expanded(GROUP3_BASE, GROUP3_EXTRA, GROUP3_EXPANDED)
    write_summary(group2_combined, group3_combined)

    print(
        f"[DONE] Group2 expanded games={group2_combined['appid'].nunique() if not group2_combined.empty else 0} | "
        f"Group3 expanded games={group3_combined['appid'].nunique() if not group3_combined.empty else 0}"
    )
    print(f"FERTIG in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
