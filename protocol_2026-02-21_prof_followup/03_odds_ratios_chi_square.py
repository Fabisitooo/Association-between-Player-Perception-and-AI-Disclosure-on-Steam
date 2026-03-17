import argparse
import math
from collections import Counter

import pandas as pd

from analysis_utils import (
    document_token_sets,
    ensure_directories,
    load_negative_reviews,
    load_positive_reviews,
    slugify,
)
from config import OUTPUT_DIR


def chi_square_pvalue_df1(chi_square_value: float) -> float:
    chi_square_value = max(0.0, float(chi_square_value))
    return math.erfc(math.sqrt(chi_square_value / 2.0))


def benjamini_hochberg(p_values):
    n = len(p_values)
    if n == 0:
        return []

    sorted_idx = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    min_so_far = 1.0

    for rank, idx in reversed(list(enumerate(sorted_idx, start=1))):
        raw_q = (p_values[idx] * n) / rank
        min_so_far = min(min_so_far, raw_q)
        adjusted[idx] = min(1.0, min_so_far)

    return adjusted


def compute_word_statistics(scope_df, min_doc_frequency=20, min_token_length=3):
    neg_texts = scope_df[scope_df["sentiment"] == "negative"]["review_text"]
    pos_texts = scope_df[scope_df["sentiment"] == "positive"]["review_text"]

    neg_docs = document_token_sets(neg_texts, min_len=min_token_length)
    pos_docs = document_token_sets(pos_texts, min_len=min_token_length)

    n_neg = len(neg_docs)
    n_pos = len(pos_docs)
    if n_neg == 0 or n_pos == 0:
        return pd.DataFrame()

    neg_counts = Counter()
    pos_counts = Counter()
    for doc in neg_docs:
        neg_counts.update(doc)
    for doc in pos_docs:
        pos_counts.update(doc)

    vocabulary = set(neg_counts) | set(pos_counts)
    rows = []

    for word in vocabulary:
        a = int(neg_counts.get(word, 0))  # neg hit
        b = int(pos_counts.get(word, 0))  # pos hit
        doc_freq = a + b
        if doc_freq < min_doc_frequency:
            continue

        c = n_neg - a  # rest neg
        d = n_pos - b  # rest pos

        # or mit haldane-anscombe
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        log2_odds = math.log2(odds_ratio)

        denom = (a + b) * (c + d) * (a + c) * (b + d)
        if denom == 0:
            chi_square = 0.0
        else:
            n_total = a + b + c + d
            chi_square = (n_total * ((a * d - b * c) ** 2)) / denom

        p_value = chi_square_pvalue_df1(chi_square)
        neg_rate = a / n_neg if n_neg > 0 else 0.0
        pos_rate = b / n_pos if n_pos > 0 else 0.0

        rows.append(
            {
                "word": word,
                "neg_docs_with_word": a,
                "pos_docs_with_word": b,
                "neg_docs_without_word": c,
                "pos_docs_without_word": d,
                "neg_rate": neg_rate,
                "pos_rate": pos_rate,
                "doc_freq_total": doc_freq,
                "odds_ratio_neg_vs_pos": odds_ratio,
                "log2_odds_neg_vs_pos": log2_odds,
                "chi_square_df1": chi_square,
                "p_value": p_value,
                "n_negative_docs": n_neg,
                "n_positive_docs": n_pos,
                "favours": "negative"
                if log2_odds > 0
                else ("positive" if log2_odds < 0 else "neutral"),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["p_value_fdr_bh"] = benjamini_hochberg(result["p_value"].tolist())
    result["significant_5pct"] = result["p_value_fdr_bh"] < 0.05

    result = result.sort_values(
        by=["p_value_fdr_bh", "chi_square_df1", "doc_freq_total"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return result


def build_pretty_table(result_df, top_n_each_direction=40):
    cols = [
        "direction",
        "word",
        "doc_freq_total",
        "neg_docs_with_word",
        "pos_docs_with_word",
        "neg_rate_pct",
        "pos_rate_pct",
        "odds_ratio_neg_vs_pos",
        "log2_odds_neg_vs_pos",
        "chi_square_df1",
        "p_value_fdr_bh",
        "significant_5pct",
    ]

    neg = result_df[result_df["favours"] == "negative"].head(top_n_each_direction).copy()
    pos = result_df[result_df["favours"] == "positive"].head(top_n_each_direction).copy()

    def prepare(df, direction):
        if df.empty:
            return pd.DataFrame(columns=cols)
        out = df.copy()
        out["direction"] = direction
        out["neg_rate_pct"] = (out["neg_rate"] * 100).round(2)
        out["pos_rate_pct"] = (out["pos_rate"] * 100).round(2)
        out["odds_ratio_neg_vs_pos"] = out["odds_ratio_neg_vs_pos"].round(3)
        out["log2_odds_neg_vs_pos"] = out["log2_odds_neg_vs_pos"].round(3)
        out["chi_square_df1"] = out["chi_square_df1"].round(3)
        out["p_value_fdr_bh"] = out["p_value_fdr_bh"].astype(float)
        return out[cols]

    return pd.concat(
        [
            prepare(neg, "negative-leaning"),
            prepare(pos, "positive-leaning"),
        ],
        ignore_index=True,
    )


def export_scope_results(scope_name, result_df, top_n=100, top_n_pretty=40):
    scope_slug = slugify(scope_name)
    full_path = OUTPUT_DIR / f"contingency_{scope_slug}.csv"
    result_df.to_csv(full_path, index=False)

    neg_top = result_df[result_df["favours"] == "negative"].head(top_n)
    pos_top = result_df[result_df["favours"] == "positive"].head(top_n)

    neg_top_path = OUTPUT_DIR / f"top_negative_words_{scope_slug}.csv"
    pos_top_path = OUTPUT_DIR / f"top_positive_words_{scope_slug}.csv"
    neg_top.to_csv(neg_top_path, index=False)
    pos_top.to_csv(pos_top_path, index=False)

    pretty = build_pretty_table(result_df, top_n_each_direction=top_n_pretty)
    pretty_path = OUTPUT_DIR / f"contingency_pretty_{scope_slug}.csv"
    pretty.to_csv(pretty_path, index=False)

    return full_path, neg_top_path, pos_top_path, pretty_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contingency Tables + Odds Ratios + Chi-Square fuer Woerter in positiven vs negativen Reviews."
    )
    parser.add_argument(
        "--min-doc-frequency",
        type=int,
        default=20,
        help="Mindestanzahl Dokumente (pos+neg), in denen ein Wort vorkommen muss.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=3,
        help="Minimale Tokenlaenge fuer die Wortanalyse.",
    )
    parser.add_argument(
        "--skip-group-splits",
        action="store_true",
        help="Nur Overall-Analyse rechnen, keine gruppenspezifischen Tabellen.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Anzahl Top-Woerter je Richtung fuer rohe Top-Tabellen.",
    )
    parser.add_argument(
        "--top-n-pretty",
        type=int,
        default=40,
        help="Anzahl Top-Woerter je Richtung fuer die lesbare Pretty-Tabelle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    neg_df = load_negative_reviews()
    pos_df = load_positive_reviews()

    if neg_df.empty:
        raise RuntimeError("Keine negativen Reviews gefunden.")
    if pos_df.empty:
        raise RuntimeError("Keine positiven Reviews gefunden. Fuehre zuerst 01_collect_positive_reviews.py aus.")

    combined = pd.concat([neg_df, pos_df], ignore_index=True)
    combined = combined[combined["review_text"].astype(str).str.len() > 0].copy()

    scope_frames = [("overall", combined)]
    if not args.skip_group_splits:
        for group_name in sorted(combined["BA_Group"].dropna().unique()):
            scope_frames.append((group_name, combined[combined["BA_Group"] == group_name]))

    summary = []
    for scope_name, scope_df in scope_frames:
        neg_scope = scope_df[scope_df["sentiment"] == "negative"]
        pos_scope = scope_df[scope_df["sentiment"] == "positive"]
        n_negative_reviews = len(neg_scope)
        n_positive_reviews = len(pos_scope)
        n_negative_games = int(neg_scope["appid"].nunique()) if "appid" in neg_scope.columns else 0
        n_positive_games = int(pos_scope["appid"].nunique()) if "appid" in pos_scope.columns else 0

        result_df = compute_word_statistics(
            scope_df,
            min_doc_frequency=args.min_doc_frequency,
            min_token_length=args.min_token_length,
        )

        if result_df.empty:
            print(f"[SKIP] {scope_name}: keine auswertbaren Daten.")
            continue

        full_path, neg_top_path, pos_top_path, pretty_path = export_scope_results(
            scope_name,
            result_df,
            top_n=args.top_n,
            top_n_pretty=args.top_n_pretty,
        )
        n_rows = len(result_df)
        n_significant = int(result_df["significant_5pct"].sum())
        summary.append(
            {
                "scope": scope_name,
                "n_negative_reviews": n_negative_reviews,
                "n_positive_reviews": n_positive_reviews,
                "n_negative_games": n_negative_games,
                "n_positive_games": n_positive_games,
                "n_words_tested": n_rows,
                "n_significant_5pct": n_significant,
                "significant_share_pct": (n_significant / n_rows * 100) if n_rows > 0 else 0.0,
                "full_table": str(full_path),
                "pretty_table": str(pretty_path),
                "top_negative": str(neg_top_path),
                "top_positive": str(pos_top_path),
            }
        )
        print(
            f"[DONE] {scope_name}: {n_rows} Woerter, {n_significant} signifikant (FDR<0.05) | "
            f"neg reviews={n_negative_reviews}, pos reviews={n_positive_reviews}, "
            f"neg games={n_negative_games}, pos games={n_positive_games}"
        )

    summary_df = pd.DataFrame(summary).sort_values("scope").reset_index(drop=True)
    if not summary_df.empty:
        summary_df["significant_share_pct"] = summary_df["significant_share_pct"].round(2)

    summary_path = OUTPUT_DIR / "contingency_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
