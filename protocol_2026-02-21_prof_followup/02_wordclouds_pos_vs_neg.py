import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import ImageDraw, ImageFont

from analysis_utils import (
    ensure_directories,
    load_negative_reviews,
    load_positive_reviews,
    slugify,
    term_frequencies,
)
from config import OUTPUT_DIR

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False


_ORIG_TEXTBBOX = ImageDraw.ImageDraw.textbbox


def _patch_wordcloud_textbbox():
    def patched_textbbox(
        self,
        xy,
        text,
        font=None,
        anchor=None,
        spacing=4,
        align="left",
        direction=None,
        features=None,
        language=None,
        stroke_width=0,
        embedded_color=False,
    ):
        try:
            return _ORIG_TEXTBBOX(
                self,
                xy,
                text,
                font=font,
                anchor=anchor,
                spacing=spacing,
                align=align,
                direction=direction,
                features=features,
                language=language,
                stroke_width=stroke_width,
                embedded_color=embedded_color,
            )
        except ValueError as exc:
            # Pillow 9 on some systems rejects TransposedFont inside textbbox.
            if "Only supported for TrueType fonts" in str(exc) and isinstance(
                font, ImageFont.TransposedFont
            ):
                inner = font.font
                x0, y0, x1, y1 = inner.getbbox(text)
                width = x1 - x0
                height = y1 - y0
                if getattr(font, "orientation", None) is not None:
                    width, height = height, width
                x, y = xy
                return (x, y, x + width, y + height)
            raise

    ImageDraw.ImageDraw.textbbox = patched_textbbox


if HAS_WORDCLOUD:
    _patch_wordcloud_textbbox()

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 28,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
)


def detect_font_path():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return str(p)
    return None


def filter_frequencies(freq: Counter, min_frequency: int) -> Counter:
    return Counter({w: c for w, c in freq.items() if c >= min_frequency})


def apply_contrastive_filter(
    target_freq: Counter,
    other_freq: Counter,
    min_ratio: float,
    min_delta: int,
) -> Counter:
    kept = {}
    for word, target_count in target_freq.items():
        other_count = int(other_freq.get(word, 0))
        ratio = (target_count + 1) / (other_count + 1)
        delta = target_count - other_count
        if ratio >= min_ratio and delta >= min_delta:
            kept[word] = target_count
    return Counter(kept)


def save_top_words_csv(freq: Counter, output_csv, top_n: int = 300):
    if not freq:
        pd.DataFrame(columns=["word", "count"]).to_csv(output_csv, index=False)
        return

    top = freq.most_common(top_n)
    pd.DataFrame(top, columns=["word", "count"]).to_csv(output_csv, index=False)


def save_wordcloud(freq: Counter, title: str, output_png):
    font_path = detect_font_path()
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=250,
        collocations=False,
        font_path=font_path,
        prefer_horizontal=1.0,
        max_font_size=220,
        min_font_size=18,
    ).generate_from_frequencies(dict(freq))

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=22, pad=18)
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def _normalized_freq(freq: Counter, scale: int = 1000) -> dict:
    if not freq:
        return {}
    max_count = max(freq.values())
    if max_count <= 0:
        return {}
    return {word: (count / max_count) * scale for word, count in freq.items()}


def save_combined_sentiment_wordcloud(
    negative_freq: Counter,
    positive_freq: Counter,
    title: str,
    output_png,
):
    font_path = detect_font_path()
    neg_norm = _normalized_freq(negative_freq)
    pos_norm = _normalized_freq(positive_freq)

    combined = {}
    color_map = {}
    negative_color = "#B22222"
    positive_color = "#0F766E"

    for word, weight in neg_norm.items():
        combined[word] = weight
        color_map[word] = negative_color

    for word, weight in pos_norm.items():
        if word not in combined or weight > combined[word]:
            combined[word] = weight
            color_map[word] = positive_color

    wc = WordCloud(
        width=1800,
        height=1100,
        background_color="white",
        max_words=250,
        collocations=False,
        font_path=font_path,
        prefer_horizontal=1.0,
        max_font_size=260,
        min_font_size=20,
        random_state=42,
        scale=2,
    ).generate_from_frequencies(combined)

    def color_func(word, *args, **kwargs):
        return color_map.get(word, "#333333")

    wc = wc.recolor(color_func=color_func, random_state=42)

    fig = plt.figure(figsize=(16, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=28, pad=18)
    # The paper caption explains the color coding; skipping an in-image legend
    # keeps the word area larger and avoids tiny labels in IEEE's column width.
    plt.tight_layout(pad=0.2)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_bar_fallback(freq: Counter, title: str, output_png, top_n: int = 40):
    top = freq.most_common(top_n)
    if not top:
        return

    words = [w for w, _ in top][::-1]
    counts = [c for _, c in top][::-1]

    fig = plt.figure(figsize=(14, 10))
    plt.barh(words, counts)
    plt.title(f"{title} (Fallback: Top Words)", fontsize=22, pad=16)
    plt.xlabel("Count", fontsize=18)
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erstellt Wordclouds (oder Fallback-Barcharts) fuer positive und negative Reviews."
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=15,
        help="Minimale Haeufigkeit eines Wortes fuer Visualisierung.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=3,
        help="Minimale Tokenlaenge fuer die Wortanalyse.",
    )
    parser.add_argument(
        "--min-contrast-ratio",
        type=float,
        default=1.3,
        help="Kontrastfilter: (target+1)/(other+1) muss >= diesem Wert sein.",
    )
    parser.add_argument(
        "--min-contrast-delta",
        type=int,
        default=10,
        help="Kontrastfilter: target_count - other_count muss >= diesem Wert sein.",
    )
    parser.add_argument(
        "--disable-contrastive-filter",
        action="store_true",
        help="Deaktiviert den Kontrastfilter und nutzt nur Min-Frequency.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    neg_df = load_negative_reviews()
    pos_df = load_positive_reviews()

    if neg_df.empty:
        raise RuntimeError("Keine negativen Reviews gefunden. Pruefe die BA_Group*.csv Dateien.")
    if pos_df.empty:
        raise RuntimeError(
            "Keine positiven Reviews gefunden. Fuehre zuerst 01_collect_positive_reviews.py aus."
        )

    combined = pd.concat([neg_df, pos_df], ignore_index=True)
    combined = combined[combined["review_text"].astype(str).str.len() > 0].copy()

    summary_rows = []

    scopes = [("overall", combined)]
    for group_name in sorted(combined["BA_Group"].dropna().unique()):
        scopes.append((group_name, combined[combined["BA_Group"] == group_name]))

    for scope_name, scope_df in scopes:
        scope_slug = slugify(scope_name)
        sentiment_raw_freq = {}
        filtered_by_sentiment = {}
        for sentiment in ("negative", "positive"):
            subset = scope_df[scope_df["sentiment"] == sentiment]
            sentiment_raw_freq[sentiment] = term_frequencies(
                subset["review_text"], min_len=args.min_token_length
            )

        for sentiment in ("negative", "positive"):
            subset = scope_df[scope_df["sentiment"] == sentiment]
            freq = sentiment_raw_freq[sentiment]
            filtered = filter_frequencies(freq, min_frequency=args.min_frequency)
            other_sentiment = "positive" if sentiment == "negative" else "negative"
            contrastive_applied = not args.disable_contrastive_filter
            if contrastive_applied:
                filtered = apply_contrastive_filter(
                    target_freq=filtered,
                    other_freq=sentiment_raw_freq[other_sentiment],
                    min_ratio=args.min_contrast_ratio,
                    min_delta=args.min_contrast_delta,
                )
            filtered_by_sentiment[sentiment] = filtered

            csv_path = OUTPUT_DIR / f"top_words_{scope_slug}_{sentiment}.csv"
            save_top_words_csv(filtered, csv_path)

            png_path = OUTPUT_DIR / f"wordcloud_{scope_slug}_{sentiment}.png"
            title = f"{scope_name} | {sentiment.title()} Reviews"
            visualization_used = "bar_fallback"

            if filtered:
                if HAS_WORDCLOUD:
                    try:
                        save_wordcloud(filtered, title, png_path)
                        visualization_used = "wordcloud"
                    except Exception as exc:
                        print(
                            f"[WARN] Wordcloud fehlgeschlagen fuer '{scope_name}'/{sentiment}: {exc}. "
                            "Nutze Fallback-Barchart."
                        )
                        save_bar_fallback(filtered, title, png_path)
                        visualization_used = "bar_fallback"
                else:
                    save_bar_fallback(filtered, title, png_path)
                    visualization_used = "bar_fallback"

            summary_rows.append(
                {
                    "scope": scope_name,
                    "sentiment": sentiment,
                    "n_reviews": len(subset),
                    "n_games": int(subset["appid"].nunique()) if "appid" in subset.columns else 0,
                    "n_unique_words_after_filter": len(filtered),
                    "contrastive_filter": contrastive_applied,
                    "min_contrast_ratio": args.min_contrast_ratio if contrastive_applied else 0.0,
                    "min_contrast_delta": args.min_contrast_delta if contrastive_applied else 0,
                    "visualization": visualization_used,
                    "output_csv": str(csv_path),
                    "output_image": str(png_path),
                }
            )

        if scope_name == "overall" and HAS_WORDCLOUD:
            try:
                combined_png = OUTPUT_DIR / "wordcloud_overall_combined.png"
                save_combined_sentiment_wordcloud(
                    negative_freq=filtered_by_sentiment.get("negative", Counter()),
                    positive_freq=filtered_by_sentiment.get("positive", Counter()),
                    title="",
                    output_png=combined_png,
                )
            except Exception as exc:
                print(f"[WARN] Kombinierte Wordcloud fehlgeschlagen: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "wordcloud_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("FERTIG: Wordcloud/Fallback-Render abgeschlossen")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
