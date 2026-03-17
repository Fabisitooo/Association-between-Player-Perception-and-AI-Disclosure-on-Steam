import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageDraw, ImageFont

from config import OUTPUT_DIR, PROJECT_ROOT

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


def detect_font_path():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def normalize_strengths(weights, scale=1000.0):
    if not weights:
        return {}
    max_weight = max(weights.values())
    if max_weight <= 0:
        return {}
    return {word: (value / max_weight) * scale for word, value in weights.items()}


def build_scope_frequencies(csv_path: Path, max_words_total: int):
    df = pd.read_csv(csv_path)
    df = df[df["significant_5pct"] == True].copy()

    half = max_words_total // 2

    negative = (
        df[df["favours"] == "negative"]
        .sort_values("odds_ratio_neg_vs_pos", ascending=False)
        .head(half)
    )
    positive = (
        df[df["favours"] == "positive"]
        .assign(positive_strength=lambda x: 1.0 / x["odds_ratio_neg_vs_pos"])
        .sort_values("positive_strength", ascending=False)
        .head(half)
    )

    negative_weights = {
        row.word: float(row.odds_ratio_neg_vs_pos)
        for row in negative.itertuples(index=False)
    }
    positive_weights = {
        row.word: float(row.positive_strength) for row in positive.itertuples(index=False)
    }

    return normalize_strengths(negative_weights), normalize_strengths(positive_weights)


def save_combined_odds_wordcloud(negative_weights, positive_weights, output_png: Path):
    if not HAS_WORDCLOUD:
        raise RuntimeError("wordcloud package is not available.")

    font_path = detect_font_path()
    negative_color = "#B22222"
    positive_color = "#0F766E"

    combined = {}
    color_map = {}

    for word, weight in negative_weights.items():
        combined[word] = weight
        color_map[word] = negative_color

    for word, weight in positive_weights.items():
        if word not in combined or weight > combined[word]:
            combined[word] = weight
        color_map[word] = positive_color

    wc = WordCloud(
        width=1800,
        height=1100,
        background_color="white",
        max_words=len(combined),
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
    plt.tight_layout(pad=0.2)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Creates combined odds-ratio word clouds for negative and positive review language."
    )
    parser.add_argument(
        "--max-words-total",
        type=int,
        default=250,
        help="Total number of words shown per combined word cloud.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scope_map = {
        "overall": (
            OUTPUT_DIR / "contingency_overall.csv",
            OUTPUT_DIR / "wordcloud_odds_overall.png",
            PROJECT_ROOT / "wordcloud_odds_overall.png",
        ),
        "newly_ai_labeled": (
            OUTPUT_DIR / "contingency_group_1_native_ai.csv",
            OUTPUT_DIR / "wordcloud_odds_newly_ai_labeled.png",
            PROJECT_ROOT / "wordcloud_odds_newly_ai_labeled.png",
        ),
        "without_ai_label": (
            OUTPUT_DIR / "contingency_group_3_control_no_ai.csv",
            OUTPUT_DIR / "wordcloud_odds_without_ai_label.png",
            PROJECT_ROOT / "wordcloud_odds_without_ai_label.png",
        ),
    }

    for _, (input_csv, output_png, root_png) in scope_map.items():
        negative_weights, positive_weights = build_scope_frequencies(
            input_csv, args.max_words_total
        )
        save_combined_odds_wordcloud(negative_weights, positive_weights, output_png)
        root_png.write_bytes(output_png.read_bytes())
        print(f"[OK] {output_png}")
        print(f"[OK] {root_png}")


if __name__ == "__main__":
    main()
