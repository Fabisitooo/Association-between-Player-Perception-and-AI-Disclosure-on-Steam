"""
LIWC-22 analysis for the Steam AI disclosure paper.

Outputs:
- liwc/outputs/liwc_results_table.csv
- liwc/outputs/liwc_results_table.tex
- liwc/outputs/liwc_descriptive_stats.csv
- liwc/outputs/plot_liwc_*.png
"""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LIWC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LIWC_DIR / "outputs"
DEFAULT_INPUT_FILE = LIWC_DIR / "LIWC-22_Results.csv"
CLI_OUTPUT_FILE = LIWC_DIR / "output" / "liwc_results.csv"
CLI_CHUNK_GLOB = "liwc_results_part_*.csv"

N_BOOTSTRAP = 10_000
BOOTSTRAP_BATCH_SIZE = 250
RANDOM_SEED = 42

GROUP_AI = "AI-Label"
GROUP_NO_LABEL = "No-Label"
GROUP_ALIASES = {
    "AI-Label": GROUP_AI,
    "No-Label": GROUP_NO_LABEL,
    "No AI-Label": GROUP_NO_LABEL,
    "No_AI_Label": GROUP_NO_LABEL,
}

GROUP_ORDER = [GROUP_AI, GROUP_NO_LABEL]
GROUP_COLORS = {
    GROUP_AI: "#2563EB",
    GROUP_NO_LABEL: "#F97316",
}

METRICS = [
    ("Authentic", "Authentic"),
    ("Tone", "Tone"),
    ("WPS", "WPS"),
    ("BigWords", "BigWords"),
    ("Cognition", "Cognition"),
    ("Affect", "Affect"),
    ("tone_pos", "tone_pos"),
    ("tone_neg", "tone_neg"),
    ("emo_pos", "emo_pos"),
    ("emo_neg", "emo_neg"),
]


def normalise_group(label):
    """Map equivalent input labels to the two paper groups."""
    label = str(label).strip()
    return GROUP_ALIASES.get(label, label)


def find_column(df, requested_name):
    """Find a LIWC column by exact name first, then case-insensitive name."""
    if requested_name in df.columns:
        return requested_name

    lower_map = {col.lower(): col for col in df.columns}
    match = lower_map.get(requested_name.lower())
    if match:
        return match

    raise KeyError(
        f"Required LIWC metric '{requested_name}' not found. "
        f"Available columns include: {', '.join(df.columns[:20])}..."
    )


def read_liwc_export():
    """Load a single LIWC export or concatenate chunked CLI exports."""
    if DEFAULT_INPUT_FILE.exists():
        return pd.read_csv(DEFAULT_INPUT_FILE)

    if CLI_OUTPUT_FILE.exists():
        return pd.read_csv(CLI_OUTPUT_FILE)

    chunk_files = sorted((LIWC_DIR / "output").glob(CLI_CHUNK_GLOB))
    if chunk_files:
        return pd.concat((pd.read_csv(path) for path in chunk_files), ignore_index=True)

    raise FileNotFoundError(
        "Missing LIWC output. Expected one of: "
        f"{DEFAULT_INPUT_FILE}, {CLI_OUTPUT_FILE}, or {LIWC_DIR / 'output' / CLI_CHUNK_GLOB}"
    )

def load_data():
    df = read_liwc_export()
    group_column = None
    for candidate in ("label_group", "BA_Group"):
        if candidate in df.columns:
            group_column = candidate
            break

    if group_column is None:
        raise KeyError("Input must contain either a 'label_group' or 'BA_Group' column.")

    df["label_group"] = df[group_column].map(normalise_group)
    missing_groups = [group for group in GROUP_ORDER if group not in set(df["label_group"])]
    if missing_groups:
        raise ValueError(f"Missing required groups after normalisation: {missing_groups}")

    metric_columns = {}
    for requested_name, display_name in METRICS:
        source_col = find_column(df, requested_name)
        metric_columns[display_name] = source_col
        df[source_col] = pd.to_numeric(df[source_col], errors="coerce")

    return df, metric_columns


def bootstrap_mean(values, rng, n_bootstrap=N_BOOTSTRAP, batch_size=BOOTSTRAP_BATCH_SIZE):
    """Bootstrap the mean in batches to avoid allocating a huge index matrix."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty metric vector.")

    boot_means = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        size = stop - start
        idx = rng.integers(0, len(values), size=(size, len(values)))
        boot_means[start:stop] = values[idx].mean(axis=1)

    return boot_means


def metric_stats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def analyse_metric(df, metric_name, source_col, rng):
    ai = metric_stats(df.loc[df["label_group"] == GROUP_AI, source_col])
    no_label = metric_stats(df.loc[df["label_group"] == GROUP_NO_LABEL, source_col])

    ai_boot = bootstrap_mean(ai["values"], rng)
    no_label_boot = bootstrap_mean(no_label["values"], rng)
    diff_boot = ai_boot - no_label_boot

    ai_ci_low, ai_ci_high = np.percentile(ai_boot, [2.5, 97.5])
    no_ci_low, no_ci_high = np.percentile(no_label_boot, [2.5, 97.5])
    diff_ci_low, diff_ci_high = np.percentile(diff_boot, [2.5, 97.5])

    diff = ai["mean"] - no_label["mean"]
    p_value = float(np.mean(diff_boot <= 0))

    row = {
        "Metric": metric_name,
        "N_AI": ai["n"],
        "N_NoLabel": no_label["n"],
        "Mean_AI": ai["mean"],
        "SD_AI": ai["sd"],
        "Mean_NoLabel": no_label["mean"],
        "SD_NoLabel": no_label["sd"],
        "Diff": diff,
        "CI_low": float(diff_ci_low),
        "CI_high": float(diff_ci_high),
        "p": p_value,
        "AI_CI_low": float(ai_ci_low),
        "AI_CI_high": float(ai_ci_high),
        "NoLabel_CI_low": float(no_ci_low),
        "NoLabel_CI_high": float(no_ci_high),
    }

    plot_stats = {
        GROUP_AI: {
            "mean": ai["mean"],
            "ci_low": float(ai_ci_low),
            "ci_high": float(ai_ci_high),
        },
        GROUP_NO_LABEL: {
            "mean": no_label["mean"],
            "ci_low": float(no_ci_low),
            "ci_high": float(no_ci_high),
        },
    }

    descriptive_rows = [
        {
            "Metric": metric_name,
            "Group": GROUP_AI,
            "N": ai["n"],
            "Mean": ai["mean"],
            "SD": ai["sd"],
        },
        {
            "Metric": metric_name,
            "Group": GROUP_NO_LABEL,
            "N": no_label["n"],
            "Mean": no_label["mean"],
            "SD": no_label["sd"],
        },
    ]

    return row, plot_stats, descriptive_rows


def latex_escape(value):
    return str(value).replace("_", r"\_")


def format_num(value, digits=2):
    return f"{value:.{digits}f}"


def format_p(value):
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def write_latex_table(results_df, output_path):
    lines = [
        r"\begin{table}[!t]",
        r"\caption{LIWC-22 comparison between AI-labeled and non-labeled games.}",
        r"\label{tab:liwc_results}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"Metric & AI-Label M (SD) & No-Label M (SD) & Diff. & 95\% CI & p \\",
        r"\hline",
    ]

    for _, row in results_df.iterrows():
        ai_ms = f"{format_num(row['Mean_AI'])} ({format_num(row['SD_AI'])})"
        no_ms = f"{format_num(row['Mean_NoLabel'])} ({format_num(row['SD_NoLabel'])})"
        ci = f"[{format_num(row['CI_low'])}, {format_num(row['CI_high'])}]"
        line = (
            f"{latex_escape(row['Metric'])} & {ai_ms} & {no_ms} & "
            f"{format_num(row['Diff'])} & {ci} & {format_p(row['p'])} \\\\"
        )
        lines.append(line)

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def safe_plot_name(metric_name):
    name = metric_name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return f"plot_liwc_{name.strip('_')}.png"


def plot_metric(metric_name, plot_stats, output_dir):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    means = [plot_stats[group]["mean"] for group in GROUP_ORDER]
    lower_errors = [
        plot_stats[group]["mean"] - plot_stats[group]["ci_low"] for group in GROUP_ORDER
    ]
    upper_errors = [
        plot_stats[group]["ci_high"] - plot_stats[group]["mean"] for group in GROUP_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(GROUP_ORDER))
    ax.bar(
        x,
        means,
        yerr=np.array([lower_errors, upper_errors]),
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        edgecolor="white",
        width=0.65,
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "#111827"},
    )

    for idx, value in enumerate(means):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

    max_high = max(plot_stats[group]["ci_high"] for group in GROUP_ORDER)
    y_top = max_high * 1.18 if max_high > 0 else 1
    ax.set_ylim(0, y_top)
    ax.set_xticks(x, GROUP_ORDER)
    ax.set_ylabel("LIWC-22 score")
    ax.set_xlabel("")
    ax.set_title(f"LIWC-22 {metric_name}: AI-Label vs. No-Label", fontsize=13)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out = output_dir / safe_plot_name(metric_name)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    df, metric_columns = load_data()
    print("=== LIWC-22 Analysis ===")
    print(f"Input rows: {len(df):,}")
    print(df["label_group"].value_counts().reindex(GROUP_ORDER).to_string())
    print(f"Bootstrap samples per metric: {N_BOOTSTRAP:,}")

    result_rows = []
    descriptive_rows = []
    plot_outputs = []

    for metric_name, source_col in metric_columns.items():
        print(f"Analysing {metric_name}...")
        row, plot_stats, metric_descriptive = analyse_metric(df, metric_name, source_col, rng)
        result_rows.append(row)
        descriptive_rows.extend(metric_descriptive)
        plot_outputs.append(plot_metric(metric_name, plot_stats, OUTPUT_DIR))

    results_df = pd.DataFrame(result_rows)
    results_df = results_df[
        [
            "Metric",
            "Mean_AI",
            "Mean_NoLabel",
            "Diff",
            "CI_low",
            "CI_high",
            "p",
            "SD_AI",
            "SD_NoLabel",
            "N_AI",
            "N_NoLabel",
            "AI_CI_low",
            "AI_CI_high",
            "NoLabel_CI_low",
            "NoLabel_CI_high",
        ]
    ]
    descriptive_df = pd.DataFrame(descriptive_rows)

    results_csv = OUTPUT_DIR / "liwc_results_table.csv"
    descriptive_csv = OUTPUT_DIR / "liwc_descriptive_stats.csv"
    latex_table = OUTPUT_DIR / "liwc_results_table.tex"

    results_df.to_csv(results_csv, index=False, float_format="%.6f")
    descriptive_df.to_csv(descriptive_csv, index=False, float_format="%.6f")
    write_latex_table(results_df, latex_table)

    print("\nSaved outputs:")
    print(f"- {results_csv}")
    print(f"- {descriptive_csv}")
    print(f"- {latex_table}")
    for path in plot_outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
