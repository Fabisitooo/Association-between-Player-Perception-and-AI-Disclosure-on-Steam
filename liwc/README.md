# LIWC Workflow

> Last updated: 2026-05-12

This folder documents the LIWC-22 workflow for the Steam review corpus.

The repository includes aggregate LIWC outputs in `liwc/outputs/`. Review-level LIWC exports are not committed because they contain review text.

## Prepare Input

```bash
python liwc/prepare_liwc_input.py
```

Output:

- `liwc/input/steam_reviews_for_liwc.csv`
- `liwc/input/steam_reviews_for_liwc_clean.csv`
- `liwc/input/chunks/*.csv`

The file contains one row per review with:

- `review_id`
- `appid`
- `game_name`
- `source_BA_Group`
- `BA_Group` (`AI-Label` or `No AI-Label`)
- `sentiment`
- `timestamp`
- `review_text`

## Run Official LIWC-22 CLI

Check readiness:

```bash
python liwc/check_liwc_ready.py
```

After LIWC-22 is installed and activated:

```bash
LIWC-22-license-server
python liwc/run_liwc_cli.py
```

The runner uses the official CLI mode documented by LIWC:

```bash
LIWC-22-cli --mode wc --input liwc/input/steam_reviews_for_liwc_clean.csv --output liwc/output/liwc_results.csv
```

If the local CLI requires CSV-specific options, run:

```bash
LIWC-22-cli --mode wc --help
```

and adjust `liwc/run_liwc_cli.py`.

## Analyze LIWC Output

After the LIWC-22 export exists, run:

```bash
python liwc/analyze_liwc_results.py
```

The analysis script looks for:

- `liwc/LIWC-22_Results.csv`
- `liwc/output/liwc_results.csv`
- `liwc/output/liwc_results_part_*.csv`

Outputs are written to `liwc/outputs/`.

Committed aggregate outputs:

- `liwc_results_table.csv`
- `liwc_results_table.tex`
- `liwc_descriptive_stats.csv`
- `plot_liwc_*.png`
