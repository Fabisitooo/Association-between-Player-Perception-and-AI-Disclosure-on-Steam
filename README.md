# Steam AI Disclosure Review Analysis

> Last updated: 2026-05-12

This repository contains the code, processed data, and derived outputs used in the exploratory study on AI disclosure and player review language on Steam.

It is organized as a compact replication package for inspecting the analysis workflow and reproducing the reported tables and figures.

## Repository Layout

- `BA_Group1_Post2024.csv`
- `BA_Group2_Pre2024.csv`
- `BA_Group2_BEFORE_2024.csv`
- `BA_Group3_Control.csv`
  - Final negative-review group datasets used as the main input for the analyses.
- `protocol_2026-02-21_prof_followup/`
  - Follow-up analysis scripts, helper functions, processed follow-up data, and derived outputs.
- `liwc/`
  - LIWC-22 preparation scripts, aggregate LIWC result tables, and LIWC figure outputs.

## Analysis Groups

The current short-paper analysis collapses the original four source groups into two reporting groups:

- `AI-Label`: newly AI-labeled games plus recent reviews for retroactively AI-labeled games.
- `No AI-Label`: historic pre-disclosure reviews plus control games without AI disclosure.

The original source group is preserved as `source_BA_Group` where exported.

## Main Scripts

The core analysis lives in `protocol_2026-02-21_prof_followup/`.

- `01_collect_positive_reviews.py`
  - Loads or builds the positive-review comparison sets.
- `02_wordclouds_pos_vs_neg.py`
  - Creates overall and group-level word-frequency visualizations.
- `03_odds_ratios_chi_square.py`
  - Builds document-level contingency tables, odds ratios, chi-square tests, and FDR-corrected significance summaries.
- `04_threshold_sensitivity.py`
  - Runs the threshold sensitivity analysis for the combined AI-criticism keyword rate.
- `05_variance_bootstrap.py`
  - Estimates uncertainty for the combined AI-criticism keyword rate via cluster bootstrap at the game level.
- `07_ai_filter_keyword_wordclouds.py`
  - Produces AI-focused keyword summaries and visualizations.
- `08_odds_ratio_wordclouds.py`
  - Creates the combined odds-ratio word clouds used for the significant positive/negative term comparison.

The `06*` scripts document expansion and candidate-search steps used during dataset construction. They are included for transparency, but they are not required for the final reported outputs.

## Reproducing Figures And Tables

From the repository root:

```bash
python protocol_2026-02-21_prof_followup/01_collect_positive_reviews.py
python protocol_2026-02-21_prof_followup/02_wordclouds_pos_vs_neg.py
python protocol_2026-02-21_prof_followup/03_odds_ratios_chi_square.py
python protocol_2026-02-21_prof_followup/04_threshold_sensitivity.py
python protocol_2026-02-21_prof_followup/05_variance_bootstrap.py
python protocol_2026-02-21_prof_followup/07_ai_filter_keyword_wordclouds.py
python protocol_2026-02-21_prof_followup/08_odds_ratio_wordclouds.py
```

The scripts write derived outputs to `protocol_2026-02-21_prof_followup/outputs/`.

Current two-group output files use `ai_label`, `no_ai_label`, or `ai_criticism` in the filename. Older `group_1`, `group_2`, `group_3`, `strict`, and `soft` files may still exist from earlier four-group runs and should not be used for the current short-paper version.

## LIWC-22

The `liwc/outputs/` folder contains aggregate LIWC-22 tables and figures used for the current paper version. The raw LIWC result export is not included because it contains review-level text.

To rerun the LIWC workflow with an active LIWC-22 installation:

```bash
python liwc/prepare_liwc_input.py
python liwc/check_liwc_ready.py
python liwc/run_liwc_cli.py
python liwc/analyze_liwc_results.py
```

The official LIWC-22 CLI is not included in this repository and requires an active LIWC-22 license.

## Notes On Included Data

This repository includes processed group-level datasets and derived result files. It is not intended to be a full archive of every intermediate scrape, local scratch file, raw LIWC export, or LaTeX build artifact created during the project.

## Development Note

Some scripting assistance was used during development. Dataset construction, analysis runs, and reported outputs were reviewed manually.
