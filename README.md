## Contents

- `BA_Group1_Post2024.csv`
- `BA_Group2_Pre2024.csv`
- `BA_Group2_BEFORE_2024.csv`
- `BA_Group3_Control.csv`
  - Main negative-review group datasets used as core inputs.
- `protocol_2026-02-21_prof_followup/`
  - Analysis scripts, helper files, processed follow-up inputs, and generated outputs.

## Main scripts

The main analysis scripts are located in `protocol_2026-02-21_prof_followup/`.

- `01_collect_positive_reviews.py`
- `02_wordclouds_pos_vs_neg.py`
- `03_odds_ratios_chi_square.py`
- `04_threshold_sensitivity.py`
- `05_variance_bootstrap.py`
- `07_ai_filter_keyword_wordclouds.py`
- `08_odds_ratio_wordclouds.py`

The `06*` scripts document the candidate-search and dataset-expansion steps used during data construction and are included for transparency.

## Running the analysis

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

Derived tables and figures are written to `protocol_2026-02-21_prof_followup/outputs/`.

## Development note

Some scripting assistance was used during development. Dataset construction, analysis runs, and reported outputs were reviewed manually.
