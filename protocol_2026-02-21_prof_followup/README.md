## Main files

- `config.py`
  - Shared paths and constants.
- `analysis_utils.py`
  - Loading helpers and text-processing utilities.
- `custom_stopwords.txt`
  - Additional stop words used by the word-frequency analyses.
- `01_collect_positive_reviews.py`
  - Builds or refreshes the positive-review comparison sets.
- `02_wordclouds_pos_vs_neg.py`
  - Creates the overall and group-level word-frequency visualizations.
- `03_odds_ratios_chi_square.py`
  - Runs the document-level contingency analysis with odds ratios, chi-square tests, and FDR correction.
- `04_threshold_sensitivity.py`
  - Evaluates how the keyword-rate results change across review-count thresholds.
- `05_variance_bootstrap.py`
  - Estimates uncertainty with cluster bootstrap sampling at the game level.
- `08_odds_ratio_wordclouds.py`
  - Creates the combined odds-ratio word clouds for the significant positive/negative term comparison.
- `06*`
  - Candidate-search and expansion scripts for the group-update steps.
- `07_ai_filter_keyword_wordclouds.py`
  - Produces AI-focused keyword summaries and related visualizations.
- `requirements.txt`
  - Minimal dependency list for the follow-up scripts.

## Data and outputs

- `data/`
  - Processed review data and group-level follow-up inputs.
- `outputs/`
  - Derived tables and figures produced by the follow-up analyses.

## Development note

Some scripting assistance was used during development. Data checks, runs, and reported outputs were reviewed manually.
