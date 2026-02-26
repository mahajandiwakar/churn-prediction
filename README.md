# Churn Analysis

End-to-end churn prediction pipeline.
The code processes five relational tables, engineers 32 point-in-time features,
runs a rolling walk-forward cross-validation, and produces an interpretable
ensemble model with detailed diagnostics.

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/mahajandiwakar/churn-prediction.git
cd churn-prediction

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the data
# Dataset: https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset/data
cp /path/to/downloaded/csvs/*.csv data/

# 5. Run the full pipeline
python main.py

# Run specific parts only (e.g. EDA + modelling)
python main.py --parts 3 4 5 6

# Point at a different data folder
python main.py --data-dir /some/other/folder

# Save models and metrics to a custom directory
python main.py --parts 4 5 --output-dir my_models/
```

---

## The six pipeline parts

| Part | Module | What it does |
|------|--------|-------------|
| 1 | `data_loader` | Load tables, parse dates, print null audit |
| 2 | `data_loader` | Internal consistency / leakage checks |
| 3 | `eda` |  EDA plots and print summaries |
| 4 | `features` | Build monthly feature snapshots |
| 5 | `models` | Rolling walk-forward CV + final model training; saves `.pkl` and `metrics.json` |
| 6 | `models` | Diagnostic + importance |

---

## `predict.py` — scoring CLI

Standalone script that loads saved `.pkl` files and scores accounts.
No training required if models already exist in `--model-dir`.

```
python predict.py [--model-dir PATH] [--data-dir PATH]
                  [--input CSV] [--output CSV] [--threshold FLOAT]
                  [--from-folder PATH] [--snapshot-date DATE]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir PATH` | `outputs` | Directory containing `.pkl` files and `metrics.json`. |
| `--input CSV` | *(none)* | Pre-engineered feature CSV with the 32 model columns. Omit to score the full dataset from `--data-dir`. |
| `--from-folder PATH` | *(none)* | Folder containing the five raw RavenStack CSVs. Feature engineering runs automatically before scoring. Cannot be combined with `--input`. |
| `--snapshot-date DATE` | today | ISO date (e.g. `2025-01-01`) controlling the point-in-time cutoff when using `--from-folder` or the default data-dir path. |

**Auto-fallback behaviour**

- **No models found** in `--model-dir` → the pipeline (Parts 4+5) runs automatically to train and save them first.
- **No `--input` and no `--from-folder`** → the full dataset from `--data-dir` (`data/` by default) is loaded and the full feature engineering pipeline runs before scoring. Equivalent to passing `--from-folder data/`.


**Output columns**

The output CSV contains all columns from the input plus two appended columns:

| Column | Description |
|--------|-------------|
| `churn_probability` | Ensemble soft-vote score in [0, 1] — `LR×0.4 + RF×0.6` |
| `churn_prediction` | Binary label — `1` if `churn_probability ≥ threshold`, else `0` |

---

## `main.py` — CLI reference

```
python main.py [--parts N …] [--data-dir PATH] [--output-dir PATH]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--parts N …` | interactive prompt | Space-separated list of parts to run (1–6). Omit to be asked interactively. |
| `--data-dir PATH` | `data` | Folder containing the five `ravenstack_*.csv` files. |
| `--output-dir PATH` | `outputs` | Directory where `.pkl` model files and `metrics.json` are written after Part 5. |

**Examples**

```bash
# Run everything interactively
python main.py

# Feature engineering + modelling only (saves models automatically)
python main.py --parts 4 5

```

---

## Data schema (five tables)

| Table | Grain | Key columns |
|-------|-------|-------------|
| `accounts` | 1 row / account | `account_id`, `signup_date`, `churn_flag`, `plan_tier`, `industry` |
| `subscriptions` | 1 row / subscription line | `subscription_id`, `account_id`, `start_date`, `end_date`, `mrr_amount`, `seats` |
| `feature_usage` | 1 row / daily usage event | `subscription_id`, `usage_date`, `usage_count`, `error_count` |
| `support_tickets` | 1 row / ticket | `ticket_id`, `account_id`, `submitted_at`, `resolution_time_hours`, `satisfaction_score` |
| `churn_events` | 1 row / churn event | `account_id`, `churn_date`, `reason_code`, `is_reactivation` |

---

## Feature engineering

All 32 features are computed strictly from data available **before** snapshot date T.
The forward label is `did the account churn in [T, T + 30 days)?`


**Feature groups**

| Group | Count | Highlights |
|-------|-------|-----------|
| Account Lifecycle | 6 | `tenure_days`, `months_since_upgrade` |
| Commercial | 4 | `mrr_at_snapshot`, `engagement_per_seat` |
| Usage 30d Windows | 7 | `usage_decline_pct`, `went_dark`, `error_rate_change` |
| Usage 90–180d | 4 | `ratio_30_180`  |
| Support | 3 | `tickets_last_30`, `satisfaction_score` |
| Subscription | 4 | `recent_downgrade`, `raw error counts` |
| Historical | 1 | `n_prior_churns`  |
| Static | 3 | `industry_enc`, `referral_source_enc`, `plan_tier_enc` |

---

## Modelling approach

### Why rolling walk-forward CV?

A single train/test split produces one number that depends on whether the test
months were easy or hard.  The rolling distribution — 8 folds expanding from
April 2024 through November 2024 — shows the expected performance envelope
across the full range of months the model will face in production.

### Model choice

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Linear constraint prevents overfitting on the small (~500 row) dataset |
| Random Forest | Captures non-linear feature interactions |
| Gradient Boosting | Strong baseline; used for comparison |
| Ensemble (LR×0.4 + RF×0.6) | combines LR's regularisation with RF's expressivity |

### Final results (Oct–Nov 2024 test window)

| Model | ROC-AUC | PR-AUC | 
|-------|---------|--------|
| Logistic Regression | ~0.59 | ~0.18 | 
| Random Forest | ~0.61 | ~0.19 |
| Gradient Boosting | ~0.59 | ~0.18 |
| Ensemble | ~0.615** | ~0.196** | 

> Exact numbers vary slightly with each run due to permutation importance randomness. The ensemble consistently leads on PR-AUC.

**Class imbalance**: 6.8% churn rate (13.7:1 ratio). `class_weight='balanced'` is applied to LR and RF; `sample_weight` to GBM.

---

## License

MIT
