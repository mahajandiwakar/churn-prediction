"""
predict.py
----------
Standalone churn prediction CLI.

Usage
-----
# Score the full dataset from the data/ folder (default — no arguments needed)
python predict.py

# Point at a folder of raw CSVs — features are generated automatically
python predict.py --from-folder /path/to/csv/folder --output predictions.csv

# Specify the snapshot date when using --from-folder (default: today)
python predict.py --from-folder data/ --snapshot-date 2025-01-01

Auto-behaviour
--------------
- If the required .pkl files are missing from --model-dir, the pipeline
  runs automatically to train and save them first.
- If --input is omitted and --from-folder is not given, all snapshots from
  the --data-dir folder are loaded and the full feature engineering pipeline
  is run before scoring.
- When --from-folder is supplied, the five raw RavenStack CSVs in that
  folder are loaded and the full feature engineering pipeline is run before
  scoring.  --snapshot-date controls the point-in-time cutoff (defaults to
  today's date).
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

import config

_REQUIRED_PKLS = ["logistic_regression.pkl", "random_forest.pkl", "scaler.pkl"]
_METRICS_FILE  = "metrics.json"


def _models_exist(model_dir: str) -> bool:
    return all(os.path.exists(os.path.join(model_dir, f)) for f in _REQUIRED_PKLS + [_METRICS_FILE])


def _auto_train(data_dir: str, output_dir: str) -> None:
    print("Models not found — running training pipeline (Parts 4+5)...")
    import data_loader
    import features
    import models

    tables = data_loader.load_data(data_dir)
    features.set_globals(tables)
    snap_cache = features.build_snapshot_cache()
    feat_cols  = features.get_feature_columns(snap_cache)

    import models as _models
    folds_df, rolling_summary = _models.run_rolling_cv(snap_cache, feat_cols)
    result = _models.train_final_models(snap_cache, feat_cols)
    result["folds_df"]        = folds_df
    result["rolling_summary"] = rolling_summary

    _models.save_models(
        result["fitted_models"],
        result["scaler"],
        feat_cols,
        result["cv_results"],
        result["folds_df"],
        result["rolling_summary"],
        output_dir=output_dir,
    )
    print()


def _load_from_folder(folder: str, snapshot_date: str) -> pd.DataFrame:
    """Load raw CSVs from *folder*, run feature engineering, and return the feature matrix."""
    import data_loader
    import features

    tables = data_loader.load_data(folder)
    features.set_globals(tables)
    df = features.build_features(snapshot_date)
    print(f"Built features for snapshot {snapshot_date!r} from {folder!r}: {len(df):,} rows")
    return df



# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def run_predict(
    model_dir: str,
    data_dir:  str,
    input_path: str | None,
    output_path: str,
    threshold_override: float | None,
    folder_path: str | None = None,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    # 1. Auto-train if needed
    if not _models_exist(model_dir):
        _auto_train(data_dir, model_dir)

    # 2. Load artefacts
    def _load(fname):
        with open(os.path.join(model_dir, fname), "rb") as f:
            return pickle.load(f)

    lr     = _load("logistic_regression.pkl")
    rf     = _load("random_forest.pkl")
    scaler = _load("scaler.pkl")

    with open(os.path.join(model_dir, _METRICS_FILE)) as f:
        metrics = json.load(f)

    meta          = metrics["metadata"]
    feat_cols     = meta["feature_columns"]
    lr_weight     = meta["ensemble_weights"]["lr"]
    rf_weight     = meta["ensemble_weights"]["rf"]
    threshold     = threshold_override if threshold_override is not None \
                    else config.DEFAULT_THRESHOLD

    # 3. Load input data
    if folder_path:
        if snapshot_date is None:
            from datetime import date
            snapshot_date = date.today().isoformat()
        input_df = _load_from_folder(folder_path, snapshot_date)
        missing  = [c for c in feat_cols if c not in input_df.columns]
        if missing:
            raise ValueError(
                f"Generated feature matrix is missing {len(missing)} expected column(s): {missing}"
            )
    elif input_path:
        input_df = pd.read_csv(input_path)
        missing  = [c for c in feat_cols if c not in input_df.columns]
        if missing:
            raise ValueError(
                f"Input CSV is missing {len(missing)} required feature column(s): {missing}"
            )
        print(f"Loaded input from {input_path!r}: {len(input_df):,} rows")
    else:
        if snapshot_date is None:
            from datetime import date
            snapshot_date = date.today().isoformat()
        input_df = _load_from_folder(data_dir, snapshot_date)
        missing = [c for c in feat_cols if c not in input_df.columns]
        if missing:
            raise ValueError(
                f"Generated feature matrix is missing {len(missing)} expected column(s): {missing}"
            )

    # 4. Score
    X      = input_df[feat_cols].copy()
    p_lr   = lr.predict_proba(scaler.transform(X))[:, 1]
    p_rf   = rf.predict_proba(X)[:, 1]
    p_ens  = lr_weight * p_lr + rf_weight * p_rf

    # 5. Apply threshold
    preds = (p_ens >= threshold).astype(int)

    # 6. Build output
    out_df = input_df.copy()
    out_df["churn_probability"] = np.round(p_ens, 6)
    out_df["churn_prediction"]  = preds

    out_df.to_csv(output_path, index=False)

    n_churners = preds.sum()
    print(f"\nScored {len(out_df):,} accounts  |  "
          f"threshold={threshold:.3f}  |  "
          f"predicted churners: {n_churners} ({n_churners / len(out_df):.1%})")
    print(f"Predictions saved to {output_path!r}")
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RavenStack churn prediction — score accounts with saved models"
    )
    parser.add_argument(
        "--model-dir", type=str, default=config.OUTPUT_DIR,
        help=f"Directory containing .pkl files and metrics.json (default: {config.OUTPUT_DIR!r})"
    )
    parser.add_argument(
        "--data-dir", type=str, default=config.DATA_DIR,
        help=f"Raw data directory, used if auto-training or loading test window "
             f"(default: {config.DATA_DIR!r})"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="CSV with pre-engineered feature columns. Omit to score the full dataset from --data-dir."
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Output CSV path (default: predictions.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Classification threshold (default: value stored in metrics.json)"
    )
    parser.add_argument(
        "--from-folder", type=str, default=None,
        help="Folder containing the five raw RavenStack CSVs. Features are generated "
             "automatically before scoring. Cannot be combined with --input."
    )
    parser.add_argument(
        "--snapshot-date", type=str, default=None,
        help="ISO date for the feature snapshot when using --from-folder "
             "(e.g. 2025-01-01). Defaults to today's date."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.from_folder and args.input:
        import sys as _sys
        print("Error: --from-folder and --input are mutually exclusive.", file=_sys.stderr)
        _sys.exit(1)
    run_predict(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        input_path=args.input,
        output_path=args.output,
        threshold_override=args.threshold,
        folder_path=args.from_folder,
        snapshot_date=args.snapshot_date,
    )
