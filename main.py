# main.py — RavenStack churn analysis pipeline
#
# python main.py                 # run everything
# python main.py --parts 1 3 4  # run specific parts
#
# Set DATA_DIR in config.py before running.

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
import data_loader
import eda
import features
import models


# ---------------------------------------------------------------------------
# Individual part runners
# ---------------------------------------------------------------------------

def run_part1(tables: dict) -> None:
    """Data loading, null audit, and class distribution summary."""
    print("\n-- Part 1: Data Loading & Quality --")
    data_loader.audit_nulls(tables)
    data_loader.print_target_summary(tables["accounts"])


def run_part2(tables: dict) -> None:
    """Four internal consistency / leakage checks."""
    print("\n-- Part 2: Flag Investigation & Consistency Checks --")
    data_loader.audit_flags(tables)


def run_part3(tables: dict) -> None:
    """Full EDA: demographics, revenue, behavioural signals."""
    print("\n-- Part 3: EDA --")
    accts = tables["accounts"]
    subs  = tables["subscriptions"]
    fu    = tables["feature_usage"]
    ce    = tables["churn_events"]
    st    = tables["support_tickets"]

    eda.plot_churn_by_industry(accts)
    eda.plot_churn_by_channel(accts)
    eda.plot_churn_by_tenure(accts)
    eda.print_revenue_impact(accts, subs)
    eda.plot_revenue_segments(accts, subs)
    eda.plot_behavioural_signals(accts, subs, fu, ce)
    eda.print_usage_summary(accts, subs, fu)
    eda.plot_support_analysis(accts, st)


def run_part4(tables: dict) -> tuple[dict, list[str]]:
    """Build and cache feature matrices for all 11 monthly snapshots."""
    print("\n-- Part 4: Feature Engineering --")
    features.set_globals(tables)

    snap_cache = features.build_snapshot_cache()
    feat_cols  = features.get_feature_columns(snap_cache)
    print(f"\n{len(feat_cols)} features: {feat_cols}")
    return snap_cache, feat_cols


def run_part5(snap_cache: dict, feat_cols: list[str], output_dir: str = config.OUTPUT_DIR) -> dict:
    """Rolling walk-forward CV + final model training on Jan–Sep window."""
    print("\n-- Part 5: Modelling --")

    folds_df, rolling_summary = models.run_rolling_cv(snap_cache, feat_cols)
    result = models.train_final_models(snap_cache, feat_cols)

    # sanity check: is the fixed split in line with the rolling range?
    ens_roc   = result["cv_results"]["Ensemble (LR×0.4 + RF×0.6)"]["roc"]
    roll_mean = rolling_summary["Ensemble"]["mean_roc"]
    roll_std  = rolling_summary["Ensemble"]["std_roc"]
    print(f"\nEnsemble Oct–Nov ROC: {ens_roc:.4f} | "
          f"Rolling mean±std: {roll_mean:.4f}±{roll_std:.4f}")

    # Store rolling artefacts for Part 6
    result["folds_df"]        = folds_df
    result["rolling_summary"] = rolling_summary

    print(f"\n-- Saving models and metrics to {output_dir!r} --")
    models.save_models(
        result["fitted_models"],
        result["scaler"],
        feat_cols,
        result["cv_results"],
        result["folds_df"],
        result["rolling_summary"],
        output_dir=output_dir,
    )
    return result


def run_part6(result: dict, feat_cols: list[str]) -> None:
    """Diagnostics (D1) + permutation importance interpretability."""
    print("\n-- Part 6: Diagnostics & Interpretability --")

    models.plot_diagnostic_d1(
        result["folds_df"],
        result["cv_results"],
        result["rolling_summary"],
    )

    pi = models.compute_permutation_importance(
        result["fitted_models"],
        result["X_test"],
        result["y_test"],
        feat_cols,
    )
    print("\nTop-5 features by permutation importance:")
    for _, row in pi.head(5).iterrows():
        print(f"  {row['label']:<40} Δ ROC = {row['mean']:+.4f} ± {row['std']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PART_DESCRIPTIONS = {
    1: "Data Loading & Quality",
    2: "Flag Investigation & Consistency Checks",
    3: "Exploratory Data Analysis",
    4: "Feature Engineering",
    5: "Predictive Modelling",
    6: "Diagnostics & Interpretability",
}


def prompt_parts() -> set[int]:
    print("\nAvailable parts:")
    for num, desc in PART_DESCRIPTIONS.items():
        print(f"  {num}: {desc}")
    raw = input("\nParts to run (e.g. 1 2 3), or Enter for all: ").strip()
    if not raw:
        return set(PART_DESCRIPTIONS.keys())
    chosen = {int(x) for x in raw.split() if x.isdigit()} & set(PART_DESCRIPTIONS.keys())
    return chosen or set(PART_DESCRIPTIONS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RavenStack churn analysis — end-to-end pipeline"
    )
    parser.add_argument(
        "--parts", nargs="*", type=int,
        default=None,
        help="Which parts to run (default: ask interactively).  Example: --parts 4 5 6"
    )
    parser.add_argument(
        "--data-dir", type=str, default=config.DATA_DIR,
        help=f"Folder containing the five CSV files (default: {config.DATA_DIR!r})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=config.OUTPUT_DIR,
        help=f"Directory for saved models and metrics (default: {config.OUTPUT_DIR!r})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("RavenStack Churn Analysis")

    if args.parts is None:
        parts = prompt_parts()
    else:
        parts = set(args.parts)

    print(f"Running parts: {sorted(parts)}")
    print(f"Data directory: {args.data_dir!r}")

    # Always load data first (needed by every part)
    tables = data_loader.load_data(args.data_dir)

    snap_cache, feat_cols, result = None, None, None

    if 1 in parts:
        run_part1(tables)

    if 2 in parts:
        run_part2(tables)

    if 3 in parts:
        run_part3(tables)

    if 4 in parts or 5 in parts or 6 in parts:
        snap_cache, feat_cols = run_part4(tables)

    if 5 in parts or 6 in parts:
        if snap_cache is None or feat_cols is None:
            snap_cache, feat_cols = run_part4(tables)
        result = run_part5(snap_cache, feat_cols, output_dir=args.output_dir)

    if 6 in parts:
        if result is None:
            raise RuntimeError("Part 6 requires Part 5 to run first. Add 5 to --parts.")
        run_part6(result, feat_cols)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
