"""
models.py
---------
Predictive modelling for the RavenStack churn dataset.

Three sections
--------------
  1. Rolling walk-forward cross-validation
     Expanding-window CV across 11 monthly snapshots.
     Primary performance claim: mean ± std ROC-AUC and PR-AUC over
     7 stable folds (April excluded — only 3 training snapshots).

  2. Final model training
     Train on all 9 Jan–Sep snapshots; evaluate on Oct–Nov test window.
     Three base models (LR, RF, GBM) + a soft-vote ensemble (LR×0.4 + RF×0.6).

  3. Interpretability
     Permutation importance on the held-out test set (correct metric for
     detecting features that generalise vs. fit training quirks).
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection         import permutation_importance

import config


def run_rolling_cv(
    snap_cache: dict[str, pd.DataFrame],
    features:   list[str],
    verbose:    bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Expanding-window walk-forward cross-validation across all snapshots.

    Fold k trains on snapshots[0..k-1] and tests on snapshots[k].
    Trains four models per fold: LR, RF, GBM, and an LR+RF soft-vote ensemble.

    Parameters
    ----------
    snap_cache : dict[str, pd.DataFrame]
        Output of ``features.build_snapshot_cache()``.
    features : list[str]
        Column names to use as model inputs.
    verbose : bool
        Print per-fold results if True.

    Returns
    -------
    folds_df : pd.DataFrame
        One row per fold with test_month, n_train, ROC-AUC and PR-AUC columns
        for every model.
    rolling_summary : dict
        Mean / std ROC and PR for each model over stable folds (n_train ≥ 4).
    """
    all_snaps = config.ALL_SNAPSHOTS
    min_train = config.MIN_CV_TRAIN

    if verbose:
        print("\nRunning rolling walk-forward CV across all 11 snapshots...")
        print(f"{'Fold':>6}  {'Train':>5}  {'Pos':>5}  "
              f"{'LR ROC':>8}  {'RF ROC':>8}  {'GBM ROC':>8}  {'Ens ROC':>8}")
        print("─" * 70)

    fold_records = []

    for i in range(min_train, len(all_snaps)):
        train_snaps = all_snaps[:i]
        test_snap   = all_snaps[i]

        tr   = pd.concat([snap_cache[d] for d in train_snaps], ignore_index=True)
        te   = snap_cache[test_snap]
        X_tr, y_tr = tr[features], tr["churn_flag"]
        X_te, y_te = te[features], te["churn_flag"]

        # Scale for LR only (fit on train fold, never on test)
        sc        = StandardScaler()
        X_tr_sc   = sc.fit_transform(X_tr)
        X_te_sc   = sc.transform(X_te)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
        )
        lr.fit(X_tr_sc, y_tr)
        p_lr = lr.predict_proba(X_te_sc)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=config.RANDOM_STATE,
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]

        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            random_state=config.RANDOM_STATE,
        )
        gb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        p_gb = gb.predict_proba(X_te)[:, 1]

        p_ens = config.LR_WEIGHT * p_lr + config.RF_WEIGHT * p_rf

        row = {"test_month": test_snap[:7], "n_train": len(train_snaps),
               "n_pos": int(y_te.sum()), "n_tot": len(y_te)}
        for model_name, probs in [("LR", p_lr), ("RF", p_rf), ("GBM", p_gb), ("Ensemble", p_ens)]:
            row[f"{model_name}_roc"] = roc_auc_score(y_te, probs)
            row[f"{model_name}_pr"]  = average_precision_score(y_te, probs)
        fold_records.append(row)

        if verbose:
            print(f"{test_snap[:7]:>6}  {len(train_snaps):>5}  {int(y_te.sum()):>5}  "
                  f"{row['LR_roc']:>8.4f}  {row['RF_roc']:>8.4f}  "
                  f"{row['GBM_roc']:>8.4f}  {row['Ensemble_roc']:>8.4f}")

    folds_df = pd.DataFrame(fold_records)
    stable   = folds_df[folds_df["n_train"] >= 4]

    rolling_summary = {
        m: {
            "mean_roc": stable[f"{m}_roc"].mean(),
            "std_roc":  stable[f"{m}_roc"].std(),
            "mean_pr":  stable[f"{m}_pr"].mean(),
            "std_pr":   stable[f"{m}_pr"].std(),
        }
        for m in ["LR", "RF", "GBM", "Ensemble"]
    }

    if verbose:
        _print_cv_summary(stable, rolling_summary, folds_df, min_train)

    return folds_df, rolling_summary


class _NpEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays to plain Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# 2. Final model training + evaluation
# ---------------------------------------------------------------------------
def train_final_models(
    snap_cache: dict[str, pd.DataFrame],
    features:   list[str],
    verbose:    bool = True,
) -> dict:
    """
    Train three base models + ensemble on all training snapshots.
    Evaluate on the held-out Oct–Nov test window.

    Parameters
    ----------
    snap_cache : dict[str, pd.DataFrame]
        Output of ``features.build_snapshot_cache()``.
    features : list[str]
        Feature column names.
    verbose : bool
        Print a results table if True.

    Returns
    -------
    dict with keys:
        fitted_models  — {name: fitted sklearn estimator}
        cv_results     — {name: {probs, roc, pr_auc, f1, threshold}}
        X_train, y_train, X_test, y_test
        X_train_sc, X_test_sc  — scaled versions (LR only)
        scaler         — fitted StandardScaler
        best_name      — model name with highest PR-AUC
    """
    train_df = pd.concat([snap_cache[d] for d in config.TRAIN_SNAPSHOTS], ignore_index=True)
    test_df  = pd.concat([snap_cache[d] for d in config.TEST_SNAPSHOTS],  ignore_index=True)

    X_train, y_train = train_df[features], train_df["churn_flag"]
    X_test,  y_test  = test_df[features],  test_df["churn_flag"]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit on train only
    X_test_sc  = scaler.transform(X_test)

    if verbose:
        print(f"\nTrain: {len(X_train):,} rows | {y_train.sum():.0f} churners ({y_train.mean():.1%})")
        print(f"Test:  {len(X_test):,} rows  | {y_test.sum():.0f} churners  ({y_test.mean():.1%})")
        print(f"\n{'Model':<28} {'ROC-AUC':>9} {'PR-AUC':>8} {'Best F1':>9} {'Threshold':>11}")
        print("-" * 68)

    base_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=config.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            random_state=config.RANDOM_STATE),
    }

    cv_results: dict    = {}
    fitted_models: dict = {}

    for name, mdl in base_models.items():
        Xtr = X_train_sc if name == "Logistic Regression" else X_train
        Xte = X_test_sc  if name == "Logistic Regression" else X_test
        sw  = compute_sample_weight("balanced", y_train) if name == "Gradient Boosting" else None
        kw  = {"sample_weight": sw} if sw is not None else {}

        mdl.fit(Xtr, y_train, **kw)
        fitted_models[name] = mdl

        probs              = mdl.predict_proba(Xte)[:, 1]
        roc                = roc_auc_score(y_test, probs)
        pr                 = average_precision_score(y_test, probs)
        prec_c, rec_c, thr = precision_recall_curve(y_test, probs)
        f1_c               = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-6)
        best_i             = np.argmax(f1_c)
        best_th            = thr[best_i] if best_i < len(thr) else thr[-1]

        cv_results[name] = {"probs": probs, "roc": roc, "pr_auc": pr,
                            "f1": f1_c[best_i], "threshold": best_th}
        if verbose:
            print(f"  {name:<26} {roc:>9.4f} {pr:>8.4f} {f1_c[best_i]:>9.4f} {best_th:>11.4f}")

    # Soft-vote ensemble
    p_lr  = cv_results["Logistic Regression"]["probs"]
    p_rf  = cv_results["Random Forest"]["probs"]
    p_ens = config.LR_WEIGHT * p_lr + config.RF_WEIGHT * p_rf

    ens_roc  = roc_auc_score(y_test, p_ens)
    ens_pr   = average_precision_score(y_test, p_ens)
    prec_e, rec_e, thr_e = precision_recall_curve(y_test, p_ens)
    f1_e     = 2 * (prec_e * rec_e) / (prec_e + rec_e + 1e-6)
    best_i_e = np.argmax(f1_e)
    best_t_e = thr_e[best_i_e] if best_i_e < len(thr_e) else thr_e[-1]

    ens_name = "Ensemble (LR×0.4 + RF×0.6)"
    cv_results[ens_name] = {"probs": p_ens, "roc": ens_roc, "pr_auc": ens_pr,
                            "f1": f1_e[best_i_e], "threshold": best_t_e}
    if verbose:
        print(f"  {ens_name:<26} {ens_roc:>9.4f} {ens_pr:>8.4f} "
              f"{f1_e[best_i_e]:>9.4f} {best_t_e:>11.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["pr_auc"])
    if verbose:
        print(f"\n  Best model (PR-AUC): {best_name}")
        _print_band_evaluation(cv_results, y_test)

    return {
        "fitted_models": fitted_models, "cv_results": cv_results,
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
        "scaler": scaler, "best_name": best_name,
    }


def save_models(
    fitted_models:   dict,
    scaler,
    feat_cols:       list[str],
    cv_results:      dict,
    folds_df:        pd.DataFrame,
    rolling_summary: dict,
    output_dir:      str = "outputs",
) -> None:
    """
    Persist fitted models, scaler, and all metrics to disk.

    Outputs
    -------
    <output_dir>/logistic_regression.pkl
    <output_dir>/random_forest.pkl
    <output_dir>/gradient_boosting.pkl
    <output_dir>/scaler.pkl
    <output_dir>/metrics.json
    """
    os.makedirs(output_dir, exist_ok=True)

    name_to_file = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "Gradient Boosting":   "gradient_boosting.pkl",
    }
    for name, fname in name_to_file.items():
        path = os.path.join(output_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(fitted_models[name], f)
        print(f"  Saved {path}")

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved {scaler_path}")

    ens_name = "Ensemble (LR×0.4 + RF×0.6)"
    metrics = {
        "metadata": {
            "feature_columns":   feat_cols,
            "ensemble_weights":  {"lr": config.LR_WEIGHT, "rf": config.RF_WEIGHT},
            "default_threshold": float(cv_results[ens_name]["threshold"]),
            "train_snapshots":   config.TRAIN_SNAPSHOTS,
            "test_snapshots":    config.TEST_SNAPSHOTS,
        },
        "rolling_cv": {
            "summary": rolling_summary,
            "folds":   folds_df.to_dict("records"),
        },
        "final_test": {
            name: {
                "roc_auc":   res["roc"],
                "pr_auc":    res["pr_auc"],
                "f1":        res["f1"],
                "threshold": res["threshold"],
            }
            for name, res in cv_results.items()
        },
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=_NpEncoder)
    print(f"  Saved {metrics_path}")


def plot_diagnostic_d1(
    folds_df:        pd.DataFrame,
    cv_results:      dict,
    rolling_summary: dict,
) -> None:
    """
    Diagnostic D1: overlay the fixed Oct–Nov split result on the rolling
    CV distribution to check whether the test window was representative.

    A ★ falling inside the rolling box confirms the test split is neither
    lucky nor unlucky — it represents the expected production range.
    """
    plt.rcParams.update(config.PLOT_STYLE)
    stable   = folds_df[folds_df["n_train"] >= 4]
    name_map = {
        "LR":       "Logistic Regression",
        "RF":       "Random Forest",
        "GBM":      "Gradient Boosting",
        "Ensemble": "Ensemble (LR×0.4 + RF×0.6)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("D1 — Fixed Split vs Rolling Distribution",
                 fontsize=13, fontweight="bold", color=config.NAVY, y=1.01)

    for ax, metric, label in zip(axes, ["_roc", "_pr"], ["ROC-AUC", "PR-AUC"]):
        cv_key = "roc" if metric == "_roc" else "pr_auc"
        for idx, (m, color) in enumerate(config.MODEL_COLORS.items()):
            vals = stable[f"{m}{metric}"].values
            ax.boxplot(vals, positions=[idx], patch_artist=True, widths=0.4,
                       boxprops=dict(facecolor=color, alpha=0.5),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=color), capprops=dict(color=color),
                       flierprops=dict(marker="o", color=color, alpha=0.4))
            fixed = cv_results[name_map[m]][cv_key]
            ax.scatter([idx], [fixed], marker="*", s=200, color=color,
                       zorder=5, edgecolors="white", linewidth=1)
        ax.set_xticks(range(len(config.MODEL_COLORS)))
        ax.set_xticklabels(list(config.MODEL_COLORS.keys()), fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: rolling box + fixed split (★)",
                     fontsize=11, fontweight="bold", color=config.NAVY)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"\n{'Model':<12}  {'Rolling mean':>13} {'Rolling std':>12}  "
          f"{'Fixed split':>12} Verdict")
    print("─" * 78)
    for m in config.MODEL_COLORS:
        r_mean = rolling_summary[m]["mean_roc"]
        r_std  = rolling_summary[m]["std_roc"]
        fixed  = cv_results[name_map[m]]["roc"]
        z      = (fixed - r_mean) / r_std if r_std > 0 else 0
        verdict = "representative" if abs(z) < 1.0 else ("above avg" if z > 0 else "below avg")
        print(f"  {m:<10}  {r_mean:>13.4f} {r_std:>12.4f}  {fixed:>12.4f}  {z:>+9.2f}  {verdict}")
    print("\n★ = fixed split result.  Within the rolling box → representative test window.")

def compute_permutation_importance(
    fitted_models: dict,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    features:      list[str],
    n_repeats:     int = 10,
) -> pd.DataFrame:
    """
    Compute permutation importance for the Random Forest on the held-out test set.

    Permutation importance measures the drop in test ROC-AUC when a feature
    is randomly shuffled.  It captures held-out contribution (not training
    influence), so it correctly identifies features that generalise vs.
    those that merely overfit training patterns.

    Parameters
    ----------
    fitted_models : dict
        Output of ``train_final_models()["fitted_models"]``.
    X_test, y_test
        Held-out test features and labels.
    features : list[str]
        Feature column names in the same order used during training.
    n_repeats : int
        Number of shuffle repetitions per feature (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: feature, label, mean, std — sorted by mean descending.
    """
    rf_model = fitted_models["Random Forest"]
    perm = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=n_repeats, scoring="roc_auc",
        random_state=config.RANDOM_STATE,
    )
    pi = pd.DataFrame({
        "feature": features,
        "label":   [config.FEATURE_LABELS.get(f, f) for f in features],
        "mean":    perm.importances_mean,
        "std":     perm.importances_std,
    }).sort_values("mean", ascending=False).reset_index(drop=True)

    def _signal_tier(m: float) -> str:
        if m > 0.020: return "***"
        if m > 0.008: return "**"
        if m > 0.002: return "*"
        if m > 0:     return "-"
        return "nil"

    print(f"\n{'Rank':<5} {'Feature':<35} {'Delta ROC':>10} {'Std':>7}  Signal")
    print("-" * 65)
    for i, row in pi.head(15).iterrows():
        print(f"  {i+1:<3} {row['label']:<35} {row['mean']:>+10.4f} "
              f"{row['std']:>7.4f}  {_signal_tier(row['mean'])}")
    print("\nSignal key: *** >0.020  ** >0.008  * >0.002  - marginal  nil ≤0")

    return pi

def _print_cv_summary(
    stable:          pd.DataFrame,
    rolling_summary: dict,
    folds_df:        pd.DataFrame,
    min_train:       int,
) -> None:
    print(f"\n{'Model':<12}  {'Mean ROC':>9} {'±Std':>7} {'Min':>7} {'Max':>7}  "
          f"{'Mean PR':>9} {'±Std':>7}")
    print("─" * 68)
    for m in ["LR", "RF", "GBM", "Ensemble"]:
        r = stable[f"{m}_roc"]
        p = stable[f"{m}_pr"]
        print(f"  {m:<10}  {r.mean():>9.4f} {r.std():>7.4f} "
              f"{r.min():>7.4f} {r.max():>7.4f}  {p.mean():>9.4f} {p.std():>7.4f}")


def _print_band_evaluation(cv_results: dict, y_test: pd.Series) -> None:
    """Print top-20 % precision, capture rate, and lift for every model."""
    print(f"\n{'Model':<30} {'Precision@20%':>14} {'Capture@20%':>12} {'Lift':>7}")
    print("-" * 68)
    for name, res in cv_results.items():
        n_top     = len(y_test) // 5
        top_idx   = np.argsort(res["probs"])[::-1][:n_top]
        prec_top  = y_test.reset_index(drop=True).iloc[top_idx].mean()
        cap_top   = y_test.reset_index(drop=True).iloc[top_idx].sum() / y_test.sum()
        lift_top  = prec_top / y_test.mean()
        print(f"  {name:<28} {prec_top:>14.3f} {cap_top:>12.3f} {lift_top:>7.2f}x")
