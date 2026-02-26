"""
features.py
-----------
Point-in-time feature engineering for churn model.

The central function ``build_features(snapshot_date)`` returns one row per
account.  Every value it computes uses only data available *before* the
snapshot date T, so the same function can be used for:

  • Historical back-testing / rolling walk-forward CV
  • Final model training
  • Production daily scoring (call with today's date)

Feature groups (32 features total)
------------------------------------
  Account Lifecycle   — tenure, plan history, trial flag
  Commercial          — MRR, seats, active subscription count
  Usage 30d Windows   — last-30 vs prev-30 counts, decline %, error rate
  Usage 90-180d       — 90d total, 3-month slope, volatility, ratio_30_180
  Support             — ticket count, resolution time, satisfaction score
  Subscription        — recent downgrade, raw error counts, seat snapshot
  Historical          — n_prior_churns (repeat-churner signal)
  Static              — industry, referral source, plan tier (encoded)

Global data dependencies
-------------------------
The function reads from five DataFrames that must be loaded before calling
``build_snapshot_cache``:

  accounts, subscriptions, feature_usage, support_tickets, churn_events

Pass them via ``set_globals()`` before any call to ``build_features()``.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

_accounts:        pd.DataFrame | None = None
_subscriptions:   pd.DataFrame | None = None
_feature_usage:   pd.DataFrame | None = None
_support_tickets: pd.DataFrame | None = None
_churn_events:    pd.DataFrame | None = None


def set_globals(tables: dict[str, pd.DataFrame]) -> None:
    """
    Inject the five data tables into this module's namespace.

    Call this once after ``data_loader.load_data()`` before using
    ``build_features`` or ``build_snapshot_cache``.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        The dict returned by ``data_loader.load_data()``.
    """
    global _accounts, _subscriptions, _feature_usage, _support_tickets, _churn_events
    _accounts        = tables["accounts"]
    _subscriptions   = tables["subscriptions"]
    _feature_usage   = tables["feature_usage"]
    _support_tickets = tables["support_tickets"]
    _churn_events    = tables["churn_events"]


# ---------------------------------------------------------------------------
# Core: single-snapshot feature matrix
# ---------------------------------------------------------------------------
def build_features(snapshot_date: str) -> pd.DataFrame:
    """
    Build a point-in-time feature matrix for every account.

    All features are computed from data strictly before *snapshot_date* T.
    The forward label is whether the account churned in [T, T + 30d).

    Parameters
    ----------
    snapshot_date : str
        ISO date string, e.g. ``"2024-06-01"``.

    Returns
    -------
    pd.DataFrame
        One row per account (500 rows for the RavenStack dataset).
        Columns: account_id | <32 features> | churn_flag | snapshot_date.
    """
    if _accounts is None:
        raise RuntimeError("Call set_globals(tables) before build_features().")

    T = pd.Timestamp(snapshot_date)

    
    churn_label = (
        _churn_events[
            (_churn_events["churn_date"] >= T) &
            (_churn_events["churn_date"] <  T + pd.Timedelta(days=config.PRED_DAYS))
        ][["account_id"]].drop_duplicates().assign(churn_flag=1)
    )
    labels = _accounts[["account_id"]].merge(churn_label, on="account_id", how="left")
    labels["churn_flag"] = labels["churn_flag"].fillna(0).astype(int)

    # Pre-join feature_usage → account_id once (avoids repeated lookups)
    u = _feature_usage.merge(
        _subscriptions[["subscription_id", "account_id"]], on="subscription_id", how="left"
    )

    
    ul30 = (
        u[(u["usage_date"] >= T - pd.Timedelta(days=30)) & (u["usage_date"] < T)]
        .groupby("account_id")
        .agg({"usage_count": "sum", "usage_duration_secs": "sum", "error_count": "sum"})
        .reset_index()
        .rename(columns={
            "usage_count":        "usage_count_last_30",
            "usage_duration_secs":"usage_duration_last_30",
            "error_count":        "error_count_last_30",
        })
    )
    up30 = (
        u[(u["usage_date"] >= T - pd.Timedelta(days=60)) &
          (u["usage_date"] <  T - pd.Timedelta(days=30))]
        .groupby("account_id")
        .agg({"usage_count": "sum", "error_count": "sum"})
        .reset_index()
        .rename(columns={"usage_count": "usage_count_prev_30",
                         "error_count": "error_count_prev_30"})
    )

    uf = ul30.merge(up30, on="account_id", how="left").fillna(0)

    uf["usage_decline_pct"]  = (
        (uf["usage_count_last_30"] - uf["usage_count_prev_30"])
        / (uf["usage_count_prev_30"] + 1)
    )
    uf["error_rate_last_30"] = uf["error_count_last_30"] / uf["usage_count_last_30"].clip(lower=1)
    uf["error_rate_change"]  = (
        uf["error_rate_last_30"]
        - (uf["error_count_prev_30"] / uf["usage_count_prev_30"].clip(lower=1))
    )
    # went_dark: had activity in 30–60d window but zero in last 30d
    uf["went_dark"] = (
        (uf["usage_count_last_30"] == 0) & (uf["usage_count_prev_30"] > 0)
    ).astype(int)

    # Engagement per seat (latest subscription at T)
    latest_subs = (
        _subscriptions[_subscriptions["start_date"] < T]
        .sort_values("start_date").groupby("account_id").tail(1)[["account_id", "seats"]]
    )
    uf = uf.merge(latest_subs, on="account_id", how="left")
    uf["seats"]               = uf["seats"].fillna(1)
    uf["engagement_per_seat"] = uf["usage_count_last_30"] / uf["seats"]
    ul90 = (
        u[(u["usage_date"] >= T - pd.Timedelta(days=90)) & (u["usage_date"] < T)]
        .groupby("account_id")["usage_count"].sum().reset_index()
        .rename(columns={"usage_count": "usage_count_last_90"})
    )
    uf = uf.merge(ul90, on="account_id", how="left").fillna(0)

    def _month_bucket(days_start: int, days_end: int, col: str) -> pd.DataFrame:
        return (
            u[(u["usage_date"] >= T - pd.Timedelta(days=days_end)) &
              (u["usage_date"] <  T - pd.Timedelta(days=days_start))]
            .groupby("account_id")["usage_count"].sum().reset_index()
            .rename(columns={"usage_count": col})
        )

    m0    = _month_bucket(0,  30,  "_m0")
    m1    = _month_bucket(30, 60,  "_m1")
    m2    = _month_bucket(60, 90,  "_m2")
    ul180 = (
        u[(u["usage_date"] >= T - pd.Timedelta(days=180)) & (u["usage_date"] < T)]
        .groupby("account_id")["usage_count"].sum().reset_index()
        .rename(columns={"usage_count": "_u180"})
    )

    slope_df = (
        _accounts[["account_id"]]
        .merge(m0,    on="account_id", how="left")
        .merge(m1,    on="account_id", how="left")
        .merge(m2,    on="account_id", how="left")
        .merge(ul180, on="account_id", how="left")
        .fillna(0)
    )
    slope_df["usage_3m_slope"]   = (slope_df["_m0"] - slope_df["_m2"]) / 2
    slope_df["usage_volatility"] = slope_df[["_m0", "_m1", "_m2"]].std(axis=1)
    slope_df["ratio_30_180"] = slope_df["_m0"] / (slope_df["_u180"] + 1)

    uf = uf.merge(
        slope_df[["account_id", "usage_3m_slope", "usage_volatility", "ratio_30_180"]],
        on="account_id", how="left",
    ).fillna(0)

    tl30 = (
        _support_tickets[
            (_support_tickets["submitted_at"] >= T - pd.Timedelta(days=30)) &
            (_support_tickets["submitted_at"] <  T)
        ]
        .groupby("account_id")
        .agg({"ticket_id": "count", "resolution_time_hours": "mean",
              "satisfaction_score": "mean"})
        .reset_index()
        .rename(columns={"ticket_id": "tickets_last_30"})
    )
    tf = tl30.fillna(0)

    recent_downgrade = (
        _subscriptions[
            (_subscriptions["downgrade_flag"] == 1) &
            (_subscriptions["start_date"] >= T - pd.Timedelta(days=60)) &
            (_subscriptions["start_date"] <  T)
        ]
        .groupby("account_id").size().reset_index(name="recent_downgrade")
    )
    recent_downgrade["recent_downgrade"] = 1

    paid_mrr = (
        _subscriptions[
            (_subscriptions["start_date"] < T) &
            (_subscriptions["end_date"].isna() | (_subscriptions["end_date"] >= T)) &
            (_subscriptions["is_trial"] == False)   # noqa: E712
        ]
        .groupby("account_id").agg(
            mrr_at_snapshot   = ("mrr_amount",      "sum"),
            seats_at_snapshot = ("seats",           "sum"),
            n_active_subs     = ("subscription_id", "count"),
        ).reset_index()
    )
    
    last_upgrade = (
        _subscriptions[
            (_subscriptions["upgrade_flag"] == 1) &
            (_subscriptions["start_date"] < T)
        ]
        .groupby("account_id")["start_date"].max().reset_index()
        .rename(columns={"start_date": "last_upgrade_date"})
    )
    last_upgrade["months_since_upgrade"] = (
        (T - last_upgrade["last_upgrade_date"]).dt.days / 30
    ).clip(upper=24)

    plan_changes = (
        _subscriptions[
            (_subscriptions["start_date"] < T) &
            ((_subscriptions["upgrade_flag"] == 1) | (_subscriptions["downgrade_flag"] == 1))
        ]
        .groupby("account_id").size().reset_index(name="total_plan_changes")
    )

    current_plan = (
        _subscriptions[_subscriptions["start_date"] < T]
        .sort_values("start_date").groupby("account_id").tail(1)[["account_id", "start_date"]]
    )
    current_plan["days_on_current_plan"] = (T - current_plan["start_date"]).dt.days

    n_prior_churns = (
        _churn_events[_churn_events["churn_date"] < T]
        .groupby("account_id").size().reset_index(name="n_prior_churns")
    )

    acct = _accounts.copy()
    acct["tenure_days"] = (T - acct["signup_date"]).dt.days.clip(lower=0)
    acct["tenure_bucket"] = np.select(
        [acct["tenure_days"] <= 90,
         acct["tenure_days"] <= 180,
         acct["tenure_days"] <= 365],
        [0, 1, 2], default=3,
    )
    le = LabelEncoder()
    for col in ["industry", "referral_source", "plan_tier"]:
        acct[col + "_enc"] = le.fit_transform(acct[col].astype(str))

    df = (
        acct[["account_id", "tenure_days", "tenure_bucket", "is_trial",
              "industry_enc", "referral_source_enc", "plan_tier_enc"]]
        .merge(uf,                                                on="account_id", how="left")
        .merge(tf,                                                on="account_id", how="left")
        .merge(recent_downgrade,                                  on="account_id", how="left")
        .merge(paid_mrr,                                          on="account_id", how="left")
        .merge(last_upgrade[["account_id", "months_since_upgrade"]], on="account_id", how="left")
        .merge(plan_changes,                                      on="account_id", how="left")
        .merge(current_plan[["account_id", "days_on_current_plan"]], on="account_id", how="left")
        .merge(n_prior_churns,                                    on="account_id", how="left")
        .merge(labels,                                            on="account_id", how="left")
        .fillna(0)
    )

    # Accounts that never upgraded: months_since_upgrade = 0 is misleading.
    # Set to full tenure so the model reads "never grown their plan".
    never_upgraded = df["months_since_upgrade"] == 0
    df.loc[never_upgraded, "months_since_upgrade"] = df.loc[never_upgraded, "tenure_days"] / 30

    df["snapshot_date"] = snapshot_date
    return df

def build_snapshot_cache(
    snapshots: list[str] | None = None,
    verbose:   bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Pre-build feature matrices for every snapshot date and cache them.

    Calling ``build_features`` once per snapshot up-front is much faster
    than rebuilding inside each CV fold, and guarantees every fold sees
    exactly the same data.

    Parameters
    ----------
    snapshots : list[str], optional
        Snapshot dates to build.  Defaults to ``config.ALL_SNAPSHOTS``.
    verbose : bool
        Print progress if True.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps snapshot date string → feature matrix DataFrame.
    """
    if snapshots is None:
        snapshots = config.ALL_SNAPSHOTS

    if verbose:
        print(f"Building snapshot cache ({len(snapshots)} snapshots × ~500 accounts each)...")

    cache = {}
    for date in snapshots:
        cache[date] = build_features(date)
        if verbose:
            print(f"  {date}", end="  ")
    if verbose:
        print()
    return cache


def get_feature_columns(snap_cache: dict[str, pd.DataFrame]) -> list[str]:
    """
    Extract the ordered list of feature column names from a snapshot cache.

    Excludes ``account_id``, ``churn_flag``, ``snapshot_date``, and any
    internal temporary columns (prefixed with ``_``).

    Parameters
    ----------
    snap_cache : dict[str, pd.DataFrame]
        Output of ``build_snapshot_cache()``.

    Returns
    -------
    list[str]
    """
    sample = next(iter(snap_cache.values()))
    return [
        c for c in sample.columns
        if c not in ("account_id", "churn_flag", "snapshot_date")
        and not c.startswith("_")
    ]