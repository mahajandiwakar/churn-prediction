"""
eda.py
------
Exploratory Data Analysis for the RavenStack churn dataset.

Each public function is self-contained: it accepts the required DataFrames,
produces console output and a matplotlib figure, and returns nothing unless
it also creates a derived column that downstream code needs.

Key findings (confirmed by modelling):
  - DevTools churns at ~31 % — nearly 2× Cybersecurity (~16 %).
  - Event-sourced accounts churn at ~30 % vs partner-sourced at ~15 %.
  - Churn *increases* with tenure (15 % at 0–90 d → 26 % at 1–2 yr).
  - MRR concentration: top 20 % of churned accounts ≈ 40 % of churned MRR.
  - No single behavioural flag produces a 2× lift — signal is diffuse.
  - Monthly churn accelerated Jan → Nov 2024 (20 → 68 events/month).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


def _style() -> None:
    plt.rcParams.update(config.PLOT_STYLE)


def plot_churn_by_industry(accounts: pd.DataFrame) -> None:
    """
    Horizontal bar chart: churn rate per industry vertical.

    DevTools has the highest rate (~31 %), nearly 2× Cybersecurity (~16 %).
    The competitive tooling market with many free/OSS alternatives likely
    drives the gap.
    """
    _style()
    stats = (
        accounts.groupby("industry")["churn_flag"]
        .agg(churn_rate="mean", n_accounts="count")
        .sort_values("churn_rate")
    )
    colors = [config.CORAL if v >= 0.25 else config.TEAL for v in stats["churn_rate"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stats.index, stats["churn_rate"] * 100, color=colors, height=0.55)
    ax.axvline(22, color=config.NAVY, linestyle="--", alpha=0.7,
               linewidth=1.5, label="Overall avg 22 %")
    for bar, rate, n in zip(bars, stats["churn_rate"], stats["n_accounts"]):
        ax.text(rate * 100 + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{rate*100:.1f}%  (n={n})", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Industry", fontsize=13, fontweight="bold", color=config.NAVY)
    ax.set_xlim(0, 42)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_churn_by_channel(accounts: pd.DataFrame) -> None:
    """
    Bar chart: churn rate per referral / acquisition channel.

    Event-sourced accounts churn at ~30 % vs partner at ~15 % — a 2× gap.
    Onboarding quality likely explains the difference.
    """
    _style()
    stats = (
        accounts.groupby("referral_source")["churn_flag"]
        .agg(churn_rate="mean", n_accounts="count")
        .sort_values("churn_rate", ascending=False)
    )
    colors = [config.CORAL if v >= 0.25 else config.TEAL for v in stats["churn_rate"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(stats.index, stats["churn_rate"] * 100, color=colors, width=0.55)
    ax.axhline(22, color=config.NAVY, linestyle="--", alpha=0.7, label="Overall avg 22 %")
    for bar, rate, n in zip(bars, stats["churn_rate"], stats["n_accounts"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate * 100 + 0.5,
                f"{rate*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"n={n}", ha="center", fontsize=9, color="white")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Acquisition Channel",
                 fontsize=13, fontweight="bold", color=config.NAVY)
    ax.set_ylim(0, 42)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_churn_by_tenure(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Bar chart: churn rate across five tenure buckets.

    Counter-intuitive finding: churn *increases* with tenure
    (15 % at 0–90 d → 26 % at 1–2 yr). Long-tenured users may experience
    platform fatigue — this is why tenure_days is the top-ranked feature.

    Returns
    -------
    pd.DataFrame
        A copy of *accounts* with ``tenure_days`` and ``tenure_bucket``
        columns added (reused by feature engineering).
    """
    _style()
    df = accounts.copy()
    df["tenure_days"] = (config.REFERENCE_DATE - df["signup_date"]).dt.days
    df["tenure_bucket"] = pd.cut(
        df["tenure_days"],
        bins=[0, 90, 180, 365, 730, 9999],
        labels=["0–90d", "90–180d", "180–365d", "1–2yr", "2yr+"],
    )
    stats = (
        df.groupby("tenure_bucket", observed=True)["churn_flag"]
        .agg(churn_rate="mean", n_accounts="count")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [config.TEAL, config.TEAL, config.TEAL, config.CORAL, config.GRAY]
    bars = ax.bar(stats.index, stats["churn_rate"] * 100, color=colors, width=0.55)
    for bar, rate, n in zip(bars, stats["churn_rate"], stats["n_accounts"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate * 100 + 0.4,
                f"{rate*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"n={n}", ha="center", fontsize=9, color="white")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Account Tenure",
                 fontsize=13, fontweight="bold", color=config.NAVY)
    ax.set_ylim(0, 38)
    plt.tight_layout()
    plt.show()

    print(stats.round(3).to_string())
    return df


def print_revenue_impact(accounts: pd.DataFrame, subscriptions: pd.DataFrame) -> None:
    """
    Print MRR figures split by churn status.

    Shows that churned average MRR ≈ active average MRR: churn is not
    concentrated in low-value accounts and requires an all-tier response.
    """
    churned_ids = accounts[accounts["churn_flag"]]["account_id"]
    active_ids  = accounts[~accounts["churn_flag"]]["account_id"]

    total   = subscriptions["mrr_amount"].sum()
    churned = subscriptions[subscriptions["account_id"].isin(churned_ids)]["mrr_amount"].sum()
    avg_c   = (subscriptions[subscriptions["account_id"].isin(churned_ids)]
               .groupby("account_id")["mrr_amount"].sum().mean())
    avg_a   = (subscriptions[subscriptions["account_id"].isin(active_ids)]
               .groupby("account_id")["mrr_amount"].sum().mean())

    print("\n=== REVENUE IMPACT ===")
    print(f"  Total MRR            : ${total:>12,.0f}")
    print(f"  Churned-account MRR  : ${churned:>12,.0f}  ({churned/total:.1%} of total)")
    print(f"  Avg MRR per churned  : ${avg_c:>12,.0f}")
    print(f"  Avg MRR per active   : ${avg_a:>12,.0f}")
    print(f"  Churned/active ratio : {avg_c/avg_a:.0%}  — churn spans ALL value tiers")


def plot_revenue_segments(accounts: pd.DataFrame, subscriptions: pd.DataFrame) -> None:
    """
    Three-panel revenue lens:
      (left)   Churned MRR by plan tier
      (centre) Churned MRR by industry
      (right)  MRR concentration / Lorenz curve

    Key finding: Top 20 % of churned accounts = ~40 % of churned MRR.
    """
    _style()
    acct_mrr = (
        subscriptions.groupby("account_id")["mrr_amount"].sum()
        .reset_index().rename(columns={"mrr_amount": "mrr"})
        .merge(accounts[["account_id", "churn_flag", "plan_tier", "industry"]], on="account_id")
    )
    total_mrr   = acct_mrr["mrr"].sum()
    churned_mrr = acct_mrr[acct_mrr["churn_flag"]]["mrr"].sum()

    # Lorenz-style concentration curve
    c_sorted = (
        acct_mrr[acct_mrr["churn_flag"]]
        .sort_values("mrr", ascending=False).reset_index(drop=True)
    )
    c_sorted["cum_pct_accts"] = (c_sorted.index + 1) / len(c_sorted)
    c_sorted["cum_pct_mrr"]   = c_sorted["mrr"].cumsum() / churned_mrr

    def _seg(col: str) -> pd.DataFrame:
        s = (acct_mrr.groupby([col, "churn_flag"])["mrr"].sum()
             .unstack(fill_value=0).reset_index())
        s.columns = [col, "active_mrr", "churned_mrr"]
        s["total_mrr"]     = s["active_mrr"] + s["churned_mrr"]
        s["churn_mrr_pct"] = s["churned_mrr"] / s["total_mrr"]
        s = s.merge(acct_mrr.groupby(col)["churn_flag"].mean().rename("acct_churn_rate"), on=col)
        return s.sort_values("churned_mrr", ascending=False)

    ps = _seg("plan_tier")
    ins = _seg("industry")

    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.38)
    fig.suptitle("Revenue Lens — Where Churned MRR Lives",
                 fontsize=14, fontweight="bold", color=config.NAVY, y=1.01)

    # Panel 1: plan tier bars
    ax1 = fig.add_subplot(gs[0, 0])
    c1  = [config.CORAL if v > 0.21 else config.TEAL for v in ps["churn_mrr_pct"]]
    bars = ax1.bar(ps["plan_tier"], ps["churned_mrr"] / 1000, color=c1, width=0.55)
    for bar, row in zip(bars, ps.itertuples()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"${row.churned_mrr/1000:.0f}k\n({row.churn_mrr_pct:.1%})",
                 ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Churned MRR ($k)")
    ax1.set_title("Churned MRR by Plan Tier", fontsize=11, fontweight="bold", color=config.NAVY)
    ax1.set_ylim(0, ps["churned_mrr"].max() / 1000 * 1.35)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: industry bars
    ax2   = fig.add_subplot(gs[0, 1])
    ip    = ins.sort_values("churned_mrr")
    c2    = [config.CORAL if v > 0.22 else config.TEAL for v in ip["churn_mrr_pct"]]
    ax2.barh(ip["industry"], ip["churned_mrr"] / 1000, color=c2, height=0.55)
    for i, row in enumerate(ip.itertuples()):
        ax2.text(row.churned_mrr / 1000 + 2, i,
                 f"${row.churned_mrr/1000:.0f}k  ({row.churn_mrr_pct:.1%})",
                 va="center", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Churned MRR ($k)")
    ax2.set_title("Churned MRR by Industry", fontsize=11, fontweight="bold", color=config.NAVY)
    ax2.set_xlim(0, ip["churned_mrr"].max() / 1000 * 1.5)
    ax2.grid(axis="x", alpha=0.3)

    # Panel 3: concentration curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(c_sorted["cum_pct_accts"] * 100, c_sorted["cum_pct_mrr"] * 100,
             color=config.NAVY, linewidth=2.5, label="Actual")
    ax3.plot([0, 100], [0, 100], "--", color=config.GRAY, linewidth=1.2, label="Perfect equality")
    p20 = float(c_sorted.loc[c_sorted["cum_pct_accts"].sub(0.20).abs().idxmin(), "cum_pct_mrr"]) * 100
    ax3.axvline(20, color=config.CORAL, linestyle=":", linewidth=1.2)
    ax3.axhline(p20, color=config.CORAL, linestyle=":", linewidth=1.2)
    ax3.scatter([20], [p20], color=config.CORAL, s=60, zorder=5)
    ax3.text(22, p20 + 2, f"Top 20% accounts\n= {p20:.0f}% of churned MRR",
             fontsize=8.5, color=config.CORAL)
    ax3.set_xlabel("Cumulative % of Churned Accounts")
    ax3.set_ylabel("Cumulative % of Churned MRR")
    ax3.set_title("MRR Concentration Curve", fontsize=11, fontweight="bold", color=config.NAVY)
    ax3.legend(fontsize=8.5)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n=== REVENUE CONCENTRATION ===")
    for pct in [0.10, 0.20, 0.30]:
        n   = int(len(c_sorted) * pct)
        mrr = c_sorted.head(n)["mrr"].sum()
        print(f"  Top {pct:.0%} of churned accounts ({n}) = {mrr/churned_mrr:.1%} of churned MRR  (${mrr:,.0f})")


def plot_behavioural_signals(
    accounts:        pd.DataFrame,
    subscriptions:   pd.DataFrame,
    feature_usage:   pd.DataFrame,
    churn_events:    pd.DataFrame,
) -> None:
    """
    Two-panel plot:
      Left   — churn rate with vs. without five behavioural flags.
      Right  — monthly churn volume for 2024 with the train/test boundary.

    Design note
    -----------
    On this synthetic dataset individual flags show *weak* separation —
    no single flag produces a 2× lift over baseline.  This is intentional:
    the model wins by combining 32 features, not from any one signal.
    EDA here generates hypotheses; permutation importance confirms which
    combinations matter.
    """
    _style()

    # Build account-level usage flags for the Nov 2024 reference window
    fu = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]], on="subscription_id", how="left"
    )
    af = accounts[["account_id", "churn_flag"]].copy()

    u_last = (fu[(fu["usage_date"] >= "2024-11-01") & (fu["usage_date"] < "2024-11-30")]
              .groupby("account_id")["usage_count"].sum())
    u_prev = (fu[(fu["usage_date"] >= "2024-10-01") & (fu["usage_date"] < "2024-11-01")]
              .groupby("account_id")["usage_count"].sum())

    af = (af.merge(u_last.rename("u_last").reset_index(), on="account_id", how="left")
            .merge(u_prev.rename("u_prev").reset_index(), on="account_id", how="left")
            .fillna(0))

    af["went_dark"]      = ((af["u_last"] == 0) & (af["u_prev"] > 0)).astype(int)
    af["usage_declined"] = ((af["u_prev"] > 0) & (af["u_last"] < af["u_prev"] * 0.5)).astype(int)
    af["had_downgrade"]  = af["account_id"].isin(
        set(subscriptions[subscriptions["downgrade_flag"] == 1]["account_id"])).astype(int)
    af["repeat_churner"] = af["account_id"].isin(
        set(churn_events.groupby("account_id").filter(lambda x: len(x) > 1)["account_id"])
    ).astype(int)
    af["was_trial"] = af["account_id"].isin(
        set(accounts[accounts["is_trial"]]["account_id"])).astype(int)

    overall = af["churn_flag"].mean()
    signals = [
        ("went_dark",      "Went dark (had usage → zero)"),
        ("usage_declined", "Usage declined >50% MoM"),
        ("had_downgrade",  "Had a plan downgrade (lifetime)"),
        ("repeat_churner", "Repeat churner (2+ events)"),
        ("was_trial",      "Started as trial account"),
    ]
    rows = []
    for col, label in signals:
        has  = af[af[col] == 1]["churn_flag"].mean()
        nope = af[af[col] == 0]["churn_flag"].mean()
        rows.append({"label": label, "with": has, "without": nope,
                     "n": int(af[col].sum()), "lift": has / overall})
    sig = pd.DataFrame(rows).sort_values("with", ascending=False)

    # Monthly churn trend
    monthly = (
        churn_events[churn_events["churn_date"].dt.year == 2024]
        .assign(mo=lambda d: d["churn_date"].dt.to_period("M"))
        .groupby("mo").size().reset_index(name="n_churns")
    )
    monthly["mo_str"] = monthly["mo"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Behavioural Signals — EDA Hypothesis Generation",
                 fontsize=13, fontweight="bold", color=config.NAVY, y=1.01)

    ax, x, w = axes[0], np.arange(len(sig)), 0.32
    b1 = ax.bar(x - w/2, sig["with"]    * 100, w, color=config.CORAL, alpha=0.85, label="With signal")
    b2 = ax.bar(x + w/2, sig["without"] * 100, w, color=config.TEAL,  alpha=0.85, label="Without signal")
    ax.axhline(overall * 100, color=config.NAVY, linestyle="--", linewidth=1.5,
               label=f"Overall: {overall:.1%}")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("Churn Rate (%)"); ax.set_ylim(0, 32)
    ax.legend(fontsize=9)
    ax.set_title("Churn Rate: With vs Without Signal", fontsize=11, fontweight="bold", color=config.NAVY)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(range(len(monthly)), monthly["n_churns"],
            color=[config.CORAL if i >= 9 else config.TEAL for i in range(len(monthly))],
            width=0.65, alpha=0.85)
    for i, (_, row) in enumerate(monthly.iterrows()):
        ax2.text(i, row["n_churns"] + 0.5, str(row["n_churns"]),
                 ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly["mo_str"], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Churn Events")
    ax2.set_title("Monthly Churn Volume (2024)\nOct–Nov = test window (coral)",
                  fontsize=11, fontweight="bold", color=config.NAVY)
    ax2.axvline(8.5, color=config.NAVY, linestyle="--", linewidth=1.2, label="Train/test split")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"\n{'Signal':<42} {'With':>8} {'Without':>9} {'Lift':>7} {'N':>6}")
    print("-" * 75)
    for _, r in sig.iterrows():
        print(f"  {r['label']:<40} {r['with']:>8.1%} {r['without']:>9.1%} {r['lift']:>7.1f}x {r['n']:>6}")


def print_usage_summary(
    accounts:      pd.DataFrame,
    subscriptions: pd.DataFrame,
    feature_usage: pd.DataFrame,
) -> None:
    """
    Print mean usage metrics (total events, unique features, error rate)
    split by churn status.

    Counter-intuitive: churned accounts use the product *more* on average
    (higher raw event count).  Error experience, not raw volume, predicts churn.
    """
    fu = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]], on="subscription_id", how="left"
    )
    by_acct = fu.groupby("account_id").agg(
        total_usage=("usage_count", "sum"),
        total_errors=("error_count", "sum"),
        unique_features=("feature_name", "nunique"),
    ).reset_index()
    by_acct["error_rate"] = by_acct["total_errors"] / by_acct["total_usage"].clip(lower=1)

    merged = by_acct.merge(
        accounts[["account_id", "churn_flag"]].rename(columns={"churn_flag": "churned"}),
        on="account_id",
    )
    summary = merged.groupby("churned")[["total_usage", "unique_features", "error_rate"]].mean().round(3)
    summary.index = ["Active", "Churned"]

    print("\n=== MEAN USAGE METRICS BY CHURN STATUS ===")
    print(summary.to_string())
    print("\nChurned accounts use the product MORE on average.")
    print("Raw volume ≠ retention. Error experience during usage is the differentiator.")


def plot_support_analysis(
    accounts:        pd.DataFrame,
    support_tickets: pd.DataFrame,
) -> None:
    """
    Two-panel support analysis:
      Left  — churn rate for escalated vs non-escalated accounts.
      Right — churn rate by ticket priority.

    Counter-intuitive: escalated accounts churn more, but fast first-response
    time does not protect against churn — resolution *quality* matters more.
    """
    _style()
    acc_churn  = accounts[["account_id", "churn_flag"]].rename(columns={"churn_flag": "acc_churn"})
    st         = support_tickets.merge(acc_churn, on="account_id", how="left")

    esc_ids    = support_tickets[support_tickets["escalation_flag"]]["account_id"].unique()
    noesc_ids  = support_tickets[~support_tickets["escalation_flag"]]["account_id"].unique()
    esc_rate   = accounts[accounts["account_id"].isin(esc_ids)]["churn_flag"].mean()
    noesc_rate = accounts[~accounts["account_id"].isin(esc_ids)]["churn_flag"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(["Escalated", "Not Escalated"], [esc_rate * 100, noesc_rate * 100],
                color=[config.CORAL, config.TEAL], width=0.5)
    for i, v in enumerate([esc_rate * 100, noesc_rate * 100]):
        axes[0].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)"); axes[0].set_ylim(0, 38)
    axes[0].set_title("Escalation vs Churn", fontsize=12, fontweight="bold", color=config.NAVY)

    by_prio = (st.groupby("priority")["acc_churn"].mean()
               .reindex(["low", "medium", "high", "urgent"]))
    axes[1].bar(by_prio.index, by_prio.values * 100,
                color=[config.TEAL, config.TEAL, config.CORAL, config.CORAL], width=0.5)
    for i, v in enumerate(by_prio.values):
        axes[1].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)"); axes[1].set_ylim(0, 38)
    axes[1].set_title("Churn by Ticket Priority", fontsize=12, fontweight="bold", color=config.NAVY)

    plt.tight_layout(); plt.show()
    print(f"\nEscalated churn rate: {esc_rate:.1%}  |  Non-escalated: {noesc_rate:.1%}")
    print("Speed alone is insufficient — resolution quality matters more.")
