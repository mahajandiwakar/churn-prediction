"""
config.py
---------
Single source of truth for every project-level constant.
Edit DATA_DIR before running to point at your CSV folder.
"""

import pandas as pd
DATA_DIR   = "data"     # folder containing the five ravenstack_*.csv files
OUTPUT_DIR = "outputs"  # directory for saved figures / artefacts
REFERENCE_DATE = pd.Timestamp("2024-12-31")  # "today" for static tenure calcs

# Monthly snapshot dates used for feature engineering + rolling CV
ALL_SNAPSHOTS = [
    "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
    "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
    "2024-09-01", "2024-10-01", "2024-11-01",
]
TRAIN_SNAPSHOTS = ALL_SNAPSHOTS[:9]   # Jan–Sep 2024  (training window)
TEST_SNAPSHOTS  = ALL_SNAPSHOTS[9:]   # Oct–Nov 2024  (held-out test window)
PRED_DAYS = 30   # forward label window  — did the account churn in [T, T+30d)?
OBS_DAYS  = 90   # primary observation window for rolling usage features
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.3   # classification threshold for churn prediction
MIN_CV_TRAIN = 3    # folds with fewer training snapshots are flagged, not dropped
# Soft-vote ensemble weights (LR × LR_WEIGHT + RF × RF_WEIGHT)
LR_WEIGHT = 0.4
RF_WEIGHT = 0.6
NAVY  = "#1E2761"
TEAL  = "#028090"
CORAL = "#F96167"
GOLD  = "#F9C74F"
GRAY  = "#64748B"
LIGHT = "#CADCFC"

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        False,
    "font.size":        11,
}

# Model colour map used in rolling-CV plots
MODEL_COLORS = {
    "LR":       TEAL,
    "RF":       NAVY,
    "GBM":      CORAL,
    "Ensemble": GOLD,
}
FEATURE_LABELS = {
    "tenure_days":            "Account Tenure (days)",
    "tenure_bucket":          "Tenure Bucket",
    "is_trial":               "Is Trial Account",
    "months_since_upgrade":   "Months Since Last Upgrade",
    "days_on_current_plan":   "Days on Current Plan",
    "total_plan_changes":     "Total Plan Changes",
    "mrr_at_snapshot":        "MRR at Snapshot",
    "seats_at_snapshot":      "Seats at Snapshot",
    "n_active_subs":          "Active Subscriptions",
    "engagement_per_seat":    "Usage Events per Seat",
    "usage_count_last_30":    "Usage Events – Last 30d",
    "usage_count_prev_30":    "Usage Events – Prev 30d",
    "usage_duration_last_30": "Usage Duration – Last 30d",
    "usage_decline_pct":      "Usage Decline % (30d vs 60d)",
    "went_dark":              "Went Dark (was active → 0)",
    "usage_count_last_90":    "Usage Events – Last 90d",
    "usage_3m_slope":         "3-Month Usage Slope",
    "usage_volatility":       "Usage Volatility (3m std)",
    "ratio_30_180":           "Collapse Ratio (30d / 180d)",
    "error_count_last_30":    "Error Count – Last 30d",
    "error_count_prev_30":    "Error Count – Prev 30d",
    "error_rate_last_30":     "Error Rate – Last 30d",
    "error_rate_change":      "Error Rate Change",
    "tickets_last_30":        "Support Tickets – Last 30d",
    "resolution_time_hours":  "Avg Ticket Resolution (hrs)",
    "satisfaction_score":     "Avg Satisfaction Score",
    "industry_enc":           "Industry",
    "referral_source_enc":    "Acquisition Channel",
    "plan_tier_enc":          "Plan Tier",
    "seats":                  "Contracted Seats",
    "n_prior_churns":         "Prior Churn Events",
    "recent_downgrade":       "Recent Downgrade (60d)",
}

FEATURE_GROUPS = {
    "Account Lifecycle":    [
        "tenure_days", "tenure_bucket", "is_trial",
        "months_since_upgrade", "days_on_current_plan", "total_plan_changes",
    ],
    "Commercial Footprint": [
        "mrr_at_snapshot", "seats_at_snapshot", "n_active_subs", "engagement_per_seat",
    ],
    "Usage Trend (90d)": [
        "usage_count_last_90", "usage_3m_slope", "usage_volatility", "ratio_30_180",
    ],
    "Usage 30d Windows": [
        "usage_count_last_30", "usage_count_prev_30",
        "usage_duration_last_30", "usage_decline_pct", "went_dark",
    ],
    "Error Signals": [
        "error_count_last_30", "error_count_prev_30",
        "error_rate_last_30", "error_rate_change",
    ],
    "Support Signals":   ["tickets_last_30", "resolution_time_hours", "satisfaction_score"],
    "Static Attributes": ["industry_enc", "referral_source_enc", "plan_tier_enc", "seats"],
    "Repeat / History":  ["n_prior_churns", "recent_downgrade"],
}
