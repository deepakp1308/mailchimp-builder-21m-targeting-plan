"""Rebuild parquet from cached raw_full CSVs (skip BigQuery)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ["SKIP_BQ"] = "1"

import build_features_full as bf

RAW = bf.RAW_DIR
if not os.path.exists(os.path.join(RAW, "customer_base.csv")):
    raise SystemExit("Missing cached CSVs in data/raw_full/")

print("Rebuilding from cache...")
df_base = bf.pd.read_csv(os.path.join(RAW, "customer_base.csv"))
df_builder = bf.pd.read_csv(os.path.join(RAW, "builder_90d.csv"))
df_builder_30d = bf.pd.read_csv(os.path.join(RAW, "builder_30d.csv"))
df_builder_prev30d = bf.pd.read_csv(os.path.join(RAW, "builder_prev30d.csv"))
df_health = bf.pd.read_csv(os.path.join(RAW, "customer_health.csv"))
df_mrr = bf.pd.read_csv(os.path.join(RAW, "mrr_status.csv"))

# Deduplicate health (safety)
df_health = df_health.drop_duplicates("user_id", keep="first")

df = df_base.copy()
for name, right_df in [
    ("builder_90d", df_builder),
    ("builder_30d", df_builder_30d),
    ("builder_prev30d", df_builder_prev30d),
    ("health", df_health),
    ("mrr", df_mrr),
]:
    print(f"Joining {name} ({len(right_df):,})...")
    df = df.merge(right_df, on="user_id", how="left")

df = df.drop_duplicates("user_id", keep="first")

# Reuse derived feature logic by exec from module tail
import numpy as np

behavior_cols = [
    c for c in df.columns
    if any(x in c for x in ["creates", "publishes", "tests", "events", "days", "delivered", "orders", "revenue"])
    and c not in ("total_revenue", "avg_mrr", "plan_amount", "total_revenue_3mo")
]
df[behavior_cols] = df[behavior_cols].fillna(0)
df["email_completion_rate"] = (df["email_publishes_90d"] / df["email_creates_90d"].replace(0, float("nan"))).fillna(0).clip(0, 1)
df["email_abandonment_rate"] = 1 - df["email_completion_rate"]
df["test_no_send_rate"] = ((df["email_tests_90d"] > 0) & (df["email_publishes_90d"] == 0)).astype(int)
df["create_no_publish_rate"] = ((df["email_creates_90d"] > 0) & (df["email_publishes_90d"] == 0)).astype(int)
df["creates_trend"] = df["email_creates_30d"] - df["email_creates_prev30d"]
df["publishes_trend"] = df["email_publishes_30d"] - df["email_publishes_prev30d"]
df["events_trend"] = df["total_builder_events_30d"] - df["total_builder_events_prev30d"]
df["builder_maturity"] = bf.pd.cut(df["email_publishes_90d"], bins=[-1, 0, 2, 10, 50, float("inf")], labels=["none", "low", "medium", "high", "power"])
conditions = [
    (df["email_creates_90d"] == 0) & (df["email_publishes_90d"] == 0),
    (df["email_creates_90d"] > 0) & (df["email_publishes_90d"] == 0),
    (df["email_publishes_90d"] > 0) & (df["email_publishes_90d"] <= 2),
    (df["email_publishes_90d"] > 2),
]
df["bulk_email_stage"] = np.select(conditions, ["1_Unexplored", "2_Explore", "3_Try", "4_Establish"], default="5_Abandon")
df["friction_score"] = (df["test_no_send_rate"] * 0.3 + df["create_no_publish_rate"] * 0.3 + df["email_abandonment_rate"] * 0.4).clip(0, 1)
df["mrr_band"] = bf.pd.cut(df["avg_mrr"], bins=[-1, 0, 20, 50, 100, 300, float("inf")], labels=["free", "<$20", "$20-50", "$50-100", "$100-300", "$300+"])
df["tenure_band"] = bf.pd.cut(df["tenure_months"], bins=[-1, 1, 3, 6, 12, 24, float("inf")], labels=["<1mo", "1-3mo", "3-6mo", "6-12mo", "12-24mo", "24mo+"])
df["eligible_rendering_fix"] = ((df["email_publishes_90d"] > 0) & (df["friction_score"] > 0.3)).astype(int)
df["eligible_brandkit"] = ((df["email_creates_90d"] > 2) & (df["ca_template_creates_90d"] == 0)).astype(int)
df["eligible_universal_content"] = (df["email_publishes_90d"] >= 3).astype(int)
df["eligible_ai_builder"] = ((df["ai_content_creates_90d"] == 0) & (df["email_creates_90d"] > 0)).astype(int)
df["eligible_template_improvement"] = ((df["email_creates_90d"] > 0) & (df["email_completion_rate"] < 0.5)).astype(int)
df["eligible_omnichannel"] = ((df["email_publishes_90d"] > 0) & (df["sms_creates_90d"] == 0)).astype(int)
df["eligible_activation"] = df["bulk_email_stage"].isin(["1_Unexplored", "2_Explore", "5_Abandon"]).astype(int)
df["eligible_code_mode"] = (df["package"].isin(["premium_monthly_plan_v0", "premium_annual_plan_v0", "legacy monthly"]) & (df["email_creates_90d"] > 0)).astype(int)
elig_cols = [c for c in df.columns if c.startswith("eligible_")]
df["total_eligible_initiatives"] = df[elig_cols].sum(axis=1)
df["data_source"] = "bigquery_full_population"
for col in df.columns:
    if "date" in col.lower():
        df[col] = bf.pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

out = os.path.join(bf.DATA_DIR, "customers_engineered.parquet")
df.to_parquet(out, index=False)
print(f"Saved {len(df):,} customers to {out}")
