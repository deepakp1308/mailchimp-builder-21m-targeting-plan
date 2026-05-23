"""
Phase 1 (full population): Build customer feature table for ALL eligible customers.
Runs one BigQuery aggregation per table (no user sampling), caches CSVs, joins locally.
"""
import os
import time

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "mc-business-intelligence"
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_full")

client = bigquery.Client(project=PROJECT_ID)
os.makedirs(RAW_DIR, exist_ok=True)


def run_query(sql, label, cache_name, force=False):
    cache_path = os.path.join(RAW_DIR, f"{cache_name}.csv")
    if not force and os.path.exists(cache_path):
        print(f"  Loading cached: {label}...", end=" ", flush=True)
        df = pd.read_csv(cache_path)
        print(f"done ({len(df):,} rows)")
        return df

    print(f"  Running: {label}...", end=" ", flush=True)
    t0 = time.time()
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)
    print(f"done ({len(df):,} rows, {time.time() - t0:.1f}s)")
    df.to_csv(cache_path, index=False)
    return df


def run_batched_query(sql_template, label, cache_name, n_batches=20, force=False):
    cache_path = os.path.join(RAW_DIR, f"{cache_name}.csv")
    if not force and os.path.exists(cache_path):
        print(f"  Loading cached: {label}...", end=" ", flush=True)
        df = pd.read_csv(cache_path)
        print(f"done ({len(df):,} rows)")
        return df

    print(f"  Running: {label} ({n_batches} BQ buckets)...")
    frames = []
    t0 = time.time()
    for bucket in range(n_batches):
        sql = sql_template.format(bucket=bucket, n_batches=n_batches)
        part = client.query(sql).to_dataframe(create_bqstorage_client=False)
        frames.append(part)
        print(f"    bucket {bucket + 1}/{n_batches}: {len(part):,} rows")
    df = pd.concat(frames, ignore_index=True)
    print(f"  done ({len(df):,} rows, {time.time() - t0:.1f}s)")
    df.to_csv(cache_path, index=False)
    return df


print("=" * 60)
print("PHASE 1 (FULL POPULATION): All eligible customers from BigQuery")
print("=" * 60)

CUSTOMER_BASE_SQL = """
SELECT
  u.user_id,
  u.primary_plan_type,
  COALESCE(u.current_marketing_plan_order.package_details, 'free') AS package,
  COALESCE(u.current_marketing_plan_order.amount, 0) AS plan_amount,
  COALESCE(u.average_monthly_revenue, 0) AS avg_mrr,
  COALESCE(u.total_revenue, 0) AS total_revenue,
  u.country,
  u.analytics_country_group AS country_group,
  u.ecomm_level,
  u.is_high_value,
  u.customer_success_tier,
  DATE_DIFF(CURRENT_DATE(), DATE(u.created_at), DAY) AS tenure_days,
  DATE_DIFF(CURRENT_DATE(), DATE(u.created_at), MONTH) AS tenure_months,
  COALESCE(u.list_size, 0) AS list_size,
  COALESCE(u.subscribed_size, 0) AS subscribed_size,
  u.list_count,
  u.active AS is_active_30d,
  DATE(u.last_login_at) AS last_login_date,
  ind.intuit_vertical,
  ind.naics_icp,
  ind.mm_sb_se_flag AS business_size,
  ind.sbb_vs_pbb AS business_type
FROM `mc-business-intelligence.bi_reporting.users` u
LEFT JOIN `mc-business-intelligence.bi_finance.user_industry` ind
  ON u.user_id = ind.user_id
WHERE
  u.primary_plan_type IN ('monthly', 'payg')
  OR (
    u.primary_plan_type = 'free'
    AND u.active = TRUE
    AND u.last_login_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  )
"""

BUILDER_90D_SQL = """
SELECT
  user_id,
  COUNTIF(action = 'create' AND object = 'email') AS email_creates_90d,
  COUNTIF(action = 'publish' AND object = 'email') AS email_publishes_90d,
  COUNTIF(action = 'test' AND object = 'email') AS email_tests_90d,
  COUNTIF(
    action = 'create' AND object = 'email'
    AND EXISTS(
      SELECT 1 FROM UNNEST(properties) p
      WHERE p.key = 'is_creative_assistant_template' AND p.value = 'true'
    )
  ) AS ca_template_creates_90d,
  COUNTIF(
    action = 'create' AND object = 'email'
    AND EXISTS(
      SELECT 1 FROM UNNEST(properties) p
      WHERE p.key = 'has_ai_content' AND p.value = 'true'
    )
  ) AS ai_content_creates_90d,
  COUNTIF(action = 'create' AND object = 'sms') AS sms_creates_90d,
  COUNTIF(action = 'publish' AND object = 'sms') AS sms_publishes_90d,
  COUNTIF(action = 'create' AND object IN ('customer_journey', 'automation')) AS automation_creates_90d,
  COUNTIF(action = 'publish' AND object IN ('customer_journey', 'automation')) AS automation_publishes_90d,
  COUNT(DISTINCT DATE(timestamp)) AS builder_active_days_90d,
  COUNT(*) AS total_builder_events_90d
FROM `mc-business-intelligence.bi_activities.users_activities`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  AND object IN ('email', 'sms', 'customer_journey', 'automation')
GROUP BY user_id
"""

BUILDER_30D_SQL = """
SELECT
  user_id,
  COUNTIF(action = 'create' AND object = 'email') AS email_creates_30d,
  COUNTIF(action = 'publish' AND object = 'email') AS email_publishes_30d,
  COUNTIF(action = 'test' AND object = 'email') AS email_tests_30d,
  COUNT(*) AS total_builder_events_30d
FROM `mc-business-intelligence.bi_activities.users_activities`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND object IN ('email', 'sms', 'customer_journey', 'automation')
GROUP BY user_id
"""

BUILDER_PREV30D_SQL = """
SELECT
  user_id,
  COUNTIF(action = 'create' AND object = 'email') AS email_creates_prev30d,
  COUNTIF(action = 'publish' AND object = 'email') AS email_publishes_prev30d,
  COUNT(*) AS total_builder_events_prev30d
FROM `mc-business-intelligence.bi_activities.users_activities`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 60 DAY)
  AND timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND object IN ('email', 'sms', 'customer_journey', 'automation')
GROUP BY user_id
"""

HEALTH_BATCH_SQL = """
SELECT
  h.user_id,
  AVG(h.open_rate) AS avg_open_rate,
  AVG(h.click_rate) AS avg_click_rate,
  AVG(h.bounce_rate) AS avg_bounce_rate,
  SUM(h.delivered) AS total_delivered_3mo,
  SUM(h.revenue_usd) AS total_revenue_3mo,
  SUM(h.orders) AS total_orders_3mo,
  AVG(h.overall_health_score) AS avg_health_score,
  AVG(h.engagement_score) AS avg_engagement_score,
  AVG(h.deliverability_score) AS avg_deliverability_score,
  AVG(h.revenue_score) AS avg_revenue_score
FROM `mc-business-intelligence.bi_customer.customer_outcomes_monthly` h
WHERE h.month >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)
  AND MOD(ABS(FARM_FINGERPRINT(CAST(h.user_id AS STRING))), {n_batches}) = {bucket}
GROUP BY h.user_id
"""

MRR_SQL = """
SELECT
  user_id,
  current_total_monthly_cloud_revenue_net AS current_mrr_net,
  previous_total_monthly_cloud_revenue_net AS prev_mrr_net,
  monthly_cloud_status AS mrr_status,
  primary_plan_type AS current_plan_type_finance
FROM `mc-business-intelligence.bi_finance.user_cloud_monthly_status`
WHERE month = (
  SELECT MAX(month)
  FROM `mc-business-intelligence.bi_finance.user_cloud_monthly_status`
)
"""

print("\n[1/6] Customer base (full eligible population)...")
df_base = run_query(CUSTOMER_BASE_SQL, "customer_base_full", "customer_base")

print("\n[2/6] Builder behavior (90d, all users)...")
df_builder = run_query(BUILDER_90D_SQL, "builder_90d_full", "builder_90d")

print("\n[3/6] Builder behavior (30d)...")
df_builder_30d = run_query(BUILDER_30D_SQL, "builder_30d_full", "builder_30d")

print("\n[4/6] Builder behavior (prior 30d)...")
df_builder_prev30d = run_query(BUILDER_PREV30D_SQL, "builder_prev30d_full", "builder_prev30d")

print("\n[5/6] Customer health (3mo, batched by user_id hash)...")
df_health = run_batched_query(HEALTH_BATCH_SQL, "customer_health_full", "customer_health", n_batches=20)

print("\n[6/6] MRR / churn signals (latest month, all users)...")
df_mrr = run_query(MRR_SQL, "mrr_status_full", "mrr_status")

print("\n[JOINING] Merging all feature groups...")
df = df_base.copy()
for name, right_df in [
    ("builder_90d", df_builder),
    ("builder_30d", df_builder_30d),
    ("builder_prev30d", df_builder_prev30d),
    ("health", df_health),
    ("mrr", df_mrr),
]:
    print(f"  Joining {name} ({len(right_df):,} rows)...")
    df = df.merge(right_df, on="user_id", how="left")

print("\n[DEDUP] Removing duplicate user_ids...")
before = len(df)
df = df.drop_duplicates(subset="user_id", keep="first")
print(f"  {before:,} → {len(df):,} rows")

print("\n[DERIVED] Computing derived features...")
behavior_cols = [
    c
    for c in df.columns
    if any(
        x in c
        for x in [
            "creates", "publishes", "tests", "events", "days",
            "delivered", "orders", "revenue",
        ]
    )
    and c not in ("total_revenue", "avg_mrr", "plan_amount", "total_revenue_3mo")
]
df[behavior_cols] = df[behavior_cols].fillna(0)

df["email_completion_rate"] = (
    df["email_publishes_90d"] / df["email_creates_90d"].replace(0, float("nan"))
).fillna(0).clip(0, 1)
df["email_abandonment_rate"] = 1 - df["email_completion_rate"]
df["test_no_send_rate"] = (
    (df["email_tests_90d"] > 0) & (df["email_publishes_90d"] == 0)
).astype(int)
df["create_no_publish_rate"] = (
    (df["email_creates_90d"] > 0) & (df["email_publishes_90d"] == 0)
).astype(int)
df["creates_trend"] = df["email_creates_30d"] - df["email_creates_prev30d"]
df["publishes_trend"] = df["email_publishes_30d"] - df["email_publishes_prev30d"]
df["events_trend"] = df["total_builder_events_30d"] - df["total_builder_events_prev30d"]

df["builder_maturity"] = pd.cut(
    df["email_publishes_90d"],
    bins=[-1, 0, 2, 10, 50, float("inf")],
    labels=["none", "low", "medium", "high", "power"],
)

conditions = [
    (df["email_creates_90d"] == 0) & (df["email_publishes_90d"] == 0),
    (df["email_creates_90d"] > 0) & (df["email_publishes_90d"] == 0),
    (df["email_publishes_90d"] > 0) & (df["email_publishes_90d"] <= 2),
    (df["email_publishes_90d"] > 2),
]
choices = ["1_Unexplored", "2_Explore", "3_Try", "4_Establish"]
df["bulk_email_stage"] = np.select(conditions, choices, default="5_Abandon")

df["friction_score"] = (
    df["test_no_send_rate"] * 0.3
    + df["create_no_publish_rate"] * 0.3
    + df["email_abandonment_rate"] * 0.4
).clip(0, 1)

df["mrr_band"] = pd.cut(
    df["avg_mrr"],
    bins=[-1, 0, 20, 50, 100, 300, float("inf")],
    labels=["free", "<$20", "$20-50", "$50-100", "$100-300", "$300+"],
)
df["tenure_band"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 1, 3, 6, 12, 24, float("inf")],
    labels=["<1mo", "1-3mo", "3-6mo", "6-12mo", "12-24mo", "24mo+"],
)

df["eligible_rendering_fix"] = (
    (df["email_publishes_90d"] > 0) & (df["friction_score"] > 0.3)
).astype(int)
df["eligible_brandkit"] = (
    (df["email_creates_90d"] > 2) & (df["ca_template_creates_90d"] == 0)
).astype(int)
df["eligible_universal_content"] = (df["email_publishes_90d"] >= 3).astype(int)
df["eligible_ai_builder"] = (
    (df["ai_content_creates_90d"] == 0) & (df["email_creates_90d"] > 0)
).astype(int)
df["eligible_template_improvement"] = (
    (df["email_creates_90d"] > 0) & (df["email_completion_rate"] < 0.5)
).astype(int)
df["eligible_omnichannel"] = (
    (df["email_publishes_90d"] > 0) & (df["sms_creates_90d"] == 0)
).astype(int)
df["eligible_activation"] = df["bulk_email_stage"].isin(
    ["1_Unexplored", "2_Explore", "5_Abandon"]
).astype(int)
df["eligible_code_mode"] = (
    df["package"].isin(
        ["premium_monthly_plan_v0", "premium_annual_plan_v0", "legacy monthly"]
    )
    & (df["email_creates_90d"] > 0)
).astype(int)

elig_cols = [c for c in df.columns if c.startswith("eligible_")]
df["total_eligible_initiatives"] = df[elig_cols].sum(axis=1)
df["data_source"] = "bigquery_full_population"

for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

out_path = os.path.join(DATA_DIR, "customers_engineered.parquet")
print(f"\n[SAVE] Writing {len(df):,} customers to {out_path}...")
df.to_parquet(out_path, index=False)

print(f"\n{'=' * 60}")
print("PHASE 1 (FULL POPULATION) COMPLETE")
print(f"{'=' * 60}")
print(f"Total customers: {len(df):,}")
print(f"Paid: {len(df[df['primary_plan_type'].isin(['monthly', 'payg'])]):,}")
print(f"Free (active): {len(df[df['primary_plan_type'] == 'free']):,}")
print(f"Avg MRR: ${df['avg_mrr'].mean():.2f}")
print(f"Mean friction: {df['friction_score'].mean():.3f}")
print(f"Mean completion rate: {df['email_completion_rate'].mean():.3f}")
