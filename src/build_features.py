"""
Phase 1: Build customer-level feature table from BigQuery.
Queries multiple tables, joins locally, outputs to parquet.
"""
import pandas as pd
from google.cloud import bigquery
import os, sys, time

PROJECT_ID = "mc-business-intelligence"
client = bigquery.Client(project=PROJECT_ID)

def run_query(sql, label=""):
    print(f"  Running: {label}...", end=" ", flush=True)
    t0 = time.time()
    job = client.query(sql)
    df = job.to_dataframe(create_bqstorage_client=False)
    print(f"done ({len(df):,} rows, {time.time()-t0:.1f}s)")
    return df

print("=" * 60)
print("PHASE 1: Building customer feature table from BigQuery")
print("=" * 60)

# ── 1. Customer Base (paid + active free) ──────────────────
print("\n[1/7] Customer profile + plan + MRR...")
df_base = run_query("""
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
  OR (u.primary_plan_type = 'free'
      AND u.active = TRUE
      AND u.last_login_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY))
""", "customer_base")
print(f"  Customer base: {len(df_base):,} rows")

user_ids_csv = ",".join(str(x) for x in df_base['user_id'].head(100))

# ── 2. Builder Behavior (last 90d from users_activities) ──
print("\n[2/7] Builder behavior features (last 90d)...")
df_builder = run_query("""
SELECT
  user_id,
  COUNTIF(action = 'create' AND object = 'email') AS email_creates_90d,
  COUNTIF(action = 'publish' AND object = 'email') AS email_publishes_90d,
  COUNTIF(action = 'test' AND object = 'email') AS email_tests_90d,
  COUNTIF(action = 'create' AND object = 'email'
    AND EXISTS(SELECT 1 FROM UNNEST(properties) p WHERE p.key = 'is_creative_assistant_template' AND p.value = 'true')
  ) AS ca_template_creates_90d,
  COUNTIF(action = 'create' AND object = 'email'
    AND EXISTS(SELECT 1 FROM UNNEST(properties) p WHERE p.key = 'has_ai_content' AND p.value = 'true')
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
""", "builder_behavior_90d")

# ── 3. Builder Behavior (last 30d for trend) ──
print("\n[3/7] Builder behavior features (last 30d for trend)...")
df_builder_30d = run_query("""
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
""", "builder_behavior_30d")

# ── 4. Builder Behavior (prior 30d for trend delta) ──
print("\n[4/7] Builder behavior features (prior 30d for trend)...")
df_builder_prev30d = run_query("""
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
""", "builder_behavior_prev30d")

# ── 5. Customer Health / Campaign Performance ──
print("\n[5/7] Customer health + campaign performance...")
df_health = run_query("""
SELECT
  user_id,
  AVG(open_rate) AS avg_open_rate,
  AVG(click_rate) AS avg_click_rate,
  AVG(bounce_rate) AS avg_bounce_rate,
  SUM(delivered) AS total_delivered_3mo,
  SUM(revenue_usd) AS total_revenue_3mo,
  SUM(orders) AS total_orders_3mo,
  AVG(overall_health_score) AS avg_health_score,
  AVG(engagement_score) AS avg_engagement_score,
  AVG(deliverability_score) AS avg_deliverability_score,
  AVG(revenue_score) AS avg_revenue_score
FROM `mc-business-intelligence.bi_customer.customer_outcomes_monthly`
WHERE month >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)
GROUP BY user_id
""", "customer_health_3mo")

# ── 6. Skipped: product_journey_monthly is aggregate-only (no user_id) ──
# We'll derive builder stage from behavioral features after join.
print("\n[6/7] Product journey stage — will derive from behavior features (no user-level table available)")
df_journey = None

# ── 7. MRR Change / Churn Signals ──
print("\n[7/7] MRR change + churn signals...")
df_mrr = run_query("""
SELECT
  user_id,
  current_total_monthly_cloud_revenue_net AS current_mrr_net,
  previous_total_monthly_cloud_revenue_net AS prev_mrr_net,
  monthly_cloud_status AS mrr_status,
  primary_plan_type AS current_plan_type_finance
FROM `mc-business-intelligence.bi_finance.user_cloud_monthly_status`
WHERE month = (SELECT MAX(month) FROM `mc-business-intelligence.bi_finance.user_cloud_monthly_status`)
""", "mrr_change")

# ── JOIN EVERYTHING ──────────────────────────────────────
print("\n[JOINING] Merging all feature groups...")
df = df_base.copy()
join_list = [
    ("builder_90d", df_builder),
    ("builder_30d", df_builder_30d),
    ("builder_prev30d", df_builder_prev30d),
    ("health", df_health),
    ("mrr", df_mrr),
]
if df_journey is not None:
    join_list.append(("journey", df_journey))
for name, right_df in join_list:
    print(f"  Joining {name} ({len(right_df):,} rows)...")
    df = df.merge(right_df, on='user_id', how='left')

# ── COMPUTE DERIVED FEATURES ────────────────────────────
print("\n[DERIVED] Computing derived features...")

# Fill NAs for behavioral features
behavior_cols = [c for c in df.columns if any(x in c for x in ['creates', 'publishes', 'tests', 'events', 'days', 'delivered', 'orders', 'revenue'])]
df[behavior_cols] = df[behavior_cols].fillna(0)

# Completion rate
df['email_completion_rate'] = (
    df['email_publishes_90d'] / df['email_creates_90d'].replace(0, float('nan'))
).fillna(0).clip(0, 1)

# Abandonment rate
df['email_abandonment_rate'] = 1 - df['email_completion_rate']

# Test-but-never-send rate
df['test_no_send_rate'] = (
    (df['email_tests_90d'] > 0) & (df['email_publishes_90d'] == 0)
).astype(int)

# Create-but-never-publish rate
df['create_no_publish_rate'] = (
    (df['email_creates_90d'] > 0) & (df['email_publishes_90d'] == 0)
).astype(int)

# Trend: creates 30d vs prev 30d
df['creates_trend'] = df['email_creates_30d'] - df['email_creates_prev30d']
df['publishes_trend'] = df['email_publishes_30d'] - df['email_publishes_prev30d']
df['events_trend'] = df['total_builder_events_30d'] - df['total_builder_events_prev30d']

# Builder maturity tier
df['builder_maturity'] = pd.cut(
    df['email_publishes_90d'],
    bins=[-1, 0, 2, 10, 50, float('inf')],
    labels=['none', 'low', 'medium', 'high', 'power']
)

# Derive product journey stage from behavior
import numpy as np
conditions = [
    (df['email_creates_90d'] == 0) & (df['email_publishes_90d'] == 0),
    (df['email_creates_90d'] > 0) & (df['email_publishes_90d'] == 0),
    (df['email_publishes_90d'] > 0) & (df['email_publishes_90d'] <= 2),
    (df['email_publishes_90d'] > 2),
]
choices = ['1_Unexplored', '2_Explore', '3_Try', '4_Establish']
df['bulk_email_stage'] = np.select(conditions, choices, default='5_Abandon')

# Friction score (composite)
df['friction_score'] = (
    df['test_no_send_rate'] * 0.3 +
    df['create_no_publish_rate'] * 0.3 +
    df['email_abandonment_rate'] * 0.4
).clip(0, 1)

# MRR band
df['mrr_band'] = pd.cut(
    df['avg_mrr'],
    bins=[-1, 0, 20, 50, 100, 300, float('inf')],
    labels=['free', '<$20', '$20-50', '$50-100', '$100-300', '$300+']
)

# Tenure band
df['tenure_band'] = pd.cut(
    df['tenure_months'],
    bins=[-1, 1, 3, 6, 12, 24, float('inf')],
    labels=['<1mo', '1-3mo', '3-6mo', '6-12mo', '12-24mo', '24mo+']
)

# Initiative eligibility flags
df['eligible_rendering_fix'] = ((df['email_publishes_90d'] > 0) & (df['friction_score'] > 0.3)).astype(int)
df['eligible_brandkit'] = ((df['email_creates_90d'] > 2) & (df['ca_template_creates_90d'] == 0)).astype(int)
df['eligible_universal_content'] = ((df['email_publishes_90d'] >= 3)).astype(int)
df['eligible_ai_builder'] = ((df['ai_content_creates_90d'] == 0) & (df['email_creates_90d'] > 0)).astype(int)
df['eligible_template_improvement'] = ((df['email_creates_90d'] > 0) & (df['email_completion_rate'] < 0.5)).astype(int)
df['eligible_omnichannel'] = ((df['email_publishes_90d'] > 0) & (df['sms_creates_90d'] == 0)).astype(int)
df['eligible_activation'] = ((df['bulk_email_stage'].isin(['1_Unexplored', '2_Explore', '5_Abandon'])) if 'bulk_email_stage' in df.columns else False).astype(int)
df['eligible_code_mode'] = ((df['package'].isin(['premium_monthly_plan_v0', 'premium_annual_plan_v0', 'legacy monthly'])) & (df['email_creates_90d'] > 0)).astype(int)

# Total eligible initiatives
elig_cols = [c for c in df.columns if c.startswith('eligible_')]
df['total_eligible_initiatives'] = df[elig_cols].sum(axis=1)

# ── SAVE ─────────────────────────────────────────────────
out_path = "data/customers_engineered.parquet"
print(f"\n[SAVE] Writing {len(df):,} customers to {out_path}...")
df.to_parquet(out_path, index=False)

print(f"\n{'=' * 60}")
print(f"PHASE 1 COMPLETE")
print(f"{'=' * 60}")
print(f"Total customers: {len(df):,}")
print(f"Paid customers: {len(df[df['primary_plan_type'] == 'monthly']):,}")
print(f"Free (active): {len(df[df['primary_plan_type'] == 'free']):,}")
print(f"Features: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(f"\nPlan distribution:")
print(df['package'].value_counts().head(10))
print(f"\nBuilder maturity distribution:")
print(df['builder_maturity'].value_counts())
print(f"\nMRR band distribution:")
print(df['mrr_band'].value_counts())
print(f"\nMean friction score: {df['friction_score'].mean():.3f}")
print(f"Mean completion rate: {df['email_completion_rate'].mean():.3f}")
print(f"\nInitiative eligibility:")
for c in elig_cols:
    print(f"  {c}: {df[c].sum():,} ({100*df[c].mean():.1f}%)")
