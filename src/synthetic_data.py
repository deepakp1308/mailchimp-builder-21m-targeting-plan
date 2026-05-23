"""
Generate a synthetic customers_engineered.parquet for modeling development.
Distributions calibrated to known Mailchimp metrics:
- ~1M paid customers
- Package mix: essential 42%, standard 42%, premium 1.5%, legacy 14%
- MRR: median $45, mean $106, P75 $100, P90 $240
- Bulk churn: 40% annually
- Builder activation: 60.6%
- Email completion rate: ~65% avg
- Ecomm connected: ~40%
"""
import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)
N = 1_000_000

print("=" * 60)
print("SYNTHETIC DATA GENERATOR — Mailchimp Builder Analysis")
print(f"Generating {N:,} customers...")
print("=" * 60)

# --- User IDs ---
user_ids = np.arange(1, N + 1)

# --- Package distribution ---
packages = np.random.choice(
    ['essential', 'standard', 'premium', 'legacy'],
    size=N,
    p=[0.42, 0.42, 0.015, 0.145]
)

# --- MRR (log-normal calibrated to median=45, mean~106) ---
mrr_by_pkg = {
    'essential': (np.log(30), 0.8),
    'standard': (np.log(55), 0.85),
    'premium': (np.log(200), 0.7),
    'legacy': (np.log(25), 0.9),
}
avg_mrr = np.zeros(N)
for pkg in ['essential', 'standard', 'premium', 'legacy']:
    mask = packages == pkg
    mu, sigma = mrr_by_pkg[pkg]
    avg_mrr[mask] = np.random.lognormal(mu, sigma, mask.sum()).clip(5, 2000)

plan_amount = avg_mrr * np.random.uniform(0.85, 1.15, N)
total_revenue = avg_mrr * np.random.uniform(3, 60, N)

# --- Tenure ---
tenure_months = np.random.exponential(18, N).clip(1, 120).astype(int)
tenure_days = tenure_months * 30 + np.random.randint(-10, 10, N)

# --- List size (log-normal) ---
list_size = np.random.lognormal(np.log(500), 1.5, N).clip(10, 500_000).astype(int)
subscribed_size = (list_size * np.random.uniform(0.4, 0.95, N)).astype(int)
list_count = np.random.choice([1, 2, 3, 4, 5, 8, 12], N, p=[0.35, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03])

# --- Country ---
countries = np.random.choice(
    ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'BR', 'IN', 'OTHER'],
    N, p=[0.45, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.12]
)
country_groups = np.where(np.isin(countries, ['US', 'CA']), 'NA',
                 np.where(np.isin(countries, ['GB', 'DE', 'FR']), 'EU',
                 np.where(np.isin(countries, ['AU']), 'APAC', 'ROW')))

# --- Ecomm ---
ecomm_levels = np.random.choice(
    ['none', 'basic', 'standard', 'advanced'],
    N, p=[0.60, 0.15, 0.15, 0.10]
)

# --- Business attributes ---
business_size = np.random.choice(['SB', 'MM', 'SE'], N, p=[0.70, 0.25, 0.05])
business_type = np.random.choice(['SBB', 'PBB'], N, p=[0.65, 0.35])
intuit_vertical = np.random.choice(
    ['retail', 'services', 'health', 'tech', 'food', 'education', 'other'],
    N, p=[0.22, 0.20, 0.10, 0.12, 0.08, 0.08, 0.20]
)

# --- Builder behavior (90d) ---
builder_activated = np.random.random(N) < 0.606

email_creates_90d = np.where(builder_activated,
    np.random.poisson(5.5, N).clip(0, 80), 0)
email_publishes_90d = np.where(
    email_creates_90d > 0,
    (email_creates_90d * np.random.beta(4, 2.2, N)).astype(int).clip(0),
    0
)
email_tests_90d = np.where(
    email_creates_90d > 0,
    np.random.poisson(1.5, N).clip(0, 30), 0)

ca_template_creates_90d = np.where(
    email_creates_90d > 0,
    np.random.binomial(email_creates_90d, 0.25, N), 0)
ai_content_creates_90d = np.where(
    email_creates_90d > 0,
    np.random.binomial(email_creates_90d, 0.15, N), 0)
sms_creates_90d = np.where(
    builder_activated, np.random.poisson(0.8, N).clip(0, 20), 0)
sms_publishes_90d = np.where(
    sms_creates_90d > 0,
    (sms_creates_90d * np.random.beta(3, 2, N)).astype(int), 0)
automation_creates_90d = np.where(
    builder_activated, np.random.poisson(0.6, N).clip(0, 15), 0)
automation_publishes_90d = np.where(
    automation_creates_90d > 0,
    (automation_creates_90d * np.random.beta(3, 3, N)).astype(int), 0)
builder_active_days_90d = np.where(
    builder_activated, np.random.poisson(12, N).clip(1, 90), 0)
total_builder_events_90d = (email_creates_90d + email_publishes_90d +
    email_tests_90d + sms_creates_90d + automation_creates_90d +
    np.random.poisson(3, N))

# --- Builder behavior (30d) ---
email_creates_30d = (email_creates_90d * np.random.uniform(0.25, 0.45, N)).astype(int)
email_publishes_30d = (email_publishes_90d * np.random.uniform(0.25, 0.45, N)).astype(int)
email_tests_30d = (email_tests_90d * np.random.uniform(0.2, 0.5, N)).astype(int)
total_builder_events_30d = (total_builder_events_90d * np.random.uniform(0.25, 0.45, N)).astype(int)

# --- Builder behavior (prev 30d) ---
email_creates_prev30d = (email_creates_90d * np.random.uniform(0.20, 0.40, N)).astype(int)
email_publishes_prev30d = (email_publishes_90d * np.random.uniform(0.20, 0.40, N)).astype(int)
total_builder_events_prev30d = (total_builder_events_90d * np.random.uniform(0.20, 0.40, N)).astype(int)

# --- Health metrics ---
avg_open_rate = np.random.beta(5, 15, N).clip(0.02, 0.7)
avg_click_rate = avg_open_rate * np.random.uniform(0.05, 0.3, N)
avg_bounce_rate = np.random.beta(1.5, 30, N).clip(0, 0.15)
total_delivered_3mo = np.where(email_publishes_90d > 0,
    list_size * email_publishes_90d * np.random.uniform(0.7, 1.0, N), 0).astype(int)
total_revenue_3mo = np.where(ecomm_levels != 'none',
    np.random.lognormal(np.log(500), 1.8, N).clip(0, 200_000), 0)
total_orders_3mo = np.where(ecomm_levels != 'none',
    np.random.poisson(15, N).clip(0, 500), 0)
avg_health_score = np.random.beta(6, 4, N) * 100
avg_engagement_score = np.random.beta(5, 5, N) * 100
avg_deliverability_score = np.random.beta(8, 2, N) * 100
avg_revenue_score = np.random.beta(3, 5, N) * 100

# --- MRR status ---
mrr_statuses = np.random.choice(
    ['new', 'expansion', 'contraction', 'churn', 'resurrection', 'flat'],
    N, p=[0.08, 0.12, 0.10, 0.04, 0.03, 0.63]
)
current_mrr_net = avg_mrr * np.where(mrr_statuses == 'churn', 0,
    np.where(mrr_statuses == 'contraction', np.random.uniform(0.5, 0.9, N),
    np.where(mrr_statuses == 'expansion', np.random.uniform(1.05, 1.5, N), 1.0)))
prev_mrr_net = avg_mrr

# --- Churn indicator (40% annual ≈ 4.2% monthly) ---
churned = np.random.random(N) < 0.042

# --- Derived features ---
email_completion_rate = np.where(email_creates_90d > 0,
    email_publishes_90d / email_creates_90d, 0).clip(0, 1)

email_abandonment_rate = 1 - email_completion_rate

test_no_send_rate = ((email_tests_90d > 0) & (email_publishes_90d == 0)).astype(int)
create_no_publish_rate = ((email_creates_90d > 0) & (email_publishes_90d == 0)).astype(int)

creates_trend = email_creates_30d - email_creates_prev30d
publishes_trend = email_publishes_30d - email_publishes_prev30d
events_trend = total_builder_events_30d - total_builder_events_prev30d

builder_maturity = pd.cut(
    pd.Series(email_publishes_90d),
    bins=[-1, 0, 2, 10, 50, float('inf')],
    labels=['none', 'low', 'medium', 'high', 'power']
)

conditions = [
    (email_creates_90d == 0) & (email_publishes_90d == 0),
    (email_creates_90d > 0) & (email_publishes_90d == 0),
    (email_publishes_90d > 0) & (email_publishes_90d <= 2),
    (email_publishes_90d > 2),
]
choices = ['1_Unexplored', '2_Explore', '3_Try', '4_Establish']
bulk_email_stage = np.select(conditions, choices, default='1_Unexplored')

friction_score = (
    test_no_send_rate * 0.3 +
    create_no_publish_rate * 0.3 +
    email_abandonment_rate * 0.4
).clip(0, 1)

mrr_band = pd.cut(
    pd.Series(avg_mrr),
    bins=[-1, 0, 20, 50, 100, 300, float('inf')],
    labels=['free', '<$20', '$20-50', '$50-100', '$100-300', '$300+']
)

tenure_band = pd.cut(
    pd.Series(tenure_months),
    bins=[-1, 1, 3, 6, 12, 24, float('inf')],
    labels=['<1mo', '1-3mo', '3-6mo', '6-12mo', '12-24mo', '24mo+']
)

# --- Initiative eligibility ---
eligible_rendering_fix = ((email_publishes_90d > 0) & (friction_score > 0.3)).astype(int)
eligible_brandkit = ((email_creates_90d > 2) & (ca_template_creates_90d == 0)).astype(int)
eligible_universal_content = (email_publishes_90d >= 3).astype(int)
eligible_ai_builder = ((ai_content_creates_90d == 0) & (email_creates_90d > 0)).astype(int)
eligible_template_improvement = ((email_creates_90d > 0) & (email_completion_rate < 0.5)).astype(int)
eligible_omnichannel = ((email_publishes_90d > 0) & (sms_creates_90d == 0)).astype(int)
eligible_activation = np.isin(bulk_email_stage, ['1_Unexplored', '2_Explore']).astype(int)
eligible_code_mode = (np.isin(packages, ['premium']) & (email_creates_90d > 0)).astype(int)

is_active_30d = (~churned).astype(int)
is_high_value = (avg_mrr >= 200).astype(int)
cs_tiers = np.random.choice(['none', 'low', 'mid', 'high'], N, p=[0.60, 0.20, 0.15, 0.05])
naics_icp = np.random.choice(['icp_strong', 'icp_moderate', 'icp_weak'], N, p=[0.30, 0.45, 0.25])

# --- Build DataFrame ---
df = pd.DataFrame({
    'user_id': user_ids,
    'primary_plan_type': 'monthly',
    'package': packages,
    'plan_amount': plan_amount.round(2),
    'avg_mrr': avg_mrr.round(2),
    'total_revenue': total_revenue.round(2),
    'country': countries,
    'country_group': country_groups,
    'ecomm_level': ecomm_levels,
    'is_high_value': is_high_value,
    'customer_success_tier': cs_tiers,
    'tenure_days': tenure_days,
    'tenure_months': tenure_months,
    'list_size': list_size,
    'subscribed_size': subscribed_size,
    'list_count': list_count,
    'is_active_30d': is_active_30d,
    'intuit_vertical': intuit_vertical,
    'naics_icp': naics_icp,
    'business_size': business_size,
    'business_type': business_type,
    'email_creates_90d': email_creates_90d,
    'email_publishes_90d': email_publishes_90d,
    'email_tests_90d': email_tests_90d,
    'ca_template_creates_90d': ca_template_creates_90d,
    'ai_content_creates_90d': ai_content_creates_90d,
    'sms_creates_90d': sms_creates_90d,
    'sms_publishes_90d': sms_publishes_90d,
    'automation_creates_90d': automation_creates_90d,
    'automation_publishes_90d': automation_publishes_90d,
    'builder_active_days_90d': builder_active_days_90d,
    'total_builder_events_90d': total_builder_events_90d,
    'email_creates_30d': email_creates_30d,
    'email_publishes_30d': email_publishes_30d,
    'email_tests_30d': email_tests_30d,
    'total_builder_events_30d': total_builder_events_30d,
    'email_creates_prev30d': email_creates_prev30d,
    'email_publishes_prev30d': email_publishes_prev30d,
    'total_builder_events_prev30d': total_builder_events_prev30d,
    'avg_open_rate': avg_open_rate.round(4),
    'avg_click_rate': avg_click_rate.round(4),
    'avg_bounce_rate': avg_bounce_rate.round(4),
    'total_delivered_3mo': total_delivered_3mo,
    'total_revenue_3mo': total_revenue_3mo.round(2),
    'total_orders_3mo': total_orders_3mo,
    'avg_health_score': avg_health_score.round(2),
    'avg_engagement_score': avg_engagement_score.round(2),
    'avg_deliverability_score': avg_deliverability_score.round(2),
    'avg_revenue_score': avg_revenue_score.round(2),
    'current_mrr_net': current_mrr_net.round(2),
    'prev_mrr_net': prev_mrr_net.round(2),
    'mrr_status': mrr_statuses,
    'current_plan_type_finance': 'monthly',
    'email_completion_rate': email_completion_rate.round(4),
    'email_abandonment_rate': email_abandonment_rate.round(4),
    'test_no_send_rate': test_no_send_rate,
    'create_no_publish_rate': create_no_publish_rate,
    'creates_trend': creates_trend,
    'publishes_trend': publishes_trend,
    'events_trend': events_trend,
    'builder_maturity': builder_maturity,
    'bulk_email_stage': bulk_email_stage,
    'friction_score': friction_score.round(4),
    'mrr_band': mrr_band,
    'tenure_band': tenure_band,
    'eligible_rendering_fix': eligible_rendering_fix,
    'eligible_brandkit': eligible_brandkit,
    'eligible_universal_content': eligible_universal_content,
    'eligible_ai_builder': eligible_ai_builder,
    'eligible_template_improvement': eligible_template_improvement,
    'eligible_omnichannel': eligible_omnichannel,
    'eligible_activation': eligible_activation,
    'eligible_code_mode': eligible_code_mode,
    'churned': churned.astype(int),
    'total_eligible_initiatives': (eligible_rendering_fix + eligible_brandkit +
        eligible_universal_content + eligible_ai_builder +
        eligible_template_improvement + eligible_omnichannel +
        eligible_activation + eligible_code_mode),
})

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "customers_engineered.parquet")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_parquet(out_path, index=False)

print(f"\nSaved {len(df):,} rows to {out_path}")
print(f"Columns: {len(df.columns)}")
print(f"\nPackage distribution:")
print(df['package'].value_counts())
print(f"\nMRR stats: median=${df['avg_mrr'].median():.0f}, mean=${df['avg_mrr'].mean():.0f}, P75=${df['avg_mrr'].quantile(0.75):.0f}, P90=${df['avg_mrr'].quantile(0.90):.0f}")
print(f"Builder activation: {(df['email_creates_90d'] > 0).mean():.1%}")
print(f"Avg completion rate (active): {df.loc[df['email_creates_90d'] > 0, 'email_completion_rate'].mean():.1%}")
print(f"Ecomm connected: {(df['ecomm_level'] != 'none').mean():.1%}")
print(f"Churn rate (monthly): {df['churned'].mean():.1%}")
print(f"\n⚠️  SYNTHETIC DATA — for model development only")
