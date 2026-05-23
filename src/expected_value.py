"""
Phase 4: Expected Value scoring with survival-adjusted CLV.
For each customer × eligible initiative, compute EV with confidence discount.
"""
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 4: Expected Value Scoring")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} customers")

# --- Survival-adjusted CLV using Kaplan-Meier ---
print("\nFitting Kaplan-Meier survival model...")
kmf = KaplanMeierFitter()

durations = df['tenure_months'].clip(1, 120).values
event_observed = df['churned'].values if 'churned' in df.columns else (df['mrr_status'] == 'churn').astype(int).values

kmf.fit(durations, event_observed=event_observed)
median_survival = kmf.median_survival_time_
print(f"  Median survival: {median_survival:.1f} months")

# Per-segment survival: group by cluster_name
segment_survival = {}
if 'cluster_name' in df.columns:
    for seg_name in df['cluster_name'].unique():
        seg_mask = df['cluster_name'] == seg_name
        seg_dur = durations[seg_mask]
        seg_event = event_observed[seg_mask]
        kmf_seg = KaplanMeierFitter()
        kmf_seg.fit(seg_dur, event_observed=seg_event)
        survival_12 = kmf_seg.predict(12)
        expected_months = min(kmf_seg.median_survival_time_, 60)
        if np.isnan(expected_months) or np.isinf(expected_months):
            expected_months = 24
        segment_survival[seg_name] = {
            'survival_12m': float(survival_12),
            'expected_retention_months': float(expected_months),
        }
        print(f"  {seg_name}: 12mo survival={survival_12:.1%}, expected retention={expected_months:.1f}mo")

# Compute survival-adjusted CLV
df['expected_retention_months'] = df['cluster_name'].map(
    {k: v['expected_retention_months'] for k, v in segment_survival.items()}
).fillna(18)
df['survival_clv'] = df['avg_mrr'] * df['expected_retention_months']
print(f"\nMean survival-adjusted CLV: ${df['survival_clv'].mean():,.0f}")

# --- Initiative definitions ---
INITIATIVES = {
    'rendering_fix': {
        'desc': 'Fix rendering/preview friction',
        'eligible_col': 'eligible_rendering_fix',
        'propensity_col': 'p_completion',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.90,
    },
    'brandkit': {
        'desc': 'Brand Kit adoption campaign',
        'eligible_col': 'eligible_brandkit',
        'propensity_col': 'p_completion',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.80,
    },
    'universal_content': {
        'desc': 'Universal Content Blocks expansion',
        'eligible_col': 'eligible_universal_content',
        'propensity_col': 'p_upgrade',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.85,
    },
    'ai_builder': {
        'desc': 'AI Builder adoption push',
        'eligible_col': 'eligible_ai_builder',
        'propensity_col': 'p_activation',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.75,
    },
    'template_improvement': {
        'desc': 'Template completion rate improvement',
        'eligible_col': 'eligible_template_improvement',
        'propensity_col': 'p_completion',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.85,
    },
    'omnichannel': {
        'desc': 'SMS + Omnichannel cross-sell',
        'eligible_col': 'eligible_omnichannel',
        'propensity_col': 'p_upgrade',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.70,
    },
    'activation': {
        'desc': 'Builder activation for dormant users',
        'eligible_col': 'eligible_activation',
        'propensity_col': 'p_activation',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.60,
    },
    'churn_prevention': {
        'desc': 'Proactive churn prevention',
        'eligible_col': None,
        'propensity_col': 'p_churn',
        'uplift_scenarios': {'low': 0.05, 'base': 0.10, 'high': 0.15},
        'reachability': 0.80,
    },
}

# --- Confidence discount components ---
def compute_confidence_discount(n_segment, predicted_prob, actual_rate):
    sample_score = min(1.0, np.log(n_segment + 1) / np.log(1000 + 1))
    treatment_score = 0.4  # no experiment data
    # Calibration score per decile
    if np.isnan(predicted_prob) or np.isnan(actual_rate):
        calibration_score = 0.5
    else:
        calibration_score = 1.0 - abs(predicted_prob - actual_rate)
        calibration_score = max(0.1, calibration_score)
    return (sample_score * treatment_score * calibration_score) ** (1/3)

# --- Compute EV per customer × initiative ---
print("\nComputing Expected Values for each customer × initiative...")

ev_records = []
for init_name, init_config in INITIATIVES.items():
    elig_col = init_config['eligible_col']
    prop_col = init_config['propensity_col']
    reachability = init_config['reachability']

    if elig_col and elig_col in df.columns:
        eligible_mask = df[elig_col] == 1
    elif init_name == 'churn_prevention':
        eligible_mask = df['p_churn'] > 0.3 if 'p_churn' in df.columns else pd.Series([False] * len(df))
    else:
        continue

    n_eligible = eligible_mask.sum()
    if n_eligible == 0:
        print(f"  {init_name}: 0 eligible — skipping")
        continue

    eligible_df = df[eligible_mask].copy()

    if prop_col not in eligible_df.columns:
        eligible_df[prop_col] = 0.5

    p_success = eligible_df[prop_col].clip(0.01, 0.99)

    # Actual rate for calibration
    if init_name == 'churn_prevention':
        actual_rate = eligible_df['churned'].mean() if 'churned' in eligible_df.columns else 0.04
    else:
        actual_rate = p_success.mean()

    # Confidence discount for each customer
    confidence = eligible_df.apply(
        lambda row: compute_confidence_discount(n_eligible, row.get(prop_col, 0.5), actual_rate),
        axis=1
    )

    for scenario in ['low', 'base', 'high']:
        uplift = init_config['uplift_scenarios'][scenario]

        ev = (
            1.0 *                   # eligibility (already filtered)
            reachability *
            p_success *
            uplift *
            eligible_df['survival_clv'] *
            confidence
        )

        if scenario == 'base':
            for idx, (uid, ev_val) in enumerate(zip(eligible_df['user_id'], ev)):
                ev_records.append({
                    'user_id': uid,
                    'initiative': init_name,
                    'ev_base': ev_val,
                    'ev_low': ev_val * (init_config['uplift_scenarios']['low'] / uplift),
                    'ev_high': ev_val * (init_config['uplift_scenarios']['high'] / uplift),
                    'p_success': p_success.iloc[idx],
                    'reachability': reachability,
                    'confidence': confidence.iloc[idx],
                    'survival_clv': eligible_df['survival_clv'].iloc[idx],
                })

    print(f"  {init_name}: {n_eligible:,} eligible, "
          f"mean EV_base=${ev.mean():.2f}, total=${ev.sum():,.0f}")

ev_df = pd.DataFrame(ev_records)
print(f"\nTotal EV records: {len(ev_df):,}")

# --- Customer-level: argmax EV → assign best initiative ---
print("\nAssigning best initiative per customer (argmax EV)...")
best_per_customer = ev_df.loc[ev_df.groupby('user_id')['ev_base'].idxmax()].copy()
best_per_customer = best_per_customer.rename(columns={'initiative': 'best_initiative'})

# Merge back to main df
df = df.merge(
    best_per_customer[['user_id', 'best_initiative', 'ev_base', 'ev_low', 'ev_high',
                        'p_success', 'confidence']],
    on='user_id', how='left'
)

# Summary by initiative
init_summary = best_per_customer.groupby('best_initiative').agg(
    customers=('user_id', 'count'),
    total_ev_base=('ev_base', 'sum'),
    total_ev_low=('ev_low', 'sum'),
    total_ev_high=('ev_high', 'sum'),
    avg_ev=('ev_base', 'mean'),
    avg_p_success=('p_success', 'mean'),
    avg_confidence=('confidence', 'mean'),
).sort_values('total_ev_base', ascending=False)

print(f"\n{'Initiative Summary (Best-initiative per customer)':^70}")
print("-" * 70)
print(f"{'Initiative':<25} {'Cust':>8} {'EV Base':>12} {'EV Low':>12} {'EV High':>12}")
print("-" * 70)
for name, row in init_summary.iterrows():
    print(f"  {name:<23} {row['customers']:>8,} ${row['total_ev_base']:>11,.0f} "
          f"${row['total_ev_low']:>11,.0f} ${row['total_ev_high']:>11,.0f}")
print("-" * 70)
print(f"  {'TOTAL':<23} {init_summary['customers'].sum():>8,} "
      f"${init_summary['total_ev_base'].sum():>11,.0f} "
      f"${init_summary['total_ev_low'].sum():>11,.0f} "
      f"${init_summary['total_ev_high'].sum():>11,.0f}")

# Save
df.to_parquet(DATA_PATH, index=False)
init_summary.to_csv(os.path.join(OUT_DIR, "initiative_summary.csv"))
ev_df.to_csv(os.path.join(OUT_DIR, "customer_scores.csv"), index=False)

print(f"\nSaved customer_scores.csv ({len(ev_df):,} rows)")
print(f"Saved initiative_summary.csv")
print(f"Updated parquet with best_initiative assignments")
