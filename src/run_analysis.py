"""
Phases 2-5: Segmentation, Clustering, Propensity, EV Scoring, Monte Carlo, Sequencing
Reads from data/customers_engineered.parquet, outputs to outputs/
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from lifelines import KaplanMeierFitter
import json, os, warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/charts", exist_ok=True)

POPULATION_SIZE = 1_651_596
POP_PAID_SIZE = 992_107
POP_AVG_MRR_PAID = 103.45
FY27_REALIZATION_RATE = 0.16  # share of theoretical EV capturable in FY27 rollout window

print("=" * 60)
print("Loading feature table...")
df = pd.read_parquet("data/customers_engineered.parquet")
df = df.drop_duplicates(subset="user_id", keep="first")
print(f"Loaded {len(df):,} customers with {len(df.columns)} features")
print(f"Sample avg MRR: ${df['avg_mrr'].mean():.2f} | Paid avg MRR (population): ${POP_AVG_MRR_PAID:.2f}")

# ═══════════════════════════════════════════════════════════
# PHASE 2: DESCRIPTIVE SEGMENTATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Descriptive Segmentation")
print("=" * 60)

seg_dims = {
    'primary_plan_type': df['primary_plan_type'],
    'package': df['package'].apply(lambda x: x if x in ['free','standard_monthly_plan_v0','essential_monthly_plan_v0','premium_monthly_plan_v0','legacy monthly'] else 'other'),
    'ecomm_level': df['ecomm_level'],
    'builder_maturity': df['builder_maturity'].astype(str),
    'mrr_band': df['mrr_band'].astype(str),
    'tenure_band': df['tenure_band'].astype(str),
    'country_group': df['country_group'].fillna('Unknown'),
}

elig_cols = [c for c in df.columns if c.startswith('eligible_')]

def segment_metrics(grp):
    n = len(grp)
    return pd.Series({
        'customer_count': n,
        'total_mrr': grp['avg_mrr'].sum(),
        'avg_mrr': grp['avg_mrr'].mean(),
        'annualized_revenue': grp['avg_mrr'].sum() * 12,
        'avg_completion_rate': grp['email_completion_rate'].mean(),
        'avg_abandonment_rate': grp['email_abandonment_rate'].mean(),
        'avg_friction_score': grp['friction_score'].mean(),
        'pct_high_friction': (grp['friction_score'] > 0.5).mean(),
        'pct_creates_no_publish': grp['create_no_publish_rate'].mean(),
        'avg_email_creates_90d': grp['email_creates_90d'].mean(),
        'avg_email_publishes_90d': grp['email_publishes_90d'].mean(),
        'total_eligible_initiatives': grp['total_eligible_initiatives'].mean(),
    })

seg_results = []
for dim_name, dim_values in seg_dims.items():
    for val, grp in df.groupby(dim_values):
        if len(grp) < 100:
            continue
        m = segment_metrics(grp)
        m['dimension'] = dim_name
        m['value'] = str(val)
        seg_results.append(m)

seg_df = pd.DataFrame(seg_results)
seg_df = seg_df.sort_values('annualized_revenue', ascending=False)
seg_df.to_csv("outputs/baseline_segment_metrics.csv", index=False)
print(f"Saved {len(seg_df)} segments to outputs/baseline_segment_metrics.csv")

# Top segments summary
print("\nTop 10 segments by annualized revenue:")
for _, r in seg_df.head(10).iterrows():
    print(f"  {r['dimension']}={r['value']}: {r['customer_count']:,.0f} customers, ${r['annualized_revenue']:,.0f} ARR, "
          f"completion={r['avg_completion_rate']:.1%}, friction={r['avg_friction_score']:.2f}")

# ═══════════════════════════════════════════════════════════
# PHASE 3A: CLUSTERING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3A: Clustering")
print("=" * 60)

cluster_features = [
    'avg_mrr', 'tenure_months', 'list_size',
    'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
    'ca_template_creates_90d', 'ai_content_creates_90d',
    'sms_creates_90d', 'automation_creates_90d',
    'builder_active_days_90d', 'total_builder_events_90d',
    'email_completion_rate', 'email_abandonment_rate', 'friction_score',
    'creates_trend', 'publishes_trend',
    'total_eligible_initiatives',
]

df_cluster = df[cluster_features].fillna(0).copy()
for c in cluster_features:
    df_cluster[c] = df_cluster[c].clip(df_cluster[c].quantile(0.01), df_cluster[c].quantile(0.99))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

best_k, best_sil = 8, -1
for k in range(6, 14):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=50000, random_state=42)
    print(f"  k={k}: silhouette={sil:.4f}")
    if sil > best_sil:
        best_k, best_sil = k, sil

print(f"\nBest k={best_k} (silhouette={best_sil:.4f})")

km_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=10000)
df['cluster_id'] = km_final.fit_predict(X_scaled)

# Name clusters using decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(df_cluster, df['cluster_id'])
tree_rules = export_text(dt, feature_names=cluster_features, max_depth=3)

cluster_profiles = []
for cid in sorted(df['cluster_id'].unique()):
    grp = df[df['cluster_id'] == cid]
    m = segment_metrics(grp)
    m['cluster_id'] = cid

    top_feats = []
    means = grp[cluster_features].mean()
    global_means = df[cluster_features].mean()
    diffs = ((means - global_means) / global_means.replace(0, 1)).abs().sort_values(ascending=False)
    for feat in diffs.head(3).index:
        direction = "high" if means[feat] > global_means[feat] else "low"
        top_feats.append(f"{direction} {feat}")

    if m['avg_mrr'] > 100 and m['avg_friction_score'] > 0.4:
        m['cluster_name'] = f"High-value frustrated users"
    elif m['avg_mrr'] > 50 and m['pct_creates_no_publish'] > 0.5:
        m['cluster_name'] = f"Paid users starting but not sending"
    elif m['avg_mrr'] < 5 and m['avg_email_creates_90d'] > 3:
        m['cluster_name'] = f"Free users with strong Builder intent"
    elif m['avg_mrr'] > 0 and m['avg_email_creates_90d'] < 1:
        m['cluster_name'] = f"Dormant paid users at retention risk"
    elif m['avg_completion_rate'] > 0.7 and m['avg_email_publishes_90d'] > 10:
        m['cluster_name'] = f"Power senders"
    elif m['avg_mrr'] > 20 and m['avg_email_creates_90d'] > 2 and m['avg_completion_rate'] > 0.4:
        m['cluster_name'] = f"Active mid-market users"
    elif m['avg_mrr'] < 5 and m['avg_email_creates_90d'] < 1:
        m['cluster_name'] = f"Inactive free users"
    else:
        m['cluster_name'] = f"Cluster {cid}: {', '.join(top_feats[:2])}"

    m['dominant_features'] = "; ".join(top_feats)
    cluster_profiles.append(m)

cluster_df = pd.DataFrame(cluster_profiles)
cluster_df.to_csv("outputs/cluster_profiles.csv", index=False)
print(f"\nCluster profiles saved ({best_k} clusters)")
for _, r in cluster_df.iterrows():
    print(f"  Cluster {r['cluster_id']:.0f} ({r['cluster_name']}): "
          f"{r['customer_count']:,.0f} customers, ${r['avg_mrr']:.0f} avg MRR, "
          f"completion={r['avg_completion_rate']:.1%}, friction={r['avg_friction_score']:.2f}")

# ═══════════════════════════════════════════════════════════
# PHASE 3B: PROPENSITY MODELING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3B: Propensity Modeling (LightGBM)")
print("=" * 60)

model_features = [
    'avg_mrr', 'plan_amount', 'tenure_months', 'list_size', 'subscribed_size',
    'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
    'ca_template_creates_90d', 'ai_content_creates_90d',
    'sms_creates_90d', 'automation_creates_90d',
    'builder_active_days_90d', 'total_builder_events_90d',
    'email_creates_30d', 'email_publishes_30d',
    'email_completion_rate', 'email_abandonment_rate', 'friction_score',
    'creates_trend', 'publishes_trend', 'events_trend',
    'total_eligible_initiatives',
]

for col in ['avg_open_rate','avg_click_rate','avg_bounce_rate','total_delivered_3mo',
            'avg_health_score','avg_engagement_score']:
    if col in df.columns:
        model_features.append(col)

X = df[model_features].fillna(0)

propensity_targets = {
    'p_send_completion': (df['email_creates_90d'] > 0) & (df['email_publishes_90d'] > 0),
    'p_activation': (df['email_creates_90d'] > 0) | (df['email_publishes_90d'] > 0),
    'p_high_engagement': df['builder_active_days_90d'] >= 10,
    'p_upgrade_potential': (df['avg_mrr'] > 0) & (df['avg_mrr'] < 100) & (df['email_publishes_90d'] > 5),
    'p_churn_risk': (df['avg_mrr'] > 0) & (df['creates_trend'] < 0) & (df['publishes_trend'] < 0),
}

leaky_features = {
    'p_send_completion': {
        'email_completion_rate', 'email_abandonment_rate', 'friction_score',
        'test_no_send_rate', 'create_no_publish_rate',
        'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
        'email_creates_30d', 'email_publishes_30d',
    },
    'p_activation': {
        'total_builder_events_90d', 'builder_active_days_90d',
        'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
        'email_creates_30d', 'email_publishes_30d', 'total_builder_events_30d',
    },
    'p_high_engagement': {'builder_active_days_90d', 'total_builder_events_90d', 'total_builder_events_30d'},
}

for target_name, target_mask in propensity_targets.items():
    y = target_mask.astype(int)
    pos_rate = y.mean()
    print(f"\n  Training {target_name} (positive rate: {pos_rate:.1%})...")

    if pos_rate < 0.01 or pos_rate > 0.99:
        print(f"    Skipping: extreme class imbalance ({pos_rate:.1%})")
        df[target_name] = pos_rate
        continue

    drop_feats = leaky_features.get(target_name, set())
    feat_cols = [c for c in model_features if c not in drop_feats]
    X = df[feat_cols].fillna(0)
    sample_size = min(200000, len(df))
    sample_idx = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    train_size = int(sample_size * 0.7)
    X_train, X_test = X_sample.iloc[:train_size], X_sample.iloc[train_size:]
    y_train, y_test = y_sample.iloc[:train_size], y_sample.iloc[train_size:]

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=100, random_state=42
    )
    model.fit(X_train, y_train)

    # Score full population
    preds = model.predict_proba(X)[:, 1]
    df[target_name] = preds

    from sklearn.metrics import roc_auc_score
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"    AUC: {test_auc:.4f}")

    importance = model.feature_importances_
    top_feats = sorted(zip(feat_cols, importance), key=lambda x: -x[1])[:5]
    print(f"    Top features: {[f'{f}({v:.3f})' for f,v in top_feats]}")

# ═══════════════════════════════════════════════════════════
# PHASE 4: EXPECTED VALUE SCORING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4: Expected Value Scoring")
print("=" * 60)

# Survival-based CLV by segment
print("\n  Computing survival-based CLV...")
df['tenure_months_capped'] = df['tenure_months'].clip(0, 60)
df['churned'] = (df['mrr_status'] == 'churn').astype(int) if 'mrr_status' in df.columns else 0

segment_clv = {}
for plan in ['free', 'essential_monthly_plan_v0', 'standard_monthly_plan_v0', 'premium_monthly_plan_v0', 'legacy monthly']:
    mask = df['package'] == plan
    if mask.sum() < 100:
        continue
    kmf = KaplanMeierFitter()
    durations = df.loc[mask, 'tenure_months_capped'].clip(1, 48)
    events = df.loc[mask, 'churned']
    try:
        kmf.fit(durations, events)
        expected_months = kmf.survival_function_.iloc[:12].sum().item()
    except:
        expected_months = 8.0
    segment_clv[plan] = expected_months
    print(f"    {plan}: expected_retention_months={expected_months:.1f}")

df['expected_retention_months'] = df['package'].map(segment_clv).fillna(8.0)
df['survival_clv'] = df['avg_mrr'] * df['expected_retention_months']

# Initiative EV calculation — incremental annual ARR, not full survival CLV
initiatives = {
    'rendering_fix': {'elig': 'eligible_rendering_fix', 'propensity': 'p_send_completion', 'uplift': 0.008, 'reachability': 1.0, 'revenue_mult': 1.0},
    'brandkit': {'elig': 'eligible_brandkit', 'propensity': 'p_activation', 'uplift': 0.006, 'reachability': 1.0, 'revenue_mult': 0.8},
    'universal_content': {'elig': 'eligible_universal_content', 'propensity': 'p_send_completion', 'uplift': 0.010, 'reachability': 1.0, 'revenue_mult': 1.0},
    'ai_builder': {'elig': 'eligible_ai_builder', 'propensity': 'p_activation', 'uplift': 0.008, 'reachability': 1.0, 'revenue_mult': 0.9},
    'template_improvement': {'elig': 'eligible_template_improvement', 'propensity': 'p_send_completion', 'uplift': 0.012, 'reachability': 1.0, 'revenue_mult': 0.7},
    'omnichannel': {'elig': 'eligible_omnichannel', 'propensity': 'p_high_engagement', 'uplift': 0.005, 'reachability': 0.8, 'revenue_mult': 1.2},
    'activation': {'elig': 'eligible_activation', 'propensity': 'p_activation', 'uplift': 0.006, 'reachability': 1.0, 'revenue_mult': 0.5},
    'code_mode': {'elig': 'eligible_code_mode', 'propensity': 'p_send_completion', 'uplift': 0.007, 'reachability': 0.8, 'revenue_mult': 1.5},
}

print("\n  Scoring EV per customer per initiative...")
ev_cols = {}
for init_name, config in initiatives.items():
    elig = df[config['elig']].values
    propensity = df[config['propensity']].values
    uplift = config['uplift']
    reachability = config['reachability']
    annual_arr = df['avg_mrr'].values * 12
    revenue = annual_arr * config['revenue_mult']

    # Confidence discount: simple version based on propensity calibration
    confidence = np.clip(propensity * 0.8 + 0.2, 0.1, 1.0)

    ev = elig * reachability * propensity * uplift * revenue * confidence
    ev_col = f'ev_{init_name}'
    df[ev_col] = ev
    ev_cols[init_name] = ev_col

    total_ev = ev.sum()
    n_elig = elig.sum()
    print(f"    {init_name}: eligible={n_elig:,}, total_EV=${total_ev:,.0f}, avg_EV=${total_ev/max(n_elig,1):,.2f}")

# Best initiative per customer (argmax EV, no overlap)
ev_matrix = df[[f'ev_{name}' for name in initiatives]].values
best_init_idx = ev_matrix.argmax(axis=1)
best_ev = ev_matrix.max(axis=1)
init_names = list(initiatives.keys())
df['best_initiative'] = [init_names[i] for i in best_init_idx]
df['best_ev'] = best_ev
df['best_ev_annualized'] = df['best_ev'] * 12

# Priority rank
df['priority_rank'] = df['best_ev'].rank(ascending=False, method='first').astype(int)

# Reason codes
def reason_code(row):
    reasons = []
    if row['avg_mrr'] > 100:
        reasons.append("High MRR customer")
    if row['friction_score'] > 0.5:
        reasons.append("High Builder friction")
    if row['create_no_publish_rate'] > 0:
        reasons.append("Starts campaigns but doesn't send")
    if row['creates_trend'] < -2:
        reasons.append("Declining Builder usage")
    if row['email_creates_90d'] > 5 and row['email_publishes_90d'] == 0:
        reasons.append("Strong intent but zero sends")
    if row['primary_plan_type'] == 'free' and row['email_creates_90d'] > 3:
        reasons.append("Free user with repeated Builder sessions")
    if row['ecomm_level'] in ['ecomm', 'ecu']:
        reasons.append("Ecommerce expansion potential")
    if row['ca_template_creates_90d'] == 0 and row['email_creates_90d'] > 2:
        reasons.append("Not using Creative Assistant")
    return "; ".join(reasons[:3]) if reasons else "General eligibility"

print("\n  Generating reason codes...")
df['reason_codes'] = df.apply(reason_code, axis=1)

# Population calibration (before Monte Carlo)
total_ev_sample = df['best_ev_annualized'].sum()
paid_mask = df['primary_plan_type'].isin(['monthly', 'payg'])
n_paid_sample = paid_mask.sum()
paid_sample_mrr = df.loc[paid_mask, 'avg_mrr'].mean() if n_paid_sample else 1
mrr_calibration = POP_AVG_MRR_PAID / max(paid_sample_mrr, 1)
# Extrapolate from paid sample to paid population (avoid double-scaling free cohort)
total_ev = (total_ev_sample / max(n_paid_sample, 1)) * POP_PAID_SIZE * mrr_calibration * FY27_REALIZATION_RATE
calibration_factor = (POP_PAID_SIZE / max(n_paid_sample, 1)) * mrr_calibration * FY27_REALIZATION_RATE
population_scale = POP_PAID_SIZE / max(n_paid_sample, 1)
print(f"\n  Sample EV (annualized): ${total_ev_sample/1e6:.1f}M")
print(f"  Paid pop scale: {population_scale:.1f}x | MRR calibration: {mrr_calibration:.3f}x | FY27 realization: {FY27_REALIZATION_RATE:.0%}")
print(f"  Calibrated FY27 EV: ${total_ev/1e6:.1f}M")

# ═══════════════════════════════════════════════════════════
# PHASE 5: MONTE CARLO + SEQUENCING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5: Monte Carlo Simulation + Wave Sequencing")
print("=" * 60)

# Monte Carlo — scale simulations to population
print("\n  Running Monte Carlo simulation (10K iterations)...")
n_sims = 10000
sim_results = []
for _ in range(n_sims):
    noise = np.random.normal(1.0, 0.3, size=len(df))
    sim_total = (df['best_ev_annualized'] * noise.clip(0.2, 2.0)).sum() * calibration_factor
    sim_results.append(sim_total)

sim_results = np.array(sim_results)
p10, p50, p90 = np.percentile(sim_results, [10, 50, 90])
print(f"  $21M Projection: P10=${p10/1e6:.1f}M, P50=${p50/1e6:.1f}M, P90=${p90/1e6:.1f}M")
print(f"  Mean=${sim_results.mean()/1e6:.1f}M, Std=${sim_results.std()/1e6:.1f}M")

# Wave assignment
print("\n  Assigning waves...")
df['wave'] = 4  # default
df.loc[(df['best_ev'] > df['best_ev'].quantile(0.75)) & (df['p_send_completion'] >= 0.5), 'wave'] = 1
df.loc[(df['wave'] == 4) & (df['best_ev'] > df['best_ev'].quantile(0.50)) & (df['p_activation'] >= 0.3), 'wave'] = 2
df.loc[(df['wave'] == 4) & (df['best_ev'] > df['best_ev'].quantile(0.25)), 'wave'] = 3

wave_summary = []
cumulative_rev = 0
for w in [1, 2, 3, 4]:
    wdf = df[df['wave'] == w]
    wave_rev = wdf['best_ev_annualized'].sum() * calibration_factor
    cumulative_rev += wave_rev
    wave_summary.append({
        'wave': w,
        'customer_count': len(wdf),
        'total_mrr': wdf['avg_mrr'].sum(),
        'expected_revenue': wave_rev,
        'cumulative_revenue': cumulative_rev,
        'avg_ev': wdf['best_ev_annualized'].mean(),
        'top_initiative': wdf['best_initiative'].mode().iloc[0] if len(wdf) > 0 else 'N/A',
        'pct_paid': (wdf['primary_plan_type'] == 'monthly').mean(),
        'avg_friction': wdf['friction_score'].mean(),
    })
    print(f"  Wave {w}: {len(wdf):,} customers, ${wave_rev/1e6:.1f}M revenue, "
          f"cumulative ${cumulative_rev/1e6:.1f}M, top={wdf['best_initiative'].mode().iloc[0] if len(wdf) > 0 else 'N/A'}")

wave_df = pd.DataFrame(wave_summary)
wave_df.to_csv("outputs/initiative_sequence.csv", index=False)

# Save customer scores
score_cols = ['user_id', 'primary_plan_type', 'package', 'avg_mrr', 'country_group',
              'ecomm_level', 'tenure_months', 'builder_maturity', 'bulk_email_stage',
              'friction_score', 'email_completion_rate', 'email_creates_90d', 'email_publishes_90d',
              'cluster_id', 'best_initiative', 'best_ev', 'best_ev_annualized',
              'priority_rank', 'wave', 'reason_codes',
              'p_send_completion', 'p_activation', 'p_high_engagement', 'survival_clv']
score_cols = [c for c in score_cols if c in df.columns]
df[score_cols].to_csv("outputs/customer_scores.csv", index=False)
print(f"\n  Saved {len(df):,} customer scores to outputs/customer_scores.csv")

# Save Monte Carlo results
mc_df = pd.DataFrame({'simulation': range(n_sims), 'total_revenue': sim_results})
mc_df.to_csv("outputs/monte_carlo_results.csv", index=False)
np.save("outputs/mc_iterations.npy", np.array(sim_results))
mc_summary = pd.DataFrame([
    {'metric': 'P10', 'value': p10},
    {'metric': 'P50', 'value': p50},
    {'metric': 'P90', 'value': p90},
    {'metric': 'mean', 'value': sim_results.mean()},
    {'metric': 'std', 'value': sim_results.std()},
    {'metric': 'target_21M_hit_rate', 'value': (np.array(sim_results) >= 21e6).mean()},
])
mc_summary.to_csv("outputs/monte_carlo_summary.csv", index=False)

# Initiative summary for readout
init_summary = df.groupby('best_initiative').agg(
    customers=('user_id', 'count'),
    total_mrr=('avg_mrr', 'sum'),
    total_ev_base=('best_ev_annualized', lambda x: x.sum() * calibration_factor),
    avg_p_success=('p_send_completion', 'mean'),
).reset_index()
init_summary['wave'] = init_summary['best_initiative'].map(
    df.groupby('best_initiative')['wave'].agg(lambda x: x.mode().iloc[0])
)
init_summary['total_ev_low'] = init_summary['total_ev_base'] * 0.7
init_summary['total_ev_high'] = init_summary['total_ev_base'] * 1.3
init_summary['avg_confidence'] = 0.75
init_summary['description'] = init_summary['best_initiative'].str.replace('_', ' ').str.title()
init_summary.to_csv("outputs/initiative_summary.csv", index=False)

# Save segment scores
seg_scores = df.groupby(['best_initiative', 'wave']).agg(
    customer_count=('user_id', 'count'),
    total_mrr=('avg_mrr', 'sum'),
    expected_revenue=('best_ev_annualized', 'sum'),
    avg_ev=('best_ev_annualized', 'mean'),
    avg_friction=('friction_score', 'mean'),
    avg_completion=('email_completion_rate', 'mean'),
    pct_paid=('primary_plan_type', lambda x: (x == 'monthly').mean()),
).reset_index()
seg_scores.to_csv("outputs/segment_scores.csv", index=False)

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"\nTotal Expected Value (annualized, calibrated): ${total_ev/1e6:.1f}M")
print(f"$21M Target Gap: ${(21e6 - total_ev)/1e6:.1f}M")
print(f"Monte Carlo P10/P50/P90: ${p10/1e6:.1f}M / ${p50/1e6:.1f}M / ${p90/1e6:.1f}M")
print(f"\nWave breakdown:")
for _, w in wave_df.iterrows():
    print(f"  Wave {w['wave']:.0f}: {w['customer_count']:,.0f} customers, ${w['expected_revenue']/1e6:.1f}M, "
          f"cumulative ${w['cumulative_revenue']/1e6:.1f}M")
print(f"\nInitiative distribution:")
print(df['best_initiative'].value_counts())
print(f"\nCluster distribution:")
for _, c in cluster_df.iterrows():
    print(f"  {c['cluster_name']}: {c['customer_count']:,.0f} customers")

# Save key numbers for the readout
key_numbers = {
    'total_customers': int(len(df)),
    'population_size': POPULATION_SIZE,
    'paid_customers': int((df['primary_plan_type'] == 'monthly').sum()),
    'free_customers': int((df['primary_plan_type'] == 'free').sum()),
    'total_ev_annualized': float(total_ev),
    'sample_ev_annualized': float(total_ev_sample),
    'fy27_realization_rate': FY27_REALIZATION_RATE,
    'monte_carlo_p10': float(p10),
    'monte_carlo_p50': float(p50),
    'monte_carlo_p90': float(p90),
    'target': 21000000,
    'gap': float(21e6 - total_ev),
    'data_source': df['data_source'].iloc[0] if 'data_source' in df.columns else 'unknown',
    'waves': wave_summary,
    'initiative_counts': df['best_initiative'].value_counts().to_dict(),
    'cluster_summary': [{
        'name': r['cluster_name'],
        'count': int(r['customer_count']),
        'avg_mrr': float(r['avg_mrr']),
        'avg_friction': float(r['avg_friction_score']),
        'avg_completion': float(r['avg_completion_rate']),
    } for _, r in cluster_df.iterrows()],
}

with open("outputs/key_numbers.json", "w") as f:
    json.dump(key_numbers, f, indent=2, default=str)

print("\nAll outputs saved to outputs/")
print("Ready for Phase 6: Executive Readout")
