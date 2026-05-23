"""
Phase 3a: Customer clustering via MiniBatch KMeans + DecisionTree naming.
Segments ~1M customers into behavioral clusters for targeting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import silhouette_score
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 3a: Customer Clustering")
print("=" * 60)

# --- Load data ---
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found. Run synthetic_data.py first.")
    sys.exit(1)

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} customers, {len(df.columns)} columns")

# --- Feature selection ---
CLUSTER_FEATURES = [
    'email_creates_90d', 'email_publishes_90d', 'email_tests_90d',
    'email_completion_rate', 'email_abandonment_rate', 'friction_score',
    'builder_active_days_90d', 'avg_mrr', 'tenure_months', 'list_size',
]

X_raw = df[CLUSTER_FEATURES].copy()
X_raw = X_raw.fillna(0)

# Log-transform skewed features
for col in ['avg_mrr', 'list_size', 'email_creates_90d', 'email_publishes_90d',
            'email_tests_90d', 'builder_active_days_90d']:
    X_raw[col] = np.log1p(X_raw[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"Features: {CLUSTER_FEATURES}")

# --- Test k values ---
print("\nTesting k values...")
k_candidates = [6, 8, 10, 12]
results = []
models = {}

for k in k_candidates:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=5)
    labels = km.fit_predict(X_scaled)
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(50000, len(X_scaled)), replace=False)
    sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
    inertia = km.inertia_
    results.append({'k': k, 'silhouette': sil, 'inertia': inertia})
    models[k] = (km, labels)
    print(f"  k={k:2d}: silhouette={sil:.4f}, inertia={inertia:,.0f}")

results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
best_k = int(best_k)
print(f"\nBest k={best_k} (highest silhouette)")

best_km, best_labels = models[best_k]
df['cluster_id'] = best_labels

# --- DecisionTree for naming rules ---
print(f"\nFitting DecisionTree (max_depth=3) for cluster naming...")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_raw, best_labels)
tree_text = export_text(dt, feature_names=CLUSTER_FEATURES)
print(tree_text[:2000])

# --- Name clusters from profiles ---
print("\nGenerating cluster profiles...")
profiles = []
for cid in range(best_k):
    mask = df['cluster_id'] == cid
    seg = df[mask]
    n = len(seg)

    avg_mrr_val = seg['avg_mrr'].mean()
    avg_creates = seg['email_creates_90d'].mean()
    avg_publishes = seg['email_publishes_90d'].mean()
    avg_completion = seg['email_completion_rate'].mean()
    avg_friction = seg['friction_score'].mean()
    avg_tenure = seg['tenure_months'].mean()
    avg_list = seg['list_size'].mean()
    avg_active_days = seg['builder_active_days_90d'].mean()
    churn_rate = seg['churned'].mean() if 'churned' in seg.columns else 0

    # Auto-name based on dominant behavioral signals
    if avg_creates < 1:
        name = "Dormant"
    elif avg_completion > 0.8 and avg_mrr_val > 100:
        name = "Power Senders"
    elif avg_completion > 0.7 and avg_mrr_val <= 100:
        name = "Steady Senders"
    elif avg_friction > 0.5 and avg_creates > 3:
        name = "High-Friction Builders"
    elif avg_creates > 5 and avg_completion < 0.4:
        name = "Prolific Abandoners"
    elif avg_active_days > 15 and avg_completion > 0.5:
        name = "Engaged Regulars"
    elif avg_tenure < 6 and avg_creates > 0:
        name = "New Explorers"
    elif avg_mrr_val > 150 and avg_completion < 0.5:
        name = "High-Value At-Risk"
    elif avg_list > 5000 and avg_completion > 0.5:
        name = "Enterprise Senders"
    elif avg_active_days < 5 and avg_creates > 0:
        name = "Light Dabblers"
    else:
        name = f"Segment {cid}"

    # Deduplicate names
    existing_names = [p['cluster_name'] for p in profiles]
    if name in existing_names:
        name = f"{name} ({cid})"

    profiles.append({
        'cluster_id': cid,
        'cluster_name': name,
        'customer_count': n,
        'pct_of_total': round(100 * n / len(df), 1),
        'avg_mrr': round(avg_mrr_val, 2),
        'total_mrr': round(seg['avg_mrr'].sum(), 0),
        'avg_creates_90d': round(avg_creates, 1),
        'avg_publishes_90d': round(avg_publishes, 1),
        'avg_completion_rate': round(avg_completion, 3),
        'avg_friction_score': round(avg_friction, 3),
        'avg_tenure_months': round(avg_tenure, 1),
        'avg_list_size': round(avg_list, 0),
        'avg_active_days_90d': round(avg_active_days, 1),
        'churn_rate': round(churn_rate, 4),
        'pct_essential': round((seg['package'] == 'essential').mean() * 100, 1),
        'pct_standard': round((seg['package'] == 'standard').mean() * 100, 1),
        'pct_premium': round((seg['package'] == 'premium').mean() * 100, 1),
    })

profiles_df = pd.DataFrame(profiles)
profiles_df.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"), index=False)

# Save cluster assignments back into parquet
name_map = dict(zip(profiles_df['cluster_id'], profiles_df['cluster_name']))
df['cluster_name'] = df['cluster_id'].map(name_map)
df.to_parquet(DATA_PATH, index=False)

print(f"\nCluster profiles saved to outputs/cluster_profiles.csv")
print(f"Updated parquet with cluster assignments")
print(f"\n{'Cluster Profiles':^60}")
print("-" * 60)
for _, row in profiles_df.iterrows():
    print(f"  [{row['cluster_id']}] {row['cluster_name']}: "
          f"{row['customer_count']:,} customers ({row['pct_of_total']}%), "
          f"MRR=${row['avg_mrr']:.0f}, completion={row['avg_completion_rate']:.0%}")

print(f"\nTotal MRR: ${profiles_df['total_mrr'].sum():,.0f}")
print(f"Decision tree accuracy: {dt.score(X_raw, best_labels):.1%}")
