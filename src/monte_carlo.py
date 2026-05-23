"""
Phase 5a: Monte Carlo simulation for revenue uncertainty quantification.
10,000 iterations sampling from uncertainty distributions.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 5a: Monte Carlo Simulation")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} customers")

# Filter to customers with initiative assignments
scored = df[df['best_initiative'].notna()].copy()
print(f"Customers with initiatives: {len(scored):,}")

# Group by initiative for simulation
init_groups = scored.groupby('best_initiative').agg(
    n_customers=('user_id', 'count'),
    mean_p_success=('p_success', 'mean'),
    std_p_success=('p_success', 'std'),
    mean_ev=('ev_base', 'mean'),
    total_ev_base=('ev_base', 'sum'),
    total_ev_low=('ev_low', 'sum'),
    total_ev_high=('ev_high', 'sum'),
    mean_clv=('survival_clv', 'mean'),
    std_clv=('survival_clv', 'std'),
    mean_mrr=('avg_mrr', 'mean'),
    mean_confidence=('confidence', 'mean'),
).reset_index()

print(f"\nInitiative groups: {len(init_groups)}")
for _, row in init_groups.iterrows():
    print(f"  {row['best_initiative']}: {row['n_customers']:,} customers, "
          f"base EV=${row['total_ev_base']:,.0f}")

# --- Monte Carlo ---
N_ITERATIONS = 10_000
np.random.seed(42)

print(f"\nRunning {N_ITERATIONS:,} Monte Carlo iterations...")
iteration_totals = np.zeros(N_ITERATIONS)

for _, init_row in init_groups.iterrows():
    name = init_row['best_initiative']
    n = int(init_row['n_customers'])
    mean_p = init_row['mean_p_success']
    std_p = max(init_row['std_p_success'], 0.05)

    # Beta distribution for P(success)
    alpha_p = max(mean_p * ((mean_p * (1 - mean_p) / (std_p ** 2)) - 1), 1.01)
    beta_p = max((1 - mean_p) * ((mean_p * (1 - mean_p) / (std_p ** 2)) - 1), 1.01)

    mean_clv = init_row['mean_clv']
    std_clv = max(init_row['std_clv'], mean_clv * 0.3)

    # LogNormal params for revenue
    sigma_rev = np.sqrt(np.log(1 + (std_clv / mean_clv) ** 2))
    mu_rev = np.log(mean_clv) - sigma_rev ** 2 / 2

    for i in range(N_ITERATIONS):
        sampled_p = np.random.beta(alpha_p, beta_p)
        sampled_uplift = np.random.normal(0.10, 0.03)  # base=10%, sd=3%
        sampled_uplift = np.clip(sampled_uplift, 0.02, 0.25)
        sampled_rev = np.random.lognormal(mu_rev, sigma_rev)
        confidence = init_row['mean_confidence']

        ev_iter = n * sampled_p * sampled_uplift * sampled_rev * confidence
        iteration_totals[i] += ev_iter

# --- Results ---
p10 = np.percentile(iteration_totals, 10)
p25 = np.percentile(iteration_totals, 25)
p50 = np.percentile(iteration_totals, 50)
p75 = np.percentile(iteration_totals, 75)
p90 = np.percentile(iteration_totals, 90)
mean_total = iteration_totals.mean()

print(f"\n{'Monte Carlo Results':^50}")
print("-" * 50)
print(f"  P10 (pessimistic):  ${p10:>14,.0f}")
print(f"  P25:                ${p25:>14,.0f}")
print(f"  P50 (median):       ${p50:>14,.0f}")
print(f"  Mean:               ${mean_total:>14,.0f}")
print(f"  P75:                ${p75:>14,.0f}")
print(f"  P90 (optimistic):   ${p90:>14,.0f}")
print(f"\n  $21M target hit rate: {(iteration_totals >= 21_000_000).mean():.1%}")

# --- Plotly histogram ---
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=iteration_totals / 1e6,
    nbinsx=80,
    marker_color='#2E86AB',
    opacity=0.85,
    name='Simulated Revenue',
))

for label, val, color in [
    ('P10', p10, '#E63946'),
    ('P50', p50, '#1D3557'),
    ('P90', p90, '#457B9D'),
    ('$21M Target', 21e6, '#E76F51'),
]:
    fig.add_vline(x=val / 1e6, line_dash="dash", line_color=color, line_width=2,
                  annotation_text=f"{label}: ${val/1e6:.1f}M",
                  annotation_position="top")

fig.update_layout(
    title=dict(text="Monte Carlo Revenue Simulation (10,000 iterations)", font=dict(size=16)),
    xaxis_title="Total Revenue Impact ($M)",
    yaxis_title="Frequency",
    template="plotly_white",
    width=900,
    height=500,
    showlegend=False,
    font=dict(family="Arial", size=12),
)

chart_path = os.path.join(CHARTS_DIR, "monte_carlo_histogram.html")
fig.write_html(chart_path, include_plotlyjs='cdn')
print(f"\nChart saved to {chart_path}")

# Save summary
summary = pd.DataFrame({
    'metric': ['P10', 'P25', 'P50', 'Mean', 'P75', 'P90', 'target_21M_hit_rate'],
    'value': [p10, p25, p50, mean_total, p75, p90, (iteration_totals >= 21_000_000).mean()],
})
summary.to_csv(os.path.join(OUT_DIR, "monte_carlo_summary.csv"), index=False)

# Save raw iterations for readout embedding
np.save(os.path.join(OUT_DIR, "mc_iterations.npy"), iteration_totals)
print("Monte Carlo complete.")
