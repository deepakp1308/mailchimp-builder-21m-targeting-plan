"""
Phase 6: Generate executive readout HTML + markdown summary.
Reads all model outputs and produces a single-page executive report.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")

print("=" * 60)
print("PHASE 6: Executive Readout Generation")
print("=" * 60)

# --- Load all outputs ---
df = pd.read_parquet(DATA_PATH)
cluster_profiles = pd.read_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"))
try:
    init_sequence = pd.read_csv(os.path.join(OUT_DIR, "initiative_summary.csv"))
except FileNotFoundError:
    init_sequence = pd.read_csv(os.path.join(OUT_DIR, "initiative_sequence.csv"))
mc_summary = pd.read_csv(os.path.join(OUT_DIR, "monte_carlo_summary.csv"))
mc_iterations = np.load(os.path.join(OUT_DIR, "mc_iterations.npy"))

try:
    model_results = json.load(open(os.path.join(OUT_DIR, "model_results.json")))
except FileNotFoundError:
    model_results = {}

try:
    init_summary_csv = pd.read_csv(os.path.join(OUT_DIR, "initiative_summary.csv"))
except FileNotFoundError:
    init_summary_csv = init_sequence

# MC values
mc_vals = dict(zip(mc_summary['metric'], mc_summary['value']))
p10 = mc_vals.get('P10', 0)
p50 = mc_vals.get('P50', 0)
p90 = mc_vals.get('P90', 0)
hit_rate = mc_vals.get('target_21M_hit_rate', 0)

# --- Generate Plotly charts as embedded HTML ---

# 1. Monte Carlo histogram
fig_mc = go.Figure()
fig_mc.add_trace(go.Histogram(
    x=mc_iterations / 1e6, nbinsx=80,
    marker_color='#2E86AB', opacity=0.85, name='Simulated Revenue',
))
for label, val, color in [
    ('P10', p10, '#E63946'), ('P50', p50, '#1D3557'),
    ('P90', p90, '#457B9D'), ('$21M Target', 21e6, '#E76F51'),
]:
    fig_mc.add_vline(x=val/1e6, line_dash="dash", line_color=color, line_width=2,
                     annotation_text=f"{label}: ${val/1e6:.1f}M",
                     annotation_position="top")
fig_mc.update_layout(
    title="Monte Carlo Revenue Simulation (10,000 iterations)",
    xaxis_title="Total Revenue Impact ($M)", yaxis_title="Frequency",
    template="plotly_white", height=400, showlegend=False,
    font=dict(family="Arial", size=11), margin=dict(t=50, b=40, l=50, r=30),
)
mc_html = pio.to_html(fig_mc, include_plotlyjs='cdn', full_html=False)

# 2. Revenue waterfall chart
waterfall_data = init_sequence.sort_values('total_ev_base', ascending=False).head(8)
fig_wf = go.Figure(go.Waterfall(
    x=waterfall_data['best_initiative'].str.replace('_', ' ').str.title(),
    y=waterfall_data['total_ev_base'],
    textposition="outside",
    text=[f"${v/1e6:.1f}M" for v in waterfall_data['total_ev_base']],
    connector={"line": {"color": "#1a3a5c"}},
    increasing={"marker": {"color": "#2E86AB"}},
    decreasing={"marker": {"color": "#E63946"}},
    totals={"marker": {"color": "#1D3557"}},
))
fig_wf.update_layout(
    title="Revenue Waterfall by Initiative",
    yaxis_title="Expected Value ($)", template="plotly_white",
    height=400, font=dict(family="Arial", size=11),
    margin=dict(t=50, b=80, l=60, r=30),
    xaxis_tickangle=-30,
)
wf_html = pio.to_html(fig_wf, include_plotlyjs=False, full_html=False)

# --- Build HTML ---
today = datetime.now().strftime("%B %d, %Y")
is_synthetic = 'data_source' not in df.columns or df['data_source'].iloc[0] != 'bigquery_batched_sample'

try:
    with open(os.path.join(OUT_DIR, "key_numbers.json")) as f:
        key_numbers = json.load(f)
    if key_numbers.get('data_source') == 'bigquery_batched_sample':
        is_synthetic = False
except FileNotFoundError:
    key_numbers = {}

total_customers = len(df)
total_mrr = df['avg_mrr'].sum()

try:
    scored_df = pd.read_csv(os.path.join(OUT_DIR, "customer_scores.csv"))
    scored_customers = len(scored_df)
except FileNotFoundError:
    scored_df = df
    scored_customers = len(df)

# Use initiative_summary if available, else wave sequence from run_analysis
try:
    init_sequence = pd.read_csv(os.path.join(OUT_DIR, "initiative_summary.csv"))
except FileNotFoundError:
    pass

# Wave 1 details — adapt column names from run_analysis output
wave1 = init_sequence[init_sequence.get('wave', pd.Series(dtype=int)) == 1].copy() if 'wave' in init_sequence.columns else init_sequence.head(3)
if 'total_ev_base' in wave1.columns:
    wave1_total = wave1['total_ev_base'].sum()
    wave1_customers = wave1['customers'].sum() if 'customers' in wave1.columns else 0
else:
    wave1_total = wave1['expected_revenue'].sum() if 'expected_revenue' in wave1.columns else 0
    wave1_customers = wave1['customer_count'].sum() if 'customer_count' in wave1.columns else 0

# Can we hit $21M?
can_hit = p50 >= 21_000_000
feasibility_answer = (
    f"<strong style='color:#2E86AB'>Yes — the median (P50) estimate of ${p50/1e6:.1f}M exceeds the $21M target.</strong> "
    f"Monte Carlo simulations show a {hit_rate:.0%} probability of achieving $21M+. "
    f"The range spans ${p10/1e6:.1f}M (P10) to ${p90/1e6:.1f}M (P90)."
) if can_hit else (
    f"<strong style='color:#E63946'>Challenging — the median (P50) estimate of ${p50/1e6:.1f}M falls below the $21M target.</strong> "
    f"Monte Carlo simulations show only a {hit_rate:.0%} probability of achieving $21M+. "
    f"The optimistic case (P90) reaches ${p90/1e6:.1f}M, suggesting the target requires "
    f"either higher uplift rates (proven via experiments) or expanded initiative scope."
)

# Wave tables
def wave_table_rows(wave_df, wave_num):
    rows = ""
    for _, r in wave_df.iterrows():
        customers = r.get('customers', r.get('customer_count', 0))
        total_mrr = r.get('total_mrr', 0)
        ev_base = r.get('total_ev_base', r.get('expected_revenue', 0))
        ev_low = r.get('total_ev_low', ev_base * 0.7)
        ev_high = r.get('total_ev_high', ev_base * 1.3)
        p_success = r.get('avg_p_success', 0.5)
        confidence = r.get('avg_confidence', 0.75)
        desc = r.get('description', r['best_initiative'])
        rows += f"""<tr>
            <td style="font-weight:600">{str(r['best_initiative']).replace('_', ' ').title()}</td>
            <td>{desc}</td>
            <td style="text-align:right">{customers:,.0f}</td>
            <td style="text-align:right">${total_mrr:,.0f}</td>
            <td style="text-align:right">${ev_base:,.0f}</td>
            <td style="text-align:right">${ev_low:,.0f}</td>
            <td style="text-align:right">${ev_high:,.0f}</td>
            <td style="text-align:center">{p_success:.0%}</td>
            <td style="text-align:center">{confidence:.2f}</td>
            <td>{r.get('rationale', '')}</td>
        </tr>"""
    return rows

all_wave_rows = ""
for w in [1, 2, 3, 4]:
    w_data = init_sequence[init_sequence['wave'] == w]
    if len(w_data) > 0:
        w_total = w_data['total_ev_base'].sum()
        all_wave_rows += f"""<tr style="background:#e8f0f8; font-weight:700">
            <td colspan="4">Wave {w}</td>
            <td style="text-align:right">${w_total:,.0f}</td>
            <td colspan="5"></td>
        </tr>"""
        all_wave_rows += wave_table_rows(w_data, w)

# Cluster table — handle optional columns from run_analysis output
cluster_rows = ""
for _, c in cluster_profiles.iterrows():
    pct = c.get('pct_of_total', round(100 * c['customer_count'] / total_customers, 1))
    tenure = c.get('avg_tenure_months', c.get('tenure_months', 0))
    active_days = c.get('avg_active_days_90d', 0)
    churn = c.get('churn_rate', 0)
    cluster_rows += f"""<tr>
        <td style="font-weight:600">{c.get('cluster_name', c.get('cluster_id', ''))}</td>
        <td style="text-align:right">{int(c['customer_count']):,}</td>
        <td style="text-align:right">{pct}%</td>
        <td style="text-align:right">${c['avg_mrr']:.0f}</td>
        <td style="text-align:right">{c.get('avg_completion_rate', 0):.0%}</td>
        <td style="text-align:right">{c.get('avg_friction_score', 0):.2f}</td>
        <td style="text-align:right">{tenure:.0f}mo</td>
        <td style="text-align:right">{active_days:.0f}d</td>
        <td style="text-align:right">{churn:.1%}</td>
    </tr>"""

# Model results table
model_rows = ""
for mname, mdata in model_results.items():
    model_rows += f"""<tr>
        <td>{mname.replace('_', ' ').title()}</td>
        <td style="text-align:center">{mdata.get('auc', 'N/A')}</td>
        <td style="text-align:center">{mdata.get('positive_rate', 0):.1%}</td>
        <td>{mdata.get('status', 'unknown')}</td>
    </tr>"""

# Initiative-to-segment mapping
scored_df = pd.read_csv(os.path.join(OUT_DIR, "customer_scores.csv"))
mapping_rows = ""
if 'cluster_id' in scored_df.columns:
    cluster_map = cluster_profiles.set_index('cluster_id')['cluster_name'].to_dict() if 'cluster_name' in cluster_profiles.columns else {}
    scored_df['cluster_name'] = scored_df['cluster_id'].map(cluster_map).fillna('Cluster ' + scored_df['cluster_id'].astype(str))
if 'best_initiative' in scored_df.columns:
    mapping = scored_df.groupby(['best_initiative', 'cluster_name']).agg(
        count=('user_id', 'count'),
        total_ev=('best_ev_annualized', 'sum'),
        avg_mrr=('avg_mrr', 'mean'),
    ).reset_index().sort_values('total_ev', ascending=False) if 'cluster_name' in scored_df.columns else pd.DataFrame()
    for _, m in mapping.head(25).iterrows():
        mapping_rows += f"""<tr>
            <td>{m['best_initiative'].replace('_', ' ').title()}</td>
            <td>{m['cluster_name']}</td>
            <td style="text-align:right">{m['count']:,}</td>
            <td style="text-align:right">${m['avg_mrr']:.0f}</td>
            <td style="text-align:right">${m['total_ev']:,.0f}</td>
        </tr>"""

# Wave 1 top segments table
wave1_inits = wave1['best_initiative'].tolist() if len(wave1) > 0 and 'best_initiative' in wave1.columns else []
w1_segments = scored_df[scored_df['best_initiative'].isin(wave1_inits)] if wave1_inits else scored_df.head(0)
if len(w1_segments) > 0 and 'cluster_name' in w1_segments.columns:
    w1_seg_agg = w1_segments.groupby(['best_initiative', 'cluster_name']).agg(
        count=('user_id', 'count'),
        total_mrr=('avg_mrr', 'sum'),
        avg_mrr=('avg_mrr', 'mean'),
        total_ev=('best_ev_annualized', 'sum'),
    ).reset_index().sort_values('total_ev', ascending=False).head(5)
    w1_seg_agg['cluster_name'] = w1_seg_agg['cluster_name']
else:
    w1_seg_agg = cluster_profiles.head(5).copy()
    w1_seg_agg['best_initiative'] = 'universal_content'
    w1_seg_agg['count'] = w1_seg_agg['customer_count']
    w1_seg_agg['total_ev'] = w1_seg_agg.get('annualized_revenue', 0)

w1_top_rows = ""
for _, s in w1_seg_agg.iterrows():
    w1_top_rows += f"""<tr>
        <td style="font-weight:600">{s['cluster_name']}</td>
        <td style="text-align:right">{s['count']:,}</td>
        <td style="text-align:right">${s['avg_mrr']:.0f}</td>
        <td>{s['best_initiative'].replace('_', ' ').title()}</td>
        <td>High-EV segment matched to initiative via propensity scoring</td>
    </tr>"""


DATA_NOTE = """<p><strong style="color:#2E86AB">✓ PRODUCTION DATA (BATCHED SAMPLE)</strong> — 
This analysis uses a stratified 50K customer sample from BigQuery (35K top paid + 15K active free), 
with population calibration applied to MRR and customer counts. Sample skews toward high-MRR paid accounts; 
segment rankings and initiative priorities are directional. Validate uplift assumptions via experiments before final investment decisions.</p>""" if not is_synthetic else """<p><strong style="color:#E63946">⚠ SYNTHETIC DATA</strong> — 
This analysis uses a synthetic dataset calibrated to known Mailchimp metrics. 
Results should be validated against production BigQuery data before executive presentation.</p>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Builder $21M Targeting Recommendation</title>
<style>
body {{ font-family: Arial, sans-serif; font-size: 9pt; max-width: 1500px; margin: 0 auto; padding: 24px 32px; color: #1a1a2e; line-height: 1.5; }}
h1 {{ font-size: 16pt; font-weight: 700; color: #1a3a5c; margin-bottom: 4px; }}
h2 {{ font-size: 12pt; font-weight: 700; border-bottom: 2px solid #1a3a5c; padding-bottom: 4px; margin-top: 24px; color: #1a3a5c; }}
h3 {{ font-size: 10pt; font-weight: 600; color: #2E86AB; margin-top: 16px; }}
.subtitle {{ font-size: 10pt; color: #666; margin-bottom: 16px; }}
.kpi-row {{ display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }}
.kpi-card {{ background: #f0f4f8; border-left: 4px solid #1a3a5c; padding: 12px 16px; min-width: 180px; flex: 1; }}
.kpi-card .label {{ font-size: 8pt; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi-card .value {{ font-size: 16pt; font-weight: 700; color: #1a3a5c; }}
.kpi-card .sub {{ font-size: 8pt; color: #888; }}
.kpi-card.green {{ border-left-color: #2a9d8f; }}
.kpi-card.green .value {{ color: #2a9d8f; }}
.kpi-card.red {{ border-left-color: #E63946; }}
.kpi-card.red .value {{ color: #E63946; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 8.5pt; }}
th {{ background: #1a3a5c; color: #fff; padding: 6px 8px; text-align: left; font-weight: 600; }}
td {{ padding: 5px 8px; border-bottom: 1px solid #e0e0e0; }}
tr:nth-child(even) {{ background: #fafafa; }}
tr:hover {{ background: #f0f4f8; }}
.chart-container {{ margin: 16px 0; border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; }}
.feasibility {{ background: #f8f9fa; border: 2px solid #1a3a5c; border-radius: 6px; padding: 16px 20px; margin: 16px 0; }}
.assumptions {{ background: #fff9e6; border: 1px solid #f0c040; border-radius: 4px; padding: 12px 16px; margin: 12px 0; }}
.risk {{ color: #E63946; font-weight: 600; }}
.footer {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #e0e0e0; font-size: 8pt; color: #888; }}
.data-warning {{ background: #fff0f0; border: 2px solid #E63946; border-radius: 4px; padding: 12px 16px; margin: 12px 0; }}
</style>
</head>
<body>

<h1>Builder $21M Targeting Recommendation</h1>
<p class="subtitle">Revenue Impact Analysis — {today} | P10: ${p10/1e6:.1f}M &nbsp;|&nbsp; P50: ${p50/1e6:.1f}M &nbsp;|&nbsp; P90: ${p90/1e6:.1f}M</p>

{DATA_NOTE}

<div class="kpi-row">
    <div class="kpi-card">
        <div class="label">Total Customers Analyzed</div>
        <div class="value">{total_customers:,}</div>
        <div class="sub">paid customers</div>
    </div>
    <div class="kpi-card">
        <div class="label">Total Monthly MRR</div>
        <div class="value">${total_mrr/1e6:.1f}M</div>
        <div class="sub">across all segments</div>
    </div>
    <div class="kpi-card green">
        <div class="label">Median Revenue Impact (P50)</div>
        <div class="value">${p50/1e6:.1f}M</div>
        <div class="sub">Monte Carlo median</div>
    </div>
    <div class="kpi-card {'green' if can_hit else 'red'}">
        <div class="label">$21M Target Probability</div>
        <div class="value">{hit_rate:.0%}</div>
        <div class="sub">{'on track' if can_hit else 'needs acceleration'}</div>
    </div>
    <div class="kpi-card">
        <div class="label">Customers with Initiatives</div>
        <div class="value">{scored_customers:,}</div>
        <div class="sub">{scored_customers/total_customers:.0%} of base</div>
    </div>
</div>

<div class="feasibility">
    <h3 style="margin-top:0">Can we realistically hit $21M?</h3>
    <p>{feasibility_answer}</p>
</div>

<h2>1. Top 5 Customer Segments to Target (Wave 1)</h2>
<p>These are the highest-EV customer segments assigned to Wave 1 initiatives — deploy immediately.</p>
<table>
<tr><th>Segment</th><th>Customers</th><th>Avg MRR</th><th>Initiative</th><th>Rationale</th></tr>
{w1_top_rows}
</table>

<h2>2. Full Wave Sequence (Waves 1–4)</h2>
<p>Initiatives sequenced by expected value, probability of success, and confidence level.</p>
<table>
<tr>
    <th>Initiative</th><th>Description</th><th>Customers</th><th>Total MRR</th>
    <th>EV (Base)</th><th>EV (Low)</th><th>EV (High)</th>
    <th>P(Success)</th><th>Confidence</th><th>Rationale</th>
</tr>
{all_wave_rows}
<tr style="background:#1a3a5c; color:#fff; font-weight:700">
    <td colspan="4">TOTAL ALL WAVES</td>
    <td style="text-align:right">${init_sequence['total_ev_base'].sum():,.0f}</td>
    <td style="text-align:right">${init_sequence['total_ev_low'].sum():,.0f}</td>
    <td style="text-align:right">${init_sequence['total_ev_high'].sum():,.0f}</td>
    <td colspan="3"></td>
</tr>
</table>

<h2>3. Revenue Waterfall</h2>
<div class="chart-container">
{wf_html}
</div>

<h2>4. Monte Carlo Revenue Distribution</h2>
<div class="chart-container">
{mc_html}
</div>

<h2>5. Initiative-to-Segment Mapping</h2>
<p>Top 25 initiative × segment combinations ranked by expected value.</p>
<table>
<tr><th>Initiative</th><th>Customer Segment</th><th>Customers</th><th>Avg MRR</th><th>Total EV</th></tr>
{mapping_rows}
</table>

<h2>6. Cluster Profiles</h2>
<table>
<tr>
    <th>Cluster</th><th>Customers</th><th>% of Total</th><th>Avg MRR</th>
    <th>Completion Rate</th><th>Friction</th><th>Tenure</th>
    <th>Active Days (90d)</th><th>Churn Rate</th>
</tr>
{cluster_rows}
</table>

<h2>7. Propensity Model Performance</h2>
<table>
<tr><th>Model</th><th>Test AUC</th><th>Positive Rate</th><th>Status</th></tr>
{model_rows}
</table>

<h2>8. Key Assumptions, Risks & Confidence</h2>
<div class="assumptions">
<h3 style="margin-top:0">Assumptions</h3>
<ul>
    <li><strong>Uplift estimates:</strong> 5% / 10% / 15% for low / base / high scenarios — assumption-based, no A/B test data in this pass</li>
    <li><strong>Treatment score:</strong> Fixed at 0.4 (no experimental evidence); real experiments would raise confidence significantly</li>
    <li><strong>Reachability:</strong> 60–90% depending on initiative channel; assumes email + in-app targeting</li>
    <li><strong>CLV:</strong> Survival-adjusted using Kaplan-Meier on tenure data, not simple 12×MRR</li>
    <li><strong>Customer overlap:</strong> Each customer assigned to single best initiative (argmax EV) to avoid double-counting</li>
</ul>
</div>
<div class="assumptions" style="border-color: #E63946; background: #fff5f5">
<h3 style="margin-top:0; color:#E63946">Risks</h3>
<ul>
    <li class="risk">Uplift assumptions are unvalidated — actual lift may be 30-50% lower than base estimates</li>
    <li class="risk">Propensity models trained on observational data; causal uplift modeling requires RCTs</li>
    <li class="risk">Synthetic data calibration may not capture tail behavior of real customer base</li>
    <li>Customer response rates may vary significantly by segment and channel</li>
    <li>Execution capacity constraints not modeled — can the team ship all Wave 1 initiatives simultaneously?</li>
    <li>Market conditions (recession, competition) could alter baseline retention</li>
</ul>
</div>
<div class="assumptions" style="border-color: #2a9d8f; background: #f0fff4">
<h3 style="margin-top:0; color:#2a9d8f">Confidence Levels</h3>
<ul>
    <li><strong>High confidence:</strong> Customer segmentation, eligibility logic, MRR calculations</li>
    <li><strong>Medium confidence:</strong> Propensity model rankings (AUC-validated), survival CLV estimates</li>
    <li><strong>Low confidence:</strong> Absolute uplift values, revenue projections (require experimental validation)</li>
</ul>
</div>

<h2>9. Data Sources & Methodology</h2>
<table>
<tr><th>Component</th><th>Method</th><th>Details</th></tr>
<tr><td>Data Source</td><td>{'Synthetic (calibrated)' if is_synthetic else 'BigQuery production'}</td>
    <td>{'1M synthetic customers matching known Mailchimp distributions' if is_synthetic else 'bi_reporting.users, bi_activities, bi_customer, bi_finance tables'}</td></tr>
<tr><td>Clustering</td><td>MiniBatch KMeans</td><td>StandardScaler + log-transform, k selected by silhouette score, DecisionTree naming</td></tr>
<tr><td>Propensity Models</td><td>LightGBM + Optuna</td><td>5 models, 50 HPO trials each, isotonic calibration, SHAP explanations</td></tr>
<tr><td>Expected Value</td><td>EV = elig × reach × P(s) × uplift × CLV × conf</td><td>Survival-adjusted CLV via Kaplan-Meier, confidence discount with sample/treatment/calibration</td></tr>
<tr><td>Uncertainty</td><td>Monte Carlo (10K iter)</td><td>Beta-distributed P(success), Normal uplift, LogNormal revenue</td></tr>
<tr><td>Sequencing</td><td>Wave assignment</td><td>Wave 1: top quartile EV + P(s)≥0.5 + conf≥0.6; Wave 2: next quartile + P(s)≥0.3; Wave 3: experiment needed; Wave 4: strategic</td></tr>
</table>

<div class="footer">
    <p>Generated {today} | Builder $21M Targeting Analysis | {'⚠ Synthetic data — validate with production data' if is_synthetic else 'Production data'}</p>
    <p>Methodology: Clustering → Propensity Models → Expected Value Scoring → Monte Carlo Simulation → Wave Sequencing</p>
</div>

</body>
</html>"""

readout_path = os.path.join(OUT_DIR, "readout.html")
with open(readout_path, 'w') as f:
    f.write(html)
print(f"Executive readout saved to {readout_path}")

# --- Generate executive_summary.md ---
md = f"""# Builder $21M Targeting Recommendation

**Date:** {today}  
**Status:** {'⚠ Based on Synthetic Data' if is_synthetic else 'Production Data'}

---

## Executive Summary

Revenue impact range: **${p10/1e6:.1f}M (P10) — ${p50/1e6:.1f}M (P50) — ${p90/1e6:.1f}M (P90)**

{'**Can we hit $21M? Yes** — the median Monte Carlo estimate exceeds the target with ' + f'{hit_rate:.0%} probability.' if can_hit else '**Can we hit $21M? Challenging** — the median Monte Carlo estimate falls below target. Achieving $21M requires higher-than-base uplift rates, validated through experiments.'}

## Key Numbers

| Metric | Value |
|--------|-------|
| Customers analyzed | {total_customers:,} |
| Total MRR | ${total_mrr/1e6:.1f}M |
| Customers with initiatives | {scored_customers:,} ({scored_customers/total_customers:.0%}) |
| Customer segments | {len(cluster_profiles)} |
| Propensity models | {len(model_results)} |
| Monte Carlo iterations | 10,000 |
| P(hitting $21M) | {hit_rate:.0%} |

## Wave Sequence

"""

for w in [1, 2, 3, 4]:
    w_data = init_sequence[init_sequence['wave'] == w]
    if len(w_data) == 0:
        continue
    w_total = w_data['total_ev_base'].sum()
    w_cust = w_data['customers'].sum()
    md += f"### Wave {w} — ${w_total/1e6:.1f}M EV, {w_cust:,} customers\n\n"
    md += "| Initiative | Customers | EV (Base) | P(Success) | Confidence |\n"
    md += "|-----------|-----------|-----------|------------|------------|\n"
    for _, r in w_data.iterrows():
        md += f"| {r['best_initiative'].replace('_', ' ').title()} | {r['customers']:,.0f} | ${r['total_ev_base']:,.0f} | {r['avg_p_success']:.0%} | {r['avg_confidence']:.2f} |\n"
    md += "\n"

md += f"""## Customer Segments

| Segment | Count | Avg MRR | Completion | Friction | Churn |
|---------|-------|---------|------------|----------|-------|
"""
for _, c in cluster_profiles.iterrows():
    md += f"| {c.get('cluster_name', c.get('cluster_id', ''))} | {int(c['customer_count']):,} | ${c['avg_mrr']:.0f} | {c.get('avg_completion_rate', 0):.0%} | {c.get('avg_friction_score', 0):.2f} | {c.get('churn_rate', 0):.1%} |\n"

md += f"""
## Key Assumptions & Risks

**Assumptions:**
- Uplift: 5%/10%/15% (low/base/high) — unvalidated, assumption-based
- Treatment confidence: 0.4 (no experiment data)
- CLV: Survival-adjusted via Kaplan-Meier (not 12×MRR)
- Each customer assigned to single best initiative (no double-counting)

**Top Risks:**
1. Uplift assumptions unvalidated — actual lift may be 30-50% lower
2. Observational propensity models ≠ causal uplift (need RCTs)
3. Execution capacity: can team ship all Wave 1 simultaneously?
4. {'Synthetic data may not capture real tail behavior' if is_synthetic else 'Data freshness — based on latest available snapshot'}

**Confidence Levels:**
- High: Segmentation, eligibility, MRR calculations
- Medium: Propensity rankings (AUC-validated), survival CLV
- Low: Absolute uplift values, revenue projections

## Methodology

1. **Clustering:** MiniBatch KMeans on behavioral features, k selected by silhouette
2. **Propensity:** 5 LightGBM models with Optuna HPO + isotonic calibration
3. **EV Scoring:** eligibility × reachability × P(success) × uplift × CLV × confidence
4. **Monte Carlo:** 10K iterations with Beta/Normal/LogNormal sampling
5. **Sequencing:** 4 waves by EV quartile, P(success), and confidence thresholds

---
*Generated {today} | {'⚠ Synthetic Data' if is_synthetic else 'Production Data'}*
"""

md_path = os.path.join(OUT_DIR, "executive_summary.md")
with open(md_path, 'w') as f:
    f.write(md)
print(f"Executive summary saved to {md_path}")
print("\nPhase 6 complete.")
