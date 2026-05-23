"""
Phase 5b: Initiative sequencing into Waves 1-4.
Wave 1: top quartile EV, P(success)>=0.5, confidence>=0.6
Wave 2: next quartile, P(success)>=0.3
Wave 3: experiment needed
Wave 4: strategic/future
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers_engineered.parquet")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 5b: Initiative Sequencing")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)
scored = df[df['best_initiative'].notna()].copy()
print(f"Scored customers: {len(scored):,}")

# Compute initiative-level aggregates
init_agg = scored.groupby('best_initiative').agg(
    customers=('user_id', 'count'),
    total_ev_base=('ev_base', 'sum'),
    total_ev_low=('ev_low', 'sum'),
    total_ev_high=('ev_high', 'sum'),
    avg_ev=('ev_base', 'mean'),
    avg_p_success=('p_success', 'mean'),
    avg_confidence=('confidence', 'mean'),
    avg_mrr=('avg_mrr', 'mean'),
    total_mrr=('avg_mrr', 'sum'),
).reset_index()

init_agg = init_agg.sort_values('total_ev_base', ascending=False).reset_index(drop=True)
ev_75 = init_agg['total_ev_base'].quantile(0.75)
ev_50 = init_agg['total_ev_base'].quantile(0.50)

# --- Wave assignment ---
def assign_wave(row):
    if (row['total_ev_base'] >= ev_75 and
        row['avg_p_success'] >= 0.5 and
        row['avg_confidence'] >= 0.6):
        return 1
    elif (row['total_ev_base'] >= ev_50 and
          row['avg_p_success'] >= 0.3):
        return 2
    elif row['avg_confidence'] < 0.5:
        return 3
    else:
        return 4

init_agg['wave'] = init_agg.apply(assign_wave, axis=1)

# Force at least one init per wave if there's enough data
if (init_agg['wave'] == 1).sum() == 0 and len(init_agg) >= 2:
    init_agg.loc[init_agg.index[0], 'wave'] = 1
    init_agg.loc[init_agg.index[1], 'wave'] = 1

# Rationale
RATIONALE = {
    'rendering_fix': 'Reduces builder friction → directly increases email completion rate and reduces abandonment',
    'template_improvement': 'Improves template quality → higher completion rate → more sends → retention',
    'brandkit': 'Brand consistency reduces design friction → faster email creation cycles',
    'universal_content': 'Content reuse drives volume → upgrade path to Standard/Premium',
    'ai_builder': 'AI adoption accelerates creation → expands builder value proposition',
    'omnichannel': 'SMS cross-sell expands ARPU → new revenue stream per customer',
    'activation': 'Converts dormant users → new builder pipeline → downstream revenue',
    'churn_prevention': 'Retains at-risk revenue → highest CLV protection per dollar spent',
}

init_agg['rationale'] = init_agg['best_initiative'].map(RATIONALE).fillna('Strategic initiative for builder growth')
init_agg['description'] = init_agg['best_initiative'].map({
    k: v['desc'] for k, v in {
        'rendering_fix': {'desc': 'Fix rendering/preview friction'},
        'brandkit': {'desc': 'Brand Kit adoption campaign'},
        'universal_content': {'desc': 'Universal Content Blocks expansion'},
        'ai_builder': {'desc': 'AI Builder adoption push'},
        'template_improvement': {'desc': 'Template completion rate improvement'},
        'omnichannel': {'desc': 'SMS + Omnichannel cross-sell'},
        'activation': {'desc': 'Builder activation for dormant users'},
        'churn_prevention': {'desc': 'Proactive churn prevention'},
    }.items()
}).fillna(init_agg['best_initiative'])

init_agg = init_agg.sort_values(['wave', 'total_ev_base'], ascending=[True, False])

# Save
out_path = os.path.join(OUT_DIR, "initiative_sequence.csv")
init_agg.to_csv(out_path, index=False)

print(f"\n{'Initiative Sequence':^70}")
print("=" * 70)
for wave in [1, 2, 3, 4]:
    wave_data = init_agg[init_agg['wave'] == wave]
    if len(wave_data) == 0:
        continue
    wave_total = wave_data['total_ev_base'].sum()
    wave_customers = wave_data['customers'].sum()
    print(f"\n  Wave {wave} — {len(wave_data)} initiatives, "
          f"{wave_customers:,} customers, ${wave_total:,.0f} EV")
    print(f"  {'─' * 66}")
    for _, row in wave_data.iterrows():
        print(f"    {row['best_initiative']:<25} "
              f"Cust={row['customers']:>7,}  "
              f"EV=${row['total_ev_base']:>12,.0f}  "
              f"P(s)={row['avg_p_success']:.2f}  "
              f"Conf={row['avg_confidence']:.2f}")

total_ev = init_agg['total_ev_base'].sum()
print(f"\n{'─' * 70}")
print(f"  Total EV (all waves): ${total_ev:,.0f}")
print(f"  vs $21M target: {'ACHIEVABLE' if total_ev >= 21_000_000 else 'GAP of $' + f'{21_000_000 - total_ev:,.0f}'}")
print(f"\nSaved to {out_path}")
