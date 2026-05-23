# Builder $21M Targeting Recommendation

**Date:** May 23, 2026  
**Status:** Production Data

---

## Executive Summary

Revenue impact range: **$20.1M (P10) — $20.1M (P50) — $20.1M (P90)**

**Can we hit $21M? Challenging** — the median Monte Carlo estimate falls below target. Achieving $21M requires higher-than-base uplift rates, validated through experiments.

## Key Numbers

| Metric | Value |
|--------|-------|
| Customers analyzed | 1,651,592 |
| Total MRR | $108.1M |
| Customers with initiatives | 1,651,592 (100%) |
| Customer segments | 13 |
| Propensity models | 5 |
| Monte Carlo iterations | 10,000 |
| P(hitting $21M) | 0% |

## Wave Sequence

### Wave 1 — $18.1M EV, 549,557 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Code Mode | 10,399 | $342,700 | 99% | 0.75 |
| Universal Content | 539,158 | $17,723,331 | 100% | 0.75 |

### Wave 2 — $2.0M EV, 181,480 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Ai Builder | 139,062 | $1,496,803 | 74% | 0.75 |
| Brandkit | 628 | $4,687 | 82% | 0.75 |
| Omnichannel | 3,339 | $9 | 42% | 0.75 |
| Template Improvement | 38,451 | $507,438 | 100% | 0.75 |

### Wave 3 — $0.0M EV, 263,090 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Activation | 263,090 | $181 | 0% | 0.75 |

### Wave 4 — $0.0M EV, 657,465 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Rendering Fix | 657,465 | $4,470 | 49% | 0.75 |

## Customer Segments

| Segment | Count | Avg MRR | Completion | Friction | Churn |
|---------|-------|---------|------------|----------|-------|
| Dormant paid users at retention risk | 149,322 | $61 | 0% | 0.41 | 0.0% |
| Active mid-market users | 277,782 | $33 | 93% | 0.03 | 0.0% |
| Active mid-market users | 37,079 | $676 | 64% | 0.15 | 0.0% |
| Power senders | 104,424 | $88 | 76% | 0.10 | 0.0% |
| Active mid-market users | 195,805 | $39 | 43% | 0.23 | 0.0% |
| Power senders | 33,836 | $433 | 80% | 0.08 | 0.0% |
| Active mid-market users | 254,185 | $45 | 83% | 0.07 | 0.0% |
| Dormant paid users at retention risk | 345,413 | $20 | 0% | 0.40 | 0.0% |
| Cluster 8: low creates_trend, high email_tests_90d | 23,196 | $73 | 33% | 0.36 | 0.0% |
| Cluster 9: high creates_trend, high friction_score | 132,430 | $13 | 0% | 0.75 | 0.0% |
| Power senders | 4,573 | $561 | 72% | 0.11 | 0.0% |
| Power senders | 73,135 | $94 | 75% | 0.10 | 0.0% |
| Power senders | 20,412 | $107 | 72% | 0.12 | 0.0% |

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
4. Data freshness — based on latest available snapshot

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
*Generated May 23, 2026 | Production Data*
