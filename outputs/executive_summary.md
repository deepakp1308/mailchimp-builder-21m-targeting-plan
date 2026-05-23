# Builder $21M Targeting Recommendation

**Date:** May 23, 2026  
**Status:** ⚠ Based on Synthetic Data

---

## Executive Summary

Revenue impact range: **$31.7M (P10) — $69.4M (P50) — $175.4M (P90)**

**Can we hit $21M? Yes** — the median Monte Carlo estimate exceeds the target with 97% probability.

## Key Numbers

| Metric | Value |
|--------|-------|
| Customers analyzed | 1,000,000 |
| Total MRR | $59.7M |
| Customers with initiatives | 983,468 (98%) |
| Customer segments | 12 |
| Propensity models | 5 |
| Monte Carlo iterations | 10,000 |
| P(hitting $21M) | 97% |

## Wave Sequence

### Wave 1 — $53.7M EV, 284,946 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Ai Builder | 241,678 | $45,819,359 | 99% | 0.71 |
| Brandkit | 43,268 | $7,858,256 | 99% | 0.64 |

### Wave 3 — $0.0M EV, 6 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Churn Prevention | 6 | $771 | 75% | 0.44 |

### Wave 4 — $7.1M EV, 698,516 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Universal Content | 192,790 | $5,308,471 | 12% | 0.73 |
| Omnichannel | 43,664 | $972,595 | 12% | 0.74 |
| Activation | 396,305 | $625,144 | 1% | 0.73 |
| Template Improvement | 52,615 | $117,474 | 1% | 0.73 |
| Rendering Fix | 13,142 | $31,272 | 1% | 0.73 |

## Customer Segments

| Segment | Count | Avg MRR | Completion | Friction | Churn |
|---------|-------|---------|------------|----------|-------|
| Segment 0 | 107,616 | $28 | 58% | 0.17 | 4.2% |
| Dormant | 119,001 | $25 | 0% | 0.40 | 4.2% |
| Dormant (2) | 110,205 | $127 | 0% | 0.40 | 4.3% |
| Segment 3 | 96,369 | $41 | 70% | 0.12 | 4.2% |
| Segment 4 | 109,623 | $42 | 34% | 0.26 | 4.2% |
| Segment 5 | 90,573 | $158 | 57% | 0.17 | 4.2% |
| Dormant (6) | 118,562 | $35 | 0% | 0.40 | 4.2% |
| Segment 7 | 28,344 | $59 | 0% | 0.94 | 4.3% |
| Dormant (8) | 49,199 | $53 | 0% | 0.40 | 4.3% |
| Segment 9 | 42,519 | $31 | 66% | 0.14 | 4.2% |
| Segment 10 | 95,078 | $49 | 58% | 0.17 | 4.2% |
| Segment 11 | 32,911 | $75 | 46% | 0.21 | 4.2% |

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
4. Synthetic data may not capture real tail behavior

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
*Generated May 23, 2026 | ⚠ Synthetic Data*
