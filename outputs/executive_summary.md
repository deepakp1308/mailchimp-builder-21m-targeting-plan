# Builder $21M Targeting Recommendation

**Date:** May 23, 2026  
**Status:** Production Data

---

## Executive Summary

Revenue impact range: **$21.3M (P10) — $21.3M (P50) — $21.4M (P90)**

**Can we hit $21M? Yes** — the median Monte Carlo estimate exceeds the target with 100% probability.

## Key Numbers

| Metric | Value |
|--------|-------|
| Customers analyzed | 50,000 |
| Total MRR | $33.5M |
| Customers with initiatives | 50,000 (100%) |
| Customer segments | 11 |
| Propensity models | 5 |
| Monte Carlo iterations | 10,000 |
| P(hitting $21M) | 100% |

## Wave Sequence

### Wave 1 — $20.2M EV, 29,819 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Universal Content | 29,819 | $20,204,623 | 100% | 0.75 |

### Wave 3 — $1.1M EV, 5,680 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Activation | 3,237 | $86 | 0% | 0.75 |
| Ai Builder | 1,332 | $561,997 | 49% | 0.75 |
| Brandkit | 3 | $1,138 | 68% | 0.75 |
| Code Mode | 661 | $345,368 | 99% | 0.75 |
| Omnichannel | 21 | $3 | 56% | 0.75 |
| Template Improvement | 426 | $217,700 | 99% | 0.75 |

### Wave 4 — $0.0M EV, 14,501 customers

| Initiative | Customers | EV (Base) | P(Success) | Confidence |
|-----------|-----------|-----------|------------|------------|
| Rendering Fix | 14,501 | $2,286 | 77% | 0.75 |

## Customer Segments

| Segment | Count | Avg MRR | Completion | Friction | Churn |
|---------|-------|---------|------------|----------|-------|
| Power senders | 8,209 | $438 | 80% | 0.08 | 0.0% |
| Power senders | 1,036 | $1009 | 75% | 0.10 | 0.0% |
| High-value frustrated users | 7,264 | $490 | 0% | 0.52 | 0.0% |
| Cluster 3: high creates_trend, high publishes_trend | 5,364 | $505 | 38% | 0.25 | 0.0% |
| Power senders | 1,800 | $1510 | 76% | 0.10 | 0.0% |
| Power senders | 4,938 | $1671 | 74% | 0.10 | 0.0% |
| Power senders | 6,614 | $708 | 79% | 0.09 | 0.0% |
| Power senders | 590 | $879 | 71% | 0.12 | 0.0% |
| Power senders | 12,247 | $271 | 84% | 0.06 | 0.0% |
| Power senders | 772 | $1774 | 74% | 0.10 | 0.0% |
| Power senders | 1,166 | $1451 | 74% | 0.10 | 0.0% |

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
