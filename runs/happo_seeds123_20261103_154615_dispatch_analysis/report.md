# Frozen-policy dispatch diagnostics

This report describes executed frozen-policy trajectories and same-day counterfactuals. It does not retrain policies or establish long-horizon optimality. Fixed training scenarios and the held-out representative-year sequence are reported separately.

Violation frequency counts active reservoir-days. Relative severity averages the deficit divided by its effective requirement across **all active days**. Conditional physical severity averages only violating days. Classification percentages use **violations** as their denominator; they are not percentages of all days.

A meeting same-state action establishes same-day reachability under the tested mapping and fixed other commands. A physical-limit classification applies to that starting state; it does not prove that earlier policy decisions could not have preserved more water or head. Proof flags can overlap and must not be summed as disjoint causes.

The alternative release/storage deltas do not price future water value. Positive release deltas generally draw additional water; negative end-storage deltas indicate lower ending storage. Other-reservoir violations are reported where measured.

Recorded discount factor: **0.99**. Returns use episode-start discounting; service counts, severity, and generation below are undiscounted.

Rows analyzed: **109,500**. Individual mixed-return accounting and repeated system/legacy accounting remain separate in `summary.json`. Global, system, and legacy CSV scalars repeat across stations and are counted once per day.

## Frequency and severity

| Scope | Term | Active days | Violations | Frequency | Relative severity | Conditional deficit |
|---|---|---:|---:|---:|---:|---:|
| Held-out representative-year sequence | ecological flow | 27,375 | 2,566 | 9.37% | 1.38% | 565.65 m³/s |
| Held-out representative-year sequence | guaranteed output | 27,375 | 13,580 | 49.61% | 21.05% | 1,739.21 MW |
| Fixed training scenarios | ecological flow | 82,125 | 9,184 | 11.18% | 1.84% | 576.99 m³/s |
| Fixed training scenarios | guaranteed output | 82,125 | 39,466 | 48.06% | 20.07% | 1,720.36 MW |

## Held-out representative-year sequence

| Reservoir | Term | Frequency | Relative severity | Meeting grid alternative / violations | Classification mix among violations |
|---|---|---:|---:|---:|---|
| BHT | ecological flow | 9.92% | 1.28% | 100.00% | same day policy alternative: 100.00% |
| BHT | guaranteed output | 51.84% | 24.63% | 61.17% | same day policy alternative: 61.17%; same state head limit: 25.26%; same state water limit: 13.57% |
| THR | ecological flow | 21.55% | 3.84% | 100.00% | same day policy alternative: 100.00% |
| THR | guaranteed output | 48.02% | 12.98% | 96.27% | same day policy alternative: 96.27%; same state head limit: 3.73% |
| WDD | ecological flow | 3.89% | 0.36% | 100.00% | same day policy alternative: 100.00% |
| WDD | guaranteed output | 44.29% | 15.19% | 80.66% | same day policy alternative: 80.66%; same state head limit: 12.91%; same state water limit: 6.43% |
| XJB | ecological flow | 5.77% | 0.70% | 100.00% | same day policy alternative: 100.00% |
| XJB | guaranteed output | 50.26% | 23.54% | 50.65% | same day policy alternative: 50.65%; same state water limit: 36.74%; same state head limit: 12.61% |
| XLD | ecological flow | 5.74% | 0.70% | 100.00% | same day policy alternative: 100.00% |
| XLD | guaranteed output | 53.63% | 28.92% | 55.69% | same day policy alternative: 55.69%; same state head limit: 29.19%; same state water limit: 15.12% |

### Measured state and alternative costs

| Reservoir | Term | Mean head (m) | Mean ending storage (billion m³) | Meeting alternative days | Mean release delta (m³/s) | Mean ending-storage delta (million m³) |
|---|---|---:|---:|---:|---:|---:|
| BHT | ecological flow | 175.74 | 9.09 | 543 | 166.37 | -14.37 |
| BHT | guaranteed output | 175.74 | 9.09 | 1,736 | 1,221.44 | -105.53 |
| THR | ecological flow | 67.46 | 18.11 | 1,180 | 1,156.44 | -99.92 |
| THR | guaranteed output | 67.46 | 18.11 | 2,531 | 2,344.20 | -202.54 |
| WDD | ecological flow | 132.11 | 3.18 | 213 | 95.73 | -8.27 |
| WDD | guaranteed output | 132.11 | 3.18 | 1,956 | 640.83 | -55.37 |
| XJB | ecological flow | 97.14 | 4.27 | 316 | 163.63 | -14.14 |
| XJB | guaranteed output | 97.14 | 4.27 | 1,394 | 1,367.39 | -118.14 |
| XLD | ecological flow | 163.68 | 5.88 | 314 | 159.46 | -13.78 |
| XLD | guaranteed output | 163.68 | 5.88 | 1,635 | 1,757.94 | -151.89 |

## Fixed training scenarios

| Reservoir | Term | Frequency | Relative severity | Meeting grid alternative / violations | Classification mix among violations |
|---|---|---:|---:|---:|---|
| BHT | ecological flow | 11.64% | 1.82% | 99.27% | same day policy alternative: 99.27%; same state water limit: 0.73% |
| BHT | guaranteed output | 51.26% | 23.16% | 62.65% | same day policy alternative: 62.65%; same state head limit: 23.06%; same state water limit: 14.29% |
| THR | ecological flow | 24.09% | 4.50% | 99.39% | same day policy alternative: 99.39%; same state water limit: 0.61% |
| THR | guaranteed output | 44.32% | 13.51% | 92.44% | same day policy alternative: 92.44%; same state head limit: 7.32%; same state water limit: 0.23% |
| WDD | ecological flow | 5.35% | 0.73% | 99.20% | same day policy alternative: 99.20%; same state water limit: 0.80% |
| WDD | guaranteed output | 40.93% | 14.22% | 79.72% | same day policy alternative: 79.72%; same state head limit: 10.83%; same state water limit: 9.45% |
| XJB | ecological flow | 7.30% | 1.05% | 99.58% | same day policy alternative: 99.58%; same state water limit: 0.42% |
| XJB | guaranteed output | 49.88% | 21.87% | 53.69% | same day policy alternative: 53.69%; same state water limit: 33.42%; same state head limit: 12.89% |
| XLD | ecological flow | 7.54% | 1.08% | 100.00% | same day policy alternative: 100.00% |
| XLD | guaranteed output | 53.89% | 27.58% | 59.90% | same day policy alternative: 59.90%; same state head limit: 24.48%; same state water limit: 15.62% |

### Measured state and alternative costs

| Reservoir | Term | Mean head (m) | Mean ending storage (billion m³) | Meeting alternative days | Mean release delta (m³/s) | Mean ending-storage delta (million m³) |
|---|---|---:|---:|---:|---:|---:|
| BHT | ecological flow | 175.77 | 9.09 | 1,898 | 194.71 | -16.82 |
| BHT | guaranteed output | 175.77 | 9.09 | 5,275 | 1,094.33 | -94.55 |
| THR | ecological flow | 67.56 | 18.18 | 3,932 | 1,195.98 | -103.33 |
| THR | guaranteed output | 67.56 | 18.18 | 6,729 | 2,558.85 | -221.08 |
| WDD | ecological flow | 132.23 | 3.19 | 871 | 133.39 | -11.52 |
| WDD | guaranteed output | 132.23 | 3.19 | 5,359 | 629.88 | -54.42 |
| XJB | ecological flow | 97.14 | 4.27 | 1,194 | 190.58 | -16.47 |
| XJB | guaranteed output | 97.14 | 4.27 | 4,399 | 1,211.19 | -104.65 |
| XLD | ecological flow | 164.15 | 5.92 | 1,239 | 183.59 | -15.86 |
| XLD | guaranteed output | 164.15 | 5.92 | 5,302 | 1,613.93 | -139.44 |

## Accounting and coverage

| Seed | Scenario | Scope | Days | Generation (TWh) | Discounted system return | Discounted legacy return |
|---|---|---|---:|---:|---:|---:|
| 1 | dry_2007 | Fixed training scenarios | 1825 | 1,184.34 | 31.67 | 4.99 |
| 1 | held_out_sequence | Held-out representative-year sequence | 1825 | 1,180.09 | 31.60 | 4.59 |
| 1 | median_1991 | Fixed training scenarios | 1825 | 1,210.06 | 19.36 | 0.40 |
| 1 | wet_1973 | Fixed training scenarios | 1825 | 1,371.64 | 58.68 | 14.59 |
| 2 | dry_2007 | Fixed training scenarios | 1825 | 1,190.61 | 42.73 | 8.75 |
| 2 | held_out_sequence | Held-out representative-year sequence | 1825 | 1,187.93 | 40.28 | 7.42 |
| 2 | median_1991 | Fixed training scenarios | 1825 | 1,217.60 | 28.74 | 3.55 |
| 2 | wet_1973 | Fixed training scenarios | 1825 | 1,380.97 | 68.35 | 17.83 |
| 3 | dry_2007 | Fixed training scenarios | 1825 | 1,180.69 | 26.50 | 2.75 |
| 3 | held_out_sequence | Held-out representative-year sequence | 1825 | 1,174.57 | 24.50 | 1.66 |
| 3 | median_1991 | Fixed training scenarios | 1825 | 1,205.96 | 15.54 | -1.38 |
| 3 | wet_1973 | Fixed training scenarios | 1825 | 1,365.10 | 53.16 | 12.21 |

## Artifacts and limits

- `summary.json`: full per-seed/scenario/station/term aggregates, calendar-month and episode-year breakdowns, classification-conditioned head/storage statistics, proof flags, measured alternative costs, and provenance.
- `service_frequency_and_severity.png`: frequency and all-active-day relative severity.
- `classification_composition.png`: observed classification fractions among violations.
- `seasonal_violations.png`: actual calendar-month violation frequencies.

All daily rows are correlated observations. Same-day counterfactuals do not establish causation, optimality, or the attainability of an alternative starting state. Missing measurements remain null rather than becoming zeros.

## Classification definitions

- **requirement inactive**: The effective service requirement is zero.
- **no violation**: The executed service meets its requirement within reward tolerance.
- **same day policy alternative**: Another available discrete command meets this service from the same saved state; future water value is unassessed.
- **discrete grid gap**: A continuous command meets the service but no configured discrete command does.
- **same state capacity limit**: The service exceeds its optimistic outlet or turbine/installed capacity bound.
- **same state water limit**: Releasable water above the current next-state storage floor cannot meet the service even at optimistic maximum head for power.
- **same state dispatch limit**: The current feasible release ceiling cannot meet the service even at optimistic maximum head for power.
- **same state head limit**: All continuous current-state power choices fail despite optimistic water, dispatch and hardware bounds.
