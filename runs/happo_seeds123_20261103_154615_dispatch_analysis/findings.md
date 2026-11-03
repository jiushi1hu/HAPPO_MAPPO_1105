# Findings from the original saved policies

The three final checkpoints were replayed on all three fixed training scenarios
and the held-out representative-year sequence: 12 complete episodes, 21,900
environment-days and 109,500 reservoir-day records. Replayed generation, reward
totals and station service counts reproduce the original evaluations exactly.
Selected alternative commands are independently verified through full-cascade
physics; no baseline command, learned weight or reward setting is changed.

This is the original **1.0 / 1.0 service-threshold experiment**, with gamma 0.99
and 100 action bins. The current working YAML now uses 0.4 / 0.4. The analysis
preserves it and uses an isolated copy of the exact 55 source/configuration files
fingerprinted in the original checkpoints. See `manifest.json` for provenance.

## Main result

Most actual failures have a feasible **individual same-day alternative** under
the current model. The following percentages are conditional on actual service
violations, not percentages of all operating days.

| Classification | Fixed training scenarios | Held-out sequence |
|---|---:|---:|
| Guaranteed output: a discrete meeting alternative exists | 68.58% (27,064 / 39,466) | 68.13% (9,252 / 13,580) |
| Guaranteed output: current-state head limitation | 16.28% | 17.17% |
| Guaranteed output: current-state water limitation | 15.14% | 14.70% |
| Ecology: a discrete meeting alternative exists | 99.46% (9,134 / 9,184) | 100% (2,566 / 2,566) |
| Ecology: current-state water limitation | 0.54% | 0% |

No discrete-grid-gap, installed/turbine-capacity, or dispatch-only classification
occurred. This supports studying policy tradeoffs and seasonal coordination
before treating finer action bins as the primary remedy. The head model remains
an assumed storage-sensitivity scenario, so physical conclusions are conditional
on that model.

## Water costs and scope

On fixed-scenario violations with a meeting discrete command, the least-release
meeting alternative needs an average additional release of:

- Guaranteed output: **1,487 m³/s**, reducing that reservoir's end-of-day storage
  by **128.5 million m³** relative to the original same-day action.
- Ecological flow: **618 m³/s**, reducing own end storage by **53.4 million m³**.

None of those individually tested minimum-meeting commands increases another
reservoir's same-day guaranteed-output or ecological **violation count**. This
does not establish unchanged shortfall severity or future performance.

Each alternative starts from the original state and preserves the other actors'
original commands. The alternatives cannot be combined across dates or stations
as an executable strategy, and their storage costs must not be summed into a
claim about a feasible cumulative intervention. Current-state head/water limits
may themselves reflect previous policy decisions; they do not establish an
unavoidable long-horizon violation rate.

## Where to focus

- **THR ecological flow:** 3,956 fixed-scenario violations, approximately 43% of
  all ecological failures; 3,932 (99.39%) have a same-day meeting command. The
  mean extra release for these witnesses is 1,196 m³/s, with 103.3 million m³
  less own end storage. All 1,180 held-out THR ecological failures are individually
  reachable under this same-state test.
- **XLD guaranteed output:** 8,852 fixed-scenario violations; 59.90% have a
  discrete meeting command, 24.48% are current-state head limited, and 15.62%
  current-state water limited. XJB has the strongest water-limited share of its
  guaranteed-output failures (33.42%).
- **Seasonality:** fixed-scenario guaranteed-output frequency approaches 99–100%
  in February–March and is near zero in August–October. Ecological frequency
  peaks in March (44.59% fixed, 39.01% held-out). This points to seasonal service
  and water-allocation questions, rather than generic optimizer divergence.

## Recommended next controlled experiment

Use paired continuation replays for representative THR ecological and XLD power
failures: change only one command to the minimum-meeting action, then allow the
original frozen actors to react for the rest of the saved horizon. Compare total
and subsequent deficits, generation, ending storage, mixed returns, system return
and legacy return with the unchanged baseline. Include failures early and late
in the episode and different source years.

That continuation experiment can test whether retaining water was worthwhile
under the current objective. The present same-day analysis alone cannot decide
whether a reward change improves the full-horizon operation problem.

## Files

- `report.md`: detailed frequency, severity, classification, state and cost tables.
- `summary.json`: all seed/scenario/station/service/month/episode-year aggregates.
- `manifest.json`: exact checkpoint/data/source fingerprints and replay checks.
- `seedN_SCENARIO_dispatch.csv`: full physical and frozen learning evidence.
- `classification_composition.png`, `service_frequency_and_severity.png`, and
  `seasonal_violations.png`: the three exported diagnostic figures.
