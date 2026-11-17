# Research artifact inventory

The repository contains the analysis and calibration materials described below. The numerical analysis begins from the released derived frames; the original third-party Git histories and source blobs are outside this release.

| Component | Public files | Scope |
| :- | :- | :- |
| Five-state Git parser | `src/history_measurement.py`, `src/construct_rules.py`, `src/maven_models.py`, `src/version_rules.py` | Snapshot parsing and state accounting |
| Original parser tests | `tests/test_measurement.py`, `tests/test_construct.py`, `tests/test_models_and_versions.py` | 56 parser/version/construct tests; the full repository suite also contains estimator and replay checks |
| Monte Carlo calibration | `scripts/simulate_calibration.py`, `src/synthetic_cohorts.py`, `src/estimators.py`, `configs/calibration.json` | Eight scenarios, 1000 replicates each, with the recorded seeds and parameters |
| Calibration reference results | `results/calibration/` | All 8000 retained replicates, event-time/cause diagnostics, scenario summaries and aggregate summary |
| Econometric and competing-risk analysis | `src/signal_models.py`, `src/measurement_analysis.py`, `src/activity_analysis.py`, `src/estimators.py` | Analysis from supplied panels and covariates |
| Activity-stratified outputs | `results/reference/activity_analysis.json`, `results/reference/measurement_summary.json` | Observed-data numerical summaries, including post-hoc activity strata |
| Precision scenarios | `src/precision_simulation.py`, `results/precision/scenarios.csv` | Precision-analysis implementation and recorded scenario results |
| Measurement comparison | `data/measurement_variants.json`, `results/measurement/` | Paired measurement definitions and recorded contrasts |
| Environment | `requirements.txt`, `requirements.lock`, `configs/environment.json` | Pinned direct dependencies and the tested transitive dependency closure for Python 3.14.5 |
| Integrity | `DATA_MANIFEST.json`, `RELEASE_SHA256.txt` | Explicit data provenance/license authorization and SHA-256 checksums |

## Tables and figures

The notation table defines symbols and has no empirical CSV input. The remaining tables draw on the following released records; the numerical summaries retain estimates that are not represented by a single CSV row.

| Manuscript table | Public input/reference locations |
| :- | :- |
| Observability funnel | `results/observability/`, `data/git_observations.parquet`, `results/reference/measurement_summary.json` |
| Registry-eligibility inflation | `data/measurement_variants.json`, `data/baseline_constraints.csv`, `results/measurement/`, `results/figures/registry_inflation.csv`, `results/reference/measurement_summary.json` |
| Exploitation-score contrasts | `data/signal_panels/epss_panel.npz`, `results/signals/epss_effects.csv`, `results/reference/measurement_summary.json`, `results/reference/activity_analysis.json` |
| Advisory contrasts | `data/signal_panels/advisory_panel.npz`, `data/signal_panels/kev_panel.npz`, `results/signals/`, `results/reference/measurement_summary.json`, `results/reference/activity_analysis.json` |
| Workflow traces | `data/workflow/`, `results/workflow/`, `results/reference/measurement_summary.json` |
| Competing risks | `data/first_exit.parquet`, `data/signal_panels/competing_events.npz`, `results/signals/cif_180d.csv`, `results/reference/measurement_summary.json`, `results/reference/activity_analysis.json` |

Figure 1 is a conceptual architecture diagram and has no empirical CSV. The exact numeric inputs for Figures 2 to 5 are `results/figures/registry_inflation.csv`, `results/figures/event_time_contrasts.csv`, `results/figures/advisory_and_bot_contrasts.csv` and `results/figures/competing_incidence.csv`, respectively. The observed-data replay regenerates and checks the corresponding numerical summaries; the figure CSVs are also supplied as recorded artifacts.

These mappings identify supporting artifacts without treating source-control observability as complete or automated candidate removals as human-verified removals. Provider attribution and retained terms are in DATA.md.
