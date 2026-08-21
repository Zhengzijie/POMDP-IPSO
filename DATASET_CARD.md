# Dataset Card

## Dataset summary

`data/synthetic/anonymized_synthetic_19_scenarios.csv` is a deterministic, algorithmically generated benchmark for multi-UAV inspection scheduling. It contains 574 synthetic task records distributed across 19 scenarios with 13–46 tasks per scenario.

The dataset was created because the original operational coordinates were not available for public release. It is designed to reproduce the reported problem-size profile and to permit fair seeded comparisons; it is not a geographic transformation or anonymized export of field coordinates.

## Provenance and generation

The generator is implemented in `scripts/run_scheduling_benchmark.py`. Scenario `i` uses NumPy random generator seed `41000 + i`. Coordinates follow synthetic corridor-like centerlines with bounded random perturbations. Task type, priority, service time, and time windows are sampled from the documented distributions in the program. Running the script regenerates the CSV deterministically.

## Schema

| Field | Type | Meaning |
|---|---:|---|
| `scenario` | integer | Scenario identifier, 1–19 |
| `task_id` | integer | Task identifier within a scenario |
| `x_m`, `y_m`, `z_m` | float | Synthetic Cartesian coordinates in metres |
| `priority` | integer | Synthetic priority level, 1–5 |
| `task_type` | integer | Synthetic task class, 0–5 |
| `service_s` | float | Required service duration in seconds |
| `earliest_s` | float | Earliest synthetic service time in seconds |
| `latest_s` | float | Latest synthetic service time in seconds |

The scheduling metadata records UAV count, optimizer budget, objective weights, task-count profile, and all evaluation seeds. Controller metadata records the normalized environment, training seeds, test seeds, UAV/obstacle counts, and training budget.

## Related results

- `results/scheduling/scheduling_raw_30_seeds.csv`: 2,850 rows (19 scenarios × 30 seeds × 5 algorithms).
- `results/controller/controller_raw_30_test_seeds.csv`: 150 rows (30 test seeds × 5 methods).
- Other CSV files provide summaries and paired nonparametric tests.

Runtime columns are machine-dependent measurements and may differ when regenerated. Objective values and seeded synthetic data should be deterministic up to numerical differences among platforms and library versions. GPU training can introduce small nondeterministic variation even when seeds are set.

## Intended uses

- Reproducing the public scheduling and controller comparisons.
- Testing optimization, multi-agent control, statistical evaluation, and plotting pipelines.
- Extending the benchmark with clearly documented methods under comparable budgets.

## Out-of-scope uses

- Inferring or locating real power-line infrastructure.
- Operational flight planning or safety certification.
- Treating synthetic performance as field validation.
- Making claims about universal algorithm superiority.

## Privacy, sensitivity, and ethics

The released data are synthetic and contain no personal data, protected attributes, account identifiers, infrastructure names, or real geolocation. No human-subject annotations are included. The repository must not be combined with restricted operational data without the data owner’s authorization and an appropriate release review.

## License and maintenance

The dataset is distributed under the repository’s MIT License. Corrections should be submitted through a GitHub Issue with the affected file, row or seed, expected behavior, and reproduction command.
