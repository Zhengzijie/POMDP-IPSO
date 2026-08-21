# Hierarchical POMDP-IPSO for Multi-UAV Inspection

Reproducibility materials for the manuscript **“A Hierarchical Dual-Stream POMDP–IPSO Framework for Multi-UAV Cooperative Inspection Scheduling and Control”** by Zijie Zheng and Zhijun Zhang.

This repository contains the public benchmark code, deterministic synthetic task data, complete run-level results, statistical summaries, and publication-quality figures used in the revised study.

> [!IMPORTANT]
> The original operational coordinates, original simulator, trained checkpoints, and field logs were not available for public release. The included 19-scenario/574-task dataset and controller environment are a **reconstructed synthetic reproducibility benchmark**. They are not field measurements and must not be represented as the restricted operational data.

## What is included

- A synthetic scheduling benchmark with 19 scenarios and 574 tasks.
- IPSO, PSO, GA, ACO, and DE under a common objective and equal 416-evaluation budget.
- Thirty fixed scheduling seeds, scenario-level summaries, Wilcoxon tests, and Holm correction.
- A reconstructed 3-D controller benchmark with APF, A*, RRT*, MAPPO, and dual-stream MAPPO.
- Five controller training seeds and 30 unseen evaluation seeds.
- Raw CSV outputs, metadata, 600 dpi PNG figures, and vector PDF figures.

## Repository layout

```text
.
├── scripts/                 Reproduction programs
├── data/
│   ├── synthetic/           Public synthetic task records
│   └── metadata/            Seeds, budgets, and environment settings
├── results/
│   ├── scheduling/          Raw and summarized optimizer results
│   └── controller/          Raw and summarized controller results
├── figures/                 600 dpi PNG and vector PDF outputs
├── docs/                    Detailed reproduction and results notes
├── tests/                   Repository and data-integrity checks
├── DATASET_CARD.md          Dataset provenance, fields, and limitations
├── CITATION.cff             GitHub citation metadata
└── LICENSE                  MIT license
```

## Quick start

Python 3.10 or newer is supported. Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate the architecture and trajectory figures:

```bash
python scripts/generate_revision_figures.py
```

Run the scheduling benchmark:

```bash
python scripts/run_scheduling_benchmark.py
```

Run the controller benchmark (a CUDA GPU is recommended but not required):

```bash
python scripts/run_controller_benchmark.py
```

The scripts use explicit seeds and write to `data/`, `results/`, and `figures/`. Existing tracked outputs are included so that readers can inspect the results without rerunning the computational experiments. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for output maps, validation commands, and hardware notes.

## Main benchmark interpretation

In the reconstructed scheduling benchmark, IPSO has lower aggregate mean fitness than standard PSO and DE. The paired scenario-level test supports the IPSO–PSO and IPSO–DE differences after Holm correction, while the comparisons with GA and ACO are not significant at 0.05. IPSO is not best in every scenario.

In the reconstructed controller benchmark, dual-stream MAPPO improves success over standard MAPPO in the paired evaluation, but classical methods remain competitive or better on some metrics. These results support bounded comparative claims, not universal superiority.

Exact values and uncertainty are reported in [docs/RESULTS.md](docs/RESULTS.md) and the CSV files under `results/`.

## Data and ethics

All released coordinates are algorithmically generated and contain no people, account identifiers, infrastructure names, or operational geolocation. The repository intentionally excludes manuscript files, reviewer correspondence, restricted coordinates, field logs, and credentials. See [DATASET_CARD.md](DATASET_CARD.md).

## Citation

Use GitHub’s **Cite this repository** feature, which reads `CITATION.cff`. Until the article receives final bibliographic details, cite the software repository using the authors, repository title, version, and release year. After publication, add the article DOI and repository URL to `CITATION.cff`.

## Contributing and support

Please use GitHub Issues for reproducibility defects and proposed improvements. Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before contributing. Questions about the scientific study should be directed to the corresponding author through the contact information in the published article.

## License

Code, synthetic data, results, figures, and documentation in this repository are released under the [MIT License](LICENSE), unless a file explicitly states otherwise.
