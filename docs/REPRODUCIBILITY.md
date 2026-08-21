# Reproducibility Guide

## Environment

- Python: 3.10 or newer; 3.11 recommended.
- Scheduling and drawing: CPU is sufficient.
- Controller training: CUDA GPU recommended; CPU execution is supported but slower.
- Tested dependency families are recorded in `requirements.txt`.

Create and activate a virtual environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For repository tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Programs and outputs

### Figure reconstruction

```bash
python scripts/generate_revision_figures.py
```

Writes Fig. 1 and Fig. 2 to `figures/` in 600 dpi PNG and vector PDF formats.

### Scheduling benchmark

```bash
python scripts/run_scheduling_benchmark.py
```

Regenerates:

- `data/synthetic/anonymized_synthetic_19_scenarios.csv`;
- `data/metadata/scheduling_benchmark_metadata.json`;
- all `results/scheduling/*.csv` files;
- `figures/Fig7_reproducible_scheduling_benchmark.png` and `.pdf`.

Every algorithm uses the same objective and 416 objective evaluations per run. The script executes IPSO, PSO, GA, ACO, and DE for 30 fixed seeds in each of 19 scenarios. The paired tests use scenario means and Holm-adjust four scheduling comparisons.

### Controller benchmark

```bash
python scripts/run_controller_benchmark.py
```

Regenerates:

- `data/metadata/controller_benchmark_metadata.json`;
- all `results/controller/*.csv` files;
- `figures/Fig8_controller_baselines.png` and `.pdf`.

The script reconstructs a normalized 3-D simulator, trains standard and dual-stream parameter-sharing MAPPO with seeds 71–75, and evaluates APF, A*, RRT*, MAPPO, and dual-stream MAPPO on unseen seeds 9101–9130.

## Determinism and expected differences

Task generation and scheduling seeds are explicit. CSV ordering and scenario values should be reproducible. Runtime measurements are expected to vary with processor, GPU, operating system, and system load. PyTorch GPU kernels and different library versions can introduce small floating-point or training differences; therefore validate the direction and uncertainty of comparisons rather than requiring byte-identical controller metrics.

The program records whether CUDA was used in controller metadata. A regenerated metadata file may say `cpu` instead of `cuda` without indicating an error.

## Integrity checks

Run:

```bash
python -m pytest
python -m compileall -q scripts
```

The tests verify the expected scenario/task counts, raw-result dimensions, algorithm sets, metadata seeds, figure pairs, and absence of non-finite numeric CSV values.

## Scientific boundary

These experiments are a reconstructed public benchmark. They complement, but do not replace, the restricted operational scenarios described in the manuscript. The original simulator, trained checkpoint, operational coordinates, and field logs are not part of this repository.
