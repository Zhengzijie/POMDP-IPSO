# Included Benchmark Results

These values describe the reconstructed synthetic benchmark, not field trials. Lower scheduling fitness is better.

## Scheduling aggregate

| Algorithm | Fitness, mean ± SD | Runtime, mean (s) | Mean convergence iteration |
|---|---:|---:|---:|
| IPSO | 0.458 ± 0.112 | 0.122 | 18.98 |
| ACO | 0.456 ± 0.111 | 0.433 | 22.11 |
| GA | 0.463 ± 0.115 | 0.132 | 24.04 |
| PSO | 0.481 ± 0.124 | 0.121 | 23.17 |
| DE | 0.498 ± 0.129 | 0.142 | 19.85 |

Paired one-sided Wilcoxon tests on 19 scenario means, with Holm correction, give adjusted `p = 1.72e-5` versus PSO and `p = 7.63e-6` versus DE. Adjusted values versus GA (`p = 0.134`) and ACO (`p = 0.779`) are not significant at 0.05. IPSO wins against PSO in 18 of 19 scenario means, but it is not universally best.

## Controller aggregate

| Method | Success, mean ± SD (%) | Minimum separation, mean | Path length, mean | Runtime, mean (s) | Collision rate |
|---|---:|---:|---:|---:|---:|
| APF | 96.0 ± 8.1 | 0.055 | 1.390 | 0.017 | 0.200 |
| A* | 90.0 ± 10.2 | 0.047 | 1.226 | 2.019 | 0.500 |
| RRT* | 89.3 ± 10.1 | 0.039 | 1.256 | 0.023 | 0.533 |
| Dual-stream MAPPO | 89.3 ± 10.1 | 0.059 | 3.239 | 1.380 | 0.533 |
| MAPPO | 83.3 ± 7.6 | 0.054 | 3.183 | 0.919 | 0.833 |

The paired success comparison between dual-stream MAPPO and standard MAPPO gives `p = 0.00135`. Comparisons with APF, A*, and RRT* are not significant at 0.05. Classical methods remain competitive or better on some metrics, and the learned controllers trade runtime and path length for their policy structure.

Use the run-level CSVs for alternative analyses. Do not infer more precision than the experimental design supports.
