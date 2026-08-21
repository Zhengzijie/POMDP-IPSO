import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_synthetic_dataset_shape_and_profile():
    rows = read_csv("data/synthetic/anonymized_synthetic_19_scenarios.csv")
    assert len(rows) == 574
    counts = {}
    for row in rows:
        counts[int(row["scenario"])] = counts.get(int(row["scenario"]), 0) + 1
    assert list(counts) == list(range(1, 20))
    assert list(counts.values()) == [13, 15, 16, 19, 22, 25, 26, 28, 29, 31, 32, 34, 35, 37, 39, 41, 42, 44, 46]


def test_scheduling_results_are_complete():
    rows = read_csv("results/scheduling/scheduling_raw_30_seeds.csv")
    assert len(rows) == 19 * 30 * 5
    assert {row["algorithm"] for row in rows} == {"IPSO", "PSO", "GA", "ACO", "DE"}
    assert len({row["seed"] for row in rows}) == 30
    assert {int(row["evaluation_budget"]) for row in rows} == {416}


def test_controller_results_are_complete():
    rows = read_csv("results/controller/controller_raw_30_test_seeds.csv")
    assert len(rows) == 30 * 5
    assert {row["method"] for row in rows} == {"APF", "A*", "RRT*", "MAPPO", "Dual-stream MAPPO"}
    assert len({row["seed"] for row in rows}) == 30


def test_metadata_and_figure_pairs():
    scheduling = json.loads((ROOT / "data/metadata/scheduling_benchmark_metadata.json").read_text(encoding="utf-8"))
    controller = json.loads((ROOT / "data/metadata/controller_benchmark_metadata.json").read_text(encoding="utf-8"))
    assert scheduling["total_tasks"] == 574
    assert scheduling["evaluation_budget"] == 416
    assert scheduling["seeds"] == list(range(202601, 202631))
    assert controller["training_seeds"] == [71, 72, 73, 74, 75]
    assert controller["test_seeds"] == list(range(9101, 9131))
    for stem in (
        "Fig1_hierarchical_architecture",
        "Fig2_multiview_trajectories",
        "Fig7_reproducible_scheduling_benchmark",
        "Fig8_controller_baselines",
    ):
        assert (ROOT / "figures" / f"{stem}.png").stat().st_size > 0
        assert (ROOT / "figures" / f"{stem}.pdf").stat().st_size > 0


def test_numeric_csv_values_are_finite():
    for path in list((ROOT / "data").rglob("*.csv")) + list((ROOT / "results").rglob("*.csv")):
        rows = read_csv(path.relative_to(ROOT).as_posix())
        for row in rows:
            for value in row.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                assert math.isfinite(numeric), f"Non-finite value in {path}: {value}"
