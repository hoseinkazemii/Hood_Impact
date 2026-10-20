"""Near-clone design-difference R2 on acceleration curves, and its run backfill."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluate_design_difference_r2 import REPORT_NAME, design_difference_report, main
from mesh_change_metrics import near_clone_difference_r2

TIMES = np.array([0., .001, .002])


def split(runs, curves):
    """Dataset in run order plus flattened curves (one row per run)."""
    dataset = SimpleNamespace(run_numbers=list(runs), time_arrays=[TIMES] * len(runs))
    return dataset, np.asarray(curves, dtype=float).reshape(-1)


def test_perfect_differences_score_one_and_identical_curves_score_zero():
    # Designs 0 and 1 are near-clones; 3 locations each, samples_per_design = 3.
    base = np.array([[0., 5., 1.], [2., 7., 3.], [4., 4., 4.]])
    delta = np.array([[1., -1., 2.], [-2., 0., 3.], [-3., 1., -1.]])  # zero mean
    dataset, truth = split([1, 2, 3, 4, 5, 6], np.vstack((base, base + delta)))
    clusters = {"A": (0, 1)}
    perfect = near_clone_difference_r2(dataset, truth, truth, 3, clusters)
    assert perfect["r2"] == pytest.approx(1)
    assert perfect["design_pairs"] == [[0, 1]] and perfect["num_matched_curves"] == 3
    _, same_curve = split(dataset.run_numbers, np.vstack((base + 9, base + 9)))
    collapsed = near_clone_difference_r2(dataset, same_curve, truth, 3, clusters)
    assert collapsed["r2"] == pytest.approx(0)
    assert collapsed["predicted_difference_rms_g"] == 0
    assert collapsed["true_difference_rms_g"] == pytest.approx(np.sqrt(np.mean(delta ** 2)))


def test_only_near_clone_pairs_at_shared_locations_are_scored():
    rng = np.random.default_rng(0)
    # Design 0 has locations 1-3, design 1 only 2-3, design 2 (other cluster) all three.
    runs = [1, 2, 3, 5, 6, 7, 8, 9]
    truth = rng.normal(size=(8, 3))
    prediction = truth + rng.normal(scale=.3, size=(8, 3))
    dataset, flat_truth = split(runs, truth)
    clusters = {"A": (0, 1), "B": (2,)}
    reference = near_clone_difference_r2(dataset, prediction.reshape(-1), flat_truth, 3, clusters)
    assert reference["design_pairs"] == [[0, 1]] and reference["num_matched_curves"] == 2
    changed = prediction.copy()
    changed[0] += 100  # design 0, location 1: design 1 has no partner run
    changed[5:] -= 50  # design 2 is in another cluster
    again = near_clone_difference_r2(dataset, changed.reshape(-1), flat_truth, 3, clusters)
    assert again["r2"] == pytest.approx(reference["r2"])
    assert near_clone_difference_r2(dataset, flat_truth, flat_truth, 3, {"A": (0,), "B": (1, 2)})["design_pairs"] == [[1, 2]]
    assert near_clone_difference_r2(dataset, flat_truth, flat_truth, 3, {})["r2"] is None


def test_mismatched_time_grids_are_rejected():
    dataset, truth = split([1, 4], np.ones((2, 3)))
    dataset.time_arrays = [TIMES, TIMES + 1e-4]
    with pytest.raises(ValueError, match="Time grids differ"):
        near_clone_difference_r2(dataset, truth, truth, 3, {"A": (0, 1)})


def test_report_skips_formats_without_a_cluster_map():
    dataset, truth = split([1, 2], np.ones((2, 3)))
    report = design_difference_report("legacy", 1, {"train": (dataset, truth, truth)})
    assert report["train"]["r2"] is None and "legacy" in report["note"]


def test_backfill_reads_saved_histories_in_any_row_order(tmp_path):
    rng = np.random.default_rng(1)
    # 1704 numbering: designs 4/5 (test, cluster B) and 0/1/6 (train; 6 is cluster C).
    def frame(designs, locations):
        rows = []
        for design in designs:
            for location in locations:
                truth = rng.normal(size=3)
                rows.append(pd.DataFrame({"run_number": design * 142 + location, "time": TIMES,
                                          "acceleration_true_g": truth,
                                          "acceleration_pred_g": truth + rng.normal(scale=.2, size=3)}))
        return pd.concat(rows, ignore_index=True)
    train, test = frame((0, 1, 6), (1, 2, 3)), frame((4, 5), (1, 2))
    (tmp_path / "config.json").write_text(json.dumps({"data": {"data_format": "euroncap1704", "samples_per_design": 142}}))
    train.sample(frac=1, random_state=2).to_csv(tmp_path / "train_acceleration_histories.csv", index=False)
    test.sample(frac=1, random_state=3).to_csv(tmp_path / "test_acceleration_histories.csv", index=False)
    main([str(tmp_path)])
    report = json.loads((tmp_path / REPORT_NAME).read_text())
    assert report["train"]["design_pairs"] == [[0, 1]] and report["train"]["num_matched_curves"] == 3
    assert report["test"]["design_pairs"] == [[4, 5]] and report["test"]["num_matched_curves"] == 2

    def expected(saved, a, b, locations):
        def curves(design, column):
            return np.concatenate([saved[saved.run_number == design * 142 + location][column].to_numpy()
                                   for location in locations])
        d = curves(b, "acceleration_true_g") - curves(a, "acceleration_true_g")
        p = curves(b, "acceleration_pred_g") - curves(a, "acceleration_pred_g")
        return 1 - np.sum((p - d) ** 2) / np.sum((d - d.mean()) ** 2)
    assert report["train"]["r2"] == pytest.approx(expected(train, 0, 1, (1, 2, 3)))
    assert report["test"]["r2"] == pytest.approx(expected(test, 4, 5, (1, 2)))
