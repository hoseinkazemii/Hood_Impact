"""A held-out design scored against its near-clone siblings in training (test 7)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mesh_change_metrics import held_out_versus_siblings
from mesh_design_clusters import GEOMETRY_CLUSTERS

SAMPLES = 142


def split(designs, curve, locations=(1, 2)):
    """(dataset, predictions, targets) with prediction = target + 1 g everywhere."""
    runs = [design * SAMPLES + location for design in designs for location in locations]
    times = [np.linspace(0, .02, 4) for _ in runs]
    targets = np.concatenate([curve(run) for run in runs])
    dataset = SimpleNamespace(run_numbers=runs, time_arrays=times,
                              indentor_positions=np.array([[run % SAMPLES, 0.0] for run in runs]))
    return dataset, targets + 1.0, targets


def curve(run):
    design, location = (run - 1) // SAMPLES, (run - 1) % SAMPLES + 1
    return np.array([0.0, 10.0, 5.0, 1.0]) * location + (20.0 if design == 7 else design)


def test_sibling_mean_baseline_and_cross_split_pairs():
    result = held_out_versus_siblings(split([0, 1, 6, 8, 9, 10], curve), split([7], curve), SAMPLES,
                                      GEOMETRY_CLUSTERS)
    assert result["design_pairs"] == [[6, 7], [7, 8], [7, 9]]
    assert result["num_test_curves_with_siblings"] == 2
    # Siblings 6, 8, 9 sit at offsets 6, 8, 9 g; design 7 at 20 g: mean misses by 12.33 g.
    assert result["sibling_mean_rmse_g"] == pytest.approx(20 - np.mean([6, 8, 9]))
    assert result["model_rmse_g"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(1.0)  # a constant 1 g offset cancels in differences


def test_whole_cluster_held_out_has_nothing_to_compare():
    result = held_out_versus_siblings(split([0, 1, 6, 7], curve), split([4, 5], curve), SAMPLES, GEOMETRY_CLUSTERS)
    assert result["r2"] is None and result["sibling_mean_rmse_g"] is None and result["design_pairs"] == []


def test_training_with_design_7_held_out_reports_it_against_its_siblings(tmp_path, monkeypatch):
    import torch
    import train_mesh_impact_history as training
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    runs = list(range(1, 12 * SAMPLES + 1))
    data = {"run_numbers": runs,
            "indentor_positions": np.array([[(run - 1) % SAMPLES, 0.0] for run in runs], dtype=np.float32),
            "mesh_geometries": [rng.normal(size=(5, 3)).astype(np.float32) for _ in runs],
            "time_arrays": [np.linspace(0, .02, 3, dtype=np.float32) for _ in runs],
            "accelerations": [rng.normal(size=3).astype(np.float32) for _ in runs]}
    monkeypatch.setattr(training.DataPreprocessor, "load_all_data", lambda self: data)
    try:
        training.main(["--data-format", "euroncap1704", "--device", "cpu", "--epochs", "1",
                       "--batch-size", "256", "--width", "8", "--num-heads", "2", "--num-latents", "2",
                       "--latent-layers", "1", "--temporal-layers", "1", "--mesh-nodes", "all",
                       "--impactor-nodes", "2", "--test-designs", "7", "--val-designs", "5",
                       "--wandb-mode", "disabled", "--output-dir", str(tmp_path / "run")])
    finally:
        torch.set_num_threads(previous)
    for directory in (tmp_path / "run", tmp_path / "run" / "last_checkpoint"):
        report = json.loads((directory / "design_difference_r2.json").read_text())
        assert report["test"]["r2"] is None  # one test design: no pair inside the split
        assert report["test_vs_train"]["design_pairs"] == [[6, 7], [7, 8], [7, 9]]
        assert report["test_vs_train"]["num_test_curves_with_siblings"] == SAMPLES
        assert report["test_vs_train"]["sibling_mean_rmse_g"] > 0
