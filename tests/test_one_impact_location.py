"""Single-location, geometry-only training: no location input, frame-only meshes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mesh_impact_history import MeshImpactHistoryNet


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(**kwargs):
    torch.manual_seed(5)
    options = dict(width=16, num_heads=2, num_latents=6, latent_layers=1, temporal_layers=1,
                   dropout=0, neighborhood_layers=1, neighborhood_k=3, impactor_nodes=2,
                   impact_conditioning="none")
    return MeshImpactHistoryNet(**{**options, **kwargs}).eval()


def predict(model, mesh, impact):
    times = torch.tensor([-.8, -.1, .7])
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long), impact[None],
                 times, torch.zeros(3, dtype=torch.long))


def test_no_location_model_has_no_location_parameters_and_only_global_tokens():
    model = make_model()
    for name in ("impact_embedding", "impact_film", "local_log_precision"):
        assert not hasattr(model, name)
    assert model.num_global_latents == model.num_latents
    mesh = torch.randn(14, 3)
    tokens, _, _, _, bias = model._mesh_pooling_inputs(mesh, torch.tensor([.3, -.4]), None)
    torch.testing.assert_close(tokens, model.query_norm(model.latent_queries))
    torch.testing.assert_close(bias[:, 2:], torch.zeros(6, 12))
    assert torch.isneginf(bias[:, :2]).all()  # The headform rows stay out of pooling.


@pytest.mark.parametrize("decoder", ["temporal", "mesh_only"])
def test_no_location_prediction_ignores_the_impact_but_not_the_geometry(decoder):
    model = make_model(decoder=decoder)
    mesh = torch.randn(14, 3)
    first = predict(model, mesh, torch.tensor([-.5, .2]))
    torch.testing.assert_close(first, predict(model, mesh, torch.tensor([.9, -1.3])), atol=0, rtol=0)
    moved = mesh.clone()
    moved[5:9, 2] += .5  # Raise part of the hood.
    assert not torch.allclose(first, predict(model, moved, torch.tensor([-.5, .2])))


def test_no_location_backward_reaches_every_parameter():
    model = make_model().train()
    mesh = torch.randn(14, 3)
    predict(model, mesh, torch.zeros(2)).square().sum().backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.grad is None or not parameter.grad.abs().sum() > 0]
    assert not missing


def write_deck(path, design, extra_inner):
    """Tiny 1704-style deck: headform first, then outer, inner and hinge shells."""
    nodes = [(1, 0, 0, 9), (2, 1, 0, 9), (3, 1, 1, 9), (4, 0, 1, 9),          # headform
             (5, 0, 0, 5), (6, 3, 0, 5), (7, 3, 3, 5), (8, 0, 3, 5)]          # outer skin
    inner = [(9, 0, 0, 2 + .1 * design), (10, 3, 0, 2), (11, 3, 3, 2), (12, 0, 3, 2)]
    inner += [(13 + i, .5 + i, 1.5, 2.2 + .05 * design) for i in range(extra_inner)]
    hinges = [(40, 5, 5, 1), (41, 6, 5, 1), (42, 6, 6, 1), (43, 5, 6, 1)]
    elements = {"Rigid Body1_2-1-3": [(1, 1, 2, 3, 4)], "Hood_Outer-1-2": [(2, 5, 6, 7, 8)],
                "Hood_Inner-1-2": [(3, 9, 10, 11, 12)] + [(4 + i, 9, 10, 13 + i, 12) for i in range(extra_inner)],
                "Hinges-1-2": [(30, 40, 41, 42, 43)]}
    lines = ["*HEADING", "*NODE"]
    lines += [f"{n}, {x}, {y}, {z}" for n, x, y, z in nodes + inner + hinges]
    lines.append("*ELEMENT, TYPE=S4")
    lines += [", ".join(map(str, element)) for group in elements.values() for element in group]
    for name, group in elements.items():
        lines += [f'*ELSET, ELSET="{name}"', ", ".join(str(element[0]) for element in group)]
    lines.append('*RIGID BODY, REF NODE=1, ELSET="Rigid Body1_2-1-3"')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return np.array([xyz for _, *xyz in nodes + inner + hinges], dtype=np.float32)


def test_inner_mesh_keeps_the_headform_first_and_only_the_inner_frame(tmp_path):
    from utils.utils import Config, DataPreprocessor

    full = write_deck(tmp_path / "HoodImpact_1.inp", design=2, extra_inner=3)
    config = Config(data_format="euroncap1704", inp_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    np.testing.assert_array_equal(DataPreprocessor(config).load_mesh_geometry(1), full)
    config.mesh_parts = "inner"
    inner = DataPreprocessor(config).load_mesh_geometry(1)
    np.testing.assert_array_equal(inner, np.concatenate((full[:4], full[8:15])))


def test_splits_allow_no_validation_design_but_never_an_empty_train_or_test_split():
    from train_mesh_impact_history import resolve_splits

    config = SimpleNamespace(samples_per_design=142)
    data = {"run_numbers": [142 * design + 3 for design in range(12)]}
    split = resolve_splits(data, config, [7], [])
    assert split["validation"] == {"design_ids": [], "run_numbers": []}
    assert split["test"]["run_numbers"] == [142 * 7 + 3]
    assert len(split["train"]["run_numbers"]) == 11
    with pytest.raises(ValueError, match="test design list"):
        resolve_splits(data, config, [], [11])
    with pytest.raises(ValueError, match="train split is empty"):
        resolve_splits(data, config, list(range(11)), [11])


def test_location_selection_is_limited_to_the_full_grid():
    from train_mesh_impact_history import selected_location_runs

    config = SimpleNamespace(data_format="euroncap1704", samples_per_design=142, num_samples=1704)
    args = SimpleNamespace(locations=[137, 9], hic_range_threshold_percent=None, location_holdout=None)
    runs = selected_location_runs(args, config)
    assert runs == sorted([142 * d + 137 for d in range(12)] + [142 * d + 9 for d in range(12)])
    for bad in ([0], [143], [9, 9]):
        with pytest.raises(ValueError):
            selected_location_runs(SimpleNamespace(**{**vars(args), "locations": bad}), config)
    with pytest.raises(ValueError, match="replaces"):
        selected_location_runs(SimpleNamespace(**{**vars(args), "hic_range_threshold_percent": 0.0}), config)
    with pytest.raises(ValueError, match="complete"):
        selected_location_runs(args, SimpleNamespace(**{**vars(config), "num_samples": 60}))


def test_one_location_training_writes_train_predictions_and_the_location_report(tmp_path):
    from train_mesh_impact_history import HistoryPredictor, main

    location = 3
    inp_dir, acceleration_dir = tmp_path / "inp_files", tmp_path / "acc"
    inp_dir.mkdir()
    acceleration_dir.mkdir()
    times = np.linspace(0, .025, 101)
    meshes = {}
    for design in range(12):
        run = 142 * design + location
        meshes[run] = write_deck(inp_dir / f"HoodImpact_{run}.inp", design, extra_inner=design % 3)
        acceleration = 60 + 5 * design + (40 + 3 * design) * np.sin(np.pi * times / .025) ** 2
        pd.DataFrame({"Time": times, "A(in g)": acceleration}).to_csv(
            acceleration_dir / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    coords = tmp_path / "ImpactCoords_1704.csv"
    pd.DataFrame({"X1": np.full(1704, 1.5), "X2": np.full(1704, 1.5), "X3": np.full(1704, 9.0)}).to_csv(
        coords, index=False)
    run_dir = tmp_path / "run"
    metrics = main([
        "--data-format", "euroncap1704", "--inp-dir", str(inp_dir), "--impact-coords-path", str(coords),
        "--acceleration-dir", str(acceleration_dir), "--locations", str(location), "--mesh-parts", "inner",
        "--test-designs", "7", "--val-designs", "--impact-conditioning", "none", "--impactor-nodes", "4",
        "--epochs", "2", "--batch-size", "4", "--width", "16", "--num-heads", "2", "--num-latents", "4",
        "--latent-layers", "1", "--temporal-layers", "1", "--dropout", "0", "--time-subsample-stride", "4",
        "--device", "cpu", "--wandb-mode", "disabled", "--output-dir", str(run_dir),
    ])
    assert all(np.isfinite(value) for value in metrics.values())

    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["data"]["locations"] == [location]
    assert saved["preprocessing"]["mesh_parts"] == "inner"
    assert saved["training"]["checkpoint_selection"] == "training_designs_eval_loss"
    assert saved["architecture"]["kwargs"]["impact_conditioning"] == "none"
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    assert splits["validation"]["design_ids"] == [] and splits["test"]["design_ids"] == [7]
    train = pd.read_csv(run_dir / "train_acceleration_histories.csv")
    assert sorted(train.run_number.unique()) == [142 * d + location for d in range(12) if d != 7]
    assert set(json.loads((run_dir / "train_metrics.json").read_text())) == {"mse", "rmse", "mae", "r2"}

    report = json.loads((run_dir / "location_report" / "report.json").read_text(encoding="utf-8"))
    (summary,) = report["locations"]
    assert summary["test_designs"] == [7] and len(summary["train_designs"]) == 11
    test = summary["test"]["7"]
    assert test["cluster_siblings_in_training"] == [6, 8, 9]
    assert np.isfinite(test["cluster_sibling_mean"]["rmse_g"]) and np.isfinite(test["training_mean"]["rmse_g"])
    assert {"0-1", "4-5", "10-11"} <= set(summary["within_cluster_difference_r2"])
    table = pd.read_csv(run_dir / "location_report" / "summary.csv")
    assert len(table) == 12 and table.hic15_true_full_resolution.notna().all()
    for figure in summary["figures"]:
        assert (run_dir / "location_report" / figure).is_file()
    assert summary["figures"] == [f"location_{location}_all_designs.png", f"location_{location}_designs_4_5.png"]

    # Reload predicts the exported test history from the frame-only mesh alone.
    predictor = HistoryPredictor.from_run(run_dir)
    run = 142 * 7 + location
    frame = np.concatenate((meshes[run][:4], meshes[run][8:-4]))
    exported = pd.read_csv(run_dir / "test_acceleration_histories.csv")
    np.testing.assert_allclose(predictor.predict(frame, np.array([1.5, 1.5])),
                               exported.acceleration_pred_g, rtol=2e-5, atol=2e-5)
