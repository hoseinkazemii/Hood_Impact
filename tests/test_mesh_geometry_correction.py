"""Direct correction behavior, optimization, isolation, and portable reload."""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from geometry_correction_basis import GeometryCorrectionBasis
from mesh_geometry_correction import CoefficientHistoryCorrection, GeometryCorrectionPredictor, load_correction_geometry
from mesh_impact_history import MeshImpactHistoryNet
from train_mesh_impact_history import HistoryPredictor
import train_mesh_geometry_correction as pipeline
from utils.utils import DataPreprocessor
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_initial_identity_and_zero_geometry_correction():
    model = CoefficientHistoryCorrection(3, width=8, layers=1)
    impact, times = torch.randn(2, 2), torch.linspace(0, .025, 63)
    assert torch.equal(model(torch.randn(2, 3), impact, times), torch.zeros(2, 63))
    with torch.no_grad():
        model.response_functions[-1].weight.normal_()
        model.response_functions[-1].bias.normal_()
    assert torch.equal(model(torch.zeros(2, 3), impact, times), torch.zeros(2, 63))


def test_direct_geometry_dependence_and_fourier_seconds():
    model = CoefficientHistoryCorrection(2, width=8, layers=1, fourier_num_frequencies=2,
                                          fourier_min_frequency_hz=20, fourier_max_frequency_hz=40)
    impact, times = torch.zeros(2, 2), torch.tensor([0., .0125])
    with torch.no_grad():
        model.response_functions[-1].bias.copy_(torch.tensor([3., -2.]))
    coefficients = torch.tensor([[1., 0.], [0., 1.]])
    torch.testing.assert_close(model(coefficients, impact, times), torch.tensor([[3., 3.], [-2., -2.]]))
    captured = []
    hook = model.response_functions[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    model(coefficients, impact, times)
    hook.remove()
    torch.testing.assert_close(captured[0][0, 0, 2:], torch.tensor([0., 1., 0., 1.]))
    torch.testing.assert_close(captured[0][0, 1, 2:], torch.tensor([1., 0., 0., -1.]), atol=2e-6, rtol=0)
    loss = (model(coefficients, impact, times) - 10).square().mean()
    loss.backward()
    assert model.response_functions[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [{"coefficient_width": 0}, {"width": 0}, {"layers": -1},
                                    {"fourier_min_frequency_hz": 0}, {"fourier_max_frequency_hz": 1}])
def test_rejects_invalid_model_settings(kwargs):
    with pytest.raises(ValueError):
        CoefficientHistoryCorrection(**{"coefficient_width": 2, **kwargs})


def _write_deck(path, mesh):
    rows = ["*NODE"] + [f"{i},{x},{y},{z}" for i, (x, y, z) in enumerate(mesh, 1)]
    rows += ["*ELEMENT, TYPE=S3R", "1,1,2,3", "2,4,5,6",
             '*ELSET, ELSET="Hood_Outer-1-2"', "1", '*ELSET, ELSET="Hood_Inner-1-2"', "2"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def toy_baseline(tmp_path):
    """Actual saved baseline + actual small decks, targets and frozen scalers."""
    source, root = tmp_path / "baseline", tmp_path / "data"
    source.mkdir()
    (root / "inp_files").mkdir(parents=True)
    (root / "output_history_acc").mkdir()
    times = np.linspace(0, .024, 7, dtype=np.float32)
    meshes = []
    for design in range(4):
        mesh = np.asarray([[0, 0, 0], [10, 0, 0], [0, 10, 0],
                           [0, 0, -10 + design], [10, 0, -10], [0, 10, -10]], dtype=np.float32)
        meshes.append(mesh)
    positions = np.asarray([[0., 0.], [5., 5.]] * 4, dtype=np.float32)
    pre = DataPreprocessor(SimpleNamespace(prediction_target="acceleration", samples_per_design=2))
    pre.mesh_scaler.fit(np.concatenate(meshes[:2]))
    pre.indentor_scaler.fit(positions[:4])
    pre.time_scaler.fit(times[:, None])
    pre.accel_scaler.fit(np.asarray([[20.], [40.]]))
    pre.hic_scaler.fit([[0.], [1.]])
    pre._fitted = True
    pre.save_scalers(str(source / "scalers.joblib"))
    torch.manual_seed(9)
    kwargs = dict(width=8, num_heads=2, num_latents=2, latent_layers=1, temporal_layers=1,
                  dropout=0., neighborhood_layers=0, impactor_nodes=0, impact_conditioning="film",
                  time_encoding="fourier", fourier_num_frequencies=2,
                  fourier_min_frequency_hz=20., fourier_max_frequency_hz=40.)
    model = MeshImpactHistoryNet(**kwargs)
    model.set_coordinate_scalers(pre.mesh_scaler.mean_, pre.mesh_scaler.scale_,
                                 pre.indentor_scaler.mean_, pre.indentor_scaler.scale_)
    model.set_time_scaler(pre.time_scaler.mean_[0], pre.time_scaler.scale_[0])
    torch.save({"model_state_dict": model.state_dict()}, source / "hood_impact_best_model.pt")
    splits = {"train": {"design_ids": [0, 1], "run_numbers": [1, 2, 3, 4]},
              "validation": {"design_ids": [2], "run_numbers": [5, 6]},
              "test": {"design_ids": [3], "run_numbers": [7, 8]}}
    config = dict(format_version=1, architecture=dict(name="MeshImpactHistoryNet", kwargs=kwargs),
                  data=dict(data_format="euroncap1704", num_samples=8, samples_per_design=2,
                            inp_dir=str(root / "inp_files"), impact_coords_path=str(root / "ImpactCoords_1704.csv"),
                            acceleration_dir=str(root / "output_history_acc")),
                  preprocessing=dict(prediction_target="acceleration", time_subsample_stride=1, max_train_time=None),
                  prediction_grid=dict(same_grid_for_all_loaded_runs=True, num_time_points=len(times)))
    pipeline.write_json(source / "config.json", config)
    pipeline.write_json(source / "splits.json", splits)
    np.save(source / "prediction_times.npy", times)
    pd.DataFrame(dict(X1=positions[:, 0], X2=positions[:, 1])).to_csv(root / "ImpactCoords_1704.csv", index=False)
    for run in range(1, 9):
        _write_deck(root / "inp_files" / f"HoodImpact_{run}.inp", meshes[(run - 1) // 2])
    labels = np.asarray(["outer"] * 3 + ["inner"] * 3)
    basis = GeometryCorrectionBasis.fit(meshes[:2], [0, 1], [labels, labels], anchor_spacing_mm=10., profile_k=2)
    predictor = HistoryPredictor(model, pre, times)
    for run in range(1, 9):
        design = (run - 1) // 2
        coefficients, _ = basis.transform(meshes[design], labels)
        impact = positions[run - 1]
        base = predictor.predict(meshes[design], impact)
        delta = coefficients[0] * (3 * np.sin(2 * np.pi * 20 * times) + 2 * np.cos(2 * np.pi * 40 * times) + .1 * impact[0])
        pd.DataFrame({"Time": times, "A(in g)": base + delta}).to_csv(
            root / "output_history_acc" / f"HoodImpact_{run}_SAE1000_interp1000.csv", index=False)
    return source, root


def _args(source, root, output, epochs=60):
    return ["--baseline-run", str(source), "--data-root", str(root), "--output-dir", str(output),
            "--device", "cpu", "--epochs", str(epochs), "--batch-size", "4", "--lr", ".02",
            "--correction-width", "16", "--correction-layers", "1", "--profile-k", "2",
            "--fourier-num-frequencies", "2", "--fourier-max-frequency-hz", "40",
            "--wandb-mode", "disabled", "--no-plots"]


def test_end_to_end_frozen_baseline_training_and_portable_inference(tmp_path, toy_baseline):
    source, root = toy_baseline
    output = tmp_path / "correction"
    source_hashes = {name: pipeline.file_sha256(source / name) for name in pipeline.BASELINE_FILES}
    accesses = []
    real_load = pipeline.load_selected_data
    def checked_load(pre, runs):
        accesses.extend(runs)
        if 7 in runs:
            assert (output / "correction_best.pt").is_file()
            assert len(torch.load(output / "correction_last.pt", weights_only=True)["history"]["train_losses"]) == 60
        return real_load(pre, runs)
    with mock.patch.object(pipeline, "load_selected_data", side_effect=checked_load):
        metrics = pipeline.main(_args(source, root, output))
    assert metrics["corrected"]["acceleration"]["rmse"] < metrics["baseline"]["acceleration"]["rmse"] * .15
    assert source_hashes == {name: pipeline.file_sha256(source / name) for name in pipeline.BASELINE_FILES}
    assert accesses == [1, 2, 3, 4, 5, 6, 7, 8]
    assert json.loads((output / "baseline_integrity.json").read_text())["unchanged"]
    basis = GeometryCorrectionBasis.load(output / "geometry_basis.npz")
    assert basis.training_design_ids == [0, 1]
    assert (output / "test_hic_per_curve.csv").is_file()
    train_metrics = json.loads((output / "train_metrics.json").read_text())
    assert train_metrics["corrected"]["design_sensitivity"]["design_difference_skill"] > .95
    predictor = GeometryCorrectionPredictor.from_run(output)
    assert not any(parameter.requires_grad for parameter in predictor.baseline.model.parameters())
    components = predictor.predict_inp(root / "inp_files" / "HoodImpact_7.inp", [0., 0.], return_components=True)
    frame = pd.read_csv(output / "test_acceleration_histories.csv")
    expected = frame.loc[frame.run_number == 7, "acceleration_pred_g"].to_numpy()
    np.testing.assert_allclose(components["corrected_g"], expected, atol=5e-6, rtol=1e-6)
    np.testing.assert_allclose(components["corrected_g"], components["baseline_g"] + components["correction_g"])
    # Frozen re-evaluation must reuse cached predictions and preserve results.
    with mock.patch.object(HistoryPredictor, "predict", side_effect=AssertionError("Cached baseline must be reused")):
        again = pipeline.main(["--evaluate-run", str(output), "--data-root", str(root), "--device", "cpu", "--no-plots"])
    assert again["corrected"]["acceleration"] == metrics["corrected"]["acceleration"]
    # Portable inference refuses silently changed baseline preprocessing.
    snapshot_config = output / "baseline" / "config.json"
    original_bytes = snapshot_config.read_bytes()
    snapshot_config.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="baseline artifact changed"):
        GeometryCorrectionPredictor.from_run(output)
    snapshot_config.write_bytes(original_bytes)
    # A changed geometry artifact cannot be paired with the old response weights.
    basis_path = output / "geometry_basis.npz"
    original_basis = basis_path.read_bytes()
    basis_path.write_bytes(original_basis + b"changed")
    with pytest.raises(ValueError, match="geometry basis changed"):
        GeometryCorrectionPredictor.from_run(output)
    basis_path.write_bytes(original_basis)


def test_resume_preserves_optimizer_schedule_and_predictions(tmp_path, toy_baseline):
    source, root = toy_baseline
    interrupted, resumed, uninterrupted = [tmp_path / name for name in ("interrupted", "resumed", "uninterrupted")]
    real_save = pipeline._save_checkpoint
    def stop_after_save(path, *arguments):
        real_save(path, *arguments)
        if Path(path).name == "correction_last.pt" and arguments[3] == 3:
            raise RuntimeError("simulated timeout")
    with mock.patch.object(pipeline, "_save_checkpoint", side_effect=stop_after_save):
        with pytest.raises(RuntimeError, match="simulated timeout"):
            pipeline.main(_args(source, root, interrupted, epochs=8))
    assert not (interrupted / "test_cache.npz").exists()
    with mock.patch.object(GeometryCorrectionBasis, "fit", side_effect=AssertionError("Never refit on resume")):
        pipeline.main(["--resume-from", str(interrupted), "--output-dir", str(resumed), "--device", "cpu",
                       "--wandb-mode", "disabled", "--no-plots"])
    pipeline.main(_args(source, root, uninterrupted, epochs=8))
    actual = torch.load(resumed / "correction_last.pt", weights_only=True)
    expected = torch.load(uninterrupted / "correction_last.pt", weights_only=True)
    assert actual["history"] == expected["history"]
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    for key, value in actual["model_state_dict"].items():
        torch.testing.assert_close(value, expected["model_state_dict"][key], rtol=0, atol=0)


def test_refuses_cohort_mismatch_and_existing_output(tmp_path, toy_baseline):
    source, root = toy_baseline
    times = np.load(source / "prediction_times.npy")
    times[-1] += .001
    np.save(source / "prediction_times.npy", times)
    with pytest.raises(ValueError, match="time grid"):
        pipeline.main(_args(source, root, tmp_path / "mismatched", epochs=1))
    with pytest.raises(FileExistsError):
        pipeline.main(_args(source, root, tmp_path / "mismatched", epochs=1))
