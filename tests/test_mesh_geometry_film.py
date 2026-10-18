"""Geometry memory injection (FiLM) in MeshImpactHistoryNet."""

import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from mesh_impact_history import GeometryMemoryEncoder, MeshImpactHistoryNet, saved_model_kwargs


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(geometry_film="all", seed=31, **kwargs):
    torch.manual_seed(seed)
    options = dict(width=16, num_heads=2, num_latents=6, latent_layers=2, temporal_layers=2,
                   dropout=0, neighborhood_layers=2, neighborhood_k=3, impactor_nodes=2)
    return MeshImpactHistoryNet(geometry_film=geometry_film, **{**options, **kwargs}).eval()


def heads(model):
    return [*model.latent_geometry_film, *getattr(model, "node_geometry_film", ())]


def perturb(modules, std=.2):
    with torch.no_grad():
        for module in modules:
            module.affine.weight.normal_(0, std)
            module.affine.bias.normal_(0, std)


def predict(model, mesh, impact):
    times = torch.tensor([-.8, -.1, .7])
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long), impact[None],
                 times, torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("mode", ["latent", "all"])
def test_injected_model_starts_as_the_uninjected_model_with_the_same_seed(mode):
    plain, injected = make_model("none", dropout=.1), make_model(mode, dropout=.1)
    shared = plain.state_dict()
    for key, value in shared.items():
        torch.testing.assert_close(injected.state_dict()[key], value, atol=0, rtol=0)
    assert set(injected.state_dict()) > set(shared)
    mesh, membership = torch.randn(20, 3), torch.tensor([0] * 9 + [1] * 11)
    impact, times, time_batch = torch.randn(2, 2), torch.randn(7), torch.tensor([0, 0, 0, 1, 1, 1, 1])
    for training in (False, True):
        outputs = []
        for model in (plain, injected):
            model.train(training)
            torch.manual_seed(5)  # Same dropout draws: the injection consumes none.
            outputs.append(model(mesh, membership, impact, times, time_batch))
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)


def test_none_adds_nothing_and_saved_configs_without_the_key_replay_uninjected():
    default = MeshImpactHistoryNet(width=16, num_heads=2, num_latents=4)
    assert default.geometry_film == "none"
    assert not [key for key in default.state_dict() if "geometry" in key]
    for saved in ({}, {"impact_conditioning": "film", "time_encoding": "fourier"}):
        assert MeshImpactHistoryNet(**saved_model_kwargs(saved), width=16, num_heads=2,
                                    num_latents=4).geometry_film == "none"
    # Frequencies follow from the saved octave count, never from the checkpoint.
    assert not [key for key in make_model().state_dict() if key.startswith("geometry_encoder.frequencies")]


def test_encoder_uses_the_papers_sinusoidal_octaves():
    encoder = GeometryMemoryEncoder(16, 5)
    torch.testing.assert_close(encoder.frequencies, torch.pi * 2.0 ** torch.arange(5.))
    assert encoder.network[0].in_features == 3 * (1 + 2 * 5)
    assert encoder(torch.randn(7, 3)).shape == (7, 16)


@pytest.mark.parametrize("mode, sites", [("latent", 2), ("all", 4)])
def test_every_site_receives_gradient_and_changes_the_prediction(mode, sites):
    model = make_model(mode)
    assert len(heads(model)) == sites
    mesh, impact = torch.randn(14, 3), torch.tensor([.3, -.2])
    predict(model, mesh, impact).square().mean().backward()
    # Zero heads still learn: their weight gradients are nonzero from step one.
    for head in heads(model):
        assert head.affine.weight.grad.abs().sum() > 0
    reference = predict(model, mesh, impact).detach()
    for head in heads(model):
        perturb([head])
        assert not torch.allclose(predict(model, mesh, impact), reference)
        with torch.no_grad():
            head.affine.weight.zero_()
            head.affine.bias.zero_()
    perturb(heads(model))
    model.zero_grad()
    predict(model, mesh, impact).square().mean().backward()
    assert all(p.grad.abs().sum() > 0 for p in model.geometry_encoder.parameters())


@pytest.mark.parametrize("mode", ["latent", "all"])
def test_token_memory_is_the_pooling_weighted_node_memory(mode):
    model = make_model(mode)
    perturb(heads(model))
    mesh, impact = torch.randn(14, 3), torch.tensor([.3, -.2])
    with torch.no_grad():
        _, geometry = model._encode_mesh_and_geometry(mesh, impact, model.impact_embedding(impact))
        weights = torch.cat([w for _, w in model.iter_pooling_attention(mesh, impact)], dim=1)
        node = model.geometry_encoder(mesh).reshape(len(mesh), 2, 8).transpose(0, 1)
    expected = (weights @ node).transpose(0, 1).reshape(6, 16)
    torch.testing.assert_close(geometry, expected, atol=1e-5, rtol=1e-5)
    # The headform rows get no pooling weight, so they add no geometry either.
    assert (weights[..., :2] == 0).all()


def test_attention_export_applies_the_node_injection_like_forward():
    model = make_model("all")
    perturb(heads(model))
    mesh, impact = torch.randn(14, 3), torch.tensor([.3, -.2])
    condition = model.impact_embedding(impact)
    with torch.no_grad():
        exported = model._mesh_pooling_inputs(mesh, impact, condition)
        forward = model._mesh_pooling_inputs(mesh, impact, condition, model.geometry_encoder(mesh))
        for a, b in zip(exported, forward):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for head in model.node_geometry_film:
            head.affine.weight.zero_()
            head.affine.bias.zero_()
        uninjected = model._mesh_pooling_inputs(mesh, impact, condition)
    assert not torch.allclose(exported[2], uninjected[2])  # Node keys carry the injection.


@pytest.mark.parametrize("mode", ["latent", "all"])
def test_impact_still_reaches_the_output_only_through_hood_node_content(mode):
    """With node values zeroed and a constant geometry memory, location is inert."""
    model = make_model(mode)
    perturb(heads(model))
    with torch.no_grad():
        model.impact_film.affine.weight.normal_(0, .1)
        model.mesh_key_value.weight[16:].zero_()
        model.mesh_key_value.bias[16:].zero_()
    mesh, impacts = torch.randn(14, 3), (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))
    varying = [predict(model, mesh, impact) for impact in impacts]
    assert not torch.allclose(*varying)  # The geometry memory itself does follow location.
    with torch.no_grad():
        norm = model.geometry_encoder.network[-1]
        norm.weight.zero_()
        norm.bias.normal_()
    constant = [predict(model, mesh, impact) for impact in impacts]
    torch.testing.assert_close(constant[0], constant[1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["latent", "all"])
def test_moving_headform_rows_cannot_change_the_injected_prediction(mode):
    model = make_model(mode)
    perturb(heads(model))
    mesh, impact = torch.randn(14, 3), torch.tensor([.3, -.2])
    moved = mesh.clone()
    moved[:2] += 5.0
    torch.testing.assert_close(predict(model, moved, impact), predict(model, mesh, impact), atol=0, rtol=0)


def test_injected_batches_keep_meshes_separate_and_ignore_node_order():
    model = make_model("all", latent_layers=1)
    perturb(heads(model))
    mesh = torch.randn(17, 3)
    membership = torch.tensor([0] * 8 + [1] * 9)
    times, time_batch = torch.tensor([-.8, 0, .8, -.5, .5]), torch.tensor([0, 0, 0, 1, 1])
    impact = torch.randn(2, 2)
    expected = model(mesh, membership, impact, times, time_batch)
    # Keep each mesh's two leading headform rows first; permute its hood rows.
    order = torch.cat((torch.tensor([0, 1]), 2 + torch.randperm(6), torch.tensor([8, 9]), 10 + torch.randperm(7)))
    permuted = model(mesh[order], membership[order], impact, times, time_batch)
    torch.testing.assert_close(permuted, expected, atol=1e-6, rtol=1e-5)
    single = model(mesh[:8], membership[:8], impact[:1], times[:3], time_batch[:3])
    torch.testing.assert_close(single, expected[:3], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("options", [
    {"geometry_film": "first"}, {"geometry_film": ""}, {"geometry_film": None}, {"geometry_film": True},
    {"geometry_film": "all", "neighborhood_layers": 0},
    {"geometry_fourier_octaves": 0}, {"geometry_fourier_octaves": True}, {"geometry_fourier_octaves": 2.0},
])
def test_invalid_geometry_configuration_fails(options):
    with pytest.raises(ValueError, match="geometry|neighborhood"):
        MeshImpactHistoryNet(width=16, num_heads=2, num_latents=4, **options)


def test_change_patch_model_refuses_geometry_injection():
    from mesh_change_attention import MeshChangeAttentionNet
    with pytest.raises(ValueError, match="geometry_film"):
        MeshChangeAttentionNet(change_anchors=2, change_k=3, width=16, num_heads=2, num_latents=4,
                               geometry_film="latent")


def test_training_cli_rejects_an_unknown_mode():
    from train_mesh_impact_history import parse_args
    assert parse_args([]).geometry_film == "none"
    assert parse_args(["--geometry-film", "all"]).geometry_film == "all"
    with pytest.raises(SystemExit):
        parse_args(["--geometry-film", "decoder"])


def test_injected_model_trains_saves_and_reloads_through_the_training_entry_point():
    from train_mesh_impact_history import DataPreprocessor, HistoryPredictor, main

    rng = np.random.default_rng(14)
    data = {key: [] for key in ("run_numbers", "mesh_geometries", "indentor_positions", "time_arrays", "accelerations")}
    times = np.linspace(0, .025, 7, dtype=np.float32)
    for design in (0, 6, 10, 11, 4, 5):
        for location in range(4):
            data["run_numbers"].append(design * 4 + location + 1)
            data["mesh_geometries"].append((rng.normal(size=(10 + location, 3)) + design).astype(np.float32))
            data["indentor_positions"].append([location * 10., location * 3.])
            data["time_arrays"].append(times.copy())
            data["accelerations"].append((20 + design * 3 + (10 + location) * np.sin(times * 100)).astype(np.float32))
    data["indentor_positions"] = np.asarray(data["indentor_positions"], dtype=np.float32)
    with tempfile.TemporaryDirectory() as folder, mock.patch.object(DataPreprocessor, "load_all_data", return_value=data):
        main(["--data-format", "euroncap1704", "--num-samples", "24", "--samples-per-design", "4",
              "--epochs", "2", "--batch-size", "4", "--width", "16", "--num-heads", "2", "--num-latents", "4",
              "--latent-layers", "2", "--temporal-layers", "1", "--neighborhood-layers", "2", "--neighborhood-k", "3",
              "--impactor-nodes", "2", "--geometry-film", "all", "--geometry-fourier-octaves", "3",
              "--neighbor-cache-dir", str(Path(folder) / "neighbor_graphs"),
              "--device", "cpu", "--wandb-mode", "disabled", "--output-dir", folder])
        root = Path(folder)
        kwargs = json.loads((root / "config.json").read_text())["architecture"]["kwargs"]
        assert kwargs["geometry_film"] == "all" and kwargs["geometry_fourier_octaves"] == 3
        checkpoint = torch.load(root / "hood_impact_best_model.pt", weights_only=True)
        trained = checkpoint["model_state_dict"]
        assert any(trained[f"latent_geometry_film.{i}.affine.weight"].abs().sum() > 0 for i in range(2))
        predictor = HistoryPredictor.from_run(folder)
        assert predictor.model.geometry_film == "all"
        exported = pd.read_csv(root / "test_acceleration_histories.csv")
        for i in range(16, 24):
            predicted = predictor.predict(data["mesh_geometries"][i], data["indentor_positions"][i])
            np.testing.assert_allclose(predicted, exported.loc[exported.run_number == data["run_numbers"][i],
                                                               "acceleration_pred_g"], atol=3e-5, rtol=3e-5)
