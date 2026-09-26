"""Behavioral checks for neighborhood learning with ordinary acceleration MSE."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from mesh_impact_history import MeshImpactHistoryNet
from mesh_neighborhood import NeighborGraphCache, NeighborhoodAttentionBlock, geometric_neighbors


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return MeshImpactHistoryNet(width=16, num_heads=2, num_latents=4,
                                latent_layers=1, temporal_layers=1, dropout=0,
                                neighborhood_layers=2, neighborhood_k=3,
                                neighborhood_chunk_size=3, **kwargs)


def test_neighbors_use_physical_xyz_and_handle_single_nodes_and_ties():
    # Standardized Euclidean distance would pick node 1; physical distance picks 2.
    normalized = torch.tensor([[0., 0, 0], [.1, 0, 0], [0, .9, 0], [0, 0, 2.]])
    neighbors = geometric_neighbors(normalized * torch.tensor([100., 1, 1]), 2)
    assert set(neighbors[0].tolist()) == {0, 2}
    assert geometric_neighbors(normalized[:1], 16).tolist() == [[0]]
    # A grid creates equidistant candidates exactly at the kNN boundary.
    points = torch.cartesian_prod(torch.arange(3.), torch.arange(3.), torch.arange(2.))
    permutation = torch.randperm(len(points))
    before = geometric_neighbors(points, 4)
    after = permutation[geometric_neighbors(points[permutation], 4)]
    torch.testing.assert_close(after, before[permutation])


def test_local_block_reads_neighbors_and_not_unconnected_nodes():
    torch.manual_seed(10)
    block = NeighborhoodAttentionBlock(16, 2, 0, 20., 3).eval()
    points = torch.tensor([[0., 0, 0], [1., 0, 0], [3., 0, 0], [100., 0, 0]])
    nodes = torch.randn(4, 16, requires_grad=True)
    output = block(nodes, points, geometric_neighbors(points, 2))
    output[0].square().sum().backward()
    assert nodes.grad[1].abs().sum() > 0
    assert nodes.grad[2:].abs().sum() == 0


def test_chunk_checkpointing_preserves_outputs_and_parameter_gradients():
    torch.manual_seed(11)
    chunked = NeighborhoodAttentionBlock(16, 2, 0, 20., 3).train()
    full = NeighborhoodAttentionBlock(16, 2, 0, 20., 100).train()
    full.load_state_dict(chunked.state_dict())
    points, nodes = torch.randn(12, 3), torch.randn(12, 16)
    neighbors = geometric_neighbors(points, 4)
    a, b = chunked(nodes, points, neighbors), full(nodes, points, neighbors)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for p, q in zip(chunked.parameters(), full.parameters()):
        assert p.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=2e-5)


def test_new_model_is_node_order_invariant_isolates_meshes_and_backpropagates():
    torch.manual_seed(12)
    model = small_model().eval()
    mesh = torch.randn(17, 3)
    membership = torch.tensor([0] * 8 + [1] * 9)
    times = torch.tensor([-.8, 0, .8, -.5, .5])
    time_batch = torch.tensor([0, 0, 0, 1, 1])
    impact = torch.randn(2, 2)
    expected = model(mesh, membership, impact, times, time_batch)
    permutation = torch.randperm(len(mesh))
    actual = model(mesh[permutation], membership[permutation], impact, times, time_batch)
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
    single = model(mesh[:8], membership[:8], impact[:1], times[:3], time_batch[:3])
    torch.testing.assert_close(single, expected[:3], atol=1e-6, rtol=1e-5)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(mesh, membership, impact, times, time_batch).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("options", [
    {"neighborhood_layers": -1}, {"neighborhood_layers": True}, {"neighborhood_k": 0},
    {"neighborhood_chunk_size": 0}, {"neighborhood_scale_mm": float("nan")}, {"neighborhood_scale_mm": 0},
    {"impactor_nodes": -1}, {"impactor_nodes": True}, {"impactor_nodes": 1.5},
])
def test_invalid_neighborhood_configuration_fails(options):
    with pytest.raises(ValueError):
        MeshImpactHistoryNet(**options)


def test_neighborhood_mse_train_save_and_reload_with_train_only_scalers():
    from train_mesh_impact_history import DataPreprocessor, HistoryPredictor, main

    # Four locations per design; training A/C/D, validation 11, test whole B.
    # Mock only disk loading so the real split, scalers, sampler, optimizer,
    # checkpoint selection, exports and inference all execute.
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
              "--latent-layers", "1", "--temporal-layers", "1", "--neighborhood-layers", "2", "--neighborhood-k", "3",
              "--impactor-nodes", "2",
              "--device", "cpu", "--wandb-mode", "disabled", "--output-dir", folder])
        root = Path(folder)
        saved = json.loads((root / "config.json").read_text())
        history = json.loads((root / "training_history.json").read_text())
        splits = json.loads((root / "splits.json").read_text())
        assert splits["train"]["design_ids"] == [0, 6, 10]
        assert splits["validation"]["design_ids"] == [11]
        assert splits["test"]["design_ids"] == [4, 5]
        assert saved["training"]["loss"] == "normalized_acceleration_mse"
        assert saved["training"]["batch_sampling"] == "shuffle"
        assert saved["architecture"]["kwargs"]["neighborhood_layers"] == 2
        assert saved["architecture"]["kwargs"]["impactor_nodes"] == 2
        assert np.isfinite(history["train_losses"]).all()
        predictor = HistoryPredictor.from_run(folder)
        np.testing.assert_allclose(predictor.preprocessor.mesh_scaler.mean_,
                                   np.concatenate(data["mesh_geometries"][:12]).astype(np.float64).mean(0))
        exported = pd.read_csv(root / "test_acceleration_histories.csv")
        for i in range(16, 24):
            predicted = predictor.predict(data["mesh_geometries"][i], data["indentor_positions"][i])
            np.testing.assert_allclose(predicted, exported.loc[exported.run_number == data["run_numbers"][i], "acceleration_pred_g"],
                                       atol=3e-5, rtol=3e-5)
        frame = pd.read_csv(root / "training_history.csv")
        np.testing.assert_allclose(frame.train_mse_normalized, history["train_losses"])
        assert "train_objective" not in frame


# The leading nodes of a 1704 .inp *NODE block are the rigid headform. It moves
# with the impact location, so holding it out is what makes the structural
# neighbor graph constant across every impact on one design.
def headform_mesh(headform_x):
    """Two headform nodes, then six structural nodes one unit apart."""
    headform = torch.tensor([[headform_x, .1, 0.], [headform_x, .2, 0.]])
    structure = torch.tensor([[float(i), 0., 0.] for i in range(6)])
    return torch.cat((headform, structure))


def test_headform_holdout_keeps_local_attention_structural_and_passes_it_through():
    torch.manual_seed(3)
    held = small_model(impactor_nodes=2).eval()
    nodes = torch.randn(8, 16)
    near, far = headform_mesh(0.), headform_mesh(100.)
    with torch.no_grad():
        first, second = held._encode_neighborhoods(near, nodes), held._encode_neighborhoods(far, nodes)
    # Structural rows never see the headform, so they cannot follow it.
    torch.testing.assert_close(first[2:], second[2:], atol=0, rtol=0)
    # Held-out rows reach pooling exactly as embedded, unchanged by the blocks.
    torch.testing.assert_close(first[:2], nodes[:2], atol=0, rtol=0)

    # Without the holdout the same weights do follow it, so the test above is
    # measuring the exclusion rather than a headform too far away to matter.
    plain = small_model(impactor_nodes=0).eval()
    plain.load_state_dict(held.state_dict())
    with torch.no_grad():
        moved = plain._encode_neighborhoods(near, nodes), plain._encode_neighborhoods(far, nodes)
    assert not torch.allclose(moved[0][2:], moved[1][2:])


def test_neighbor_graph_is_built_once_per_geometry_and_stays_out_of_the_checkpoint():
    model = small_model(impactor_nodes=2).eval()
    nodes = torch.randn(8, 16)
    with torch.no_grad():
        for headform_x in (0., 30., 60.):          # three impacts, one design
            model._encode_neighborhoods(headform_mesh(headform_x), nodes)
    assert (model.neighbor_cache.misses, model.neighbor_cache.hits) == (1, 2)
    with torch.no_grad():                           # a second design must miss
        model._encode_neighborhoods(headform_mesh(0.) * 2.0, nodes)
    assert (model.neighbor_cache.misses, model.neighbor_cache.hits) == (2, 2)
    assert not [key for key in model.state_dict() if "neighbor_cache" in key]


def test_cache_returns_the_true_graph_and_bounds_its_own_size():
    cache = NeighborGraphCache(capacity=2)
    points = [torch.randn(9, 3) for _ in range(3)]
    for mesh in points:
        torch.testing.assert_close(cache.neighbors(mesh, 3), geometric_neighbors(mesh, 3), atol=0, rtol=0)
    assert len(cache.entries) == 2 and cache.misses == 3
    # Distinct k on one mesh is a distinct graph, never a stale hit.
    fresh = NeighborGraphCache()
    torch.testing.assert_close(fresh.neighbors(points[0], 2), geometric_neighbors(points[0], 2), atol=0, rtol=0)
    torch.testing.assert_close(fresh.neighbors(points[0], 4), geometric_neighbors(points[0], 4), atol=0, rtol=0)
    assert fresh.misses == 2
    with pytest.raises(ValueError):
        NeighborGraphCache(capacity=0)


def test_holdout_leaving_too_little_structure_is_rejected():
    model = small_model(impactor_nodes=5).eval()
    with pytest.raises(ValueError, match="structural nodes"):
        model._encode_neighborhoods(torch.randn(6, 3), torch.randn(6, 16))
