"""Behavioral checks for neighborhood learning with ordinary acceleration MSE."""

import json
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
    {"neighborhood_chunk_size": 0}, {"neighborhood_scale_mm": float("nan")}, {"neighborhood_scale_mm": -1.0},
])
def test_invalid_neighborhood_configuration_fails(options):
    with pytest.raises(ValueError):
        MeshImpactHistoryNet(**options)


def test_trained_run_reloads_its_neighbor_graphs_without_a_new_search(tmp_path):
    from toy_local_edits import impact, mesh, training_args, write_dataset
    from train_mesh_impact_history import HistoryPredictor, main

    root = write_dataset(tmp_path / "data")
    run = tmp_path / "run"
    main(training_args(root, run, "--epochs", "2", "--neighbor-cache-dir", str(tmp_path / "graphs")))
    saved = json.loads((run / "config.json").read_text())
    assert saved["training"]["batch_sampling"] == "shuffle"
    assert saved["architecture"]["kwargs"]["neighborhood_layers"] == 2
    # N1-N4: one graph per design, shared by all of its impact locations.
    assert saved["neighbor_graph_cache"]["unique_geometries"] == saved["neighbor_graph_cache"]["built"] == 4
    history = json.loads((run / "training_history.json").read_text())
    np.testing.assert_allclose(pd.read_csv(run / "training_history.csv").train_mse_normalized, history["train_losses"])
    predictor = HistoryPredictor.from_run(run)
    exported = pd.read_csv(run / "test_acceleration_histories.csv")
    with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected repeated search")):
        for location in (1, 2):
            predicted = predictor.predict(mesh(4), impact(location))
            np.testing.assert_allclose(predicted, exported.loc[exported.run_number == 142 * 4 + location,
                                                               "acceleration_pred_g"], atol=3e-5, rtol=3e-5)
    assert predictor.model.neighbor_cache.builds == 0
    assert predictor.model.neighbor_cache.disk_hits == 1


def test_neighbor_graph_is_built_once_per_geometry_and_stays_out_of_the_checkpoint():
    model = small_model().eval()
    nodes = torch.randn(6, 16)
    structure = torch.tensor([[float(i), 0., 0.] for i in range(6)])
    with torch.no_grad():
        for _ in range(3):                          # three impacts, one design
            model._encode_neighborhoods(structure, nodes)
    assert (model.neighbor_cache.misses, model.neighbor_cache.hits) == (1, 2)
    with torch.no_grad():                           # a second design must miss
        model._encode_neighborhoods(structure * 2.0, nodes)
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


def test_a_single_node_mesh_is_rejected():
    model = small_model().eval()
    with pytest.raises(ValueError, match="at least two nodes"):
        model._encode_neighborhoods(torch.randn(1, 3), torch.randn(1, 16))
