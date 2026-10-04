"""Persisted geometric lists are portable across models, scalers and devices."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from mesh_impact_history import MeshImpactHistoryNet
from mesh_neighborhood import NeighborGraphCache, geometric_neighbors, prepare_model_neighbor_graphs


def test_fresh_process_reuses_saved_lists_without_running_search(tmp_path):
    points = torch.randn(23, 3)
    first = NeighborGraphCache(directory=tmp_path)
    expected = first.neighbors(points, 7)
    assert first.builds == first.writes == 1
    assert expected.dtype == torch.long
    with np.load(first.path_for(points, 7), allow_pickle=False) as saved:
        assert saved["neighbors"].dtype == np.int32
    second = NeighborGraphCache(directory=tmp_path)
    with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected kNN search")):
        actual = second.neighbors(points, 7)
        again = second.neighbors(points, 7)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual is again
    assert second.builds == 0 and second.disk_hits == 1 and second.hits == 1


def test_geometry_order_and_k_have_distinct_files(tmp_path):
    torch.manual_seed(2)
    points = torch.randn(17, 3)
    variants = [(points, 5), (points * torch.tensor([1., 1., 2.]), 5), (points.flip(0), 5), (points, 8)]
    cache = NeighborGraphCache(directory=tmp_path)
    for geometry, count in variants:
        actual = cache.neighbors(geometry, count)
        torch.testing.assert_close(actual, geometric_neighbors(geometry, count), atol=0, rtol=0)
    assert len(list(tmp_path.glob("*.npz"))) == 4 and cache.builds == 4


@pytest.mark.parametrize("damage", ["truncated", "wrong_format", "out_of_range", "checksum"])
def test_corrupt_graph_is_rebuilt_instead_of_used(tmp_path, damage):
    points = torch.randn(13, 3)
    cache = NeighborGraphCache(directory=tmp_path)
    expected = cache.neighbors(points, 4)
    path = cache.path_for(points, 4)
    if damage == "truncated":
        path.write_bytes(b"incomplete")
    elif damage == "wrong_format":
        with path.open("wb") as file:
            np.save(file, np.arange(5))
    else:
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        payload["neighbors"][0, 0] = len(points) if damage == "out_of_range" else (int(payload["neighbors"][0, 0]) + 1) % len(points)
        np.savez(path, **payload)
    second = NeighborGraphCache(directory=tmp_path)
    actual = second.neighbors(points, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert second.invalid_files == second.builds == second.writes == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_registration_uses_raw_graph_and_keeps_a_snapshot_of_raw_xyz(tmp_path):
    raw = torch.randn(12, 3)
    original = raw.clone()
    runtime = raw + 100.
    cache = NeighborGraphCache(directory=tmp_path)
    cache.neighbors(runtime, 4)
    cache.register_geometry(raw, runtime, 4)
    assert cache.path_for(original, 4).is_file()
    raw.add_(5.)  # The caller is free to reuse or modify its input array.
    cache.entries.clear()
    with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected kNN search")):
        restored = cache.register_geometry(original, runtime, 4)
    torch.testing.assert_close(restored, geometric_neighbors(original, 4), atol=0, rtol=0)


def test_disk_reload_after_memory_eviction_does_not_repeat_search(tmp_path):
    meshes = [torch.randn(9, 3), torch.randn(11, 3)]
    cache = NeighborGraphCache(capacity=1, directory=tmp_path)
    expected = cache.neighbors(meshes[0], 3)
    cache.neighbors(meshes[1], 3)
    with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected kNN search")):
        torch.testing.assert_close(cache.neighbors(meshes[0], 3), expected, atol=0, rtol=0)
    assert cache.builds == 2 and cache.disk_hits == 1 and len(cache.entries) == 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_raw_geometry_reuses_graph_across_scalers_and_headform_positions(tmp_path, device):
    torch.manual_seed(8)
    structure = (torch.randn(19, 3) * torch.tensor([100., 20., 5.]) + torch.tensor([1000., -50., 800.])).numpy()
    first = NeighborGraphCache(directory=tmp_path)
    expected = first.neighbors(torch.from_numpy(structure), 6)
    meshes = [np.concatenate((np.full((2, 3), impact, dtype=np.float32), structure)) for impact in (0., 100.)]
    for statistics in ([structure], [structure, structure + [100., 15., 2.]]):
        scaler = StandardScaler().fit(np.concatenate(statistics))
        model = MeshImpactHistoryNet(width=16, num_heads=2, num_latents=4, latent_layers=1,
            temporal_layers=1, dropout=0, neighborhood_layers=2, neighborhood_k=6, impactor_nodes=2).to(device)
        model.set_coordinate_scalers(scaler.mean_, scaler.scale_, scaler.mean_[:2], scaler.scale_[:2])
        preprocessor = SimpleNamespace(transform_mesh=scaler.transform)
        with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected kNN search")):
            report = prepare_model_neighbor_graphs(model, preprocessor, meshes, tmp_path)
            for mesh in meshes:
                normalized = torch.as_tensor(scaler.transform(mesh), dtype=torch.float32, device=device)
                coordinates = normalized[2:] * model.mesh_scale
                torch.testing.assert_close(model.neighbor_cache.neighbors(coordinates, 6).cpu(), expected, atol=0, rtol=0)
                nodes = torch.randn(len(mesh), 16, device=device, requires_grad=True)
                output = model._encode_neighborhoods(normalized, nodes)
                output.square().mean().backward()
                assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.neighborhood_blocks.parameters())
        assert report["unique_structural_geometries"] == report["loaded_from_disk"] == 1
        assert report["built"] == 0
        assert not any("neighbor_cache" in key for key in model.state_dict())
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_precompute_checks_first_and_last_impact_and_reuses_graph(tmp_path):
    from precompute_mesh_neighbors_1704 import main
    structure = np.arange(27, dtype=np.float32).reshape(9, 3)
    def load_mesh(run):
        return np.concatenate((np.full((2, 3), run, dtype=np.float32), structure))
    args = ["--data-root", str(tmp_path), "--impactor-nodes", "2", "--k", "4", "--designs", "0", "1"]
    with mock.patch("precompute_mesh_neighbors_1704.DataPreprocessor.load_mesh_geometry", side_effect=load_mesh):
        result = main(args)
        assert result["unique_files"] == 1 and len(result["designs"]) == 2
        assert (tmp_path / "neighbor_graphs" / "manifest_k4.json").is_file()
        with mock.patch("mesh_neighborhood.geometric_neighbors", side_effect=AssertionError("Unexpected kNN search")):
            second = main(args)
            assert second["built"] == 0 and second["loaded_from_disk"] == 1
    with mock.patch("precompute_mesh_neighbors_1704.DataPreprocessor.load_mesh_geometry",
                    side_effect=lambda run: np.full((11, 3), run, dtype=np.float32)):
        with pytest.raises(ValueError, match="differs between runs"):
            main(args)
