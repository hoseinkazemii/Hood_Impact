"""Change-token behavior, checkpoint portability and mesh isolation."""

import numpy as np
import pytest
import torch

from mesh_change_attention import MeshChangeAttentionNet


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_and_mesh():
    torch.manual_seed(21)
    model = MeshChangeAttentionNet(width=16, num_heads=2, num_latents=4,
        latent_layers=1, temporal_layers=1, dropout=0, change_anchors=3,
        change_k=4, change_layers=2, change_chunk_size=3)
    model.set_change_atlas({
        "anchors_mm": np.array([[100., 0, 0], [101., 2, 0]]),
        "reference_offsets_mm": np.zeros((2, 3)),
        "reference_distance_profiles_mm": np.array([[0., 1., 2., 3.]] * 2),
        "change_scores_mm": np.array([2., 1.]),
    })
    model.set_coordinate_scalers([0., 0., 0.], [100., 100., 10.], [0., 0.], [100., 100.])
    mesh = torch.tensor([[0., 0, 0], [.01, .01, 0], [1., 0, 0],
                         [1.01, .01, .01], [1., .02, 0], [1.02, .02, .01]])
    return model, mesh


def predict(model, mesh, impact=None):
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long),
                 torch.zeros(1, 2) if impact is None else impact,
                 torch.tensor([-.7, 0., .6]), torch.zeros(3, dtype=torch.long))


def test_atlas_is_required_and_inputs_validated():
    model, mesh = model_and_mesh()
    model.change_anchor_count.zero_()
    with pytest.raises(RuntimeError, match="training-only"):
        predict(model, mesh)
    for kwargs in ({"change_k": 0}, {"change_layers": True}, {"change_scale_mm": float("nan")},
                   {"change_anchors": -1}, {"neighborhood_layers": 2}):
        with pytest.raises(ValueError):
            MeshChangeAttentionNet(**kwargs)


def test_change_selection_is_impact_independent_remote_and_permutation_invariant():
    model, mesh = model_and_mesh()
    model.eval()
    xyz = mesh * model.mesh_scale
    anchors = model.change_anchor_xyz[:2]
    indices = model._change_indices(xyz, anchors)
    assert not ({0, 1} & set(indices.flatten().tolist()))
    expected = predict(model, mesh)
    predict(model, mesh, torch.tensor([[.9, .7]]))
    assert len(model._patch_cache) == 1
    torch.testing.assert_close(model._change_indices(xyz, anchors), indices)
    permutation = torch.tensor([5, 3, 1, 4, 0, 2])
    torch.testing.assert_close(predict(model, mesh[permutation]), expected, atol=2e-6, rtol=2e-5)
    # Remote geometric change has a differentiable path into predicted histories.
    differentiable_mesh = mesh.clone().requires_grad_(True)
    predict(model, differentiable_mesh).square().sum().backward()
    assert differentiable_mesh.grad[2:].abs().sum() > 0
    changed = mesh.clone()
    changed[2:, 2] += .1
    assert not torch.allclose(predict(model, changed), expected, atol=1e-6, rtol=1e-6)


def test_batch_isolation_strict_reload_and_mixed_precision_gradients():
    model, mesh = model_and_mesh()
    model.eval()
    second = mesh.clone()
    second[-1, -1] += .05
    expected = predict(model, mesh)
    together = model(torch.cat((mesh, second)), torch.tensor([0] * 6 + [1] * 6),
                     torch.zeros(2, 2), torch.tensor([-.7, 0., .6] * 2),
                     torch.tensor([0] * 3 + [1] * 3))
    torch.testing.assert_close(together[:3], expected, atol=1e-6, rtol=1e-5)
    reloaded, _ = model_and_mesh()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    reloaded.eval()
    torch.testing.assert_close(predict(reloaded, mesh), expected)
    assert not any("cache" in key or "graph" in key for key in model.state_dict())
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        predict(model, mesh).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_change_branch_excludes_moving_headform_and_handles_small_meshes():
    model, mesh = model_and_mesh()
    model.impactor_nodes = 2
    model.eval()
    impact = torch.zeros(2)
    condition = model.impact_embedding(impact)
    original = model._encode_changes(mesh, impact, condition)
    changed = mesh.clone()
    changed[:2] += 100
    torch.testing.assert_close(model._encode_changes(changed, impact, condition), original)
    assert len(model._patch_cache) == 1
    assert model._encode_changes(mesh[:-1], impact, condition).shape == (2, 16)
    with pytest.raises(ValueError, match="structural"):
        model._encode_changes(mesh[:3], impact, condition)


def test_patch_reference_conventions_ignore_duplicate_rows_and_match_nearest_ties():
    from scipy.spatial import cKDTree

    model, mesh = model_and_mesh()
    xyz = mesh * model.mesh_scale
    duplicate = torch.cat((xyz, xyz[2:]))
    anchors = torch.tensor([[100., .5, 0.], [101., 1., .1]])
    # Exercise tied nearest distances and coincident rows in the current mesh.
    indices = model._change_indices(duplicate, anchors)
    unique = np.unique(duplicate.numpy(), axis=0)
    distances, _ = cKDTree(unique).query(anchors.numpy(), k=model.change_k)
    nearest = cKDTree(unique).query(anchors.numpy(), k=1)[1]
    torch.testing.assert_close(duplicate[indices[:, 0]], torch.tensor(unique[nearest]))
    np.testing.assert_allclose(torch.linalg.vector_norm(duplicate[indices] - anchors[:, None], dim=-1),
                               distances, atol=1e-6)
