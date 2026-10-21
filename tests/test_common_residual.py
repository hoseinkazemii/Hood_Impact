import numpy as np
import pytest
import torch

from common_residual import (
    DEVIATION_CHANNELS, CommonResidualNet, deviation_features, family_references, hic_pair_weights,
    reformulation_losses, torch_hic,
)
from hic15 import batched_hic
from mesh_impact_history import MeshImpactHistoryNet

SMALL = {"width": 16, "num_heads": 2, "num_latents": 4, "latent_layers": 1, "temporal_layers": 1, "dropout": 0.0}


def geometry(ids, xyz, gap=None, panel=None):
    count = len(ids)
    normal = np.tile([0.0, 0.0, 1.0], (count, 1))
    return {"ids": np.asarray(ids), "xyz": np.asarray(xyz, float), "normal": normal,
            "gap": np.zeros(count) if gap is None else np.asarray(gap, float),
            "panel": np.zeros(count, int) if panel is None else np.asarray(panel)}


def test_torch_hic_matches_the_exact_numpy_hic():
    times = np.linspace(0, 0.025, 63)
    rng = np.random.default_rng(0)
    curves = np.abs(rng.normal(60, 40, (4, 63))) * np.exp(-((times - 0.008) / 0.004) ** 2)
    expected = batched_hic(times, curves)
    actual = torch_hic(torch.tensor(curves), torch.tensor(times)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_torch_hic_is_differentiable():
    times = torch.linspace(0, 0.025, 63, dtype=torch.float64)
    curve = (100 * torch.exp(-((times - 0.008) / 0.003) ** 2)).requires_grad_(True)
    torch_hic(curve[None], times).sum().backward()
    assert torch.isfinite(curve.grad).all() and curve.grad.abs().sum() > 0


def test_family_references_group_by_node_ids_and_use_training_designs_only():
    geometries = {0: geometry([1, 2], [[0, 0, 0], [1, 0, 0]]), 1: geometry([1, 2], [[0, 0, 2], [1, 0, 2]]),
                  2: geometry([1, 2], [[0, 0, 100], [1, 0, 100]]), 3: geometry([5, 6, 7], np.zeros((3, 3)))}
    family_of, references = family_references(geometries, train_designs=[0, 1, 3])
    assert family_of == {0: 0, 1: 0, 2: 0, 3: 1}
    np.testing.assert_allclose(references[0]["xyz"][:, 2], [1.0, 1.0])  # design 2 (held out) excluded
    with pytest.raises(ValueError, match="no training design"):
        family_references(geometries, train_designs=[0])


def test_deviation_features_are_zero_for_the_reference_and_scale_the_edit():
    base = geometry([1, 2, 3], np.zeros((3, 3)), gap=[5, 5, 5], panel=[0, 1, 3])
    edited = geometry([1, 2, 3], [[0, 0, 0], [0, 0, 20.0], [0, 0, 0]], gap=[5, 1, 5], panel=[0, 1, 3])
    _, (reference,) = family_references({0: base}, [0])
    np.testing.assert_allclose(deviation_features(base, reference)[:, :8], 0)
    features = deviation_features(edited, reference)
    assert features.shape == (3, len(DEVIATION_CHANNELS))
    np.testing.assert_allclose(features[1, :4], [0, 0, 2.0, 2.0])     # dz / 10 mm and |dx| / 10 mm
    np.testing.assert_allclose(features[1, 7], -0.4)                   # gap change / 10 mm
    np.testing.assert_allclose(features[:, 8:], np.eye(4)[[0, 1, 3]])  # panel one-hot


def loss_inputs():
    torch.manual_seed(0)
    target = 50 + 10 * torch.randn(4, 63)
    mean = target.mean(0, keepdim=True).expand(4, -1).clone()
    group = torch.tensor([0, 0, 0, 1])
    return target, mean, group, {"mu": 50.0, "sigma": 40.0, "residual": 10.0}


def exact_outputs(target, mean, scales):
    return (mean - scales["mu"]) / scales["sigma"], (target - mean) / scales["residual"]


def test_reformulation_losses_vanish_for_a_perfect_prediction():
    target, mean, group, scales = loss_inputs()
    common, residual = exact_outputs(target, mean, scales)
    weights = {"abs": 1, "common": 1, "delta": 1, "pair": 1, "hic": 1}
    total, terms, prediction = reformulation_losses(common, residual, target, mean, group, torch.ones(4), scales,
                                                    weights, torch.linspace(0, 0.025, 63))
    assert total.item() == pytest.approx(0, abs=1e-8)
    torch.testing.assert_close(prediction, target)


def test_pair_loss_penalizes_identical_curves_and_ignores_a_shared_error():
    target, mean, group, scales = loss_inputs()
    common, residual = exact_outputs(target, mean, scales)
    weights = {"pair": 1.0}
    times = torch.linspace(0, 0.025, 63)
    # Shifting every design at a location by the same curve leaves design differences intact.
    _, shared, _ = reformulation_losses(common, residual + torch.randn(1, 63), target, mean, group, torch.ones(4),
                                        scales, weights, times)
    assert shared["pair"] == pytest.approx(0, abs=1e-10)
    # Collapsed: the same residual for every design.
    collapsed = torch.zeros_like(residual)
    _, terms, _ = reformulation_losses(common, collapsed, target, mean, group, torch.ones(4), scales, weights, times)
    true = residual[:3]
    expected = np.mean([((true[i] - true[j]) ** 2).mean().item() for i, j in ((0, 1), (0, 2), (1, 2))])
    assert terms["pair"] == pytest.approx(expected, rel=1e-6)


def test_hic_pair_weights_have_unit_mean_before_clipping():
    hic = np.array([[100, 200, 300], [100, 260, 900], [100, 140, 300]], float)
    weights = hic_pair_weights(hic, [0, 1, 2], maximum=np.inf)
    assert weights[0] == 0 and weights.mean() == pytest.approx(1.0)
    assert hic_pair_weights(hic, [0, 1, 2], maximum=1.5).max() == pytest.approx(1.5)


def test_mesh_model_extra_node_features_default_is_unchanged():
    torch.manual_seed(0)
    plain = MeshImpactHistoryNet(**SMALL)
    torch.manual_seed(0)
    explicit = MeshImpactHistoryNet(**SMALL, node_extra_features=0)
    assert plain.state_dict().keys() == explicit.state_dict().keys()
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], value)
    with pytest.raises(ValueError, match="node_extra_features"):
        MeshImpactHistoryNet(**SMALL, node_extra_features=-1)


def test_extra_node_features_reach_the_output_and_shapes_are_checked():
    torch.manual_seed(0)
    model = MeshImpactHistoryNet(**SMALL, node_extra_features=2).eval()
    mesh = torch.randn(30, 5)
    args = (torch.zeros(30, dtype=torch.long), torch.zeros(1, 2), torch.linspace(-1, 1, 7), torch.zeros(7, dtype=torch.long))
    with torch.no_grad():
        base = model(mesh, *args)
        changed = mesh.clone()
        changed[:, 3:] += 1.0
        assert not torch.allclose(model(changed, *args), base)
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        model(mesh[:, :3], *args)


def test_common_residual_net_returns_both_paths():
    torch.manual_seed(0)
    model = CommonResidualNet(SMALL, common_width=16).eval()
    channels = 3 + len(DEVIATION_CHANNELS)
    mesh = torch.randn(50, channels)
    mesh_batch = torch.repeat_interleave(torch.arange(2), torch.tensor([20, 30]))
    with torch.no_grad():
        common, residual = model(mesh, mesh_batch, torch.zeros(2, 2), torch.linspace(-1, 1, 9),
                                 torch.linspace(0, 0.025, 9))
    assert common.shape == residual.shape == (2, 9)
    torch.testing.assert_close(common[0], common[1])  # same impact point -> same common curve
