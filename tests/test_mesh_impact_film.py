"""Check the intended information paths of the XYZ-only, FiLM-conditioned network.

The impact location should choose *where* the pooling queries read the hood and
never supply content of its own.
"""

from unittest import mock

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
    torch.manual_seed(31)
    return MeshImpactHistoryNet(width=16, num_heads=2, num_latents=6,
                                latent_layers=1, temporal_layers=2, dropout=0,
                                neighborhood_layers=2, neighborhood_k=3, **kwargs).eval()


def predict(model, mesh, impact):
    times = torch.tensor([-.8, -.1, .7])
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long), impact[None],
                 times, torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("local_latents", [0, 3])
def test_impact_reaches_output_only_through_hood_node_values(local_latents):
    """With node values zeroed, no geometry-free path may carry the impact."""
    impacts = (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))
    mesh = torch.randn(14, 3)
    model = make_model(local_latents=local_latents)
    with torch.no_grad():
        model.impact_film.affine.weight.normal_(0, .1)
    assert not torch.allclose(predict(model, mesh, impacts[0]), predict(model, mesh, impacts[1]))
    with torch.no_grad():
        model.mesh_key_value.weight[16:].zero_()
        model.mesh_key_value.bias[16:].zero_()
    torch.testing.assert_close(predict(model, mesh, impacts[0]), predict(model, mesh, impacts[1]), atol=0, rtol=0)


def test_fixed_mesh_memory_removes_every_direct_impact_path_to_decoder():
    model = make_model()
    mesh = torch.randn(14, 3)
    fixed_memory = torch.randn(6, 16)
    with mock.patch.object(model, "_encode_mesh", return_value=fixed_memory):
        outputs = [predict(model, mesh, impact) for impact in (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))]
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_film_starts_at_identity_and_loss_trains_both_scale_and_shift():
    model = make_model()
    normalized_tokens = model.query_norm(model.latent_queries)
    condition = model.impact_embedding(torch.tensor([.3, -.2]))
    torch.testing.assert_close(model.impact_film(normalized_tokens, condition), normalized_tokens)
    mesh, times = torch.randn(14, 3), torch.tensor([-.8, -.1, .7])
    prediction = model(mesh, torch.zeros(14, dtype=torch.long), torch.tensor([[.3, -.2]]),
                       times, torch.zeros(3, dtype=torch.long))
    prediction.square().mean().backward()
    gradient = model.impact_film.affine.weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient[:16].abs().sum() > 0
    assert gradient[16:].abs().sum() > 0
