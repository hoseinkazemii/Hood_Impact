"""Check the intended information paths of XYZ-only, FiLM-conditioned models."""

from unittest import mock

import pytest
import torch

from mesh_impact_history import MeshImpactHistoryNet, saved_model_kwargs


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
                                neighborhood_layers=2, neighborhood_k=3,
                                impactor_nodes=2, **kwargs).eval()


def test_impact_changes_pooling_queries_and_prior_but_not_encoded_node_keys_or_values():
    model = make_model()
    # Exercise learned modulation beyond the identity initialization.
    with torch.no_grad():
        model.impact_film.affine.weight.normal_(0, .1)
    mesh = torch.randn(14, 3)
    impacts = (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))
    first, second = [model._mesh_pooling_inputs(mesh, p, model.impact_embedding(p)) for p in impacts]
    for index in (2, 3):  # All node keys/values, after both neighborhood layers.
        torch.testing.assert_close(first[index], second[index], atol=0, rtol=0)
    assert not torch.allclose(first[0], second[0])  # FiLM-conditioned residual tokens.
    assert not torch.allclose(first[1], second[1])  # Queries used by cross-attention.
    torch.testing.assert_close(first[4][:3], torch.zeros(3, len(mesh)))
    assert not torch.allclose(first[4][3:], second[4][3:])


@pytest.mark.parametrize("decoder", ["temporal", "mesh_only"])
def test_fixed_mesh_memory_removes_every_direct_impact_path_to_decoder(decoder):
    model = make_model(decoder=decoder)
    mesh = torch.randn(14, 3)
    times = torch.tensor([-.8, -.1, .7])
    fixed_memory = torch.randn(6, 16)
    with mock.patch.object(model, "_encode_mesh", return_value=fixed_memory):
        outputs = [model(mesh, torch.zeros(14, dtype=torch.long), impact[None],
                         times, torch.zeros(3, dtype=torch.long))
                   for impact in (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))]
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


def test_historical_configuration_keeps_six_features_and_original_state_keys():
    model = make_model(**saved_model_kwargs({}))
    mesh, impact = torch.randn(7, 3), torch.tensor([.2, -.3])
    features = model.node_features(mesh, impact)
    assert features.shape == (7, 6)
    torch.testing.assert_close(features[:, :3], mesh)
    torch.testing.assert_close(features[:, 3:5], mesh[:, :2] - impact)
    assert not any("film" in key for key in model.state_dict())
    assert saved_model_kwargs({"impact_conditioning": "film"})["impact_conditioning"] == "film"


@pytest.mark.parametrize("mode", ["additive", "", None, True, ["film"]])
def test_invalid_conditioning_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="impact_conditioning"):
        make_model(impact_conditioning=mode)
