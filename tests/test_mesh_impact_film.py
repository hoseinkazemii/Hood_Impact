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
    assert not torch.allclose(first[0], second[0])  # FiLM-conditioned query tokens.
    assert not torch.allclose(first[1], second[1])  # Queries used by cross-attention.
    # Global tokens: no radial prior on hood nodes; the two headform rows are masked.
    torch.testing.assert_close(first[4][:3, 2:], torch.zeros(3, len(mesh) - 2))
    assert torch.isneginf(first[4][:, :2]).all()
    assert not torch.allclose(first[4][3:, 2:], second[4][3:, 2:])


def predict(model, mesh, impact):
    times = torch.tensor([-.8, -.1, .7])
    return model(mesh, torch.zeros(len(mesh), dtype=torch.long), impact[None],
                 times, torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("decoder", ["temporal", "mesh_only"])
def test_impact_reaches_output_only_through_hood_node_values(decoder):
    """With node values zeroed, no geometry-free path may carry the impact."""
    impacts = (torch.tensor([-.5, .2]), torch.tensor([.7, -.8]))
    mesh = torch.randn(14, 3)
    for residual, expect_equal in ((False, True), (True, False)):
        model = make_model(decoder=decoder, pooling_query_residual=residual)
        with torch.no_grad():
            model.impact_film.affine.weight.normal_(0, .1)
            model.mesh_key_value.weight[16:].zero_()
            model.mesh_key_value.bias[16:].zero_()
        first, second = (predict(model, mesh, impact) for impact in impacts)
        if expect_equal:
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        else:
            assert not torch.allclose(first, second)


def test_excluded_headform_rows_cannot_change_the_prediction():
    mesh, impact = torch.randn(14, 3), torch.tensor([.3, -.2])
    moved = mesh.clone()
    moved[:2] += 5.0  # Only the two leading headform rows move.
    for pool, expect_equal in ((False, True), (True, False)):
        model = make_model(pool_impactor_nodes=pool)
        first, second = predict(model, mesh, impact), predict(model, moved, impact)
        if expect_equal:
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        else:
            assert not torch.allclose(first, second)
    weights = torch.cat([w for _, w in make_model().iter_pooling_attention(mesh, impact)], dim=1)
    assert (weights[..., :2] == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_shortcut_switches_add_no_parameters_and_old_configs_replay_with_both_on():
    keys = [set(make_model(pooling_query_residual=r, pool_impactor_nodes=p).state_dict())
            for r in (False, True) for p in (False, True)]
    assert all(k == keys[0] for k in keys)
    fresh = make_model()
    assert not fresh.pooling_query_residual and not fresh.pool_impactor_nodes
    film_config = {"impact_conditioning": "film", "time_encoding": "fourier"}
    for saved in ({}, film_config):
        restored = saved_model_kwargs(saved)
        assert restored["pooling_query_residual"] is True and restored["pool_impactor_nodes"] is True
    explicit = saved_model_kwargs({**film_config, "pooling_query_residual": False, "pool_impactor_nodes": False})
    assert explicit["pooling_query_residual"] is False and explicit["pool_impactor_nodes"] is False


@pytest.mark.parametrize("name", ["pooling_query_residual", "pool_impactor_nodes"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_shortcut_switches_require_booleans(name, value):
    with pytest.raises(ValueError, match=name):
        make_model(**{name: value})


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
