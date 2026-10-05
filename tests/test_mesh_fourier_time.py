"""Physical Fourier phases, gradients, and historical checkpoint compatibility."""
import math

import numpy as np
import pytest
import torch

from mesh_impact_history import MeshImpactHistoryNet, saved_model_kwargs


def small_model(**options):
    return MeshImpactHistoryNet(width=16, num_heads=2, num_latents=4,
        latent_layers=1, temporal_layers=1, dropout=0, **options)


@pytest.mark.parametrize("stride", [16, 4, 1])
def test_fourier_features_match_physical_sine_cosine_on_sampled_grids(stride):
    model = small_model()
    times = torch.linspace(0, .025, 1000)[::stride]
    mean, scale = float(times.mean()), float(times.std(correction=0))
    model.set_time_scaler(mean, scale)
    features = model.time_features((times-mean)/scale)
    frequencies = torch.tensor([20., 40., 80., 160., 320., 640.])
    phase = times[:, None] * (2 * math.pi * frequencies)
    expected = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
    assert features.shape == (len(times), 12)
    torch.testing.assert_close(model.time_frequencies_hz, frequencies)
    torch.testing.assert_close(features, expected, atol=2e-5, rtol=0)
    # Every consecutive pair represents a point on the unit circle.
    torch.testing.assert_close(features.reshape(len(times), 6, 2).square().sum(-1), torch.ones(len(times), 6))
    torch.testing.assert_close(features[0], torch.tensor([0., 1.] * 6), atol=3e-6, rtol=0)
    assert model.time_embedding[0].in_features == 12


def test_physical_phases_are_independent_of_the_choice_of_time_scaler():
    times = torch.tensor([0., .004, .008, .015, .024])
    a, b = small_model(), small_model()
    a.set_time_scaler(.012, .007)
    b.set_time_scaler(.021, .003)
    fa = a.time_features((times-.012)/.007)
    fb = b.time_features((times-.021)/.003)
    torch.testing.assert_close(fa, fb, atol=2e-5, rtol=0)


def test_feature_derivative_includes_the_physical_time_scale():
    model = small_model(fourier_num_frequencies=1, fourier_min_frequency_hz=40., fourier_max_frequency_hz=40.)
    model.set_time_scaler(.00625, .007)
    normalized = torch.tensor([0.], requires_grad=True)
    # At t=6.25 ms, the 40 Hz phase is pi/2: d cos / ds = -2*pi*40*time_scale.
    features = model.time_features(normalized)
    derivative = torch.autograd.grad(features[0, 1], normalized)[0]
    torch.testing.assert_close(derivative, torch.tensor([-2*math.pi*40*.007]))


@pytest.mark.parametrize("impact_conditioning", ["film", "legacy_additive"])
def test_historical_configs_without_time_encoding_replay_the_five_feature_model(impact_conditioning):
    kwargs = dict(width=16, num_heads=2, num_latents=4, latent_layers=1,
                  temporal_layers=1, dropout=0, impact_conditioning=impact_conditioning)
    old = MeshImpactHistoryNet(time_encoding="legacy_five", **kwargs).eval()
    restored = MeshImpactHistoryNet(**saved_model_kwargs(kwargs)).eval()
    restored.load_state_dict(old.state_dict(), strict=True)
    assert restored.time_encoding == "legacy_five"
    assert restored.time_embedding[0].in_features == 5
    assert not any(key.startswith(("time_mean", "time_scale", "time_frequencies")) for key in restored.state_dict())
    t = torch.tensor([-1., -.5, 0., .5, 1.])
    expected = torch.stack((t, t.square(), t.pow(3), t.tanh(), torch.exp(-t.square())), -1)
    torch.testing.assert_close(restored.time_features(t), expected, atol=0, rtol=0)
    mesh, impact = torch.randn(8, 3), torch.randn(1, 2)
    membership = torch.zeros(8, dtype=torch.long)
    time_batch = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(old(mesh, membership, impact, t, time_batch),
                                   restored(mesh, membership, impact, t, time_batch), atol=0, rtol=0)


@pytest.mark.parametrize("options", [
    {"time_encoding":"unknown"}, {"time_encoding":None}, {"time_encoding":["fourier"]},
    {"fourier_num_frequencies":0}, {"fourier_num_frequencies":True},
    {"fourier_min_frequency_hz":0}, {"fourier_min_frequency_hz":float("nan")},
    {"fourier_max_frequency_hz":float("inf")}, {"fourier_max_frequency_hz":10},
    {"fourier_num_frequencies":1},
])
def test_invalid_fourier_settings_are_rejected(options):
    with pytest.raises(ValueError):
        small_model(**options)


@pytest.mark.parametrize("mean,scale", [(float("nan"),1), (0,0), (0,-1), (0,float("inf"))])
def test_invalid_time_scaler_is_rejected(mean, scale):
    with pytest.raises(ValueError):
        small_model().set_time_scaler(mean,scale)


def test_custom_frequency_band_and_scaler_survive_strict_state_reload():
    options = dict(fourier_num_frequencies=3, fourier_min_frequency_hz=25., fourier_max_frequency_hz=400.)
    original = small_model(**options).eval()
    original.set_time_scaler(.0124, .0073)
    restored = small_model(**options).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(restored.time_frequencies_hz, torch.tensor([25.,100.,400.]))
    times = torch.linspace(-1.7,1.7,63)
    torch.testing.assert_close(original.time_features(times),restored.time_features(times),atol=0,rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("stride", [16, 4])
def test_fourier_channels_and_decoder_receive_finite_training_gradients(device, stride):
    torch.manual_seed(4)
    model = small_model(neighborhood_layers=2, neighborhood_k=4).to(device).train()
    times = torch.linspace(0,.025,1000,device=device)[::stride]
    mean,scale = float(times.mean()),float(times.std(correction=0))
    model.set_time_scaler(mean,scale)
    mesh = torch.randn(10,3,device=device)
    prediction = model(mesh,torch.zeros(10,dtype=torch.long,device=device),
        torch.tensor([[.1,.2]],device=device),(times-mean)/scale,
        torch.zeros(len(times),dtype=torch.long,device=device))
    target = torch.sin(2*math.pi*320*times)
    (prediction-target).square().mean().backward()
    assert prediction.shape == (len(times),) and torch.isfinite(prediction).all()
    weights = model.time_embedding[0].weight
    assert torch.isfinite(weights.grad).all()
    assert torch.all(weights.grad.abs().sum(0) > 0)  # Every sine/cosine input is trainable downstream.
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
