import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "abaqus_scripts"))

import design_difference_data as dd  # noqa: E402
from analyze_solver_factorial import energy_share_removed  # noqa: E402
from edit_operator import EDIT_CHANNELS, NODE_FEATURES, EditOperator  # noqa: E402
from make_solver_factorial_decks import (  # noqa: E402
    CONTACT_OFFSET_MM, fixed_increment_lines, gap_start_lift, sphere_penetration,
)
from oracle_delta_features import geometry_features  # noqa: E402


# --------------------------------------------------------------------------- metric
def test_difference_r2_is_zero_for_a_collapsed_predictor_and_one_for_truth():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(3, 142, 63))
    truth -= truth.mean()
    assert dd.difference_r2(np.zeros_like(truth), truth) == pytest.approx(0.0, abs=1e-12)
    assert dd.difference_r2(truth, truth) == pytest.approx(1.0)
    assert dd.difference_r2(-truth, truth) == pytest.approx(-3.0)


def test_pair_geometry_requires_corresponding_nodes():
    base = {"ids": np.arange(4), "xyz": np.zeros((4, 3)), "gap": np.ones(4), "normal": np.zeros((4, 3)),
            "area": np.ones(4), "panel": np.zeros(4, int)}
    edited = {**base, "xyz": base["xyz"].copy()}
    edited["xyz"][2] = [0.0, 0.0, 3.0]
    pair = dd.pair_geometry(base, edited)
    assert pair["moved"].tolist() == [False, False, True, False]
    np.testing.assert_allclose(pair["dx"][2], [0, 0, 3])
    with pytest.raises(ValueError, match="different mesh families"):
        dd.pair_geometry(base, {**edited, "ids": np.arange(1, 5)})


def test_distance_bands():
    assert dd.distance_band([0, 99.9, 100, 249, 250, 1e4]).tolist() == ["near", "near", "mid", "mid", "far", "far"]


# --------------------------------------------------------------------------- solver factorial decks
def deck_lines(terminator="\n"):
    return [line + terminator for line in
            ("*STEP, NAME=\"Explicit Dynamic Step1\", NLGEOM=YES", "*DYNAMIC, EXPLICIT, SCALE FACTOR=1.",
             " , 0.025", "*CONTACT, OP=NEW")]


@pytest.mark.parametrize("terminator", ["\n", "\r\n"])
def test_fixed_increment_rewrites_only_the_dynamic_card(terminator):
    replaced = fixed_increment_lines(deck_lines(terminator), 4.5e-7)
    assert replaced == {1: "*DYNAMIC, EXPLICIT, DIRECT USER CONTROL" + terminator,
                        2: "4.5e-07, 0.025" + terminator}


def test_fixed_increment_rejects_bad_decks():
    lines = deck_lines()
    with pytest.raises(ValueError, match="exactly one"):
        fixed_increment_lines(lines + lines, 4.5e-7)
    with pytest.raises(ValueError, match="smaller than the step period"):
        fixed_increment_lines(lines, 0.05)
    with pytest.raises(ValueError, match="finite and positive"):
        fixed_increment_lines(lines, -1.0)


def plane(slope=0.0):
    x, y = np.meshgrid(np.linspace(-150, 150, 121), np.linspace(-150, 150, 121))
    return np.column_stack((x.ravel(), y.ravel(), slope * x.ravel()))


@pytest.mark.parametrize("slope", [0.0, 0.3])
def test_gap_start_lift_reaches_the_requested_clearance(slope):
    hood = plane(slope)
    centre = np.array([0.0, 0.0, 82.5 - 1.6])  # 1.6 mm sphere-node overclosure on the flat plane
    target = -(CONTACT_OFFSET_MM + 1.0)
    lift, penetration = gap_start_lift(hood, centre, target)
    assert penetration == pytest.approx(target, abs=1e-3)
    assert sphere_penetration(hood, centre + [0, 0, lift]) == pytest.approx(target, abs=1e-3)
    if slope == 0.0:
        assert lift == pytest.approx(1.6 - target, abs=1e-3)
    else:  # the closest node sits off the vertical, so more lift is needed
        assert lift > 1.6 - target


# --------------------------------------------------------------------------- delta features
def test_geometry_features_follow_the_edit_direction():
    rng = np.random.default_rng(1)
    edit = {"x": rng.uniform(-100, 100, (30, 3)), "dx": rng.normal(0, 2, (30, 3)), "dgap": rng.normal(0, 1, 30),
            "normal": np.tile([0.0, 0.0, 1.0], (30, 1)), "panel": rng.integers(0, 4, 30),
            "distance_xy": np.array([10.0, 300.0]), "distance_3d": np.array([12.0, 310.0])}
    xy = np.array([[0.0, 0.0], [250.0, 0.0]])
    forward, names = geometry_features(edit, xy, 1)
    backward, _ = geometry_features(edit, xy, -1)
    for i, name in enumerate(names):
        if name.startswith(("dz_sum", "dn_sum")):
            np.testing.assert_allclose(backward[:, i], -forward[:, i])
        elif name.startswith(("abs_sum", "max_abs", "n_moved", "log_distance")):
            np.testing.assert_allclose(backward[:, i], forward[:, i])
    close = [i for i, n in enumerate(names) if n.startswith("gap_close")]
    opened = [i for i, n in enumerate(names) if n.startswith("gap_open")]
    np.testing.assert_allclose(backward[:, close], -forward[:, opened])


# --------------------------------------------------------------------------- edit operator
def operator_inputs(batch=5, nodes=12, context_dim=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    relative = 200 * torch.randn(batch, nodes, 3, generator=generator)
    features = torch.randn(batch, nodes, NODE_FEATURES, generator=generator)
    edit = torch.randn(batch, nodes, EDIT_CHANNELS, generator=generator)
    mask = torch.ones(batch, nodes, dtype=torch.bool)
    mask[:, -3:] = False
    edit[:, -3:] = 0.0  # padding carries no edit, as in PairData
    context = torch.randn(batch, context_dim, generator=generator)
    rho = torch.randn(batch, generator=generator)
    return relative, features, edit, mask, context, rho


@pytest.mark.parametrize("operator", ["linear", "attention"])
def test_edit_operator_is_antisymmetric_and_zero_for_no_edit(operator):
    torch.manual_seed(0)
    model = EditOperator(num_times=63, context_dim=4, operator=operator, nuisance="dt", width=32,
                         basis_size=8, num_heads=4).eval()
    relative, features, edit, mask, context, rho = operator_inputs()
    with torch.no_grad():
        geometric, nuisance = model(relative, features, edit, mask, context, rho)
        flipped_geometric, flipped_nuisance = model(relative, features, -edit, mask, context, -rho)
        zero_geometric, zero_nuisance = model(relative, features, torch.zeros_like(edit), mask, context,
                                              torch.zeros_like(rho))
    assert geometric.abs().max() > 0 and nuisance.abs().max() > 0
    torch.testing.assert_close(flipped_geometric, -geometric)
    torch.testing.assert_close(flipped_nuisance, -nuisance)
    assert zero_geometric.abs().max() == 0 and zero_nuisance.abs().max() == 0


@pytest.mark.parametrize("operator", ["linear", "attention"])
def test_edit_operator_ignores_padding(operator):
    torch.manual_seed(0)
    model = EditOperator(num_times=63, context_dim=4, operator=operator, nuisance="none", width=32,
                         basis_size=8, num_heads=4).eval()
    relative, features, edit, mask, context, rho = operator_inputs()
    changed_relative, changed_features = relative.clone(), features.clone()
    changed_relative[:, -3:] += 500.0
    changed_features[:, -3:] = torch.randn_like(changed_features[:, -3:])
    with torch.no_grad():
        reference, _ = model(relative, features, edit, mask, context, rho)
        padded, _ = model(changed_relative, changed_features, edit, mask, context, rho)
    torch.testing.assert_close(padded, reference)


def test_edit_operator_rejects_an_empty_model():
    with pytest.raises(ValueError, match="needs a geometric operator"):
        EditOperator(num_times=63, context_dim=4, operator="none", nuisance="none")


# --------------------------------------------------------------------------- factorial analysis
def test_energy_share_removed():
    assert energy_share_removed([2.0, 2.0], [1.0, 1.0]) == pytest.approx(0.75)
    assert energy_share_removed([2.0, np.nan], [2.0, 1.0]) == pytest.approx(0.0)
    assert np.isnan(energy_share_removed([np.nan], [1.0]))
