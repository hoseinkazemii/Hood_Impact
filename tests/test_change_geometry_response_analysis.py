import numpy as np

from analyze_change_geometry_response import (
    WITHIN_PAIRS, family_label_permutation, family_rank_residual, spatial_association,
)


def test_control_removes_common_impact_pattern():
    x = np.tile(np.arange(10.), (4, 1))
    y = x * np.arange(1, 5)[:, None] + np.arange(4)[:, None]
    result = spatial_association(x, y)
    assert result["mean_within_pair_spearman"] == 1
    assert np.isnan(result["pair_and_location_rank_residual_correlation"])


def test_two_design_families_cannot_supply_pair_ordering_evidence():
    x = np.arange(len(WITHIN_PAIRS), dtype=float)
    residual = family_rank_residual(x)
    assert residual[WITHIN_PAIRS.index((4, 5))] == 0
    assert residual[WITHIN_PAIRS.index((10, 11))] == 0
    result = family_label_permutation(x, x)
    assert np.isclose(result["observed_family_centered_rank_correlation"], 1)
    assert result["num_design_label_permutations"] == 576
    assert 0 < result["fraction_permutations_at_least_observed"] < .05


def test_constant_geometry_has_no_measurable_association():
    result = family_label_permutation(np.zeros(14), np.arange(14.))
    assert result["fraction_permutations_at_least_observed"] is None
    assert np.isnan(result["observed_family_centered_rank_correlation"])
