"""Check interpretable variance components with unbalanced design clusters."""

import numpy as np
import pytest

from analysis_change_responses import variance_decomposition


def test_variance_decomposition_weights_each_design_and_preserves_within_variation():
    curves = np.array([[[0.]], [[2.]], [[10.]]])
    result = variance_decomposition(curves, np.array(["A", "A", "B"]))
    assert result["total_variance_g2"][0] == pytest.approx(56 / 3)
    assert result["within_cluster_variance_g2"][0] == pytest.approx(2 / 3)
    assert result["between_cluster_variance_g2"][0] == pytest.approx(18)
    # A common response at each time/location never changes design variation.
    shifted = variance_decomposition(curves + 100, np.array(["A", "A", "B"]))
    for key in result:
        np.testing.assert_allclose(result[key], shifted[key])


def test_variance_decomposition_does_not_invent_variation_at_constant_locations():
    curves = np.ones((4, 2, 3))
    result = variance_decomposition(curves, np.array(["A", "A", "B", "B"]))
    for values in result.values():
        np.testing.assert_array_equal(values, np.zeros(2))
