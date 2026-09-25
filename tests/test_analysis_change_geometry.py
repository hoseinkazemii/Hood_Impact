import numpy as np
import pytest

from analysis_change_geometry import analyze_geometry, frozen_patch_coverage


def test_patch_coverage_uses_xyz_and_is_invariant_to_duplicates_and_order():
    points = np.array([[0., 0, 0], [1., 0, 100], [2., 0, 0]])
    a, mask, radii = frozen_patch_coverage(points, [[0, 0, 0]], 2)
    b, other, same_radii = frozen_patch_coverage(points[[2, 0, 0, 1]], [[0, 0, 0]], 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(mask, other)
    np.testing.assert_array_equal(radii, same_radii)
    assert mask.tolist() == [True, False, True]


def test_symmetric_changed_coverage_and_local_denominator_detect_removed_remote_feature():
    a = np.array([[0., 0, 0], [100., 0, 0]])
    b = np.array([[0., 0, 0]])
    summary, local, arrays = analyze_geometry({0: a, 1: b}, [[0., 0, 0]],
                                              {1: [0., 0], 142: [100., 0]}, neighborhood_k=1)
    row = summary.iloc[0]
    assert row.changed_fraction == pytest.approx(1/3)
    assert row.changed_coverage_fraction == 0
    assert row.nn_rms_mm == pytest.approx(np.sqrt(10000/3))
    assert local.loc[local.location_id == 1, "local_50mm_nn_rms_mm"].item() == 0
    assert local.loc[local.location_id == 142, "local_50mm_nn_rms_mm"].item() == 100
    np.testing.assert_array_equal(arrays["pair_0_1_d0_xyz"], [[100, 0, 0]])


def test_identical_clouds_have_zero_difference_without_invented_changed_coverage():
    points = np.array([[0., 0, 0], [1., 0, 0]])
    summary, local, _ = analyze_geometry({4: points, 5: points[::-1]}, [[0, 0, 0]],
                                         {1: [0, 0]}, splits={"test": [4, 5]})
    assert summary.iloc[0].nn_rms_mm == 0
    assert summary.iloc[0].changed_fraction == 0
    assert np.isnan(summary.iloc[0].changed_coverage_fraction)
    assert summary.iloc[0].involves_test
    assert np.isnan(local.iloc[0].min_changed_distance_xy_mm)
