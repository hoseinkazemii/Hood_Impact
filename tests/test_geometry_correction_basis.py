"""Geometry-only training fit, sensitivity, invariance, and span diagnostics."""
import numpy as np
import pytest

from geometry_correction_basis import GeometryCorrectionBasis


def geometry_series():
    points = np.array([[0., 0., 0.], [20., 0., 0.], [40., 0., 0.],
                       [0., 20., 0.], [20., 20., 0.]])
    a = points.copy()
    b = points.copy()
    b[0, 2] += 4.
    c = points.copy()
    c[-1, 1] += 3.
    return [a, b, c]


def fit(geometries=None, **kwargs):
    geometries = geometry_series() if geometries is None else geometries
    return GeometryCorrectionBasis.fit(geometries, list(range(len(geometries))),
                                       profile_k=3, anchor_spacing_mm=5., **kwargs)


def test_retains_all_resolved_modes_centers_and_reconstructs_training_geometry():
    geometries = geometry_series()
    basis = fit(geometries)
    assert basis.coefficient_width == 2
    assert basis.num_features == basis.num_anchors * 6
    assert basis.config["fit_split"] == "train"
    coefficients = []
    for index, points in enumerate(geometries):
        coefficient, diagnostic = basis.transform(points)
        coefficients.append(coefficient)
        np.testing.assert_allclose(coefficient, basis.training_coefficients[index], atol=1e-7)
        assert diagnostic["relative_reconstruction_error"] < 1e-12
        assert diagnostic["reconstruction_rms_mm"] < 1e-12
    np.testing.assert_allclose(np.mean(coefficients, axis=0), 0., atol=1e-7)
    np.testing.assert_allclose(basis.components @ basis.components.T, np.eye(2), atol=1e-12)


def test_cloud_order_duplicates_and_design_order_cannot_change_fit():
    rng = np.random.default_rng(33)
    original = geometry_series()
    expected = fit(original)
    shuffled = [np.vstack([points[rng.permutation(len(points))], points[:2]]) for points in original]
    actual = GeometryCorrectionBasis.fit(shuffled[::-1], [2, 1, 0], profile_k=3, anchor_spacing_mm=5.)
    for field in ("anchors_mm", "anchor_labels", "components", "feature_mean_mm", "feature_scale_mm",
                  "coefficient_scale", "training_coefficients"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=field)
    assert actual.config == expected.config
    assert actual.training_design_ids == [0, 1, 2]
    c0, _ = actual.transform(original[0])
    c1, _ = actual.transform(shuffled[0])
    np.testing.assert_array_equal(c0, c1)


def test_part_labels_keep_inner_edits_separate_from_unchanged_outer():
    outer = np.array([[0., 0., 2.], [20., 0., 2.], [40., 0., 2.]])
    inner = outer - [0., 0., 2.]
    moved_inner = inner.copy()
    moved_inner[1, 2] += 20.
    labels = np.array(["outer"] * 3 + ["inner"] * 3)
    geometries = [np.vstack([outer, inner]), np.vstack([outer, moved_inner])]
    basis = GeometryCorrectionBasis.fit(geometries, [4, 5], [labels, labels],
                                        profile_k=2, anchor_spacing_mm=10.)
    assert basis.config["part_aware"]
    assert set(basis.anchor_labels) == {"outer", "inner"}
    descriptor_a = basis._descriptor({"outer": outer, "inner": inner}, basis.anchors_mm,
                                     basis.anchor_labels, 2).reshape(basis.num_anchors, 5)
    descriptor_b = basis._descriptor({"outer": outer, "inner": moved_inner}, basis.anchors_mm,
                                     basis.anchor_labels, 2).reshape(basis.num_anchors, 5)
    np.testing.assert_array_equal(descriptor_a[basis.anchor_labels == "outer"],
                                  descriptor_b[basis.anchor_labels == "outer"])
    assert np.any(descriptor_a[basis.anchor_labels == "inner"] != descriptor_b[basis.anchor_labels == "inner"])
    c0, _ = basis.transform(geometries[0], labels)
    c1, _ = basis.transform(geometries[1], labels)
    assert np.linalg.norm(c1 - c0) > 1.
    with pytest.raises(ValueError, match="match fitted groups"):
        basis.transform(geometries[0])


def test_sparse_edit_is_not_discarded_behind_larger_family_change():
    spine = np.column_stack([np.arange(100.) * 5., np.zeros(100), np.zeros(100)])
    geometries = []
    for i in range(4):
        points = spine.copy()
        points[:, 2] = 100. if i >= 2 else 0.
        points[50, 1] += .5 * (i % 2)
        geometries.append(points)
    basis = GeometryCorrectionBasis.fit(geometries, [0, 1, 2, 3], profile_k=4,
                                        anchor_spacing_mm=10., noise_floor_mm=.1)
    # More than a single broad family mode survives, and both sibling edits
    # remain distinguishable even though only one of 100 nodes was changed.
    assert basis.coefficient_width >= 2
    for a, b in ((0, 1), (2, 3)):
        ca, da = basis.transform(geometries[a])
        cb, db = basis.transform(geometries[b])
        assert np.linalg.norm(cb - ca) > .1
        assert da["relative_reconstruction_error"] < 1e-12
        assert db["relative_reconstruction_error"] < 1e-12


def test_held_out_projection_cannot_mutate_fit_and_reports_unseen_direction():
    a = np.array([[0., 0., 0.], [50., 0., 0.]])
    b = a.copy()
    b[0, 2] += 4.
    basis = fit([a, b])
    saved_arrays = {field: getattr(basis, field).copy() for field in (
        "anchors_mm", "components", "feature_mean_mm", "feature_scale_mm", "coefficient_scale")}
    unseen = a.copy()
    unseen[1, 1] += 8.
    coefficient, diagnostics = basis.transform(unseen)
    assert coefficient.shape == (1,)
    assert diagnostics["out_of_span_energy_fraction"] > .5
    assert diagnostics["reconstruction_rms_mm"] > 1.
    assert diagnostics["reconstruction_max_mm"] > 5.
    for field, array in saved_arrays.items():
        np.testing.assert_array_equal(getattr(basis, field), array)
    # Editing inputs after fit also cannot alter the frozen artifact.
    a[:] = -100.
    b[:] = 100.
    for field, array in saved_arrays.items():
        np.testing.assert_array_equal(getattr(basis, field), array)


def test_roundtrip_is_pickle_free_and_preserves_projection(tmp_path):
    basis = fit()
    path = tmp_path / "geometry_basis.npz"
    basis.save(path)
    with np.load(path, allow_pickle=False) as artifact:
        assert artifact["metadata_json"].dtype.kind == "U"
        assert artifact["anchor_labels"].dtype.kind == "U"
    loaded = GeometryCorrectionBasis.load(path)
    assert loaded.config == basis.config
    assert loaded.training_design_ids == basis.training_design_ids
    for geometry in geometry_series():
        expected_c, expected_d = basis.transform(geometry)
        actual_c, actual_d = loaded.transform(geometry)
        np.testing.assert_array_equal(actual_c, expected_c)
        assert actual_d == expected_d


def test_noise_floor_does_not_whiten_tiny_geometry_noise():
    a = np.array([[0., 0., 0.], [20., 0., 0.]])
    b = a.copy()
    b[0, 2] = 1e-7
    basis = fit([a, b], noise_floor_mm=.1)
    assert basis.feature_scale_mm.min() >= .1
    assert np.all(basis.coefficient_scale >= basis.config["mode_coefficient_noise_floor"])
    c, _ = basis.transform(b)
    assert np.linalg.norm(c) < 1e-4


def test_explicit_coefficient_scale_floor_is_respected():
    basis = fit(coefficient_scale_floor=100.)
    assert np.all(basis.coefficient_scale >= 100.)


def test_small_cloud_profiles_pad_in_physical_units_and_signed_offsets():
    a, b = np.array([[0., 0., 0.]]), np.array([[0., 0., 4.]])
    basis = GeometryCorrectionBasis.fit([a, b], [3, 9], anchor_spacing_mm=10., profile_k=4)
    np.testing.assert_array_equal(basis.anchors_mm, b)
    # b is nearer to the center of the occupied voxel than a.
    descriptor = basis._descriptor({"all": a}, basis.anchors_mm, basis.anchor_labels, 4)
    np.testing.assert_allclose(descriptor, [0., 0., -4., 4., 4., 4., 4.])
    np.testing.assert_allclose(basis.feature_mean_mm, [0., 0., -2., 2., 2., 2., 2.])


def test_exact_nearest_ties_have_lexicographic_signed_offset():
    a = np.array([[-1., 0., 0.], [1., 0., 0.]])
    feature = GeometryCorrectionBasis._descriptor({"all": a}, np.array([[0., 0., 0.]]),
                                                 np.array(["all"]), 1)
    np.testing.assert_array_equal(feature, [-1., 0., 0., 1.])


@pytest.mark.parametrize("kwargs", [
    {"anchor_spacing_mm": 0}, {"anchor_spacing_mm": True}, {"anchor_spacing_mm": np.nan},
    {"noise_floor_mm": -1}, {"noise_floor_mm": np.inf}, {"profile_k": 0},
    {"profile_k": True}, {"profile_k": 1.5}, {"rank_rtol": 0}, {"rank_rtol": 1},
    {"coefficient_scale_floor": 0},
])
def test_invalid_settings_fail(kwargs):
    settings = dict(anchor_spacing_mm=5., profile_k=3)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        GeometryCorrectionBasis.fit(geometry_series(), [0, 1, 2], **settings)


@pytest.mark.parametrize("geometries, ids, match", [
    ([], [], "at least two"),
    ([np.zeros((1, 3))], [0], "at least two"),
    ([np.zeros((1, 3)), np.ones((1, 3))], [0, 0], "design IDs"),
    ([np.zeros((1, 3)), np.ones((1, 3))], [0, True], "design IDs"),
    ([np.zeros((1, 3)), np.ones((1, 3))], [0, "1"], "design IDs"),
    ([np.zeros((1, 3)), np.ones((1, 2))], [0, 1], "shape"),
    ([np.zeros((1, 3)), np.full((1, 3), np.nan)], [0, 1], "nonfinite"),
    ([np.zeros((1, 3)), np.empty((0, 3))], [0, 1], "shape"),
])
def test_invalid_inputs_fail(geometries, ids, match):
    with pytest.raises(ValueError, match=match):
        GeometryCorrectionBasis.fit(geometries, ids)


def test_identical_geometries_have_clear_error_even_for_nonbinary_decimal_profiles():
    a = np.array([[.1, .2, .3], [40.25, 0., .15]])
    with pytest.raises(ValueError, match="no numerically resolved"):
        GeometryCorrectionBasis.fit([a, a.copy(), a.copy()], [0, 1, 2])


def test_incompatible_part_groups_fail_before_fit():
    clouds = [np.zeros((2, 3)), np.ones((2, 3))]
    with pytest.raises(ValueError, match="same part label groups"):
        GeometryCorrectionBasis.fit(clouds, [0, 1], [np.array(["inner", "outer"]), np.array(["inner", "inner"])])
    with pytest.raises(ValueError, match="one-dimensional"):
        GeometryCorrectionBasis.fit(clouds, [0, 1], [np.array([1]), np.array([1, 1])])
