"""Change-anchor behavior with unordered, unequal training-only point clouds."""

import numpy as np
import pytest

from mesh_change_atlas import fit_change_atlas, group_designs_by_geometry


def fit(meshes, **kwargs):
    settings = dict(num_anchors=3, candidate_spacing_mm=.1,
                    anchor_separation_mm=10., impactor_nodes=0, neighborhood_k=4)
    settings.update(kwargs)
    return fit_change_atlas(meshes, **settings)


def assert_same_atlas(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key], err_msg=key)
        else:
            assert value == right[key], key


def test_far_away_nudge_and_removed_feature_are_found_with_unequal_mesh_counts():
    unchanged = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    a = np.vstack([unchanged, [100., 0., 0.], [200., 0., 0.]])
    b = np.vstack([unchanged, [104., 0., 0.]])
    atlas = fit({4: a, 5: b}, num_anchors=10)
    assert atlas["train_design_ids"] == [4, 5]
    assert atlas["num_anchors"] == 3  # no unchanged padding anchors
    assert set(atlas["anchors_mm"][:, 0]) == {100., 104., 200.}
    assert np.all(atlas["change_scores_mm"] > 0)
    assert atlas["change_scores_mm"][0] == 48.  # missing feature dominates
    assert atlas["reference_offsets_mm"].shape == (3, 3)
    assert atlas["reference_distance_profiles_mm"].shape == (3, 4)


def test_atlas_is_invariant_to_structure_node_order_dictionary_order_and_duplicates():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(70, 3)) * 20.
    b = np.vstack([a[:50] + [0., 0., 2.], a[50:]])
    expected = fit({0: a, 8: b}, candidate_spacing_mm=5.)
    actual = fit({8: b[rng.permutation(len(b))],
                  0: np.vstack([a[rng.permutation(len(a))], a[:3]])}, candidate_spacing_mm=5.)
    assert_same_atlas(expected, actual)


def test_only_structural_changes_count_and_headform_motion_cannot_move_anchors():
    structure = np.array([[0., 0., 0.], [20., 0., 0.], [30., 0., 0.]])
    first = np.vstack([[1000., 0., 0.], structure])
    moved_headform = np.vstack([[-1000., 0., 0.], structure])
    with pytest.raises(ValueError, match="no measurable geometry changes"):
        fit({0: first, 1: moved_headform}, impactor_nodes=1)
    changed_structure = structure.copy()
    changed_structure[-1, 2] = 2.
    atlas = fit({0: first, 1: np.vstack([[-1000., 0., 0.], changed_structure])}, impactor_nodes=1)
    reference = fit({0: structure, 1: changed_structure})
    np.testing.assert_array_equal(atlas["anchors_mm"], reference["anchors_mm"])
    np.testing.assert_array_equal(atlas["change_scores_mm"], reference["change_scores_mm"])


def test_profiles_and_offsets_have_physical_units_and_pad_short_meshes():
    atlas = fit({3: np.array([[0., 0., 0.]]), 9: np.array([[0., 0., 4.]])}, num_anchors=1)
    np.testing.assert_allclose(atlas["change_scores_mm"], [2.])
    np.testing.assert_allclose(atlas["reference_offsets_mm"], [[0., 0., 2.]])
    np.testing.assert_allclose(atlas["reference_distance_profiles_mm"], [[2., 2., 2., 2.]])
    assert atlas["anchors_mm"].dtype == np.float32


def test_frozen_training_atlas_is_independent_of_held_out_geometry_and_labels():
    # The function sees only the explicit training mapping. Unused objects must
    # never influence the artifact, including designs whose responses vary most.
    all_meshes = {0: np.array([[0., 0., 0.]]), 1: np.array([[0., 0., 1.]]),
                  10: np.array([[0., 0., 1000.]])}
    atlas = fit({key: all_meshes[key] for key in (0, 1)})
    all_meshes[10][:] = np.nan
    again = fit({key: all_meshes[key] for key in (0, 1)})
    assert_same_atlas(atlas, again)
    assert atlas["fit_split"] == "train"
    assert atlas["train_design_ids"] == [0, 1]
    # Returned arrays are owned by the artifact, not views of a mutable input.
    all_meshes[0][:] = 100.
    assert np.max(atlas["anchors_mm"]) <= 1.


def test_spatial_separation_retains_distant_changes_before_dense_cluster():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [100., 0., 0.]])
    b = a + [0., 0., 2.]
    atlas = fit({0: a, 1: b}, num_anchors=2, anchor_separation_mm=20.)
    assert np.linalg.norm(atlas["anchors_mm"][0] - atlas["anchors_mm"][1]) >= 20.
    # Once the preferred separation is exhausted, additional *changed* sites
    # remain available; this must neither duplicate anchors nor add zero scores.
    full = fit({0: a, 1: b}, num_anchors=6, anchor_separation_mm=20.)
    assert len(np.unique(full["anchors_mm"], axis=0)) == 6
    assert np.all(full["change_scores_mm"] > 0)


@pytest.mark.parametrize("kwargs", [
    {"num_anchors": 0}, {"num_anchors": True}, {"num_anchors": 2.5},
    {"neighborhood_k": 0}, {"neighborhood_k": False},
    {"impactor_nodes": -1}, {"impactor_nodes": .5},
    {"candidate_spacing_mm": 0}, {"candidate_spacing_mm": np.nan},
    {"anchor_separation_mm": np.inf}, {"anchor_separation_mm": -1},
])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        fit({0: np.zeros((2, 3)), 1: np.ones((2, 3))}, **kwargs)


@pytest.mark.parametrize("meshes, match", [
    ({}, "at least two"),
    ({0: np.zeros((1, 3))}, "at least two"),
    ({0: np.zeros((2, 3)), 1: np.zeros((2, 2))}, "shape"),
    ({0: np.zeros((2, 3)), 1: np.full((2, 3), np.nan)}, "nonfinite"),
    ({0: np.zeros((2, 3)), 1: np.empty((0, 3))}, "no structural"),
    ({0: np.zeros((2, 3)), -1: np.ones((2, 3))}, "design ID"),
    ({0: np.zeros((2, 3)), "1": np.ones((2, 3))}, "design ID"),
])
def test_invalid_meshes_fail(meshes, match):
    with pytest.raises(ValueError, match=match):
        fit(meshes)


def base_shape_families(offset_mm, nudge_mm):
    """Two families whose base shapes differ far more than siblings differ.

    Siblings 0/1 carry a nudge at x=0; siblings 2/3 carry one at x=50. The whole
    second family also sits ``offset_mm`` away in z, mimicking a different base
    panel. Pooled scoring must prefer that offset; within-family scoring must not.
    """
    spine = np.stack([np.arange(0., 101., 5.), np.zeros(21), np.zeros(21)], axis=1)
    designs = {}
    for index in range(4):
        points = spine.copy()
        points[:, 2] += offset_mm * (index >= 2)
        points[0 if index < 2 else 10, 2] += nudge_mm * (index % 2)
        designs[index] = points
    return designs


def test_pooled_scoring_spends_every_anchor_on_the_base_shape_difference():
    atlas = fit(base_shape_families(30., 3.), num_anchors=4, candidate_spacing_mm=1.,
                anchor_separation_mm=2., neighborhood_k=2)
    assert atlas["change_scope"] == "all_designs"
    assert atlas["num_scoring_families"] == 1
    # Every anchor scores at the 30 mm base-shape scale, never the 3 mm nudge
    # scale, so the sibling differences receive no dedicated representation.
    assert np.all(atlas["change_scores_mm"] > 10.)


def test_within_family_scoring_finds_the_sibling_nudges_instead():
    designs = base_shape_families(30., 3.)
    families = {0: "a", 1: "a", 2: "b", 3: "b"}
    atlas = fit(designs, num_anchors=4, candidate_spacing_mm=1., anchor_separation_mm=2.,
                neighborhood_k=2, groups_by_design=families)
    assert atlas["change_scope"] == "within_family"
    assert atlas["scoring_families"] == [[0, 1], [2, 3]]
    assert atlas["num_scoring_families"] == 2
    # Both sibling nudge sites are anchored, and every score now sits at the
    # nudge scale rather than the far larger base-shape offset.
    assert {0., 50.} <= set(atlas["anchors_mm"][:, 0].tolist())
    assert np.all(atlas["change_scores_mm"] < 10.)
    # Reference geometry still averages every supplied design, not one family.
    pooled = fit(designs, num_anchors=4, candidate_spacing_mm=1., anchor_separation_mm=2.,
                 neighborhood_k=2)
    assert pooled["reference_distance_profiles_mm"].shape == atlas["reference_distance_profiles_mm"].shape


def test_grouping_recovers_families_from_training_geometry_alone():
    grouping = group_designs_by_geometry(base_shape_families(30., 3.), impactor_nodes=0,
                                         tolerance_mm=.5, max_family_fraction=.3)
    assert grouping["families"] == [[0, 1], [2, 3]]
    assert grouping["design_family"] == {0: 0, 1: 0, 2: 1, 3: 1}
    assert grouping["num_families"] == 2
    assert grouping["max_within_family_moved_fraction"] < grouping["min_cross_family_moved_fraction"]
    assert grouping["separation_margin"] > 1
    assert grouping["moved_fraction_matrix"].shape == (4, 4)


def test_grouping_reads_only_the_supplied_designs():
    designs = base_shape_families(30., 3.)
    expected = group_designs_by_geometry({key: designs[key] for key in (0, 1, 2)}, impactor_nodes=0,
                                         tolerance_mm=.5, max_family_fraction=.3)
    designs[3][:] = np.nan
    again = group_designs_by_geometry({key: designs[key] for key in (0, 1, 2)}, impactor_nodes=0,
                                      tolerance_mm=.5, max_family_fraction=.3)
    assert expected["families"] == again["families"] == [[0, 1], [2]]


def test_identical_designs_collapse_into_one_family_with_infinite_margin():
    spine = np.stack([np.arange(0., 21., 1.), np.zeros(21), np.zeros(21)], axis=1)
    grouping = group_designs_by_geometry({0: spine, 1: spine.copy()}, impactor_nodes=0)
    assert grouping["families"] == [[0, 1]]
    assert grouping["separation_margin"] is None  # no cross-family pair exists
    assert grouping["max_within_family_moved_fraction"] == 0


def test_min_change_excludes_negligible_sites_from_the_anchor_fallback():
    a = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    b = a + [[0., 0., 4.], [0., 0., .02], [0., 0., 0.]]
    # Without a floor, the 0.02 mm and 0 mm sites pad the request to three.
    assert fit({0: a, 1: b}, num_anchors=3, anchor_separation_mm=100.)["num_anchors"] == 3
    atlas = fit({0: a, 1: b}, num_anchors=3, anchor_separation_mm=100., min_change_mm=.5)
    assert atlas["min_change_mm"] == .5
    assert atlas["change_tolerance_mm"] == .5
    # Only the 4 mm site survives; the 0.02 mm and unchanged sites cannot pad it.
    assert set(atlas["anchors_mm"][:, 0].tolist()) == {0.}
    assert atlas["num_changed_candidates"] == atlas["num_anchors"] < 3


def test_single_design_families_cannot_supply_within_family_scoring():
    designs = base_shape_families(30., 3.)
    with pytest.raises(ValueError, match="single design"):
        fit(designs, groups_by_design={index: index for index in range(4)})


@pytest.mark.parametrize("groups, match", [
    ({0: "a", 1: "a", 2: "b"}, "exactly the supplied"),
    ({0: "a", 1: "a", 2: "b", 3: "b", 4: "c"}, "exactly the supplied"),
    ("not a mapping", "must be a mapping"),
])
def test_invalid_groupings_fail(groups, match):
    with pytest.raises(ValueError, match=match):
        fit(base_shape_families(30., 3.), groups_by_design=groups)


@pytest.mark.parametrize("kwargs", [
    {"min_change_mm": -1}, {"min_change_mm": np.nan}, {"min_change_mm": True},
])
def test_invalid_min_change_fails(kwargs):
    with pytest.raises(ValueError, match="min_change_mm"):
        fit({0: np.zeros((2, 3)), 1: np.ones((2, 3))}, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"tolerance_mm": 0}, {"tolerance_mm": np.inf}, {"max_family_fraction": 1.},
    {"max_family_fraction": 0}, {"max_family_fraction": -.1}, {"impactor_nodes": -1},
    {"impactor_nodes": .5}, {"tolerance_mm": -1},
])
def test_invalid_grouping_settings_fail(kwargs):
    settings = {"impactor_nodes": 0, **kwargs}
    with pytest.raises(ValueError):
        group_designs_by_geometry(base_shape_families(30., 3.), **settings)
