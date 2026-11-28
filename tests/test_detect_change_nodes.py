"""ID-free change-node detection (detect_change_nodes.py) on synthetic shells."""
import numpy as np
import pytest

pytest.importorskip("pyvista")

from detect_change_nodes import Surface, detect_edit, hysteresis, mesh_edges, triangulate  # noqa: E402


def grid(n, size=200.0, height=None):
    """Quad grid on [0, size]^2; z = height(x, y) or 0."""
    axis = np.linspace(0.0, size, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = np.zeros_like(x) if height is None else height(x, y)
    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    index = np.arange(n * n).reshape(n, n)
    faces = [(index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
             for i in range(n - 1) for j in range(n - 1)]
    return points, faces


def bump(x, y):
    return 5.0 * np.exp(-((x - 100.0) ** 2 + (y - 100.0) ** 2) / (2 * 12.0 ** 2))


def test_triangulate_and_edges():
    assert triangulate([(0, 1, 2, 3), (3, 2, 4)]).tolist() == [[0, 1, 2], [0, 2, 3], [3, 2, 4]]
    assert len(mesh_edges([(0, 1, 2, 3)])) == 4


def test_hysteresis_keeps_seeded_regions_only():
    edges = np.array([[0, 1], [1, 2], [3, 4]])
    low = np.array([1, 1, 1, 1, 1], bool)
    high = np.array([0, 0, 1, 0, 0], bool)
    assert hysteresis(low, high, edges, 5).tolist() == [True, True, True, False, False]
    assert not hysteresis(low, high, edges, 5, min_nodes=4).any()


def test_morphed_mesh_is_exact_and_order_free():
    points, faces = grid(41)
    reference = Surface(points, faces)
    moved = points.copy()
    moved[:, 2] = bump(moved[:, 0], moved[:, 1])
    truth = moved[:, 2] >= 0.01
    perm = np.random.default_rng(0).permutation(len(points))
    inverse = np.argsort(perm)
    edited = Surface(moved[perm], [tuple(inverse[list(f)]) for f in faces])
    on_reference, on_edited = detect_edit(reference, edited)
    assert on_reference.shared_fraction > 0.5
    assert np.array_equal(on_reference.changed, truth)
    assert np.array_equal(on_edited.changed[inverse], truth)
    # A reference node's displacement is its offset to the edited surface: the normal
    # part of the true vertical move, so a little shorter on the bump's slopes.
    lift = -on_reference.displacement[truth, 2]
    assert (lift > 0).all() and (on_reference.distance[truth] <= moved[truth, 2] + 1e-9).all()
    assert (on_reference.distance[truth] >= 0.9 * moved[truth, 2]).all()


def test_remeshed_design_finds_the_bump_and_nothing_else():
    points, faces = grid(41)  # 5 mm reference
    reference = Surface(points, faces)
    new_points, new_faces = grid(29, height=bump)  # ~7.1 mm, no node in common away from corners
    on_reference, on_edited = detect_edit(reference, Surface(new_points, new_faces))
    assert on_reference.shared_fraction < 0.5
    height = bump(points[:, 0], points[:, 1])
    assert on_reference.changed[height > 0.5].all()   # every node that moved > 0.5 mm
    # The coarse facets straddle the bump's tail by ~0.01 mm: far from it, nothing.
    assert not on_reference.changed[height < 0.001].any()
    new_height = bump(new_points[:, 0], new_points[:, 1])
    assert on_edited.changed[new_height > 0.5].all()
    assert not on_edited.changed[new_height < 0.001].any()


def test_no_edit_on_curved_remesh_flags_nothing():
    def flange(x, y):
        # Flat plate bending up past x = 130 mm (curvature 0.008 / mm), like a hood fillet.
        return 0.004 * np.maximum(x - 130.0, 0.0) ** 2

    reference = Surface(*grid(41, height=flange))   # 5 mm
    remeshed = Surface(*grid(23, height=flange))    # 9.1 mm: chords sag up to ~0.08 mm
    on_reference, on_edited = detect_edit(reference, remeshed)
    assert on_reference.distance.max() > 0.04  # the chordal gap is real
    assert not on_reference.changed.any() and not on_edited.changed.any()
    on_reference, _ = detect_edit(reference, remeshed, kappa=0.0)
    assert on_reference.changed.any()  # without the tolerance the fillet would be flagged
