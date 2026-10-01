import numpy as np

from analyze_hic30_feature_response import curve_gram, feature_associations
from analyze_hic30_feature_identifiability import (
    single_feature_ceiling, cluster_label_ceiling, between_cluster_fraction, within_family,
    within_family_directions, family_relabelings, squared_associations, permutation_scan,
)

CLUSTERS = [(0, 1, 2), (3, 4), (5, 6, 7)]


def test_single_feature_ceiling_is_reached_by_the_first_component_only():
    rng = np.random.default_rng(3)
    curves = rng.normal(size=(8, 2, 40))
    gram = curve_gram(curves)
    ceiling = single_feature_ceiling(gram)
    centered = curves-curves.mean(axis=0)
    for il in range(2):
        u = np.linalg.svd(centered[:, il], full_matrices=False)[0][:, 0]
        best = feature_associations(u[:, None, None], gram, np.arange(8))[0, 0, il]
        np.testing.assert_allclose(best, ceiling[il])
    random = feature_associations(rng.normal(size=(8, 50, 1)), gram, np.arange(8))[:, 0]
    assert np.all(random <= ceiling+1e-12)


def test_cluster_label_ceiling_bounds_every_cluster_constant_feature():
    rng = np.random.default_rng(4)
    curves = rng.normal(size=(8, 3, 30))
    gram = curve_gram(curves)
    ceiling = cluster_label_ceiling(gram, CLUSTERS)
    assert np.all(ceiling <= single_feature_ceiling(gram)+1e-12)
    labels = np.zeros((8, 200))
    for j, members in enumerate(CLUSTERS):
        labels[list(members)] = rng.normal(size=200)
    r2 = feature_associations(labels[:, :, None], gram, np.arange(8))[:, 0]
    assert np.all(r2 <= ceiling+1e-12)
    np.testing.assert_allclose(r2.max(axis=0), ceiling, rtol=.05)


def test_pure_cluster_curves_are_fully_between_cluster_and_reach_one():
    shape = np.sin(np.linspace(0, 3, 25))
    level = np.array([0., 0, 0, 2, 2, 5, 5, 5])
    curves = (level[:, None, None]*shape)+10
    gram = curve_gram(curves)
    np.testing.assert_allclose(between_cluster_fraction(gram, CLUSTERS), 1)
    np.testing.assert_allclose(cluster_label_ceiling(gram, CLUSTERS), 1)
    noise = np.random.default_rng(5).normal(size=(8, 1, 25))
    for members in CLUSTERS:
        noise[list(members)] -= noise[list(members)].mean(axis=0)
    np.testing.assert_allclose(between_cluster_fraction(curve_gram(noise+3), CLUSTERS), 0, atol=1e-12)


def test_within_family_directions_ignore_cluster_constant_offsets():
    rng = np.random.default_rng(6)
    features = np.round(rng.normal(size=(8, 5, 2)), 2)
    offset = np.zeros_like(features)
    for members in CLUSTERS:
        offset[list(members)] = rng.integers(-50, 50, size=(5, 2))
    precision = np.array([.01, .01])
    a = within_family_directions(features, CLUSTERS, precision)
    b = within_family_directions(features+offset, CLUSTERS, precision)
    np.testing.assert_allclose(a[0], b[0], atol=1e-9)
    constant = within_family_directions(offset, CLUSTERS, precision)
    assert len(constant[0]) == 0


def test_relabelings_stay_inside_families():
    perms = family_relabelings([(0, 1, 2, 3), (6, 7, 8, 9)])
    assert perms.shape == (576, 8)
    assert len({tuple(p) for p in perms}) == 576
    assert np.all(np.sort(perms[:, :4], axis=1) == np.arange(4))
    assert np.all(np.sort(perms[:, 4:], axis=1) == np.arange(4, 8))


def test_squared_associations_match_the_feature_response_definition():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 6))
    curves = rng.normal(size=(8, 3, 20))
    gram = curve_gram(curves-curves.mean(axis=0))
    centered = (x-x.mean(axis=0)).T
    unit = centered/np.linalg.norm(centered, axis=1, keepdims=True)
    expected = feature_associations(x[:, :, None], curve_gram(curves), np.arange(8))[:, 0]
    np.testing.assert_allclose(squared_associations(unit, gram), expected, rtol=1e-10)


def test_planted_within_family_signal_gets_the_smallest_p_value():
    rng = np.random.default_rng(8)
    families = [(0, 1, 2, 3), (4, 5, 6, 7)]
    x = np.array([-3., -1, 1, 3, -2, .5, 1, 2.5])
    curves = x[:, None, None]*rng.normal(size=(1, 2, 30))+rng.normal(size=(1, 2, 30))
    features = np.column_stack([x, rng.normal(size=8)])[:, :, None]
    directions, _, _ = within_family_directions(features, families, np.array([1e-6]))
    perms = family_relabelings(families)
    scan = permutation_scan(directions, curve_gram(within_family(curves, families)), perms)
    np.testing.assert_allclose(scan['best'], 1)
    np.testing.assert_allclose(scan['p'], 1/576)
    assert np.all(scan['adjusted'] <= 2/576)
