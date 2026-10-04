"""Find geometry-change neighborhoods using training designs alone.

The atlas is an unsupervised, physical-coordinate preprocessing artifact. It
does not accept acceleration histories, impact locations, or a test dataset.
Callers must pass only the unique meshes belonging to their training split and
save the resulting atlas with the checkpoint. Validation/test meshes are then
queried against these frozen anchors, never included in this fit.

Meshes need not share node counts, node numbering, or topology. The only order
assumption is the existing dataset convention that the first ``impactor_nodes``
rows belong to the rigid headform. Remaining rows are unordered structural XYZ.
Nearest-surface variation detects geometric displacements and missing/added
regions, though it can also respond to differences in mesh sampling density.
No dense node-by-node distance matrix is constructed.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Hashable, Mapping

import numpy as np
from scipy.spatial import cKDTree


CHANGE_TOLERANCE_MM = 1e-6


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive_real(name: str, value: float) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _voxel_representatives(points: np.ndarray, spacing: float) -> np.ndarray:
    """One actual union point per fixed-origin voxel, independent of row order."""
    # ``unique`` also provides lexicographic ordering for deterministic ties.
    points = np.unique(points, axis=0)
    voxel = np.floor(points / spacing)
    center = (voxel + 0.5) * spacing
    radius_squared = np.square(points - center).sum(axis=1)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], radius_squared,
                        voxel[:, 2], voxel[:, 1], voxel[:, 0]))
    ordered_voxel = voxel[order]
    first = np.r_[True, np.any(ordered_voxel[1:] != ordered_voxel[:-1], axis=1)]
    return points[order[first]]


def _spread_changed_candidates(
    candidates: np.ndarray, scores: np.ndarray, count: int, separation: float,
    tolerance: float = CHANGE_TOLERANCE_MM,
) -> np.ndarray:
    """Prioritize changes, then fill uncovered space without unchanged anchors."""
    positive = np.flatnonzero(scores > tolerance)
    if not len(positive):
        raise ValueError(
            "Training structural meshes have no measurable geometry changes "
            f"above {tolerance:g} mm; cannot fit a change atlas."
        )
    points, weights = candidates[positive], scores[positive]
    # Candidate indices already have deterministic spatial ordering.
    priority = np.argsort(-weights, kind="stable")
    available = np.ones(len(positive), dtype=bool)
    nearest_selected_squared = np.full(len(positive), np.inf)
    selected: list[int] = []
    for _ in range(min(count, len(positive))):
        separated = available & (nearest_selected_squared >= separation ** 2)
        if np.any(separated):
            chosen = int(priority[np.flatnonzero(separated[priority])[0]])
        else:
            # A small change region may not contain ``count`` separated nodes.
            # Fill from its remaining most spatially distinct changed points.
            chosen = int(np.argmax(np.where(available, nearest_selected_squared, -1.0)))
        selected.append(int(positive[chosen]))
        available[chosen] = False
        nearest_selected_squared = np.minimum(
            nearest_selected_squared, np.square(points - points[chosen]).sum(axis=1)
        )
    return np.asarray(selected, dtype=np.int64)


def _structural_clouds(
    meshes_by_design: Mapping[int, np.ndarray], impactor_nodes: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Validate the mapping and return sorted IDs with deduplicated hood points."""
    if not isinstance(meshes_by_design, Mapping) or len(meshes_by_design) < 2:
        raise ValueError("meshes_by_design must contain at least two training designs")
    design_ids = [_integer("training design ID", key, 0) for key in meshes_by_design]
    design_ids.sort()
    structures: list[np.ndarray] = []
    for design_id in design_ids:
        try:
            mesh = np.asarray(meshes_by_design[design_id], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Training design {design_id} must be a numeric XYZ array") from exc
        if mesh.ndim != 2 or mesh.shape[1] != 3:
            raise ValueError(f"Training design {design_id} must have shape (N, 3)")
        if not np.isfinite(mesh).all():
            raise ValueError(f"Training design {design_id} contains nonfinite coordinates")
        if len(mesh) <= impactor_nodes:
            raise ValueError(f"Training design {design_id} has no structural nodes after headform exclusion")
        # Canonical ordering makes cKDTree's exact-distance ties independent of
        # the original mesh numbering. Duplicates cannot alter shape evidence.
        structures.append(np.unique(mesh[impactor_nodes:], axis=0))
    return design_ids, structures


def group_designs_by_geometry(
    meshes_by_design: Mapping[int, np.ndarray],
    impactor_nodes: int = 286,
    tolerance_mm: float = 0.5,
    max_family_fraction: float = 0.05,
) -> dict[str, Any]:
    """Partition TRAIN designs into near-clone families from geometry alone.

    Two designs join a family when the symmetric fraction of structural points
    whose nearest surface distance to the other design exceeds ``tolerance_mm``
    stays below ``max_family_fraction``. Families are the connected components
    of that relation, so a chain of near-clones stays together.

    This exists because a DOE built by nudging one base shape produces both a
    few large *base shape* differences and the many small local features that
    actually distinguish sibling designs. Pooling every design into one variance
    lets the base shape difference dominate, and the small features receive no
    anchors. Grouping is therefore part of anchor selection, not a data label.

    Only the supplied training meshes are read; no design outside the mapping
    contributes, and no response, impact location, or externally supplied
    cluster table is used. ``separation_margin`` is the ratio between the
    smallest cross-family and largest within-family fraction. A value near one
    means the threshold sits inside a continuum rather than a real gap, so the
    caller should treat the partition as unreliable and inspect it.
    """
    impactor_nodes = _integer("impactor_nodes", impactor_nodes, 0)
    tolerance_mm = _positive_real("tolerance_mm", tolerance_mm)
    max_family_fraction = _positive_real("max_family_fraction", max_family_fraction)
    if not max_family_fraction < 1:
        raise ValueError("max_family_fraction must be below one")
    design_ids, structures = _structural_clouds(meshes_by_design, impactor_nodes)
    trees = [cKDTree(points) for points in structures]

    size = len(design_ids)
    moved = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            forward = float(np.mean(trees[j].query(structures[i], k=1)[0] > tolerance_mm))
            backward = float(np.mean(trees[i].query(structures[j], k=1)[0] > tolerance_mm))
            moved[i, j] = moved[j, i] = max(forward, backward)

    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(size):
        for j in range(i + 1, size):
            if moved[i, j] < max_family_fraction:
                roots = find(i), find(j)
                if roots[0] != roots[1]:
                    parent[roots[0]] = roots[1]
    members: dict[int, list[int]] = {}
    for index in range(size):
        members.setdefault(find(index), []).append(index)
    # Order families by their lowest design ID so the result is deterministic.
    families = sorted((sorted(design_ids[i] for i in group) for group in members.values()),
                      key=lambda group: group[0])
    design_family = {design: number for number, group in enumerate(families) for design in group}

    same = [moved[i, j] for i in range(size) for j in range(i + 1, size)
            if design_family[design_ids[i]] == design_family[design_ids[j]]]
    cross = [moved[i, j] for i in range(size) for j in range(i + 1, size)
             if design_family[design_ids[i]] != design_family[design_ids[j]]]
    within_max, cross_min = (max(same) if same else None), (min(cross) if cross else None)
    margin = None
    if within_max is not None and cross_min is not None:
        margin = float("inf") if within_max == 0 else cross_min / within_max
    return {
        "design_family": design_family,
        "families": families,
        "num_families": len(families),
        "tolerance_mm": tolerance_mm,
        "max_family_fraction": max_family_fraction,
        "max_within_family_moved_fraction": within_max,
        "min_cross_family_moved_fraction": cross_min,
        "separation_margin": margin,
        "moved_fraction_design_ids": design_ids,
        "moved_fraction_matrix": moved.astype(np.float32),
    }


def _scoring_families(
    groups_by_design: Mapping[int, Hashable], design_ids: list[int],
) -> list[list[int]]:
    """Rows of ``nearest_points`` per family, keeping only comparable families."""
    if not isinstance(groups_by_design, Mapping):
        raise ValueError("groups_by_design must be a mapping from design ID to family")
    if {_integer("grouped design ID", key, 0) for key in groups_by_design} != set(design_ids):
        raise ValueError("groups_by_design must label exactly the supplied training designs")
    rows: dict[Hashable, list[int]] = {}
    for row, design_id in enumerate(design_ids):
        rows.setdefault(groups_by_design[design_id], []).append(row)
    # A family holding one design has no within-family variation to measure.
    comparable = [group for group in rows.values() if len(group) > 1]
    if not comparable:
        raise ValueError(
            "Within-family scoring needs at least one family with two training "
            "designs; every supplied family holds a single design."
        )
    return comparable


def fit_change_atlas(
    meshes_by_design: Mapping[int, np.ndarray],
    num_anchors: int = 32,
    candidate_spacing_mm: float = 10.0,
    anchor_separation_mm: float = 30.0,
    impactor_nodes: int = 286,
    neighborhood_k: int = 256,
    groups_by_design: Mapping[int, Hashable] | None = None,
    min_change_mm: float = 0.0,
) -> dict[str, Any]:
    """Fit immutable-at-inference change anchors from TRAIN meshes in mm.

    For each voxel-balanced union candidate ``c``, each training design supplies
    its nearest structural point ``p_d(c)``. The score is
    ``sqrt(mean_d(||p_d(c) - mean_d(p_d(c))||^2))`` in mm. Thus no correspondence
    between node indices is assumed. Holes/additions can change the nearest
    point even when a point exists in only one design.

    With ``groups_by_design``, the score instead pools *within-family* variance,
    ``sqrt(mean_g(mean_{d in g}(||p_d(c) - mean_{d in g}(p_d(c))||^2)))`` over the
    families ``g`` holding at least two designs. Use this whenever the designs
    form near-clone families: a shared base-shape difference between families is
    far larger than the local features separating siblings, so pooled scoring
    spends every anchor on the base shape and none on the local features.
    ``group_designs_by_geometry`` derives the families from the same training
    meshes. Reference offsets and profiles stay means over all supplied designs.

    ``min_change_mm`` raises the floor a candidate must clear to be eligible,
    which keeps near-zero numerical differences from becoming anchors once the
    genuinely changed sites are exhausted.

    Returns ``anchors_mm`` (A, 3), ``change_scores_mm`` (A,), training-mean
    ``reference_offsets_mm`` (A, 3) of the nearest point from each anchor, and
    ``reference_distance_profiles_mm`` (A, K), the training-mean sorted KNN
    distances. Profiles for meshes with fewer than K structural nodes repeat
    their furthest available distance. Float arrays use float32. ``A`` may be
    smaller than ``num_anchors`` when fewer candidates exhibit a real change;
    no unchanged anchors are inserted. Metadata contains the exact training
    design IDs, settings, and candidate counts. Identical structural meshes
    fail explicitly instead of producing arbitrary neighborhoods.

    This is a fit operation, not an inference-time mesh comparison: changing
    the supplied design set changes the atlas. A caller must never pass held
    out designs here. Test geometry may only be used later as a model input.
    """
    num_anchors = _integer("num_anchors", num_anchors, 1)
    neighborhood_k = _integer("neighborhood_k", neighborhood_k, 1)
    impactor_nodes = _integer("impactor_nodes", impactor_nodes, 0)
    candidate_spacing_mm = _positive_real("candidate_spacing_mm", candidate_spacing_mm)
    anchor_separation_mm = _positive_real("anchor_separation_mm", anchor_separation_mm)
    if (isinstance(min_change_mm, (bool, np.bool_)) or not isinstance(min_change_mm, Real)
            or not np.isfinite(min_change_mm) or min_change_mm < 0):
        raise ValueError("min_change_mm must be finite and nonnegative")
    design_ids, structures = _structural_clouds(meshes_by_design, impactor_nodes)

    candidates = _voxel_representatives(np.concatenate(structures), candidate_spacing_mm)
    trees = [cKDTree(points) for points in structures]
    # O(designs * candidates), never O(mesh_nodes ** 2). Keeping the nearest
    # point field is modest compared with storing full KNN for every candidate.
    nearest_points = np.stack([
        points[tree.query(candidates, k=1)[1]]
        for points, tree in zip(structures, trees)
    ])
    # Offsets and profiles describe the training mean shape for every design,
    # so they stay pooled even when only within-family variance ranks anchors.
    reference_points = nearest_points.mean(axis=0)
    if groups_by_design is None:
        families = [list(range(len(design_ids)))]
    else:
        families = _scoring_families(groups_by_design, design_ids)
    variances = [
        np.square(nearest_points[rows] - nearest_points[rows].mean(axis=0)).sum(axis=2).mean(axis=0)
        for rows in families
    ]
    scores = np.sqrt(np.mean(variances, axis=0))
    tolerance = max(CHANGE_TOLERANCE_MM, float(min_change_mm))
    selected = _spread_changed_candidates(candidates, scores, num_anchors, anchor_separation_mm, tolerance)
    anchors = candidates[selected]
    distance_profiles = []
    for points, tree in zip(structures, trees):
        count = min(neighborhood_k, len(points))
        distances = np.asarray(tree.query(anchors, k=count)[0]).reshape(len(anchors), count)
        if count < neighborhood_k:
            distances = np.pad(distances, ((0, 0), (0, neighborhood_k - count)), mode="edge")
        distance_profiles.append(distances)

    return {
        "schema_version": 1,
        "fit_split": "train",
        "train_design_ids": design_ids,
        "num_training_designs": len(design_ids),
        "num_anchors_requested": num_anchors,
        "num_anchors": len(anchors),
        "candidate_spacing_mm": candidate_spacing_mm,
        "anchor_separation_mm": anchor_separation_mm,
        "impactor_nodes": impactor_nodes,
        "neighborhood_k": neighborhood_k,
        "change_tolerance_mm": tolerance,
        "min_change_mm": float(min_change_mm),
        "change_scope": "all_designs" if groups_by_design is None else "within_family",
        "scoring_families": [[design_ids[row] for row in rows] for rows in families],
        "num_scoring_families": len(families),
        "num_candidates": len(candidates),
        "num_changed_candidates": int(np.count_nonzero(scores > tolerance)),
        "structural_node_counts": [len(points) for points in structures],
        "anchors_mm": anchors.astype(np.float32),
        "change_scores_mm": scores[selected].astype(np.float32),
        "reference_offsets_mm": (reference_points[selected] - anchors).astype(np.float32),
        "reference_distance_profiles_mm": np.mean(distance_profiles, axis=0).astype(np.float32),
    }
