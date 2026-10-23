"""Change nodes: the hood nodes that differ between near-clone designs.

Two designs are near-clones when their meshes share node IDs and fewer than
``NEAR_CLONE_MAX_NODES`` nodes moved by more than ``MOVED_THRESHOLD_MM`` between
them. On the 1704 set that keeps 46-245 nodes per pair
(figures/design_pair_mesh_differences_1704/pair_summary.csv); every other
same-mesh pair moves ~15k nodes through the morph between clusters. A design's
change nodes are the union over its near-clone partners, which is the row rule
of the pair matrix. All members of a cluster therefore share one node set
(A 188, B 110, C 333, D 192 nodes) and differ only in the coordinates there.

The edits only add. 0-1 and 0-2 sum to 1-2, and every changed node rises
(+Z) from 4 to 5 and from 10 to 11. Each cluster's base is the member lowest
at its change nodes (0, 4, 6 and 10), and a design's displacement is measured
from that base, so the base itself has none.

Held-out designs enter through their geometry only: the change set of design 4
comes from comparing its deck with design 5's. No acceleration is read.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from design_difference_data import PANELS, load_design_geometry

NEAR_CLONE_MAX_NODES = 300
MOVED_THRESHOLD_MM = 0.1
# The base must be no higher than any other member at every change node.
BASE_TOLERANCE_MM = 1e-3
# Columns of DesignChange["static"]: base-hood position and panel gap, and the
# design's displacement and gap change from that base, in mm.
STATIC_CHANNELS = ("base_x", "base_y", "base_z", "dx", "dy", "dz", "base_gap", "gap_change", "outer")


def inp_node_ids(path):
    """Node IDs of the first ``*NODE`` block, in the row order the training mesh uses.

    Mirrors utils.DataPreprocessor._load_mesh_from_inp, which keeps only the
    coordinates of the same lines.
    """
    ids, recording = [], False
    with open(path, "r") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            if not recording:
                recording = text.upper() == "*NODE"
                continue
            if text.startswith("*"):
                break
            parts = text.split(",")
            if len(parts) >= 4:
                ids.append(int(parts[0]))
    return np.array(ids)


def near_clone_changes(geometries, max_nodes=NEAR_CLONE_MAX_NODES, threshold_mm=MOVED_THRESHOLD_MM):
    """Change nodes and displacements of every design, from hood geometry alone.

    ``geometries`` maps design -> dict with ``ids``, ``xyz`` (mm), ``gap`` (mm)
    and ``panel`` (index into PANELS), as design_difference_data.load_design_geometry
    returns. Returns ({design: change}, report). A change holds the cluster, its
    base, the change ``node_ids`` and ``static`` rows (len(node_ids),
    len(STATIC_CHANNELS)). A design without a near-clone has no change nodes.
    """
    designs = sorted(geometries)
    pairs, moved = [], {}
    for a, b in combinations(designs, 2):
        if not np.array_equal(geometries[a]["ids"], geometries[b]["ids"]):
            continue
        mask = np.linalg.norm(geometries[b]["xyz"] - geometries[a]["xyz"], axis=1) > threshold_mm
        near = int(mask.sum()) < max_nodes
        pairs.append({"designs": [a, b], "nodes_moved": int(mask.sum()), "near_clone": near})
        if near:
            moved[(a, b)] = mask

    # Clusters are the connected groups of near-clone pairs.
    cluster_of = {design: design for design in designs}

    def root(design):
        while cluster_of[design] != design:
            design = cluster_of[design]
        return design

    for a, b in moved:
        cluster_of[root(b)] = root(a)
    clusters = {}
    for design in designs:
        clusters.setdefault(root(design), []).append(design)

    changes, cluster_report = {}, []
    for members in clusters.values():
        reference = geometries[members[0]]
        union = np.zeros(len(reference["ids"]), dtype=bool)
        for pair, mask in moved.items():
            if pair[0] in members:
                union |= mask
        heights = {d: geometries[d]["xyz"][union, 2] for d in members}
        base = min(members, key=lambda d: (heights[d].mean() if union.any() else 0.0, d))
        below = [d for d in members if np.any(heights[d] < heights[base] - BASE_TOLERANCE_MM)]
        if below:
            raise ValueError(f"Cluster {members}: no member is lowest at every change node "
                             f"(designs {below} dip below {base}); the base rule assumes edits only raise nodes")
        cluster_report.append({"designs": members, "base": base, "change_nodes": int(union.sum())})
        outer = PANELS.index("outer")
        for design in members:
            rows = np.zeros_like(union)
            for partner in members:
                if partner != design:
                    rows |= moved.get((min(design, partner), max(design, partner)), False)
            geometry, base_geometry = geometries[design], geometries[base]
            static = np.column_stack((
                base_geometry["xyz"][rows], geometry["xyz"][rows] - base_geometry["xyz"][rows],
                base_geometry["gap"][rows], geometry["gap"][rows] - base_geometry["gap"][rows],
                (base_geometry["panel"][rows] == outer).astype(float),
            ))
            changes[design] = {"cluster": members, "base": base,
                               "node_ids": geometry["ids"][rows], "static": static.astype(np.float32)}
    report = {
        "rule": {"near_clone_max_nodes": max_nodes, "moved_threshold_mm": threshold_mm,
                 "change_nodes": "union of nodes moved against every near-clone partner",
                 "displacement_from": "cluster base: the member lowest at every change node"},
        "static_channels": list(STATIC_CHANNELS),
        "same_mesh_pairs": pairs,
        "clusters": cluster_report,
        "designs": {design: {"base": change["base"], "change_nodes": len(change["node_ids"]),
                             "max_displacement_mm": float(np.linalg.norm(change["static"][:, 3:6], axis=1).max())
                             if len(change["node_ids"]) else 0.0}
                    for design, change in changes.items()},
    }
    return changes, report


def find_design_changes(data_dir, designs, samples_per_design):
    """near_clone_changes on the decks, plus each change node's training-mesh row.

    Rows index the mesh of every run of the design (first ``*NODE`` block of its
    deck, headform included); the trainer checks them against the loaded meshes.
    """
    data_dir = Path(data_dir)
    geometries = {design: load_design_geometry(design, data_dir) for design in designs}
    changes, report = near_clone_changes(geometries)
    for design, change in changes.items():
        deck = data_dir / "inp_files" / f"HoodImpact_{design * samples_per_design + 1}.inp"
        row_of = {node: row for row, node in enumerate(inp_node_ids(deck))}
        missing = [node for node in change["node_ids"] if node not in row_of]
        if missing:
            raise ValueError(f"Design {design}: change nodes {missing[:5]} are not in {deck.name}")
        change["rows"] = np.array([row_of[node] for node in change["node_ids"]], dtype=np.int64)
    return changes, report
