"""Exploratory all-design geometry diagnostics against an already frozen atlas.

This module does not train a model or refit an atlas. Held-out designs are
explicitly included for the user's post-training analysis. Nearest-node
distances measure unordered point-cloud mismatch, not continuous surface
distance; remeshing and sampling density can influence them. Exact node-ID
displacements are additionally reported only when connectivity is identical.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from abaqus_scripts.inp_geom import Deck

ROOT = Path(__file__).resolve().parent
FAMILIES = {"A": [0, 1, 2, 3], "B": [4, 5], "C": [6, 7, 8, 9], "D": [10, 11]}
RADII_MM = (50., 100., 200., 400.)


def _cloud(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Structural points must be a nonempty finite (N,3) array")
    return np.unique(points, axis=0)


def frozen_patch_coverage(points, anchors_mm, neighborhood_k=256):
    """Deduplicated physical-XYZ KNN patch membership, with no atlas fitting."""
    points = _cloud(points)
    anchors = np.asarray(anchors_mm, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not len(anchors) or not np.isfinite(anchors).all():
        raise ValueError("anchors_mm must be a nonempty finite (A,3) array")
    if isinstance(neighborhood_k, bool) or not isinstance(neighborhood_k, int) or neighborhood_k < 1:
        raise ValueError("neighborhood_k must be a positive integer")
    count = min(neighborhood_k, len(points))
    distances, indices = cKDTree(points).query(anchors, k=count)
    indices = np.asarray(indices).reshape(len(anchors), count)
    mask = np.zeros(len(points), dtype=bool)
    mask[indices.ravel()] = True
    return points, mask, np.asarray(distances).reshape(len(anchors), count)[:, -1]


def analyze_geometry(meshes_by_design, anchors_mm, impact_xy_by_location,
                     neighborhood_k=256, tolerance_mm=.5,
                     families=FAMILIES, splits=None):
    """Analyze STRUCTURAL-only clouds; returns summary frames + within-family arrays.

    ``impact_xy_by_location`` maps actual integer location IDs to physical XY.
    Pair metrics concatenate both nearest-neighbor directions, so both missing
    and added regions count. Local RMS and changed fractions use all structural
    nodes within each XY radius as their denominator (not only changed nodes).
    ``*_changed_fraction_max_direction`` additionally matches the atlas family
    grouping statistic. Distances to change sites use XY, while geometric
    mismatch and atlas KNN use XYZ. No responses are read here.
    """
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be finite and positive")
    designs = sorted(meshes_by_design)
    family = {d: name for name, ids in families.items() for d in ids}
    split = {d: name for name, ids in (splits or {}).items() for d in ids}
    prepared = {d: frozen_patch_coverage(meshes_by_design[d], anchors_mm, neighborhood_k) for d in designs}
    trees = {d: cKDTree(prepared[d][0]) for d in designs}
    summary, local_rows, arrays = [], [], {}
    for i, j in combinations(designs, 2):
        a, ca, ra = prepared[i]
        b, cb, rb = prepared[j]
        da, db = trees[j].query(a)[0], trees[i].query(b)[0]
        ma, mb = da > tolerance_mm, db > tolerance_mm
        xyz = np.concatenate((a, b))
        distance = np.concatenate((da, db))
        changed = np.concatenate((ma, mb))
        covered = np.concatenate((ca, cb))
        change_xyz = xyz[changed]
        same = family.get(i) == family.get(j) and i in family and j in family
        base = dict(design_i=i, design_j=j, pair=f"{i}-{j}", family_i=family.get(i, ""),
                    family_j=family.get(j, ""), within_family=same,
                    split_i=split.get(i, "unspecified"), split_j=split.get(j, "unspecified"),
                    involves_test=split.get(i) == "test" or split.get(j) == "test",
                    involves_validation=split.get(i) == "validation" or split.get(j) == "validation")
        row = dict(base, nodes_i=len(a), nodes_j=len(b), nn_rms_mm=float(np.sqrt(np.mean(distance**2))),
                   nn_mean_mm=float(distance.mean()), nn_p95_mm=float(np.percentile(distance, 95)),
                   nn_p99_mm=float(np.percentile(distance, 99)), nn_max_mm=float(distance.max()),
                   changed_nodes_i=int(ma.sum()), changed_nodes_j=int(mb.sum()),
                   changed_fraction=float(changed.mean()),
                   changed_fraction_max_direction=float(max(ma.mean(), mb.mean())),
                   patch_union_nodes_i=int(ca.sum()), patch_union_nodes_j=int(cb.sum()),
                   patch_total_fraction=float(covered.mean()),
                   changed_coverage_fraction=float(covered[changed].mean()) if changed.any() else np.nan,
                   changed_coverage_fraction_i=float(ca[ma].mean()) if ma.any() else np.nan,
                   changed_coverage_fraction_j=float(cb[mb].mean()) if mb.any() else np.nan,
                   mismatch_energy_coverage_fraction=float(np.sum(distance[covered]**2) / np.sum(distance**2))
                   if np.any(distance) else np.nan,
                   patch_k_radius_median_i_mm=float(np.median(ra)),
                   patch_k_radius_median_j_mm=float(np.median(rb)))
        summary.append(row)
        if same:
            for d, points, values, mask, patch in ((i, a, da, ma, ca), (j, b, db, mb, cb)):
                prefix = f"pair_{i}_{j}_d{d}"
                arrays[prefix + "_xyz"] = points[mask].astype(np.float32)
                arrays[prefix + "_distance_mm"] = values[mask].astype(np.float32)
                arrays[prefix + "_covered"] = patch[mask]
        squared = distance**2
        for loc, impact in impact_xy_by_location.items():
            xy_distance = np.linalg.norm(xyz[:, :2] - np.asarray(impact), axis=1)
            change_distances = xy_distance[changed]
            item = dict(base, location_id=int(loc), impact_x_mm=float(impact[0]), impact_y_mm=float(impact[1]),
                        min_changed_distance_xy_mm=float(change_distances.min()) if changed.any() else np.nan,
                        median_changed_distance_xy_mm=float(np.median(change_distances)) if changed.any() else np.nan)
            for radius in RADII_MM:
                inside = xy_distance <= radius
                stem = f"local_{radius:g}mm"
                item[stem + "_nodes"] = int(inside.sum())
                item[stem + "_nn_rms_mm"] = float(np.sqrt(squared[inside].mean())) if inside.any() else np.nan
                item[stem + "_changed_fraction"] = float(changed[inside].mean()) if inside.any() else np.nan
            local_rows.append(item)
        print(f"Geometry pair {i}-{j}: changed {row['changed_fraction']:.3%}, atlas coverage "
              f"{row['changed_coverage_fraction']:.1%}", flush=True)
    return pd.DataFrame(summary), pd.DataFrame(local_rows), arrays


def load_canonical_geometry(data_dir, design_ids=range(12), impactor_nodes=286):
    """Read just the first impact deck per design; validate actual headform set."""
    meshes, decks, audit = {}, {}, []
    for d in design_ids:
        path = Path(data_dir) / "inp_files" / f"HoodImpact_{142*d+1}.inp"
        deck = Deck(path)
        headform = deck.impactor_node_ids()
        if set(list(deck.nodes)[:impactor_nodes]) != headform:
            raise ValueError(f"Design {d}: leading {impactor_nodes} nodes do not match the headform set")
        points = np.asarray(list(deck.nodes.values()), dtype=float)[impactor_nodes:]
        meshes[d], decks[d] = points, deck
        audit.append(dict(design_id=d, canonical_run=142*d+1, path=str(path.resolve()),
                          headform_nodes=len(headform), structural_nodes=len(points),
                          inner_nodes=len(deck.elset_nodes("Hood_Inner-1-2")),
                          outer_nodes=len(deck.elset_nodes("Hood_Outer-1-2"))))
    return meshes, decks, audit


def aligned_geometry_metrics(decks, tolerance_mm=.5):
    """Node-ID displacement only after verifying structural shell connectivity."""
    records = []
    for i, j in combinations(sorted(decks), 2):
        a, b = decks[i], decks[j]
        ai, bi = a.impactor_node_ids(), b.impactor_node_ids()
        na, nb = set(a.nodes) - ai, set(b.nodes) - bi
        ea = {k: v for k, v in a.elems.items() if not set(v) & ai}
        eb = {k: v for k, v in b.elems.items() if not set(v) & bi}
        comparable = na == nb and ea == eb
        row = dict(design_i=i, design_j=j, aligned_same_nodes_and_connectivity=comparable,
                   aligned_geometry_identical=False, aligned_rms_mm=np.nan,
                   aligned_max_mm=np.nan, aligned_changed_nodes=np.nan,
                   aligned_changed_fraction=np.nan, aligned_changed_inner_nodes=np.nan,
                   aligned_changed_outer_nodes=np.nan)
        if comparable:
            ids = sorted(na)
            difference = np.linalg.norm(np.asarray([a.nodes[n] for n in ids]) - np.asarray([b.nodes[n] for n in ids]), axis=1)
            changed_ids = {n for n, delta in zip(ids, difference) if delta > tolerance_mm}
            row.update(aligned_geometry_identical=bool(np.all(difference == 0)),
                       aligned_rms_mm=float(np.sqrt(np.mean(difference**2))), aligned_max_mm=float(difference.max()),
                       aligned_changed_nodes=len(changed_ids), aligned_changed_fraction=len(changed_ids)/len(ids),
                       aligned_changed_inner_nodes=len(changed_ids & a.elset_nodes("Hood_Inner-1-2")),
                       aligned_changed_outer_nodes=len(changed_ids & a.elset_nodes("Hood_Outer-1-2")))
        records.append(row)
    return pd.DataFrame(records)


def plot_family_changes(decks, anchors, arrays, summary, impact_xy, out_dir, k=256):
    """Actual inner-panel shell faces with nearby atlas nodes and sibling changes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    all_values = np.concatenate([v for key, v in arrays.items() if key.endswith("_distance_mm")])
    vmax = max(1., float(np.percentile(all_values, 99)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 12), sharex=True, sharey=True, layout="constrained")
    for ax, (family, members) in zip(axes.flat, FAMILIES.items()):
        representative = members[0]
        deck = decks[representative]
        faces = [np.asarray([deck.nodes[n] for n in deck.elems[e]])[:, [1, 0]]
                 for e in deck.elsets["Hood_Inner-1-2"]]
        ax.add_collection(PolyCollection(faces, facecolors="#e1e6eb", edgecolors="#bcc5ce", linewidths=.17, rasterized=True))
        inner = np.asarray([deck.nodes[n] for n in sorted(deck.elset_nodes("Hood_Inner-1-2"))])
        points, mask, _ = frozen_patch_coverage(np.asarray(list(deck.nodes.values()))[286:], anchors, k)
        is_inner = cKDTree(inner).query(points)[0] < 1e-5
        patch_points = points[mask & is_inner]
        ax.scatter(patch_points[:, 1], patch_points[:, 0], color="#43bfc4", s=2, alpha=.55, linewidths=0, rasterized=True)
        selected = summary.loc[summary.within_family & (summary.family_i == family)]
        family_points, family_values = [], []
        for i, j in combinations(members, 2):
            for d in (i, j):
                prefix = f"pair_{i}_{j}_d{d}"
                xyz = arrays[prefix + "_xyz"]
                own_inner = np.asarray([decks[d].nodes[n] for n in decks[d].elset_nodes("Hood_Inner-1-2")])
                include = cKDTree(own_inner).query(xyz)[0] < 1e-3
                family_points.append(xyz[include])
                family_values.append(arrays[prefix + "_distance_mm"][include])
        xy = np.concatenate(family_points)
        values = np.concatenate(family_values)
        order = np.argsort(values)
        dots = ax.scatter(xy[order, 1], xy[order, 0], c=values[order], cmap="inferno", norm=Normalize(.5, vmax),
                          s=8, linewidths=0, zorder=3, rasterized=True)
        ax.scatter(anchors[:, 1], anchors[:, 0], s=28, facecolors="none", edgecolors="#168a95", linewidths=.8, zorder=4)
        for location in (1, 142):
            impact = impact_xy[location]
            ax.scatter(impact[1], impact[0], marker="*", s=120, color="#2773ce", edgecolors="white", linewidths=.6, zorder=5)
            ax.annotate(str(location), (impact[1], impact[0]), xytext=(7, 6), textcoords="offset points", color="#155099", fontsize=10)
        state = "training" if family in "AC" else "test 4/5" if family == "B" else "train 10 / validation 11"
        coverage = selected.changed_coverage_fraction
        ax.set_title(f"Family {family}: designs {', '.join(map(str, members))} ({state})\n"
                     f"Changed-node patch coverage: {coverage.min():.0%}–{coverage.max():.0%}", fontsize=11)
        ax.set(aspect="equal", xlabel="X2 / Y (mm)", ylabel="X1 / X (mm)")
        ax.autoscale_view()
        ax.grid(alpha=.13)
    figure.suptitle("Small within-family geometry changes and the frozen 32 × 256 atlas\n"
                   "Inner hood only; colors show symmetric nearest-node mismatch > 0.5 mm; outer skin hidden", fontsize=14)
    figure.colorbar(dots, ax=list(axes.flat), shrink=.6, label="Nearest-node mismatch (mm; upper colors clipped at 99th percentile)")
    handles = [Line2D([], [], marker='o', color='none', markerfacecolor='#43bfc4', markeredgecolor='none', label='Frozen patch nodes on reference inner hood'),
               Line2D([], [], marker='o', color='none', markerfacecolor='none', markeredgecolor='#168a95', label='Frozen anchor'),
               Line2D([], [], marker='*', color='none', markerfacecolor='#2773ce', markeredgecolor='none', label='Impact locations 1 and 142')]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"within_family_changes_and_frozen_atlas.{suffix}", dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/mesh_change_attention/20260922_170037_3194307_1704")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/change_geometry_analysis/geometry")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atlas = np.load(args.run_dir / "change_atlas.npz", allow_pickle=False)
    metadata = json.loads((args.run_dir / "change_atlas.json").read_text())
    saved_splits = json.loads((args.run_dir / "splits.json").read_text())
    splits = {name: contents["design_ids"] for name, contents in saved_splits.items()}
    meshes, decks, audit = load_canonical_geometry(args.data_dir)
    impacts = pd.read_csv(args.data_dir / "ImpactCoords_1704.csv")[["X1", "X2"]].to_numpy().reshape(12, 142, 2)
    if not np.allclose(impacts, impacts[:1], atol=.001, rtol=0):
        raise ValueError("Impact locations differ among designs")
    impact_xy = {loc+1: xy for loc, xy in enumerate(impacts[0])}
    summary, local, arrays = analyze_geometry(meshes, atlas["anchors_mm"], impact_xy,
                                              neighborhood_k=metadata["neighborhood_k"], splits=splits)
    summary = summary.merge(aligned_geometry_metrics(decks), on=["design_i", "design_j"], validate="one_to_one")
    summary.to_csv(args.out_dir / "pairwise_geometry.csv", index=False)
    local.to_csv(args.out_dir / "pair_location_geometry.csv", index=False)
    np.savez_compressed(args.out_dir / "within_family_changed_nodes.npz", **arrays)
    within = summary.loc[summary.within_family].copy()
    within.to_csv(args.out_dir / "within_family_geometry.csv", index=False)
    local.loc[local.within_family & local.location_id.isin([1, 142])].to_csv(args.out_dir / "within_family_locations_1_142.csv", index=False)
    overview = {
        "analysis_only": True, "all_12_designs_explicitly_authorized": True,
        "atlas_refitted": False, "model_or_training_code_modified": False,
        "atlas_source": str((args.run_dir / "change_atlas.npz").resolve()),
        "atlas_train_design_ids": metadata["train_design_ids"],
        "atlas_scoring_families": metadata["scoring_families"],
        "atlas_num_anchors": int(len(atlas["anchors_mm"])), "atlas_neighborhood_k": metadata["neighborhood_k"],
        "change_threshold_mm": .5, "splits": splits, "geometry_file_audit": audit,
        "distance_definition": "Symmetric structural XYZ nearest-node distances; deduplicated unordered clouds. Headform excluded. Local radius and impact proximity use XY.",
        "limitation": "Point sampling/remeshing can affect NN distances; not a surface metric or causal test. Aligned displacement is separately supplied only for identical nodes/connectivity.",
        "within_family_pairs": within.replace({np.nan: None}).to_dict(orient="records"),
    }
    (args.out_dir / "geometry_summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    plot_family_changes(decks, atlas["anchors_mm"], arrays, summary, impact_xy, args.out_dir, metadata["neighborhood_k"])
    print("Saved geometry analysis:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
