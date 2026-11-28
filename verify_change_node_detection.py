"""Check the ID-free change-node detector (detect_change_nodes.py) on the 72 local edits.

Node IDs are used here only to build each edited design (design 6's inner panel plus
the generator's dz) and to score the result. The detector sees point arrays and
element lists; in every test the edited design's node order is shuffled.

Reference: design 6's inner panel as simulated (18,182 nodes, 17,586 elements, ~10 mm).
Edited design n, four ways:
  same_mesh        design 6's elements with the nodes moved (a morphed mesh)
  flat_remesh_hH   gmsh remesh of the edited FE surface, quad-dominant, size H mm
                   (14 -> ~19k nodes, 20 -> ~10k). New nodes lie on the old facets.
  cad_remesh_hH    the same new nodes projected onto a smooth stand-in for the edited
                   CAD surface: butterfly subdivision of the edited FE mesh, which passes
                   through its nodes and bulges where the panel curves. Real meshes of a
                   CAD part sit on the CAD surface, not on the old elements.
  cad_remesh_h14 + gauss  extra Gaussian noise along each node's normal (sigma mm)
Sides scored:
  reference        which of design 6's nodes changed; truth = the generator's nodes
  edited           which of the edited design's nodes changed; truth = the generator's
                   dz carried to each node (|dz| >= 0.01 mm); for cad_* through the
                   subdivision, so it includes the ring the smooth surface spreads it to
  reference_naive  baseline: nearest-node distance with the same robust threshold
fp_far counts false flags not adjacent to a true node (one element ring).

  python verify_change_node_detection.py            # all 72 designs, 12 processes
Writes runs/change_node_detection/<stamp>/: per_design.csv, summary.csv,
summary.json, examples.npz (for plot_change_node_detection.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from detect_change_nodes import MAD_TO_SIGMA, Surface, detect_edit, hysteresis, triangulate

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "Data" / "HoodImpact_LocalEdits"
INNER_PANEL_ELSET = "Hood_Inner-1-2"
FLOOR_MM = 0.01
DZ_BANDS = ((0.01, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, math.inf))
FEATURE_ANGLE_DEG = 20.0  # gmsh patch split; at 40 degrees the main panel fails to mesh
EXAMPLE_DESIGNS = (54, 72, 1)
EXAMPLE_TEST = ("cad_remesh_h14", 0.0)

_WORKER = None  # per process: reference data, see _init_worker


def load_reference(deck_path):
    """(node IDs, {ID: row}, (N, 3) points, faces as row tuples) of design 6's inner panel."""
    sys.path.insert(0, str(ROOT / "abaqus_scripts"))
    from inp_geom import Deck

    deck = Deck(str(deck_path))
    elements = [deck.elems[e] for e in deck.elsets[INNER_PANEL_ELSET]]
    ids = np.array(sorted({n for e in elements for n in e}), dtype=np.int64)
    row = {int(n): i for i, n in enumerate(ids)}
    faces = [tuple(row[n] for n in e) for e in elements]
    points = np.array([deck.nodes[int(n)] for n in ids], dtype=np.float64)
    return ids, row, points, faces


def butterfly(points, faces, flat=None):
    """(points, triangles, flat) of one interpolating butterfly subdivision step.

    Butterfly rings next to sharp bends, which a CAD surface does not do, so the new
    point of every edge whose stencil touches a sharp edge (dihedral > the feature
    angle) or the boundary stays at the flat midpoint. ``flat`` = (new point index,
    edge ends) from design 6; pass it back for an edited design so both use one
    feature set.
    """
    import pyvista as pv

    triangles = triangulate(faces)
    mesh = pv.PolyData(points, np.hstack((np.full((len(triangles), 1), 3), triangles)).ravel())
    fine = mesh.subdivide(1, subfilter="butterfly")
    fine_points = np.asarray(fine.points, dtype=np.float64)
    if not np.allclose(fine_points[:len(points)], points, atol=1e-6):
        raise RuntimeError("Butterfly output does not start with the input points")
    if flat is None:
        surface = Surface(points, [tuple(t) for t in triangles])
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        owner = np.tile(np.arange(len(triangles)), 3)
        edges = np.sort(edges, axis=1)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, owner = edges[order], owner[order]
        unique, start, count = np.unique(edges, axis=0, return_index=True, return_counts=True)
        sharp_node = np.zeros(len(points), bool)
        for (a, b), s, c in zip(unique, start, count):
            if c != 2 or abs(surface.normals[owner[s]] @ surface.normals[owner[s + 1]]) < math.cos(
                    math.radians(FEATURE_ANGLE_DEG)):
                sharp_node[[a, b]] = True
        # Stencil reach: the edge's ends and the far corners of its two triangles.
        far = {}
        for t in triangles:
            for i in range(3):
                key = tuple(sorted((t[i], t[(i + 1) % 3])))
                far.setdefault(key, []).append(t[(i + 2) % 3])
        near_sharp = np.array([sharp_node[a] or sharp_node[b] or sharp_node[far[(a, b)]].any()
                               for a, b in unique])
        # Linear subdivision numbers its new points like butterfly does and puts each
        # exactly at its edge's midpoint: match edges to new points through it.
        linear = mesh.subdivide(1, subfilter="linear")
        if not np.array_equal(linear.faces, fine.faces):
            raise RuntimeError("Linear and butterfly subdivision number their points differently")
        midpoints = 0.5 * (points[unique[:, 0]] + points[unique[:, 1]])
        gap, index = cKDTree(np.asarray(linear.points, dtype=np.float64)[len(points):]).query(midpoints)
        if gap.max() > 1e-3 or len(np.unique(index)) != len(index):  # VTK keeps float32 points
            raise RuntimeError("Could not match subdivision points to their edges")
        flat = (index[near_sharp] + len(points), unique[near_sharp])
    new_index, ends = flat
    fine_points[new_index] = 0.5 * (points[ends[:, 0]] + points[ends[:, 1]])
    return fine_points, fine.faces.reshape(-1, 4)[:, 1:].copy(), flat


def _init_worker(deck_path):
    global _WORKER
    ids, row, points, faces = load_reference(deck_path)
    cad_points, cad_triangles, flat = butterfly(points, faces)
    _WORKER = {"row": row, "points": points, "faces": faces, "reference": Surface(points, faces),
               "cad_points": cad_points, "cad_triangles": cad_triangles, "cad_flat": flat}


def ground_truth(changes_dir, design, row, points):
    """(rows, dz) of design N<design> from the generator's csv, checked against the deck."""
    table = pd.read_csv(Path(changes_dir) / f"N{design}_changed_nodes.csv")
    rows = np.array([row[int(n)] for n in table["label"]], dtype=np.int64)
    error = np.abs(points[rows] - table[["x", "y", "z_design6"]].to_numpy(np.float64)).max()
    if error > 0.05:
        raise ValueError(f"N{design}: csv base coordinates differ from design 6 by {error:.3g} mm")
    return rows, table["dz"].to_numpy(np.float64)


REMESH_ANGLES_DEG = (FEATURE_ANGLE_DEG, 22.0, 18.0, 25.0, 15.0)  # gmsh retries; see remesh()


def remesh(points, faces, size_mm):
    """Independent quad-dominant gmsh remesh: (points, faces, patch angle used).

    gmsh's discrete reparametrization occasionally fails on a patch layout
    ("Wrong topology of boundary mesh") or leaves a patch unmeshed; the next patch
    angle in REMESH_ANGLES_DEG is tried then.
    """
    errors = []
    for angle in REMESH_ANGLES_DEG:
        try:
            return (*_remesh_once(points, faces, size_mm, angle), angle)
        except Exception as error:  # gmsh raises plain Exception
            errors.append(f"{angle:g} deg: {error}")
    raise RuntimeError("gmsh could not remesh at any patch angle: " + "; ".join(errors))


def _remesh_once(points, faces, size_mm, angle_deg):
    import gmsh

    triangles = triangulate(faces)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("edited")
        entity = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, entity, list(range(1, len(points) + 1)), points.ravel().tolist())
        gmsh.model.mesh.addElementsByType(entity, 2, [], (triangles + 1).ravel().tolist())
        gmsh.model.mesh.classifySurfaces(math.radians(angle_deg), True, True, math.pi)
        gmsh.model.mesh.createGeometry()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8 * size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        empty = [tag for _, tag in gmsh.model.getEntities(2)
                 if not sum(len(n) for n in gmsh.model.mesh.getElements(2, tag)[2])]
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        types, _, element_nodes = gmsh.model.mesh.getElements(2)
    finally:
        gmsh.finalize()
    if empty:
        raise RuntimeError(f"gmsh left surfaces {empty} unmeshed")
    tags = np.asarray(tags, dtype=np.int64)
    lookup = np.full(tags.max() + 1, -1, dtype=np.int64)
    lookup[tags] = np.arange(len(tags))
    blocks = [lookup[np.asarray(nodes, dtype=np.int64)].reshape(-1, {2: 3, 3: 4}[int(kind)])
              for kind, nodes in zip(types, element_nodes)]
    used = np.unique(np.concatenate([b.ravel() for b in blocks]))
    compact = np.full(len(tags), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    new_points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[used]
    new_faces = [tuple(f) for b in blocks for f in compact[b].tolist()]
    return new_points, new_faces


def area_ratio(a: Surface, b: Surface) -> float:
    def area(s):
        c = s.points[s.triangles]
        return 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1).sum()
    return float(area(a) / area(b))


def shuffled(points, faces, rng):
    """Same mesh with node order and element order permuted; returns (points, faces, perm)."""
    perm = rng.permutation(len(points))  # new node j is old node perm[j]
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    new_faces = [tuple(inverse[list(f)].tolist()) for f in faces]
    order = rng.permutation(len(new_faces))
    return points[perm], [new_faces[i] for i in order], perm


def interpolate(surface: Surface, values, query):
    """Node ``values`` of ``surface`` at the closest points to ``query`` (barycentric)."""
    closest, triangle = surface.closest(query)
    corners = surface.points[surface.triangles[triangle]]
    v0, v1, v2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], closest - corners[:, 0]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    determinant = d00 * d11 - d01 * d01
    determinant = np.where(np.abs(determinant) > 0, determinant, 1.0)
    b = (d11 * d20 - d01 * d21) / determinant
    c = (d00 * d21 - d01 * d20) / determinant
    weights = np.clip(np.stack((1 - b - c, b, c), axis=1), 0, 1)
    weights /= weights.sum(1, keepdims=True)
    return (weights * values[surface.triangles[triangle]]).sum(1)


def gap(nodes: Surface, other: Surface):
    """Distance (mm) from each node to the other surface: the change a detector can see."""
    return np.linalg.norm(nodes.points - other.closest(nodes.points)[0], axis=1)


def vertex_normals(surface: Surface):
    """One incident triangle's unit normal per node (the noise direction; sign irrelevant)."""
    normals = np.zeros_like(surface.points)
    for corner in range(3):
        normals[surface.triangles[:, corner]] = surface.normals
    return normals


def dilate(mask, edges):
    out = mask.copy()
    out[edges[mask[edges[:, 0]], 1]] = True
    out[edges[mask[edges[:, 1]], 0]] = True
    return out


def score(pred, truth, dz_abs, observable, edges, distance=None):
    """distance: the detector's node-to-surface distance, to size the false flags."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp, fn = pred & truth, pred & ~truth, ~pred & truth
    ring1 = dilate(truth, edges)
    far, far2 = fp & ~ring1, fp & ~dilate(ring1, edges)
    out = {"n_true": int(truth.sum()), "n_pred": int(pred.sum()), "tp": int(tp.sum()),
           "fp": int(fp.sum()), "fn": int(fn.sum()), "fp_far": int(far.sum()), "fp_far2": int(far2.sum())}
    if distance is not None:
        out["fp_far_distance_max_mm"] = float(distance[far].max()) if far.any() else 0.0
        out["fp_far2_distance_max_mm"] = float(distance[far2].max()) if far2.any() else 0.0
    out["precision"] = out["tp"] / out["n_pred"] if out["n_pred"] else float("nan")
    out["precision_near"] = (out["n_pred"] - out["fp_far"]) / out["n_pred"] if out["n_pred"] else float("nan")
    out["recall"] = out["tp"] / out["n_true"] if out["n_true"] else float("nan")
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    energy = np.square(dz_abs)
    out["energy_recall"] = float(energy[tp].sum() / energy[truth].sum()) if truth.any() else float("nan")
    # Truth nodes that are not off the other surface at all: tangential slides.
    out["fn_tangential"] = int((fn & (observable < FLOOR_MM)).sum())
    for k, (low, high) in enumerate(DZ_BANDS):
        band = truth & (dz_abs >= low) & (dz_abs < high)
        out[f"band{k}_n"], out[f"band{k}_tp"] = int(band.sum()), int((band & pred).sum())
    return out


def displacement_scores(detection, hit, dz_true, observable):
    """Token displacement (closest edited point - design-6 node) vs the generator's (0, 0, dz)."""
    estimate = -detection.displacement[hit]
    if not len(estimate):
        return {}
    truth = np.zeros_like(estimate)
    truth[:, 2] = dz_true
    spread = np.square(dz_true - dz_true.mean()).sum()
    return {
        "disp_rmse_mm": float(np.sqrt(np.square(estimate - truth).sum(1).mean())),
        "disp_z_r2": float(1 - np.square(estimate[:, 2] - dz_true).sum() / spread) if spread > 0 else float("nan"),
        "disp_normal_rmse_mm": float(np.sqrt(np.square(np.linalg.norm(estimate, axis=1) - observable).mean())),
        "disp_ratio_median": float(np.median(np.linalg.norm(estimate, axis=1) / np.abs(dz_true))),
    }


def naive_nearest_node(reference: Surface, edited_points):
    """Baseline: distance from each design-6 node to the nearest edited node, robust threshold."""
    distance, _ = cKDTree(edited_points).query(reference.points)
    median = np.median(distance)
    sigma = MAD_TO_SIGMA * np.median(np.abs(distance - median))
    low, high = max(median + 3 * sigma, FLOOR_MM), max(median + 6 * sigma, 2 * FLOOR_MM)
    return hysteresis(distance > low, distance > high, reference.edges(), len(distance), 10)


def run_design(job):
    design, changes_dir, noise_levels, sizes = job
    w = _WORKER
    base_points, base_faces, reference = w["points"], w["faces"], w["reference"]
    rng = np.random.default_rng(1000 + design)
    rows, dz = ground_truth(changes_dir, design, w["row"], base_points)
    dz_full = np.zeros(len(base_points))
    dz_full[rows] = dz
    truth_reference = dz_full != 0
    edited_points = base_points.copy()
    edited_points[:, 2] += dz_full
    true_edited = Surface(edited_points, base_faces)
    cad_points, cad_triangles, _ = butterfly(edited_points, base_faces, w["cad_flat"])
    if not np.array_equal(cad_triangles, w["cad_triangles"]):
        raise RuntimeError("Butterfly subdivision changed connectivity between design 6 and the edit")
    cad_dz = cad_points[:, 2] - w["cad_points"][:, 2]
    cad_edited = Surface(cad_points, cad_triangles)
    # Geometric change at each design-6 node (from the truth): distance to the true edited surface.
    observable_reference = gap(reference, true_edited)
    base_edges = reference.edges()
    records, examples, timing = [], {}, {}

    def evaluate(test, noise, points, faces, truth_new, dz_new, observable_new, store):
        points, faces, perm = shuffled(points, faces, rng)
        edited = Surface(points, faces)
        on_reference, on_edited = detect_edit(reference, edited)
        pred_new = np.zeros(len(perm), bool)
        pred_new[perm] = on_edited.changed
        distance_new = np.zeros(len(perm))
        distance_new[perm] = on_edited.distance
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(len(perm))
        new_edges = perm[edited.edges()]  # back to the unshuffled node order
        hit = on_reference.changed & truth_reference
        common = {"design": design, "test": test, "noise_mm": noise, "nodes_edited": len(points)}
        records.append({**common, "side": "reference",
                        **score(on_reference.changed, truth_reference, np.abs(dz_full), observable_reference,
                                base_edges, on_reference.distance),
                        "sigma_mm": on_reference.sigma, "tau_low_median_mm": float(np.median(on_reference.tau_low)),
                        **displacement_scores(on_reference, hit, dz_full[hit], observable_reference[hit])})
        records.append({**common, "side": "edited",
                        **score(pred_new, truth_new, np.abs(dz_new), observable_new, new_edges, distance_new),
                        "sigma_mm": on_edited.sigma, "tau_low_median_mm": float(np.median(on_edited.tau_low))})
        if test != "same_mesh":
            naive = naive_nearest_node(reference, points)
            records.append({**common, "side": "reference_naive",
                            **score(naive, truth_reference, np.abs(dz_full), observable_reference, base_edges)})
        if store:
            examples.update({"reference_pred": on_reference.changed, "edited_points": points[inverse],
                             "edited_pred": pred_new, "edited_truth": truth_new,
                             "naive_pred": naive})

    t = time.time()
    observable_same = gap(true_edited, reference)
    evaluate("same_mesh", 0.0, edited_points, base_faces, truth_reference, dz_full, observable_same, False)
    timing["same_mesh"] = time.time() - t

    for size in sizes:
        t = time.time()
        new_points, new_faces, angle = remesh(edited_points, base_faces, size)
        timing[f"remesh_h{size:g}"] = time.time() - t
        timing[f"patch_angle_h{size:g}"] = angle
        flat = Surface(new_points, new_faces)
        dz_flat = interpolate(true_edited, dz_full, new_points)
        evaluate(f"flat_remesh_h{size:g}", 0.0, new_points, new_faces, np.abs(dz_flat) >= FLOOR_MM, dz_flat,
                 gap(flat, reference), False)
        # CAD stand-in: the same nodes on the smooth edited surface.
        projected, _ = cad_edited.closest(new_points)
        cad_mesh = Surface(projected, new_faces)
        dz_cad = interpolate(cad_edited, cad_dz, projected)
        truth_cad = np.abs(dz_cad) >= FLOOR_MM
        observable_cad = gap(cad_mesh, reference)
        test = f"cad_remesh_h{size:g}"
        normals = vertex_normals(cad_mesh)
        for noise in (noise_levels if size == sizes[0] else [0.0]):
            noisy = projected + noise * rng.standard_normal(len(projected))[:, None] * normals
            store = design in EXAMPLE_DESIGNS and (test, noise) == EXAMPLE_TEST
            evaluate(test, noise, noisy, new_faces, truth_cad, dz_cad, observable_cad, store)
        if size == sizes[0]:
            timing["area_ratio"] = area_ratio(cad_mesh, reference)
    if design in EXAMPLE_DESIGNS:
        examples["truth_reference"] = truth_reference
        examples["dz"] = dz_full
    return records, (design, examples), timing


def sagitta_summary(reference: Surface):
    node = reference.curvature(45.0)[0] * reference.element_size()[0] ** 2 / 8.0
    return {"median_mm": float(np.median(node)), "p90_mm": float(np.percentile(node, 90)),
            "p99_mm": float(np.percentile(node, 99)), "max_mm": float(node.max())}


def summarize(frame):
    rows = []
    for (test, noise, side), group in frame.groupby(["test", "noise_mm", "side"], sort=False):
        tp, fp, fn = group["tp"].sum(), group["fp"].sum(), group["fn"].sum()
        row = {"test": test, "noise_mm": noise, "side": side, "designs": len(group),
               "precision": tp / (tp + fp) if tp + fp else float("nan"),
               "precision_near": 1 - group["fp_far"].sum() / (tp + fp) if tp + fp else float("nan"),
               "recall": tp / (tp + fn), "recall_min": group["recall"].min(),
               "energy_recall": float(np.average(group["energy_recall"])),
               "energy_recall_min": group["energy_recall"].min(),
               "fp_total": int(fp), "fp_far_total": int(group["fp_far"].sum()),
               "fp_far2_total": int(group["fp_far2"].sum()), "fn_total": int(fn),
               "fn_tangential_total": int(group["fn_tangential"].sum()),
               "designs_with_fp_far": int((group["fp_far"] > 0).sum())}
        for k in range(len(DZ_BANDS)):
            n = group[f"band{k}_n"].sum()
            row[f"recall_band{k}"] = group[f"band{k}_tp"].sum() / n if n else float("nan")
        for column in ("fp_far_distance_max_mm", "fp_far2_distance_max_mm"):
            if column in group and group[column].notna().any():
                row[column] = float(group[column].max())
        for column in ("sigma_mm", "tau_low_median_mm", "nodes_edited", "disp_rmse_mm", "disp_z_r2",
                       "disp_normal_rmse_mm", "disp_ratio_median"):
            if column in group and group[column].notna().any():
                row[f"{column}_median"] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--designs", default="1-72", help="e.g. 1-72 or 54,55")
    parser.add_argument("--noise-mm", default="0,0.03,0.1,0.3", help="Gaussian noise on cad_remesh at the first size")
    parser.add_argument("--remesh-mm", default="14,20")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    designs = []
    for part in args.designs.split(","):
        low, _, high = part.partition("-")
        designs.extend(range(int(low), int(high or low) + 1))
    noise_levels = [float(v) for v in args.noise_mm.split(",")]
    sizes = [float(v) for v in args.remesh_mm.split(",")]
    deck = args.data_root / "inp_files" / "HoodImpact_1.inp"
    changes_dir = args.data_root / "design_changes"
    output = args.output_dir or ROOT / "runs" / "change_node_detection" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    _, _, base_points, base_faces = load_reference(deck)
    sagitta = sagitta_summary(Surface(base_points, base_faces))
    print(f"Design 6 inner panel: {len(base_points)} nodes, {len(base_faces)} elements; sagitta per node "
          f"median {sagitta['median_mm']:.3f}, p90 {sagitta['p90_mm']:.3f}, p99 {sagitta['p99_mm']:.3f} mm")

    start = time.time()
    records, examples, timing = [], {}, []
    jobs = [(d, changes_dir, noise_levels, sizes) for d in designs]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(deck,)) as pool:
        for done, (rows, (design, example), seconds) in enumerate(pool.map(run_design, jobs), 1):
            records.extend(rows)
            timing.append(seconds)
            if example:
                examples[design] = example
            if done % 12 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} designs, {time.time() - start:.0f} s", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_design.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    flat = {"base_points": base_points, "base_faces": np.array([list(f) + [-1] * (4 - len(f)) for f in base_faces])}
    for design, example in examples.items():
        flat.update({f"N{design}_{key}": value for key, value in example.items()})
    np.savez_compressed(output / "examples.npz", **flat)
    timing = pd.DataFrame(timing)
    angles = [c for c in timing.columns if c.startswith("patch_angle")]
    for column in angles:
        print(f"gmsh patch angle used ({column}): {timing[column].value_counts().to_dict()}")
    (output / "summary.json").write_text(json.dumps({
        "designs": designs, "noise_mm": noise_levels, "remesh_mm": sizes, "floor_mm": FLOOR_MM,
        "dz_bands_mm": [list(b) for b in DZ_BANDS], "sagitta_design6": sagitta,
        "cad_remesh_area_ratio_min": float(timing["area_ratio"].min()),
        "gmsh_patch_angle_counts": {c: {f"{k:g}": int(v) for k, v in timing[c].value_counts().items()}
                                    for c in angles},
        "median_seconds": timing.drop(columns=["area_ratio", *angles]).median().to_dict(),
        "summary": summary.to_dict(orient="records"),
    }, indent=1, default=float))
    pd.set_option("display.width", 250)
    columns = ["test", "noise_mm", "side", "precision", "precision_near", "recall", "recall_min", "energy_recall_min",
               "fp_far_total", "fp_far2_total", "fp_far_distance_max_mm", "designs_with_fp_far", "fn_tangential_total", "recall_band0", "recall_band1",
               "recall_band2", "recall_band3"]
    print(summary[columns].round(3).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
