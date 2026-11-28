"""Check the ID-free change detector (detect_change_nodes.py) on the B1-B4 benchmarks.

Each benchmark family is one mesh; an edit morphs it (benchmarks/local_edit/), so
the generator's change node IDs are exact. They mix three kinds of node:
  interior        moved by the morph inside the solid: no geometry changed there
  boundary, slid  moved along an unchanged boundary (a fin wall that got taller)
  boundary, off   moved off the old boundary: the geometric change
Only the last kind exists without node IDs. Truth here = that kind, on both sides:
  reference  base boundary nodes off the edited boundary (geometry removed or moved)
  edited     edited boundary nodes off the base boundary (geometry added or moved)
computed from the exact (ID-built) geometry. Node IDs are used only to build the
edited designs and to score; the detector sees shuffled boundary points and cells.

Tests per design, the base mesh being the reference:
  morph          the generator's edited mesh, nodes shuffled
  remesh         the edited boundary re-discretized: 2D curves resampled at 1.37x
                 the base spacing between kept corners; B2's skin remeshed by gmsh
  remesh_cad     the same nodes on a smooth stand-in for the CAD boundary (2D:
                 cubic spline through the edited boundary nodes between corners;
                 B2: butterfly subdivision, see verify_change_node_detection.py)
  remesh_cad + noise  Gaussian noise along the normal, sigma = 1% / 3% of the
                 boundary spacing
A no-edit pair (base vs its own remesh_cad) checks the false-flag rate without labels.

  python verify_change_node_detection_benchmarks.py [--root D:/.../LocalEditBenchmarks/pilot]
Writes runs/change_node_detection_benchmarks/<stamp>/: per_design.csv, summary.csv,
composition.csv, examples.npz.
"""
from __future__ import annotations

import argparse
import glob
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import verify_change_node_detection as hood
from detect_change_nodes import Surface, detect_edit, mesh_boundary

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"D:\Papers\Hood_Impact\LocalEditBenchmarks\pilot")
BENCHMARKS = ("B1_plate", "B2_elbow", "B3_heatsink", "B4_cantilever")
FLOOR_MM = 0.01
RESAMPLE = 1.37          # new boundary spacing / base spacing: no node lands on an old one
CORNER_DEG = 30.0        # 2D resampling keeps nodes with a sharper turn (a remesher keeps corners)
NOISE_FRACTIONS = (0.01, 0.03)


def load(path):
    z = np.load(path)
    return {"xyz": np.asarray(z["xyz"], np.float64), "elements": np.asarray(z["elements"], np.int64) - 1,
            "change": np.asarray(z["change_node_ids"], np.int64) - 1}


def shuffle_surface(surface: Surface, rng):
    """Same boundary with node order and cell order permuted; returns (Surface, perm)."""
    perm = rng.permutation(len(surface.points))
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    faces = [tuple(inverse[list(f)]) for f in surface.faces]
    faces = [faces[i] for i in rng.permutation(len(faces))]
    return Surface(surface.points[perm], faces), perm


# ----------------------------------------------------------------- 2D boundary resampling
def loops(surface: Surface):
    """Closed node loops of a curve boundary (every node has two segments)."""
    neighbours = [[] for _ in surface.points]
    for a, b in surface.cells:
        neighbours[a].append(b)
        neighbours[b].append(a)
    if any(len(n) != 2 for n in neighbours):
        raise ValueError("2D boundary is not a set of closed loops")
    seen, out = np.zeros(len(surface.points), bool), []
    for start in range(len(surface.points)):
        if seen[start]:
            continue
        loop, previous, node = [start], -1, start
        seen[start] = True
        while True:
            nxt = neighbours[node][0] if neighbours[node][0] != previous else neighbours[node][1]
            if nxt == start:
                break
            loop.append(nxt)
            seen[nxt] = True
            previous, node = node, nxt
        out.append(np.array(loop))
    return out


def resample_curve(surface: Surface, spacing, smooth):
    """New closed-loop boundary through the same shape: corners kept, runs between them
    resampled at ``spacing`` (linear along the polyline, or a cubic spline if smooth)."""
    points, faces = [], []
    for loop in loops(surface):
        p = surface.points[loop, :2]
        before, after = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
        cosine = np.einsum("ij,ij->i", before, after) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1))
        corner = np.flatnonzero(np.arccos(np.clip(cosine, -1, 1)) > math.radians(CORNER_DEG))
        if len(corner) == 0:
            runs = [np.append(np.arange(len(p)), 0)]  # one closed smooth run
        else:
            runs = [np.arange(c, (corner[(k + 1) % len(corner)] - c) % len(p) + c + 1) % len(p)
                    if len(corner) > 1 else np.append(np.arange(c, c + len(p)) % len(p), c)
                    for k, c in enumerate(corner)]
        start = len(points)
        for run in runs:
            q = p[run]
            arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))))
            count = max(int(round(arc[-1] / spacing)), 1)
            # Interior samples offset by half a step: none falls on an old node.
            s = (np.arange(count) + 0.5) * arc[-1] / count
            if smooth and len(q) >= 4:
                closed = len(corner) == 0
                spline = CubicSpline(arc, q, bc_type="periodic" if closed else "not-a-knot")
                new = spline(s)
            else:
                new = np.column_stack([np.interp(s, arc, q[:, i]) for i in range(2)])
            if len(corner):
                points.append(q[0])  # the corner itself
            points.extend(new)
        n = len(points) - start
        faces.extend((start + i, start + (i + 1) % n) for i in range(n))
    return Surface(np.array(points), faces)


# ----------------------------------------------------------------- 3D skin (B2)
def remesh_skin(surface: Surface, spacing, smooth, flat=None):
    """gmsh remesh of a 3D boundary surface, optionally projected onto its butterfly CAD."""
    new_points, new_faces, _ = hood.remesh(surface.points, surface.faces, spacing)
    if smooth:
        cad_points, cad_triangles, flat = hood.butterfly(surface.points, surface.faces, flat)
        new_points, _ = Surface(cad_points, [tuple(t) for t in cad_triangles]).closest(new_points)
    return Surface(new_points, new_faces), flat


def normals_at_nodes(surface: Surface):
    normals = np.zeros_like(surface.points)
    for corner in range(surface.cells.shape[1]):
        normals[surface.cells[:, corner]] = surface.normals
    return normals


def carry(surface: Surface, values, query):
    """Node ``values`` of ``surface`` at the closest points to ``query``: linear along a
    segment, barycentric in a triangle."""
    if surface.dim == 2:
        return hood.interpolate(surface, values, query)
    closest, cell = surface.closest(query)
    a, b = surface.points[surface.cells[cell, 0]], surface.points[surface.cells[cell, 1]]
    t = np.clip(np.einsum("ij,ij->i", closest - a, b - a) / np.maximum(np.einsum("ij,ij->i", b - a, b - a), 1e-24),
                0, 1)
    return (1 - t) * values[surface.cells[cell, 0]] + t * values[surface.cells[cell, 1]]


def gap(nodes: Surface, other: Surface):
    return np.linalg.norm(nodes.points - other.closest(nodes.points)[0], axis=1)


def ring(mask, edges):
    out = mask.copy()
    out[edges[mask[edges[:, 0]], 1]] = True
    out[edges[mask[edges[:, 1]], 0]] = True
    return out


def score(pred, truth, deviation, edges, distance):
    tp, fp, fn = pred & truth, pred & ~truth, ~pred & truth
    far = fp & ~ring(truth, edges)
    energy = np.square(deviation)
    return {"n_true": int(truth.sum()), "n_pred": int(pred.sum()), "tp": int(tp.sum()), "fp": int(fp.sum()),
            "fn": int(fn.sum()), "fp_far": int(far.sum()),
            "fp_far_distance_max_mm": float(distance[far].max()) if far.any() else 0.0,
            "fn_deviation_max_mm": float(deviation[fn].max()) if fn.any() else 0.0,
            "energy_recall": float(energy[tp].sum() / energy[truth].sum()) if truth.any() else float("nan")}


def run_benchmark(folder: Path, rng):
    base = load(next(folder.glob("base/*_mesh.npz")))
    reference, reference_nodes = mesh_boundary(base["xyz"], base["elements"])
    spacing = float(np.median(reference.element_size()[0])) * RESAMPLE
    is_curve = reference.dim == 1
    records, composition, examples = [], [], {}

    def remeshed(surface, smooth, flat=None):
        if is_curve:
            return resample_curve(surface, spacing, smooth), flat
        return remesh_skin(surface, spacing, smooth, flat)

    # No-edit pair: base vs its own smooth remesh. No labels needed.
    null, flat = remeshed(reference, True)
    on_reference, on_edited = detect_edit(reference, null)
    records.append({"benchmark": folder.name, "design": "base (no edit)", "test": "remesh_cad", "noise_fraction": 0.0,
                    "side": "reference", "n_true": 0, "n_pred": int(on_reference.changed.sum()),
                    "fp": int(on_reference.changed.sum()), "fp_far": int(on_reference.changed.sum()),
                    "nodes": len(reference.points)})
    records.append({"benchmark": folder.name, "design": "base (no edit)", "test": "remesh_cad", "noise_fraction": 0.0,
                    "side": "edited", "n_true": 0, "n_pred": int(on_edited.changed.sum()),
                    "fp": int(on_edited.changed.sum()), "fp_far": int(on_edited.changed.sum()),
                    "nodes": len(null.points)})

    for path in sorted(folder.glob("e*/*_mesh.npz")):
        design = path.parent.name
        edit = load(path)
        if not np.array_equal(edit["elements"], base["elements"]):
            raise ValueError(f"{design}: not a morph of the base mesh")
        true_edited, edited_nodes = mesh_boundary(edit["xyz"], edit["elements"])
        if not np.array_equal(edited_nodes, reference_nodes):
            raise ValueError(f"{design}: boundary nodes differ from the base")
        moved = np.zeros(len(base["xyz"]), bool)
        moved[edit["change"]] = True
        move = np.linalg.norm(edit["xyz"] - base["xyz"], axis=1)
        # Geometric truth from the exact geometry.
        deviation_reference = gap(reference, true_edited)       # base boundary node -> edited boundary
        deviation_edited = gap(true_edited, reference)          # edited boundary node -> base boundary
        truth_reference = deviation_reference >= FLOOR_MM
        on_boundary = np.isin(edit["change"], reference_nodes)
        boundary_moved = moved[reference_nodes]
        composition.append({
            "benchmark": folder.name, "design": design, "id_change_nodes": int(moved.sum()),
            "interior": int((~on_boundary).sum()), "boundary": int(on_boundary.sum()),
            "boundary_off_reference": int((boundary_moved & truth_reference).sum()),
            "boundary_new_off_base": int((boundary_moved & (deviation_edited >= FLOOR_MM)).sum()),
            "boundary_slid_only": int((boundary_moved & ~truth_reference & (deviation_edited < FLOOR_MM)).sum()),
            "max_move_mm": float(move.max())})
        if not (truth_reference <= boundary_moved).all():
            raise RuntimeError(f"{design}: a base node that did not move is off the edited boundary")

        tests = [("morph", 0.0, None), ("remesh", 0.0, False), ("remesh_cad", 0.0, True)]
        tests += [("remesh_cad", f, True) for f in NOISE_FRACTIONS]
        cache = {}
        for test, noise, smooth in tests:
            if smooth is None:
                candidate = true_edited
                truth_edited, dev_edited = deviation_edited >= FLOOR_MM, deviation_edited
            else:
                if smooth not in cache:
                    surface, _ = remeshed(true_edited, smooth, flat)
                    # The edit's own offset carried to the new nodes (not their distance
                    # to the base, which on a curved part includes every node's sag).
                    cache[smooth] = (surface, carry(true_edited, deviation_edited, surface.points))
                surface, clean = cache[smooth]
                normals = normals_at_nodes(surface)
                points = surface.points + noise * spacing / RESAMPLE * rng.standard_normal(len(surface.points))[:, None] * normals
                candidate = Surface(points, surface.faces)
                truth_edited, dev_edited = clean >= FLOOR_MM, clean
            shuffled, perm = shuffle_surface(candidate, rng)
            on_reference, on_edited = detect_edit(reference, shuffled)
            pred_edited = np.zeros(len(perm), bool)
            pred_edited[perm] = on_edited.changed
            distance_edited = np.zeros(len(perm))
            distance_edited[perm] = on_edited.distance
            common = {"benchmark": folder.name, "design": design, "test": test, "noise_fraction": noise}
            records.append({**common, "side": "reference", "nodes": len(reference.points),
                            **score(on_reference.changed, truth_reference, deviation_reference, reference.edges(),
                                    on_reference.distance),
                            "id_boundary_found": int((on_reference.changed & boundary_moved).sum()),
                            "id_boundary": int(boundary_moved.sum()),
                            "displacement_error_mm": float(np.sqrt(np.mean(np.square(
                                on_reference.distance - deviation_reference)[on_reference.changed])))
                            if on_reference.changed.any() else float("nan")})
            records.append({**common, "side": "edited", "nodes": len(candidate.points),
                            **score(pred_edited, truth_edited, dev_edited, perm[shuffled.edges()], distance_edited)})
            if test == "remesh_cad" and noise == NOISE_FRACTIONS[0]:
                examples[f"{folder.name}/{design}/reference_points"] = reference.points
                examples[f"{folder.name}/{design}/reference_truth"] = truth_reference
                examples[f"{folder.name}/{design}/reference_pred"] = on_reference.changed
                examples[f"{folder.name}/{design}/edited_points"] = candidate.points
                examples[f"{folder.name}/{design}/edited_truth"] = truth_edited
                examples[f"{folder.name}/{design}/edited_pred"] = pred_edited
    return records, composition, examples, spacing


def summarize(frame):
    rows = []
    edited = frame[frame["design"] != "base (no edit)"]
    for (bench, test, noise, side), g in edited.groupby(["benchmark", "test", "noise_fraction", "side"], sort=False):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        rows.append({"benchmark": bench, "test": test, "noise_fraction": noise, "side": side,
                     "recall": tp / (tp + fn) if tp + fn else float("nan"),
                     "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "precision_near": 1 - g["fp_far"].sum() / (tp + fp) if tp + fp else float("nan"),
                     "energy_recall_min": g["energy_recall"].min(),
                     "fn_deviation_max_mm": g["fn_deviation_max_mm"].max(),
                     "fp_far_total": int(g["fp_far"].sum()),
                     "id_boundary_recall": (g["id_boundary_found"].sum() / g["id_boundary"].sum())
                     if "id_boundary" in g and g["id_boundary"].notna().any() else float("nan"),
                     "displacement_error_mm": g["displacement_error_mm"].median() if "displacement_error_mm" in g
                     else float("nan")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "runs" / "change_node_detection_benchmarks" / datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    records, composition, examples, spacings = [], [], {}, {}
    for name in BENCHMARKS:
        folder = args.root / name
        if not folder.exists():
            print(f"skip {name}: no {folder}")
            continue
        r, c, e, spacing = run_benchmark(folder, rng)
        records += r
        composition += c
        examples.update(e)
        spacings[name] = spacing
        print(f"{name}: {len(c)} edits, resample spacing {spacing:.2f} mm", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_design.csv", index=False)
    composition = pd.DataFrame(composition)
    composition.to_csv(output / "composition.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    np.savez_compressed(output / "examples.npz", **examples)
    pd.set_option("display.width", 250)
    print("\nWhat the node-ID change sets contain:")
    print(composition.groupby("benchmark")[["id_change_nodes", "interior", "boundary", "boundary_off_reference",
                                            "boundary_new_off_base", "boundary_slid_only"]].sum().to_string())
    print("\nNo-edit pair (base vs its own smooth remesh), false flags:")
    null = frame[frame["design"] == "base (no edit)"]
    print(null[["benchmark", "side", "nodes", "fp"]].to_string(index=False))
    print("\nDetection vs the geometric truth:")
    print(summary.round(3).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
