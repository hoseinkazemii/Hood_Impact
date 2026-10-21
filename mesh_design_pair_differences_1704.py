"""Node-level geometry differences between every pair of the 12 hood designs.

Two designs in the same mesh family (0-5 or 6-11) share node IDs and shell
connectivity, so their difference is exact: the displacement of each node ID
from one deck to the other. Across families there is no node correspondence;
there we report, for every node of one design, the distance to the closest
point on the *same part's* surface (outer, inner, hinges) of the other design.
That distance is never exactly zero, because two different discretisations of
a curved surface do not coincide, so a small floor is expected.

Outputs (in --out-dir):
  pair_summary.csv / pair_summary.json   one row per pair (66 rows)
  viewer_data.bin (+ .b64.txt) + viewer_meta.json  packed meshes and fields for the 3D viewer
                                         (gzip; per-design coordinates and exact-pair
                                         deltas stored sparsely above 1e-4 mm)
"""
from __future__ import annotations

import gzip
import argparse
import base64
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from abaqus_scripts.inp_geom import Deck
from mesh_design_clusters import GEOMETRY_CLUSTERS

ROOT = Path(__file__).resolve().parent
PARTS = {"outer": "Hood_Outer-1-2", "inner": "Hood_Inner-1-2", "hinges": "Hinges-1-2"}
THRESHOLDS_MM = (0.001, 0.01, 0.1, 1.0, 5.0)
DISTANCE_QUANTUM_MM = 0.01
# Designs 0 and 6 print coordinates at full double precision, the others are
# rounded near 1e-6 mm; deltas below this are rounding, not geometry.
SPARSE_TOLERANCE_MM = 1e-4


def load_design(data_dir, design):
    deck = Deck(data_dir / "inp_files" / f"HoodImpact_{142 * design + 1}.inp")
    impactor = deck.impactor_node_ids()
    elements = {}
    for part, elset in PARTS.items():
        for eid in deck.elsets[elset]:
            if set(deck.elems[eid]) & impactor:
                raise ValueError(f"Design {design}: headform node in {elset}")
            elements[eid] = (part, deck.elems[eid])
    node_ids = np.array(sorted({n for _, conn in elements.values() for n in conn}))
    index = {n: i for i, n in enumerate(node_ids)}
    xyz = np.array([deck.nodes[n] for n in node_ids])
    # A node shared by two parts is labelled with the first part listed.
    part_of = np.full(len(node_ids), -1)
    for code, part in reversed(list(enumerate(PARTS))):
        for _, conn in (v for v in elements.values() if v[0] == part):
            part_of[[index[n] for n in conn]] = code
    eids = sorted(elements)
    conn = [tuple(index[n] for n in elements[e][1]) for e in eids]
    elem_part = np.array([list(PARTS).index(elements[e][0]) for e in eids])
    return dict(design=design, node_ids=node_ids, xyz=xyz, part=part_of,
                elem_ids=np.array(eids), conn=conn, elem_part=elem_part)


def part_surface(geom, code):
    cells = [c for c, p in zip(geom["conn"], geom["elem_part"]) if p == code]
    faces = np.concatenate([[len(c), *c] for c in cells])
    return pv.PolyData(geom["xyz"], faces).clean().triangulate()


def same_mesh(a, b):
    return (np.array_equal(a["node_ids"], b["node_ids"]) and np.array_equal(a["elem_ids"], b["elem_ids"])
            and a["conn"] == b["conn"])


def surface_distance(src, dst):
    """|closest-surface distance| from every node of src to dst, part by part."""
    out = np.empty(len(src["xyz"]))
    for code in range(len(PARTS)):
        mask = src["part"] == code
        dist = pv.PolyData(src["xyz"][mask]).compute_implicit_distance(part_surface(dst, code))
        out[mask] = np.abs(dist["implicit_distance"])
    return out


def stats(values, part):
    row = dict(n_nodes=int(len(values)), max_mm=float(values.max()), mean_mm=float(values.mean()),
               rms_mm=float(np.sqrt(np.mean(values ** 2))), n_exact_zero=int(np.sum(values == 0)))
    for t in THRESHOLDS_MM:
        row[f"n_gt_{t:g}mm"] = int(np.sum(values > t))
    for code, name in enumerate(PARTS):
        row[f"{name}_n_gt_0.001mm"] = int(np.sum(values[part == code] > 0.001))
        row[f"{name}_n_gt_0.1mm"] = int(np.sum(values[part == code] > 0.1))
        row[f"{name}_max_mm"] = float(values[part == code].max())
    return row


def triangles(geom):
    tris, part = [], []
    for c, p in zip(geom["conn"], geom["elem_part"]):
        tris.append(c[:3])
        part.append(p)
        if len(c) == 4:
            tris.append((c[0], c[2], c[3]))
            part.append(p)
    return np.array(tris, dtype=np.uint32), np.array(part, dtype=np.uint8)


def element_edges(geom):
    """Unique element edges (no quad diagonals) and the part each belongs to."""
    edges = {}
    for c, p in zip(geom["conn"], geom["elem_part"]):
        for k in range(len(c)):
            edges.setdefault(tuple(sorted((c[k], c[(k + 1) % len(c)]))), p)
    return np.array(list(edges), dtype=np.uint32), np.array(list(edges.values()), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/design_pair_mesh_differences_1704")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cluster_of = {d: name for name, members in GEOMETRY_CLUSTERS.items() for d in members}

    designs = [load_design(args.data_dir, d) for d in range(12)]
    # Mesh families: designs whose node IDs and connectivity are identical.
    family_of, reps = {}, []
    for g in designs:
        match = next((f for f, r in enumerate(reps) if same_mesh(designs[r], g)), None)
        if match is None:
            reps.append(g["design"])
            match = len(reps) - 1
        family_of[g["design"]] = match
    print("Mesh families:", {f: [d for d in family_of if family_of[d] == f] for f in range(len(reps))}, flush=True)

    center = np.concatenate([g["xyz"] for g in designs]).mean(0)
    blob, meta = bytearray(), dict(parts=list(PARTS), distance_quantum_mm=DISTANCE_QUANTUM_MM,
                                   clusters=GEOMETRY_CLUSTERS, families=[], designs=[], pairs=[],
                                   center_mm=center.tolist(), sparse_tolerance_mm=SPARSE_TOLERANCE_MM)

    def put(array):
        array = np.ascontiguousarray(array)
        while len(blob) % 4:
            blob.append(0)
        entry = dict(offset=len(blob), dtype=str(array.dtype), length=int(array.size))
        blob.extend(array.tobytes())
        return entry

    # Display base per family: a rounded deck, so other designs differ sparsely.
    reps = [[d for d in family_of if family_of[d] == f][1] for f in range(len(reps))]
    for f, r in enumerate(reps):
        g = designs[r]
        tris, tri_part = triangles(g)
        edges, edge_part = element_edges(g)
        meta["families"].append(dict(family=f, designs=[d for d in family_of if family_of[d] == f],
                                     n_nodes=len(g["xyz"]), n_elems=len(g["conn"]),
                                     triangles=put(tris), tri_part=put(tri_part),
                                     edges=put(edges), edge_part=put(edge_part),
                                     part=put(g["part"].astype(np.uint8)),
                                     node_ids=put(g["node_ids"].astype(np.uint32)),
                                     base_design=r, base_xyz=put((g["xyz"] - center).astype(np.float32))))
    for g in designs:
        base = designs[reps[family_of[g["design"]]]]
        moved = np.flatnonzero(np.linalg.norm(g["xyz"] - base["xyz"], axis=1) > SPARSE_TOLERANCE_MM)
        meta["designs"].append(dict(design=g["design"], cluster=cluster_of[g["design"]],
                                    family=family_of[g["design"]], n_moved_vs_base=int(len(moved)),
                                    moved_idx=put(moved.astype(np.uint32)),
                                    moved_xyz=put((g["xyz"][moved] - center).astype(np.float32))))

    rows = []
    for i, j in itertools.combinations(range(12), 2):
        a, b = designs[i], designs[j]
        row = dict(design_a=i, design_b=j, cluster_a=cluster_of[i], cluster_b=cluster_of[j])
        if family_of[i] == family_of[j]:
            delta = b["xyz"] - a["xyz"]
            disp = np.linalg.norm(delta, axis=1)
            row.update(method="exact_node_id_displacement", **stats(disp, a["part"]),
                       max_abs_dx_mm=float(np.abs(delta[:, 0]).max()),
                       max_abs_dy_mm=float(np.abs(delta[:, 1]).max()),
                       max_abs_dz_mm=float(np.abs(delta[:, 2]).max()))
            idx = np.flatnonzero(disp > SPARSE_TOLERANCE_MM)
            meta["pairs"].append(dict(a=i, b=j, method="exact", idx=put(idx.astype(np.uint32)),
                                      delta=put(delta[idx].astype(np.float32))))
        else:
            d_ab, d_ba = surface_distance(a, b), surface_distance(b, a)
            sa, sb = stats(d_ab, a["part"]), stats(d_ba, b["part"])
            row.update(method="closest_surface_distance_same_part", **sa,
                       **{f"reverse_{k}": v for k, v in sb.items()})
            q = lambda v: np.minimum(np.round(v / DISTANCE_QUANTUM_MM), 65535).astype(np.uint16)
            meta["pairs"].append(dict(a=i, b=j, method="surface", dist_a=put(q(d_ab)), dist_b=put(q(d_ba))))
        rows.append(row)
        print(f"{i:2d}-{j:2d} {row['method'][:5]}  max {row['max_mm']:8.3f}  rms {row['rms_mm']:7.3f}  "
              f">0.1mm {row['n_gt_0.1mm']:6d}/{row['n_nodes']}", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / "pair_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "pair_summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keep = ("max_mm", "rms_mm", "mean_mm", "n_nodes", *(f"n_gt_{t:g}mm" for t in THRESHOLDS_MM),
            *(f"{name}_n_gt_{t}mm" for name in PARTS for t in ("0.001", "0.1")))
    for p, r in zip(meta["pairs"], rows):
        # summary_a is measured on design a's nodes, summary_b on design b's.
        p["summary_a"] = {k: r[k] for k in keep}
        p["summary_b"] = {k: r.get(f"reverse_{k}", r[k]) for k in keep}
    packed = gzip.compress(bytes(blob), 9)
    (args.out_dir / "viewer_data.bin").write_bytes(packed)
    # Artifact hosting serves text but not raw binary, so the viewer reads this copy.
    (args.out_dir / "viewer_data.b64.txt").write_bytes(base64.b64encode(packed))
    (args.out_dir / "viewer_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"Wrote {args.out_dir} ({len(blob) / 1e6:.1f} MB binary)", flush=True)


if __name__ == "__main__":
    main()
