"""
Parse all 60 HoodImpact_*.inp files, extract the HOOD SHELL mesh (nodes used by
S3R/S4 shell elements, i.e. excluding impactor / reference / connector nodes),
fingerprint each geometry, and discover the true design grouping.

Outputs a cache (parsed meshes) + a printed report + a CSV mapping
run -> design / impact location.
"""

import os
import re
import json
import hashlib
import pickle

import numpy as np
import pandas as pd

INP_DIR = "./Data/HoodImpact_60_IndustryLike/inp_files"
IMPACT_CSV = "./Data/HoodImpact_60_IndustryLike/ImpactCoords_60.csv"
CACHE = "./Data/HoodImpact_60_IndustryLike/_parsed_hood_meshes.pkl"
N = 60


def parse_inp(path):
    """Return dict with full nodes {id:(x,y,z)} and shell element connectivity."""
    nodes = {}
    elems_s3 = []   # (n1,n2,n3)
    elems_s4 = []   # (n1,n2,n3,n4)

    mode = None  # 'node', 's3', 's4', or None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("*"):
                u = s.upper()
                if u == "*NODE":
                    mode = "node"
                elif u.startswith("*ELEMENT") and "S3R" in u:
                    mode = "s3"
                elif u.startswith("*ELEMENT") and ("TYPE=S4" in u.replace(" ", "")):
                    mode = "s4"
                else:
                    mode = None  # any other keyword ends the current block
                continue

            if mode == "node":
                p = s.split(",")
                if len(p) >= 4:
                    nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
            elif mode == "s3":
                p = s.split(",")
                if len(p) >= 4:
                    elems_s3.append((int(p[1]), int(p[2]), int(p[3])))
            elif mode == "s4":
                p = s.split(",")
                if len(p) >= 5:
                    elems_s4.append((int(p[1]), int(p[2]), int(p[3]), int(p[4])))

    return nodes, elems_s3, elems_s4


def hood_mesh(nodes, elems_s3, elems_s4):
    """Build a compact hood mesh: (coords array, remapped tri faces, quad faces)."""
    used = set()
    for e in elems_s3:
        used.update(e)
    for e in elems_s4:
        used.update(e)
    used = sorted(used)
    remap = {nid: i for i, nid in enumerate(used)}
    coords = np.array([nodes[nid] for nid in used], dtype=np.float64)
    tris = np.array([[remap[a] for a in e] for e in elems_s3], dtype=np.int64) if elems_s3 else np.zeros((0, 3), int)
    quads = np.array([[remap[a] for a in e] for e in elems_s4], dtype=np.int64) if elems_s4 else np.zeros((0, 4), int)
    return coords, tris, quads


def fingerprint(coords):
    """Order-independent hash of geometry (rounded coords)."""
    r = np.round(coords, 2)
    order = np.lexsort((r[:, 2], r[:, 1], r[:, 0]))
    b = r[order].tobytes()
    return hashlib.md5(b).hexdigest()[:12]


def main():
    impact = pd.read_csv(IMPACT_CSV)
    impact.columns = [c.strip() for c in impact.columns]

    meshes = {}
    recs = []
    for run in range(1, N + 1):
        path = os.path.join(INP_DIR, f"HoodImpact_{run}.inp")
        nodes, s3, s4 = parse_inp(path)
        coords, tris, quads = hood_mesh(nodes, s3, s4)
        meshes[run] = {"coords": coords, "tris": tris, "quads": quads}

        fp = fingerprint(coords)
        bb_min = coords.min(0)
        bb_max = coords.max(0)
        recs.append({
            "run": run,
            "n_total_nodes": len(nodes),
            "n_hood_nodes": len(coords),
            "n_tris": len(tris),
            "n_quads": len(quads),
            "fp": fp,
            "xmin": bb_min[0], "xmax": bb_max[0],
            "ymin": bb_min[1], "ymax": bb_max[1],
            "zmin": bb_min[2], "zmax": bb_max[2],
            "imp_x1": impact.iloc[run - 1]["X1"],
            "imp_x2": impact.iloc[run - 1]["X2"],
            "imp_x3": impact.iloc[run - 1]["X3"],
        })
        print(f"run {run:2d}: hood_nodes={len(coords):6d} tris={len(tris):4d} "
              f"quads={len(quads):6d} fp={fp}")

    df = pd.DataFrame(recs)

    # group by geometry fingerprint -> unique designs
    uniq_fps = list(dict.fromkeys(df["fp"]))  # preserve order of first appearance
    fp_to_design = {fp: i for i, fp in enumerate(uniq_fps)}
    df["design_geom"] = df["fp"].map(fp_to_design)

    print("\n" + "=" * 70)
    print(f"UNIQUE HOOD GEOMETRIES (by node-coordinate hash): {len(uniq_fps)}")
    print("=" * 70)
    for fp, d in fp_to_design.items():
        runs = df.loc[df["fp"] == fp, "run"].tolist()
        print(f"  design_geom {d:2d}  fp={fp}  -> runs {runs}")

    # unique impact locations
    imp_xy = df[["imp_x1", "imp_x2"]].round(3)
    uniq_loc = imp_xy.drop_duplicates().reset_index(drop=True)
    loc_to_id = {tuple(r): i for i, r in enumerate(uniq_loc.values)}
    df["loc_id"] = [loc_to_id[(round(a, 3), round(b, 3))] for a, b in zip(df["imp_x1"], df["imp_x2"])]
    print("\n" + "=" * 70)
    print(f"UNIQUE IMPACT (X1,X2) LOCATIONS: {len(uniq_loc)}")
    print("=" * 70)
    for i, r in uniq_loc.iterrows():
        runs = df.loc[df["loc_id"] == i, "run"].tolist()
        print(f"  loc {i}: ({r['imp_x1']:.2f}, {r['imp_x2']:.2f})  -> {len(runs)} runs")

    # crosstab design_geom x loc
    print("\nDesign(geom) x Location count table:")
    print(pd.crosstab(df["design_geom"], df["loc_id"]))

    df.to_csv("./Data/HoodImpact_60_IndustryLike/design_structure.csv", index=False)
    with open(CACHE, "wb") as f:
        pickle.dump(meshes, f)
    print(f"\nSaved structure CSV and mesh cache ({CACHE}).")


if __name__ == "__main__":
    main()
