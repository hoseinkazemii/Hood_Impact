"""
Visualize the HoodImpact_60_IndustryLike dataset: 12 designs x 5 impact locations.

Produces (in ./figures/hood_60_industrylike/):
  1. hood_3d_hero.png         - 3D shell surfaces of 2 representative hoods + impacts
  2. hood_topview_heightmap.png - top view height map + 5 numbered impact locations
  3. designs_12_grid_height.png - 3x4 grid, hood height map per design + impacts
  4. designs_12_grid_HIC.png    - 3x4 grid, impacts colored/sized by official HIC
  5. impact_locations.png       - the 5 impact locations (top view) + coordinate table
  6. dataset_summary.csv        - per-run design/location/HIC table

Requires the mesh cache produced by analyze_inp_designs.py
(./Data/HoodImpact_60_IndustryLike/_parsed_hood_meshes.pkl).
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401 (registers 3d)

DATA = "./Data/HoodImpact_60_IndustryLike"
CACHE = os.path.join(DATA, "_parsed_hood_meshes.pkl")
IMPACT_CSV = os.path.join(DATA, "ImpactCoords_60.csv")
HIC_CSV = os.path.join(DATA, "output_scalar_HIC.csv")
OUT = "./figures/hood_60_industrylike"
os.makedirs(OUT, exist_ok=True)

N = 60
SPD = 5                      # runs per design
N_DESIGNS = N // SPD         # 12
N_LOC = SPD                  # 5 impact locations


# ---------------------------------------------------------------- helpers
def build_triangles(tris, quads):
    """Combine S3R tris and S4 quads (split into 2 tris) into one (M,3) array."""
    parts = []
    if len(tris):
        parts.append(tris)
    if len(quads):
        parts.append(quads[:, [0, 1, 2]])
        parts.append(quads[:, [0, 2, 3]])
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), int)


def load_inputs():
    meshes = pickle.load(open(CACHE, "rb"))
    impact = pd.read_csv(IMPACT_CSV)
    impact.columns = [c.strip() for c in impact.columns]
    hic = pd.read_csv(HIC_CSV)
    hic.columns = [c.strip() for c in hic.columns]
    hic_map = dict(zip(hic["Job ID"].astype(int), hic["HIC Value"].astype(float)))
    return meshes, impact, hic_map


def impact_xyz(impact, run):
    r = impact.iloc[run - 1]
    return float(r["X1"]), float(r["X2"]), float(r["X3"])


# ---------------------------------------------------------------- figures
def fig_hero_3d(meshes, impact):
    """3D shell surfaces of a coarse (run1) and fine (run31) hood + impact points."""
    reps = [(1, "Design 0 (runs 1-5)  37,979 nodes"),
            (31, "Design 6 (runs 31-35)  39,059 nodes")]
    fig = plt.figure(figsize=(16, 7))
    for k, (run, title) in enumerate(reps):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        c = meshes[run]["coords"]
        tri = build_triangles(meshes[run]["tris"], meshes[run]["quads"])
        ax.plot_trisurf(c[:, 0], c[:, 1], c[:, 2], triangles=tri,
                        cmap="viridis", linewidth=0, antialiased=False, alpha=0.95)
        # 5 impact locations for this design
        for loc in range(N_LOC):
            x, y, z = impact_xyz(impact, run + loc)
            ax.scatter([x], [y], [z + 8], color="red", s=60, depthshade=False)
            ax.text(x, y, z + 30, f"L{loc+1}", color="red", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X1 (mm)"); ax.set_ylabel("X2 (mm)"); ax.set_zlabel("X3 (mm)")
        ax.view_init(elev=55, azim=-60)
        try:
            ax.set_box_aspect((np.ptp(c[:, 0]), np.ptp(c[:, 1]), np.ptp(c[:, 2]) * 3))
        except Exception:
            pass
    fig.suptitle("Hood shell mesh (two base resolutions) with 5 headform impact locations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "hood_3d_hero.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_topview_heightmap(meshes, impact):
    run = 1
    c = meshes[run]["coords"]
    tri = build_triangles(meshes[run]["tris"], meshes[run]["quads"])
    triang = Triangulation(c[:, 0], c[:, 1], tri)
    fig, ax = plt.subplots(figsize=(11, 11))
    tpc = ax.tripcolor(triang, c[:, 2], shading="gouraud", cmap="viridis")
    fig.colorbar(tpc, ax=ax, shrink=0.7, label="Height X3 (mm)")
    for loc in range(N_LOC):
        x, y, _ = impact_xyz(impact, run + loc)
        ax.scatter([x], [y], s=180, facecolors="none", edgecolors="red", linewidths=2.2)
        ax.scatter([x], [y], s=18, color="red")
        ax.annotate(f"L{loc+1}\n({x:.0f}, {y:.0f})", (x, y), textcoords="offset points",
                    xytext=(10, 10), color="red", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("X1 (mm)"); ax.set_ylabel("X2 (mm)")
    ax.set_title("Hood top view (height map) with the 5 impact locations", fontweight="bold")
    p = os.path.join(OUT, "hood_topview_heightmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_designs_grid_height(meshes, impact):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for d in range(N_DESIGNS):
        ax = axes.flat[d]
        run = d * SPD + 1
        c = meshes[run]["coords"]
        tri = build_triangles(meshes[run]["tris"], meshes[run]["quads"])
        triang = Triangulation(c[:, 0], c[:, 1], tri)
        ax.tripcolor(triang, c[:, 2], shading="gouraud", cmap="viridis")
        for loc in range(N_LOC):
            x, y, _ = impact_xyz(impact, run + loc)
            ax.scatter([x], [y], s=70, facecolors="none", edgecolors="red", linewidths=1.6)
            ax.text(x, y, f"{loc+1}", color="red", fontsize=8, fontweight="bold",
                    ha="center", va="center")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Design {d}  |  runs {run}-{run+4}  |  {len(c):,} nodes", fontsize=11)
    fig.suptitle("12 hood designs (top-view height map) — each impacted at the same 5 locations",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = os.path.join(OUT, "designs_12_grid_height.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_designs_grid_hic(meshes, impact, hic_map):
    # global HIC range for consistent color scale
    hics = np.array([hic_map[r] for r in range(1, N + 1)])
    vmin, vmax = hics.min(), hics.max()
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    sc = None
    for d in range(N_DESIGNS):
        ax = axes.flat[d]
        run = d * SPD + 1
        c = meshes[run]["coords"]
        tri = build_triangles(meshes[run]["tris"], meshes[run]["quads"])
        triang = Triangulation(c[:, 0], c[:, 1], tri)
        # light grey hood backdrop
        ax.tripcolor(triang, np.ones(len(c)), shading="gouraud",
                     cmap="Greys", vmin=0, vmax=3)
        xs, ys, hs = [], [], []
        for loc in range(N_LOC):
            x, y, _ = impact_xyz(impact, run + loc)
            h = hic_map[run + loc]
            xs.append(x); ys.append(y); hs.append(h)
            ax.annotate(f"{h:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, fontweight="bold")
        sc = ax.scatter(xs, ys, c=hs, s=260, cmap="inferno", vmin=vmin, vmax=vmax,
                        edgecolors="k", linewidths=0.8, zorder=5)
        for loc, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y, f"{loc+1}", color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Design {d}  |  runs {run}-{run+4}", fontsize=11)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, location="right", pad=0.02)
    cbar.set_label("Official HIC value", fontsize=12)
    fig.suptitle("HIC safety map: 5 impact locations per design (color/number = HIC)",
                 fontsize=16, fontweight="bold")
    p = os.path.join(OUT, "designs_12_grid_HIC.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_impact_locations(impact, hic_map):
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    # the 5 locations are the same across designs -> take from design 0
    rows = []
    for loc in range(N_LOC):
        x, y, z = impact_xyz(impact, loc + 1)
        ax.scatter([x], [y], s=240, color="tab:red", zorder=5)
        ax.annotate(f"L{loc+1}", (x, y), textcoords="offset points", xytext=(12, 8),
                    fontsize=13, fontweight="bold")
        mean_hic = np.mean([hic_map[d * SPD + loc + 1] for d in range(N_DESIGNS)])
        rows.append([f"L{loc+1}", f"{x:.1f}", f"{y:.1f}", f"{z:.1f}", f"{mean_hic:.0f}"])
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("X1 (mm)"); ax.set_ylabel("X2 (mm)")
    ax.set_title("The 5 impact locations (shared by all 12 designs)", fontweight="bold")

    axt.axis("off")
    tbl = axt.table(cellText=rows,
                    colLabels=["Loc", "X1 (mm)", "X2 (mm)", "X3 (mm)", "mean HIC\n(over 12 designs)"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.2)
    axt.set_title("Impact location coordinates", fontweight="bold")
    p = os.path.join(OUT, "impact_locations.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def write_summary_csv(meshes, impact, hic_map):
    rows = []
    for run in range(1, N + 1):
        d = (run - 1) // SPD
        loc = (run - 1) % SPD
        x, y, z = impact_xyz(impact, run)
        rows.append({
            "run": run, "design": d, "impact_loc": loc + 1,
            "imp_x1": x, "imp_x2": y, "imp_x3": z,
            "n_hood_nodes": len(meshes[run]["coords"]),
            "official_HIC": hic_map[run],
        })
    df = pd.DataFrame(rows)
    p = os.path.join(OUT, "dataset_summary.csv")
    df.to_csv(p, index=False)
    print("wrote", p)
    # quick console pivot: design x location HIC
    piv = df.pivot(index="design", columns="impact_loc", values="official_HIC")
    print("\nOfficial HIC by design (rows) x impact location (cols):")
    print(piv.round(0).to_string())


def main():
    meshes, impact, hic_map = load_inputs()
    write_summary_csv(meshes, impact, hic_map)
    fig_topview_heightmap(meshes, impact)
    fig_impact_locations(impact, hic_map)
    fig_designs_grid_height(meshes, impact)
    fig_designs_grid_hic(meshes, impact, hic_map)
    fig_hero_3d(meshes, impact)
    print(f"\nAll figures written to: {OUT}")


if __name__ == "__main__":
    main()
