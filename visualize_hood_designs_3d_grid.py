"""
3D shell-mesh renders of all 12 hood designs in one 3x4 grid, each with its 5
headform impact locations marked (raised stems + star markers so they are not
occluded by the surface).

Reuses the mesh cache from analyze_inp_designs.py
(./Data/HoodImpact_60_IndustryLike/_parsed_hood_meshes.pkl).

Also writes one high-res standalone 3D figure per design (subfolder per_design/).
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401 (registers 3d)

DATA = "./Data/HoodImpact_60_IndustryLike"
CACHE = os.path.join(DATA, "_parsed_hood_meshes.pkl")
IMPACT_CSV = os.path.join(DATA, "ImpactCoords_60.csv")
OUT = "./figures/hood_60_industrylike"
PER_DESIGN = os.path.join(OUT, "per_design_3d")
os.makedirs(PER_DESIGN, exist_ok=True)

N = 60
SPD = 5
N_DESIGNS = N // SPD
N_LOC = SPD
Z_EXAG = 3.0          # vertical exaggeration so hood curvature is visible
ELEV, AZIM = 52, -60


def build_triangles(tris, quads):
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
    return meshes, impact


def impact_xyz(impact, run):
    r = impact.iloc[run - 1]
    return float(r["X1"]), float(r["X2"]), float(r["X3"])


def draw_design(ax, meshes, impact, design, show_labels=True, marker_size=70):
    run = design * SPD + 1
    c = meshes[run]["coords"]
    tri = build_triangles(meshes[run]["tris"], meshes[run]["quads"])

    ax.plot_trisurf(c[:, 0], c[:, 1], c[:, 2], triangles=tri,
                    cmap="viridis", linewidth=0, antialiased=False, alpha=0.97)

    zspan = float(np.ptp(c[:, 2]))
    raise_h = 0.9 * zspan          # how high above the local surface to lift markers
    for loc in range(N_LOC):
        x, y, z = impact_xyz(impact, run + loc)
        z_top = z + raise_h
        ax.plot([x, x], [y, y], [z, z_top], color="red", linewidth=1.4, zorder=10)
        ax.scatter([x], [y], [z_top], marker="*", s=marker_size * 2.2,
                   color="red", edgecolors="black", linewidths=0.6,
                   depthshade=False, zorder=11)
        if show_labels:
            ax.text(x, y, z_top + 0.12 * zspan, f"L{loc+1}",
                    color="darkred", fontsize=9, fontweight="bold", ha="center")

    ax.set_xlabel("X1 (mm)", fontsize=8, labelpad=-2)
    ax.set_ylabel("X2 (mm)", fontsize=8, labelpad=-2)
    ax.set_zlabel("X3 (mm)", fontsize=8, labelpad=-4)
    ax.view_init(elev=ELEV, azim=AZIM)
    try:
        ax.set_box_aspect((np.ptp(c[:, 0]), np.ptp(c[:, 1]), zspan * Z_EXAG))
    except Exception:
        pass
    return len(c)


def fig_grid(meshes, impact):
    fig = plt.figure(figsize=(24, 17))
    for d in range(N_DESIGNS):
        ax = fig.add_subplot(3, 4, d + 1, projection="3d")
        n = draw_design(ax, meshes, impact, d, show_labels=True, marker_size=60)
        ax.set_title(f"Design {d}  |  runs {d*SPD+1}-{d*SPD+SPD}  |  {n:,} nodes",
                     fontsize=12, fontweight="bold", pad=0)
        ax.tick_params(labelsize=6)
    fig.suptitle("3D hood shell mesh — all 12 designs (red stars = 5 headform impact locations)",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(OUT, "designs_12_grid_3d.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def per_design_figs(meshes, impact):
    for d in range(N_DESIGNS):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        n = draw_design(ax, meshes, impact, d, show_labels=True, marker_size=110)
        ax.set_title(f"Design {d}  |  runs {d*SPD+1}-{d*SPD+SPD}  |  {n:,} nodes",
                     fontsize=13, fontweight="bold")
        p = os.path.join(PER_DESIGN, f"design_{d:02d}_3d.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


def main():
    meshes, impact = load_inputs()
    fig_grid(meshes, impact)
    per_design_figs(meshes, impact)
    print(f"\nDone. Grid + 12 standalone 3D figures in: {OUT}")


if __name__ == "__main__":
    main()
