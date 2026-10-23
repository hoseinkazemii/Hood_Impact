"""Overlay all 12 ground-truth histories at the 10 most and least HIC-varied locations.

Each page pairs the 12 acceleration curves with each design's inner panel and
hinges (outer skin hidden: it is identical across designs) and the headform
footprint at that location. Colors follow the four geometry clusters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyvista as pv

from abaqus_scripts.inp_geom import Deck
from visualize_design_mesh_comparison_1704 import read_clusters, shell_mesh

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/mesh_impact_history/20261004_050747_3303364_1704"
PER_DESIGN = 142
COLORS = {0: "#0b3d91", 1: "#2f6fd1", 2: "#69a7ef", 3: "#1593b8",
          4: "#d95f02", 5: "#f0a030",
          6: "#1b7837", 7: "#4daf4a", 8: "#93d07a", 9: "#00695c",
          10: "#6a3d9a", 11: "#b07cd8"}
STYLES = ["-", "--", "-.", ":"]
PANEL, HINGE, MARK = "#a8c2d2", "#2b2f36", "#e0115f"
WINDOW = (560, 650)


def design_geometry(data_dir, design):
    deck = Deck(data_dir / "inp_files" / f"HoodImpact_{PER_DESIGN * design + 1}.inp")
    meshes = {name: shell_mesh(deck, deck.elsets[f"{name}-1-2"]) for name in ("Hood_Inner", "Hinges")}
    used = set(meshes["Hood_Inner"]["node_id"]) | set(meshes["Hinges"]["node_id"])
    if used & set(deck.elset_nodes("Hood_Outer-1-2")) or used & deck.impactor_node_ids():
        raise ValueError(f"Design {design}: outer-skin or headform nodes in plotted mesh")
    return meshes


def headform_footprint(data_dir, location):
    """XY center and radius of the actual headform nodes placed at this location."""
    deck = Deck(data_dir / "inp_files" / f"HoodImpact_{location}.inp")
    xyz = np.array([deck.nodes[n] for n in deck.impactor_node_ids()])
    center = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
    return center, float(np.linalg.norm(xyz[:, :2] - center, axis=1).max())


def render_design(meshes, center, scale, footprints):
    """One shared top-view camera; returns {location: image, None: no marker}."""
    p = pv.Plotter(off_screen=True, window_size=WINDOW)
    p.set_background("white")
    kw = dict(smooth_shading=False, ambient=.35, diffuse=.65, specular=.15, reset_camera=False)
    p.add_mesh(meshes["Hood_Inner"], color=PANEL, **kw)
    # Hinges sit below the inner-panel flange; lift them in z so the
    # orthographic top view shows them at their true XY position.
    p.add_mesh(meshes["Hinges"].translate((0, 0, 150), inplace=False), color=HINGE, **kw)
    p.camera_position = [center + np.array([0, 0, 3000.]), center, (0, 1, 0)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = scale
    images = {None: p.screenshot(return_img=True)}
    z = 720.0
    for location, (xy, radius) in footprints.items():
        ring = pv.Circle(radius=radius, resolution=120).translate((*xy, z), inplace=False)
        actors = [p.add_mesh(ring, color=MARK, opacity=.22, reset_camera=False),
                  p.add_mesh(pv.Circle(radius=radius, resolution=120).extract_feature_edges(
                      boundary_edges=True, feature_edges=False, manifold_edges=False,
                      non_manifold_edges=False).translate((*xy, z + 1), inplace=False),
                      color=MARK, line_width=3, reset_camera=False),
                  p.add_mesh(pv.Sphere(radius=12, center=(*xy, z + 2)), color=MARK, reset_camera=False)]
        images[location] = p.screenshot(return_img=True)
        for actor in actors:
            p.remove_actor(actor)
    p.close()
    return images


def thumbnail(ax, image, design, labels, hic=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS[design])
        spine.set_linewidth(2.2)
    ax.set_title(labels[design], fontsize=10.5, color=COLORS[design], fontweight="bold", pad=4)
    ax.add_line(Line2D([.03, .30], [-.045, -.045], transform=ax.transAxes, clip_on=False, color=COLORS[design],
                       linestyle=STYLES[STYLE_INDEX[design]], lw=2.4))
    if hic is not None:
        ax.text(.97, -.045, f"HIC15 {hic:,.0f}", transform=ax.transAxes, ha="right", va="center",
                fontsize=9.5, color="#1f2937")


STYLE_INDEX = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--out-dir", type=Path, default=RUN / "all_designs_hic_overlays")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clusters = read_clusters()
    cluster_of = {d: name for name, members in clusters.items() for d in members}
    for members in clusters.values():
        STYLE_INDEX.update({d: i for i, d in enumerate(members)})
    splits = json.loads((RUN / "splits.json").read_text())
    split_of = {d: name for name, s in splits.items() if isinstance(s, dict) for d in s["design_ids"]}
    split_of = {d: {"validation": "val"}.get(s, s) for d, s in split_of.items()}
    labels = {d: f"Design {d} | {cluster_of[d]} | {split_of[d]}" for d in range(12)}

    variation = pd.read_csv(RUN / "hic_location_variation.csv")
    selections = {}
    for name, lowest in (("top10", False), ("lowest10", True)):
        top = (variation.sort_values(["range_percent", "location"], ascending=[lowest, True])
               .head(10).copy())
        top.insert(0, "rank", range(1, 11))
        selections[name] = top
    locations = sorted({int(l) for top in selections.values() for l in top.location})

    footprints = {loc: headform_footprint(args.data_dir, loc) for loc in locations}
    print(f"Headform footprints from design-0 decks; radius {min(r for _, r in footprints.values()):.1f}"
          f"-{max(r for _, r in footprints.values()):.1f} mm", flush=True)
    geometry = [design_geometry(args.data_dir, d) for d in range(12)]
    points = np.concatenate([g["Hood_Inner"].points for g in geometry])
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    scale = 0.52 * max(hi[1] - lo[1], (hi[0] - lo[0]) * WINDOW[1] / WINDOW[0])
    images = [render_design(g, center, scale, footprints) for g in geometry]
    print("Rendered inner panel + hinges for all 12 designs", flush=True)

    curves = {}
    for d in range(12):
        for loc in locations:
            number = PER_DESIGN * d + loc
            truth = pd.read_csv(args.data_dir / "output_history_acc"
                                / f"HoodImpact_{number}_SAE1000_interp1000.csv",
                                usecols=["Time", "A(in g)"])
            assert len(truth) == 1000 and np.isfinite(truth["A(in g)"]).all()
            curves[d, loc] = truth

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    for name, top in selections.items():
        lowest = name == "lowest10"
        path = args.out_dir / f"{name}_locations_all_12_designs_with_geometry.pdf"
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(16, 13))
            grid = GridSpec(3, 4, figure=fig, left=.02, right=.98, top=.9, bottom=.03,
                            wspace=.06, hspace=.18)
            for d in range(12):
                thumbnail(fig.add_subplot(grid[d // 4, d % 4]), images[d][None], d, labels)
            fig.suptitle("Design geometries: inner panel + hinges (outer skin hidden, identical in all designs)",
                         fontsize=17, fontweight="bold", y=.975)
            fig.text(.5, .935, "Top view, same camera and scale for every design | hinges in dark grey (drawn on top) | "
                     "color and line style match the acceleration plots on the following pages",
                     ha="center", fontsize=11, color="#4b5563")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

            for row in top.itertuples():
                loc = int(row.location)
                fig = plt.figure(figsize=(22, 11))
                grid = GridSpec(3, 7, figure=fig, width_ratios=[1.45, 1.45, .06, 1, 1, 1, 1],
                                left=.05, right=.99, top=.86, bottom=.08, wspace=.05, hspace=.24)
                ax = fig.add_subplot(grid[:, :2])
                peak = 0.0
                for d in range(12):
                    truth = curves[d, loc]
                    ax.plot(truth["Time"] * 1000, truth["A(in g)"], color=COLORS[d],
                            linestyle=STYLES[STYLE_INDEX[d]], lw=1.7,
                            label=f"{labels[d]} | HIC15 {getattr(row, f'hic_design_{d}'):,.0f}")
                    peak = max(peak, truth["A(in g)"].max())
                ax.set_xlim(0, curves[0, loc]["Time"].max() * 1000)
                ax.set_ylim(0, np.ceil(peak * 1.02 / 25) * 25)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Resultant acceleration (g)")
                ax.set_title("Ground truth, all 12 designs", fontsize=15, pad=10)
                ax.grid(alpha=.2)
                ax.legend(frameon=False, loc="upper right", fontsize=8.6, ncol=1, handlelength=2.8)
                for d in range(12):
                    hic = getattr(row, f"hic_design_{d}")
                    thumbnail(fig.add_subplot(grid[d // 4, 3 + d % 4]), images[d][loc], d, labels, hic)
                rank_label = "Lowest HIC variation rank" if lowest else "HIC variation rank"
                fig.suptitle(f"Impact location {loc} | {rank_label} #{row.rank}",
                             fontsize=20, fontweight="bold", y=.965)
                fig.text(.5, .915, f"HIC15 range / mean across all 12 designs: {row.range_percent:.1f}%"
                         f"  (min {row.hic_min:,.0f}, max {row.hic_max:,.0f})   |   "
                         "pink disk = headform outline (82.5 mm radius) at this location",
                         ha="center", fontsize=12)
                fig.text(.5, .025, "Ground truth SAE1000 histories, 1,000 points | geometry: Hood_Inner + "
                         "Hinges element sets, top view, common camera/scale | ranking: "
                         f"{RUN.name}/hic_location_variation.csv", ha="center", fontsize=9, color="#555555")
                pdf.savefig(fig, facecolor="white")
                plt.close(fig)
        print(f"Saved {path}", flush=True)
        top.to_csv(args.out_dir / f"{name}_hic_location_ranking.csv", index=False)
        print(top[["rank", "location", "range_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
