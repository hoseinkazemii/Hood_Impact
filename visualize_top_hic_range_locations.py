"""Locate the impact points whose HIC15 varies most between hood designs.

Ranks the 142 EuroNCAP impact locations by the spread of HIC15 across the 12
designs and shows where the worst ones sit on the hood, next to the geometry
that differs between designs. These are the points where a design change moves
the injury number most, so they are the ones a geometry-aware surrogate has to
get right.

Two spreads are reported, because they are not the same quantity:

  * ``range_percent_all_designs``   max-min over all 12 designs, the figure's
                                    ranking; it includes the solver's own
                                    run-to-run scatter between near-clones;
  * ``range_percent_by_cluster``    max-min over the 4 geometry-cluster means,
                                    which is the real design effect with that
                                    scatter averaged out.

Run with no arguments to use the repository's canonical dataset layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter

from visualize_design_sensitivity_1704 import (
    IMPACTOR_NODES,
    LOCS_PER_DESIGN,
    NUM_DESIGNS,
    cluster_designs,
    hic15,
    load_histories,
    load_nodes,
    style_axes,
)

MARKER = "#1c5cab"  # 6.6:1 against the white rank label it carries
CONTEXT = "#c3c2b7"
HOOD = "#dcdbd3"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"


def build_figure(table, xy, all_range, hood, geometry, out_dir, top):
    grid_x, grid_y, spread = geometry
    fig = plt.figure(figsize=(15.0, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], left=0.05, right=0.975,
                          top=0.815, bottom=0.085, wspace=0.22)

    # -- where they are ------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax, grid_axis=None)
    ax.scatter(hood[::7, 0], hood[::7, 1], s=0.7, color=HOOD, linewidths=0, zorder=1)
    ax.contour(grid_x, grid_y, uniform_filter(np.nan_to_num(spread), size=6), levels=[2.0],
               colors=[MUTED], linewidths=1.1, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=26, color=CONTEXT, linewidths=0, zorder=3)
    ax.scatter(table["X1"], table["X2"], s=250, color=MARKER, edgecolors=SURFACE,
               linewidths=1.2, zorder=4)
    for row in table.itertuples():
        ax.annotate(str(row.rank), xy=(row.X1, row.X2), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="#ffffff", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("X1 (mm)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("X2 (mm)", fontsize=9.5, color=INK_2)
    ax.set_title(f"(a)  the {top} most design-sensitive impact points\n"
                 f"numbered by rank; small dots are the other "
                 f"{len(xy) - top} locations",
                 fontsize=10, color=INK, pad=9, loc="left")
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=1.1, label="hood geometry differs between designs"),
        Line2D([], [], marker="o", lw=0, markerfacecolor=MARKER, markeredgecolor=SURFACE,
               markersize=9, label=f"top {top} by HIC15 spread"),
        Line2D([], [], marker="o", lw=0, markerfacecolor=CONTEXT, markeredgecolor=CONTEXT,
               markersize=5, label="other impact locations"),
    ], loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK_2, handlelength=1.6,
        borderaxespad=0.2)

    # -- how big the spread is ----------------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    style_axes(ax, grid_axis="x")
    y = np.arange(len(table))[::-1]
    ax.barh(y, table["range_percent_all_designs"], height=0.66, color=MARKER, zorder=3)
    for position, row in zip(y, table.itertuples()):
        ax.text(row.range_percent_all_designs + 1.1, position,
                f"{row.range_percent_all_designs:.0f}%   "
                f"HIC {row.hic_min:.0f}–{row.hic_max:.0f}",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [f"{row.rank}.  loc {row.loc}" for row in table.itertuples()],
                  fontsize=8.5)
    ax.set_xlim(0, table["range_percent_all_designs"].max() * 1.62)
    ax.set_xlabel("HIC15 spread across the 12 designs (% of the location mean)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("(b)  how much the design moves HIC15 there\n"
                 "max − min over the 12 designs, with the absolute span",
                 fontsize=10, color=INK, pad=9, loc="left")

    inside = int(table["inside_changed_geometry"].sum())
    fig.text(0.05, 0.962,
             f"The {top} impact points where the hood design changes HIC15 the most",
             fontsize=14.5, color=INK, fontweight="bold", ha="left", va="center")
    fig.text(0.05, 0.918,
             f"12 designs × 142 EuroNCAP locations, all simulated. {inside} of the {top} "
             f"sit inside the region where the geometry differs; the median location "
             f"across the hood spreads only {100 * np.median(all_range):.0f}%.",
             fontsize=9.5, color=MUTED, ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "top_hic_range_locations.png"
    pdf = out_dir / "top_hic_range_locations.pdf"
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)
    plt.close(fig)
    return png, pdf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path,
                        default=root / "Data" / "HoodImpact_1704_EuroNCAP")
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    parser.add_argument("--cache", type=Path,
                        default=root / "figures" / "_acc_1704_cache.npz")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    if args.top < 1 or args.top > LOCS_PER_DESIGN:
        parser.error(f"--top must be between 1 and {LOCS_PER_DESIGN}")

    acc, times = load_histories(args.data_dir / "output_history_acc", args.cache)
    hic = hic15(acc.reshape(-1, acc.shape[2]), float(np.diff(times).mean()))
    hic = hic.reshape(NUM_DESIGNS, LOCS_PER_DESIGN)

    meshes = [load_nodes(args.data_dir / "inp_files", LOCS_PER_DESIGN * d + 1)
              for d in range(NUM_DESIGNS)]
    _, clusters, _, geometry = cluster_designs(meshes)
    hood = meshes[0][:-IMPACTOR_NODES]
    locations = pd.read_csv(args.data_dir / "impact_locations_142.csv")
    xy = locations[["X1", "X2"]].to_numpy()

    all_range = (hic.max(axis=0) - hic.min(axis=0)) / hic.mean(axis=0)
    cluster_means = np.array([hic[members].mean(axis=0) for members in clusters])
    cluster_range = ((cluster_means.max(axis=0) - cluster_means.min(axis=0))
                     / cluster_means.mean(axis=0))

    # Which side of the 2 mm geometry-change boundary each impact point sits on.
    grid_x, grid_y, spread = geometry
    field = uniform_filter(np.nan_to_num(spread), size=6)
    col = np.abs(grid_x[0, :][None, :] - xy[:, 0:1]).argmin(axis=1)
    row = np.abs(grid_y[:, 0][None, :] - xy[:, 1:2]).argmin(axis=1)
    local_change = field[row, col]

    order = np.argsort(all_range)[::-1][:args.top]
    table = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "loc": locations["loc"].to_numpy()[order],
        "X1": xy[order, 0],
        "X2": xy[order, 1],
        "hic_min": hic[:, order].min(axis=0),
        "hic_max": hic[:, order].max(axis=0),
        "hic_mean": hic[:, order].mean(axis=0),
        "range_percent_all_designs": 100 * all_range[order],
        "range_percent_by_cluster": 100 * cluster_range[order],
        "argmin_design": hic[:, order].argmin(axis=0),
        "argmax_design": hic[:, order].argmax(axis=0),
        "local_geometry_change_mm": local_change[order],
        "inside_changed_geometry": local_change[order] > 2.0,
    })

    png, pdf = build_figure(table, xy, all_range, hood, geometry, args.out_dir, args.top)
    csv = args.out_dir / "top_hic_range_locations.csv"
    table.to_csv(csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.drop(columns=["X1", "X2"]).to_string(index=False,
              float_format=lambda v: f"{v:.1f}"))
    print(f"\nOf the top {args.top}: {int(table['inside_changed_geometry'].sum())} sit inside "
          f"the >2 mm geometry-change region.")
    print(f"Median spread over all {LOCS_PER_DESIGN} locations: {100*np.median(all_range):.1f}%")
    print(f"\nSaved {png}\n      {pdf}\n      {csv}")
    return table


if __name__ == "__main__":
    main()
