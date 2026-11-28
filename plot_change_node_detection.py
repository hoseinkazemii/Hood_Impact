"""One summary figure for verify_change_node_detection.py.

  python plot_change_node_detection.py runs/change_node_detection/<stamp>
Writes <stamp>/change_node_detection.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BANDS = ["0.01-0.1", "0.1-0.5", "0.5-2", "≥ 2"]
TESTS = [("same_mesh", "same mesh (morph)", BLUE), ("flat_remesh_h14", "remesh 14 mm", ORANGE),
         ("cad_remesh_h14", "remesh 14 mm on smooth CAD", AQUA), ("cad_remesh_h20", "remesh 20 mm on smooth CAD", YELLOW)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


MIRROR_Y_MM = 155.34  # every hood edit is mirrored about this line; the maps show one half


def node_map(ax, points, truth, pred, title, window_from):
    half = window_from[window_from[:, 1] >= MIRROR_Y_MM - 20]
    window_from = half if len(half) else window_from
    lo = window_from.min(0) - 30
    hi = window_from.max(0) + 30
    inside = np.all((points[:, :2] >= lo[:2]) & (points[:, :2] <= hi[:2]), axis=1)
    tp, fn, fp = pred & truth & inside, ~pred & truth & inside, pred & ~truth & inside
    ax.scatter(*points[inside & ~truth & ~pred, :2].T, s=4, color="#cfcdc7", linewidths=0)
    for mask, color, label in ((tp, BLUE, "found"), (fn, ORANGE, "missed"), (fp, AQUA, "flagged, not in the edit list")):
        ax.scatter(*points[mask, :2].T, s=16, color=color, linewidths=0, label=f"{label} ({mask.sum()})")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.01), labelcolor=MUTED,
              handletextpad=0.2, borderaxespad=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.run / "summary.csv")
    examples = np.load(args.run / "examples.npz")
    base = examples["base_points"]

    fig = plt.figure(figsize=(15, 9.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.42, wspace=0.18)
    maps = [("N54", "reference", "N54 crease (test design)\ndesign-6 nodes, one mirror half"),
            ("N54", "edited", "N54 crease\nthe remeshed design's own nodes"),
            ("N72", "reference", "N72, smallest edit (50 nodes)\ndesign-6 nodes"),
            ("N1", "reference", "N1 beam notch\ndesign-6 nodes, one mirror half")]
    for column, (design, side, title) in enumerate(maps):
        ax = fig.add_subplot(grid[0, column])
        truth = examples[f"{design}_truth_reference"]
        if side == "reference":
            node_map(ax, base, truth, examples[f"{design}_reference_pred"], title, base[truth])
        else:
            points = examples[f"{design}_edited_points"]
            node_map(ax, points, examples[f"{design}_edited_truth"], examples[f"{design}_edited_pred"], title,
                     base[truth])

    # Recall by |dz| band, design-6 side, noise-free tests.
    ax = fig.add_subplot(grid[1, :2])
    style(ax)
    width = 0.2
    x = np.arange(len(BANDS))
    for k, (test, label, color) in enumerate(TESTS):
        row = summary[(summary["test"] == test) & (summary["noise_mm"] == 0) & (summary["side"] == "reference")]
        values = [row[f"recall_band{b}"].iloc[0] for b in range(4)]
        ax.bar(x + (k - 1.5) * width, values, width * 0.9, color=color, label=label)
    naive = summary[(summary["test"] == "cad_remesh_h14") & (summary["noise_mm"] == 0)
                    & (summary["side"] == "reference_naive")]
    ax.scatter(x, [naive[f"recall_band{b}"].iloc[0] for b in range(4)], marker="_", s=260, color=INK, zorder=3,
               label="nearest-node baseline, remesh 14 mm on CAD")
    ax.set_xticks(x, [f"|dz| {b} mm" for b in BANDS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of design-6 change nodes found")
    ax.set_title("Recall by how far the node moved (all 72 designs pooled, design-6 side)", fontsize=10,
                 color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.1), labelcolor=MUTED)

    # Noise sweep on the CAD remesh.
    ax = fig.add_subplot(grid[1, 2:])
    style(ax)
    sweep = summary[summary["test"] == "cad_remesh_h14"].sort_values("noise_mm")
    series = [("reference", "recall", BLUE, "-", "recall, design-6 nodes"),
              ("edited", "recall", ORANGE, "-", "recall, remeshed nodes"),
              ("reference", "energy_recall_min", AQUA, "-", "worst design's share of edit dz² found"),
              ("reference", "precision_near", INK, "--", "precision (flags within one element of the edit)")]
    for side, column, color, dash, label in series:
        part = sweep[sweep["side"] == side]
        ax.plot(np.arange(len(part)), part[column], dash, color=color, linewidth=2, marker="o", markersize=5,
                label=label)
    ax.set_xticks(np.arange(len(sweep["noise_mm"].unique())),
                  [f"{v:g}" for v in sorted(sweep["noise_mm"].unique())])
    ax.set_xlabel("extra Gaussian noise on the new nodes, sigma (mm)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Robustness: remesh 14 mm on smooth CAD plus noise", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=MUTED)

    fig.suptitle("Finding the edited nodes without node IDs: point-to-surface deviation + curvature tolerance "
                 "(maps: remesh 14 mm on smooth CAD, no noise)", fontsize=11, color=INK, x=0.06, ha="left", y=0.98)
    out = args.run / "change_node_detection.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
