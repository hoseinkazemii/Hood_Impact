"""Maps of the ID-free change detection on B1-B4 (verify_change_node_detection_benchmarks.py).

  python plot_change_node_detection_benchmarks.py runs/change_node_detection_benchmarks/<stamp>
Writes <stamp>/change_node_detection_benchmarks.png (remesh_cad, noise 1% of the spacing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, GRID = "#fcfcfb", "#0b0b0b", "#cfcdc7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
PANELS = [("B1_plate", "e4_slot_diagonal", "B1 plate: hole to diagonal slot"),
          ("B2_elbow", "e2_to_intrados", "B2 elbow: defect moved to intrados"),
          ("B3_heatsink", "e1_keepout_centre", "B3 heat sink: centre fins cut (removed)"),
          ("B3_heatsink", "e4_raise_centre", "B3 heat sink: centre fins raised (added)"),
          ("B4_cantilever", "e3_add_pair", "B4 cantilever: notch pair added")]


def project(points, around):
    """2D view: x, y for planar parts; for 3D the plane of the edited region (PCA)."""
    if np.ptp(points[:, 2]) < 1e-9:
        return points[:, :2], around[:, :2]
    centre = around.mean(0)
    _, _, axes = np.linalg.svd(around - centre, full_matrices=False)
    keep = np.abs((points - centre) @ axes[2]) < 3 * np.ptp(around @ axes[2]) + 5  # this side of the wall
    view = (points - centre) @ axes[:2].T
    view[~keep] = np.nan
    return view, (around - centre) @ axes[:2].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    data = np.load(args.run / "examples.npz")
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 4.6), facecolor=SURFACE)
    for ax, (bench, design, title) in zip(axes, PANELS):
        key = f"{bench}/{design}/"
        base, truth, pred = data[key + "reference_points"], data[key + "reference_truth"], data[key + "reference_pred"]
        new, new_pred = data[key + "edited_points"], data[key + "edited_pred"]
        region = np.vstack((base[truth], new[new_pred])) if truth.any() or new_pred.any() else base
        base2, region2 = project(base, region)
        new2, _ = project(new, region)
        lo, hi = np.nanmin(region2, 0), np.nanmax(region2, 0)
        pad = 0.35 * max(hi - lo) + 1.0
        lo, hi = lo - pad, hi + pad

        def inside(p):
            return np.all((p >= lo) & (p <= hi), axis=1)

        ax.scatter(*base2[inside(base2) & ~truth & ~pred].T, s=3, color=GRID, linewidths=0)
        for mask, color, label in ((truth & pred, BLUE, "base boundary: found"),
                                   (truth & ~pred, ORANGE, "base boundary: missed"),
                                   (~truth & pred, AQUA, "base boundary: extra flag")):
            mask = mask & inside(base2)
            ax.scatter(*base2[mask].T, s=14, color=color, linewidths=0, label=f"{label} ({mask.sum()})")
        flagged = new_pred & inside(new2)
        ax.scatter(*new2[flagged].T, s=22, facecolors="none", edgecolors=VIOLET, linewidths=0.9,
                   label=f"new boundary: flagged ({flagged.sum()})")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.01), labelcolor="#52514e")
    fig.suptitle("ID-free change detection on remeshed B1-B4 edits (boundary resampled on a smooth CAD stand-in, "
                 "noise 1% of the spacing). B2 is viewed in the plane of the defect.",
                 fontsize=10, color=INK, x=0.01, ha="left")
    out = args.run / "change_node_detection_benchmarks.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
