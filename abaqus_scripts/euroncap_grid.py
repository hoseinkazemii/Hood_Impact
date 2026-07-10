"""
euroncap_grid.py
================
Define the 50 shared headform impact locations for the 600-sample dataset
(12 designs x 50 locations), following the Euro NCAP pedestrian headform
GRID method (Euro NCAP Pedestrian/VRU Testing Protocol, TB 024) adapted to a
standalone hood model:

  * Euro NCAP marks the bonnet with grid points on 100 mm spacing: rows along
    Wrap Around Distance (WAD) lines 100 mm apart, columns 100 mm apart
    laterally, starting from the vehicle centreline.
  * WAD is measured with a tape along the outer surface. With no bumper or
    ground in the model, we measure the wrap distance ALONG THE HOOD OUTER
    SURFACE from the hood leading edge (front edge, x ~ -493) instead; rows
    are lines of constant surface arc-length ("WADh").
  * The headform (165 mm dia) must land fully on the hood, so grid points are
    kept >= INSET (100 mm > 82.5 mm radius) inside the outer-panel boundary
    in plan view. This stands in for the protocol's bonnet side/rear
    reference-line bounds, which need the full vehicle to construct.
  * The full 100 mm grid has more points than the 50-per-design budget, so
    the 50 are chosen from the grid by FARTHEST-POINT (maximin) sampling,
    seeded with the 5 original impact locations of the 60-sample dataset so
    the new points spread away from data we already have. Deterministic.

The grid geometry is built on design 0 (run 1); all 12 designs share the same
hood outline (verified: identical bounding boxes across all 60 decks), and
their local geometry differences change the surface arc-length only at the
sub-mm level.

Run from the Code/ directory (or anywhere -- paths are script-relative):
    python abaqus_scripts/euroncap_grid.py
Outputs:
    Data/HoodImpact_600_EuroNCAP/impact_locations_50.csv   (the 50 targets)
    Data/HoodImpact_600_EuroNCAP/impact_grid_candidates.csv (full 100mm grid)
    figures/hood600/impact_grid_50.png                      (overview figure)
"""
import os
import csv
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CODE_ROOT = os.path.dirname(HERE)

from inp_geom import Deck, OuterSurface, points_inside, dist_to_polyline, polygon_x_span

BASE_DECK = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike",
                         "inp_files", "HoodImpact_1.inp")
OUT_DIR = os.path.join(CODE_ROOT, "Data", "HoodImpact_600_EuroNCAP")
FIG_DIR = os.path.join(CODE_ROOT, "figures", "hood600")

PITCH = 100.0          # mm, Euro NCAP grid spacing (rows and columns)
INSET = 100.0          # mm, min plan-view distance of a point to the hood edge
CENTERLINE_Y = 155.3432  # vehicle centreline (exactly L1/L3's X2; = outline mid-Y)
FINE_STEP = 2.0        # mm, x-step for the surface arc-length integration
N_SELECT = 50

# the 5 original locations of the 60-sample dataset (seeds for maximin sampling)
ORIGINAL_5 = np.array([
    [-100.275, 155.3432],
    [104.9198, 143.5854],
    [405.7401, 151.9838],
    [73.04483, 9.210356],
    [144.7636, -145.321],
])


def build_candidates(surface):
    """All Euro-NCAP-style grid points (x, y, wadh) that fit on the hood."""
    poly = surface.outline
    ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
    print("hood outline: x [%.1f, %.1f]  y [%.1f, %.1f]  mid-y %.3f"
          % (poly[:, 0].min(), poly[:, 0].max(), ymin, ymax, 0.5 * (ymin + ymax)))

    cands = []
    kmax = int(np.ceil((ymax - ymin) / PITCH))
    for k in range(-kmax, kmax + 1):
        y = CENTERLINE_Y + k * PITCH
        span = polygon_x_span(poly, y)
        if span is None:
            continue
        x0, x1 = span
        if x1 - x0 < 2 * FINE_STEP:
            continue
        # surface profile along this column, front -> rear
        xs = np.arange(x0 + 1.0, x1 - 1.0, FINE_STEP)
        if len(xs) < 3:
            continue
        zs = surface.surf_z(xs, np.full_like(xs, y))
        ds = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])       # arc length from front edge
        # rows: constant arc-length lines every PITCH mm
        s_targets = np.arange(PITCH, s[-1], PITCH)
        x_rows = np.interp(s_targets, s, xs)
        for wadh, x in zip(s_targets, x_rows):
            cands.append((x, y, wadh, k))

    cands = np.array(cands, float)
    # keep only points at least INSET inside the outline
    inside = points_inside(cands[:, :2], poly)
    cands = cands[inside]
    far = dist_to_polyline(cands[:, :2], poly) >= INSET
    cands = cands[far]
    print("candidate grid points (100 mm pitch, %.0f mm inset): %d" % (INSET, len(cands)))
    return cands


def farthest_point_select(cands_xy, seeds_xy, n_select):
    """Greedy maximin selection of n_select points from cands, seeded with
    seeds (which are 'already chosen' but not part of the returned set)."""
    chosen = []
    # distance of every candidate to the closest already-chosen point
    dmin = np.min(
        np.linalg.norm(cands_xy[:, None, :] - seeds_xy[None, :, :], axis=2), axis=1)
    for _ in range(n_select):
        i = int(np.argmax(dmin))
        chosen.append(i)
        d_new = np.linalg.norm(cands_xy - cands_xy[i], axis=1)
        dmin = np.minimum(dmin, d_new)
    return chosen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("parsing %s ..." % BASE_DECK)
    deck = Deck(BASE_DECK)
    surface = OuterSurface(deck)

    cands = build_candidates(surface)
    order = farthest_point_select(cands[:, :2], ORIGINAL_5, N_SELECT)
    sel = cands[order]

    # selection quality: min pairwise distance among the 50 (+ 5 originals)
    all_pts = np.vstack([sel[:, :2], ORIGINAL_5])
    dm = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=2)
    np.fill_diagonal(dm, np.inf)
    print("selected %d locations; min pairwise spacing %.1f mm "
          "(incl. the 5 originals)" % (len(sel), dm.min()))

    # -- write CSVs ----------------------------------------------------------
    loc_csv = os.path.join(OUT_DIR, "impact_locations_50.csv")
    with open(loc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loc", "X1", "X2", "wadh_mm", "col_k"])
        for i, (x, y, wadh, k) in enumerate(sel, start=1):
            w.writerow([i, "%.4f" % x, "%.4f" % y, "%.1f" % wadh, int(k)])
    print("wrote %s" % loc_csv)

    cand_csv = os.path.join(OUT_DIR, "impact_grid_candidates.csv")
    with open(cand_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X1", "X2", "wadh_mm", "col_k"])
        for x, y, wadh, k in cands:
            w.writerow(["%.4f" % x, "%.4f" % y, "%.1f" % wadh, int(k)])
    print("wrote %s" % cand_csv)

    # -- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    poly = surface.outline
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=1.5, label="hood outer boundary")
    ax.scatter(cands[:, 0], cands[:, 1], s=8, c="0.75", zorder=1,
               label="Euro NCAP 100 mm grid (%d pts)" % len(cands))
    ax.scatter(sel[:, 0], sel[:, 1], s=42, c="tab:blue", zorder=3,
               label="selected 50 (farthest-point)")
    for i, (x, y) in enumerate(sel[:, :2], start=1):
        ax.annotate(str(i), (x, y), fontsize=6, ha="center", va="center",
                    color="white", zorder=4)
    ax.scatter(ORIGINAL_5[:, 0], ORIGINAL_5[:, 1], s=90, marker="X",
               c="tab:red", zorder=5, label="original 5 (60-sample set)")
    ax.axhline(CENTERLINE_Y, color="tab:green", lw=0.8, ls="--",
               label="centreline y=%.1f" % CENTERLINE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("X1 (mm, front edge at left)")
    ax.set_ylabel("X2 (mm)")
    ax.set_title("600-sample dataset: 50 Euro-NCAP-grid impact locations "
                 "(shared by all 12 designs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "impact_grid_50.png")
    fig.savefig(out_png, dpi=140)
    print("wrote %s" % out_png)


if __name__ == "__main__":
    main()
