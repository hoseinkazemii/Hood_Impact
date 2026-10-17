"""Plot the unseen-location holdout on a sample hood shell mesh (top view)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd

from location_holdout import DEFAULT_SPLIT_FILE, load_split
from visualize_hic_threshold_locations import shell_polygons

TRAIN_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data/HoodImpact_1704_EuroNCAP"))
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--design", type=int, default=0, help="Zero-based design whose mesh is drawn.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/location_holdout"))
    args = parser.parse_args()
    split = load_split(args.split)
    grid = pd.read_csv(args.data_root / "impact_locations_142.csv")
    test = grid["loc"].isin(split["test_locations"])
    run = 142 * args.design + 1
    polygons = shell_polygons(args.data_root / f"inp_files/HoodImpact_{run}.inp")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 8.4))
    fig.subplots_adjust(left=.10, right=.97, bottom=.11, top=.80)
    ax.add_collection(PolyCollection(polygons, facecolors="#f0f2f4", edgecolors="#c9cfd5",
                                     linewidths=.16, rasterized=True))
    # Mirror line of the lattice: col_k = 0. Mirror images share a set.
    centre = grid.loc[grid.col_k == 0, "X2"].mean()
    ax.axvline(centre, color="#9aa3ab", linewidth=.8, linestyle=(0, (4, 3)), zorder=2)
    ax.text(centre, grid.X1.max() + 45, "lateral mirror line", ha="center", va="bottom",
            fontsize=8.5, color="#566574")
    seen, unseen = grid.loc[~test], grid.loc[test]
    ax.scatter(seen.X2, seen.X1, s=58, marker="o", c=TRAIN_COLOR, edgecolors="white", linewidths=1.2,
               label=f"Training / validation designs (0–3, 6–11): {len(seen)} locations", zorder=4)
    ax.scatter(unseen.X2, unseen.X1, s=62, marker="D", c=TEST_COLOR, edgecolors="white", linewidths=1.2,
               label=f"Test designs (4, 5): {len(unseen)} locations", zorder=4)
    for row in grid.itertuples():
        ax.annotate(str(row.loc), (row.X2, row.X1), xytext=(0, -9.5), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color="#4b5563", zorder=5)
    xy = pd.concat([pd.DataFrame(p, columns=["x", "y"]) for p in polygons])
    ax.set_xlim(xy.x.min() - 65, xy.x.max() + 65)
    ax.set_ylim(xy.y.min() - 60, xy.y.max() + 90)
    ax.set_aspect("equal")
    ax.set_xlabel("X2 (mm)")
    ax.set_ylabel("X1 (mm)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#b7bec5")
    ax.tick_params(colors="#4b5563", labelsize=9)
    fig.suptitle("Unseen-location holdout", x=.10, y=.965, ha="left",
                 fontsize=22, fontweight="bold", color="#172b3a")
    fig.text(.10, .905, "Checkerboard on the Euro-NCAP lattice: no test location is trained on, "
             "on any design", fontsize=12.5, color="#415466")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.10, .875),
               frameon=False, fontsize=10, ncol=1, borderaxespad=0)
    fig.text(.10, .04, "Split = parity of wadh_mm/100 + col_k  ·  639 train / 71 validation / 142 test runs  "
             "·  numbers are location IDs", fontsize=9, color="#566574")
    fig.text(.10, .017, f"Top view · Sample design {args.design}, run {run} · Actual shell mesh · "
             "Rigid headform omitted", fontsize=9, color="#566574")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"location_holdout_design{args.design}"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png').resolve()}")


if __name__ == "__main__":
    main()
