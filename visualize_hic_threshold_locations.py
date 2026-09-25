"""Plot each HIC variation threshold on the same sample hood shell mesh."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def shell_polygons(path):
    """Read actual shell connectivity; omit the leading 286 headform nodes."""
    nodes, shells, mode = {}, [], None
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                keyword = line.upper()
                mode = ("node" if keyword == "*NODE" else
                        "shell" if keyword.startswith("*ELEMENT,") and
                        any(f"TYPE={kind}" in keyword.replace(" ", "")
                            for kind in ("S3R", "S4")) else None)
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if mode == "node":
                nodes[int(parts[0])] = np.array(parts[1:4], float)
            elif mode == "shell":
                shells.append([int(part) for part in parts[1:]])
    headform = set(list(nodes)[:286])
    polygons = [np.array([nodes[node][[1, 0]] for node in shell])
                for shell in shells if not headform.intersection(shell)]
    if not polygons:
        raise ValueError("No structural shell elements found")
    return polygons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data/HoodImpact_1704_EuroNCAP"))
    parser.add_argument("--variation-csv", type=Path,
                        default=Path("runs/hic_location_filter/hic_location_variation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/hic_threshold_locations"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[10, 15, 20, 25, 30, 40, 50, 75, 100])
    args = parser.parse_args()
    variation = pd.read_csv(args.variation_csv)
    locations = pd.read_csv(args.data_root / "impact_locations_142.csv")
    table = locations.merge(variation, left_on="loc", right_on="location", validate="one_to_one")
    if len(table) != 142 or not np.isfinite(table[["X1", "X2", "range_percent"]]).all().all():
        raise ValueError("Expected all 142 locations with finite coordinates and HIC ranges")
    polygons = shell_polygons(args.data_root / "inp_files/HoodImpact_1.inp")
    coordinates = np.concatenate(polygons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    with PdfPages(args.output_dir / "all_thresholds.pdf") as combined:
        for threshold in args.thresholds:
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Thresholds must be finite and nonnegative")
            selected = table.range_percent >= threshold
            count = int(selected.sum())
            counts[f"{threshold:g}"] = count
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.subplots_adjust(left=.10, right=.97, bottom=.13, top=.81)
            ax.add_collection(PolyCollection(polygons, facecolors="#f0f2f4", edgecolors="#c9cfd5",
                                             linewidths=.16, rasterized=True))
            excluded = table.loc[~selected]
            kept = table.loc[selected]
            ax.scatter(excluded.X2, excluded.X1, s=27, marker="x", c="#929ba4", linewidths=1,
                       label=f"Excluded: {142 - count}", zorder=3)
            ax.scatter(kept.X2, kept.X1, s=53, c="#0068b5", edgecolors="white", linewidths=.75,
                       label=f"Retained for training / validation / test: {count}", zorder=4)
            ax.set_xlim(coordinates[:, 0].min() - 65, coordinates[:, 0].max() + 65)
            ax.set_ylim(coordinates[:, 1].min() - 60, coordinates[:, 1].max() + 60)
            ax.set_aspect("equal")
            ax.set_xlabel("X2 (mm)")
            ax.set_ylabel("X1 (mm)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("bottom", "left"):
                ax.spines[spine].set_color("#b7bec5")
            ax.tick_params(colors="#4b5563", labelsize=9)
            fig.suptitle(f"HIC variation ≥ {threshold:g}%", x=.10, y=.96, ha="left",
                         fontsize=23, fontweight="bold", color="#172b3a")
            fig.text(.10, .90, f"{count} of 142 impact locations retained  |  {100 * count / 142:.1f}% of locations",
                     fontsize=13, color="#415466")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.10, .87),
                       frameon=False, fontsize=10, ncol=2, borderaxespad=0)
            if count == 0:
                ax.text(.5, .52, "No impact locations meet this threshold", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color="#172b3a",
                        bbox={"facecolor": "white", "edgecolor": "#c9cfd5", "pad": 10}, zorder=5)
            fig.text(.10, .045, "Variation = 100 × (max HIC15 − min HIC15) / mean HIC15 across all 12 designs",
                     fontsize=9, color="#566574")
            fig.text(.10, .022, "Top view · Sample design 0, run 1 · Actual shell mesh · Rigid headform omitted",
                     fontsize=9, color="#566574")
            stem = args.output_dir / f"hic_threshold_{threshold:g}pct"
            fig.savefig(stem.with_suffix(".png"), dpi=220)
            fig.savefig(stem.with_suffix(".pdf"), dpi=220)
            combined.savefig(fig, dpi=220)
            plt.close(fig)
            table.loc[selected].to_csv(args.output_dir / f"retained_locations_{threshold:g}pct.csv", index=False)
    (args.output_dir / "threshold_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"Figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
