"""How much does the hood design change the acceleration history?

The 1704-run EuroNCAP set is a complete 12 designs x 142 impact locations grid,
so the design effect and the location effect can be separated exactly: every
design is struck at every location. This script answers one question --
*do the subtle geometry differences between the 12 designs produce a
meaningful change in the acceleration history, and where?* -- as a single
four-panel figure.

    (a)-(c) the 12 histories at three impact locations, from the least to the
        most design-sensitive, coloured by geometry cluster;
    (d) a plan view of the hood: where on the panel the design matters;
    (e) exact variance decomposition of peak |a| and HIC15 into location,
        design, and their interaction;
    (f) the distribution over all 142 locations of the design-induced HIC
        range, against the trained surrogate's own error.

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter

NUM_DESIGNS = 12
LOCS_PER_DESIGN = 142
IMPACTOR_NODES = 286  # trailing node IDs in every deck belong to the headform

# Validated categorical palette: worst CVD dE 9.23, worst normal-vision dE 16.32
# (OKLab x100, all-pairs, light surface). Aqua sits at 2.74:1 on the light
# surface, so panel (a) direct-labels the cluster means -- identity never rests
# on colour alone.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Best-checkpoint test error of runs/mesh_impact_history/20260904_220022_3086980_1704,
# used in panel (d) as the resolution floor: a design effect smaller than this
# cannot be told apart from the surrogate's own error.
SURROGATE_HIC_MAPE = 5.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_nodes(inp_dir: Path, run: int) -> np.ndarray:
    """Parse the first *NODE block of a deck into an (N, 3) point cloud."""
    coords, recording = [], False
    with open(inp_dir / f"HoodImpact_{run}.inp") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not recording:
                if stripped.upper() == "*NODE":
                    recording = True
                continue
            if stripped.startswith("*"):
                break
            parts = stripped.split(",")
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords)


def load_histories(acc_dir: Path, cache: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Return acceleration (12, 142, T) in g and the shared time vector."""
    if cache is not None and cache.exists():
        stored = np.load(cache)
        return stored["acc"].astype(np.float64), stored["time"].astype(np.float64)
    acc, times = None, None
    for design in range(NUM_DESIGNS):
        for loc in range(1, LOCS_PER_DESIGN + 1):
            run = LOCS_PER_DESIGN * design + loc
            frame = pd.read_csv(
                acc_dir / f"HoodImpact_{run}_SAE1000_interp1000.csv",
                usecols=["Time", "A(in g)"],
            )
            if acc is None:
                times = frame["Time"].to_numpy(np.float64)
                acc = np.empty((NUM_DESIGNS, LOCS_PER_DESIGN, len(times)), np.float64)
            acc[design, loc - 1] = frame["A(in g)"].to_numpy(np.float64)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, acc=acc, time=times)
    return acc, times


def hic15(acc: np.ndarray, dt: float, window: float = 0.015) -> np.ndarray:
    """Full HIC15 search over every window up to 15 ms, for a stack of histories."""
    magnitude = np.abs(acc)
    integral = np.concatenate(
        [
            np.zeros((acc.shape[0], 1)),
            np.cumsum(0.5 * (magnitude[:, 1:] + magnitude[:, :-1]) * dt, axis=1),
        ],
        axis=1,
    )
    best = np.zeros(acc.shape[0])
    for steps in range(1, int(np.floor(window / dt)) + 1):
        span = steps * dt
        candidate = ((integral[:, steps:] - integral[:, :-steps]) / span) ** 2.5 * span
        np.maximum(best, candidate.max(axis=1), out=best)
    return best


def cluster_designs(meshes: list[np.ndarray], tol_mm: float = 0.5):
    """Group designs whose hood surfaces agree, and grid the geometry difference.

    The 12 decks carry two mesh topologies, so designs cannot be compared node
    by node across families. Every hood is resampled onto one common XY grid
    and compared as a surface instead.
    """
    hoods = [mesh[:-IMPACTOR_NODES] for mesh in meshes]
    stacked = np.concatenate(hoods)
    x_lo, x_hi = np.percentile(stacked[:, 0], [2, 98])
    y_lo, y_hi = np.percentile(stacked[:, 1], [2, 98])
    grid_x, grid_y = np.meshgrid(np.linspace(x_lo, x_hi, 90), np.linspace(y_lo, y_hi, 90))
    surfaces = np.stack(
        [griddata(h[:, :2], h[:, 2], (grid_x, grid_y), method="linear") for h in hoods]
    )
    valid = np.isfinite(surfaces).all(axis=0)

    distance = np.array(
        [
            [np.abs(surfaces[i][valid] - surfaces[j][valid]).mean() for j in range(NUM_DESIGNS)]
            for i in range(NUM_DESIGNS)
        ]
    )
    labels, clusters = -np.ones(NUM_DESIGNS, int), []
    for design in range(NUM_DESIGNS):
        if labels[design] >= 0:
            continue
        members = np.where(distance[design] <= tol_mm)[0]
        labels[members] = len(clusters)
        clusters.append(members.tolist())

    spread = np.abs(surfaces - surfaces.mean(axis=0)).max(axis=0)
    spread[~valid] = np.nan
    return labels, clusters, distance, (grid_x, grid_y, spread)


def variance_shares(values: np.ndarray) -> dict:
    """Exact two-way decomposition on the balanced 12 x 142 grid."""
    grand = values.mean()
    by_loc, by_design = values.mean(axis=0), values.mean(axis=1)
    total = ((values - grand) ** 2).sum()
    location = NUM_DESIGNS * ((by_loc - grand) ** 2).sum()
    design = LOCS_PER_DESIGN * ((by_design - grand) ** 2).sum()
    interaction = ((values - by_design[:, None] - by_loc[None, :] + grand) ** 2).sum()
    return {
        "location": float(100 * location / total),
        "design": float(100 * design / total),
        "interaction": float(100 * interaction / total),
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def style_axes(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def build_figure(acc, times, peak, hic, labels, clusters, geometry, impact_xy, hood, out_dir):
    hic_range = (hic.max(axis=0) - hic.min(axis=0)) / hic.mean(axis=0)
    peak_range = (peak.max(axis=0) - peak.min(axis=0)) / peak.mean(axis=0)
    order = np.argsort(hic_range)
    picks = [order[0], order[len(order) // 2], order[-1]]
    tags = ["least design-sensitive location", "median location", "most design-sensitive location"]
    time_ms = times * 1e3

    fig = plt.figure(figsize=(15.6, 10.2), facecolor=SURFACE)
    # Two independent bands: the history row is tight, the summary row needs
    # room for a colorbar and two-line category labels.
    gs_top = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.835, bottom=0.545, wspace=0.22)
    gs_bottom = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.435, bottom=0.075, wspace=0.58)

    # -- (a) the histories themselves ---------------------------------------- #
    for column, (loc_index, tag) in enumerate(zip(picks, tags)):
        ax = fig.add_subplot(gs_top[0, column])
        style_axes(ax)
        for design in range(NUM_DESIGNS):
            ax.plot(time_ms, acc[design, loc_index], lw=0.9, alpha=0.45,
                    color=CLUSTER_COLORS[labels[design]], zorder=2)
        for cluster_id, members in enumerate(clusters):
            mean_curve = acc[members, loc_index].mean(axis=0)
            ax.plot(time_ms, mean_curve, lw=2.0, color=CLUSTER_COLORS[cluster_id], zorder=3,
                    solid_capstyle="round")
            if column == 2:  # relief for the sub-3:1 hue: label, never colour alone
                anchor = int(len(time_ms) * (0.19 + 0.17 * cluster_id))
                ax.annotate(
                    f"cluster {'ABCD'[cluster_id]}",
                    xy=(time_ms[anchor], mean_curve[anchor]),
                    xytext=(0, 11 if cluster_id % 2 == 0 else -17),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=CLUSTER_COLORS[cluster_id], fontweight="bold", zorder=5,
                    path_effects=[withStroke(linewidth=2.4, foreground=SURFACE)],
                )
        ax.set_title(
            f"({'abc'[column]})  loc {loc_index + 1} — {tag}\nHIC15 spans "
            f"{hic[:, loc_index].min():.0f}–{hic[:, loc_index].max():.0f}"
            f"  ({100 * hic_range[loc_index]:.0f}% of the mean)",
            fontsize=9.5, color=INK, pad=9, loc="left",
        )
        ax.set_xlabel("time (ms)", fontsize=9, color=INK_2)
        if column == 0:
            ax.set_ylabel("acceleration (g)", fontsize=9, color=INK_2)
        ax.set_xlim(0, time_ms[-1])

    legend = fig.legend(
        handles=[
            Line2D([], [], color=CLUSTER_COLORS[i], lw=2.0,
                   label=f"cluster {'ABCD'[i]} — designs {', '.join(str(d) for d in members)}")
            for i, members in enumerate(clusters)
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.900), ncol=4, frameon=False,
        fontsize=8.5, handlelength=1.8, columnspacing=2.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)

    # -- (d) where on the hood the design matters ---------------------------- #
    ax = fig.add_subplot(gs_bottom[0, 0])
    style_axes(ax, grid_axis=None)
    ax.scatter(hood[::9, 0], hood[::9, 1], s=1.0, color="#cfcec5", linewidths=0, zorder=1)
    grid_x, grid_y, spread = geometry
    ax.contour(grid_x, grid_y, uniform_filter(np.nan_to_num(spread), size=6), levels=[2.0],
               colors=[MUTED], linewidths=1.1, zorder=2)
    cmap = LinearSegmentedColormap.from_list("hood_blue", SEQUENTIAL)
    points = ax.scatter(
        impact_xy[:, 0], impact_xy[:, 1], c=100 * hic_range, cmap=cmap, s=46,
        edgecolors=SURFACE, linewidths=0.7, zorder=3, vmin=0, vmax=60,
    )
    bar = fig.colorbar(points, ax=ax, fraction=0.040, pad=0.025, shrink=0.78)
    bar.set_label("HIC15 range (%)", fontsize=8, color=INK_2, labelpad=3)
    bar.ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.8)
    bar.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title("(d)  where the design matters\n142 EuroNCAP impact points, plan view",
                 fontsize=9.5, color=INK, pad=9, loc="left")
    ax.set_xlabel("X1 (mm)", fontsize=9, color=INK_2)
    ax.set_ylabel("X2 (mm)", fontsize=9, color=INK_2)
    outline = Line2D([], [], color=MUTED, lw=1.1, label="hood geometry differs\nbetween designs by >2 mm")
    ax.legend(handles=[outline], loc="lower left", frameon=False, fontsize=7,
              labelcolor=INK_2, handlelength=1.4, borderaxespad=0.1)

    # -- (e) exact variance decomposition ------------------------------------ #
    ax = fig.add_subplot(gs_bottom[0, 1])
    style_axes(ax, grid_axis="x")
    peak_shares, hic_shares = variance_shares(peak), variance_shares(hic)
    rows = ["interaction", "design", "location"]
    row_labels = ["design × location", "design", "impact location"]
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [peak_shares[k] for k in rows], height=0.34, color=BLUE,
            label="peak |a|", zorder=3)
    ax.barh(y - 0.19, [hic_shares[k] for k in rows], height=0.34, color=BLUE_LIGHT,
            label="HIC15", zorder=3)
    for index, key in enumerate(rows):
        for offset, shares in ((0.19, peak_shares), (-0.19, hic_shares)):
            ax.text(shares[key] + 1.8, index + offset, f"{shares[key]:.1f}%", va="center",
                    fontsize=8, color=INK_2)
    ax.set_yticks(y, row_labels, fontsize=8.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of total variance across all 1704 runs (%)", fontsize=9, color=INK_2)
    ax.set_title("(e)  what drives the response\nbalanced 12 × 142 decomposition",
                 fontsize=9.5, color=INK, pad=9, loc="left")
    ax.annotate("a uniform stiffness shift is negligible;\nthe design acts through the location",
                xy=(0.97, 0.63), xycoords="axes fraction", ha="right", va="center",
                fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc=(0.60, 0.28), labelcolor=INK_2)

    # -- (f) is the design effect above the surrogate's own error? ------------ #
    # Designs inside one cluster have near-identical hoods, so the spread they
    # show is the solver's own run-to-run scatter. That is the floor a real
    # geometry effect has to clear.
    ax = fig.add_subplot(gs_bottom[0, 2])
    style_axes(ax)
    cluster_means = np.array([hic[members].mean(axis=0) for members in clusters])
    geometry_range = (cluster_means.max(axis=0) - cluster_means.min(axis=0)) / cluster_means.mean(axis=0)
    noise_range = np.array(
        [(hic[m].max(axis=0) - hic[m].min(axis=0)) / hic[m].mean(axis=0) for m in clusters]
    ).mean(axis=0)
    fraction = 100 * np.arange(1, LOCS_PER_DESIGN + 1) / LOCS_PER_DESIGN
    ax.plot(100 * np.sort(noise_range), fraction, lw=1.6, color=MUTED, zorder=3,
            solid_capstyle="round", label="within a cluster\n(near-identical hoods: solver noise)")
    ax.plot(100 * np.sort(geometry_range), fraction, lw=2.0, color=BLUE, zorder=4,
            solid_capstyle="round", label="between clusters\n(the real geometry effect)")
    ax.axvline(SURROGATE_HIC_MAPE, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"surrogate error\n({SURROGATE_HIC_MAPE:.0f}% MAPE)",
        xy=(SURROGATE_HIC_MAPE, 2), xytext=(7, 2), textcoords="offset points",
        fontsize=7.5, color=MUTED, va="bottom",
    )
    clears = 100 * (geometry_range > noise_range).mean()
    ax.annotate(
        f"the geometry effect clears the noise\nfloor at {clears:.0f}% of locations "
        f"(median {np.median(geometry_range / noise_range):.1f}×)",
        xy=(0.97, 0.33), xycoords="axes fraction", ha="right", fontsize=8.5, color=INK_2,
    )
    ax.set_xlabel("HIC15 range at one impact location (% of mean)", fontsize=9, color=INK_2)
    ax.set_ylabel("locations at or below (%)", fontsize=9, color=INK_2)
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 100)
    ax.set_title("(f)  is the design effect real?\ngeometry effect against the noise floor",
                 fontsize=9.5, color=INK, pad=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_2,
              handlelength=1.6, borderaxespad=0.4)

    fig.text(0.055, 0.965,
             "Subtle hood geometry changes do move the acceleration history — but the impact location decides whether it matters",
             fontsize=14, color=INK, fontweight="bold", ha="left", va="center")
    fig.text(0.055, 0.930,
             "12 designs × 142 EuroNCAP impact locations, every combination simulated (1704 runs). "
             "(a)–(c) show all 12 histories at one location; thin lines are individual designs, thick lines are cluster means.",
             fontsize=9, color=MUTED, ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / "design_sensitivity_1704.png", out_dir / "design_sensitivity_1704.pdf"
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    fig.savefig(pdf, facecolor=SURFACE)
    plt.close(fig)
    return png, pdf, peak_shares, hic_shares, hic_range, peak_range, geometry_range, noise_range


# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path,
                        default=root / "Data" / "HoodImpact_1704_EuroNCAP")
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    parser.add_argument("--cache", type=Path,
                        default=root / "figures" / "_acc_1704_cache.npz")
    args = parser.parse_args(argv)

    acc, times = load_histories(args.data_dir / "output_history_acc", args.cache)
    dt = float(np.diff(times).mean())
    peak = np.abs(acc).max(axis=2)
    hic = hic15(acc.reshape(-1, acc.shape[2]), dt).reshape(NUM_DESIGNS, LOCS_PER_DESIGN)

    inp_dir = args.data_dir / "inp_files"
    meshes = [load_nodes(inp_dir, LOCS_PER_DESIGN * d + 1) for d in range(NUM_DESIGNS)]
    labels, clusters, distance, geometry = cluster_designs(meshes)
    hood = meshes[0][:-IMPACTOR_NODES]
    impact_xy = pd.read_csv(args.data_dir / "impact_locations_142.csv")[["X1", "X2"]].to_numpy()

    png, pdf, peak_shares, hic_shares, hic_range, peak_range, geometry_range, noise_range = (
        build_figure(acc, times, peak, hic, labels, clusters, geometry, impact_xy, hood,
                     args.out_dir)
    )

    within = np.concatenate([(hic[m] - hic[m].mean(axis=0)).ravel() for m in clusters])
    # Does the design matter most near the geometry that actually changed?
    grid_x, grid_y, spread = geometry
    changed = np.column_stack([grid_x[spread > 2.0], grid_y[spread > 2.0]])
    to_change = np.array([np.linalg.norm(changed - xy, axis=1).min() for xy in impact_xy])
    summary = {
        "runs": int(acc.shape[0] * acc.shape[1]),
        "geometry_clusters": {"ABCD"[i]: m for i, m in enumerate(clusters)},
        "max_within_cluster_surface_difference_mm": float(
            max(distance[np.ix_(m, m)].max() for m in clusters)
        ),
        "min_between_cluster_surface_difference_mm": float(
            min(distance[np.ix_(a, b)].min()
                for i, a in enumerate(clusters) for b in clusters[i + 1:])
        ),
        "variance_shares_percent": {"peak_abs_acceleration": peak_shares, "hic15": hic_shares},
        "hic_design_range_percent": {
            "median": float(100 * np.median(hic_range)),
            "p90": float(100 * np.percentile(hic_range, 90)),
            "max": float(100 * hic_range.max()),
        },
        "peak_design_range_percent": {
            "median": float(100 * np.median(peak_range)),
            "p90": float(100 * np.percentile(peak_range, 90)),
            "max": float(100 * peak_range.max()),
        },
        "within_cluster_hic_std_percent_of_mean": float(100 * within.std(ddof=1) / hic.mean()),
        # The honest design signal: cluster-level range, against the run-to-run
        # scatter shown by designs whose hoods are identical to 0.06 mm.
        "hic_geometry_effect_vs_noise_floor": {
            "geometry_range_median_percent": float(100 * np.median(geometry_range)),
            "noise_floor_median_percent": float(100 * np.median(noise_range)),
            "median_ratio": float(np.median(geometry_range / noise_range)),
            "locations_clearing_noise_floor_percent": float(
                100 * (geometry_range > noise_range).mean()
            ),
            "locations_above_surrogate_error_percent": float(
                100 * (geometry_range > SURROGATE_HIC_MAPE / 100).mean()
            ),
        },
        # Fixed millimetre bins would be empty for some thresholds, so compare
        # the nearest and farthest quartile of locations instead.
        "hic_range_vs_distance_to_changed_geometry": {
            "pearson_r": float(np.corrcoef(geometry_range, to_change)[0, 1]),
            "nearest_quartile_median_percent": float(
                100 * np.median(geometry_range[to_change <= np.percentile(to_change, 25)])
            ),
            "farthest_quartile_median_percent": float(
                100 * np.median(geometry_range[to_change >= np.percentile(to_change, 75)])
            ),
            "max_distance_mm": float(to_change.max()),
        },
    }
    (args.out_dir / "design_sensitivity_1704.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {png}\n      {pdf}")
    return summary


if __name__ == "__main__":
    main()
