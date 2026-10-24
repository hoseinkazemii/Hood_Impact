"""Do the selected change nodes relate to the near-clone response differences?

Every near-clone pair (same mesh, fewer than 300 nodes moved by more than
0.1 mm; design_change_nodes.py) at every one of the 142 impact locations gives
one row of ``pair_location_table.csv``:

* ``rms_diff_g``: RMS over time of a_b(t) - a_a(t) on the model's 63-point grid;
* ``distance_mm``: XY distance from the impact point to the nearest node that
  moved between the two designs (the change tokens that tell the pair apart);
* ``hic_diff_pct`` / ``peak_diff_g``: HIC15 and peak acceleration of the design
  that is raised at that nearest node, minus the other design's;
* ``dt_mismatch``: |log| of the pair's ratio of explicit increments, the proxy
  for a different stable time step (within_family_dt_artifact).

Figures (figures/change_node_relevance/):
1. maps: where each pair differs, with its changed nodes;
2. distance: difference vs distance, a random-location check, and the far-field
   floor against the time-step mismatch;
3. direction: whether the raised design responds harder;
4. timing: when the differences appear, near and far from the edit;
5. examples: one pair's curves near, mid-way and far from its edit.
"""
import io
from itertools import combinations
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from design_change_nodes import MOVED_THRESHOLD_MM, NEAR_CLONE_MAX_NODES
from design_difference_data import (
    DEFAULT_DATA_DIR, MODEL_TIME_STRIDE, PANELS, design_increments, load_design_geometry, load_histories,
    load_locations, load_run_table,
)
from hic15 import batched_hic

OUT = Path(__file__).resolve().parent / "figures" / "change_node_relevance"
BINS = (0, 50, 100, 150, 250, 400, np.inf)
CENTERS = np.array([25, 75, 125, 200, 325, 550])
BIN_LABELS = ["0–50", "50–100", "100–150", "150–250", "250–400", "400+"]
BANDS = (("near, < 100 mm", 0, 100), ("mid, 100–250 mm", 100, 250), ("far, ≥ 250 mm", 250, np.inf))
DT_MATCHED = 0.10  # |log increment ratio| below this counts as the same time step
CONTROL_DRAWS = 50
GROUPS = {"train pairs (A, C)": lambda a, b: a not in (4, 5, 10, 11),
          "10–11 (validation)": lambda a, b: (a, b) == (10, 11),
          "4–5 (test)": lambda a, b: (a, b) == (4, 5)}

# Palette (dataviz reference instance): categorical slots, sequential blue, recessive ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    """Write the PNG, retrying: an open viewer or file scanner can briefly lock the old file."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi)
    plt.close(fig)
    for attempt in range(10):
        try:
            (OUT / name).write_bytes(buffer.getvalue())
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(1)


def binned_median(distance, value):
    out = []
    for low, high in zip(BINS[:-1], BINS[1:]):
        inside = value[(distance >= low) & (distance < high)]
        out.append(np.median(inside) if len(inside) else np.nan)
    return np.array(out)


def build_table(rng):
    geometries = {d: load_design_geometry(d, DEFAULT_DATA_DIR) for d in range(12)}
    times, acceleration = load_histories(DEFAULT_DATA_DIR)
    on_grid = acceleration[..., ::MODEL_TIME_STRIDE]
    hic, peak = batched_hic(times, acceleration), acceleration.max(axis=-1)
    increments = design_increments(load_run_table(DEFAULT_DATA_DIR))
    impact_xy = load_locations(DEFAULT_DATA_DIR)[["X1", "X2"]].to_numpy()
    rows, pairs, abs_diff = [], {}, {}
    for a, b in combinations(range(12), 2):
        if not np.array_equal(geometries[a]["ids"], geometries[b]["ids"]):
            continue
        moved = np.linalg.norm(geometries[b]["xyz"] - geometries[a]["xyz"], axis=1) > MOVED_THRESHOLD_MM
        if not 0 < moved.sum() < NEAR_CLONE_MAX_NODES:
            continue
        nodes, rise = geometries[a]["xyz"][moved], geometries[b]["xyz"][moved, 2] - geometries[a]["xyz"][moved, 2]
        distance, nearest = cKDTree(nodes[:, :2]).query(impact_xy)
        # +1 when design b is the raised one at the nearest changed node.
        sign = np.where(rise[nearest] > 0, 1.0, -1.0)
        control = np.array([cKDTree(nodes[:, :2] - nodes[:, :2].mean(0) + impact_xy[rng.integers(len(impact_xy))])
                            .query(impact_xy)[0] for _ in range(CONTROL_DRAWS)])
        pairs[(a, b)] = {"nodes": nodes, "control": control}
        abs_diff[(a, b)] = np.abs(acceleration[b] - acceleration[a])
        rows.append(pd.DataFrame({
            "a": a, "b": b, "loc": np.arange(1, 143), "x_mm": impact_xy[:, 0], "y_mm": impact_xy[:, 1],
            "changed_nodes": int(moved.sum()), "distance_mm": distance,
            "rms_diff_g": np.sqrt(np.mean((on_grid[b] - on_grid[a]) ** 2, axis=-1)),
            "hic_diff_pct": sign * (hic[b] - hic[a]) / (0.5 * (hic[a] + hic[b])) * 100,
            "peak_diff_g": sign * (peak[b] - peak[a]),
            "dt_mismatch": abs(np.log(increments[a] / increments[b])),
        }))
    table = pd.concat(rows, ignore_index=True)
    table["group"] = [next(name for name, keep in GROUPS.items() if keep(a, b)) for a, b in zip(table.a, table.b)]
    return table, pairs, abs_diff, times, acceleration, geometries, impact_xy


def figure_maps(table, pairs, geometries):
    outer = geometries[0]["xyz"][geometries[0]["panel"] == PANELS.index("outer")][::6]
    outer_c = geometries[6]["xyz"][geometries[6]["panel"] == PANELS.index("outer")][::6]
    keys = list(pairs)
    fig, axes = plt.subplots(4, 4, figsize=(15, 15.5), constrained_layout=True)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax, (a, b) in zip(axes.flat, keys):
        rows = table[(table.a == a) & (table.b == b)]
        hull = outer if a < 6 else outer_c
        ax.scatter(hull[:, 1], hull[:, 0], s=0.6, color="#d9d8d3", linewidths=0, rasterized=True)
        dots = ax.scatter(rows.y_mm, rows.x_mm, c=rows.rms_diff_g, cmap=SEQUENTIAL, vmin=0, vmax=25, s=46,
                          edgecolors="white", linewidths=0.6, zorder=3)
        nodes = pairs[(a, b)]["nodes"]
        ax.scatter(nodes[:, 1], nodes[:, 0], s=4, color=ORANGE, linewidths=0, zorder=4)
        dt = rows.dt_mismatch.iloc[0]
        ax.set_title(f"{a}–{b} · {rows.changed_nodes.iloc[0]} nodes · Δt mismatch {dt:.2f}"
                     + (" (test)" if (a, b) == (4, 5) else " (val)" if (a, b) == (10, 11) else ""),
                     loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(dots, ax=axes[-1, len(keys) % 4:], orientation="horizontal", fraction=0.5, shrink=0.8)
    bar.set_label("RMS acceleration difference between the two designs (g)")
    axes[-1, -1].scatter([], [], s=20, color=ORANGE, label="nodes that differ between the pair")
    axes[-1, -1].legend(loc="center", frameon=False, fontsize=10)
    fig.suptitle("Top view: impact points coloured by how much the pair differs; orange = changed nodes",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.set_dpi(130)
    save(fig, "fig1_maps.png")


def figure_distance(table, pairs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    ax = axes[0]
    control_d = np.concatenate([pairs[(a, b)]["control"].ravel() for a, b in pairs])
    control_v = np.concatenate([np.tile(table[(table.a == a) & (table.b == b)].rms_diff_g.to_numpy(), CONTROL_DRAWS)
                                for a, b in pairs])
    ax.scatter(np.minimum(table.distance_mm, 800), np.minimum(table.rms_diff_g, 35), s=6, color="#86b6ef",
               alpha=0.45, linewidths=0, label="one pair at one location")
    ax.plot(CENTERS, binned_median(table.distance_mm.to_numpy(), table.rms_diff_g.to_numpy()), "-o", color=BLUE,
            lw=2, ms=7, label="median, the pair's changed nodes")
    ax.plot(CENTERS, binned_median(control_d, control_v), "--o", color=MUTED, lw=2, ms=7, mfc="white",
            label="median, same nodes at a random spot")
    style(ax, "A · Difference vs distance to the changed nodes")
    ax.set_ylim(0, 36); ax.set_xlim(0, 820)
    ax.set_xlabel("distance from impact to nearest changed node (mm)")
    ax.set_ylabel("RMS acceleration difference (g)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for (name, rows), color in zip(table.groupby("group", sort=False), (BLUE, ORANGE, AQUA)):
        ax.plot(CENTERS, binned_median(rows.distance_mm.to_numpy(), rows.rms_diff_g.to_numpy()), "-o",
                color=color, lw=2, ms=7, label=name)
    style(ax, "B · Median by pair group")
    ax.set_ylim(0, 22); ax.set_xlim(0, 620)
    ax.set_xlabel("distance from impact to nearest changed node (mm)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    per_pair = table.groupby(["a", "b"]).apply(lambda r: pd.Series({
        "dt": r.dt_mismatch.iloc[0],
        "near": r.rms_diff_g[r.distance_mm < 100].median(),
        "far": r.rms_diff_g[r.distance_mm >= 250].median()}), include_groups=False).reset_index()
    # Near/far as dark/light of the map's sequential hue; panel B's colours name pair groups.
    ax.scatter(per_pair.dt, per_pair.near, s=60, color="#184f95", label="near the edit (< 100 mm)", zorder=3,
               edgecolors="white", linewidths=1)
    ax.scatter(per_pair.dt, per_pair.far, s=60, color="#86b6ef", label="far from the edit (≥ 250 mm)", zorder=3,
               edgecolors="white", linewidths=1)
    for row in per_pair.itertuples():
        ax.annotate(f"{row.a}–{row.b}", (row.dt, row.far), textcoords="offset points", xytext=(5, -10),
                    fontsize=8, color=MUTED)
    style(ax, "C · Per pair, against the time-step mismatch")
    ax.set_xlabel("time-step mismatch |log(increments a / increments b)|")
    ax.set_ylabel("median RMS difference (g)")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.set_dpi(140)
    save(fig, "fig2_distance.png")
    return per_pair


def figure_direction(table):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=1)
    ax.scatter(np.minimum(table.distance_mm, 800), np.clip(table.hic_diff_pct, -40, 40), s=6, color="#86b6ef",
               alpha=0.5, linewidths=0, label="one pair at one location")
    ax.plot(CENTERS, binned_median(table.distance_mm.to_numpy(), table.hic_diff_pct.to_numpy()), "-o", color=BLUE,
            lw=2, ms=7, label="median")
    style(ax, "A · HIC15 of the raised design minus the flat one")
    ax.set_xlabel("distance from impact to nearest changed node (mm)")
    ax.set_ylabel("HIC15 difference (% of the pair mean)")
    ax.set_ylim(-42, 42); ax.set_xlim(0, 820)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    width = 0.38
    for offset, column, color, label in ((-width / 2, "hic_diff_pct", BLUE, "HIC15"),
                                         (width / 2, "peak_diff_g", AQUA, "peak acceleration")):
        share = [100 * (table[column][(table.distance_mm >= lo) & (table.distance_mm < hi)] > 0).mean()
                 for lo, hi in zip(BINS[:-1], BINS[1:])]
        ax.bar(np.arange(len(share)) + offset, share, width=width - 0.04, color=color, label=label)
    ax.axhline(50, color=MUTED, lw=1, ls="--")
    ax.text(-0.45, 51.5, "coin flip", ha="left", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    style(ax, "B · How often the raised design responds harder")
    ax.set_xlabel("distance from impact to nearest changed node (mm)")
    ax.set_ylabel("% of pair-locations where raised > flat")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.set_dpi(140)
    save(fig, "fig3_direction.png")


def figure_timing(table, abs_diff, times, acceleration):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    ms = times * 1000
    matched = table.dt_mismatch < DT_MATCHED
    for ax, (title, keep) in zip(axes[:2], ((f"A · Same time step (mismatch < {DT_MATCHED})", matched),
                                            ("B · Different time step", ~matched))):
        for (label, lo, hi), color in zip(BANDS, (BLUE, ORANGE, MUTED)):
            rows = table[keep & (table.distance_mm >= lo) & (table.distance_mm < hi)]
            curves = np.stack([abs_diff[(r.a, r.b)][r.loc - 1] for r in rows.itertuples()])
            ax.plot(ms, curves.mean(0), color=color, lw=2, label=f"{label} ({len(rows)})")
        style(ax, title)
        ax.set_xlabel("time (ms)")
        ax.set_ylim(0, 30); ax.set_xlim(0, 25)
        ax.legend(frameon=False, fontsize=9, title="impact location", title_fontsize=9)
    axes[0].set_ylabel("mean |a_b(t) − a_a(t)| (g)")
    ax = axes[2]
    mean_curve = acceleration.reshape(-1, acceleration.shape[-1]).mean(0)
    band = np.percentile(acceleration.reshape(-1, acceleration.shape[-1]), [25, 75], axis=0)
    ax.fill_between(ms, band[0], band[1], color="#cde2fb", label="25–75% of all 1704 runs")
    ax.plot(ms, mean_curve, color=BLUE, lw=2, label="mean")
    style(ax, "C · For reference: the acceleration itself")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("acceleration (g)"); ax.set_xlim(0, 25)
    ax.legend(frameon=False, fontsize=9)
    fig.set_dpi(140)
    save(fig, "fig4_timing.png")
    pairs_used = sorted({(r.a, r.b) for r in table[matched].itertuples()})
    return pairs_used


def figure_examples(table, times, acceleration, pair=(6, 9)):
    a, b = pair
    rows = table[(table.a == a) & (table.b == b)]
    picks = [rows.loc[rows.distance_mm.idxmin()],
             rows.loc[(rows.distance_mm - 180).abs().idxmin()],
             rows.loc[(rows.distance_mm - 450).abs().idxmin()]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True, constrained_layout=True)
    for ax, row, name in zip(axes, picks, ("near", "mid-way", "far")):
        loc = int(row["loc"])
        ax.plot(times * 1000, acceleration[a, loc - 1], color=BLUE, lw=2, label=f"design {a} (base)")
        ax.plot(times * 1000, acceleration[b, loc - 1], color=ORANGE, lw=2, label=f"design {b} (edited)")
        style(ax, f"{name}: location {loc}, {row.distance_mm:.0f} mm from the edit · RMS diff {row.rms_diff_g:.1f} g")
        ax.set_xlabel("time (ms)"); ax.set_xlim(0, 25)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("acceleration (g)")
    fig.set_dpi(140)
    save(fig, "fig5_examples.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    rng = np.random.default_rng(0)
    table, pairs, abs_diff, times, acceleration, geometries, _ = build_table(rng)
    table.to_csv(OUT / "pair_location_table.csv", index=False)
    figure_maps(table, pairs, geometries)
    per_pair = figure_distance(table, pairs)
    figure_direction(table)
    matched_pairs = figure_timing(table, abs_diff, times, acceleration)
    figure_examples(table, times, acceleration)

    near, far = table.distance_mm < 100, table.distance_mm >= 250
    summary = {
        "points": len(table), "pairs": len(pairs),
        "spearman_distance_vs_rms_diff": spearmanr(table.distance_mm, table.rms_diff_g)[0],
        "median_rms_diff_by_bin_g": dict(zip(BIN_LABELS, binned_median(table.distance_mm.to_numpy(),
                                                                       table.rms_diff_g.to_numpy()).round(2))),
        "raised_has_higher_hic_pct": {label: 100 * (table.hic_diff_pct[(table.distance_mm >= lo)
                                                                      & (table.distance_mm < hi)] > 0).mean()
                                      for label, lo, hi in BANDS},
        "median_hic_diff_pct": {label: table.hic_diff_pct[(table.distance_mm >= lo) & (table.distance_mm < hi)].median()
                                for label, lo, hi in BANDS},
        "median_peak_diff_g": {label: table.peak_diff_g[(table.distance_mm >= lo) & (table.distance_mm < hi)].median()
                               for label, lo, hi in BANDS},
        "same_time_step_pairs": [list(p) for p in matched_pairs],
        "per_pair": per_pair.round(3).to_dict(orient="records"),
        "near_points": int(near.sum()), "far_points": int(far.sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
