import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.path import Path as MplPath


DEFAULT_INPUT = "HoodImpactor_100_SAE1000.csv"
DEFAULT_OUTPUT = "single_acceleration_history_visual"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a slide-ready visualization for one hood impact acceleration history."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Acceleration CSV to plot. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file stem. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG export resolution. Default: 300",
    )
    return parser.parse_args()


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_drop_scene(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    clean_axes(ax)

    hood_color = "#58606a"
    hood_shadow = "#c9ced6"
    ball_color = "#d64b3a"
    accent = "#1f6f8b"

    # Hood side profile.
    hood_vertices = [
        (0.08, 0.26),
        (0.23, 0.34),
        (0.48, 0.40),
        (0.76, 0.39),
        (0.92, 0.32),
        (0.90, 0.22),
        (0.10, 0.18),
        (0.08, 0.26),
    ]
    hood_codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    hood = patches.PathPatch(
        MplPath(hood_vertices, hood_codes),
        facecolor="#e9edf2",
        edgecolor=hood_color,
        linewidth=2.0,
    )
    ax.add_patch(hood)

    ax.plot([0.16, 0.88], [0.20, 0.24], color=hood_shadow, lw=4, solid_capstyle="round")
    ax.plot([0.34, 0.68], [0.37, 0.38], color="#ffffff", lw=2, alpha=0.85)

    # Single impact location and sensor cue.
    impact_xy = (0.55, 0.39)
    ax.scatter(*impact_xy, s=120, color=accent, zorder=4)
    ax.scatter(*impact_xy, s=330, facecolor="none", edgecolor=accent, lw=1.6, alpha=0.35, zorder=3)
    ax.plot([impact_xy[0], impact_xy[0]], [0.23, impact_xy[1]], color=accent, lw=2)
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.49, 0.16),
            0.12,
            0.07,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#ffffff",
            edgecolor=accent,
            linewidth=1.6,
        )
    )

    # Ball/head impactor and drop arrow.
    ball_center = (impact_xy[0], 0.78)
    ax.add_patch(
        patches.Circle(
            ball_center,
            0.075,
            facecolor=ball_color,
            edgecolor="#8f2b23",
            linewidth=2.0,
            zorder=5,
        )
    )
    ax.add_patch(
        patches.Circle(
            (ball_center[0] - 0.025, ball_center[1] + 0.025),
            0.022,
            facecolor="#ffffff",
            edgecolor="none",
            alpha=0.55,
            zorder=6,
        )
    )
    ax.annotate(
        "",
        xy=(impact_xy[0], impact_xy[1] + 0.045),
        xytext=(impact_xy[0], ball_center[1] - 0.095),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ball_color),
    )

    ax.text(0.10, 0.91, "Head impactor drop", fontsize=15, fontweight="bold", color="#1f2933")
    ax.text(0.10, 0.845, "One hood location", fontsize=11, color="#4b5563")
    ax.text(0.14, 0.105, "Hood", fontsize=11, color="#4b5563")
    ax.text(0.64, 0.165, "sensor", fontsize=10, color=accent, va="center")


def load_acceleration(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Time", "A(in g)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    time_ms = df["Time"].to_numpy(dtype=float) * 1000.0
    acc_g = df["A(in g)"].to_numpy(dtype=float)
    return time_ms, acc_g


def draw_acceleration_history(ax, time_ms, acc_g, label):
    line_color = "#1f6f8b"
    fill_color = "#b8dbe7"
    peak_color = "#d64b3a"

    peak_idx = int(np.nanargmax(acc_g))
    peak_time = time_ms[peak_idx]
    peak_acc = acc_g[peak_idx]

    ax.plot(time_ms, acc_g, color=line_color, lw=2.4)
    ax.fill_between(time_ms, acc_g, 0, color=fill_color, alpha=0.40)
    ax.scatter([peak_time], [peak_acc], s=58, color=peak_color, zorder=5)
    ax.annotate(
        f"Peak {peak_acc:.0f} g",
        xy=(peak_time, peak_acc),
        xytext=(peak_time + 3.0, peak_acc * 0.92),
        fontsize=11,
        color="#1f2933",
        arrowprops=dict(arrowstyle="->", lw=1.3, color=peak_color),
    )

    ax.set_title("Recorded acceleration history", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.text(
        0.015,
        0.965,
        f"Example trace: {label}",
        transform=ax.transAxes,
        fontsize=10,
        color="#6b7280",
        ha="left",
        va="top",
    )
    ax.set_xlabel("Time after impact (ms)", fontsize=11)
    ax.set_ylabel("Acceleration (g)", fontsize=11)
    ax.set_xlim(time_ms.min(), time_ms.max())
    ax.set_ylim(0, max(peak_acc * 1.15, 20))
    ax.grid(True, color="#d9dee7", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(colors="#4b5563")


def make_visual(input_csv, output_stem, dpi):
    input_path = Path(input_csv)
    time_ms, acc_g = load_acceleration(input_path)

    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.35],
        left=0.055,
        right=0.965,
        top=0.80,
        bottom=0.14,
        wspace=0.19,
    )

    scene_ax = fig.add_subplot(grid[0, 0])
    plot_ax = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "One hood impact produces one acceleration-time record",
        x=0.055,
        y=0.955,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#111827",
    )

    draw_drop_scene(scene_ax)
    draw_acceleration_history(plot_ax, time_ms, acc_g, input_path.stem.replace("_", " "))
    png_path = Path(f"{output_stem}.png")
    svg_path = Path(f"{output_stem}.svg")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return png_path, svg_path


if __name__ == "__main__":
    args = parse_args()
    png, svg = make_visual(args.input, args.output, args.dpi)
    print(f"Saved {png}")
    print(f"Saved {svg}")
