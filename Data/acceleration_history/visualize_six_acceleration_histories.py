import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SAMPLES = ["1", "2", "3", "4", "5", "6"]
DEFAULT_OUTPUT = "six_acceleration_histories"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize acceleration histories for six hood impact samples."
    )
    parser.add_argument(
        "samples",
        nargs="*",
        default=DEFAULT_SAMPLES,
        help=(
            "Six sample IDs or CSV paths to plot. "
            "IDs are expanded to HoodImpactor_<ID>_SAE1000.csv. "
            f"Default: {' '.join(DEFAULT_SAMPLES)}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing acceleration CSV files. Default: current directory.",
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
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Plot the six histories on one shared axis instead of a 2x3 panel.",
    )
    return parser.parse_args()


def sample_to_path(sample, data_dir):
    sample_path = Path(sample)
    if sample_path.suffix.lower() == ".csv":
        return sample_path if sample_path.is_absolute() else data_dir / sample_path
    return data_dir / f"HoodImpactor_{sample}_SAE1000.csv"


def label_from_path(path):
    name = path.stem
    if name.startswith("HoodImpactor_") and name.endswith("_SAE1000"):
        return f"Sample {name.removeprefix('HoodImpactor_').removesuffix('_SAE1000')}"
    return name.replace("_", " ")


def load_acceleration(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Time", "A(in g)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    time_ms = df["Time"].to_numpy(dtype=float) * 1000.0
    acc_g = df["A(in g)"].to_numpy(dtype=float)
    return time_ms, acc_g


def load_samples(samples, data_dir):
    if len(samples) != 6:
        raise ValueError(f"Expected exactly 6 samples, got {len(samples)}.")

    histories = []
    for sample in samples:
        csv_path = sample_to_path(sample, data_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {csv_path}")
        time_ms, acc_g = load_acceleration(csv_path)
        histories.append(
            {
                "path": csv_path,
                "label": label_from_path(csv_path),
                "time_ms": time_ms,
                "acc_g": acc_g,
                "peak_idx": int(np.nanargmax(acc_g)),
            }
        )
    return histories


def style_axis(ax):
    ax.grid(True, color="#d9dee7", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(colors="#4b5563", labelsize=9)


def set_common_limits(axes, histories):
    x_min = min(np.nanmin(history["time_ms"]) for history in histories)
    x_max = max(np.nanmax(history["time_ms"]) for history in histories)
    y_max = max(np.nanmax(history["acc_g"]) for history in histories)

    for ax in np.ravel(axes):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max(y_max * 1.12, 20))


def draw_panel(histories):
    colors = ["#1f6f8b", "#d64b3a", "#2f855a", "#805ad5", "#b7791f", "#2b6cb0"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, facecolor="white")
    fig.suptitle(
        "Acceleration-Time Histories for Six Hood Impact Samples",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )

    for ax, history, color in zip(np.ravel(axes), histories, colors):
        time_ms = history["time_ms"]
        acc_g = history["acc_g"]
        peak_idx = history["peak_idx"]
        peak_time = time_ms[peak_idx]
        peak_acc = acc_g[peak_idx]

        ax.plot(time_ms, acc_g, color=color, linewidth=1.8)
        ax.fill_between(time_ms, acc_g, 0, color=color, alpha=0.16)
        ax.scatter([peak_time], [peak_acc], s=36, color="#111827", zorder=5)
        ax.set_title(
            f"{history['label']} | peak {peak_acc:.0f} g",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color="#1f2933",
            pad=8,
        )
        style_axis(ax)

    set_common_limits(axes, histories)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time after impact (ms)", fontsize=10, color="#374151")
    for ax in axes[:, 0]:
        ax.set_ylabel("Acceleration (g)", fontsize=10, color="#374151")

    fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.08, hspace=0.28, wspace=0.13)
    return fig


def draw_overlay(histories):
    colors = ["#1f6f8b", "#d64b3a", "#2f855a", "#805ad5", "#b7791f", "#2b6cb0"]

    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor="white")
    fig.suptitle(
        "Acceleration-Time Histories for Six Hood Impact Samples",
        x=0.065,
        y=0.955,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )

    for history, color in zip(histories, colors):
        time_ms = history["time_ms"]
        acc_g = history["acc_g"]
        peak_acc = acc_g[history["peak_idx"]]
        ax.plot(time_ms, acc_g, color=color, linewidth=2.0, label=f"{history['label']} ({peak_acc:.0f} g)")

    set_common_limits([ax], histories)
    style_axis(ax)
    ax.set_xlabel("Time after impact (ms)", fontsize=11, color="#374151")
    ax.set_ylabel("Acceleration (g)", fontsize=11, color="#374151")
    ax.legend(title="Peak acceleration", frameon=False, loc="upper right", fontsize=10, title_fontsize=10)

    fig.subplots_adjust(left=0.075, right=0.975, top=0.88, bottom=0.12)
    return fig


def save_figure(fig, output_stem, dpi):
    png_path = Path(f"{output_stem}.png")
    svg_path = Path(f"{output_stem}.svg")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    histories = load_samples(args.samples, data_dir)
    fig = draw_overlay(histories) if args.overlay else draw_panel(histories)
    png_path, svg_path = save_figure(fig, args.output, args.dpi)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
