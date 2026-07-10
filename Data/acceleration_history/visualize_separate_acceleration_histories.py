import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SAMPLES = ["501", "502", "574", "889", "623", "628"]
DEFAULT_PREFIX = "acceleration_history_sample"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save one acceleration-history PNG for each selected hood impact sample."
    )
    parser.add_argument(
        "samples",
        nargs="*",
        default=DEFAULT_SAMPLES,
        help=(
            "Sample IDs or CSV paths to plot. IDs are expanded to "
            "HoodImpactor_<ID>_SAE1000.csv. "
            f"Default: {' '.join(DEFAULT_SAMPLES)}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing acceleration CSV files. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where PNG files are saved. Default: current directory.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Output file prefix. Default: {DEFAULT_PREFIX}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG export resolution. Default: 300",
    )
    return parser.parse_args()


def sample_to_path(sample, data_dir):
    sample_path = Path(sample)
    if sample_path.suffix.lower() == ".csv":
        return sample_path if sample_path.is_absolute() else data_dir / sample_path
    return data_dir / f"HoodImpactor_{sample}_SAE1000.csv"


def sample_id_from_path(path):
    name = path.stem
    if name.startswith("HoodImpactor_") and name.endswith("_SAE1000"):
        return name.removeprefix("HoodImpactor_").removesuffix("_SAE1000")
    return name


def load_acceleration(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Time", "A(in g)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    time_ms = df["Time"].to_numpy(dtype=float) * 1000.0
    acc_g = df["A(in g)"].to_numpy(dtype=float)
    return time_ms, acc_g


def load_histories(samples, data_dir):
    histories = []
    for sample in samples:
        csv_path = sample_to_path(sample, data_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {csv_path}")

        time_ms, acc_g = load_acceleration(csv_path)
        peak_idx = int(np.nanargmax(acc_g))
        histories.append(
            {
                "sample_id": sample_id_from_path(csv_path),
                "path": csv_path,
                "time_ms": time_ms,
                "acc_g": acc_g,
                "peak_idx": peak_idx,
            }
        )
    return histories


def style_axis(ax):
    ax.grid(True, color="#d9dee7", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(colors="#4b5563")


def draw_history(history, y_limit):
    time_ms = history["time_ms"]
    acc_g = history["acc_g"]
    peak_idx = history["peak_idx"]
    peak_time = time_ms[peak_idx]
    peak_acc = acc_g[peak_idx]

    line_color = "#1f6f8b"
    fill_color = "#b8dbe7"
    peak_color = "#d64b3a"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.plot(time_ms, acc_g, color=line_color, linewidth=2.2)
    ax.fill_between(time_ms, acc_g, 0, color=fill_color, alpha=0.40)
    ax.scatter([peak_time], [peak_acc], s=54, color=peak_color, zorder=5)
    ax.annotate(
        f"Peak {peak_acc:.0f} g",
        xy=(peak_time, peak_acc),
        xytext=(peak_time + 1.5, min(peak_acc * 1.08, y_limit * 0.92)),
        fontsize=11,
        color="#1f2933",
        arrowprops=dict(arrowstyle="->", lw=1.2, color=peak_color),
    )

    ax.set_title(
        f"Acceleration History - Sample {history['sample_id']}",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    ax.set_xlabel("Time after impact (ms)", fontsize=11, color="#374151")
    ax.set_ylabel("Acceleration (g)", fontsize=11, color="#374151")
    ax.set_xlim(np.nanmin(time_ms), np.nanmax(time_ms))
    ax.set_ylim(0, y_limit)
    style_axis(ax)
    fig.tight_layout()
    return fig


def save_histories(histories, output_dir, prefix, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    y_limit = max(max(np.nanmax(history["acc_g"]) for history in histories) * 1.15, 20)

    output_paths = []
    for history in histories:
        fig = draw_history(history, y_limit)
        output_path = output_dir / f"{prefix}_{history['sample_id']}.png"
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    histories = load_histories(args.samples, data_dir)
    output_paths = save_histories(histories, output_dir, args.prefix, args.dpi)

    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
