"""Focused acceleration-history plots for HoodImpact_1704_EuroNCAP.

Plots only the resultant acceleration column, ``A(in g)``, used as the
time-history target by the current TemporalDeepONet data loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


N_DESIGNS = 12
N_LOCATIONS = 142
N_RUNS = N_DESIGNS * N_LOCATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot acceleration histories by hood design.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/HoodImpact_1704_EuroNCAP"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/hoodimpact_1704_acceleration_histories"),
    )
    return parser.parse_args()


def load_histories(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    history_dir = data_dir / "output_history_acc"
    time = None
    histories = None

    for run in range(1, N_RUNS + 1):
        path = history_dir / f"HoodImpact_{run}_SAE1000_interp1000.csv"
        values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 5))
        run_time = values[:, 0]
        acceleration_g = values[:, 1]

        if time is None:
            time = run_time
            histories = np.empty((N_RUNS, len(time)), dtype=np.float64)
        elif not np.array_equal(run_time, time):
            raise ValueError(f"run {run} does not use the common time grid")

        histories[run - 1] = acceleration_g

    assert time is not None and histories is not None
    return time, histories.reshape(N_DESIGNS, N_LOCATIONS, -1)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_all_histories_by_design(
    time_ms: np.ndarray, histories: np.ndarray, output_dir: Path
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(19, 13), sharex=True, sharey=True)
    palette = sns.color_palette("colorblind", N_DESIGNS)

    for design, ax in enumerate(axes.flat):
        design_histories = histories[design]
        color = palette[design]
        ax.plot(
            time_ms,
            design_histories.T,
            color=color,
            alpha=0.075,
            linewidth=0.55,
        )
        q25, median, q75 = np.percentile(design_histories, [25, 50, 75], axis=0)
        ax.fill_between(time_ms, q25, q75, color=color, alpha=0.28, label="P25-P75")
        ax.plot(time_ms, median, color="black", linewidth=1.8, label="median")
        ax.set_title(
            f"Design {design + 1:02d} · 142 histories\n"
            f"peak range {design_histories.max(axis=1).min():.0f}–"
            f"{design_histories.max(axis=1).max():.0f} g"
        )
        ax.grid(alpha=0.22)
        if design // 4 == 2:
            ax.set_xlabel("Time (ms)")
        if design % 4 == 0:
            ax.set_ylabel("Resultant acceleration (g)")

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Acceleration histories within each hood design\n"
        "thin lines: individual impact locations; band/black line: IQR and median",
        fontsize=19,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save(fig, output_dir / "01_acceleration_histories_by_design.png")


def plot_design_median_comparison(
    time_ms: np.ndarray, histories: np.ndarray, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8))
    palette = sns.color_palette("turbo", N_DESIGNS)
    medians = np.median(histories, axis=1)

    for design in range(N_DESIGNS):
        ax.plot(
            time_ms,
            medians[design],
            color=palette[design],
            linewidth=1.8,
            label=f"Design {design + 1:02d}",
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Median resultant acceleration (g)")
    ax.set_title("Direct comparison of the 12 design-level median histories", fontweight="bold")
    ax.legend(ncol=3, frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, output_dir / "02_design_median_history_comparison.png")


def select_central_locations(data_dir: Path) -> pd.DataFrame:
    locations = pd.read_csv(data_dir / "impact_locations_142.csv")
    selected = []
    for wadh in sorted(locations["wadh_mm"].unique()):
        band = locations[locations["wadh_mm"] == wadh]
        central_index = band["col_k"].abs().idxmin()
        selected.append(locations.loc[central_index])
    # Six bands cover the hood without making the comparison unreadable.
    indices = np.linspace(0, len(selected) - 1, 6).round().astype(int)
    return pd.DataFrame(selected).iloc[indices].reset_index(drop=True)


def plot_same_locations_across_designs(
    time_ms: np.ndarray,
    histories: np.ndarray,
    data_dir: Path,
    output_dir: Path,
) -> None:
    selected = select_central_locations(data_dir)
    palette = sns.color_palette("turbo", N_DESIGNS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey=True)

    for ax, location in zip(axes.flat, selected.itertuples()):
        loc_index = int(location.loc) - 1
        for design in range(N_DESIGNS):
            ax.plot(
                time_ms,
                histories[design, loc_index],
                color=palette[design],
                linewidth=1.15,
                alpha=0.90,
                label=f"D{design + 1:02d}",
            )
        ax.set_title(
            f"Location {int(location.loc)} · WADH {int(location.wadh_mm)} mm\n"
            f"X1={location.X1:.0f}, X2={location.X2:.0f} mm"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Resultant acceleration (g)")
        ax.grid(alpha=0.22)

    axes[0, 0].legend(ncol=3, frameon=False, fontsize=7)
    fig.suptitle(
        "Same impact location compared across all 12 designs",
        fontsize=19,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save(fig, output_dir / "03_same_locations_across_designs.png")


def plot_all_locations_across_designs(
    time_ms: np.ndarray,
    histories: np.ndarray,
    data_dir: Path,
    output_dir: Path,
    locations_per_page: int = 12,
) -> None:
    """Create paginated same-location comparisons for all 142 locations."""
    locations = pd.read_csv(data_dir / "impact_locations_142.csv")
    # Spatial ordering keeps neighboring WADH rows and lateral columns together.
    locations = locations.sort_values(["wadh_mm", "col_k", "loc"]).reset_index(drop=True)
    page_dir = output_dir / "all_locations_across_designs"
    page_dir.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("turbo", N_DESIGNS)
    global_max = float(histories.max())
    y_max = 25.0 * np.ceil(1.03 * global_max / 25.0)
    index_rows = []
    n_pages = int(np.ceil(len(locations) / locations_per_page))

    for page_index in range(n_pages):
        start = page_index * locations_per_page
        stop = min(start + locations_per_page, len(locations))
        page_locations = locations.iloc[start:stop]
        fig, axes = plt.subplots(
            3,
            4,
            figsize=(19, 13),
            sharex=True,
            sharey=True,
        )

        for panel_index, (ax, location) in enumerate(
            zip(axes.flat, page_locations.itertuples()), start=1
        ):
            loc_index = int(location.loc) - 1
            for design in range(N_DESIGNS):
                ax.plot(
                    time_ms,
                    histories[design, loc_index],
                    color=palette[design],
                    linewidth=1.0,
                    alpha=0.90,
                    label=f"D{design + 1:02d}",
                )
            ax.set_title(
                f"Loc {int(location.loc)} · WADH {int(location.wadh_mm)} · "
                f"k={int(location.col_k):+d}\n"
                f"X1={location.X1:.0f}, X2={location.X2:.0f} mm",
                fontsize=10,
            )
            ax.set_xlim(time_ms[0], time_ms[-1])
            ax.set_ylim(-5, y_max)
            ax.grid(alpha=0.20)
            if (panel_index - 1) // 4 == 2:
                ax.set_xlabel("Time (ms)")
            if (panel_index - 1) % 4 == 0:
                ax.set_ylabel("Resultant acceleration (g)")
            index_rows.append(
                {
                    "page": page_index + 1,
                    "panel": panel_index,
                    "loc": int(location.loc),
                    "wadh_mm": float(location.wadh_mm),
                    "col_k": int(location.col_k),
                    "X1": float(location.X1),
                    "X2": float(location.X2),
                }
            )

        for ax in axes.flat[len(page_locations) :]:
            ax.set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=palette[d], linewidth=2, label=f"D{d + 1:02d}")
            for d in range(N_DESIGNS)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=12,
            frameon=False,
            fontsize=8,
        )
        wadh_min = int(page_locations["wadh_mm"].min())
        wadh_max = int(page_locations["wadh_mm"].max())
        wadh_text = str(wadh_min) if wadh_min == wadh_max else f"{wadh_min}–{wadh_max}"
        fig.suptitle(
            f"Same impact location across 12 designs — page {page_index + 1}/{n_pages}\n"
            f"spatial order · WADH {wadh_text} mm · common y-scale 0–{y_max:.0f} g",
            fontsize=18,
            fontweight="bold",
            y=1.005,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.935))
        page_path = page_dir / f"page_{page_index + 1:02d}_of_{n_pages:02d}.png"
        save(fig, page_path)

    pd.DataFrame(index_rows).to_csv(page_dir / "location_page_index.csv", index=False)
    print(f"wrote {page_dir / 'location_page_index.csv'}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    time, histories = load_histories(data_dir)
    time_ms = time * 1e3
    plot_all_histories_by_design(time_ms, histories, output_dir)
    plot_design_median_comparison(time_ms, histories, output_dir)
    plot_same_locations_across_designs(time_ms, histories, data_dir, output_dir)
    plot_all_locations_across_designs(time_ms, histories, data_dir, output_dir)
    print(f"acceleration-history figures complete: {output_dir}")


if __name__ == "__main__":
    main()
