"""Exploratory visualizations for the 1,704-run EuroNCAP hood dataset.

The script treats the data as the balanced factorial design it is:
12 hood geometries x 142 shared impact locations.  It validates the metadata,
loads every 1,000-point acceleration history, derives waveform features, and
creates model-oriented plots plus machine-readable summaries.

Usage
-----
python visualize_hoodimpact_1704_eda.py
python visualize_hoodimpact_1704_eda.py --skip-mesh
python visualize_hoodimpact_1704_eda.py --output-dir figures/my_eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import numpy as np
import pandas as pd
import seaborn as sns


# The exported files use the simulation convention 1 g = 9,800 mm/s^2.
G_MM_S2 = 9800.0
N_DESIGNS = 12
N_LOCATIONS = 142
N_RUNS = N_DESIGNS * N_LOCATIONS

COLORS = {
    "ink": "#172033",
    "muted": "#667085",
    "blue": "#1676B8",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#7A5195",
    "light": "#E8EEF5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HoodImpact_1704_EuroNCAP."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/HoodImpact_1704_EuroNCAP"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/hoodimpact_1704_eda"),
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip the 12 representative .inp point-cloud panels.",
    )
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int = 190) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = pd.read_csv(data_dir / "manifest_1704.csv")
    metrics = pd.read_csv(data_dir / "generation_metrics.csv")
    locations = pd.read_csv(data_dir / "impact_locations_142.csv")
    coords = pd.read_csv(data_dir / "ImpactCoords_1704.csv")

    require_columns(
        manifest,
        {
            "run",
            "design",
            "loc",
            "X1",
            "X2",
            "X3",
            "dz_mm",
            "center_gap_mm",
            "sphere_pen_mm",
            "lift_mm",
            "surf_z_mm",
            "source_subset",
        },
        "manifest_1704.csv",
    )
    require_columns(
        metrics,
        {"run", "status", "solve_s", "n_raw_points", "peak_g", "hic15"},
        "generation_metrics.csv",
    )
    require_columns(locations, {"loc", "X1", "X2", "wadh_mm", "col_k"}, "impact_locations_142.csv")
    require_columns(coords, {"X1", "X2", "X3"}, "ImpactCoords_1704.csv")

    expected_runs = np.arange(1, N_RUNS + 1)
    if not np.array_equal(np.sort(manifest["run"].to_numpy()), expected_runs):
        raise ValueError("manifest run IDs are not exactly 1..1704")
    if manifest["run"].duplicated().any() or metrics["run"].duplicated().any():
        raise ValueError("duplicate run IDs found in metadata")
    if set(metrics["status"].astype(str).str.lower()) != {"ok"}:
        bad = metrics.loc[metrics["status"].astype(str).str.lower() != "ok", "run"]
        raise ValueError(f"non-ok generation rows found: {bad.tolist()[:10]}")

    formula_run = N_LOCATIONS * manifest["design"].to_numpy() + manifest["loc"].to_numpy()
    if not np.array_equal(formula_run, manifest["run"].to_numpy()):
        raise ValueError("run != 142 * design + loc for one or more rows")
    if not (manifest.groupby("design").size() == N_LOCATIONS).all():
        raise ValueError("each design must contain 142 locations")
    if not (manifest.groupby("loc").size() == N_DESIGNS).all():
        raise ValueError("each location must appear in all 12 designs")

    manifest_sorted = manifest.sort_values("run").reset_index(drop=True)
    coords_delta = np.abs(
        manifest_sorted[["X1", "X2", "X3"]].to_numpy()
        - coords[["X1", "X2", "X3"]].to_numpy()
    )

    keep_metrics = metrics[
        ["run", "solve_s", "n_raw_points", "peak_g", "hic15", "finished_at"]
    ].rename(
        columns={
            "peak_g": "reported_peak_g",
            "hic15": "reported_hic15",
        }
    )
    frame = manifest.merge(keep_metrics, on="run", how="inner", validate="one_to_one")
    frame = frame.merge(
        locations[["loc", "wadh_mm", "col_k"]],
        on="loc",
        how="left",
        validate="many_to_one",
    )
    frame = frame.sort_values("run").reset_index(drop=True)
    frame["design_label"] = frame["design"].map(lambda value: f"D{value + 1:02d}")
    frame["source_group"] = np.where(
        frame["loc"] <= 50, "retained 50", "new 92"
    )

    checks = {
        "runs": int(len(frame)),
        "all_status_ok": True,
        "run_formula_mismatches": 0,
        "max_impact_coordinate_mismatch_mm": float(coords_delta.max()),
        "source_counts": {
            str(k): int(v) for k, v in frame["source_group"].value_counts().items()
        },
    }
    return frame, locations.sort_values("loc").reset_index(drop=True), checks


def load_histories(
    data_dir: Path, frame: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    history_dir = data_dir / "output_history_acc"
    reference_time: np.ndarray | None = None
    waveforms: np.ndarray | None = None
    records: list[dict] = []
    nonfinite_values = 0
    nonmonotonic_histories = 0
    resampled_histories = 0
    max_magnitude_error_g = 0.0

    for position, run in enumerate(frame["run"].astype(int)):
        path = history_dir / f"HoodImpact_{run}_SAE1000_interp1000.csv"
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"unexpected history shape for run {run}: {values.shape}")

        time = values[:, 0]
        components = values[:, 1:4]
        magnitude_mm_s2 = values[:, 4]
        acceleration_g = values[:, 5]
        finite_mask = np.isfinite(values)
        nonfinite_values += int(values.size - finite_mask.sum())
        if not finite_mask.all():
            raise ValueError(f"non-finite history value in run {run}")
        if np.any(np.diff(time) <= 0):
            nonmonotonic_histories += 1

        calculated_magnitude_g = np.linalg.norm(components, axis=1) / G_MM_S2
        max_magnitude_error_g = max(
            max_magnitude_error_g,
            float(np.max(np.abs(calculated_magnitude_g - acceleration_g))),
        )

        if reference_time is None:
            reference_time = time.copy()
            waveforms = np.empty((len(frame), len(reference_time)), dtype=np.float64)
            plotted_g = acceleration_g
        elif len(time) == len(reference_time) and np.allclose(
            time, reference_time, rtol=0.0, atol=1e-10
        ):
            plotted_g = acceleration_g
        else:
            plotted_g = np.interp(reference_time, time, acceleration_g)
            resampled_histories += 1

        assert waveforms is not None
        waveforms[position] = plotted_g
        peak_index = int(np.argmax(acceleration_g))
        peak_g = float(acceleration_g[peak_index])
        above_half = np.flatnonzero(acceleration_g >= 0.5 * peak_g)
        half_max_span_ms = float(
            (time[above_half[-1]] - time[above_half[0]]) * 1e3
        )
        tail_mask = time >= (time[-1] - 0.002)
        records.append(
            {
                "run": int(run),
                "history_points": int(len(time)),
                "duration_ms": float((time[-1] - time[0]) * 1e3),
                "median_dt_us": float(np.median(np.diff(time)) * 1e6),
                "calculated_peak_g": peak_g,
                "time_to_peak_ms": float(time[peak_index] * 1e3),
                "half_max_span_ms": half_max_span_ms,
                "rms_g": float(np.sqrt(np.mean(acceleration_g**2))),
                "impulse_g_ms": float(np.trapezoid(acceleration_g, time) * 1e3),
                "end_g": float(acceleration_g[-1]),
                "end_to_peak_pct": float(100.0 * acceleration_g[-1] / peak_g),
                "tail_2ms_rms_g": float(np.sqrt(np.mean(acceleration_g[tail_mask] ** 2))),
                "peak_abs_A1_g": float(np.max(np.abs(components[:, 0])) / G_MM_S2),
                "peak_abs_A2_g": float(np.max(np.abs(components[:, 1])) / G_MM_S2),
                "peak_abs_A3_g": float(np.max(np.abs(components[:, 2])) / G_MM_S2),
            }
        )
        if (position + 1) % 250 == 0 or position + 1 == len(frame):
            print(f"loaded histories: {position + 1}/{len(frame)}")

    assert reference_time is not None and waveforms is not None
    history_metrics = pd.DataFrame.from_records(records)
    checks = {
        "history_files_loaded": int(len(history_metrics)),
        "history_points_min": int(history_metrics["history_points"].min()),
        "history_points_max": int(history_metrics["history_points"].max()),
        "nonfinite_history_values": int(nonfinite_values),
        "nonmonotonic_histories": int(nonmonotonic_histories),
        "histories_resampled_for_plotting": int(resampled_histories),
        "max_vector_magnitude_error_g": float(max_magnitude_error_g),
    }
    return history_metrics, reference_time, waveforms, checks


def exhaustive_hic15_common_grid(
    time: np.ndarray,
    waveforms: np.ndarray,
    max_window_s: float = 0.015,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate every admissible sampled time window for every waveform.

    The dataset has one common time grid, so window start/end indices and
    durations can be reused. Processing a few runs at a time bounds memory.
    """
    if time.ndim != 1 or waveforms.ndim != 2 or waveforms.shape[1] != len(time):
        raise ValueError("time and waveforms have incompatible shapes")
    starts, ends = np.triu_indices(len(time), k=1)
    durations = time[ends] - time[starts]
    valid = durations <= max_window_s
    starts = starts[valid]
    ends = ends[valid]
    durations = durations[valid]

    integration_steps = 0.5 * (waveforms[:, 1:] + waveforms[:, :-1]) * np.diff(time)
    cumulative = np.zeros_like(waveforms, dtype=np.float64)
    cumulative[:, 1:] = np.cumsum(integration_steps, axis=1)
    best_hic = np.empty(len(waveforms), dtype=np.float64)
    best_start = np.empty(len(waveforms), dtype=np.int32)
    best_end = np.empty(len(waveforms), dtype=np.int32)

    for lower in range(0, len(waveforms), chunk_size):
        upper = min(lower + chunk_size, len(waveforms))
        averages = (
            cumulative[lower:upper, ends] - cumulative[lower:upper, starts]
        ) / durations
        np.maximum(averages, 0.0, out=averages)
        scores = np.power(averages, 2.5)
        scores *= durations
        maximizing_window = np.argmax(scores, axis=1)
        rows = np.arange(upper - lower)
        best_hic[lower:upper] = scores[rows, maximizing_window]
        best_start[lower:upper] = starts[maximizing_window]
        best_end[lower:upper] = ends[maximizing_window]
    return best_hic, best_start, best_end


def add_history_metrics(
    frame: pd.DataFrame,
    histories: pd.DataFrame,
    time: np.ndarray,
    waveforms: np.ndarray,
) -> pd.DataFrame:
    print("recomputing exhaustive HIC15 over every admissible window ...")
    canonical_hic, window_start, window_end = exhaustive_hic15_common_grid(time, waveforms)
    histories = histories.copy()
    histories["hic15"] = canonical_hic
    histories["hic_window_start_ms"] = time[window_start] * 1e3
    histories["hic_window_end_ms"] = time[window_end] * 1e3
    histories["hic_window_duration_ms"] = (
        histories["hic_window_end_ms"] - histories["hic_window_start_ms"]
    )
    result = frame.merge(histories, on="run", how="inner", validate="one_to_one")
    result["peak_g_difference"] = (
        result["calculated_peak_g"] - result["reported_peak_g"]
    )
    result["log10_hic15"] = np.log10(result["hic15"])
    result["hic15_stored_shortfall"] = result["hic15"] - result["reported_hic15"]
    result["hic15_stored_shortfall_pct"] = (
        100.0 * result["hic15_stored_shortfall"] / result["hic15"]
    )
    return result


def matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    return (
        frame.pivot(index="design", columns="loc", values=value)
        .reindex(index=range(N_DESIGNS), columns=range(1, N_LOCATIONS + 1))
        .to_numpy()
    )


def balanced_variance_components(values: np.ndarray) -> dict[str, float]:
    """Two-way balanced sums of squares (one observation per design/location)."""
    grand = float(np.mean(values))
    design_mean = np.mean(values, axis=1, keepdims=True)
    location_mean = np.mean(values, axis=0, keepdims=True)
    design_ss = values.shape[1] * float(np.sum((design_mean - grand) ** 2))
    location_ss = values.shape[0] * float(np.sum((location_mean - grand) ** 2))
    interaction = values - design_mean - location_mean + grand
    interaction_ss = float(np.sum(interaction**2))
    total_ss = float(np.sum((values - grand) ** 2))
    return {
        "design": design_ss / total_ss,
        "location": location_ss / total_ss,
        "interaction": interaction_ss / total_ss,
    }


def annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        va="bottom",
    )


def plot_hic15_validation(
    frame: pd.DataFrame,
    time: np.ndarray,
    waveforms: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12))

    ax = axes[0, 0]
    scatter = ax.scatter(
        frame["reported_hic15"],
        frame["hic15"],
        c=frame["hic15_stored_shortfall_pct"],
        cmap="rocket_r",
        s=18,
        alpha=0.72,
        linewidth=0,
    )
    limits = [
        float(min(frame["reported_hic15"].min(), frame["hic15"].min())),
        float(max(frame["reported_hic15"].max(), frame["hic15"].max())),
    ]
    ax.plot(limits, limits, color=COLORS["ink"], linestyle="--", linewidth=1.5, label="1:1")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stored generation_metrics hic15")
    ax.set_ylabel("Exhaustive HIC15 recomputed from history")
    ax.set_title("Stored shortcut vs exhaustive definition")
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, pad=0.015, label="Stored shortfall (% of exhaustive)")
    annotate_panel(ax, "A")

    ax = axes[0, 1]
    sns.histplot(
        frame,
        x="hic15_stored_shortfall_pct",
        bins=45,
        color=COLORS["orange"],
        ax=ax,
    )
    for threshold, linestyle in [(1, "--"), (5, "-."), (10, ":")]:
        count = int((frame["hic15_stored_shortfall_pct"] > threshold).sum())
        ax.axvline(threshold, color=COLORS["ink"], linestyle=linestyle, linewidth=1)
        ax.text(
            threshold + 0.5,
            ax.get_ylim()[1] * (0.92 - 0.10 * [1, 5, 10].index(threshold)),
            f">{threshold}%: {count} runs",
            fontsize=8,
        )
    ax.set_xlabel("Stored HIC15 shortfall relative to exhaustive (%)")
    ax.set_ylabel("Runs")
    ax.set_title("Magnitude of the stored-target bias")
    annotate_panel(ax, "B")

    ax = axes[1, 0]
    error_matrix = matrix(frame, "hic15_stored_shortfall_pct")
    image = ax.imshow(
        error_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="rocket_r",
        vmin=0,
        vmax=float(np.percentile(error_matrix, 99)),
    )
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.1)
    ax.set_xticks([0, 24, 49, 74, 99, 124, 141], [1, 25, 50, 75, 100, 125, 142])
    ax.set_yticks(range(N_DESIGNS), [f"D{i + 1:02d}" for i in range(N_DESIGNS)])
    ax.set_xlabel("Location (canonical FPS order)")
    ax.set_ylabel("Design")
    ax.set_title("Where the shortcut misses (color clipped at P99)")
    fig.colorbar(image, ax=ax, pad=0.015, label="Stored shortfall (%)")
    annotate_panel(ax, "C")

    ax = axes[1, 1]
    worst = frame.loc[frame["hic15_stored_shortfall_pct"].idxmax()]
    run = int(worst["run"])
    run_index = run - 1
    time_ms = time * 1e3
    ax.plot(time_ms, waveforms[run_index], color=COLORS["ink"], linewidth=1.8)
    ax.axvspan(
        worst["hic_window_start_ms"],
        worst["hic_window_end_ms"],
        color=COLORS["orange"],
        alpha=0.24,
        label=f"max HIC window ({worst['hic_window_duration_ms']:.2f} ms)",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Resultant acceleration (g)")
    ax.set_title(
        f"Largest discrepancy: run {run} · stored {worst['reported_hic15']:.0f} "
        f"vs exhaustive {worst['hic15']:.0f}"
    )
    ax.legend(frameon=False, fontsize=9)
    annotate_panel(ax, "D")

    fig.suptitle(
        "HIC15 target validation — exhaustive search changes the supplied scalar target",
        fontsize=18,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "00_hic15_validation.png")


def plot_overview(
    frame: pd.DataFrame,
    locations: pd.DataFrame,
    time: np.ndarray,
    waveforms: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))

    ax = axes[0, 0]
    retained = locations["loc"] <= 50
    ax.scatter(
        locations.loc[retained, "X2"],
        locations.loc[retained, "X1"],
        s=38,
        color=COLORS["blue"],
        label="retained 50",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.scatter(
        locations.loc[~retained, "X2"],
        locations.loc[~retained, "X1"],
        s=38,
        color=COLORS["orange"],
        label="new 92",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X2 lateral position (mm)")
    ax.set_ylabel("X1 longitudinal position (mm)")
    ax.set_title("142 shared impact locations")
    ax.legend(frameon=False, fontsize=9)
    annotate_panel(ax, "A")

    ax = axes[0, 1]
    sns.histplot(frame, x="hic15", bins=35, color=COLORS["purple"], ax=ax)
    median_hic = float(frame["hic15"].median())
    ax.axvline(median_hic, color=COLORS["ink"], linestyle="--", linewidth=1.6)
    ax.text(
        median_hic,
        ax.get_ylim()[1] * 0.92,
        f" median {median_hic:,.0f}",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("HIC15")
    ax.set_ylabel("Runs")
    ax.set_title("HIC15 distribution")
    annotate_panel(ax, "B")

    ax = axes[0, 2]
    sns.boxplot(
        data=frame,
        x="design_label",
        y="hic15",
        hue="source_group",
        palette=[COLORS["blue"], COLORS["orange"]],
        fliersize=1.5,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("Hood design")
    ax.set_ylabel("HIC15")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Location subset", frameon=False, fontsize=8, title_fontsize=8)
    ax.set_title("Response by design and source subset")
    annotate_panel(ax, "C")

    ax = axes[1, 0]
    hic_matrix = matrix(frame, "hic15")
    image = ax.imshow(
        hic_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=float(hic_matrix.min()), vmax=float(hic_matrix.max())),
    )
    ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.2)
    ax.set_xticks([0, 24, 49, 74, 99, 124, 141], [1, 25, 50, 75, 100, 125, 142])
    ax.set_yticks(range(N_DESIGNS), [f"D{i + 1:02d}" for i in range(N_DESIGNS)])
    ax.set_xlabel("Impact location (canonical FPS order)")
    ax.set_ylabel("Design")
    ax.set_title("HIC15: every design-location pair (log color)")
    fig.colorbar(image, ax=ax, pad=0.02, label="HIC15")
    annotate_panel(ax, "D")

    ax = axes[1, 1]
    scatter = ax.scatter(
        frame["calculated_peak_g"],
        frame["hic15"],
        c=frame["half_max_span_ms"],
        cmap="viridis",
        s=16,
        alpha=0.72,
        linewidth=0,
    )
    pearson = frame[["calculated_peak_g", "hic15"]].corr().iloc[0, 1]
    ax.text(
        0.04,
        0.96,
        f"Pearson r = {pearson:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set_xlabel("Peak resultant acceleration (g)")
    ax.set_ylabel("HIC15")
    ax.set_title("Peak acceleration is not the whole HIC story")
    fig.colorbar(scatter, ax=ax, pad=0.02, label="Span above half-peak (ms)")
    annotate_panel(ax, "E")

    ax = axes[1, 2]
    q05, q25, q50, q75, q95 = np.percentile(waveforms, [5, 25, 50, 75, 95], axis=0)
    time_ms = time * 1e3
    ax.fill_between(time_ms, q05, q95, color=COLORS["blue"], alpha=0.14, label="P05-P95")
    ax.fill_between(time_ms, q25, q75, color=COLORS["blue"], alpha=0.28, label="P25-P75")
    ax.plot(time_ms, q50, color=COLORS["ink"], linewidth=2.0, label="median")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Resultant acceleration (g)")
    ax.set_title("All 1,704 acceleration histories")
    ax.legend(frameon=False, fontsize=9)
    annotate_panel(ax, "F")

    fig.suptitle(
        "HoodImpact_1704_EuroNCAP — dataset overview",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "01_dataset_overview.png")


def plot_spatial_hic_maps(frame: pd.DataFrame, output_dir: Path) -> None:
    values = frame["hic15"].to_numpy()
    norm = LogNorm(vmin=float(values.min()), vmax=float(values.max()))
    fig, axes = plt.subplots(3, 4, figsize=(17, 14), sharex=True, sharey=True)
    scatter = None
    for design, ax in enumerate(axes.flat):
        group = frame[frame["design"] == design]
        scatter = ax.scatter(
            group["X2"],
            group["X1"],
            c=group["hic15"],
            cmap="magma",
            norm=norm,
            s=37,
            edgecolor="white",
            linewidth=0.25,
        )
        worst = group.loc[group["hic15"].idxmax()]
        ax.scatter(
            [worst["X2"]],
            [worst["X1"]],
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.4,
        )
        ax.set_aspect("equal")
        ax.set_title(
            f"D{design + 1:02d} · median {group['hic15'].median():.0f}\n"
            f"max L{int(worst['loc'])}: {worst['hic15']:.0f}",
            fontsize=10,
        )
        ax.grid(alpha=0.15)
        if design // 4 == 2:
            ax.set_xlabel("X2 (mm)")
        if design % 4 == 0:
            ax.set_ylabel("X1 (mm)")
    assert scatter is not None
    fig.suptitle(
        "Spatial HIC15 maps — cyan ring marks each design's worst location",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.95, right=0.89, hspace=0.20, wspace=0.10)
    colorbar_ax = fig.add_axes([0.915, 0.17, 0.016, 0.66])
    cbar = fig.colorbar(scatter, cax=colorbar_ax)
    cbar.set_label("HIC15 (logarithmic color scale)")
    save_figure(fig, output_dir / "02_spatial_hic_maps.png")


def plot_spatial_response_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    spatial = frame.groupby("loc", as_index=False).agg(
        X1=("X1", "first"),
        X2=("X2", "first"),
        wadh_mm=("wadh_mm", "first"),
        hic_mean=("hic15", "mean"),
        hic_std=("hic15", "std"),
        peak_mean=("calculated_peak_g", "mean"),
        peak_std=("calculated_peak_g", "std"),
    )
    panels = [
        ("hic_mean", "Mean HIC15 across designs", "magma"),
        ("hic_std", "HIC15 standard deviation", "rocket"),
        ("peak_mean", "Mean peak acceleration (g)", "viridis"),
        ("peak_std", "Peak acceleration standard deviation (g)", "crest"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    for panel, ((column, title, cmap), ax) in enumerate(zip(panels, axes.flat)):
        scatter = ax.scatter(
            spatial["X2"],
            spatial["X1"],
            c=spatial[column],
            cmap=cmap,
            s=74,
            edgecolor="white",
            linewidth=0.45,
        )
        extreme = spatial.nlargest(3, column)
        for row in extreme.itertuples():
            ax.annotate(
                f"L{row.loc}",
                (row.X2, row.X1),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=COLORS["ink"],
            )
        ax.set_aspect("equal")
        ax.set_xlabel("X2 lateral position (mm)")
        ax.set_ylabel("X1 longitudinal position (mm)")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, pad=0.015, shrink=0.82)
        annotate_panel(ax, chr(ord("A") + panel))
    fig.suptitle(
        "Across-design spatial response: mean behavior and interaction hot spots",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "02_spatial_response_summary.png")


def plot_design_location_response(frame: pd.DataFrame, output_dir: Path) -> dict:
    hic = matrix(frame, "hic15")
    peak = matrix(frame, "calculated_peak_g")
    peak_time = matrix(frame, "time_to_peak_ms")
    end_ratio = matrix(frame, "end_to_peak_pct")

    components = {
        "HIC15": balanced_variance_components(hic),
        "log10(HIC15)": balanced_variance_components(np.log10(hic)),
        "peak_g": balanced_variance_components(peak),
        "time_to_peak_ms": balanced_variance_components(peak_time),
    }

    fig = plt.figure(figsize=(19, 12))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.9], hspace=0.28, wspace=0.25)
    heat_specs = [
        (hic, "HIC15", "magma", LogNorm(vmin=float(hic.min()), vmax=float(hic.max()))),
        (peak, "Peak acceleration (g)", "viridis", None),
        (peak_time, "Time to peak (ms)", "crest", None),
        (end_ratio, "End / peak acceleration (%)", "rocket", PowerNorm(gamma=0.55)),
    ]
    for index, (values, title, cmap, norm) in enumerate(heat_specs):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.axvline(49.5, color="cyan", linestyle="--", linewidth=1.1)
        ax.set_xticks([0, 24, 49, 74, 99, 124, 141], [1, 25, 50, 75, 100, 125, 142])
        ax.set_yticks(range(N_DESIGNS), [f"D{i + 1:02d}" for i in range(N_DESIGNS)])
        ax.set_xlabel("Location")
        ax.set_ylabel("Design")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, pad=0.015, shrink=0.85)
        annotate_panel(ax, chr(ord("A") + index))

    ax = fig.add_subplot(grid[0, 2])
    component_df = (
        pd.DataFrame(components).T[["design", "location", "interaction"]]
        .mul(100.0)
    )
    component_df = component_df.rename(
        columns={"interaction": "non-additive residual"}
    )
    component_df.plot(
        kind="bar",
        stacked=True,
        color=[COLORS["blue"], COLORS["orange"], COLORS["purple"]],
        ax=ax,
        width=0.72,
    )
    ax.set_ylabel("Share of total variation (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(
        title="Variation component",
        frameon=True,
        framealpha=0.82,
        fontsize=6.5,
        title_fontsize=7,
        loc="lower left",
    )
    ax.set_ylim(0, 100)
    ax.set_title("Balanced design/location decomposition")
    annotate_panel(ax, "E")

    ax = fig.add_subplot(grid[1, 2])
    loc_summary = frame.groupby("loc")["hic15"].agg(["min", "median", "max"])
    x = loc_summary.index.to_numpy()
    ax.fill_between(
        x,
        loc_summary["min"].to_numpy(),
        loc_summary["max"].to_numpy(),
        color=COLORS["orange"],
        alpha=0.18,
        label="min-max across designs",
    )
    ax.plot(x, loc_summary["median"], color=COLORS["ink"], linewidth=1.6, label="median")
    ax.axvline(50.5, color=COLORS["blue"], linestyle="--", linewidth=1.2, label="source boundary")
    ax.set_xlabel("Location (canonical FPS order)")
    ax.set_ylabel("HIC15")
    ax.set_title("Location effect across the 12 designs")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "F")

    fig.suptitle(
        "Design × location response structure",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    save_figure(fig, output_dir / "03_design_location_response.png")
    return components


def nearest_quantile_runs(frame: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    chosen = []
    for quantile in quantiles:
        target = frame["hic15"].quantile(quantile)
        index = (frame["hic15"] - target).abs().idxmin()
        row = frame.loc[index].copy()
        row["requested_quantile"] = quantile
        chosen.append(row)
    return pd.DataFrame(chosen).drop_duplicates("run")


def plot_acceleration_dynamics(
    frame: pd.DataFrame,
    time: np.ndarray,
    waveforms: np.ndarray,
    output_dir: Path,
) -> None:
    time_ms = time * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    percentiles = np.percentile(waveforms, [1, 5, 25, 50, 75, 95, 99], axis=0)
    ax.fill_between(time_ms, percentiles[0], percentiles[6], color=COLORS["blue"], alpha=0.10, label="P01-P99")
    ax.fill_between(time_ms, percentiles[1], percentiles[5], color=COLORS["blue"], alpha=0.18, label="P05-P95")
    ax.fill_between(time_ms, percentiles[2], percentiles[4], color=COLORS["blue"], alpha=0.30, label="P25-P75")
    ax.plot(time_ms, percentiles[3], color=COLORS["ink"], linewidth=2, label="median")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Resultant acceleration (g)")
    ax.set_title("Population waveform envelope")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "A")

    ax = axes[0, 1]
    cmap = plt.get_cmap("turbo", N_DESIGNS)
    for design in range(N_DESIGNS):
        design_rows = frame.index[frame["design"] == design].to_numpy()
        ax.plot(
            time_ms,
            waveforms[design_rows].mean(axis=0),
            color=cmap(design),
            linewidth=1.35,
            label=f"D{design + 1:02d}",
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean acceleration (g)")
    ax.set_title("Mean waveform for each hood design")
    ax.legend(ncol=3, frameon=False, fontsize=7)
    annotate_panel(ax, "B")

    ax = axes[1, 0]
    vmax = float(np.percentile(waveforms, 99.5))
    image = ax.imshow(
        waveforms,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], 1, len(frame)],
        cmap="magma",
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
        interpolation="nearest",
    )
    for boundary in range(N_LOCATIONS, N_RUNS, N_LOCATIONS):
        ax.axhline(boundary + 0.5, color="white", alpha=0.28, linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Run (design-major order)")
    ax.set_title("All waveforms (color clipped at P99.5)")
    fig.colorbar(image, ax=ax, pad=0.015, label="Acceleration (g)")
    annotate_panel(ax, "C")

    ax = axes[1, 1]
    selected = nearest_quantile_runs(frame, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    selected_colors = sns.color_palette("viridis", n_colors=len(selected))
    for color, (_, row) in zip(selected_colors, selected.iterrows()):
        idx = int(row["run"]) - 1
        q = int(round(float(row["requested_quantile"]) * 100))
        ax.plot(
            time_ms,
            waveforms[idx],
            color=color,
            linewidth=1.45,
            label=f"P{q:02d}: run {idx + 1}, HIC={row['hic15']:.0f}",
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Resultant acceleration (g)")
    ax.set_title("Representative histories selected by HIC quantile")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "D")

    fig.suptitle("Acceleration-history dynamics", fontsize=19, fontweight="bold", y=1.005)
    fig.tight_layout()
    save_figure(fig, output_dir / "04_acceleration_dynamics.png")


def temporal_subsampling_summary(
    time: np.ndarray, waveforms: np.ndarray, strides: tuple[int, ...] = (2, 4, 8, 16, 32)
) -> tuple[pd.DataFrame, dict]:
    full_peak = waveforms.max(axis=1).astype(np.float64)
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for stride in strides:
        sample_indices = np.arange(0, waveforms.shape[1], stride)
        sampled_peak = waveforms[:, sample_indices].max(axis=1).astype(np.float64)
        loss_pct = 100.0 * (1.0 - sampled_peak / full_peak)
        worst_index = int(np.argmax(loss_pct))
        for run_index, loss in enumerate(loss_pct):
            rows.append({"run": run_index + 1, "stride": stride, "peak_loss_pct": float(loss)})
        summary[str(stride)] = {
            "points": int(len(sample_indices)),
            "effective_dt_ms": float(np.median(np.diff(time[sample_indices])) * 1e3),
            "last_time_ms": float(time[sample_indices[-1]] * 1e3),
            "median_peak_retained_pct": float(100.0 - np.median(loss_pct)),
            "p05_peak_retained_pct": float(100.0 - np.percentile(loss_pct, 95)),
            "worst_peak_retained_pct": float(100.0 - loss_pct[worst_index]),
            "runs_losing_over_1_pct": int(np.sum(loss_pct > 1.0)),
            "runs_losing_over_5_pct": int(np.sum(loss_pct > 5.0)),
            "runs_losing_over_10_pct": int(np.sum(loss_pct > 10.0)),
            "worst_run": worst_index + 1,
        }
    return pd.DataFrame(rows), summary


def plot_temporal_subsampling(
    frame: pd.DataFrame,
    time: np.ndarray,
    waveforms: np.ndarray,
    output_dir: Path,
) -> dict:
    losses, summary = temporal_subsampling_summary(time, waveforms)
    strides = sorted(losses["stride"].unique())
    colors = dict(zip(strides, sns.color_palette("viridis", n_colors=len(strides))))
    time_ms = time * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    for stride in strides:
        values = np.sort(losses.loc[losses["stride"] == stride, "peak_loss_pct"].to_numpy())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cumulative, color=colors[stride], linewidth=1.8, label=f"stride {stride}")
    ax.axvline(5, color=COLORS["orange"], linestyle="--", linewidth=1.1)
    ax.axvline(10, color=COLORS["orange"], linestyle=":", linewidth=1.1)
    ax.set_xlabel("Peak missed by subsampling (%)")
    ax.set_ylabel("Fraction of runs")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.set_title("Empirical CDF of peak loss")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "A")

    ax = axes[0, 1]
    positions = np.arange(len(strides))
    median_loss = [100 - summary[str(s)]["median_peak_retained_pct"] for s in strides]
    p95_loss = [100 - summary[str(s)]["p05_peak_retained_pct"] for s in strides]
    worst_loss = [100 - summary[str(s)]["worst_peak_retained_pct"] for s in strides]
    ax.plot(positions, median_loss, marker="o", label="median loss", color=COLORS["blue"])
    ax.plot(positions, p95_loss, marker="o", label="P95 loss", color=COLORS["orange"])
    ax.plot(positions, worst_loss, marker="o", label="worst loss", color=COLORS["purple"])
    ax.set_xticks(positions, [str(s) for s in strides])
    ax.set_xlabel("Stride")
    ax.set_ylabel("Native peak missed (%)")
    ax.set_title("Peak fidelity vs temporal stride")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "B")

    default_stride = 16
    worst_run = int(summary[str(default_stride)]["worst_run"])
    worst_index = worst_run - 1
    peak_index = int(np.argmax(waveforms[worst_index]))
    zoom = (time_ms >= max(0.0, time_ms[peak_index] - 2.0)) & (
        time_ms <= min(time_ms[-1], time_ms[peak_index] + 2.0)
    )
    ax = axes[1, 0]
    ax.plot(time_ms[zoom], waveforms[worst_index, zoom], color=COLORS["ink"], linewidth=2.0, label="native 1,000 points")
    for stride, marker in [(8, "o"), (16, "s"), (32, "^")]:
        indices = np.arange(0, len(time), stride)
        local = indices[zoom[indices]]
        ax.scatter(
            time_ms[local],
            waveforms[worst_index, local],
            s=28,
            marker=marker,
            color=colors[stride],
            label=f"stride {stride}",
            zorder=4,
        )
    run_row = frame.loc[frame["run"] == worst_run].iloc[0]
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(
        f"Worst stride-16 peak miss: run {worst_run} "
        f"(D{int(run_row['design']) + 1:02d}, L{int(run_row['loc'])})"
    )
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "C")

    ax = axes[1, 1]
    threshold_labels = [">1%", ">5%", ">10%"]
    width = 0.16
    for position, stride in enumerate(strides):
        counts = [
            summary[str(stride)]["runs_losing_over_1_pct"],
            summary[str(stride)]["runs_losing_over_5_pct"],
            summary[str(stride)]["runs_losing_over_10_pct"],
        ]
        percentages = 100.0 * np.asarray(counts) / len(frame)
        ax.bar(
            np.arange(3) + (position - 2) * width,
            percentages,
            width=width,
            color=colors[stride],
            label=f"stride {stride}",
        )
    ax.set_xticks(np.arange(3), threshold_labels)
    ax.set_xlabel("Peak-loss threshold")
    ax.set_ylabel("Runs exceeding threshold (%)")
    ax.set_title("How often subsampling materially misses a peak")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    annotate_panel(ax, "D")

    default = summary["16"]
    fig.suptitle(
        "Temporal subsampling fidelity\n"
        f"Current stride 16: {default['points']} points, Δt={default['effective_dt_ms']:.4f} ms, "
        f"last sample={default['last_time_ms']:.4f} ms",
        fontsize=18,
        fontweight="bold",
        y=1.015,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "08_temporal_subsampling_diagnostic.png")
    losses.to_csv(output_dir / "temporal_subsampling_peak_loss.csv", index=False)
    return summary


def plot_feature_relationships(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))

    relationships = [
        ("calculated_peak_g", "Peak acceleration (g)"),
        ("impulse_g_ms", "Acceleration impulse (g·ms)"),
        ("rms_g", "RMS acceleration (g)"),
        ("time_to_peak_ms", "Time to peak (ms)"),
        ("half_max_span_ms", "Span above half-peak (ms)"),
    ]
    for panel, ((feature, label), ax) in enumerate(zip(relationships, axes.flat[:5])):
        scatter = ax.scatter(
            frame[feature],
            frame["hic15"],
            c=frame["design"],
            cmap="turbo",
            s=15,
            alpha=0.65,
            linewidth=0,
        )
        correlation = frame[[feature, "hic15"]].corr(method="spearman").iloc[0, 1]
        ax.text(
            0.04,
            0.96,
            f"Spearman ρ = {correlation:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax.set_xlabel(label)
        ax.set_ylabel("HIC15")
        ax.set_title(f"HIC15 vs {label.lower()}")
        annotate_panel(ax, chr(ord("A") + panel))
    ax = axes[1, 2]
    correlation_columns = [
        "hic15",
        "calculated_peak_g",
        "rms_g",
        "impulse_g_ms",
        "time_to_peak_ms",
        "half_max_span_ms",
        "end_to_peak_pct",
        "peak_abs_A1_g",
        "peak_abs_A2_g",
        "peak_abs_A3_g",
        "X1",
        "X2",
        "X3",
        "surf_z_mm",
    ]
    labels = [
        "HIC15",
        "peak",
        "RMS",
        "impulse",
        "t_peak",
        "half-max span",
        "end/peak",
        "peak |A1|",
        "peak |A2|",
        "peak |A3|",
        "X1",
        "X2",
        "X3",
        "surface Z",
    ]
    correlations = frame[correlation_columns].corr(method="spearman")
    sns.heatmap(
        correlations,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    ax.set_title("Feature correlation map")
    annotate_panel(ax, "F")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=plt.get_cmap("turbo")(design / (N_DESIGNS - 1)),
            markeredgecolor="none",
            label=f"D{design + 1:02d}",
        )
        for design in range(N_DESIGNS)
    ]
    fig.legend(
        handles=legend_handles,
        title="Point color = hood design",
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Response-feature relationships", fontsize=19, fontweight="bold", y=1.005)
    fig.subplots_adjust(top=0.94, bottom=0.10, hspace=0.30, wspace=0.30)
    save_figure(fig, output_dir / "05_feature_relationships.png")


def load_model_mesh_nodes(path: Path) -> np.ndarray:
    """Match DataPreprocessor._load_mesh_from_inp: first exact *NODE block."""
    coordinates: list[list[float]] = []
    recording = False
    with path.open("r") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            if not recording:
                if text.upper() == "*NODE":
                    recording = True
                continue
            if text.startswith("*"):
                break
            parts = text.split(",")
            if len(parts) >= 4:
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coordinates:
        raise ValueError(f"no nodes parsed from {path}")
    return np.asarray(coordinates, dtype=np.float32)


def plot_model_mesh_inputs(frame: pd.DataFrame, data_dir: Path, output_dir: Path) -> dict:
    meshes: list[np.ndarray] = []
    representative_runs: list[int] = []
    for design in range(N_DESIGNS):
        run = design * N_LOCATIONS + 1
        representative_runs.append(run)
        path = data_dir / "inp_files" / f"HoodImpact_{run}.inp"
        mesh = load_model_mesh_nodes(path)
        meshes.append(mesh)
        print(f"parsed model mesh: design {design + 1:02d}, run {run}, {len(mesh):,} nodes")

    all_z = np.concatenate([points[:, 2] for points in meshes])
    norm = PowerNorm(
        gamma=0.75,
        vmin=float(np.percentile(all_z, 1)),
        vmax=float(np.percentile(all_z, 99)),
    )
    fig, axes = plt.subplots(3, 4, figsize=(17, 14), sharex=True, sharey=True)
    scatter = None
    for design, (ax, points, run) in enumerate(zip(axes.flat, meshes, representative_runs)):
        # Drawing order by height keeps upper surfaces visible. Downsample only
        # for rendering; the title reports the full node count consumed by code.
        order = np.argsort(points[:, 2])
        plot_points = points[order][::3]
        scatter = ax.scatter(
            plot_points[:, 1],
            plot_points[:, 0],
            c=plot_points[:, 2],
            cmap="viridis",
            norm=norm,
            s=0.7,
            alpha=0.72,
            linewidth=0,
            rasterized=True,
        )
        impacts = frame[frame["design"] == design]
        ax.scatter(
            impacts["X2"],
            impacts["X1"],
            s=11,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.42,
            alpha=0.60,
        )
        first = impacts.loc[impacts["run"] == run].iloc[0]
        ax.scatter([first["X2"]], [first["X1"]], marker="x", color="red", s=32, linewidth=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"D{design + 1:02d} · run {run} · {len(points):,} nodes")
        ax.grid(alpha=0.12)
        if design // 4 == 2:
            ax.set_xlabel("X2 (mm)")
        if design % 4 == 0:
            ax.set_ylabel("X1 (mm)")
    assert scatter is not None
    fig.suptitle(
        "Representative point clouds exactly as the current loader reads them\n"
        "white rings: 142 impact sites; red ×: impactor location in the representative deck",
        fontsize=17,
        fontweight="bold",
        y=0.997,
    )
    fig.subplots_adjust(top=0.93, right=0.89, hspace=0.20, wspace=0.10)
    colorbar_ax = fig.add_axes([0.915, 0.17, 0.016, 0.66])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.set_label("Node X3 (mm; color clipped P01-P99)")
    save_figure(fig, output_dir / "06_model_input_mesh_by_design.png")
    return {
        f"design_{design + 1:02d}": {
            "representative_run": representative_runs[design],
            "nodes": int(len(meshes[design])),
            "min_xyz_mm": [float(x) for x in meshes[design].min(axis=0)],
            "max_xyz_mm": [float(x) for x in meshes[design].max(axis=0)],
        }
        for design in range(N_DESIGNS)
    }


def plot_mesh_loader_diagnostic(
    frame: pd.DataFrame, data_dir: Path, output_dir: Path
) -> dict:
    run_a, run_b = 1, N_LOCATIONS
    mesh_a = load_model_mesh_nodes(data_dir / "inp_files" / f"HoodImpact_{run_a}.inp")
    mesh_b = load_model_mesh_nodes(data_dir / "inp_files" / f"HoodImpact_{run_b}.inp")
    if mesh_a.shape != mesh_b.shape:
        raise ValueError("same-design diagnostic decks have different mesh shapes")
    displacement = mesh_b.astype(np.float64) - mesh_a.astype(np.float64)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    moved = displacement_norm > 1e-4
    static = ~moved
    moved_count = int(moved.sum())
    mean_translation = displacement[moved].mean(axis=0)
    impact_a = frame.loc[frame["run"] == run_a, ["X1", "X2", "X3"]].iloc[0].to_numpy(float)
    impact_b = frame.loc[frame["run"] == run_b, ["X1", "X2", "X3"]].iloc[0].to_numpy(float)
    impact_translation = impact_b - impact_a

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    static_points = mesh_a[static][::8]
    ax.scatter(static_points[:, 1], static_points[:, 0], s=0.8, color="#B8C1CC", alpha=0.45, linewidth=0, label="static assembly")
    ax.scatter(mesh_a[moved, 1], mesh_a[moved, 0], s=6, color=COLORS["blue"], alpha=0.75, linewidth=0, label=f"run {run_a} moving nodes")
    ax.scatter(mesh_b[moved, 1], mesh_b[moved, 0], s=6, color=COLORS["orange"], alpha=0.75, linewidth=0, label=f"run {run_b} moving nodes")
    ax.scatter([impact_a[1], impact_b[1]], [impact_a[0], impact_b[0]], marker="x", s=65, color=[COLORS["blue"], COLORS["orange"]], linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlabel("X2 (mm)")
    ax.set_ylabel("X1 (mm)")
    ax.set_title("Top view: same design, two impact locations")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    annotate_panel(ax, "A")

    ax = axes[0, 1]
    ax.scatter(static_points[:, 0], static_points[:, 2], s=0.8, color="#B8C1CC", alpha=0.45, linewidth=0)
    ax.scatter(mesh_a[moved, 0], mesh_a[moved, 2], s=6, color=COLORS["blue"], alpha=0.75, linewidth=0)
    ax.scatter(mesh_b[moved, 0], mesh_b[moved, 2], s=6, color=COLORS["orange"], alpha=0.75, linewidth=0)
    ax.scatter([impact_a[0], impact_b[0]], [impact_a[2], impact_b[2]], marker="x", s=65, color=[COLORS["blue"], COLORS["orange"]], linewidth=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X1 (mm)")
    ax.set_ylabel("X3 (mm)")
    ax.set_title("Side view: translated headform within the NODE block")
    annotate_panel(ax, "B")

    ax = axes[1, 0]
    first_rows = min(450, len(displacement_norm))
    ax.plot(np.arange(1, first_rows + 1), displacement_norm[:first_rows], color=COLORS["purple"], linewidth=1.3)
    ax.axvline(moved_count + 0.5, color=COLORS["orange"], linestyle="--", linewidth=1.2)
    ax.text(
        moved_count * 0.5,
        float(displacement_norm[moved].mean()) * 0.55,
        f"{moved_count} translated nodes\n({100*moved_count/len(mesh_a):.3f}% of loader input)",
        ha="center",
        fontsize=9,
    )
    ax.text(moved_count + 8, 0.05 * float(displacement_norm[moved].mean()), "remaining nodes: zero displacement", fontsize=8)
    ax.set_xlabel("Loader row (first 450 shown)")
    ax.set_ylabel("Coordinate displacement, run 142 − run 1 (mm)")
    ax.set_title("Only the first contiguous node block segment moves")
    annotate_panel(ax, "C")

    ax = axes[1, 1]
    positions = np.arange(3)
    width = 0.34
    ax.bar(positions - width / 2, mean_translation, width, color=COLORS["purple"], label="mean node translation")
    ax.bar(positions + width / 2, impact_translation, width, color=COLORS["green"], label="impact-coordinate translation")
    for x, value in zip(positions, mean_translation):
        ax.text(x - width / 2, value + np.sign(value) * 18, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(positions, ["ΔX1", "ΔX2", "ΔX3"])
    ax.set_ylabel("Translation (mm)")
    ax.set_title("Moving nodes follow the impact location rigidly")
    ax.legend(frameon=False, fontsize=8)
    annotate_panel(ax, "D")

    fig.suptitle(
        "Current mesh-loader diagnostic — the input contains the translated headform",
        fontsize=18,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "06b_mesh_loader_headform_diagnostic.png")
    return {
        "comparison_runs": [run_a, run_b],
        "total_loader_nodes": int(len(mesh_a)),
        "translated_nodes": moved_count,
        "translated_fraction_pct": float(100 * moved_count / len(mesh_a)),
        "mean_node_translation_xyz_mm": [float(value) for value in mean_translation],
        "impact_translation_xyz_mm": [float(value) for value in impact_translation],
        "max_static_node_displacement_mm": float(displacement_norm[static].max()),
        "max_moving_translation_residual_mm": float(
            np.linalg.norm(displacement[moved] - mean_translation, axis=1).max()
        ),
    }


def plot_source_subset_comparison(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    metrics = [
        ("hic15", "HIC15"),
        ("calculated_peak_g", "Peak acceleration (g)"),
        ("time_to_peak_ms", "Time to peak (ms)"),
        ("end_to_peak_pct", "End / peak (%)"),
        ("solve_s", "Solver wall time (s)"),
        ("n_raw_points", "Solver-history points before resampling"),
    ]
    palette = {"retained 50": COLORS["blue"], "new 92": COLORS["orange"]}
    for panel, ((column, label), ax) in enumerate(zip(metrics, axes.flat)):
        sns.violinplot(
            data=frame,
            x="source_group",
            y=column,
            hue="source_group",
            palette=palette,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(label)
        annotate_panel(ax, chr(ord("A") + panel))
    fig.suptitle(
        "Retained-50 vs newly generated-92 location subsets\n"
        "Differences can reflect location selection as well as generation provenance",
        fontsize=18,
        fontweight="bold",
        y=1.015,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "07_source_subset_comparison.png")


def plot_design_similarity_and_splits(frame: pd.DataFrame, output_dir: Path) -> dict:
    hic_profiles = frame.pivot(index="loc", columns="design", values="hic15")
    peak_profiles = frame.pivot(index="loc", columns="design", values="calculated_peak_g")
    hic_correlation = hic_profiles.corr(method="spearman")
    peak_correlation = peak_profiles.corr(method="spearman")
    labels = [f"D{i + 1:02d}" for i in range(N_DESIGNS)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for panel, (correlation, title, ax) in enumerate(
        [
            (hic_correlation, "Pairwise HIC15 profile correlation", axes[0, 0]),
            (peak_correlation, "Pairwise peak-g profile correlation", axes[0, 1]),
        ]
    ):
        sns.heatmap(
            correlation,
            vmin=0.65,
            vmax=1.0,
            cmap="mako",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar_kws={"label": "Spearman correlation", "shrink": 0.75},
            ax=ax,
        )
        for boundary in (4, 6, 10):
            ax.axhline(boundary, color="white", linewidth=1.5)
            ax.axvline(boundary, color="white", linewidth=1.5)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.set_xlabel("Design")
        ax.set_ylabel("Design")
        ax.set_title(title)
        annotate_panel(ax, chr(ord("A") + panel))

    ax = axes[1, 0]
    cluster_names = np.array(["A"] * 4 + ["B"] * 2 + ["C"] * 4 + ["D"] * 2)
    cluster_colors = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7"}
    wadh_profiles = frame.groupby(["design", "wadh_mm"], as_index=False)["hic15"].mean()
    for design in range(N_DESIGNS):
        group = wadh_profiles[wadh_profiles["design"] == design]
        cluster = cluster_names[design]
        ax.plot(
            group["wadh_mm"],
            group["hic15"],
            marker="o",
            markersize=3,
            linewidth=1.25,
            alpha=0.78,
            color=cluster_colors[cluster],
            label=f"D{design + 1:02d} (cluster {cluster})",
        )
    ax.set_xlabel("WADH band (mm)")
    ax.set_ylabel("Mean HIC15")
    ax.set_title("Design profiles across physical WADH bands")
    ax.legend(ncol=2, frameon=False, fontsize=7)
    annotate_panel(ax, "C")

    ax = axes[1, 1]
    # Codes: 0=train, 1=validation, 2=test, 3=mixed within design.
    split_masks = np.zeros((3, N_DESIGNS), dtype=int)
    split_masks[0, :] = 3
    split_masks[1, 10] = 1
    split_masks[1, 2] = 2
    split_masks[2, 0:4] = 2
    split_masks[2, 4:6] = 1
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    split_cmap = ListedColormap(["#4DAF4A", "#FFB000", "#D62728", "#7B68A6"])
    ax.imshow(split_masks, aspect="auto", cmap=split_cmap, vmin=-0.5, vmax=3.5)
    ax.set_xticks(range(N_DESIGNS), labels, rotation=45)
    ax.set_yticks(
        range(3),
        ["random by run\n(leaks every design)", "single-design holdout\n(current defaults)", "whole-cluster holdout\n(example)"],
    )
    for x in range(N_DESIGNS + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1)
    for y in range(4):
        ax.axhline(y - 0.5, color="white", linewidth=1)
    ax.set_xlabel("Hood design")
    ax.set_title("What different split strategies actually test")
    ax.legend(
        handles=[
            Patch(facecolor="#4DAF4A", label="train"),
            Patch(facecolor="#FFB000", label="validation"),
            Patch(facecolor="#D62728", label="test"),
            Patch(facecolor="#7B68A6", label="mixed train/val/test"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    annotate_panel(ax, "D")

    fig.suptitle(
        "Design similarity and generalization splits\n"
        "White lines mark known near-clone geometry clusters A/B/C/D",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "09_design_similarity_and_split_masks.png")
    hic_correlation.index = labels
    hic_correlation.columns = labels
    peak_correlation.index = labels
    peak_correlation.columns = labels
    hic_correlation.to_csv(output_dir / "design_hic_spearman_correlation.csv")
    peak_correlation.to_csv(output_dir / "design_peak_g_spearman_correlation.csv")

    off_diagonal = ~np.eye(N_DESIGNS, dtype=bool)
    return {
        "mean_pairwise_hic_spearman": float(hic_correlation.to_numpy()[off_diagonal].mean()),
        "min_pairwise_hic_spearman": float(hic_correlation.to_numpy()[off_diagonal].min()),
        "mean_pairwise_peak_g_spearman": float(peak_correlation.to_numpy()[off_diagonal].mean()),
        "min_pairwise_peak_g_spearman": float(peak_correlation.to_numpy()[off_diagonal].min()),
        "known_near_clone_clusters": {
            "A": [1, 2, 3, 4],
            "B": [5, 6],
            "C": [7, 8, 9, 10],
            "D": [11, 12],
        },
    }


def numeric_summary(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "mean": float(series.mean()),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(quantiles.loc[1.0]),
        "std": float(series.std()),
    }


def write_outputs(
    frame: pd.DataFrame,
    checks: dict,
    variance_components: dict,
    mesh_summary: dict,
    mesh_loader_diagnostic: dict,
    temporal_subsampling: dict,
    design_similarity: dict,
    output_dir: Path,
) -> None:
    frame.to_csv(output_dir / "per_run_eda_metrics.csv", index=False)
    design_summary = frame.groupby(["design", "design_label"]).agg(
        runs=("run", "size"),
        hic_mean=("hic15", "mean"),
        hic_median=("hic15", "median"),
        hic_min=("hic15", "min"),
        hic_max=("hic15", "max"),
        peak_g_mean=("calculated_peak_g", "mean"),
        peak_g_max=("calculated_peak_g", "max"),
        median_time_to_peak_ms=("time_to_peak_ms", "median"),
        median_end_to_peak_pct=("end_to_peak_pct", "median"),
    )
    design_summary.to_csv(output_dir / "per_design_summary.csv")

    top_hic = frame.nlargest(15, "hic15")[
        ["run", "design_label", "loc", "X1", "X2", "X3", "hic15", "calculated_peak_g", "time_to_peak_ms"]
    ]
    top_hic.to_csv(output_dir / "top_15_hic_runs.csv", index=False)
    frame[["run", "design", "loc", "reported_hic15", "hic15", "hic15_stored_shortfall", "hic15_stored_shortfall_pct"]].rename(
        columns={"hic15": "exhaustive_hic15"}
    ).to_csv(output_dir / "canonical_hic15.csv", index=False)

    summary = {
        "dataset": "HoodImpact_1704_EuroNCAP",
        "checks": checks,
        "hic15": numeric_summary(frame["hic15"]),
        "reported_hic15": numeric_summary(frame["reported_hic15"]),
        "peak_g": numeric_summary(frame["calculated_peak_g"]),
        "time_to_peak_ms": numeric_summary(frame["time_to_peak_ms"]),
        "half_max_span_ms": numeric_summary(frame["half_max_span_ms"]),
        "end_to_peak_pct": numeric_summary(frame["end_to_peak_pct"]),
        "solver_seconds": numeric_summary(frame["solve_s"]),
        "max_abs_reported_vs_history_peak_difference_g": float(frame["peak_g_difference"].abs().max()),
        "hic15_recalculation": {
            "mean_exhaustive_minus_stored": float(frame["hic15_stored_shortfall"].mean()),
            "max_exhaustive_minus_stored": float(frame["hic15_stored_shortfall"].max()),
            "max_stored_shortfall_pct": float(frame["hic15_stored_shortfall_pct"].max()),
            "runs_stored_shortfall_over_1_pct": int((frame["hic15_stored_shortfall_pct"] > 1).sum()),
            "runs_stored_shortfall_over_5_pct": int((frame["hic15_stored_shortfall_pct"] > 5).sum()),
            "runs_stored_shortfall_over_10_pct": int((frame["hic15_stored_shortfall_pct"] > 10).sum()),
            "runs_stored_shortfall_over_25_pct": int((frame["hic15_stored_shortfall_pct"] > 25).sum()),
        },
        "histories_with_end_above_5_percent_of_peak": int((frame["end_to_peak_pct"] > 5).sum()),
        "histories_with_end_above_10_percent_of_peak": int((frame["end_to_peak_pct"] > 10).sum()),
        "variance_components": variance_components,
        "representative_meshes": mesh_summary,
        "mesh_loader_diagnostic": mesh_loader_diagnostic,
        "temporal_subsampling": temporal_subsampling,
        "design_similarity": design_similarity,
    }
    with (output_dir / "eda_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    hic = summary["hic15"]
    peak = summary["peak_g"]
    t_peak = summary["time_to_peak_ms"]
    stored_hic = summary["reported_hic15"]
    hic_check = summary["hic15_recalculation"]
    end5 = summary["histories_with_end_above_5_percent_of_peak"]
    end10 = summary["histories_with_end_above_10_percent_of_peak"]
    peak_corr = frame[["calculated_peak_g", "hic15"]].corr().iloc[0, 1]
    impulse_corr = frame[["impulse_g_ms", "hic15"]].corr(method="spearman").iloc[0, 1]
    vc = variance_components["log10(HIC15)"]
    stride16 = temporal_subsampling["16"]
    worst = frame.loc[frame["hic15"].idxmax()]
    best = frame.loc[frame["hic15"].idxmin()]

    report = f"""# HoodImpact_1704_EuroNCAP exploratory summary

Generated from all **{len(frame):,} runs**: {N_DESIGNS} designs × {N_LOCATIONS} impact locations.

## Main findings

- Metadata are complete and balanced: every design has 142 locations, every location appears in all 12 designs, and every generation row is `ok`.
- Exhaustively recomputed HIC15 ranges from **{hic['min']:.1f}** to **{hic['max']:.1f}** (median **{hic['median']:.1f}**, mean **{hic['mean']:.1f}**, P95 **{hic['p95']:.1f}**).
- The supplied `generation_metrics.csv` HIC shortcut is not exhaustive: its mean is **{stored_hic['mean']:.1f}** versus **{hic['mean']:.1f}** after correction. It is low by more than 5% for **{hic_check['runs_stored_shortfall_over_5_pct']:,} runs** and more than 10% for **{hic_check['runs_stored_shortfall_over_10_pct']:,} runs**. Use `canonical_hic15.csv` for a scalar-HIC target.
- Peak resultant acceleration ranges from **{peak['min']:.1f} g** to **{peak['max']:.1f} g** (median **{peak['median']:.1f} g**).
- Time to peak ranges from **{t_peak['min']:.2f} ms** to **{t_peak['max']:.2f} ms** (median **{t_peak['median']:.2f} ms**).
- Peak acceleration and HIC15 have Pearson correlation **{peak_corr:.3f}**; acceleration impulse and HIC15 have Spearman correlation **{impulse_corr:.3f}**. Waveform shape therefore carries information beyond the single peak.
- On log10(HIC15), balanced variation is approximately **{100*vc['design']:.1f}% design**, **{100*vc['location']:.1f}% location**, and **{100*vc['interaction']:.1f}% non-additive design×location residual** (interaction and simulation/error variation cannot be separated without replicated cells).
- **{end5:,} / {len(frame):,}** histories end above 5% of their peak; **{end10:,} / {len(frame):,}** end above 10%. These 25 ms records are not equally settled at the right boundary.
- Default stride 16 retains a median **{stride16['median_peak_retained_pct']:.1f}%** of the native peak, but **{stride16['runs_losing_over_5_pct']:,} runs** lose more than 5% and the worst retains only **{stride16['worst_peak_retained_pct']:.1f}%**. Stride 16 yields {stride16['points']} points and stops at {stride16['last_time_ms']:.3f} ms rather than 25 ms.
- Lowest HIC15: run {int(best['run'])} (D{int(best['design'])+1:02d}, location {int(best['loc'])}) = **{best['hic15']:.1f}**. Highest: run {int(worst['run'])} (D{int(worst['design'])+1:02d}, location {int(worst['loc'])}) = **{worst['hic15']:.1f}**.

## Modeling implications

- Split by **design**, not randomly by run, when testing generalization to unseen hood geometries. A random split would place the same design geometry in train and test.
- The four documented near-clone geometry groups are A=D01-D04, B=D05-D06, C=D07-D10, and D=D11-D12. A single-design holdout still leaks a close clone; use whole-cluster holdouts for a strict geometry-generalization test.
- Preserve `loc`, `design`, and `source_subset` in evaluation tables so spatial and provenance-specific errors remain visible.
- The first 50 and remaining 92 locations are different selected subsets, so their distribution differences should not automatically be interpreted as a simulation-pipeline batch effect.
- Inspect the late-time tail before shortening the training window. Some histories retain meaningful acceleration at 25 ms.
- The mesh figure mirrors the current loader's exact first `*NODE` block. Confirm that this is the intended geometry scope before committing to a PointNet-style encoder.
- In one same-design comparison, **{mesh_loader_diagnostic.get('translated_nodes', 0)} nodes ({mesh_loader_diagnostic.get('translated_fraction_pct', 0):.3f}%)** move rigidly with impact position while all other loader nodes are unchanged. Thus the mesh input already encodes impact location through the included headform, in addition to the separately supplied X1/X2 coordinates.

## Outputs

- `00_hic15_validation.png`: stored shortcut versus exhaustive HIC15 and the worst missed window.
- `01_dataset_overview.png`: coverage, HIC distribution, design effects, response matrix, peak/HIC relation, waveform envelope.
- `02_spatial_hic_maps.png`: one spatial response map per design.
- `02_spatial_response_summary.png`: across-design mean and variability maps for HIC and peak g.
- `03_design_location_response.png`: design-location matrices and balanced variation decomposition.
- `04_acceleration_dynamics.png`: waveform envelopes, design means, all-run heatmap, representative curves.
- `05_feature_relationships.png`: HIC relationships and feature correlations.
- `06_model_input_mesh_by_design.png`: representative point clouds as read by the current loader (unless `--skip-mesh`).
- `06b_mesh_loader_headform_diagnostic.png`: same-design proof that 286 headform nodes move with impact position.
- `07_source_subset_comparison.png`: retained/new subset distributions.
- `08_temporal_subsampling_diagnostic.png`: peak fidelity at strides 2, 4, 8, 16, and 32.
- `09_design_similarity_and_split_masks.png`: target-profile similarity and split-strategy consequences.
- CSV and JSON files: reusable per-run, per-design, extreme-run, validation, and summary data.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    print(f"wrote {output_dir / 'per_run_eda_metrics.csv'}")
    print(f"wrote {output_dir / 'per_design_summary.csv'}")
    print(f"wrote {output_dir / 'top_15_hic_runs.csv'}")
    print(f"wrote {output_dir / 'eda_summary.json'}")
    print(f"wrote {output_dir / 'README.md'}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    frame, locations, metadata_checks = load_metadata(data_dir)
    histories, time, waveforms, history_checks = load_histories(data_dir, frame)
    frame = add_history_metrics(frame, histories, time, waveforms)
    checks = {**metadata_checks, **history_checks}

    plot_hic15_validation(frame, time, waveforms, output_dir)
    plot_overview(frame, locations, time, waveforms, output_dir)
    plot_spatial_hic_maps(frame, output_dir)
    plot_spatial_response_summary(frame, output_dir)
    variance_components = plot_design_location_response(frame, output_dir)
    plot_acceleration_dynamics(frame, time, waveforms, output_dir)
    plot_feature_relationships(frame, output_dir)
    plot_source_subset_comparison(frame, output_dir)
    temporal_subsampling = plot_temporal_subsampling(frame, time, waveforms, output_dir)
    design_similarity = plot_design_similarity_and_splits(frame, output_dir)
    mesh_summary = {}
    mesh_loader_diagnostic = {}
    if not args.skip_mesh:
        mesh_summary = plot_model_mesh_inputs(frame, data_dir, output_dir)
        mesh_loader_diagnostic = plot_mesh_loader_diagnostic(frame, data_dir, output_dir)
    write_outputs(
        frame,
        checks,
        variance_components,
        mesh_summary,
        mesh_loader_diagnostic,
        temporal_subsampling,
        design_similarity,
        output_dir,
    )
    print(f"\nEDA complete: {output_dir}")


if __name__ == "__main__":
    main()
