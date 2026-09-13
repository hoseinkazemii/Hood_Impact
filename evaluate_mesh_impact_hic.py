"""Score the predicted acceleration histories on the quantity that matters: HIC15.

The network predicts an acceleration history; the head-injury criterion is the
engineering output derived from it.  This command reads the histories exported
by ``train_mesh_impact_history.py`` (``test_acceleration_histories.csv``),
integrates both the simulated and the predicted signal with the HIC15 formula,
and reports R^2 and the error distribution over every held-out impact.

HIC15 = max over t1 < t2, t2 - t1 <= 15 ms of
        (t2 - t1) * [ (1 / (t2 - t1)) * integral_{t1}^{t2} a(t) dt ] ^ 2.5
with ``a`` the resultant head acceleration in g and time in seconds.

Ground truth is reported at three levels, because they are not the same number:

  * ``hic_ground_truth``      the saved sampled simulation history, on the same
                              time grid the model predicts on, so this is the
                              target the model can actually attain;
  * ``hic_full_resolution``   the full-resolution SAE1000 history the sampled
                              grid was drawn from, which isolates how much HIC
                              accuracy the time subsampling costs;
  * ``hic_solver_reference``  the ``hic15`` column recorded in the dataset's
                              ``generation_metrics.csv`` at generation time.

The last two need the local dataset directory and are omitted when it is absent.

Note on ``hic_solver_reference``: it does not maximise over the window length.
On the 1704 EuroNCAP set it reproduces, to six decimals, the HIC obtained when
only the widest admissible window (~15 ms) is evaluated for each start time, so
it under-reports true HIC15 whenever a shorter window is the maximiser.  It is
reported here as a provenance check on the recorded labels, not as a reference
for the model, and ``hic_full_resolution`` is the correct HIC of the same
signal.

Usage:
    python evaluate_mesh_impact_hic.py runs/mesh_impact_history/<run>
    python evaluate_mesh_impact_hic.py runs/mesh_impact_history/<run> --no-history-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HIC_WINDOW_S = 0.015
BAND_LIMITS_PERCENT = (5.0, 10.0, 15.0, 20.0)
_HISTORY_COLUMNS = ("run_number", "time", "acceleration_true_g", "acceleration_pred_g")


# --------------------------------------------------------------------------- HIC


def cumulative_trapezoid(time, acceleration):
    """Running integral of the acceleration over the saved samples."""
    integral = np.zeros(len(acceleration), dtype=np.float64)
    integral[1:] = np.cumsum(0.5 * (acceleration[1:] + acceleration[:-1]) * np.diff(time))
    return integral


def hic15(time, acceleration, max_window=HIC_WINDOW_S):
    """Return ``(hic, window_start_s, window_end_s)`` for one history.

    Every admissible ``(t1, t2)`` pair on the saved grid is evaluated, so the
    result is the exact maximum over that grid rather than a search heuristic.
    A window whose mean acceleration is negative cannot maximise HIC and the
    2.5 power is undefined there, so such windows score zero.
    """
    time = np.asarray(time, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if time.ndim != 1 or time.shape != acceleration.shape or len(time) < 2:
        raise ValueError("Time and acceleration must be matching vectors of at least two samples")
    if not (np.isfinite(time).all() and np.isfinite(acceleration).all()):
        raise ValueError("Time and acceleration must be finite")
    if not np.all(np.diff(time) > 0):
        raise ValueError("Time must be strictly increasing")

    integral = cumulative_trapezoid(time, acceleration)
    window = time[np.newaxis, :] - time[:, np.newaxis]
    inside = (window > 0.0) & (window <= max_window)
    if not inside.any():
        raise ValueError(f"No sample pair falls inside the {max_window * 1000:g} ms HIC window")

    mean_acceleration = np.zeros_like(window)
    np.divide(
        integral[np.newaxis, :] - integral[:, np.newaxis],
        window,
        out=mean_acceleration,
        where=inside,
    )
    np.clip(mean_acceleration, 0.0, None, out=mean_acceleration)
    candidates = np.where(inside, window * mean_acceleration**2.5, 0.0)
    start, end = np.unravel_index(int(np.argmax(candidates)), candidates.shape)
    return float(candidates[start, end]), float(time[start]), float(time[end])


# ------------------------------------------------------------------------ loading


def load_test_histories(run_dir):
    """Group the exported test predictions by run, preserving the saved order."""
    frame = pd.read_csv(Path(run_dir) / "test_acceleration_histories.csv")
    missing = sorted(set(_HISTORY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"test_acceleration_histories.csv is missing columns: {missing}")
    if frame.empty:
        raise ValueError("test_acceleration_histories.csv contains no rows")
    return {
        int(run_number): group[list(_HISTORY_COLUMNS[1:])].to_numpy(dtype=np.float64)
        for run_number, group in frame.groupby("run_number", sort=True)
    }


def load_full_resolution_history(data_dir, run_number):
    """Read the full-resolution SAE1000 history for one run, or ``None``."""
    if data_dir is None:
        return None
    matches = sorted(Path(data_dir).glob(f"output_history_acc/HoodImpact_{run_number}_*.csv"))
    if len(matches) != 1:
        return None
    frame = pd.read_csv(matches[0])
    if not {"Time", "A(in g)"}.issubset(frame.columns):
        return None
    return frame["Time"].to_numpy(np.float64), frame["A(in g)"].to_numpy(np.float64)


def load_solver_reference_hic(data_dir):
    """HIC15 recorded when the dataset was generated, keyed by run number."""
    if data_dir is None:
        return {}
    path = Path(data_dir) / "generation_metrics.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if not {"run", "hic15"}.issubset(frame.columns):
        return {}
    frame = frame.dropna(subset=["run", "hic15"])
    return {int(run): float(value) for run, value in zip(frame["run"], frame["hic15"])}


def load_impact_coordinates(data_dir, run_dir):
    """Impact XY per location id, from the dataset or from a saved run report."""
    if data_dir is not None:
        matches = sorted(Path(data_dir).glob("impact_locations_*.csv"))
        for path in matches:
            frame = pd.read_csv(path)
            if {"loc", "X1", "X2"}.issubset(frame.columns):
                return {
                    int(loc): (float(x), float(y))
                    for loc, x, y in zip(frame["loc"], frame["X1"], frame["X2"])
                }
    fallback = Path(run_dir) / "per_history_performance.csv"
    if fallback.is_file():
        frame = pd.read_csv(fallback)
        if {"location_id", "impact_x_mm", "impact_y_mm"}.issubset(frame.columns):
            return {
                int(loc): (float(x), float(y))
                for loc, x, y in zip(
                    frame["location_id"], frame["impact_x_mm"], frame["impact_y_mm"]
                )
            }
    return {}


def resolve_data_dir(run_dir, explicit):
    """Locate the dataset directory; the path recorded in config is a cluster path."""
    if explicit is not None:
        data_dir = Path(explicit).resolve()
        if not (data_dir / "output_history_acc").is_dir():
            raise ValueError(f"{data_dir} does not contain output_history_acc/")
        return data_dir
    config_path = Path(run_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    recorded = config.get("data", {}).get("acceleration_dir")
    if not recorded:
        return None
    recorded_root = Path(recorded).parent
    for candidate in (recorded_root, Path(__file__).resolve().parent / "Data" / recorded_root.name):
        if (candidate / "output_history_acc").is_dir():
            return candidate.resolve()
    return None


# ------------------------------------------------------------------------ metrics


def r2_score(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((prediction - truth) ** 2)) / denominator


def error_summary(truth, prediction, label):
    """R^2 and the error distribution of one HIC estimate against a reference."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[finite], prediction[finite]
    if not len(truth):
        raise ValueError(f"No finite pairs available for '{label}'")
    error = prediction - truth
    relative = 100.0 * error / truth
    summary = {
        "comparison": label,
        "n": int(len(truth)),
        "r2": r2_score(truth, prediction),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": float(np.mean(np.abs(relative))),
        "median_absolute_percentage_error": float(np.median(np.abs(relative))),
        "max_absolute_percentage_error": float(np.max(np.abs(relative))),
        "bias": float(np.mean(error)),
        "reference_mean": float(truth.mean()),
        "reference_range": [float(truth.min()), float(truth.max())],
    }
    for limit in BAND_LIMITS_PERCENT:
        summary[f"within_{limit:g}_percent"] = float(np.mean(np.abs(relative) <= limit) * 100.0)
    return summary


def evaluate_run(run_dir, data_dir=None, samples_per_design=None, max_window=HIC_WINDOW_S):
    """Compute per-impact HIC and acceleration metrics for every test history."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if samples_per_design is None:
        samples_per_design = int(config.get("data", {}).get("samples_per_design", 142))

    histories = load_test_histories(run_dir)
    solver_reference = load_solver_reference_hic(data_dir)
    coordinates = load_impact_coordinates(data_dir, run_dir)

    records = []
    for run_number, values in sorted(histories.items()):
        time, truth, prediction = values[:, 0], values[:, 1], values[:, 2]
        hic_truth, truth_start, truth_end = hic15(time, truth, max_window)
        hic_prediction, prediction_start, prediction_end = hic15(time, prediction, max_window)
        residual = prediction - truth
        location_id = (run_number - 1) % samples_per_design + 1
        impact_x, impact_y = coordinates.get(location_id, (np.nan, np.nan))
        record = {
            "run_number": run_number,
            "design_id": (run_number - 1) // samples_per_design,
            "location_id": location_id,
            "impact_x_mm": impact_x,
            "impact_y_mm": impact_y,
            "hic_ground_truth": hic_truth,
            "hic_prediction": hic_prediction,
            "hic_error": hic_prediction - hic_truth,
            "hic_relative_error_pct": 100.0 * (hic_prediction - hic_truth) / hic_truth,
            "hic_window_start_true_ms": truth_start * 1000.0,
            "hic_window_duration_true_ms": (truth_end - truth_start) * 1000.0,
            "hic_window_start_pred_ms": prediction_start * 1000.0,
            "hic_window_duration_pred_ms": (prediction_end - prediction_start) * 1000.0,
            "acceleration_rmse_g": float(np.sqrt(np.mean(residual**2))),
            "acceleration_mae_g": float(np.mean(np.abs(residual))),
            "acceleration_r2": r2_score(truth, prediction),
            "true_peak_g": float(truth.max()),
            "predicted_peak_g": float(prediction.max()),
        }

        full_resolution = load_full_resolution_history(data_dir, run_number)
        if full_resolution is not None:
            hic_full, _, _ = hic15(full_resolution[0], full_resolution[1], max_window)
            record["hic_full_resolution"] = hic_full
            record["hic_relative_error_pct_vs_full_resolution"] = (
                100.0 * (hic_prediction - hic_full) / hic_full
            )
            record["sampling_relative_error_pct"] = 100.0 * (hic_truth - hic_full) / hic_full
        if run_number in solver_reference:
            record["hic_solver_reference"] = solver_reference[run_number]
        records.append(record)

    results = pd.DataFrame(records)
    summaries = [
        error_summary(
            results["hic_ground_truth"],
            results["hic_prediction"],
            "prediction vs sampled simulation (the target the model can attain)",
        )
    ]
    if "hic_full_resolution" in results:
        summaries.append(
            error_summary(
                results["hic_full_resolution"],
                results["hic_prediction"],
                "prediction vs full-resolution simulation (the engineering answer)",
            )
        )
        summaries.append(
            error_summary(
                results["hic_full_resolution"],
                results["hic_ground_truth"],
                "sampled simulation vs full-resolution simulation (time-subsampling penalty)",
            )
        )
        if "hic_solver_reference" in results:
            summaries.append(
                error_summary(
                    results["hic_solver_reference"],
                    results["hic_full_resolution"],
                    "correct HIC15 vs the label recorded at dataset generation "
                    "(provenance check on the labels, not on the model)",
                )
            )

    ordered = [values for _, values in sorted(histories.items())]
    pooled_truth = np.concatenate([values[:, 1] for values in ordered])
    pooled_prediction = np.concatenate([values[:, 2] for values in ordered])
    pooled_residual = pooled_prediction - pooled_truth
    worst = results.reindex(
        results["hic_relative_error_pct"].abs().sort_values(ascending=False).index
    ).head(10)
    report = {
        "run_dir": str(run_dir),
        "source": "test_acceleration_histories.csv (best validation checkpoint)",
        "num_test_impacts": int(len(results)),
        "test_design_ids": sorted({int(value) for value in results["design_id"]}),
        "hic_window_s": max_window,
        "time_grid": {
            "num_points": int(len(ordered[0])),
            "time_step_ms": float(np.diff(ordered[0][:, 0]).mean() * 1000.0),
            "time_subsample_stride": config.get("preprocessing", {}).get("time_subsample_stride"),
        },
        "dataset_dir": str(data_dir) if data_dir is not None else None,
        "hic": {summary["comparison"]: summary for summary in summaries},
        "acceleration": {
            "pooled_rmse_g": float(np.sqrt(np.mean(pooled_residual**2))),
            "pooled_mae_g": float(np.mean(np.abs(pooled_residual))),
            "pooled_r2": r2_score(pooled_truth, pooled_prediction),
            "mean_per_history_r2": float(results["acceleration_r2"].mean()),
        },
        "worst_hic_cases": worst[
            [
                "run_number",
                "location_id",
                "hic_ground_truth",
                "hic_prediction",
                "hic_error",
                "hic_relative_error_pct",
            ]
        ].to_dict(orient="records"),
    }
    if {"hic_solver_reference", "hic_full_resolution"}.issubset(results.columns):
        shortfall = 100.0 * (
            results["hic_full_resolution"] - results["hic_solver_reference"]
        ) / results["hic_solver_reference"]
        report["recorded_label_check"] = {
            "note": (
                "generation_metrics.csv hic15 evaluates only the widest admissible "
                "window per start time, so it under-reports HIC15 whenever a shorter "
                "window is the maximiser"
            ),
            "labels_matching_correct_hic": int((shortfall.abs() <= 0.01).sum()),
            "labels_under_reporting": int((shortfall > 0.01).sum()),
            "mean_shortfall_percent": float(shortfall.mean()),
            "max_shortfall_percent": float(shortfall.max()),
        }
    return results, report


# ----------------------------------------------------------------------- plotting


def export_parity_plot(results, path, tolerance=10.0, title=None):
    """Predicted vs simulated HIC with a tolerance band, plus the relative error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = results["hic_ground_truth"].to_numpy(np.float64)
    prediction = results["hic_prediction"].to_numpy(np.float64)
    relative = 100.0 * (prediction - truth) / truth
    inside = np.abs(relative) <= tolerance
    fraction = tolerance / 100.0
    summary = error_summary(truth, prediction, "parity")

    fig, (parity_ax, error_ax) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    try:
        low = min(truth.min(), prediction.min())
        high = max(truth.max(), prediction.max())
        pad = 0.04 * (high - low)
        line = np.array([low - pad, high + pad])

        parity_ax.fill_between(line, line * (1 - fraction), line * (1 + fraction),
                               color="#4c72b0", alpha=0.13, lw=0,
                               label=f"$\\pm${tolerance:g}% band", zorder=1)
        parity_ax.plot(line, line * (1 + fraction), color="#4c72b0", lw=1.0, ls="--", alpha=0.7, zorder=2)
        parity_ax.plot(line, line * (1 - fraction), color="#4c72b0", lw=1.0, ls="--", alpha=0.7, zorder=2)
        parity_ax.plot(line, line, color="0.25", lw=1.6, label="1:1", zorder=3)
        parity_ax.scatter(truth[inside], prediction[inside], s=34, facecolor="#2a7f62",
                          edgecolor="white", linewidth=0.5, alpha=0.9, zorder=4,
                          label=f"within {tolerance:g}%  (n={int(inside.sum())})")
        parity_ax.scatter(truth[~inside], prediction[~inside], s=44, facecolor="#c8553d",
                          edgecolor="white", linewidth=0.5, alpha=0.95, marker="D", zorder=5,
                          label=f"outside  (n={int((~inside).sum())})")
        parity_ax.set(xlim=(line[0], line[1]), ylim=(line[0], line[1]),
                      xlabel="Simulated HIC15", ylabel="Predicted HIC15",
                      title="Predicted vs simulated HIC")
        parity_ax.set_aspect("equal", adjustable="box")
        parity_ax.grid(alpha=0.25, lw=0.6)
        parity_ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
        stats = (f"$R^2$ = {summary['r2']:.4f}\n"
                 f"RMSE = {summary['rmse']:.1f}\n"
                 f"MAE  = {summary['mae']:.1f}\n"
                 f"MAPE = {summary['mape_percent']:.2f}%\n"
                 f"bias = {summary['bias']:+.1f}\n"
                 f"within {tolerance:g}% : {inside.mean() * 100:.1f}%")
        parity_ax.text(0.98, 0.02, stats, transform=parity_ax.transAxes, ha="right", va="bottom",
                       fontsize=9.5, family="monospace",
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                 edgecolor="0.75", alpha=0.94))

        error_ax.axhspan(-tolerance, tolerance, color="#4c72b0", alpha=0.13, lw=0,
                         label=f"$\\pm${tolerance:g}% band")
        error_ax.axhline(tolerance, color="#4c72b0", lw=1.0, ls="--", alpha=0.7)
        error_ax.axhline(-tolerance, color="#4c72b0", lw=1.0, ls="--", alpha=0.7)
        error_ax.axhline(0.0, color="0.25", lw=1.6)
        error_ax.scatter(truth[inside], relative[inside], s=34, facecolor="#2a7f62",
                         edgecolor="white", linewidth=0.5, alpha=0.9)
        error_ax.scatter(truth[~inside], relative[~inside], s=44, facecolor="#c8553d",
                         edgecolor="white", linewidth=0.5, alpha=0.95, marker="D")
        error_ax.set(xlabel="Simulated HIC15",
                     ylabel="Relative error  (pred $-$ simulation) / simulation  [%]",
                     title="Relative error vs HIC magnitude")
        error_ax.grid(alpha=0.25, lw=0.6)
        error_ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

        fig.suptitle(title or f"HIC15 from predicted acceleration  |  n = {len(results)}",
                     fontsize=11.5, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(path, dpi=300)
    finally:
        plt.close(fig)
    return path


def export_error_distribution_plot(results, path):
    """Where the HIC error sits: distribution, history-error trend, location map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relative = results["hic_relative_error_pct"].to_numpy(np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    try:
        axes[0].hist(relative, bins=24, color="#4c72b0", edgecolor="white")
        axes[0].axvline(0.0, color="0.25", lw=1.4)
        axes[0].axvline(float(np.mean(relative)), color="#c8553d", lw=1.4, ls="--",
                        label=f"mean {np.mean(relative):+.2f}%")
        axes[0].set(xlabel="HIC relative error (%)", ylabel="Test impacts",
                    title="HIC error distribution")
        axes[0].legend(fontsize=9, framealpha=0.92)
        axes[0].grid(alpha=0.22, lw=0.6)

        axes[1].scatter(results["acceleration_rmse_g"], np.abs(relative), s=32,
                        facecolor="#2a7f62", edgecolor="white", linewidth=0.5, alpha=0.9)
        axes[1].set(xlabel="Acceleration RMSE (g)", ylabel="|HIC relative error| (%)",
                    title="History error vs HIC error")
        axes[1].grid(alpha=0.22, lw=0.6)

        coordinates = results[["impact_x_mm", "impact_y_mm"]].to_numpy(np.float64)
        if np.isfinite(coordinates).all():
            limit = float(np.max(np.abs(relative)))
            scatter = axes[2].scatter(coordinates[:, 1], coordinates[:, 0], c=relative, s=64,
                                      cmap="coolwarm", vmin=-limit, vmax=limit,
                                      edgecolor="white", linewidth=0.5)
            fig.colorbar(scatter, ax=axes[2], label="HIC relative error (%)")
            axes[2].set(xlabel="Impact Y (mm)", ylabel="Impact X (mm)",
                        title="HIC error by impact location")
            axes[2].set_aspect("equal", adjustable="datalim")
            axes[2].grid(alpha=0.22, lw=0.6)
        else:
            axes[2].set_axis_off()
            axes[2].text(0.5, 0.5, "Impact coordinates unavailable", ha="center", va="center")

        fig.tight_layout()
        fig.savefig(path, dpi=300)
    finally:
        plt.close(fig)
    return path


# ------------------------------------------------------------------------ command


def print_report(report):
    print(f"\nTest impacts: {report['num_test_impacts']}  "
          f"(held-out design IDs {report['test_design_ids']})")
    grid = report["time_grid"]
    print(f"Time grid: {grid['num_points']} points, {grid['time_step_ms']:.4f} ms step "
          f"(subsample stride {grid['time_subsample_stride']})")
    acceleration = report["acceleration"]
    print(f"\nAcceleration history: R2 = {acceleration['pooled_r2']:.4f}   "
          f"RMSE = {acceleration['pooled_rmse_g']:.3f} g   "
          f"MAE = {acceleration['pooled_mae_g']:.3f} g")
    print("\nHIC15:")
    for label, summary in report["hic"].items():
        print(f"\n  {label}")
        print(f"    R2 = {summary['r2']:.4f}   RMSE = {summary['rmse']:.2f}   "
              f"MAE = {summary['mae']:.2f}   MAPE = {summary['mape_percent']:.2f}%   "
              f"bias = {summary['bias']:+.2f}")
        print("    " + "   ".join(
            f"within {limit:g}%: {summary[f'within_{limit:g}_percent']:.1f}%"
            for limit in BAND_LIMITS_PERCENT
        ))
    print("\nWorst 10 impacts by relative HIC error:")
    print(f"    {'run':>5} {'loc':>4} {'HIC true':>10} {'HIC pred':>10} {'error':>9} {'rel %':>8}")
    for case in report["worst_hic_cases"]:
        print(f"    {case['run_number']:5d} {case['location_id']:4d} "
              f"{case['hic_ground_truth']:10.1f} {case['hic_prediction']:10.1f} "
              f"{case['hic_error']:9.1f} {case['hic_relative_error_pct']:8.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path,
                        help="run directory containing test_acceleration_histories.csv")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset directory holding output_history_acc/ (auto-detected)")
    parser.add_argument("--samples-per-design", type=int, default=None)
    parser.add_argument("--max-window", type=float, default=HIC_WINDOW_S,
                        help="HIC averaging window in seconds (default: 0.015)")
    parser.add_argument("--tolerance", type=float, default=10.0,
                        help="parity-plot error band half-width in percent (default: 10)")
    parser.add_argument("--no-history-plots", action="store_true",
                        help="skip the per-impact acceleration plots, PDF, and gallery")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    data_dir = resolve_data_dir(run_dir, args.data_dir)
    if data_dir is None:
        print("Full-resolution dataset not found locally; reporting sampled-grid HIC only.")
    else:
        print(f"Dataset directory: {data_dir}")

    results, report = evaluate_run(run_dir, data_dir, args.samples_per_design, args.max_window)
    results_path = run_dir / "hic_results_pred_vs_gt.csv"
    results.to_csv(results_path, index=False)
    report_path = run_dir / "hic_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\nWrote: {results_path}")
    print(f"Wrote: {report_path}")

    parity_path = export_parity_plot(
        results,
        run_dir / f"hic_pred_vs_gt_{args.tolerance:g}pct.png",
        args.tolerance,
        title=f"HIC15 from predicted acceleration  |  {run_dir.name}  |  n = {len(results)}",
    )
    print(f"Wrote: {parity_path}")
    distribution_path = export_error_distribution_plot(
        results, run_dir / "hic_error_distribution.png"
    )
    print(f"Wrote: {distribution_path}")

    if not args.no_history_plots:
        from visualize_mesh_impact_test_set import export_all_test_plots

        exports = export_all_test_plots(run_dir, results)
        print(f"Wrote: {exports['png_count']} acceleration plots in {exports['png_directory']}")
        print(f"Wrote: {exports['pdf_path']} ({exports['pdf_page_count']} pages)")
        print(f"Wrote: {exports['gallery_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
