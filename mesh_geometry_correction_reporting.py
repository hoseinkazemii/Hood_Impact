"""Compare frozen and corrected responses on exactly the same supplied curves.

Only post-prediction values are read. HIC15 is calculated on the supplied
prediction grid, never presented as HIC of the original full-resolution data.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_design_sensitivity import write_sensitivity_report
from hic15 import batched_hic
from mesh_change_metrics import design_sensitivity_metrics


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate(records, prediction_times, samples_per_design):
    times = np.asarray(prediction_times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("prediction_times must be a finite, strictly increasing vector of at least two seconds values")
    if not records:
        raise ValueError("records must contain at least one evaluated history")
    normalized = []
    runs, design_locations, inferred_counts = set(), set(), set()
    for record in records:
        row = {key: _integer(record[key], key, 0 if key == "design_id" else 1)
               for key in ("run_number", "design_id", "location_id")}
        if row["run_number"] in runs or (row["design_id"], row["location_id"]) in design_locations:
            raise ValueError("Duplicate run number or design/location history")
        runs.add(row["run_number"])
        design_locations.add((row["design_id"], row["location_id"]))
        xy = np.asarray(record["impact_xy"], dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("impact_xy must contain two finite physical-millimetre coordinates")
        row["impact_xy"] = xy
        for key in ("ground_truth", "baseline", "corrected"):
            array = np.asarray(record[key], dtype=np.float64)
            if array.shape != times.shape or not np.isfinite(array).all():
                raise ValueError(f"{key} must be a finite physical-g vector matching prediction_times exactly")
            row[key] = array
        if row["design_id"]:
            numerator = row["run_number"] - row["location_id"]
            count, remainder = divmod(numerator, row["design_id"])
            if remainder or count < 1:
                raise ValueError("run_number disagrees with explicit design_id/location_id")
            inferred_counts.add(count)
        normalized.append(row)
    if samples_per_design is None:
        if len(inferred_counts) > 1:
            raise ValueError("Inconsistent samples_per_design inferred from run identifiers")
        samples_per_design = next(iter(inferred_counts), max(row["location_id"] for row in normalized))
    samples_per_design = _integer(samples_per_design, "samples_per_design", 1)
    for row in normalized:
        if row["location_id"] > samples_per_design or row["run_number"] != row["design_id"] * samples_per_design + row["location_id"]:
            raise ValueError("run_number disagrees with explicit design_id/location_id and samples_per_design")
    return sorted(normalized, key=lambda row: row["run_number"]), times, samples_per_design


def _accuracy(truth, prediction):
    residual = prediction - truth
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.square(truth - np.mean(truth)).sum())
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(residual))),
            "r2": 1.0 - float(np.square(residual).sum()) / denominator if denominator > 0 else None}


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sensitivity_delta(baseline, corrected):
    keys = ("design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine", "difference_gain")
    delta = {key: corrected[key] - baseline[key]
             if baseline[key] is not None and corrected[key] is not None else None for key in keys}
    before, after = baseline["difference_amplitude_ratio"], corrected["difference_amplitude_ratio"]
    delta["amplitude_ratio_distance_to_one_change"] = abs(after - 1) - abs(before - 1) if before is not None and after is not None else None
    return delta


def _plot_parity(output_dir, prefix, truth, baseline, corrected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    low = min(float(values.min()) for values in (truth, baseline, corrected))
    high = max(float(values.max()) for values in (truth, baseline, corrected))
    padding = max((high - low) * .05, 1.)
    for ax, values, label, color in zip(axes, (baseline, corrected), ("Frozen baseline", "Geometry correction"), ("#737373", "#1765a1")):
        ax.scatter(truth, values, s=18, alpha=.65, color=color)
        ax.plot([low - padding, high + padding], [low - padding, high + padding], "k--", linewidth=1)
        ax.set(xlabel="Ground-truth HIC15 (supplied grid)", ylabel="Predicted HIC15 (supplied grid)", title=label,
               xlim=(low - padding, high + padding), ylim=(low - padding, high + padding))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    fig.suptitle("HIC15 parity on sampled prediction histories (15 ms window)")
    try:
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_hic_pred_vs_gt.png", dpi=180)
    finally:
        plt.close(fig)


def _plot_histories(output_dir, prefix, records, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    directory = output_dir / f"{prefix}_acceleration_plots"
    directory.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for record in records:
        groups[record["location_id"]].append(record)
    for location, rows in sorted(groups.items()):
        columns = min(len(rows), 3)
        nrows = (len(rows) + columns - 1) // columns
        fig, axes = plt.subplots(nrows, columns, figsize=(5.1 * columns, 3.5 * nrows), squeeze=False, sharex=True, sharey=True)
        for ax, record in zip(axes.flat, rows):
            ax.plot(times * 1000, record["ground_truth"], color="#202020", linewidth=1.8)
            ax.plot(times * 1000, record["baseline"], color="#969696", linestyle="--", linewidth=1.6)
            ax.plot(times * 1000, record["corrected"], color="#1765a1", linewidth=1.6)
            ax.set(title=f"Design {record['design_id']} | run {record['run_number']}", xlabel="Time (ms)", ylabel="Acceleration (g)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(rows):]:
            ax.set_visible(False)
        xy = rows[0]["impact_xy"]
        fig.suptitle(f"Impact location {location} | XY = ({xy[0]:.2f}, {xy[1]:.2f}) mm")
        fig.legend(handles=[Line2D([0], [0], color="#202020", label="Ground truth"),
                            Line2D([0], [0], color="#969696", linestyle="--", label="Frozen baseline"),
                            Line2D([0], [0], color="#1765a1", label="Corrected")],
                   loc="lower center", ncol=3, frameon=False)
        try:
            fig.tight_layout(rect=(0, .06, 1, .94))
            fig.savefig(directory / f"location_{location:03d}.png", dpi=160)
        finally:
            plt.close(fig)


def export_correction_evaluation(output_dir, records, prediction_times, prefix="test", make_plots=True, *, samples_per_design=None):
    """Export baseline/correction metrics and plots on identical supplied curves.

    Each record has run_number, zero-based design_id, one-based location_id,
    physical-mm impact_xy, and ground_truth/baseline/corrected 1D physical-g
    histories. prediction_times is their shared physical-seconds grid. Pass
    samples_per_design explicitly for single-design subsets; otherwise it is
    inferred from the run/design/location relationship where possible.

    Absolute acceleration metrics pool all supplied points. Absolute HIC15
    metrics weight each history equally. Sensitivity uses all unordered pairs
    at each matched location, including zero-variation ground truth; unmatched
    single-design locations are excluded only from sensitivity. Baseline and
    correction always share the exact cohort and reference values.
    """
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix or prefix in (".", "..") or any(char in prefix for char in "/\\"):
        raise ValueError("prefix must be a nonempty filename component")
    records, times, count = _validate(list(records), prediction_times, samples_per_design)
    truth = np.stack([row["ground_truth"] for row in records])
    baseline = np.stack([row["baseline"] for row in records])
    corrected = np.stack([row["corrected"] for row in records])
    dataset = SimpleNamespace(run_numbers=[row["run_number"] for row in records],
                              time_arrays=[times for _ in records],
                              indentor_positions=[row["impact_xy"] for row in records])
    sensitivities = {name: design_sensitivity_metrics(dataset, prediction.ravel(), truth.ravel(), count)
                     for name, prediction in (("baseline", baseline), ("corrected", corrected))}
    hic = {name: batched_hic(times, values) for name, values in (("ground_truth", truth), ("baseline", baseline), ("corrected", corrected))}
    if not all(np.isfinite(values).all() for values in hic.values()):
        raise ValueError("Computed HIC15 must remain finite")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"schema_version": 1, "prefix": prefix,
               "cohort": {"run_numbers": dataset.run_numbers,
                          "design_ids": sorted({row["design_id"] for row in records}),
                          "location_ids": sorted({row["location_id"] for row in records}),
                          "num_histories": len(records), "points_per_history": len(times),
                          "samples_per_design": count,
                          "prediction_times_seconds": times.tolist(), "impact_xy_checked": True},
               "hic15_source": "supplied sampled acceleration histories; seconds; 15 ms maximum window",
               "aggregation": {"acceleration": "all supplied points equally weighted",
                               "hic15": "one HIC15 value per supplied curve, equally weighted",
                               "design_sensitivity": "all same-location unordered design pairs and time samples equally weighted; no interpolation"}}
    for name, prediction in (("baseline", baseline), ("corrected", corrected)):
        artifact_prefix = f"baseline_{prefix}" if name == "baseline" else prefix
        frame = pd.DataFrame({"run_number": np.repeat(dataset.run_numbers, len(times)),
                              "design_id": np.repeat([row["design_id"] for row in records], len(times)),
                              "location_id": np.repeat([row["location_id"] for row in records], len(times)),
                              "impact_x_mm": np.repeat([row["impact_xy"][0] for row in records], len(times)),
                              "impact_y_mm": np.repeat([row["impact_xy"][1] for row in records], len(times)),
                              "time": np.tile(times, len(records)),
                              "acceleration_true_g": truth.ravel(), "acceleration_pred_g": prediction.ravel()})
        history_path = output_dir / f"{artifact_prefix}_acceleration_histories.csv"
        frame.to_csv(history_path, index=False)
        sensitivity = sensitivities[name]
        sensitivity.update(source_history=history_path.name, source_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
                           evaluated_run_numbers=dataset.run_numbers,
                           location_ids=metrics["cohort"]["location_ids"])
        metrics[name] = {"acceleration": _accuracy(truth, prediction),
                         "hic15": _accuracy(hic["ground_truth"], hic[name]),
                         "design_sensitivity": sensitivity}
        if make_plots:
            write_sensitivity_report(output_dir, sensitivity, artifact_prefix)
        else:
            _write_json(output_dir / f"{artifact_prefix}_design_sensitivity.json", sensitivity)
            pd.json_normalize(sensitivity["per_location"], sep="_").to_csv(
                output_dir / f"{artifact_prefix}_design_sensitivity_by_location.csv", index=False)
    comparison = {"definition": "corrected minus frozen baseline; smaller errors and larger skill/alignment are better; amplitude is best near one",
                  "acceleration": {key: metrics["corrected"]["acceleration"][key] - metrics["baseline"]["acceleration"][key]
                                   if metrics["corrected"]["acceleration"][key] is not None and metrics["baseline"]["acceleration"][key] is not None else None
                                   for key in ("mse", "rmse", "mae", "r2")},
                  "hic15": {key: metrics["corrected"]["hic15"][key] - metrics["baseline"]["hic15"][key]
                            if metrics["corrected"]["hic15"][key] is not None and metrics["baseline"]["hic15"][key] is not None else None
                            for key in ("mse", "rmse", "mae", "r2")},
                  "design_sensitivity": _sensitivity_delta(sensitivities["baseline"], sensitivities["corrected"]),
                  "hic15_design_sensitivity": _sensitivity_delta(sensitivities["baseline"]["hic15"], sensitivities["corrected"]["hic15"])}
    metrics["comparison"] = comparison
    hic_frame = pd.DataFrame({"run_number": dataset.run_numbers,
                              "design_id": [row["design_id"] for row in records],
                              "location_id": [row["location_id"] for row in records],
                              "impact_x_mm": [row["impact_xy"][0] for row in records],
                              "impact_y_mm": [row["impact_xy"][1] for row in records],
                              "hic15_true_sampled_grid": hic["ground_truth"],
                              "hic15_baseline_sampled_grid": hic["baseline"],
                              "hic15_corrected_sampled_grid": hic["corrected"]})
    hic_frame.to_csv(output_dir / f"{prefix}_hic_per_curve.csv", index=False)
    _write_json(output_dir / f"{prefix}_correction_comparison.json", {"cohort": metrics["cohort"], **comparison})
    _write_json(output_dir / f"{prefix}_metrics.json", metrics)
    if prefix == "test":
        _write_json(output_dir / "metrics.json", metrics)
    if make_plots:
        _plot_parity(output_dir, prefix, hic["ground_truth"], hic["baseline"], hic["corrected"])
        _plot_histories(output_dir, prefix, records, times)
    return metrics
