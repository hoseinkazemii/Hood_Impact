"""Report matched-location design differences without loading or retraining a model."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mesh_change_metrics import design_sensitivity_metrics


def sensitivity_log_values(report):
    """Stable W&B names, omitting undefined normalized metrics."""
    values = {}
    for prefix, section in (("test/design_sensitivity", report),
                            ("test/design_sensitivity/hic15", report["hic15"])):
        for key in ("design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine",
                    "difference_gain", "same_impact_difference_rmse_g", "same_impact_difference_rmse"):
            value = section.get(key)
            if value is not None:
                values[f"{prefix}/{key}"] = float(value)
    return values


def write_sensitivity_report(output_dir, report, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_design_sensitivity"
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    frame = pd.json_normalize(report["per_location"], sep="_")
    frame.to_csv(output_dir / f"{stem}_by_location.csv", index=False)
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x = frame.location_id.to_numpy()
    for ax, truth, prediction, unit in (
        (axes[0], "true_across_design_rms_spread_g", "predicted_across_design_rms_spread_g", "Acceleration spread (g)"),
        (axes[1], "hic15_true_across_design_rms_spread", "hic15_predicted_across_design_rms_spread", "HIC15 spread"),
    ):
        ax.plot(x, frame[truth], "o-", markersize=3, label="Ground truth", color="#1765a1")
        ax.plot(x, frame[prediction], "o-", markersize=3, label="Prediction", color="#df7126")
        ax.set_ylabel(unit)
        ax.legend()
    axes[2].plot(x, frame.design_difference_skill, "o-", markersize=3, label="Acceleration")
    axes[2].plot(x, frame.hic15_design_difference_skill, "o-", markersize=3, label="HIC15")
    axes[2].axhline(0, color="black", linestyle="--", label="Zero predicted difference")
    axes[2].axhline(1, color="gray", linestyle=":", label="Correct differences")
    axes[2].set(xlabel="Impact location ID", ylabel="Design-difference skill")
    axes[2].legend(ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    def label(value):
        return "undefined" if value is None else f"{value:.4f}"
    fig.suptitle(f"Matched-location design sensitivity | {output_dir.name}\n"
                 f"Acceleration skill {label(report['design_difference_skill'])} | "
                 f"amplitude ratio {label(report['difference_amplitude_ratio'])} | "
                 f"HIC15 skill {label(report['hic15']['design_difference_skill'])}")
    try:
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=180)
    finally:
        plt.close(fig)


def evaluate_saved_run(run_dir, locations=None, history_name="test_acceleration_histories.csv"):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    count = config["data"]["samples_per_design"]
    path = run_dir / history_name
    frame = pd.read_csv(path)
    required = ["run_number", "time", "acceleration_true_g", "acceleration_pred_g"]
    if any(name not in frame for name in required) or frame.empty:
        raise ValueError(f"{path} must contain nonempty saved histories with columns {required}")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(frame.run_number, np.floor(frame.run_number)).all():
        raise ValueError("Saved histories must be finite and run IDs must be integers")
    if locations is not None:
        available = set((frame.run_number.astype(int) - 1) % count + 1)
        if not set(locations) <= available:
            raise ValueError(f"Requested locations absent from {run_dir}: {set(locations) - available}")
        frame = frame[((frame.run_number.astype(int) - 1) % count + 1).isin(locations)]
    frame = frame.sort_values(["run_number", "time"])
    groups = list(frame.groupby("run_number", sort=True))
    data = SimpleNamespace(run_numbers=[int(run) for run, _ in groups],
                           time_arrays=[g.time.to_numpy() for _, g in groups], indentor_positions=None)
    if {"impact_x_mm", "impact_y_mm"} <= set(frame):
        for _, group in groups:
            if group[["impact_x_mm", "impact_y_mm"]].drop_duplicates().shape[0] != 1:
                raise ValueError("Impact coordinates change within a saved history")
        data.indentor_positions = [g[["impact_x_mm", "impact_y_mm"]].iloc[0].to_numpy() for _, g in groups]
    report = design_sensitivity_metrics(data, frame.acceleration_pred_g.to_numpy(),
                                        frame.acceleration_true_g.to_numpy(), count)
    report.update(source_history=history_name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  evaluated_run_numbers=data.run_numbers,
                  location_ids=sorted({(run - 1) % count + 1 for run in data.run_numbers}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--locations", nargs="+", type=int, help="Explicit shared cohort when comparing filtered runs")
    parser.add_argument("--prefix", default="test", help="Artifact prefix; use a different prefix for cohort subsets")
    args = parser.parse_args(argv)
    for directory in args.run_dirs:
        report = evaluate_saved_run(directory, args.locations)
        write_sensitivity_report(directory, report, args.prefix)
        print(directory.name, json.dumps({key: report[key] for key in (
            "num_matched_locations", "design_difference_skill", "difference_amplitude_ratio", "difference_alignment_cosine")}))


if __name__ == "__main__":
    main()
