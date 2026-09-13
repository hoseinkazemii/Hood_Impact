"""Run a trained MeshImpactHistoryNet over every impact location of several designs.

The held-out design 2 has three geometric near-twins in the training set
(designs 0, 1 and 3, within 0.024 mm mean surface distance).  This command
predicts all 142 impact locations for each requested design with one saved
checkpoint, exports the histories next to their simulated ground truth, and
compares the designs against the held-out one.

Two questions it answers directly:

  * how much does the *simulation* change between these designs, at a fixed
    impact location (the across-design spread of the ground truth); and
  * how much does the *model* change (the across-design spread of the
    prediction).  A model that reads geometry should reproduce the first
    spread; a model that has learned only the impact location will not.

Designs the checkpoint trained on are labelled ``train`` in every output, so
their errors are never mistaken for generalization.

Usage:
    python infer_designs_mesh_impact_history.py runs/mesh_impact_history/<run>
    python infer_designs_mesh_impact_history.py runs/mesh_impact_history/<run> \
        --designs 0 1 2 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from evaluate_mesh_impact_hic import hic15, r2_score, resolve_data_dir
from train_mesh_impact_history import HistoryPredictor
from utils.utils import Config, DataPreprocessor

_COLORS = ("#1965b0", "#dc7014", "#2a7f62", "#8c4a9c", "#b03060", "#666666")


def load_splits(run_dir):
    """Which design IDs the checkpoint trained, validated and tested on."""
    path = Path(run_dir) / "splits.json"
    if not path.is_file():
        return {}
    splits = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(design): name
        for name, group in splits.items()
        for design in group.get("design_ids", [])
    }


def build_source_preprocessor(run_dir, data_dir, output_dir):
    """A preprocessor that loads meshes and histories exactly as training did."""
    saved = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    data = saved["data"]
    preprocessing = saved["preprocessing"]
    config = Config(
        data_format=data["data_format"],
        samples_per_design=data["samples_per_design"],
        num_samples=data["num_samples"],
        inp_dir=str(Path(data_dir) / "inp_files"),
        impact_coords_path=str(Path(data_dir) / Path(data["impact_coords_path"]).name),
        acceleration_dir=str(Path(data_dir) / "output_history_acc"),
        prediction_target="acceleration",
        time_subsample_stride=preprocessing["time_subsample_stride"],
        max_train_time=preprocessing["max_train_time"],
        output_dir=str(output_dir),
    )
    return DataPreprocessor(config), config


def infer_designs(run_dir, data_dir, designs, device, output_dir, progress_every=25):
    """Predict every impact location of each design and pair it with the truth."""
    run_dir = Path(run_dir).resolve()
    predictor = HistoryPredictor.from_run(run_dir, device=device)
    source, config = build_source_preprocessor(run_dir, data_dir, output_dir)
    impact_coords = source.load_impact_coords()
    split_of = load_splits(run_dir)
    samples_per_design = config.samples_per_design

    histories, records = [], []
    started = time.perf_counter()
    total = len(designs) * samples_per_design
    for design in designs:
        for location in range(1, samples_per_design + 1):
            run_number = design * samples_per_design + location
            mesh = source.load_mesh_geometry(run_number)
            impact = impact_coords.iloc[run_number - 1][["X1", "X2"]].to_numpy(np.float32)
            times, truth = source.load_acceleration_history(run_number)
            if not np.allclose(times, predictor.time_points, atol=1e-9):
                raise ValueError(
                    f"Run {run_number}: sampled time grid differs from the checkpoint grid"
                )
            prediction = predictor.predict(mesh, impact).astype(np.float64)
            truth = truth.astype(np.float64)
            residual = prediction - truth
            hic_truth = hic15(times.astype(np.float64), truth)[0]
            hic_prediction = hic15(times.astype(np.float64), prediction)[0]
            histories.append(pd.DataFrame({
                "run_number": run_number,
                "design_id": design,
                "location_id": location,
                "split": split_of.get(design, "unknown"),
                "time": times.astype(np.float64),
                "acceleration_true_g": truth,
                "acceleration_pred_g": prediction,
            }))
            records.append({
                "run_number": run_number,
                "design_id": design,
                "location_id": location,
                "split": split_of.get(design, "unknown"),
                "impact_x_mm": float(impact[0]),
                "impact_y_mm": float(impact[1]),
                "num_mesh_nodes": int(len(mesh)),
                "acceleration_rmse_g": float(np.sqrt(np.mean(residual**2))),
                "acceleration_mae_g": float(np.mean(np.abs(residual))),
                "acceleration_r2": r2_score(truth, prediction),
                "hic_ground_truth": hic_truth,
                "hic_prediction": hic_prediction,
                "hic_error": hic_prediction - hic_truth,
                "hic_relative_error_pct": 100.0 * (hic_prediction - hic_truth) / hic_truth,
                "true_peak_g": float(truth.max()),
                "predicted_peak_g": float(prediction.max()),
            })
            done = len(records)
            if done == 1 or done % progress_every == 0 or done == total:
                rate = done / (time.perf_counter() - started)
                print(f"  inferred {done}/{total} runs ({rate:.1f} runs/s)", flush=True)

    return pd.concat(histories, ignore_index=True), pd.DataFrame(records)


def summarize_designs(metrics, histories):
    """Pooled accuracy per design, keeping the train/test label attached."""
    rows = []
    for design, group in metrics.groupby("design_id"):
        runs = histories[histories["design_id"] == design]
        truth = runs["acceleration_true_g"].to_numpy(np.float64)
        prediction = runs["acceleration_pred_g"].to_numpy(np.float64)
        hic_truth = group["hic_ground_truth"].to_numpy(np.float64)
        hic_prediction = group["hic_prediction"].to_numpy(np.float64)
        rows.append({
            "design_id": int(design),
            "split": group["split"].iloc[0],
            "num_locations": int(len(group)),
            "acceleration_r2": r2_score(truth, prediction),
            "acceleration_rmse_g": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "acceleration_mae_g": float(np.mean(np.abs(prediction - truth))),
            "hic_r2": r2_score(hic_truth, hic_prediction),
            "hic_mape_percent": float(np.mean(np.abs(group["hic_relative_error_pct"]))),
            "hic_bias": float(np.mean(group["hic_error"])),
            "mean_true_hic": float(hic_truth.mean()),
            "mean_predicted_hic": float(hic_prediction.mean()),
        })
    return pd.DataFrame(rows).sort_values("design_id").reset_index(drop=True)


def across_design_spread(histories, reference_design):
    """Per-location spread of truth and prediction across the compared designs.

    Both are measured the same way: the RMS difference of each design from the
    reference design, at matching impact location and time sample.  If the
    simulations differ but the predictions do not, the model is not reading
    geometry.
    """
    pivot = histories.pivot_table(
        index=["location_id", "time"],
        columns="design_id",
        values=["acceleration_true_g", "acceleration_pred_g"],
    )
    designs = sorted({design for _, design in pivot.columns})
    if reference_design not in designs:
        raise ValueError(f"Reference design {reference_design} is not among {designs}")
    rows = []
    for design in designs:
        if design == reference_design:
            continue
        truth_difference = (
            pivot[("acceleration_true_g", design)] - pivot[("acceleration_true_g", reference_design)]
        ).to_numpy(np.float64)
        prediction_difference = (
            pivot[("acceleration_pred_g", design)] - pivot[("acceleration_pred_g", reference_design)]
        ).to_numpy(np.float64)
        rows.append({
            "design_id": int(design),
            "reference_design": int(reference_design),
            "truth_rms_difference_g": float(np.sqrt(np.mean(truth_difference**2))),
            "prediction_rms_difference_g": float(np.sqrt(np.mean(prediction_difference**2))),
            "truth_max_difference_g": float(np.max(np.abs(truth_difference))),
            "prediction_max_difference_g": float(np.max(np.abs(prediction_difference))),
        })
    spread = pd.DataFrame(rows)
    spread["prediction_over_truth_ratio"] = (
        spread["prediction_rms_difference_g"] / spread["truth_rms_difference_g"]
    )
    return spread


def hic_spread(metrics, reference_design):
    """Same comparison on HIC: how much the designs differ, truth vs prediction."""
    truth = metrics.pivot(index="location_id", columns="design_id", values="hic_ground_truth")
    prediction = metrics.pivot(index="location_id", columns="design_id", values="hic_prediction")
    rows = []
    for design in truth.columns:
        if design == reference_design:
            continue
        rows.append({
            "design_id": int(design),
            "reference_design": int(reference_design),
            "truth_rms_difference": float(
                np.sqrt(np.mean((truth[design] - truth[reference_design]) ** 2))
            ),
            "prediction_rms_difference": float(
                np.sqrt(np.mean((prediction[design] - prediction[reference_design]) ** 2))
            ),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- plotting


def _design_label(design, split):
    return f"Design {design} ({split})"


def export_location_overlays(histories, metrics, path, locations):
    """One panel per location: every design's simulation and prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = sorted(histories["design_id"].unique())
    split_of = metrics.drop_duplicates("design_id").set_index("design_id")["split"]
    columns = 2
    rows = int(np.ceil(len(locations) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.4 * columns, 4.5 * rows), squeeze=False)
    try:
        for axis, location in zip(axes.flat, locations):
            for index, design in enumerate(designs):
                group = histories[
                    (histories["design_id"] == design) & (histories["location_id"] == location)
                ]
                time_ms = group["time"].to_numpy() * 1000.0
                color = _COLORS[index % len(_COLORS)]
                axis.plot(time_ms, group["acceleration_true_g"], color=color, lw=1.9,
                          label=f"{_design_label(design, split_of[design])} — simulation")
                axis.plot(time_ms, group["acceleration_pred_g"], color=color, lw=1.5, ls="--",
                          alpha=0.85, label=f"{_design_label(design, split_of[design])} — prediction")
            axis.set(xlabel="Time (ms)", ylabel="Acceleration (g)",
                     title=f"Impact location {location}")
            axis.grid(alpha=0.22, lw=0.6)
        for axis in list(axes.flat)[len(locations):]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=7.5, ncol=2, framealpha=0.92, loc="upper right")
        fig.suptitle(
            "Simulation (solid) and prediction (dashed) for near-identical designs, "
            "one saved checkpoint",
            fontsize=13, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(path, dpi=190)
    finally:
        plt.close(fig)
    return path


def export_spread_plot(history_spread, hic_difference, path, reference_design):
    """Does the model separate these designs as much as the simulation does?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (history_axis, hic_axis) = plt.subplots(1, 2, figsize=(13.0, 5.4))
    try:
        positions = np.arange(len(history_spread))
        width = 0.38
        history_axis.bar(positions - width / 2, history_spread["truth_rms_difference_g"],
                         width, color="#1965b0", label="Simulation")
        history_axis.bar(positions + width / 2, history_spread["prediction_rms_difference_g"],
                         width, color="#dc7014", label="Model prediction")
        history_axis.set_xticks(positions)
        history_axis.set_xticklabels([f"design {int(d)}" for d in history_spread["design_id"]])
        history_axis.set(ylabel="RMS difference from the reference design (g)",
                         title=f"Acceleration: separation from design {reference_design}")
        history_axis.legend(frameon=False)
        history_axis.grid(alpha=0.22, lw=0.6, axis="y")
        for position, ratio in zip(positions, history_spread["prediction_over_truth_ratio"]):
            history_axis.text(position, 0, f"  ratio {ratio:.2f}", ha="center", va="bottom",
                              fontsize=9, color="#333333")

        positions = np.arange(len(hic_difference))
        hic_axis.bar(positions - width / 2, hic_difference["truth_rms_difference"],
                     width, color="#1965b0", label="Simulation")
        hic_axis.bar(positions + width / 2, hic_difference["prediction_rms_difference"],
                     width, color="#dc7014", label="Model prediction")
        hic_axis.set_xticks(positions)
        hic_axis.set_xticklabels([f"design {int(d)}" for d in hic_difference["design_id"]])
        hic_axis.set(ylabel="RMS HIC15 difference from the reference design",
                     title=f"HIC15: separation from design {reference_design}")
        hic_axis.legend(frameon=False)
        hic_axis.grid(alpha=0.22, lw=0.6, axis="y")

        fig.suptitle(
            "How far each design sits from the held-out design, in the simulation and in the model",
            fontsize=12.5, y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        fig.savefig(path, dpi=220)
    finally:
        plt.close(fig)
    return path


def export_design_metric_plot(summary, path, reference_design):
    """Per-design accuracy, with the held-out design marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    panels = (
        ("acceleration_r2", "Acceleration R²", None),
        ("hic_r2", "HIC15 R²", None),
        ("hic_mape_percent", "HIC15 MAPE (%)", None),
    )
    try:
        for axis, (column, title, _) in zip(axes, panels):
            colors = [
                "#c8553d" if int(design) == reference_design else "#2a7f62"
                for design in summary["design_id"]
            ]
            axis.bar(np.arange(len(summary)), summary[column], color=colors, width=0.62)
            axis.set_xticks(np.arange(len(summary)))
            axis.set_xticklabels(
                [f"{int(row.design_id)}\n({row.split})" for row in summary.itertuples()],
                fontsize=9,
            )
            axis.set(title=title, xlabel="Design")
            axis.grid(alpha=0.22, lw=0.6, axis="y")
            for position, value in enumerate(summary[column]):
                axis.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(
            f"One checkpoint across near-identical designs — red is the held-out design "
            f"{reference_design}",
            fontsize=12.5, y=0.985,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(path, dpi=220)
    finally:
        plt.close(fig)
    return path


def export_design_galleries(run_dir, output_dir, histories, metrics, designs):
    """One per-impact plot, PDF and HTML gallery per design, as for the test set.

    Each design gets its own ``design_<id>_acceleration_plots/`` directory so a
    training design's report is never mistaken for the held-out one.
    """
    from visualize_mesh_impact_test_set import export_all_test_plots

    exports = {}
    for design in designs:
        design_histories = histories[histories["design_id"] == design]
        design_metrics = metrics[metrics["design_id"] == design]
        if design_metrics.empty:
            raise ValueError(f"No inferred runs for design {design}")
        split = str(design_metrics["split"].iloc[0])
        print(f"\nExporting design {design} ({split}): {len(design_metrics)} impacts")
        exports[int(design)] = export_all_test_plots(
            run_dir,
            design_metrics,
            output_dir=output_dir,
            histories=design_histories,
            slug=f"design_{design}",
            heading=f"Design {design} ({split}) — all acceleration predictions",
            noun=f"design {design} impacts",
        )
    return exports


# ------------------------------------------------------------------------ command


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--designs", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Zero-based design IDs to run (default: 0 1 2 3)")
    parser.add_argument("--reference-design", type=int, default=2,
                        help="Design the others are compared against (default: 2, the held-out one)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <run_dir>/design_comparison")
    parser.add_argument("--overlay-locations", type=int, nargs="+", default=None,
                        help="Impact locations to overlay (default: six spread over the HIC range)")
    parser.add_argument("--reuse-saved", action="store_true",
                        help="Skip inference and reuse the CSVs already in the output directory")
    parser.add_argument("--plot-designs", type=int, nargs="+", default=None,
                        help="Designs to give a per-impact plot gallery (default: every "
                             "inferred design except the reference, whose gallery the "
                             "test report already holds)")
    parser.add_argument("--no-history-plots", action="store_true",
                        help="Skip the per-impact plot galleries")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    data_dir = resolve_data_dir(run_dir, args.data_dir)
    if data_dir is None:
        raise SystemExit("The dataset directory was not found; pass --data-dir")
    output_dir = (args.output_dir or run_dir / "design_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {run_dir}\nDataset:    {data_dir}\nDevice:     {args.device}")
    print(f"Designs:    {args.designs} (reference: {args.reference_design})")

    histories_path = output_dir / "design_acceleration_histories.csv"
    metrics_path = output_dir / "design_per_history_metrics.csv"
    if args.reuse_saved:
        if not (histories_path.is_file() and metrics_path.is_file()):
            raise SystemExit(f"--reuse-saved needs {histories_path.name} and {metrics_path.name}")
        print(f"Reusing saved inference results in {output_dir}")
        histories = pd.read_csv(histories_path)
        metrics = pd.read_csv(metrics_path)
        missing = sorted(set(args.designs) - set(metrics["design_id"]))
        if missing:
            raise SystemExit(f"The saved results do not cover designs {missing}; rerun without --reuse-saved")
        histories = histories[histories["design_id"].isin(args.designs)]
        metrics = metrics[metrics["design_id"].isin(args.designs)]
    else:
        histories, metrics = infer_designs(
            run_dir, data_dir, args.designs, args.device, output_dir
        )
        histories.to_csv(histories_path, index=False)
        metrics.to_csv(metrics_path, index=False)

    summary = summarize_designs(metrics, histories)
    history_spread = across_design_spread(histories, args.reference_design)
    hic_difference = hic_spread(metrics, args.reference_design)
    report = {
        "run_dir": str(run_dir),
        "dataset_dir": str(data_dir),
        "designs": list(args.designs),
        "reference_design": args.reference_design,
        "locations_per_design": int(metrics.groupby("design_id").size().iloc[0]),
        "per_design": summary.to_dict(orient="records"),
        "across_design_separation_acceleration": history_spread.to_dict(orient="records"),
        "across_design_separation_hic": hic_difference.to_dict(orient="records"),
    }
    report_path = output_dir / "design_comparison_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    reference = metrics[metrics["design_id"] == args.reference_design].sort_values(
        "hic_ground_truth"
    )
    locations = args.overlay_locations or [
        int(reference["location_id"].iloc[index])
        for index in np.linspace(0, len(reference) - 1, 6).astype(int)
    ]
    overlay_path = export_location_overlays(
        histories, metrics, output_dir / "design_history_overlays.png", locations
    )
    spread_path = export_spread_plot(
        history_spread, hic_difference, output_dir / "design_separation.png",
        args.reference_design,
    )
    metric_path = export_design_metric_plot(
        summary, output_dir / "design_metric_comparison.png", args.reference_design
    )

    print("\nPer-design accuracy from the single held-out-design-2 checkpoint:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:9.4f}"))
    print(f"\nSeparation from design {args.reference_design} (acceleration, g RMS):")
    print(history_spread.to_string(index=False, float_format=lambda value: f"{value:9.4f}"))
    print(f"\nSeparation from design {args.reference_design} (HIC15 RMS):")
    print(hic_difference.to_string(index=False, float_format=lambda value: f"{value:9.3f}"))
    for path in (histories_path, metrics_path, report_path, overlay_path, spread_path, metric_path):
        print(f"Wrote: {path}")

    if not args.no_history_plots:
        plot_designs = args.plot_designs
        if plot_designs is None:
            plot_designs = [d for d in args.designs if d != args.reference_design]
        exports = export_design_galleries(run_dir, output_dir, histories, metrics, plot_designs)
        for design, export in exports.items():
            print(f"Design {design}: {export['png_count']} plots in {export['png_directory']}")
            print(f"           {export['pdf_path']} ({export['pdf_page_count']} pages)")
            print(f"           {export['gallery_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
