"""Export HIC15 results for a local-edit training run (train_mesh_impact_history.py).

Reads the run's exported histories (train / validation / test, best checkpoint) and
``local_edit_runs.csv``, recomputes exact HIC15 for every run, and adds:

  * the full-resolution simulation HIC (1000-point SAE1000 history) of each run and of
    design 6 at the same location, to show what the 63-point time grid costs;
  * design-blind baselines: copy design 6 (same location), location mean (mean HIC of
    the training designs at that location, own design left out), and sibling copy
    (training designs with the same edit site and sign, from ``local_edit_runs.csv``);
  * the edit effect on HIC, change = 100 * (HIC / HIC_D6 - 1) at the same location.
    Its R2 is taken about zero, so copying design 6 scores 0, matching
    ``local_edit_metrics.json``.

Outputs (in the run directory):
    hic_results_pred_vs_gt.csv   one row per run, every split
    hic_metrics.json             per split, per class (train) and per role (val/test)
    hic_pred_vs_gt.png           HIC parity, train / validation / test
    hic_change_pred_vs_gt.png    HIC change vs design 6, model and sibling copy
    hic_by_location_<design>.png held-out designs, location by location

Usage:
    python evaluate_local_edits_hic.py runs/mesh_impact_history/local_edits/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hic15 import batched_hic

SPLITS = {"train": "train_acceleration_histories.csv",
          "validation": "validation_acceleration_histories.csv",
          "test": "test_acceleration_histories.csv"}
BANDS = (5.0, 10.0)
BASE = "D6"
# Reference palette slots 1-3 plus a neutral for the design-6 reference.
SIM, MODEL, SIBLING, D6_COLOR = "#2a78d6", "#eb6834", "#1baf7a", "#6f6e69"


# ------------------------------------------------------------------------- data


def resolve_data_dir(run_dir, explicit):
    if explicit is not None:
        return Path(explicit).resolve()
    recorded = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["data"]["acceleration_dir"]
    name = Path(recorded).parent.name
    for candidate in (Path(recorded).parent, Path(__file__).resolve().parent / "Data" / name):
        if (candidate / "output_history_acc").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset {name} not found; pass --data-dir")


def load_histories(run_dir):
    """{sample: (time, truth, prediction)} over every split."""
    curves = {}
    for split, file_name in SPLITS.items():
        frame = pd.read_csv(run_dir / file_name)
        for sample, group in frame.groupby("run_number", sort=True):
            curves[int(sample)] = (group["time"].to_numpy(np.float64),
                                   group["acceleration_true_g"].to_numpy(np.float64),
                                   group["acceleration_pred_g"].to_numpy(np.float64))
    return curves


def load_full_resolution(data_dir, run):
    matches = sorted((data_dir / "output_history_acc").glob(f"HoodImpact_{run}_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one history file for run {run}, found {len(matches)}")
    frame = pd.read_csv(matches[0], usecols=["Time", "A(in g)"])
    return frame["Time"].to_numpy(np.float64), frame["A(in g)"].to_numpy(np.float64)


def hic_full(histories):
    """HIC15 of full-resolution histories, batched over those sharing a time grid."""
    out = np.empty(len(histories))
    groups = {}
    for index, (time, _) in enumerate(histories):
        groups.setdefault((len(time), round(time[1] - time[0], 12), round(time[-1], 9)), []).append(index)
    for indices in groups.values():
        time = histories[indices[0]][0]
        out[indices] = batched_hic(time, np.stack([histories[i][1] for i in indices]))
    return out


# ----------------------------------------------------------------------- metrics


def r2(truth, prediction, about=None):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    centre = truth.mean() if about is None else about
    denominator = float(np.sum((truth - centre) ** 2))
    return float(1.0 - np.sum((prediction - truth) ** 2) / denominator) if denominator else float("nan")


def hic_summary(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    error = prediction - truth
    relative = 100.0 * error / truth
    summary = {"n": int(len(truth)), "r2": r2(truth, prediction),
               "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
               "mape_pct": float(np.mean(np.abs(relative))), "bias": float(error.mean()),
               "max_abs_pct": float(np.abs(relative).max())}
    for band in BANDS:
        summary[f"within_{band:g}_pct"] = float(np.mean(np.abs(relative) <= band) * 100.0)
    return summary


def change_summary(true_change, predicted_change):
    true_change, predicted_change = np.asarray(true_change, float), np.asarray(predicted_change, float)
    large = np.abs(true_change) >= 3.0
    return {"n": int(len(true_change)),
            "r2_about_zero": r2(true_change, predicted_change, about=0.0),
            "mae_pct_points": float(np.mean(np.abs(predicted_change - true_change))),
            "true_change_mean_abs_pct": float(np.mean(np.abs(true_change))),
            "sign_agreement_pct_where_true_change_ge_3pct":
                float(np.mean(np.sign(predicted_change[large]) == np.sign(true_change[large])) * 100.0)
                if large.any() else None,
            "runs_with_true_change_ge_3pct": int(large.sum())}


def summarize(rows):
    sib = rows[rows["hic_sibling"].notna()]
    loc = rows[rows["hic_location_mean"].notna()]
    return {
        "runs": int(len(rows)),
        "designs": sorted(rows["design"].unique(), key=lambda n: int(n[1:])),
        "hic_sampled_grid": {
            "model": hic_summary(rows["hic_true"], rows["hic_pred"]),
            "copy_d6": hic_summary(rows["hic_true"], rows["hic_d6"]),
            "location_mean": hic_summary(loc["hic_true"], loc["hic_location_mean"]) if len(loc) else None,
            "sibling_copy": hic_summary(sib["hic_true"], sib["hic_sibling"]) if len(sib) else None,
            "model_on_sibling_runs": hic_summary(sib["hic_true"], sib["hic_pred"]) if len(sib) else None,
        },
        "hic_full_resolution": {
            "model": hic_summary(rows["hic_true_full"], rows["hic_pred"]),
            "copy_d6": hic_summary(rows["hic_true_full"], rows["hic_d6_full"]),
            "sampled_simulation": hic_summary(rows["hic_true_full"], rows["hic_true"]),
        },
        "hic_change_vs_d6": {
            "model": change_summary(rows["hic_change_true_pct"], rows["hic_change_pred_pct"]),
            "copy_d6": change_summary(rows["hic_change_true_pct"], np.zeros(len(rows))),
            "location_mean": change_summary(loc["hic_change_true_pct"], loc["hic_change_location_mean_pct"])
            if len(loc) else None,
            "sibling_copy": change_summary(sib["hic_change_true_pct"], sib["hic_change_sibling_pct"])
            if len(sib) else None,
            "model_on_sibling_runs": change_summary(sib["hic_change_true_pct"], sib["hic_change_pred_pct"])
            if len(sib) else None,
        },
    }


# ------------------------------------------------------------------------ build


def build_results(run_dir, data_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stride = int(config["preprocessing"]["time_subsample_stride"])
    runs = pd.read_csv(run_dir / "local_edit_runs.csv")
    manifest = pd.read_csv(data_dir / "manifest_local_edits.csv")
    curves = load_histories(run_dir)
    missing = set(runs["sample"]) - set(curves)
    if missing:
        raise ValueError(f"{len(missing)} runs in local_edit_runs.csv have no exported history")

    times = {len(curves[s][0]) for s in runs["sample"]}
    if len(times) != 1:
        raise ValueError("Exported histories are not on one time grid")
    grid = curves[int(runs["sample"].iloc[0])][0]

    # Sampled-grid HIC, recomputed from the exported histories.
    order = runs["sample"].to_numpy()
    recorded = runs[["hic_true", "hic_pred"]].to_numpy()
    runs["hic_true"] = batched_hic(grid, np.stack([curves[s][1] for s in order]))
    runs["hic_pred"] = batched_hic(grid, np.stack([curves[s][2] for s in order]))
    recompute_error = float(np.abs(runs[["hic_true", "hic_pred"]].to_numpy() - recorded).max())

    # Full resolution: every edited run, and design 6 at each location.
    full = [load_full_resolution(data_dir, int(run)) for run in runs["source_run"]]
    sampling_error = max(float(np.abs(f[1][::stride][:len(grid)] - curves[s][1]).max())
                         for f, s in zip(full, order))
    grid_error = max(float(np.abs(f[0][::stride][:len(grid)] - grid).max()) for f in full)
    runs["hic_true_full"] = hic_full(full)

    base = manifest[manifest["design"] == BASE]
    base_full = {int(r.location): load_full_resolution(data_dir, int(r.run)) for r in base.itertuples()}
    locations = sorted(base_full)
    base_full_hic = dict(zip(locations, hic_full([base_full[l] for l in locations])))
    base_sampled_hic = dict(zip(locations, batched_hic(
        grid, np.stack([base_full[l][1][::stride][:len(grid)] for l in locations]))))
    d6_check = float(np.max(np.abs(runs["location"].map(base_sampled_hic) - runs["hic_d6"])))
    runs["hic_d6_full"] = runs["location"].map(base_full_hic)

    # Location mean of the training designs' simulated HIC, own design left out.
    train = runs[runs["split"] == "train"]
    sums = train.groupby("location")["hic_true"].agg(["sum", "count"])
    own = runs["split"].eq("train")
    total = runs["location"].map(sums["sum"]) - np.where(own, runs["hic_true"], 0.0)
    count = runs["location"].map(sums["count"]) - own.astype(int)
    runs["hic_location_mean"] = np.where(count > 0, total / count.where(count > 0), np.nan)

    for key in ("true", "pred", "sibling", "location_mean"):
        runs[f"hic_change_{key}_pct"] = 100.0 * (runs[f"hic_{key}"] / runs["hic_d6"] - 1.0)
    runs["hic_change_true_full_pct"] = 100.0 * (runs["hic_true_full"] / runs["hic_d6_full"] - 1.0)
    runs["hic_error_pct"] = 100.0 * (runs["hic_pred"] / runs["hic_true"] - 1.0)
    runs["hic_error_vs_full_pct"] = 100.0 * (runs["hic_pred"] / runs["hic_true_full"] - 1.0)

    checks = {
        "recomputed_hic_vs_local_edit_runs_max_abs": recompute_error,
        "sampled_truth_equals_full_resolution_every_nth_point_max_abs_g": sampling_error,
        "time_grid_max_abs_diff_s": grid_error,
        "d6_sampled_hic_vs_local_edit_runs_max_abs": d6_check,
        "time_subsample_stride": stride,
        "time_points": int(len(grid)),
        "time_step_ms": float(np.diff(grid).mean() * 1000.0),
    }
    return runs, checks


def build_report(run_dir, runs, checks):
    report = {
        "run_dir": str(run_dir),
        "source": "best-validation checkpoint histories (train/validation/test_acceleration_histories.csv)",
        "hic": "HIC15, exact search over every admissible window (hic15.batched_hic)",
        "hic_change": "100 * (HIC / HIC_D6 - 1), design 6 at the same location; R2 about zero (copy D6 = 0)",
        "baselines": {
            "copy_d6": "design 6's HIC at the same location (design-blind)",
            "location_mean": "mean simulated HIC of the training designs at that location, own design left out",
            "sibling_copy": "HIC of the mean curve of training designs with the same edit site and sign (local_edit_runs.csv)",
        },
        "checks": checks,
        "splits": {},
    }
    for split in SPLITS:
        rows = runs[runs["split"] == split]
        entry = summarize(rows)
        group = "class" if split == "train" else "role"
        entry[f"by_{group}"] = {key: summarize(g) for key, g in rows.groupby(group)}
        if split != "train":
            entry["by_design"] = {key: summarize(g) for key, g in rows.groupby("design")}
        report["splits"][split] = entry
    return report


# ----------------------------------------------------------------------- plots


def _style(ax):
    ax.grid(alpha=0.22, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_parity(runs, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    low = min(runs["hic_true"].min(), runs["hic_pred"].min()) * 0.95
    high = max(runs["hic_true"].max(), runs["hic_pred"].max()) * 1.03
    line = np.array([low, high])
    for ax, split in zip(axes, SPLITS):
        rows = runs[runs["split"] == split]
        stats = report["splits"][split]["hic_sampled_grid"]
        names = ", ".join(report["splits"][split]["designs"]) if split != "train" else "70 designs"
        ax.fill_between(line, line * 0.9, line * 1.1, color=SIM, alpha=0.08, lw=0, label="±10%")
        ax.plot(line, line, color="0.3", lw=1.2, label="1:1")
        size, alpha = (10, 0.45) if split == "train" else (46, 0.95)
        ax.scatter(rows["hic_true"], rows["hic_pred"], s=size, color=MODEL, alpha=alpha,
                   edgecolor="white", linewidth=0.4, label="model", zorder=3)
        ax.set(xlim=line, ylim=line, xlabel="Simulated HIC15", ylabel="Predicted HIC15",
               title=f"{split} ({names}, n = {len(rows)})")
        ax.set_aspect("equal", adjustable="box")
        _style(ax)
        model, copy = stats["model"], stats["copy_d6"]
        text = (f"R² model   {model['r2']:.3f}\n"
                f"R² copy D6 {copy['r2']:.3f}\n"
                f"MAPE model {model['mape_pct']:.1f}%\n"
                f"MAPE copy  {copy['mape_pct']:.1f}%\n"
                f"within 10% {model['within_10_pct']:.0f}%")
        ax.text(0.97, 0.03, text, transform=ax.transAxes, ha="right", va="bottom", family="monospace",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(f"HIC15 from predicted acceleration (63-point grid)  |  {Path(report['run_dir']).name}",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_change(runs, report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    values = runs[["hic_change_true_pct", "hic_change_pred_pct"]].to_numpy()
    limit = float(np.nanmax(np.abs(values))) * 1.05
    line = np.array([-limit, limit])
    for ax, split in zip(axes, SPLITS):
        rows = runs[runs["split"] == split]
        stats = report["splits"][split]["hic_change_vs_d6"]
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvline(0, color="0.75", lw=0.8)
        ax.plot(line, line, color="0.3", lw=1.2, label="1:1")
        sib = rows[rows["hic_sibling"].notna()]
        size, alpha = (10, 0.4) if split == "train" else (46, 0.95)
        ax.scatter(sib["hic_change_true_pct"], sib["hic_change_sibling_pct"], s=size, facecolor="none",
                   edgecolor=SIBLING, linewidth=1.0, alpha=alpha, label="sibling copy", zorder=2)
        ax.scatter(rows["hic_change_true_pct"], rows["hic_change_pred_pct"], s=size, color=MODEL,
                   alpha=alpha, edgecolor="white", linewidth=0.4, label="model", zorder=3)
        ax.set(xlim=line, ylim=line, xlabel="Simulated HIC change vs design 6 (%)",
               ylabel="Predicted HIC change vs design 6 (%)", title=f"{split} (n = {len(rows)})")
        ax.set_aspect("equal", adjustable="box")
        _style(ax)
        model, sibling = stats["model"], stats["sibling_copy"]
        text = (f"R² model    {model['r2_about_zero']:.2f}\n"
                f"R² sibling  {sibling['r2_about_zero']:.2f}  (n={sibling['n']})\n"
                f"MAE model   {model['mae_pct_points']:.1f} pts\n"
                f"MAE sibling {sibling['mae_pct_points']:.1f} pts\n"
                f"MAE copy D6 {stats['copy_d6']['mae_pct_points']:.1f} pts")
        ax.text(0.97, 0.03, text, transform=ax.transAxes, ha="right", va="bottom", family="monospace",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Edit effect on HIC15: 100 (HIC / HIC_D6 - 1) at the same location.  "
                 "Copy design 6 = 0 (R² 0)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_by_location(rows, split, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    role_order = {"near": 0, "twin": 1, "far": 2}
    rows = rows.assign(_r=rows["role"].map(role_order)).sort_values(["_r", "dist_to_edit_mm"])
    x = np.arange(len(rows))
    labels = [f"{l}\n{r} {d:.0f}" for l, r, d in zip(rows["location"], rows["role"], rows["dist_to_edit_mm"])]
    design = rows["design"].iloc[0]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True)
    top.scatter(x, rows["hic_d6"], marker="_", s=260, color=D6_COLOR, linewidth=2.2, label="design 6", zorder=2)
    top.scatter(x, rows["hic_true"], s=60, color=SIM, label="simulation", zorder=4)
    top.scatter(x, rows["hic_pred"], s=60, marker="D", color=MODEL, label="model", zorder=5)
    top.scatter(x, rows["hic_sibling"], s=60, facecolor="none", edgecolor=SIBLING, linewidth=1.4,
                label="sibling copy", zorder=3)
    top.set(ylabel="HIC15", title=f"{design} ({split}): HIC15 by impact location "
                                   "(sorted by role, then distance to the edit)")
    _style(top)
    top.legend(ncol=4, loc="upper left", fontsize=9, frameon=False)
    top.set_ylim(top=top.get_ylim()[1] + 0.12 * np.ptp(top.get_ylim()))
    width = 0.27
    bottom.axhline(0, color="0.4", lw=0.9)
    bottom.axhspan(-1.4, 1.4, color="0.85", alpha=0.6, lw=0, label="D6 mirror-pair floor (±1.4%)")
    bottom.bar(x - width, rows["hic_change_true_pct"], width, color=SIM, label="simulation")
    bottom.bar(x, rows["hic_change_pred_pct"], width, color=MODEL, label="model")
    bottom.bar(x + width, rows["hic_change_sibling_pct"].fillna(0), width, color=SIBLING, label="sibling copy")
    bottom.set(ylabel="HIC change vs design 6 (%)", xticks=x)
    bottom.set_xticklabels(labels, fontsize=8)
    bottom.set_xlabel("impact location / role / distance to edit (mm)")
    _style(bottom)
    bottom.legend(ncol=4, loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------- command


def print_report(report):
    print(f"\n{'split':<11}{'runs':>5} | {'HIC R2':>7} {'copyD6':>7} {'locmean':>8} {'MAPE%':>6} "
          f"{'w/in10%':>8} | {'dHIC R2':>8} {'sibling':>8} {'MAE pts':>8} {'sib pts':>8} {'D6 pts':>7}")
    for split, entry in report["splits"].items():
        h, c = entry["hic_sampled_grid"], entry["hic_change_vs_d6"]
        print(f"{split:<11}{entry['runs']:>5} | {h['model']['r2']:7.3f} {h['copy_d6']['r2']:7.3f} "
              f"{h['location_mean']['r2']:8.3f} {h['model']['mape_pct']:6.2f} {h['model']['within_10_pct']:8.1f} | "
              f"{c['model']['r2_about_zero']:8.3f} {c['sibling_copy']['r2_about_zero']:8.3f} "
              f"{c['model']['mae_pct_points']:8.2f} {c['sibling_copy']['mae_pct_points']:8.2f} "
              f"{c['copy_d6']['mae_pct_points']:7.2f}")
    print("\nChecks:", json.dumps(report["checks"], indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="HoodImpact_LocalEdits directory (auto-detected under Data/)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    data_dir = resolve_data_dir(run_dir, args.data_dir)

    runs, checks = build_results(run_dir, data_dir)
    report = build_report(run_dir, runs, checks)
    columns = ["split", "sample", "source_run", "design", "class", "site", "location", "role", "dist_to_edit_mm",
               "hic_true", "hic_pred", "hic_error_pct", "hic_true_full", "hic_error_vs_full_pct",
               "hic_d6", "hic_d6_full", "hic_sibling", "siblings", "hic_location_mean",
               "hic_change_true_pct", "hic_change_pred_pct", "hic_change_sibling_pct",
               "hic_change_location_mean_pct", "hic_change_true_full_pct",
               "effect_rms_true_g", "error_rms_g", "difference_r2"]
    runs[columns].to_csv(run_dir / "hic_results_pred_vs_gt.csv", index=False)
    (run_dir / "hic_metrics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    plot_parity(runs, report, run_dir / "hic_pred_vs_gt.png")
    plot_change(runs, report, run_dir / "hic_change_pred_vs_gt.png")
    for split in ("validation", "test"):
        for design, rows in runs[runs["split"] == split].groupby("design"):
            plot_by_location(rows, split, run_dir / f"hic_by_location_{design}.png")
    print_report(report)
    print(f"\nWrote HIC exports to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
