"""Training-fit and test report for single-location, geometry-only runs.

Reads a run directory written by ``train_mesh_impact_history.py --locations``
and, for each location, compares predicted and true acceleration histories of
every design:

  * ``location_<L>_all_designs.png/pdf``  one panel per design: ground truth
    (full 1000-point history) against the prediction, split in the title;
  * ``location_<L>_designs_4_5.png/pdf``  the ground-truth | prediction layout
    of the top-10 HIC-variation figures, for designs 4 and 5;
  * ``summary.csv``  per design: split, cluster, HIC15 and initial peak
    (0-3 ms) true vs predicted, RMSE and R2 on the prediction grid;
  * ``report.json``  per location: how much of the design-to-design variation
    the training fit captures, within-cluster difference R2, and every test
    design against two references that use no model -- the mean of its
    geometry-cluster siblings in training, and the mean of all training designs.

HIC15 is compared on the saved prediction grid for both curves (exact search,
``hic15.batched_hic``); the full-resolution ground-truth HIC15 is listed too.

    python report_one_impact_location.py runs/mesh_impact_history/<run>
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hic15 import batched_hic
from mesh_design_clusters import GEOMETRY_CLUSTERS

SAMPLES_PER_DESIGN = 142
INITIAL_PEAK_WINDOW_S = 0.003
CLUSTER_OF = {design: name for name, members in GEOMETRY_CLUSTERS.items() for design in members}
TRUTH, PREDICTION, INK_2, TEST_PANEL = "#0b0b0b", "#2a78d6", "#52514e", "#f1f0ea"
PAIR_DESIGNS = (4, 5)
PAIR_COLORS = ("#1f77b4", "#ff7f0e")  # design 4 / design 5 in the top-10 HIC-variation figures


def r2(truth, prediction):
    residual = np.sum((truth - prediction) ** 2)
    total = np.sum((truth - truth.mean()) ** 2)
    return float(1 - residual / total) if total > 0 else None


def curve_metrics(times, truth, prediction):
    hic = batched_hic(times, np.stack((truth, prediction)))
    early = times <= INITIAL_PEAK_WINDOW_S
    return {
        "rmse_g": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "r2": r2(truth, prediction),
        "hic15_true": float(hic[0]),
        "hic15_pred": float(hic[1]),
        "hic15_error_percent": float(100 * (hic[1] - hic[0]) / hic[0]),
        "initial_peak_true_g": float(truth[early].max()),
        "initial_peak_pred_g": float(prediction[early].max()),
    }


def load_curves(run_dir):
    """{location: {design: (split, times, truth, prediction)}} from the exported CSVs."""
    curves = {}
    for split in ("train", "test"):
        path = run_dir / f"{split}_acceleration_histories.csv"
        if not path.is_file():
            continue
        for run, rows in pd.read_csv(path).groupby("run_number"):
            rows = rows.sort_values("time")
            design, location = divmod(int(run) - 1, SAMPLES_PER_DESIGN)
            curves.setdefault(location + 1, {})[design] = (
                split, rows.time.to_numpy(), rows.acceleration_true_g.to_numpy(),
                rows.acceleration_pred_g.to_numpy())
    if not curves:
        raise FileNotFoundError(f"No train/test acceleration histories in {run_dir}")
    return curves


def full_truth(acceleration_dir, run):
    """Full-resolution history (s, g), or None when the source file is not here."""
    path = Path(acceleration_dir) / f"HoodImpact_{run}_SAE1000_interp1000.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, usecols=["Time", "A(in g)"])
    return frame.Time.to_numpy(), frame["A(in g)"].to_numpy()


def reference_metrics(times, truth, reference):
    return None if reference is None else curve_metrics(times, truth, reference)


def location_report(location, designs, acceleration_dir):
    """Per-design rows and the location's summary dictionary."""
    rows, full = [], {}
    for design, (split, times, truth, prediction) in sorted(designs.items()):
        run = SAMPLES_PER_DESIGN * design + location
        full[design] = full_truth(acceleration_dir, run)
        full_hic = None if full[design] is None else float(batched_hic(*full[design]))
        rows.append({"location": location, "design": design, "run_number": run, "split": split,
                     "cluster": CLUSTER_OF.get(design), **curve_metrics(times, truth, prediction),
                     "hic15_true_full_resolution": full_hic})
    train = {d: v for d, v in designs.items() if v[0] == "train"}
    summary = {"location": location, "train_designs": sorted(train),
               "test_designs": sorted(d for d, v in designs.items() if v[0] == "test")}
    if train:
        truth = np.stack([v[2] for v in train.values()])
        prediction = np.stack([v[3] for v in train.values()])
        collapsed = truth.mean(axis=0)
        # Skill against predicting the training mean for every design: 1 means
        # every design's own curve, 0 means no design-to-design variation at all.
        summary["train_r2"] = r2(truth.ravel(), prediction.ravel())
        summary["train_variation_skill"] = float(
            1 - np.sum((prediction - truth) ** 2) / np.sum((collapsed - truth) ** 2))
        summary["within_cluster_difference_r2"] = {
            f"{a}-{b}": r2(train[a][2] - train[b][2], train[a][3] - train[b][3])
            for a, b in combinations(sorted(train), 2) if CLUSTER_OF.get(a) == CLUSTER_OF.get(b)}
    summary["test"] = {}
    for design in summary["test_designs"]:
        _, times, truth, prediction = designs[design]
        siblings = [d for d in train if CLUSTER_OF.get(d) == CLUSTER_OF.get(design)]
        summary["test"][str(design)] = {
            "model": curve_metrics(times, truth, prediction),
            "cluster_sibling_mean": reference_metrics(
                times, truth, np.mean([train[d][2] for d in siblings], axis=0) if siblings else None),
            "cluster_siblings_in_training": siblings,
            "training_mean": reference_metrics(
                times, truth, np.mean([v[2] for v in train.values()], axis=0) if train else None),
        }
    return rows, summary, full


def style(ax):
    ax.grid(alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_all_designs(location, designs, rows, full, run_name, out_dir):
    by_design = {row["design"]: row for row in rows}
    count = len(designs)
    columns = min(4, count)
    lines = -(-count // columns)
    fig, axes = plt.subplots(lines, columns, figsize=(3.6 * columns, 2.9 * lines + 1.1),
                             sharex=True, sharey=True, squeeze=False)
    peak = 0.0
    for ax, (design, (split, times, truth, prediction)) in zip(axes.flat, sorted(designs.items())):
        reference = full[design] or (times, truth)
        ax.plot(reference[0] * 1e3, reference[1], color=TRUTH, lw=1.1)
        ax.plot(times * 1e3, prediction, color=PREDICTION, lw=1.8)
        peak = max(peak, reference[1].max(), prediction.max())
        row = by_design[design]
        label = "TEST" if split == "test" else "train"
        ax.set_title(f"Design {design} ({row['cluster']}) | {label}\n"
                     f"HIC15 {row['hic15_true']:.0f} true, {row['hic15_pred']:.0f} predicted",
                     fontsize=9.5, color=TRUTH, fontweight="bold" if split == "test" else "normal")
        if split == "test":
            ax.set_facecolor(TEST_PANEL)
        style(ax)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    axes[0, 0].set_ylim(0, np.ceil(peak / 25) * 25)
    axes[0, 0].set_xlim(0, 25)
    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Acceleration (g)")
    fig.legend(handles=[Line2D([], [], color=TRUTH, lw=1.1, label="Ground truth (1,000 points)"),
                        Line2D([], [], color=PREDICTION, lw=1.8, label="Prediction")],
               loc="upper right", frameon=False, ncol=2)
    fig.suptitle(f"Impact location {location}: every design, ground truth vs prediction",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, f"Model: {run_name} | HIC15 on the prediction grid", ha="center",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return save(fig, out_dir / f"location_{location}_all_designs")


def plot_design_pair(location, designs, full, run_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    peak = 0.0
    for index, design in enumerate(PAIR_DESIGNS):
        split, times, truth, prediction = designs[design]
        reference = full[design] or (times, truth)
        line = dict(color=PAIR_COLORS[index], linestyle=["-", "--"][index], lw=2,
                    label=f"Design {design} ({split})")
        axes[0].plot(reference[0] * 1e3, reference[1], **line)
        axes[1].plot(times * 1e3, prediction, **line)
        peak = max(peak, reference[1].max(), prediction.max())
    for ax, title in zip(axes, ("Ground truth", "Prediction")):
        ax.set_title(title, fontsize=16, pad=12)
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, np.ceil(peak / 25) * 25)
        ax.legend(frameon=False, loc="upper right")
        style(ax)
    axes[0].set_ylabel("Resultant acceleration (g)")
    fig.suptitle(f"Impact location {location} | designs 4 and 5, geometry-only model",
                 fontsize=19, fontweight="bold", y=0.96)
    fig.text(0.5, 0.04, f"GT: 1,000 points | Prediction: {len(designs[PAIR_DESIGNS[0]][1])} saved points | "
             f"Model: {run_name}", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.15, wspace=0.08)
    return save(fig, out_dir / f"location_{location}_designs_4_5")


def save(fig, stem):
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=200, facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def write_report(run_dir, acceleration_dir=None):
    """Write figures, summary.csv and report.json; return the report directory."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    acceleration_dir = acceleration_dir or config["data"]["acceleration_dir"]
    out_dir = run_dir / "location_report"
    out_dir.mkdir(exist_ok=True)
    all_rows, summaries = [], []
    for location, designs in sorted(load_curves(run_dir).items()):
        rows, summary, full = location_report(location, designs, acceleration_dir)
        summary["figures"] = [plot_all_designs(location, designs, rows, full, run_dir.name, out_dir).name]
        if all(design in designs for design in PAIR_DESIGNS):
            summary["figures"].append(plot_design_pair(location, designs, full, run_dir.name, out_dir).name)
        all_rows += rows
        summaries.append(summary)
    pd.DataFrame(all_rows).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps({"run": run_dir.name, "locations": summaries}, indent=2)
                                         + "\n", encoding="utf-8")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--acceleration-dir", type=Path,
                        help="Full-resolution histories; defaults to the run's saved acceleration_dir.")
    args = parser.parse_args(argv)
    print(f"Saved {write_report(args.run_dir, args.acceleration_dir)}")


if __name__ == "__main__":
    main()
