"""Near-clone design-difference R2 of a mesh-history run, train and test separately.

Take two near-clone designs (same entry of ``mesh_design_clusters.GEOMETRY_CLUSTERS``)
at the same impact location and subtract their acceleration curves, once in the
simulation (d) and once in the prediction (p). Score with ordinary R2,

    r2 = 1 - sum((p - d)^2) / sum((d - mean d)^2)

pooled over sampled times, locations and the near-clone pairs inside one split.
0 means the model predicts the same curve for both designs; 1 is perfect.

Training writes ``design_difference_r2.json`` for every run. This command
backfills it from a run's saved ``train_acceleration_histories.csv`` and
``test_acceleration_histories.csv`` without loading the model.

Usage:
    python evaluate_design_difference_r2.py runs/mesh_impact_history/<run> [<run> ...]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mesh_change_metrics import near_clone_difference_r2
from mesh_design_clusters import GEOMETRY_CLUSTERS

REPORT_NAME = "design_difference_r2.json"
SPLITS = ("train", "test")
DEFINITION = ("d(t) = a_b(t) - a_a(t) for near-clone designs a < b at the same impact location, p(t) the same "
              "from predictions; r2 = 1 - sum((p - d)^2) / sum((d - mean d)^2) pooled over sampled times, "
              "locations and pairs of the split. 0 = same predicted curve for both designs, 1 = perfect.")


def design_difference_report(data_format, samples_per_design, splits):
    """``splits`` maps a split name to (dataset, predictions, targets) in dataset order."""
    clusters = GEOMETRY_CLUSTERS if data_format == "euroncap1704" else {}
    report = {"metric": "R2 of near-clone acceleration difference curves", "definition": DEFINITION,
              "clusters": {name: list(members) for name, members in clusters.items()}}
    if not clusters:
        report["note"] = f"No near-clone cluster map for data format {data_format!r}; nothing to score"
    for name, (dataset, predictions, targets) in splits.items():
        report[name] = near_clone_difference_r2(dataset, predictions, targets, samples_per_design, clusters)
    return report


def difference_log_values(report):
    return {f"{name}/design_difference_r2": report[name]["r2"]
            for name in SPLITS if report.get(name, {}).get("r2") is not None}


def write_design_difference_report(output_dir, report):
    path = Path(output_dir) / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def saved_split(path):
    """Histories exported by training, as (dataset, predictions, targets)."""
    frame = pd.read_csv(path).sort_values(["run_number", "time"], kind="stable")
    groups = list(frame.groupby("run_number", sort=True))
    dataset = SimpleNamespace(run_numbers=[int(run) for run, _ in groups],
                              time_arrays=[group.time.to_numpy() for _, group in groups])
    return dataset, frame.acceleration_pred_g.to_numpy(), frame.acceleration_true_g.to_numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["data"]
        splits, missing = {}, []
        for name in SPLITS:
            path = run_dir / f"{name}_acceleration_histories.csv"
            if path.is_file():
                splits[name] = saved_split(path)
            else:
                missing.append(path.name)
        report = design_difference_report(data["data_format"], data["samples_per_design"], splits)
        if missing:
            report["missing_histories"] = missing
        path = write_design_difference_report(run_dir, report)
        scores = {name: report[name]["r2"] for name in SPLITS if name in report}
        print(f"{run_dir.name}: {scores}" + (f" (missing {', '.join(missing)})" if missing else "") + f" -> {path}")


if __name__ == "__main__":
    main()
