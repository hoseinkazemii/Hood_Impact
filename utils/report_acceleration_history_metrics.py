import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange


SAMPLE_RE = re.compile(r"test_sample_(\d+)\.csv$")
REQUIRED_COLUMNS = {"time", "ground_truth", "prediction"}


@njit(parallel=True, fastmath=True)
def compute_hic_batch(times, targets, predictions, max_window):
    n_samples, n_steps = targets.shape
    target_hic = np.zeros(n_samples, dtype=np.float64)
    prediction_hic = np.zeros(n_samples, dtype=np.float64)

    for sample_idx in prange(n_samples):
        target_integral = np.zeros(n_steps, dtype=np.float64)
        prediction_integral = np.zeros(n_steps, dtype=np.float64)

        for i in range(1, n_steps):
            dt = times[sample_idx, i] - times[sample_idx, i - 1]
            target_integral[i] = (
                target_integral[i - 1]
                + 0.5
                * (targets[sample_idx, i] + targets[sample_idx, i - 1])
                * dt
            )
            prediction_integral[i] = (
                prediction_integral[i - 1]
                + 0.5
                * (predictions[sample_idx, i] + predictions[sample_idx, i - 1])
                * dt
            )

        best_target = 0.0
        best_prediction = 0.0

        for i in range(n_steps):
            for j in range(i + 1, n_steps):
                window = times[sample_idx, j] - times[sample_idx, i]
                if window > max_window:
                    break
                if window <= 0.0:
                    continue

                target_avg = (target_integral[j] - target_integral[i]) / window
                target_value = window * (target_avg ** 2.5)
                if target_value > best_target:
                    best_target = target_value

                prediction_avg = (
                    prediction_integral[j] - prediction_integral[i]
                ) / window
                prediction_value = window * (prediction_avg ** 2.5)
                if prediction_value > best_prediction:
                    best_prediction = prediction_value

        target_hic[sample_idx] = best_target
        prediction_hic[sample_idx] = best_prediction

    return target_hic, prediction_hic


def sample_id_from_path(path):
    match = SAMPLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse sample id from {path}")
    return int(match.group(1))


def find_sample_files(root):
    files_by_id = {}
    for path in Path(root).rglob("test_sample_*.csv"):
        sample_id = sample_id_from_path(path)
        if sample_id in files_by_id:
            raise ValueError(
                "Duplicate sample id "
                f"{sample_id}: {files_by_id[sample_id]} and {path}"
            )
        files_by_id[sample_id] = path

    if not files_by_id:
        raise FileNotFoundError(f"No test_sample_*.csv files found under {root}")

    return [files_by_id[k] for k in sorted(files_by_id)]


def r2_score_from_arrays(targets, predictions):
    residual = targets - predictions
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    if denominator == 0:
        return np.nan
    return 1.0 - np.sum(residual**2) / denominator


def parse_design_ids(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_records(sample_files, samples_per_design, design_ids, max_window):
    records = []
    hic_groups = {}

    for sample_index, path in enumerate(sample_files):
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        time = df["time"].to_numpy(np.float64)
        target = df["ground_truth"].to_numpy(np.float64)
        prediction = df["prediction"].to_numpy(np.float64)
        error = prediction - target

        block_index = sample_index // samples_per_design
        run_number = np.nan
        if "run_number" in df.columns:
            run_number = int(df["run_number"].iloc[0])

        design_id = np.nan
        if "design_id" in df.columns:
            design_id = int(df["design_id"].iloc[0])
        elif design_ids is not None and block_index < len(design_ids):
            design_id = design_ids[block_index]

        record = {
            "sample_index": sample_index,
            "sample_id": sample_id_from_path(path),
            "run_number": run_number,
            "test_block": block_index,
            "design_id": design_id,
            "n_time_points": len(error),
            "sse": float(np.sum(error**2)),
            "sae": float(np.sum(np.abs(error))),
            "accel_rmse": float(np.sqrt(np.mean(error**2))),
            "accel_mae": float(np.mean(np.abs(error))),
            "path": str(path),
        }
        records.append(record)

        hic_groups.setdefault(len(time), []).append((len(records) - 1, time, target, prediction))

    for group in hic_groups.values():
        times = np.vstack([item[1] for item in group])
        targets = np.vstack([item[2] for item in group])
        predictions = np.vstack([item[3] for item in group])
        target_hic, prediction_hic = compute_hic_batch(
            times,
            targets,
            predictions,
            max_window,
        )

        for item, hic_target, hic_prediction in zip(group, target_hic, prediction_hic):
            record = records[item[0]]
            record["hic_ground_truth"] = float(hic_target)
            record["hic_prediction"] = float(hic_prediction)
            record["hic_error"] = float(hic_prediction - hic_target)
            record["hic_abs_error"] = float(abs(hic_prediction - hic_target))

    return pd.DataFrame(records)


def summarize_overall(per_sample):
    total_sse = per_sample["sse"].sum()
    total_sae = per_sample["sae"].sum()
    total_points = per_sample["n_time_points"].sum()
    hic_target = per_sample["hic_ground_truth"].to_numpy(np.float64)
    hic_prediction = per_sample["hic_prediction"].to_numpy(np.float64)
    hic_error = hic_prediction - hic_target

    return pd.DataFrame(
        [
            {
                "n_samples": len(per_sample),
                "n_time_points": int(total_points),
                "pointwise_accel_rmse": float(np.sqrt(total_sse / total_points)),
                "pointwise_accel_mae": float(total_sae / total_points),
                "mean_sample_accel_rmse": float(per_sample["accel_rmse"].mean()),
                "median_sample_accel_rmse": float(per_sample["accel_rmse"].median()),
                "hic_rmse": float(np.sqrt(np.mean(hic_error**2))),
                "hic_mae": float(np.mean(np.abs(hic_error))),
                "hic_bias": float(np.mean(hic_error)),
                "hic_r2": float(r2_score_from_arrays(hic_target, hic_prediction)),
            }
        ]
    )


def summarize_blocks(per_sample):
    rows = []
    group_cols = ["test_block", "design_id"]
    for group_keys, group in per_sample.groupby(group_cols, dropna=False):
        hic_target = group["hic_ground_truth"].to_numpy(np.float64)
        hic_prediction = group["hic_prediction"].to_numpy(np.float64)
        hic_error = hic_prediction - hic_target
        total_sse = group["sse"].sum()
        total_sae = group["sae"].sum()
        total_points = group["n_time_points"].sum()

        rows.append(
            {
                "test_block": group_keys[0],
                "design_id": group_keys[1],
                "n_samples": len(group),
                "start_sample_id": int(group["sample_id"].min()),
                "end_sample_id": int(group["sample_id"].max()),
                "pointwise_accel_rmse": float(np.sqrt(total_sse / total_points)),
                "pointwise_accel_mae": float(total_sae / total_points),
                "mean_sample_accel_rmse": float(group["accel_rmse"].mean()),
                "mean_sample_accel_mae": float(group["accel_mae"].mean()),
                "hic_rmse": float(np.sqrt(np.mean(hic_error**2))),
                "hic_mae": float(np.mean(np.abs(hic_error))),
                "hic_bias": float(np.mean(hic_error)),
                "hic_r2": float(r2_score_from_arrays(hic_target, hic_prediction)),
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report acceleration-history and HIC metrics from exported "
            "test_sample_*.csv files. Files are discovered recursively."
        )
    )
    parser.add_argument("predictions_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-window", type=float, default=0.015)
    parser.add_argument("--samples-per-design", type=int, default=50)
    parser.add_argument(
        "--test-design-ids",
        default=None,
        help="Optional comma-separated design IDs matching the sorted test blocks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or args.predictions_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_ids = parse_design_ids(args.test_design_ids)
    sample_files = find_sample_files(args.predictions_dir)
    per_sample = load_records(
        sample_files,
        args.samples_per_design,
        design_ids,
        args.max_window,
    )
    overall = summarize_overall(per_sample)
    blocks = summarize_blocks(per_sample)

    per_sample_path = output_dir / "acceleration_hic_per_sample_metrics.csv"
    overall_path = output_dir / "acceleration_hic_overall_metrics.csv"
    blocks_path = output_dir / "acceleration_hic_block_metrics.csv"

    per_sample.to_csv(per_sample_path, index=False)
    overall.to_csv(overall_path, index=False)
    blocks.to_csv(blocks_path, index=False)

    row = overall.iloc[0]
    print(f"Samples: {int(row['n_samples'])}")
    print(f"Pointwise acceleration RMSE: {row['pointwise_accel_rmse']:.6f} g")
    print(f"Mean per-sample acceleration RMSE: {row['mean_sample_accel_rmse']:.6f} g")
    print(f"HIC RMSE: {row['hic_rmse']:.6f}")
    print(f"Wrote: {per_sample_path}")
    print(f"Wrote: {overall_path}")
    print(f"Wrote: {blocks_path}")


if __name__ == "__main__":
    main()
