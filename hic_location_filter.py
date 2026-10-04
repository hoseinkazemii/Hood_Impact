"""Select 1704 impact locations by full-history HIC15 range across designs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hic15 import batched_hic


def summarize(hic, threshold):
    hic = np.asarray(hic, dtype=float)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("HIC range threshold must be finite and nonnegative")
    if hic.ndim != 2 or hic.shape[0] < 2 or not np.isfinite(hic).all() or (hic < 0).any():
        raise ValueError("Expected finite nonnegative HIC values for at least two designs")
    mean = hic.mean(axis=0)
    spread = np.divide(100 * np.ptp(hic, axis=0), mean,
                       out=np.zeros_like(mean), where=mean > 0)
    table = pd.DataFrame({"location": np.arange(1, hic.shape[1] + 1),
                          "hic_min": hic.min(axis=0), "hic_max": hic.max(axis=0),
                          "hic_mean": mean, "range_percent": spread,
                          "selected": spread >= threshold})
    for design in range(hic.shape[0]):
        table[f"hic_design_{design}"] = hic[design]
    report = {"threshold_percent": float(threshold),
              "definition": "100 * (max(HIC15) - min(HIC15)) / mean(HIC15)",
              "selection_design_ids": list(range(hic.shape[0])),
              "hic_source": "full SAE1000_interp1000 histories; exact trapezoidal HIC15",
              "selected_locations": table.loc[table.selected, "location"].tolist(),
              "total_locations": hic.shape[1],
              "threshold_counts": {str(t): int((spread >= t).sum())
                                   for t in (10, 15, 20, 25, 30, 40, 50, 75, 100)}}
    return table, report


def analyze(acceleration_dir, threshold=10):
    values = []
    for design in range(12):
        histories, reference = [], None
        for location in range(1, 143):
            run = design * 142 + location
            path = Path(acceleration_dir) / f"HoodImpact_{run}_SAE1000_interp1000.csv"
            frame = pd.read_csv(path, usecols=["Time", "A(in g)"])
            times = frame["Time"].to_numpy(float)
            acceleration = frame["A(in g)"].to_numpy(float)
            if (len(times) < 2 or not np.isfinite(times).all()
                    or not np.isfinite(acceleration).all() or np.any(np.diff(times) <= 0)):
                raise ValueError(f"Invalid history: {path}")
            if reference is None:
                reference = times
            if not np.array_equal(reference, times):
                raise ValueError(f"Time grids differ within design: {path}")
            histories.append(np.abs(acceleration))
        values.append(batched_hic(reference, histories))
    return summarize(values, threshold)


def filter_data(data, selected_locations, samples_per_design=142):
    selected = set(selected_locations)
    if not selected or not selected <= set(range(1, samples_per_design + 1)):
        raise ValueError("No qualifying locations or invalid location IDs")
    indices = [i for i, run in enumerate(data["run_numbers"])
               if (run - 1) % samples_per_design + 1 in selected]
    return {key: value[indices] if isinstance(value, np.ndarray)
            else [value[i] for i in indices] for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceleration-dir", type=Path,
                        default=Path("Data/HoodImpact_1704_EuroNCAP/output_history_acc"))
    parser.add_argument("--threshold", type=float, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hic_location_filter"))
    args = parser.parse_args()
    table, report = analyze(args.acceleration_dir, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "hic_location_variation.csv", index=False)
    (args.output_dir / "hic_location_filter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
