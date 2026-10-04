"""
analyze_noise_floor.py
======================
Read the solved noise-floor runs and answer one question: how much of the
design-4 vs design-5 acceleration difference is reproduced by a perturbation
that cannot matter physically?

Three quantities per impact location, all RMS over the acceleration history:

    determinism   |control - delivered archive|
                  the same deck solved twice in different places. Bounds what
                  the pipeline can resolve at all.

    jitter        |control - jittered|
                  identical physics, hood nodes displaced by 1e-3 or 1e-1 mm.
                  Any difference here is chaotic amplification of roundoff-scale
                  input change, not a design effect.

    design        |design 4 - design 5|
                  the difference a surrogate is being asked to predict.

The headline number is jitter / design. Near 1, the design difference is mostly
irreducible and no surrogate should be expected to predict it; the modelling
goal has to be restated. Near 0, the difference is real signal that the current
models simply fail to capture, and more modelling is justified.

HIC15 is reported alongside, since that is the engineering quantity: a 10 g
acceleration difference matters only insofar as it moves HIC.

Usage
-----
    python abaqus_scripts/analyze_noise_floor.py
    python abaqus_scripts/analyze_noise_floor.py --stride 16   # model's grid
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from hic15 import batched_hic

DEFAULT_NOISE = os.path.join(CODE_ROOT, "Data", "HoodImpact_NoiseFloor")
DEFAULT_DATA = os.path.join(CODE_ROOT, "Data", "HoodImpact_1704_EuroNCAP")
MANIFEST = "manifest_noise_floor.csv"


def read_history(acc_dir, run):
    path = os.path.join(acc_dir, "HoodImpact_%d_SAE1000_interp1000.csv" % run)
    if not os.path.exists(path):
        return None, None
    frame = pd.read_csv(path)
    return frame["Time"].to_numpy(float), frame["A(in g)"].to_numpy(float)


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def summarize(label, values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {"comparison": label, "n": 0}
    return {"comparison": label, "n": int(len(values)), "mean_g": float(values.mean()),
            "median_g": float(np.median(values)), "min_g": float(values.min()),
            "max_g": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-dir", default=DEFAULT_NOISE)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--stride", type=int, default=1,
                        help="subsample the history; 16 matches the surrogate's grid")
    parser.add_argument("--out", default=None, help="JSON report path")
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be a positive integer")

    manifest_path = os.path.join(args.noise_dir, MANIFEST)
    if not os.path.exists(manifest_path):
        raise SystemExit("missing %s; run make_noise_floor_decks.py first" % manifest_path)
    manifest = pd.read_csv(manifest_path)
    noise_acc = os.path.join(args.noise_dir, "output_history_acc")
    data_acc = os.path.join(args.data_dir, "output_history_acc")
    if not os.path.isdir(noise_acc):
        raise SystemExit("no solved runs in %s; submit delta/hood_noise_floor.sbatch" % noise_acc)

    jitter_variants = sorted(v for v in manifest["variant"].unique() if v != "control")
    rows, missing = [], []
    for location, group in manifest.groupby("location", sort=True):
        entry = {"location": int(location), "band": group["band"].iloc[0],
                 "source_run": int(group["source_run"].iloc[0]),
                 "pair_run": int(group["pair_run"].iloc[0])}
        control_run = group.loc[group.variant == "control", "synthetic_run"]
        if control_run.empty:
            continue
        time, control = read_history(noise_acc, int(control_run.iloc[0]))
        if control is None:
            missing.append((int(location), "control"))
            continue

        # The archive is the delivered solve of the very same deck.
        archive_time, archive = read_history(data_acc, entry["source_run"])
        pair_time, pair = read_history(data_acc, entry["pair_run"])
        for name, other_time in (("archive", archive_time), ("pair", pair_time)):
            if other_time is not None and not np.allclose(other_time, time, rtol=0, atol=1e-12):
                raise SystemExit("location %d: %s history is on a different time grid; "
                                 "refusing to compare by index" % (location, name))

        step = slice(None, None, args.stride)
        t = time[step]
        curves = {"control": control[step]}
        for variant in jitter_variants:
            match = group.loc[group.variant == variant, "synthetic_run"]
            _, values = read_history(noise_acc, int(match.iloc[0])) if not match.empty else (None, None)
            if values is None:
                missing.append((int(location), variant))
            else:
                curves[variant] = values[step]
        if archive is not None:
            curves["archive"] = archive[step]
        if pair is not None:
            curves["pair"] = pair[step]

        for name, values in curves.items():
            entry["hic_%s" % name] = float(batched_hic(t, values))
        if "archive" in curves:
            entry["determinism_g"] = rms(curves["control"], curves["archive"])
        for variant in jitter_variants:
            if variant in curves:
                entry["%s_g" % variant] = rms(curves["control"], curves[variant])
                entry["%s_dhic" % variant] = entry["hic_%s" % variant] - entry["hic_control"]
        if "pair" in curves:
            entry["design_g"] = rms(curves["control"], curves["pair"])
            entry["design_dhic"] = entry["hic_pair"] - entry["hic_control"]
        rows.append(entry)

    if not rows:
        raise SystemExit("no solved locations found in %s" % noise_acc)
    table = pd.DataFrame(rows).sort_values("design_g", ascending=False)

    grid = "full 1000-point SAE1000 grid" if args.stride == 1 else \
        "stride %d (the surrogate's %d-point grid)" % (args.stride, len(t))
    print("Noise floor for design 4, %d locations, %s\n" % (len(table), grid))
    header = ["loc", "band", "design", "determ"] + [v.replace("jitter_", "jit") for v in jitter_variants]
    print("  %-5s %-5s %8s %8s" % tuple(header[:4]) + "".join("%10s" % h for h in header[4:]))
    print("  " + "-" * (28 + 10 * len(jitter_variants)))
    for record in table.to_dict("records"):
        # Index by column name: itertuples() mangles names holding a dot.
        line = "  %-5d %-5s %8.2f %8.2f" % (
            record["location"], record["band"],
            record.get("design_g", float("nan")), record.get("determinism_g", float("nan")))
        for variant in jitter_variants:
            line += "%10.2f" % record.get("%s_g" % variant, float("nan"))
        print(line)

    print("\nRMS acceleration difference (g):")
    report = {"grid": grid, "num_locations": int(len(table)), "comparisons": []}
    for label, column in ([("design 4 vs design 5 (the target)", "design_g"),
                           ("control vs delivered archive (determinism)", "determinism_g")] +
                          [("control vs %s mm jitter" % v.split("_", 1)[1], "%s_g" % v)
                           for v in jitter_variants]):
        if column not in table:
            continue
        stats = summarize(label, table[column].tolist())
        report["comparisons"].append(stats)
        if stats["n"]:
            print("  %-42s mean %6.2f  median %6.2f  max %6.2f"
                  % (label, stats["mean_g"], stats["median_g"], stats["max_g"]))

    print("\nHIC15 shift relative to the control solve:")
    for label, column in ([("design 4 vs design 5", "design_dhic")] +
                          [("%s mm jitter" % v.split("_", 1)[1], "%s_dhic" % v)
                           for v in jitter_variants]):
        if column in table:
            values = table[column].abs()
            print("  %-42s mean %6.1f  median %6.1f  max %6.1f"
                  % (label, values.mean(), values.median(), values.max()))

    if "design_g" in table:
        print("\nFraction of the design difference reproduced by a meaningless perturbation:")
        print("  (ratio of medians within a band; a per-location mean would be")
        print("   dominated by quiet locations whose design difference is near zero)")
        for variant in jitter_variants:
            column = "%s_g" % variant
            if column not in table:
                continue
            amplitude = variant.split("_", 1)[1]
            entry = {}
            for band in ("high", "low", "manual"):
                subset = table[table["band"] == band]
                if subset.empty or not subset["design_g"].median():
                    continue
                jitter_median = float(subset[column].median())
                design_median = float(subset["design_g"].median())
                entry[band] = {"ratio_of_medians": jitter_median / design_median,
                               "median_jitter_g": jitter_median,
                               "median_design_g": design_median}
                print("  %s mm jitter, %-6s band: %5.2f   (%.2f g jitter vs %.2f g design)"
                      % (amplitude, band, jitter_median / design_median,
                         jitter_median, design_median))
            report.setdefault("jitter_to_design_ratio", {})[amplitude] = entry
        print("\n  Near 1: the design difference is largely irreducible and a surrogate")
        print("          cannot be expected to predict it. Restate the goal.")
        print("  Near 0: the difference is real signal the current models miss.")

    if missing:
        print("\nWARNING: %d run(s) not solved yet: %s" % (len(missing), missing[:10]))

    out = args.out or os.path.join(args.noise_dir, "noise_floor_report.json")
    table.to_csv(os.path.join(args.noise_dir, "noise_floor_per_location.csv"), index=False)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\nWrote %s\n      %s" % (out, os.path.join(args.noise_dir, "noise_floor_per_location.csv")))


if __name__ == "__main__":
    main()
