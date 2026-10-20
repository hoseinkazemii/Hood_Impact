"""
analyze_solver_factorial.py
===========================
Decompose the design-4 vs design-5 acceleration difference into solver set-up
and geometry, using the solver-factorial runs (make_solver_factorial_decks.py)
and, when solved, the noise-floor runs (make_noise_floor_decks.py).

Per impact location, all RMS over the acceleration history (g):

    native            |a5 - a4| in the delivered dataset (automatic, design-
                      dependent time step; centre-clearance start)
    fixed_dt          same, both designs at one user-controlled time step
    gap_start         same, headform starting clear of the hood
    fixed_dt_gap_start  both controls: the geometry-only difference
    dt_effect_d       |a_d(fixed dt) - a_d(native)|: how much the time step
                      alone moves one design's response
    clean_jitter      |clean - clean with 1e-3 mm hood jitter| on design 4:
                      the chaotic floor under the clean settings, the right
                      yardstick for the clean difference
    jitter            |control - 1e-3 mm jitter| on design 4 under the native
                      settings (noise-floor experiment), the yardstick for the
                      native difference
    determinism       |control - delivered archive| on design 4

Locations are grouped by XY distance from the impact point to the nearest node
the 4-5 edit moves by more than 0.5 mm (near < 100 mm <= mid < 250 mm <= far).

Questions it answers:

1. Artifact share: how much of the native difference energy disappears with
   both controls? 1 - sum(clean^2) / sum(native^2), per band. A large share in
   the far band confirms the time-step / start-state artifact.
2. Genuine signal: is the clean difference near the edit larger than the
   chaotic floor? clean / clean_jitter per location; well above 1 means a
   surrogate has something real to learn there.

Partial results are fine: it reports whatever has been solved so far.

Usage
-----
    python abaqus_scripts/analyze_solver_factorial.py
    python abaqus_scripts/analyze_solver_factorial.py --stride 16   # surrogate grid
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import design_difference_data as dd  # noqa: E402
from hic15 import batched_hic  # noqa: E402

DEFAULT_FACTORIAL = os.path.join(CODE_ROOT, "Data", "HoodImpact_SolverFactorial")
DEFAULT_NOISE = os.path.join(CODE_ROOT, "Data", "HoodImpact_NoiseFloor")
DEFAULT_DATA = os.path.join(CODE_ROOT, "Data", "HoodImpact_1704_EuroNCAP")
CELLS = ("native", "fixed_dt", "gap_start", "fixed_dt_gap_start")
JITTER_CELL = "fixed_dt_gap_start_jitter"
EARLY_MS = 3.0
ONSET_G = 3.0
# A fixed increment above the stable limit blows up; flag anything implausible.
IMPLAUSIBLE_PEAK_G = 1000.0


def read_history(acc_dir, run):
    path = os.path.join(acc_dir, "HoodImpact_%d_SAE1000_interp1000.csv" % run)
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return frame["Time"].to_numpy(float), frame["A(in g)"].to_numpy(float)


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def onset_ms(time_ms, a, b, threshold=ONSET_G):
    hit = np.nonzero(np.abs(np.asarray(a) - np.asarray(b)) > threshold)[0]
    return float(time_ms[hit[0]]) if len(hit) else np.nan


def energy_share_removed(native, clean):
    native, clean = np.asarray(native, float), np.asarray(clean, float)
    keep = np.isfinite(native) & np.isfinite(clean)
    if not keep.any() or not np.sum(native[keep] ** 2):
        return np.nan
    return float(1.0 - np.sum(clean[keep] ** 2) / np.sum(native[keep] ** 2))


def collect(args):
    """{(cell, design, location): curve} on one time grid, plus jitter/determinism curves."""
    factorial = pd.read_csv(os.path.join(args.factorial_dir, "manifest_solver_factorial.csv"))
    factorial_acc = os.path.join(args.factorial_dir, "output_history_acc")
    data_acc = os.path.join(args.data_dir, "output_history_acc")
    locations = sorted(factorial["location"].unique())
    designs = sorted(factorial["design"].unique())
    if designs != [4, 5]:
        raise SystemExit("the decomposition expects designs 4 and 5, found %s" % designs)
    curves, missing, implausible = {}, [], []
    reference_time = None

    def keep(key, history, source):
        nonlocal reference_time
        if history is None:
            missing.append((source,) + key)
            return
        time, values = history
        if reference_time is None:
            reference_time = time
        # 10 ns against a 25 us sample spacing: tolerates float32 end times.
        elif not np.allclose(time, reference_time, rtol=0, atol=1e-8):
            raise SystemExit("%s %s is on a different time grid" % (source, key))
        if not np.all(np.isfinite(values)) or np.abs(values).max() > IMPLAUSIBLE_PEAK_G:
            implausible.append((source,) + key)
            return
        curves[key] = values

    for location in locations:
        for design in designs:
            keep(("native", design, location), read_history(data_acc, dd.run_number(design, location)), "archive")
    for row in factorial.itertuples():
        if row.cell == "native":
            continue  # the delivered archive is the native cell
        keep((row.cell, row.design, row.location), read_history(factorial_acc, row.synthetic_run), "factorial")

    noise_manifest = os.path.join(args.noise_dir, "manifest_noise_floor.csv")
    if os.path.exists(noise_manifest):
        noise = pd.read_csv(noise_manifest)
        noise_acc = os.path.join(args.noise_dir, "output_history_acc")
        for row in noise.itertuples():
            if row.design == 4 and row.location in locations:
                keep(("noise_" + row.variant, 4, row.location),
                     read_history(noise_acc, row.synthetic_run), "noise")
    return curves, reference_time, locations, missing, implausible


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factorial-dir", default=DEFAULT_FACTORIAL)
    parser.add_argument("--noise-dir", default=DEFAULT_NOISE)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--stride", type=int, default=1, help="16 matches the surrogate's 63-point grid")
    parser.add_argument("--out-dir", default=None, help="defaults to the factorial directory")
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be a positive integer")
    out_dir = args.out_dir or args.factorial_dir
    os.makedirs(out_dir, exist_ok=True)

    curves, time, locations, missing, implausible = collect(args)
    step = slice(None, None, args.stride)
    time = time[step]
    time_ms = time * 1e3
    early = time_ms <= EARLY_MS
    curves = {key: values[step] for key, values in curves.items()}

    geometry = dd.pair_geometry(dd.load_design_geometry(4, args.data_dir), dd.load_design_geometry(5, args.data_dir))
    runs = dd.load_run_table(args.data_dir)
    centre = runs[runs.design == 4].set_index("loc").loc[locations, ["X1", "X2", "X3"]].to_numpy()
    distance_xy, _ = dd.edit_distance(geometry, centre)
    penetration = runs[runs.design == 4].set_index("loc").loc[locations, "sphere_pen_mm"].to_numpy()

    rows = []
    for index, location in enumerate(locations):
        entry = {"location": int(location), "distance_xy_mm": float(distance_xy[index]),
                 "band": dd.distance_band(distance_xy[index]).item(),
                 "native_sphere_pen_mm": float(penetration[index])}
        for cell in CELLS:
            a4, a5 = curves.get((cell, 4, location)), curves.get((cell, 5, location))
            if a4 is None or a5 is None:
                continue
            entry["%s_g" % cell] = rms(a5, a4)
            entry["%s_early_g" % cell] = rms(a5[early], a4[early])
            entry["%s_onset_ms" % cell] = onset_ms(time_ms, a5, a4)
            entry["%s_dhic" % cell] = float(batched_hic(time, a5) - batched_hic(time, a4))
        for design in (4, 5):
            native, fixed = curves.get(("native", design, location)), curves.get(("fixed_dt", design, location))
            if native is not None and fixed is not None:
                entry["dt_effect_d%d_g" % design] = rms(fixed, native)
        clean, jittered = curves.get(("fixed_dt_gap_start", 4, location)), curves.get((JITTER_CELL, 4, location))
        if clean is not None and jittered is not None:
            entry["clean_jitter_g"] = rms(clean, jittered)
            entry["clean_jitter_early_g"] = rms(clean[early], jittered[early])
        control = curves.get(("noise_control", 4, location))
        if control is not None:
            if ("native", 4, location) in curves:
                entry["determinism_g"] = rms(control, curves[("native", 4, location)])
            for key, values in curves.items():
                if key[0].startswith("noise_jitter") and key[2] == location:
                    entry["%s_g" % key[0].replace("noise_", "")] = rms(control, values)
        rows.append(entry)
    table = pd.DataFrame(rows).sort_values("distance_xy_mm")
    grid = "full 1000-point grid" if args.stride == 1 else "stride %d (%d points)" % (args.stride, len(time))
    suffix = "" if args.stride == 1 else "_stride%d" % args.stride
    table.to_csv(os.path.join(out_dir, "solver_factorial_per_location%s.csv" % suffix), index=False)

    print("Design 4 vs 5 RMS difference (g) by cell, %s; sorted by XY distance to the edit\n" % grid)
    columns = [c for c in ("native_g", "fixed_dt_g", "gap_start_g", "fixed_dt_gap_start_g",
                           "clean_jitter_g", "dt_effect_d4_g", "dt_effect_d5_g", "jitter_0.001_g",
                           "jitter_0.1_g", "determinism_g") if c in table]
    shown = table[["location", "distance_xy_mm", "native_sphere_pen_mm"] + columns]
    print(shown.to_string(index=False, float_format=lambda v: "%.2f" % v))

    summary = {"grid": grid, "locations": len(table), "bands": {}}
    print("\nBy distance band (median g, and the share of native difference energy each control removes):")
    for name, _, _ in dd.DISTANCE_BANDS:
        band = table[table["band"] == name]
        if band.empty:
            continue
        stats = {"n": int(len(band))}
        for column in columns:
            stats["median_" + column] = float(band[column].median())
        for cell in ("fixed_dt", "gap_start", "fixed_dt_gap_start"):
            if "%s_g" % cell in band and "native_g" in band:
                stats["energy_removed_by_" + cell] = energy_share_removed(band["native_g"], band["%s_g" % cell])
        for numerator, denominator, label in (("fixed_dt_gap_start_g", "clean_jitter_g", "clean_over_clean_jitter"),
                                              ("native_g", "jitter_0.001_g", "native_over_native_jitter")):
            if numerator in band and denominator in band:
                ratio = (band[numerator] / band[denominator]).replace([np.inf, -np.inf], np.nan)
                stats["ratio_" + label] = float(ratio.median())
        summary["bands"][name] = stats
        print("  %-4s n=%-2d " % (name, stats["n"]) + "  ".join(
            "%s %.2f" % (key[len("median_"):-len("_g")], value)
            for key, value in stats.items() if key.startswith("median_")))
        removed = {key: value for key, value in stats.items() if key.startswith("energy_removed_by_")}
        if removed:
            print("       energy removed: " + "  ".join(
                "%s %.0f%%" % (key.replace("energy_removed_by_", ""), 100 * value) for key, value in removed.items()))
        ratios = {key: value for key, value in stats.items() if key.startswith("ratio_")}
        if ratios:
            print("       median ratios:  " + "  ".join(
                "%s %.2f" % (key.replace("ratio_", ""), value) for key, value in ratios.items()))

    print("\nReading:")
    print("  energy removed by fixed_dt_gap_start in the far band near 100%  -> the far-field 4-5")
    print("    difference is a solver artifact; do not ask a geometry model to learn it.")
    print("  clean_over_clean_jitter well above 1 in the near band -> real geometry signal a")
    print("    surrogate should learn; near 1 everywhere -> the near-clone difference is chaos.")
    if implausible:
        print("\nWARNING: %d run(s) with non-finite or |a| > %g g were excluded (unstable fixed increment?): %s"
              % (len(implausible), IMPLAUSIBLE_PEAK_G, implausible[:6]))
    if missing:
        print("\n%d curve(s) not solved yet, e.g. %s" % (len(missing), missing[:6]))
    summary["missing"] = len(missing)
    summary["implausible"] = [list(map(str, key)) for key in implausible]
    dd.write_json(os.path.join(out_dir, "solver_factorial_report%s.json" % suffix), summary)
    if not table.empty and "native_g" in table:
        plot(table, columns, os.path.join(out_dir, "solver_factorial%s.png" % suffix), grid)
    print("\nWrote %s" % out_dir)


def plot(table, columns, path, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"native_g": "native", "fixed_dt_g": "fixed dt", "gap_start_g": "gap start",
              "fixed_dt_gap_start_g": "fixed dt + gap (clean)",
              "clean_jitter_g": "clean, 1e-3 mm jitter (design 4)",
              "jitter_0.001_g": "native, 1e-3 mm jitter (design 4)"}
    shown = [c for c in labels if c in columns]
    x = np.arange(len(table))
    width = 0.8 / max(len(shown), 1)
    figure, axis = plt.subplots(figsize=(max(8, 0.7 * len(table)), 4.5))
    for i, column in enumerate(shown):
        axis.bar(x + (i - (len(shown) - 1) / 2) * width, table[column], width, label=labels[column])
    axis.set_xticks(x)
    axis.set_xticklabels(["%d\n%.0f mm" % (r.location, r.distance_xy_mm) for r in table.itertuples()], fontsize=8)
    axis.set_ylabel("design 4 vs 5 RMS difference (g)")
    axis.set_xlabel("impact location / XY distance to the edit")
    axis.set_title("Solver factorial, %s" % grid)
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
