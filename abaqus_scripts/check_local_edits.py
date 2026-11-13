"""
check_local_edits.py
====================
Quality checks for the local-edit dataset (local_edit_doe.py), run on Delta by
delta/hood_local_edits.sbatch and again on Windows once the results are back.

Each edited run is compared with the base design D6 at the same location, on
the shared 1000-point SAE1000 grid (resultant acceleration in g):

    rms_g      RMS of (design - D6) over the whole history
    dhic_pct   exact HIC15 change, 100 * (HIC_design / HIC_D6 - 1)

and each twin run with the same design at the location it mirrors.

--pilot is the gate before the full campaign. It passes only if
  1. every pilot run finished (status ok) and has its SAE1000 csv;
  2. every pilot run took the same number of increments (the fixed time step
     is in force, so designs cannot differ by time step);
  3. the far control differs from D6 by at most --far-limit-g RMS;
  4. the largest near-location difference is at least twice the far-control
     difference (clean local signal above the far-field floor).
Exit code 0 = pass, 1 = fail. Without --pilot the same numbers are reported for
every finished run, grouped by role, and nothing is gated.

Usage (from the Code directory):
    python abaqus_scripts/check_local_edits.py --pilot --metrics <work>/generation_metrics.csv
    python abaqus_scripts/check_local_edits.py            # after the campaign, on Windows
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE_ROOT)

import local_edit_doe as doe  # noqa: E402
from hic15 import batched_hic  # noqa: E402


def load_curve(acc_dir, run):
    path = os.path.join(acc_dir, "HoodImpact_%d_SAE1000_interp1000.csv" % run)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df["Time"].to_numpy(float), df["A(in g)"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=doe.OUT_DIR)
    ap.add_argument("--metrics", default=None, help="generation_metrics.csv (default: <data-dir>/generation_metrics.csv)")
    ap.add_argument("--pilot", action="store_true", help="check the pilot runs and gate")
    ap.add_argument("--far-limit-g", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()

    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    manifest = list(csv.DictReader(open(os.path.join(args.data_dir, doe.MANIFEST_CSV))))
    rows = [r for r in manifest if r["pilot"] == "1"] if args.pilot else manifest
    metrics_path = args.metrics or os.path.join(args.data_dir, "generation_metrics.csv")
    status, increments = {}, {}
    if os.path.exists(metrics_path):
        for m in csv.DictReader(open(metrics_path)):
            run = int(m["run"])
            if m["status"] == "ok" or run not in status:   # a later ok wins over an earlier failure
                status[run] = m["status"]
                increments[run] = m.get("n_raw_points", "")
    acc_dir = os.path.join(args.data_dir, "output_history_acc")
    curves, hic = {}, {}
    for r in rows:
        run = int(r["run"])
        c = load_curve(acc_dir, run) if status.get(run) == "ok" else None
        if c is not None:
            curves[run] = c
    if curves:
        runs = sorted(curves)
        hics = batched_hic(curves[runs[0]][0], np.stack([curves[k][1] for k in runs]))
        hic = dict(zip(runs, hics))
    base = {r["location"]: int(r["run"]) for r in manifest if r["design"] == doe.BASE_DESIGN}
    by_design_loc = {(r["design"], r["location"]): int(r["run"]) for r in manifest}

    say("Local-edit dataset check (%s): %d runs considered, %d finished" % (
        "pilot" if args.pilot else "all", len(rows), len(curves)))
    failed = [int(r["run"]) for r in rows if status.get(int(r["run"])) not in (None, "ok")]
    pending = [int(r["run"]) for r in rows if int(r["run"]) not in status]
    if failed:
        say("  FAILED runs: %s" % failed[:40])
    if pending:
        say("  not run yet: %d" % len(pending))
    inc = {increments[k] for k in curves if increments.get(k)}
    say("  increments per run: %s" % (", ".join(sorted(inc)) if inc else "n/a"))

    # differences
    table = []
    for r in rows:
        run = int(r["run"])
        if r["design"] == doe.BASE_DESIGN or run not in curves:
            continue
        b = base[r["location"]]
        rec = dict(run=run, design=r["design"], location=int(r["location"]), role=r["role"],
                   dist_mm=float(r["dist_to_edit_mm"]))
        if b in curves:
            rec["rms_g"] = float(np.sqrt(np.mean((curves[run][1] - curves[b][1]) ** 2)))
            rec["dhic_pct"] = 100.0 * (hic[run] / hic[b] - 1.0)
            rec["hic"], rec["hic_d6"] = hic[run], hic[b]
        if r["role"] == "twin":
            # the near location of the same design whose mirror this is
            twin_of = [n for n in manifest if n["design"] == r["design"] and n["role"] == "near"
                       and int(n["col_k"]) == -int(r["col_k"]) and abs(float(n["x_mm"]) - float(r["x_mm"])) < 1.0]
            if twin_of and int(twin_of[0]["run"]) in curves:
                m = int(twin_of[0]["run"])
                rec["twin_rms_g"] = float(np.sqrt(np.mean((curves[run][1] - curves[m][1]) ** 2)))
                rec["twin_of"] = int(twin_of[0]["location"])
                bt, bm = base.get(r["location"]), base.get(twin_of[0]["location"])
                if bt in curves and bm in curves:
                    rec["d6_twin_rms_g"] = float(np.sqrt(np.mean((curves[bt][1] - curves[bm][1]) ** 2)))
        table.append(rec)
    df = pd.DataFrame(table)
    if len(df):
        pd.set_option("display.width", 200)
        if args.pilot:
            say(df.round(2).to_string(index=False))
        else:
            for role, g in df.groupby("role"):
                say("  %-5s %4d runs  RMS vs D6 median %.2f g  p90 %.2f g  |dHIC| median %.1f%%  p90 %.1f%%" % (
                    role, len(g), g["rms_g"].median(), g["rms_g"].quantile(.9),
                    g["dhic_pct"].abs().median(), g["dhic_pct"].abs().quantile(.9)))
            near = df[df.role == "near"]
            for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 300)):
                g = near[(near.dist_mm >= lo) & (near.dist_mm < hi)]
                if len(g):
                    say("  near %3d-%3d mm: %4d runs  RMS median %.2f g  |dHIC| median %.1f%%"
                        % (lo, hi, len(g), g["rms_g"].median(), g["dhic_pct"].abs().median()))
            designs_csv = os.path.join(args.data_dir, doe.DESIGNS_CSV)
            if os.path.exists(designs_csv) and "dhic_pct" in near:
                meta = {r["design"]: r for r in csv.DictReader(open(designs_csv))}
                g_all = near.dropna(subset=["dhic_pct"]).assign(
                    group=lambda t: [meta[d]["kind"] + " " + meta[d]["variant"] for d in t.design])
                say("  near runs by kind and variant (dHIC vs D6 at the same location):")
                for grp, g in g_all.groupby("group", sort=False):
                    say("    %-12s %2d designs %4d runs  |dHIC| median %5.1f%%  p90 %5.1f%%  max %5.1f%%  RMS median %.2f g"
                        % (grp, g.design.nunique(), len(g), g.dhic_pct.abs().median(), g.dhic_pct.abs().quantile(.9),
                           g.dhic_pct.abs().max(), g.rms_g.median()))
            if "twin_rms_g" in df:
                t = df.dropna(subset=["twin_rms_g"])
                say("  twins: design at mirror location RMS median %.2f g, p90 %.2f g; D6 median %.2f g"
                    % (t["twin_rms_g"].median(), t["twin_rms_g"].quantile(.9), t["d6_twin_rms_g"].median()))
            qa = os.path.join(args.data_dir, "qa_local_edits.csv")
            df.to_csv(qa, index=False)
            say("  per-run table: %s" % qa)

    ok = True
    if args.pilot:
        reasons = []
        if failed or pending or len(curves) < len(rows):
            reasons.append("%d of %d pilot runs did not finish" % (len(rows) - len(curves), len(rows)))
        if len(inc) != 1:
            reasons.append("pilot runs took different numbers of increments: %s" % sorted(inc))
        far = df[df.role == "far"]["rms_g"] if len(df) and "rms_g" in df else pd.Series(dtype=float)
        near = df[df.role == "near"]["rms_g"] if len(df) and "rms_g" in df else pd.Series(dtype=float)
        if far.empty or near.empty:
            reasons.append("no near/far comparison with D6 available")
        else:
            if far.max() > args.far_limit_g:
                reasons.append("far control differs from D6 by %.2f g RMS (limit %.1f)" % (far.max(), args.far_limit_g))
            if near.max() < 2.0 * far.max():
                reasons.append("near difference %.2f g is not above twice the far-field %.2f g" % (near.max(), far.max()))
        ok = not reasons
        say("")
        say("PILOT GATE: %s" % ("PASS" if ok else "FAIL"))
        for reason in reasons:
            say("  - " + reason)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
