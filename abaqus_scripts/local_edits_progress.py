"""
local_edits_progress.py
=======================
Progress of the local-edit campaign: which runs are done (never solved again),
which finished but are not recorded yet (adopted by the next job, not solved
again), and what is solved next. Standard library only; run on the Delta login
node from the Code directory:

    python abaqus_scripts/local_edits_progress.py
    python abaqus_scripts/local_edits_progress.py --list     # also list the run numbers

A run counts as done only if it finished on the campaign's core count (8, or
LOCAL_EDITS_CPUS in Data/HoodImpact_LocalEdits/campaign_layout.env): runs
finished on another count are solved again (hood_local_edits.sbatch).
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "Data", "HoodImpact_LocalEdits")
WORK = os.environ.get("LOCAL_EDITS_WORK", "/u/hkazemi/hood_local_edits_scratch")
RUNS_PER_H = {1: 7.2, 8: 15.0}   # 12 x 1 campaign; 5 x 8 measured by the cpus test (job 23905334)


def campaign_layout():
    """(cpus, redo) as hood_local_edits.sbatch reads them: env > campaign_layout.env > 8, redo on."""
    vals = {}
    path = os.path.join(DATA, "campaign_layout.env")
    if os.path.exists(path):
        for line in open(path):
            key, _, val = line.partition("=")
            if key.strip() in ("LOCAL_EDITS_CPUS", "LOCAL_EDITS_REDO_OTHER_CPUS"):
                vals[key.strip()] = val.split("#")[0].strip()
    for key in ("LOCAL_EDITS_CPUS", "LOCAL_EDITS_REDO_OTHER_CPUS"):
        if os.environ.get(key):
            vals[key] = os.environ[key]
    return int(vals.get("LOCAL_EDITS_CPUS") or 8), vals.get("LOCAL_EDITS_REDO_OTHER_CPUS", "1") == "1"


def row_cpus(m):
    try:
        return int(float(m.get("cpus") or 1))
    except ValueError:
        return 1
GROUPS = [("pilot + design 6", None), ("cut-line crease", ("cutline", "K")), ("cut-line standard", ("cutline", "S")),
          ("single-beam crease", ("single", "K")), ("cut-line half crease", ("cutline", "KH")),
          ("cut-line deeper", ("cutline", "D")), ("single-beam notches", "single")]


def ranges(runs):
    out, runs = [], sorted(runs)
    for r in runs:
        if out and r == out[-1][1] + 1:
            out[-1][1] = r
        else:
            out.append([r, r])
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)


def completed(run):
    d = os.path.join(WORK, "run_%d" % run)
    sta = os.path.join(d, "HoodImpact_%d.sta" % run)
    try:
        with open(sta, errors="ignore") as f:
            ok = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in f.read()
    except OSError:
        return False
    return ok and os.path.exists(os.path.join(d, "HoodImpact_%d_raw_acc.csv" % run))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the run numbers of each state")
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(os.path.join(DATA, "manifest_local_edits.csv"))))
    meta = {r["design"]: r for r in csv.DictReader(open(os.path.join(DATA, "local_edit_designs.csv")))}
    metrics = os.path.join(WORK, "generation_metrics.csv")
    if not os.path.exists(metrics):
        metrics = os.path.join(DATA, "generation_metrics.csv")
    cpus, redo = campaign_layout()
    status, other = {}, set()
    if os.path.exists(metrics):
        for m in csv.DictReader(open(metrics)):
            run = int(m["run"])
            if m["status"] == "ok" and redo and row_cpus(m) != cpus:
                other.add(run)                   # finished on another core count: solved again
                continue
            if m["status"] == "ok" or run not in status:
                status[run] = m["status"]
    acc = os.path.join(DATA, "output_history_acc")

    state = {}
    for r in manifest:
        run = int(r["run"])
        if status.get(run) == "ok":
            state[run] = "done"
        elif run in other:
            state[run] = "other_cpus"
        elif os.path.exists(os.path.join(acc, "HoodImpact_%d_SAE1000_interp1000.csv" % run)) and completed(run):
            state[run] = "unrecorded"
        elif run in status:
            state[run] = "failed"
        else:
            state[run] = "todo"

    def group(r):
        if r["design"] == "D6" or r["pilot"] == "1":
            return GROUPS[0][0]
        m = meta[r["design"]]
        for name, key in GROUPS[1:]:
            if key == (m["kind"], m["variant"]) or key == m["kind"]:
                return name

    n = len(manifest)
    cnt = {s: sum(1 for v in state.values() if v == s)
           for s in ("done", "unrecorded", "other_cpus", "failed", "todo")}
    print("metrics: %s" % metrics)
    print("%d runs in the plan; campaign layout cpus=%d per solve" % (n, cpus))
    print("  done on %2d cores (ok, never solved again)       %5d" % (cpus, cnt["done"]))
    print("  finished, not recorded (adopted by the next job) %5d" % cnt["unrecorded"])
    print("  finished on another core count (solved again)    %5d" % cnt["other_cpus"])
    print("  failed or interrupted (solved again)             %5d" % cnt["failed"])
    print("  not started                                      %5d" % cnt["todo"])
    left = cnt["other_cpus"] + cnt["failed"] + cnt["todo"]
    rate = RUNS_PER_H.get(cpus, RUNS_PER_H[8])
    print("  left to solve %d runs = about %.0f h at ~%.0f runs/h (%.1f jobs of 48 h)"
          % (left, left / rate, rate, left / rate / 46))
    print("\nby group, in solve order:")
    for name, _ in GROUPS:
        rs = [r for r in manifest if group(r) == name]
        d = sum(1 for r in rs if state[int(r["run"])] in ("done", "unrecorded"))
        print("  %-22s %4d / %4d done" % (name, d, len(rs)))
    order = sorted(manifest, key=lambda r: int(r.get("solve_order") or r["run"]))
    nxt = []
    for r in order:
        if state[int(r["run"])] in ("other_cpus", "failed", "todo") and r["design"] not in nxt:
            nxt.append(r["design"])
    print("\nnext designs: %s" % " ".join(nxt[:8]))
    if args.list:
        for s in ("done", "unrecorded", "other_cpus", "failed"):
            runs = [k for k, v in state.items() if v == s]
            print("\n%s runs (%d): %s" % (s, len(runs), ranges(runs) or "-"))


if __name__ == "__main__":
    main()
