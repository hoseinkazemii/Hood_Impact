"""
local_edits_progress.py
=======================
Progress of the local-edit campaign: which runs are done (never solved again),
which finished but are not recorded yet (adopted by the next job, not solved
again), and what is solved next. Standard library only; run on the Delta login
node from the Code directory:

    python abaqus_scripts/local_edits_progress.py
    python abaqus_scripts/local_edits_progress.py --list     # also list the run numbers
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "Data", "HoodImpact_LocalEdits")
WORK = os.environ.get("LOCAL_EDITS_WORK", "/u/hkazemi/hood_local_edits_scratch")
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
    status = {}
    if os.path.exists(metrics):
        for m in csv.DictReader(open(metrics)):
            if m["status"] == "ok" or int(m["run"]) not in status:
                status[int(m["run"])] = m["status"]
    acc = os.path.join(DATA, "output_history_acc")

    state = {}
    for r in manifest:
        run = int(r["run"])
        if status.get(run) == "ok":
            state[run] = "done"
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
    cnt = {s: sum(1 for v in state.values() if v == s) for s in ("done", "unrecorded", "failed", "todo")}
    print("metrics: %s" % metrics)
    print("%d runs in the plan" % n)
    print("  done (ok, never solved again)                    %5d" % cnt["done"])
    print("  finished, not recorded (adopted by the next job) %5d" % cnt["unrecorded"])
    print("  failed or interrupted (solved again)             %5d" % cnt["failed"])
    print("  not started                                      %5d" % cnt["todo"])
    left = cnt["failed"] + cnt["todo"]
    print("  left to solve %d runs = about %.0f h on 12 lanes (%.1f jobs of 48 h)" % (left, left / 6.6, left / 6.6 / 46))
    print("\nby group, in solve order:")
    for name, _ in GROUPS:
        rs = [r for r in manifest if group(r) == name]
        d = sum(1 for r in rs if state[int(r["run"])] in ("done", "unrecorded"))
        print("  %-22s %4d / %4d done" % (name, d, len(rs)))
    order = sorted(manifest, key=lambda r: int(r.get("solve_order") or r["run"]))
    nxt = []
    for r in order:
        if state[int(r["run"])] in ("failed", "todo") and r["design"] not in nxt:
            nxt.append(r["design"])
    print("\nnext designs: %s" % " ".join(nxt[:8]))
    if args.list:
        for s in ("done", "unrecorded", "failed"):
            runs = [k for k, v in state.items() if v == s]
            print("\n%s runs (%d): %s" % (s, len(runs), ranges(runs) or "-"))


if __name__ == "__main__":
    main()
