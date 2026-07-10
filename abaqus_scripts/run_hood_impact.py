"""
run_hood_impact.py
==================
Driver to REPRODUCE the delivered hood-impact acceleration histories from the
partner's input decks, on a single CPU core.

Key idea
--------
The 60 files in  Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_<n>.inp
are NOT just meshes -- each one is a *complete, self-contained Abaqus/Explicit
job*: hood mesh + rigid headform + materials + fasteners/couplings + contact +
the 11.1 m/s initial velocity + the acceleration history request on the
headform reference node. So to regenerate a sample we just SOLVE the deck and
read the headform acceleration back out -- we do not rebuild any physics.

This script, for each requested run id:
    1. solves  HoodImpact_<n>.inp  with `abq2025 ... cpus=1`            (Abaqus/Explicit)
    2. extracts the raw headform acceleration history from the .odb     (extract_acc_odb.py)
    3. SAE-1000 filters + resamples to 1000 pts, and (optionally)       (postprocess_acc.py)
       compares against the delivered reference file

Each run is solved inside its own output folder under --work-dir so the
(large) Abaqus output files stay organised and runs don't collide. If
--scratch-dir is provided, Abaqus temp files are also directed to a per-run
folder under that root.

Examples
--------
    # verify run 1 against the delivered data, with an overlay plot
    python run_hood_impact.py --runs 1 --compare --plot

    # solve several runs (still 1 core each, sequentially)
    python run_hood_impact.py --runs 1,6,31

    # re-do only the post-processing for a run already solved
    python run_hood_impact.py --runs 1 --skip-solve --compare --plot

NOTE: a full explicit solve of these ~40k-element decks to 25 ms on one core
is not instant (typically tens of minutes to a couple of hours). Start with a
single run to confirm the pipeline before launching a batch.
"""
import os
import sys
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
DEFAULT_INP_DIR = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "inp_files")
DEFAULT_REF_DIR = os.path.join(CODE_ROOT, "Data", "HoodImpact_60_IndustryLike", "output_history_acc")
DEFAULT_ABAQUS = r"C:\SIMULIA\Commands\abq2025.bat"
EXTRACTOR = os.path.join(HERE, "extract_acc_odb.py")
POSTPROC = os.path.join(HERE, "postprocess_acc.py")


def parse_runs(spec):
    """'1,6,31' or '1-5' or '1-3,10' -> sorted list of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def find_abaqus(user_value):
    if user_value:
        return user_value
    if os.path.exists(DEFAULT_ABAQUS):
        return DEFAULT_ABAQUS
    # fall back to whatever is on PATH
    return "abaqus"


def abaqus_env(scratch_dir=None):
    env = os.environ.copy()
    if scratch_dir:
        os.makedirs(scratch_dir, exist_ok=True)
        for key in ("TMPDIR", "TMP", "TEMP"):
            env[key] = scratch_dir
    return env


def run_cmd(cmd, cwd, env=None):
    print("\n$ %s   (cwd=%s)" % (" ".join('"%s"' % c if " " in c else c for c in cmd), cwd))
    # shell=True so the .bat launcher resolves correctly on Windows
    completed = subprocess.run(cmd, cwd=cwd, shell=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError("command failed (exit %d): %s" % (completed.returncode, cmd))


def solve(abaqus, inp_path, job, work_dir, cpus, scratch_dir=None):
    """Run Abaqus/Explicit on one deck, 1 core, blocking until finished."""
    cmd = [
        abaqus,
        "job=%s" % job,
        "input=%s" % inp_path,
        "cpus=%d" % cpus,
        "double=both",       # double precision packager+solver (recommended for explicit)
        "ask_delete=OFF",    # overwrite previous output without prompting
    ]
    if scratch_dir:
        cmd.append("scratch=%s" % scratch_dir)
    cmd.append("interactive")  # block until the analysis completes
    # a stale lock from a killed/crashed previous attempt blocks this one
    lck = os.path.join(work_dir, "%s.lck" % job)
    if os.path.exists(lck):
        os.remove(lck)
    run_cmd(cmd, cwd=work_dir, env=abaqus_env(scratch_dir))


def extract(abaqus, odb_path, raw_csv, work_dir, scratch_dir=None):
    cmd = [abaqus, "python", EXTRACTOR, "--", odb_path, raw_csv]
    run_cmd(cmd, cwd=work_dir, env=abaqus_env(scratch_dir))


def postprocess(run, raw_csv, out_dir, reference, plot):
    cmd = [sys.executable, POSTPROC, "--raw", raw_csv, "--run", str(run), "--out-dir", out_dir]
    if reference:
        cmd += ["--reference", reference]
    if plot:
        cmd += ["--plot"]
    run_cmd(cmd, cwd=out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="run id(s), e.g. '1' or '1,6,31' or '1-5'")
    ap.add_argument("--inp-dir", default=DEFAULT_INP_DIR)
    ap.add_argument("--ref-dir", default=DEFAULT_REF_DIR)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"),
                    help="output root; each run solves in <work-dir>/run_<n>")
    ap.add_argument("--abaqus", default=None, help="path to abaqus launcher (default abq2025.bat)")
    ap.add_argument("--scratch-dir", default=None,
                    help="Abaqus scratch/temp root; each run uses <scratch-dir>/run_<n>")
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--compare", action="store_true", help="compare against delivered reference")
    ap.add_argument("--plot", action="store_true", help="save overlay plot (implies --compare data load)")
    ap.add_argument("--skip-solve", action="store_true", help="reuse an existing .odb")
    ap.add_argument("--skip-extract", action="store_true", help="reuse an existing raw csv")
    args = ap.parse_args()

    abaqus = find_abaqus(args.abaqus)
    runs = parse_runs(args.runs)
    print("Abaqus launcher : %s" % abaqus)
    print("Runs            : %s" % runs)
    print("CPUs per job    : %d" % args.cpus)

    os.makedirs(args.work_dir, exist_ok=True)

    for run in runs:
        job = "HoodImpact_%d" % run
        inp_path = os.path.join(args.inp_dir, "%s.inp" % job)
        if not os.path.exists(inp_path):
            print("!! skipping run %d: %s not found" % (run, inp_path))
            continue

        run_dir = os.path.join(args.work_dir, "run_%d" % run)
        os.makedirs(run_dir, exist_ok=True)

        odb_path = os.path.join(run_dir, "%s.odb" % job)
        raw_csv = os.path.join(run_dir, "%s_raw_acc.csv" % job)
        reference = os.path.join(args.ref_dir, "%s_SAE1000_interp1000.csv" % job) \
            if (args.compare or args.plot) else None

        print("\n" + "=" * 72)
        print("RUN %d" % run)
        print("=" * 72)

        run_scratch = os.path.join(args.scratch_dir, "run_%d" % run) \
            if args.scratch_dir else None

        if not args.skip_solve:
            solve(abaqus, inp_path, job, run_dir, args.cpus, run_scratch)
        if not args.skip_extract:
            extract(abaqus, odb_path, raw_csv, run_dir, run_scratch)
        postprocess(run, raw_csv, run_dir, reference, args.plot)

    print("\nDone. Per-run outputs are under %s" % args.work_dir)


if __name__ == "__main__":
    main()
