"""
batch_verify_60.py
==================
Batch verification of the delivered HoodImpact_60_IndustryLike dataset:
solve all 60 delivered decks with Abaqus/Explicit, extract the headform
acceleration, SAE1000-filter it, and compare each run against the delivered
`output_history_acc` reference.

Parallelisation: N concurrent SINGLE-CORE jobs (default --lanes 6), not one
multi-core job. Reason (discovered 2026-07-05): this machine's 3DEXPERIENCE
DSLS licence crashes on any cpus>1 checkout --
    "Not enough parallel licenses for a 6 cpu job"
    "Exception: DriverLM: can't parse host/port from umbrella"
i.e. the *parallel-token* checkout path cannot parse the comma-separated
multi-host licence-server string. The single-core checkout path works (proven
by the verified run 1) and the EDU token pool is large, so N independent
cpus=1 jobs are both licence-safe and faster than one N-core job anyway.

Robustness notes:
    * abq2025.bat returns EXIT CODE 0 even when the analysis fails, so success
      is detected from artefacts instead: .sta must contain
      "THE ANALYSIS HAS COMPLETED SUCCESSFULLY", and the extraction must
      produce the raw csv.
    * up to 3 solve attempts per run (licence-server hiccups), 90 s apart
    * job starts are throttled (>=15 s apart) to avoid licence-checkout races
    * per-run Abaqus console output -> <run_dir>/solve_console.txt
    * incremental metrics CSV (work/verification_metrics.csv), resume-safe:
      runs already marked ok are skipped on restart
    * DISK CLEANUP: each solve leaves ~865 MB (.odb .abq .pac .stt .mdl .res);
      60 runs = ~52 GB which does NOT fit on this disk. Big solver files are
      deleted after a successful extraction; the raw acceleration csv (full-
      precision solver output), the SAE1000 csv, the overlay png and the small
      text logs are kept, so post-processing can be redone without re-solving.

Usage (from d:\\Papers\\Hood_Impact\\Code):
    python -u abaqus_scripts\\batch_verify_60.py --lanes 6
    python -u abaqus_scripts\\batch_verify_60.py --runs 5-10 --lanes 2
"""
import os
import csv
import sys
import time
import argparse
import datetime
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_hood_impact as rhi
import postprocess_acc as pp

BIG_EXTENSIONS = (".odb", ".abq", ".pac", ".stt", ".mdl", ".res", ".sel",
                  ".lck", ".023")
METRIC_FIELDS = ["run", "status", "cpus", "solve_s", "n_raw_points",
                 "rmse_g", "peak_g_mine", "peak_g_ref", "peak_pct_err",
                 "hic_mine", "hic_ref", "hic_pct_err", "finished_at", "note"]

SOLVE_ATTEMPTS = 3
RETRY_WAIT_S = 90
LAUNCH_SPACING_S = 15

log_lock = threading.Lock()
metrics_lock = threading.Lock()
postproc_lock = threading.Lock()   # matplotlib is not thread-safe
launch_lock = threading.Lock()
_last_launch = [0.0]


class Log(object):
    """print to stdout AND append to a log file (thread-safe, line-flushed)."""
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg=""):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = "[%s] %s" % (stamp, msg)
        with log_lock:
            print(line, flush=True)
            self.f.write(line + "\n")


def load_done(metrics_csv):
    done = set()
    if os.path.exists(metrics_csv):
        with open(metrics_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(int(row["run"]))
    return done


def append_metrics(metrics_csv, row):
    with metrics_lock:
        new = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def cleanup_big_files(run_dir, log, keep_odb=False):
    freed = 0
    for name in os.listdir(run_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in BIG_EXTENSIONS and not (keep_odb and ext == ".odb"):
            path = os.path.join(run_dir, name)
            try:
                freed += os.path.getsize(path)
                os.remove(path)
            except OSError:
                pass
    if freed:
        log("   cleanup: freed %.0f MB in %s" % (freed / 1e6, os.path.basename(run_dir)))


def run_logged(cmd, cwd, console_path):
    """Run a command with stdout+stderr appended to a per-run console file."""
    with open(console_path, "a") as f:
        f.write("\n$ %s\n" % " ".join(cmd))
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def throttled_launch_slot():
    """Ensure >=LAUNCH_SPACING_S between successive Abaqus job starts."""
    with launch_lock:
        wait = _last_launch[0] + LAUNCH_SPACING_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.time()


def sta_completed(sta_path):
    if not os.path.exists(sta_path):
        return False
    try:
        with open(sta_path, errors="ignore") as f:
            return "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in f.read()
    except OSError:
        return False


def console_tail(console_path, n=5):
    try:
        with open(console_path, errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        return " | ".join(lines[-n:])
    except OSError:
        return ""


def process_one(run, args, abaqus, log):
    """solve (cpus=1) -> extract -> postprocess -> compare -> plot -> cleanup"""
    job = "HoodImpact_%d" % run
    inp_path = os.path.join(args.inp_dir, "%s.inp" % job)
    if not os.path.exists(inp_path):
        return {"run": run, "status": "missing_inp", "note": inp_path}

    run_dir = os.path.join(args.work_dir, "run_%d" % run)
    os.makedirs(run_dir, exist_ok=True)
    odb_path = os.path.join(run_dir, "%s.odb" % job)
    raw_csv = os.path.join(run_dir, "%s_raw_acc.csv" % job)
    sta_path = os.path.join(run_dir, "%s.sta" % job)
    console = os.path.join(run_dir, "solve_console.txt")
    reference = os.path.join(args.ref_dir, "%s_SAE1000_interp1000.csv" % job)

    row = {"run": run, "cpus": args.cpus, "note": ""}

    # --- solve --------------------------------------------------------------
    solve_cmd = [abaqus, "job=%s" % job, "input=%s" % inp_path,
                 "cpus=%d" % args.cpus, "double=both", "ask_delete=OFF",
                 "interactive"]
    t0 = time.time()
    solved = sta_completed(sta_path) and os.path.exists(odb_path)  # leftover?
    if solved:
        log("   run %d: reusing existing completed solve" % run)
    for attempt in range(1, SOLVE_ATTEMPTS + 1):
        if solved:
            break
        # stale lock files from a killed attempt block the next one
        lck = os.path.join(run_dir, "%s.lck" % job)
        if os.path.exists(lck):
            os.remove(lck)
        throttled_launch_slot()
        log("   run %d: solve attempt %d starting" % (run, attempt))
        run_logged(solve_cmd, run_dir, console)
        solved = sta_completed(sta_path) and os.path.exists(odb_path)
        if not solved:
            log("   run %d: solve attempt %d FAILED -- %s"
                % (run, attempt, console_tail(console)))
            if attempt < SOLVE_ATTEMPTS:
                time.sleep(RETRY_WAIT_S)
    if not solved:
        cleanup_big_files(run_dir, log)
        row.update(status="solve_failed",
                   note=console_tail(console)[:200])
        return row
    row["solve_s"] = round(time.time() - t0, 1)

    # --- extract raw acceleration from the odb --------------------------------
    extract_cmd = [abaqus, "python", rhi.EXTRACTOR, "--", odb_path, raw_csv]
    run_logged(extract_cmd, run_dir, console)
    if not os.path.exists(raw_csv):
        cleanup_big_files(run_dir, log, keep_odb=True)  # keep odb for debugging
        row.update(status="extract_failed", note=console_tail(console)[:200])
        return row

    # --- postprocess + compare + plot ----------------------------------------
    try:
        with postproc_lock:
            df = pp.process_raw(raw_csv)
            out_csv = os.path.join(run_dir, "%s_SAE1000_interp1000.csv" % job)
            df.to_csv(out_csv, index=False)
            m = pp.compare(df, reference)
            pp.overlay_plot(df, reference,
                            os.path.join(run_dir, "%s_acc_overlay.png" % job), run)
        row["n_raw_points"] = sum(1 for _ in open(raw_csv)) - 1
        for k in ("rmse_g", "peak_g_mine", "peak_g_ref", "peak_pct_err",
                  "hic_mine", "hic_ref", "hic_pct_err"):
            row[k] = round(m[k], 4)
        row["status"] = "ok"
    except Exception as e:
        row.update(status="postproc_failed", note=str(e)[:200])
    finally:
        cleanup_big_files(run_dir, log)

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="1-60")
    ap.add_argument("--lanes", type=int, default=6,
                    help="number of concurrent single-core Abaqus jobs")
    ap.add_argument("--cpus", type=int, default=1,
                    help="cpus per job (leave 1: cpus>1 crashes the DSLS "
                         "parallel-licence checkout on this machine)")
    ap.add_argument("--inp-dir", default=rhi.DEFAULT_INP_DIR)
    ap.add_argument("--ref-dir", default=rhi.DEFAULT_REF_DIR)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--abaqus", default=None)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    log = Log(os.path.join(args.work_dir, "batch_log.txt"))
    metrics_csv = os.path.join(args.work_dir, "verification_metrics.csv")

    abaqus = rhi.find_abaqus(args.abaqus)
    runs = rhi.parse_runs(args.runs)
    done = load_done(metrics_csv)
    todo = [r for r in runs if r not in done]

    log("=" * 70)
    log("BATCH VERIFICATION: %d runs, %d lanes x cpus=%d"
        % (len(runs), args.lanes, args.cpus))
    log("already ok (skipped): %s" % (sorted(done & set(runs)) or "none"))
    log("to solve            : %d runs" % len(todo))
    log("=" * 70)

    t_start = time.time()
    n_done = n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        futures = {pool.submit(process_one, run, args, abaqus, log): run
                   for run in todo}
        for fut in as_completed(futures):
            run = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"run": run, "status": "crashed", "note": str(e)[:200]}
            row["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            append_metrics(metrics_csv, row)
            n_done += 1

            if row["status"] == "ok":
                n_ok += 1
                elapsed = time.time() - t_start
                rate = elapsed / n_done
                eta = datetime.datetime.now() + datetime.timedelta(
                    seconds=(len(todo) - n_done) * rate)
                log("RUN %d ok: solve=%.0fs rmse=%.3fg peak_err=%.2f%% "
                    "hic_err=%.2f%%  [%d/%d, ETA ~%s]"
                    % (run, row.get("solve_s", -1), row["rmse_g"],
                       row["peak_pct_err"], row["hic_pct_err"],
                       n_done, len(todo), eta.strftime("%H:%M")))
            else:
                n_fail += 1
                log("RUN %d FAILED: %s (%s)"
                    % (run, row["status"], row.get("note", "")))

            if n_done % 10 == 0 or n_done == len(todo):
                log("MILESTONE %d/%d done (%d ok, %d failed)"
                    % (n_done, len(todo), n_ok, n_fail))

    log("BATCH COMPLETE: %d ok, %d failed. Metrics -> %s"
        % (n_ok, n_fail, metrics_csv))


if __name__ == "__main__":
    main()
