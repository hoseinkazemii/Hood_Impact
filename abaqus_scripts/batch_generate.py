"""
batch_generate.py
=================
Production driver for the 600-sample dataset: solve the generated
HoodImpact_<run>.inp decks with Abaqus/Explicit, extract the headform
acceleration, SAE1000-filter it, compute HIC15, and keep only the small
result files. There is no reference comparison for the new runs.

Parallelisation: N concurrent SINGLE-CORE solves (--lanes). On Delta the
constraint is the FlexNet license pool (65 "abaqus" tokens; one cpus=1 job
checks out 5 tokens -> 13 lanes max; default 12 leaves headroom for other
license users) and the CPU-hour budget (single-core solves are the most
core-hour-efficient way to produce samples).

Robustness:
    * success detected from artefacts, not exit codes: .sta must contain
      "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" and the extraction must
      produce the raw csv
    * up to 3 solve attempts per run (license hiccups), 90 s apart
    * job starts throttled >=15 s apart to avoid license-checkout races
    * per-run Abaqus console -> <run_dir>/solve_console.txt
    * incremental metrics CSV, resume-safe: runs marked ok are skipped, so
      the same command can simply be resubmitted after a walltime kill
    * optional --deadline (the Slurm job's end): a run is only started when
      it is expected to finish before it (90th-percentile solve time so far,
      +10%, +10 min), so a job ends cleanly instead of being killed mid-solve;
      the runs left over are solved by the next job
    * big solver files (.odb .abq .pac .stt ...) deleted after successful
      extraction; kept: raw acceleration csv, SAE1000 csv, logs
    * a run that finished and published its csv but got no metrics row (its
      job died first) is adopted from its raw csv instead of solved again
    * logging never raises (a stale Slurm .out on NFS once stopped a job), and
      if the main loop dies the queued runs are cancelled, not started blind
    * --runs-file: solve the runs in the file's order (the campaign's priority)

Outputs (dataset layout mirrors the delivered 60-sample set):
    <out-dir>/output_history_acc/HoodImpact_<run>_SAE1000_interp1000.csv
    <work-dir>/run_<n>/HoodImpact_<n>_raw_acc.csv       (full-precision raw)
    <work-dir>/generation_metrics.csv                    (per-run status/HIC)

Usage on Delta (inside the Slurm job; see delta/hood600.sbatch):
    python -u abaqus_scripts/batch_generate.py \
        --inp-dir  <...>/HoodImpact_600_EuroNCAP/inp_files \
        --work-dir <...>/work --scratch-dir <...>/abaqus_scratch \
        --out-dir <...>/HoodImpact_600_EuroNCAP \
        --runs 1-600 --lanes 12
"""
import os
import csv
import sys
import time
import shutil
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
                  ".lck", ".023", ".prt", ".sim", ".dat")
METRIC_FIELDS = ["run", "status", "cpus", "solve_s", "n_raw_points",
                 "peak_g", "hic15", "finished_at", "note"]

SOLVE_ATTEMPTS = 3
RETRY_WAIT_S = 90
LAUNCH_SPACING_S = 15
DEFAULT_RUN_S = 4 * 3600      # expected run time until 5 solve times are known
END_MARGIN_S = 600            # extraction, post-processing and job wrap-up

log_lock = threading.Lock()
metrics_lock = threading.Lock()
postproc_lock = threading.Lock()
launch_lock = threading.Lock()
_last_launch = [0.0]


class Log(object):
    """Console + file log that never raises: a log write that fails (e.g. the
    Slurm .out deleted or replaced from another node on NFS -> ESTALE) must not
    stop the solves; job 23838174 lost 4.5 h that way."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg=""):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = "[%s] %s" % (stamp, msg)
        with log_lock:
            try:
                print(line, flush=True)
            except (OSError, ValueError):
                pass
            try:
                self.f.write(line + "\n")
            except (OSError, ValueError):
                try:                       # reopen once (a stale handle stays stale)
                    self.f = open(self.path, "a", buffering=1)
                    self.f.write(line + "\n")
                except (OSError, ValueError):
                    pass


def load_done(metrics_csv):
    done = set()
    if os.path.exists(metrics_csv):
        with open(metrics_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(int(row["run"]))
    return done


class RunClock(object):
    """Says whether a run started now is expected to finish before the deadline."""

    def __init__(self, deadline, metrics_csv):
        self.deadline = deadline
        self.times = []
        self.lock = threading.Lock()
        if os.path.exists(metrics_csv):
            with open(metrics_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "ok" and row.get("solve_s"):
                        self.times.append(float(row["solve_s"]))

    def add(self, solve_s):
        with self.lock:
            self.times.append(float(solve_s))

    def expected_s(self):
        with self.lock:
            if len(self.times) < 5:
                return DEFAULT_RUN_S
            t = sorted(self.times)
            return 1.1 * t[int(0.9 * (len(t) - 1))] + END_MARGIN_S

    def fits(self):
        return self.deadline is None or time.time() + self.expected_s() <= self.deadline


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
    if freed > 1e8:
        log("   cleanup: freed %.0f MB in %s" % (freed / 1e6, os.path.basename(run_dir)))


def cleanup_scratch_dir(scratch_dir, log):
    if scratch_dir and os.path.isdir(scratch_dir):
        try:
            shutil.rmtree(scratch_dir)
        except OSError as e:
            log("   cleanup: could not remove scratch %s (%s)" % (scratch_dir, e))


def run_logged(cmd, cwd, console_path, env=None):
    with open(console_path, "a") as f:
        f.write("\n$ %s\n" % " ".join(cmd))
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)


def throttled_launch_slot():
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


def adopt_finished(run, args, log):
    """A run that completed and published its result but got no metrics row
    (its job died before recording it): rebuild the row instead of solving again."""
    job = "HoodImpact_%d" % run
    run_dir = os.path.join(args.work_dir, "run_%d" % run)
    raw_csv = os.path.join(run_dir, "%s_raw_acc.csv" % job)
    published = os.path.join(args.out_dir, "output_history_acc", "%s_SAE1000_interp1000.csv" % job)
    if not (sta_completed(os.path.join(run_dir, "%s.sta" % job))
            and os.path.exists(raw_csv) and os.path.exists(published)):
        return None
    try:
        with postproc_lock:
            df = pp.process_raw(raw_csv)
        t = df["Time"].to_numpy(float)
        g = df["A(in g)"].to_numpy(float)
        row = {"run": run, "status": "ok", "cpus": args.cpus, "solve_s": "",
               "n_raw_points": sum(1 for _ in open(raw_csv)) - 1,
               "peak_g": round(float(g.max()), 4), "hic15": round(pp.compute_hic(t, g), 4),
               "note": "adopted: finished earlier, not recorded"}
        out_csv = os.path.join(run_dir, "%s_SAE1000_interp1000.csv" % job)
        df.to_csv(out_csv, index=False)
        shutil.copy2(out_csv, published)      # republish from the raw csv (same result)
    except Exception as e:
        log("   run %d: could not adopt the earlier result (%s); solving again" % (run, e))
        return None
    log("   run %d: adopted the earlier finished result" % run)
    return row


def process_one(run, args, abaqus, log, clock):
    """solve (cpus=1) -> extract -> SAE1000 + HIC -> publish csv -> cleanup"""
    job = "HoodImpact_%d" % run
    inp_path = os.path.join(args.inp_dir, "%s.inp" % job)
    if not os.path.exists(inp_path):
        return {"run": run, "status": "missing_inp", "note": inp_path}
    adopted = adopt_finished(run, args, log)
    if adopted:
        return adopted
    if not clock.fits():
        return {"run": run, "status": "deferred"}

    run_dir = os.path.join(args.work_dir, "run_%d" % run)
    os.makedirs(run_dir, exist_ok=True)
    run_scratch = os.path.join(args.scratch_dir, "run_%d" % run) \
        if args.scratch_dir else None
    if run_scratch:
        os.makedirs(run_scratch, exist_ok=True)
    odb_path = os.path.join(run_dir, "%s.odb" % job)
    raw_csv = os.path.join(run_dir, "%s_raw_acc.csv" % job)
    sta_path = os.path.join(run_dir, "%s.sta" % job)
    console = os.path.join(run_dir, "solve_console.txt")

    row = {"run": run, "cpus": args.cpus, "note": ""}

    # --- solve ---------------------------------------------------------------
    solve_cmd = [abaqus, "job=%s" % job, "input=%s" % inp_path,
                 "cpus=%d" % args.cpus, "double=both", "ask_delete=OFF",
                 "interactive"]
    if run_scratch:
        solve_cmd.insert(-1, "scratch=%s" % run_scratch)
    t0 = time.time()
    solved = sta_completed(sta_path) and os.path.exists(odb_path)
    if solved:
        log("   run %d: reusing existing completed solve" % run)
    for attempt in range(1, SOLVE_ATTEMPTS + 1):
        if solved:
            break
        if not clock.fits():
            cleanup_big_files(run_dir, log)
            cleanup_scratch_dir(run_scratch, log)
            return {"run": run, "status": "deferred"}
        lck = os.path.join(run_dir, "%s.lck" % job)
        if os.path.exists(lck):
            os.remove(lck)
        throttled_launch_slot()
        log("   run %d: solve attempt %d starting" % (run, attempt))
        run_logged(solve_cmd, run_dir, console, env=rhi.abaqus_env(run_scratch))
        solved = sta_completed(sta_path) and os.path.exists(odb_path)
        if not solved:
            log("   run %d: solve attempt %d FAILED -- %s"
                % (run, attempt, console_tail(console)))
            if attempt < SOLVE_ATTEMPTS:
                time.sleep(RETRY_WAIT_S)
    if not solved:
        cleanup_big_files(run_dir, log)
        cleanup_scratch_dir(run_scratch, log)
        row.update(status="solve_failed", note=console_tail(console)[:200])
        return row
    row["solve_s"] = round(time.time() - t0, 1)

    # --- extract raw acceleration ---------------------------------------------
    extract_cmd = [abaqus, "python", rhi.EXTRACTOR, "--", odb_path, raw_csv]
    run_logged(extract_cmd, run_dir, console, env=rhi.abaqus_env(run_scratch))
    if not os.path.exists(raw_csv):
        cleanup_big_files(run_dir, log, keep_odb=True)
        cleanup_scratch_dir(run_scratch, log)
        row.update(status="extract_failed", note=console_tail(console)[:200])
        return row

    # --- SAE1000 filter + HIC + publish ----------------------------------------
    try:
        with postproc_lock:
            df = pp.process_raw(raw_csv)
        out_csv = os.path.join(run_dir, "%s_SAE1000_interp1000.csv" % job)
        df.to_csv(out_csv, index=False)
        acc_dir = os.path.join(args.out_dir, "output_history_acc")
        os.makedirs(acc_dir, exist_ok=True)
        shutil.copy2(out_csv, os.path.join(acc_dir, os.path.basename(out_csv)))

        t = df["Time"].to_numpy(float)
        g = df["A(in g)"].to_numpy(float)
        row["n_raw_points"] = sum(1 for _ in open(raw_csv)) - 1
        row["peak_g"] = round(float(g.max()), 4)
        row["hic15"] = round(pp.compute_hic(t, g), 4)
        row["status"] = "ok"
    except Exception as e:
        row.update(status="postproc_failed", note=str(e)[:200])
    finally:
        cleanup_big_files(run_dir, log)
        cleanup_scratch_dir(run_scratch, log)

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="1-600", help="run numbers, solved in increasing order")
    ap.add_argument("--runs-file", default=None,
                    help="file with one run number per line, solved in that order "
                         "(replaces --runs)")
    ap.add_argument("--lanes", type=int, default=12,
                    help="concurrent single-core Abaqus jobs (12 x 5 = 60 of "
                         "the 65 license tokens)")
    ap.add_argument("--cpus", type=int, default=1,
                    help="cpus per solve (keep 1: most core-hour-efficient, "
                         "and lanes are sized for 5-token serial checkouts)")
    ap.add_argument("--inp-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--scratch-dir", default=None,
                    help="Abaqus scratch/temp root; each run uses "
                         "<scratch-dir>/run_<n>")
    ap.add_argument("--out-dir", required=True,
                    help="dataset root; SAE1000 csvs land in "
                         "<out-dir>/output_history_acc/")
    ap.add_argument("--abaqus", default=None,
                    help="abaqus launcher (default: 'abaqus' on PATH)")
    ap.add_argument("--deadline", type=float, default=None,
                    help="unix time by which every run must be finished (the "
                         "Slurm job's end); runs that would not finish in time "
                         "are left for the next job")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log = Log(os.path.join(args.work_dir, "batch_log.txt"))
    metrics_csv = os.path.join(args.work_dir, "generation_metrics.csv")

    abaqus = rhi.find_abaqus(args.abaqus)
    if args.runs_file:
        with open(args.runs_file) as f:
            runs = [int(s) for s in f.read().split()]
        if len(set(runs)) != len(runs):
            raise SystemExit("%s lists a run twice" % args.runs_file)
    else:
        runs = rhi.parse_runs(args.runs)
    done = load_done(metrics_csv)
    todo = [r for r in runs if r not in done]
    clock = RunClock(args.deadline, metrics_csv)

    log("=" * 70)
    log("BATCH GENERATION: %d runs requested, %d lanes x cpus=%d"
        % (len(runs), args.lanes, args.cpus))
    log("abaqus  : %s" % abaqus)
    log("already ok (skipped): %d" % len(done & set(runs)))
    log("to solve            : %d runs" % len(todo))
    if args.deadline is not None:
        log("deadline            : %s (expected run time now %.0f min)"
            % (datetime.datetime.fromtimestamp(args.deadline).strftime("%m-%d %H:%M"),
               clock.expected_s() / 60))
    log("=" * 70)

    t_start = time.time()
    n_done = n_ok = n_fail = n_deferred = 0
    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        # workers take the runs in this order (the solve order)
        futures = {pool.submit(process_one, run, args, abaqus, log, clock): run
                   for run in todo}
        try:
            for fut in as_completed(futures):
                run = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {"run": run, "status": "crashed", "note": str(e)[:200]}
                if row["status"] == "deferred":
                    n_deferred += 1
                    continue
                row["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                try:
                    append_metrics(metrics_csv, row)
                except OSError as e:
                    log("WARNING: could not record run %d in %s (%s); the next job adopts it"
                        % (run, metrics_csv, e))
                n_done += 1
                report(log, row, run, n_done, len(todo), t_start, clock)
                if row["status"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    log("MILESTONE %d/%d done (%d ok, %d failed)"
                        % (n_done, len(todo), n_ok, n_fail))
        except BaseException:
            # never let the queued runs start without anyone recording them
            for f in futures:
                f.cancel()
            raise

    if n_deferred:
        log("DEADLINE: %d runs not started (expected %.0f min each, too little "
            "time left); the next job solves them" % (n_deferred, clock.expected_s() / 60))
    log("BATCH COMPLETE: %d ok, %d failed. Metrics -> %s"
        % (n_ok, n_fail, metrics_csv))


def report(log, row, run, n_done, n_todo, t_start, clock):
    if row["status"] == "ok":
        if row.get("solve_s") not in (None, ""):
            clock.add(row["solve_s"])
        rate = (time.time() - t_start) / n_done
        eta = datetime.datetime.now() + datetime.timedelta(seconds=(n_todo - n_done) * rate)
        log("RUN %d ok: solve=%ss peak=%.1fg HIC15=%.0f  [%d/%d, ETA ~%s]%s"
            % (run, row.get("solve_s", "?"), row["peak_g"], row["hic15"], n_done, n_todo,
               eta.strftime("%m-%d %H:%M"), "  (%s)" % row["note"] if row.get("note") else ""))
    else:
        log("RUN %d FAILED: %s (%s)" % (run, row["status"], row.get("note", "")))


if __name__ == "__main__":
    main()
