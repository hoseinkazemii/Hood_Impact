"""
cpus_scaling_test.py
====================
One Delta job (delta/hood_cpus_test.sbatch, ~1.5 h) that answers, before the
local-edit campaign leaves single-core lanes:

  1. tokens  -- how many FlexNet tokens does one cpus=N solve check out?
                Abaqus' rule is floor(5 * N**0.422) "abaqus" tokens
                (1->5, 2->6, 4->8, 8->12, 16->16, 32->21), so 65 tokens buy far
                more than 13 cores when each solve uses several. The academic
                file also carries a 'parallel' feature (16384), so the real
                count is read from the license server, not assumed.
  2. speed   -- how fast is one solve on N cores, alone and when L of them
                share the node (the campaign layout L lanes x N cores)?
  3. noise   -- does solving on N cores keep what the dataset is for: the
                difference between an edited design and D6 at the same impact
                location? N cores means N domains, a different floating-point
                order, and contact/plasticity/buckling can grow that roundoff.
                The pilot runs (D6, N1, N2 at 4 locations, all finished on 1
                core) are solved again on N cores and the difference curves
                (design - D6) compared with the 1-core ones; one run is solved
                twice on N cores to check the result repeats bit for bit.

It picks the layout (lanes x cpus) with the highest measured throughput that
fits the token pool and the cores of this job, and writes it to
<out-dir>/layout.env with the noise verdict. hood_local_edits.sbatch uses it
once copied to Data/HoodImpact_LocalEdits/campaign_layout.env.

Phases (each solve in its own folder under --work-dir; nothing touches the
campaign's work folder, metrics or published curves):
  solo    the deck of one finished run, cut to the first part of the step
          (longer for more cores, so every solve analyses for about a minute;
          field-output interval scaled with it), solved once per cpus count,
          one after another, so each solve has the license and the cores to
          itself. The .sta is polled every second for the increment rate, and
          the license server is asked which tokens this node holds meanwhile.
  layout  the best --layouts candidates predicted from solo: all lanes of the
          layout solve a short deck at once; the rate is taken over the time
          all lanes were analysing together (memory-bandwidth contention).
  noise   the --noise-runs (default: the pilot runs) in full on the chosen
          layout, extracted and SAE1000 filtered exactly like the campaign, and
          compared run by run and pair by pair with their published 1-core
          curves. Their solve times are the campaign's real per-run time.
A solve that would not finish before --deadline is not started; the phases can
be run later on their own, e.g. --phases noise (it reads results.json).

Outputs in --out-dir: report.txt, results.json, layout.env, lmstat/, and
noise/ (noise_runs.csv, noise_pairs.csv, noise_pairs.png and the N-core curves
HoodImpact_<run>_cpus<N>[_repeat]_SAE1000_interp1000.csv).
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import run_hood_impact as rhi       # noqa: E402
import postprocess_acc as pp        # noqa: E402
import batch_generate as bg         # noqa: E402

DEFAULT_DATA = os.path.join(CODE_ROOT, "Data", "HoodImpact_LocalEdits")
DEFAULT_SERVER = "27000@130.126.241.64"
POLL_S = 1.0
LAUNCH_GAP_S = 2.0
EXTRACT_GUESS_S = 90.0          # extraction + SAE1000 per run, until the noise phase measures it
# noise verdict (suggested): for every near/twin pair the N-core difference curve
# must match the 1-core one this well, and the HIC change by this many points
PAIR_R2_MIN = 0.95
PAIR_DHIC_PP_MAX = 0.5

# numeric .sta line: increment, step time, total time [, wall hh:mm:ss, ...]
STA_LINE = re.compile(r"^\s*(\d+)\s+-?\d*\.\d+E[-+]\d+\s+-?\d*\.\d+E[-+]\d+(.*)$", re.I)
WALL = re.compile(r"\b(\d+):(\d\d):(\d\d)\b")
USERS_OF = re.compile(r"Users of (\S+?):\s*\(Total of (\d+) licenses? issued;\s*Total of (\d+) licenses? in use\)")
N_LICENSES = re.compile(r",\s*(\d+) licenses?\s*$")


def tokens_rule(n):
    """Abaqus' documented token count for an n-core analysis."""
    return int(5 * n ** 0.422 + 1e-9)


# ----------------------------------------------------------------------------
# logging / report
# ----------------------------------------------------------------------------
class Report(object):
    def __init__(self, path):
        self.path = path
        self.lines = []

    def __call__(self, msg=""):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            print("[%s] %s" % (stamp, msg), flush=True)
        except (OSError, ValueError):
            pass
        self.lines.append(msg)

    def save(self):
        try:
            with open(self.path, "w") as f:
                f.write("\n".join(self.lines) + "\n")
        except OSError:
            pass


# ----------------------------------------------------------------------------
# deck, .sta, license server
# ----------------------------------------------------------------------------
def read_deck(path):
    with open(path, newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    return text.split(nl), nl


def step_card(lines):
    """(line index, dt, period) of the *DYNAMIC, EXPLICIT data line."""
    for i, line in enumerate(lines):
        if line.upper().startswith("*DYNAMIC"):
            j = i + 1
            while lines[j].startswith("**"):
                j += 1
            f = [s.strip() for s in lines[j].split(",")]
            return j, float(f[0]), float(f[1])
    raise ValueError("no *DYNAMIC card")


def deck_increments(deck):
    lines, _ = read_deck(deck)
    _, dt, period = step_card(lines)
    return int(round(period / dt)), dt, period


def write_short_deck(deck, out, frac):
    """Copy of `deck` with the step cut to frac of its length and the field-output
    interval cut with it (same output per increment). Returns its increment count."""
    lines, nl = read_deck(deck)
    j, dt, period = step_card(lines)
    f = lines[j].split(",")
    f[1] = " %.6g" % (period * frac)
    lines[j] = ",".join(f)
    for k, line in enumerate(lines):
        if line.upper().startswith("*OUTPUT, FIELD"):
            lines[k] = re.sub(r"(NUMBER INTERVAL\s*=\s*)(\d+)",
                              lambda m: m.group(1) + str(max(1, int(round(int(m.group(2)) * frac)))),
                              line, flags=re.I)
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        fh.write(nl.join(lines))
    os.replace(tmp, out)
    return int(round(period * frac / dt))


def read_sta(path):
    """[(increment, wall_s or None)] of the numeric .sta lines."""
    rows = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                m = STA_LINE.match(line)
                if m:
                    w = WALL.search(m.group(2))
                    wall = int(w.group(1)) * 3600 + int(w.group(2)) * 60 + int(w.group(3)) if w else None
                    rows.append((int(m.group(1)), wall))
    except OSError:
        pass
    return rows


def parse_lmstat(text, host):
    """{feature: {issued, in_use, mine}}; mine = tokens held by checkouts from `host`."""
    feats, cur = {}, None
    host_re = re.compile(r"\b%s\b" % re.escape(host), re.I) if host else None
    for line in text.splitlines():
        m = USERS_OF.search(line)
        if m:
            cur = m.group(1)
            feats[cur] = {"issued": int(m.group(2)), "in_use": int(m.group(3)), "mine": 0}
            continue
        if cur and host_re and " start " in line and host_re.search(line):
            n = N_LICENSES.search(line)
            feats[cur]["mine"] += int(n.group(1)) if n else 1
    return feats


class License(object):
    def __init__(self, abaqus, server, out_dir):
        self.abaqus, self.server, self.out_dir = abaqus, server, out_dir
        self.host = socket.gethostname().split(".")[0]
        self.n = 0

    def snapshot(self, label):
        """Query the server; the raw text is kept in <out>/lmstat/<n>_<label>.txt."""
        try:
            p = subprocess.run([self.abaqus, "licensing", "lmstat", "-a", "-c", self.server],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=90)
            text = p.stdout or ""
        except (OSError, subprocess.SubprocessError) as e:
            text = "lmstat failed: %s" % e
        self.n += 1
        d = os.path.join(self.out_dir, "lmstat")
        os.makedirs(d, exist_ok=True)
        try:
            with open(os.path.join(d, "%02d_%s.txt" % (self.n, label)), "w") as f:
                f.write(text)
        except OSError:
            pass
        feats = parse_lmstat(text, self.host)
        return feats or None


def tokens_held(during, base):
    """Tokens this test holds per feature: this node's checkouts, else the rise in use."""
    if not during:
        return {}, "none"
    mine = {f: v["mine"] for f, v in during.items() if v["mine"] > 0}
    if mine:
        return mine, "lmstat (this node)"
    rise = {}
    for f, v in during.items():
        b = base.get(f, {}).get("in_use", 0) if base else 0
        if v["in_use"] > b:
            rise[f] = v["in_use"] - b
    return rise, "lmstat (rise in use)" if rise else "none"


def campaign_running():
    try:
        p = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", "hood_local_edits",
                            "-t", "RUNNING", "-o", "%A"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60)
        return p.stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []


# ----------------------------------------------------------------------------
# one Abaqus solve, monitored
# ----------------------------------------------------------------------------
class Solve(object):
    def __init__(self, abaqus, tag, deck, cpus, mode, work_dir):
        self.abaqus, self.tag, self.deck, self.cpus, self.mode = abaqus, tag, deck, cpus, mode
        self.job = "cpus_test_%s" % tag
        self.dir = os.path.join(work_dir, tag)
        self.scratch = os.path.join(self.dir, "scratch")
        self.sta = os.path.join(self.dir, self.job + ".sta")
        self.odb = os.path.join(self.dir, self.job + ".odb")
        self.console_path = os.path.join(self.dir, "solve_console.txt")
        self.proc = None
        self.t0 = self.t_end = self.last_change = None
        self.samples = []            # (poll time, increment), whenever the increment changed
        self.status = "pending"

    def start(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.scratch)
        cmd = [self.abaqus, "job=%s" % self.job, "input=%s" % self.deck, "cpus=%d" % self.cpus,
               "double=both", "ask_delete=OFF", "scratch=%s" % self.scratch]
        if self.cpus > 1 and self.mode:
            cmd.append("mp_mode=%s" % self.mode)
        cmd.append("interactive")
        self._console = open(self.console_path, "w")
        self._console.write("$ %s\n" % " ".join(cmd))
        self._console.flush()
        self.t0 = self.last_change = time.time()
        kw = {"start_new_session": True} if os.name == "posix" else {}
        self.proc = subprocess.Popen(cmd, cwd=self.dir, stdout=self._console, stderr=subprocess.STDOUT,
                                     env=rhi.abaqus_env(self.scratch), **kw)
        self.status = "running"

    def _sample(self, now):
        rows = read_sta(self.sta)
        if rows:
            inc = rows[-1][0]
            if not self.samples or inc != self.samples[-1][1]:
                self.samples.append((now, inc))
                self.last_change = now

    def poll(self, now):
        if self.status != "running":
            return
        self._sample(now)
        if self.proc.poll() is not None:
            self._sample(now)
            self.t_end = now
            self._close()
            self.status = "ok" if bg.sta_completed(self.sta) else "failed"

    def kill(self, why):
        if self.status != "running":
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(self.proc.pid, signal.SIGTERM)
            else:
                self.proc.terminate()
            self.proc.wait(30)
        except (OSError, subprocess.SubprocessError):
            try:
                if hasattr(os, "killpg"):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                else:
                    self.proc.kill()
            except OSError:
                pass
        self.t_end = time.time()
        self._close()
        self.status = why

    def _close(self):
        try:
            self._console.close()
        except (OSError, AttributeError):
            pass

    # ---- extraction (same extractor as the campaign), in the background ------
    def start_extract(self, raw_csv):
        self.raw_csv = raw_csv
        self.xproc = None
        self.extract_s = None
        self._xcon = open(self.console_path, "a")
        self.t_x0 = time.time()
        try:
            self.xproc = subprocess.Popen([self.abaqus, "python", rhi.EXTRACTOR, "--", self.odb, raw_csv],
                                          cwd=self.dir, stdout=self._xcon, stderr=subprocess.STDOUT,
                                          env=rhi.abaqus_env(self.scratch))
        except OSError as e:
            self._xcon.write("extraction could not start: %s\n" % e)

    def extract_poll(self):
        """True once the extraction has ended (successfully or not)."""
        if self.xproc is not None and self.xproc.poll() is None:
            return False
        try:
            self._xcon.close()
        except OSError:
            pass
        self.extract_s = round(time.time() - self.t_x0, 1)
        return True

    def in_analysis(self):
        return any(i > 0 for _, i in self.samples)

    def note(self):
        return bg.console_tail(self.console_path, 3)[:240]

    # ---- numbers -------------------------------------------------------------
    def summary(self, n_full):
        """Rates and the full-run projection of a finished solve."""
        out = {"status": self.status, "total_s": round(self.t_end - self.t0, 1) if self.t_end else None}
        if self.status != "ok":
            out["note"] = self.note()
            return out
        if not self.samples or self.samples[-1][1] <= 0:
            out["note"] = "no increments seen in the .sta"
            return out
        n_inc = self.samples[-1][1]
        pos = [(t, i) for t, i in self.samples if i > 0]
        sta = [(i, w) for i, w in read_sta(self.sta) if i > 0 and w is not None]
        # own wall clock first (the .sta time column could be CPU time summed over cores)
        if len(pos) >= 2 and pos[-1][0] - pos[0][0] >= 10:
            rate = (pos[-1][1] - pos[0][1]) / (pos[-1][0] - pos[0][0])
            method = "poll"
        elif len(sta) >= 2 and sta[-1][1] > sta[0][1]:
            rate = (sta[-1][0] - sta[0][0]) / float(sta[-1][1] - sta[0][1])
            method = ".sta time column"
        elif len(pos) >= 2 and pos[-1][0] - pos[0][0] >= 2:
            rate = (pos[-1][1] - pos[0][1]) / (pos[-1][0] - pos[0][0])
            method = "poll (short)"
        else:
            out["note"] = "the solve was too short to time"
            return out
        overhead = out["total_s"] - n_inc / rate
        out.update(n_inc=n_inc, rate_inc_s=round(rate, 3), s_per_kinc=round(1000.0 / rate, 2),
                   rate_from=method, overhead_s=round(overhead, 1),
                   full_solve_pred_s=round(overhead + n_full / rate, 0))
        return out

    def inc_at(self, t):
        ts = [s[0] for s in self.samples]
        inc = [s[1] for s in self.samples]
        return float(np.interp(t, ts, inc))


def run_group(solves, lic, label, deadline, stall_s, say):
    """Start the solves LAUNCH_GAP_S apart, poll until all end, and read the
    license server once all of them are analysing. Returns that snapshot."""
    snap = None
    pending = list(solves)
    next_start = 0.0
    t_all_in = None
    while pending or any(s.status == "running" for s in solves):
        now = time.time()
        if pending and now >= next_start:      # launches are spaced, but polling never stops
            pending.pop(0).start()
            next_start = time.time() + LAUNCH_GAP_S
        time.sleep(POLL_S)
        now = time.time()
        for s in solves:
            s.poll(now)
            if s.status == "running":
                if deadline and now > deadline:
                    s.kill("deadline")
                    say("   %s: killed at the deadline" % s.tag)
                elif now - s.last_change > stall_s:
                    s.kill("stalled")
                    say("   %s: no progress for %d s, killed -- %s" % (s.tag, stall_s, s.note()))
        if snap is None and not pending and all(s.status == "running" and s.in_analysis() for s in solves):
            if t_all_in is None:
                t_all_in = now
            elif now - t_all_in >= 3:
                snap = lic.snapshot(label)
    return snap


def run_pool(solves, lanes, lic, label, deadline, stall_s, est_s, say):
    """Solve at most `lanes` at a time, like the campaign: a solve starts only if
    it is expected (est_s) to end before the deadline, and each finished solve is
    extracted in the background while the lane takes the next one. Returns the
    license snapshot taken while `lanes` solves were analysing together."""
    pending, running, extracting = list(solves), [], []
    snap, next_start = None, 0.0
    while pending or running or extracting:
        now = time.time()
        if pending and len(running) < lanes and now >= next_start:
            if deadline and now + est_s > deadline:
                say("  deadline: %d solves not started (%s)" % (len(pending), " ".join(s.tag for s in pending)))
                for s in pending:
                    s.status = "not_started"
                pending = []
            else:
                s = pending.pop(0)
                s.start()
                running.append(s)
                next_start = time.time() + LAUNCH_GAP_S
        time.sleep(POLL_S)
        now = time.time()
        for s in list(running):
            s.poll(now)
            if s.status == "running":
                if deadline and now > deadline:
                    s.kill("deadline")
                    say("   %s: killed at the deadline" % s.tag)
                elif now - s.last_change > stall_s:
                    s.kill("stalled")
                    say("   %s: no progress for %d s, killed -- %s" % (s.tag, stall_s, s.note()))
            if s.status != "running":
                running.remove(s)
                if s.status == "ok":
                    s.start_extract(os.path.join(s.dir, "raw_acc.csv"))
                    extracting.append(s)
                else:
                    say("   %s: %s -- %s" % (s.tag, s.status.upper(), s.note()))
        for s in list(extracting):
            if s.extract_poll():
                extracting.remove(s)
                bg.cleanup_big_files(s.dir, say)
                shutil.rmtree(s.scratch, ignore_errors=True)
                say("   %s: solved in %.0f s, extracted in %.0f s%s" % (
                    s.tag, s.t_end - s.t0, s.extract_s, "" if os.path.exists(s.raw_csv) else "  EXTRACTION FAILED"))
        if snap is None and running and len(running) == min(lanes, len(solves)) \
                and all(s.in_analysis() for s in running):
            snap = lic.snapshot(label)
    return snap


def ok_runs_1core(metrics_csv):
    """Runs whose latest ok row was solved on 1 core."""
    last = {}
    if metrics_csv and os.path.exists(metrics_csv):
        for r in csv.DictReader(open(metrics_csv)):
            if r.get("status") == "ok":
                last[int(r["run"])] = bg.row_cpus(r)
    return {run for run, c in last.items() if c == 1}


def noise_plan(data_dir, metrics_csv, runs_arg, repeat_arg):
    """Solve order for the noise phase, the (design, D6) pairs, and the manifest.
    Pairs are formed with the base design (role 'base', D6) at the same location;
    the order completes pairs early; the repeat solve goes third."""
    man = {int(r["run"]): r for r in csv.DictReader(open(os.path.join(data_dir, "manifest_local_edits.csv")))}
    ok1 = ok_runs_1core(metrics_csv)

    def usable(run):
        return (run in man and run in ok1
                and os.path.exists(os.path.join(data_dir, "inp_files", "HoodImpact_%d.inp" % run))
                and os.path.exists(os.path.join(data_dir, "output_history_acc",
                                                "HoodImpact_%d_SAE1000_interp1000.csv" % run)))
    if runs_arg == "pilot":
        cand = [run for run in sorted(man) if man[run]["pilot"] == "1"]
    else:
        cand = rhi.parse_runs(runs_arg)
    skipped = [run for run in cand if not usable(run)]
    cand = [run for run in cand if usable(run)]
    base_at = {man[r]["location"]: r for r in man if man[r]["role"] == "base"}
    order, seen = [], set()
    designs = [r for r in cand if man[r]["role"] != "base"]
    for r in designs:
        b = base_at.get(man[r]["location"])
        if b in cand and b not in seen:
            order.append(b)
            seen.add(b)
        if r not in seen:
            order.append(r)
            seen.add(r)
    order += [r for r in cand if r not in seen]
    pairs = [(r, base_at[man[r]["location"]]) for r in designs if base_at.get(man[r]["location"]) in cand]
    plan = [(r, False) for r in order]
    if repeat_arg != "none" and order:
        rep = int(repeat_arg) if repeat_arg else (designs[0] if designs else order[0])
        if rep in order:
            plan.insert(min(2, len(plan)), (rep, True))
    return plan, pairs, man, skipped


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def evaluate_noise(solves, pairs, man, data_dir, out_dir, n, mode, lanes, say):
    """Compare the N-core solves with their published 1-core curves, run by run and
    as (design - D6) difference curves; check the repeated solve; write noise/."""
    nd = os.path.join(out_dir, "noise")
    os.makedirs(nd, exist_ok=True)
    one, multi, raw, rows, times = {}, {}, {}, [], []
    for s in solves:
        lab = man[s.run]
        base = {"run": s.run, "repeat": int(s.repeat), "design": lab["design"], "location": lab["location"],
                "role": lab["role"]}
        if s.status != "ok" or not os.path.exists(getattr(s, "raw_csv", "") or ""):
            base["status"] = s.status if s.status != "ok" else "extract_failed"
            rows.append(base)
            continue
        new = pp.process_raw(s.raw_csv)
        new.to_csv(os.path.join(nd, "HoodImpact_%d_cpus%d%s_SAE1000_interp1000.csv"
                                % (s.run, n, "_repeat" if s.repeat else "")), index=False)
        raw[(s.run, s.repeat)] = pd.read_csv(s.raw_csv).to_numpy(float)
        ref = pd.read_csv(os.path.join(data_dir, "output_history_acc", "HoodImpact_%d_SAE1000_interp1000.csv" % s.run))
        t = ref["Time"].to_numpy(float)
        g1 = ref["A(in g)"].to_numpy(float)
        gn = np.interp(t, new["Time"].to_numpy(float), new["A(in g)"].to_numpy(float))
        if s.repeat:
            multi[(s.run, True)] = gn
            continue
        times.append((s.t_end - s.t0, s.extract_s))
        one[s.run], multi[s.run] = (t, g1), gn
        h1, hn = pp.compute_hic(t, g1), pp.compute_hic(t, gn)
        base.update(status="ok", n_raw_points=len(raw[(s.run, False)]), solve_s=round(s.t_end - s.t0),
                    extract_s=s.extract_s, rms_g=round(rms(gn - g1), 3), max_abs_g=round(float(np.abs(gn - g1).max()), 3),
                    peak_1core=round(float(g1.max()), 2), peak_ncore=round(float(gn.max()), 2),
                    hic_1core=round(h1, 1), hic_ncore=round(hn, 1), dhic_pct=round(100.0 * (hn / h1 - 1), 3))
        rows.append(base)

    out = {"cpus": n, "mode": mode, "lanes": lanes, "runs": rows}
    say("")
    say("  same run, 1 core vs %d cores (SAE1000 resultant, g):" % n)
    say("    run design  loc  role   incs    solve s   RMS g   max g   peak 1->%d       HIC15 1->%d (change)" % (n, n))
    for r in rows:
        if r.get("status") != "ok":
            say("    %3d %-6s %4s  %-5s  %s" % (r["run"], r["design"], r["location"], r["role"], r["status"].upper()))
            continue
        say("    %3d %-6s %4s  %-5s  %6d  %7d  %6.3f  %6.2f  %6.1f -> %6.1f  %6.0f -> %6.0f (%+.2f%%)" % (
            r["run"], r["design"], r["location"], r["role"], r["n_raw_points"], r["solve_s"], r["rms_g"],
            r["max_abs_g"], r["peak_1core"], r["peak_ncore"], r["hic_1core"], r["hic_ncore"], r["dhic_pct"]))
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        out["floor_rms_g_median"] = float(np.median([r["rms_g"] for r in ok_rows]))
        out["floor_dhic_abs_median"] = float(np.median([abs(r["dhic_pct"]) for r in ok_rows]))
        say("    -> roundoff alone (1 vs %d cores, same geometry): median %.3f g RMS, median |HIC change| %.2f%%"
            % (n, out["floor_rms_g_median"], out["floor_dhic_abs_median"]))

    # repeatability: the same deck twice on N cores
    rep = [k for k in raw if k[1]]
    if rep:
        run = rep[0][0]
        if (run, False) in raw:
            a, b = raw[(run, False)], raw[(run, True)]
            same = a.shape == b.shape and np.array_equal(a, b)
            d = float(np.abs(multi[run] - multi[(run, True)]).max()) if run in multi else float("nan")
            out["repeat"] = {"run": run, "bit_identical": bool(same), "max_abs_g": round(d, 6)}
            say("  repeatability: run %d solved twice on %d cores -> %s" % (
                run, n, "bit-identical raw history" if same else "DIFFERENT (max |diff| %.4f g)" % d))
    else:
        say("  repeatability: not checked (the repeat solve did not finish)")

    # pairs: (design - D6) at the same location, the quantity the dataset is for
    prow = []
    for d, b in pairs:
        if d not in multi or b not in multi:
            continue
        t = one[d][0]
        d1 = one[d][1] - one[b][1]
        dn = multi[d] - multi[b]
        err = dn - d1
        sst = float(np.sum((d1 - d1.mean()) ** 2))
        hd1, hb1 = pp.compute_hic(t, one[d][1]), pp.compute_hic(t, one[b][1])
        hdn, hbn = pp.compute_hic(t, multi[d]), pp.compute_hic(t, multi[b])
        dh1, dhn = 100.0 * (hd1 / hb1 - 1), 100.0 * (hdn / hbn - 1)
        prow.append({"design": man[d]["design"], "run": d, "base_run": b, "location": man[d]["location"],
                     "role": man[d]["role"], "dist_to_edit_mm": man[d].get("dist_to_edit_mm", ""),
                     "signal_rms_g": round(rms(d1), 3), "error_rms_g": round(rms(err), 3),
                     "error_over_signal": round(rms(err) / max(rms(d1), 1e-12), 3),
                     "r2": round(1.0 - float(np.sum(err ** 2)) / sst, 4) if sst > 0 else float("nan"),
                     "dhic_1core_pct": round(dh1, 2), "dhic_ncore_pct": round(dhn, 2),
                     "dhic_diff_pp": round(dhn - dh1, 2)})
    out["pairs"] = prow
    say("")
    say("  design - D6 at the same location (what the model learns), 1 core vs %d cores:" % n)
    say("    design  loc  role   dist mm   signal RMS g  error RMS g  error/signal    R2     HIC change 1 core -> %d cores" % n)
    for p in prow:
        say("    %-6s %4s  %-5s  %7s   %10.2f   %10.3f   %10.2f   %6.3f   %+6.2f%% -> %+6.2f%%  (%+.2f pts)" % (
            p["design"], p["location"], p["role"], p["dist_to_edit_mm"] or "-", p["signal_rms_g"],
            p["error_rms_g"], p["error_over_signal"], p["r2"], p["dhic_1core_pct"], p["dhic_ncore_pct"],
            p["dhic_diff_pp"]))

    # verdict
    near = [p for p in prow if p["role"] in ("near", "twin")]
    far = [p for p in prow if p["role"] == "far"]
    det = (out.get("repeat") or {}).get("bit_identical")
    bad = [p for p in near if not (p["r2"] >= PAIR_R2_MIN and abs(p["dhic_diff_pp"]) <= PAIR_DHIC_PP_MAX)]
    if det is False:
        verdict = "FAIL: the same deck on %d cores does not repeat; stay on single-core lanes" % n
    elif not near:
        verdict = "INCOMPLETE: no near/twin pair finished"
    elif bad:
        verdict = ("FAIL: %d of %d near/twin pairs change (need R2 >= %.2f and HIC change within %.1f pts);"
                   " stay on single-core lanes" % (len(bad), len(near), PAIR_R2_MIN, PAIR_DHIC_PP_MAX))
    else:
        verdict = ("PASS: every near/twin difference keeps its shape (R2 >= %.2f) and HIC change (within %.1f pts)%s"
                   % (PAIR_R2_MIN, PAIR_DHIC_PP_MAX, "" if det else "; repeatability NOT checked"))
    out["verdict"] = verdict
    say("  NOISE VERDICT (%d cores): %s" % (n, verdict))
    for f in far:
        say("  far control (%s mm from the edit): difference %.2f g RMS; changing the core count moves it by"
            " %.2f g RMS (%.0f%%) -> %s" % (
                f["dist_to_edit_mm"], f["signal_rms_g"], f["error_rms_g"], 100 * f["error_over_signal"],
                "the far-field difference is at the solver-noise level" if f["error_over_signal"] > 0.5
                else "the far-field difference stands above solver noise"))

    if times:
        out["calibration"] = {"cpus": n, "lanes": lanes, "solve_s": float(np.median([t for t, _ in times])),
                              "extract_s": float(np.median([x for _, x in times if x is not None] or [EXTRACT_GUESS_S]))}
    pd.DataFrame(rows).to_csv(os.path.join(nd, "noise_runs.csv"), index=False)
    pd.DataFrame(prow).to_csv(os.path.join(nd, "noise_pairs.csv"), index=False)
    if prow:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            k = len(prow)
            fig, axes = plt.subplots(k, 1, figsize=(8, 2.2 * k), sharex=True, squeeze=False)
            for ax, p in zip(axes[:, 0], prow):
                t = one[p["run"]][0] * 1000
                ax.plot(t, one[p["run"]][1] - one[p["base_run"]][1], "k-", lw=1.2, label="1 core")
                ax.plot(t, multi[p["run"]] - multi[p["base_run"]], "r--", lw=1.0, label="%d cores" % n)
                ax.set_ylabel("g")
                ax.set_title("%s - D6 at location %s (%s): R2 %.3f, HIC change %+.2f%% -> %+.2f%%" % (
                    p["design"], p["location"], p["role"], p["r2"], p["dhic_1core_pct"], p["dhic_ncore_pct"]),
                    fontsize=9)
            axes[0, 0].legend(fontsize=8)
            axes[-1, 0].set_xlabel("time (ms)")
            fig.tight_layout()
            fig.savefig(os.path.join(nd, "noise_pairs.png"), dpi=110)
            plt.close(fig)
        except Exception as e:      # plotting is optional
            say("  (no plot: %s)" % e)
    say("  tables and curves: %s" % nd)
    return out


def cleanup(s, say, keep_odb=False):
    bg.cleanup_big_files(s.dir, say, keep_odb=keep_odb)
    shutil.rmtree(s.scratch, ignore_errors=True)


# ----------------------------------------------------------------------------
# campaign numbers
# ----------------------------------------------------------------------------
def production_stats(metrics_csv, run):
    """median cpus=1 solve time, the reference run's own row, ok runs per cpus."""
    out = {"median_solve_s": None, "ref_row": None, "ok_by_cpus": {}}
    if not metrics_csv or not os.path.exists(metrics_csv):
        return out
    df = pd.read_csv(metrics_csv, dtype=str)
    ok = df[df["status"] == "ok"].copy()
    ok["cpus"] = ok["cpus"].fillna("1").replace("", "1").astype(int)
    out["ok_by_cpus"] = {int(k): int(v) for k, v in ok.drop_duplicates("run", keep="last")
                         .groupby("cpus")["run"].count().items()}
    one = pd.to_numeric(ok.loc[ok["cpus"] == 1, "solve_s"], errors="coerce").dropna()
    if len(one):
        out["median_solve_s"] = float(one.median())
    ref = ok[ok["run"] == str(run)]
    if len(ref):
        out["ref_row"] = ref.iloc[-1].to_dict()
    return out


def pilot_far_floor(data_dir):
    """(rms_g, dhic_pct) of the pilot far control, the far-field difference floor."""
    try:
        with open(os.path.join(data_dir, "pilot_report.txt")) as f:
            for line in f:
                tok = line.split()
                if len(tok) >= 7 and tok[3] == "far":
                    return float(tok[5]), float(tok[6])
    except (OSError, ValueError):
        pass
    return None


def candidates(solo, avail, max_cores, extract_s):
    out = []
    for e in solo:
        if e.get("status") != "ok" or not e.get("full_solve_pred_s"):
            continue
        n = e["cpus"]
        lanes = max_cores // n
        toks = e["tokens"] or {"abaqus": tokens_rule(n)}
        for f, k in toks.items():
            if k > 0 and f in avail:
                lanes = min(lanes, avail[f] // k)
        if lanes < 1:
            continue
        run_s = e["full_solve_pred_s"] + extract_s
        out.append({"cpus": n, "mode": e["mode"], "lanes": lanes, "cores": lanes * n,
                    "tokens": {f: k * lanes for f, k in toks.items()},
                    "run_s": round(run_s), "runs_per_h": round(lanes * 3600.0 / run_s, 2),
                    "extract_s_used": extract_s, "measured": "solo only (no contention)"})
    return sorted(out, key=lambda c: -c["runs_per_h"])


def adjust(row, extract_s, ratio):
    """Copy of a layout row with its solve time scaled by `ratio` (measured full solve
    / its projection from the short solo deck) and the measured extraction time."""
    r = dict(row)
    if r.get("measured") == "campaign":
        return r
    used = r.get("extract_s_used", EXTRACT_GUESS_S)
    r["run_s"] = round((r["run_s"] - used) * ratio + extract_s)
    r["runs_per_h"] = round(r["lanes"] * 3600.0 / r["run_s"], 2)
    return r


def pick(pool):
    """Highest runs/h; within 3% of it, the layout with the fewest cores (core-hours)."""
    best = max(c["runs_per_h"] for c in pool)
    close = [c for c in pool if c["runs_per_h"] >= 0.97 * best]
    return min(close, key=lambda c: (c["cores"], -c["runs_per_h"]))


def fmt_tokens(t):
    return " ".join("%s=%d" % (f, k) for f, k in sorted(t.items())) if t else "?"


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=1, help="a run already solved with cpus=1 (default 1, a pilot run)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--metrics", default=None, help="campaign generation_metrics.csv (cpus=1 times, reference run)")
    ap.add_argument("--work-dir", required=True, help="scratch folder for the test solves")
    ap.add_argument("--out-dir", default=None, help="default <data-dir>/cpus_test")
    ap.add_argument("--cpus", default="1 4 8 16 32",
                    help="solo cpus counts; 'N' or 'N:mode' (mode threads|mpi), space separated")
    ap.add_argument("--mode", default="threads", help="mp_mode for cpus>1 when not given per entry")
    ap.add_argument("--base-frac", type=float, default=0.04,
                    help="solo deck length as a fraction of the step for cpus=1; x sqrt(cpus) for more cores")
    ap.add_argument("--max-frac", type=float, default=0.3)
    ap.add_argument("--phases", default="solo,layout,noise", help="any of solo,layout,noise ('full' = noise)")
    ap.add_argument("--layouts", type=int, default=3, help="how many candidate layouts to measure together")
    ap.add_argument("--max-cores", type=int, default=None,
                    help="cores the campaign job may use (default: this job's cores)")
    ap.add_argument("--reserve", type=int, default=0,
                    help="tokens of each feature to leave free for other license users")
    ap.add_argument("--noise-cpus", "--full-cpus", type=int, default=None,
                    help="cores per solve in the noise phase (default: the chosen layout's)")
    ap.add_argument("--noise-runs", default="pilot",
                    help="'pilot' (runs 1-11: D6, N1, N2 at 4 locations) or a run list like 1-11,20")
    ap.add_argument("--noise-repeat", default="",
                    help="run solved twice for the repeatability check (default: the first edited run; 'none')")
    ap.add_argument("--total-runs", type=int, default=None, help="campaign size (default: manifest rows)")
    ap.add_argument("--deadline", type=float, default=None, help="unix time; nothing runs past it")
    ap.add_argument("--stall-s", type=float, default=600, help="kill a solve whose .sta has not moved for this long")
    ap.add_argument("--server", default=os.environ.get("ABAQUSLM_LICENSE_FILE") or DEFAULT_SERVER)
    ap.add_argument("--abaqus", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "cpus_test")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    say = Report(os.path.join(out_dir, "report.txt"))
    res_path = os.path.join(out_dir, "results.json")
    phases = [{"full": "noise"}.get(p.strip(), p.strip()) for p in args.phases.split(",") if p.strip()]
    abaqus = args.abaqus or "abaqus"
    alloc = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    max_cores = args.max_cores or alloc
    deadline = args.deadline

    deck = os.path.join(args.data_dir, "inp_files", "HoodImpact_%d.inp" % args.run)
    if not os.path.exists(deck):
        raise SystemExit("missing deck %s" % deck)
    n_full, dt, period = deck_increments(deck)
    prod = production_stats(args.metrics, args.run)
    total_runs = args.total_runs
    if total_runs is None:
        man = os.path.join(args.data_dir, "manifest_local_edits.csv")
        total_runs = sum(1 for _ in open(man)) - 1 if os.path.exists(man) else None

    res = {}
    if "solo" not in phases and os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        say.lines = res.get("report_lines", [])
    res.setdefault("solo", [])
    res.setdefault("layouts", [])

    def save():
        res["report_lines"] = say.lines
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)
        say.save()

    say("=" * 78)
    say("cpus scaling test  %s  job %s on %s (%d cores, %s)" % (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), os.environ.get("SLURM_JOB_ID", "-"),
        socket.gethostname(), alloc, ",".join(phases)))
    say("deck HoodImpact_%d.inp: dt %.3g s x %.4g s = %d increments; license %s"
        % (args.run, dt, period, n_full, args.server))
    if prod["median_solve_s"]:
        say("campaign so far: cpus=1 lanes, median solve %.0f s (%.2f s per 1000 increments); ok runs by cpus %s"
            % (prod["median_solve_s"], prod["median_solve_s"] / n_full * 1000, prod["ok_by_cpus"]))
    if deadline:
        say("deadline %s" % datetime.datetime.fromtimestamp(deadline).strftime("%H:%M:%S"))
    busy = campaign_running()
    if busy:
        say("STOP: hood_local_edits job(s) %s are running and hold the license; hold or end them first." % busy)
        save()
        sys.exit(2)

    lic = License(abaqus, args.server, out_dir)
    base = lic.snapshot("baseline")
    if base:
        say("license now: " + "; ".join("%s %d/%d in use" % (f, v["in_use"], v["issued"])
                                        for f, v in sorted(base.items()) if v["in_use"] or f in ("abaqus", "parallel")))
        avail = {f: v["issued"] - v["in_use"] - args.reserve for f, v in base.items()}
    else:
        say("WARNING: could not read the license server; token counts fall back to the Abaqus rule")
        avail = {"abaqus": 65 - args.reserve}
    res["avail"] = avail

    # ---------------------------------------------------------------- solo
    if "solo" in phases:
        say("")
        say("SOLO: one solve at a time on the start of the step (mp_mode %s for cpus>1)" % args.mode)
        say("  cpus  mode     incs   s/1000inc  speedup  overhead  full solve (pred)  tokens held [rule]")
        res["solo"] = []
        for spec in args.cpus.split():
            n, _, mode = spec.partition(":")
            n = int(n)
            mode = (mode or args.mode) if n > 1 else None
            if deadline and time.time() + 420 > deadline:
                say("  cpus %d: skipped, too close to the deadline" % n)
                continue
            if campaign_running():
                say("STOP: a hood_local_edits job started meanwhile; the test no longer owns the license.")
                break
            frac = min(args.max_frac, args.base_frac * math.sqrt(n))
            short = os.path.join(args.work_dir, "short_%04d.inp" % int(round(frac * 1000)))
            n_inc = write_short_deck(deck, short, frac)
            s = Solve(abaqus, "solo%d%s" % (n, "_" + mode if mode and mode != "threads" else ""),
                      short, n, mode, args.work_dir)
            snap = run_group([s], lic, s.tag, deadline, args.stall_s, say)
            e = s.summary(n_full)
            e.update(cpus=n, mode=mode, frac=round(frac, 4), deck_incs=n_inc)
            e["tokens"], e["token_method"] = tokens_held(snap, base)
            e["tokens_rule"] = tokens_rule(n)
            res["solo"].append(e)
            cleanup(s, say)
            if e["status"] == "ok":
                one = [x for x in res["solo"] if x["cpus"] == 1 and x.get("rate_inc_s")]
                sp = e["rate_inc_s"] / one[0]["rate_inc_s"] if one else float("nan")
                say("  %4d  %-7s %6d  %9.2f  %7.2f  %6.0f s  %8.0f s = %4.1f h  %s [%d]"
                    % (n, mode or "-", e["n_inc"], e["s_per_kinc"], sp, e["overhead_s"],
                       e["full_solve_pred_s"], e["full_solve_pred_s"] / 3600.0,
                       fmt_tokens(e["tokens"]), e["tokens_rule"]))
            else:
                say("  %4d  %-7s %s after %s s -- %s" % (n, mode or "-", e["status"].upper(),
                                                         e.get("total_s"), e.get("note", "")))
            save()

    # -------------------------------------------------------------- layouts
    extract_s = ((res.get("noise") or {}).get("calibration") or {}).get("extract_s") or EXTRACT_GUESS_S
    cands = candidates(res["solo"], avail, max_cores, extract_s)
    res["candidates"] = cands
    if "layout" in phases and cands:
        say("")
        say("LAYOUT: the best predicted lanes x cpus (max %d cores, tokens available %s) solving together"
            % (max_cores, fmt_tokens({f: v for f, v in avail.items() if f in ("abaqus", "parallel", "explicit")})))
        res["layouts"] = []
        todo = [c for c in cands if c["lanes"] >= 2 and c["cores"] <= alloc][:args.layouts]
        for c in todo:
            solo = [e for e in res["solo"] if e["cpus"] == c["cpus"] and e["mode"] == c["mode"]][0]
            frac = min(args.max_frac, 1.5 * args.base_frac * math.sqrt(c["cpus"]))
            n_inc_layout = int(round(frac * n_full))
            est = solo["overhead_s"] + 2.0 * n_inc_layout / solo["rate_inc_s"] + 60
            if deadline and time.time() + est > deadline:
                say("  %d x %d: skipped, would not finish before the deadline (~%.0f min)" % (c["lanes"], c["cpus"], est / 60))
                continue
            if campaign_running():
                say("STOP: a hood_local_edits job started meanwhile; the test no longer owns the license.")
                break
            short = os.path.join(args.work_dir, "short_%04d.inp" % int(round(frac * 1000)))
            write_short_deck(deck, short, frac)
            lanes = [Solve(abaqus, "lay%dx%d_%d" % (c["lanes"], c["cpus"], k), short, c["cpus"], c["mode"], args.work_dir)
                     for k in range(c["lanes"])]
            snap = run_group(lanes, lic, "lay%dx%d" % (c["lanes"], c["cpus"]), deadline, args.stall_s, say)
            e = dict(c)
            e["status"] = "ok" if all(s.status == "ok" for s in lanes) else \
                ",".join(sorted(set(s.status for s in lanes)))
            if snap:
                e["license_in_use"] = {f: v["in_use"] for f, v in snap.items() if v["in_use"]}
            if e["status"] == "ok":
                starts = [min(t for t, i in s.samples if i > 0) for s in lanes]
                ends = [s.samples[-1][0] for s in lanes]
                w0, w1 = max(starts), min(ends)
                if w1 - w0 >= 20:
                    rates = [(s.inc_at(w1) - s.inc_at(w0)) / (w1 - w0) for s in lanes]
                    e["window_s"] = round(w1 - w0, 1)
                else:
                    rates = [s.summary(n_full).get("rate_inc_s") for s in lanes]
                    rates = [r for r in rates if r]
                    e["window_s"] = round(max(w1 - w0, 0), 1)
                    e["note"] = "lanes overlapped only %.0f s; whole-solve rates used" % max(w1 - w0, 0)
                if not rates:
                    e["status"] = "untimed"
                    say("  %d x %d cpus: finished but too short to time" % (c["lanes"], c["cpus"]))
                    for s in lanes:
                        cleanup(s, say)
                    res["layouts"].append(e)
                    continue
                lane_rate = float(np.mean(rates))
                e["lane_rate_inc_s"] = round(lane_rate, 3)
                e["contention"] = round(solo["rate_inc_s"] / lane_rate, 3)
                e["run_s"] = round(solo["overhead_s"] + n_full / lane_rate + extract_s)
                e["runs_per_h"] = round(c["lanes"] * 3600.0 / e["run_s"], 2)
                e["measured"] = "together (%d lanes)" % c["lanes"]
                say("  %d x %2d cpus (%d cores, %s): %.2f s/1000inc per lane, %.0f%% slower than alone"
                    " -> %.0f s per run, %.2f runs/h%s"
                    % (c["lanes"], c["cpus"], c["cores"], fmt_tokens(e["tokens"]), 1000.0 / lane_rate,
                       100 * (e["contention"] - 1), e["run_s"], e["runs_per_h"],
                       "  (%s)" % e["note"] if e.get("note") else ""))
            else:
                notes = "; ".join("%s: %s" % (s.tag, s.note()) for s in lanes if s.status != "ok")[:400]
                say("  %d x %d cpus: %s -- %s" % (c["lanes"], c["cpus"], e["status"].upper(), notes))
            for s in lanes:
                cleanup(s, say)
            res["layouts"].append(e)
            save()

    # ------------------------------------------- provisional choice (sets the full phase's cpus)
    measured = [e for e in res.get("layouts", []) if e.get("status") == "ok"]
    measured += [c for c in cands if c["lanes"] == 1]
    failed = {(e["lanes"], e["cpus"]) for e in res.get("layouts", []) if e.get("status") != "ok"}
    pool = measured or [c for c in cands if (c["lanes"], c["cpus"]) not in failed]
    pre = pick(pool) if pool else {}

    # ------------------------------------------------------------------ noise
    noise_n = args.noise_cpus or pre.get("cpus")
    if "noise" in phases and noise_n:
        same = noise_n == pre.get("cpus")
        noise_mode = (pre.get("mode") if same else args.mode) if noise_n > 1 else None
        if same:
            lanes_n = pre["lanes"]
        else:
            e1 = [e for e in res["solo"] if e["cpus"] == noise_n and e.get("tokens")]
            toks = e1[0]["tokens"] if e1 else {"abaqus": tokens_rule(noise_n)}
            lanes_n = max(1, min([alloc // noise_n] + [avail[f] // k for f, k in toks.items() if k > 0 and f in avail]))
        plan, pairs, man, skipped = noise_plan(args.data_dir, args.metrics, args.noise_runs, args.noise_repeat)
        say("")
        say("NOISE: %d runs finished on 1 core, solved again on %d cores (%s), %d at a time, as the campaign would"
            % (len({r for r, _ in plan}), noise_n, noise_mode or "-", lanes_n))
        if skipped:
            say("  left out (not ok on 1 core, or deck/curve missing): %s" % skipped)
        if same and pre.get("run_s"):
            est = (pre["run_s"] - pre.get("extract_s_used", EXTRACT_GUESS_S)) * 1.25 + 120
        else:
            solo = [e for e in res["solo"] if e["cpus"] == noise_n and e.get("full_solve_pred_s")]
            est = solo[0]["full_solve_pred_s"] * 1.5 + 120 if solo else 3 * 3600
        say("  %d solves incl. one repeat, ~%.0f min each, %d rounds; order %s"
            % (len(plan), est / 60, math.ceil(len(plan) / float(lanes_n)),
               " ".join("%d%s" % (r, "(again)" if rep else "") for r, rep in plan)))
        if noise_n == 1:
            say("  the chosen layout is single-core; nothing to compare")
        elif not plan:
            say("  no finished 1-core runs to compare")
        elif campaign_running():
            say("STOP: a hood_local_edits job started meanwhile; the test no longer owns the license.")
        else:
            solves = []
            for run, rep in plan:
                s = Solve(abaqus, "n%d%s" % (run, "r" if rep else ""),
                          os.path.join(args.data_dir, "inp_files", "HoodImpact_%d.inp" % run),
                          noise_n, noise_mode, args.work_dir)
                s.run, s.repeat = run, rep
                solves.append(s)
            snap = run_pool(solves, lanes_n, lic, "noise", deadline, args.stall_s, est, say)
            if snap:
                say("  license while %d solved together: %s" % (lanes_n, "; ".join(
                    "%s %d/%d in use" % (f, v["in_use"], v["issued"]) for f, v in sorted(snap.items()) if v["in_use"])))
            res["noise"] = evaluate_noise(solves, pairs, man, args.data_dir, out_dir, noise_n, noise_mode,
                                          lanes_n, say)
        save()

    # ------------------------------------------- final choice
    noise = res.get("noise") or {}
    cal = noise.get("calibration") or {}
    extract_now = cal.get("extract_s") or extract_s
    ratio = 1.0
    if pre and cal.get("solve_s") and cal.get("cpus") == pre.get("cpus") and cal.get("lanes") == pre.get("lanes"):
        ratio = cal["solve_s"] / max(pre["run_s"] - pre.get("extract_s_used", EXTRACT_GUESS_S), 1.0)
    ch = adjust(pre, extract_now, ratio) if pre else None
    res["choice"] = ch
    redo = 0 if ch is None or ch["cpus"] == 1 else 1

    # --------------------------------------------------------------- summary
    say("")
    say("LAYOUTS (runs per hour of the whole job; cpus=1 x 12 lanes is the campaign so far)")
    rows = []
    if prod["median_solve_s"]:
        rows.append({"lanes": 12, "cpus": 1, "cores": 12, "run_s": prod["median_solve_s"] + extract_now,
                     "runs_per_h": 12 * 3600.0 / (prod["median_solve_s"] + extract_now), "measured": "campaign"})
    seen = set()
    for e in [e for e in res.get("layouts", []) if e.get("status") == "ok"] + cands:
        key = (e["lanes"], e["cpus"])
        if key not in seen:
            seen.add(key)
            rows.append(ch if ch and key == (ch["lanes"], ch["cpus"]) else adjust(e, extract_now, ratio))
    rows = rows[:1] + sorted(rows[1:], key=lambda r: -r["runs_per_h"]) if prod["median_solve_s"] else \
        sorted(rows, key=lambda r: -r["runs_per_h"])
    remaining = None
    if total_runs:
        done_same = prod["ok_by_cpus"].get(ch["cpus"], 0) if ch else 0
        done_any = sum(prod["ok_by_cpus"].values())
        remaining = total_runs - (done_same if redo else done_any)
    say("  lanes x cpus  cores  per run   runs/h  %s  measured" % ("campaign left" if remaining else ""))
    for r in rows:
        left = "  %5.0f h" % (remaining / r["runs_per_h"]) if remaining else ""
        say("  %5d x %-4d %6d  %6.0f s  %6.2f %s  %s%s" % (
            r["lanes"], r["cpus"], r["cores"], r["run_s"], r["runs_per_h"], left, r["measured"],
            "  <- chosen" if ch and (r["lanes"], r["cpus"]) == (ch["lanes"], ch["cpus"]) else ""))
    if ratio != 1.0:
        say("  (solve times scaled x%.2f: the full noise-phase solves on %d x %d against the short-deck"
            " projection; extraction %.0f s measured)" % (ratio, cal["lanes"], cal["cpus"], extract_now))
    if ch:
        hours = remaining / ch["runs_per_h"] if remaining else None
        verdict = noise.get("verdict") if noise.get("cpus") == ch["cpus"] else None
        say("")
        say("CHOSEN: %d lanes x %d cpus (%s), %d cores, %s tokens, ~%.0f s per run, %.2f runs/h  [%s]"
            % (ch["lanes"], ch["cpus"], ch.get("mode") or "-", ch["cores"], fmt_tokens(ch["tokens"]),
               ch["run_s"], ch["runs_per_h"], ch["measured"]))
        if hours:
            say("  %d runs left -> %.0f h of solving (%d x 48-h jobs, or %d x 12-h jobs), ~%.0f core-hours"
                % (remaining, hours, math.ceil(hours / 46.0), math.ceil(hours / 11.0), hours * ch["cores"]))
            if prod["median_solve_s"]:
                h1 = (total_runs - sum(prod["ok_by_cpus"].values())) / rows[0]["runs_per_h"]
                say("  (12 single-core lanes: %.0f h of solving, ~%.0f core-hours)" % (h1, h1 * 12))
        say("  noise verdict at %d cores: %s" % (ch["cpus"], verdict or "NOT TESTED"))
        mem = max(24, int(math.ceil(ch["cores"] * 1.0)))
        env = os.path.join(out_dir, "layout.env")
        try:
            env_rel = os.path.relpath(env, CODE_ROOT)
        except ValueError:
            env_rel = env
        with open(env, "w") as f:
            f.write("# Fastest layout measured by cpus_scaling_test.py, job %s, %s.\n"
                    % (os.environ.get("SLURM_JOB_ID", "-"), datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            f.write("# NOISE VERDICT at %d cores: %s\n" % (ch["cpus"], verdict or "NOT TESTED"))
            f.write("# Use only if the verdict is PASS (see report.txt):\n")
            f.write("#   cp %s Data/HoodImpact_LocalEdits/campaign_layout.env\n" % env_rel)
            f.write("#   sbatch -c %d --mem=%dg --array=1-4%%1 abaqus_scripts/delta/hood_local_edits.sbatch\n"
                    % (ch["cores"], mem))
            f.write("# Environment variables of the same name override these lines.\n")
            f.write("LOCAL_EDITS_CPUS=%d\n" % ch["cpus"])
            f.write("LOCAL_EDITS_LANES=%d\n" % ch["lanes"])
            f.write("LOCAL_EDITS_MP_MODE=%s\n" % (ch.get("mode") or ""))
            f.write("LOCAL_EDITS_RUN_S=%d\n" % int(math.ceil(ch["run_s"] * 1.15)))
            f.write("# 1 = re-solve the runs finished on another core count: one solver setting for the whole dataset\n")
            f.write("LOCAL_EDITS_REDO_OTHER_CPUS=%d\n" % redo)
        say("  written %s" % env_rel)
        if ch["cpus"] > 1 and not verdict:
            say("  NOTE: the noise test has not run at %d cores. To run only that part:" % ch["cpus"])
            say("  sbatch --export=ALL,CPUS_TEST_PHASES=noise,CPUS_TEST_NOISE_CPUS=%d abaqus_scripts/delta/hood_cpus_test.sbatch"
                % ch["cpus"])
    else:
        say("no layout could be chosen (no solo solve finished)")
    save()

if __name__ == "__main__":
    main()
