# Local-edit dataset on Delta

1582 Abaqus/Explicit solves for learning local design changes:

- 72 beam edits of hood design 6 (N1–N72), 20 impact locations each = 1440 runs;
- design 6 itself ("D6") at all 142 Euro-NCAP locations = 142 runs.

The edits come in two families:

| family | designs | what |
|---|---|---|
| single-beam notches | N1–N49 without N4 (48) | one beam (a diagonal pair, the centre beam or the cross beam) made shallower or deeper over ±60 mm: standard, half, deeper, length variants at 12 sites |
| cut lines | N4, N50–N68 (20) | the centre beam **and** both diagonals edited at about the same x (−125, −50, 30, 100, 460), which cuts every load path across the hood there: standard (±60 mm), crease (20 mm raise over ±30 mm), half crease, deeper |
| single-beam creases | N69–N72 (4) | the crease on one beam only, at x = 30 and x = 100, to separate the crease from the cut line |

Why the cut lines: the old design 7 raised all three beams by 18–20 mm over ~40 mm at x ≈ 80 and
changed the clean-start HIC by −15 to −42 % over a wide region. The single-beam notches change it
by 1–4 %, about the run-to-run noise (1.4 %). Runs 1–1162 (D6 and N1–N51) were planned first and
keep their numbers; N52–N72 are runs 1163–1582.

**Solve order** (manifest column `solve_order`): pilot, D6, cut-line creases, cut-line standard,
single-beam creases, half creases, cut-line deeper, then the single-beam notches.

The plan is fixed in `Data/HoodImpact_LocalEdits/`, written by `abaqus_scripts/local_edit_doe.py`:

| file | content |
|---|---|
| `local_edit_doe.csv` | one row per notch: design, site, centre, axis, floor gap, dz, half-length, mirror |
| `local_edit_designs.csv` | one row per design: variant, nodes moved, checks, its near / twin / far locations |
| `manifest_local_edits.csv` | one row per run: design, location, role, distance to the edit, headform shift |
| `design_changes/<design>_changed_nodes.csv` | label, x, y, z of design 6, dz of every moved node |

The 3-D viewer of the designs and their locations is at
https://claude.ai/artifact/5ASvaZRHvCKbfzYovfZskC

## What every run has in common

Each deck is `HoodImpact_31.inp` (design 6) with three kinds of change. Every other line is unchanged.

1. **The design's inner-panel node z values.** Design 6 itself has none.
2. **The headform position.** It is rigidly moved to the location, and its contact surface starts
   **1.0 mm clear** of the hood's outer contact surface, so there is no initial overclosure.
3. **The time step.** `*DYNAMIC, EXPLICIT, DIRECT USER CONTROL` with a **fixed 4.5e-7 s increment**
   for every run. The deck's variable mass scaling keeps every element's stable increment at or above
   5e-7 s, so this increment is stable for every design. It also means no two designs can differ by
   time step, which is what contaminated the delivered near-clone pairs. Every edited element keeps
   its edges ≥ 3.0 mm (≥ 5.25 mm for the smooth notches); at the aluminium plate wave speed
   (5.07 km/s) that is a stable increment ≥ 5.9e-7 s, so mass scaling adds no mass to any edit.

## Which locations each design is run at (all 72 designs: 16 + 2 + 2)

| role | count | what |
|---|---|---|
| near | 16 | the 16 locations nearest the edit on the +y half and the centreline (the 16th lies 115–278 mm away) |
| twin | 2 | mirror images (`col_k -> -col_k`) of the two nearest off-centre near locations: checks the symmetry and measures noise where the signal is largest |
| far | 2 | controls 300–450 mm from the edit, where no change is expected |

Distances are measured in plan view from the impact point to the nearest node moved by more than
0.5 mm. The mirrored notch on the −y side gives the same response at the mirrored location, so near
locations are taken on one half only; the twins confirm this.

## 0. Upload (from Windows, Code directory)

`local_edits_delta.tar.gz` holds the scripts (with Linux line endings), the plan csvs and the base
deck. It overwrites these scripts on Delta with the current Windows versions:

```powershell
scp local_edits_delta.tar.gz hkazemi@dt-login04.delta.ncsa.illinois.edu:<Code dir on Delta>/
```

On Delta:

```bash
cd <Code dir on Delta>
tar -xzf local_edits_delta.tar.gz
```

**Adding designs to a running campaign.** An update tarball (`local_edits_update.tar.gz`, unpacked the same way) holds only the changed scripts and the
three plan csvs. It can be unpacked while a job runs: the running job keeps its loaded code and run
list, and the next job writes the missing decks and solves in the new order. Runs already finished
are never redone.

**Never delete, move or overwrite `hood_local_edits_<jobid>.out` while that job runs** (copying it is
fine). On the NFS home a deleted log turns into a stale file handle in the running job; job
23838174 lost 4.5 h that way before the logging was made failure-proof.

## 1. Pre-flight (login node, 1 minute)

```bash
quota -s     # needs ~27 GB free in home: 7.5 GB decks, ~7 GB raw csvs, ~11 GB solver scratch
abaqus licensing lmstat -c 27000@130.126.241.64 -i | head -20   # expiry after today, ABAQUSLM UP
conda activate hood_impact
python abaqus_scripts/make_local_edit_decks.py --runs 1-11 --verify 1   # writes and checks the 11 pilot decks
```

If home is short of space, the old `inp_files` folders of `HoodImpact_600_EuroNCAP` and
`HoodImpact_1104_EuroNCAP_Remaining` can be deleted: the local-edit campaign does not use them.

## 2. Submit once

```bash
sbatch --array=1-4%1 abaqus_scripts/delta/hood_local_edits.sbatch
```

The job finds its Code directory from the script's own path, so it reads and writes
`<Code dir>/Data/HoodImpact_LocalEdits` wherever you submit from. That has to be the folder the
tarball was unpacked in. The job:

1. writes the decks that are missing (all 1582 take about 15 minutes);
2. solves the 11 pilot runs, only if the pilot gate has not passed yet (`pilot_gate_passed`);
3. runs the pilot gate (`check_local_edits.py --pilot`, see `Data/HoodImpact_LocalEdits/pilot_report.txt`);
4. solves the rest in solve order on 5 lanes of 8-core solves (`mp_mode=threads`, 12 FlexNet tokens
   each, 60 of the 65; section 2b). The job asks for 64 cores and 64 GB: the 40 the solves use, plus
   the headroom they had in the test.
5. solves again every run that finished on another core count, i.e. the 23 runs done on single-core
   lanes before 2026-11-14, so the whole dataset comes from one solver setting.

**Budget.** A run takes about 1000–1300 s on 8 cores with 5 solving together (55.6k increments),
about 15 runs per hour. 1582 runs come to about 100–110 h of wall time, or about 6,500–7,000
core-hours at 64 cores. That is half the time of 12 single-core lanes, at about 2.5 times the
core-hours.

**48-hour jobs.** The campaign runs as 48-h jobs (Delta's cpu maximum) of about 700 runs each, so
3 jobs. The queue wait turned out about the same for 12-h and 48-h requests (~20 h), so fewer,
longer jobs finish the campaign sooner. A lane only starts a run that is expected to finish before
the job ends, so no solve is cut off half-way. For a different length, e.g.
`sbatch -t 24:00:00 --array=1-6%1 abaqus_scripts/delta/hood_local_edits.sbatch`.

**Follow-up jobs.** Two ways to run the jobs one after another; finished runs are never redone
either way.

- **Job array (preferred):** `sbatch --array=1-4%1 abaqus_scripts/delta/hood_local_edits.sbatch`.
  3 tasks should finish the campaign; the 4th is slack. All 4 tasks are eligible from the start and gain queue age while the one before them runs, so
  each next task starts sooner. The task that finishes the last run, or one that finishes no run,
  cancels the pending tasks.
- **Chain:** plain `sbatch abaqus_scripts/delta/hood_local_edits.sbatch`. Each job queues its own
  follow-up (`afterany`) as soon as it starts, but a follow-up only gains queue age once the job
  before it has ended. The chain stops when every run is done, after 30 follow-ups, or when a whole
  job finished no run.

Only one of these jobs may run at a time (two would need 120 of the 65 license tokens): a job that
finds another one running stops.

**Right after submitting**, `squeue -u hkazemi` shows the array as `<jobid>_[1-4%1]`, pending, and
later one running task plus the rest pending. If the queue is empty a minute after a task started,
it stopped at a check; `job_logs/status.txt` says which.

**Tracking.** Everything is written inside `Data/HoodImpact_LocalEdits/`, so downloading that
folder shows the state:

| file | content |
|---|---|
| `job_logs/status.txt` | one line per stage of every job: start, follow-up queued, decks ready, pilot gate, finished, or the error that stopped it |
| `job_logs/hood_local_edits_<jobid>.out` | the Slurm log, refreshed every 10 minutes and when the job ends |
| `generation_metrics.csv` | one row per finished run (`status`, `n_raw_points`), refreshed every 10 minutes |
| `output_history_acc/` | results, one csv per finished run, written as each run completes |

On Delta:

```bash
python abaqus_scripts/local_edits_progress.py --list   # done runs (never solved again), what is next
squeue -u hkazemi
sprio -j <jobid>                              # while pending: why it waits (AGE, FAIRSHARE)
cat Data/HoodImpact_LocalEdits/job_logs/status.txt
tail -f hood_local_edits_<jobid>.out         # in the folder you submitted from
scancel -u hkazemi -n hood_local_edits       # stops the running task and the pending ones
```

## 2b. Why 5 lanes x 8 cores (the cpus test)

The license has 65 `abaqus` tokens, not 65 cores. A solve on N cores checks out
`floor(5 * N**0.422)` tokens: 1 core costs 5, 8 cores 12, 16 cores 16. Single-core lanes therefore
stop at 13 cores, while the same 65 tokens can run, for example, 5 lanes x 8 cores (40 cores) or
4 lanes x 16 cores (64 cores). Editing the count in the license file does not work: every FEATURE
line is signed (`SIGN=`), and a changed line is rejected outright.

**Result** (job 23905334, 2026-11-13; everything in `Data/HoodImpact_LocalEdits/cpus_test/`):

| | 1 core | 4 cores | 8 cores | 16 cores | 32 cores |
|---|---|---|---|---|---|
| tokens held (lmstat) | 5 | 8 | 12 | 16 | 21 |
| speed, one solve alone | 1x | 3.4x | 6.0x | 9.6x | 11.8x |

| layout, solving together | runs/h | campaign |
|---|---|---|
| 12 x 1 (the campaign until 2026-11-14) | ~7 | ~220 h |
| **5 x 8** (40 of 64 cores busy) | **~15–17** | **~95–110 h** |
| 8 x 4 (32 of 64 cores busy) | ~16 (projected) | ~100 h |
| 4 x 16 (64 of 64 cores busy; each solve 3x slower than alone) | ~12 | ~130 h |

Noise, the 11 pilot runs solved again on 5 x 8: **PASS**.
- 1 vs 8 cores on the same deck: HIC within 0.03%, peaks within 0.15 g, curves within 0.02–0.24 g
  RMS.
- Every design - D6 difference keeps R2 >= 0.998 and its HIC change within 0.03 points. N1 at
  location 9 shows 0.992 only because the 1-core run 6 turns up in the last 0.1 ms of the window.
- Run 5 solved twice on 8 cores is bit-identical.
- The far pair's 3 g difference (379 mm from the edit) is 100x the core-count noise, so it is a
  real effect of the edit.

The campaign therefore runs 5 x 8 by default (`hood_local_edits.sbatch`).

**Changing the layout.**
- **Overrides:** `Data/HoodImpact_LocalEdits/campaign_layout.env` or `--export` with
  `LOCAL_EDITS_CPUS`, `_LANES`, `_MP_MODE`, `_RUN_S`, `_REDO_OTHER_CPUS`; `--export` wins. Match
  them with `-c`/`--mem`.
- **Back to single-core lanes:**
  `sbatch -c 12 --mem=24g --export=ALL,LOCAL_EDITS_CPUS=1,LOCAL_EDITS_LANES=12 --array=1-8%1 abaqus_scripts/delta/hood_local_edits.sbatch`.
  The runs finished on 8 cores are then solved again on 1 core.
- **Never mix core counts** in the dataset.
- **Testing another core count:** a new core count needs its own noise test first, with
  `hood_cpus_test.sbatch` and `cpus_scaling_test.py` (removed on 2026-11-27; restore them from git,
  commit 65a938f).

**How the test worked.** One job (`hood_cpus_test.sbatch`, ~1.5 h on 64 cores; job 23905334,
report in `Data/HoodImpact_LocalEdits/cpus_test/`) timed the start of run 1's step on 1-32 cores and
read the tokens each held from the license server, ran the 3 best predicted layouts together, then
solved the 11 pilot runs in full on the chosen layout. Using N cores splits the hood into N domains,
which changes the floating-point order, so the noise this adds to (design - D6) had to stay far
below the edit effects: PASS required run 5 to repeat bit for bit and every near/twin pair to keep
R2 >= 0.95 and its HIC change within 0.5 percentage points. The scripts were removed with the other
finished campaigns on 2026-11-27 and are in git history (commit 65a938f).

## 3. The pilot gate

The pilot is runs 1–11:

- D6, N1 (12.5 mm raise) and N2 (20 mm deeper) at N1's two nearest locations (17 and 21 mm from the
  notch) and at its first twin;
- D6 and N1 at N1's first far control (379 mm away).

It passes only if all four of these hold:

- all 11 runs finished;
- all 11 took the same number of increments, which shows the fixed step is in force;
- the far control differs from D6 by at most 5 g RMS;
- the near difference is at least twice the far one.

Under the old settings the far-field difference was as large as the near one; the gate is there so
that the campaign's core-hours are not spent if the clean settings don't fix that. If it fails, nothing more
is solved and `pilot_gate_failed` is written. Read `pilot_report.txt` and decide:

- to ignore it: `sbatch --export=ALL,SKIP_PILOT_GATE=1 abaqus_scripts/delta/hood_local_edits.sbatch`;
- to rerun the pilot: delete `pilot_gate_failed` and submit again.

## 4. Bring the results back

```bash
cd <Code dir on Delta>
tar -czf local_edits_results.tar.gz \
    Data/HoodImpact_LocalEdits/output_history_acc \
    Data/HoodImpact_LocalEdits/generation_metrics.csv \
    Data/HoodImpact_LocalEdits/pilot_report.txt \
    Data/HoodImpact_LocalEdits/qa_report.txt
# optional, ~5 GB: full-rate raw accelerations (Time, A1, A2, A3 at every increment)
tar -czf local_edits_raw.tar.gz -C /u/hkazemi/hood_local_edits_scratch $(cd /u/hkazemi/hood_local_edits_scratch && ls -d run_*/HoodImpact_*_raw_acc.csv)
```

```powershell
scp hkazemi@dt-login04.delta.ncsa.illinois.edu:<Code dir on Delta>/local_edits_results.tar.gz D:\Papers\Hood_Impact\Code\
cd D:\Papers\Hood_Impact\Code ; tar -xzf local_edits_results.tar.gz
python abaqus_scripts/check_local_edits.py      # QA by role and distance, writes qa_local_edits.csv
```

The results are `output_history_acc/HoodImpact_<run>_SAE1000_interp1000.csv`. Each has 1000 points
over 0–25 ms with columns A1, A2, A3 and `A(in g)`, on the same time grid for every run.

The manifest maps each run to its design, location and role. Compute HIC with `hic15.batched_hic`,
not the `hic15` column of `generation_metrics.csv`: that column only searches the longest windows and
under-reports HIC.
