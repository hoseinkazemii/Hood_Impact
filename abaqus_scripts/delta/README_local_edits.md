# Local-edit dataset on Delta

1162 Abaqus/Explicit solves for learning local design changes:

- 51 beam-notch designs of hood design 6 (N1–N51), 20 impact locations each = 1020 runs;
- design 6 itself ("D6") at all 142 Euro-NCAP locations = 142 runs.

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
   time step, which is what contaminated the delivered near-clone pairs.

## Which locations each design is run at (all 51 designs: 16 + 2 + 2)

| role | count | what |
|---|---|---|
| near | 16 | the 16 locations nearest the edit on the +y half and the centreline (the 16th lies 115–278 mm away, median 182 mm) |
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

## 1. Pre-flight (login node, 1 minute)

```bash
quota -s     # needs ~25 GB free in home: 5.5 GB decks, ~5 GB raw csvs, ~11 GB solver scratch
abaqus licensing lmstat -c 27000@130.126.241.64 -i | head -20   # expiry after today, ABAQUSLM UP
conda activate hood_impact
python abaqus_scripts/make_local_edit_decks.py --runs 1-11 --verify 1   # writes and checks the 11 pilot decks
```

If home is short of space, the old `inp_files` folders of `HoodImpact_600_EuroNCAP` and
`HoodImpact_1104_EuroNCAP_Remaining` can be deleted: they can be regenerated with
`generate_impact_decks.py`.

## 2. Submit once

```bash
sbatch abaqus_scripts/delta/hood_local_edits.sbatch
```

The job finds its Code directory from the script's own path, so it reads and writes
`<Code dir>/Data/HoodImpact_LocalEdits` wherever you submit from. That has to be the folder the
tarball was unpacked in. The job:

1. writes all 1162 decks (about 10 minutes);
2. solves the 11 pilot runs (one round of 12 lanes, about 2 h);
3. runs the pilot gate (`check_local_edits.py --pilot`, see `Data/HoodImpact_LocalEdits/pilot_report.txt`);
4. solves the rest on 12 single-core lanes (5 FlexNet tokens each, 60 of the 65).

**Budget.** A run takes about 2 h on one Delta core: 55.6k increments at ~0.13 s each. 1162 runs on
12 lanes come to about 195 h of wall time, or about 2,350 core-hours.

**Follow-up jobs.** Each job queues its own follow-up (`afterany`), so a walltime kill continues on
its own; finished runs are never redone. The chain stops when every run is done, after 6
follow-ups, or when a whole job finished no run.

**Right after submitting**, `squeue -u hkazemi` should show the running job and a
follow-up waiting with reason `(Dependency)`. If the queue is empty a minute after the job started,
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
squeue -u hkazemi
cat Data/HoodImpact_LocalEdits/job_logs/status.txt
tail -f hood_local_edits_<jobid>.out         # in the folder you submitted from
scancel -u hkazemi -n hood_local_edits       # stops the running job and the queued follow-up
```

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
that 2,350 core-hours are not spent if the clean settings don't fix that. If it fails, nothing more
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
