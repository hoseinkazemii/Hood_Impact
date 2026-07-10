# Generating the 600-sample hood-impact dataset on Delta

12 designs x 50 Euro-NCAP-grid impact locations = 600 Abaqus/Explicit solves.

Assumes the repo on Delta mirrors the local `Code/` layout (abaqus_scripts/,
Data/HoodImpact_60_IndustryLike/, ...). All commands run FROM the Code
directory. Everything -- the 600 decks, the SAE1000 output histories, and the
transient solver scratch files -- lives under `$HOME` (`/u/hkazemi`); see
"Storage, resolved" below for why, and don't move it back to `/work` without
re-verifying (see the same section).

Budget: ~450-500 of the 1000 CPU-hours on `bbqg-delta-cpu`. Wall time: at the
~3600 s/run observed on Delta (vs ~2650 s/run on the original local machine),
600 runs / 12 lanes is ~50 h -- just over one job's 47.5 h walltime cap, so
**expect to resubmit `hood600.sbatch` once** to finish the last runs; this is
by design (the driver skips everything already marked `ok`), not a failure.

---

## Storage, resolved (read this before changing paths)

Two unrelated storage problems blocked this pipeline for a while; both are
now fixed and the defaults below already reflect the fix.

1. **Home was genuinely full.** An old, unrelated 40 GB directory
   (`~/projects/mitral_valve`, leftover from a different project, including a
   git repo with large pack files) had home at 99.5/100 GB. Deleting it
   dropped usage to ~6 GB. **Always check `quota -s` first** if you see
   `OSError: [Errno 122] Disk quota exceeded` -- don't assume it's `/work`.

2. **`/work` (both `/work/nvme/bbqg` and `/work/hdd/bbqg`) was unreliable for
   real Abaqus writes**, independent of (1). `df` on `/work` intermittently
   reported the whole mount as 100% full / 0 avail even when the project's
   own `quota -s` allocation showed real headroom, and Abaqus's packager
   stage (`.cps`/`.stt` files) failed there with disk-space errors that a
   plain 1 MB write-probe in the same directory did not reproduce. This
   looked like live capacity/reliability pressure on Delta's shared `/work`
   Lustre pool, not anything specific to this pipeline. **Verified fix:**
   moved the transient solver scratch off `/work` entirely, onto
   `$HOME` (`/u/hkazemi/hood600_scratch`) -- confirmed working with a full,
   real solve (see the "Reference" section for the observed accuracy).

Since the batch driver deletes each run's big solver files right after
extraction (`cleanup_big_files`), peak scratch usage stays bounded to
`lanes x ~900 MB` regardless of total runs completed -- about 11 GB at the
production job's 12 lanes. Home's quota (100 GB soft / 103 GB hard) covers
that plus the ~4.8 GB of decks and ~70 MB of output csvs comfortably, now
that (1) is fixed. There is no need to symlink anything to `/work` or
`/projects` unless home fills up again.

The Slurm wrappers use `/u/hkazemi/hood600_scratch` as the scratch root by
default; override at submit time if ever needed:
```bash
sbatch --export=ALL,HOOD600_WORK=/some/other/path abaqus_scripts/delta/smoke_test.sbatch
```

---

## 0. One-time: sync the scripts to Delta

From `D:\Papers\Hood_Impact\Code` on Windows (single Duo prompt; the tarball
`hood600_scripts.tar.gz` is already built):

```powershell
scp hood600_scripts.tar.gz hkazemi@dt-login04.delta.ncsa.illinois.edu:<Code dir on Delta>/
```

On Delta, from the Code dir:
```bash
tar -xzf hood600_scripts.tar.gz
```

## 1. Smoke test (already passed 2026-07-08)

```bash
cd <Code dir on Delta>
conda activate hood_impact
python abaqus_scripts/generate_impact_decks.py --designs 0 --locs 1,32
sbatch abaqus_scripts/delta/smoke_test.sbatch
```

Confirmed on job 19992517: license OK from a compute node; delivered run 1
vs. reference -- peak g 0.01% off, HIC15 -0.03% off, RMSE 0.27 g; generated
runs 1 and 32 both solved (HIC15 1171 and 875) and their SAE1000 csvs are
already in `Data/HoodImpact_600_EuroNCAP/output_history_acc/` -- they count
toward the 600, the production run will skip them.

## 2. Generate the remaining 598 decks

```bash
python abaqus_scripts/generate_impact_decks.py      # all 600, ~3 min, ~4.8 GB, into Data/
```

Expected: every design reports `center clearance = 2.006 mm`, ~12-16
locations per design lifted (penetration cap 1.65 mm), ends with
`manifest_600.csv (600 rows)` + `ImpactCoords_600.csv`. Decks 1 and 32 are
rewritten identically to the smoke-test versions.

## 3. Production run (12 lanes, ~45-50 h wall, ~450-500 core-hours)

```bash
sbatch abaqus_scripts/delta/hood600.sbatch
```

* Progress: `tail -f hood600_<jobid>.out` -- one line per finished run with
  ETA; milestones every 20 runs. Runs 1 and 32 are skipped (already ok).
* **Expect to resubmit once** near the walltime cap (see budget note above)
  -- just `sbatch abaqus_scripts/delta/hood600.sbatch` again; completed runs
  are skipped via `Data/HoodImpact_600_EuroNCAP/generation_metrics.csv`
  (also mirrored at `~/hood600_scratch/runs/generation_metrics.csv` while
  running).
* License hiccups: 3 attempts per run, 90 s apart; launches staggered 15 s.

## 4. Bring the dataset back to Windows

Already in the repo's Data folder on Delta -- just zip and scp:

```bash
cd <Code dir on Delta>
tar -czf hood600_results.tar.gz \
    Data/HoodImpact_600_EuroNCAP/output_history_acc \
    Data/HoodImpact_600_EuroNCAP/manifest_600.csv \
    Data/HoodImpact_600_EuroNCAP/ImpactCoords_600.csv \
    Data/HoodImpact_600_EuroNCAP/impact_locations_50.csv \
    Data/HoodImpact_600_EuroNCAP/generation_metrics.csv
```

```powershell
scp hkazemi@dt-login04.delta.ncsa.illinois.edu:<Code dir on Delta>/hood600_results.tar.gz D:\Papers\Hood_Impact\Code\
cd D:\Papers\Hood_Impact\Code ; tar -xzf hood600_results.tar.gz
```

Full-precision raw acceleration csvs (~2 GB, optional) stay on scratch:
`~/hood600_scratch/runs/run_*/HoodImpact_*_raw_acc.csv`.

---

## Reference

* Run numbering: `run = 50*design + loc` (design 0-11, loc 1-50), i.e.
  `design = (run-1)//50` -- same convention as the delivered set.
* License math: one `cpus=1` job = 5 tokens; pool = 65 => 13 lanes max; the
  job uses 12. Do NOT raise `--cpus`: multicore burns ~40% more of the
  CPU-hour allocation for the same 600 samples.
* Headform placement: partner's measured convention (sphere bottom 2.006 mm
  above the outer skin under the center), with true sphere penetration capped
  at the delivered decks' own worst case (1.65 mm) by lifting -- see
  `manifest_600.csv` columns `sphere_pen_mm` / `lift_mm`.
* Smoke-test accuracy (job 19992517, 2026-07-08): delivered run 1 peak g
  147.33 vs ref 147.32 (0.01%), HIC15 1283.0 vs ref 1283.5 (-0.03%), RMSE
  0.27 g. Generated run 1: peak 122.3 g, HIC15 1171. Generated run 32: peak
  131.0 g, HIC15 875. Solve time ~3601 s/run on Delta (home-filesystem
  scratch) vs ~2650 s/run on the original local machine (local NVMe).
