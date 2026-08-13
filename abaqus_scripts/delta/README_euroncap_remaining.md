# Completing the EuroNCAP-style impact grid on Delta

The candidate CSV contains 142 data rows (the apparent 143rd line is its
header). The completed dataset used 50 of those locations for every design,
so the exact complement is:

- 92 remaining locations per design
- 12 designs x 92 = 1,104 new simulations
- 600 existing + 1,104 new = 1,704 total EuroNCAP-style simulations

The five IndustryLike locations are not part of this count or the new target
set. The 60 IndustryLike input decks are still required on Delta only as the
12 finite-element design/physics templates (runs 1, 6, ..., 56).

## Submit

After pulling this change, start from the repository's `Code` directory:

```bash
sbatch abaqus_scripts/delta/hood1104_remaining.sbatch
```

That one command creates any missing `.inp` decks, checks that the 92 new
locations are disjoint from the prior 50 and cover the full 142-point grid,
validates all 1,104 deck IDs, and starts the parallel Abaqus run.

The generated dataset is isolated from the completed 600 runs:

```text
Data/HoodImpact_1104_EuroNCAP_Remaining/
  impact_locations_remaining_92.csv
  manifest_1104.csv
  ImpactCoords_1104.csv
  inp_files/HoodImpact_1.inp ... HoodImpact_1104.inp
  output_history_acc/HoodImpact_1_SAE1000_interp1000.csv ...
```

Resume after a walltime stop by submitting the exact same command. The wrapper
uses `/u/hkazemi/hood1104_remaining_scratch` by default and skips runs already
marked `ok` there. Do not point it at the old `hood600_scratch`, whose run IDs
would collide. Do not overlap two submissions: the wrapper holds a lock on the
shared resume state and refuses to start while another job is using it.
Override storage only when necessary:

```bash
sbatch --export=ALL,HOOD1104_WORK=/safe/persistent/path \
  abaqus_scripts/delta/hood1104_remaining.sbatch
```

The wrapper exits successfully only after all 1,104 runs have both an `ok`
metric and an acceleration-history CSV. If Slurm reports the job as failed
after individual Abaqus failures, use the same submission command; successful
runs remain skipped and unfinished ones are retried.

Monitor with:

```bash
squeue -u hkazemi
tail -f hood1104_<jobid>.out
```

At 12 serial Abaqus lanes, prior timings imply roughly 95-125 ideal wall-hours,
or about 4-5 submissions with the current 30-hour limit. Check the current CPU
allocation before starting; this expansion may require more than 1,000
CPU-hours. Allow roughly 4.2 GiB for the new decks, about 11 GiB peak transient
solver space at 12 lanes, and roughly 3.7 GiB for retained raw histories under
the work root, plus the much smaller filtered result histories.
