#!/bin/bash
# Complete the project's EuroNCAP-style grid: 12 designs x 92 previously
# unused locations = 1,104 new Abaqus/Explicit solves. Together with the
# completed 600-run dataset, this gives 12 x 142 = 1,704 EuroNCAP samples.
#
# SUBMIT FROM THE Code DIRECTORY (the repo root containing abaqus_scripts/
# and Data/):
#     sbatch abaqus_scripts/delta/hood1104_remaining.sbatch
#
# The wrapper first creates any missing decks, validates the exact complement,
# and then runs 12 concurrent single-core solves. It is resume-safe: submit the
# same command again after every walltime stop. Keep the work root unchanged so
# completed runs remain recorded in generation_metrics.csv.
#
# Expected from prior timings: roughly 95-125 ideal wall-hours in total, or
# about 4-5 submissions at this 30-hour walltime. Check the current allocation
# balance before submitting; the expansion may require >1,000 CPU-hours.
#
# Monitor: squeue -u hkazemi ; tail -f hood1104_<jobid>.out
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood1104
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH -t 30:00:00
#SBATCH -o hood1104_%j.out
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
DATASET="$ROOT/Data/HoodImpact_1104_EuroNCAP_Remaining"
LOCATIONS="$DATASET/impact_locations_remaining_92.csv"
TEMPLATE_DIR="$ROOT/Data/HoodImpact_60_IndustryLike/inp_files"

if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -d "$ROOT/Data" ]; then
    echo "ERROR: submit from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
for required in \
    "$LOCATIONS" \
    "$ROOT/Data/HoodImpact_600_EuroNCAP/impact_locations_50.csv" \
    "$ROOT/Data/HoodImpact_600_EuroNCAP/impact_grid_candidates.csv" \
    "$ROOT/Data/HoodImpact_60_IndustryLike/ImpactCoords_60.csv"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: missing required input: $required"
        exit 1
    fi
done
for base_run in 1 6 11 16 21 26 31 36 41 46 51 56; do
    template="$TEMPLATE_DIR/HoodImpact_${base_run}.inp"
    if [ ! -f "$template" ]; then
        echo "ERROR: missing ignored design-template deck: $template"
        exit 1
    fi
done

# Use a new persistent namespace. Reusing hood600_scratch would make its old
# run IDs look complete and incorrectly skip new simulations.
umask 002
WORK="${HOOD1104_WORK:-/u/hkazemi/hood1104_remaining_scratch}"
SCRATCH="${HOOD1104_SCRATCH:-$WORK/abaqus_scratch}"
export ROOT DATASET LOCATIONS WORK SCRATCH
mkdir -p "$DATASET" "$WORK" "$SCRATCH"
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to protect the shared resume state"
    exit 1
fi
exec 9>"$WORK/hood1104.lock"
if ! flock -n 9; then
    echo "ERROR: another hood1104 job is already using $WORK"
    echo "Do not overlap resubmissions; wait for the existing job to end."
    exit 1
fi

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact
if ! command -v abaqus >/dev/null 2>&1; then
    echo "ERROR: abaqus launcher is not available on PATH"
    exit 1
fi

cd "$ROOT"
echo "=== node: $(hostname), $(date), repo: $ROOT ==="
echo "=== storage check ==="
quota -s 2>/dev/null || true
df -h "$ROOT" "$WORK" "$SCRATCH" 2>/dev/null || true
python - <<'PY'
import os

for key in ("WORK", "SCRATCH"):
    path = os.environ[key]
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".hood1104_write_probe")
    with open(probe, "wb") as f:
        f.write(b"0" * 1048576)
        f.flush()
        os.fsync(f.fileno())
    os.remove(probe)
    print("write ok:", path)
PY
abaqus licensing ru | head -n 6 || true

echo "=== generating missing decks (existing decks are preserved) ==="
python -u abaqus_scripts/generate_impact_decks.py \
    --locations "$LOCATIONS" \
    --out-dir "$DATASET" \
    --skip-existing

echo "=== validating 92-location complement and 1,104 decks ==="
python - <<'PY'
import csv
import glob
import os
import re

root = os.environ["ROOT"]
dataset = os.environ["DATASET"]

def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

old = rows(os.path.join(
    root, "Data", "HoodImpact_600_EuroNCAP", "impact_locations_50.csv"))
new = rows(os.path.join(dataset, "impact_locations_remaining_92.csv"))
candidates = rows(os.path.join(
    root, "Data", "HoodImpact_600_EuroNCAP", "impact_grid_candidates.csv"))

def key(row):
    return int(row["col_k"]), round(float(row["wadh_mm"]), 6)

old_keys = {key(row) for row in old}
new_keys = {key(row) for row in new}
candidate_keys = {key(row) for row in candidates}
assert len(old_keys) == 50, "expected 50 unique prior locations"
assert len(new_keys) == 92, "expected 92 unique remaining locations"
assert old_keys.isdisjoint(new_keys), "old and remaining locations overlap"
assert old_keys | new_keys == candidate_keys, "locations do not cover the grid"

deck_paths = glob.glob(os.path.join(dataset, "inp_files", "HoodImpact_*.inp"))
deck_ids = set()
for path in deck_paths:
    match = re.fullmatch(r"HoodImpact_(\d+)\.inp", os.path.basename(path))
    if match:
        assert os.path.getsize(path) >= 1000000, "incomplete deck: %s" % path
        deck_ids.add(int(match.group(1)))
assert deck_ids == set(range(1, 1105)), "deck IDs must be exactly 1..1104"

manifest = rows(os.path.join(dataset, "manifest_1104.csv"))
assert len(manifest) == 1104, "manifest must have 1,104 rows"
assert {int(row["run"]) for row in manifest} == set(range(1, 1105))
for design in range(12):
    assert sum(int(row["design"]) == design for row in manifest) == 92
print("validated: 50 prior + 92 remaining = 142 locations; 1,104 decks ready")
PY

echo "=== solving runs 1-1104 with 12 serial Abaqus lanes ==="
python -u abaqus_scripts/batch_generate.py \
    --inp-dir "$DATASET/inp_files" \
    --work-dir "$WORK/runs" \
    --scratch-dir "$SCRATCH/runs" \
    --out-dir "$DATASET" \
    --runs 1-1104 --lanes 12

echo "=== copying metrics into the dataset folder ==="
cp -f "$WORK/runs/generation_metrics.csv" "$DATASET/" || true

echo "=== verifying all 1,104 simulations completed successfully ==="
python - <<'PY'
import csv
import glob
import os
import re

dataset = os.environ["DATASET"]
metrics_path = os.path.join(os.environ["WORK"], "runs", "generation_metrics.csv")
with open(metrics_path, newline="") as f:
    metric_rows = list(csv.DictReader(f))
ok_runs = {int(row["run"]) for row in metric_rows if row.get("status") == "ok"}

output_ids = set()
pattern = re.compile(r"HoodImpact_(\d+)_SAE1000_interp1000\.csv")
for path in glob.glob(os.path.join(dataset, "output_history_acc", "*.csv")):
    match = pattern.fullmatch(os.path.basename(path))
    if match:
        output_ids.add(int(match.group(1)))

expected = set(range(1, 1105))
missing_ok = sorted(expected - ok_runs)
missing_outputs = sorted(expected - output_ids)
if missing_ok or missing_outputs:
    print("INCOMPLETE: resubmit the same sbatch command to retry unfinished runs")
    if missing_ok:
        print("runs without an ok metric (%d): %s" %
              (len(missing_ok), missing_ok[:30]))
    if missing_outputs:
        print("runs without an output history (%d): %s" %
              (len(missing_outputs), missing_outputs[:30]))
    raise SystemExit(1)
print("complete: 1,104 ok metrics and 1,104 acceleration histories")
PY
echo "=== done: $(date) ==="
