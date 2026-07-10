#!/bin/bash
# Production job: solve all 600 generated decks with 12 concurrent single-core
# Abaqus/Explicit lanes (60 of the 65 FlexNet tokens; ~5 left for other users
# of the license server).
#
# SUBMIT FROM THE Code DIRECTORY (the repo root that contains abaqus_scripts/
# and Data/):
#     sbatch abaqus_scripts/delta/hood600.sbatch
#
# The dataset lands in the repo:
#     Data/HoodImpact_600_EuroNCAP/output_history_acc/HoodImpact_<run>_SAE1000_interp1000.csv
# Transient solver files + raw acceleration csvs + metrics stay on project work
# scratch: /work/hdd/bbqg/hkazemi/hood600_work/
#
# Budget: ~600 x ~2650 s / 12 lanes = ~37 h wall, ~450 core-hours of the
# 1000 h bbqg-delta-cpu balance. If the job hits the walltime, JUST SUBMIT IT
# AGAIN -- the driver is resume-safe and skips every run already marked ok in
# the metrics csv (the smoke test's runs 1 and 32 are skipped the same way).
#
# Monitor:  squeue -u hkazemi ; tail -f hood600_<jobid>.out
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH -t 47:30:00
#SBATCH -o hood600_%j.out
set -u

ROOT="$SLURM_SUBMIT_DIR"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -d "$ROOT/Data" ]; then
    echo "ERROR: submit this from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
# Override these at submit time if a Delta quota is tight:
#     sbatch --export=ALL,HOOD600_WORK=/path,HOOD600_SCRATCH=/path ...
umask 002
WORK="${HOOD600_WORK:-/u/hkazemi/hood600_scratch}"
SCRATCH="${HOOD600_SCRATCH:-$WORK/abaqus_scratch}"
export WORK SCRATCH
mkdir -p "$WORK" "$SCRATCH"

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact

cd "$ROOT"
echo "=== node: $(hostname), $(date), repo: $ROOT ==="
echo "=== storage check ==="
quota -s 2>/dev/null || true
df -h "$ROOT" "$WORK" "$SCRATCH" 2>/dev/null || true
python - <<'PY' || exit 1
import os
for key in ("WORK", "SCRATCH"):
    path = os.environ[key]
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".hood_write_probe")
    with open(probe, "wb") as f:
        f.write(b"0" * 1048576)
        f.flush()
        os.fsync(f.fileno())
    os.remove(probe)
    print("write ok:", path)
PY
abaqus licensing ru | head -n 6 || true

python -u abaqus_scripts/batch_generate.py \
    --inp-dir  "$ROOT/Data/HoodImpact_600_EuroNCAP/inp_files" \
    --work-dir "$WORK/runs" \
    --scratch-dir "$SCRATCH/runs" \
    --out-dir  "$ROOT/Data/HoodImpact_600_EuroNCAP" \
    --runs 1-600 --lanes 12

echo "=== copying metrics into the repo dataset folder ==="
cp -f "$WORK/runs/generation_metrics.csv" "$ROOT/Data/HoodImpact_600_EuroNCAP/" || true
echo "=== done: $(date) ==="
