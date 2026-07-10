#!/bin/bash
# Smoke test on a Delta CPU node BEFORE the 600-run batch.
#
# SUBMIT FROM THE Code DIRECTORY (the repo root that contains abaqus_scripts/
# and Data/):
#     sbatch abaqus_scripts/delta/smoke_test.sbatch
#
# It validates, in one ~1 h job (3 cores, ~2.5 core-hours):
#   1. the FlexNet license server is reachable from a COMPUTE node (only the
#      login node has been tested so far)
#   2. accuracy: delivered run 1 re-solved with Abaqus 2026 and compared
#      against the delivered reference (expect RMSE < ~0.5 g, HIC err < ~1%)
#   3. two REAL samples of the 600 produced end-to-end with the production
#      driver: run 1 (design 0, far rear-corner location) and run 32
#      (design 0, the worst-case lifted location). Their SAE1000 csvs land in
#      Data/HoodImpact_600_EuroNCAP/output_history_acc/ and already count --
#      the production job will skip nothing (they are in its metrics file).
#   4. per-run solve time on a Delta core -> confirms the production job's
#      47.5 h walltime (local machine: ~2650 s/run)
#
# Prerequisite (login node, from the Code dir):
#     conda activate hood_impact
#     python abaqus_scripts/generate_impact_decks.py --designs 0 --locs 1,32
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8g
#SBATCH -t 03:00:00
#SBATCH -o smoke_%j.out
set -u

ROOT="$SLURM_SUBMIT_DIR"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -d "$ROOT/Data" ]; then
    echo "ERROR: submit this from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
# Transient solver files (~1 GB per lane) go to project work storage, NOT
# the repo. Override these at submit time if a Delta quota is tight:
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

echo "=== 1. license check from the compute node ==="
abaqus licensing ru | head -n 30 || echo "LICENSE CHECK FAILED"

echo "=== 2. delivered run 1 vs reference  +  3. generated runs 1,32 (parallel) ==="
python -u abaqus_scripts/run_hood_impact.py --runs 1 \
    --work-dir "$WORK/smoke_delivered" \
    --scratch-dir "$SCRATCH/smoke_delivered" \
    --compare --plot > "$WORK/smoke_delivered_run1.log" 2>&1 &

python -u abaqus_scripts/batch_generate.py --runs 1,32 --lanes 2 \
    --inp-dir  "$ROOT/Data/HoodImpact_600_EuroNCAP/inp_files" \
    --work-dir "$WORK/runs" \
    --scratch-dir "$SCRATCH/runs" \
    --out-dir  "$ROOT/Data/HoodImpact_600_EuroNCAP" \
    > "$WORK/smoke_generated.log" 2>&1 &
wait

echo ""
echo "=== delivered run 1 (peak/HIC should be within ~1% of reference) ==="
tail -n 15 "$WORK/smoke_delivered_run1.log"
echo ""
echo "=== generated runs 1 and 32 (look for 'RUN n ok: solve=...s peak=... HIC15=...') ==="
tail -n 15 "$WORK/smoke_generated.log"
echo ""
echo "=== produced dataset files ==="
ls -l "$ROOT/Data/HoodImpact_600_EuroNCAP/output_history_acc/" 2>/dev/null
echo "=== done: $(date) ==="
