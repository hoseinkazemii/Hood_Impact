#!/bin/bash
# Design-difference audit: is the design-4 vs design-5 acceleration difference
# geometry, solver set-up, or chaos? Solves two deck sets in one job, so the
# FlexNet pool (65 tokens, 5 per cpus=1 solve, 13 lanes max) is never
# oversubscribed by two jobs at once:
#
#   1. Data/HoodImpact_SolverFactorial  (make_solver_factorial_decks.py)
#      105 decks: designs 4 and 5 at 15 locations x {fixed dt + gap start,
#      fixed dt, gap start}, plus design 4 under fixed dt + gap start with a
#      1e-3 mm hood jitter (the chaotic floor under the clean settings).
#      Runs are numbered cell-major with the clean cell first.
#   2. Data/HoodImpact_NoiseFloor       (make_noise_floor_decks.py)
#      30 decks: design 4 at 10 locations x {control, 1e-3 mm, 1e-1 mm jitter}
#      under the native settings.
#
# SUBMIT FROM THE Code DIRECTORY (holding abaqus_scripts/ and Data/). Decks are
# generated here if absent, which needs Data/HoodImpact_1704_EuroNCAP
# (inp_files + output_history_acc + manifest_1704.csv). Otherwise upload the
# two folders generated on Windows (~330 MB + ~130 MB).
#     sbatch abaqus_scripts/delta/hood_design_difference_audit.sbatch
#
# Budget (Delta ~0.133 s per explicit increment on one core): fixed-dt decks
# take ~55.6k increments (~2 h), native design 4 ~26.6k (~1 h), design 5 ~40.3k
# (~1.5 h). ~222 core-hours in total, ~19 h wall on 12 lanes. Resume-safe: if
# the walltime kills it, submit again; runs marked ok are skipped.
#
# Results land in:
#     Data/HoodImpact_SolverFactorial/output_history_acc/HoodImpact_<n>_SAE1000_interp1000.csv
#     Data/HoodImpact_NoiseFloor/output_history_acc/HoodImpact_<n>_SAE1000_interp1000.csv
# Bring both output_history_acc folders and generation_metrics.csv files back,
# then run (on Windows):
#     python abaqus_scripts/analyze_solver_factorial.py
#     python abaqus_scripts/analyze_noise_floor.py
#
# Monitor:  squeue -u hkazemi ; tail -f hood_dd_audit_<jobid>.out
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood_dd_audit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36g
#SBATCH -t 30:00:00
#SBATCH -o hood_dd_audit_%j.out
set -u

ROOT="$SLURM_SUBMIT_DIR"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -d "$ROOT/Data" ]; then
    echo "ERROR: submit this from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
LANES="${AUDIT_LANES:-12}"
umask 002
WORK="${AUDIT_WORK:-/u/hkazemi/hood_dd_audit_scratch}"
SCRATCH="${AUDIT_SCRATCH:-$WORK/abaqus_scratch}"
export WORK SCRATCH
mkdir -p "$WORK" "$SCRATCH"

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact

cd "$ROOT"
echo "=== node: $(hostname), $(date), repo: $ROOT, lanes: $LANES ==="
df -h "$ROOT" "$WORK" "$SCRATCH" 2>/dev/null || true
abaqus licensing ru | head -n 6 || true

solve_set() {
    # $1 dataset dir, $2 generator script, $3 work subdirectory
    local dataset="$1" generator="$2" sub="$3"
    if [ ! -d "$dataset/inp_files" ]; then
        echo "=== generating decks with $generator ==="
        python -u "$generator" || return 1
    else
        echo "=== reusing existing decks in $dataset/inp_files ==="
    fi
    local count
    count=$(ls -1 "$dataset/inp_files"/HoodImpact_*.inp 2>/dev/null | wc -l)
    if [ "$count" -lt 2 ]; then
        echo "ERROR: found $count decks in $dataset/inp_files"
        return 1
    fi
    echo "=== solving $count decks of $dataset ($(date)) ==="
    python -u abaqus_scripts/batch_generate.py \
        --inp-dir  "$dataset/inp_files" \
        --work-dir "$WORK/$sub" \
        --scratch-dir "$SCRATCH/$sub" \
        --out-dir  "$dataset" \
        --runs "1-${count}" --lanes "$LANES"
    cp -f "$WORK/$sub/generation_metrics.csv" "$dataset/" || true
}

# Factorial first: its clean cell and clean jitter floor answer the question.
solve_set "$ROOT/Data/HoodImpact_SolverFactorial" abaqus_scripts/make_solver_factorial_decks.py factorial || exit 1
solve_set "$ROOT/Data/HoodImpact_NoiseFloor" abaqus_scripts/make_noise_floor_decks.py noise_floor || exit 1

echo "=== done: $(date) ==="
echo "Bring back Data/HoodImpact_SolverFactorial/{output_history_acc,generation_metrics.csv}"
echo "and Data/HoodImpact_NoiseFloor/{output_history_acc,generation_metrics.csv}, then run"
echo "  python abaqus_scripts/analyze_solver_factorial.py"
