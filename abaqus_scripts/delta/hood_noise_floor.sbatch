#!/bin/bash
# Noise-floor experiment: solve the 30 decks written by
# abaqus_scripts/make_noise_floor_decks.py (10 impact locations on design 4 x
# {control, jitter 1e-3 mm, jitter 1e-1 mm}).
#
# The question: designs 4 and 5 differ by up to 29 g RMS at some impact
# locations while their hoods differ at 0.26 % of nodes. How much of that does
# an explicit solve produce from a perturbation with no physical meaning?
#
# SUBMIT FROM THE Code DIRECTORY (the repo root holding abaqus_scripts/ and
# Data/), after generating the decks:
#     python abaqus_scripts/make_noise_floor_decks.py
#     sbatch abaqus_scripts/delta/hood_noise_floor.sbatch
#
# Results land in:
#     Data/HoodImpact_NoiseFloor/output_history_acc/HoodImpact_<n>_SAE1000_interp1000.csv
# Transient solver files stay on work scratch.
#
# Budget: 30 x ~2650 s / 10 lanes = ~2.2 h wall, ~22 core-hours. The driver is
# resume-safe: if the walltime kills it, just submit again and it skips every
# run already marked ok in the metrics csv.
#
# Control and perturbed decks are solved in the SAME job so that the only
# difference between them is the perturbation, not the environment.
#
# Monitor:  squeue -u hkazemi ; tail -f hood_noise_floor_<jobid>.out
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood_noise_floor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20g
#SBATCH -t 08:00:00
#SBATCH -o hood_noise_floor_%j.out
set -u

ROOT="$SLURM_SUBMIT_DIR"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -d "$ROOT/Data" ]; then
    echo "ERROR: submit this from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
DATASET="$ROOT/Data/HoodImpact_NoiseFloor"
umask 002
WORK="${NOISE_FLOOR_WORK:-/u/hkazemi/hood_noise_floor_scratch}"
SCRATCH="${NOISE_FLOOR_SCRATCH:-$WORK/abaqus_scratch}"
export WORK SCRATCH
mkdir -p "$WORK" "$SCRATCH"

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact

cd "$ROOT"
echo "=== node: $(hostname), $(date), repo: $ROOT ==="

# The decks are ~130 MB, so they are generated here rather than committed.
# Generation is deterministic (fixed seed), and the log records the selection.
if [ ! -d "$DATASET/inp_files" ]; then
    echo "=== generating noise-floor decks ==="
    python -u abaqus_scripts/make_noise_floor_decks.py || exit 1
else
    echo "=== reusing existing decks in $DATASET/inp_files ==="
fi
NUM_DECKS=$(ls -1 "$DATASET/inp_files"/HoodImpact_*.inp 2>/dev/null | wc -l)
if [ "$NUM_DECKS" -lt 2 ]; then
    echo "ERROR: found $NUM_DECKS decks in $DATASET/inp_files; expected the full set."
    exit 1
fi

echo "=== solving $NUM_DECKS noise-floor decks ==="
df -h "$ROOT" "$WORK" "$SCRATCH" 2>/dev/null || true
abaqus licensing ru | head -n 6 || true

python -u abaqus_scripts/batch_generate.py \
    --inp-dir  "$DATASET/inp_files" \
    --work-dir "$WORK/runs" \
    --scratch-dir "$SCRATCH/runs" \
    --out-dir  "$DATASET" \
    --runs "1-${NUM_DECKS}" --lanes 10

echo "=== copying metrics into the dataset folder ==="
cp -f "$WORK/runs/generation_metrics.csv" "$DATASET/" || true
echo "=== done: $(date) ==="
echo "Now run:  python abaqus_scripts/analyze_noise_floor.py"
