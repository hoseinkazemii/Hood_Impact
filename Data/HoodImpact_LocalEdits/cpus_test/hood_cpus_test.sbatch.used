#!/bin/bash
# Test (~1.5 h, 2h15 limit) before the local-edit campaign leaves single-core
# lanes. With the license to itself it measures:
#   * speed: how many FlexNet tokens one solve on N cores checks out (N = 1 4 8
#     16 32), how fast it is alone and as a whole layout (lanes x cores) sharing
#     the node, and so which layout finishes the campaign soonest in 65 tokens
#   * noise: the 11 pilot runs (D6, N1, N2 at 4 locations; finished on 1 core)
#     solved again on that layout, run 5 twice. Verdict PASS only if the same
#     deck repeats bit for bit and every near/twin difference curve
#     (design - D6) keeps its shape (R2 >= 0.95) and HIC change (within 0.5 pts)
# Driver and details: abaqus_scripts/cpus_scaling_test.py. Test solves run in
# their own scratch folder; the campaign's metrics and curves are not touched.
#
# The test needs every token: no hood_local_edits job may run meanwhile.
#     squeue -u hkazemi -n hood_local_edits     # running + pending campaign jobs
#     scontrol hold <each pending job id>       # keeps the next one from starting
#     (let the running one end, or scancel it: its unfinished runs are redone later)
#     cd <Code dir> && sbatch abaqus_scripts/delta/hood_cpus_test.sbatch
# It stops at once if a campaign job is running.
#
# Result: Data/HoodImpact_LocalEdits/cpus_test/report.txt, noise/ (tables, the
# N-core curves, noise_pairs.png) and layout.env. Only if the NOISE VERDICT is
# PASS, run the campaign with that layout:
#     cp Data/HoodImpact_LocalEdits/cpus_test/layout.env Data/HoodImpact_LocalEdits/campaign_layout.env
#     scancel <the held campaign jobs>   # they were submitted with 12 cores
#     sbatch -c <cores> --mem=<mem> --array=1-4%1 abaqus_scripts/delta/hood_local_edits.sbatch
# Otherwise release the held jobs (scontrol release <id>) and stay single-core.
#
# Options (sbatch --export=ALL,NAME=value,...):
#     CPUS_TEST_PHASES=solo,layout,noise  e.g. =noise to repeat only the noise test
#     CPUS_TEST_CPUS="1 4 8 16 32"        solo core counts ('8:mpi' tries MPI instead of threads)
#     CPUS_TEST_NOISE_CPUS=8              cores per solve in the noise test (default: the chosen layout's)
#     CPUS_TEST_NOISE_RUNS=pilot          runs re-solved (finished on 1 core), e.g. 1-11,12-16
#     CPUS_TEST_RUN=1                     the run whose deck the speed phases cut short
#     CPUS_TEST_RESERVE=0                 tokens to leave free for other license users
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood_cpus_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64g
#SBATCH -t 02:15:00
#SBATCH -o hood_cpus_test_%j.out
set -u

SELF="abaqus_scripts/delta/hood_cpus_test.sbatch"
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
case "$SCRIPT_PATH" in /*) ;; ?*) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;; esac
case "$SCRIPT_PATH" in
    */"$SELF") ROOT="${SCRIPT_PATH%/"$SELF"}" ;;
    *) ROOT="$SLURM_SUBMIT_DIR" ;;
esac
echo "=== node: $(hostname), $(date), job $SLURM_JOB_ID, ${SLURM_CPUS_PER_TASK:-?} cores ==="
if [ ! -f "$ROOT/abaqus_scripts/cpus_scaling_test.py" ]; then
    echo "ERROR: cannot find the Code directory. Submit as: cd <Code dir> && sbatch $SELF"
    exit 1
fi
DATA="$ROOT/Data/HoodImpact_LocalEdits"
OUT="$DATA/cpus_test"
mkdir -p "$OUT"
cp -f "$0" "$OUT/hood_cpus_test.sbatch.used" 2>/dev/null

BUSY=$(squeue -h -u "$USER" -n hood_local_edits -t RUNNING -o %A 2>/dev/null | tr '\n' ' ')
if [ -n "${BUSY// /}" ]; then
    echo "STOP: hood_local_edits job(s) $BUSY are running and hold the license tokens."
    echo "      Hold the pending ones (scontrol hold <id>), wait for the running one to end, then resubmit."
    exit 1
fi

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact || { echo "ERROR: conda activate hood_impact failed"; exit 1; }

WORK="${CPUS_TEST_WORK:-/u/hkazemi/hood_cpus_test_scratch}"
CAMPAIGN_METRICS="${LOCAL_EDITS_WORK:-/u/hkazemi/hood_local_edits_scratch}/generation_metrics.csv"
[ -f "$CAMPAIGN_METRICS" ] || CAMPAIGN_METRICS="$DATA/generation_metrics.csv"
mkdir -p "$WORK"

# finish 3 min before the time limit so the report is always written
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
DEADLINE=""
case "$END" in 20[0-9][0-9]-*) DEADLINE=$(( $(date -d "$END" +%s) - 180 )) ;; esac

echo "=== cpu, allocation ==="
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)" || true
accounts 2>/dev/null | head -n 12 || true

cd "$ROOT"
python -u abaqus_scripts/cpus_scaling_test.py \
    --run "${CPUS_TEST_RUN:-1}" --data-dir "$DATA" --metrics "$CAMPAIGN_METRICS" \
    --work-dir "$WORK" --out-dir "$OUT" --cpus "${CPUS_TEST_CPUS:-1 4 8 16 32}" \
    --phases "${CPUS_TEST_PHASES:-solo,layout,noise}" --reserve "${CPUS_TEST_RESERVE:-0}" \
    --noise-runs "${CPUS_TEST_NOISE_RUNS:-pilot}" \
    ${CPUS_TEST_NOISE_CPUS:+--noise-cpus "$CPUS_TEST_NOISE_CPUS"} \
    ${DEADLINE:+--deadline "$DEADLINE"}
RC=$?

rm -rf "$WORK"
cp -f "$SLURM_SUBMIT_DIR/hood_cpus_test_${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null
echo "=== done (exit $RC): $OUT/report.txt ==="
[ -f "$OUT/layout.env" ] && cat "$OUT/layout.env"
exit $RC
