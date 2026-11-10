#!/bin/bash
# Local-edit dataset: 51 beam-notch designs of hood design 6 x 20 impact
# locations each + design 6 itself at all 142 locations = 1162 Abaqus/Explicit
# solves, every one with a fixed 4.5e-7 s increment and the headform starting
# 1.0 mm clear of the hood. Plan: Data/HoodImpact_LocalEdits/manifest_local_edits.csv
# (abaqus_scripts/local_edit_doe.py); details in delta/README_local_edits.md.
#
# What one submission does:
#   1. writes the 1162 decks if absent (make_local_edit_decks.py, ~10 min, 5.5 GB)
#   2. solves the 11 pilot runs (one round of 12 lanes, ~2 h)
#   3. pilot gate (check_local_edits.py --pilot): every pilot run finished,
#      all took the same number of increments, the far control matches design 6
#      and the near difference stands clearly above it. FAIL stops everything
#      (and cancels the queued follow-up job); read pilot_report.txt.
#   4. solves the remaining runs, 12 single-core lanes (5 FlexNet tokens each)
#   5. writes qa_report.txt over everything finished so far
#
# Walltime: ~2 h per run on one Delta core (55.6k increments), so the campaign
# needs ~195 h of wall time on 12 lanes (~2,350 core-hours). The job queues
# its own follow-up (afterany) at the start, so a walltime kill continues
# automatically; runs marked ok are never redone. Chaining stops when all runs
# are done, after LOCAL_EDITS_CHAIN_LEFT follow-ups (default 6), or when a
# whole job made no progress.
#
# SUBMIT FROM THE Code DIRECTORY (holding abaqus_scripts/ and Data/):
#     sbatch abaqus_scripts/delta/hood_local_edits.sbatch
#
# Monitor:  squeue -u hkazemi ; tail -f hood_local_edits_<jobid>.out
# Stop:     scancel -u hkazemi -n hood_local_edits
#
#SBATCH -A bbqg-delta-cpu
#SBATCH -p cpu
#SBATCH -J hood_local_edits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36g
#SBATCH -t 47:30:00
#SBATCH -o hood_local_edits_%j.out
set -u

ROOT="$SLURM_SUBMIT_DIR"
SELF="abaqus_scripts/delta/hood_local_edits.sbatch"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -f "$ROOT/$SELF" ]; then
    echo "ERROR: submit this from the Code directory (needs abaqus_scripts/ and Data/)"
    exit 1
fi
DATA="$ROOT/Data/HoodImpact_LocalEdits"
MANIFEST="$DATA/manifest_local_edits.csv"
LANES="${LOCAL_EDITS_LANES:-12}"
CHAIN_LEFT="${LOCAL_EDITS_CHAIN_LEFT:-6}"
PREV_OK="${LOCAL_EDITS_PREV_OK:--1}"
umask 002
WORK="${LOCAL_EDITS_WORK:-/u/hkazemi/hood_local_edits_scratch}"
SCRATCH="$WORK/abaqus_scratch"
METRICS="$WORK/generation_metrics.csv"
mkdir -p "$WORK" "$SCRATCH"

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact

cd "$ROOT"
echo "=== node: $(hostname), $(date), job $SLURM_JOB_ID, repo: $ROOT, lanes: $LANES, chain left: $CHAIN_LEFT ==="
if [ ! -f "$MANIFEST" ] || [ ! -f "$DATA/local_edit_doe.csv" ]; then
    echo "ERROR: missing $MANIFEST or local_edit_doe.csv -- copy Data/HoodImpact_LocalEdits/*.csv from Windows"
    exit 1
fi
if [ -f "$DATA/pilot_gate_failed" ]; then
    echo "STOP: the pilot gate failed earlier (see $DATA/pilot_report.txt)."
    echo "Remove $DATA/pilot_gate_failed to try again, or resubmit with --export=ALL,SKIP_PILOT_GATE=1 to ignore it."
    [ "${SKIP_PILOT_GATE:-0}" = "1" ] || exit 1
fi

N_TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
PILOT_RUNS=$(python -c "import csv; print(','.join(r['run'] for r in csv.DictReader(open('$MANIFEST')) if r['pilot'] == '1'))")
count_ok() {
    python - "$METRICS" <<'EOF'
import csv, os, sys
p = sys.argv[1]
print(len({r["run"] for r in csv.DictReader(open(p)) if r["status"] == "ok"}) if os.path.exists(p) else 0)
EOF
}
OK_START=$(count_ok)
echo "runs: $N_TOTAL in the plan, $OK_START already ok; pilot runs: $PILOT_RUNS"
if [ "$OK_START" -ge "$N_TOTAL" ]; then
    echo "=== nothing left to solve ==="
    exit 0
fi
if [ "$OK_START" -eq "$PREV_OK" ]; then
    echo "STOP: the previous job finished no run ($OK_START ok before and after). Check its log, then resubmit by hand."
    exit 1
fi

# queue the follow-up now: it starts when this job ends, even after a walltime kill
NEXT=""
if [ "$CHAIN_LEFT" -gt 0 ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,LOCAL_EDITS_CHAIN_LEFT=$((CHAIN_LEFT - 1)),LOCAL_EDITS_PREV_OK="$OK_START" \
        "$ROOT/$SELF" 2>&1) && echo "queued follow-up job $NEXT" \
        || { echo "WARNING: could not queue a follow-up ($NEXT); resubmit by hand if this job hits its walltime"; NEXT=""; }
fi

echo "=== storage ==="
quota -s 2>/dev/null | head -n 8 || true
df -h "$ROOT" "$WORK" 2>/dev/null || true
abaqus licensing ru 2>&1 | head -n 6 || true

echo "=== decks ($(date)) ==="
python -u abaqus_scripts/make_local_edit_decks.py --skip-existing --verify 50 || {
    echo "ERROR: deck generation failed"; [ -n "$NEXT" ] && scancel "$NEXT"; exit 1; }
N_DECKS=$(ls -1 "$DATA/inp_files"/HoodImpact_*.inp 2>/dev/null | wc -l)
if [ "$N_DECKS" -ne "$N_TOTAL" ]; then
    echo "ERROR: $N_DECKS decks for $N_TOTAL runs"; [ -n "$NEXT" ] && scancel "$NEXT"; exit 1
fi

solve() {
    python -u abaqus_scripts/batch_generate.py \
        --inp-dir "$DATA/inp_files" --work-dir "$WORK" --scratch-dir "$SCRATCH" \
        --out-dir "$DATA" --runs "$1" --lanes "$LANES"
    cp -f "$METRICS" "$DATA/generation_metrics.csv" 2>/dev/null || true
}

if [ ! -f "$DATA/pilot_gate_passed" ]; then
    echo "=== pilot: runs $PILOT_RUNS ($(date)) ==="
    solve "$PILOT_RUNS"
    if python -u abaqus_scripts/check_local_edits.py --pilot --metrics "$METRICS" --out "$DATA/pilot_report.txt"; then
        touch "$DATA/pilot_gate_passed"
    elif [ "${SKIP_PILOT_GATE:-0}" = "1" ]; then
        echo "WARNING: pilot gate failed, continuing because SKIP_PILOT_GATE=1"
    else
        touch "$DATA/pilot_gate_failed"
        echo "STOP: pilot gate failed -- see $DATA/pilot_report.txt. No further runs are solved."
        [ -n "$NEXT" ] && scancel "$NEXT" && echo "cancelled follow-up job $NEXT"
        exit 1
    fi
fi

echo "=== campaign: runs 1-$N_TOTAL ($(date)) ==="
solve "1-$N_TOTAL"
python -u abaqus_scripts/check_local_edits.py --metrics "$METRICS" --out "$DATA/qa_report.txt" || true

OK_END=$(count_ok)
echo "=== $(date): $OK_END of $N_TOTAL runs ok ==="
if [ "$OK_END" -ge "$N_TOTAL" ] && [ -n "$NEXT" ]; then
    scancel "$NEXT" && echo "all done: cancelled follow-up job $NEXT"
fi
echo "Bring back Data/HoodImpact_LocalEdits/{output_history_acc,generation_metrics.csv,pilot_report.txt,qa_report.txt}"
