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
# Tracking: Data/HoodImpact_LocalEdits/job_logs/status.txt gets one line per
# stage of every job; the Slurm log and generation_metrics.csv are copied into
# Data/HoodImpact_LocalEdits every 10 minutes and when the job ends; finished
# results appear in Data/HoodImpact_LocalEdits/output_history_acc as they come.
#
# Walltime: ~2 h per run on one Delta core (55.6k increments), so the campaign
# needs ~195 h of wall time on 12 lanes (~2,350 core-hours). The job queues
# its own follow-up (afterany) at the start, so a walltime kill continues
# automatically; runs marked ok are never redone. Chaining stops when all runs
# are done, after LOCAL_EDITS_CHAIN_LEFT follow-ups (default 6), or when a
# whole job made no progress.
#
# Submit (the Code directory is found from this script's own path, so the
# current directory does not matter; the Slurm log goes to the current directory):
#     cd <Code dir> && sbatch abaqus_scripts/delta/hood_local_edits.sbatch
#
# Monitor:  squeue -u hkazemi ; tail -f hood_local_edits_<jobid>.out
#           cat Data/HoodImpact_LocalEdits/job_logs/status.txt
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

SELF="abaqus_scripts/delta/hood_local_edits.sbatch"
# Slurm runs a spooled copy of this script, so read its submitted path from Slurm.
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
case "$SCRIPT_PATH" in /*) ;; ?*) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;; esac
case "$SCRIPT_PATH" in
    */"$SELF") ROOT="${SCRIPT_PATH%/"$SELF"}" ;;
    *) ROOT="$SLURM_SUBMIT_DIR" ;;
esac
echo "=== node: $(hostname), $(date), job $SLURM_JOB_ID ==="
echo "script: ${SCRIPT_PATH:-?}  ->  Code directory: $ROOT  (submitted from $SLURM_SUBMIT_DIR)"
if [ ! -d "$ROOT/abaqus_scripts" ] || [ ! -f "$ROOT/$SELF" ]; then
    echo "ERROR: cannot find the Code directory (abaqus_scripts/ next to Data/). Submit as:"
    echo "    cd <Code dir> && sbatch abaqus_scripts/delta/hood_local_edits.sbatch"
    exit 1
fi
DATA="$ROOT/Data/HoodImpact_LocalEdits"
MANIFEST="$DATA/manifest_local_edits.csv"
LOGDIR="$DATA/job_logs"
mkdir -p "$LOGDIR"
status() { echo "$(date '+%F %T') job $SLURM_JOB_ID: $*" | tee -a "$LOGDIR/status.txt"; }

LANES="${LOCAL_EDITS_LANES:-12}"
CHAIN_LEFT="${LOCAL_EDITS_CHAIN_LEFT:-6}"
PREV_OK="${LOCAL_EDITS_PREV_OK:--1}"
umask 002
WORK="${LOCAL_EDITS_WORK:-/u/hkazemi/hood_local_edits_scratch}"
SCRATCH="$WORK/abaqus_scratch"
METRICS="$WORK/generation_metrics.csv"
OUTFILE="$SLURM_SUBMIT_DIR/hood_local_edits_${SLURM_JOB_ID}.out"
mkdir -p "$WORK" "$SCRATCH"

sync_logs() {
    cp -f "$OUTFILE" "$LOGDIR/" 2>/dev/null
    [ -f "$METRICS" ] && cp -f "$METRICS" "$DATA/generation_metrics.csv" 2>/dev/null
    return 0
}
SYNC_PID=""
trap 'kill $SYNC_PID 2>/dev/null; sync_logs' EXIT
( while sleep 600; do sync_logs; done ) &
SYNC_PID=$!
status "started on $(hostname); Code directory $ROOT; log $OUTFILE"

if [ ! -f "$MANIFEST" ] || [ ! -f "$DATA/local_edit_doe.csv" ]; then
    status "ERROR: missing $MANIFEST or local_edit_doe.csv -- unpack local_edits_delta.tar.gz in $ROOT"
    exit 1
fi
if [ ! -f "$ROOT/Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_31.inp" ]; then
    status "ERROR: missing the base deck Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_31.inp"
    exit 1
fi
if [ -f "$DATA/pilot_gate_failed" ] && [ "${SKIP_PILOT_GATE:-0}" != "1" ]; then
    status "STOP: the pilot gate failed earlier (see pilot_report.txt). Delete pilot_gate_failed to retry,"
    status "      or resubmit with --export=ALL,SKIP_PILOT_GATE=1 to ignore it."
    exit 1
fi

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact || { status "ERROR: conda activate hood_impact failed"; exit 1; }

cd "$ROOT"
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
status "runs: $N_TOTAL in the plan, $OK_START already ok; lanes $LANES; follow-ups left $CHAIN_LEFT"
if [ "$OK_START" -ge "$N_TOTAL" ]; then
    status "all runs are ok; nothing left to solve"
    exit 0
fi
if [ "$OK_START" -eq "$PREV_OK" ]; then
    status "STOP: the previous job finished no run ($OK_START ok before and after). Check its log, then resubmit by hand."
    exit 1
fi

# queue the follow-up now: it starts when this job ends, even after a walltime kill
NEXT=""
if [ "$CHAIN_LEFT" -gt 0 ]; then
    if NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
            --export=ALL,LOCAL_EDITS_CHAIN_LEFT=$((CHAIN_LEFT - 1)),LOCAL_EDITS_PREV_OK="$OK_START" \
            "$ROOT/$SELF" 2>&1); then
        status "queued follow-up job $NEXT"
    else
        status "WARNING: could not queue a follow-up ($NEXT); resubmit by hand if this job hits its walltime"
        NEXT=""
    fi
fi
stop_chain() { [ -n "$NEXT" ] && scancel "$NEXT" && status "cancelled follow-up job $NEXT"; return 0; }

echo "=== storage ==="
quota -s 2>/dev/null | head -n 8 || true
df -h "$ROOT" "$WORK" 2>/dev/null || true
abaqus licensing ru 2>&1 | head -n 6 || true

status "writing decks"
if ! python -u abaqus_scripts/make_local_edit_decks.py --skip-existing --verify 50; then
    status "ERROR: deck generation failed (see the log)"; stop_chain; exit 1
fi
N_DECKS=$(ls -1 "$DATA/inp_files"/HoodImpact_*.inp 2>/dev/null | wc -l)
if [ "$N_DECKS" -ne "$N_TOTAL" ]; then
    status "ERROR: $N_DECKS decks for $N_TOTAL runs"; stop_chain; exit 1
fi
status "decks ready: $N_DECKS"

solve() {
    python -u abaqus_scripts/batch_generate.py \
        --inp-dir "$DATA/inp_files" --work-dir "$WORK" --scratch-dir "$SCRATCH" \
        --out-dir "$DATA" --runs "$1" --lanes "$LANES"
    sync_logs
}

if [ ! -f "$DATA/pilot_gate_passed" ]; then
    status "pilot: solving runs $PILOT_RUNS"
    solve "$PILOT_RUNS"
    if python -u abaqus_scripts/check_local_edits.py --pilot --metrics "$METRICS" --out "$DATA/pilot_report.txt"; then
        touch "$DATA/pilot_gate_passed"
        status "pilot gate PASSED (pilot_report.txt)"
    elif [ "${SKIP_PILOT_GATE:-0}" = "1" ]; then
        status "WARNING: pilot gate failed, continuing because SKIP_PILOT_GATE=1"
    else
        touch "$DATA/pilot_gate_failed"
        status "STOP: pilot gate FAILED -- see pilot_report.txt; nothing further is solved"
        stop_chain
        exit 1
    fi
fi

status "campaign: solving runs 1-$N_TOTAL ($(count_ok) ok so far)"
solve "1-$N_TOTAL"
python -u abaqus_scripts/check_local_edits.py --metrics "$METRICS" --out "$DATA/qa_report.txt" || true

OK_END=$(count_ok)
status "finished this job: $OK_END of $N_TOTAL runs ok"
if [ "$OK_END" -ge "$N_TOTAL" ]; then
    stop_chain
    status "ALL DONE. Bring back Data/HoodImpact_LocalEdits (output_history_acc, generation_metrics.csv, *_report.txt)"
fi
