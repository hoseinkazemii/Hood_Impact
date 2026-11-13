#!/bin/bash
# Local-edit dataset: 72 beam edits of hood design 6 x 20 impact locations each
# + design 6 itself at all 142 locations = 1582 Abaqus/Explicit solves, every
# one with a fixed 4.5e-7 s increment and the headform starting 1.0 mm clear
# of the hood. Plan: Data/HoodImpact_LocalEdits/manifest_local_edits.csv
# (abaqus_scripts/local_edit_doe.py); details in delta/README_local_edits.md.
#
# What one submission does:
#   1. writes the decks that are missing (make_local_edit_decks.py; all 1582
#      take ~15 min and 7.5 GB, designs added to the plan later only theirs)
#   2. solves the 11 pilot runs (one round of 12 lanes, ~2 h)
#   3. pilot gate (check_local_edits.py --pilot): every pilot run finished,
#      all took the same number of increments, the far control matches design 6
#      and the near difference stands clearly above it. FAIL stops everything
#      (and cancels the queued follow-up job); read pilot_report.txt.
#   4. solves the remaining runs in the manifest's solve_order (design 6 first,
#      then the cut-line designs, then the single-beam notches), 12 single-core
#      lanes (5 FlexNet tokens each) or the layout in campaign_layout.env (below)
#   5. writes qa_report.txt over everything finished so far
#
# Layout (lanes x cores per solve). Without Data/HoodImpact_LocalEdits/
# campaign_layout.env: 12 lanes x 1 core, 60 of the 65 tokens. A solve on N
# cores takes floor(5 * N**0.422) tokens (8 -> 12, 16 -> 16), so multi-core
# lanes use far more cores for the same 65 tokens. hood_cpus_test.sbatch
# measures the fastest layout and writes cpus_test/layout.env; copy it to
# campaign_layout.env and submit with the cores and memory it names, e.g.
#     sbatch -c 64 --mem=64g --array=1-8%1 abaqus_scripts/delta/hood_local_edits.sbatch
# Its lines (LOCAL_EDITS_CPUS, _LANES, _MP_MODE, _RUN_S, _REDO_OTHER_CPUS) can
# also be given with --export, which wins over the file; lanes are cut to what
# the job's cores allow. Chained follow-ups inherit the job's cores and memory.
#
# Tracking: Data/HoodImpact_LocalEdits/job_logs/status.txt gets one line per
# stage of every job; the Slurm log and generation_metrics.csv are copied into
# Data/HoodImpact_LocalEdits every 10 minutes and when the job ends; finished
# results appear in Data/HoodImpact_LocalEdits/output_history_acc as they come.
#
# Walltime: ~1.75 h per run on one Delta core (55.6k increments), ~6.6 runs/h
# on 12 lanes, so the campaign needs ~240 h of wall time (~2,900 core-hours),
# done as 48-h jobs (Delta's cpu maximum; ~300 runs each, so ~5-6 jobs). The
# queue wait was about the same for 12-h and 48-h requests, so fewer, longer
# jobs finish sooner. 12 cores = the default 12 single-core lanes (see Layout).
# A lane only starts a run that is expected to finish before the job's end
# (batch_generate.py --deadline), so no solve is killed half-way; the job
# queues its own follow-up (afterany) at the start, with the same time limit,
# and the follow-up solves what is left; runs marked ok are never redone.
# Chaining stops when all runs are done, after LOCAL_EDITS_CHAIN_LEFT
# follow-ups (default 30), or when a whole job made no progress.
#
# Submit (the Code directory is found from this script's own path, so the
# current directory does not matter; the Slurm log goes to the current directory):
#     cd <Code dir> && sbatch --array=1-8%1 abaqus_scripts/delta/hood_local_edits.sbatch
# Progress (which runs are done and will not be solved again):
#     python abaqus_scripts/local_edits_progress.py
# As a job array throttled to one task at a time, every task is eligible from the
# start and gains queue age while the one before it runs, so the next one starts
# sooner; a follow-up queued with a dependency only starts ageing when the job
# before it ends. Without --array the job chains itself as described above.
# The task that finishes the last run (or one that makes no progress) cancels
# the array's pending tasks. Only one of these jobs may run at a time (license):
# a job that finds another one running stops.
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
#SBATCH --mem=24g
#SBATCH -t 48:00:00
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
# stop what would run after this job: the queued follow-up, or the array's pending tasks
NEXT=""
stop_chain() {
    if [ -n "$ARRAY_ID" ]; then
        scancel --state=PENDING "$ARRAY_ID" && status "cancelled the pending tasks of job array $ARRAY_ID"
    elif [ -n "$NEXT" ]; then
        scancel "$NEXT" && status "cancelled follow-up job $NEXT"
    fi
    return 0
}

# layout: --export variables > campaign_layout.env > 12 single-core lanes
LAYOUT_FILE="$DATA/campaign_layout.env"
if [ -f "$LAYOUT_FILE" ]; then
    while IFS='=' read -r KEY VAL; do
        case "$KEY" in
            LOCAL_EDITS_CPUS|LOCAL_EDITS_LANES|LOCAL_EDITS_MP_MODE|LOCAL_EDITS_RUN_S|LOCAL_EDITS_REDO_OTHER_CPUS)
                VAL="${VAL%%#*}"; VAL="${VAL//[[:space:]]/}"
                [ -z "${!KEY+x}" ] && printf -v "$KEY" '%s' "$VAL" ;;
        esac
    done < "$LAYOUT_FILE"
fi
CPUS="${LOCAL_EDITS_CPUS:-1}"
LANES="${LOCAL_EDITS_LANES:-12}"
MP_MODE="${LOCAL_EDITS_MP_MODE:-}"
RUN_S="${LOCAL_EDITS_RUN_S:-}"
REDO="${LOCAL_EDITS_REDO_OTHER_CPUS:-0}"
ALLOC="${SLURM_CPUS_PER_TASK:-12}"
CHAIN_LEFT="${LOCAL_EDITS_CHAIN_LEFT:-30}"
PREV_OK="${LOCAL_EDITS_PREV_OK:--1}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-}"     # set when submitted as a job array (--array=1-8%1)
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
    stop_chain
    exit 1
fi
if [ ! -f "$ROOT/Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_31.inp" ]; then
    status "ERROR: missing the base deck Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_31.inp"
    stop_chain
    exit 1
fi
if [ -f "$DATA/pilot_gate_failed" ] && [ "${SKIP_PILOT_GATE:-0}" != "1" ]; then
    status "STOP: the pilot gate failed earlier (see pilot_report.txt). Delete pilot_gate_failed to retry,"
    status "      or resubmit with --export=ALL,SKIP_PILOT_GATE=1 to ignore it."
    stop_chain
    exit 1
fi

export ABAQUSLM_LICENSE_FILE=27000@130.126.241.64
export PATH=/scratch/bbqg/hkazemi/software/DassaultSystemes/SIMULIA/Commands:$PATH
source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact || { status "ERROR: conda activate hood_impact failed"; stop_chain; exit 1; }

cd "$ROOT"
N_TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
PILOT_RUNS=$(python -c "import csv; print(','.join(r['run'] for r in csv.DictReader(open('$MANIFEST')) if r['pilot'] == '1'))")
# the campaign's solve order (manifest column solve_order; run order if absent)
ORDER="$WORK/solve_order.txt"
python - "$MANIFEST" "$ORDER" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
key = (lambda r: int(r["solve_order"])) if "solve_order" in rows[0] else (lambda r: int(r["run"]))
open(sys.argv[2], "w").write("\n".join(r["run"] for r in sorted(rows, key=key)) + "\n")
EOF
[ -s "$ORDER" ] || { status "ERROR: could not write the solve order $ORDER"; stop_chain; exit 1; }
# ok runs; with REDO=1 only those solved on $CPUS cores count
count_ok() {
    python - "$METRICS" "$CPUS" "$REDO" <<'EOF'
import csv, os, sys
p, cpus, redo = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
def same(r):
    try:
        return int(float(r.get("cpus") or 1)) == cpus
    except ValueError:
        return False
print(len({r["run"] for r in csv.DictReader(open(p)) if r["status"] == "ok" and (not redo or same(r))})
      if os.path.exists(p) else 0)
EOF
}
if [ "$CPUS" -gt "$ALLOC" ]; then
    status "ERROR: the layout needs $CPUS cores per solve but this job has $ALLOC; submit with -c $((CPUS * LANES)) (see $LAYOUT_FILE)"
    stop_chain
    exit 1
fi
if [ $((LANES * CPUS)) -gt "$ALLOC" ]; then
    status "WARNING: $LANES lanes x $CPUS cores do not fit the job's $ALLOC cores; using $((ALLOC / CPUS)) lanes"
    LANES=$((ALLOC / CPUS))
fi
LAYOUT="lanes $LANES x cpus $CPUS${MP_MODE:+ ($MP_MODE)}, ~$((LANES * $(python -c "print(int(5 * $CPUS ** 0.422 + 1e-9))"))) tokens"
[ "$REDO" = "1" ] && LAYOUT="$LAYOUT; re-solving runs finished with another cpus count"
OK_START=$(count_ok)
if [ -n "$ARRAY_ID" ]; then
    status "runs: $N_TOTAL in the plan, $OK_START already ok; $LAYOUT; task ${SLURM_ARRAY_TASK_ID:-?} of job array $ARRAY_ID"
else
    status "runs: $N_TOTAL in the plan, $OK_START already ok; $LAYOUT; follow-ups left $CHAIN_LEFT"
fi
if [ "$OK_START" -ge "$N_TOTAL" ]; then
    status "all runs are ok; nothing left to solve"
    stop_chain
    exit 0
fi
if [ "$OK_START" -eq "$PREV_OK" ]; then
    status "STOP: the previous job finished no run ($OK_START ok before and after). Check its log, then resubmit by hand."
    exit 1
fi
# two of these jobs at once would need 2 x 60 of the 65 license tokens
OTHERS=$(squeue -h -u "$USER" -n hood_local_edits -t RUNNING -o %A 2>/dev/null | grep -vx "$SLURM_JOB_ID" | tr '\n' ' ')
if [ -n "${OTHERS// /}" ]; then
    status "STOP: another hood_local_edits job is running ($OTHERS); only one may run at a time (license)."
    stop_chain
    exit 1
fi
# the cpus test (hood_cpus_test.sbatch) needs every token: wait for it (max ~140 min)
for WAIT_MIN in $(seq 1 140); do
    TEST=$(squeue -h -u "$USER" -n hood_cpus_test -t RUNNING -o %A 2>/dev/null | tr '\n' ' ')
    [ -z "${TEST// /}" ] && break
    [ "$WAIT_MIN" = 1 ] && status "waiting for the cpus test job $TEST to end (it needs the license to itself)"
    sleep 60
done

# this job's end (lanes start no run that would not finish by then) and time limit
DEADLINE="${SLURM_JOB_END_TIME:-}"
if [ -z "$DEADLINE" ]; then
    END_STR=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
    case "$END_STR" in 20[0-9][0-9]-*) DEADLINE=$(date -d "$END_STR" +%s 2>/dev/null) || DEADLINE="" ;; esac
fi
case "$DEADLINE" in *[!0-9]*|"") DEADLINE="" ;; esac
[ -n "$DEADLINE" ] && [ "$DEADLINE" -le "$(date +%s)" ] && DEADLINE=""
TIME_LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null)
case "$TIME_LIMIT" in *[!0-9:-]*|"") TIME_LIMIT="" ;; esac
if [ -n "$DEADLINE" ]; then
    status "time limit ${TIME_LIMIT:-?}; no run is started that would end after $(date -d "@$DEADLINE" '+%F %T')"
else
    status "WARNING: could not read this job's end time; runs in progress at the walltime are lost"
fi

# chain mode: queue the follow-up now; it starts when this job ends, even after a
# walltime kill. A job array needs none: its next task is already waiting (and ageing).
if [ -z "$ARRAY_ID" ] && [ "$CHAIN_LEFT" -gt 0 ]; then
    if NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" ${TIME_LIMIT:+--time="$TIME_LIMIT"} \
            --cpus-per-task="$ALLOC" ${SLURM_MEM_PER_NODE:+--mem="${SLURM_MEM_PER_NODE}M"} \
            --export=ALL,LOCAL_EDITS_CHAIN_LEFT=$((CHAIN_LEFT - 1)),LOCAL_EDITS_PREV_OK="$OK_START" \
            "$ROOT/$SELF" 2>&1); then
        status "queued follow-up job $NEXT"
    else
        status "WARNING: could not queue a follow-up ($NEXT); resubmit by hand if this job hits its walltime"
        NEXT=""
    fi
fi

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

solve() {   # solve --runs <list> | --runs-file <file>
    python -u abaqus_scripts/batch_generate.py \
        --inp-dir "$DATA/inp_files" --work-dir "$WORK" --scratch-dir "$SCRATCH" \
        --out-dir "$DATA" --lanes "$LANES" --cpus "$CPUS" ${MP_MODE:+--mp-mode "$MP_MODE"} \
        ${RUN_S:+--default-run-s "$RUN_S"} $([ "$REDO" = "1" ] && echo --redo-other-cpus) \
        ${DEADLINE:+--deadline "$DEADLINE"} "$@"
    sync_logs
}

if [ ! -f "$DATA/pilot_gate_passed" ]; then
    status "pilot: solving runs $PILOT_RUNS"
    solve --runs "$PILOT_RUNS"
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

NEXT_UP=$(python - "$ORDER" "$MANIFEST" "$METRICS" <<'EOF'
import csv, os, sys
done = {r["run"] for r in csv.DictReader(open(sys.argv[3])) if r["status"] == "ok"} if os.path.exists(sys.argv[3]) else set()
design = {r["run"]: r["design"] for r in csv.DictReader(open(sys.argv[2]))}
nxt = []
for run in open(sys.argv[1]).read().split():
    if run not in done and design[run] not in nxt:
        nxt.append(design[run])
print(" ".join(nxt[:6]))
EOF
)
status "campaign: solving the $N_TOTAL runs in solve order ($(count_ok) ok so far; next designs: $NEXT_UP)"
solve --runs-file "$ORDER"
python -u abaqus_scripts/check_local_edits.py --metrics "$METRICS" --out "$DATA/qa_report.txt" || true

OK_END=$(count_ok)
status "finished this job: $OK_END of $N_TOTAL runs ok"
if [ -n "$ARRAY_ID" ] && [ "$OK_END" -lt "$N_TOTAL" ] && [ "$OK_END" -le "$OK_START" ]; then
    status "STOP: this job finished no run ($OK_START ok before and after). Check this log, then resubmit by hand."
    stop_chain
fi
if [ "$OK_END" -ge "$N_TOTAL" ]; then
    stop_chain
    status "ALL DONE. Bring back Data/HoodImpact_LocalEdits (output_history_acc, generation_metrics.csv, *_report.txt)"
fi
