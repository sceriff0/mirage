#!/bin/bash
#SBATCH --job-name=regbench-rung
# One unit of one run at one rung of the memory ladder: a case for a method that runs
# per pair, a patient for one that runs per patient. SLURM_ARRAY_TASK_ID is the unit
# (`run --index`). Submitted by submit_cohort.sh, which gives the allocation and these
# arguments (one line of `python -m benchmarks.regbench runs`, then the rung):
#
#   run_rung.sbatch <name> <method> <label|-> <opts|-> <image> <gpu 0|1> <dataset>
#                   <attempt> <memory GB> <realcheck 0|1>
#                   <this ladder's earlier job ids, comma-separated|->
#                   <task: first|climb|again> <time limit of this task>
#
# Each rung is two arrays (benchmarks/regbench/ladder.py has the rules):
#   first / climb   the attempt at this rung. `climb` (every rung above the first) exits
#                   at once unless the unit's last attempt ended for memory.
#   again           the SAME rung once more, and only when the attempt before was lost
#                   to the scheduler (node failure, preemption, a cancelled job, a task
#                   that never started). One retry per rung.
# A unit whose results are all ok already is not run again (REGBENCH_FORCE=1 does). A
# failed registration is recorded in run.json and scored at the unregistered pose, so
# the task exits 0 either way.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" gpu="$6" dataset="$7"
attempt="$8" mem_gb="$9" realcheck="${10}" previous="${11:--}" task="${12:-first}"
limit="${13:-}"
retry=0
[ "$task" = again ] && retry=1
export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb" REGBENCH_GPU="$gpu"
export REGBENCH_RETRY="$retry" REGBENCH_TIME_LIMIT="$limit"

# The gate's verdict for this run, as the gate job left it. The dependency on that job
# already keeps a blocked run from starting; this is the second lock, for a scheduler
# that lets a dependent job through.
gate="$REGBENCH_OUT/logs/gates/$name.tsv"
status=not_checked blocked=1
if [ -s "$gate" ]; then
    IFS=$'\t' read -r _ status _ blocked _ < "$gate"
fi
if [ "$blocked" != 0 ]; then
    echo "$name: the gate says $status and the run is not allowed to start unchecked; nothing run"
    echo "REGBENCH_END kind=run run=$name dataset=$dataset unit=index_${SLURM_ARRAY_TASK_ID:-0} rung=$attempt status=blocked cause=gate_blocked whose=launcher log=${REGBENCH_LOG:--}"
    exit 0
fi
export REGBENCH_SELFCHECK_STATUS="$status"

index="${SLURM_ARRAY_TASK_ID:-0}"
work="${REGBENCH_WORK:-${TMPDIR:-/tmp}}/regbench-$name-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}-$index"
args=(run --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --dataset "$dataset"
      --method "$method" --index "$index" --work "$work")
[ "$label" != - ] && args+=(--label "$label")
if [ "$opts" != - ]; then
    IFS=, read -ra items <<< "$opts"
    for o in "${items[@]}"; do args+=(--opt "$o"); done
fi
[ "$realcheck" = 1 ] && args+=(--realcheck)
[ "${REGBENCH_FORCE:-0}" = 1 ] && args+=(--force)
if [ "$task" != first ]; then
    args+=(--ladder "$task")
    # What the scheduler says of this unit's earlier tasks: state, elapsed and CPU time,
    # one line per task. ladder.py matches a record that still says "killed" to the task
    # that wrote it (run.json: launch_id is <array job>_<index>), so it learns why the
    # task ended and what it cost.
    if [ "$previous" != - ] && command -v sacct >/dev/null; then
        ids=""
        IFS=, read -ra earlier <<< "$previous"
        for id in "${earlier[@]}"; do ids="${ids:+$ids,}${id}_${index}"; done
        mkdir -p "$REGBENCH_OUT/logs/accounting"
        acct="$REGBENCH_OUT/logs/accounting/$name-$dataset-$index-$attempt-$retry.psv"
        if sacct -n -X -P -j "$ids" -o JobID,State,ElapsedRaw,TotalCPU > "$acct" 2>/dev/null \
            && [ -s "$acct" ]; then
            args+=(--accounting "$acct")
        fi
    fi
fi
regbench_py "$image" "${args[@]}"
code=$?
echo "$name $dataset unit $index rung $attempt $task (${mem_gb} GB): python exited $code"
rm -rf "$work"
regbench_end run "$code" "run=$name" "dataset=$dataset" "unit=index_$index" "rung=$attempt"
exit 0
