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
#                   <stages: -|jobs|local>
#
# `jobs`: the run's registration is several scheduler jobs per round (stage.sbatch) and
# this task is the round's last one. It runs nothing of the registration: it reads the
# stage jobs' records, moves the benchmark's points through the transform and runs the
# method's writer (STITCH), with the pipeline's allocation for that process.
# `local` (REGBENCH_STARE_EXECUTION=local, the CPU cross): such a run as one job per
# unit, this task starting the stages itself: `run --stages local`, the tiles as many at
# a time as its option workers says (submit_cohort.sh passes workers=REGBENCH_CPUS).
# `-`: the run is one job per unit. A run whose registration is stage jobs and that is
# handed `-` is refused by `run` before it touches a record: its end line says
# cause=not_submitted whose=launcher, and nothing is registered here in their place.
#
# Each rung is two arrays (benchmarks/regbench/ladder.py has the rules):
#   first / climb   the attempt at this rung. `climb` (every rung above the first) exits
#                   at once unless the unit's last attempt ended for memory.
#   again           the SAME rung once more, and only when the attempt before was lost
#                   to the scheduler (node failure, preemption, a cancelled job, a task
#                   that never started). One retry per rung.
# A record that still says "killed" is settled by the scheduler's accounting of the task
# that wrote it. When the accounting has no END state for that task yet, this task asks
# again every REGBENCH_ACCT_POLL_S (30) seconds for at most REGBENCH_ACCT_WAIT_S (600);
# without one after that, the climb task runs the unit again at its own rung.
# A unit whose results are all ok already is not run again (REGBENCH_FORCE=1 does). A
# failed registration is recorded in run.json and scored at the unregistered pose, so
# the task exits 0 either way.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" gpu="$6" dataset="$7"
attempt="$8" mem_gb="$9" realcheck="${10}" previous="${11:--}" task="${12:-first}"
limit="${13:-}" stages="${14:--}"
# stated here and nowhere else: a run that is one job per unit states nothing, and a
# `local` one states it on its own command line (`run --stages local`)
unset REGBENCH_STAGES
stated=()
case "$stages" in
    -) ;;
    local) stated=(--stages local) ;;
    *) export REGBENCH_STAGES="$stages" ;;
esac
retry=0
[ "$task" = again ] && retry=1
export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb" REGBENCH_GPU="$gpu"
export REGBENCH_RETRY="$retry" REGBENCH_TIME_LIMIT="$limit"

# The selfcheck is REPORTED, never a gate: this task does not wait for it and does not
# read its verdict (the scorer reads it from logs/verdicts.tsv).

index="${SLURM_ARRAY_TASK_ID:-0}"
regbench_start run "run=$name" "label=$label" "dataset=$dataset" "unit=index_$index" \
    "rung_gb=$mem_gb" "attempt=$attempt" "task=$task"
work="${REGBENCH_WORK:-${TMPDIR:-/tmp}}/regbench-$name-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}-$index"
args=(run --cases "$(regbench_cases_of "$dataset")" --out "$REGBENCH_OUT" --dataset "$dataset"
      --method "$method" --index "$index" --work "$work")
[ "$label" != - ] && args+=(--label "$label")
if [ "$opts" != - ]; then
    IFS=, read -ra items <<< "$opts"
    for o in "${items[@]}"; do args+=(--opt "$o"); done
fi
[ "$realcheck" = 1 ] && args+=(--realcheck)
args+=(${stated[@]+"${stated[@]}"})
[ "${REGBENCH_FORCE:-0}" = 1 ] && args+=(--force)
# How long a task waits for the scheduler's accounting to name the END state of an
# earlier task whose record still says "killed", and how often it asks.
: "${REGBENCH_ACCT_WAIT_S:=600}"
: "${REGBENCH_ACCT_POLL_S:=30}"
case "$REGBENCH_ACCT_WAIT_S" in ''|*[!0-9]*) REGBENCH_ACCT_WAIT_S=600 ;; esac
case "$REGBENCH_ACCT_POLL_S" in ''|*[!0-9]*|0) REGBENCH_ACCT_POLL_S=30 ;; esac
acct_dir="$REGBENCH_OUT/logs/accounting"
acct="$acct_dir/$name-$dataset-$index-$attempt-$retry.psv"
wanted="$acct.wanted"
# look <job ids, comma-separated>: what the accounting says of them now, into $acct:
# each task's own line (its state and elapsed time) and its steps' lines, which carry
# the CPU time (asked for the allocations alone, sacct reports that as zero)
look() {
    sacct -n -P -j "$1" -o JobID,State,ElapsedRaw,TotalCPU > "$acct.new" 2>/dev/null \
        && [ -s "$acct.new" ] && mv "$acct.new" "$acct"
}
# all_ended: has every task named in $wanted a line in $acct whose state is an end state
# (benchmarks/regbench/ladder.py: STATES_NOT_ENDED is this list)
all_ended() {
    [ -s "$acct" ] || return 1
    awk -F'|' '
        NR == FNR { want[$1] = 1; next }
        ($1 in want) && $2 != "" && $2 !~ /^(PENDING|RUNNING|RESIZING|SUSPENDED|COMPLETING)/ { got[$1] = 1 }
        END { for (id in want) if (!(id in got)) exit 1 }' "$wanted" "$acct"
}
acct_args=()
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
        mkdir -p "$acct_dir"
        rm -f "$acct" "$acct.new" "$wanted"
        look "$ids" && acct_args+=(--accounting "$acct")
        # a record that still says "killed" and has no end state there: Python names the
        # tasks in $wanted and exits 75 without deciding (ladder.EXIT_ACCOUNTING_PENDING)
        acct_args+=(--accounting-wanted "$wanted")
    else
        # no accounting to ask: nothing to wait for
        acct_args+=(--accounting-waited)
    fi
fi
regbench_py "$image" "${args[@]}" ${acct_args[@]+"${acct_args[@]}"}
code=$?
if [ "$code" = 75 ] && [ -s "$wanted" ]; then
    # One look is not the answer: the accounting can lag behind a task that has just
    # ended. Ask until it names an end state for every task Python wants, or the wait
    # is over; then Python decides with what there is.
    want_ids="$(grep -E '^[0-9]+(_[0-9]+)?$' "$wanted" | paste -sd, -)"
    waited=0
    while [ -n "$want_ids" ] && ! all_ended && [ "$waited" -lt "$REGBENCH_ACCT_WAIT_S" ]; do
        sleep "$REGBENCH_ACCT_POLL_S"
        waited=$((waited + REGBENCH_ACCT_POLL_S))
        look "$want_ids" || echo "accounting: sacct gave nothing for $want_ids after ${waited}s"
    done
    if all_ended; then
        echo "accounting: an end state for $want_ids after ${waited}s"
    else
        echo "accounting: NO end state for every task of $(paste -sd, "$wanted") after ${waited}s of at most ${REGBENCH_ACCT_WAIT_S}s (REGBENCH_ACCT_WAIT_S); deciding without it"
    fi
    acct_args=(--accounting-waited)
    [ -s "$acct" ] && acct_args+=(--accounting "$acct")
    regbench_py "$image" "${args[@]}" "${acct_args[@]}"
    code=$?
fi
echo "$name $dataset unit $index rung $attempt $task (${mem_gb} GB): python exited $code"
if [ "$code" = 137 ]; then
    # Python ended by signal 9 and this script goes on: the kernel's out-of-memory
    # killer took it (the scheduler's own kill ends the script too). Noted for the next
    # task of the ladder and for the settle job, whatever the accounting says later.
    # run.json: launch_id (resources.host_info)
    launch="${SLURM_JOB_ID:-}"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        launch="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    if [ -n "$launch" ]; then
        mkdir -p "$acct_dir/exits" && echo "$code" > "$acct_dir/exits/$launch"
    fi
    # and on the end line, which Python never wrote: the cause that note settles to
    # (ladder.settle: oom), and that only the next rung can run the unit
    # (failure_log.chain_wants): the watcher submits no same-rung retry for it
    if [ -n "${REGBENCH_END_FILE:-}" ] && [ ! -s "$REGBENCH_END_FILE" ]; then
        echo "REGBENCH_END kind=run run=$name dataset=$dataset unit=index_$index rung=$attempt status=failed cause=oom whose=method exit=$code wants=climb log=${REGBENCH_LOG:--}" \
            > "$REGBENCH_END_FILE"
    fi
fi
rm -rf "$work"
regbench_end run "$code" "run=$name" "dataset=$dataset" "unit=index_$index" "rung=$attempt"
exit 0
