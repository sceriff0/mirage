#!/bin/bash
#SBATCH --job-name=regbench-rung
# One unit of one run at one rung of the memory ladder: a case for a method that runs
# per pair, a patient for one that runs per patient. SLURM_ARRAY_TASK_ID is the unit
# (`run --index`). Submitted by submit_cohort.sh, which gives the allocation and these
# arguments (one line of `python -m benchmarks.regbench runs`, then the rung):
#
#   run_rung.sbatch <name> <method> <label|-> <opts|-> <image> <gpu 0|1> <dataset>
#                   <attempt> <memory GB> <realcheck 0|1> <previous rung's job id|->
#
# A task of a rung above the first exits at once unless its unit's last attempt ran out
# of memory (`run --retry-oom`; benchmarks/regbench/ladder.py has the rule). A failed
# registration is recorded in run.json and scored at the unregistered pose, so the task
# exits 0 either way.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" gpu="$6" dataset="$7"
attempt="$8" mem_gb="$9" realcheck="${10}" previous="${11:--}"
export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb" REGBENCH_GPU="$gpu"

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
if [ "$attempt" -gt 1 ]; then
    args+=(--retry-oom)
    # what the scheduler says ended this unit at the rung below (ladder.py: it decides
    # only where the driver's own record still says "killed")
    if [ "$previous" != - ] && command -v sacct >/dev/null; then
        state="$(sacct -n -X -P -j "${previous}_${index}" -o State 2>/dev/null | head -n 1)"
        [ -n "$state" ] && args+=(--previous-state "$state")
    fi
fi
regbench_py "$image" "${args[@]}"
echo "$name $dataset unit $index rung $attempt (${mem_gb} GB): python exited $?"
rm -rf "$work"
exit 0
