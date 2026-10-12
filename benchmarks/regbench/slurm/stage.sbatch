#!/bin/bash
#SBATCH --job-name=regbench-stage
# One stage job of one round, for a run whose registration is several scheduler jobs per
# round, as the pipeline runs it (`runs`, column `stages`; benchmarks/regbench/stages.py):
# prep (the preprocessing row only), coarse, tile or solve. Submitted by
# submit_cohort.sh, which gives the allocation (the pipeline's, for that process and
# attempt) and these arguments:
#
#   stage.sbatch <name> <method> <label|-> <opts|-> <image> <dataset> <stage>
#                <attempt 1-4> <memory GB> <time limit> <layout file|-> <offset|->
#
# SLURM_ARRAY_TASK_ID is the round (`stage --index`), or, with a layout and an offset
# (the first attempt of the tiles), offset + task id is one tile counted over every
# round's tiles laid end to end (`stage --flat`).
#
# Attempt 1 runs the job. Attempts 2 to 4 exit at once unless the attempt before ended
# the way the pipeline retries (conf/base.config: exit 104, 134, 135, 137, 139, 140 or
# 143, or killed without a record), and then run it with the attempt's memory and time.
# A tile task of attempt 2 to 4 is one job per round: it runs the round's tiles that
# are due, one after another. A job whose stage before it is not ok does nothing. The
# stage's record says what became of it; the task exits 0 either way.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" dataset="$6" stage="$7"
attempt="$8" mem_gb="$9" limit="${10:-}" layout="${11:--}" offset="${12:--}"
export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb" REGBENCH_GPU=0
export REGBENCH_TIME_LIMIT="$limit"

task="${SLURM_ARRAY_TASK_ID:-0}"
# the second lock behind the dependency on the gate job, as in run_rung.sbatch
gate="$REGBENCH_OUT/logs/gates/$name.tsv"
status=not_checked blocked=1
if [ -s "$gate" ]; then
    IFS=$'\t' read -r _ status _ blocked _ < "$gate"
fi
if [ "$blocked" != 0 ]; then
    echo "$name: the gate says $status and the run is not allowed to start unchecked; nothing run"
    echo "REGBENCH_END kind=stage run=$name dataset=$dataset unit=task_$task stage=$stage attempt=$attempt status=blocked cause=gate_blocked whose=launcher log=${REGBENCH_LOG:--}"
    exit 0
fi

args=(stage --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --dataset "$dataset"
      --method "$method" --stage "$stage" --attempt "$attempt")
[ "$label" != - ] && args+=(--label "$label")
if [ "$opts" != - ]; then
    IFS=, read -ra items <<< "$opts"
    for o in "${items[@]}"; do args+=(--opt "$o"); done
fi
if [ "$offset" != - ]; then
    args+=(--flat "$((offset + task))" --layout "$layout")
else
    args+=(--index "$task")
fi
regbench_py "$image" "${args[@]}"
code=$?
echo "$name $dataset $stage task $task attempt $attempt (${mem_gb} GB): python exited $code"
regbench_end stage "$code" "run=$name" "dataset=$dataset" "unit=task_$task" "stage=$stage" "attempt=$attempt"
exit 0
