#!/bin/bash
#SBATCH --job-name=regbench-selfcheck
# One run's selfcheck, in that run's image with that run's options: do the method's
# points follow the image its own writer produced? Submitted by submit_cohort.sh:
#
#   selfcheck.sbatch <name> <method> <label|-> <opts|-> <image> <gpu 0|1> <mask|->
#                    <name of the run whose selfcheck gives the anchor|->
#                    <attempt> <memory GB> <previous attempt's job id|->
#
# The record is $REGBENCH_OUT/logs/selfcheck/<name>/selfcheck.json. It is kept: the gate
# job reads it, and an anchored run's selfcheck reads the anchor run's (`--m0-from`).
# The selfcheck climbs in memory like a run: an attempt above the first runs only when
# the one before ended for memory (`selfcheck --retry-memory`), and otherwise leaves the
# record as it is and exits with its status. Exit: 0 ok, 1 inconsistent, 2 not checked,
# 3 crashed.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" gpu="$6" mask="$7" anchor="$8"
attempt="$9" mem_gb="${10}" previous="${11:--}"
export REGBENCH_ATTEMPT="$attempt" REGBENCH_MEM_GB="$mem_gb" REGBENCH_GPU="$gpu"

root="$REGBENCH_OUT/logs/selfcheck"
args=(selfcheck --method "$method" --work "$root/$name")
[ "$label" != - ] && args+=(--label "$label")
if [ "$opts" != - ]; then
    IFS=, read -ra items <<< "$opts"
    for o in "${items[@]}"; do args+=(--opt "$o"); done
fi
[ "$mask" != - ] && args+=(--mask "$mask")
[ "$anchor" != - ] && args+=(--m0-from "$root/$anchor")
if [ "$attempt" -gt 1 ]; then
    args+=(--retry-memory)
    if [ "$previous" != - ] && command -v sacct >/dev/null; then
        state="$(sacct -n -X -P -j "$previous" -o State 2>/dev/null | head -n 1)"
        [ -n "$state" ] && args+=(--previous-state "$state")
    fi
fi
regbench_py "$image" "${args[@]}"
code=$?
if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
    cat "$REGBENCH_END_FILE"
else
    # no record line: the selfcheck process died. Not evidence about the method.
    echo "REGBENCH_END kind=selfcheck run=$name rung=$attempt status=no_end_line cause=selfcheck_not_checked whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
