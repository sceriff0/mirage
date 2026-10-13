#!/bin/bash
#SBATCH --job-name=regbench-gate
# May one run start? Two forms, both submitted by submit_cohort.sh.
#
#   gate.sbatch selfcheck <run> <the run whose selfcheck gates it> <allowed 0|1>
#       reads $REGBENCH_OUT/logs/selfcheck/<gating run>/selfcheck.json and writes
#       $REGBENCH_OUT/logs/gates/<run>.tsv:
#       run, status, rows to drop (or -), blocked 0|1, the reference comparison (or -)
#
#   gate.sbatch realcheck <run> <method> <label|-> <dataset> <variant> <allowed 0|1>
#       reads the realcheck of the run's first unit on that dataset and writes
#       $REGBENCH_OUT/logs/gates/<run>.realcheck.tsv: run, status, -, blocked 0|1, -
#
# It exits non-zero when the run is blocked, and the run's arrays wait for it with
# `afterok`: a blocked run never starts, and only that run. <allowed> is 1 for a run
# named in REGBENCH_ALLOW_UNCHECKED: it starts when its selfcheck ran and could not
# decide (not_checked). Inconsistent, crashed, a flagged realcheck and a selfcheck that
# left no record of this submission block it all the same.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
kind="$1"; shift
mkdir -p "$REGBENCH_OUT/logs/gates"
if [ "$kind" = selfcheck ]; then
    run="$1" gating="$2" allowed="$3"
    args=(gate --name "$gating" --for "$run"
          --selfcheck "$REGBENCH_OUT/logs/selfcheck/$gating/selfcheck.json"
          --record "$REGBENCH_OUT/logs/gates/$run.tsv")
else
    run="$1" method="$2" label="$3" dataset="$4" variant="$5" allowed="$6"
    args=(realgate --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --dataset "$dataset"
          --name "$run" --method "$method" --variant "$variant" --index 0
          --record "$REGBENCH_OUT/logs/gates/$run.realcheck.tsv")
    [ "$label" != - ] && args+=(--label "$label")
fi
[ "$allowed" = 1 ] && args+=(--allow-unchecked)
# 2: the owner's override of a measured disagreement (REGBENCH_ALLOW_INCONSISTENT);
# the selfcheck gate alone knows it
[ "$allowed" = 2 ] && [ "$kind" = selfcheck ] && args+=(--allow-inconsistent)
# 3: the selfcheck is advice, not a gate (REGBENCH_SELFCHECK_ADVISORY=1)
[ "$allowed" = 3 ] && [ "$kind" = selfcheck ] && args+=(--advisory)
regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}"
code=$?
# a gate that could not run is the benchmark's failure, and the run stays blocked
if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
    cat "$REGBENCH_END_FILE"
else
    echo "REGBENCH_END kind=gate run=$run gate=$kind status=no_end_line cause=selfcheck_not_checked whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
