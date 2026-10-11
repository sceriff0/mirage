#!/bin/bash
#SBATCH --job-name=regbench-score
# The end of a cohort submission, in the scorer's image. Two forms:
#
#   score_cohort.sbatch tissue   the tissue-level pieces of every written image (OpenCV)
#   score_cohort.sbatch score    the tables: every row the gates dropped is left out
#                                (`score --drop`), every blocked run and dropped row is
#                                named in the summary (`score --gates`), and the
#                                pipeline's own steps are read from a trace when
#                                REGBENCH_PIPELINE_TRACE names one
#
#   score_cohort.sbatch score <external dataset>
#                                an external landmark set (hyreco_a, hyreco_b, anhir_tune, anhir), alone:
#                                its own tables under $REGBENCH_OUT/tables/<dataset>/,
#                                never the cohort's
#
# Both read the gates' verdicts from $REGBENCH_OUT/logs/gates. A row is dropped when its
# own selfcheck failed and it does not gate its method.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
kind="$1" external="${2:-}"
workers="${SLURM_CPUS_PER_TASK:-1}"
if [ "$kind" = tissue ]; then
    regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$REGBENCH_CASES" \
        --out "$REGBENCH_OUT" --workers "$workers"
    code=$?
    echo "tissue: python exited $code (the scorer computes whatever is missing)"
    echo "REGBENCH_END kind=tissue status=$([ "$code" = 0 ] && echo ok || echo failed) cause=$([ "$code" = 0 ] && echo - || echo tissue_missing) whose=$([ "$code" = 0 ] && echo - || echo benchmark) exit=$code log=${REGBENCH_LOG:--}"
    exit 0
fi

# The scheduler's accounting of every job of this submission, BEFORE scoring: a result
# that still says "killed" (the last rung has no later task to ask) gets the cause and
# the elapsed time the scheduler names, so it is not scored as a method failure when the
# node failed, and its cost is not zero.
jobs="$REGBENCH_OUT/logs/jobs.tsv"
if [ -s "$jobs" ] && command -v sacct >/dev/null; then
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    acct="$REGBENCH_OUT/logs/accounting.psv"
    if sacct -n -X -P -j "$ids" -o JobID,State,ElapsedRaw,TotalCPU > "$acct" 2>/dev/null \
        && [ -s "$acct" ]; then
        regbench_py "$REGBENCH_SCORER_IMAGE" settle --out "$REGBENCH_OUT" --accounting "$acct"
    fi
fi

gates="$REGBENCH_OUT/logs/gates"
args=(score --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --workers "$workers")
# only what THIS submission ran or reused: a blocked run's older folders are not scored
[ -n "${REGBENCH_SUBMISSION:-}" ] && args+=(--submission "$REGBENCH_SUBMISSION")
# the gates' verdicts, for the cohort and for an external set alike: a run a gate
# blocked is not a method failure (ANHIR's scorer lists it apart; HyReCo's ignores it)
args+=(--gates "$gates")
[ -n "$external" ] && args+=(--dataset "$external")
for f in "$gates"/*.tsv; do
    [ -s "$f" ] || continue
    IFS=$'\t' read -r _ _ dropped _ _ < "$f"
    [ "${dropped:--}" = - ] && continue
    IFS=, read -ra rows <<< "$dropped"
    for v in "${rows[@]}"; do args+=(--drop "$v"); done
done
if [ -z "$external" ] && [ -n "${REGBENCH_PIPELINE_TRACE:-}" ]; then
    args+=(--pipeline-trace "$REGBENCH_PIPELINE_TRACE")
    [ -n "${REGBENCH_PIPELINE_TRACE_LABEL:-}" ] && \
        args+=(--pipeline-trace-label "$REGBENCH_PIPELINE_TRACE_LABEL")
fi
regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}"
code=$?

# the scheduler's own accounting of every job, as a cross-check of the in-process numbers
if [ -s "$jobs" ] && command -v sacct >/dev/null; then
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    sacct -j "$ids" --parsable2 --units=M \
        --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,UserCPU,SystemCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,AllocCPUS,ReqTRES,AllocTRES,NodeList \
        > "$REGBENCH_OUT/tables/slurm_accounting.psv" || true
    cp "$jobs" "$REGBENCH_OUT/tables/slurm_jobs.tsv"
    echo "scheduler accounting -> $REGBENCH_OUT/tables/slurm_accounting.psv"
fi
if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
    cat "$REGBENCH_END_FILE"
else
    echo "REGBENCH_END kind=score status=failed cause=unreadable_record whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
