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
#                                an external landmark set (hyreco_a, hyreco_b), alone:
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
    echo "tissue: python exited $? (the scorer computes whatever is missing)"
    exit 0
fi

gates="$REGBENCH_OUT/logs/gates"
args=(score --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --workers "$workers")
if [ -n "$external" ]; then
    args+=(--dataset "$external")
else
    args+=(--gates "$gates")
fi
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
jobs="$REGBENCH_OUT/logs/jobs.tsv"
if [ -s "$jobs" ] && command -v sacct >/dev/null; then
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    sacct -j "$ids" --parsable2 --units=M \
        --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,UserCPU,SystemCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,AllocCPUS,ReqTRES,AllocTRES,NodeList \
        > "$REGBENCH_OUT/tables/slurm_accounting.psv" || true
    cp "$jobs" "$REGBENCH_OUT/tables/slurm_jobs.tsv"
    echo "scheduler accounting -> $REGBENCH_OUT/tables/slurm_accounting.psv"
fi
exit "$code"
