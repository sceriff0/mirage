#!/bin/bash
#SBATCH --job-name=regbench-score
# The end of a cohort submission, in the scorer's image. Four forms:
#
#   score_cohort.sbatch settle   FIRST, before any case is scored: every result that still
#                                says "killed" is given the cause and the elapsed time the
#                                scheduler's accounting names (`regbench settle`: the last
#                                rung has no later task to ask). It rewrites those
#                                results' run.json, so it runs before the case tasks read
#                                them; it leaves $REGBENCH_OUT/logs/settled naming the
#                                submission when it has done so
#   score_cohort.sbatch tissue   the tissue-level pieces of every written image (OpenCV)
#   score_cohort.sbatch case <dataset>
#                                ONE case of a cohort dataset, the SLURM_ARRAY_TASK_ID-th
#                                (`tissue --index`, then `score-case`): the tissue
#                                pieces of its written images, then its rows, its pair
#                                records side by side on the task's CPUs, kept under
#                                $REGBENCH_OUT/cache/scores. No table is written. A task
#                                that is killed keeps the pair records it finished
#   score_cohort.sbatch score    the tables: every row the gates dropped is left out
#                                (`score --drop`) and every blocked run and dropped row
#                                is named in the summary (`score --gates`). A case whose
#                                record is still valid is read; every other case is
#                                computed here, so the tables never depend on the array
#                                above. Every cost in them was measured by this
#                                benchmark: nothing is read from another run. It settles
#                                the killed results itself ONLY when the settle job left
#                                no word of having done it (a node without `sacct`, a
#                                job that was lost): the cases that changes are computed
#                                again here, never read from a record made before it
#
#   score_cohort.sbatch score <external dataset>
#                                an external landmark set (hyreco_a, hyreco_b, anhir_tune, anhir), alone:
#                                its own tables under $REGBENCH_OUT/tables/<dataset>/,
#                                never the cohort's
#
# `case` and `score` read the gates' verdicts from $REGBENCH_OUT/logs/gates. A row is
# dropped when its own selfcheck failed and it does not gate its method. Both are given
# the same submission and the same dropped rows: a case's record is reused only by a
# scorer that was (benchmarks/regbench/score.py: case_key).
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
kind="$1" external="${2:-}"
workers="${SLURM_CPUS_PER_TASK:-1}"
gates="$REGBENCH_OUT/logs/gates"

# what `score-case` and `score` are both told: the submission, and the dropped rows
shared=()
shared_args() {
    local f dropped rows v
    # only what THIS submission ran or reused: a blocked run's older folders are not scored
    [ -n "${REGBENCH_SUBMISSION:-}" ] && shared+=(--submission "$REGBENCH_SUBMISSION")
    for f in "$gates"/*.tsv; do
        [ -s "$f" ] || continue
        IFS=$'\t' read -r _ _ dropped _ _ < "$f"
        [ "${dropped:--}" = - ] && continue
        IFS=, read -ra rows <<< "$dropped"
        for v in "${rows[@]}"; do shared+=(--drop "$v"); done
    done
    return 0
}

# The scheduler's accounting of every job of this submission, and the results that
# still say "killed" settled against it: such a result gets the cause and the elapsed
# time the scheduler names, so it is not scored as a method failure when the node failed,
# and its cost is not zero. Returns 0 when the accounting was read and applied; then
# logs/settled names the submission it was done for.
settled="$REGBENCH_OUT/logs/settled"
settle_killed() {
    local jobs="$REGBENCH_OUT/logs/jobs.tsv" acct="$REGBENCH_OUT/logs/accounting.psv" ids
    if [ ! -s "$jobs" ]; then
        echo "settle: no $jobs (nothing was submitted into this folder by the launcher)"
        return 1
    fi
    if ! command -v sacct >/dev/null; then
        echo "settle: no sacct on this node"
        return 1
    fi
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    if ! sacct -n -X -P -j "$ids" -o JobID,State,ElapsedRaw,TotalCPU > "$acct" 2>/dev/null \
        || [ ! -s "$acct" ]; then
        echo "settle: sacct gave no accounting of the jobs in $jobs"
        return 1
    fi
    regbench_py "$REGBENCH_SCORER_IMAGE" settle --out "$REGBENCH_OUT" --accounting "$acct" \
        || return 1
    echo "${REGBENCH_SUBMISSION:-}" > "$settled"
}
already_settled() {
    [ -f "$settled" ] && [ "$(head -n 1 "$settled")" = "${REGBENCH_SUBMISSION:-}" ]
}

if [ "$kind" = settle ]; then
    rm -f "$settled"
    if settle_killed; then
        echo "REGBENCH_END kind=settle status=ok cause=- whose=- exit=0 log=${REGBENCH_LOG:--}"
    else
        # the scorer tries again; if it cannot either, such results stay "killed"
        echo "REGBENCH_END kind=settle status=failed cause=settle_unfinished whose=benchmark exit=1 log=${REGBENCH_LOG:--}"
    fi
    exit 0
fi
if [ "$kind" = case ]; then
    : "${SLURM_ARRAY_TASK_ID:?this task is one case of an array: submit with --array}"
    shared_args
    # the tissue-level pieces of this case's written images first (kept beside each
    # image; the tissue job and the scorer read them and compute only what is missing).
    # A failure here costs the case nothing: its rows are scored all the same
    regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$REGBENCH_CASES" \
        --out "$REGBENCH_OUT" --dataset "$external" --index "$SLURM_ARRAY_TASK_ID" \
        --workers "$workers" \
        || echo "tissue $external index $SLURM_ARRAY_TASK_ID: not computed here (the tissue job and the scorer compute whatever is missing)"
    regbench_py "$REGBENCH_SCORER_IMAGE" score-case --cases "$REGBENCH_CASES" \
        --out "$REGBENCH_OUT" --dataset "$external" --index "$SLURM_ARRAY_TASK_ID" \
        --workers "$workers" ${shared[@]+"${shared[@]}"}
    code=$?
    echo "score-case $external index $SLURM_ARRAY_TASK_ID: python exited $code (the scorer computes whatever is missing)"
    if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
        cat "$REGBENCH_END_FILE"
    else
        echo "REGBENCH_END kind=score_case dataset=$external unit=index_$SLURM_ARRAY_TASK_ID status=no_end_line cause=score_case_unfinished whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
    fi
    exit "$code"
fi
if [ "$kind" = tissue ]; then
    regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$REGBENCH_CASES" \
        --out "$REGBENCH_OUT" --workers "$workers"
    code=$?
    echo "tissue: python exited $code (the scorer computes whatever is missing)"
    echo "REGBENCH_END kind=tissue status=$([ "$code" = 0 ] && echo ok || echo failed) cause=$([ "$code" = 0 ] && echo - || echo tissue_missing) whose=$([ "$code" = 0 ] && echo - || echo benchmark) exit=$code log=${REGBENCH_LOG:--}"
    exit 0
fi

# The killed results were settled by the settle job, before the case tasks read them.
# Only when that job left no word of it for this submission is it done here, BEFORE
# scoring: a case whose result this changes has a record made for the result as it was,
# which the scorer does not read (its key holds every run.json's bytes), so its rows are
# made again below.
jobs="$REGBENCH_OUT/logs/jobs.tsv"
if already_settled; then
    echo "settle: done by the settle job, before the cases were scored"
else
    echo "settle: the settle job left no record for this submission; settling here (a case whose result changes is scored again in this job)"
    settle_killed || echo "settle: not done: a result the scheduler ended stays 'killed' in the tables"
fi

shared_args
args=(score --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --workers "$workers"
      ${shared[@]+"${shared[@]}"})
# the gates' verdicts, for the cohort and for an external set alike: a run a gate
# blocked is not a method failure (ANHIR's scorer lists it apart; HyReCo's ignores it)
args+=(--gates "$gates")
[ -n "$external" ] && args+=(--dataset "$external")
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
