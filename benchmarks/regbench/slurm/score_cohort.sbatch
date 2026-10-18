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
#   score_cohort.sbatch tissue [<dataset>...]
#                                the tissue-level pieces of every written image (OpenCV)
#   score_cohort.sbatch case <dataset>
#                                ONE case of a cohort dataset, the SLURM_ARRAY_TASK_ID-th
#                                (`tissue --index`, then `score-case`): the tissue
#                                pieces of its written images, then its rows, its pair
#                                records side by side on the task's CPUs, kept under
#                                $REGBENCH_OUT/cache/scores. No table is written. A task
#                                that is killed keeps the pair records it finished
#   score_cohort.sbatch score    the tables. FIRST `regbench verdict`: what every run's
#                                selfcheck and realcheck said, into
#                                $REGBENCH_OUT/logs/verdicts.tsv; then `score
#                                --verdicts` names them in launch.csv, in the summary and
#                                in every row's selfcheck_status. Reported, never a
#                                gate: no row is left out for them. A case whose
#                                record is still valid is read; every other case is
#                                computed here, so the tables never depend on the array
#                                above. Every cost in them was measured by this
#                                benchmark: nothing is read from another run. It settles
#                                the killed results itself ONLY when the settle job left
#                                no word of having done it (a node without `sacct`, a
#                                job that was lost): the cases that changes are computed
#                                again here, never read from a record made before it
#
#
# `tissue` and `score` take the datasets of the submission when it named a public one
# (regbench.PUBLIC): each public dataset is read from its own cases folder
# (REGBENCH_CASES_<NAME>, env.sh: regbench_cases_of) and scored ALONE into
# $REGBENCH_OUT/tables/<dataset>/, never pooled with the cohort. Without them, the cohort.
#
# `case` and `score` are given the same submission: a case's record is reused only by
# a scorer that was (benchmarks/regbench/score.py: case_key). The checks' verdicts do
# not enter a case's record: `score` sets them on the rows after reading the records.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
kind="$1" dataset="${2:-}"  # `case` takes ONE dataset; `tissue` and `score` any number
sets=("${@:2}")
n_sets=$(( $# > 1 ? $# - 1 : 0 ))
workers="${SLURM_CPUS_PER_TASK:-1}"
verdicts="$REGBENCH_OUT/logs/verdicts.tsv"
case "$kind" in
    case) regbench_start "score_$kind" "dataset=$dataset" "unit=index_${SLURM_ARRAY_TASK_ID:--}" ;;
    score) export REGBENCH_RECORD="$REGBENCH_OUT/tables/summary.md"; regbench_start score ;;
    *) regbench_start "$kind" ;;
esac

# what `score-case` and `score` are both told: the submission
shared=()
shared_args() {
    # only what THIS submission ran or reused: an older submission's folders are not scored
    [ -n "${REGBENCH_SUBMISSION:-}" ] && shared+=(--submission "$REGBENCH_SUBMISSION")
    return 0
}

# The scheduler's accounting of every job of this submission, and the results that
# still say "killed" settled against it: such a result gets the cause and the elapsed
# time the scheduler names, so it is not scored as a method failure when the node failed,
# and its cost is not zero. Returns 0 when the accounting was read and applied; then
# logs/settled names the submission it was done for.
settled="$REGBENCH_OUT/logs/settled"
# settle_ids <jobs.tsv>: the ids of this submission's run jobs, comma-separated. The
# submission's own ids are logs/submitted/<submission>.ids (the launcher's, its
# watchers' and its tile submitters'); without that file (a folder filled before it was
# written), every run job of jobs.tsv.
settle_ids() {
    local mine="$REGBENCH_OUT/logs/submitted/${REGBENCH_SUBMISSION:-none}.ids"
    awk -F'\t' -v mine="$mine" '
        BEGIN { own = 0; while ((getline id < mine) > 0) { want[id] = 1; own = 1 } }
        $2 ~ /-[0-9]+g(-again)?$/ && $2 !~ /^rb-sc-/ && (!own || ($3 in want)) && !seen[$3]++ { print $3 }
    ' "$1" | paste -sd, -
}
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
    # Only a RUN TASK writes a run.json that can still say killed (run_rung.sbatch:
    # the jobs named rb-<run>-<dataset>[-first|-last]-<GB>g[-again]), and only this
    # submission's are settled (ladder.mine). jobs.tsv holds every job ever submitted
    # into the folder, the tile arrays of hundreds of tasks each included: asked for
    # all of them, with their steps, sacct could outlast this job's time limit.
    ids="$(settle_ids "$jobs")"
    if [ -z "$ids" ]; then
        echo "settle: no run task of submission ${REGBENCH_SUBMISSION:-?} in $jobs"
        return 1
    fi
    echo "settle: the accounting of $(tr ',' '\n' <<< "$ids" | wc -l | tr -d ' ') run job(s) of submission ${REGBENCH_SUBMISSION:-?}"
    # with the jobs' steps (no -X): a job's line alone reports no CPU time, its steps
    # do (`regbench ... settle` adds them up: ladder.parse_accounting)
    if ! sacct -n -P -j "$ids" -o JobID,State,ElapsedRaw,TotalCPU > "$acct" 2>/dev/null \
        || [ ! -s "$acct" ]; then
        echo "settle: sacct gave no accounting of the jobs in $jobs"
        return 1
    fi
    regbench_py "$REGBENCH_SCORER_IMAGE" settle --out "$REGBENCH_OUT" --accounting "$acct" \
        --cases "$REGBENCH_CASES" || return 1
    echo "${REGBENCH_SUBMISSION:-}" > "$settled"
}
already_settled() {
    [ -f "$settled" ] && [ "$(head -n 1 "$settled")" = "${REGBENCH_SUBMISSION:-}" ]
}

if [ "$kind" = settle ]; then
    rm -f "$settled"
    if settle_killed; then
        regbench_say_end "REGBENCH_END kind=settle status=ok cause=- whose=- exit=0 log=${REGBENCH_LOG:--}"
    else
        # the scorer tries again; if it cannot either, such results stay "killed"
        regbench_say_end "REGBENCH_END kind=settle status=failed cause=settle_unfinished whose=benchmark exit=1 log=${REGBENCH_LOG:--}"
    fi
    exit 0
fi
if [ "$kind" = case ]; then
    : "${SLURM_ARRAY_TASK_ID:?this task is one case of an array: submit with --array}"
    shared_args
    # the tissue-level pieces of this case's written images first (kept beside each
    # image; the tissue job and the scorer read them and compute only what is missing).
    # A failure here costs the case nothing: its rows are scored all the same
    regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$(regbench_cases_of "$dataset")" \
        --out "$REGBENCH_OUT" --dataset "$dataset" --index "$SLURM_ARRAY_TASK_ID" \
        --workers "$workers" \
        || echo "tissue $dataset index $SLURM_ARRAY_TASK_ID: not computed here (the tissue job and the scorer compute whatever is missing)"
    regbench_py "$REGBENCH_SCORER_IMAGE" score-case --cases "$(regbench_cases_of "$dataset")" \
        --out "$REGBENCH_OUT" --dataset "$dataset" --index "$SLURM_ARRAY_TASK_ID" \
        --workers "$workers" ${shared[@]+"${shared[@]}"}
    code=$?
    echo "score-case $dataset index $SLURM_ARRAY_TASK_ID: python exited $code (the scorer computes whatever is missing)"
    if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
        sed '$d' "$REGBENCH_END_FILE"
        regbench_say_end "$(tail -n 1 "$REGBENCH_END_FILE")"
    else
        regbench_say_end "REGBENCH_END kind=score_case dataset=$dataset unit=index_$SLURM_ARRAY_TASK_ID status=no_end_line cause=score_case_unfinished whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
    fi
    exit "$code"
fi
if [ "$kind" = tissue ]; then
    code=0
    if [ "$n_sets" -eq 0 ]; then
        regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$REGBENCH_CASES" \
            --out "$REGBENCH_OUT" --workers "$workers" || code=$?
    else
        for ds in ${sets[@]+"${sets[@]}"}; do
            regbench_py "$REGBENCH_SCORER_IMAGE" tissue --cases "$(regbench_cases_of "$ds")" \
                --out "$REGBENCH_OUT" --dataset "$ds" --workers "$workers" || code=$?
        done
    fi
    echo "tissue: python exited $code (the scorer computes whatever is missing)"
    regbench_say_end "REGBENCH_END kind=tissue status=$([ "$code" = 0 ] && echo ok || echo failed) cause=$([ "$code" = 0 ] && echo - || echo tissue_missing) whose=$([ "$code" = 0 ] && echo - || echo benchmark) exit=$code log=${REGBENCH_LOG:--}"
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
# what every run's selfcheck and realcheck said, FIRST: reported, never a gate. The runs
# are those this submission selected (submit_cohort.sh leaves their names beside its job
# ids); without that file, every active run. `verdict` always exits 0.
verdict=(verdict --out "$REGBENCH_OUT")
[ -n "${REGBENCH_SUBMISSION:-}" ] && verdict+=(--submission "$REGBENCH_SUBMISSION")
runs_file="$REGBENCH_OUT/logs/submitted/${REGBENCH_SUBMISSION:-none}.runs"
[ -s "$runs_file" ] && verdict+=(--runs "$(head -n 1 "$runs_file")")
rm -f "$verdicts"
regbench_py "$REGBENCH_SCORER_IMAGE" "${verdict[@]}"
echo "verdict: python exited $? ($verdicts; the tables say no selfcheck was found for a run it does not name)"
# its end line is not this job's: the scorer's own is the one the log ends with
if [ -n "${REGBENCH_END_FILE:-}" ]; then rm -f "$REGBENCH_END_FILE"; fi
args=(score --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --workers "$workers"
      ${shared[@]+"${shared[@]}"})
args+=(--verdicts "$verdicts")
# the cohort's manifest, for the first column of attrition.csv: REGBENCH_MANIFEST when
# this job has it, else the copy `submit_cohort.sh --prepare` left beside the cases;
# copied under the results (which every image can read)
manifest="${REGBENCH_MANIFEST:-}"
[ -n "$manifest" ] && [ -r "$manifest" ] || manifest="$REGBENCH_CASES/manifest.csv"
if [ -r "$manifest" ]; then
    cp "$manifest" "$REGBENCH_OUT/logs/manifest.csv" \
        && args+=(--manifest "$REGBENCH_OUT/logs/manifest.csv")
fi
code=0
cohort=()
for ds in ${sets[@]+"${sets[@]}"}; do [ "$ds" = multiplex ] && cohort=(--dataset multiplex); done
if [ "$n_sets" -eq 0 ] || [ "${#cohort[@]}" -gt 0 ]; then
    regbench_py "$REGBENCH_SCORER_IMAGE" "${args[@]}" ${cohort[@]+"${cohort[@]}"} || code=$?
fi
# each public dataset alone, from its own cases folder, into tables/<dataset>/
for ds in ${sets[@]+"${sets[@]}"}; do
    [ "$ds" = multiplex ] && continue
    regbench_py "$REGBENCH_SCORER_IMAGE" score --cases "$(regbench_cases_of "$ds")" \
        --dataset "$ds" --out "$REGBENCH_OUT" --workers "$workers" \
        ${shared[@]+"${shared[@]}"} --verdicts "$verdicts" || code=$?
done
# the chains no link of whose watcher finished (fleet wave 2, F1): named in the summary,
# never left to be inferred from rows that say `infrastructure`
if [ -s "$REGBENCH_OUT/tables/summary.md" ]; then
    regbench_py "$REGBENCH_SCORER_IMAGE" status --out "$REGBENCH_OUT" \
        --summary-to "$REGBENCH_OUT/tables/summary.md" \
        ${REGBENCH_SUBMISSION:+--submission "$REGBENCH_SUBMISSION"} \
        || echo "status: the chains no watcher finished could not be added to summary.md (see above)"
fi

# the scheduler's own accounting of every job, as a cross-check of the in-process numbers
if [ -s "$jobs" ] && command -v sacct >/dev/null; then
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    if sacct -j "$ids" --parsable2 --units=M \
        --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,UserCPU,SystemCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,AllocCPUS,ReqTRES,AllocTRES,NodeList \
        > "$REGBENCH_OUT/tables/slurm_accounting.psv"; then
        echo "scheduler accounting -> $REGBENCH_OUT/tables/slurm_accounting.psv"
    else
        # said, and the partial file is not left to pass for the table
        echo "scheduler accounting: sacct FAILED (exit $?): $REGBENCH_OUT/tables/slurm_accounting.psv is not written; the cross-check table is missing"
        rm -f "$REGBENCH_OUT/tables/slurm_accounting.psv"
    fi
    cp "$jobs" "$REGBENCH_OUT/tables/slurm_jobs.tsv"
fi
if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
    sed '$d' "$REGBENCH_END_FILE"
    regbench_say_end "$(tail -n 1 "$REGBENCH_END_FILE")"
else
    regbench_say_end "REGBENCH_END kind=score status=failed cause=unreadable_record whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
