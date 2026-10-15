#!/bin/bash
# Submit the cohort benchmark: every run of the run list behind its own check, each on a
# four-rung memory ladder, then the scorer. From a login node, in two steps:
#
#   benchmarks/regbench/slurm/submit_cohort.sh --prepare [-- extra sbatch args]
#   (wait for the prepare jobs; look at their logs)
#   benchmarks/regbench/slurm/submit_cohort.sh [--dry-run] [--runs "a b"] \
#       [--datasets "multiplex synthetic"] [-- extra sbatch args]
#
# and, for a results folder whose registrations are done, the scoring alone:
#
#   benchmarks/regbench/slurm/submit_cohort.sh --score-only [--dry-run] \
#       [--datasets "multiplex synthetic"] [-- extra sbatch args]
#
# THE DEFAULT is `multiplex`, then `synthetic`: every array of a run on `synthetic` waits
# for that run's last array on `multiplex`, so the cohort is done first. `semisynth` and
# the HyReCo sets are DORMANT: implemented and tested, never part of the default, and
# submitted only when --datasets names them (the README lists what is known to be wrong
# with each before any use).
#
# --dry-run (or REGBENCH_DRY=1) prints every sbatch command with its dependencies, a job
# name in angle brackets standing for each job id, and submits nothing.
#
# THE PREFLIGHT runs before the first sbatch of a real submission, on the login node, and
# alone with --preflight (same --runs and --datasets; nothing is submitted): each image a
# selected run uses is started exactly as a job starts it and asked whether the run's
# modules import, whether the method's package is the version its adapter enforces,
# whether numpy resolves inside the image, and whether the results and work folders can
# be written (`regbench preflight`, benchmarks/regbench/preflight.py). One line per
# failure, and a failure stops the launch. REGBENCH_PREFLIGHT=0 skips it (a login node
# without the container runtime) and says so.
#
# THE PINNED STARE SOURCE, for the images without a `stare` package
# (REGBENCH_STARE_SRC_IMAGES), is fetched once into $REGBENCH_OUT/cache before the
# preflight and before the first sbatch of every real submission, WHETHER OR NOT the
# preflight runs: the jobs of those images import it, and a job may have no network.
#
# THE RUN LIST is read from `python -m benchmarks.regbench runs` and from nowhere else:
# this script names no method. What it submits, in order:
#
#   1. per run that gates a run: its selfcheck at 64 GB and its watcher (retry.sbatch),
#      which submits it at 256 and then 512 GB only when the attempt before did not
#      end ok. An attempt above the first runs only when the one before ended for memory. A run whose
#      results another reads (column `after`) hands its selfcheck's work to that one
#      (`--m0-from`), which waits for it.
#   2. per run: a gate job (`regbench gate`). It fails when the selfcheck that gates the
#      run is anything but ok, and then THAT run alone never starts. The rows it says to
#      drop are collected for the scorer.
#   3. per run and dataset: the memory ladder, 16 -> 64 -> 256 -> 512 GB at 4 CPUs. ONE
#      array is submitted here, the 16 GB attempt, and one small job after it, the
#      ladder's watcher (retry.sbatch). A RETRY IS SUBMITTED ONLY AFTER A FAILURE: the
#      watcher submits the same rung once more for a unit the scheduler lost (node
#      failure, preemption, a cancelled job, a task that never started) and the next
#      rung for a unit that ran out of memory, as an array of those units alone, one
#      step after the other. A ladder whose units are all ok puts nothing more in the
#      queue. A task of a later step still decides for itself and exits at once unless
#      its unit's last run.json names the cause that step retries. One task is
#      one case for a run that works per pair and one patient for a run that works per
#      patient. A run that reads another's results waits for that run's last array on
#      the same dataset. A unit whose results are all ok already (an earlier
#      submission into the same REGBENCH_OUT) is reused, not run: REGBENCH_FORCE=1.
#      A run whose registration is several scheduler jobs per round (column `stages`:
#      the pipeline's own shape) has a chain of stage jobs instead, each with the
#      pipeline's allocation for that process: per attempt an array over the rounds for
#      COARSE and for SOLVE, and last the round's last job on its own rungs, all
#      submitted here. THE TILE JOBS ARE NOT: one task per tile would be hundreds of
#      thousands of queued tasks. Between COARSE and SOLVE sits the run's TILE SUBMITTER
#      (feed_tiles.sbatch), a small job on a compute node that calls sbatch itself, as
#      a Nextflow head job does: still one job per tile with the pipeline's allocation,
#      in arrays of at most REGBENCH_STAGE_ARRAY_MAX tasks, never more than
#      its share of REGBENCH_STAGE_QUEUE_MAX (a total over the launch) of them pending or
#      running. SOLVE waits for it.
#      `stage_chain` and `staged` below.
#   4. for a run whose realcheck is a gate (column `realcheck`): the first unit of the
#      first dataset goes alone, with `--realcheck`; a second gate job reads its
#      realcheck.json; everything else of that run waits for it.
#   5. after everything, in the scorer's image: FIRST the settle job (`rb-settle`: every
#      result that still says "killed" gets the cause the scheduler's accounting names;
#      it rewrites those results' run.json, which is why every scoring job waits for
#      it); then per dataset ONE ARRAY with a task per case (the tissue-level pieces of
#      the case, then `score-case`: the case's rows, kept as its record under
#      $REGBENCH_OUT/cache/scores), then the tissue job (the pieces of any case whose
#      task did not make them), then the scorer (`score --drop ...`), which reads the
#      records, computes whatever case has none that is still valid, and writes the
#      tables. The tables are the same whether the array ran, half of it did, or none
#      of it.
#
# --score-only submits step 5 and nothing before it: no selfcheck, no gate, no
# registration. It scores what the LAST submission into REGBENCH_OUT ran or reused
# (its id is kept in $REGBENCH_OUT/logs/submission; REGBENCH_SUBMISSION names another),
# with the gate records that submission left. Nothing is run again, and a case whose
# record is still valid is not computed again.
#
# Nothing waits on the login node and no job waits on `sbatch --wait`. One kind of job
# submits jobs: the tile submitter of a run whose registration is several jobs per round.
#
# A SECOND SUBMISSION into a REGBENCH_OUT whose earlier jobs are still pending or running
# is REFUSED (its gate records would replace the ones those jobs read). Wait, cancel them
# (the refusal prints the line), or use another REGBENCH_OUT. --score-only is refused
# the same way (it would score, and settle, results whose tasks are still running):
# REGBENCH_SCORE_WHILE_QUEUED=1 scores anyway. --prepare is not refused.
#
# A SUBMISSION ID IS USED ONCE per REGBENCH_OUT. A call that would run anything under an
# id this folder already holds a submission of (logs/submitted/<id>.ids) is refused: the
# tile submitters of the new call would take the old call's lists for their own and feed
# nothing. Leave REGBENCH_SUBMISSION unset and every call gets its own; what is ok is
# reused whatever the id.
#
# EVERYTHING THAT CAN STOP THE SCRIPT IS READ BEFORE THE FIRST sbatch: the run list, the
# cases, the preflight, the pipeline's allocation per stage, and the tile layout of
# every run whose registration is several jobs per round. After the first sbatch only
# the scheduler can stop it, and whatever does (an sbatch it refuses, any error of this
# script) ends the same way: how many jobs of this call are already queued, where their
# ids are, and the scancel line that removes them. Nothing is left half-submitted
# without that being said.
#
# AN EXTERNAL LANDMARK SET (HyReCo) is submitted on its own, never with the cohort:
#
#   submit_cohort.sh --prepare --datasets hyreco_a      (then the same without --prepare)
#   submit_cohort.sh --prepare --datasets hyreco_b      (then the same without --prepare)
#
# `hyreco_a` is the TUNING set and `hyreco_b` the REPORT set: subset B is prepared with
# the settings of the prepared subset-A cases and with nothing else. Such a submission
# reads its own run list (`runs --dataset <name>`: only the runs with an input rule for
# brightfield), prepares one pair per array task, submits no tissue job, sends no first
# unit ahead for a realcheck (both are defined on nuclei), and its scorer writes
# $REGBENCH_OUT/tables/<name>/ and none of the cohort's tables. Use a REGBENCH_OUT of
# its own.
#
# ANHIR's TRAINING PAIRS are another external landmark set, in two datasets from one
# prepare (README, "ANHIR: the challenge's training pairs"):
#
#   submit_cohort.sh --prepare --datasets anhir         (writes anhir AND anhir_tune)
#   REGBENCH_ANHIR_TUNE=1 submit_cohort.sh --datasets anhir_tune
#                                                       the preprocessing candidates
#                                                       alone, on the tuning side (PAUSED
#                                                       with stare_prep: refused unless
#                                                       REGBENCH_ANHIR_TUNE=1)
#   python -m benchmarks.regbench prep-select --cases "$REGBENCH_CASES" \
#       --out "$REGBENCH_OUT" --freeze                  (then commit prep_frozen.json)
#   submit_cohort.sh --datasets anhir                   every run ANHIR takes, on all 230
#                                                       training pairs
#
# Neither is part of the default. While prep_frozen.json says `provisional`, every
# submission that holds the preprocessing row says so, in a dry run too.
#
# ENVIRONMENT (env.sh has the paths and the images)
#   REGBENCH_CASES, REGBENCH_OUT      prepared cases; results
#   REGBENCH_SIF_<image>              one per image of the run list (column `image`)
#   REGBENCH_SBATCH_GPU (--gres=gpu:1)  what a task that needs a GPU asks for: one GPU.
#                                     Export the site's own form where it names the
#                                     card, e.g. "--gres=gpu:nvidia_h200:1". Set EMPTY,
#                                     the runs that need a GPU are refused.
#   REGBENCH_TIME (24:00:00)          per round, every rung. A run that works per patient
#                                     gets it times the rounds of the largest patient,
#                                     never above the site's longest job (7 days)
#   REGBENCH_SELFCHECK_TIME (24:00:00)
#   REGBENCH_ACCT_WAIT_S (600)        the longest a ladder task waits for the scheduler's
#                                     accounting to name the END state of an earlier
#                                     task whose result still says "killed"
#   REGBENCH_ACCT_POLL_S (30)         seconds between two looks at it (run_rung.sbatch)
#   REGBENCH_ARRAY_LIMIT (20)         tasks of one array that run at once (%N); not
#                                     applied to the tile arrays, whose limit is
#                                     REGBENCH_STAGE_QUEUE_MAX
#   REGBENCH_STAGE_ARRAY_MAX (100)    the most tasks in one tile array
#   REGBENCH_STAGE_QUEUE_MAX (200)    the most tile tasks pending or running at a time
#                                     in the WHOLE LAUNCH: a total, split evenly over the
#                                     runs with a tile submitter that feed at once (all
#                                     of them, or REGBENCH_RUN_LANES when fewer; floor,
#                                     at least 1 each); each submitter's ceiling is its share
#   REGBENCH_STAGE_FEED_MIN (25)      the submitter waits for that many free slots
#                                     before it submits an array (fewer when fewer
#                                     tiles are left)
#   REGBENCH_STAGE_POLL_S (60)        seconds between two looks at the queue
#   REGBENCH_STAGE_FEEDERS (2)        tile submitters chained per run and dataset: the
#                                     second goes on where the first stopped when the
#                                     first is ended at its time limit
#   REGBENCH_STAGE_FEED_TIME (7-00:00:00)  the time limit of one of them (the site's
#                                     longest job)
#   REGBENCH_STAGE_SQUEUE_TRIES (30), REGBENCH_STAGE_STALL_S (3600),
#   REGBENCH_STAGE_STALL_MAX_S (172800), REGBENCH_STAGE_HELD_REASONS,
#   REGBENCH_STAGE_WAIT_REASONS
#                                     when a tile submitter stops waiting and gives up
#                                     (feed_tiles.sbatch has the rules)
#                                     The dry run prints how many sbatch calls are made
#                                     now, how many tile tasks are fed in later, and
#                                     the most tasks in the queue at once
#   REGBENCH_RUN_LANES (0)            runs in flight at once (0: no limit); see the ladders
#   REGBENCH_RUN_ORDER                the order the runs are SUBMITTED in, and so dealt
#                                     onto the lanes: family or run names, space-separated
#                                     (`runs --order`); the runs it does not name follow
#                                     in the run list's order. A run that reads another's
#                                     results is never put before it (a `#` line says
#                                     when it was moved). Default: `default`, the run
#                                     list's own first runs (`runs --order default`
#                                     prints the order), on the cohort
#                                     and the synthetic set; set EMPTY, and on an
#                                     external set, the run list's order.
#                                     Nothing but the order of submission changes: not
#                                     the run list, not a table, not a result
#   REGBENCH_SCORE_CPUS (8)           per task of the per-case scoring arrays: a case's
#                                     pair records are computed on that many processes
#   REGBENCH_SCORE_MEM (64G)          per task of those arrays, and of the tissue and
#                                     score jobs (which keep 4 CPUs and one day)
#   REGBENCH_SCORE_TIME (24:00:00)    per task of those arrays. A task that runs out of
#                                     it keeps the pair records it finished: submit
#                                     again with --score-only and it goes on from there
#   REGBENCH_FORCE=1                  run every unit again, also those already ok
#   REGBENCH_SUBMISSION               this call's id (default: time and process id).
#                                     Records of another submission are evidence of
#                                     nothing: not for a gate, the ladder or the scorer.
#                                     With --score-only: the submission to score. An id
#                                     already used for a submission into REGBENCH_OUT
#                                     is refused for another one
#   REGBENCH_SCORE_WHILE_QUEUED=1     --score-only although jobs of an earlier call
#                                     into REGBENCH_OUT are still pending or running
#   REGBENCH_SELFCHECK_ADVISORY=1     the selfcheck is advice, not a gate: EVERY run
#                                     starts whatever its selfcheck says, and the gate
#                                     record, the end line and the summary keep the
#                                     verdict. launch.sbatch sets it; unset, a
#                                     selfcheck that is not ok blocks its run
#   REGBENCH_ALLOW_INCONSISTENT       run names that start although their selfcheck
#                                     MEASURED a disagreement (`inconsistent`): the
#                                     owner's override. The verdict is recorded as it
#                                     is; a crash or a missing record still blocks
#   REGBENCH_ALLOW_UNCHECKED          run names that start although their selfcheck ran
#                                     and could not decide (`not_checked`). Nothing
#                                     else is let through: not `inconsistent`, not
#                                     `crashed`, not a flagged realcheck, not a
#                                     selfcheck that left no record
#   REGBENCH_MANIFEST                 with --prepare: the cohort's rounds.csv
#   REGBENCH_TRUTH_PATIENTS           with --prepare: the patients of the known-truth
#                                     cases, comma-separated (default: all)
#   REGBENCH_PREPARE_ARGS             with --prepare: more `prepare multiplex` options
#   REGBENCH_PREPARE_FORCE=1          with --prepare: redo cases that exist
#   REGBENCH_NUCLEI                   with --prepare: where each slide's BARE nuclei are
#                                     kept (default $REGBENCH_CASES/_nuclei). An array
#                                     over the manifest's slides runs before `prepare
#                                     multiplex`: an existing label image is checked,
#                                     and only a slide without one is segmented again
#   REGBENCH_STARDIST_MODEL, REGBENCH_STARDIST_MODEL_DIR
#                                     the StarDist model the arms ran (the pipeline's
#                                     segmentation_model and segmentation_model_dir).
#                                     Unset, both are read from the site config; set,
#                                     they win. One of the two routes is REQUIRED for
#                                     that step, with REGBENCH_SIF_STARDIST
#   REGBENCH_SITE_CONFIG              the site config to read them from, and any
#                                     override of the StarDist settings (seg_n_tiles_*,
#                                     seg_pmin, seg_pmax, seg_prob_thresh,
#                                     seg_expand_distance). Default: conf/ieo.config of
#                                     this checkout if it exists; `none` reads no file.
#                                     Where each value came from is printed (`# StarDist
#                                     re-run:` lines), in a dry run too
#   REGBENCH_STARDIST_CPU=1           run that StarDist without a GPU (else it asks for
#                                     REGBENCH_SBATCH_GPU, as the arms did)
#   REGBENCH_NUCLEI_FORCE=1           decide EVERY slide again, whatever is kept, usable
#                                     nuclei included: a kept re-run is set aside and
#                                     StarDist runs again, a label image is checked
#                                     again. (Without it a kept re-run is set aside and
#                                     redone by itself when its settings no longer
#                                     resolve to the same values, or when its slide is
#                                     no longer the file it was computed from.)
#   REGBENCH_NUCLEI_SKIP=1            prepare without that step: cases whose slides have
#                                     no nuclei are scored on the outlines alone
#   REGBENCH_NUCLEI_FALLBACK=shrink   for a slide still without nuclei, use its outline
#                                     shrunk by 10 px (labelled, never exact). Off unless set
#   REGBENCH_NUCLEI_CPUS (8), REGBENCH_NUCLEI_MEM (128G), REGBENCH_NUCLEI_TIME (08:00:00)
#   REGBENCH_HYRECO_A                 with --prepare --datasets hyreco_a: the unpacked
#                                     HyReCo/ folder (it holds HE/ and PHH3/)
#   REGBENCH_HYRECO_A_ARGS            more `prepare hyreco --subset a` options: the
#                                     settings being tried (--proxy, --reduce)
#   REGBENCH_HYRECO_B_HE, REGBENCH_HYRECO_B_PHH3
#                                     with --prepare --datasets hyreco_b: the HE/ and
#                                     PHH3/ folders of HyReCo-Additional (two archives)
#   REGBENCH_HYRECO_TUNED             the cases folder holding the prepared hyreco_a
#                                     cases subset B takes its settings from (default:
#                                     REGBENCH_CASES)
#   REGBENCH_HYRECO_B_ARGS            more `prepare hyreco --subset b` options (never a
#                                     setting: --proxy and --reduce are refused there)
#   REGBENCH_ANHIR                    for EVERY ANHIR submission, prepare and runs: the
#                                     folder holding dataset_medium.csv, images/ and
#                                     landmarks/ (the challenge's unpacked archives;
#                                     only read). The cases name its images, so it is
#                                     bound into every image (env.sh)
#   REGBENCH_ANHIR_ARGS               more `prepare anhir` options (--allow-incomplete)
#   REGBENCH_HEAD_IMAGE               the image this script itself reads the run list
#                                     with (default STARE_REGBENCH)
# The rungs, 4 CPUs and the 7-day limit are frozen (benchmarks/regbench/ladder.py).
# No thread variable is exported anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# who starts a run's stage processes is stated per run by its own job script; a value
# left in the caller's shell would reach every job (--export=ALL) and every image
unset REGBENCH_STAGES
# shellcheck source=benchmarks/regbench/slurm/env.sh
source "$HERE/env.sh"

RUNGS_GB="16 64 256 512"
SELFCHECK_RUNGS_GB="64 256 512"
CPUS=4
MAX_SECONDS=$((7 * 24 * 3600))
: "${REGBENCH_TIME:=24:00:00}"
: "${REGBENCH_SELFCHECK_TIME:=24:00:00}"
: "${REGBENCH_ARRAY_LIMIT:=20}"
: "${REGBENCH_RUN_LANES:=0}"
: "${REGBENCH_SBATCH_GPU=--gres=gpu:1}"
: "${REGBENCH_STAGE_ARRAY_MAX:=100}"
: "${REGBENCH_STAGE_QUEUE_MAX:=200}"
: "${REGBENCH_STAGE_FEED_MIN:=25}"
: "${REGBENCH_STAGE_POLL_S:=60}"
: "${REGBENCH_STAGE_FEEDERS:=2}"
: "${REGBENCH_STAGE_FEED_TIME:=7-00:00:00}"
: "${REGBENCH_SCORE_CPUS:=8}"
: "${REGBENCH_SCORE_MEM:=64G}"
: "${REGBENCH_FORCE:=0}"
export REGBENCH_STAGE_ARRAY_MAX REGBENCH_STAGE_FEED_MIN REGBENCH_STAGE_POLL_S
: "${REGBENCH_HEAD_IMAGE:=STARE_REGBENCH}"
DRY="${REGBENCH_DRY:-0}"
# multiplex first, then synthetic. semisynth and hyreco_* are dormant: only when named.
# The ANHIR sets (anhir, anhir_tune) are never part of the default either.
DEFAULT_DATASETS="multiplex synthetic"
DORMANT_DATASETS="semisynth hyreco_a hyreco_b"
prepare=0 score_only=0 preflight_only=0 only_runs="" datasets="$DEFAULT_DATASETS"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --prepare) prepare=1; shift ;;
        --score-only) score_only=1; shift ;;
        --preflight) preflight_only=1; shift ;;
        --runs) only_runs="$2"; shift 2 ;;
        --datasets) datasets="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
[ "$DRY" = 0 ] || DRY=1

die() { echo "submit_cohort: $*" >&2; exit 2; }
# However this script ends once a job of this call is queued, short of its last line
# (FINISHED is set just before it): say what is queued and how to remove it (and, in a
# dry run, remove the scratch folder). The exit code is not what decides: a signal
# reaches the EXIT trap with the code of the last command, which may be 0.
N_QUEUED=0
FINISHED=0
STOPPED_BY=""
IN_SBATCH=0
PENDING_EXIT=""
scratch_is_temporary=0
# shellcheck disable=SC2329  # called by the EXIT trap below
on_exit() {
    local code=$?
    [ "$scratch_is_temporary" = 1 ] && rm -rf "$scratch"
    if [ "$FINISHED" != 1 ] && [ "$N_QUEUED" -gt 0 ]; then
        {
            echo "submit_cohort: stopped with exit $code after the first sbatch (${STOPPED_BY:-the reason is above})."
            echo "submit_cohort: $N_QUEUED job(s) of this call are already queued (submission $REGBENCH_SUBMISSION); the rest was NOT submitted, so the graph is incomplete."
            echo "submit_cohort: their ids, one per line: $IDS_FILE (with names: $JOBS_FILE)"
            echo "submit_cohort: cancel them:  scancel \$(cat \"$IDS_FILE\")"
        } >&2
    fi
}
trap on_exit EXIT
# A signal ends the script with the shell's code for it (128 + the signal), through the
# EXIT trap. One that arrives while an sbatch is on its way is kept until that job's id
# is in the ids file: a job the scheduler took is never missing from the notice.
# shellcheck disable=SC2329  # called by the signal traps below
on_signal() {
    STOPPED_BY="signal $1"
    [ "$IN_SBATCH" = 1 ] || exit "$2"
    PENDING_EXIT="$2"
}
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
has() { case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac; }
# whole numbers of at least <least>: a zero here is an array of no task, or a loop
# that never ends
need_int() {
    local var="$1" least="$2" value="${!1}"
    case "$value" in ''|*[!0-9]*) die "$var='$value': a whole number of at least $least. Nothing was submitted." ;; esac
    [ "$((10#$value))" -ge "$least" ] || die "$var='$value': a whole number of at least $least. Nothing was submitted."
    printf -v "$var" '%s' "$((10#$value))"
}
need_int REGBENCH_ARRAY_LIMIT 1
need_int REGBENCH_STAGE_ARRAY_MAX 1
need_int REGBENCH_STAGE_QUEUE_MAX 1
need_int REGBENCH_STAGE_FEED_MIN 1
need_int REGBENCH_STAGE_POLL_S 1
need_int REGBENCH_STAGE_FEEDERS 1
need_int REGBENCH_RUN_LANES 0
need_int REGBENCH_SCORE_CPUS 1
# read by the jobs, not here (feed_tiles.sbatch and run_rung.sbatch have the defaults):
# a value that is no whole number stops the launch, not a job days later
for pair in REGBENCH_STAGE_STALL_S:1 REGBENCH_STAGE_STALL_MAX_S:1 REGBENCH_STAGE_SQUEUE_TRIES:1 \
            REGBENCH_ACCT_WAIT_S:0 REGBENCH_ACCT_POLL_S:1; do
    var="${pair%%:*}"
    [ -z "${!var+set}" ] || need_int "$var" "${pair##*:}"
done
[ "$preflight_only" = 1 ] && DRY=0   # it starts the images; it submits nothing
[ "$prepare" = 1 ] && [ "$score_only" = 1 ] && die "--prepare and --score-only are two different steps; name one. Nothing was submitted."

# one id per call: every record of this submission carries it. --score-only makes no
# record of its own: it scores an EARLIER submission, the last one into REGBENCH_OUT
# unless REGBENCH_SUBMISSION names another.
if [ "$score_only" = 1 ]; then
    # The ids the folder's selfcheck records carry, each record PARSED by the reader the
    # gates use (`regbench submissions`, benchmarks/regbench/ladder.py): one id per line.
    found="$(regbench_py "$REGBENCH_HEAD_IMAGE" submissions --out "$REGBENCH_OUT" | cut -f1)" \
        || die "--score-only: cannot read the selfcheck records under $REGBENCH_OUT/logs/selfcheck (python -m benchmarks.regbench submissions). Nothing was submitted."
    if [ -z "${REGBENCH_SUBMISSION:-}" ]; then
        if [ -s "$REGBENCH_OUT/logs/submission" ]; then
            REGBENCH_SUBMISSION="$(head -n 1 "$REGBENCH_OUT/logs/submission")"
        else
            # a folder filled before that file was written: the id is in its selfcheck records
            [ -n "$found" ] || die "--score-only: no submission to score under $REGBENCH_OUT (no logs/submission, and no selfcheck record names one). Export REGBENCH_SUBMISSION=<the id an earlier submit_cohort.sh printed as '# submission  -> ID'; it is also in every run.json, key 'submissions'>. Nothing was submitted."
            [ "$(wc -l <<< "$found")" -eq 1 ] || die "--score-only: $REGBENCH_OUT holds records of more than one submission ($(paste -sd' ' - <<< "$found")) and no logs/submission says which was last. Export REGBENCH_SUBMISSION=<one of them>. Nothing was submitted."
            REGBENCH_SUBMISSION="$found"
        fi
        echo "# --score-only: scoring submission $REGBENCH_SUBMISSION"
    fi
    # the id is the user's, or the file's: said aloud when no selfcheck record is of it
    if [ -n "$found" ] && ! grep -qxF -- "$REGBENCH_SUBMISSION" <<< "$found"; then
        echo "# WARNING --score-only: no selfcheck record under $REGBENCH_OUT is of submission $REGBENCH_SUBMISSION (the records there are of: $(paste -sd' ' - <<< "$found")). The scorer takes only what $REGBENCH_SUBMISSION ran or reused, so every row may read 'not run': check the id."
    fi
fi
: "${REGBENCH_SUBMISSION:=$(date +%Y%m%dT%H%M%S)-$$}"
export REGBENCH_SUBMISSION
for ds in $datasets; do
    has "$DORMANT_DATASETS" "$ds" && echo "# $ds is DORMANT: submitted because it was named. Read \"Dormant datasets: known defects before any use\" in benchmarks/regbench/README.md first."
done

# An external landmark set is submitted alone: its own run list, its own tables.
EXTERNAL="" n_named=0
for ds in $datasets; do
    n_named=$((n_named + 1))
    case "$ds" in hyreco_a|hyreco_b|anhir|anhir_tune) EXTERNAL="$ds" ;; esac
done
if [ -n "$EXTERNAL" ] && [ "$n_named" -ne 1 ]; then
    die "--datasets '$datasets': an external landmark set (hyreco_a, hyreco_b, anhir_tune, anhir) is submitted alone, with its own run list and its own tables. hyreco_a and anhir_tune are tuning sets. Nothing was submitted."
fi
# ANHIR's tuning side is PAUSED with `stare_prep`, the only row that reads it
# (variants.TUNING_PAUSED): it is submitted only when asked for by name AND by this variable.
if [ "$EXTERNAL" = anhir_tune ] && [ "${REGBENCH_ANHIR_TUNE:-0}" != 1 ]; then
    die "--datasets anhir_tune: ANHIR's tuning side is PAUSED with stare_prep, the only row that reads it (benchmarks/regbench/variants.py, TUNING_PAUSED). Export REGBENCH_ANHIR_TUNE=1 to submit it anyway. Nothing was submitted."
fi

# seconds <[D-]HH:MM:SS> ; limit <seconds> -> D-HH:MM:SS, never above the site's maximum
# Every call is a plain assignment made before the first sbatch (`x="$(seconds ...)"`):
# there a time that is not one stops the script. Nested in another substitution, or in
# an argument of sbatch, the `die` would end a subshell only.
seconds() {
    local t="$1" d=0 h m s part
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    IFS=: read -r h m s <<< "$t"
    for part in "$d" "${h:-}" "${m:-}" "${s:-}"; do
        case "$part" in ''|*[!0-9]*) die "a time must be [D-]HH:MM:SS, got '$1'. Nothing was submitted." ;; esac
    done
    echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s ))
}
limit() {
    local s="$1"
    [ "$s" -gt "$MAX_SECONDS" ] && s="$MAX_SECONDS"
    printf '%d-%02d:%02d:%02d' $((s / 86400)) $((s % 86400 / 3600)) $((s % 3600 / 60)) $((s % 60))
}

# submit <job name> <sbatch arguments...>: sets JOB_ID. Dry: prints the command.
N_JOBS=0
N_TASKS=0
JOB_ID=""
JOBS_FILE=""
submit() {
    local name="$1" arg range; shift
    N_JOBS=$((N_JOBS + 1))
    range=""
    for arg in "$@"; do case "$arg" in --array=*) range="${arg#--array=}"; range="${range%%\%*}" ;; esac; done
    if [ -n "$range" ]; then
        N_TASKS=$((N_TASKS + ${range#*-} - ${range%%-*} + 1))
    else
        N_TASKS=$((N_TASKS + 1))
    fi
    if [ "$DRY" = 1 ]; then
        printf 'sbatch --parsable --job-name=%s' "$name"
        printf ' %s' "$@"
        printf '\n'
        JOB_ID="<$name>"
        return 0
    fi
    local got
    IN_SBATCH=1
    if ! got="$(sbatch --parsable --job-name="$name" "$@" < /dev/null 2>&1)"; then
        IN_SBATCH=0
        refused "$name" "$got"
    fi
    # sbatch went through: the job id is the LAST line it printed. A site's submit
    # plugin may print lines of its own before it ("sbatch: slurm_job_submit: Job
    # assigned to GPU nodes"), and the job is in the queue all the same: reading that as
    # a refusal left a queued job that no list named.
    JOB_ID="$(awk 'NF { last = $0 } END { print last }' <<< "$got")"
    [ "$JOB_ID" = "$got" ] || awk -v id="$JOB_ID" '$0 != id && NF { print "# sbatch said: " $0 }' <<< "$got" >&2
    JOB_ID="${JOB_ID%%;*}"
    case "$JOB_ID" in ''|*[!0-9_]*) IN_SBATCH=0; refused "$name" "sbatch printed no job id: $got" ;; esac
    # the id first: whatever fails after this line, the job is in the file the
    # closing notice (on_exit) names
    echo "$JOB_ID" >> "$IDS_FILE"
    N_QUEUED=$((N_QUEUED + 1))
    IN_SBATCH=0
    # the last submission that ran anything into this folder: what --score-only scores.
    # Written with the first job the scheduler took: a call that queued nothing (stopped
    # by the preflight, a check, or a refused first sbatch) is not a submission.
    if [ "$N_QUEUED" = 1 ] && [ "$prepare" != 1 ] && [ "$score_only" != 1 ]; then
        echo "$REGBENCH_SUBMISSION" > "$REGBENCH_OUT/logs/submission"
    fi
    printf '%s\t%s\n' "$name" "$JOB_ID"
    printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$name" "$JOB_ID" >> "$JOBS_FILE"
    [ -z "$PENDING_EXIT" ] || exit "$PENDING_EXIT"
}
# refused <job name> <the scheduler's words>: an sbatch that did not go through. The
# jobs before it, if any, ARE queued, and some of them wait for jobs that will never
# exist: on_exit says how many, where their ids are and how to cancel them.
refused() {
    {
        echo "submit_cohort: sbatch REFUSED the job '$1': $2"
        if [ "$N_QUEUED" -gt 0 ]; then
            echo "submit_cohort: submit again once that reason is gone (a limit on queued jobs: see README, \"What the site must allow\"), after cancelling what is queued:"
        else
            echo "submit_cohort: submit again once that reason is gone (a limit on queued jobs: see README, \"What the site must allow\"). Nothing was submitted."
        fi
    } >&2
    exit 2
}
# every job: the caller's own options, logs in one place, never pending for ever on a
# dependency that can no longer be met, and never requeued: a task the scheduler starts
# a second time would find only the placeholder its first start left and read it as the
# method's death. A node failure ends the task as NODE_FAIL, and the `again` task of the
# same rung runs the unit once more
common() {
    COMMON=(--export=ALL --kill-on-invalid-dep=yes --no-requeue)
    [ ${#extra[@]} -gt 0 ] && COMMON+=("${extra[@]}")
    return 0
}
common
LOG_ONE="--output=$REGBENCH_OUT/logs/%x-%j.log"
LOG_ARRAY="--output=$REGBENCH_OUT/logs/%x-%A_%a.log"
round_s="$(seconds "$REGBENCH_TIME")"
sc_s="$(seconds "$REGBENCH_SELFCHECK_TIME")"
prep_s="$(seconds "${REGBENCH_PREPARE_TIME:-24:00:00}")"
feed_s="$(seconds "$REGBENCH_STAGE_FEED_TIME")"
score_s="$(seconds "${REGBENCH_SCORE_TIME:-24:00:00}")"
nuclei_s="$(seconds "${REGBENCH_NUCLEI_TIME:-08:00:00}")"
# shellcheck disable=SC2206  # the site's GPU request is several words on purpose
GPU_ARGS=(${REGBENCH_SBATCH_GPU:-})

if [ "$DRY" = 1 ]; then
    scratch="$(mktemp -d)"
    scratch_is_temporary=1
else
    mkdir -p "$REGBENCH_OUT/logs/gates" "$REGBENCH_OUT/logs/selfcheck"
    scratch="$REGBENCH_OUT/logs"
    JOBS_FILE="$REGBENCH_OUT/logs/jobs.tsv"
    mkdir -p "$REGBENCH_OUT/logs/submitted"
    # every job id of THIS call, one per line (the tile submitters add theirs)
    IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.ids"
    # the jobs of --prepare are not a submission's: their ids have a file of their own
    [ "$prepare" = 1 ] && IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.prepare.ids"
    export REGBENCH_IDS_FILE="$IDS_FILE"
    # A second submission while jobs of an earlier one are still in the queue: their
    # tasks read the gate records and the stage files this call would replace. And a
    # scoring: it would score, and settle, results whose tasks are still running.
    if [ "$prepare" != 1 ] && [ "$preflight_only" != 1 ] && [ -s "$JOBS_FILE" ] \
        && ! { [ "$score_only" = 1 ] && [ "${REGBENCH_SCORE_WHILE_QUEUED:-0}" = 1 ]; }; then
        in_queue="$(squeue -h -u "$(id -un)" -o '%F %A' 2>/dev/null)" \
            || die "$REGBENCH_OUT already holds a submission ($JOBS_FILE) and the queue cannot be read (squeue), so it cannot be shown that its jobs have ended. Nothing was submitted."
        still="$(tr ' ' '\n' <<< "$in_queue" | grep -Fxf <(cut -f3 "$JOBS_FILE" | cut -d_ -f1 | sort -u) | sort -u | paste -sd' ' - || true)"
        if [ -n "$still" ] && [ "$score_only" = 1 ]; then
            die "--score-only: $(wc -w <<< "$still" | tr -d ' ') job(s) of an earlier call into $REGBENCH_OUT are still pending or running ($still). Scoring now would score what is there so far and settle the results of tasks that have not ended, so it is refused. Wait for them; or cancel them:  scancel $still ; or score what is there all the same with REGBENCH_SCORE_WHILE_QUEUED=1. Nothing was submitted."
        elif [ -n "$still" ]; then
            die "$(wc -w <<< "$still" | tr -d ' ') job(s) of an earlier submission into $REGBENCH_OUT are still pending or running ($still). A second submission would replace the gate records and stage files their tasks read, so it is refused. Wait for them; or cancel them:  scancel $still ; or submit into another REGBENCH_OUT. Nothing was submitted."
        fi
    fi
    # An id is one submission's: the stage files, the tile submitters' lists and the job
    # ids of an earlier call under the same id would be taken for this call's.
    if [ "$prepare" != 1 ] && [ "$score_only" != 1 ] && [ "$preflight_only" != 1 ] && [ -e "$IDS_FILE" ]; then
        die "the submission id '$REGBENCH_SUBMISSION' was already used for a submission into $REGBENCH_OUT ($IDS_FILE). A second one under the same id is refused: its tile submitters would find the first one's lists of what is due and what was fed ($REGBENCH_OUT/logs/stages/$REGBENCH_SUBMISSION/) and feed nothing. Unset REGBENCH_SUBMISSION (every call then gets an id of its own) or export a new one; every unit that is ok is reused whatever the id. Nothing was submitted."
    fi
fi

# ── the run list, from the harness alone ──────────────────────────────────────────────
table="$scratch/runs.tsv"
if [ -n "$EXTERNAL" ]; then
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --dataset "$EXTERNAL" > "$table"
else
    regbench_py "$REGBENCH_HEAD_IMAGE" runs > "$table"
fi
[ -s "$table" ] || die "the run list is empty (python -m benchmarks.regbench runs)"
# REGBENCH_RUN_ORDER: the same rows in the order they are to be submitted. Every loop
# below reads the table top to bottom, and a job only ever waits for a job submitted
# before it, so the harness never puts a run before the run whose results it reads.
# Unset, the cohort's default: ASHLAR first. An external set has other runs (no ASHLAR).
[ -n "$EXTERNAL" ] || : "${REGBENCH_RUN_ORDER=default}"
if [ -n "${REGBENCH_RUN_ORDER:-}" ]; then
    of=()
    [ -n "$EXTERNAL" ] && of=(--dataset "$EXTERNAL")
    regbench_py "$REGBENCH_HEAD_IMAGE" runs ${of[@]+"${of[@]}"} --order "$REGBENCH_RUN_ORDER" > "$table.ordered" \
        || die "REGBENCH_RUN_ORDER='$REGBENCH_RUN_ORDER' is not an order of the run list (see the line above). Nothing was submitted."
    mv "$table.ordered" "$table"
    echo "# run order: $(cut -f1 "$table" | paste -sd' ' -)"
    regbench_py "$REGBENCH_HEAD_IMAGE" runs ${of[@]+"${of[@]}"} --order "$REGBENCH_RUN_ORDER" --order-notes \
        | while IFS= read -r note; do echo "# run order: $note"; done
fi
if [ -n "$EXTERNAL" ]; then
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --dataset "$EXTERNAL" --excluded | \
        while IFS=$'\t' read -r name why; do echo "# $EXTERNAL: $name is not run ($why)"; done
fi
[ "$EXTERNAL" = anhir_tune ] && echo "# anhir_tune: the preprocessing candidates and no other run (the tuning side is the only data that choice may look at)"
REGBENCH_SCORER_IMAGE="$(regbench_py "$REGBENCH_HEAD_IMAGE" runs --scorer-image)"
export REGBENCH_SCORER_IMAGE

all_names="" selected=""
while IFS=$'\t' read -r name _; do all_names="$all_names $name"; done < "$table"
for r in $only_runs; do has "$all_names" "$r" || die "--runs: no run named '$r' (see: python -m benchmarks.regbench runs)"; done
for r in $all_names; do
    if [ -z "$only_runs" ] || has "$only_runs" "$r"; then selected="$selected $r"; fi
done
for r in ${REGBENCH_ALLOW_UNCHECKED:-}; do has "$all_names" "$r" || die "REGBENCH_ALLOW_UNCHECKED: no run named '$r'"; done
for r in ${REGBENCH_ALLOW_INCONSISTENT:-}; do has "$all_names" "$r" || die "REGBENCH_ALLOW_INCONSISTENT: no run named '$r'"; done
# --score-only: no run is selected, so nothing of steps 1 to 4 is submitted
if [ "$score_only" = 1 ]; then
    [ -z "$only_runs" ] || echo "# --score-only: --runs is not used (the scorer scores every run of the submission)"
    selected=""
fi
# the preprocessing row: said aloud while its preprocessing has not been selected
IFS=$'\t' read -r prep_status prep_candidate prep_run < <(regbench_py "$REGBENCH_HEAD_IMAGE" runs --prep-status) \
    || die "cannot read the frozen preprocessing (benchmarks/regbench/prep_frozen.json)"
if has "$selected" "$prep_run"; then
    if [ "$prep_status" = selected ]; then
        echo "# $prep_run: preprocessing '$prep_candidate', selected on ANHIR's tuning side (prep_frozen.json)"
    else
        echo "# WARNING $prep_run: its preprocessing ('$prep_candidate') is PROVISIONAL: the selection on ANHIR's tuning side has not been run. Its results will say so, and are run again once prep_frozen.json names the selected candidate (submit_cohort.sh --datasets anhir_tune, then prep-select --freeze)."
    fi
fi

# ── prepare, on its own: array sizes are read from what it writes ─────────────────────
if [ "$prepare" = 1 ]; then
    [ -n "$EXTERNAL" ] || [ -n "${REGBENCH_MANIFEST:-}" ] || die "--prepare needs REGBENCH_MANIFEST (the rounds.csv of python -m benchmarks.regbench.arms_manifest)"
    force=()
    [ "${REGBENCH_PREPARE_FORCE:-0}" = 1 ] && force=(--force)
    alloc=(--cpus-per-task="$CPUS" --mem="${REGBENCH_PREPARE_MEM:-64G}"
           --time="$(limit "$prep_s")")
    export REGBENCH_PREPARE_IMAGE="$REGBENCH_SCORER_IMAGE"
    patients=()
    [ -n "${REGBENCH_TRUTH_PATIENTS:-}" ] && patients=(--patients "$REGBENCH_TRUTH_PATIENTS")
    # shellcheck disable=SC2206  # extra `prepare multiplex` options, several words
    more=(${REGBENCH_PREPARE_ARGS:-})
    for ds in $datasets; do
        array=() log="$LOG_ONE" after=()
        case "$ds" in
            multiplex)
                what=(multiplex --manifest "$REGBENCH_MANIFEST" ${more[@]+"${more[@]}"})
                # kept beside the cases it made: the score job reads it from there for
                # attrition.csv, in a later call that no longer has REGBENCH_MANIFEST
                if [ "$DRY" != 1 ]; then
                    { mkdir -p "$REGBENCH_CASES" && { [ "$REGBENCH_MANIFEST" -ef "$REGBENCH_CASES/manifest.csv" ] || cp "$REGBENCH_MANIFEST" "$REGBENCH_CASES/manifest.csv"; }; } \
                        || die "--prepare: cannot copy the manifest $REGBENCH_MANIFEST to $REGBENCH_CASES/manifest.csv. Nothing was submitted."
                fi
                # the bare nuclei of every native slide first: what exists is checked,
                # StarDist runs again only for a slide that has none (nuclei.sbatch)
                if [ "${REGBENCH_NUCLEI_SKIP:-0}" != 1 ]; then
                    : "${REGBENCH_NUCLEI:=$REGBENCH_CASES/_nuclei}"
                    export REGBENCH_NUCLEI REGBENCH_MANIFEST
                    # the model and settings of the re-run, and where each is from: the
                    # environment, else the site config (conf/ieo.config, REGBENCH_SITE_CONFIG)
                    stardist="$(regbench_py "$REGBENCH_HEAD_IMAGE" nuclei --manifest "$REGBENCH_MANIFEST" \
                        --out "$REGBENCH_NUCLEI" --settings)" || die "--prepare: the bare nuclei need the StarDist model the arms ran (the pipeline's segmentation_model and segmentation_model_dir): export REGBENCH_STARDIST_MODEL and REGBENCH_STARDIST_MODEL_DIR, or point REGBENCH_SITE_CONFIG at the site config that sets them (default: conf/ieo.config of this checkout). REGBENCH_NUCLEI_SKIP=1 prepares without them: nuclei numbers are then absent wherever no label image exists. Nothing was submitted."
                    while IFS= read -r said; do echo "# StarDist re-run: $said"; done <<< "$stardist"
                    nuc_gpu=()
                    if [ "${REGBENCH_STARDIST_CPU:-0}" = 0 ]; then
                        [ -n "${REGBENCH_SBATCH_GPU:-}" ] || die "--prepare: the StarDist re-run of the bare nuclei asks for a GPU, as the arms did, and REGBENCH_SBATCH_GPU is not set. Export the site's request, or REGBENCH_STARDIST_CPU=1 to run it on CPUs. Nothing was submitted."
                        nuc_gpu=("${GPU_ARGS[@]}")
                    fi
                    if [ -z "$(_regbench_var SIF STARDIST)$(_regbench_var CONDA STARDIST)$(_regbench_var VENV STARDIST)" ]; then
                        if [ "$DRY" = 1 ] || [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" = 1 ]; then
                            echo "# not set: REGBENCH_SIF_STARDIST (the StarDist re-run would use the python on PATH)"
                        else
                            die "not set: REGBENCH_SIF_STARDIST (the pipeline's StarDist image, bolt3x/mirage-stardist:1.0.0), needed to segment the slides that have no nuclear mask. Nothing was submitted."
                        fi
                    fi
                    n_slides="$(regbench_py "$REGBENCH_HEAD_IMAGE" nuclei --manifest "$REGBENCH_MANIFEST" \
                        --out "$REGBENCH_NUCLEI" --count)" || die "cannot size the nuclei array from $REGBENCH_MANIFEST"
                    [ "${REGBENCH_NUCLEI_FORCE:-0}" = 1 ] && echo "# REGBENCH_NUCLEI_FORCE=1: every slide is checked and segmented again (--export=ALL carries it to the array)"
                    submit "rb-nuclei" "${COMMON[@]}" "$LOG_ARRAY" \
                        --cpus-per-task="${REGBENCH_NUCLEI_CPUS:-8}" --mem="${REGBENCH_NUCLEI_MEM:-128G}" \
                        --time="$(limit "$nuclei_s")" \
                        ${nuc_gpu[@]+"${nuc_gpu[@]}"} \
                        --array="0-$((n_slides - 1))%$REGBENCH_ARRAY_LIMIT" "$HERE/nuclei.sbatch"
                    # afterany: a slide whose re-run failed leaves its cases without
                    # nuclei, with the reason; they are still prepared on the outlines
                    after=(--dependency="afterany:$JOB_ID")
                    what+=(--nuclei-dir "$REGBENCH_NUCLEI")
                fi
                [ -n "${REGBENCH_NUCLEI_FALLBACK:-}" ] && what+=(--nuclei-fallback "$REGBENCH_NUCLEI_FALLBACK")
                ;;
            semisynth) what=(semisynth --truth --manifest "$REGBENCH_MANIFEST" ${patients[@]+"${patients[@]}"}) ;;
            synthetic) what=(synthetic --suite truth) ;;
            hyreco_a)
                [ -n "${REGBENCH_HYRECO_A:-}" ] || die "--prepare --datasets hyreco_a needs REGBENCH_HYRECO_A (the unpacked HyReCo/ folder, holding HE/ and PHH3/). Nothing was submitted."
                # shellcheck disable=SC2206  # the settings being tried, several words
                tried=(${REGBENCH_HYRECO_A_ARGS:-})
                what=(hyreco --subset a --data-root "$REGBENCH_HYRECO_A" ${tried[@]+"${tried[@]}"}) ;;
            hyreco_b)
                [ -n "${REGBENCH_HYRECO_B_HE:-}" ] && [ -n "${REGBENCH_HYRECO_B_PHH3:-}" ] || die "--prepare --datasets hyreco_b needs REGBENCH_HYRECO_B_HE and REGBENCH_HYRECO_B_PHH3 (the HE/ and PHH3/ folders of HyReCo-Additional). Nothing was submitted."
                # shellcheck disable=SC2206  # several words; a setting is refused by prepare itself
                given=(${REGBENCH_HYRECO_B_ARGS:-})
                what=(hyreco --subset b --ref-dir "$REGBENCH_HYRECO_B_HE" --mov-dir "$REGBENCH_HYRECO_B_PHH3"
                      --settings-from "${REGBENCH_HYRECO_TUNED:-$REGBENCH_CASES}" ${given[@]+"${given[@]}"}) ;;
            anhir|anhir_tune)
                [ -n "${REGBENCH_ANHIR:-}" ] || die "--prepare --datasets $ds needs REGBENCH_ANHIR (the folder holding dataset_medium.csv, images/ and landmarks/). Nothing was submitted."
                [ "$ds" = anhir_tune ] && echo "# anhir_tune is written by the prepare of anhir: one prepare, both datasets"
                # shellcheck disable=SC2206  # several words
                given=(${REGBENCH_ANHIR_ARGS:-})
                what=(anhir --source "$REGBENCH_ANHIR" ${given[@]+"${given[@]}"}) ;;
            *) die "--datasets: '$ds' is not one of multiplex, semisynth, synthetic, hyreco_a, hyreco_b, anhir, anhir_tune" ;;
        esac
        if [ -n "$EXTERNAL" ]; then
            # one pair per task: a HyReCo slide is about 21 gigapixels, and a proxy of
            # each image is written here
            case "$ds" in
                hyreco_*) count=(hyreco --subset "${ds#hyreco_}") ;;
                *) count=(anhir --source "$REGBENCH_ANHIR") ;;
            esac
            n_pairs="$(regbench_py "$REGBENCH_HEAD_IMAGE" prepare "${count[@]}" --cases "$REGBENCH_CASES" \
                --count)" || die "cannot size the $ds prepare array"
            array=(--array="0-$((n_pairs - 1))%$REGBENCH_ARRAY_LIMIT")
            log="$LOG_ARRAY"
        fi
        submit "rb-prepare-$ds" "${COMMON[@]}" "$log" "${alloc[@]}" ${array[@]+"${array[@]}"} \
            ${after[@]+"${after[@]}"} "$HERE/prepare.sbatch" "${what[@]}" ${force[@]+"${force[@]}"}
    done
    echo "prepare: $N_JOBS job(s). When they are done, run this script again without --prepare."
    FINISHED=1
    exit 0
fi

# ── refusals, before anything is submitted ────────────────────────────────────────────
case "$EXTERNAL" in anhir|anhir_tune)
    [ -n "${REGBENCH_ANHIR:-}" ] || die "--datasets $EXTERNAL needs REGBENCH_ANHIR for the runs too (the folder the cases were prepared from): its cases name the original images inside it, and it is bound into every image from that variable. Nothing was submitted." ;;
esac
gpu_runs="" images=""
while IFS=$'\t' read -r name _ _ _ image gpu _; do
    has "$selected" "$name" || continue
    [ "$gpu" = 1 ] && gpu_runs="$gpu_runs $name"
    has "$images" "$image" || images="$images $image"
done < "$table"
if [ -n "$gpu_runs" ] && [ -z "${REGBENCH_SBATCH_GPU:-}" ]; then
    die "these runs need a GPU and REGBENCH_SBATCH_GPU is empty:$gpu_runs. Export the site's request (e.g. REGBENCH_SBATCH_GPU=\"--gres=gpu:nvidia_h200:1\"), or leave them out with --runs. Nothing was submitted."
fi
missing=""
for image in $images $REGBENCH_SCORER_IMAGE; do
    [ -n "$(_regbench_var SIF "$image")$(_regbench_var CONDA "$image")$(_regbench_var VENV "$image")" ] \
        || has "$missing" "REGBENCH_SIF_$image" || missing="$missing REGBENCH_SIF_$image"
done
if [ -n "$missing" ]; then
    if [ "$DRY" = 1 ] || [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" = 1 ]; then
        echo "# not set:$missing (those jobs would use the python on PATH)"
    else
        die "not set:$missing. Each image of the run list needs its path (or REGBENCH_ALLOW_PATH_PYTHON=1 to use the python on PATH). Nothing was submitted."
    fi
fi

# ── the cases: every array is sized from what is prepared ─────────────────────────────
# A synthetic case prepared from another truth_spec.json is refused by the scorer, which
# then writes no table for any dataset: refused here, by the same rule, before a job runs
if has "$datasets" synthetic; then
    regbench_py "$REGBENCH_HEAD_IMAGE" list --cases "$REGBENCH_CASES" --dataset synthetic --current-spec > /dev/null \
        || die "the synthetic cases under $REGBENCH_CASES are not all of the current truth_spec.json (line above): the scorer would refuse the whole result at the end. Nothing was submitted."
fi
sized=""
for ds in $datasets; do
    read -r n_cases n_groups max_rounds < <(regbench_py "$REGBENCH_HEAD_IMAGE" list \
        --cases "$REGBENCH_CASES" --dataset "$ds" --sizes) || die "cannot list the $ds cases under $REGBENCH_CASES"
    if [ "${n_cases:-0}" -eq 0 ]; then
        echo "# $ds: no prepared cases under $REGBENCH_CASES, left out (submit_cohort.sh --prepare)"
        continue
    fi
    sized="$sized $ds"
    printf -v "CASES_$ds" '%s' "$n_cases"
    printf -v "GROUPS_$ds" '%s' "$n_groups"
    printf -v "ROUNDS_$ds" '%s' "$max_rounds"
    echo "# $ds: $n_cases case(s), $n_groups patient(s), at most $max_rounds round(s) per patient"
done
[ -n "$sized" ] || die "no prepared cases under $REGBENCH_CASES for any of: $datasets. Run with --prepare first. Nothing was submitted."
# the cohort first, whatever the order they were named in
ordered=""
has "$sized" multiplex && ordered="multiplex"
for ds in $sized; do [ "$ds" = multiplex ] || ordered="$ordered $ds"; done
sized="$ordered"
first_dataset="${sized# }"; first_dataset="${first_dataset%% *}"

# ── the preflight: every image, started as a job starts it, before the first sbatch ───
# The pinned STARE source, for the images of this call that have none inside (and are
# images: an environment or the python on PATH has its own). Not part of the preflight:
# the jobs import it whether or not the preflight runs.
fetch_stare_source() {
    local image src
    for image in $images; do
        has "$REGBENCH_STARE_SRC_IMAGES" "$image" || continue
        [ -n "$(_regbench_var SIF "$image")" ] || continue
        mkdir -p "$REGBENCH_OUT" || die "cannot create REGBENCH_OUT=$REGBENCH_OUT. Nothing was submitted."
        src="$(regbench_fetch_stare)" || die "the image $image has no STARE package and the pinned STARE source could not be fetched (line above; REGBENCH_STARE_SRC names a copy that is already there). Its jobs would fail at their first import. Nothing was submitted."
        echo "# STARE source for the images without one ($REGBENCH_STARE_SRC_IMAGES) -> $src"
        break
    done
}
preflight() {
    local seen="" failed=0 name method image out code key where=() slides=() ds
    if [ "${REGBENCH_PREFLIGHT:-1}" = 0 ]; then
        echo "# WARNING preflight SKIPPED (REGBENCH_PREFLIGHT=0): nothing checked that the images hold what their runs import"
        return 0
    fi
    mkdir -p "$REGBENCH_OUT" || die "cannot create REGBENCH_OUT=$REGBENCH_OUT. Nothing was submitted."
    where=(--writable "$REGBENCH_OUT")
    if [ -n "${REGBENCH_WORK:-}" ]; then
        where+=(--writable "$REGBENCH_WORK")
    else
        echo "# preflight: REGBENCH_WORK is not set: a job's work folder is its own TMPDIR, which cannot be checked from here"
    fi
    # one prepared case of each dataset: its slides must be readable inside the image
    slides=(--cases "$REGBENCH_CASES")
    for ds in $sized; do slides+=(--dataset "$ds"); done
    while IFS=$'\t' read -r name method _ _ image _; do
        has "$selected" "$name" || continue
        key="$image:$method"
        has "$seen" "$key" && continue
        seen="$seen $key"
        out="$(regbench_py "$image" preflight --method "$method" --image "$image" "${where[@]}" "${slides[@]}" 2>&1)" && code=0 || code=$?
        grep -E '^preflight( FAIL)? \[' <<< "$out" | grep -E 'FAIL| (ok|[0-9]+ failure\(s\))$' || true
        if [ "$code" != 0 ]; then
            failed=1
            grep -q '^preflight FAIL \[' <<< "$out" \
                || echo "preflight FAIL [$image $method] the image did not start, or has neither python nor python3, or the harness does not import there (exit $code): $(tail -n 3 <<< "$out" | tr '\n' ' ')"
        fi
    done < "$table"
    image="$REGBENCH_SCORER_IMAGE"
    out="$(regbench_py "$image" preflight --scorer --image "$image" --writable "$REGBENCH_OUT" 2>&1)" && code=0 || code=$?
    grep -E '^preflight( FAIL)? \[' <<< "$out" | grep -E 'FAIL| (ok|[0-9]+ failure\(s\))$' || true
    if [ "$code" != 0 ]; then
        failed=1
        grep -q '^preflight FAIL \[' <<< "$out" \
            || echo "preflight FAIL [$image the scorer] the image did not start, or has neither python nor python3 (exit $code): $(tail -n 3 <<< "$out" | tr '\n' ' ')"
    fi
    [ "$failed" = 0 ] || die "the preflight failed (the 'preflight FAIL' lines above, one per failure). Nothing was submitted."
    echo "# preflight: ok"
}
if [ "$preflight_only" = 1 ]; then
    fetch_stare_source
    preflight
    echo "# --preflight: nothing was submitted"
    exit 0
fi
if [ "$DRY" = 1 ]; then
    echo "# preflight: not run in a dry run (submit_cohort.sh --preflight runs it alone; a real submission runs it first)"
    echo "# STARE source: not fetched in a dry run (a real submission fetches it before its first sbatch)"
else
    fetch_stare_source
    preflight
fi

# ── everything else that can stop this script, read before the first sbatch ───────────
# the pipeline's allocation per stage and attempt, for the runs that are several jobs
# per round: ALLOC_<stage>_<attempt>="<cpus> <GB> <time>"
any_staged=0
STITCH_S=0
while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ realcheck stages; do
    has "$selected" "$name" || continue
    [ "${stages:--}" = - ] && continue
    any_staged=1
    [ "${realcheck:--}" = - ] || die "$name runs as several jobs per round and has a realcheck gate: the launcher sends no first unit ahead for such a run. Nothing was submitted."
    for stage in ${stages//,/ }; do
        case "$stage" in prep|coarse|tile|solve|stitch) ;; *) die "$name: no job is defined for the stage '$stage' (column stages of the run list). Nothing was submitted." ;; esac
    done
done < "$table"
if [ "$any_staged" = 1 ]; then
    alloc_rows="$(regbench_py "$REGBENCH_HEAD_IMAGE" runs --stage-allocation)" \
        || die "cannot read the pipeline's allocation per stage (python -m benchmarks.regbench runs --stage-allocation). Nothing was submitted."
    while IFS=$'\t' read -r stage attempt cpus gb time; do
        [ -n "$stage" ] || continue
        printf -v "ALLOC_${stage}_$attempt" '%s %s %s' "$cpus" "$gb" "$time"
    done <<< "$alloc_rows"
    for stage in coarse tile solve stitch; do
        for attempt in 1 2 3 4; do
            row="ALLOC_${stage}_$attempt"
            [ -n "${!row:-}" ] || die "the pipeline's allocation has no line for $stage, attempt $attempt (python -m benchmarks.regbench runs --stage-allocation printed: $(tr '\n\t' '; ' <<< "$alloc_rows")). Nothing was submitted."
            # every time of the table is a time, and the writer's four are summed here
            read -r _ _ stage_time <<< "${!row}"
            stage_s="$(seconds "$stage_time")"
            [ "$stage" != stitch ] || STITCH_S=$((STITCH_S + stage_s))
        done
    done
fi
# the tile layout of every such run on every dataset: one line per round, where its
# tiles start in the arrays. Kept per submission: the tile tasks read it, and a later
# submission must not move it under them. Sets LAYOUT_<ds>_<run>, ROUNDS_OF_<ds>_<run>,
# TILES_OF_<ds>_<run>.
N_TILE_RUNS=0 TILE_RUNS=""
for ds in $sized; do
    while IFS=$'\t' read -r name method label opts image _ _ _ _ _ _ _ _ _ _ stages; do
        has "$selected" "$name" || continue
        [ "${stages:--}" = - ] && continue
        # a run's datasets follow one another, so it is ONE submitter's worth of queue
        case ",$stages," in *,tile,*) has "$TILE_RUNS" "$name" || { TILE_RUNS="$TILE_RUNS $name"; N_TILE_RUNS=$((N_TILE_RUNS + 1)); } ;; esac
        layout="$REGBENCH_OUT/logs/stages/$REGBENCH_SUBMISSION/$name-$ds.tsv"
        made="$layout"
        if [ "$DRY" = 1 ]; then
            made="$scratch/stages/$name-$ds.tsv"
            layout="$REGBENCH_OUT/logs/stages/<submission>/$name-$ds.tsv"
        fi
        what=(stages --cases "$REGBENCH_CASES" --dataset "$ds" --method "$method" --layout "$made")
        [ "$label" != - ] && what+=(--label "$label")
        if [ "$opts" != - ]; then
            IFS=, read -ra items <<< "$opts"
            for o in "${items[@]}"; do what+=(--opt "$o"); done
        fi
        read -r n_rounds n_tiles < <(regbench_py "$image" "${what[@]}") \
            || die "cannot size the tile arrays of $name on $ds (python -m benchmarks.regbench stages). Nothing was submitted."
        case "${n_rounds:-}:${n_tiles:-}" in *[!0-9:]*|:*|*:) die "cannot size the tile arrays of $name on $ds (python -m benchmarks.regbench stages printed '${n_rounds:-} ${n_tiles:-}'). Nothing was submitted." ;; esac
        [ "$n_tiles" -gt 0 ] && [ "$n_rounds" -gt 0 ] || die "$name on $ds: no tile to register (python -m benchmarks.regbench stages printed '$n_rounds $n_tiles'). Nothing was submitted."
        if [ "$DRY" != 1 ]; then
            # the caller's own sbatch arguments, for the jobs the submitter submits
            : > "${made%.tsv}.sbatch_args"
            [ ${#extra[@]} -gt 0 ] && printf '%s\n' "${extra[@]}" > "${made%.tsv}.sbatch_args"
        fi
        printf -v "LAYOUT_${ds}_$name" '%s' "$layout"
        printf -v "ROUNDS_OF_${ds}_$name" '%s' "$n_rounds"
        printf -v "TILES_OF_${ds}_$name" '%s' "$n_tiles"
    done < "$table"
done
# THE TILE QUEUE CEILING IS A TOTAL: REGBENCH_STAGE_QUEUE_MAX is divided by the number of
# runs with a tile submitter that can feed AT ONCE (floor, at least 1) and each submitter
# is given its share, REGBENCH_STAGE_QUEUE_RUN, which feed_tiles.sbatch reads. Set before
# the first sbatch. That number is N_TILE_RUNS, or REGBENCH_RUN_LANES when fewer: every
# job of a laned run (its feeders included) waits `afterany` for the LAST job of the run
# before it on its lane (LANE_LAST_<lane>, below), and that last job (stitch) waits for
# the run's last feeder, which ends only once each of its tile tasks has left the queue
# (feed_tiles.sbatch drains before it writes `.done`). So at most `lanes` submitters feed
# at a time, and `lanes` shares stay within the total.
N_FEEDING="$N_TILE_RUNS"
if [ "$REGBENCH_RUN_LANES" -gt 0 ] && [ "$REGBENCH_RUN_LANES" -lt "$N_FEEDING" ]; then
    N_FEEDING="$REGBENCH_RUN_LANES"
fi
STAGE_QUEUE_RUN="$REGBENCH_STAGE_QUEUE_MAX"
if [ "$N_FEEDING" -gt 0 ]; then
    STAGE_QUEUE_RUN=$((REGBENCH_STAGE_QUEUE_MAX / N_FEEDING))
    [ "$STAGE_QUEUE_RUN" -ge 1 ] || STAGE_QUEUE_RUN=1
    echo "# tile queue: REGBENCH_STAGE_QUEUE_MAX=$REGBENCH_STAGE_QUEUE_MAX split over $N_FEEDING tile submitter(s) feeding at once ($N_TILE_RUNS run(s) with one, REGBENCH_RUN_LANES=$REGBENCH_RUN_LANES): $STAGE_QUEUE_RUN each"
fi
export REGBENCH_STAGE_QUEUE_RUN="$STAGE_QUEUE_RUN"
# ── 1. selfchecks: the runs that gate a selected run, and the runs they are anchored on
needed=""
while IFS=$'\t' read -r name _ _ _ _ _ _ after gated_by _; do
    has "$selected" "$name" || continue
    has "$needed" "$gated_by" || needed="$needed $gated_by"
done < "$table"
# a gating run that reads another's results needs that one's selfcheck too
while IFS=$'\t' read -r name _ _ _ _ _ _ after _; do
    has "$needed" "$name" || continue
    [ "$after" = - ] || has "$needed" "$after" || needed="$needed $after"
done < "$table"

# RETRIES ARE SUBMITTED AFTER A FAILURE, NEVER BEFORE IT. Of a chain (a memory ladder, the
# attempts of a stage) only the FIRST array is submitted here, with one small job after
# it, the chain's watcher (retry.sbatch). The other steps are written down as a plan
# (plan_step: $REGBENCH_OUT/logs/retry/<submission>/<job prefix>/step-<k>), and the
# watcher submits a step only for the units whose last task failed in a way that step
# retries, then waits for it. A chain whose units are all ok puts nothing more in the
# queue. Everything that waited for a chain's last array waits for its watcher.
N_LATER=0 PLAN="" PLAN_SHOWN="" PLAN_N=0
plan_open() {  # plan_open <job prefix>
    PLAN_N=0
    PLAN_SHOWN="$REGBENCH_OUT/logs/retry/$REGBENCH_SUBMISSION/$1"
    PLAN="$PLAN_SHOWN"
    if [ "$DRY" = 1 ]; then
        PLAN="$scratch/retry/$1"
        PLAN_SHOWN="$REGBENCH_OUT/logs/retry/<submission>/$1"
    fi
    # the folder above it was made, and shown to be writable, before the first sbatch
    rm -rf "$PLAN"
    mkdir -p "$PLAN"
}
plan_step() {  # plan_step <again|climb|stage> <job name> <sbatch arguments...>
    local kind="$1" jobname="$2"; shift 2
    PLAN_N=$((PLAN_N + 1))
    N_LATER=$((N_LATER + 1))
    printf '%s\n' "$kind" "$jobname" "$@" > "$PLAN/step-$PLAN_N"
    if [ "$DRY" = 1 ]; then
        printf '# later, only after a failure: sbatch --parsable --job-name=%s' "$jobname"
        printf ' %s' "$@"
        printf '\n'
    fi
}
# plan_watch <job prefix> <first job id> <first job name> <array range|-> <dependencies
# of the first job> [<run whose gate it obeys|->]: the chain's watcher; sets JOB_ID. It
# has the first job's own dependencies too, so a chain a gate stops has no watcher
# either. A range of `-` is a chain of single jobs (a selfcheck), and a run of `-` a
# chain that runs BEFORE its run's gate.
plan_watch() {
    local dep="afterany:$2"
    [ -z "$5" ] || dep="$5,$dep"
    submit "$1-retry" "${COMMON[@]}" "$LOG_ONE" --dependency="$dep" \
        --cpus-per-task=1 --mem=1G --time="$(limit "$MAX_SECONDS")" \
        "$HERE/retry.sbatch" "${6:-$name}" "$PLAN_SHOWN" "$2" "$3" "$4"
}

# The chains' retry plans (plan_open, above) are written while the jobs are submitted.
# Their folder is made HERE, where one that cannot be written still stops the launch
# with nothing in the queue.
retry_root="$REGBENCH_OUT/logs/retry/$REGBENCH_SUBMISSION"
[ "$DRY" = 1 ] && retry_root="$scratch/retry"
mkdir -p "$retry_root" 2>/dev/null && [ -w "$retry_root" ] \
    || die "cannot write the retry plans to $retry_root. Nothing was submitted."
# read here, before the first sbatch, like every other variable of the caller
ADVISORY="${REGBENCH_SELFCHECK_ADVISORY:-0}"
sc_time="$(limit "$sc_s")"
while IFS=$'\t' read -r name method label opts image gpu _ after _ mask _; do
    has "$needed" "$name" || continue
    gres=()
    [ "$gpu" = 1 ] && gres=(${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
    base="" anchor=-
    if [ "$after" != - ]; then
        anchor="$after"
        last="SC_LAST_$after"
        base="afterok:${!last}"
    fi
    # the first rung is submitted; the others are steps of the selfcheck's watcher,
    # submitted only when the attempt before did not end ok (and such a job still runs
    # the selfcheck only when that attempt ended for memory). The watcher ends ok only
    # when the selfcheck did: a run anchored on this one waits for it with `afterok`.
    attempt=0 first_id="" first_name=""
    plan_open "rb-sc-$name"
    for gb in $SELFCHECK_RUNGS_GB; do
        attempt=$((attempt + 1))
        if [ "$attempt" = 1 ]; then
            deps=()
            [ -n "$base" ] && deps=(--dependency="$base")
            submit "rb-sc-$name-${gb}g" "${COMMON[@]}" "$LOG_ONE" ${deps[@]+"${deps[@]}"} \
                --cpus-per-task="$CPUS" --mem="${gb}G" --time="$sc_time" ${gres[@]+"${gres[@]}"} \
                "$HERE/selfcheck.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                "$mask" "$anchor" "$attempt" "$gb" -
            first_id="$JOB_ID" first_name="rb-sc-$name-${gb}g"
            continue
        fi
        plan_step notok "rb-sc-$name-${gb}g" "${COMMON[@]}" "$LOG_ONE" \
            --cpus-per-task="$CPUS" --mem="${gb}G" --time="$sc_time" ${gres[@]+"${gres[@]}"} \
            "$HERE/selfcheck.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
            "$mask" "$anchor" "$attempt" "$gb" "@PREVIOUS@"
    done
    plan_watch "rb-sc-$name" "$first_id" "$first_name" - "$base" -
    printf -v "SC_LAST_$name" '%s' "$JOB_ID"
done < "$table"

# ── 2. one gate per run ───────────────────────────────────────────────────────────────
small=(--cpus-per-task=1 --mem=4G --time=0-01:00:00)
while IFS=$'\t' read -r name _ _ _ _ _ _ _ gated_by _; do
    has "$selected" "$name" || continue
    allowed=0
    has "${REGBENCH_ALLOW_UNCHECKED:-}" "$name" && allowed=1
    has "${REGBENCH_ALLOW_INCONSISTENT:-}" "$name" && allowed=2
    [ "$ADVISORY" = 1 ] && allowed=3
    # an earlier submission's verdict must not be what a task of this one finds
    [ "$DRY" = 1 ] || rm -f "$REGBENCH_OUT/logs/gates/$name.tsv" "$REGBENCH_OUT/logs/gates/$name.realcheck.tsv"
    last="SC_LAST_$gated_by"
    submit "rb-gate-$name" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:${!last}" "${small[@]}" \
        "$HERE/gate.sbatch" selfcheck "$name" "$gated_by" "$allowed"
    printf -v "GATE_$name" '%s' "$JOB_ID"
done < "$table"

# ── 3 and 4. the ladders ──────────────────────────────────────────────────────────────
# ladder <job prefix> <array range> <realcheck 0|1> <dependencies> <time>: the first
# rung's array, then the plan of the others (per rung the attempt, then the same rung
# again for a unit the scheduler lost) and the watcher; sets LADDER_LAST to the watcher.
# Reads the current row's variables. Every later task is given the ids of all the
# ladder's earlier jobs: it asks the scheduler what became of its unit in each.
# With a sixth argument `jobs` (a run whose registration is several jobs per round) this
# is each round's LAST job: it holds the benchmark's own work (the points, the tissue
# reduction), so it has the benchmark's allocation, the same ladder, and inside it the
# method's writer runs as a process of its own with the pipeline's time limit and
# retries (methods/stare.py: write_native). Its time is one round's plus the writer's
# four attempts.
ladder() {
    local prefix="$1" range="$2" real="$3" base="$4" time="$5" staged="${6:--}"
    local attempt=0 gb task jobname cpus="$CPUS" rungs="$RUNGS_GB" first_id="" first_name=""
    # STITCH_S: the writer's four attempts, summed before the first sbatch
    [ "$staged" = - ] || time="$(limit $((round_s + STITCH_S)))"
    plan_open "$prefix"
    for gb in $rungs; do
        attempt=$((attempt + 1))
        for task in attempt again; do
            jobname="$prefix-${gb}g"
            if [ "$task" = again ]; then
                jobname="$jobname-again"
            elif [ "$attempt" = 1 ]; then
                task=first
            else
                task=climb
            fi
            if [ "$task" = first ]; then
                # the one array of the chain that is submitted here
                submit "$jobname" "${COMMON[@]}" "$LOG_ARRAY" --dependency="$base" \
                    --array="$range%$REGBENCH_ARRAY_LIMIT" --cpus-per-task="$cpus" --mem="${gb}G" \
                    --time="$time" ${gres[@]+"${gres[@]}"} \
                    "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                    "$ds" "$attempt" "$gb" "$real" - "$task" "$time" "$staged"
                first_id="$JOB_ID" first_name="$jobname"
                continue
            fi
            # every other one is a step of the plan: submitted by the chain's watcher,
            # after a failure, for the units that failed
            plan_step "$task" "$jobname" "${COMMON[@]}" "$LOG_ARRAY" \
                --cpus-per-task="$cpus" --mem="${gb}G" \
                --time="$time" ${gres[@]+"${gres[@]}"} \
                "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                "$ds" "$attempt" "$gb" "$real" "@EARLIER@" "$task" "$time" "$staged"
        done
    done
    plan_watch "$prefix" "$first_id" "$first_name" "$range" "$base"
    LADDER_LAST="$JOB_ID"
}

# staged <dependencies>: the jobs of a run whose registration is several scheduler jobs
# per round (column `stages`), each stage with the pipeline's allocation for that process
# (`regbench stages --allocation`, benchmarks/regbench/stages.py), every job after the one
# before it (`afterany`); sets LADDER_LAST to the last job of the last stage. Reads the
# current row's variables.
#   prep, coarse, solve   attempt 1 as an array over the rounds, and the stage's
#                         watcher (retry.sbatch): attempts 2 to 4 are submitted by it,
#                         only for the rounds whose attempt before was not ok, and
#                         such a task still exits at once unless that attempt ended
#                         the way the pipeline retries
#   tile                  NOT submitted here: REGBENCH_STAGE_FEEDERS tile submitters
#                         (feed_tiles.sbatch), one after the other. The first submits
#                         ONE TASK PER TILE (the count of a round is known from its
#                         reference's size alone, `regbench stages`) in arrays of at
#                         most REGBENCH_STAGE_ARRAY_MAX tasks, keeping at most
#                         its share of REGBENCH_STAGE_QUEUE_MAX of them pending or running, then
#                         per attempt 2 to 4 one task per round that has a tile to
#                         retry; a later one goes on where the one before stopped and
#                         exits at once when nothing is left
#   stitch                the round's last job (run_rung.sbatch, `jobs`), on the
#                         benchmark's memory ladder: it collects the stage records, moves
#                         the benchmark's points, and runs the writer as its own process
stage_chain() {
    local stage="$1" range="$2" dep="$3" attempt alloc cpus gb time first_id="" first_name=""
    plan_open "rb-$name-$ds-$stage"
    for attempt in 1 2 3 4; do
        alloc="ALLOC_${stage}_$attempt"; read -r cpus gb time <<< "${!alloc}"
        if [ "$attempt" = 1 ]; then
            submit "rb-$name-$ds-$stage-a$attempt" "${COMMON[@]}" "$LOG_ARRAY" --dependency="$dep" \
                --array="$range%$REGBENCH_ARRAY_LIMIT" --cpus-per-task="$cpus" --mem="${gb}G" --time="$time" \
                "$HERE/stage.sbatch" "$name" "$method" "$label" "$opts" "$image" "$ds" "$stage" \
                "$attempt" "$gb" "$time"
            first_id="$JOB_ID" first_name="rb-$name-$ds-$stage-a$attempt"
            continue
        fi
        plan_step stage "rb-$name-$ds-$stage-a$attempt" "${COMMON[@]}" "$LOG_ARRAY" \
            --cpus-per-task="$cpus" --mem="${gb}G" --time="$time" \
            "$HERE/stage.sbatch" "$name" "$method" "$label" "$opts" "$image" "$ds" "$stage" \
            "$attempt" "$gb" "$time"
    done
    plan_watch "rb-$name-$ds-$stage" "$first_id" "$first_name" "$range" "$dep"
}
staged() {
    local base="$1" layout n_rounds n_tiles stage dep alloc cpus gb time
    local attempt per_attempt link var
    # sized before the first sbatch (above): the layout file is written, the counts known
    var="LAYOUT_${ds}_$name"; layout="${!var}"
    var="ROUNDS_OF_${ds}_$name"; n_rounds="${!var}"
    var="TILES_OF_${ds}_$name"; n_tiles="${!var}"
    dep="$base"
    for stage in ${stages//,/ }; do
        case "$stage" in
            prep|coarse|solve)
                stage_chain "$stage" "0-$((n_rounds - 1))" "$dep"
                dep="$base,afterany:$JOB_ID" ;;
            tile)
                per_attempt=()
                for attempt in 1 2 3 4; do
                    alloc="ALLOC_tile_$attempt"; read -r cpus gb time <<< "${!alloc}"
                    per_attempt+=("$cpus:$gb:$time")
                done
                for link in $(seq 1 "$REGBENCH_STAGE_FEEDERS"); do
                    submit "rb-$name-$ds-feed-$link" "${COMMON[@]}" "$LOG_ONE" --dependency="$dep" \
                        --cpus-per-task=1 --mem=1G --time="$feed_time" "$HERE/feed_tiles.sbatch" \
                        "$name" "$method" "$label" "$opts" "$image" "$ds" "$layout" \
                        "$n_rounds" "$n_tiles" "$link" "${per_attempt[@]}"
                    dep="$base,afterany:$JOB_ID"
                done
                N_FED=$((N_FED + n_tiles))
                has "$FED_RUNS" "$name" || FED_RUNS="$FED_RUNS $name" ;;
            stitch)
                ladder "rb-$name-$ds-last" "0-$((n_rounds - 1))" 0 "$dep" - jobs ;;
        esac
    done
    echo "# $name on $ds: $n_rounds round(s) as separate jobs ($stages); $n_tiles tile job(s) fed in later by its tile submitter, in arrays of at most $REGBENCH_STAGE_ARRAY_MAX, at most $STAGE_QUEUE_RUN in the queue (its share of the total $REGBENCH_STAGE_QUEUE_MAX)"
}

N_FED=0 FED_RUNS=""
feed_time="$(limit "$feed_s")"
finals="" prev_ds="" n_laned=0
for ds in $sized; do
    n_cases="CASES_$ds"; n_groups="GROUPS_$ds"; rounds="ROUNDS_$ds"
    while IFS=$'\t' read -r name method label opts image gpu scope after _ _ _ _ _ _ realcheck stages; do
        has "$selected" "$name" || continue
        gres=()
        [ "$gpu" = 1 ] && gres=(${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
        if [ "$scope" = group ]; then
            n="${!n_groups}"; time="$(limit $((round_s * ${!rounds})))"
        else
            n="${!n_cases}"; time="$(limit "$round_s")"
        fi
        gate="GATE_$name"
        base="afterok:${!gate}"
        if [ "$after" != - ]; then
            last="LAST_${ds}_$after"
            if [ -n "${!last:-}" ]; then
                base="$base,afterany:${!last}"
            else
                echo "# $name reads the results of $after, which is not submitted here: they must already be under $REGBENCH_OUT"
            fi
        fi
        # one dataset after the other, per run: the cohort (multiplex) is done before
        # the same run starts on the next dataset
        if [ -n "$prev_ds" ]; then
            last="LAST_${prev_ds}_$name"
            [ -n "${!last:-}" ] && base="$base,afterany:${!last}"
        fi
        # at most REGBENCH_RUN_LANES runs at once: the selected runs are dealt onto that
        # many lanes in the order of the run list, and a run starts only when the run
        # before it on its lane has ended, whatever became of it (0: no such limit)
        if [ "$REGBENCH_RUN_LANES" -gt 0 ]; then
            lane="LANE_OF_$name"
            if [ -z "${!lane:-}" ]; then
                printf -v "$lane" '%s' $((n_laned % REGBENCH_RUN_LANES))
                n_laned=$((n_laned + 1))
            fi
            lane="LANE_LAST_${!lane}"
            [ -n "${!lane:-}" ] && base="$base,afterany:${!lane}"
        fi
        if [ "${stages:--}" != - ]; then
            staged "$base"
            printf -v "LAST_${ds}_$name" '%s' "$LADDER_LAST"
            finals="$finals $LADDER_LAST"
            if [ "$REGBENCH_RUN_LANES" -gt 0 ]; then
                lane="LANE_OF_$name"
                printf -v "LANE_LAST_${!lane}" '%s' "$LADDER_LAST"
            fi
            continue
        fi
        start=0
        if [ "${realcheck:--}" != - ]; then
            real="REAL_$name"
            if [ "$ds" = "$first_dataset" ]; then
                allowed=0
                has "${REGBENCH_ALLOW_UNCHECKED:-}" "$name" && allowed=1
                ladder "rb-$name-$ds-first" "0-0" 1 "$base" "$time"
                finals="$finals $LADDER_LAST"
                submit "rb-realgate-$name" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$LADDER_LAST" \
                    "${small[@]}" "$HERE/gate.sbatch" realcheck "$name" "$method" "$label" \
                    "$ds" "$realcheck" "$allowed"
                printf -v "$real" '%s' "$JOB_ID"
                start=1
            fi
            base="$base,afterok:${!real}"
        fi
        if [ "$n" -gt "$start" ]; then
            ladder "rb-$name-$ds" "$start-$((n - 1))" 0 "$base" "$time"
            printf -v "LAST_${ds}_$name" '%s' "$LADDER_LAST"
            finals="$finals $LADDER_LAST"
        elif [ "$start" = 1 ]; then
            real="REAL_$name"
            printf -v "LAST_${ds}_$name" '%s' "${!real}"
        fi
        last="LAST_${ds}_$name"
        if [ "$REGBENCH_RUN_LANES" -gt 0 ] && [ -n "${!last:-}" ]; then
            lane="LANE_OF_$name"
            printf -v "LANE_LAST_${!lane}" '%s' "${!last}"
        fi
    done < "$table"
    prev_ds="$ds"
done

# ── 5. per-case scoring and tissue, then the scorer, in the scorer's image ────────────
dep=""
for j in $finals; do dep="${dep:+$dep:}$j"; done
while IFS=$'\t' read -r name _; do
    has "$selected" "$name" || continue
    gate="GATE_$name"
    dep="${dep:+$dep:}${!gate}"
done < "$table"
# after everything this call submitted (--score-only submitted nothing before this)
after_all=()
[ -n "$dep" ] && after_all=(--dependency="afterany:$dep")
# The killed results are settled against the scheduler's accounting FIRST, in a job of
# its own that every scoring job waits for: it rewrites those results' run.json, and a
# case scored before that would leave a record of the result as it was (which the scorer
# does not read, so the case would be scored twice). afterany: whatever became of the
# registrations, and whatever becomes of this job, the scoring goes on.
submit "rb-settle" "${COMMON[@]}" "$LOG_ONE" ${after_all[@]+"${after_all[@]}"} "${small[@]}" \
    "$HERE/score_cohort.sbatch" settle
settle_job="$JOB_ID"
if [ -n "$EXTERNAL" ]; then
    # no tissue job (it is defined on nuclear channels) and no per-case array (a landmark
    # set has no pair records: its scorer reads landmarks); the set's own scorer and tables
    submit "rb-score-$EXTERNAL" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$settle_job" --cpus-per-task="$CPUS" \
        --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 "$HERE/score_cohort.sbatch" score "$EXTERNAL"
else
    # one task per case: its rows, kept as a record the scorer reads. afterany: a task
    # that fails or is killed leaves its case to the scorer, which computes what is missing
    score_time="$(limit "$score_s")"
    scored=""
    for ds in $sized; do
        n_cases="CASES_$ds"
        submit "rb-score-case-$ds" "${COMMON[@]}" "$LOG_ARRAY" --dependency="afterany:$settle_job" \
            --array="0-$((${!n_cases} - 1))%$REGBENCH_ARRAY_LIMIT" --cpus-per-task="$REGBENCH_SCORE_CPUS" \
            --mem="$REGBENCH_SCORE_MEM" --time="$score_time" "$HERE/score_cohort.sbatch" case "$ds"
        scored="$scored:$JOB_ID"
    done
    # each case task also made the tissue pieces of its case: this job makes those of a
    # case whose task did not, after them (never beside them: two jobs would be writing
    # the same pieces)
    submit "rb-tissue" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany${scored}" --cpus-per-task="$CPUS" \
        --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 "$HERE/score_cohort.sbatch" tissue
    submit "rb-score" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$JOB_ID$scored" --cpus-per-task="$CPUS" \
        --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 "$HERE/score_cohort.sbatch" score
fi

[ "$N_LATER" = 0 ] || echo "# retries: none is in the queue. Up to $N_LATER retry array(s) are written down as plans ($REGBENCH_OUT/logs/retry/) and submitted by the chains' watchers (retry.sbatch) only after a failure, for the units that failed"
echo "# $N_JOBS job(s)$([ "$DRY" = 1 ] && echo ' would be submitted; nothing was' || echo ' submitted'), $N_TASKS task(s) in them (an array counts once as a job and once per task): $N_JOBS sbatch call(s) $([ "$DRY" = 1 ] && echo 'would be made now' || echo 'were made now')"
if [ "$N_FED" -gt 0 ]; then
    n_fed_runs="$(wc -w <<< "$FED_RUNS" | tr -d ' ')"
    echo "# later, by the tile submitters (feed_tiles.sbatch): $N_FED tile task(s), one per tile, in arrays of at most $REGBENCH_STAGE_ARRAY_MAX; plus, per attempt 2 to 4, one retry task per round that has a tile to retry"
    if [ "$N_FEEDING" -lt "$n_fed_runs" ]; then
        echo "# the most in the queue at once: the $N_TASKS task(s) above plus at most $STAGE_QUEUE_RUN tile task(s) for each of the $N_FEEDING of the $n_fed_runs run(s) with a tile submitter that feed at once (${FED_RUNS# }; REGBENCH_RUN_LANES=$REGBENCH_RUN_LANES; REGBENCH_STAGE_QUEUE_MAX=$REGBENCH_STAGE_QUEUE_MAX is the total they share): $((N_TASKS + N_FEEDING * STAGE_QUEUE_RUN)) task(s)"
    else
        echo "# the most in the queue at once: the $N_TASKS task(s) above plus at most $STAGE_QUEUE_RUN tile task(s) for each of the $n_fed_runs run(s) with a tile submitter (${FED_RUNS# }; REGBENCH_STAGE_QUEUE_MAX=$REGBENCH_STAGE_QUEUE_MAX is the total they share): $((N_TASKS + n_fed_runs * STAGE_QUEUE_RUN)) task(s)"
    fi
fi
[ "$DRY" = 1 ] || echo "# job ids    -> $IDS_FILE (this call's, one per line; cancel everything: scancel \$(cat \"$IDS_FILE\"))"
if [ "$score_only" = 1 ]; then
    echo "# submission  -> $REGBENCH_SUBMISSION (scored as it is under $REGBENCH_OUT: nothing is run again)"
else
    echo "# submission  -> $REGBENCH_SUBMISSION (a unit already ok under $REGBENCH_OUT is reused$([ "$REGBENCH_FORCE" = 1 ] && echo ': NO, REGBENCH_FORCE=1 runs everything again'))"
fi
[ -n "$EXTERNAL" ] || echo "# case records -> $REGBENCH_OUT/cache/scores/<dataset>/<case>/ (kept; a valid one is read, never computed twice)"
echo "# gates       -> $REGBENCH_OUT/logs/gates/<run>.tsv (run, status, rows dropped, blocked)"
if [ -n "$EXTERNAL" ]; then
    echo "# results     -> $REGBENCH_OUT/tables/$EXTERNAL/summary.md (this set's own tables; none of the cohort's)"
    [ "$EXTERNAL" = hyreco_a ] && echo "# hyreco_a is the TUNING set: nothing in its tables is a result"
    [ "$EXTERNAL" = anhir_tune ] && echo "# anhir_tune is the TUNING side: nothing in its tables is a result. When it is done: python -m benchmarks.regbench prep-select --cases $REGBENCH_CASES --out $REGBENCH_OUT --freeze, commit benchmarks/regbench/prep_frozen.json, then submit_cohort.sh --datasets anhir"
    [ "$EXTERNAL" = anhir ] && echo "# anhir: the challenge's TRAINING pairs; no number is comparable with a leaderboard row (hidden evaluation pairs)"
else
    echo "# results     -> $REGBENCH_OUT/tables/summary.md: it names every blocked run and dropped row"
fi
[ -n "${REGBENCH_ALLOW_UNCHECKED:-}" ] && echo "# allowed to start unchecked: $REGBENCH_ALLOW_UNCHECKED"
[ "$ADVISORY" = 1 ] && echo "# the selfchecks are ADVISORY (REGBENCH_SELFCHECK_ADVISORY=1): no run is blocked by one; each verdict is in $REGBENCH_OUT/logs/gates/<run>.tsv and in the summary"
[ -n "${REGBENCH_ALLOW_INCONSISTENT:-}" ] && echo "# OVERRIDE: allowed to start although the selfcheck is inconsistent: $REGBENCH_ALLOW_INCONSISTENT (the gate record and the summary keep the verdict)"
FINISHED=1
exit 0
