#!/bin/bash
# Submit the cohort benchmark: every run of the run list on a five-rung memory ladder,
# beside the selfchecks (REPORTED, NEVER A GATE: no run waits for one), then the scorer.
# From a login node, in three steps:
#
#   benchmarks/regbench/slurm/submit_cohort.sh --inputs [-- extra sbatch args]
#   (wait for the input step: the pipeline's own preprocessing of the cohort,
#   slurm/inputs.sbatch, one Nextflow head job)
#   benchmarks/regbench/slurm/submit_cohort.sh --prepare [-- extra sbatch args]
#   (wait for the prepare jobs; look at their logs)
#   benchmarks/regbench/slurm/submit_cohort.sh [--dry-run] [--runs "a b"] \
#       [--datasets "multiplex"] [-- extra sbatch args]
#
# and, for a results folder whose registrations are done, the scoring alone:
#
#   benchmarks/regbench/slurm/submit_cohort.sh --score-only [--dry-run] \
#       [--datasets "multiplex"] [-- extra sbatch args]
#
# THE DEFAULT DATASET is `multiplex`, the cohort, and the default submission is the cohort
# alone. --datasets may also name a public dataset (regbench.PUBLIC: PUBLIC_DATASETS below),
# whose prepared cases are under REGBENCH_CASES_<NAME> (env.sh: regbench_cases_of); it is
# run with its own run list (`runs --dataset <name>`) and scored into tables of its own,
# never pooled with the cohort. Any other name is refused.
#
# THE SYNTHETIC SET (`synthetic`, T32: cost only, never pooled, never in --all) is made
# by this launcher: --prepare --datasets synthetic (REGBENCH_CASES_SYNTHETIC, and
# REGBENCH_SYNTH_SOURCE, a cohort slide whose nuclear channel is the texture) submits
# slurm/synth.sbatch's plan job, its array (one task per patient and panel) and its finish
# job; a complete set is reused, a planned one is continued. It is then launched by name
# (--datasets synthetic) and scored for cost alone: no per-case or tissue scoring.
#
# HyReCo (`hyreco`, the one landmark set): download/hyreco.sh fetches the FULL set by
# default, the core zip and both Additional zips (HYRECO_ADDITIONAL=0: the core zip
# alone), and `prepare hyreco` puts subset A and subset B into one cases folder
# (REGBENCH_CASES_HYRECO). This launcher reads neither the download nor its switch.
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
#   1. per run that checks a selected run (column `checked_by`): its selfcheck at 64 GB
#      and its watcher (retry.sbatch), which submits it at 256 and then 512 GB only when
#      the attempt before did not end ok. An attempt above the first runs only when the
#      one before ended for memory. A run whose results another reads (column `after`)
#      hands its selfcheck's work to that one (`--m0-from`), which waits for it with
#      `afterany`: whatever the anchor's selfcheck said. THE SELFCHECK IS REPORTED, NEVER
#      A GATE: no ladder waits for it, no run is kept from starting and no row is left
#      out because of it (user ruling: run, report, never gate). Its verdict is read by
#      the scorer (`regbench verdict`, step 5).
#   2. (there is no step 2 any more: the gate jobs are gone)
#   3. per run and dataset: the memory ladder, 16 -> 64 -> 256 -> 512 -> 700 GB at 4 CPUs. ONE
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
#   4. for a run with a realcheck (column `realcheck`): the first unit of the first
#      dataset runs with `--realcheck`, in a ladder of its own SUBMITTED BESIDE the rest
#      of the run, which does not wait for it. Reported, never a gate.
#   5. after everything, in the scorer's image: FIRST the settle job (`rb-settle`: every
#      result that still says "killed" gets the cause the scheduler's accounting names;
#      it rewrites those results' run.json, which is why every scoring job waits for
#      it); then per dataset ONE ARRAY with a task per case (the tissue-level pieces of
#      the case, then `score-case`: the case's rows, kept as its record under
#      $REGBENCH_OUT/cache/scores), then the tissue job (the pieces of any case whose
#      task did not make them), then the scorer, which first writes what every run's
#      selfcheck and realcheck said (`regbench verdict` -> logs/verdicts.tsv), then
#      reads the records, computes whatever case has none that is still valid, and
#      writes the tables (`score --verdicts`). The tables are the same whether the array
#      ran, half of it did, or none of it. Everything here waits (`afterany`) for every
#      ladder, and for NO selfcheck (Task 28, B5): the selfchecks never gate, nor do they
#      delay the tables. When there are selfchecks, `rb-rescore` (score_cohort.sbatch
#      rescore) waits for the scorer AND every selfcheck's watcher, settles again and
#      writes the tables again, with every verdict in them; the case records are read
#      (the selfcheck statuses are set on the rows, not keyed), so it computes nothing.
#
#   6. with --compute-all, after the scorer and the re-score: compute-all, the WHOLE pipeline traced per
#      patient and registration backend (compute_all.sbatch, one Nextflow head job per
#      run; `python -m benchmarks.regbench compute-all plan` lists them, every valis
#      patient then every tiled one). The heads are dealt onto REGBENCH_COMPUTE_LANES
#      lanes: each waits (afterany) for the head before it on its lane, and the first of
#      each lane for the scorer of step 5. So they run LAST and ALONE: a cost measured
#      under contention the benchmark made itself is not a cost.
#   7. then `rb-score-compute`, the scorer again (score_cohort.sbatch score) after every
#      head: the case records are kept, so it only adds pipeline_cost*.csv and their
#      block of summary.md.
#
# --compute-only submits steps 6 and 7 and nothing else, for a results folder whose
# benchmark is done: the heads start at once (on their lanes), and the scorer scores the
# LAST submission into REGBENCH_OUT, as --score-only does. Both need REGBENCH_INPUT.
#
# --score-only submits step 5 and nothing before it: no selfcheck and no
# registration. It scores what the LAST submission into REGBENCH_OUT ran or reused
# (its id is kept in $REGBENCH_OUT/logs/submission; REGBENCH_SUBMISSION names another),
# with the selfcheck records that submission left. Nothing is run again, and a case whose
# record is still valid is not computed again.
#
# Nothing waits on the login node and no job waits on `sbatch --wait`. One kind of job
# submits jobs: the tile submitter of a run whose registration is several jobs per round.
#
# A SECOND SUBMISSION into a REGBENCH_OUT whose earlier jobs are still pending or running
# is REFUSED (its stage files and retry plans would replace the ones those jobs read). Wait, cancel them
# (the refusal prints the line), or use another REGBENCH_OUT. --score-only is refused
# the same way (it would score, and settle, results whose tasks are still running):
# REGBENCH_SCORE_WHILE_QUEUED=1 scores anyway. --prepare is not refused.
#
# A RUN IS DROPPED, NEVER THE SUBMISSION (Task 28, B3/B4). A run whose image fails the
# preflight, whose GPU or image path is not set, or whose own configuration cannot be
# submitted (a staged run with a realcheck, a stage no job is defined for, tile arrays
# that cannot be sized) is left out, with a row of $REGBENCH_OUT/submit_dropped.csv
# (`run,reason,detail`; reason preflight_failed or run_config), and the others are
# submitted; the dropped ones are listed again at the end, and the scorer makes each a
# row of losses.csv. Only when no run is left does the launch stop. The scorer's own
# image failing its preflight still stops it: nothing could be scored.
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
#   REGBENCH_GPU_ARRAY_LIMIT (REGBENCH_ARRAY_LIMIT)  the same, for every array of a run
#                                     that asks for a GPU (column `gpu`): its first rung,
#                                     its watcher's retry arrays (the chain's plan names
#                                     its limit: <plan folder>/limit, read by
#                                     retry.sbatch) and its realcheck ladder; and the
#                                     nuclei array when StarDist runs on a GPU. Each task
#                                     of such an array holds one GPU, so this is the
#                                     GPUs one array can hold at once. Unset: the array
#                                     limit, as before (launch.sbatch sets 4)
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
#   REGBENCH_REPEAT (1)               the same submission N times, repeat k into
#                                     $REGBENCH_OUT/rep<k> (the CPU cross's repetitions,
#                                     README "The CPU cross"), one after the other (its
#                                     first jobs afterany the last of the one before);
#                                     1 is the launch as it was
#   REGBENCH_WAVES (0)                1: one method at a time, each method's first jobs
#                                     after every job of the method before it (afterany):
#                                     a timing replicate with no other method on the nodes
#   REGBENCH_RUN_LANES (0)            runs in flight at once (0: no limit); see the ladders
#   REGBENCH_RUN_ORDER                the order the runs are SUBMITTED in, and so dealt
#                                     onto the lanes: family or run names, space-separated
#                                     (`runs --order`); the runs it does not name follow
#                                     in the run list's order. A run that reads another's
#                                     results is never put before it (a `#` line says
#                                     when it was moved). Default: `default`, the run
#                                     list's own first runs (`runs --order default`
#                                     prints the order): ASHLAR on the cohort, STARE
#                                     first on a public set;
#                                     set EMPTY, the run list's order.
#                                     Nothing but the order of submission changes: not
#                                     the run list, not a table, not a result
#   REGBENCH_SCORE_CPUS (8)           per task of the per-case scoring arrays: a case's
#                                     pair records are computed on that many processes
#   REGBENCH_SCORE_MEM (64G)          per task of those arrays, and of the tissue and
#                                     score jobs (which keep 4 CPUs and one day)
#   REGBENCH_SCORE_TIME (24:00:00)    per task of those arrays. A task that runs out of
#                                     it keeps the pair records it finished: submit
#                                     again with --score-only and it goes on from there
#   REGBENCH_SETTLE_TIME (04:00:00)   the settle job's time limit
#   REGBENCH_RETRY_WATCHERS (2)       links of each chain's watcher (retry.sbatch), one
#                                     after the other (`afterany`), each with the site's
#                                     longest time: a link ended at its limit or lost
#                                     with its node leaves the next to go on from the
#                                     step it was at. What waits for a chain waits for
#                                     its LAST link
#   REGBENCH_RUNG_CHECK (1)           before the first sbatch (and with --preflight):
#                                     ask `sinfo` for the largest node of the partition
#                                     (and constraint) the jobs are given. A rung no
#                                     node can grant is said loudly and is NOT planned:
#                                     the units that would climb to it are written down
#                                     as the launcher's (not_submitted, ladder_cut_gb),
#                                     never left at the rung below as the method's
#                                     out-of-memory. 0: not asked
#   REGBENCH_FIRST_RUNG               "<run>=<GB> ...": the ladder of each named run starts
#                                     at that rung (one of the rungs, 16 64 256 512 700)
#                                     and never tries the ones below it; e.g.
#                                     "valis_mirage=256 valis120=256 valis120_hires=256".
#                                     A run not named starts at the first; one in no run
#                                     list of the launch is said and ignored. A GB that
#                                     is not a rung, or one no node can grant,
#                                     stops the launch before any sbatch. Neither this
#                                     file nor ladder.py is in a result's reuse key: what
#                                     is ok stays ok; the rung a result ran at is billed
#                                     as reserved memory (the GB*h columns)
#   REGBENCH_FORCE=1                  run every unit again, also those already ok
#   REGBENCH_SUBMISSION               this call's id (default: time and process id).
#                                     Records of another submission are evidence of
#                                     nothing: not for a verdict, the ladder or the scorer.
#                                     With --score-only: the submission to score. An id
#                                     already used for a submission into REGBENCH_OUT
#                                     is refused for another one
#   REGBENCH_SCORE_WHILE_QUEUED=1     --score-only although jobs of an earlier call
#                                     into REGBENCH_OUT are still pending or running
#   (the three switches that once let a run start past its selfcheck are gone with the
#   gate; a call that still exports one is told it is ignored)
#   REGBENCH_INPUT                    with --inputs, --compute-all, --compute-only: the
#                                     cohort's samplesheet (the pipeline's --input)
#   REGBENCH_COMPUTE_LANES (2)        with --compute-all/--compute-only: compute-all heads
#                                     in flight at once (each is a Nextflow head whose
#                                     processes are jobs of their own)
#   REGBENCH_COMPUTE_TIME (7-00:00:00)  the time limit of one compute-all head
#   REGBENCH_INPUTS                   the input folder: the input step writes
#                                     preprocess/ into it, --prepare writes rounds.csv
#                                     and outlines/ there (launch.sbatch: inputs/ of
#                                     REGBENCH_ROOT)
#   REGBENCH_INPUTS_TIME (2-00:00:00) with --inputs: the head job's time limit
#   REGBENCH_INPUTS_FORCE=1           with --inputs: run it again, the earlier one moved
#                                     to preprocess.replaced/ (never deleted)
#   REGBENCH_NF_PROFILE               with --inputs and compute-all: the pipeline's
#                                     -profile (default singularity,ieo, as the arms ran)
#   REGBENCH_MANIFEST                 with --prepare: the cohort's rounds.csv. Unset, it
#                                     is $REGBENCH_INPUTS/rounds.csv, written here from
#                                     the input step's preprocessed.csv
#                                     (python -m benchmarks.regbench.manifest). Set, the
#                                     file is taken as it is (one written from the arms'
#                                     results before 2026-10-16 is read as before)
#   REGBENCH_PREPARE_ARGS             with --prepare: more `prepare multiplex` options
#   REGBENCH_PREPARE_FORCE=1          with --prepare: redo cases that exist
#   REGBENCH_NUCLEI                   with --prepare: where each slide's BARE nuclei are
#                                     kept (default $REGBENCH_CASES/_nuclei). An array
#                                     over the manifest's slides runs before `prepare
#                                     multiplex`: StarDist segments each slide, and its
#                                     cell mask is traced to the slide's outlines
#   REGBENCH_STARDIST_MODEL, REGBENCH_STARDIST_MODEL_DIR
#                                     the site's StarDist model (the pipeline's
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
#   REGBENCH_STARDIST_MODEL_DIR_WAS   the folder the site's models were MOVED from: a
#                                     kept re-run recorded with that folder is compared
#                                     as if made in the one that resolves now (said in
#                                     the log). Any other folder is still a change
#   REGBENCH_NUCLEI_FALLBACK=shrink   for a slide still without nuclei, use its outline
#                                     shrunk by 10 px (labelled, never exact). Off unless set
#   REGBENCH_NUCLEI_CPUS (8), REGBENCH_NUCLEI_MEM (128G), REGBENCH_NUCLEI_TIME (08:00:00)
#   REGBENCH_HEAD_IMAGE               the image this script itself reads the run list
#                                     with (default STARE_REGBENCH)
#   REGBENCH_CPUS (4)                 every ladder task's --cpus-per-task (run_rung.sbatch,
#                                     recorded in run.json as cpus_requested): 4 or 8,
#                                     anything else is refused before any sbatch.
#                                     4 CPUs is every method's allocation; REGBENCH_CPUS
#                                     exists for the CPU cross only (user ruling
#                                     2026-10-16). The other jobs keep their own.
#   REGBENCH_STARE_EXECUTION (jobs)   `local`: every run whose registration is several
#                                     jobs per round (column `stages`: STARE) is one job
#                                     per case on the ordinary ladder instead, its stages
#                                     started inside that job (`run --stages local`,
#                                     option workers=REGBENCH_CPUS): no stage chain, no
#                                     tile submitter. Its results get the reuse key's
#                                     execution `local`, never one of the staged results'.
#                                     For the CPU cross only (README, "The CPU cross")
# The rungs, 4 CPUs and the 7-day limit are frozen (benchmarks/regbench/ladder.py);
# REGBENCH_CPUS moves the ladder's CPUs for the CPU cross alone.
# No thread variable is exported anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# who starts a run's stage processes is stated per run by its own job script; a value
# left in the caller's shell would reach every job (--export=ALL) and every image
unset REGBENCH_STAGES
# shellcheck source=benchmarks/regbench/slurm/env.sh
source "$HERE/env.sh"

RUNGS_GB="16 64 256 512 700"
# first_rung_of <run>: its REGBENCH_FIRST_RUNG, else the first rung; rungs_of <run>:
# the rungs its ladder climbs, from there up
first_rung_of() {
    local pair
    for pair in ${REGBENCH_FIRST_RUNG:-}; do
        [ "${pair%%=*}" = "$1" ] && { printf '%s\n' "${pair#*=}"; return 0; }
    done
    printf '%s\n' "${RUNGS_GB%% *}"
}
rungs_of() {
    local from gb out=""
    from="$(first_rung_of "$1")"
    for gb in $RUNGS_GB; do [ "$gb" -ge "$from" ] && out="$out $gb"; done
    printf '%s\n' "${out# }"
}
# the nuclei step's rungs above its first request (REGBENCH_NUCLEI_MEM, 128G): the
# pipeline's SEGMENT is 128 GB x task.attempt
NUCLEI_RUNGS_GB="256 512"
SELFCHECK_RUNGS_GB="64 256 512"
CPUS=4
# 4 CPUs is every method's allocation; REGBENCH_CPUS exists for the CPU cross only (user
# ruling 2026-10-16). It is every ladder task's --cpus-per-task and nothing else's:
# the selfchecks, the prepare, settle, tissue and score jobs keep CPUS. env.sh (sourced
# above) defaults it to 4; checked below, before anything is submitted.
LADDER_CPUS="${REGBENCH_CPUS:-4}"
# REGBENCH_STARE_EXECUTION: who starts the stage processes of a run whose registration is
# several jobs per round (column `stages`). `jobs` (the default): the scheduler, a chain
# of stage jobs. `local`: the run's own job on the ordinary ladder, one job per case,
# `--stages local` with the option workers=$LADDER_CPUS (the CPU cross).
STARE_EXECUTION="${REGBENCH_STARE_EXECUTION:-jobs}"
MAX_SECONDS=$((7 * 24 * 3600))
: "${REGBENCH_TIME:=24:00:00}"
: "${REGBENCH_SELFCHECK_TIME:=24:00:00}"
: "${REGBENCH_ARRAY_LIMIT:=20}"
: "${REGBENCH_GPU_ARRAY_LIMIT:=$REGBENCH_ARRAY_LIMIT}"
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
: "${REGBENCH_SETTLE_TIME:=04:00:00}"
: "${REGBENCH_RETRY_WATCHERS:=2}"
: "${REGBENCH_RUNG_CHECK:=1}"
: "${REGBENCH_FORCE:=0}"
export REGBENCH_STAGE_ARRAY_MAX REGBENCH_STAGE_FEED_MIN REGBENCH_STAGE_POLL_S
: "${REGBENCH_HEAD_IMAGE:=STARE_REGBENCH}"
DRY="${REGBENCH_DRY:-0}"
# The cohort (multiplex) is the default dataset; the public datasets are named on request
# (regbench.PUBLIC, in its order; test_submit_cohort.py holds the two equal).
DEFAULT_DATASETS="multiplex"
PUBLIC_DATASETS="hyreco mcmicro phenocycler rashid synthetic"
# each public dataset's kind (regbench.PUBLIC), and the kinds score() can score
# (score.SCORED_KINDS); a test keeps both equal to Python. Methods never run ahead of their
# scorer: a dataset of a kind not scored is refused up front (every kind is, since Task 21)
PUBLIC_KINDS="hyreco=landmarks mcmicro=nuclei phenocycler=nuclei rashid=nuclei synthetic=synthetic"
SCORED_KINDS="tissue nuclei landmarks synthetic"
# the public datasets scored for COST alone (cli.COST_ONLY_SETS; a test keeps them equal):
# named only (never in --cases-there's yes), prepared here (synth.sbatch), and given no
# per-case or tissue scoring job
COST_ONLY_SETS="synthetic"
# the public LANDMARK sets (kind landmarks; a test keeps them equal to regbench.PUBLIC):
# their scorer (regbench.landmarks) reads no per-case record and no tissue piece, so they
# are given no per-case scoring job, and no tissue job when they are all a call scores
LANDMARK_SETS="hyreco"
prepare=0 inputs=0 score_only=0 preflight_only=0 only_runs="" datasets="$DEFAULT_DATASETS"
compute_all=0 compute_only=0 cases_there=0
ORIG_ARGS=("$@")  # REGBENCH_REPEAT starts this script again with them
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --inputs) inputs=1; shift ;;
        --prepare) prepare=1; shift ;;
        --score-only) score_only=1; shift ;;
        --compute-all) compute_all=1; shift ;;
        --compute-only) compute_only=1; shift ;;
        --preflight) preflight_only=1; shift ;;
        --cases-there) cases_there=1; shift ;;
        --runs) only_runs="$2"; shift 2 ;;
        --datasets) datasets="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
[ "$DRY" = 0 ] || DRY=1
# the switches of the selfcheck gate, which no longer exists: said, and nothing else
for v in REGBENCH_SELFCHECK_ADVISORY REGBENCH_ALLOW_UNCHECKED REGBENCH_ALLOW_INCONSISTENT; do
    if [ -n "${!v:-}" ]; then echo "# ignored: $v (the selfcheck no longer gates)"; fi
done

die() { echo "submit_cohort: $*" >&2; exit 2; }
# the verdict lines of a preflight's output; none is a legitimate answer (the grep exits 1)
preflight_lines() {
    local lines
    lines="$(grep -E '^preflight( FAIL)? \[' <<< "$1" | grep -E 'FAIL| (ok|[0-9]+ failure\(s\))$')" || lines=""
    [ -z "$lines" ] || printf '%s\n' "$lines"
}
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
    # in the launch job (launch.sbatch): its `[regbench] end` line
    if [ -n "${REGBENCH_LAUNCH_T0:-}" ]; then
        echo "[regbench] end kind=launch job=${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:--} outcome=$([ "$code" = 0 ] && echo ok || echo "exit_$code") wall=$(( $(date +%s) - REGBENCH_LAUNCH_T0 ))s peak_rss=- record=${IDS_FILE:--} exit=$code queued=$N_QUEUED"
    fi
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
# real_file <path>: the file's path with every symlink resolved (no `realpath -m` on macOS)
real_file() {
    local p="$1" n=0 t
    while [ -L "$p" ] && [ "$n" -lt 40 ]; do
        t="$(readlink "$p")"
        case "$t" in /*) p="$t" ;; *) p="$(dirname "$p")/$t" ;; esac
        n=$((n + 1))
    done
    printf '%s/%s\n' "$(cd "$(dirname "$p")" && pwd -P)" "$(basename "$p")"
}
# drop_run <run> <reason> <detail>: the run is left out of this submission (removed from
# `selected`) and is a row of submit_dropped.csv (B3/B4); the others go on
DROPPED_ROWS="" DROPPED_RUNS=""
csv_field() { local v="${1//\"/\"\"}"; printf '"%s"' "${v//$'\n'/ }"; }
write_dropped() {
    [ "$DRY" = 1 ] && return 0
    [ -n "${REGBENCH_OUT:-}" ] || return 0
    mkdir -p "$REGBENCH_OUT" 2>/dev/null || return 0
    { echo "run,reason,detail"; printf '%s' "$DROPPED_ROWS"; } > "$REGBENCH_OUT/submit_dropped.csv.new" \
        && mv "$REGBENCH_OUT/submit_dropped.csv.new" "$REGBENCH_OUT/submit_dropped.csv"
}
drop_run() {
    local run="$1" reason="$2" detail="$3" r kept=""
    has "$selected" "$run" || return 0
    for r in $selected; do [ "$r" = "$run" ] || kept="$kept $r"; done
    selected="$kept"
    DROPPED_RUNS="$DROPPED_RUNS $run"
    DROPPED_ROWS="$DROPPED_ROWS$run,$reason,$(csv_field "$detail")"$'\n'
    echo "# DROPPED $run ($reason): $detail. The other runs are submitted (submit_dropped.csv)"
    write_dropped
}
# none_left <what dropped them>: no run is left to submit
SELECTED_ANY=0
none_left() {
    [ "$SELECTED_ANY" = 1 ] || return 0  # --score-only / --compute-only select none
    [ -n "${selected// /}" ] && return 0
    die "every selected run was dropped ($1:$DROPPED_RUNS; the lines above). Nothing was submitted."
}
# case_sizes <dataset> <cases folder>: "<cases> <patients> <most rounds>" of what is
# prepared there (`regbench list --sizes`). THE predicate of "the dataset has prepared
# cases" (n_cases > 0): the launch below sizes its arrays from it, and --cases-there
# (launch.sbatch --all) asks it before the launch is started
case_sizes() {
    regbench_py "$REGBENCH_HEAD_IMAGE" list --cases "$2" --dataset "$1" --sizes
}
# --cases-there: one line per dataset (the cohort, then every public one), tab-separated
# `<dataset> yes|no <why>`: whether a launch could take it now (launch.sbatch --all
# reads it to name the datasets of its launch stage). Submits nothing, reads nothing else
if [ "$cases_there" = 1 ]; then
    for ds in multiplex $PUBLIC_DATASETS; do
        why="" ds_cases=""
        if has "$COST_ONLY_SETS" "$ds"; then
            why="named only: cost alone, launched by --datasets $ds and never by --all"
        elif [ "$ds" != multiplex ]; then
            kind=""
            for k in $PUBLIC_KINDS; do [ "${k%%=*}" != "$ds" ] || kind="${k#*=}"; done
            if ! has "$SCORED_KINDS" "$kind"; then
                why="a $kind dataset, and score() scores ${SCORED_KINDS// / and } only"
            elif ! ds_cases="$(regbench_cases_of "$ds")"; then
                why="REGBENCH_CASES_$(tr '[:lower:]' '[:upper:]' <<< "$ds") is not set"
            elif [ ! -d "$ds_cases" ]; then
                why="its cases folder $ds_cases does not exist"
            fi
        else
            ds_cases="$(regbench_cases_of multiplex)"
        fi
        if [ -z "$why" ]; then
            n_cases=""
            if ! read -r n_cases _ < <(REGBENCH_PUBLIC_SETS="${ds#multiplex}" case_sizes "$ds" "$ds_cases"); then
                why="cannot list the cases under $ds_cases"
            elif [ "${n_cases:-0}" -eq 0 ]; then
                why="no prepared cases under $ds_cases"
            fi
        fi
        if [ -z "$why" ]; then
            printf '%s\tyes\t%s case(s) under %s\n' "$ds" "$n_cases" "$ds_cases"
        else
            printf '%s\tno\t%s\n' "$ds" "$why"
        fi
    done
    exit 0
fi
# --datasets is checked up front, before anything is read or submitted: the cohort, or a
# public dataset whose cases folder is named (REGBENCH_CASES_<NAME>)
public_sets=""
DATASETS_SAID="the cohort is multiplex; the public datasets are ${PUBLIC_DATASETS// /, }"
[ -n "${datasets// /}" ] || die "--datasets is empty: $DATASETS_SAID. Nothing was submitted."
for ds in $datasets; do
    [ "$ds" = multiplex ] && continue
    has "$PUBLIC_DATASETS" "$ds" || die "--datasets '$ds': no such dataset ($DATASETS_SAID). Nothing was submitted."
    [ "$prepare" != 1 ] || has "$COST_ONLY_SETS $LANDMARK_SETS" "$ds" || die "--prepare --datasets '$ds': this launcher prepares the cohort (multiplex), the synthetic set and the landmark set (hyreco) only. Nothing was submitted."
    # hyreco is prepared from the folder download/hyreco.sh unpacked into, read INSIDE the
    # image: named in REGBENCH_HYRECO_DATA, checked here, bound below (REGBENCH_BIND)
    if [ "$prepare" = 1 ] && [ "$ds" = hyreco ]; then
        [ -n "${REGBENCH_HYRECO_DATA:-}" ] || die "--prepare --datasets hyreco needs REGBENCH_HYRECO_DATA: the folder download/hyreco.sh unpacked into (HyReCo/{HE,PHH3}, and HyReCo-Additional/{HE,PHH3} when downloaded). Nothing was submitted."
        for hy_s in HE PHH3; do
            [ -d "$REGBENCH_HYRECO_DATA/HyReCo/$hy_s" ] || die "--prepare --datasets hyreco: REGBENCH_HYRECO_DATA=$REGBENCH_HYRECO_DATA has no HyReCo/$hy_s (download/hyreco.sh leaves HyReCo/HE and HyReCo/PHH3 there). Nothing was submitted."
        done
    fi
    kind=""
    for k in $PUBLIC_KINDS; do [ "${k%%=*}" != "$ds" ] || kind="${k#*=}"; done
    has "$SCORED_KINDS" "$kind" \
        || die "--datasets '$ds': a $kind dataset, and score() scores ${SCORED_KINDS// / and } only: its methods do not run ahead of their scorer. Nothing was submitted."
    ds_var="REGBENCH_CASES_$(tr '[:lower:]' '[:upper:]' <<< "$ds")"
    ds_cases="$(regbench_cases_of "$ds")" || die "--datasets '$ds': $ds_var is not set: export the folder its cases were prepared under. Nothing was submitted."
    # the synthetic set's folder is made by its --prepare (bound into the images from the
    # first call on, so made here; not in a dry run, which writes nothing)
    if [ "$prepare" = 1 ]; then
        [ "$DRY" = 1 ] || [ -d "$ds_cases" ] || mkdir -p "$ds_cases" \
            || die "--prepare --datasets '$ds': cannot make $ds_var=$ds_cases. Nothing was submitted."
    else
        [ -d "$ds_cases" ] || die "--datasets '$ds': $ds_var=$ds_cases does not exist: export the folder its cases were prepared under. Nothing was submitted."
    fi
    has "$public_sets" "$ds" || public_sets="${public_sets:+$public_sets }$ds"
done
# the public datasets of this submission, and only those, are bound into the images
# (env.sh: _regbench_public_cases); set always, so a value left in a profile is not read
export REGBENCH_PUBLIC_SETS="$public_sets"

# whole numbers of at least <least>: a zero here is an array of no task, or a loop
# that never ends
need_int() {
    local var="$1" least="$2" value="${!1}"
    case "$value" in ''|*[!0-9]*) die "$var='$value': a whole number of at least $least. Nothing was submitted." ;; esac
    [ "$((10#$value))" -ge "$least" ] || die "$var='$value': a whole number of at least $least. Nothing was submitted."
    printf -v "$var" '%s' "$((10#$value))"
}
need_int REGBENCH_ARRAY_LIMIT 1
need_int REGBENCH_GPU_ARRAY_LIMIT 1
need_int REGBENCH_STAGE_ARRAY_MAX 1
need_int REGBENCH_STAGE_QUEUE_MAX 1
need_int REGBENCH_STAGE_FEED_MIN 1
need_int REGBENCH_STAGE_POLL_S 1
need_int REGBENCH_STAGE_FEEDERS 1
need_int REGBENCH_RUN_LANES 0
need_int REGBENCH_SCORE_CPUS 1
need_int REGBENCH_RETRY_WATCHERS 1
# REGBENCH_REPEAT=N (T31 R13, the CPU cross's two repetitions per point): the same
# submission N times, repeat k into $REGBENCH_OUT/rep<k> (its own results, so no repeat
# reuses another's; `cpu_scaling` reads each rep<k> as one repeat). 1 (the default) is
# the launch as it always was
: "${REGBENCH_REPEAT:=1}"
need_int REGBENCH_REPEAT 1
if [ "$REGBENCH_REPEAT" -gt 1 ] && [ -z "${REGBENCH_REPEAT_INDEX:-}" ]; then
    [ -n "${REGBENCH_OUT:-}" ] || die "REGBENCH_REPEAT=$REGBENCH_REPEAT needs REGBENCH_OUT (each repeat is a folder of it). Nothing was submitted."
    # one after the other (T31 fix M3): repeat k's first jobs wait afterany for repeat
    # k-1's last ones (its scorers), so no two repeats share the nodes' time
    last_file="$(mktemp "${TMPDIR:-/tmp}/regbench_repeat.XXXXXX")" || die "REGBENCH_REPEAT: no temporary file. Nothing was submitted."
    after=""
    for ((k = 1; k <= REGBENCH_REPEAT; k++)); do
        echo "# repeat $k of $REGBENCH_REPEAT: REGBENCH_OUT=$REGBENCH_OUT/rep$k${after:+ (after repeat $((k - 1)): afterany:$after)}"
        : > "$last_file"
        REGBENCH_REPEAT_INDEX="$k" REGBENCH_OUT="$REGBENCH_OUT/rep$k" \
            REGBENCH_REPEAT_AFTER="$after" REGBENCH_REPEAT_LAST_FILE="$last_file" \
            bash "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} || { rm -f "$last_file"; exit 1; }
        after="$(cat "$last_file")"
    done
    rm -f "$last_file"
    exit 0
fi
# REGBENCH_WAVES=1 (T31 R14, a timing replicate): the runs are submitted one method at
# a time, each method's first jobs after every job of the method before it in the run
# list (afterany), so no two methods share the nodes' time. Off (0) by default
: "${REGBENCH_WAVES:=0}"
case "$REGBENCH_WAVES" in 0|1) ;; *) die "REGBENCH_WAVES='$REGBENCH_WAVES': 0 or 1. Nothing was submitted." ;; esac
# the CPU cross: 4 (every method's allocation) or 8, and who starts STARE's stages
case "$LADDER_CPUS" in
    4|8) ;;
    *) die "REGBENCH_CPUS='$LADDER_CPUS': the ladder's CPUs are 4 or 8 (4 is every method's allocation; 8 is the CPU cross's). Nothing was submitted." ;;
esac
case "$STARE_EXECUTION" in
    jobs|local) ;;
    *) die "REGBENCH_STARE_EXECUTION='$STARE_EXECUTION': jobs (the stage jobs, the default) or local (the stages inside one job per case, the CPU cross). Nothing was submitted." ;;
esac
# read by the jobs, not here (feed_tiles.sbatch and run_rung.sbatch have the defaults):
# a value that is no whole number stops the launch, not a job days later
for pair in REGBENCH_STAGE_STALL_S:1 REGBENCH_STAGE_STALL_MAX_S:1 REGBENCH_STAGE_SQUEUE_TRIES:1 \
            REGBENCH_ACCT_WAIT_S:0 REGBENCH_ACCT_POLL_S:1; do
    var="${pair%%:*}"
    [ -z "${!var+set}" ] || need_int "$var" "${pair##*:}"
done
[ "$preflight_only" = 1 ] && DRY=0   # it starts the images; it submits nothing
[ "$prepare" = 1 ] && [ "$score_only" = 1 ] && die "--prepare and --score-only are two different steps; name one. Nothing was submitted."
[ "$inputs" = 1 ] && [ "$prepare$score_only$preflight_only" != 000 ] && die "--inputs and --prepare/--score-only/--preflight are two different steps; name one. Nothing was submitted."
# compute-all: after the benchmark (--compute-all), or alone (--compute-only)
computing=0
if [ "$compute_all" = 1 ] || [ "$compute_only" = 1 ]; then
    computing=1
    [ "$compute_all$compute_only" != 11 ] || die "--compute-all and --compute-only: name one (--compute-only is compute-all without the benchmark before it). Nothing was submitted."
    [ "$inputs$prepare$score_only$preflight_only" = 0000 ] || die "--compute-all/--compute-only and --inputs/--prepare/--score-only/--preflight are two different steps; name one. Nothing was submitted."
    [ -n "${REGBENCH_INPUT:-}" ] || die "--compute-all/--compute-only need REGBENCH_INPUT: the cohort's samplesheet (the pipeline's --input). Nothing was submitted."
    [ -f "$REGBENCH_INPUT" ] || die "REGBENCH_INPUT=$REGBENCH_INPUT is not there. Nothing was submitted."
    : "${REGBENCH_COMPUTE_LANES:=2}"
    need_int REGBENCH_COMPUTE_LANES 1
    : "${REGBENCH_NF_PROFILE:=singularity,ieo}"
    export REGBENCH_INPUT REGBENCH_NF_PROFILE
fi
# what is scored is an EARLIER submission's: --score-only, and --compute-only
scoring_earlier=0
{ [ "$score_only" = 1 ] || [ "$compute_only" = 1 ]; } && scoring_earlier=1
step_flag="--score-only"
[ "$compute_only" = 1 ] && step_flag="--compute-only"

# one id per call: every record of this submission carries it. --score-only makes no
# record of its own: it scores an EARLIER submission, the last one into REGBENCH_OUT
# unless REGBENCH_SUBMISSION names another.
if [ "$scoring_earlier" = 1 ]; then
    # The ids the folder's selfcheck records carry, each record PARSED by the reader the
    # verdicts use (`regbench submissions`, benchmarks/regbench/ladder.py): one id per line.
    found="$(regbench_py "$REGBENCH_HEAD_IMAGE" submissions --out "$REGBENCH_OUT" | cut -f1)" \
        || die "$step_flag: cannot read the selfcheck records under $REGBENCH_OUT/logs/selfcheck (python -m benchmarks.regbench submissions). Nothing was submitted."
    if [ -z "${REGBENCH_SUBMISSION:-}" ]; then
        if [ -s "$REGBENCH_OUT/logs/submission" ]; then
            REGBENCH_SUBMISSION="$(head -n 1 "$REGBENCH_OUT/logs/submission")"
        else
            # a folder filled before that file was written: the id is in its selfcheck records
            [ -n "$found" ] || die "$step_flag: no submission to score under $REGBENCH_OUT (no logs/submission, and no selfcheck record names one). Export REGBENCH_SUBMISSION=<the id an earlier submit_cohort.sh printed as '# submission  -> ID'; it is also in every run.json, key 'submissions'>. Nothing was submitted."
            [ "$(wc -l <<< "$found")" -eq 1 ] || die "$step_flag: $REGBENCH_OUT holds records of more than one submission ($(paste -sd' ' - <<< "$found")) and no logs/submission says which was last. Export REGBENCH_SUBMISSION=<one of them>. Nothing was submitted."
            REGBENCH_SUBMISSION="$found"
        fi
        echo "# $step_flag: scoring submission $REGBENCH_SUBMISSION"
    fi
    # the id is the user's, or the file's: said aloud when no selfcheck record is of it
    if [ -n "$found" ] && ! grep -qxF -- "$REGBENCH_SUBMISSION" <<< "$found"; then
        echo "# WARNING $step_flag: no selfcheck record under $REGBENCH_OUT is of submission $REGBENCH_SUBMISSION (the records there are of: $(paste -sd' ' - <<< "$found")). The scorer takes only what $REGBENCH_SUBMISSION ran or reused, so every row may read 'not run': check the id."
    fi
fi
: "${REGBENCH_SUBMISSION:=$(date +%Y%m%dT%H%M%S)-$$}"
export REGBENCH_SUBMISSION
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
# twice <D-HH:MM:SS>: twice a time limit() made, never above the site's maximum: the
# `again` task's time (ladder.TIME_FACTOR, ruling 2026-10-21). Its input is always
# limit()'s own output, so it parses without refusing anything (no `die`).
twice() {
    local t="$1" d=0 h m s
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    IFS=: read -r h m s <<< "$t"
    limit $(( (((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s) * 2 ))
}

# submit <job name> <sbatch arguments...>: sets JOB_ID. Dry: prints the command.
N_JOBS=0
N_TASKS=0
JOB_ID=""
JOBS_FILE=""
# array_size <a-b,c,...>: how many tasks an --array list has
array_size() {
    awk -F, '{ for (i = 1; i <= NF; i++) { n = split($i, r, "-"); t += (n == 2 ? r[2] - r[1] + 1 : 1) } } END { print t + 0 }' <<< "$1"
}
REPEAT_SINKS="" ALL_SUBMITTED=""
submit() {
    local name="$1" arg range; shift
    N_JOBS=$((N_JOBS + 1))
    # REGBENCH_REPEAT (T31 fix M3): a job with no dependency of its own is a first job,
    # and waits for the repeat before (every other job waits for one of those)
    if [ -n "${REGBENCH_REPEAT_AFTER:-}" ]; then
        case " $* " in *" --dependency="*) ;; *) set -- --dependency="afterany:$REGBENCH_REPEAT_AFTER" "$@" ;; esac
    fi
    range=""
    for arg in "$@"; do case "$arg" in --array=*) range="${arg#--array=}"; range="${range%%\%*}" ;; esac; done
    if [ -n "$range" ]; then
        N_TASKS=$((N_TASKS + $(array_size "$range")))
    else
        N_TASKS=$((N_TASKS + 1))
    fi
    if [ "$DRY" = 1 ]; then
        printf 'sbatch --parsable --job-name=%s' "$name"
        printf ' %s' "$@"
        printf '\n'
        JOB_ID="<$name>"
        repeat_note "$name"
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
    if [ "$N_QUEUED" = 1 ] && [ "$prepare" != 1 ] && [ "$inputs" != 1 ] && [ "$scoring_earlier" != 1 ]; then
        echo "$REGBENCH_SUBMISSION" > "$REGBENCH_OUT/logs/submission"
    fi
    printf '%s\t%s\n' "$name" "$JOB_ID"
    printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$name" "$JOB_ID" >> "$JOBS_FILE"
    repeat_note "$name"
    [ -z "$PENDING_EXIT" ] || exit "$PENDING_EXIT"
}
# the jobs the next repeat waits for: the scorers, which wait for everything else of
# this one; a submission with no scorer gives every job it submitted
repeat_note() {
    ALL_SUBMITTED="${ALL_SUBMITTED:+$ALL_SUBMITTED:}$JOB_ID"
    case "$1" in rb-score|rb-rescore|rb-tissue|rb-score-compute) REPEAT_SINKS="${REPEAT_SINKS:+$REPEAT_SINKS:}$JOB_ID" ;; esac
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
settle_s="$(seconds "$REGBENCH_SETTLE_TIME")"
# the synthetic set's steps (--prepare --datasets synthetic): plan, each array task, finish
synth_plan_s="$(seconds "${REGBENCH_SYNTH_PLAN_TIME:-04:00:00}")"
synth_s="$(seconds "${REGBENCH_SYNTH_TIME:-04:00:00}")"
synth_finish_s="$(seconds 01:00:00)"
# shellcheck disable=SC2206  # the site's GPU request is several words on purpose
GPU_ARGS=(${REGBENCH_SBATCH_GPU:-})

# ── the input step, on its own: one Nextflow head job, nothing of the benchmark ───────
if [ "$inputs" = 1 ]; then
    [ -n "${REGBENCH_INPUT:-}" ] || die "--inputs needs REGBENCH_INPUT: the cohort's samplesheet (the pipeline's --input). Nothing was submitted."
    [ -f "$REGBENCH_INPUT" ] || die "--inputs: REGBENCH_INPUT=$REGBENCH_INPUT is not there. Nothing was submitted."
    [ -n "${REGBENCH_INPUTS:-}" ] || die "--inputs needs REGBENCH_INPUTS: the folder the input step writes into (launch.sbatch: inputs/ of REGBENCH_ROOT). Nothing was submitted."
    : "${REGBENCH_NF_PROFILE:=singularity,ieo}"
    export REGBENCH_INPUT REGBENCH_INPUTS REGBENCH_NF_PROFILE
    if [ -e "$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv" ]; then
        [ "${REGBENCH_INPUTS_FORCE:-0}" = 1 ] || die "--inputs: $REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv already exists: the input step has run (next: --prepare). REGBENCH_INPUTS_FORCE=1 runs it again; the earlier one is moved aside to $REGBENCH_INPUTS/preprocess.replaced/, never deleted. Nothing was submitted."
        echo "# REGBENCH_INPUTS_FORCE=1: the input step runs again; $REGBENCH_INPUTS/preprocess is moved aside to preprocess.replaced/ when the job starts"
        export REGBENCH_INPUTS_FORCE
    fi
    inputs_s="$(seconds "${REGBENCH_INPUTS_TIME:-2-00:00:00}")"
    if [ "$DRY" != 1 ]; then
        mkdir -p "$REGBENCH_OUT/logs/submitted" || die "cannot write $REGBENCH_OUT/logs. Nothing was submitted."
        JOBS_FILE="$REGBENCH_OUT/logs/jobs.tsv"
        IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.inputs.ids"
    fi
    echo "# the input step: $REGBENCH_INPUT -> $REGBENCH_INPUTS/preprocess (profile $REGBENCH_NF_PROFILE, site config ${REGBENCH_SITE_CONFIG:-conf/ieo.config})"
    submit "rb-inputs" "${COMMON[@]}" "$LOG_ONE" --cpus-per-task=2 --mem=8G \
        --time="$(limit "$inputs_s")" "$HERE/inputs.sbatch"
    echo "inputs: 1 job. When it is done (REGBENCH_END kind=inputs status=ok in its log), run this script again with --prepare."
    FINISHED=1
    exit 0
fi

# AN EARLIER RESULTS FOLDER OF THE SAME CASES. The default REGBENCH_OUT is named after
# the checkout's commit (env.sh), so a launch from a newer commit starts an EMPTY folder
# and reuses nothing an earlier launch finished. Said loudly; the default is not changed.
# An earlier folder is one of the same cases when its logs/cases (written below by every
# launch) names them; a folder written before that file existed is named as unknown.
earlier_default_folders() {
    local dir cases n
    [ "${REGBENCH_OUT_DEFAULTED:-0}" = 1 ] || return 0
    for dir in "$REGBENCH_RESULTS_HOME"/mirage-*; do
        [ -d "$dir/logs" ] || continue
        [ "$dir" -ef "$REGBENCH_OUT" ] && continue
        cases=""
        [ -s "$dir/logs/cases" ] && cases="$(head -n 1 "$dir/logs/cases")"
        if [ -n "$cases" ] && [ "$cases" != "$REGBENCH_CASES" ] && ! [ "$cases" -ef "$REGBENCH_CASES" ]; then
            continue
        fi
        n="$(find "$dir" -mindepth 4 -maxdepth 4 -name run.json 2>/dev/null | wc -l | tr -d ' ')"
        [ -n "$cases" ] || [ "$n" -gt 0 ] || continue
        {
            echo "# WARNING results folder: REGBENCH_OUT is the default for this commit ($REGBENCH_OUT), and an EARLIER default-named folder holds $n result(s) $([ -n "$cases" ] && echo "of the same cases ($cases)" || echo "of cases it did not record (written before logs/cases existed)"): $dir"
            echo "# WARNING results folder: nothing in it is reused by this launch. To reuse it (every unit that is ok there is not run again): export REGBENCH_OUT=$dir"
        } | tee /dev/stderr
    done
    return 0
}
earlier_default_folders

# A RESULTS FOLDER SLIMMED BY `regbench slim` (benchmarks/regbench/paper/slim.py). At
# `csv` its maps (warped.npz) and records are gone: no result in it is reusable, so every
# unit runs again (the run's own reuse check, methods.record_gaps, finds no warped.npz),
# and it can no longer be scored (score refuses it). Said here; `images` changes nothing.
# Asked only when the folder holds SLIMMED.txt, so any other launch is unchanged.
if [ -e "$REGBENCH_OUT/SLIMMED.txt" ]; then
    slimmed="$(regbench_py "$REGBENCH_HEAD_IMAGE" slim --status --root "$REGBENCH_OUT")" \
        || die "cannot read $REGBENCH_OUT/SLIMMED.txt (python -m benchmarks.regbench slim --status). Nothing was submitted."
    if [ "$slimmed" = "slimmed: csv" ]; then
        [ "$score_only" = 1 ] && die "--score-only: $REGBENCH_OUT is $slimmed ($REGBENCH_OUT/SLIMMED.txt): its maps and records are removed, so it can no longer be scored (score refuses it). Nothing was submitted."
        echo "# WARNING results folder: $REGBENCH_OUT is $slimmed ($REGBENCH_OUT/SLIMMED.txt): its maps (warped.npz) and records are removed, so NO result in it is reusable: every unit runs again" | tee /dev/stderr
    else
        echo "# results folder: $REGBENCH_OUT is $slimmed ($REGBENCH_OUT/SLIMMED.txt): its written images are removed; its maps are kept, so its results are reused as usual"
    fi
fi

if [ "$DRY" = 1 ]; then
    scratch="$(mktemp -d)"
    scratch_is_temporary=1
else
    mkdir -p "$REGBENCH_OUT/logs/selfcheck"
    # the cases this folder's results are of (read by a later launch's warning above)
    printf '%s\n' "$REGBENCH_CASES" > "$REGBENCH_OUT/logs/cases" \
        || die "cannot write $REGBENCH_OUT/logs/cases. Nothing was submitted."
    scratch="$REGBENCH_OUT/logs"
    JOBS_FILE="$REGBENCH_OUT/logs/jobs.tsv"
    mkdir -p "$REGBENCH_OUT/logs/submitted"
    # every job id of THIS call, one per line (the tile submitters add theirs)
    IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.ids"
    # the jobs of --prepare are not a submission's: their ids have a file of their own
    [ "$prepare" = 1 ] && IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.prepare.ids"
    # nor those of --compute-only, which runs after a submission and scores it again
    [ "$compute_only" = 1 ] && IDS_FILE="$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.compute.ids"
    export REGBENCH_IDS_FILE="$IDS_FILE"
    # A second submission while jobs of an earlier one are still in the queue: their
    # tasks read the stage files and retry plans this call would replace. And a
    # scoring: it would score, and settle, results whose tasks are still running.
    if [ "$prepare" != 1 ] && [ "$preflight_only" != 1 ] && [ -s "$JOBS_FILE" ] \
        && ! { [ "$score_only" = 1 ] && [ "${REGBENCH_SCORE_WHILE_QUEUED:-0}" = 1 ]; }; then
        # a queue that does not answer is asked again (B13): REGBENCH_SQUEUE_TRIES (5)
        # times, REGBENCH_SQUEUE_WAIT_S (10) seconds apart, before the launch stops
        tries=0
        until in_queue="$(squeue -h -u "$(id -un)" -o '%F %A' 2>/dev/null)"; do
            tries=$((tries + 1))
            [ "$tries" -ge "${REGBENCH_SQUEUE_TRIES:-5}" ] \
                && die "$REGBENCH_OUT already holds a submission ($JOBS_FILE) and the queue cannot be read (squeue), asked $tries times ${REGBENCH_SQUEUE_WAIT_S:-10}s apart, so it cannot be shown that its jobs have ended. Nothing was submitted."
            echo "# the queue could not be read (squeue), try $tries of ${REGBENCH_SQUEUE_TRIES:-5}: asked again in ${REGBENCH_SQUEUE_WAIT_S:-10}s"
            sleep "${REGBENCH_SQUEUE_WAIT_S:-10}"
        done
        still="$(tr ' ' '\n' <<< "$in_queue" | grep -Fxf <(cut -f3 "$JOBS_FILE" | cut -d_ -f1 | sort -u) | sort -u | paste -sd' ' -)" || still=""
        if [ -n "$still" ] && [ "$compute_only" = 1 ]; then
            die "--compute-only: $(wc -w <<< "$still" | tr -d ' ') job(s) of an earlier call into $REGBENCH_OUT are still pending or running ($still). compute-all runs ALONE, after them: a cost measured beside them is not a cost. Wait for them; or cancel them:  scancel $still ; or use --compute-all in the submission itself. Nothing was submitted."
        elif [ -n "$still" ] && [ "$score_only" = 1 ]; then
            die "--score-only: $(wc -w <<< "$still" | tr -d ' ') job(s) of an earlier call into $REGBENCH_OUT are still pending or running ($still). Scoring now would score what is there so far and settle the results of tasks that have not ended, so it is refused. Wait for them; or cancel them:  scancel $still ; or score what is there all the same with REGBENCH_SCORE_WHILE_QUEUED=1. Nothing was submitted."
        elif [ -n "$still" ]; then
            die "$(wc -w <<< "$still" | tr -d ' ') job(s) of an earlier submission into $REGBENCH_OUT are still pending or running ($still). A second submission would replace the stage files their tasks read, so it is refused. Wait for them; or cancel them:  scancel $still ; or submit into another REGBENCH_OUT. Nothing was submitted."
        fi
    fi
    # An id is one submission's: the stage files, the tile submitters' lists and the job
    # ids of an earlier call under the same id would be taken for this call's.
    if [ "$prepare" != 1 ] && [ "$scoring_earlier" != 1 ] && [ "$preflight_only" != 1 ] && [ -e "$IDS_FILE" ]; then
        die "the submission id '$REGBENCH_SUBMISSION' was already used for a submission into $REGBENCH_OUT ($IDS_FILE). A second one under the same id is refused: its tile submitters would find the first one's lists of what is due and what was fed ($REGBENCH_OUT/logs/stages/$REGBENCH_SUBMISSION/) and feed nothing. Unset REGBENCH_SUBMISSION (every call then gets an id of its own) or export a new one; every unit that is ok is reused whatever the id. Nothing was submitted."
    fi
fi

# ── the run list, from the harness alone ──────────────────────────────────────────────
table="$scratch/runs.tsv"
regbench_py "$REGBENCH_HEAD_IMAGE" runs > "$table"
[ -s "$table" ] || die "the run list is empty (python -m benchmarks.regbench runs)"
# REGBENCH_RUN_ORDER: the same rows in the order they are to be submitted. Every loop
# below reads the table top to bottom, and a job only ever waits for a job submitted
# before it, so the harness never puts a run before the run whose results it reads.
# Unset, the cohort's default: ASHLAR first.
: "${REGBENCH_RUN_ORDER=default}"
if [ -n "${REGBENCH_RUN_ORDER:-}" ]; then
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --order "$REGBENCH_RUN_ORDER" > "$table.ordered" \
        || die "REGBENCH_RUN_ORDER='$REGBENCH_RUN_ORDER' is not an order of the run list (see the line above). Nothing was submitted."
    mv "$table.ordered" "$table"
    echo "# run order: $(cut -f1 "$table" | paste -sd' ' -)"
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --order "$REGBENCH_RUN_ORDER" --order-notes \
        | while IFS= read -r note; do echo "# run order: $note"; done
fi
REGBENCH_SCORER_IMAGE="$(regbench_py "$REGBENCH_HEAD_IMAGE" runs --scorer-image)"
export REGBENCH_SCORER_IMAGE
# A run whose `stages` column (the 16th) reads `local` is ALWAYS its stages inside one
# job on the ordinary ladder (T31 R2: stare_local4, STARE at 4 CPUs in one job, a cost
# row of stare's): OWN_LOCAL, with the workers its own options name (never appended),
# and its column read as `-` like every LOCAL_RUN's. The other runs' rows are unchanged
own_local() {  # <run list>
    local name stages
    while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ _ stages; do
        [ "${stages:-}" = local ] || continue
        has "$OWN_LOCAL" "$name" || OWN_LOCAL="$OWN_LOCAL $name"
        has "$LOCAL_RUNS" "$name" || LOCAL_RUNS="$LOCAL_RUNS $name"
    done < "$1"
    if ! awk -F'\t' -v OFS='\t' 'NF >= 16 && $16 == "local" { $16 = "-" } { print }' "$1" > "$1.own" \
        || ! mv "$1.own" "$1"; then
        die "cannot read the run list's stages column ($1). Nothing was submitted."
    fi
}
# REGBENCH_STARE_EXECUTION=local: the run list's `stages` column (the 16th) is read as
# `-` by every loop below, so no stage chain, layout or tile submitter is made; the runs
# it named are LOCAL_RUNS, laddered one job per case with `local` (ladder, below)
LOCAL_RUNS="" OWN_LOCAL=""
own_local "$table"
if [ "$STARE_EXECUTION" = local ]; then
    while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ _ stages; do
        [ "${stages:--}" = - ] || LOCAL_RUNS="$LOCAL_RUNS $name"
    done < "$table"
    if ! awk -F'\t' -v OFS='\t' 'NF >= 16 { $16 = "-" } { print }' "$table" > "$table.local" \
        || ! mv "$table.local" "$table"; then
        die "cannot read the run list's stages column as local ($table). Nothing was submitted."
    fi
    echo "# STARE: stages run inside one job of $LADDER_CPUS CPUs (local)${LOCAL_RUNS:+:$LOCAL_RUNS}"
fi

# a public dataset's own run list (`runs --dataset <name>`: its regime's runs, less those
# without an input rule for it), in the same order and with the same reading of the stages
# column; the cohort's is $table (table_of)
table_of() {
    if [ "$1" = multiplex ]; then echo "$table"; else echo "$scratch/runs-$1.tsv"; fi
}
for ds in $datasets; do
    [ "$ds" = multiplex ] && continue
    t="$(table_of "$ds")"
    order=()
    [ -n "${REGBENCH_RUN_ORDER:-}" ] && order=(--order "$REGBENCH_RUN_ORDER")
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --dataset "$ds" ${order[@]+"${order[@]}"} > "$t" \
        || die "cannot read the run list of $ds (python -m benchmarks.regbench runs --dataset $ds). Nothing was submitted."
    [ -s "$t" ] || die "the run list of $ds is empty (python -m benchmarks.regbench runs --dataset $ds). Nothing was submitted."
    # its own order (a public set's default: STARE first), not the cohort's line above
    [ -z "${REGBENCH_RUN_ORDER:-}" ] || echo "# run order ($ds): $(cut -f1 "$t" | paste -sd' ' -)"
    own_local "$t"
    if [ "$STARE_EXECUTION" = local ]; then
        while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ _ stages; do
            [ "${stages:--}" = - ] || has "$LOCAL_RUNS" "$name" || LOCAL_RUNS="$LOCAL_RUNS $name"
        done < "$t"
        if ! awk -F'\t' -v OFS='\t' 'NF >= 16 { $16 = "-" } { print }' "$t" > "$t.local" \
            || ! mv "$t.local" "$t"; then
            die "cannot read the run list's stages column as local ($t). Nothing was submitted."
        fi
    fi
    echo "# $ds: $(wc -l < "$t" | tr -d ' ') run(s) of its own list (runs --dataset $ds); the scorer writes $REGBENCH_OUT/tables/$ds/"
done

# the submission's runs: the named datasets' lists, one row per run name (the first
# list that has it, the cohort's first), so a submission without multiplex selects,
# checks and sizes no run of the cohort's alone. The default launch's is $table itself
if [ "$datasets" = multiplex ]; then
    union="$table"
else
    union="$scratch/runs-union.tsv"
    lists=()
    for ds in $datasets; do lists+=("$(table_of "$ds")"); done
    awk -F'\t' '!seen[$1]++' "${lists[@]}" > "$union" \
        || die "cannot join the run lists of $datasets ($union). Nothing was submitted."
fi
all_names="" selected=""
while IFS=$'\t' read -r name _; do all_names="$all_names $name"; done < "$union"
for r in $only_runs; do has "$all_names" "$r" || die "--runs: no run named '$r' (see: python -m benchmarks.regbench runs)"; done
for r in $all_names; do
    if [ -z "$only_runs" ] || has "$only_runs" "$r"; then selected="$selected $r"; fi
done
for pair in ${REGBENCH_FIRST_RUNG:-}; do
    case "$pair" in *=*) ;; *) die "REGBENCH_FIRST_RUNG: '$pair' is not <run>=<GB>. Nothing was submitted." ;; esac
    has "$all_names" "${pair%%=*}" || echo "# REGBENCH_FIRST_RUNG: '${pair%%=*}' is in no run list of this launch: not used"
    has "$RUNGS_GB" "${pair#*=}" || die "REGBENCH_FIRST_RUNG: ${pair#*=} GB is not a rung ($RUNGS_GB). Nothing was submitted."
done
# a submission that runs anything writes its own submit_dropped.csv (header alone when
# no run is dropped); --score-only leaves the scored submission's as it is
if [ "$scoring_earlier" != 1 ] && [ "$preflight_only" != 1 ] && [ "$prepare" != 1 ] && [ -n "${selected// /}" ]; then
    write_dropped
fi
[ -n "${selected// /}" ] && [ "$scoring_earlier" != 1 ] && SELECTED_ANY=1
# --score-only: no run is selected, so nothing of steps 1 to 4 is submitted
if [ "$scoring_earlier" = 1 ]; then
    [ -z "$only_runs" ] || echo "# $step_flag: --runs is not used (the scorer scores every run of the submission)"
    selected=""
fi
# REGBENCH_STARE_EXECUTION=local into a folder that holds STARE results of its stage jobs
# is refused: those results have another reuse key (execution `jobs`), so every STARE
# unit would run again and replace them. The cross has folders of its own (README, "The
# CPU cross"). A staged launch over local results is not refused.
if [ "$STARE_EXECUTION" = local ] && [ -d "$REGBENCH_OUT" ]; then
    for r in $LOCAL_RUNS; do
        has "$selected" "$r" || continue
        has "$OWN_LOCAL" "$r" && continue  # its own reuse key: never a staged result's
        staged_found="$(regbench_py "$REGBENCH_HEAD_IMAGE" staged-results --out "$REGBENCH_OUT")" \
            || die "cannot read which STARE results under $REGBENCH_OUT ran as stage jobs (python -m benchmarks.regbench staged-results). Nothing was submitted."
        if [ -n "$staged_found" ]; then
            die "REGBENCH_STARE_EXECUTION=local into $REGBENCH_OUT, which holds $(wc -l <<< "$staged_found" | tr -d ' ') STARE result(s) of its stage jobs (execution jobs), e.g. $(head -n 1 <<< "$staged_found"): every STARE unit would run again under another reuse key and replace them. Use a separate REGBENCH_OUT for the CPU cross (README, \"The CPU cross\": one folder per CPU count). Nothing was submitted."
        fi
        break
    done
fi
# compute-all's runs, read before the first sbatch: (backend, patient) in the order
# they are submitted
COMPUTE_PLAN=""
if [ "$computing" = 1 ]; then
    COMPUTE_PLAN="$(regbench_py "$REGBENCH_HEAD_IMAGE" compute-all plan --input "$REGBENCH_INPUT" \
        --out "$REGBENCH_OUT")" \
        || die "compute-all: cannot list the runs of $REGBENCH_INPUT (python -m benchmarks.regbench compute-all plan; see above). Nothing was submitted."
    [ -n "$COMPUTE_PLAN" ] || die "compute-all: $REGBENCH_INPUT names no patient. Nothing was submitted."
    compute_s="$(seconds "${REGBENCH_COMPUTE_TIME:-7-00:00:00}")"
    echo "# compute-all: $(wc -l <<< "$COMPUTE_PLAN" | tr -d ' ') run(s) of the whole pipeline ($REGBENCH_INPUT, profile $REGBENCH_NF_PROFILE), on $REGBENCH_COMPUTE_LANES lane(s), after $([ "$compute_only" = 1 ] && echo "nothing (--compute-only)" || echo "the scorer")"
fi
# ── the memory ladder against the partition: can a node grant each rung? ─────────────
# The partition and constraint the jobs are given (the caller's sbatch arguments, else
# SBATCH_PARTITION / SBATCH_CONSTRAINT, else the cluster's default partition), and the
# largest node's memory there, from `sinfo -N -o '%P|%m|%f'` (%m: the node's memory in
# MB; %f: its features; a `*` after the partition marks the default one: the sinfo
# manual's -o field list `[MEMORY]`, not retrieved while this was written). Sets
# GRANT_MB (empty: not known) and GRANT_WHERE. A request above every node's memory is
# refused by sbatch or pends for ever: the rungs above GRANT_MB are not planned (ladder,
# below), and the units that would climb to them are written down as the launcher's.
GRANT_MB="" GRANT_WHERE="" GRANT_GPU_MB="" GRANT_GPU_WHERE="" GRANT_SAID="" GRANT_AT="" GRANT_AT_MB=""
# where <sbatch arguments...>: "<partition>|<constraint>" those arguments name (the last
# of each wins, as sbatch reads them); empty fields when they name none
where() {
    local part="" cons="" i
    local -a args=("$@")
    for i in "${!args[@]}"; do
        case "${args[$i]}" in
            --partition=*) part="${args[$i]#--partition=}" ;;
            --partition|-p) part="${args[$((i + 1))]:-}" ;;
            -p?*) part="${args[$i]#-p}" ;;
            --constraint=*) cons="${args[$i]#--constraint=}" ;;
            --constraint|-C) cons="${args[$((i + 1))]:-}" ;;
            -C?*) cons="${args[$i]#-C}" ;;
        esac
    done
    echo "$part|$cons"
}
# largest <partition> <constraint> <gres name|-> : the largest node's memory in MB among
# the nodes of the partition (the default one when empty) that meet the constraint and,
# with a gres name (`gpu`, `gpu:nvidia_h200`), have that gres (sinfo %G); empty when sinfo
# lists none. Sets SINFO_ERR when sinfo failed.
SINFO_ERR=""
largest() {
    local part="$1" cons="$2" gres="$3" nodes best
    if ! nodes="$(sinfo -h -N -o '%P|%m|%f|%G' ${part:+-p "$part"} 2>&1)"; then
        SINFO_ERR="$(head -n 1 <<< "$nodes")"
        return 0
    fi
    # the default partition's nodes when none is named; every term of an `&` constraint
    # (any of an `|` one) among the node's features; the gres among the node's
    best="$(awk -F'|' -v part="$part" -v cons="$cons" -v gres="$gres" -v any_part=0 '
        BEGIN { gsub(/[][()]/, "", cons); gsub(/\*[0-9]+/, "", cons)
                n_any = split(cons, any, "|"); n_all = split(cons, all, "&") }
        function has(f, w,   k, n, list) { n = split(f, list, ","); for (k = 1; k <= n; k++) if (list[k] == w) return 1; return 0 }
        function has_gres(g, w,   k, n, list) { n = split(g, list, ","); for (k = 1; k <= n; k++) if (index(list[k] ":", w ":") == 1) return 1; return 0 }
        { p = $1; m = $2; f = $3; g = $4; gsub(/[^0-9]/, "", m) }
        part == "" && p !~ /\*$/ { next }
        gres != "-" && !has_gres(g, gres) { next }
        { ok = 1
          if (cons != "" && n_any > 1) { ok = 0; for (k = 1; k <= n_any; k++) if (has(f, any[k])) ok = 1 }
          else if (cons != "") { for (k = 1; k <= n_all; k++) if (!has(f, all[k])) ok = 0 }
          if (ok && m + 0 > best) best = m + 0; seen = 1 }
        END { if (seen) print best }' <<< "$nodes")"
    if [ -z "$best" ] && [ -z "$part" ]; then
        # no partition is marked as the default: every node (with the gres)
        best="$(awk -F'|' -v gres="$gres" '
            function has_gres(g, w,   k, n, list) { n = split(g, list, ","); for (k = 1; k <= n; k++) if (index(list[k] ":", w ":") == 1) return 1; return 0 }
            gres != "-" && !has_gres($4, gres) { next }
            { m = $2; gsub(/[^0-9]/, "", m); if (m + 0 > best) best = m + 0; seen = 1 } END { if (seen) print best }' <<< "$nodes")"
    fi
    [ "$best" = 0 ] && best=""
    echo "$best"
}
# rung_cut_said <MB> <where> <label> <rungs...>: the warning for the rungs above <MB>
rung_cut_said() {
    local mb="$1" at="$2" label="$3" cut="" gb; shift 3
    for gb in "$@"; do
        [ $((gb * 1024)) -le "$mb" ] || has "$cut" "$gb" || cut="$cut $gb"
    done
    if [ -n "$cut" ]; then
        cut="$(sed -e 's/^ //' -e 's/ / GB, /g' <<< "$cut") GB"
        {
            echo "# WARNING memory ladder: ${label}the largest node of $at has $mb MB (sinfo), so the rung(s) $cut can NEVER be granted there."
            echo "# WARNING memory ladder: those rungs are NOT planned: a unit that runs out of memory below them is written down as the launcher's (not_submitted, ladder_cut_gb), not as the method's out-of-memory. For the full ladder (16-700 GB for every method, the user's ruling) name a partition, or drop a constraint, that reaches the site's largest node: -- --partition=..."
        } | tee /dev/stderr
        return 1
    fi
    return 0
}
rung_check() {
    local part cons w want gpu_part gpu_cons gres
    [ "$REGBENCH_RUNG_CHECK" = 0 ] && { echo "# memory ladder: not checked against the partition (REGBENCH_RUNG_CHECK=0)"; return 0; }
    w="$(where ${extra[@]+"${extra[@]}"})"
    part="${w%%|*}" cons="${w#*|}"
    [ -n "$part" ] || part="${SBATCH_PARTITION:-}"
    [ -n "$cons" ] || cons="${SBATCH_CONSTRAINT:-}"
    if ! command -v sinfo >/dev/null; then
        echo "# memory ladder: NOT checked against the partition (no sinfo here): a rung no node can grant would be found only when its step is refused, days into the run"
        return 0
    fi
    SINFO_ERR=""
    GRANT_MB="$(largest "$part" "$cons" -)"
    if [ -n "$SINFO_ERR" ]; then
        GRANT_MB=""
        echo "# memory ladder: NOT checked against the partition (sinfo failed: $SINFO_ERR)"
        return 0
    fi
    GRANT_WHERE="partition ${part:-<the default>}${cons:+, constraint $cons}"
    if [ -z "$GRANT_MB" ]; then
        echo "# memory ladder: NOT checked: sinfo lists no node of $GRANT_WHERE"
    else
        want="$(awk '{ print $1 * 1024 }' <<< "${RUNGS_GB%% *}")"
        [ "$want" -le "$GRANT_MB" ] || die "the memory ladder's first rung, ${RUNGS_GB%% *} GB, is more than the largest node of $GRANT_WHERE has ($GRANT_MB MB, sinfo): no run could start. Name a partition with larger nodes (-- --partition=...). Nothing was submitted."
        # shellcheck disable=SC2086  # the rungs are words
        rung_cut_said "$GRANT_MB" "$GRANT_WHERE" "" $RUNGS_GB $SELFCHECK_RUNGS_GB \
            && echo "# memory ladder: every rung (${RUNGS_GB// /, } GB; selfcheck ${SELFCHECK_RUNGS_GB// /, } GB) fits the largest node of $GRANT_WHERE ($GRANT_MB MB, sinfo)"
    fi
    # A TASK THAT NEEDS A GPU is given REGBENCH_SBATCH_GPU after the caller's arguments
    # (sbatch takes the last): its partition and constraint where it names them, and a
    # node that has the gres it asks for (`--gres=gpu[:type]:n`)
    GRANT_GPU_MB="" GRANT_GPU_WHERE=""
    [ "${#GPU_ARGS[@]}" -gt 0 ] || return 0
    w="$(where "${GPU_ARGS[@]}")"
    gpu_part="${w%%|*}" gpu_cons="${w#*|}"
    [ -n "$gpu_part" ] || gpu_part="$part"
    [ -n "$gpu_cons" ] || gpu_cons="$cons"
    gres="$(printf '%s\n' "${GPU_ARGS[@]}" | sed -n 's/^--gres=//p' | tail -n 1)"
    gres="${gres%%,*}"
    case "$gres" in *:[0-9]*) gres="${gres%:*}" ;; esac
    SINFO_ERR=""
    GRANT_GPU_MB="$(largest "$gpu_part" "$gpu_cons" "${gres:--}")"
    GRANT_GPU_WHERE="partition ${gpu_part:-<the default>}${gpu_cons:+, constraint $gpu_cons}${gres:+ with gres $gres}"
    if [ -n "$SINFO_ERR" ] || [ -z "$GRANT_GPU_MB" ]; then
        # held to what the CPU runs are held to, as before the GPU rows had a check
        echo "# memory ladder, GPU runs: sinfo lists no node of $GRANT_GPU_WHERE${SINFO_ERR:+ (sinfo failed: $SINFO_ERR)}: checked against $GRANT_WHERE, as the other runs"
        GRANT_GPU_MB="$GRANT_MB" GRANT_GPU_WHERE="$GRANT_WHERE"
        return 0
    fi
    want="$(awk '{ print $1 * 1024 }' <<< "${RUNGS_GB%% *}")"
    [ -z "${gpu_runs:-}" ] || [ "$want" -le "$GRANT_GPU_MB" ] || die "the memory ladder's first rung, ${RUNGS_GB%% *} GB, is more than the largest node of $GRANT_GPU_WHERE has ($GRANT_GPU_MB MB, sinfo): the runs that need a GPU ($gpu_runs ) could not start. Name the site's GPU partition and gres in REGBENCH_SBATCH_GPU, or leave those runs out with --runs. Nothing was submitted."
    # shellcheck disable=SC2086
    rung_cut_said "$GRANT_GPU_MB" "$GRANT_GPU_WHERE" "GPU runs: " $RUNGS_GB $SELFCHECK_RUNGS_GB \
        && echo "# memory ladder, GPU runs: every rung fits the largest node of $GRANT_GPU_WHERE ($GRANT_GPU_MB MB, sinfo)"
    return 0
}
# grantable <GB> [<gpu: 1>]: can a node the task may have grant it (yes when it is not
# known). Leaves what it checked against in GRANT_SAID (for plan_cut's reason).
grantable() {
    local mb="$GRANT_MB" at="$GRANT_WHERE"
    if [ "${2:-0}" = 1 ] && [ "${#GPU_ARGS[@]}" -gt 0 ]; then
        mb="$GRANT_GPU_MB" at="$GRANT_GPU_WHERE"
    fi
    GRANT_SAID="the largest node of $at has $mb MB (sinfo at launch)"
    GRANT_AT="$at" GRANT_AT_MB="$mb"
    [ -z "$mb" ] || [ $(($1 * 1024)) -le "$mb" ]
}

# RETRIES ARE SUBMITTED AFTER A FAILURE, NEVER BEFORE IT. Of a chain (a memory ladder, the
# attempts of a stage) only the FIRST array is submitted here, with one small job after
# it, the chain's watcher (retry.sbatch). The other steps are written down as a plan
# (plan_step: $REGBENCH_OUT/logs/retry/<submission>/<job prefix>/step-<k>), and the
# watcher submits a step only for the units whose last task failed in a way that step
# retries, then waits for it. A chain whose units are all ok puts nothing more in the
# queue. Everything that waited for a chain's last array waits for its watcher.
N_LATER=0 PLAN="" PLAN_SHOWN="" PLAN_N=0
plan_open() {  # plan_open <job prefix> [<array limit>]
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
    # the %N of the chain's retry arrays (retry.sbatch reads it): a GPU run's is
    # REGBENCH_GPU_ARRAY_LIMIT
    echo "${2:-$REGBENCH_ARRAY_LIMIT}" > "$PLAN/limit"
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
# plan_cut <job name> <sbatch arguments...>: a rung no node of the partition can grant
# (rung_check). Never submitted: the watcher writes the units that would climb to it
# down as the launcher's (retry.sbatch, `cut`), and the ladder ends there.
N_CUT=0
plan_cut() {
    local jobname="$1"; shift
    PLAN_N=$((PLAN_N + 1))
    N_CUT=$((N_CUT + 1))
    printf '%s\n' cut "$jobname" "$GRANT_SAID" "$@" > "$PLAN/step-$PLAN_N"
    [ "$DRY" = 1 ] && echo "# NOT PLANNED, no node of $GRANT_AT can grant it ($GRANT_AT_MB MB): $jobname"
    return 0
}
# plan_watch <job prefix> <first job id> <first job name> <array range|-> <dependencies
# of the first job> [<run|->]: the chain's watcher; sets JOB_ID. It has the first job's
# own dependencies too. A range of `-` is a chain of single jobs (a selfcheck), and a
# run of `-` a selfcheck's chain.
# A watcher is REGBENCH_RETRY_WATCHERS links: link 1 (`<prefix>-retry`) after the first
# job, each later link (`<prefix>-retry-<n>`) after the link before it, whatever became
# of that one (afterany). A link ended at its time limit, or lost with its node, leaves
# the next to go on from the step it was at (retry.sbatch: state); one that finds the
# chain finished ends at once. JOB_ID is the LAST link: what waits for the chain.
plan_watch() {
    local dep="afterany:$2" n jobname
    [ -z "$5" ] || dep="$5,$dep"
    for n in $(seq 1 "$REGBENCH_RETRY_WATCHERS"); do
        jobname="$1-retry"
        [ "$n" = 1 ] || jobname="$1-retry-$n"
        submit "$jobname" "${COMMON[@]}" "$LOG_ONE" --dependency="$dep" \
            --cpus-per-task=1 --mem=1G --time="$(limit "$MAX_SECONDS")" \
            "$HERE/retry.sbatch" "${6:-$name}" "$PLAN_SHOWN" "$2" "$3" "$4" "$n"
        dep="afterany:$JOB_ID"
    done
}

# ── prepare, on its own: array sizes are read from what it writes ─────────────────────
if [ "$prepare" = 1 ]; then
    # the rounds are the cohort's: a --prepare of the synthetic set alone reads none
    if has "$datasets" multiplex; then
    if [ -z "${REGBENCH_MANIFEST:-}" ]; then
        # the rounds of the cohort, from regbench's own input step (launch.sbatch --inputs)
        [ -n "${REGBENCH_INPUTS:-}" ] || die "--prepare needs REGBENCH_INPUTS (the input folder, whose rounds.csv is written from the input step) or REGBENCH_MANIFEST (a rounds.csv to take as it is). Nothing was submitted."
        checkpoint="$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv"
        [ -s "$checkpoint" ] || die "--prepare: $checkpoint is not there: run the input step first (launch.sbatch --inputs) and wait for it. Nothing was submitted."
        REGBENCH_MANIFEST="$REGBENCH_INPUTS/rounds.csv"
        echo "# the rounds: $checkpoint -> $REGBENCH_MANIFEST (outlines traced into $REGBENCH_INPUTS/outlines by the nuclei step)"
        regbench_exec "$REGBENCH_HEAD_IMAGE" sh -c "$REGBENCH_PYTHON" regbench-python \
            -m benchmarks.regbench.manifest --preprocessed "$checkpoint" \
            --outlines "$REGBENCH_INPUTS/outlines" --out "$REGBENCH_MANIFEST" \
            || die "--prepare: the rounds could not be written from $checkpoint (see above). Nothing was submitted."
    else
        echo "# the rounds: REGBENCH_MANIFEST=$REGBENCH_MANIFEST, taken as it is"
    fi
    [ -s "$REGBENCH_MANIFEST" ] || die "--prepare: the rounds file $REGBENCH_MANIFEST is not there. Nothing was submitted."
    fi
    force=()
    [ "${REGBENCH_PREPARE_FORCE:-0}" = 1 ] && force=(--force)
    alloc=(--cpus-per-task="$CPUS" --mem="${REGBENCH_PREPARE_MEM:-64G}"
           --time="$(limit "$prep_s")")
    export REGBENCH_PREPARE_IMAGE="$REGBENCH_SCORER_IMAGE"
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
                    # and what the manifest dropped (manifest_dropped.csv, beside it):
                    # attrition.csv's checkpoint_rounds and rows of losses.csv (S7)
                    dropped_of="$(dirname "$REGBENCH_MANIFEST")/manifest_dropped.csv"
                    if [ -f "$dropped_of" ] && ! [ "$dropped_of" -ef "$REGBENCH_CASES/manifest_dropped.csv" ]; then
                        cp "$dropped_of" "$REGBENCH_CASES/manifest_dropped.csv" \
                            || echo "# WARNING --prepare: cannot copy $dropped_of to $REGBENCH_CASES: the manifest's drops are then read from beside the manifest only"
                    fi
                fi
                # the nuclei and the outlines of every native slide first: StarDist on
                # each slide, its cell mask traced to the outlines (nuclei.sbatch).
                # Without outlines there is no case: this step is never left out
                [ -z "${REGBENCH_NUCLEI_SKIP:-}" ] || echo "# REGBENCH_NUCLEI_SKIP is no longer read: the nuclei step traces the outlines every case needs"
                : "${REGBENCH_NUCLEI:=$REGBENCH_CASES/_nuclei}"
                export REGBENCH_NUCLEI REGBENCH_MANIFEST
                # the model and settings of the re-run, and where each is from: the
                # environment, else the site config (conf/ieo.config, REGBENCH_SITE_CONFIG)
                stardist="$(regbench_py "$REGBENCH_HEAD_IMAGE" nuclei --manifest "$REGBENCH_MANIFEST" \
                    --out "$REGBENCH_NUCLEI" --settings)" || die "--prepare: the nuclei and outlines need the site's StarDist model (the pipeline's segmentation_model and segmentation_model_dir): export REGBENCH_STARDIST_MODEL and REGBENCH_STARDIST_MODEL_DIR, or point REGBENCH_SITE_CONFIG at the site config that sets them (default: conf/ieo.config of this checkout). Nothing was submitted."
                while IFS= read -r said; do echo "# StarDist re-run: $said"; done <<< "$stardist"
                nuc_gpu=()
                if [ "${REGBENCH_STARDIST_CPU:-0}" = 0 ]; then
                    [ -n "${REGBENCH_SBATCH_GPU:-}" ] || die "--prepare: the StarDist re-run of the bare nuclei asks for a GPU, as the pipeline does (seg_gpu), and REGBENCH_SBATCH_GPU is not set. Export the site's request, or REGBENCH_STARDIST_CPU=1 to run it on CPUs. Nothing was submitted."
                    nuc_gpu=("${GPU_ARGS[@]}")
                fi
                if [ -z "$(_regbench_var SIF STARDIST)$(_regbench_var CONDA STARDIST)$(_regbench_var VENV STARDIST)" ]; then
                    if [ "$DRY" = 1 ] || [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" = 1 ]; then
                        echo "# not set: REGBENCH_SIF_STARDIST (the StarDist re-run would use the python on PATH)"
                    else
                        die "not set: REGBENCH_SIF_STARDIST (the pipeline's StarDist image, bolt3x/mirage-stardist:1.0.0), needed to segment the slides. Nothing was submitted."
                    fi
                fi
                n_slides="$(regbench_py "$REGBENCH_HEAD_IMAGE" nuclei --manifest "$REGBENCH_MANIFEST" \
                    --out "$REGBENCH_NUCLEI" --count)" || die "cannot size the nuclei array from $REGBENCH_MANIFEST"
                nuc_force=()
                if [ "${REGBENCH_NUCLEI_FORCE:-0}" = 1 ]; then
                    echo "# REGBENCH_NUCLEI_FORCE=1: every slide is segmented and traced again (--export=ALL carries it to the array)"
                    nuc_force=(--force)
                fi
                # only the slides that have anything to do (fleet wave 2, F9): one that
                # has usable nuclei and its outlines would hold a GPU and 128 GB only
                # to print "nothing to segment". Each task still decides for itself.
                pending="$(regbench_py "$REGBENCH_HEAD_IMAGE" nuclei --manifest "$REGBENCH_MANIFEST" \
                    --out "$REGBENCH_NUCLEI" --pending ${nuc_force[@]+"${nuc_force[@]}"})" \
                    || die "cannot read which slides need the nuclei step from $REGBENCH_NUCLEI (nuclei --pending failed above). Nothing was submitted."
                pending="$(awk 'NF { last = $0 } END { print last }' <<< "$pending")"
                if [ -z "$pending" ]; then
                    echo "# nuclei: all $n_slides slide(s) have usable nuclei and their outlines (nuclei --pending): no nuclei job"
                else
                    echo "# nuclei: slide(s) $pending of $n_slides (nuclei --pending)"
                    rung_check
                    # THE NUCLEI STEP HAS A MEMORY LADDER (fleet wave 2, F7): a task whose
                    # StarDist run the kernel killed for memory says so on its end line
                    # (nuclei.sbatch: wants=climb), and the chain's watcher submits the
                    # next rung for those slides only. The pipeline's SEGMENT block is
                    # 128 GB x task.attempt (conf/modules.config).
                    nuc_mem="${REGBENCH_NUCLEI_MEM:-128G}"
                    nuc_alloc=(--cpus-per-task="${REGBENCH_NUCLEI_CPUS:-8}" --time="$(limit "$nuclei_s")"
                               ${nuc_gpu[@]+"${nuc_gpu[@]}"})
                    case "$nuc_mem" in
                        *[0-9]G) nuc_gb="${nuc_mem%G}" ;;
                        *) nuc_gb="" ;;
                    esac
                    [ -z "${nuc_gb//[0-9]/}" ] || nuc_gb=""
                    nuc_on_gpu=0
                    [ "${#nuc_gpu[@]}" -eq 0 ] || nuc_on_gpu=1
                    # its FIRST request too (T5): one no node can grant pends for ever
                    if [ -n "$nuc_gb" ] && ! grantable "$nuc_gb" "$nuc_on_gpu"; then
                        die "--prepare: the nuclei array asks for REGBENCH_NUCLEI_MEM=$nuc_mem per slide, and $GRANT_SAID: it could never start. Name a partition with larger nodes (REGBENCH_SBATCH_GPU for the GPU, or -- --partition=...), or lower REGBENCH_NUCLEI_MEM. Nothing was submitted."
                    fi
                    # on a GPU, each task holds one: the GPU limit
                    nuc_limit="$REGBENCH_ARRAY_LIMIT"
                    [ "$nuc_on_gpu" = 0 ] || nuc_limit="$REGBENCH_GPU_ARRAY_LIMIT"
                    submit "rb-nuclei" "${COMMON[@]}" "$LOG_ARRAY" --mem="$nuc_mem" "${nuc_alloc[@]}" \
                        --array="$pending%$nuc_limit" "$HERE/nuclei.sbatch"
                    nuc_first="$JOB_ID"
                    mkdir -p "$REGBENCH_OUT/logs/retry/$REGBENCH_SUBMISSION" 2>/dev/null \
                        || die "cannot write the retry plans under $REGBENCH_OUT/logs/retry. Nothing more was submitted."
                    plan_open rb-nuclei "$nuc_limit"
                    if [ -z "$nuc_gb" ]; then
                        echo "# nuclei: REGBENCH_NUCLEI_MEM=$nuc_mem is not <N>G: no memory ladder above it"
                    else
                        for gb in $NUCLEI_RUNGS_GB; do
                            [ "$gb" -gt "$nuc_gb" ] || continue
                            if ! grantable "$gb" "$nuc_on_gpu"; then
                                plan_cut "rb-nuclei-${gb}g" "${COMMON[@]}" "$LOG_ARRAY" --mem="${gb}G" \
                                    "${nuc_alloc[@]}" "$HERE/nuclei.sbatch"
                                break
                            fi
                            plan_step climb "rb-nuclei-${gb}g" "${COMMON[@]}" "$LOG_ARRAY" --mem="${gb}G" \
                                "${nuc_alloc[@]}" "$HERE/nuclei.sbatch"
                        done
                    fi
                    plan_watch rb-nuclei "$nuc_first" rb-nuclei "$pending" "" -
                    # afterany: a slide whose run failed has no outlines, and `prepare
                    # multiplex` drops its rounds with the reason its task wrote down
                    after=(--dependency="afterany:$JOB_ID")
                fi
                what+=(--nuclei-dir "$REGBENCH_NUCLEI")
                [ -n "${REGBENCH_NUCLEI_FALLBACK:-}" ] && what+=(--nuclei-fallback "$REGBENCH_NUCLEI_FALLBACK")
                ;;
            synthetic)
                # THE SYNTHETIC SET (T32, synth.sbatch): generated once and shared by every
                # run. A complete set is reused; a planned one is continued (its array
                # keeps every slide already written); none is planned from the source
                # slide first. Each step waits (afterany) for the one before it: a step
                # whose input is missing stops on it, naming it.
                syn_to="$(regbench_cases_of synthetic)"
                syn_grid="${REGBENCH_SYNTH_GRID:-default}"
                case "$syn_grid" in default|small) ;; *) die "REGBENCH_SYNTH_GRID='$syn_grid': default or small. Nothing more was submitted." ;; esac
                # read in a dry run too: it writes nothing
                syn_state="$(regbench_py "$REGBENCH_HEAD_IMAGE" synth --to "$syn_to" --status)" \
                    || die "--prepare --datasets synthetic: cannot read the state of $syn_to/synthetic (synth --status). Nothing more was submitted."
                syn_state="$(awk 'NF { last = $0 } END { print last }' <<< "$syn_state")"
                syn_after=()
                case "$syn_state" in
                    complete)
                        echo "# synthetic: the set under $syn_to/synthetic is complete: reused, nothing to make"
                        continue
                        ;;
                    partial)
                        die "--prepare --datasets synthetic: $syn_to/synthetic exists without its plan (synth_plan.json under _slides): not a set this launcher made or can continue. Move it aside by hand. Nothing more was submitted."
                        ;;
                    planned)
                        echo "# synthetic: the set under $syn_to/synthetic is planned, not complete: its array continues it (slides already written are kept)"
                        ;;
                    absent)
                        [ -n "${REGBENCH_SYNTH_SOURCE:-}" ] || die "--prepare --datasets synthetic needs REGBENCH_SYNTH_SOURCE: a cohort slide (OME-TIFF) whose nuclear channel is the texture, under a folder the images bind (REGBENCH_INPUTS, REGBENCH_CASES or REGBENCH_BIND). Nothing more was submitted."
                        [ -f "$REGBENCH_SYNTH_SOURCE" ] || die "--prepare --datasets synthetic: REGBENCH_SYNTH_SOURCE=$REGBENCH_SYNTH_SOURCE is not a file. Nothing more was submitted."
                        # the plan job reads it INSIDE the image: its realpath must lie under
                        # a folder regbench_exec binds (env.sh)
                        syn_real="$(real_file "$REGBENCH_SYNTH_SOURCE")" \
                            || die "--prepare --datasets synthetic: cannot resolve REGBENCH_SYNTH_SOURCE=$REGBENCH_SYNTH_SOURCE. Nothing more was submitted."
                        syn_bound=0
                        IFS=, read -r -a syn_dirs <<< "${REGBENCH_INPUTS:-},${REGBENCH_CASES:-},${REGBENCH_BIND:-}"
                        for syn_dir in "${syn_dirs[@]}"; do
                            syn_dir="${syn_dir%%:*}"
                            [ -n "$syn_dir" ] && [ -d "$syn_dir" ] || continue
                            syn_dir="$(cd "$syn_dir" && pwd -P)"
                            case "$syn_real" in "${syn_dir%/}/"*) syn_bound=1 ;; esac
                        done
                        [ "$syn_bound" = 1 ] || die "--prepare --datasets synthetic: REGBENCH_SYNTH_SOURCE=$REGBENCH_SYNTH_SOURCE (really $syn_real) is under none of the folders the images bind (REGBENCH_INPUTS, REGBENCH_CASES, REGBENCH_BIND): the plan job could not read it. Put it under one, or name its folder in REGBENCH_BIND. Nothing more was submitted."
                        # handed over as its realpath, the path the check above held
                        # bound: a link that only looks bound never reaches the image
                        syn_plan=(--source "$syn_real" --grid "$syn_grid" --seed "${REGBENCH_SYNTH_SEED:-0}")
                        [ -z "${REGBENCH_SYNTH_CHANNEL:-}" ] || syn_plan+=(--channel "$REGBENCH_SYNTH_CHANNEL")
                        submit "rb-synth-plan" "${COMMON[@]}" "$LOG_ONE" --cpus-per-task=1 --mem=16G \
                            --time="$(limit "$synth_plan_s")" "$HERE/synth.sbatch" plan "$syn_to" "${syn_plan[@]}"
                        # afterok: an array over a plan that failed would only fail 55 times
                        syn_after=(--dependency="afterok:$JOB_ID")
                        ;;
                    *) die "--prepare --datasets synthetic: synth --status said '$syn_state'. Nothing more was submitted." ;;
                esac
                syn_n="$(regbench_py "$REGBENCH_HEAD_IMAGE" synth --to "$syn_to" --grid "$syn_grid" --units)" \
                    || die "--prepare --datasets synthetic: cannot size its array (synth --units). Nothing more was submitted."
                syn_n="$(awk 'NF { last = $0 } END { print last }' <<< "$syn_n")"
                # one task per (patient, panel): a 65536 px panel is ~45 min on one CPU
                # (measured: ~2.7 min and a 1.6 GB peak per 16384 px moving panel), its
                # memory the memory-mapped texture and a few tiles
                submit "rb-synth" "${COMMON[@]}" "$LOG_ARRAY" ${syn_after[@]+"${syn_after[@]}"} \
                    --array="0-$((syn_n - 1))%$REGBENCH_ARRAY_LIMIT" --cpus-per-task=1 \
                    --mem="${REGBENCH_SYNTH_MEM:-8G}" --time="$(limit "$synth_s")" \
                    "$HERE/synth.sbatch" unit "$syn_to"
                submit "rb-synth-finish" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$JOB_ID" \
                    --cpus-per-task=1 --mem=4G --time="$(limit "$synth_finish_s")" \
                    "$HERE/synth.sbatch" finish "$syn_to"
                continue
                ;;
            hyreco)
                # ONE job: subset A and, when it was downloaded, subset B into
                # REGBENCH_CASES_HYRECO (prepare.sbatch binds that folder); the data root
                # is read inside the image, so it is bound through REGBENCH_BIND, which
                # --export=ALL carries to the job
                what=(hyreco --data-root "$REGBENCH_HYRECO_DATA")
                export REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$REGBENCH_HYRECO_DATA"
                echo "# hyreco: prepared from REGBENCH_HYRECO_DATA=$REGBENCH_HYRECO_DATA (bound into the job's image through REGBENCH_BIND) into $(regbench_cases_of hyreco)"
                ;;
            *) die "--prepare: '$ds' is not multiplex, synthetic or hyreco (this launcher prepares the cohort, the synthetic set and the landmark set only)" ;;
        esac
        submit "rb-prepare-$ds" "${COMMON[@]}" "$log" "${alloc[@]}" ${array[@]+"${array[@]}"} \
            ${after[@]+"${after[@]}"} "$HERE/prepare.sbatch" "${what[@]}" ${force[@]+"${force[@]}"}
    done
    echo "prepare: $N_JOBS job(s). When they are done, run this script again without --prepare."
    FINISHED=1
    exit 0
fi

# ── refusals, before anything is submitted ────────────────────────────────────────────
# an OWN_LOCAL run is a 4-CPU cost row (its options name workers=4): at another ladder
# CPU count it would not be what it says, so it is dropped (the CPU cross runs the
# families' rows, not it)
if [ "$LADDER_CPUS" != 4 ]; then
    for r in $OWN_LOCAL; do
        drop_run "$r" run_config "it is STARE at 4 CPUs in one job and REGBENCH_CPUS=$LADDER_CPUS"
    done
    none_left "REGBENCH_CPUS=$LADDER_CPUS"
fi
gpu_runs="" images=""
while IFS=$'\t' read -r name _ _ _ image gpu _; do
    has "$selected" "$name" || continue
    [ "$gpu" = 1 ] && gpu_runs="$gpu_runs $name"
    has "$images" "$image" || images="$images $image"
done < "$union"
if [ -n "$gpu_runs" ] && [ -z "${REGBENCH_SBATCH_GPU:-}" ]; then
    # B4: each such run is dropped, the others go on
    for r in $gpu_runs; do
        drop_run "$r" run_config "it needs a GPU and REGBENCH_SBATCH_GPU is empty (export the site's request, e.g. REGBENCH_SBATCH_GPU=\"--gres=gpu:nvidia_h200:1\")"
    done
    none_left "no GPU request"
fi
missing=""
for image in $images $REGBENCH_SCORER_IMAGE; do
    [ -n "$(_regbench_var SIF "$image")$(_regbench_var CONDA "$image")$(_regbench_var VENV "$image")" ] \
        || has "$missing" "REGBENCH_SIF_$image" || missing="$missing REGBENCH_SIF_$image"
done
if [ -n "$missing" ]; then
    if [ "$DRY" = 1 ] || [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" = 1 ]; then
        echo "# not set:$missing (those jobs would use the python on PATH)"
    elif has "$missing" "REGBENCH_SIF_$REGBENCH_SCORER_IMAGE"; then
        die "integrity refusal (the scorer's image): not set:$missing. The scorer's image needs its path (or REGBENCH_ALLOW_PATH_PYTHON=1 to use the python on PATH): nothing could be scored. Nothing was submitted."
    else
        # B4: the runs of an image with no path are dropped, the others go on
        while IFS=$'\t' read -r name _ _ _ image _; do
            has "$missing" "REGBENCH_SIF_$image" && drop_run "$name" run_config "not set: REGBENCH_SIF_$image (its image needs a path, or REGBENCH_ALLOW_PATH_PYTHON=1 to use the python on PATH)"
        done < "$union"
        none_left "images with no path"
        images=""
        while IFS=$'\t' read -r name _ _ _ image _; do
            has "$selected" "$name" && ! has "$images" "$image" && images="$images $image"
        done < "$union"
    fi
fi

# ── the cases: every array is sized from what is prepared ─────────────────────────────
sized=""
for ds in $datasets; do
    ds_cases="$(regbench_cases_of "$ds")"
    read -r n_cases n_groups max_rounds < <(case_sizes "$ds" "$ds_cases") \
        || die "cannot list the $ds cases under $ds_cases"
    if [ "${n_cases:-0}" -eq 0 ]; then
        echo "# $ds: no prepared cases under $ds_cases, left out (submit_cohort.sh --prepare)"
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
# THE SYNTHETIC SET'S RESULTS DISK (final review I1), said before the first sbatch and
# never a refusal: STARE writes every moving panel as an uncompressed single-level round,
# about 343 GB per STARE run on the default grid (synth.results_estimate, from the sizes
# in code), beside the free space of REGBENCH_OUT (its nearest folder that exists)
if has "$sized" synthetic && [ -n "${selected// /}" ] && [ "$scoring_earlier" != 1 ]; then
    if syn_res="$(regbench_py "$REGBENCH_HEAD_IMAGE" synth --to "$(regbench_cases_of synthetic)" \
            --results-estimate --runs "${selected# }")"; then
        syn_res="$(awk 'NF { last = $0 } END { print last }' <<< "$syn_res")"
        echo "# synthetic: ${syn_res#synth: }"
    else
        echo "# synthetic: results estimate: cannot be read (synth --results-estimate failed above)"
    fi
    syn_df="$REGBENCH_OUT"
    while [ ! -d "$syn_df" ] && [ "$syn_df" != / ] && [ "$syn_df" != . ]; do syn_df="$(dirname "$syn_df")"; done
    if syn_kb="$(df -Pk "$syn_df" 2>/dev/null | awk 'NR == 2 { print $4 }')" && [ -n "$syn_kb" ]; then
        echo "# synthetic: free space of REGBENCH_OUT ($REGBENCH_OUT, on $syn_df): $((syn_kb * 1024)) bytes ($(awk -v k="$syn_kb" 'BEGIN { printf "%.1f", k * 1024 / 1e9 }') GB)"
    else
        echo "# synthetic: free space of REGBENCH_OUT ($REGBENCH_OUT): cannot be read (df $syn_df)"
    fi
fi

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
    local seen="" failed=0 name method image out code key where=() slides=() ds failed_keys="" fail_said
    PF_KEYS_SAID=""
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
    for ds in $sized; do
        if [ "$ds" = multiplex ]; then
            slides+=(--dataset "$ds")
        else
            echo "# preflight: the slides of $ds are not checked (its cases are under $(regbench_cases_of "$ds"), the check reads REGBENCH_CASES)"
        fi
    done
    while IFS=$'\t' read -r name method _ _ image _; do
        has "$selected" "$name" || continue
        key="$image:$method"
        has "$seen" "$key" && continue
        seen="$seen $key"
        out="$(regbench_py "$image" preflight --method "$method" --image "$image" "${where[@]}" "${slides[@]}" 2>&1)" && code=0 || code=$?
        preflight_lines "$out"
        if [ "$code" != 0 ]; then
            grep -q '^preflight FAIL \[' <<< "$out" \
                || echo "preflight FAIL [$image $method] the image did not start, or has neither python nor python3, or the harness does not import there (exit $code): $(tail -n 3 <<< "$out" | tr '\n' ' ')"
            failed_keys="$failed_keys $key"
            fail_said="$(grep -E '^preflight FAIL \[' <<< "$out" | head -n 3 | tr '\n' ' ')" || fail_said=""
            printf -v "PF_SAID_${#failed_keys}" '%s' "${fail_said:-the image did not start, or the harness does not import there (exit $code)}"
            PF_KEYS_SAID="$PF_KEYS_SAID $key=${#failed_keys}"
        fi
    done < "$union"
    # B3: the runs of an (image, method) that failed are dropped; the others go on
    if [ -n "$failed_keys" ]; then
        local said_of n
        while IFS=$'\t' read -r name method _ _ image _; do
            has "$selected" "$name" || continue
            has "$failed_keys" "$image:$method" || continue
            n="$(tr ' ' '\n' <<< "$PF_KEYS_SAID" | sed -n "s/^$image:$method=//p" | head -n 1)" || n=""
            said_of="PF_SAID_$n"
            drop_run "$name" preflight_failed "[$image $method] ${!said_of:-the preflight failed}"
        done < "$union"
    fi
    image="$REGBENCH_SCORER_IMAGE"
    out="$(regbench_py "$image" preflight --scorer --image "$image" --writable "$REGBENCH_OUT" 2>&1)" && code=0 || code=$?
    preflight_lines "$out"
    if [ "$code" != 0 ]; then
        failed=1
        grep -q '^preflight FAIL \[' <<< "$out" \
            || echo "preflight FAIL [$image the scorer] the image did not start, or has neither python nor python3 (exit $code): $(tail -n 3 <<< "$out" | tr '\n' ' ')"
    fi
    [ "$failed" = 0 ] || die "integrity refusal (the scorer's image): the scorer's image failed the preflight (the 'preflight FAIL' lines above): nothing could be scored. Nothing was submitted."
    if [ "$preflight_only" != 1 ]; then
        none_left "the preflight"
    fi
    if [ -n "$failed_keys" ]; then
        echo "# preflight: failed for$failed_keys (the 'preflight FAIL' lines above); their runs are dropped:$DROPPED_RUNS"
        [ "$preflight_only" = 1 ] && return 1
        return 0
    fi
    echo "# preflight: ok"
}

if [ "$preflight_only" = 1 ]; then
    rung_check
    fetch_stare_source
    if ! preflight; then
        echo "# --preflight: nothing was submitted"
        exit 2
    fi
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
# a dry run asks too: it reads the scheduler and submits nothing
rung_check
# a run's own first rung (REGBENCH_FIRST_RUNG) must be one a node can grant: a ladder
# whose first rung is cut would submit nothing for that run
for r in $selected; do
    gb="$(first_rung_of "$r")"
    [ "$gb" = "${RUNGS_GB%% *}" ] && continue
    grantable "$gb" || die "REGBENCH_FIRST_RUNG: $r starts at $gb GB, more than $GRANT_SAID. Nothing was submitted."
    echo "# memory ladder: $r starts at $gb GB (REGBENCH_FIRST_RUNG): $(rungs_of "$r") GB"
done

# ── everything else that can stop this script, read before the first sbatch ───────────
# the pipeline's allocation per stage and attempt, for the runs that are several jobs
# per round: ALLOC_<stage>_<attempt>="<cpus> <GB> <time>"
any_staged=0
STITCH_S=0
while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ realcheck stages; do
    has "$selected" "$name" || continue
    [ "${stages:--}" = - ] && continue
    any_staged=1
    if [ "${realcheck:--}" != - ]; then
        drop_run "$name" run_config "it runs as several jobs per round and has a realcheck: the launcher runs no first unit with --realcheck for such a run"
        continue
    fi
    for stage in ${stages//,/ }; do
        case "$stage" in coarse|tile|solve|stitch) ;; *) drop_run "$name" run_config "no job is defined for the stage '$stage' (column stages of the run list)"; continue 2 ;; esac
    done
done < "$union"
none_left "their configuration"
any_staged=0
while IFS=$'\t' read -r name _ _ _ _ _ _ _ _ _ _ _ _ _ _ stages; do
    has "$selected" "$name" && [ "${stages:--}" != - ] && any_staged=1
done < "$union"
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
        layout="$REGBENCH_OUT/logs/stages/$REGBENCH_SUBMISSION/$name-$ds.tsv"
        made="$layout"
        if [ "$DRY" = 1 ]; then
            made="$scratch/stages/$name-$ds.tsv"
            layout="$REGBENCH_OUT/logs/stages/<submission>/$name-$ds.tsv"
        fi
        what=(stages --cases "$(regbench_cases_of "$ds")" --dataset "$ds" --method "$method" --layout "$made")
        [ "$label" != - ] && what+=(--label "$label")
        if [ "$opts" != - ]; then
            IFS=, read -ra items <<< "$opts"
            for o in "${items[@]}"; do what+=(--opt "$o"); done
        fi
        if ! read -r n_rounds n_tiles < <(regbench_py "$image" "${what[@]}"); then
            drop_run "$name" run_config "cannot size its tile arrays on $ds (python -m benchmarks.regbench stages)"
            continue
        fi
        case "${n_rounds:-}:${n_tiles:-}" in *[!0-9:]*|:*|*:)
            drop_run "$name" run_config "cannot size its tile arrays on $ds (python -m benchmarks.regbench stages printed '${n_rounds:-} ${n_tiles:-}')"
            continue ;;
        esac
        if ! { [ "$n_tiles" -gt 0 ] && [ "$n_rounds" -gt 0 ]; }; then
            drop_run "$name" run_config "no tile to register on $ds (python -m benchmarks.regbench stages printed '$n_rounds $n_tiles')"
            continue
        fi
        # a run's datasets follow one another, so it is ONE submitter's worth of queue
        case ",$stages," in *,tile,*) has "$TILE_RUNS" "$name" || { TILE_RUNS="$TILE_RUNS $name"; N_TILE_RUNS=$((N_TILE_RUNS + 1)); } ;; esac
        if [ "$DRY" != 1 ]; then
            # the caller's own sbatch arguments, for the jobs the submitter submits
            : > "${made%.tsv}.sbatch_args"
            [ ${#extra[@]} -gt 0 ] && printf '%s\n' "${extra[@]}" > "${made%.tsv}.sbatch_args"
        fi
        printf -v "LAYOUT_${ds}_$name" '%s' "$layout"
        printf -v "ROUNDS_OF_${ds}_$name" '%s' "$n_rounds"
        printf -v "TILES_OF_${ds}_$name" '%s' "$n_tiles"
    done < "$(table_of "$ds")"
done
none_left "their tile arrays"
# a run dropped at sizing on a LATER dataset was added above on an earlier one: the
# runs that feed the tile queue are rebuilt from the runs left (review M1)
kept_tile_runs="" N_TILE_RUNS=0
for name in $TILE_RUNS; do
    has "$selected" "$name" || continue
    kept_tile_runs="$kept_tile_runs $name"
    N_TILE_RUNS=$((N_TILE_RUNS + 1))
done
TILE_RUNS="$kept_tile_runs"
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
# the runs this submission selected, for the scorer's `regbench verdict`: one line,
# beside the submission's job ids (score_cohort.sbatch reads it)
if [ "$DRY" != 1 ] && [ -n "$selected" ]; then
    echo "${selected# }" > "$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.runs" \
        || die "cannot write $REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.runs. Nothing was submitted."
fi
# ── 1. selfchecks: the runs that check a selected run, and the runs they are anchored on.
# REPORTED, NEVER A GATE: no ladder below waits for one.
needed=""
while IFS=$'\t' read -r name _ _ _ _ _ _ after checked_by _; do
    has "$selected" "$name" || continue
    has "$needed" "$checked_by" || needed="$needed $checked_by"
done < "$union"
# a checking run that reads another's results needs that one's selfcheck too
while IFS=$'\t' read -r name _ _ _ _ _ _ after _; do
    has "$needed" "$name" || continue
    [ "$after" = - ] || has "$needed" "$after" || needed="$needed $after"
done < "$union"

# THE PEAK, before the first sbatch (REGBENCH_PEAK=1: launch.sbatch --all at its launch
# stage). What can run at once, from the settings and the run list as selected: the
# REGBENCH_RUN_LANES runs in flight (all of them at 0) are taken as the ones whose arrays
# hold the most, each holding its first-rung array at its limit (REGBENCH_GPU_ARRAY_LIMIT
# for a run that asks for a GPU, else REGBENCH_ARRAY_LIMIT; one task more for a run with
# a realcheck, whose own ladder is one unit); every selfcheck (one job each, outside
# the lanes); the tile tasks (the lanes' shares of REGBENCH_STAGE_QUEUE_MAX). A bound,
# never an estimate: an array of fewer units holds fewer.
if [ "${REGBENCH_PEAK:-0}" = 1 ] && [ -n "${selected// /}" ]; then
    caps="" n_sel=0
    while IFS=$'\t' read -r name _ _ _ _ gpu _ _ _ _ _ _ _ _ realcheck _; do
        has "$selected" "$name" || continue
        n_sel=$((n_sel + 1))
        cap="$REGBENCH_ARRAY_LIMIT"
        [ "$gpu" != 1 ] || cap="$REGBENCH_GPU_ARRAY_LIMIT"
        [ "${realcheck:--}" = - ] || cap=$((cap + 1))
        caps="$caps$cap $gpu"$'\n'
    done < "$union"
    in_flight="$n_sel"
    [ "$REGBENCH_RUN_LANES" -gt 0 ] && [ "$REGBENCH_RUN_LANES" -lt "$n_sel" ] && in_flight="$REGBENCH_RUN_LANES"
    # (awk reads to the end: no head, whose early exit would fail the pipe under pipefail)
    p_ladder="$(sort -k1,1nr <<< "$caps" | awk -v n="$in_flight" 'NF && ++k <= n { t += $1 } END { print t + 0 }')"
    # the GPUs: when the runs in flight are GPU runs (at most the lanes of them)
    p_gpu="$(awk '$2 == 1' <<< "$caps" | sort -k1,1nr | awk -v n="$in_flight" 'NF && ++k <= n { t += $1 } END { print t + 0 }')"
    p_gpu_runs="$(awk '$2 == 1 { k++ } END { print k + 0 }' <<< "$caps")"
    p_sc=0 p_sc_gpu=0
    while IFS=$'\t' read -r name _ _ _ _ gpu _; do
        has "$needed" "$name" || continue
        p_sc=$((p_sc + 1))
        [ "$gpu" != 1 ] || p_sc_gpu=$((p_sc_gpu + 1))
    done < "$union"
    p_tiles=0 p_tile_cpus=0 p_tile_gb=0
    if [ "$N_FEEDING" -gt 0 ] && [ -n "${ALLOC_tile_1:-}" ]; then
        p_tiles=$((N_FEEDING * STAGE_QUEUE_RUN))
        read -r p_tile_cpus p_tile_gb _ <<< "$ALLOC_tile_1"
    fi
    p_tasks=$((p_ladder + p_sc + p_tiles))
    p_cpus=$((p_ladder * LADDER_CPUS + p_sc * CPUS + p_tiles * p_tile_cpus))
    p_gb=$((p_ladder * ${RUNGS_GB%% *} + p_sc * ${SELFCHECK_RUNGS_GB%% *} + p_tiles * p_tile_gb))
    echo "# peak: at most $p_tasks task(s) at once = $p_ladder ladder task(s) ($in_flight of $n_sel run(s) in flight, REGBENCH_RUN_LANES=$REGBENCH_RUN_LANES, at most REGBENCH_ARRAY_LIMIT=$REGBENCH_ARRAY_LIMIT each, a GPU run's at most REGBENCH_GPU_ARRAY_LIMIT=$REGBENCH_GPU_ARRAY_LIMIT, one more with a realcheck) + $p_sc selfcheck(s) (outside the lanes) + $p_tiles tile task(s) (REGBENCH_STAGE_QUEUE_MAX=$REGBENCH_STAGE_QUEUE_MAX over $N_FEEDING submitter(s)): $p_cpus CPUs, at most $((p_gpu + p_sc_gpu)) GPU(s) ($p_gpu ladder task(s) when the runs in flight are GPU runs, of $p_gpu_runs, + $p_sc_gpu GPU selfcheck(s)), $p_gb GB at the first rung (${RUNGS_GB%% *} GB a ladder task, ${SELFCHECK_RUNGS_GB%% *} GB a selfcheck, ${p_tile_gb} GB a tile task); plus the small jobs (each chain's watcher link and each tile submitter: 1 CPU and 1 GB)"
fi

# The chains' retry plans (plan_open, above) are written while the jobs are submitted.
# Their folder is made HERE, where one that cannot be written still stops the launch
# with nothing in the queue.
retry_root="$REGBENCH_OUT/logs/retry/$REGBENCH_SUBMISSION"
[ "$DRY" = 1 ] && retry_root="$scratch/retry"
mkdir -p "$retry_root" 2>/dev/null && [ -w "$retry_root" ] \
    || die "cannot write the retry plans to $retry_root. Nothing was submitted."
sc_time="$(limit "$sc_s")"
while IFS=$'\t' read -r name method label opts image gpu _ after _ mask _; do
    has "$needed" "$name" || continue
    gres=()
    [ "$gpu" = 1 ] && gres=(${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
    base="" anchor=-
    if [ "$after" != - ]; then
        # it reads the anchor's selfcheck work (`--m0-from`): after it, WHATEVER the
        # anchor's selfcheck said (afterany): one selfcheck's verdict never stops another
        anchor="$after"
        last="SC_LAST_$after"
        base="afterany:${!last}"
    fi
    # the first rung is submitted; the others are steps of the selfcheck's watcher,
    # submitted only when the attempt before did not end ok (and such a job still runs
    # the selfcheck only when that attempt ended for memory).
    attempt=0 first_id="" first_name=""
    plan_open "rb-sc-$name"
    for gb in $SELFCHECK_RUNGS_GB; do
        attempt=$((attempt + 1))
        # a rung no node can grant: the selfcheck stops below it (it reports, and its
        # record says what the last attempt left)
        if [ "$attempt" -gt 1 ] && ! grantable "$gb" "$gpu"; then
            [ "$DRY" = 1 ] && echo "# NOT PLANNED, no node of $GRANT_AT can grant it ($GRANT_AT_MB MB): rb-sc-$name-${gb}g"
            break
        fi
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
done < "$union"

# the small jobs: the settle job. One sacct over this submission's run jobs and their
# steps (score_cohort.sbatch: settle_ids), then one pass over the run.json files: hours
# at most, never the day-scale scorer that would settle again if this one timed out
small=(--cpus-per-task=1 --mem=4G --time="$(limit "$settle_s")")

# ── 3 and 4. the ladders ──────────────────────────────────────────────────────────────
# deps_and <a> <b>: two --dependency lists joined with `,` (all of them), either empty
deps_and() {
    if [ -z "$1" ]; then echo "$2"; elif [ -z "$2" ]; then echo "$1"; else echo "$1,$2"; fi
}

# ladder <job prefix> <array range> <realcheck 0|1> <dependencies|""> <time>: the first
# rung's array, then the plan of the others (per rung the attempt, then the same rung
# again for a unit the scheduler lost) and the watcher; sets LADDER_LAST to the watcher.
# Reads the current row's variables. Every later task is given the ids of all the
# ladder's earlier jobs: it asks the scheduler what became of its unit in each.
# With a sixth argument `jobs` (a run whose registration is several jobs per round) this
# is each round's LAST job: it holds the benchmark's own work (the points, the tissue
# reduction), so it has the benchmark's allocation, the same ladder, and inside it the
# method's writer runs as a process of its own with the pipeline's time limit and
# retries (methods/stare.py: write_native). Its time is one round's plus the writer's
# four attempts. With `local` (REGBENCH_STARE_EXECUTION=local) it is such a run as one
# job per unit: the task starts the stages itself (run_rung.sbatch: `run --stages
# local`), with a plain round's time.
ladder() {
    local prefix="$1" range="$2" real="$3" base="$4" time="$5" staged="${6:--}"
    local attempt=0 gb task jobname cpus="$LADDER_CPUS" rungs first_id="" first_name=""
    rungs="$(rungs_of "$name")"
    # a run that asks for a GPU: every task holds one, so its arrays have the GPU limit
    local limit="$REGBENCH_ARRAY_LIMIT"
    [ "$gpu" != 1 ] || limit="$REGBENCH_GPU_ARRAY_LIMIT"
    # STITCH_S: the writer's four attempts, summed before the first sbatch (a `local`
    # run is a plain round: its stages are inside the job, its time a round's)
    [ "$staged" != jobs ] || time="$(limit $((round_s + STITCH_S)))"
    plan_open "$prefix" "$limit"
    # the row this ladder is of, for the harness that reads the chain back (chains.py)
    printf '%s\n' "$name" "$method" "$label" "$opts" "$ds" > "$PLAN/run"
    for gb in $rungs; do
        attempt=$((attempt + 1))
        if ! grantable "$gb" "$gpu"; then
            # rung_check stopped the launch when the first rung is not grantable
            plan_cut "$prefix-${gb}g" "${COMMON[@]}" "$LOG_ARRAY" \
                --cpus-per-task="$cpus" --mem="${gb}G" \
                --time="$time" ${gres[@]+"${gres[@]}"} \
                "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                "$ds" "$attempt" "$gb" "$real" "@EARLIER@" climb "$time" "$staged"
            break
        fi
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
                submit "$jobname" "${COMMON[@]}" "$LOG_ARRAY" ${base:+--dependency="$base"} \
                    --array="$range%$limit" --cpus-per-task="$cpus" --mem="${gb}G" \
                    --time="$time" ${gres[@]+"${gres[@]}"} \
                    "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                    "$ds" "$attempt" "$gb" "$real" - "$task" "$time" "$staged"
                first_id="$JOB_ID" first_name="$jobname"
                continue
            fi
            # every other one is a step of the plan: submitted by the chain's watcher,
            # after a failure, for the units that failed. An `again` task has
            # TIME_FACTOR (2) x the time, capped at the site's longest job: it retries
            # a unit that ran out of time too (ladder.py, ruling 2026-10-21)
            local step_time="$time"
            [ "$task" != again ] || step_time="$(twice "$time")"
            plan_step "$task" "$jobname" "${COMMON[@]}" "$LOG_ARRAY" \
                --cpus-per-task="$cpus" --mem="${gb}G" \
                --time="$step_time" ${gres[@]+"${gres[@]}"} \
                "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                "$ds" "$attempt" "$gb" "$real" "@EARLIER@" "$task" "$step_time" "$staged"
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
#   coarse, solve         attempt 1 as an array over the rounds, and the stage's
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
            submit "rb-$name-$ds-$stage-a$attempt" "${COMMON[@]}" "$LOG_ARRAY" ${dep:+--dependency="$dep"} \
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
            coarse|solve)
                stage_chain "$stage" "0-$((n_rounds - 1))" "$dep"
                dep="$(deps_and "$base" "afterany:$JOB_ID")" ;;
            tile)
                per_attempt=()
                for attempt in 1 2 3 4; do
                    alloc="ALLOC_tile_$attempt"; read -r cpus gb time <<< "${!alloc}"
                    per_attempt+=("$cpus:$gb:$time")
                done
                for link in $(seq 1 "$REGBENCH_STAGE_FEEDERS"); do
                    submit "rb-$name-$ds-feed-$link" "${COMMON[@]}" "$LOG_ONE" ${dep:+--dependency="$dep"} \
                        --cpus-per-task=1 --mem=1G --time="$feed_time" "$HERE/feed_tiles.sbatch" \
                        "$name" "$method" "$label" "$opts" "$image" "$ds" "$layout" \
                        "$n_rounds" "$n_tiles" "$link" "${per_attempt[@]}"
                    dep="$(deps_and "$base" "afterany:$JOB_ID")"
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
WAVE_METHOD="" WAVE_PREV="" WAVE_CUR=""
# wave_add <jobs>: what the next method's wave waits for (REGBENCH_WAVES=1)
wave_add() { [ -z "$1" ] || WAVE_CUR="${WAVE_CUR:+$WAVE_CUR:}$1"; }
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
        # no selfcheck is waited for: reported, never a gate
        base=""
        if [ "$after" != - ]; then
            last="LAST_${ds}_$after"
            if [ -n "${!last:-}" ]; then
                base="$(deps_and "$base" "afterany:${!last}")"
            else
                echo "# $name reads the results of $after, which is not submitted here: they must already be under $REGBENCH_OUT"
            fi
        fi
        # one dataset after the other, per run: the cohort (multiplex) is done before
        # the same run starts on the next dataset
        if [ -n "$prev_ds" ]; then
            last="LAST_${prev_ds}_$name"
            [ -n "${!last:-}" ] && base="$(deps_and "$base" "afterany:${!last}")"
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
            [ -n "${!lane:-}" ] && base="$(deps_and "$base" "afterany:${!lane}")"
        fi
        if [ "$REGBENCH_WAVES" = 1 ]; then
            if [ "$method" != "$WAVE_METHOD" ]; then
                WAVE_PREV="$WAVE_CUR" WAVE_CUR="" WAVE_METHOD="$method"
            fi
            [ -n "$WAVE_PREV" ] && base="$(deps_and "$base" "afterany:$WAVE_PREV")"
        fi
        if [ "${stages:--}" != - ]; then
            staged "$base"
            printf -v "LAST_${ds}_$name" '%s' "$LADDER_LAST"
            wave_add "$LADDER_LAST"
            finals="$finals $LADDER_LAST"
            if [ "$REGBENCH_RUN_LANES" -gt 0 ]; then
                lane="LANE_OF_$name"
                printf -v "LANE_LAST_${!lane}" '%s' "$LADDER_LAST"
            fi
            continue
        fi
        # a run with a realcheck: its first unit of the first dataset runs with
        # --realcheck in a ladder of its own, SUBMITTED BESIDE the rest of the run, which
        # does not wait for it (reported, never a gate). What waits for the run waits for
        # both (LAST_ holds both watchers).
        start=0 ends="" execution=-
        # REGBENCH_STARE_EXECUTION=local: its stages inside the task, the tiles
        # $LADDER_CPUS at a time (STARE's option workers)
        if has "$OWN_LOCAL" "$name"; then
            execution=local  # its own options name its workers
        elif has "$LOCAL_RUNS" "$name"; then
            execution=local
            if [ "$opts" = - ]; then opts="workers=$LADDER_CPUS"; else opts="$opts,workers=$LADDER_CPUS"; fi
        fi
        if [ "${realcheck:--}" != - ] && [ "$ds" = "$first_dataset" ]; then
            ladder "rb-$name-$ds-first" "0-0" 1 "$base" "$time" "$execution"
            finals="$finals $LADDER_LAST"
            ends="$LADDER_LAST"
            start=1
        fi
        if [ "$n" -gt "$start" ]; then
            ladder "rb-$name-$ds" "$start-$((n - 1))" 0 "$base" "$time" "$execution"
            finals="$finals $LADDER_LAST"
            ends="${ends:+$ends:}$LADDER_LAST"
        fi
        [ -n "$ends" ] && printf -v "LAST_${ds}_$name" '%s' "$ends"
        wave_add "$ends"
        last="LAST_${ds}_$name"
        if [ "$REGBENCH_RUN_LANES" -gt 0 ] && [ -n "${!last:-}" ]; then
            lane="LANE_OF_$name"
            printf -v "LANE_LAST_${!lane}" '%s' "${!last}"
        fi
    done < "$(table_of "$ds")"
    prev_ds="$ds"
done

# ── 5. per-case scoring and tissue, then the scorer, in the scorer's image ────────────
SCORER_JOB="" RESCORE_JOB=""
if [ "$compute_only" != 1 ]; then
dep=""
for j in $finals; do dep="${dep:+$dep:}$j"; done
# NOT the selfchecks (B5): the tables wait for the registrations alone, and rb-rescore
# (below) writes them again once every selfcheck's watcher has ended
sc_dep=""
for r in $needed; do
    last="SC_LAST_$r"
    [ -n "${!last:-}" ] && sc_dep="${sc_dep:+$sc_dep:}${!last}"
done
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
# one task per case: its rows, kept as a record the scorer reads. afterany: a task
# that fails or is killed leaves its case to the scorer, which computes what is missing
score_time="$(limit "$score_s")"
scored=""
tissued=0
for ds in $sized; do
    # a cost-only set (the synthetic set) has no per-case rows: its scorer reads the
    # results' resources alone. Nor has a landmark set (hyreco): its scorer reads the
    # landmarks and the results, never a per-case record or a tissue piece
    has "$COST_ONLY_SETS" "$ds" && continue
    has "$LANDMARK_SETS" "$ds" && continue
    tissued=1
    n_cases="CASES_$ds"
    submit "rb-score-case-$ds" "${COMMON[@]}" "$LOG_ARRAY" --dependency="afterany:$settle_job" \
        --array="0-$((${!n_cases} - 1))%$REGBENCH_ARRAY_LIMIT" --cpus-per-task="$REGBENCH_SCORE_CPUS" \
        --mem="$REGBENCH_SCORE_MEM" --time="$score_time" "$HERE/score_cohort.sbatch" case "$ds"
    scored="$scored:$JOB_ID"
done
# each case task also made the tissue pieces of its case: this job makes those of a
# case whose task did not, after them (never beside them: two jobs would be writing
# the same pieces)
# a submission that names a public dataset gives the tissue job and the scorer every
# dataset it sized: each public one is scored alone into tables/<dataset>/ (never
# pooled). The default submission (the cohort alone) names none: the cohort is theirs
scored_sets=()
for ds in $sized; do
    [ "$ds" = multiplex ] || { read -r -a scored_sets <<< "$sized"; break; }
done
# and no tissue job when every set is cost-only or a landmark set: the scorer then waits
# for the settle job
if [ "$tissued" = 1 ]; then
    submit "rb-tissue" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany${scored}" --cpus-per-task="$CPUS" \
        --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 "$HERE/score_cohort.sbatch" tissue \
        ${scored_sets[@]+"${scored_sets[@]}"}
else
    JOB_ID="$settle_job"
fi
submit "rb-score" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$JOB_ID$scored" --cpus-per-task="$CPUS" \
    --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 "$HERE/score_cohort.sbatch" score \
    ${scored_sets[@]+"${scored_sets[@]}"}
SCORER_JOB="$JOB_ID"
# the light re-score (B5): after the scorer and every selfcheck's watcher, settle again,
# then the tables with every verdict (the case records are read: the selfcheck statuses
# are set on the rows at table time, never in a record's key)
if [ -n "$sc_dep" ]; then
    submit "rb-rescore" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$SCORER_JOB:$sc_dep" \
        --cpus-per-task="$CPUS" --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 \
        "$HERE/score_cohort.sbatch" rescore ${scored_sets[@]+"${scored_sets[@]}"}
    RESCORE_JOB="$JOB_ID"
fi
fi

# ── 6 and 7. compute-all: the whole pipeline per patient and backend, last and alone ──
if [ "$computing" = 1 ]; then
    # the plan, written down: a head that never made its folder is still a row of
    # pipeline_cost_summary (no_trace, compute_all.planned)
    if [ "$DRY" != 1 ]; then
        mkdir -p "$REGBENCH_OUT/logs/compute"
        printf '%s\n' "$COMPUTE_PLAN" > "$REGBENCH_OUT/logs/compute/plan.tsv"
    fi
    i=0 heads=""
    while IFS=$'\t' read -r backend patient _ _; do
        [ -n "$backend" ] || continue
        lane=$((i % REGBENCH_COMPUTE_LANES))
        prev="COMPUTE_LANE_$lane"
        deps=()
        if [ -n "${!prev:-}" ]; then
            deps=(--dependency="afterany:${!prev}")
        elif [ -n "$SCORER_JOB" ]; then
            # LAST and ALONE (I3): after the scorer AND the re-score, so no selfcheck
            # and no scorer runs beside a measured pipeline
            deps=(--dependency="afterany:$SCORER_JOB${RESCORE_JOB:+:$RESCORE_JOB}")
        fi
        submit "rb-compute-$backend-$patient" "${COMMON[@]}" "$LOG_ONE" ${deps[@]+"${deps[@]}"} \
            --cpus-per-task=2 --mem=8G --time="$(limit "$compute_s")" \
            "$HERE/compute_all.sbatch" "$backend" "$patient"
        printf -v "$prev" '%s' "$JOB_ID"
        heads="${heads:+$heads:}$JOB_ID"
        i=$((i + 1))
    done <<< "$COMPUTE_PLAN"
    # its own scoring waits for its heads, which wait for rb-rescore (above): two
    # scorers never write the tables at once
    submit "rb-score-compute" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$heads${RESCORE_JOB:+:$RESCORE_JOB}" \
        --cpus-per-task="$CPUS" --mem="$REGBENCH_SCORE_MEM" --time=1-00:00:00 \
        "$HERE/score_cohort.sbatch" score
    echo "# compute-all: $i head(s) -> $REGBENCH_OUT/compute/<backend>/<patient>/ (trace/trace.txt); their end lines in $REGBENCH_OUT/logs/compute/; the tables pipeline_cost*.csv from rb-score-compute"
fi

[ "$N_CUT" = 0 ] || echo "# WARNING memory ladder: $N_CUT ladder(s) stop below a rung no node of $GRANT_WHERE can grant ($GRANT_MB MB): their units that would climb to it are written down as capped (the method's, imputed: losses.csv names the cap), the ladder stopping at the rung below"
if [ -n "$DROPPED_RUNS" ]; then
    echo "# DROPPED from this submission ($(wc -w <<< "$DROPPED_RUNS" | tr -d ' ') run(s); $([ "$DRY" = 1 ] && echo "a real submission writes them to $REGBENCH_OUT/submit_dropped.csv" || echo "$REGBENCH_OUT/submit_dropped.csv"); the scorer makes each a row of losses.csv):"
    printf '%s' "$DROPPED_ROWS" | sed 's/^/#   /'
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
if [ "$scoring_earlier" = 1 ]; then
    echo "# submission  -> $REGBENCH_SUBMISSION (scored as it is under $REGBENCH_OUT: nothing is run again)"
else
    echo "# submission  -> $REGBENCH_SUBMISSION (a unit already ok under $REGBENCH_OUT is reused$([ "$REGBENCH_FORCE" = 1 ] && echo ': NO, REGBENCH_FORCE=1 runs everything again'))"
fi
echo "# case records -> $REGBENCH_OUT/cache/scores/<dataset>/<case>/ (kept; a valid one is read, never computed twice)"
echo "# verdicts    -> $REGBENCH_OUT/logs/verdicts.tsv (written by the scorer: what every run's selfcheck and realcheck said; reported, never a gate)"
echo "# results     -> $REGBENCH_OUT/tables/summary.md: it says what every selfcheck said, beside rows that are all scored"
if [ -n "${REGBENCH_REPEAT_LAST_FILE:-}" ]; then
    printf '%s' "${REPEAT_SINKS:-$ALL_SUBMITTED}" > "$REGBENCH_REPEAT_LAST_FILE"
fi
FINISHED=1
exit 0
