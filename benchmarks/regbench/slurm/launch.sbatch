#!/bin/bash
#SBATCH --job-name=regbench-launch
#SBATCH --export=ALL
# never requeued: a requeued head would run its stage again and queue its jobs twice
# (slurm.conf JobRequeue=1, the default, requeues a batch job unless it says otherwise)
#SBATCH --no-requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=regbench-launch-%j.log
# The launch as ONE job, everything set in here.
#
# THE WHOLE BENCHMARK IN ONE COMMAND (README, "The whole benchmark in one command"):
#
#   sbatch [site sbatch options] <repository>/benchmarks/regbench/slurm/launch.sbatch --all \
#       [launcher args, e.g. --runs "a b"] [-- extra sbatch args of every job]
#   sbatch ... launch.sbatch --all --dry-run   (the plan and the sbatch lines; submits nothing)
#
# runs the input step, the prepare step and the launch as a CHAIN OF SHORT HEAD JOBS
# (job name regbench-launch): each head runs ONE stage, submits the next head (this same
# file) with afterok on that stage's jobs and --kill-on-invalid-dep=yes, and ends. No
# head sleeps waiting for another and nothing waits on the login node. The first head
# runs the first stage not done on disk; each later head is given its stage
# (--all-stage=prepare, --all-stage=launch), so the chain never loops or skips. Its
# record is $REGBENCH_ROOT/logs/chain/<chain id>.tsv (each head's line written before its
# stage runs); every head prints the scancel line that removes everything the chain has
# queued so far, before its stage and after it. Submitted again after a stop, --all skips
# the stages that are done (prepare only when its own record says it ended ok), and the
# launch reuses every unit that is ok. The flags of the next head, and why each is
# there: chain_next_head in slurm/chain.sh.
#
# The steps one by one, from the repository:
#
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --inputs    (once; wait for its job)
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --prepare   (once; wait for its jobs)
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch             (the launch)
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --dry-run   (prints, submits nothing)
#
# and the WHOLE pipeline's cost per patient and registration backend (compute-all,
# slurm/compute_all.sbatch: run last and alone, after the scorer), with the launch or
# on its own for a benchmark that is done; both need REGBENCH_INPUT:
#
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --compute-all
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --compute-only
#
# FROM ANY FOLDER: the folder it is submitted from is where the benchmark lives
# (regbench/ under it: inputs, cases, results); the repository is found from the path
# given to sbatch.
#
# THE INPUTS are regbench's own. --inputs submits ONE Nextflow head job
# (slurm/inputs.sbatch): the pipeline's preprocessing of the cohort's samplesheet
# REGBENCH_INPUT into $REGBENCH_INPUTS/preprocess. --prepare then writes the rounds
# ($REGBENCH_INPUTS/rounds.csv, python -m benchmarks.regbench.manifest) from its
# preprocessed.csv, and the nuclei step traces each slide's outlines from its own
# StarDist run into $REGBENCH_INPUTS/outlines. A REGBENCH_MANIFEST exported by hand is
# taken as it is instead (a rounds.csv written from the arms' results before
# 2026-10-16 is read as before).
#
# THE IMAGES are looked for where the site keeps its own (the cacheDir of the site
# config), under the name Nextflow gives a pulled image; one that is not there is pulled
# into that folder first. No image, no launch: the python on PATH is never used.
#
# It runs submit_cohort.sh on a compute node; every argument is the launcher's. The
# SETTINGS below are the only thing to edit. A variable already exported when sbatch is
# called wins over the line here (--export=ALL), so one launch can differ without an
# edit.
#
# THE SITE is the site config (conf/ieo.config of the checkout, which is not in the
# repository): the GPU request, the image folder and the folders bound into the images are READ from it
# when this job runs, never written down here. A site whose config names no partition
# and no account needs no sbatch option; one that does names them after `--`, and before
# the script for this job itself.
#
# This job starts the images once (the preflight) and submits; it ends when the last job
# is in the queue, and its log (regbench-launch-<job id>.log, where it was submitted
# from) is what the launcher prints. It needs what the tile submitter needs of a compute
# node: the scheduler's commands and the container runtime. A node has often no network:
# the pinned STARE source is downloaded once by a launch, so the first call (--preflight
# is enough) is made from a login node, and this job finds it there.
#
set -uo pipefail
here="${SLURM_SUBMIT_DIR:-$PWD}"
# --all / --all-stage=<stage> are this script's own (the chain, chain_run below); every
# other argument is the launcher's, and those after `--` are extra sbatch arguments
ALL=0 ALL_STAGE="" ALL_DRY=0 ALL_ARGS=() ALL_EXTRA=() seen_dd=0
for a in "$@"; do
    if [ "$seen_dd" = 1 ]; then ALL_EXTRA+=("$a"); continue; fi
    case "$a" in
        --all) ALL=1 ;;
        --all-stage=*) ALL=1 ALL_STAGE="${a#--all-stage=}" ;;
        --dry-run) ALL_DRY=1; ALL_ARGS+=("$a") ;;
        --) seen_dd=1 ;;
        *) ALL_ARGS+=("$a") ;;
    esac
done
if [ "$ALL" = 1 ]; then
    for a in ${ALL_ARGS[@]+"${ALL_ARGS[@]}"}; do
        case "$a" in --inputs|--prepare|--score-only|--compute-only|--preflight)
            echo "regbench-launch: --all runs the input step, the prepare step and the launch itself; $a is a step of its own (submit it without --all). Nothing was submitted." >&2
            exit 2 ;;
        esac
    done
    # the launcher repeats a submission before its input and prepare steps: under --all
    # each repeat would queue its own input step and the chain would see none queued.
    # Repeats are the CPU cross's, a plain launch after the benchmark is prepared
    if [ "${REGBENCH_REPEAT:-1}" != 1 ]; then
        echo "regbench-launch: --all runs ONE submission; REGBENCH_REPEAT=${REGBENCH_REPEAT} is the CPU cross's, submitted as a plain launch (without --all) once --all has prepared the benchmark. Nothing was submitted." >&2
        exit 2
    fi
    case "$ALL_STAGE" in
        ''|inputs|prepare|launch) ;;
        *) echo "regbench-launch: --all-stage=$ALL_STAGE: the stages are inputs, prepare and launch. Nothing was submitted." >&2; exit 2 ;;
    esac
fi
# the repository: REGBENCH_REPO; else where the file given to sbatch is (the scheduler
# knows its path: Command= of `scontrol show job`); else the folder submitted from
asked=""
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    asked="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -nE 's/^[[:space:]]*Command=([^[:space:]]+).*/\1/p' | head -n 1)"
fi
repo=""
for found in "${REGBENCH_REPO:-}" "${asked:+$(dirname "$asked")/../../..}" "$here" "$(dirname "$0")/../../.."; do
    [ -n "$found" ] && [ -f "$found/benchmarks/regbench/slurm/submit_cohort.sh" ] || continue
    repo="$(cd "$found" && pwd)"
    break
done
if [ -z "$repo" ]; then
    echo "regbench-launch: the repository was not found (looked at REGBENCH_REPO, at the script given to sbatch${asked:+ ($asked)} and at $here). Export REGBENCH_REPO. Nothing was submitted." >&2
    exit 2
fi
launcher="$repo/benchmarks/regbench/slurm/submit_cohort.sh"
export REGBENCH_REPO="$repo"

# ------------------------------- SETTINGS: edit here -------------------------------
# A variable exported before sbatch wins over the line here (--export=ALL). A commented
# line (`#   REGBENCH_X=...`) is the launcher's own default, shown so it can be set.
# one folder for everything of the benchmark: inputs/, cases/ and out/.
# Default: regbench/ in the folder the job is submitted from
: "${REGBENCH_ROOT:=$here/regbench}"
# the input folder: the input step's preprocessing (preprocess/), the rounds
# (rounds.csv) and the outlines the nuclei step traces (outlines/)
: "${REGBENCH_INPUTS:=$REGBENCH_ROOT/inputs}"
# REGBENCH_INPUT: the cohort's samplesheet (the pipeline's --input). Required by
# --inputs, --compute-all and --compute-only. With --all: needed by its input stage,
# and when it is set the launch stage adds --compute-all (unset: skipped, and said)
#   REGBENCH_INPUT=/path/to/input.csv
# --inputs and compute-all: the pipeline's profile (the arms' ARMS_PROFILE)
: "${REGBENCH_NF_PROFILE:=singularity,ieo}"
: "${REGBENCH_CASES:=$REGBENCH_ROOT/cases}"
: "${REGBENCH_OUT:=$REGBENCH_ROOT/out}"
# REGBENCH_MANIFEST is NOT set here: --prepare writes $REGBENCH_INPUTS/rounds.csv from
# the input step. Export it to take a rounds.csv as it is.
# ── CONCURRENCY: how much runs at once ────────────────────────────────────────────
# THE PEAK, with the values below (the launch stage prints it from the run list as
# "# peak: ..."; the launcher's dry run gives the exact queue figure):
#   ladder tasks <= RUN_LANES x ARRAY_LIMIT    one first-rung array per run in flight;
#                                              a run with a realcheck: one task more
#   GPU tasks    <= (GPU runs in flight) x GPU_ARRAY_LIMIT, one GPU each
#   + every selfcheck: one job each (64 GB, 4 CPUs), OUTSIDE the lanes, all at once
#   + tile tasks <= STAGE_QUEUE_MAX            STARE's REG-TILE jobs
#   + the small jobs: a watcher link per chain, a tile submitter per STARE run in
#     flight (1 CPU, 1 GB each)
#   CPUs = ladder tasks and selfchecks x REGBENCH_CPUS, plus tile tasks x their own
#   allocation (the pipeline's: 2 CPUs); memory at the first rung = ladder tasks x 16 GB
#   + selfchecks x 64 GB + tile tasks x 4 GB. On the cohort's list, with these values:
#   at most 197 tasks (82 ladder + 15 selfchecks + 100 tiles), 588 CPUs, 20 GPUs, 2672 GB
#
# REGBENCH_RUN_LANES: runs in flight. The runs are dealt onto this many lanes; a run
# starts when the run before it on its lane has ended (afterany). Default here 4 (the
# launcher's own: 0, every run at once). Raise it on an empty cluster, lower it when
# the per-user queue limit is hit. 0: no limit
: "${REGBENCH_RUN_LANES:=4}"
# REGBENCH_ARRAY_LIMIT: tasks of ONE array running at once (sbatch --array=...%N): a CPU
# run's ladder arrays, the nuclei array on CPUs, the per-case scoring arrays. Default
# 20. Lower it when one run's tasks crowd out the others. Not the tile arrays
: "${REGBENCH_ARRAY_LIMIT:=20}"
# REGBENCH_GPU_ARRAY_LIMIT: the same for every array of a run that asks for a GPU (its
# first rung, its watcher's retry arrays, its realcheck) and the nuclei array on a GPU.
# Each task holds ONE GPU, so this is the GPUs one array holds at once. Default here 4
# (unset in the launcher: ARRAY_LIMIT, 20 GPUs for one DeeperHistReg array). Raise it
# to what the site lets one user hold
: "${REGBENCH_GPU_ARRAY_LIMIT:=4}"
# REGBENCH_STAGE_QUEUE_MAX: tile tasks pending or running in the WHOLE launch (a total),
# split over the STARE runs feeding at once (at most the lanes): 100 over 4 lanes is 25
# each. Default here 100 (launcher: 200). Lower it when the tiles fill the queue limit
: "${REGBENCH_STAGE_QUEUE_MAX:=100}"
# REGBENCH_STAGE_ARRAY_MAX: the most tasks in ONE tile array a tile submitter submits.
# Default 100. Lower it where the site caps an array's size (MaxArraySize)
#   REGBENCH_STAGE_ARRAY_MAX=100
# REGBENCH_STAGE_FEEDERS: tile submitters chained per STARE run and dataset, each with
# the site's longest time; the second goes on where the first stopped at its limit.
# Default 2. Raise it only when one run's tiles take more than 7 days to feed
#   REGBENCH_STAGE_FEEDERS=2
# REGBENCH_COMPUTE_LANES: compute-all heads (one Nextflow head per patient and backend)
# at once, LAST and ALONE after the scorer. Default 2: more lanes is more contention
# in a cost measurement (REGBENCH_COMPUTE_TIME, 7-00:00:00: each head's time limit)
#   REGBENCH_COMPUTE_LANES=2
# REGBENCH_RETRY_WATCHERS: links of each chain's watcher (retry.sbatch: 1 CPU, 1 GB, 7
# days), one after the other: a link ended at its limit leaves the next to go on.
# Default 2. Raise it only for a ladder that climbs for more than 14 days
#   REGBENCH_RETRY_WATCHERS=2
# REGBENCH_WAVES: 1 runs one method at a time, each after every job of the method
# before it (a timing replicate, nothing else on the nodes). Default 0
#   REGBENCH_WAVES=0
# REGBENCH_REPEAT: the same submission N times, repeat k into $REGBENCH_OUT/rep<k> (the
# CPU cross's repetitions), one after the other: each repeat's first jobs wait afterany
# on the previous repeat's scorers. Default 1. Not with --all (refused): a plain launch
#   REGBENCH_REPEAT=1
# ── a task's allocation ───────────────────────────────────────────────────────────
# REGBENCH_CPUS: every ladder task's --cpus-per-task: 4 (every method's allocation, the
# default) or 8 (the CPU cross only)
#   REGBENCH_CPUS=4
# REGBENCH_TIME: a ladder task's time limit per round (a patient run gets it times its
# rounds, never above 7 days). Default 24:00:00
#   REGBENCH_TIME=24:00:00
# REGBENCH_SCORE_CPUS / REGBENCH_SCORE_MEM: per task of the per-case scoring arrays (the
# memory also of the tissue and score jobs). Default 8 CPUs, 64G
#   REGBENCH_SCORE_CPUS=8
#   REGBENCH_SCORE_MEM=64G
# The memory ladder rungs are NOT a setting: 16 -> 64 -> 256 -> 512 -> 700 GB at 4 CPUs
# (benchmarks/regbench/ladder.py, frozen); the selfchecks 64 -> 256 -> 512 GB; the
# nuclei step REGBENCH_NUCLEI_MEM (128G) -> 256 -> 512 GB. Only the first rung is
# submitted; a higher one only for the units that ran out of memory on the one below.
# REGBENCH_SITE_CONFIG: the site config the GPU request, the image folder and the bound
# folders are READ from (`none`: export REGBENCH_SBATCH_GPU and REGBENCH_BIND yourself).
# Default conf/ieo.config of this checkout (it is not in the repository)
: "${REGBENCH_SITE_CONFIG:=$repo/conf/ieo.config}"
# the images: REGBENCH_IMAGES is where the site keeps its images (default: the
# cacheDir of the site config, else images/ under REGBENCH_ROOT). One that is not
# there yet is pulled into it, under the name Nextflow gives a pulled image
IMAGES="
STARE_REGBENCH docker://bolt3x/mirage-stare-regbench:1.0.0
VALIS120       docker://bolt3x/mirage-valis120:1.0.0
DEEPERHISTREG  docker://bolt3x/mirage-deeperhistreg:1.0.0
PALOM          docker://bolt3x/mirage-palom:1.0.0
ASHLAR         docker://bolt3x/mirage-ashlar120:1.0.0
VALIS_MIRAGE   docker://bolt3x/mirage-valis100:1.0.0
STARDIST       docker://bolt3x/mirage-stardist:1.0.0
"
# -----------------------------------------------------------------------------------

# compute-all runs the pipeline on the cohort's samplesheet: no launch without it
case " $* " in
    *" --compute-all "*|*" --compute-only "*)
        if [ -z "${REGBENCH_INPUT:-}" ]; then
            echo "regbench-launch: --compute-all/--compute-only need REGBENCH_INPUT, the cohort's samplesheet (the pipeline's --input): export it, or set it in the SETTINGS. Nothing was submitted." >&2
            exit 2
        fi
        ;;
esac

# site_value KEY: the quoted value the site config gives KEY (comment lines are not read)
site_value() {
    sed -E 's#^[[:space:]]*//.*$##' "$REGBENCH_SITE_CONFIG" \
        | sed -nE "s/^[[:space:]]*(params\.)?$1[[:space:]]*=[[:space:]]*['\"]([^'\"]*)['\"].*/\2/p" \
        | head -n 1
}
site_images=""
case "$REGBENCH_SITE_CONFIG" in
    none|-) ;;
    *)
        if [ ! -f "$REGBENCH_SITE_CONFIG" ]; then
            echo "regbench-launch: the site config $REGBENCH_SITE_CONFIG is not there. Name one in REGBENCH_SITE_CONFIG, or \`none\` and export REGBENCH_SBATCH_GPU and REGBENCH_BIND yourself. Nothing was submitted." >&2
            exit 2
        fi
        # the GPU a task asks for: the pipeline's own request on this site (gpu_type)
        gpu="$(site_value gpu_type)"
        [ -z "$gpu" ] || : "${REGBENCH_SBATCH_GPU:=--gres=gpu:$gpu}"
        # the folders the pipeline binds into its images on this site (singularity
        # runOptions): the slides and the nuclei are under them
        binds="$(site_value runOptions | grep -oE -- '--bind[= ]+[^[:space:]]+' \
            | sed -E 's/^--bind[= ]+//' | paste -sd, -)"
        [ -z "$binds" ] || : "${REGBENCH_BIND:=$binds}"
        # where the pipeline keeps its pulled images on this site
        site_images="$(site_value cacheDir)"
        ;;
esac
mkdir -p "$REGBENCH_ROOT" "$REGBENCH_INPUTS" "$REGBENCH_CASES" "$REGBENCH_OUT" || exit 2
: "${REGBENCH_IMAGES:=${site_images:-$REGBENCH_ROOT/images}}"

# Every image has a file before anything is asked of it: the launcher would otherwise
# start with the python on PATH, which has not what the harness imports.
# REGBENCH_ALLOW_PATH_PYTHON=1 (the launcher's own switch) leaves the images alone.
# --inputs needs none of them: Nextflow pulls the pipeline's own images.
inputs_only=0
case " $* " in *" --inputs "*) inputs_only=1 ;; esac
if [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" != 1 ] && [ "$inputs_only" = 0 ]; then
    runtime="$(command -v apptainer || command -v singularity)" || runtime=""
    if [ -z "$runtime" ]; then
        echo "regbench-launch: neither apptainer nor singularity is on PATH here ($(hostname)). Load the site's container module before sbatch (--export=ALL carries the PATH into the job). Nothing was submitted." >&2
        exit 2
    fi
    while read -r image ref; do
        [ -n "$image" ] || continue
        var="REGBENCH_SIF_$image"
        [ -z "${!var:-}" ] || { export "${var?}"; continue; }
        # only --prepare runs StarDist (and --all, whose chain has the prepare stage)
        if [ "$image" = STARDIST ] && [ "$ALL" = 0 ]; then
            case " $* " in *" --prepare "*) ;; *) continue ;; esac
        fi
        name="${ref#docker://}"; name="${name//\//-}"; name="${name//:/-}.img"
        file=""
        short="$(tr '[:upper:]_' '[:lower:]-' <<< "$image").sif"
        for at in "$REGBENCH_IMAGES/$name" "$REGBENCH_IMAGES/regbench/$name" \
                  "$REGBENCH_IMAGES/$short" "$REGBENCH_IMAGES/regbench/$short"; do
            [ -s "$at" ] && { file="$at"; break; }
        done
        if [ -z "$file" ]; then
            file="$REGBENCH_IMAGES/$name"
            echo "regbench-launch: pulling $ref to $file"
            mkdir -p "$REGBENCH_IMAGES" || exit 2
            if ! "$runtime" pull "$file.part" "$ref" || ! mv "$file.part" "$file"; then
                rm -f "$file.part"
                echo "regbench-launch: $ref could not be pulled on $(hostname) (a node may have no network). From a node that has one:  $runtime pull $file $ref  ; then submit again. Nothing was submitted." >&2
                exit 2
            fi
        fi
        printf -v "$var" '%s' "$file"
        export "${var?}"
        echo "regbench-launch: $image -> $file"
    done <<< "$IMAGES"
fi
export REGBENCH_ROOT REGBENCH_INPUTS REGBENCH_NF_PROFILE REGBENCH_CASES REGBENCH_OUT REGBENCH_SITE_CONFIG
[ -z "${REGBENCH_MANIFEST:-}" ] || export REGBENCH_MANIFEST
[ -z "${REGBENCH_INPUT:-}" ] || export REGBENCH_INPUT
export REGBENCH_RUN_LANES REGBENCH_ARRAY_LIMIT REGBENCH_GPU_ARRAY_LIMIT REGBENCH_STAGE_QUEUE_MAX
[ -z "${REGBENCH_SBATCH_GPU:-}" ] || export REGBENCH_SBATCH_GPU
# the benchmark's own folder is read inside the images too (REGBENCH_INPUTS is bound
# by env.sh wherever it is)
# (once: a later head of --all inherits the list the head before it made)
case ",${REGBENCH_BIND:-}," in
    *",$REGBENCH_ROOT,"*) ;;
    *) [ -d "$REGBENCH_ROOT" ] && REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$REGBENCH_ROOT" ;;
esac
[ -z "${REGBENCH_BIND:-}" ] || export REGBENCH_BIND
echo "regbench-launch: repository $repo, cases $REGBENCH_CASES, results $REGBENCH_OUT, images $REGBENCH_IMAGES"
n_bound="$(tr ',' '\n' <<< "${REGBENCH_BIND:-}" | grep -c .)" || n_bound=0
echo "regbench-launch: bound: ${REGBENCH_BIND:-nothing}"
rounds="${REGBENCH_MANIFEST:-$REGBENCH_INPUTS/rounds.csv}"
echo "regbench-launch: inputs $REGBENCH_INPUTS (preprocessing $([ -s "$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv" ] && echo "done" || echo "not run yet: --inputs")), rounds file $rounds ($([ -s "$rounds" ] && echo there || echo "not written yet: --prepare"))"
echo "regbench-launch: lanes $REGBENCH_RUN_LANES, array limit $REGBENCH_ARRAY_LIMIT (GPU runs $REGBENCH_GPU_ARRAY_LIMIT), tile queue $REGBENCH_STAGE_QUEUE_MAX, GPU request ${REGBENCH_SBATCH_GPU:-one GPU}, $n_bound folder(s) bound into the images"

cd "$repo" || exit 2
echo "[regbench] start kind=launch job=${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:--} run=- label=- dataset=- unit=- rung_gb=- attempt=- out=$REGBENCH_OUT commit=$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || echo unknown) host=$(hostname 2>/dev/null || echo -) at=$(date +%FT%T) args=$*"
# --prepare writes the rounds itself (submit_cohort.sh), from the input step; its EXIT
# trap prints the launch's `[regbench] end` line
REGBENCH_LAUNCH_T0="$(date +%s)"
export REGBENCH_LAUNCH_T0
if [ "$ALL" = 1 ]; then
    # shellcheck source=benchmarks/regbench/slurm/chain.sh
    source "$repo/benchmarks/regbench/slurm/chain.sh"
    chain_run
    exit $?
fi
exec bash "$launcher" "$@"
