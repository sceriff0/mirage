#!/bin/bash
#SBATCH --job-name=regbench-launch
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=regbench-launch-%j.log
# The launch as ONE job, everything set in here. From the repository:
#
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --prepare   (once; wait for its jobs)
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch             (the launch)
#   sbatch <repository>/benchmarks/regbench/slurm/launch.sbatch --dry-run   (prints, submits nothing)
#
# FROM ANY FOLDER: the folder it is submitted from is where the benchmark lives
# (regbench/ under it: cases, results, rounds.csv) and where the arms' results are
# looked for (arm_results/); the repository is found from the path given to sbatch.
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
# one folder for everything of the benchmark: cases/, out/ and rounds.csv.
# Default: regbench/ in the folder the job is submitted from
: "${REGBENCH_ROOT:=$here/regbench}"
# the arms' results the cohort is made of (rounds.csv is written from them by
# --prepare when it is not there yet). Default: arm_results/ where submitted from
: "${REGBENCH_ARMS:=$here/arm_results}"
: "${REGBENCH_CASES:=$REGBENCH_ROOT/cases}"
: "${REGBENCH_OUT:=$REGBENCH_ROOT/out}"
# the cohort's rounds.csv (python -m benchmarks.regbench.arms_manifest)
: "${REGBENCH_MANIFEST:=$REGBENCH_ROOT/rounds.csv}"
# how much runs at once
: "${REGBENCH_RUN_LANES:=4}"          # runs in flight (0: all of them)
: "${REGBENCH_ARRAY_LIMIT:=20}"       # tasks of one array running at once
: "${REGBENCH_STAGE_QUEUE_MAX:=50}"   # tile tasks pending or running, per run that has tiles
# the site
: "${REGBENCH_SITE_CONFIG:=$repo/conf/ieo.config}"
# the images: REGBENCH_IMAGES is where the site keeps its images (default: the
# cacheDir of the site config, else images/ under REGBENCH_ROOT). One that is not
# there yet is pulled into it, under the name Nextflow gives a pulled image
IMAGES="
STARE_REGBENCH docker://bolt3x/mirage-stare-regbench:1.0.0
VALIS120       docker://bolt3x/mirage-valis120:1.0.0
DEEPERHISTREG  docker://bolt3x/mirage-deeperhistreg:1.0.0
PALOM          docker://bolt3x/mirage-palom:1.0.0
ASHLAR         docker://labsyspharm/ashlar:1.20.0
VALIS_MIRAGE   docker://cdgatenbee/valis-wsi@sha256:eac27cc599ae0e54aa01c1bef97538301994ce1abd4da44be3f3130ab85a40e6
STARDIST       docker://bolt3x/mirage-stardist:1.0.0
"
# -----------------------------------------------------------------------------------

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
mkdir -p "$REGBENCH_ROOT" "$REGBENCH_CASES" "$REGBENCH_OUT" || exit 2
: "${REGBENCH_IMAGES:=${site_images:-$REGBENCH_ROOT/images}}"

# Every image has a file before anything is asked of it: the launcher would otherwise
# start with the python on PATH, which has not what the harness imports.
# REGBENCH_ALLOW_PATH_PYTHON=1 (the launcher's own switch) leaves the images alone.
if [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" != 1 ]; then
    runtime="$(command -v apptainer || command -v singularity || true)"
    if [ -z "$runtime" ]; then
        echo "regbench-launch: neither apptainer nor singularity is on PATH here ($(hostname)). Load the site's container module before sbatch (--export=ALL carries the PATH into the job). Nothing was submitted." >&2
        exit 2
    fi
    while read -r image ref; do
        [ -n "$image" ] || continue
        var="REGBENCH_SIF_$image"
        [ -z "${!var:-}" ] || { export "${var?}"; continue; }
        # only --prepare runs StarDist
        if [ "$image" = STARDIST ]; then
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
export REGBENCH_ROOT REGBENCH_CASES REGBENCH_OUT REGBENCH_MANIFEST REGBENCH_SITE_CONFIG
export REGBENCH_RUN_LANES REGBENCH_ARRAY_LIMIT REGBENCH_STAGE_QUEUE_MAX
[ -z "${REGBENCH_SBATCH_GPU:-}" ] || export REGBENCH_SBATCH_GPU
# the arms' results and the benchmark's own folder are read inside the images too
for more in "$REGBENCH_ARMS" "$REGBENCH_ROOT"; do
    [ -d "$more" ] && REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$more"
done
[ -z "${REGBENCH_BIND:-}" ] || export REGBENCH_BIND
echo "regbench-launch: repository $repo, cases $REGBENCH_CASES, results $REGBENCH_OUT, images $REGBENCH_IMAGES"
n_bound="$(tr ',' '\n' <<< "${REGBENCH_BIND:-}" | grep -c . || true)"
echo "regbench-launch: bound: ${REGBENCH_BIND:-nothing}"
echo "regbench-launch: arms $REGBENCH_ARMS ($([ -d "$REGBENCH_ARMS" ] && echo found || echo NOT THERE)), rounds file $REGBENCH_MANIFEST ($([ -s "$REGBENCH_MANIFEST" ] && echo there || echo not written yet))"
echo "regbench-launch: lanes $REGBENCH_RUN_LANES, array limit $REGBENCH_ARRAY_LIMIT, tile queue $REGBENCH_STAGE_QUEUE_MAX, GPU request ${REGBENCH_SBATCH_GPU:-one GPU}, $n_bound folder(s) bound into the images"

cd "$repo" || exit 2
# --prepare with no rounds.csv yet: write it from the arms' results, in the image the
# launcher itself runs its Python in
case " $* " in
    *" --prepare "*)
        if [ ! -s "$REGBENCH_MANIFEST" ] && [ ! -d "$REGBENCH_ARMS" ]; then
            echo "regbench-launch: --prepare needs the cohort's rounds file and there is none at $REGBENCH_MANIFEST, nor an arms' results folder at $REGBENCH_ARMS to write it from. Export REGBENCH_ARMS (the arm results folder) or REGBENCH_MANIFEST. Nothing was submitted." >&2
            exit 2
        fi
        if [ ! -s "$REGBENCH_MANIFEST" ] && [ -d "$REGBENCH_ARMS" ]; then
            # shellcheck disable=SC1091
            source "$repo/benchmarks/regbench/slurm/env.sh"
            echo "regbench-launch: writing $REGBENCH_MANIFEST from $REGBENCH_ARMS"
            if ! regbench_exec STARE_REGBENCH sh -c "$REGBENCH_PYTHON" regbench-python \
                -m benchmarks.regbench.arms_manifest --root "$REGBENCH_ARMS" --out "$REGBENCH_MANIFEST"; then
                echo "regbench-launch: the manifest could not be written from $REGBENCH_ARMS (see above). Nothing was submitted." >&2
                rm -f "$REGBENCH_MANIFEST"
                exit 2
            fi
        fi
        ;;
esac
exec bash "$launcher" "$@"
