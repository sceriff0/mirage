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
# (regbench/ under it: cases, results, images, rounds.csv) and where the arms' results
# are looked for (arm_results/); the repository is found from the path given to sbatch.
#
# It runs submit_cohort.sh on a compute node; every argument is the launcher's. The
# SETTINGS below are the only thing to edit. A variable already exported when sbatch is
# called wins over the line here (--export=ALL), so one launch can differ without an
# edit.
#
# THE SITE is the site config (conf/ieo.config of the checkout, which is not in the
# repository): the GPU request and the folders bound into the images are READ from it
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
# one folder for everything of the benchmark: cases/, out/, images/ and rounds.csv.
# Default: regbench/ in the folder the job is submitted from
: "${REGBENCH_ROOT:=$here/regbench}"
# the arms' results the cohort is made of (rounds.csv is written from them by
# --prepare when it is not there yet). Default: arm_results/ where submitted from
: "${REGBENCH_ARMS:=$here/arm_results}"
: "${REGBENCH_CASES:=$REGBENCH_ROOT/cases}"
: "${REGBENCH_OUT:=$REGBENCH_ROOT/out}"
# the cohort's rounds.csv (python -m benchmarks.regbench.arms_manifest)
: "${REGBENCH_MANIFEST:=$REGBENCH_ROOT/rounds.csv}"
# the images, pulled once into one folder under the names the README pulls them as
: "${REGBENCH_IMAGES:=$REGBENCH_ROOT/images}"
if [ -d "$REGBENCH_IMAGES" ]; then
    : "${REGBENCH_SIF_STARE_REGBENCH:=$REGBENCH_IMAGES/stare-regbench.sif}"
    : "${REGBENCH_SIF_VALIS120:=$REGBENCH_IMAGES/valis120.sif}"
    : "${REGBENCH_SIF_DEEPERHISTREG:=$REGBENCH_IMAGES/deeperhistreg.sif}"
    : "${REGBENCH_SIF_PALOM:=$REGBENCH_IMAGES/palom.sif}"
    : "${REGBENCH_SIF_ASHLAR:=$REGBENCH_IMAGES/ashlar.sif}"
    : "${REGBENCH_SIF_VALIS_MIRAGE:=$REGBENCH_IMAGES/valis-mirage.sif}"
    : "${REGBENCH_SIF_STARDIST:=$REGBENCH_IMAGES/stardist.sif}"
else
    echo "regbench-launch: no image folder $REGBENCH_IMAGES: only the REGBENCH_SIF_* exported by the caller are used"
fi
# how much runs at once
: "${REGBENCH_RUN_LANES:=4}"          # runs in flight (0: all of them)
: "${REGBENCH_ARRAY_LIMIT:=20}"       # tasks of one array running at once
: "${REGBENCH_STAGE_QUEUE_MAX:=50}"   # tile tasks pending or running, per run that has tiles
# the site
: "${REGBENCH_SITE_CONFIG:=$repo/conf/ieo.config}"
# -----------------------------------------------------------------------------------

# site_value KEY: the quoted value the site config gives KEY (comment lines are not read)
site_value() {
    sed -E 's#^[[:space:]]*//.*$##' "$REGBENCH_SITE_CONFIG" \
        | sed -nE "s/^[[:space:]]*(params\.)?$1[[:space:]]*=[[:space:]]*['\"]([^'\"]*)['\"].*/\2/p" \
        | head -n 1
}
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
        ;;
esac
mkdir -p "$REGBENCH_ROOT" || exit 2
export REGBENCH_ROOT REGBENCH_CASES REGBENCH_OUT REGBENCH_MANIFEST REGBENCH_SITE_CONFIG
for v in STARE_REGBENCH VALIS120 DEEPERHISTREG PALOM ASHLAR VALIS_MIRAGE STARDIST; do
    v="REGBENCH_SIF_$v"
    [ -z "${!v:-}" ] || export "${v?}"
done
export REGBENCH_RUN_LANES REGBENCH_ARRAY_LIMIT REGBENCH_STAGE_QUEUE_MAX
[ -z "${REGBENCH_SBATCH_GPU:-}" ] || export REGBENCH_SBATCH_GPU
# the arms' results and the benchmark's own folder are read inside the images too
for more in "$REGBENCH_ARMS" "$REGBENCH_ROOT"; do
    [ -d "$more" ] && REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$more"
done
[ -z "${REGBENCH_BIND:-}" ] || export REGBENCH_BIND
echo "regbench-launch: cases $REGBENCH_CASES, results $REGBENCH_OUT, images $REGBENCH_IMAGES"
n_bound="$(tr ',' '\n' <<< "${REGBENCH_BIND:-}" | grep -c . || true)"
echo "regbench-launch: lanes $REGBENCH_RUN_LANES, array limit $REGBENCH_ARRAY_LIMIT, tile queue $REGBENCH_STAGE_QUEUE_MAX, GPU request ${REGBENCH_SBATCH_GPU:-one GPU}, $n_bound folder(s) bound into the images"

cd "$repo" || exit 2
# --prepare with no rounds.csv yet: write it from the arms' results, in the image the
# launcher itself runs its Python in
case " $* " in
    *" --prepare "*)
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
