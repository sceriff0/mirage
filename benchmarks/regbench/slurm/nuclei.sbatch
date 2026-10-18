#!/bin/bash
#SBATCH --job-name=regbench-nuclei
#SBATCH --output=regbench-nuclei-%A_%a.log
# The nuclei and the outlines of ONE native slide of the cohort (an array over slides):
#   sbatch --array=0-$((N - 1)) -c 8 --mem=128G -t 8:00:00 <gpu request> nuclei.sbatch
# with N = `python -m benchmarks.regbench nuclei --manifest "$REGBENCH_MANIFEST" --out "$REGBENCH_NUCLEI" --count`.
#
# 1. In the scorer's image: what is kept for the slide is looked at (a finished run is
#    reused when it is of this slide and of the settings that resolve now).
# 2. For a slide without one: the pipeline's bin/segment.py, unchanged, in the
#    pipeline's StarDist image (REGBENCH_SIF_STARDIST), with the arguments the pipeline
#    renders for --seg_method stardist (`regbench nuclei --command` prints the line).
# 3. In the scorer's image again (it has scikit-image and scipy): the run's cell mask is
#    traced to the slide's outlines (the manifest's ref_cells / mov_cells, under
#    $REGBENCH_INPUTS/outlines) by bin/mask_to_geojson.py at simplify_tolerance, the
#    nuclei are checked against them (a self-consistency check: exact must come out
#    true) and the slide's record is written. A disagreement is printed and recorded
#    (exact: false).
#
#    Nuclei under 0.90 of the outlines with exactly one nucleus are REFUSED: the slide
#    gets no nuclei, a `REFUSED <slide>` line says why, and this task exits 1.
#
# Needs REGBENCH_MANIFEST (the rounds.csv of python -m benchmarks.regbench.manifest) and
# REGBENCH_NUCLEI, and the site's model (the pipeline's segmentation_model and
# segmentation_model_dir): REGBENCH_STARDIST_MODEL and REGBENCH_STARDIST_MODEL_DIR, or,
# when they are unset, a site config that sets the two keys (REGBENCH_SITE_CONFIG, else
# conf/ieo.config of the checkout; any StarDist setting and simplify_tolerance it
# overrides is used too). REGBENCH_STARDIST_CPU=1 runs StarDist without a GPU.
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
: "${REGBENCH_MANIFEST:?export REGBENCH_MANIFEST (the rounds.csv of benchmarks.regbench.manifest)}"
: "${REGBENCH_NUCLEI:?export REGBENCH_NUCLEI (the folder that keeps the nuclei of each slide)}"
: "${SLURM_ARRAY_TASK_ID:?this task is one slide of an array: submit with --array}"
scorer="${REGBENCH_PREPARE_IMAGE:-STARE_REGBENCH}"
# nuclei_end <status> <cause|-> <wants|-> <exit>: the task's end line, where the chain's
# watcher reads it (REGBENCH_END_FILE), and as the log's last line
nuclei_end() {
    local cause="" line
    [ "$2" = - ] || cause=" cause=$2 whose=benchmark"
    line="REGBENCH_END kind=nuclei unit=index_$SLURM_ARRAY_TASK_ID status=$1$cause wants=$3 exit=$4 log=${REGBENCH_LOG:--}"
    if [ -n "${REGBENCH_END_FILE:-}" ]; then echo "$line" > "$REGBENCH_END_FILE"; fi
    regbench_say_end "$line"
}
regbench_start nuclei "unit=index_$SLURM_ARRAY_TASK_ID" "rung_gb=$(awk '{ printf "%g", $1 / 1024 }' <<< "${SLURM_MEM_PER_NODE:-0}")"
what=(nuclei --manifest "$REGBENCH_MANIFEST" --out "$REGBENCH_NUCLEI"
      --index "$SLURM_ARRAY_TASK_ID" --workers "${SLURM_CPUS_PER_TASK:-1}")
mkdir -p "$REGBENCH_NUCLEI"

# REGBENCH_NUCLEI_FORCE=1: decide the slide again whatever is kept. The first look
# below sets a kept re-run aside (usable or refused), so the command is printed for it
# and the last look reads the masks StarDist has just written
force=()
[ "${REGBENCH_NUCLEI_FORCE:-0}" = 1 ] && force=(--force)
# what is there first. It exits 1 for a slide whose kept re-run has no usable nuclei
# and needs no re-run (StarDist found none): that is this task's verdict when nothing
# is run again, not a reason to stop before asking what the slide still needs. A
# re-run under the agreement threshold is KEPT and flagged low_agreement (B9), and a
# record an earlier checkout refused is checked again here from its masks.
first=0
regbench_py "$scorer" "${what[@]}" ${force[@]+"${force[@]}"} || first=$?
line="$(regbench_py "$scorer" "${what[@]}" --command ${force[@]+"${force[@]}"})"
if [ -z "$line" ]; then
    if [ "$first" -ne 0 ]; then
        echo "slide $SLURM_ARRAY_TASK_ID: its kept StarDist re-run has no usable nuclei and is not run again (nuclei --list has the reason). Correct the model or the settings, then REGBENCH_NUCLEI_FORCE=1 runs it again. This task fails."
        nuclei_end failed benchmark - "$first"
        exit "$first"
    fi
    echo "slide $SLURM_ARRAY_TASK_ID: nothing to segment (usable nuclei exist: nuclei --list); StarDist is not run"
    nuclei_end ok - - 0
    exit 0
fi
echo "+ $line"
# where the model and each setting came from; a model folder named by the site config
# only (no REGBENCH_STARDIST_MODEL_DIR) is bound into the StarDist image here
settings="$(regbench_py "$scorer" "${what[@]}" --settings)"
while IFS= read -r said; do echo "# StarDist re-run: $said"; done <<< "$settings"
model_dir="$(awk -F'\t' '$1 == "segmentation_model_dir" { print $2 }' <<< "$settings")"
if [ -n "$model_dir" ] && [ -z "${REGBENCH_STARDIST_MODEL_DIR:-}" ]; then
    export REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$model_dir"
fi
script="$(mktemp "$REGBENCH_NUCLEI/.segment-$SLURM_ARRAY_TASK_ID-XXXXXX.sh")"
# shellcheck disable=SC2329  # the EXIT trap
on_exit() {
    local code=$?
    rm -f "$script"
    regbench_exit_said "$code"
}
trap on_exit EXIT
# the line starts with `python`: where the StarDist image has python3 only, that is it
printf 'set -euo pipefail\n%s\n%s\n' \
    'command -v python >/dev/null 2>&1 || python() { python3 "$@"; }' "$line" > "$script"
# the GPU reaches the StarDist image only (REGBENCH_GPU=1 passes --nv)
gpu=1
[ "${REGBENCH_STARDIST_CPU:-0}" != 0 ] && gpu=0
# A StarDist run that fails is WRITTEN DOWN (`nuclei --failed`): the slide's record and
# the drop reason of `prepare multiplex` name the exit code (137: the kernel's
# out-of-memory killer) instead of "outlines not traced yet". The end line asks the
# chain's watcher (retry.sbatch, submit_cohort.sh) for the next memory rung only then.
code=0
REGBENCH_GPU="$gpu" regbench_exec STARDIST bash "$script" || code=$?
if [ "$code" -ne 0 ]; then
    mem_gb="$(awk '{ printf "%g", $1 / 1024 }' <<< "${SLURM_MEM_PER_NODE:-0}")"
    [ "$mem_gb" = 0 ] && mem_gb=""
    echo "slide $SLURM_ARRAY_TASK_ID: StarDist FAILED (exit $code${mem_gb:+ at $mem_gb GB}); written down for the slide"
    regbench_py "$scorer" "${what[@]}" --failed "$code" ${mem_gb:+--mem-gb "$mem_gb"} \
        || echo "slide $SLURM_ARRAY_TASK_ID: the failure could not be written down (nuclei --failed failed above)"
    # the benchmark's own step (resources.CAUSES): its out-of-memory is benchmark_oom
    cause=benchmark wants=-
    if [ "$code" = 137 ]; then cause=benchmark_oom wants=climb; fi
    nuclei_end failed "$cause" "$wants" "$code"
    exit "$code"
fi
REGBENCH_GPU=0 regbench_py "$scorer" "${what[@]}"
nuclei_end ok - - 0
