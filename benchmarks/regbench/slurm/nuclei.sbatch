#!/bin/bash
#SBATCH --job-name=regbench-nuclei
#SBATCH --output=regbench-nuclei-%A_%a.log
# The bare nuclei of ONE native slide of the cohort (an array over slides):
#   sbatch --array=0-$((N - 1)) -c 8 --mem=128G -t 8:00:00 <gpu request> nuclei.sbatch
# with N = `python -m benchmarks.regbench nuclei --manifest "$REGBENCH_MANIFEST" --out "$REGBENCH_NUCLEI" --count`.
#
# 1. In the scorer's image: an existing label image of the slide (the manifest's
#    ref_nuclei / mov_nuclei) is checked against the slide and its published outlines.
# 2. Only for a slide left without usable nuclei: the pipeline's bin/segment.py, unchanged,
#    in the pipeline's StarDist image (REGBENCH_SIF_STARDIST), with the arguments the
#    pipeline renders for the arms (`regbench nuclei --command` prints the line).
# 3. In the scorer's image again: the re-run is compared with the published outlines and
#    the slide's record is written. A disagreement is printed and recorded (exact: false).
#
#    A re-run under 0.90 of the outlines with exactly one nucleus is REFUSED: the slide
#    gets no nuclei, a `REFUSED <slide>` line says why, and this task exits 1.
#
# Needs REGBENCH_MANIFEST and REGBENCH_NUCLEI, and the site's model (the pipeline's
# segmentation_model and segmentation_model_dir): REGBENCH_STARDIST_MODEL and
# REGBENCH_STARDIST_MODEL_DIR, or, when they are unset, a site config that sets the two
# keys (REGBENCH_SITE_CONFIG, else conf/ieo.config of the checkout; any StarDist setting
# it overrides is rendered too). REGBENCH_STARDIST_CPU=1 runs StarDist without a GPU.
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
: "${REGBENCH_MANIFEST:?export REGBENCH_MANIFEST (the rounds.csv of arms_manifest)}"
: "${REGBENCH_NUCLEI:?export REGBENCH_NUCLEI (the folder that keeps the nuclei of each slide)}"
: "${SLURM_ARRAY_TASK_ID:?this task is one slide of an array: submit with --array}"
scorer="${REGBENCH_PREPARE_IMAGE:-STARE_REGBENCH}"
what=(nuclei --manifest "$REGBENCH_MANIFEST" --out "$REGBENCH_NUCLEI"
      --index "$SLURM_ARRAY_TASK_ID" --workers "${SLURM_CPUS_PER_TASK:-1}")
mkdir -p "$REGBENCH_NUCLEI"

# REGBENCH_NUCLEI_FORCE=1: check and segment again whatever is kept (a refused re-run,
# after the model or a setting was corrected)
force=()
[ "${REGBENCH_NUCLEI_FORCE:-0}" = 1 ] && force=(--force)
# what is there first. It exits 1 for a slide whose kept re-run was REFUSED: that is
# this task's verdict when nothing is run again, not a reason to stop before asking
# what the slide still needs.
first=0
regbench_py "$scorer" "${what[@]}" ${force[@]+"${force[@]}"} || first=$?
line="$(regbench_py "$scorer" "${what[@]}" --command ${force[@]+"${force[@]}"})"
if [ -z "$line" ]; then
    if [ "$first" -ne 0 ]; then
        echo "slide $SLURM_ARRAY_TASK_ID: its StarDist re-run was REFUSED and is not run again (nuclei --list has the reason). Correct the model or the settings, then REGBENCH_NUCLEI_FORCE=1 runs it again. This task fails."
        exit "$first"
    fi
    echo "slide $SLURM_ARRAY_TASK_ID: nothing to segment (usable nuclei exist: nuclei --list); StarDist is not run"
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
trap 'rm -f "$script"' EXIT
printf 'set -euo pipefail\n%s\n' "$line" > "$script"
# the GPU reaches the StarDist image only (REGBENCH_GPU=1 passes --nv)
gpu=1
[ "${REGBENCH_STARDIST_CPU:-0}" != 0 ] && gpu=0
REGBENCH_GPU="$gpu" regbench_exec STARDIST bash "$script"
REGBENCH_GPU=0 regbench_py "$scorer" "${what[@]}"
