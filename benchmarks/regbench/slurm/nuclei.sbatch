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
# Needs REGBENCH_MANIFEST, REGBENCH_NUCLEI, REGBENCH_STARDIST_MODEL and
# REGBENCH_STARDIST_MODEL_DIR (the site's model: the pipeline's segmentation_model and
# segmentation_model_dir). REGBENCH_STARDIST_CPU=1 runs StarDist without a GPU.
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

regbench_py "$scorer" "${what[@]}"
line="$(regbench_py "$scorer" "${what[@]}" --command)"
if [ -z "$line" ]; then
    echo "slide $SLURM_ARRAY_TASK_ID: usable nuclei exist; StarDist is not run"
    exit 0
fi
echo "+ $line"
script="$(mktemp "$REGBENCH_NUCLEI/.segment-$SLURM_ARRAY_TASK_ID-XXXXXX.sh")"
trap 'rm -f "$script"' EXIT
printf 'set -euo pipefail\n%s\n' "$line" > "$script"
# the GPU reaches the StarDist image only (REGBENCH_GPU=1 passes --nv)
gpu=1
[ "${REGBENCH_STARDIST_CPU:-0}" != 0 ] && gpu=0
REGBENCH_GPU="$gpu" regbench_exec STARDIST bash "$script"
REGBENCH_GPU=0 regbench_py "$scorer" "${what[@]}"
