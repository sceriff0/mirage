#!/bin/bash
#SBATCH --job-name=regbench-prepare
#SBATCH --output=regbench-prepare-%j.log
# Prepare one dataset's cases. Arguments are passed to `regbench prepare`, e.g.
#   sbatch -c 4 --mem=64G -t 8:00:00 prepare.sbatch multiplex --manifest rounds.csv --diameter 24
# Submitted as an array (--array=0-N), each task prepares the case of its index.
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
dataset="$1"; shift
# as an array task (submit_cohort.sh --prepare, for HyReCo): one case per task
index=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && index=(--index "$SLURM_ARRAY_TASK_ID")
# REGBENCH_PREPARE_IMAGE: the image to prepare in (submit_cohort.sh sets the scorer's)
regbench_py "${REGBENCH_PREPARE_IMAGE:-stare}" prepare "$dataset" --cases "$REGBENCH_CASES" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@" ${index[@]+"${index[@]}"}
