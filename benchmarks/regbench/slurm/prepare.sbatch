#!/bin/bash
#SBATCH --job-name=regbench-prepare
#SBATCH --output=regbench-prepare-%j.log
# Prepare one dataset's cases. Arguments are passed to `regbench prepare`, e.g.
#   sbatch -c 16 --mem=64G -t 8:00:00 prepare.sbatch multiplex --manifest rounds.csv --diameter 24
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
dataset="$1"; shift
regbench_py stare prepare "$dataset" --cases "$REGBENCH_CASES" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
