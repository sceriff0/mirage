#!/bin/bash
#SBATCH --job-name=regbench-prepare
#SBATCH --output=regbench-prepare-%j.log
# Prepare one dataset's cases. Arguments are passed to `regbench prepare`, e.g.
#   sbatch -c 4 --mem=64G -t 8:00:00 prepare.sbatch multiplex --manifest rounds.csv --diameter 24
#   sbatch -c 8 --mem=32G -t 1-00:00:00 --export=ALL,REGBENCH_REPO=<repo>,\
#       REGBENCH_CASES_HYRECO=<cases>,REGBENCH_BIND=<data root>,REGBENCH_PREPARE_IMAGE=STARE_REGBENCH \
#       prepare.sbatch hyreco --data-root <data root>
# The cases go where the launcher reads them (env.sh: regbench_cases_of): the cohort's
# under REGBENCH_CASES, a public dataset's under REGBENCH_CASES_<NAME>, which this job
# binds into the image; the data a public dataset is prepared from is bound through
# REGBENCH_BIND. submit_cohort.sh --prepare submits it as ONE job per dataset (for
# hyreco with REGBENCH_HYRECO_DATA, which it binds). Submitted by hand as an array
# (--array=0-N), each task prepares the case of its index (SLURM_ARRAY_TASK_ID).
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
dataset="$1"; shift
if ! cases="$(regbench_cases_of "$dataset")"; then
    echo "regbench-prepare: REGBENCH_CASES_$(tr '[:lower:]' '[:upper:]' <<< "${dataset%%@*}") is not set: export the folder the $dataset cases are prepared under" >&2
    exit 2
fi
if [ "$dataset" != multiplex ]; then
    # made, and bound into the image (env.sh: _regbench_public_cases binds the folders
    # of REGBENCH_PUBLIC_SETS that exist): this job's dataset alone
    mkdir -p "$cases"
    export REGBENCH_PUBLIC_SETS="$dataset"
fi
regbench_start prepare "dataset=$dataset" "unit=${SLURM_ARRAY_TASK_ID:--}"
export REGBENCH_RECORD="$cases/$dataset"
# only when run as an array task by hand: one case per task
index=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && index=(--index "$SLURM_ARRAY_TASK_ID")
# THE RECORD THAT PREPARE COMPLETED: <cases>/prepared-<dataset>.ok, written only
# when the whole dataset's prepare ended ok and removed when one starts. Cases on disk are
# not that: a prepare cancelled or out of time half-way leaves some (launch.sbatch --all
# runs prepare again unless this file is there). A task of a hand-made array prepares one
# case and leaves the record alone.
done_record="$cases/prepared-$dataset.ok"
[ ${#index[@]} -gt 0 ] || rm -f "$done_record"
# REGBENCH_PREPARE_IMAGE: the image to prepare in (submit_cohort.sh sets the scorer's)
code=0
regbench_py "${REGBENCH_PREPARE_IMAGE:-stare}" prepare "$dataset" --cases "$cases" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@" ${index[@]+"${index[@]}"} || code=$?
if [ "$code" = 0 ] && [ ${#index[@]} -eq 0 ]; then
    printf 'prepare %s ended ok at %s (job %s)\n' "$dataset" "$(date +%FT%T)" "${SLURM_JOB_ID:--}" > "$done_record.new" \
        && mv "$done_record.new" "$done_record" \
        || echo "regbench-prepare: cannot write $done_record: the --all launch will run prepare again" >&2
fi
if [ "$code" = 0 ]; then
    regbench_say_end "REGBENCH_END kind=prepare dataset=$dataset status=ok cause=- exit=0 log=${REGBENCH_LOG:--}"
else
    regbench_say_end "REGBENCH_END kind=prepare dataset=$dataset status=failed cause=benchmark whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
