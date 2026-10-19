#!/bin/bash
#SBATCH --job-name=regbench-synth
#SBATCH --output=regbench-synth-%j.log
# One step of the synthetic scaling set (T32, benchmarks/regbench/synth.py), written under
# <to>/synthetic/ (REGBENCH_CASES_SYNTHETIC). submit_cohort.sh --prepare --datasets synthetic
# submits the three steps, each after the one before it (afterany):
#
#   synth.sbatch plan   <to> --source <slide> [--grid default|small] [--seed N] [--channel C]
#       ONE job: prints the bytes it will write, then fixes the set (its texture, its units).
#       Refuses a <to>/synthetic that exists: the set is generated once and shared.
#   synth.sbatch unit   <to>        (an ARRAY, one task per (patient, panel):
#       SLURM_ARRAY_TASK_ID is the unit) writes that slide; a slide already written is kept
#   synth.sbatch finish <to>
#       ONE job: writes the cases (case.json last); refuses a set with a missing slide
#
# The source slide must be readable inside the image: under REGBENCH_INPUTS, REGBENCH_CASES
# or a folder named in REGBENCH_BIND (env.sh: regbench_exec binds those).
set -euo pipefail
: "${REGBENCH_REPO:?export REGBENCH_REPO (this repository) before sbatch}"
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
step="$1" to="$2"; shift 2
case "$step" in
    plan) what=(--plan "$@") ;;
    unit)
        : "${SLURM_ARRAY_TASK_ID:?synth.sbatch unit runs as an array task (sbatch --array=0-N)}"
        what=(--unit "$SLURM_ARRAY_TASK_ID" "$@")
        ;;
    finish) what=(--finish "$@") ;;
    *) echo "synth.sbatch: the step is plan, unit or finish, not '$step'" >&2; exit 2 ;;
esac
regbench_start synth dataset=synthetic "step=$step" "unit=${SLURM_ARRAY_TASK_ID:--}"
code=0
regbench_py "${REGBENCH_PREPARE_IMAGE:-stare}" synth --to "$to" "${what[@]}" || code=$?
if [ "$code" = 0 ]; then
    regbench_say_end "REGBENCH_END kind=synth step=$step status=ok cause=- exit=0 log=${REGBENCH_LOG:--}"
else
    regbench_say_end "REGBENCH_END kind=synth step=$step status=failed cause=benchmark whose=benchmark exit=$code log=${REGBENCH_LOG:--}"
fi
exit "$code"
