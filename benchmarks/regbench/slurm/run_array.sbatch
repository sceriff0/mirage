#!/bin/bash
#SBATCH --job-name=regbench-run
#SBATCH --output=regbench-%x-%A_%a.log
# One case per array task: --index is SLURM_ARRAY_TASK_ID into `regbench list`.
# Submitted by submit.sh, which sets METHOD, DATASET and the allocation.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
work="${REGBENCH_WORK:-${TMPDIR:-/tmp}}/regbench-$METHOD-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}-${SLURM_ARRAY_TASK_ID:-0}"
args=(run --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --dataset "$DATASET" --method "$METHOD"
      --workers "$SLURM_CPUS_PER_TASK" --work "$work")
# CASE_ID addresses one named case (submit_scale.sh); otherwise the array index picks it.
if [ -n "${CASE_ID:-}" ]; then args+=(--case-id "$CASE_ID"); else args+=(--index "$SLURM_ARRAY_TASK_ID"); fi
# ARM_LABEL / ARM_OPTS (submit.sh --arm, submit_calibrate.sh) take the place of the
# per-method REGBENCH_LABEL_<M> / REGBENCH_OPTS_<M>.
label="${ARM_LABEL:-$(_regbench_var LABEL "$METHOD")}"
[ -n "$label" ] && args+=(--label "$label")
for o in ${ARM_OPTS-$(_regbench_var OPTS "$METHOD")}; do args+=(--opt "$o"); done
regbench_py "$METHOD" "${args[@]}"
rm -rf "$work"
# A failed registration is recorded in run.json and scored at the initial pose, so the
# task exits 0 either way; the scorer's n_imputed / n_failed columns count them.
exit 0
