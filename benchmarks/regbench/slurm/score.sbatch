#!/bin/bash
#SBATCH --job-name=regbench-score
#SBATCH --output=regbench-score-%j.log
# Score every variant under $REGBENCH_OUT, then pull the scheduler's own accounting of the
# run jobs as a cross-check of the in-process resource numbers.
set -euo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
regbench_py stare score --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --workers "$SLURM_CPUS_PER_TASK"
jobs="$REGBENCH_OUT/logs/jobs.tsv"
if [ -s "$jobs" ] && command -v sacct >/dev/null; then
    ids=$(cut -f3 "$jobs" | paste -sd, -)
    sacct -j "$ids" --parsable2 --units=M \
        --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,UserCPU,SystemCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,AllocCPUS,ReqTRES,AllocTRES,NodeList \
        > "$REGBENCH_OUT/tables/slurm_accounting.psv" || true
    cp "$jobs" "$REGBENCH_OUT/tables/slurm_jobs.tsv"
    echo "scheduler accounting -> $REGBENCH_OUT/tables/slurm_accounting.psv"
fi
