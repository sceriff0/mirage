#!/bin/bash
# Submit the benchmark: one array per (dataset, method), then the scorer.
#
#   export REGBENCH_CASES=... REGBENCH_OUT=... [REGBENCH_SIF_VALIS=... REGBENCH_CONDA_DEEPERHISTREG=...]
#   benchmarks/regbench/slurm/submit.sh [--datasets "synthetic semisynth multiplex"] \
#       [--methods "stare valis deeperhistreg"] [--max-parallel 20] [-- extra sbatch args]
#
# Every job is submitted at the first rung of the ladder (REGBENCH_CPUS, REGBENCH_MEM). This
# script does not escalate a run that ran out of memory and does not gate on the selfcheck.
#
# The cases must already be prepared (`python -m benchmarks.regbench prepare ...` or
# prepare.sbatch): the array sizes come from them. Extra sbatch args (after --) go to every
# job, e.g. `-- --partition=cpu --account=lab --constraint=icelake`. Pin one node type with
# --constraint if the cluster is heterogeneous, or the timings compare CPUs, not methods.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

datasets="synthetic semisynth multiplex" methods="stare valis deeperhistreg"
maxpar=20
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --max-parallel) maxpar="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
mkdir -p "$REGBENCH_OUT/logs"
cd "$REGBENCH_OUT/logs"
alloc=(--cpus-per-task="$REGBENCH_CPUS" --mem="$REGBENCH_MEM" --time="$REGBENCH_TIME")

ids=()
for ds in $datasets; do
    n=$(regbench_py stare list --cases "$REGBENCH_CASES" --dataset "$ds")
    if [ "$n" -eq 0 ]; then echo "$ds: no prepared cases under $REGBENCH_CASES, skipped"; continue; fi
    for m in $methods; do
        # shellcheck disable=SC2206
        per_method=($(_regbench_var SBATCH "$m"))
        j=$(sbatch --parsable --export="ALL,METHOD=$m,DATASET=$ds" --job-name="rb-$m-$ds" \
            --array="0-$((n - 1))%$maxpar" "${alloc[@]}" "${extra[@]}" "${per_method[@]}" \
            "$HERE/run_array.sbatch")
        j="${j%%;*}"
        ids+=("$j")
        printf '%s\t%s\t%s\n' "$ds" "$m" "$j" >> jobs.tsv
        echo "$ds / $m: job $j ($n cases)"
    done
done
[ ${#ids[@]} -gt 0 ] || { echo "nothing submitted"; exit 1; }
dep=$(IFS=:; echo "${ids[*]}")
js=$(sbatch --parsable --export=ALL --dependency=afterany:"$dep" --cpus-per-task="$REGBENCH_CPUS" \
    --mem=32G --time=04:00:00 "${extra[@]}" "$HERE/score.sbatch")
echo "score: job $js"
echo "results -> $REGBENCH_OUT  (tables/summary.md when the scorer finishes)"
