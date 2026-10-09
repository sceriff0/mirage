#!/bin/bash
# Submit the benchmark: one array per (dataset, method), then the scorer.
#
#   export REGBENCH_CASES=... REGBENCH_OUT=... [REGBENCH_SIF_VALIS=... REGBENCH_CONDA_DEEPERHISTREG=...]
#   benchmarks/regbench/slurm/submit.sh [--datasets "synthetic anhir multiplex"] \
#       [--methods "stare valis deeperhistreg"] [--arm default|recommended|both] \
#       [--max-parallel 20] [-- extra sbatch args]
#
# --arm default      each package's defaults (results named after the method)
# --arm recommended  VALIS and DeeperHistReg with the options `regbench calibrate` locked in
#                    $REGBENCH_OUT/calibration/params.lock.json (submit_calibrate.sh), or each
#                    package's documented higher-accuracy setting if there is no lock; results
#                    are named <method>_rec. STARE has one arm: its defaults.
# --arm both         (default) the two above
#
# The cases must already be prepared (`python -m regbench prepare ...` or prepare.sbatch): the
# array sizes come from them. Extra sbatch args (after --) go to every job, e.g.
# `-- --partition=cpu --account=lab --constraint=icelake`. Pin one node type with
# --constraint if the cluster is heterogeneous, or the timings compare CPUs, not methods.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

datasets="synthetic semisynth anhir hyreco multiplex" methods="stare valis deeperhistreg"
maxpar=20 arm=both
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --max-parallel) maxpar="$2"; shift 2 ;;
        --arm) arm="$2"; shift 2 ;;
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
        arms="default"
        [ "$arm" = recommended ] && arms="recommended"
        [ "$arm" = both ] && arms="default recommended"
        [ "$m" = stare ] && arms="default"
        for a in $arms; do
            exp="ALL,METHOD=$m,DATASET=$ds" name="$m"
            if [ "$a" = recommended ]; then
                opts=$(regbench_py stare calibrate --cases "$REGBENCH_CASES" \
                    --out "$REGBENCH_OUT/calibration" --opts-for "$m")
                name="${m}_rec"
                # sbatch splits --export on commas; options are space-separated key=value
                export ARM_LABEL="$name" ARM_OPTS="$opts"
                echo "$m recommended arm: ${opts:-(defaults)}"
            else
                unset ARM_LABEL ARM_OPTS
            fi
            j=$(sbatch --parsable --export="$exp" --job-name="rb-$name-$ds" \
                --array="0-$((n - 1))%$maxpar" "${alloc[@]}" "${extra[@]}" "${per_method[@]}" \
                "$HERE/run_array.sbatch")
            j="${j%%;*}"
            ids+=("$j")
            printf '%s\t%s\t%s\n' "$ds" "$name" "$j" >> jobs.tsv
            echo "$ds / $name: job $j ($n cases)"
        done
    done
done
[ ${#ids[@]} -gt 0 ] || { echo "nothing submitted"; exit 1; }
dep=$(IFS=:; echo "${ids[*]}")
js=$(sbatch --parsable --export=ALL --dependency=afterany:"$dep" --cpus-per-task="$REGBENCH_CPUS" \
    --mem=32G --time=04:00:00 "${extra[@]}" "$HERE/score.sbatch")
echo "score: job $js"
echo "results -> $REGBENCH_OUT  (tables/summary.md when the scorer finishes)"
