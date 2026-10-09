#!/bin/bash
# The scaling test in one command, from a fresh clone:
#
#   git clone -b benchmarking https://github.com/sceriff0/stare && cd stare
#   export REGBENCH_CONDA_STARE=... REGBENCH_SIF_VALIS=... REGBENCH_VENV_DEEPERHISTREG=...
#   benchmarks/regbench/slurm/submit_scale.sh [-- --partition=cpu --constraint=<node-type>]
#
# It writes synthetic slides from 1024 to 65536 px a side (one deformation, only the size
# changes), registers each with every method, and scores accuracy and cost against size.
# Per size: one job that generates the pair, then one job per method that waits for it; a
# scorer waits for everything. Jobs of one size get the SAME allocation for every method:
#
#   size     1024  2048  4096  8192  16384  32768  65536
#   memory     8G    8G   16G   32G    64G   128G   256G
#   time     0:30  0:30  1:00  2:00   4:00  12:00  24:00
#
# A method that runs out of memory or time at some size is a result, not an error: its
# run.json says so, slurm_accounting.psv has the scheduler's verdict, and the scorer counts
# it under n_failed.
#
# Options:
#   --sizes "1024 2048 4096"   a subset (default: all seven)
#   --methods "stare valis"    default: stare valis deeperhistreg
#   --mem-scale 2              multiply every memory request (e.g. if a method needs more)
#   --time-scale 2             multiply every time limit
# Paths default to $SCRATCH (or $HOME): REGBENCH_CASES=.../regbench_scale/cases,
# REGBENCH_OUT=.../regbench_scale/results-<commit>. REGBENCH_CPUS (default 8) is per job.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
base="${SCRATCH:-$HOME}/regbench_scale"
export REGBENCH_REPO="${REGBENCH_REPO:-$(cd "$HERE/../../.." && pwd)}"
export REGBENCH_CASES="${REGBENCH_CASES:-$base/cases}"
export REGBENCH_OUT="${REGBENCH_OUT:-$base/results-$(git -C "$REGBENCH_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
source "$HERE/env.sh"

all_sizes=(1024 2048 4096 8192 16384 32768 65536)
mem_gb=(8 8 16 32 64 128 256)
minutes=(30 30 60 120 240 720 1440)
sizes="${all_sizes[*]}" methods="stare valis deeperhistreg" mem_scale=1 time_scale=1
while [ $# -gt 0 ]; do
    case "$1" in
        --sizes) sizes="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --mem-scale) mem_scale="$2"; shift 2 ;;
        --time-scale) time_scale="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
mkdir -p "$REGBENCH_CASES" "$REGBENCH_OUT/logs"
cd "$REGBENCH_OUT/logs"

ids=()
for size in $sizes; do
    idx=-1
    for i in "${!all_sizes[@]}"; do [ "${all_sizes[$i]}" = "$size" ] && idx=$i; done
    [ "$idx" -ge 0 ] || { echo "size $size is not one of ${all_sizes[*]}" >&2; exit 2; }
    case_id=$(printf 'scale_%05d' "$size")
    alloc=(--cpus-per-task="$REGBENCH_CPUS" --mem="$((mem_gb[idx] * mem_scale))G"
           --time="$((minutes[idx] * time_scale))")
    # `--index` is the size's position in the suite, so the job writes exactly this case.
    jp=$(sbatch --parsable --export=ALL --job-name="rb-prep-$size" "${alloc[@]}" "${extra[@]}" \
        "$HERE/prepare.sbatch" synthetic --suite scale --index "$idx")
    jp="${jp%%;*}"
    echo "$size px: generate -> job $jp (${alloc[*]})"
    for m in $methods; do
        # shellcheck disable=SC2206
        per_method=($(_regbench_var SBATCH "$m"))
        j=$(sbatch --parsable --export=ALL,METHOD="$m",DATASET=synthetic,CASE_ID="$case_id" \
            --job-name="rb-$m-$size" --dependency=afterok:"$jp" "${alloc[@]}" "${extra[@]}" \
            "${per_method[@]}" "$HERE/run_array.sbatch")
        j="${j%%;*}"
        ids+=("$j")
        printf 'synthetic\t%s\t%s\t%s\n' "$m" "$j" "$case_id" >> jobs.tsv
        echo "$size px: $m -> job $j"
    done
done
dep=$(IFS=:; echo "${ids[*]}")
js=$(sbatch --parsable --export=ALL --dependency=afterany:"$dep" --cpus-per-task="$REGBENCH_CPUS" \
    --mem=32G --time=06:00:00 "${extra[@]}" "$HERE/score.sbatch")
echo "score: job $js"
echo "cases   -> $REGBENCH_CASES"
echo "results -> $REGBENCH_OUT/tables/summary.md  (scaling table: tables/scaling.csv)"
echo "preview at any time:  PYTHONPATH=\$REGBENCH_REPO/src:\$REGBENCH_REPO/benchmarks python -m regbench score --cases $REGBENCH_CASES --out $REGBENCH_OUT"
