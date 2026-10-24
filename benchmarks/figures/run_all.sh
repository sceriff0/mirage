#!/bin/bash
# The benchmark from samplesheet to paper figures, in three steps run from the regbench
# root (the folder whose regbench.env sets REGBENCH_CASES / REGBENCH_OUT):
#
#   bash <checkout>/benchmarks/figures/run_all.sh launch    # 1. submit; returns at once
#   bash <checkout>/benchmarks/figures/run_all.sh status    #    is the benchmark done?
#   bash <checkout>/benchmarks/figures/run_all.sh paper     # 2. once done: tables + figures
#
# 1 launch  launch.sbatch --all: inputs, prepare, every method, the scorer and compute-all,
#           as a chain of short head jobs (benchmarks/regbench/README.md, "The whole
#           benchmark in one command"). Needs REGBENCH_INPUT (the cohort's samplesheet).
#           Resubmitting after a stop reuses every unit that is ok.
# 2 paper   the harness's own paper tables and figures (`regbench paper`, `regbench plots`)
#           into $REGBENCH_OUT/paper/, then make_figures.sh (Fig. 3a-e, 4b, 4c, cost vs
#           size) into $OUTDIR. Refused while any regbench job is queued or running, or
#           before the scorer has written its tables.
#
# Run step 2 BEFORE keep_results.sh: slimming frees the written images that the crop and
# overlay figures read. Step 2 runs here, not in an image: it needs FIG_PYTHON, an
# environment of requirements/ci.txt (the scorer's image has no matplotlib or pandas).
# Extra sbatch options for step 1 (partition, account, ...) go in SBATCH_ARGS.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck disable=SC1091
[ -f ./regbench.env ] && source ./regbench.env
FIG_PYTHON="${FIG_PYTHON:-python}"
JOBS='^(regbench-launch|rb-)'

queued() { squeue --me -h -o '%j' 2>/dev/null | grep -cE "$JOBS" || true; }

case "${1:-}" in
launch)
    : "${REGBENCH_INPUT:?set REGBENCH_INPUT to the cohort samplesheet (input.csv)}"
    # shellcheck disable=SC2086  # SBATCH_ARGS is a list of options
    sbatch ${SBATCH_ARGS:-} "$repo/benchmarks/regbench/slurm/launch.sbatch" --all "${@:2}"
    echo "submitted: follow it with  bash $0 status  (logs: \$REGBENCH_ROOT/logs/)" ;;
status)
    n="$(queued)"
    echo "regbench jobs queued or running: $n"
    squeue --me -h -o '%.12i %.24j %.10T %.10M' 2>/dev/null | grep -E " (regbench-launch|rb-)" || true
    if [ -n "${REGBENCH_OUT:-}" ] && [ -f "$REGBENCH_OUT/tables/cells.csv" ]; then
        echo "scorer tables: $REGBENCH_OUT/tables ($(date -r "$REGBENCH_OUT/tables/cells.csv" '+%F %T'))"
    else
        echo "scorer tables: not written yet"
    fi
    [ "$n" = 0 ] && echo "nothing running: run  bash $0 paper" ;;
paper)
    : "${REGBENCH_CASES:?set REGBENCH_CASES (or run from the regbench root)}"
    : "${REGBENCH_OUT:?set REGBENCH_OUT (or run from the regbench root)}"
    if [ "$(queued)" != 0 ] && [ "${FORCE:-0}" != 1 ]; then
        echo "regbench jobs are still queued or running (bash $0 status); FORCE=1 to draw interim figures"
        exit 1
    fi
    for t in cells.csv summary_resources.csv resources.csv; do
        [ -f "$REGBENCH_OUT/tables/$t" ] || { echo "no $REGBENCH_OUT/tables/$t: the scorer has not run"; exit 1; }
    done
    rb() { (cd "$repo" && PYTHONPATH="$repo" MPLBACKEND=Agg "$FIG_PYTHON" -m benchmarks.regbench "$@"); }
    echo "== the harness's paper tables -> $REGBENCH_OUT/paper"
    rb paper --out "$REGBENCH_OUT" --cases "$REGBENCH_CASES"
    echo "== the harness's paper figures"
    rb plots --out "$REGBENCH_OUT"
    echo "== the publication figures"
    bash "$here/make_figures.sh" "${@:2}" ;;
*)
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
