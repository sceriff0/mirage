#!/usr/bin/env bash
# Profile ONE real mirage run's resources and hand the tables to ihc_method.
#
# Usage:
#   benchmarks/pull_run_resources.sh <run_outdir> <ihc_method_dir> [--trace <trace.txt>] [--handoff <dir>]
#
#   <run_outdir>     the --outdir of a finished mirage run (any run, not only a benchmark arm)
#   <ihc_method_dir> the sibling analysis repo; tables land in <ihc>/data/run_resources/
#   --trace          trace.txt when it is not under <run_outdir>/trace, <run_outdir>/.trace
#                    or <run_outdir>/../.trace (trace_dir defaults to `.trace` beside the
#                    LAUNCH directory, not inside --outdir)
#   --handoff        where the tables are staged first (default: benchmarks/_handoff)
#
# What it runs:  python -m benchmarks.analysis.run_resources --run <run_outdir> --outdir <handoff>/run_resources
# What it copies: run_resources_{tasks,processes,fits,summary}.csv + run_resources.dict.md
#                 -> <ihc>/data/run_resources/            (read by analysis/run_resources.Rmd)
#
# ONE RUN AT A TIME, by design (see the module docstring): the fits inside a run use
# that run's own tasks as points; pooling runs would mix configurations. Running this
# again for another run REPLACES the tables -- the page shows one run.
#
# Companion of pull_to_ihc_method.sh, which calls this for the run it publishes as
# data/mirage/ (step 4b). Kept separate so a plain pipeline run -- no arms, no sweep --
# can be profiled without an arm_results root to point at.
set -euo pipefail

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

RUN=""; IHC=""; TRACE=""; HANDOFF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace)   TRACE="$2";   shift 2 ;;
    --handoff) HANDOFF="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    -*)        echo "unexpected option: $1" >&2; usage 1 ;;
    *) if   [[ -z "$RUN" ]]; then RUN="$1"
       elif [[ -z "$IHC" ]]; then IHC="$1"
       else echo "unexpected argument: $1" >&2; usage 1; fi; shift ;;
  esac
done
[[ -n "$RUN" && -n "$IHC" ]] || usage 1
[[ -d "$RUN" ]] || { echo "no such directory: $RUN" >&2; exit 1; }
[[ -d "$IHC" ]] || { echo "no such directory: $IHC" >&2; exit 1; }
RUN="$(cd "$RUN" && pwd)"
IHC="$(cd "$IHC" && pwd)"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "$HANDOFF" ]] || HANDOFF="$PIPELINE_DIR/benchmarks/_handoff"
STAGE="$HANDOFF/run_resources"

# Resolve the interpreter the way submit_arms.sh does: a conda env can expose
# python only as python3. Assigned in a chain rather than `|| true`.
PY=$(command -v python3) || PY=$(command -v python) || PY=""
[[ -n "$PY" ]] || { echo "no python3/python on PATH" >&2; exit 1; }

echo "=== run resources: $RUN ==="
args=(--run "$RUN" --outdir "$STAGE")
[[ -n "$TRACE" ]] && args+=(--trace "$TRACE")
( cd "$PIPELINE_DIR" && "$PY" -m benchmarks.analysis.run_resources "${args[@]}" )

DEST="$IHC/data/run_resources"
mkdir -p "$DEST"
cp "$STAGE"/run_resources_tasks.csv "$STAGE"/run_resources_processes.csv \
   "$STAGE"/run_resources_fits.csv "$STAGE"/run_resources_summary.csv \
   "$STAGE"/run_resources.dict.md "$DEST"/
echo "  -> $DEST ($(ls "$DEST" | wc -l | tr -d ' ') files)"
echo
echo "In $IHC:  Rscript -e 'workflowr::wflow_build(\"analysis/run_resources.Rmd\")'"
echo "data/ is gitignored in ihc_method — nothing here is committed."
