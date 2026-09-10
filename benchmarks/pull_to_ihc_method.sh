#!/usr/bin/env bash
# Pull mirage benchmark artifacts into ihc_method/data/ — the analysis hand-off.
#
# Usage:
#   benchmarks/pull_to_ihc_method.sh <arm_results_root> <ihc_method_dir> [options]
#
#   --sweep <dir>        resource-sweep results root (run_sweep.sh's <results_root>)
#   --sweep-plan <csv>   its run plan   (default: <sweep>_plan.csv, then <sweep>/run_plan.csv)
#   --arm-plan <csv>     the arm plan   (default: <arm_root>_plan.csv)
#   --run <dir>          the full run published as data/mirage/ (default: auto-detect)
#   --handoff <dir>      where built tables are staged (default: benchmarks/_handoff)
#   --build              regenerate the tables from the roots before copying
#   -h | --help
#
# Local (both repos side by side under Github/):
#   benchmarks/pull_to_ihc_method.sh arm_results ../ihc_method --sweep sweep_results --build
#
# From a laptop, pulling off the cluster (note the trailing context: rsync runs
# over ssh only for the SOURCE, so run this ON the cluster and rsync the small
# result out, or mount the cluster path and point at it directly):
#   benchmarks/pull_to_ihc_method.sh /hpcnfs/.../arm_results ~/Github/ihc_method
#
# WHY A SCRIPT AND NOT A COPY. Four different mirage output sets land in three
# different places in ihc_method, and only ONE of them is a whole outdir:
#
#   <arm_root>/<arm>/<patient>/qc|registered   -> data/registration_arms/<arm>/...
#   <arm_root>/arms.csv                        -> data/registration_arms/arms.csv
#   the SWEEP's tables                         -> data/benchmark/
#   one representative run's outdir            -> data/mirage/<patient>/
#
# ONLY THE SMALL ARTIFACTS ARE COPIED. The registered OME-TIFFs and masks are
# multi-GB per patient and no ihc_method page reads them — every figure reads
# JSON/CSV QC. Copying the images would move terabytes to answer nothing, so the
# include filters below are an allowlist, not an optimisation.
#
# THE ARMS AND THE SWEEP ARE DIFFERENT EXPERIMENTS AND MUST NOT SHARE AN OUTDIR.
# make_tables.py and make_figures.py both default to ONE outdir pair
# (benchmarks/paper_data + benchmarks/analysis). Run them for the arms and then
# for the sweep and the second overwrites the first: measurements.csv is
# whichever ran last. Nothing errors — the file exists, carries the right
# columns and plots cleanly — but benchmark_plots.R keys on the SWEEP's axes
# (scaling_grid, target_px, n_channels), so an arm-derived measurements.csv
# renders the scaling pages with data that answers a different question. This
# script therefore builds each experiment into its OWN outdir under
# benchmarks/_handoff/ and copies only the sweep's tables to data/benchmark/.

ARM_ROOT=""; IHC=""; SWEEP_ROOT=""; SWEEP_PLAN=""; ARM_PLAN=""; SRC_RUN=""; BUILD=0
HANDOFF=""

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)      SWEEP_ROOT="$2"; shift 2 ;;
    --sweep-plan) SWEEP_PLAN="$2"; shift 2 ;;
    --arm-plan)   ARM_PLAN="$2";   shift 2 ;;
    --run)        SRC_RUN="$2";    shift 2 ;;
    --handoff)    HANDOFF="$2";    shift 2 ;;
    --build)      BUILD=1;         shift ;;
    -h|--help)    usage 0 ;;
    -*)           echo "unknown option: $1" >&2; usage 1 ;;
    *)            if   [[ -z "$ARM_ROOT" ]]; then ARM_ROOT="$1"
                  elif [[ -z "$IHC"      ]]; then IHC="$1"
                  else echo "unexpected argument: $1" >&2; usage 1; fi; shift ;;
  esac
done

[[ -n "$ARM_ROOT" ]] || { echo "missing <arm_results_root>" >&2; usage 1; }
[[ -n "$IHC"      ]] || { echo "missing <ihc_method_dir>" >&2;   usage 1; }
[[ -d "$ARM_ROOT" ]] || { echo "no such directory: $ARM_ROOT" >&2; exit 1; }
[[ -d "$IHC" ]]      || { echo "no such directory: $IHC" >&2; exit 1; }
ARM_ROOT="$(cd "$ARM_ROOT" && pwd)"
IHC="$(cd "$IHC" && pwd)"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$IHC/_workflowr.yml" ]] || {
  echo "WARNING: $IHC has no _workflowr.yml — is that really the ihc_method repo?" >&2; }

if [[ -n "$SWEEP_ROOT" ]]; then
  [[ -d "$SWEEP_ROOT" ]] || { echo "no such directory: $SWEEP_ROOT" >&2; exit 1; }
  SWEEP_ROOT="$(cd "$SWEEP_ROOT" && pwd)"
fi

# Resolve a run plan next to its results root. run_sweep.sh and the arm Makefile
# both use "<root>_plan.csv"; a plan dropped inside the root is also accepted.
resolve_plan() {
  local given="$1" root="$2"
  if [[ -n "$given" ]]; then printf '%s' "$given"; return; fi
  for c in "${root}_plan.csv" "$root/run_plan.csv" "$root/arm_plan.csv"; do
    [[ -f "$c" ]] && { printf '%s' "$c"; return; }
  done
  printf ''
}
ARM_PLAN="$(resolve_plan "$ARM_PLAN" "$ARM_ROOT")"
[[ -n "$SWEEP_ROOT" ]] && SWEEP_PLAN="$(resolve_plan "$SWEEP_PLAN" "$SWEEP_ROOT")"

PY="$(command -v python3 || command -v python)"

DEST_ARMS="$IHC/data/registration_arms"
DEST_BENCH="$IHC/data/benchmark"
# Staging lives in the repo by default, but a cluster checkout is often read-only
# and the tables are regenerable, so the location is an option.
[[ -n "$HANDOFF" ]] || HANDOFF="$PIPELINE_DIR/benchmarks/_handoff"
SWEEP_TABLES="$HANDOFF/sweep"
ARM_TABLES="$HANDOFF/arms"
mkdir -p "$DEST_ARMS" "$DEST_BENCH"

# ── 0. build ────────────────────────────────────────────────────────────────
# Opt-in because make_figures/make_tables re-read every trace under a results
# root and that is minutes on a full sweep. Without --build the script copies
# whatever tables already exist, which is what you want while a sweep is still
# running and you are refreshing one page.
if [[ "$BUILD" -eq 1 ]]; then
  echo "=== 0/5  building tables (--build) ==="
  if [[ -n "$SWEEP_ROOT" && -n "$SWEEP_PLAN" ]]; then
    mkdir -p "$SWEEP_TABLES"
    ( cd "$PIPELINE_DIR" && \
      "$PY" -m benchmarks.analysis.make_tables \
          --results-root "$SWEEP_ROOT" --run-plan "$SWEEP_PLAN" --outdir "$SWEEP_TABLES" && \
      "$PY" -m benchmarks.analysis.make_figures \
          --results-root "$SWEEP_ROOT" --run-plan "$SWEEP_PLAN" --outdir "$SWEEP_TABLES" ) \
      || echo "  WARNING: sweep table build failed — copying whatever exists" >&2
  elif [[ -n "$SWEEP_ROOT" ]]; then
    echo "  WARNING: --sweep given but no run plan found (looked for" >&2
    echo "           ${SWEEP_ROOT}_plan.csv and $SWEEP_ROOT/run_plan.csv)." >&2
    echo "           Pass --sweep-plan <csv>. Skipping the sweep build." >&2
  fi
  if [[ -n "$ARM_PLAN" ]]; then
    mkdir -p "$ARM_TABLES"
    ( cd "$PIPELINE_DIR" && \
      "$PY" -m benchmarks.analysis.make_tables \
          --results-root "$ARM_ROOT" --run-plan "$ARM_PLAN" --outdir "$ARM_TABLES" ) \
      || echo "  WARNING: arm table build failed — copying whatever exists" >&2
  else
    echo "  note: no arm run plan found; skipping the arm table build" >&2
  fi
fi

echo "=== 1/5  registration arms -> data/registration_arms/ ==="
# --prune-empty-dirs keeps the tree to arms that actually produced QC: an arm
# directory with no seg_qc.json is an arm that failed, and arm_manifest() counts
# any directory holding QC as an arm. Copying empty ones would put an empty box
# in every panel rather than an honest absence.
# The '**/' prefixes are load-bearing. An rsync include pattern containing a '/'
# is anchored to the TRANSFER ROOT, so `--include='qc/**'` matches only a
# top-level qc/ and never <arm>/<patient>/qc/ — which is where every artifact
# this script exists to copy actually lives. The first version of this filter
# silently transferred almost nothing and still exited 0.
rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='**/qc/**' \
  --include='**/registered/summary/**' \
  --include='**/csv/*.csv' \
  --include='**/trace/trace.txt' \
  --exclude='*' \
  "$ARM_ROOT"/ "$DEST_ARMS"/

# Say what was actually transferred. A filter that matches nothing is invisible
# otherwise: rsync exits 0 on an empty transfer.
n_qc=$(find "$DEST_ARMS" -name '*_seg_qc.json' 2>/dev/null | wc -l | tr -d ' ')
echo "  $n_qc seg_qc JSON(s) across $(find "$DEST_ARMS" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') arm dir(s)"
if [[ "$n_qc" -eq 0 ]]; then
  echo "  WARNING: no *_seg_qc.json copied. Either no arm has finished, or the" >&2
  echo "           runs were launched without reg_qc=2 (which is what emits them)." >&2
fi

if [[ -f "$ARM_ROOT/arms.csv" ]]; then
  cp "$ARM_ROOT/arms.csv" "$DEST_ARMS/arms.csv"
  echo "  arms.csv copied (labels explicit — directory-name parsing bypassed)"
else
  # Not fatal, but say so loudly. Without it registration_arms.R parses the
  # directory names, and a mislabelled arm renders cleanly with the conclusion
  # inverted rather than failing.
  echo "  WARNING: no arms.csv at $ARM_ROOT — the consumer will fall back to" >&2
  echo "           parsing directory names. Re-run build_arm_plan.py with" >&2
  echo "           --results-root $ARM_ROOT to generate it." >&2
fi

echo "=== 2/5  SWEEP tables -> data/benchmark/ ==="
# The nine CSVs ihc_method reads out of data/benchmark/, and who writes each:
#   make_tables : runs_master, param_matrix, registration_accuracy,
#                 registration_valis_rtre, segmentation_agreement, scaling_fits
#   make_figures: measurements, resource_stats, run_cost
# ONE source: benchmarks/_handoff/sweep, which is unambiguously the sweep's.
#
# There used to be two fallbacks -- benchmarks/paper_data and
# benchmarks/analysis, the legacy default outdirs -- searched "for a tree built
# before this script grew --sweep". They were the last surviving arm of the
# collision this hand-off split exists to close: the ARMS experiment wrote those
# same two directories (submit_arms.sh printed them as its documented next step)
# and writes the same nine filenames, so a legacy tree could hold either
# experiment's tables with nothing to tell them apart. Copying that into
# data/benchmark/ produces an R page that renders cleanly with the wrong answer.
#
# Failing loudly, naming the target that rebuilds them, is strictly better.
copied=0
for f in "$SWEEP_TABLES"/*.csv; do
  [[ -e "$f" ]] || continue
  cp "$f" "$DEST_BENCH"/ && copied=$((copied + 1))
done

# Name the tables the R pages actually open, so a missing one is visible here
# rather than as an empty panel three steps later.
missing=()
for want in runs_master.csv param_matrix.csv registration_accuracy.csv \
            registration_valis_rtre.csv segmentation_agreement.csv \
            measurements.csv resource_stats.csv run_cost.csv; do
  [[ -f "$DEST_BENCH/$want" ]] || missing+=("$want")
done
echo "  $copied table(s) copied"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "  WARNING: not present: ${missing[*]}" >&2
  echo "           Looked ONLY in $SWEEP_TABLES. The two legacy fallback directories" >&2
  echo "           were removed (see the comment above this block): the ARMS experiment" >&2
  echo "           wrote those same directories and the same filenames, so a hit there" >&2
  echo "           could silently have been the wrong experiment's tables." >&2
  echo "           Re-run with --sweep <results_root> --build, or generate them by hand" >&2
  echo "           INTO THE SWEEP ROOT:" >&2
  echo "             $PY -m benchmarks.analysis.make_tables  --results-root R --run-plan P --outdir $SWEEP_TABLES" >&2
  echo "             $PY -m benchmarks.analysis.make_figures --results-root R --run-plan P --outdir $SWEEP_TABLES" >&2
fi

echo "=== 3/5  ARM tables -> data/registration_arms/ ==="
# registration_arms.Rmd reads the per-arm QC tree copied in step 1; these tables
# are the same arms pre-aggregated, kept BESIDE the arms rather than in
# data/benchmark/ so they can never be mistaken for the sweep's.
copied=0
if [[ -d "$ARM_TABLES" ]]; then
  for f in "$ARM_TABLES"/*.csv; do
    [[ -e "$f" ]] || continue
    cp "$f" "$DEST_ARMS"/ && copied=$((copied + 1))
  done
fi
echo "  $copied table(s)$([[ $copied -eq 0 ]] && echo ' — pass --build with an arm plan to generate them')"

echo "=== 4/5  one full run -> data/mirage/ ==="
# TWO consumers read data/mirage/ and they want different things out of it:
#   registration_run_qc.Rmd  (run_qc.R)      -> <patient>/qc/** and csv/*.csv
#   the mirage cell pages    (mirage_cells.R)-> <patient>/quantification/merged_quant.csv
#                                               <patient>/cell_properties/morphology.csv
#                                               <patient>/phenotyping/phenotypes.csv (optional)
# The filter carries all of them. Copying only qc/ — as this step used to — left
# load_mirage_cells() finding no patient directories at all.
#
# The compute arm is the right source: it is the only arm that ran the FULL
# pipeline, so it is the only one whose tree carries postprocessing QC too.
if [[ -z "$SRC_RUN" ]]; then
  for cand in "$ARM_ROOT"/compute_* "$ARM_ROOT"/valis_high_micro2; do
    [[ -d "$cand" ]] && { SRC_RUN="$cand"; break; }
  done
fi
if [[ -n "$SRC_RUN" && -d "$SRC_RUN" ]]; then
  mkdir -p "$IHC/data/mirage"
  rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='**/qc/**' \
    --include='**/csv/*.csv' \
    --include='**/quantification/*.csv' \
    --include='**/cell_properties/*.csv' \
    --include='**/cell_properties/**/*.csv' \
    --include='**/phenotyping/*.csv' \
    --exclude='*' \
    "$SRC_RUN"/ "$IHC/data/mirage"/
  n_pat=$(find "$IHC/data/mirage" -name merged_quant.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "  from $(basename "$SRC_RUN") — $n_pat patient(s) with merged_quant.csv"
  [[ "$n_pat" -eq 0 ]] && {
    echo "  WARNING: no quantification/merged_quant.csv found. registration_run_qc.Rmd" >&2
    echo "           will still knit, but the mirage cell pages have nothing to load." >&2; }
else
  echo "  no compute_* or valis_high_micro2 arm found — skipped (pass --run <dir>)"
fi

echo "=== 4b   that run's resource profile -> data/run_resources/ ==="
# One run's CPU / wall-time / peak-RSS tables against input size, for
# analysis/run_resources.Rmd. Best effort: a run launched without a trace (or with
# trace_dir somewhere this cannot see) is reported, not fatal -- the other four
# hand-offs are unaffected.
if [[ -n "$SRC_RUN" && -d "$SRC_RUN" ]]; then
  if ! "$PIPELINE_DIR/benchmarks/pull_run_resources.sh" "$SRC_RUN" "$IHC" --handoff "$HANDOFF"; then
    echo "  WARNING: run_resources skipped -- no usable trace.txt for $SRC_RUN." >&2
    echo "           Run benchmarks/pull_run_resources.sh <run> $IHC --trace <trace_dir>/trace.txt by hand." >&2
  fi
else
  echo "  no run selected — skipped"
fi

echo "=== 5/5  summary ==="
printf '  %-28s %s\n' "data/registration_arms/" "$(find "$DEST_ARMS" -type f 2>/dev/null | wc -l | tr -d ' ') file(s)"
printf '  %-28s %s\n' "data/benchmark/"         "$(find "$DEST_BENCH" -type f 2>/dev/null | wc -l | tr -d ' ') file(s)"
printf '  %-28s %s\n' "data/mirage/"            "$(find "$IHC/data/mirage" -type f 2>/dev/null | wc -l | tr -d ' ') file(s)"
printf '  %-28s %s\n' "data/run_resources/"     "$(find "$IHC/data/run_resources" -type f 2>/dev/null | wc -l | tr -d ' ') file(s)"

echo
echo "Done. In $IHC:"
echo "  Rscript -e 'workflowr::wflow_build(c(\"analysis/registration_arms.Rmd\", \\"
echo "                                       \"analysis/benchmark_pipeline.Rmd\", \\"
echo "                                       \"analysis/benchmark_registration.Rmd\", \\"
echo "                                       \"analysis/registration_run_qc.Rmd\", \\"
echo "                                       \"analysis/run_resources.Rmd\"))'"
echo
echo "data/ is gitignored in ihc_method — nothing here is committed."
