#!/usr/bin/env bash
#SBATCH --job-name=mirage_all_figures
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/all_figures_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/all_figures_%j.err
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G             # submit_figures.sh's heads + renders; the other stages are small
#SBATCH --partition=normal
# ============================================================================
# EVERY FIGURE, ONE JOB: benchmark plots + image composites (+ ANHIR, hand-off, ihc_method)
# ============================================================================
# Five stages, in order. Each one runs only when its inputs are given and reports
# SKIPPED (with the reason) otherwise, so the same script serves a preview on day one
# and the final build at the end:
#
#   stats       benchmarks.analysis.make_figures, once per results root:
#                 arms   ARMS_RESULTS  + ARMS_PLAN      (SRC_DIR)
#                 sweep  SWEEP_RESULTS + SWEEP_PLAN     (SRC_DIR)
#                 drape  DRAPE_RESULTS + DRAPE_PLAN     (DRAPE_SRC) -- LEGACY, see below
#
#   ONE ROOT FOR EVERY METHOD (2026-09-29). submit_arms.sh on benchmarking_new_method now
#   runs VALIS, DRAPE, STARE v1 (pinned-code arms), ASHLAR and seg into ONE results root,
#   and arm_plan.csv carries a `method` column the analysis splits STARE from DRAPE by. So
#   point SRC_DIR at the benchmarking_new_method checkout, ARMS_* at that root, and leave
#   DRAPE_SRC/DRAPE_RESULTS/DRAPE_PLAN UNSET: the drape stage and the --append-arms
#   hand-off exist only for the old two-root layout and would add the DRAPE arms twice.
#               -> $OUT/stats/<name>/   (or $OUT/stats_preview/<name>/ with placeholders)
#   composites  benchmarks/submit_figures.sh inline (mosaic, overlay, zoom, crop, channel),
#               needs INPUT (samplesheet) and CONFIG (figures.yaml) -> $OUT/composites/
#   anhir       benchmarks.anhir.evaluate over every warped-landmark leg present under
#               ANHIR_DIR (ANHIR_LEGS) -> $ANHIR_DIR/tables/
#   handoff     benchmarks/pull_to_ihc_method.sh into IHC: arms + sweep from SRC_DIR,
#               then DRAPE (--append-arms) + ANHIR from DRAPE_SRC. REAL data only.
#   ihc         workflowr::wflow_build of the benchmark pages in IHC (IHC_BUILD=1)
#
# PLACEHOLDER_MISSING=1 (default 0) turns on the marked previews everywhere they exist:
#   stats       make_figures --placeholder-missing: synthetic points hollow/hatched, a red
#               "PLACEHOLDER -- N synthetic points" watermark, placeholders.csv sidecar
#   composites  a composite that did not render becomes a grey hatched CARD naming the
#               missing run (benchmarks/placeholder_card.py). A slide is never synthesised.
#   ihc         IHC_PLACEHOLDER_MISSING=1: figures to output/placeholders/, never
#               output/figures/, and figures/*.R refuses to run
# The hand-off never carries placeholders (pull_to_ihc_method.sh refuses a preview dir).
# A default run (PLACEHOLDER_MISSING=0) deletes every composite card first.
#
# CONCURRENCY. stats/anhir/handoff/ihc are computed INSIDE this job (no SLURM children).
# Only `composites` launches pipeline runs: one registration per arm it has to build and
# one segmentation per method, sequentially; arms given in figures.yaml as {name, dir}
# are reused, never rebuilt. SKIP_COMPOSITE_RUNS=1 skips building arms (phase 1).
#
# Submit (login node). Every knob in --export, NEVER as `VAR=x sbatch` (it does not reach
# the job at this site), and no commas inside a value:
#   B=/beegfs/scratch/ieo7660/ihc_method/benchmark
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/figures_all && cd $_
#   cp ~/pipelines/mirage/benchmarks/configs/figures.yaml .
#   sbatch --export=ALL,PLACEHOLDER_MISSING=1,\
#   ARMS_RESULTS=$B/arm_results,ARMS_PLAN=$B/arm_plan.csv,\
#   SWEEP_RESULTS=$B/bench_results,SWEEP_PLAN=$B/bench_run_plan.csv,\
#   INPUT=/beegfs/scratch/ieo7660/ihc_method/head_neck/input.csv,CONFIG=figures.yaml,\
#   ANHIR_DIR=/beegfs/scratch/ieo7660/ihc_method/anhir,IHC=$HOME/ihc_method,IHC_BUILD=1 \
#     ~/pipelines/mirage/benchmarks/submit_all_figures.sh
# Only some stages:  --export=ALL,STAGES=stats,...     See the plan first:  DRY_RUN=1
# ============================================================================
# No `set -u`: ~/.bashrc and `conda activate` read variables a batch job leaves unset.

# ---- knobs ---------------------------------------------------------------------
OUT="${OUT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"            # the checkout that ran the arms
DRAPE_SRC="${DRAPE_SRC:-}"                              # LEGACY two-root layout only; leave unset
CONDA_ENV="${CONDA_ENV:-nf-env}"
STAGES="${STAGES:-stats composites anhir handoff ihc}"
PLACEHOLDER_MISSING="${PLACEHOLDER_MISSING:-0}"
PLACEHOLDER_SEED="${PLACEHOLDER_SEED:-0}"
REG_EVAL="${REG_EVAL:-none}"
ARMS_RESULTS="${ARMS_RESULTS:-}";   ARMS_PLAN="${ARMS_PLAN:-}"
SWEEP_RESULTS="${SWEEP_RESULTS:-}"; SWEEP_PLAN="${SWEEP_PLAN:-}"
DRAPE_RESULTS="${DRAPE_RESULTS:-}"; DRAPE_PLAN="${DRAPE_PLAN:-}"
INPUT="${INPUT:-}"; CONFIG="${CONFIG:-}"
SKIP_COMPOSITE_RUNS="${SKIP_COMPOSITE_RUNS:-0}"
ANHIR_DIR="${ANHIR_DIR:-}"
ANHIR_LEGS="${ANHIR_LEGS:-drape=results_drape/tiled_warped stare=results_stare/tiled_warped valis=results_valis/valis_warped initial=results_initial_warped bunwarpj=results_bunwarpj_warped}"
IHC="${IHC:-}"; IHC_BUILD="${IHC_BUILD:-0}"
# molecular_massimo2 BEFORE paper_figures: its last chunk writes output/paired_deconv.rds,
# which Fig 5(c) (and Supplementary S11) reads -- without it that panel prints "Needs ...".
# marker_qc carries the cell-level QC (lineage leakage, marker exclusivity).
IHC_PAGES="${IHC_PAGES:-registration_arms benchmark_registration benchmark_anhir benchmark_pipeline registration_run_qc run_resources marker_qc molecular_massimo2 paper_figures}"
DRY_RUN="${DRY_RUN:-0}"
# -------------------------------------------------------------------------------

case "$PLACEHOLDER_MISSING" in 0|1) ;; *) echo "PLACEHOLDER_MISSING must be 0 or 1" >&2; exit 2 ;; esac
abs() { case "$1" in ""|/*) printf '%s' "$1" ;; *) printf '%s/%s' "$OUT" "$1" ;; esac; }
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd) || { echo "cannot use OUT=$OUT" >&2; exit 1; }
CONFIG=$(abs "$CONFIG"); INPUT=$(abs "$INPUT")

if [[ "$DRY_RUN" != 1 ]]; then
  # shellcheck disable=SC1090
  source ~/.bashrc
  conda activate "$CONDA_ENV"
fi
PY=$(command -v python3) || PY=$(command -v python) || PY=""
[[ -n "$PY" ]] || { echo "no python3 on PATH (check CONDA_ENV=$CONDA_ENV)" >&2; exit 1; }

STATUS=()                         # one "<stage>: <outcome>" line per stage, printed at the end
note() { STATUS+=("$1"); echo "[$1]"; }
run() {                           # run <cmd...>: echo it; execute unless DRY_RUN=1
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] && return 0
  "$@"
}
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
mode() { [[ "$PLACEHOLDER_MISSING" == 1 ]] && echo "PREVIEW (placeholders ON)" || echo "REAL (placeholders off)"; }

echo "=================================================="
echo "All-figures job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Mode:     $(mode)   seed $PLACEHOLDER_SEED"
echo "Stages:   $STAGES"
echo "Out:      $OUT"
echo "Checkout: $SRC_DIR   DRAPE: ${DRAPE_SRC:-<none>}"
echo "=================================================="

# ---- stats -----------------------------------------------------------------------
if want stats; then
  if [[ "$DRY_RUN" != 1 ]] && ! "$PY" -c "import pandas, matplotlib, sklearn, yaml" 2>/dev/null; then
    note "stats: FAILED -- $PY cannot import pandas/matplotlib/sklearn/yaml (conda activate $CONDA_ENV)"
  else
    sub=stats; extra=()
    if [[ "$PLACEHOLDER_MISSING" == 1 ]]; then
      sub=stats_preview; extra=(--placeholder-missing --placeholder-seed "$PLACEHOLDER_SEED")
    fi
    stats_one() {                 # stats_one <name> <checkout> <results> <plan>
      local name="$1" src="$2" res="$3" plan="$4"
      if [[ -z "$res" || -z "$plan" ]]; then note "stats/$name: SKIPPED (results root or plan not given)"; return; fi
      if [[ -z "$src" ]]; then note "stats/$name: SKIPPED (no checkout: set DRAPE_SRC)"; return; fi
      if [[ "$DRY_RUN" != 1 && ( ! -d "$res" || ! -s "$plan" ) ]]; then
        note "stats/$name: SKIPPED (missing $res or $plan)"; return; fi
      if (cd "$src" && unset PLACEHOLDER_MISSING && \
            run "$PY" -m benchmarks.analysis.make_figures --results-root "$res" --run-plan "$plan" \
                --reg-eval "$REG_EVAL" --outdir "$OUT/$sub/$name" "${extra[@]+"${extra[@]}"}"); then
        note "stats/$name: OK -> $OUT/$sub/$name"
      else
        note "stats/$name: FAILED (see above)"
      fi
    }
    stats_one arms  "$SRC_DIR"   "$ARMS_RESULTS"  "$ARMS_PLAN"
    stats_one sweep "$SRC_DIR"   "$SWEEP_RESULTS" "$SWEEP_PLAN"
    stats_one drape "$DRAPE_SRC" "$DRAPE_RESULTS" "$DRAPE_PLAN"
  fi
fi

# ---- composites --------------------------------------------------------------------
if want composites; then
  ROOT="$OUT/composites"
  if [[ -z "$INPUT" || -z "$CONFIG" ]]; then
    note "composites: SKIPPED (INPUT and CONFIG are both needed)"
  else
    mkdir -p "$ROOT"
    # A card must never outlive the render that replaces it, in either mode.
    run "$PY" "$SRC_DIR/benchmarks/placeholder_card.py" clear --root "$ROOT"
    run env ROOT="$ROOT" SRC_DIR="$SRC_DIR" CONFIG="$CONFIG" INPUT="$INPUT" \
        SKIP_REGISTRATION="$SKIP_COMPOSITE_RUNS" bash "$SRC_DIR/benchmarks/submit_figures.sh"
    rc=$?
    # submit_figures.sh exits non-zero when ANY figure failed; the ones that rendered are kept.
    outcome="OK"; (( rc != 0 )) && outcome="PARTIAL (some composites failed, see above)"
    if [[ "$PLACEHOLDER_MISSING" == 1 ]]; then
      plan="$ROOT/.launch/figure_plan.tsv"
      if [[ "$DRY_RUN" == 1 || -s "$plan" ]]; then
        run "$PY" "$SRC_DIR/benchmarks/placeholder_card.py" fill --plan "$plan" --root "$ROOT"
        outcome="$outcome; missing slots carded"
      else
        outcome="$outcome; no plan at $plan, nothing carded"
      fi
    fi
    note "composites: $outcome -> $ROOT"
  fi
fi

# ---- anhir -------------------------------------------------------------------------
if want anhir; then
  if [[ -z "$ANHIR_DIR" ]]; then
    note "anhir: SKIPPED (ANHIR_DIR not given)"
  else
    legs=(); absent=()
    for leg in $ANHIR_LEGS; do
      name="${leg%%=*}"; dir="${leg#*=}"
      [[ "$dir" = /* ]] || dir="$ANHIR_DIR/$dir"
      if [[ "$DRY_RUN" == 1 || -s "$dir/warp_index.csv" ]]; then legs+=(--warped "$name=$dir"); else absent+=("$name"); fi
    done
    if (( ${#legs[@]} == 0 )); then
      note "anhir: SKIPPED (no warped leg under $ANHIR_DIR yet)"
    elif (cd "${DRAPE_SRC:-$SRC_DIR}" && run "$PY" -m benchmarks.anhir.evaluate \
            --dataset "$ANHIR_DIR/challenge/anhir/dataset_medium.csv" \
            --landmarks-root "$ANHIR_DIR/challenge/anhir/landmarks" --status training \
            "${legs[@]}" --out "$ANHIR_DIR/tables"); then
      note "anhir: OK -> $ANHIR_DIR/tables${absent[*]:+ (not yet warped: ${absent[*]})}"
    else
      note "anhir: FAILED (see above)"
    fi
  fi
fi

# ---- handoff -----------------------------------------------------------------------
if want handoff; then
  if [[ -z "$IHC" ]]; then
    note "handoff: SKIPPED (IHC not given)"
  else
    # Pass 1 REPLACES arms.csv, pass 2 MERGES into it: the order is load-bearing.
    first=(); [[ -n "$ARMS_PLAN" ]] && first+=(--arm-plan "$ARMS_PLAN")
    [[ -n "$SWEEP_RESULTS" ]] && first+=(--sweep "$SWEEP_RESULTS")
    [[ -n "$SWEEP_PLAN" ]] && first+=(--sweep-plan "$SWEEP_PLAN")
    if [[ -n "$ARMS_RESULTS" ]] && (cd "$SRC_DIR" && unset PLACEHOLDER_MISSING && run \
          benchmarks/pull_to_ihc_method.sh "$ARMS_RESULTS" "$IHC" "${first[@]+"${first[@]}"}" --build); then
      out1="arms+sweep OK"
    else
      out1="arms+sweep SKIPPED/FAILED"
    fi
    second=(--append-arms)
    [[ -n "$DRAPE_PLAN" ]] && second+=(--arm-plan "$DRAPE_PLAN")
    [[ -n "$ANHIR_DIR" && ( "$DRY_RUN" == 1 || -d "$ANHIR_DIR/tables" ) ]] && second+=(--anhir "$ANHIR_DIR/tables")
    if [[ -n "$DRAPE_SRC" && -n "$DRAPE_RESULTS" ]] && (cd "$DRAPE_SRC" && unset PLACEHOLDER_MISSING && run \
          benchmarks/pull_to_ihc_method.sh "$DRAPE_RESULTS" "$IHC" "${second[@]}"); then
      out2="drape+anhir OK"
    else
      out2="drape+anhir SKIPPED/FAILED"
    fi
    note "handoff: $out1; $out2"
  fi
fi

# ---- ihc ---------------------------------------------------------------------------
if want ihc; then
  if [[ -z "$IHC" || "$IHC_BUILD" != 1 ]]; then
    note "ihc: SKIPPED (needs IHC and IHC_BUILD=1)"
  else
    pages=""; for p in $IHC_PAGES; do pages="$pages\"analysis/$p.Rmd\","; done
    if (cd "$IHC" && run env IHC_PLACEHOLDER_MISSING="$PLACEHOLDER_MISSING" \
          Rscript -e "workflowr::wflow_build(c(${pages%,}))"); then
      where="output/figures/"; [[ "$PLACEHOLDER_MISSING" == 1 ]] && where="output/placeholders/"
      note "ihc: OK -> $IHC/docs/*.html, PDFs in $IHC/$where"
    else
      note "ihc: FAILED (see above)"
    fi
  fi
fi

echo "=================================================="
echo "Done $(date). Mode: $(mode)"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "=================================================="
