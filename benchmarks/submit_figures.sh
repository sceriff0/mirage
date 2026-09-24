#!/usr/bin/env bash
#SBATCH --job-name=mirage_figures
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/figures_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/figures_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G             # the Nextflow heads (-Xmx4g each) + the renders
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE FIGURES -- every figure, from one samplesheet, in one job
# ============================================================================
# Driven by benchmarks/configs/figures.yaml (or CONFIG=<file>), expanded by
# benchmarks/build_figure_plan.py. The two EXPENSIVE phases are .done-marked and skipped
# on a re-submit; the figures are always redrawn, which is the point -- change a size or a
# colour, resubmit, and only the drawing happens:
#
#   1. preprocessing + one REGISTRATION per arm, and the mosaic over all of them
#      (benchmarks/submit_mosaic.sh, called here -- it already knows how to build
#      VALIS, STARE and ASHLAR and to draw their mosaic)
#   2. the overlays, off each arm's registered slides
#   3. one SEGMENTATION per method, resuming from reference_arm's registered.csv
#   4. the zoom figures, the segmented crops and the channel crops
#
# Phases 1 and 3 are the whole cost. Everything in 2 and 4 is a re-render of slides
# already on disk -- seconds each.
#
# EVERYTHING IS WRITTEN INTO THE DIRECTORY YOU SUBMIT FROM:
#   <submit dir>/preprocess_shared/, <arm>/           the runs
#   <submit dir>/mosaic/                              <pid>[_vN]_mosaic.{png,pdf}
#   <submit dir>/overlay/<arm>/f<field>_z<zoom>/      <pid>_<round>[_vN]_before/after
#   <submit dir>/seg_<method>/                        the segmentation run
#   <submit dir>/zoom/<method>/f<field>_<mask>/       <pid>_zoom.{png,pdf}
#   <submit dir>/crops/<method>/f<field>_p<px>_<mask>/<pid>_crop.{png,pdf}
#   <submit dir>/crops/channels/f<field>_p<px>/       <pid>_<channel>_crop.{png,pdf}
#
# Submit (login node):
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/figures_033 && cd $_
#   cp ~/pipelines/mirage/benchmarks/configs/figures.yaml .      # edit the axes
#   sbatch --export=ALL,CONFIG=figures.yaml ~/pipelines/mirage/benchmarks/submit_figures.sh input.csv
#
# See what it expands to first (seconds, no job):
#   python3 -m benchmarks.build_figure_plan --config figures.yaml --count
# ============================================================================
# No `set -u` / `set -o pipefail`: this job sources ~/.bashrc and runs `conda activate`,
# whose scripts read variables a batch job leaves unset. Failures are checked explicitly.

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT="${INPUT:-${1:-}}"
ROOT="${ROOT:-$SUBMIT_DIR}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"          # checkout on `benchmarking`
CONFIG="${CONFIG:-$SRC_DIR/benchmarks/configs/figures.yaml}"
PROFILES="${PROFILES:-singularity,ieo}"
SITE_CONFIG="${SITE_CONFIG:-$SRC_DIR/conf/ieo.config}"
CONDA_ENV="${CONDA_ENV:-nf-env}"
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"    # µm/px; 'auto' = the file's own calibration
SKIP_REGISTRATION="${SKIP_REGISTRATION:-0}"  # 1 = phase 1 is already done elsewhere
# -------------------------------------------------------------------------------

[[ -n "$INPUT" ]] || { echo "usage: sbatch submit_figures.sh <samplesheet.csv>  (or --export=ALL,INPUT=...)" >&2; exit 1; }
[[ "$INPUT" = /* ]] || INPUT="$SUBMIT_DIR/$INPUT"
[[ -s "$INPUT" ]] || { echo "samplesheet $INPUT not found or empty" >&2; exit 1; }
[[ "$CONFIG" = /* ]] || CONFIG="$SUBMIT_DIR/$CONFIG"
[[ -s "$CONFIG" ]] || { echo "config $CONFIG not found or empty" >&2; exit 1; }
# ROOT MUST BE ABSOLUTE, and this is not cosmetic. Several steps below run inside
# `(cd "$SRC_DIR" && ...)` while writing to "$ROOT/..."; a relative ROOT re-resolves against
# the checkout there and the write lands somewhere that does not exist. Measured as job
# 7052347: ROOT=. made every segmentation die on
#   FileNotFoundError: '.launch/seg_stardist/params.json'
# -- inside $SRC_DIR, not inside ROOT -- and phase 2 reported all three methods failed.
# INPUT and CONFIG are already absolutised just above; ROOT was the one that was not.
mkdir -p "$ROOT/.launch" || { echo "cannot create $ROOT/.launch" >&2; exit 1; }
ROOT=$(cd "$ROOT" && pwd) || { echo "cannot resolve ROOT=$ROOT" >&2; exit 1; }
cd "$ROOT" || exit 1

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"

# CellSAM downloads its weights from users.deepcell.org unless cellsam_model_path is set,
# and needs DEEPCELL_ACCESS_TOKEN in the ENVIRONMENT (nextflow.config's singularity
# .envWhitelist forwards it by name into the container). Two ways it silently is not there
# even though your interactive shell has it:
#   * most ~/.bashrc files begin with `case $- in *i*) ;; *) return;; esac`, so the `source`
#     above returns immediately in a batch job and sets nothing;
#   * `DEEPCELL_ACCESS_TOKEN=...` without `export` is a shell variable, not an environment
#     one, so it never reaches nextflow's children.
# Cover both, and say which happened.
ensure_deepcell_token() {
  if [[ -n "$DEEPCELL_ACCESS_TOKEN" ]]; then export DEEPCELL_ACCESS_TOKEN; return 0; fi
  local line
  line=$(grep -hE '^[[:space:]]*(export[[:space:]]+)?DEEPCELL_ACCESS_TOKEN=' \
           ~/.bashrc ~/.bash_profile ~/.profile 2>/dev/null | tail -1)
  if [[ -n "$line" ]]; then
    eval "${line#*export }" 2>/dev/null || eval "$line" 2>/dev/null
    export DEEPCELL_ACCESS_TOKEN
    echo "[token] DEEPCELL_ACCESS_TOKEN taken from a shell rc file (this batch shell is"
    echo "        non-interactive, so sourcing ~/.bashrc can return before setting it)"
  fi
  [[ -n "$DEEPCELL_ACCESS_TOKEN" ]]
}
command -v python3 >/dev/null || { echo "python3 not on PATH (check CONDA_ENV)" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/reg_crop.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/reg_crop.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }

export NXF_VER="${NXF_VER:-25.04.7}"
export NXF_OPTS="${NXF_OPTS:--Xms256m -Xmx4g}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
export NXF_SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR:-$SINGULARITY_CACHEDIR}"
export APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-true}"
export SINGULARITY_DISABLE_CACHE="${SINGULARITY_DISABLE_CACHE:-$APPTAINER_DISABLE_CACHE}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$NXF_SINGULARITY_CACHEDIR/.pull_tmp}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_TMPDIR"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
ensure_sif() {                     # one pull per image; `singularity exec docker://` re-pulls
  local ref="$1" f tmp                                  # on EVERY call with the cache disabled
  f="$NXF_SINGULARITY_CACHEDIR/$(printf '%s' "$ref" | tr '/:' '--').img"
  if [[ ! -s "$f" ]]; then
    tmp="$f.partial.$$"
    echo "[images] pulling docker://$ref -> $f" >&2
    if singularity pull "$tmp" "docker://$ref" >&2 && mv -f "$tmp" "$f"; then :; else
      rm -f "$tmp"
      echo "[images] WARNING: could not pull $ref; steps will fetch docker://$ref each time" >&2
      printf 'docker://%s' "$ref"; return
    fi
  fi
  printf '%s' "$f"
}
# The renders need matplotlib + scikit-image + tifffile AND imagecodecs (LZW slides; the
# segeval image cannot decode them -- job 6844142). The quantify image carries all four.
RENDER_EXEC="${RENDER_EXEC:-singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-quantify:1.0.0)}"

PLAN="$ROOT/.launch/figure_plan.tsv"
(cd "$SRC_DIR" && python3 -m benchmarks.build_figure_plan --config "$CONFIG") > "$PLAN" \
  || { echo "could not expand $CONFIG (see the error above)" >&2; exit 1; }
SUMMARY=$(cd "$SRC_DIR" && python3 -m benchmarks.build_figure_plan --config "$CONFIG" --count)

read_opt() { (cd "$SRC_DIR" && python3 -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
node = cfg
for key in sys.argv[2].split('.'):
    node = (node or {}).get(key) if isinstance(node, dict) else None
print(sys.argv[3] if node is None or node == '' else node)
" "$CONFIG" "$1" "$2"); }
# the ROI, the patients and every drawing option are decided in the plan, not here
PATIENT=$(read_opt options.patient "")
SEG_QC=$(read_opt options.seg_qc 0)
ASHLAR_TILE=$(read_opt options.ashlar_tile 1024)
ASHLAR_SHIFT_UM=$(read_opt options.ashlar_shift_um 15)
FORMATS=$(read_opt options.formats png,pdf)
DPI=$(read_opt options.dpi 100)
REF_ARM=$(read_opt reference_arm valis_high_micro2)
# segmentation.params: extra pipeline params for EVERY segmentation run, as name=value.
# This is how a backend's assets are pinned -- cellsam_model_path being the one that
# matters here: with it, CellSAM never reaches users.deepcell.org, which a compute node
# cannot do anyway (job 68633*).
SEG_PARAMS=$(cd "$SRC_DIR" && python3 -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
for k, v in ((cfg.get('segmentation') or {}).get('params') or {}).items():
    print(f'{k}={v}')
" "$CONFIG" | tr '\n' ' ')
ASHLAR_DIR="ashlar_t${ASHLAR_TILE}_s${ASHLAR_SHIFT_UM}"
arm_dir() { [[ "$1" == ashlar ]] && printf '%s' "$ASHLAR_DIR" || printf '%s' "$1"; }

echo "=================================================="
echo "Figures job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Input:    $INPUT"
echo "Config:   $CONFIG"
echo "Plan:     $SUMMARY"
echo "Root:     $ROOT   pixel size: $PIXEL_SIZE   seg QC: $SEG_QC"
echo "Checkout: $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD))"
echo "=================================================="

# ---- 1. registration: build the arms that are not already on disk -----------------
# An arm given in the YAML as {name, dir} already exists (an arms-benchmark run, say):
# it is listed here and never rebuilt. The rest are built by submit_mosaic.sh, which
# already knows how -- called, not duplicated -- and which draws the mosaic over them.
# bash 3.2 (macOS) has no associative arrays and no mapfile, and this script is read on
# both: the reused arms live in a two-column file and are looked up with awk.
ARM_DIRS="$ROOT/.launch/arm_dirs.tsv"
(cd "$SRC_DIR" && python3 -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
for e in cfg.get('arms') or []:
    if isinstance(e, dict) and e.get('dir'):
        print(e['name'], e['dir'], sep='\t')
" "$CONFIG") > "$ARM_DIRS"
REUSED=$(wc -l < "$ARM_DIRS" | tr -d ' ')
BUILD=$(cd "$SRC_DIR" && python3 -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
for e in cfg.get('arms') or []:
    if not isinstance(e, dict):
        print(e)
" "$CONFIG")

resolve() {                        # resolve <run key> -> a directory
  local label existing
  case "$1" in
    seg:*)  printf '%s' "$ROOT/seg_${1#seg:}" ;;
    arm:*)  label="${1#arm:}"
            existing=$(awk -F'\t' -v l="$label" '$1 == l { print $2; exit }' "$ARM_DIRS")
            if [[ -n "$existing" ]]; then printf '%s' "$existing"
            elif [[ "$label" == ashlar ]]; then printf '%s' "$ROOT/$ASHLAR_DIR"
            else printf '%s' "$ROOT/$label"; fi ;;
    *)      printf '%s' "$ROOT" ;;
  esac
}

if [[ "$SKIP_REGISTRATION" == "1" || -z "$BUILD" ]]; then
  echo "[phase 1] nothing to build ($REUSED arm(s) reused, SKIP_REGISTRATION=$SKIP_REGISTRATION)"
else
  echo "[phase 1] building: $(echo "$BUILD" | tr '\n' ' ')"
  # DRAW=0: it builds the arms and stops. Every mosaic is a plan row like any other figure
  # and is drawn in phase 3, with its own patch size, kind, numbers, ROI and variants --
  # letting it draw one here as well would just produce an extra, unasked-for mosaic.
  mosaic_sh="$SRC_DIR/benchmarks/submit_mosaic.sh"
  env ROOT="$ROOT" SRC_DIR="$SRC_DIR" PROFILES="$PROFILES" SITE_CONFIG="$SITE_CONFIG" \
      CONDA_ENV="$CONDA_ENV" PIXEL_SIZE="$PIXEL_SIZE" SEG_QC="$SEG_QC" PATIENT="$PATIENT" \
      ASHLAR_TILE="$ASHLAR_TILE" ASHLAR_SHIFT_UM="$ASHLAR_SHIFT_UM" DRAW=0 \
      bash "$mosaic_sh" "$INPUT" \
    || echo "[phase 1] one or more arms FAILED -- drawing from the ones that finished" >&2
  # not fatal: submit_mosaic.sh builds the arms independently, so a VALIS that died must
  # not cost the STARE overlays, the segmentation, or the channel crops. A figure whose own
  # arm is missing is reported by name in phase 3 and skipped.
fi

common=(--formats "$FORMATS" --dpi "$DPI")
[[ "$PIXEL_SIZE" != auto ]] && common+=(--pixel-size-um "$PIXEL_SIZE")

# ---- 2. one segmentation per method, resuming from the reference arm --------------
segment() {                        # segment <method>
  local method="$1" run rundir rc=0 px=() ref
  ref="$(resolve "arm:$REF_ARM")/csv/registered.csv"
  run="$ROOT/seg_$method"; rundir="$ROOT/.launch/seg_$method"
  if [[ -f "$run/.done" ]]; then echo "[seg:$method] DONE already, skipping"; return 0; fi
  [[ -s "$ref" ]] || { echo "[seg:$method] no $ref -- did phase 1 run?" >&2; return 1; }
  command -v nextflow >/dev/null \
    || { echo "[seg:$method] nextflow not on PATH (check CONDA_ENV)" >&2; return 1; }
  mkdir -p "$run/trace" "$rundir"
  [[ "$PIXEL_SIZE" != auto ]] && px=("pixel_size=$PIXEL_SIZE")
  # shellcheck disable=SC2086
  (cd "$SRC_DIR" && python3 -m benchmarks.params_json --out "$rundir/params.json" \
      "${px[@]+"${px[@]}"}" cleanup_level=none cleanup_work=false enable_trace=true \
      "trace_dir=$run/trace" "seg_method=$method" start=segmentation stop=segmentation \
      $SEG_PARAMS) \
    || { echo "[seg:$method] could not build params" >&2; return 1; }
  echo "[seg:$method] launching -> $run (from $REF_ARM)"
  (
    cd "$rundir" || exit 1
    nextflow -q run "$SRC_DIR" -profile "$PROFILES" -c "$SITE_CONFIG" \
      -work-dir "$rundir/work" -resume -params-file "$rundir/params.json" \
      --input "$ref" --outdir "$run" \
      > "$run/nextflow.stdout.log" 2> "$run/nextflow.stderr.log"
  ) || rc=$?
  if (( rc != 0 )); then
    echo "[seg:$method] FAILED (exit $rc); last lines of $run/nextflow.stdout.log:" >&2
    tail -n 25 "$run/nextflow.stdout.log" >&2
    return "$rc"
  fi
  [[ -s "$run/csv/segmented.csv" ]] || { echo "[seg:$method] no csv/segmented.csv" >&2; return 1; }
  date '+%F %T' > "$run/.done"
  echo "[seg:$method] OK"
}
methods=$(awk -F'\t' '$2 ~ /^seg:/ { sub(/^seg:/, "", $2); print $2 }' "$PLAN" | awk '!seen[$0]++')
# A backend that cannot run here (cellsam wants users.deepcell.org, and compute nodes have no
# outbound network) must cost its OWN figures and nothing else: the other methods segmented,
# and every overlay, mosaic and channel crop is independent of segmentation entirely.
SEG_FAILED=""
case " $methods " in
  *" cellsam "*)
    if ensure_deepcell_token; then
      echo "[token] DEEPCELL_ACCESS_TOKEN is set (${#DEEPCELL_ACCESS_TOKEN} chars) and exported"
    else
      echo "[token] WARNING: cellsam is requested but DEEPCELL_ACCESS_TOKEN is EMPTY." >&2
      echo "        It will try users.deepcell.org, which a compute node cannot reach." >&2
      echo "        Either export it before sbatch (--export=ALL,DEEPCELL_ACCESS_TOKEN=...)" >&2
      echo "        or pre-download the weights and set segmentation.params.cellsam_model_path." >&2
    fi ;;
esac
for m in $methods; do
  segment "$m" || SEG_FAILED="$SEG_FAILED $m"
done
if [[ -n "$SEG_FAILED" ]]; then
  echo "[phase 2] segmentation FAILED for:$SEG_FAILED -- skipping only their figures" >&2
fi

# ---- 3. every figure in the plan --------------------------------------------------
# The plan carries the tool's whole argument list, shell-quoted: this loop resolves the
# run directory, makes the output directory and runs it. No figure logic lives here.
tool_for() {                       # tool_for <kind> -> the module to run
  case "$1" in
    mosaic)  printf 'benchmarks.reg_mosaic' ;;
    overlay) printf 'benchmarks.reg_overlay' ;;
    zoom|crop) printf 'benchmarks.reg_zoom' ;;
    channel) printf 'benchmarks.reg_crop' ;;
    *)       printf '' ;;
  esac
}
arm_args() {                       # arm_args <comma-separated labels> -> dirs, then labels
  # The DIRECTORIES are printed first and the --label flags after them: ARM_DIR is
  # nargs="+", so argparse stops collecting positionals at the first flag and calls every
  # later directory "unrecognized" (job 6874795 failed exactly that way).
  local label dir dirs=() labels=()
  for label in ${1//,/ }; do
    [[ -n "$label" ]] || continue
    dir=$(resolve "arm:$label")
    [[ -s "$dir/csv/registered.csv" ]] || continue
    dirs+=("$dir"); labels+=("--label" "$(basename "$dir")=$label")
  done
  (( ${#dirs[@]} > 0 )) || return 1
  printf '%s\n' "${dirs[@]}" "${labels[@]}"
}

n=0; failed=0
while IFS=$'\t' read -r kind run out args; do
  [[ -n "$kind" ]] || continue
  tool=$(tool_for "$kind")
  [[ -n "$tool" ]] || { echo "[figures] unknown row kind: $kind" >&2; continue; }
  eval "set -- $args"
  outdir="$ROOT/$out"
  mkdir -p "$outdir"
  case " $SEG_FAILED " in
    *" ${run#seg:} "*)
      if [[ "$run" == seg:* ]]; then
        echo "[figures] SKIPPED: $kind $out (${run#seg:} did not segment)" >&2
        failed=$((failed + 1)); continue
      fi ;;
  esac
  inputs=()
  if [[ "$run" == arms:* ]]; then
    while IFS= read -r line; do inputs+=("$line"); done < <(arm_args "${run#arms:}")
    (( ${#inputs[@]} > 0 )) \
      || { echo "[figures] FAILED: $kind $out -- no arm of ${run#arms:} has registered slides" >&2
           failed=$((failed+1)); continue; }
  else
    inputs=("$(resolve "$run")")
    [[ -d "${inputs[0]}" ]] || { echo "[figures] FAILED: $kind -- no run at ${inputs[0]}" >&2; failed=$((failed+1)); continue; }
  fi
  # shellcheck disable=SC2086
  if (
    cd "$SRC_DIR" || exit 1
    SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
      $RENDER_EXEC python3 -m "$tool" "${inputs[@]}" -o "$outdir" "$@" "${common[@]}"
  ); then
    n=$((n + 1))
  else
    echo "[figures] FAILED: $kind $out" >&2
    failed=$((failed + 1))
  fi
done < "$PLAN"

echo "=================================================="
echo "Done $(date). $n figure(s) under $ROOT ($failed failed or skipped)"
[[ -n "$SEG_FAILED" ]] && echo "  segmentation failed for:$SEG_FAILED"
echo "  mosaic  $ROOT/mosaic/        overlay $ROOT/overlay/<arm>/f<field>_z<zoom>/"
echo "  zoom    $ROOT/zoom/<method>/ crops   $ROOT/crops/"
echo "=================================================="
(( failed == 0 ))
