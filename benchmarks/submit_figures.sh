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
# benchmarks/build_figure_plan.py. Four phases, each marked .done and skipped on a
# re-submit, so changing only the drawing options redraws without recomputing:
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
mkdir -p "$ROOT/.launch"
cd "$ROOT" || exit 1

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
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
ROI=$(read_opt options.roi "")
PATIENT=$(read_opt options.patient "")
SEG_QC=$(read_opt options.seg_qc 0)
ASHLAR_TILE=$(read_opt options.ashlar_tile 1024)
ASHLAR_SHIFT_UM=$(read_opt options.ashlar_shift_um 500)
OUTLINE_COLOR=$(read_opt options.outline_color "#ffd400")
NUCLEI_COLOR=$(read_opt options.nuclei_color "#00e5ff")
OUTLINE_WIDTH=$(read_opt options.outline_width 2)
CHANNEL_LABEL=$(read_opt options.channel_label DAPI)
AUTOSCALE=$(read_opt options.autoscale clean)
SAT=$(read_opt options.sat 0.35)
BG_K=$(read_opt options.bg_k 3.0)
TITLE=$(read_opt options.title "")
FORMATS=$(read_opt options.formats png,pdf)
DPI=$(read_opt options.dpi 100)
REF_ARM=$(read_opt reference_arm valis_high_micro2)
ASHLAR_DIR="ashlar_t${ASHLAR_TILE}_s${ASHLAR_SHIFT_UM}"
arm_dir() { [[ "$1" == ashlar ]] && printf '%s' "$ASHLAR_DIR" || printf '%s' "$1"; }

echo "=================================================="
echo "Figures job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Input:    $INPUT"
echo "Config:   $CONFIG"
echo "Plan:     $SUMMARY"
echo "Root:     $ROOT   pixel size: $PIXEL_SIZE   roi: ${ROI:-auto}   seg QC: $SEG_QC"
echo "Checkout: $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD))"
echo "=================================================="

# ---- 1. registration arms + the mosaic ------------------------------------------
# submit_mosaic.sh already builds VALIS, STARE and ASHLAR from a samplesheet, skips what
# is .done, and draws the mosaic over whichever arms finished. Called, not duplicated.
mosaic_rows=$(awk -F'\t' '$1 == "mosaic"' "$PLAN")
if [[ "$SKIP_REGISTRATION" == "1" ]]; then
  echo "[phase 1] SKIP_REGISTRATION=1: using the runs already under $ROOT"
else
  while IFS=$'\t' read -r _kind patch variants; do
    [[ -n "$patch" ]] || continue
    echo "[phase 1] registration + mosaic (patch ${patch} um, ${variants} variant(s))"
    mosaic_sh="$SRC_DIR/benchmarks/submit_mosaic.sh"
    mosaic_args="--patch-um $patch"
    [[ -n "$ROI" ]] && mosaic_args="$mosaic_args --roi $ROI"
    env ROOT="$ROOT" SRC_DIR="$SRC_DIR" PROFILES="$PROFILES" SITE_CONFIG="$SITE_CONFIG" \
        CONDA_ENV="$CONDA_ENV" PIXEL_SIZE="$PIXEL_SIZE" SEG_QC="$SEG_QC" PATIENT="$PATIENT" \
        ASHLAR_TILE="$ASHLAR_TILE" ASHLAR_SHIFT_UM="$ASHLAR_SHIFT_UM" VARIANTS="$variants" \
        MOSAIC_ARGS="$mosaic_args" \
        bash "$mosaic_sh" "$INPUT" \
      || { echo "[phase 1] FAILED" >&2; exit 1; }
  done <<< "$mosaic_rows"
  if [[ -z "$mosaic_rows" ]]; then
    echo "[phase 1] no mosaic row: registration must already exist under $ROOT" >&2
  fi
fi

common=(--formats "$FORMATS" --dpi "$DPI")
[[ "$PIXEL_SIZE" != auto ]] && common+=(--pixel-size-um "$PIXEL_SIZE")
[[ -n "$PATIENT" ]] && common+=(--patient "$PATIENT")

# ---- 2. one segmentation per method, resuming from the reference arm --------------
segment() {                        # segment <method>
  local method="$1" run rundir rc=0 px=() ref
  ref="$ROOT/$(arm_dir "$REF_ARM")/csv/registered.csv"
  run="$ROOT/seg_$method"; rundir="$ROOT/.launch/seg_$method"
  if [[ -f "$run/.done" ]]; then echo "[seg:$method] DONE already, skipping"; return 0; fi
  [[ -s "$ref" ]] || { echo "[seg:$method] no $ref -- did phase 1 run?" >&2; return 1; }
  command -v nextflow >/dev/null \
    || { echo "[seg:$method] nextflow not on PATH (check CONDA_ENV)" >&2; return 1; }
  mkdir -p "$run/trace" "$rundir"
  [[ "$PIXEL_SIZE" != auto ]] && px=("pixel_size=$PIXEL_SIZE")
  (cd "$SRC_DIR" && python3 -m benchmarks.params_json --out "$rundir/params.json" \
      "${px[@]+"${px[@]}"}" cleanup_level=none cleanup_work=false enable_trace=true \
      "trace_dir=$run/trace" "seg_method=$method" start=segmentation stop=segmentation) \
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
methods=$(awk -F'\t' '$1 == "zoom" || $1 == "crop" { print $2 }' "$PLAN" | awk '!seen[$0]++')
for m in $methods; do segment "$m" || { echo "[phase 2] segmentation failed" >&2; exit 1; }; done

# ---- 3. every remaining figure ----------------------------------------------------
n=0; failed=0
render() {                         # render <tool> <run> <outdir> <args...>
  local tool="$1" run="$2" out="$3"; shift 3
  mkdir -p "$out"
  # shellcheck disable=SC2086
  if (
    cd "$SRC_DIR" || exit 1
    SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
      $RENDER_EXEC python3 -m "$tool" "$run" -o "$out" "$@"
  ); then n=$((n + 1)); return 0; fi
  failed=$((failed + 1)); return 1
}

while IFS=$'\t' read -r kind a b c d e; do
  case "$kind" in
    overlay)                       # a=arm b=field_um c=zoom_um d=variants
      dir=$(arm_dir "$a")
      args=(--field-um "$b" --variants "$d" --title "$a" "${common[@]}")
      [[ "$c" != 0 ]] && args+=(--zoom-um "$c")
      [[ -n "$ROI" ]] && args+=(--roi "$ROI")
      render benchmarks.reg_overlay "$ROOT/$dir" "$ROOT/overlay/$dir/f${b}_z${c}" "${args[@]}" \
        || echo "[figures] FAILED: overlay $a field=$b zoom=$c" >&2
      ;;
    zoom|crop)                     # a=method b=field_um c=mask d=crop|crop_px
      args=(--mask "$c" --field-um "$b" --outline-color "$OUTLINE_COLOR"
            --nuclei-color "$NUCLEI_COLOR" --outline-width "$OUTLINE_WIDTH"
            --channel-label "$CHANNEL_LABEL" --title "$a" "${common[@]}")
      [[ -n "$ROI" ]] && args+=(--roi "$ROI")
      if [[ "$kind" == zoom ]]; then
        [[ "$d" == none ]] && args+=(--crop none) || args+=(--crop "$d")
        out="$ROOT/zoom/$a/f${b}_${c}"
      else
        args+=(--crop only --crop-px "$d")
        out="$ROOT/crops/$a/f${b}_p${d}_${c}"
      fi
      render benchmarks.reg_zoom "$ROOT/seg_$a" "$out" "${args[@]}" \
        || echo "[figures] FAILED: $kind $a field=$b $c $d" >&2
      ;;
    channel)                       # a=arm b=field_um c=crop_px d=channel e=color
      args=(--channel "$d" --colors "$e" --field-um "$b" --crop-px "$c"
            --autoscale "$AUTOSCALE" --sat "$SAT" --bg-k "$BG_K" --title "$TITLE"
            "${common[@]}")
      [[ -n "$ROI" ]] && args+=(--roi "$ROI")
      render benchmarks.reg_crop "$ROOT/$(arm_dir "$a")" \
        "$ROOT/crops/channels/f${b}_p${c}" "${args[@]}" \
        || echo "[figures] FAILED: channel $d field=$b px=$c" >&2
      ;;
    mosaic) ;;                     # drawn in phase 1
    "") ;;
    *) echo "[figures] unknown plan row: $kind" >&2 ;;
  esac
done < "$PLAN"

echo "=================================================="
echo "Done $(date). $n figure(s) under $ROOT ($failed failed)"
echo "  mosaic  $ROOT/mosaic/        overlay $ROOT/overlay/<arm>/f<field>_z<zoom>/"
echo "  zoom    $ROOT/zoom/<method>/ crops   $ROOT/crops/"
echo "=================================================="
(( failed == 0 ))
