#!/usr/bin/env bash
#SBATCH --job-name=mirage_zoom
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/zoom_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/zoom_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G             # one Nextflow head (-Xmx4g) + the render: whole-slide overview from a
                              # pyramid level, zoom and mask crop at full resolution
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE ZOOM figure -- whole-slide DAPI overview + zoom on segmented cells, SLURM head job
# ============================================================================
#   1. the pipeline up to SEGMENTATION (StarDist, from conf/ieo.config) -> csv/segmented.csv
#        from a samplesheet:            preprocessing -> registration (VALIS high, micro 2,
#                                       reg_qc=1) -> segmentation, one run
#        or FROM_REGISTERED=<csv/registered.csv of a finished run, e.g. a mosaic's
#                                       valis_high_micro2/>: --start segmentation only
#   2. benchmarks/reg_zoom.py -> zoom/<patient>_zoom.{png,pdf}, <patient>_zoom.json
#        left: the reference slide's DAPI, whole slide, white, a box on the zoom region
#        right: the zoom at full resolution, DAPI white, cells outlined (OUTLINE_COLOR /
#        OUTLINE_WIDTH), joined by a translucent funnel; scale bars (µm, mm from 1 mm);
#        "DAPI" and CELLS_LABEL in their colours, bottom right
#
# EVERYTHING IS WRITTEN INTO THE DIRECTORY YOU SUBMIT FROM, like the other submitters.
#
# Submit (login node):
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/zoom_033 && cd /beegfs/scratch/ieo7660/ihc_method/zoom_033
#   sbatch ~/pipelines/mirage/benchmarks/submit_zoom.sh /path/to/input.csv
#   sbatch --export=ALL,FROM_REGISTERED=/beegfs/scratch/ieo7660/ihc_method/mosaic_033/valis_high_micro2/csv/registered.csv \
#          ~/pipelines/mirage/benchmarks/submit_zoom.sh
#   sbatch --export=ALL,FIELD_UM=200,OUTLINE_COLOR=#00ff00,OUTLINE_WIDTH=2 ~/pipelines/mirage/benchmarks/submit_zoom.sh input.csv
# Re-submitting CONTINUES: the finished segmentation is skipped (.done), so a re-submit with
# other drawing options only redraws.
# ============================================================================
# No `set -u` / `set -o pipefail`: this job sources ~/.bashrc and runs `conda activate`,
# whose scripts read variables a batch job leaves unset. Failures are checked explicitly.

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT="${INPUT:-${1:-}}"                              # samplesheet (unless FROM_REGISTERED)
FROM_REGISTERED="${FROM_REGISTERED:-}"               # a finished run's csv/registered.csv
ROOT="${ROOT:-$SUBMIT_DIR}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"          # checkout on `benchmarking`
PROFILES="${PROFILES:-singularity,ieo}"
SITE_CONFIG="${SITE_CONFIG:-$SRC_DIR/conf/ieo.config}" # pins seg_method=stardist + the model
CONDA_ENV="${CONDA_ENV:-nf-env}"
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"    # µm/px; 'auto' = the file's own calibration
SEG_METHOD="${SEG_METHOD:-stardist}"
PATIENT="${PATIENT:-}"               # empty = the only patient
FIELD_UM="${FIELD_UM:-300}"          # zoom side in µm (drawn at full resolution)
ROI="${ROI:-}"                       # "Y,X" zoom top-left in the reference frame; empty = auto
MASK="${MASK:-cell}"                 # cell | nuclei | both (both = cells in OUTLINE_COLOR and
                                     # nuclei in NUCLEI_COLOR, in one image)
OUTLINE_COLOR="${OUTLINE_COLOR:-#ffd400}"  # any matplotlib colour: yellow, #00ff00, cyan, ...
OUTLINE_WIDTH="${OUTLINE_WIDTH:-1}"  # outline thickness in image px
CELLS_LABEL="${CELLS_LABEL:-StarDist cells}"
NUCLEI_COLOR="${NUCLEI_COLOR:-#00e5ff}"    # nuclear outlines under MASK=both
NUCLEI_LABEL="${NUCLEI_LABEL:-StarDist nuclei}"
CHANNEL_LABEL="${CHANNEL_LABEL:-DAPI}"      # the grey channel's name in the legend
TITLE="${TITLE:-$SEG_METHOD}"        # method, top left on the figure AND on the crop
CROP="${CROP:-none}"                 # none | also | only -- write the outlined crop ALONE as
                                     # <pid>_crop.*, beside the figure (also) or instead (only)
CROP_PX="${CROP_PX:-0}"              # its side in output px (0 = the crop's own pixels, 1:1);
                                     # FIELD_UM sets how much tissue it covers, this the file size
ZOOM_ARGS="${ZOOM_ARGS:-}"           # extra reg_zoom.py flags, e.g. "--overview-px 3000 --pmin 2"
# -------------------------------------------------------------------------------

if [[ -n "$FROM_REGISTERED" ]]; then
  [[ "$FROM_REGISTERED" = /* ]] || FROM_REGISTERED="$SUBMIT_DIR/$FROM_REGISTERED"
  [[ -s "$FROM_REGISTERED" ]] || { echo "FROM_REGISTERED $FROM_REGISTERED not found or empty" >&2; exit 1; }
else
  [[ -n "$INPUT" ]] || { echo "usage: sbatch submit_zoom.sh <samplesheet.csv>  (or --export=ALL,FROM_REGISTERED=<run>/csv/registered.csv)" >&2; exit 1; }
  [[ "$INPUT" = /* ]] || INPUT="$SUBMIT_DIR/$INPUT"
  [[ -s "$INPUT" ]] || { echo "samplesheet $INPUT not found or empty" >&2; exit 1; }
fi
mkdir -p "$ROOT/.launch"
cd "$ROOT" || exit 1

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
command -v nextflow >/dev/null || { echo "nextflow not on PATH (check CONDA_ENV)" >&2; exit 1; }
command -v python3  >/dev/null || { echo "python3 not on PATH (check CONDA_ENV)" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/reg_zoom.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/reg_zoom.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }
[[ -f "$SITE_CONFIG" ]] || { echo "site config $SITE_CONFIG missing" >&2; exit 1; }

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
# Each step runs `singularity exec <image>`. Given docker://..., Apptainer re-downloads and
# re-converts the image on EVERY call (its cache is disabled above) -- ~10 min per step on a
# real run (job 6831633). So each image is pulled ONCE, up front, into the cache under the file
# name Nextflow itself uses (registry/name:tag -> registry-name-tag.img), which also spares
# the pipeline runs their own pull. Written to a temporary name and moved into place, so a
# concurrent reader never sees half an image.
ensure_sif() {                     # ensure_sif <registry/name:tag> -> prints the local image path
  local ref="$1" f tmp
  f="$NXF_SINGULARITY_CACHEDIR/$(printf '%s' "$ref" | tr '/:' '--').img"
  if [[ ! -s "$f" ]]; then
    tmp="$f.partial.$$"
    echo "[images] pulling docker://$ref -> $f" >&2
    if singularity pull "$tmp" "docker://$ref" >&2 && mv -f "$tmp" "$f"; then
      :
    else
      rm -f "$tmp"
      echo "[images] WARNING: could not pull $ref; steps will fetch docker://$ref each time" >&2
      printf 'docker://%s' "$ref"; return
    fi
  fi
  printf '%s' "$f"
}
# The figures need matplotlib + scikit-image + tifffile AND imagecodecs: the slides are LZW,
# and the segeval image cannot decode them (job 6844142). The quantify image carries all four.
RENDER_EXEC="${RENDER_EXEC:-singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-quantify:1.0.0)}"

echo "=================================================="
echo "Zoom job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Input:    ${FROM_REGISTERED:-$INPUT}"
echo "Root:     $ROOT   seg: $SEG_METHOD ($MASK mask)   pixel size: $PIXEL_SIZE"
echo "Checkout: $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD))"
echo "=================================================="

# ---- 1. the pipeline up to segmentation ----------------------------------------
RUN="$ROOT/segmentation"
RUNDIR="$ROOT/.launch/segmentation"
if [[ -n "$FROM_REGISTERED" ]]; then
  STEP_PARAMS=(start=segmentation stop=segmentation); RUN_INPUT="$FROM_REGISTERED"
else
  STEP_PARAMS=(stop=segmentation reg_qc=1 registration_method=valis memory_mode=high reg_micro_reg=2)
  RUN_INPUT="$INPUT"
fi
if [[ -f "$RUN/.done" ]]; then
  echo "[segmentation] DONE already, skipping"
else
  mkdir -p "$RUN/trace" "$RUNDIR"
  px=(); [[ "$PIXEL_SIZE" != auto ]] && px=("pixel_size=$PIXEL_SIZE")
  (cd "$SRC_DIR" && python3 -m benchmarks.params_json --out "$RUNDIR/params.json" \
      "${px[@]+"${px[@]}"}" cleanup_level=none cleanup_work=false enable_trace=true \
      "trace_dir=$RUN/trace" "seg_method=$SEG_METHOD" "${STEP_PARAMS[@]}") \
    || { echo "could not build params" >&2; exit 1; }
  echo "[segmentation] launching -> $RUN (${STEP_PARAMS[*]}, seg_method=$SEG_METHOD)"
  rc=0
  (
    cd "$RUNDIR" || exit 1
    nextflow -q run "$SRC_DIR" -profile "$PROFILES" -c "$SITE_CONFIG" \
      -work-dir "$RUNDIR/work" -resume -params-file "$RUNDIR/params.json" \
      --input "$RUN_INPUT" --outdir "$RUN" \
      > "$RUN/nextflow.stdout.log" 2> "$RUN/nextflow.stderr.log"
  ) || rc=$?
  if (( rc != 0 )); then
    echo "[segmentation] FAILED (exit $rc); last lines of $RUN/nextflow.stdout.log:" >&2
    tail -n 25 "$RUN/nextflow.stdout.log" >&2
    echo "full log: $RUNDIR/.nextflow.log" >&2
    exit "$rc"
  fi
  date '+%F %T' > "$RUN/.done"
fi
[[ -s "$RUN/csv/segmented.csv" ]] || { echo "no $RUN/csv/segmented.csv after segmentation" >&2; exit 1; }

# ---- 2. the zoom figure ------------------------------------------------------------
ARGS=(--field-um "$FIELD_UM" --mask "$MASK" --outline-color "$OUTLINE_COLOR"
      --outline-width "$OUTLINE_WIDTH" --cells-label "$CELLS_LABEL"
      --nuclei-color "$NUCLEI_COLOR" --nuclei-label "$NUCLEI_LABEL"
      --channel-label "$CHANNEL_LABEL" --title "$TITLE"
      --crop "$CROP" --crop-px "$CROP_PX")
[[ -n "$PATIENT" ]] && ARGS+=(--patient "$PATIENT")
[[ -n "$ROI" ]] && ARGS+=(--roi "$ROI")
[[ "$PIXEL_SIZE" != auto ]] && ARGS+=(--pixel-size-um "$PIXEL_SIZE")
# shellcheck disable=SC2086
(
  cd "$SRC_DIR" || exit 1
  SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
    $RENDER_EXEC python3 -m benchmarks.reg_zoom "$RUN" -o "$ROOT/zoom" "${ARGS[@]}" $ZOOM_ARGS
) || { echo "[zoom] FAILED" >&2; exit 1; }

echo "=================================================="
if [[ "$CROP" == "only" ]]; then
  echo "Done $(date). Crop: $ROOT/zoom/<patient>_crop.{png,pdf}"
elif [[ "$CROP" == "also" ]]; then
  echo "Done $(date). Figure: $ROOT/zoom/<patient>_zoom.{png,pdf}  Crop: <patient>_crop.{png,pdf}"
else
  echo "Done $(date). Figure: $ROOT/zoom/<patient>_zoom.{png,pdf}"
fi
echo "=================================================="
