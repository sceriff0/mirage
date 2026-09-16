#!/usr/bin/env bash
#SBATCH --job-name=mirage_overlay
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/overlay_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/overlay_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G             # one Nextflow head (-Xmx4g) + the overlay render, which runs in this job
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE registration OVERLAY — DAPI Before / After for TWO slides, SLURM head job
# ============================================================================
# A FAST PROTOTYPE figure for one pair of slides of one patient. Input: a samplesheet with
# exactly two rows (the pipeline's usual columns):
#
#   patient_id,path_to_file,is_reference,channels
#   033,/.../033_DAPI_ARID1A_PDL1.nd2,false,PDL1|ARID1A|DAPI
#   033,/.../033_DAPI_CD14_CD56.nd2,false,CD56|CD14|DAPI
#
# If neither row says is_reference=true, row REF_ROW (default 1) becomes the reference.
#
#   1. one pipeline run, preprocessing -> registration (--stop registration)
#   2. benchmarks/reg_overlay.py -> <pair>/overlay/<pid>_<round>_before.{png,pdf}
#                                   <pair>/overlay/<pid>_<round>_after.{png,pdf}
#                                   <pair>/overlay/<pid>_<round>_locator.{png,pdf}, *_overlay.json
#      magenta = moving DAPI, cyan = reference DAPI (white where they overlap), µm scale bar.
#      One FIELD_UM crop, picked on tissue, or ROI="Y,X". AVOID_ROIS_JSON=<a mosaic's
#      <pid>_rois.json> keeps the crop clear of that mosaic's ROIs (applied only when both
#      registered onto the same reference slide).
#
# EVERYTHING IS WRITTEN INTO THE DIRECTORY YOU SUBMIT FROM, one sub-directory per pair
# (<method>_<slide1>__<slide2>/), like the arm and sweep submitters. Keep it off $HOME.
#
# THE PROTOTYPE HAS NO NUMBERS. SEG_QC=0 (default) runs reg_qc=1 -- the Before/After composite,
# no WARP_SEG_QC -- and the images carry NO Dice / Δ. SEG_QC=1 runs reg_qc=2 and prints the
# pipeline scorer's Dice and Δ. NUMBERS overrides: none | scorer | image | auto.
#
# Submit (login node):
#   mkdir -p /hpcnfs/home/ieo7660/pipelines/logs
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/overlay_033 && cd /beegfs/scratch/ieo7660/ihc_method/overlay_033
#   sbatch ~/pipelines/mirage/benchmarks/submit_overlay.sh /path/to/two_slides.csv
# Options go through --export (a bare VAR=x before sbatch has not reached the job here):
#   sbatch --export=ALL,METHOD=tiled,FIELD_UM=800 ~/pipelines/mirage/benchmarks/submit_overlay.sh two_slides.csv
# Re-submitting from the same directory CONTINUES: the finished registration is skipped
# (.done, Nextflow -resume), so a re-submit with another crop only redraws.
# ============================================================================
set -uo pipefail

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT_CSV="${INPUT:-${1:-}}"                         # the 2-row samplesheet; relative = to SUBMIT_DIR
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"          # checkout on `benchmarking`
PROFILES="${PROFILES:-singularity,ieo}"
SITE_CONFIG="${SITE_CONFIG:-$SRC_DIR/conf/ieo.config}"
CONDA_ENV="${CONDA_ENV:-nf-env}"
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"    # µm/px; 'auto' = the ND2's own calibration
METHOD="${METHOD:-valis}"            # valis (memory_mode high, reg_micro_reg 2) | tiled (STARE high)
SEG_QC="${SEG_QC:-0}"                # 0 = reg_qc=1, no WARP_SEG_QC (fast); 1 = reg_qc=2
NUMBERS="${NUMBERS:-}"               # empty = none at SEG_QC=0, scorer values (auto) at SEG_QC=1
REF_ROW="${REF_ROW:-1}"              # which row is the reference when neither row says so
FIELD_UM="${FIELD_UM:-500}"          # crop side in µm
ROI="${ROI:-}"                       # "Y,X" top-left in the reference frame (full-res px); empty = auto
AVOID_ROIS_JSON="${AVOID_ROIS_JSON:-}" # a mosaic's <pid>_rois.json to keep the crop clear of
# -------------------------------------------------------------------------------

[[ -n "$INPUT_CSV" ]] || { echo "usage: sbatch submit_overlay.sh <two_slides.csv>  (or --export=ALL,INPUT=...)" >&2; exit 1; }
[[ "$INPUT_CSV" = /* ]] || INPUT_CSV="$SUBMIT_DIR/$INPUT_CSV"
[[ -s "$INPUT_CSV" ]] || { echo "samplesheet $INPUT_CSV not found or empty" >&2; exit 1; }
[[ -n "$NUMBERS" ]] || { if [[ "$SEG_QC" == "1" ]]; then NUMBERS=auto; else NUMBERS=none; fi; }
case "$METHOD" in valis|tiled) ;; *) echo "METHOD must be valis or tiled, got '$METHOD'" >&2; exit 1 ;; esac

# ---- the samplesheet: two rows, one patient, exactly one reference --------------
ROWS=()
while IFS= read -r line; do [[ -n "${line//[[:space:]]/}" ]] && ROWS+=("$line"); done \
  < <(tail -n +2 "$INPUT_CSV" | tr -d '\r')
(( ${#ROWS[@]} == 2 )) || { echo "$INPUT_CSV has ${#ROWS[@]} slide rows; this script takes exactly 2" >&2; exit 1; }
PATIENT="${ROWS[0]%%,*}"
[[ "${ROWS[1]%%,*}" == "$PATIENT" ]] || { echo "the two rows are different patients (${PATIENT} vs ${ROWS[1]%%,*})" >&2; exit 1; }
n_ref=0; for r in "${ROWS[@]}"; do IFS=',' read -r _ _ isref _ <<< "$r"; [[ "$isref" == true ]] && n_ref=$((n_ref + 1)); done
(( n_ref <= 1 )) || { echo "both rows say is_reference=true" >&2; exit 1; }

slide_name() { local path; path=$(cut -d, -f2 <<< "$1"); path=$(basename "$path"); echo "${path%.*}"; }
NAME="${METHOD}_$(slide_name "${ROWS[0]}")__$(slide_name "${ROWS[1]}")"
ROOT="${ROOT:-$SUBMIT_DIR/$NAME}"
mkdir -p "$ROOT/.launch"
INPUT="$ROOT/input.csv"
{
  echo "patient_id,path_to_file,is_reference,channels"
  for i in 0 1; do
    IFS=',' read -r pid path isref channels <<< "${ROWS[$i]}"
    if (( n_ref == 0 )); then isref=false; (( i + 1 == REF_ROW )) && isref=true; fi
    echo "$pid,$path,$isref,$channels"
  done
} > "$INPUT"
echo "Samplesheet ($INPUT):"; cat "$INPUT"

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
command -v nextflow >/dev/null || { echo "nextflow not on PATH (check CONDA_ENV)" >&2; exit 1; }
command -v python3  >/dev/null || { echo "python3 not on PATH (check CONDA_ENV)" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/reg_overlay.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/reg_overlay.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }
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
sif_or_docker() {
  local ref="$1" f
  f="$NXF_SINGULARITY_CACHEDIR/$(printf '%s' "$ref" | tr '/:' '--').img"
  if [[ -f "$f" ]]; then printf '%s' "$f"; else printf 'docker://%s' "$ref"; fi
}
# reg_overlay.py needs matplotlib + scikit-image + tifffile: the segeval image carries all three.
RENDER_EXEC="${RENDER_EXEC:-singularity exec $SING_BINDS $(sif_or_docker bolt3x/mirage-segeval:1.0.0)}"

echo "=================================================="
echo "Overlay job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Input:    $INPUT_CSV"
echo "Patient $PATIENT  method $METHOD  seg QC $SEG_QC (numbers: $NUMBERS)  pixel size $PIXEL_SIZE"
echo "Root:     $ROOT"
echo "Checkout $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD))"
echo "=================================================="

# ---- 1. preprocessing + registration, one run ------------------------------------
REG_QC=1; [[ "$SEG_QC" == "1" ]] && REG_QC=2
if [[ "$METHOD" == valis ]]; then
  METHOD_PARAMS=(registration_method=valis memory_mode=high reg_micro_reg=2); LABEL="VALIS high"
else
  METHOD_PARAMS=(registration_method=tiled reg_tiled_mode=high); LABEL="STARE high"
fi
RUN="$ROOT/registration"
RUNDIR="$ROOT/.launch/registration"
if [[ -f "$RUN/.done" ]]; then
  echo "[registration] DONE already, skipping"
else
  mkdir -p "$RUN/trace" "$RUNDIR"
  px=(); [[ "$PIXEL_SIZE" != auto ]] && px=("pixel_size=$PIXEL_SIZE")
  (cd "$SRC_DIR" && python3 -m benchmarks.params_json --out "$RUNDIR/params.json" \
      "${px[@]+"${px[@]}"}" cleanup_level=none cleanup_work=false enable_trace=true \
      "trace_dir=$RUN/trace" stop=registration "reg_qc=$REG_QC" "${METHOD_PARAMS[@]}") \
    || { echo "could not build params" >&2; exit 1; }
  echo "[registration] launching -> $RUN (${METHOD_PARAMS[*]}, reg_qc=$REG_QC)"
  rc=0
  (
    cd "$RUNDIR" || exit 1
    nextflow -q run "$SRC_DIR" -profile "$PROFILES" -c "$SITE_CONFIG" \
      -work-dir "$RUNDIR/work" -resume -params-file "$RUNDIR/params.json" \
      --input "$INPUT" --outdir "$RUN" \
      > "$RUN/nextflow.stdout.log" 2> "$RUN/nextflow.stderr.log"
  ) || rc=$?
  if (( rc != 0 )); then
    echo "[registration] FAILED (exit $rc); last lines of $RUN/nextflow.stdout.log:" >&2
    tail -n 25 "$RUN/nextflow.stdout.log" >&2
    echo "full log: $RUNDIR/.nextflow.log" >&2
    exit "$rc"
  fi
  date '+%F %T' > "$RUN/.done"
fi

# ---- 2. the Before / After overlay ---------------------------------------------
EXTRA=()
if [[ -n "$AVOID_ROIS_JSON" ]]; then
  [[ "$AVOID_ROIS_JSON" = /* ]] || AVOID_ROIS_JSON="$SUBMIT_DIR/$AVOID_ROIS_JSON"
  [[ -f "$AVOID_ROIS_JSON" ]] || { echo "AVOID_ROIS_JSON $AVOID_ROIS_JSON not found" >&2; exit 1; }
  EXTRA+=(--avoid-rois-json "$AVOID_ROIS_JSON")
fi
[[ -n "$ROI" ]] && EXTRA+=(--roi "$ROI")
# shellcheck disable=SC2086
(
  cd "$SRC_DIR" || exit 1
  SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
    $RENDER_EXEC python3 -m benchmarks.reg_overlay "$RUN" --patient "$PATIENT" \
      --field-um "$FIELD_UM" --palette magenta-cyan --numbers "$NUMBERS" --title "$LABEL" \
      -o "$ROOT/overlay" "${EXTRA[@]+"${EXTRA[@]}"}"
) || { echo "[overlay] FAILED" >&2; exit 1; }

echo "=================================================="
echo "Done $(date). Images in $ROOT/overlay/: ${PATIENT}_<round>_before/after/locator.{png,pdf}"
echo "=================================================="
