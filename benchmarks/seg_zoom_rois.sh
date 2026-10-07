#!/bin/bash
# seg_zoom_rois.sh -- the segmentation zoom figure at N regions, for every backend.
#
# No SLURM header on purpose: run it in an interactive job (or on any node that has
# singularity), not with sbatch. It runs no pipeline; it only draws, from segmentation
# runs that are already on disk.
#
#   srun --pty --mem=32G -c 4 -t 2:00:00 bash          # an interactive shell, then:
#   RESULTS=/beegfs/.../benchmark/arm_results PATIENT=046 N=8 \
#     ~/pipelines/mirage/benchmarks/seg_zoom_rois.sh
#
# The N regions are picked ONCE, on the first backend's reference DAPI
# (reg_zoom --pick-rois), and every backend is then drawn at those same places, so the
# figures of one region are comparable across backends.
#
# Output (under OUT, default ./seg_zoom_rois):
#   <patient>_rois.json                        the N regions (delete it to pick again)
#   roi<k>/<backend>/<patient>_zoom.{png,pdf}  one figure per region and backend
#   roi<k>/<backend>/<patient>_zoom.json       its manifest
#
# Regions already drawn are skipped, so raising N and re-running adds only the new ones --
# but only if <patient>_rois.json is deleted first, since a larger N is a new pick.

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
RESULTS="${RESULTS:-}"                       # holds one segmentation run per backend
BACKENDS="${BACKENDS:-stardist instantseg cellsam}"   # space-separated; the first picks
ARM_PREFIX="${ARM_PREFIX:-seg_}"             # run dir = RESULTS/<ARM_PREFIX><backend>
PATIENT="${PATIENT:-}"                       # empty = the run's only patient
N="${N:-5}"                                  # how many regions
OUT="${OUT:-$PWD/seg_zoom_rois}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}" # checkout on `benchmarking`
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"            # µm/px; 'auto' = the file's own calibration
FIELD_UM="${FIELD_UM:-300}"                  # zoom side in µm (drawn at full resolution)
MASK="${MASK:-cell}"                         # cell | nuclei | both
MIN_SEP="${MIN_SEP:-0.15}"                   # min distance between regions, fraction of the
                                             # slide diagonal; lower it if N are not found
FORMATS="${FORMATS:-png,pdf}"
ZOOM_ARGS="${ZOOM_ARGS:-}"                   # extra reg_zoom.py flags, e.g. "--overview-px 3000"
SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
# -------------------------------------------------------------------------------

[[ -n "$RESULTS" && -d "$RESULTS" ]] \
  || { echo "usage: RESULTS=<dir with ${ARM_PREFIX}<backend> runs> [PATIENT=..] [N=5] $0" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/reg_zoom.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/reg_zoom.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }

# The image the other figure launchers render with (matplotlib, scikit-image, tifffile and
# imagecodecs for the LZW slides), under the file name they cache it by.
if [[ -z "${RENDER_EXEC:-}" ]]; then
  img="$SINGULARITY_CACHEDIR/bolt3x-mirage-quantify-1.0.0.img"
  [[ -s "$img" ]] || img="docker://bolt3x/mirage-quantify:1.0.0"
  RENDER_EXEC="singularity exec $SING_BINDS $img"
fi

zoom() {                          # zoom <run dir> <outdir> [reg_zoom flags...]
  local run="$1" out="$2"; shift 2
  local extra=()
  [[ -n "$PATIENT" ]] && extra+=(--patient "$PATIENT")
  [[ "$PIXEL_SIZE" != auto ]] && extra+=(--pixel-size-um "$PIXEL_SIZE")
  # shellcheck disable=SC2086
  (
    cd "$SRC_DIR" || exit 1
    SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
      $RENDER_EXEC python3 -m benchmarks.reg_zoom "$run" -o "$out" \
        --field-um "$FIELD_UM" "${extra[@]}" "$@" $ZOOM_ARGS
  )
}

read -r -a backends <<< "$BACKENDS"
for b in "${backends[@]}"; do
  [[ -s "$RESULTS/$ARM_PREFIX$b/csv/segmented.csv" ]] \
    || { echo "no $RESULTS/$ARM_PREFIX$b/csv/segmented.csv (BACKENDS / ARM_PREFIX?)" >&2; exit 1; }
done
mkdir -p "$OUT"

# ---- 1. the regions, once ------------------------------------------------------
shopt -s nullglob
rois=("$OUT"/*_rois.json)
if [[ ${#rois[@]} -eq 0 ]]; then
  zoom "$RESULTS/$ARM_PREFIX${backends[0]}" "$OUT" --pick-rois "$N" --min-sep "$MIN_SEP" \
    || { echo "[rois] FAILED" >&2; exit 1; }
  rois=("$OUT"/*_rois.json)
fi
[[ ${#rois[@]} -eq 1 ]] \
  || { echo "expected one *_rois.json in $OUT, found ${#rois[@]} (set PATIENT)" >&2; exit 1; }
# "Y,X" per line; plain python3, the file is small JSON
picks=()
while IFS= read -r line; do picks+=("$line"); done < <(python3 -c '
import json, sys
for r in json.load(open(sys.argv[1]))["rois"]:
    print("%d,%d" % (r["y"], r["x"]))' "${rois[0]}")
echo "[rois] ${#picks[@]} region(s) from ${rois[0]}"
[[ ${#picks[@]} -lt $N ]] \
  && echo "[rois] fewer than N=$N: lower MIN_SEP, or delete ${rois[0]} to pick again" >&2

# ---- 2. every region x backend -------------------------------------------------
failed=0
k=0
for roi in "${picks[@]}"; do
  k=$((k + 1))
  for b in "${backends[@]}"; do
    dest="$OUT/roi$k/$b"
    if compgen -G "$dest/*_zoom.json" >/dev/null; then
      echo "[roi$k/$b] already drawn, skipped"
      continue
    fi
    echo "[roi$k/$b] $roi"
    zoom "$RESULTS/$ARM_PREFIX$b" "$dest" --roi "$roi" --mask "$MASK" --title "$b" \
      --formats "$FORMATS" || { echo "[roi$k/$b] FAILED" >&2; failed=$((failed + 1)); }
  done
done
echo "done: $OUT  ($k region(s) x ${#backends[@]} backend(s), $failed failed)"
[[ $failed -eq 0 ]]
