#!/usr/bin/env bash
#SBATCH --job-name=mirage_mosaics
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/mosaics_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/mosaics_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=normal
#
# Draw ONLY the mosaics of a figures run that already exists. Everything else the run
# produced is left alone; nothing is registered, segmented or re-rendered.
#
#   cd /beegfs/scratch/ieo7660/ihc_method/paper_figures/033_all_in_one
#   sbatch ~/pipelines/mirage/benchmarks/draw_mosaics.sh
#
# Why it exists next to submit_figures.sh: that one draws EVERY figure of a plan, and the
# figures carry no .done marker, so asking it for one more mosaic size redraws all of them.
# This redraws only the mosaics, off runs that already exist.
#
# Knobs (--export=ALL,NAME=value):
#   PATCHES  "200 500 1000 2000"  patch sizes in µm, one mosaic each
#   KINDS    "overlay checker"    magenta/cyan overlay and/or the checkerboard
#   NUMBERS  image                auto | scorer | image | none
#   VARIANTS 4                    ROI sets per mosaic
#   ROI      ""                   "Y,X" TOP-LEFT, passed through; empty = auto per variant
#   ROI_CENTER ""                 "Y,X" CENTRE: every patch size is drawn concentric on it,
#                                 which is what makes a size series a zoom series
#   MAX_CELL_PX 1600              cap on the drawn cell; a crop larger than this is
#                                 downsampled instead of making a gigapixel canvas
#   ROWS     ""                   (round, ROI) cells; empty = every round at one ROI

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"
PATCHES="${PATCHES:-200 500 1000 2000}"
KINDS="${KINDS:-overlay checker}"
NUMBERS="${NUMBERS:-image}"
VARIANTS="${VARIANTS:-4}"
ROI="${ROI:-}"
ROI_CENTER="${ROI_CENTER:-}"
MAX_CELL_PX="${MAX_CELL_PX:-1600}"
ROWS="${ROWS:-}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"  # 1 (default) = an arm with no registration is drawn
                                     # as a flat "no data" column; 0 aborts the figure
                                     # instead, which is what a published one wants
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"
FORMATS="${FORMATS:-png,pdf}"
DPI="${DPI:-100}"
IMG="${IMG:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images/bolt3x-mirage-quantify-1.0.0.img}"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
# The renders need matplotlib + scikit-image + tifffile AND imagecodecs (the slides are LZW,
# which the segeval image cannot decode -- job 6844142). The quantify image carries all four.
RENDER_EXEC="${RENDER_EXEC:-singularity exec $SING_BINDS $IMG}"

# ROOT MUST BE ABSOLUTE: steps below run inside `(cd "$SRC_DIR" && ...)` while writing to
# "$ROOT/...", and a relative ROOT re-resolves against the checkout there. Measured as
# figures job 7052347, where ROOT=. sent every params.json into $SRC_DIR and failed all
# three segmentations with FileNotFoundError.
ROOT=$(cd "$ROOT" && pwd) || { echo "cannot resolve ROOT=$ROOT" >&2; exit 1; }
cd "$ROOT" || exit 1
[[ -x "$SRC_DIR/benchmarks/reg_mosaic.py" || -f "$SRC_DIR/benchmarks/reg_mosaic.py" ]] \
  || { echo "no reg_mosaic.py under $SRC_DIR" >&2; exit 1; }

# The arms this run built, in the order the mosaic should column them. Only those with
# registered slides are drawn -- an arm that failed is left out rather than failing the run.
# DIRECTORIES AND LABELS ARE KEPT APART: ARM_DIR is nargs="+", so argparse stops collecting
# positionals at the first flag and calls every later directory "unrecognized". The
# directories go first, the --label flags after them.
DIRS=(); LABELS=(); REAL=0
add_arm() {                        # add_arm <directory name> <column title>
  # at ALLOW_MISSING=1 an arm with no checkpoint is still passed: reg_mosaic draws it as a
  # labelled empty column, which is the point -- filtering it here would hide the gap
  if [[ ! -s "$ROOT/$1/csv/registered.csv" ]]; then
    [[ "$ALLOW_MISSING" == "1" ]] || return 0
    mkdir -p "$ROOT/$1"
  else
    REAL=$(( REAL + 1 ))
  fi
  DIRS+=("$ROOT/$1"); LABELS+=(--label "$1=$2")
}
add_arm valis_high_micro2 "VALIS high"
add_arm stare_high        "STARE high"
add_arm ashlar_t1024_s15  "ASHLAR"
# COUNT THE ARMS WITH DATA, not the columns. At ALLOW_MISSING=1 every arm is a column
# whether or not it registered anything, so a `${#DIRS[@]}` test can never fail and an
# empty ROOT would run all the way to a mosaic of nothing but grey boxes -- or, as
# measured, to a confusing "could not count the moving rounds" three lines further down.
(( REAL > 0 )) || { echo "no arm under $ROOT has csv/registered.csv" >&2; exit 1; }

# --rows = one row per moving round, counted off the reference arm's checkpoint. Only needed
# on a checkout older than the fix that made --rows default to exactly this.
if [[ -z "$ROWS" ]]; then
  ref="$ROOT/valis_high_micro2/csv/registered.csv"
  [[ -s "$ref" ]] || ref="$ROOT/stare_high/csv/registered.csv"
  ROWS=$(tail -n +2 "$ref" | awk -F, '$4 != "true"' | wc -l | tr -d ' ')
fi
(( ROWS >= 1 )) || { echo "could not count the moving rounds; pass ROWS=N" >&2; exit 1; }

echo "=================================================="
echo "Mosaics job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Root:     $ROOT"
echo "Arms:     ${DIRS[*]}"
echo "Rows:     $ROWS   patches: $PATCHES   kinds: $KINDS   numbers: $NUMBERS   variants: $VARIANTS"
echo "Cell cap: $MAX_CELL_PX px   centre: ${ROI_CENTER:-none}   roi: ${ROI:-auto}"
echo "Checkout: $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null)"
echo "=================================================="

n=0; failed=0
for patch in $PATCHES; do
  for kind in $KINDS; do
    out="$ROOT/mosaic/p${patch}_${kind}_${NUMBERS}"
    mkdir -p "$out"
    args=(--rows "$ROWS" --patch-um "$patch" --kinds "$kind" --numbers "$NUMBERS"
          --variants "$VARIANTS" --formats "$FORMATS" --dpi "$DPI")
    [[ "$PIXEL_SIZE" != auto ]] && args+=(--pixel-size-um "$PIXEL_SIZE")
    [[ "$ALLOW_MISSING" == "1" ]] || args+=(--no-allow-missing-arms)

    # The figure scales with the patch: at 0.325 um/px a 2000 um cell is 6154 px, so a
    # 10-round x 4-arm grid would be 1.5 gigapixels. --cell-in caps what is DRAWN (inches x
    # dpi), downsampling the crop instead -- which is what zooming out should look like.
    patch_px=$(awk -v u="$patch" -v s="${PIXEL_SIZE/auto/0.325}" 'BEGIN{printf "%d", u/s + 0.5}')
    if (( patch_px > MAX_CELL_PX )); then
      args+=(--cell-in "$(awk -v n="$MAX_CELL_PX" -v d="$DPI" 'BEGIN{printf "%.4f", n/d}')")
    fi

    # A size series is only a ZOOM series if the sizes share a centre: --roi is the
    # top-left, so holding it fixed slides the view as the patch grows.
    if [[ -n "$ROI_CENTER" ]]; then
      cy="${ROI_CENTER%%,*}"; cx="${ROI_CENTER##*,}"
      args+=(--roi "$(awk -v cy="$cy" -v cx="$cx" -v p="$patch_px" \
                        'BEGIN{y=cy-p/2; x=cx-p/2; if(y<0)y=0; if(x<0)x=0; printf "%d,%d", y, x}')")
    elif [[ -n "$ROI" ]]; then
      args+=(--roi "$ROI")
    fi
    echo "[mosaic] patch ${patch}um kind=$kind -> $out"
    # shellcheck disable=SC2086
    if (
      cd "$SRC_DIR" || exit 1
      SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
        $RENDER_EXEC python3 -m benchmarks.reg_mosaic \
          "${DIRS[@]}" -o "$out" "${args[@]}" "${LABELS[@]}"
    ); then
      n=$((n + 1))
    else
      echo "[mosaic] FAILED: patch ${patch}um kind=$kind" >&2
      failed=$((failed + 1))
    fi
  done
done

echo "=================================================="
echo "Done $(date). $n mosaic(s) under $ROOT/mosaic/ ($failed failed)"
echo "=================================================="
(( failed == 0 ))
