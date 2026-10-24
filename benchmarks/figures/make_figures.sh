#!/bin/bash
# Every data figure of the paper from the regbench results and one pipeline run, with
# every choice in the block below: rebuilding a figure is running this file again.
#
#   cd <regbench root> && bash <checkout>/benchmarks/figures/make_figures.sh [figure ...]
#
# figures: cost cohort crops overlay segmentation (default: all). Each figure writes
# its own _source_data.csv, which records what was chosen (case, patient, crop, zones) and
# the display settings. benchmarks/figures/README.md says what each one reproduces.
set -euo pipefail

# ── the choices ───────────────────────────────────────────────────────────────
# Each can also be set for one run in the environment (VALIS=valis120_micro bash ...);
# the paper's figures are the ones made with this file as committed.
OUTDIR="${OUTDIR:-$PWD/figures}"
SEED="${SEED-0}"                            # the jitter of every box plot

# one variant per family; the box order is this order
VALIS="${VALIS-valis120_hires}"             # VALIS as shown everywhere
STARE="${STARE-stare}"                      # STARE at the shipped stride
COHORT_METHODS="${COHORT_METHODS-Unregistered=initial,ASHLAR=ashlar,DeeperHistReg=deeperhistreg_hr_gpu_asdrawn,PALOM=palom,VALIS=$VALIS,STARE=$STARE}"
SCORE_SET="${SCORE_SET-outlines}"           # outlines (the headline) or nuclei
COST_DATASET="${COST_DATASET-multiplex}"         # peak RSS / CPU vs input size: the cohort

# Fig. 3a: rows, the patient (empty = the one closest to the median Dice of the rows'
# methods), how many rounds, the crop (empty = the densest window of reference nuclei)
CROP_METHODS="${CROP_METHODS-STARE=$STARE,VALIS=$VALIS}"
CROP_PATIENT="${CROP_PATIENT-}"
CROP_COLUMNS="${CROP_COLUMNS-4}"
CROP_UM="${CROP_UM-150}"
CROP_AT="${CROP_AT-}"                       # X,Y in reference px

# Fig. 4b: the case (empty = the closest to the median Dice of OVERLAY_MEDIAN_OF), whose
# result locates the zones, and which result the "after" panel shows
OVERLAY_CASE="${OVERLAY_CASE-}"
OVERLAY_MEDIAN_OF="${OVERLAY_MEDIAN_OF-$VALIS,$STARE}"
OVERLAY_ZONES_FROM="${OVERLAY_ZONES_FROM-$VALIS}"
OVERLAY_AFTER="${OVERLAY_AFTER-$STARE}"
OVERLAY_ZOOM_UM="${OVERLAY_ZOOM_UM-150}"
OVERLAY_ZONES=()                            # e.g. (--zone rotation=12000,8000)

# Fig. 4c: a pipeline run's slide and its cells
SEG_IMAGE="${SEG_IMAGE-}"                   # <run>/pyramid/<patient>.ome.tif
SEG_GEOJSON="${SEG_GEOJSON-}"               # <run>/geojson/<patient>_cells.geojson
SEG_ZOOM_UM="${SEG_ZOOM_UM-150}"
SEG_ZOOM_AT="${SEG_ZOOM_AT-}"               # X,Y in image px

# the zooms' display, the same in every figure (state it in the legends)
# shellcheck disable=SC2054  # 1,99.8 is one argument: the low,high percentiles
DISPLAY=(--clip 1,99.8 --clahe 0.01 --sharpen 0.6 --upsample 3)

# ── the run ───────────────────────────────────────────────────────────────────
# NOT in the scorer's image: it has no matplotlib or pandas (requirements/stare-regbench.txt).
# FIG_PYTHON is an environment of requirements/ci.txt (segeval.txt: matplotlib, pandas,
# scikit-image; stare.txt: STARE at its pinned release), the one CI tests these with. The
# scores are still the harness's own: the scripts import benchmarks.regbench from this checkout.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck disable=SC1091
[ -f ./regbench.env ] && source ./regbench.env
: "${REGBENCH_CASES:?set REGBENCH_CASES (or run from the regbench root, with its regbench.env)}"
: "${REGBENCH_OUT:?set REGBENCH_OUT (or run from the regbench root, with its regbench.env)}"
FIG_PYTHON="${FIG_PYTHON:-python}"
TABLES="$REGBENCH_OUT/tables"
mkdir -p "$OUTDIR"

fig() {
    (cd "$repo" && PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg \
        "$FIG_PYTHON" -m "benchmarks.figures.$1" "${@:2}")
}

if [ $# -gt 0 ]; then want=("$@"); else want=(cost cohort crops overlay segmentation); fi
for f in "${want[@]}"; do
    echo "== $f"
    case "$f" in
    cost)
        fig fig_synthetic_cost "$TABLES" --valis "$VALIS" --stare "$STARE" \
            --dataset "$COST_DATASET" --seed "$SEED" --outdir "$OUTDIR/cost" ;;
    cohort)
        fig fig_cohort_scores "$TABLES" --methods "$COHORT_METHODS" --set "$SCORE_SET" \
            --seed "$SEED" --outdir "$OUTDIR/cohort" ;;
    crops)
        args=(--cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --tables "$TABLES"
              --methods "$CROP_METHODS" --columns "$CROP_COLUMNS" --crop-um "$CROP_UM"
              "${DISPLAY[@]}" --outdir "$OUTDIR/crops")
        [ -n "$CROP_PATIENT" ] && args+=(--patient "$CROP_PATIENT")
        [ -n "$CROP_AT" ] && args+=(--crop "$CROP_AT")
        fig fig_crop_grid "${args[@]}" ;;
    overlay)
        args=(--cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --tables "$TABLES"
              --median-of "$OVERLAY_MEDIAN_OF" --zones-from "$OVERLAY_ZONES_FROM"
              --zoom-um "$OVERLAY_ZOOM_UM" "${DISPLAY[@]}" --outdir "$OUTDIR/overlay"
              ${OVERLAY_ZONES[@]+"${OVERLAY_ZONES[@]}"})
        [ -n "$OVERLAY_CASE" ] && args+=(--case-id "$OVERLAY_CASE")
        [ -n "$OVERLAY_AFTER" ] && args+=(--after "$OVERLAY_AFTER")
        fig fig_overlay "${args[@]}" ;;
    segmentation)
        if [ -z "$SEG_IMAGE" ] || [ -z "$SEG_GEOJSON" ]; then
            echo "segmentation: set SEG_IMAGE and SEG_GEOJSON in this file"; continue
        fi
        args=(--image "$SEG_IMAGE" --geojson "$SEG_GEOJSON" --zoom-um "$SEG_ZOOM_UM"
              "${DISPLAY[@]}" --outdir "$OUTDIR/segmentation")
        [ -n "$SEG_ZOOM_AT" ] && args+=(--zoom "$SEG_ZOOM_AT")
        fig fig_segmentation "${args[@]}" ;;
    *) echo "unknown figure $f (cost cohort crops overlay segmentation)"; exit 2 ;;
    esac
done
echo "-> $OUTDIR"
