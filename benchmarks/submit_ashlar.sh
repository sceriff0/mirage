#!/bin/bash
#SBATCH --job-name=mirage_ashlar
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/ashlar_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/ashlar_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=normal
# ------------------------------------------------------------------------------------
# ONE ASHLAR arm, on its own: the external baseline without the arms launcher.
#
# run_arms.sh runs the ASHLAR arm as one pass of the whole benchmark. This submits just
# that arm (benchmarks/run_ashlar_arm.sh) with the same containers and the same finished
# marker, for when only ASHLAR is missing or another maximum shift is wanted.
#
#   sbatch ~/pipelines/mirage/benchmarks/submit_ashlar.sh                    # 15 um, the default
#   sbatch --export=ALL,SHIFT=240 ~/pipelines/mirage/benchmarks/submit_ashlar.sh
#   sbatch --export=ALL,MODE=original ~/pipelines/mirage/benchmarks/submit_ashlar.sh
#   bash ~/pipelines/mirage/benchmarks/submit_ashlar.sh                      # interactive
#
# MODE=original runs ASHLAR AS PUBLISHED (run_ashlar_original_arm.sh): the original
# `ashlar` command on synthetic raw tiles of every cycle, its own stitching of the
# reference and its own mosaic as the registered image. MODE=layer (the default, what
# run_arms.sh runs) uses ASHLAR's cross-cycle class only, on a perfect grid, and the
# pipeline's stitcher. The two write differently named arms and can coexist.
#
# Every knob goes in --export (never `VAR=x sbatch`). The arm is named
# ashlar_t<TILE>_s<SHIFT> (as build_arm_plan.py names it), ashlar_orig_t<TILE>_s<SHIFT>
# under MODE=original, unless ARM is given.
# It needs, already finished under RESULTS: the shared preprocessing (PREPROCESS_ARM) and,
# for the Dice, the arm whose QC nuclei are reused (FROM_ARM).
# ------------------------------------------------------------------------------------

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
RESULTS="${RESULTS:-/beegfs/scratch/ieo7660/ihc_method/benchmark/arm_results}"
TILE="${TILE:-1024}"                         # px; the grid the stitched slide is cut into
OVERLAP="${OVERLAP:-0.1}"                    # fraction of a tile
SHIFT="${SHIFT:-15}"                         # ASHLAR's --maximum-shift, um (15 = its default)
MODE="${MODE:-layer}"                        # layer | original (see above)
case "$MODE" in
  layer)    arm_script=run_ashlar_arm.sh;          arm_prefix=ashlar ;;
  original) arm_script=run_ashlar_original_arm.sh; arm_prefix=ashlar_orig ;;
  *) echo "MODE=$MODE is not layer or original" >&2; exit 1 ;;
esac
ARM="${ARM:-${arm_prefix}_t${TILE}_s${SHIFT}}"
# MODE=original only: the synthetic raw tiles (run_ashlar_original_arm.sh documents them)
JITTER_UM="${JITTER_UM:-2}"                  # stage error per tile, um
NOISE_FRAC="${NOISE_FRAC:-0.01}"             # per-tile sensor noise, fraction of the range
SEED="${SEED:-0}"
FROM_ARM="${FROM_ARM:-valis_high_micro2}"    # whose QC nuclei score it
PREPROCESS_ARM="${PREPROCESS_ARM:-preprocess_shared}"
MAX_DISCARD="${MAX_DISCARD:-1}"              # as arms.yaml's max_discard_fraction
REG_QC="${REG_QC:-0}"                        # 1 = also write the Before/After QC preview
                                             # (heavy; the figures do not read it)
SEG_QC="${SEG_QC:-1}"                        # 0 = no Dice (then FROM_ARM is not needed)
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}" # checkout on `benchmarking`
IMAGES="${IMAGES:-${NXF_SINGULARITY_CACHEDIR:-${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/docker_images}}}"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
# -------------------------------------------------------------------------------

# PREPROC_CSV: the slides the tiles are cut from, when they are not this root's own
# preprocessing (submit_degraded.sh cuts them from the CLEAN slides: the tiles get their
# noise per tile, and cutting them from the noisy copies would add it twice).
PREPROC="${PREPROC_CSV:-$RESULTS/$PREPROCESS_ARM/csv/preprocessed.csv}"
[[ -s "$PREPROC" ]] || { echo "no $PREPROC: $PREPROCESS_ARM has not finished" >&2; exit 1; }
[[ "$SEG_QC" != "1" || -d "$RESULTS/$FROM_ARM" ]] \
  || { echo "no $RESULTS/$FROM_ARM: its QC nuclei score ASHLAR (SEG_QC=0 skips the Dice)" >&2; exit 1; }
[[ -x "$SRC_DIR/benchmarks/$arm_script" ]] \
  || { echo "$SRC_DIR has no benchmarks/$arm_script: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }
if [[ -f "$RESULTS/$ARM/.external_done" ]]; then
  echo "$ARM already finished ($(cat "$RESULTS/$ARM/.external_done")); remove $RESULTS/$ARM/.external_done to redo it"
  exit 0
fi

# Each step runs `singularity exec <image>`. An image missing from the cache is pulled ONCE
# into it, under the name Nextflow itself uses, written to a temporary name and moved into
# place: `singularity exec docker://...` would re-download it on every one of the steps.
image() {                         # image <registry/name:tag> -> prints the local path
  local ref="$1" f tmp
  f="$IMAGES/$(printf '%s' "$ref" | tr '/:' '--').img"
  if [[ ! -s "$f" ]]; then
    tmp="$f.partial.$$"
    echo "[images] pulling docker://$ref -> $f" >&2
    if singularity pull "$tmp" "docker://$ref" >&2 && mv -f "$tmp" "$f"; then :; else
      rm -f "$tmp"
      echo "[images] could not pull $ref" >&2
      return 1
    fi
  fi
  printf '%s' "$f"
}
a=$(image labsyspharm/ashlar:1.20.0) || exit 1
q=$(image bolt3x/mirage-stare:1.0.0) || exit 1
g=$(image bolt3x/mirage-regqc:1.0.0) || exit 1
export ASHLAR_EXEC="${ASHLAR_EXEC:-singularity exec $SING_BINDS $a}"
export QC_EXEC="${QC_EXEC:-singularity exec $SING_BINDS $q}"
export REGQC_EXEC="${REGQC_EXEC:-singularity exec $SING_BINDS $g}"
export ASHLAR_MAX_DISCARD="$MAX_DISCARD" ASHLAR_REG_QC="$REG_QC" ASHLAR_SEG_QC="$SEG_QC"
export ASHLAR_STAGE_JITTER_UM="$JITTER_UM" ASHLAR_NOISE_FRAC="$NOISE_FRAC" ASHLAR_SEED="$SEED"

echo "=================================================="
echo "ASHLAR arm $ARM ($MODE), job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)} -- $(date)"
[[ "$MODE" == "original" ]] \
  && echo "Tiles:    synthetic, stage error <= $JITTER_UM um, noise $NOISE_FRAC of the range, seed $SEED"
echo "Results:  $RESULTS   tile $TILE px, overlap $OVERLAP, maximum shift $SHIFT um"
echo "Inputs:   $PREPROC"
echo "Nuclei:   $FROM_ARM (Dice: $SEG_QC)   QC preview: $REG_QC"
echo "Solve:    $ASHLAR_EXEC"
echo "Others:   $QC_EXEC"
echo "=================================================="
mkdir -p "$RESULTS/$ARM"
if "$SRC_DIR/benchmarks/$arm_script" \
     "$RESULTS" "$ARM" "$FROM_ARM" "$PREPROC" "$TILE" "$OVERLAP" "$SHIFT"; then
  date '+%Y-%m-%dT%H:%M:%S' > "$RESULTS/$ARM/.external_done"
  echo "DONE: $RESULTS/$ARM"
else
  echo "FAILED: $ARM (see above); no finished marker written" >&2
  exit 1
fi
