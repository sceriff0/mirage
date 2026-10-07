#!/bin/bash
#SBATCH --job-name=mirage_degraded
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/degraded_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/degraded_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal
# ------------------------------------------------------------------------------------
# VALIS, STARE and ASHLAR-as-published on EQUALLY DEGRADED inputs, in one command.
#
# ASHLAR needs raw tiles, and the slides exist only stitched, so its tiles are synthetic:
# a stage error and sensor noise per tile (submit_ashlar.sh MODE=original). Comparing that
# against VALIS and STARE run on the clean slides would hand them better pixels. This job
# gives the whole slides the same sensor noise and runs all three on it, in a results root
# OF ITS OWN -- nothing under the main benchmark is read-write here:
#
#   1. noisy copies of the shared preprocessed slides        benchmarks/ashlar/degrade.py
#        -> <DEGRADED>/arm_results/preprocess_shared/
#   2. VALIS and STARE, the two arms in ARMS, from that checkpoint   submit_arms.sh
#        (the same launcher, plan, params and containers as the main benchmark)
#   3. ASHLAR as published, tiles cut from the CLEAN slides with the same noise per tile
#        and the stage error, scored on step 2's nuclei     submit_ashlar.sh MODE=original
#        (submitted as its own job: it needs far more memory than this head)
#
#   sbatch ~/pipelines/mirage/benchmarks/submit_degraded.sh
#   sbatch --export=ALL,NOISE_FRAC=0.02,JITTER_UM=3 ~/pipelines/mirage/benchmarks/submit_degraded.sh
#
# Every knob goes in --export (never `VAR=x sbatch`). Resubmitting continues: slides
# already degraded are kept, finished arms are skipped (ARMS_RESUME=1 is set), a finished
# ASHLAR arm is not redone.
# ------------------------------------------------------------------------------------

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
BENCH_DIR="${BENCH_DIR:-/beegfs/scratch/ieo7660/ihc_method/benchmark}"   # the MAIN benchmark
CLEAN_RESULTS="${CLEAN_RESULTS:-$BENCH_DIR/arm_results}"
DEGRADED="${DEGRADED:-$BENCH_DIR/degraded}"            # this experiment's own directory
ARMS="${ARMS:-valis_high_micro2 tiled_high_s64}"       # space-separated; VALIS first
NOISE_FRAC="${NOISE_FRAC:-0.01}"   # sensor noise s.d., fraction of each channel's range
JITTER_UM="${JITTER_UM:-2}"        # ASHLAR's tiles only: stage error per tile, um
SEED="${SEED:-0}"
SHIFT="${SHIFT:-15}"               # ASHLAR's --maximum-shift, um
TILE="${TILE:-1024}"
RUN_ASHLAR="${RUN_ASHLAR:-1}"      # 0 = VALIS and STARE only
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"
IMAGES="${IMAGES:-${NXF_SINGULARITY_CACHEDIR:-${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/docker_images}}}"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
# -------------------------------------------------------------------------------

RESULTS="$DEGRADED/arm_results"
CLEAN_CSV="$CLEAN_RESULTS/preprocess_shared/csv/preprocessed.csv"
[[ -s "$CLEAN_CSV" ]] || { echo "no $CLEAN_CSV: the main benchmark's preprocessing has not finished" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/ashlar/degrade.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/ashlar/degrade.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }
case "$DEGRADED" in "$CLEAN_RESULTS"|"$CLEAN_RESULTS"/*|"$BENCH_DIR")
  echo "DEGRADED=$DEGRADED is (inside) the main benchmark's results: it must be its own directory" >&2; exit 1 ;;
esac
mkdir -p "$RESULTS" "$DEGRADED/logs"

echo "=================================================="
echo "Degraded-input run, job ${SLURM_JOB_ID:-local} -- $(date)"
echo "Clean:     $CLEAN_RESULTS"
echo "Degraded:  $DEGRADED   (noise $NOISE_FRAC of the range, seed $SEED)"
echo "Arms:      $ARMS   ASHLAR as published: $RUN_ASHLAR (stage error <= $JITTER_UM um, shift $SHIFT um)"
echo "=================================================="

# ---- 1. the noisy whole slides -----------------------------------------------------
# In the STARE image, as the tile cutter: it reads the slides' compression and carries the
# pipeline's own OME-TIFF writer. Pulled once into the cache if missing.
img="$IMAGES/bolt3x-mirage-stare-1.0.0.img"
if [[ ! -s "$img" ]]; then
  echo "[images] pulling docker://bolt3x/mirage-stare:1.0.0 -> $img" >&2
  singularity pull "$img.partial.$$" docker://bolt3x/mirage-stare:1.0.0 >&2 \
    && mv -f "$img.partial.$$" "$img" || { rm -f "$img.partial.$$"; echo "could not pull the STARE image" >&2; exit 1; }
fi
DEGRADE_EXEC="${DEGRADE_EXEC:-singularity exec $SING_BINDS $img}"
# shellcheck disable=SC2086
(
  cd "$SRC_DIR" || exit 1
  SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
    $DEGRADE_EXEC python3 -m benchmarks.ashlar.degrade --csv "$CLEAN_CSV" \
      --out-root "$RESULTS/preprocess_shared" --noise-frac "$NOISE_FRAC" --seed "$SEED"
) || { echo "[degrade] FAILED" >&2; exit 1; }
# ---- 2. VALIS and STARE on them ----------------------------------------------------
# EXACT, not ONLY: these rows alone. ONLY would add everything scored on VALIS's nuclei,
# i.e. the whole benchmark; and with no preprocess row in the plan the launcher takes the
# checkpoint written above as it is.
exact="^($(echo $ARMS | tr ' ' '|'))\$"
BENCH_DIR="$DEGRADED" RESULTS="$RESULTS" EXACT="$exact" ARMS_RESUME=1 ENABLE_CSE=false \
  ARMS_CONCURRENCY="${ARMS_CONCURRENCY:-2}" \
  bash "$SRC_DIR/benchmarks/submit_arms.sh" || { echo "[arms] FAILED" >&2; exit 1; }
first="${ARMS%% *}"
compgen -G "$RESULTS/$first/*/qc/registration/*_seg_qc.json" >/dev/null \
  || { echo "[arms] $first wrote no *_seg_qc.json under $RESULTS/$first: see $DEGRADED/logs and $RESULTS/$first/nextflow.stderr.log" >&2; exit 1; }

# ---- 3. ASHLAR as published --------------------------------------------------------
if [[ "$RUN_ASHLAR" == "1" ]]; then
  sbatch --export=ALL,MODE=original,RESULTS="$RESULTS",PREPROC_CSV="$CLEAN_CSV",FROM_ARM="$first",SHIFT="$SHIFT",TILE="$TILE",JITTER_UM="$JITTER_UM",NOISE_FRAC="$NOISE_FRAC",SEED="$SEED",IMAGES="$IMAGES" \
    "$SRC_DIR/benchmarks/submit_ashlar.sh" \
    || { echo "[ashlar] could not submit; run it by hand with the --export above" >&2; exit 1; }
fi
echo "=================================================="
echo "VALIS and STARE finished: $(date)"
echo "Results:  $RESULTS   plan: $DEGRADED/arm_plan.csv"
[[ "$RUN_ASHLAR" == "1" ]] && echo "ASHLAR:   submitted as its own job -> $RESULTS/ashlar_orig_t${TILE}_s${SHIFT}"
echo "=================================================="
