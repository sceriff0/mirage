#!/usr/bin/env bash
#SBATCH --job-name=mirage_supp
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/supp_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/supp_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G             # full-resolution slide crops; no Nextflow runs here
#SBATCH --partition=normal
#
# ============================================================================
# EVERY SUPPLEMENTARY FIGURE, ONE JOB -- S2..S11 + the method mosaic
# ============================================================================
# Draws from what the arms ALREADY computed (benchmarks/submit_arms.sh: one results root
# holding VALIS, STARE v1, DRAPE, ASHLAR and the segmentation arms). Registers nothing,
# segments nothing, launches no pipeline: seconds-to-minutes per figure, so resubmit
# freely after changing benchmarks/configs/supplementary.yaml.
#
#   mirage half  benchmarks/supplementary.py (renders in bolt3x/mirage-quantify)
#     mosaic   Before | VALIS | STARE-or-DRAPE | ASHLAR, Dice per cell     [priority]
#     S2       secondary-only controls at one pinned contrast   (S2.csv in the config)
#     S3a      per-round DAPI retention             (quantification/*_round_qc.csv)
#     S4, S7   Before | VALIS | STARE/DRAPE (| ASHLAR) on one crop + matched inset
#     S5       registration cost by tier            (the Nextflow traces)
#     S6       nuclei | cell masks per backend + pairwise Dice
#     S8       Dice and displacement by case and by panel pair
#   ihc half     benchmarks/ihc/supplementary.R, run IN the ihc_method checkout (IHC=)
#     S3b      CD3+ among CD8+ per case    S9  every case by FlowPath phenotype
#     S10      CD45+ per case + cold/intermediate/hot    S11  every deconvolution method
#
# EVERY COMPARISON IN EVERY COMBINATION -- you choose by looking at OUT/index.html:
#   set    stare | drape | all      config  high | best (picks.csv)    variant  v1..vN
# and the SAME tissue in every set and config (one anchor render picks the ROIs).
#
# Submit (login node; every knob in --export, never `VAR=x sbatch`):
#   B=/beegfs/scratch/ieo7660/ihc_method/benchmark
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/supplementary && cd $_
#   sbatch --export=ALL,RESULTS=$B/arm_results,PLAN=$B/arm_plan.csv,IHC=$HOME/workflowR/ihc_method \
#     ~/pipelines/mirage/benchmarks/submit_supplementary.sh
# Only some:   ONLY=mosaic+S4       (+ separated: --export splits on commas)
# Check first: CHECK=1              (per figure READY/PARTIAL/MISSING + the AUTHORS TO
#                                   SUPPLY config items -> OUT/check.csv; draws nothing.
#                                   Also written at the start of every drawing run.)
# See first:   DRY_RUN=1            (prints every render; draws nothing)
# Tier:        a legend naming no registration tier means HIGH: the registration figures
#              are drawn at each method's high arm only (configs: [high] in the config)
# Own config:  CONFIG=supplementary.yaml   (copy benchmarks/configs/supplementary.yaml)
# Python:      the ACTIVE env is used when it imports pandas/yaml/matplotlib; else
#              CONDA_ENV (default nf-env) is activated. CONDA_ENV=<env> forces one.
# R:           RSCRIPT=/path/to/Rscript when Rscript is not on PATH (e.g. a vizu node)
# RNA (S11):   IHC_KNIT_MOLECULAR=1 knits analysis/molecular_massimo2.Rmd first, which
#              writes output/paired_deconv.rds (slow: runs immunedeconv)
# ============================================================================
# No `set -u`: ~/.bashrc and `conda activate` read variables a batch job leaves unset.

# $PWD, not $SLURM_SUBMIT_DIR: a batch job already starts in its submit dir, and inside
# an interactive allocation (srun --pty) SLURM_SUBMIT_DIR is where THAT was started.
OUT="${OUT:-$PWD}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"          # benchmarking_new_method checkout
RESULTS="${RESULTS:-/beegfs/scratch/ieo7660/ihc_method/benchmark/arm_results}"
PLAN="${PLAN:-$(dirname "$RESULTS")/arm_plan.csv}"
CONFIG="${CONFIG:-$SRC_DIR/benchmarks/configs/supplementary.yaml}"
IHC="${IHC:-}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
CHECK="${CHECK:-0}"
CONDA_ENV="${CONDA_ENV:-}"        # empty: keep the active env if it has the packages
RSCRIPT="${RSCRIPT:-Rscript}"      # the R whose library has ihc_method's packages

[[ "$CONFIG" = /* ]] || CONFIG="$OUT/$CONFIG"
for f in "$PLAN" "$CONFIG"; do
  [[ -s "$f" ]] || { echo "not found or empty: $f" >&2; exit 1; }
done
[[ -d "$RESULTS" ]] || { echo "no results root at $RESULTS" >&2; exit 1; }
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
head -n1 "$PLAN" | tr ',' '\n' | grep -qx method || {
  echo "$PLAN has no \`method\` column: rebuild it with this checkout's submit_arms.sh" >&2
  echo "(STARE and DRAPE are both registration_method=tiled; only \`method\` tells them apart)" >&2
  exit 1
}

# The orchestrator runs on the host (the renderers run in the container below), so this
# python3 needs pandas + yaml + matplotlib. An already-active env that has them is kept
# -- `bash submit_supplementary.sh` from a `conda activate`d shell -- and only otherwise
# is CONDA_ENV (default nf-env) activated.
py_ok() { python3 -c 'import pandas, yaml, matplotlib' >/dev/null 2>&1; }
if [[ -n "$CONDA_ENV" ]] || ! py_ok; then
  source ~/.bashrc
  eval "$(conda shell.bash hook 2>/dev/null)"      # `conda` in a non-interactive shell
  conda activate "${CONDA_ENV:-nf-env}"
fi
py_ok || {
  echo "$(command -v python3 || echo 'no python3') cannot import pandas/yaml/matplotlib." >&2
  echo "Activate an env that has them first, or pass CONDA_ENV=<env>." >&2
  exit 1
}

# ---- the render container (same image and pull discipline as submit_figures.sh) ----
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
export NXF_SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR:-$SINGULARITY_CACHEDIR}"
export APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-true}"
export SINGULARITY_DISABLE_CACHE="${SINGULARITY_DISABLE_CACHE:-$APPTAINER_DISABLE_CACHE}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$NXF_SINGULARITY_CACHEDIR/.pull_tmp}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_TMPDIR"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
ensure_sif() {                     # one pull per image; `singularity exec docker://` re-pulls
  local ref="$1" f tmp
  f="$NXF_SINGULARITY_CACHEDIR/$(printf '%s' "$ref" | tr '/:' '--').img"
  if [[ ! -s "$f" ]]; then
    tmp="$f.partial.$$"
    echo "[images] pulling docker://$ref -> $f" >&2
    if singularity pull "$tmp" "docker://$ref" >&2 && mv -f "$tmp" "$f"; then :; else
      rm -f "$tmp"; printf 'docker://%s' "$ref"; return
    fi
  fi
  printf '%s' "$f"
}
if [[ -z "${RENDER_EXEC+x}" ]]; then
  if command -v singularity >/dev/null; then
    RENDER_EXEC="singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-quantify:1.0.0)"
  else
    RENDER_EXEC=""                 # a workstation: the renderers run in this env
  fi
fi

echo "=================================================="
echo "Supplementary set, job ${SLURM_JOB_ID:-local} -- $(date)"
echo "Results: $RESULTS"
echo "Plan:    $PLAN"
echo "Config:  $CONFIG"
echo "Out:     $OUT"
echo "IHC:     ${IHC:-<none: S3b, S9, S10, S11 skipped>}"
echo "Render:  ${RENDER_EXEC:-<this env>}"
echo "=================================================="

STATUS=()
wants() { [[ -z "$ONLY" ]] || [[ "+$ONLY+" == *"+$1+"* ]]; }

# ---- the ihc half FIRST, so the index the mirage half writes lists it too -----------
IHC_FIGS=()
if [[ -n "$IHC" ]] && ! command -v "$RSCRIPT" >/dev/null; then
  echo "[ihc] $RSCRIPT not found: pass RSCRIPT=/path/to/Rscript (the R you run ihc_method" \
       "with; \`which Rscript\` there), or load its module first" >&2
fi
for f in S3 S9 S10 S11; do wants "$f" && IHC_FIGS+=("$f"); done
if [[ -n "$IHC" && ${#IHC_FIGS[@]} -gt 0 ]]; then
  if [[ ! -f "$IHC/figures/_common.R" ]]; then
    STATUS+=("ihc: FAILED ($IHC is not an ihc_method checkout)")
  elif [[ "$CHECK" == "1" ]]; then
    (cd "$IHC" && IHC_ROOT="$IHC" "$RSCRIPT" "$SRC_DIR/benchmarks/ihc/supplementary.R" --check) \
      && STATUS+=("ihc: CHECKED") || STATUS+=("ihc: CHECK FAILED (see the log above)")
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] (cd $IHC && $RSCRIPT $SRC_DIR/benchmarks/ihc/supplementary.R ${IHC_FIGS[*]})"
  else
    if [[ "${IHC_KNIT_MOLECULAR:-0}" == "1" ]]; then
      (cd "$IHC" && "$RSCRIPT" -e 'workflowr::wflow_build("analysis/molecular_massimo2.Rmd", view = FALSE)') \
        || echo "[ihc] knitting molecular_massimo2 failed; S11 will be skipped" >&2
    fi
    # cd into the checkout: its .Rprofile activates renv, which is where the packages are.
    if (cd "$IHC" && IHC_ROOT="$IHC" "$RSCRIPT" "$SRC_DIR/benchmarks/ihc/supplementary.R" "${IHC_FIGS[@]}"); then
      STATUS+=("ihc: OK (${IHC_FIGS[*]})")
    else
      STATUS+=("ihc: FAILED (see the log above)")
    fi
    # Collected BESIDE the mirage half: OUT/S3 holds S3a (mirage) and S3b (ihc).
    for f in "${IHC_FIGS[@]}"; do
      src="$IHC/output/figures/supplementary/$f"
      [[ -d "$src" ]] && mkdir -p "$OUT/$f" && cp -R "$src/." "$OUT/$f/"
    done
  fi
else
  STATUS+=("ihc: SKIPPED (set IHC=<ihc_method checkout>)")
fi

# ---- the mirage half ---------------------------------------------------------------
MIRAGE_FIGS=()
for f in mosaic S2 S3 S4 S5 S6 S7 S8; do wants "$f" && MIRAGE_FIGS+=("$f"); done
if (( ${#MIRAGE_FIGS[@]} > 0 )); then
  args=(--results "$RESULTS" --plan "$PLAN" --config "$CONFIG" -o "$OUT"
        --only "$(IFS=,; echo "${MIRAGE_FIGS[*]}")" --exec "$RENDER_EXEC")
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  [[ "$CHECK" == "1" ]] && args+=(--check)
  [[ -n "$IHC" ]] && args+=(--ihc "$IHC")
  if (cd "$SRC_DIR" && python3 -m benchmarks.supplementary "${args[@]}"); then
    STATUS+=("mirage: OK (${MIRAGE_FIGS[*]})")
  else
    STATUS+=("mirage: PARTIAL (a figure failed; the others are drawn -- see above)")
  fi
fi

echo "=================================================="
printf '  %s\n' "${STATUS[@]}"
echo "  inputs per figure: $OUT/check.csv"
echo "  choose here: $OUT/index.html   (arm picks: $OUT/picks.csv)"
echo "  legend values: $OUT/S*/**/*_values*.csv"
echo "=================================================="
