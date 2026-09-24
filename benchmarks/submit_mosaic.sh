#!/usr/bin/env bash
#SBATCH --job-name=mirage_mosaic
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/mosaic_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/mosaic_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G             # 2 Nextflow heads (-Xmx4g each) + the ASHLAR steps and the mosaic,
                              # which run INSIDE this job, not as SLURM children
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE registration MOSAIC — VALIS high vs STARE high vs ASHLAR, SLURM head job
# ============================================================================
# A FAST PROTOTYPE of the registration figure, outside the arm benchmark. From one
# samplesheet (the pipeline's usual columns, any number of slides per patient):
#
#   1. preprocess_shared   pipeline, --stop preprocessing              (once)
#   2. valis_high_micro2   pipeline, --start/--stop registration        } in parallel,
#      stare_high          pipeline, --start/--stop registration        } all resume from 1
#      ashlar_t1024_s30    benchmarks/run_ashlar_arm.sh                 }
#   3. mosaic/             benchmarks/reg_mosaic.py over every arm that finished:
#                          <patient>_mosaic.{png,pdf}, <patient>_locator.*, <patient>_rois.json
#
# EVERYTHING IS WRITTEN INTO THE DIRECTORY YOU SUBMIT FROM (sbatch's SLURM_SUBMIT_DIR), like
# the arm and sweep submitters -- one directory per figure. Keep it off $HOME.
#
# THE PROTOTYPE HAS NO NUMBERS. SEG_QC=0 (default) runs registration QC at reg_qc=1: the
# Before/After composite the figure is drawn from, without WARP_SEG_QC's segmentation, and the
# cells carry NO Dice / Δ. SEG_QC=1 runs reg_qc=2 -- the pipeline's matched-nucleus scorer --
# and the cells print its Dice and Δ (ASHLAR is then scored on VALIS's QC nuclei, so it waits
# for VALIS). NUMBERS overrides what the cells print: none | scorer | image | auto.
#
# Submit (login node):
#   mkdir -p /hpcnfs/home/ieo7660/pipelines/logs
#   mkdir -p /beegfs/scratch/ieo7660/ihc_method/mosaic_033 && cd /beegfs/scratch/ieo7660/ihc_method/mosaic_033
#   sbatch ~/pipelines/mirage/benchmarks/submit_mosaic.sh /path/to/input.csv
# Options go through --export (a bare VAR=x before sbatch has not reached the job here):
#   sbatch --export=ALL,SEG_QC=1,ROWS=18 ~/pipelines/mirage/benchmarks/submit_mosaic.sh input.csv
# Re-submitting from the same directory CONTINUES: finished steps carry a .done marker and are
# skipped (Nextflow runs -resume), so a re-submit with other mosaic options only redraws.
# Watch:  squeue -u $USER ; tail -f /hpcnfs/home/ieo7660/pipelines/logs/mosaic_<jobid>.out
# ============================================================================
# No `set -u` / `set -o pipefail`, like submit_arms.sh: this job sources ~/.bashrc and runs
# `conda activate`, whose scripts read variables a batch job leaves unset, so `set -u` kills
# the job there with "unbound variable". Failures are checked explicitly below instead.

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT="${INPUT:-${1:-}}"                              # the samplesheet; relative = to SUBMIT_DIR
ROOT="${ROOT:-$SUBMIT_DIR}"                          # every run, work dir and the mosaic land here
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"          # the checkout, on `benchmarking`
PROFILES="${PROFILES:-singularity,ieo}"
SITE_CONFIG="${SITE_CONFIG:-$SRC_DIR/conf/ieo.config}"
CONDA_ENV="${CONDA_ENV:-nf-env}"
PIXEL_SIZE="${PIXEL_SIZE:-0.325}"    # µm/px, as the arms (benchmark.config); 'auto' = the file's own
SEG_QC="${SEG_QC:-0}"                # 0 = reg_qc=1, no WARP_SEG_QC, no numbers; 1 = reg_qc=2 + numbers
NUMBERS="${NUMBERS:-}"               # empty = none at SEG_QC=0, scorer values (auto) at SEG_QC=1
ASHLAR_TILE="${ASHLAR_TILE:-1024}"
ASHLAR_OVERLAP="${ASHLAR_OVERLAP:-0.1}"
ASHLAR_MAX_DISCARD="${ASHLAR_MAX_DISCARD:-1}"  # tiles ASHLAR may replace with a model
                                     # prediction before solve.py aborts. 1 = never abort:
                                     # a starved budget still finishes and its discard
                                     # fraction is reported, which is what a sweep needs.
ASHLAR_SHIFT_UM="${ASHLAR_SHIFT_UM:-15}"   # ONE OR MORE budgets (space-separated) for the
                                     # cross-cycle drift; one arm and one column each.
                                     # 15 IS ASHLAR'S OWN SHIPPED DEFAULT (`ashlar --maximum-shift`,
                                     # scripts/ashlar.py), so the default column is the tool AS
                                     # PUBLISHED -- which is the comparison a reader asks for.
                                     # IT IS ALSO STARVED HERE and that must be said out loud:
                                     # measured on 033 the cycles sit ~1187 px = 386 um apart, 26x
                                     # the default budget, and ASHLAR replaces out-of-range tiles
                                     # with MODEL PREDICTIONS instead of erroring (at 30 um it
                                     # discarded 30-57%). So this column is not "ASHLAR's accuracy",
                                     # it is "ASHLAR's accuracy at its default budget, with N% of
                                     # its placements invented" -- never quote it without the
                                     # discard fraction beside it. Pass a list, e.g.
                                     # ASHLAR_SHIFT_UM="15 480", to put the tuned run next to it.
ASHLAR_REG_QC="${ASHLAR_REG_QC:-0}"  # 1 also writes ASHLAR's 8-bit Before/After composite (the
                                     # figures read the stitched slide itself; that step is sized
                                     # >=100 GB in the pipeline and runs inside this job)
PATIENT="${PATIENT:-}"               # empty = one mosaic per patient in the samplesheet
ROWS="${ROWS:-}"                     # (round, ROI) cells per mosaic; empty = the moving rounds
                                     # of the first patient (one ROI each)
DRAW="${DRAW:-1}"                    # 0 = build the arms and stop (submit_figures.sh draws
                                     # its own mosaics from the plan, so it sets this)
VARIANTS="${VARIANTS:-1}"             # draw N mosaics on different tissue and pick the best
MOSAIC_ARGS="${MOSAIC_ARGS:-}"       # extra reg_mosaic.py flags, e.g. "--orient rounds-as-rows"
# -------------------------------------------------------------------------------

[[ -n "$INPUT" ]] || { echo "usage: sbatch submit_mosaic.sh <samplesheet.csv>  (or --export=ALL,INPUT=...)" >&2; exit 1; }
[[ "$INPUT" = /* ]] || INPUT="$SUBMIT_DIR/$INPUT"
[[ -s "$INPUT" ]] || { echo "samplesheet $INPUT not found or empty" >&2; exit 1; }
[[ -n "$NUMBERS" ]] || { if [[ "$SEG_QC" == "1" ]]; then NUMBERS=auto; else NUMBERS=none; fi; }
if [[ -z "$ROWS" ]]; then
  first=$(tail -n +2 "$INPUT" | tr -d '\r' | grep -v '^[[:space:]]*$' | head -1 | cut -d, -f1)
  ROWS=$(tail -n +2 "$INPUT" | tr -d '\r' | awk -F, -v p="$first" '$1 == p && $3 != "true"' | wc -l | tr -d ' ')
fi
(( ROWS >= 1 )) || { echo "no moving slide (is_reference=false) in $INPUT" >&2; exit 1; }
mkdir -p "$ROOT/.launch"
cd "$ROOT" || exit 1

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
command -v nextflow >/dev/null || { echo "nextflow not on PATH (check CONDA_ENV)" >&2; exit 1; }
command -v python3  >/dev/null || { echo "python3 not on PATH (check CONDA_ENV)" >&2; exit 1; }
grep -q '"none"' "$SRC_DIR/benchmarks/reg_mosaic.py" 2>/dev/null \
  || { echo "$SRC_DIR predates the mosaic prototype: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }
[[ -f "$SITE_CONFIG" ]] || { echo "site config $SITE_CONFIG missing" >&2; exit 1; }

# Same engine and image-cache setup as benchmarks/submit_arms.sh (the reasons are there).
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
export ASHLAR_EXEC="${ASHLAR_EXEC:-singularity exec $SING_BINDS $(ensure_sif labsyspharm/ashlar:1.20.0)}"
export QC_EXEC="${QC_EXEC:-singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-tiled:1.0.0)}"
export REGQC_EXEC="${REGQC_EXEC:-singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-regqc:1.0.0)}"
# The figures need matplotlib + scikit-image + tifffile AND imagecodecs: the slides are LZW,
# and the segeval image cannot decode them (job 6844142). The quantify image carries all four.
MOSAIC_EXEC="${MOSAIC_EXEC:-singularity exec $SING_BINDS $(ensure_sif bolt3x/mirage-quantify:1.0.0)}"

echo "=================================================="
echo "Mosaic job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}  $(date)"
echo "Input:    $INPUT"
echo "Root:     $ROOT"
echo "Seg QC:   $SEG_QC (numbers: $NUMBERS)   rows: $ROWS   pixel size: $PIXEL_SIZE"
echo "Checkout: $SRC_DIR @ $(git -C "$SRC_DIR" rev-parse --short HEAD) ($(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD))"
echo "=================================================="

done_marker() { [[ -f "$ROOT/$1/.done" ]]; }

# run_nf <step> <input.csv> key=value...   -- one pipeline run, typed params, own trace dir
run_nf() {
  local step="$1" in_csv="$2"; shift 2
  local rundir="$ROOT/.launch/$step" outdir="$ROOT/$step"
  if done_marker "$step"; then echo "[$step] DONE already, skipping"; return 0; fi
  mkdir -p "$rundir" "$outdir/trace"
  # Integers and booleans travel in a -params-file (Nextflow 26 stringifies CLI params).
  # cleanup_level=none: the registration runs RE-ENTER preprocess_shared's published tree.
  # cleanup_work=false: keeps the work dir, so a re-submit resumes instead of redoing.
  # trace_dir per run: two heads sharing one trace file corrupt it.
  # pixel_size=auto is the pipeline default and params_json types the param as a number,
  # so it is passed only when it IS a number.
  local px=(); [[ "$PIXEL_SIZE" != auto ]] && px=("pixel_size=$PIXEL_SIZE")
  (cd "$SRC_DIR" && python3 -m benchmarks.params_json --out "$rundir/params.json" \
      "${px[@]+"${px[@]}"}" cleanup_level=none cleanup_work=false enable_trace=true \
      "trace_dir=$outdir/trace" "$@") || { echo "[$step] could not build params" >&2; return 1; }
  echo "[$step] launching -> $outdir  ($*)"
  local rc=0
  (
    cd "$rundir" || exit 1
    nextflow -q run "$SRC_DIR" \
      -profile "$PROFILES" -c "$SITE_CONFIG" \
      -work-dir "$rundir/work" -resume \
      -params-file "$rundir/params.json" \
      --input "$in_csv" --outdir "$outdir" \
      > "$outdir/nextflow.stdout.log" 2> "$outdir/nextflow.stderr.log"
  ) || rc=$?
  if (( rc != 0 )); then
    echo "[$step] FAILED (exit $rc); last lines of $outdir/nextflow.stdout.log:" >&2
    tail -n 25 "$outdir/nextflow.stdout.log" >&2
    echo "full log: $rundir/.nextflow.log" >&2
    return "$rc"
  fi
  date '+%F %T' > "$outdir/.done"
  echo "[$step] OK"
}

# ---- 1. preprocessing, once ---------------------------------------------------
run_nf preprocess_shared "$INPUT" stop=preprocessing || exit 1
PREPROC_CSV="$ROOT/preprocess_shared/csv/preprocessed.csv"
[[ -s "$PREPROC_CSV" ]] || { echo "no $PREPROC_CSV after preprocessing" >&2; exit 1; }

# ---- 2. VALIS high (micro 2), STARE high and ASHLAR ---------------------------
REG_QC=1; [[ "$SEG_QC" == "1" ]] && REG_QC=2
export ASHLAR_SEG_QC="$SEG_QC" ASHLAR_REG_QC
# every ASHLAR step at the run's pixel size, not the slide header's (0.3453 on the ND2 slides)
[[ "$PIXEL_SIZE" != auto ]] && export ASHLAR_PIXEL_SIZE_UM="$PIXEL_SIZE"
REG_COMMON=(start=registration stop=registration "reg_qc=$REG_QC")
# ASHLAR_SHIFT_UM may be a LIST: one arm and one mosaic column per budget, which is how a
# shift sweep is read -- the columns sit side by side on the same tissue.
ASHLAR_ARMS=()
for _s in $ASHLAR_SHIFT_UM; do ASHLAR_ARMS+=("ashlar_t${ASHLAR_TILE}_s${_s}"); done
ASHLAR_ARM="${ASHLAR_ARMS[0]}"     # the first one, for the SEG_QC=1 ordering below

run_ashlar_one() {                 # run_ashlar_one <arm> <shift> <arm whose QC nuclei score it>
  local arm="$1" shift_um="$2" from="$3"
  if done_marker "$arm"; then echo "[$arm] DONE already, skipping"; return 0; fi
  echo "[$arm] launching (tile $ASHLAR_TILE, overlap $ASHLAR_OVERLAP, shift ${shift_um} um, seg QC $SEG_QC, discard<=${ASHLAR_MAX_DISCARD})"
  mkdir -p "$ROOT/$arm"
  if ASHLAR_MAX_DISCARD="$ASHLAR_MAX_DISCARD" \
     "$SRC_DIR/benchmarks/run_ashlar_arm.sh" "$ROOT" "$arm" "$from" "$PREPROC_CSV" \
        "$ASHLAR_TILE" "$ASHLAR_OVERLAP" "$shift_um" \
        > "$ROOT/$arm/ashlar.stdout.log" 2> "$ROOT/$arm/ashlar.stderr.log"; then
    date '+%F %T' > "$ROOT/$arm/.done"; echo "[$arm] OK"; return 0
  fi
  echo "[$arm] FAILED; last lines of ashlar.stderr.log:" >&2
  tail -n 25 "$ROOT/$arm/ashlar.stderr.log" >&2
  return 1
}

run_ashlar() {                     # every budget; one failing does not stop the others
  local i=0 rc=0
  for _s in $ASHLAR_SHIFT_UM; do
    run_ashlar_one "${ASHLAR_ARMS[$i]}" "$_s" "$1" || rc=1
    i=$((i + 1))
  done
  return "$rc"
}

run_nf valis_high_micro2 "$PREPROC_CSV" "${REG_COMMON[@]}" \
    registration_method=valis memory_mode=high reg_micro_reg=2 &
pid_valis=$!
run_nf stare_high "$PREPROC_CSV" "${REG_COMMON[@]}" \
    registration_method=tiled reg_tiled_mode=high &
pid_stare=$!
rc_ashlar=1
if [[ "$SEG_QC" != "1" ]]; then    # nothing to wait for: ASHLAR needs only the preprocessed slides
  run_ashlar none & pid_ashlar=$!
fi
rc_valis=0; wait "$pid_valis" || rc_valis=$?
rc_stare=0; wait "$pid_stare" || rc_stare=$?
if [[ "$SEG_QC" != "1" ]]; then
  rc_ashlar=0; wait "$pid_ashlar" || rc_ashlar=$?
elif (( rc_valis == 0 )); then
  run_ashlar valis_high_micro2 && rc_ashlar=0
elif (( rc_stare == 0 )); then
  run_ashlar stare_high && rc_ashlar=0
else
  echo "[$ASHLAR_ARM] SKIP: SEG_QC=1 and neither pipeline arm finished, so there are no QC nuclei" >&2
fi

# ---- 3. the mosaic, over every arm that finished ------------------------------
ARM_DIRS=(); LABELS=()
(( rc_valis  == 0 )) && { ARM_DIRS+=("$ROOT/valis_high_micro2"); LABELS+=(--label "valis_high_micro2=VALIS high"); }
(( rc_stare  == 0 )) && { ARM_DIRS+=("$ROOT/stare_high");        LABELS+=(--label "stare_high=STARE high"); }
# ASHLAR's column needs its registered slides, not a clean exit: its last per-round step (the QC
# composite, sized >=100 GB in the pipeline) can fail in this head job after every slide was
# stitched, and the mosaic reads the stitched slides themselves.
ashlar_slides_ok() {               # ashlar_slides_ok <arm>
  local csv="$ROOT/$1/csv/registered.csv" n=0 img
  [[ -s "$csv" ]] || return 1
  while IFS= read -r img; do
    [[ -s "$img" ]] || { echo "[mosaic] ASHLAR slide missing: $img" >&2; return 1; }
    n=$((n + 1))
  done < <(tail -n +2 "$csv" | awk -F, '$4 != "true" { print $3 }')
  (( n > 0 ))
}
# Each budget is judged on ITS OWN output, not on the sweep's exit status: a run can exit
# non-zero at its last per-round step (the QC composite, sized >=100 GB) with every slide
# stitched, and one starved budget must not cost the others their columns.
(( rc_ashlar == 0 )) || echo "[mosaic] an ASHLAR budget exited non-zero; drawing the columns whose slides exist" >&2
for arm in "${ASHLAR_ARMS[@]}"; do
  ashlar_slides_ok "$arm" || continue
  ARM_DIRS+=("$ROOT/$arm")
  LABELS+=(--label "$arm=ASHLAR ${arm##*_s}um")
done
if (( ${#ARM_DIRS[@]} == 0 )); then
  echo "no arm finished; no mosaic" >&2; exit 1
fi
PATIENT_ARGS=(); [[ -n "$PATIENT" ]] && PATIENT_ARGS=(--patient "$PATIENT")
PX_ARGS=(); [[ "$PIXEL_SIZE" != auto ]] && PX_ARGS=(--pixel-size-um "$PIXEL_SIZE")
if [[ "$DRAW" != "1" ]]; then
  # same exit status as the drawing path below: an arm that failed must not be reported as
  # success just because nobody asked for a picture of it
  echo "[mosaic] DRAW=0: arms are built (${ARM_DIRS[*]}), not drawing"
  (( rc_valis == 0 && rc_stare == 0 && rc_ashlar == 0 ))
  exit $?
fi
echo "[mosaic] columns: ${ARM_DIRS[*]}"
# shellcheck disable=SC2086
(
  cd "$SRC_DIR" || exit 1
  SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
    $MOSAIC_EXEC python3 -m benchmarks.reg_mosaic "${ARM_DIRS[@]}" \
      "${PATIENT_ARGS[@]+"${PATIENT_ARGS[@]}"}" --rows "$ROWS" -o "$ROOT/mosaic" \
      --numbers "$NUMBERS" --palette magenta-cyan "${PX_ARGS[@]+"${PX_ARGS[@]}"}" \
      --variants "$VARIANTS" "${LABELS[@]}" $MOSAIC_ARGS
) || { echo "[mosaic] FAILED" >&2; exit 1; }

echo "=================================================="
if (( VARIANTS > 1 )); then
  echo "Done $(date). Mosaics: $ROOT/mosaic/<patient>_v1..v${VARIANTS}_mosaic.{png,pdf} -- pick one"
else
  echo "Done $(date). Mosaic: $ROOT/mosaic/<patient>_mosaic.{png,pdf}"
fi
echo "Exit status: valis=$rc_valis stare=$rc_stare ashlar=$rc_ashlar"
echo "=================================================="
(( rc_valis == 0 && rc_stare == 0 && rc_ashlar == 0 ))
