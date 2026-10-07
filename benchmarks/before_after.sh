#!/bin/bash
# before_after.sh -- Before | After figures of ONE registration arm, for any patients.
#
# No SLURM header on purpose: run it in an interactive job, not with sbatch. It runs no
# pipeline and compares no methods: per patient it draws the arm's Before and After of the
# whole tissue with a full-resolution inset (reg_overlay), VARIANTS insets per moving
# round, and assembles each into one figure with editable text (before_after_pairs.py).
#
#   srun --pty --mem=64G -c 4 -t 4:00:00 bash          # an interactive shell, then:
#   RESULTS=/beegfs/.../benchmark/arm_results PATIENTS="046 052" VARIANTS=5 \
#     ROUNDS="CD3_P53 CD4_CD8" ~/pipelines/mirage/benchmarks/before_after.sh
#
# Output (under OUT, default ./before_after):
#   <patient>/before_after/<patient>_<round>_v<k>.{pdf,png}   the figures
#   <patient>/<patient>_crops.csv is NOT written here; the panels stay in
#   <patient>/_anchor/ unless KEEP_PANELS=0.
#
# A patient whose figures exist is skipped, so a run that was interrupted continues;
# delete OUT/<patient> to redo it (e.g. after changing VARIANTS or ROUNDS).

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
RESULTS="${RESULTS:-}"                       # the arms' results root
ARM="${ARM:-valis_high_micro2}"              # the registration to show
PATIENTS="${PATIENTS:-}"                     # space-separated; empty = every patient of ARM
ROUNDS="${ROUNDS:-}"                         # space-separated moving rounds; empty = all
VARIANTS="${VARIANTS:-3}"                    # insets per round: one figure each
FIELD_UM="${FIELD_UM:-50000}"                # larger than any tissue = the whole tissue
ZOOM_UM="${ZOOM_UM:-300}"                    # the inset's side, drawn at full resolution
OUT="${OUT:-$PWD/before_after}"
FORMATS="${FORMATS:-pdf,png}"
KEEP_PANELS="${KEEP_PANELS:-1}"              # 0 = delete <patient>/_anchor once assembled
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}" # checkout on `benchmarking`
OVERLAY_ARGS="${OVERLAY_ARGS:-}"             # extra reg_overlay.py flags
SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
SING_BINDS="${SING_BINDS:---bind /beegfs --bind /hpcnfs}"
# -------------------------------------------------------------------------------

[[ -n "$RESULTS" && -d "$RESULTS/$ARM" ]] \
  || { echo "usage: RESULTS=<arm_results> [ARM=$ARM] [PATIENTS=..] [ROUNDS=..] [VARIANTS=3] $0" >&2
       echo "  (no $RESULTS/$ARM)" >&2; exit 1; }
[[ -f "$SRC_DIR/benchmarks/before_after_pairs.py" ]] \
  || { echo "$SRC_DIR has no benchmarks/before_after_pairs.py: git -C $SRC_DIR pull (benchmarking)" >&2; exit 1; }

# The image the other figure launchers render with, under the name they cache it by.
if [[ -z "${RENDER_EXEC:-}" ]]; then
  img="$SINGULARITY_CACHEDIR/bolt3x-mirage-quantify-1.0.0.img"
  [[ -s "$img" ]] || img="docker://bolt3x/mirage-quantify:1.0.0"
  RENDER_EXEC="singularity exec $SING_BINDS $img"
fi
run() {                           # run <python args...>, from the checkout, in the image
  # shellcheck disable=SC2086
  (
    cd "$SRC_DIR" || exit 1
    SINGULARITYENV_PYTHONPATH="$SRC_DIR" APPTAINERENV_PYTHONPATH="$SRC_DIR" PYTHONPATH="$SRC_DIR" \
      $RENDER_EXEC python3 "$@"
  )
}

if [[ -z "$PATIENTS" ]]; then     # every patient directory of the arm that holds slides
  for d in "$RESULTS/$ARM"/*/; do
    [[ -d "$d/registered" || -d "$d/qc" ]] && PATIENTS+=" $(basename "$d")"
  done
fi
read -r -a patients <<< "$PATIENTS"
[[ ${#patients[@]} -gt 0 ]] || { echo "no patient found under $RESULTS/$ARM; set PATIENTS" >&2; exit 1; }
read -r -a rounds <<< "$ROUNDS"
mkdir -p "$OUT"
echo "arm $ARM, ${#patients[@]} patient(s): ${patients[*]}; rounds: ${ROUNDS:-all}; $VARIANTS variant(s)"

failed=0
for pid in "${patients[@]}"; do
  root="$OUT/$pid"
  if compgen -G "$root/before_after/*.pdf" >/dev/null || compgen -G "$root/before_after/*.png" >/dev/null; then
    echo "[$pid] figures exist, skipped (delete $root to redo)"
    continue
  fi
  args=("$RESULTS/$ARM" -o "$root/_anchor" --patient "$pid" --field-um "$FIELD_UM"
        --zoom-um "$ZOOM_UM" --variants "$VARIANTS" --labels none --formats png)
  [[ ${#rounds[@]} -gt 0 ]] && args+=(--rounds "${rounds[@]}")
  echo "[$pid] drawing"
  # shellcheck disable=SC2086
  if ! run -m benchmarks.reg_overlay "${args[@]}" $OVERLAY_ARGS; then
    echo "[$pid] FAILED drawing" >&2; failed=$((failed + 1)); continue
  fi
  if ! run "$SRC_DIR/benchmarks/before_after_pairs.py" "$root" --formats "$FORMATS"; then
    echo "[$pid] FAILED assembling" >&2; failed=$((failed + 1)); continue
  fi
  [[ "$KEEP_PANELS" == "0" ]] && rm -rf "$root/_anchor"
done
echo "done: $OUT  (${#patients[@]} patient(s), $failed failed)"
[[ $failed -eq 0 ]]
