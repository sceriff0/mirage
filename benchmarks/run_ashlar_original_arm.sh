#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The ASHLAR arm, AS PUBLISHED: the original `ashlar` command per patient.
#
# run_ashlar_arm.sh drives ASHLAR's cross-cycle class alone (reference tiles placed by
# fiat, image written by the pipeline's stitcher). This arm runs the original program
# instead -- benchmarks/ashlar/original.py calls ashlar.scripts.ashlar.main unmodified --
# so the reference is stitched by EdgeAligner, every cycle is aligned by LayerAligner and
# the registered image is ASHLAR's own mosaic:
#
#   synthetic raw tiles (every cycle) ──> ashlar ref cyc1 cyc2 ... -o ashlar_output.ome.tif
#        │                                   │
#        └ a stage error and sensor noise    ├─> <patient>/registered/registered/*.ome.tiff
#          per tile: retile.py                │    (the mosaic, one slide per file)
#                                            └─> one manifest per cycle ──> warp_seg_qc.py
#
# THE INPUT IS THE ONE THING THAT CANNOT BE ASHLAR'S: the slides exist only stitched, so
# the raw tiles are synthesised from them (ASHLAR_STAGE_JITTER_UM, ASHLAR_NOISE_FRAC). The
# scoring is the benchmark's, on the SAME nuclei as every other arm: ASHLAR moves each
# tile rigidly, and the manifest carries exactly that field (constant over a tile, ramping
# across the overlap band where its mosaic blends two tiles).
#
# Usage and environment as run_ashlar_arm.sh:
#   run_ashlar_original_arm.sh <root> <arm> <geojson_from_arm> <preprocessed_csv> \
#                              <tile_size> <overlap_fraction> <max_shift_um>
#   ASHLAR_EXEC / QC_EXEC / REGQC_EXEC, ASHLAR_SEG_QC, ASHLAR_REG_QC, ASHLAR_MAX_DISCARD,
#   ASHLAR_PIXEL_SIZE_UM -- and:
#   ASHLAR_STAGE_JITTER_UM  each tile is cut up to this far off its nominal corner (um,
#                           default 2): the stage error ASHLAR's stitching exists to find.
#   ASHLAR_NOISE_FRAC       per-tile sensor noise, s.d. as a fraction of each channel's
#                           1st-99th percentile range (default 0.01). Not optional in
#                           practice: on tiles with IDENTICAL overlaps ASHLAR's error
#                           metric fails on a rounding difference (job 6844139).
#   ASHLAR_SEED             seed of both (default 0).
#   ASHLAR_ARGS             extra flags handed to ashlar as typed (e.g. "--filter-sigma 1").
#   ASHLAR_KEEP_WORK        1 keeps the tiles and ashlar_output.ome.tif under
#                           <root>/.launch/<arm>/ (default 0: they are several times the
#                           slides' size and the registered slides hold the same pixels).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:?results root}"
ARM="${2:?arm name}"
FROM_ARM="${3:?arm whose QC geojsons to reuse}"
PREPROC_CSV="${4:?preprocessed.csv from the shared preprocessing run}"
TILE="${5:?tile size (px)}"
OVERLAP="${6:?overlap fraction}"
MAXSHIFT="${7:?maximum shift (um)}"
MAX_DISCARD="${ASHLAR_MAX_DISCARD:-1}"
JITTER_UM="${ASHLAR_STAGE_JITTER_UM:-2}"
NOISE_FRAC="${ASHLAR_NOISE_FRAC:-0.01}"
SEED="${ASHLAR_SEED:-0}"
EXTRA_ARGS="${ASHLAR_ARGS:-}"
KEEP_WORK="${ASHLAR_KEEP_WORK:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ASHLAR_EXEC="${ASHLAR_EXEC:-}"
QC_EXEC="${QC_EXEC:-}"
REGQC_EXEC="${REGQC_EXEC:-}"
SEG_QC="${ASHLAR_SEG_QC:-1}"
REG_QC="${ASHLAR_REG_QC:-1}"
PIXEL_SIZE_OVERRIDE="${ASHLAR_PIXEL_SIZE_UM:-}"

# The pinned STARE release on the path, exactly as run_ashlar_arm.sh does and for the same
# reason: the manifest assembly (stare.manifest) is imported inside ASHLAR's image.
STARE_URL=$(grep -oE 'https://github\.com/sceriff0/stare/archive/refs/tags/[^[:space:]]+\.tar\.gz' \
  "$REPO/requirements/stare.txt" | head -1)
[[ -n "$STARE_URL" ]] || { echo "no STARE release URL in $REPO/requirements/stare.txt" >&2; exit 1; }
STARE_SRC="${ASHLAR_STARE_SRC:-$ROOT/.cache/stare-$(basename "$STARE_URL" .tar.gz)/src}"
if [[ ! -f "$STARE_SRC/stare/manifest.py" ]]; then
  _tmp=$(mktemp -d "$ROOT/.cache.XXXXXX" 2>/dev/null || { mkdir -p "$ROOT/.cache" && mktemp -d "$ROOT/.cache/dl.XXXXXX"; })
  if curl -fsSL "$STARE_URL" | tar -xz -C "$_tmp" --strip-components=1; then
    mkdir -p "$(dirname "$STARE_SRC")"
    if ! mv -n "$_tmp/src" "$STARE_SRC" 2>/dev/null; then :; fi
  fi
  rm -rf "$_tmp"
  [[ -f "$STARE_SRC/stare/manifest.py" ]] || { echo "could not fetch STARE from $STARE_URL" >&2; exit 1; }
fi
STEP_PYTHONPATH="$REPO:$STARE_SRC"
export PYTHONPATH="$STEP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$STEP_PYTHONPATH" APPTAINERENV_PYTHONPATH="$STEP_PYTHONPATH"

OUT="$ROOT/$ARM"
WORK="$ROOT/.launch/$ARM"
mkdir -p "$OUT/trace" "$OUT/csv" "$WORK"
TRACE="$OUT/trace/trace.txt"
step() {                         # step <PROCESS> <patient> [--input PATH ...] -- <command...>
  python3 "$HERE/trace_step.py" --trace "$TRACE" --process "$1" --tag "$2" "${@:3}"
}
col() { head -n1 "$2" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1; }
C_PID=$(col patient_id "$PREPROC_CSV")
C_IMG=$(col preprocessed_image "$PREPROC_CSV")
C_REF=$(col is_reference "$PREPROC_CSV")
C_CH=$(col channels "$PREPROC_CSV")
C_PX=$(col pixel_size "$PREPROC_CSV")
if [[ -z "$C_PID" || -z "$C_IMG" || -z "$C_REF" || -z "$C_CH" ]]; then
  echo "[$ARM] ERROR: $PREPROC_CSV lacks patient_id/preprocessed_image/is_reference/channels" >&2
  exit 1
fi
REG_CSV="$OUT/csv/registered.csv"
printf 'patient_id,id,registered_image,is_reference,channels,pixel_size\n' > "$REG_CSV"

patients=$(tail -n +2 "$PREPROC_CSV" | tr -d '\r' | cut -d',' -f"$C_PID" | sort -u)
[[ -n "$patients" ]] || { echo "[$ARM] ERROR: no patients in $PREPROC_CSV" >&2; exit 1; }

rc=0
for pid in $patients; do
  rows=$(tail -n +2 "$PREPROC_CSV" | tr -d '\r' | awk -F',' -v p="$pid" -v c="$C_PID" '$c==p')
  ref_row=$(echo "$rows" | awk -F',' -v r="$C_REF" '$r=="true"{print; exit}')
  if [[ -z "$ref_row" ]]; then
    echo "[$ARM/$pid] SKIP: no is_reference=true row" >&2; rc=1; continue
  fi
  gj_dir="$ROOT/$FROM_ARM/$pid/qc/registration/geojson"
  if [[ "$SEG_QC" == "1" && ! -d "$gj_dir" ]]; then
    echo "[$ARM/$pid] SKIP: $gj_dir missing — arm '$FROM_ARM' produced no QC nuclei" >&2
    rc=1; continue
  fi
  ref_px=""; [[ -n "$C_PX" ]] && ref_px=$(echo "$ref_row" | cut -d',' -f"$C_PX")
  [[ -n "$PIXEL_SIZE_OVERRIDE" ]] && ref_px="$PIXEL_SIZE_OVERRIDE"
  px_arg=(); [[ "$ref_px" =~ ^[0-9.]+$ ]] && px_arg=(--pixel-size-um "$ref_px")

  # Every cycle of the patient, the reference FIRST: that order is the ashlar command's,
  # whose first file is the cycle everything else is aligned to.
  imgs=(); names=(); chans=()
  while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    img=$(echo "$row" | cut -d',' -f"$C_IMG")
    name=$(basename "$img"); name="${name%%.*}"
    imgs+=("$img"); names+=("$name"); chans+=("$(echo "$row" | cut -d',' -f"$C_CH")")
  done < <(echo "$ref_row"; echo "$rows" | awk -F',' -v r="$C_REF" '$r!="true"{print}')
  if [[ ${#imgs[@]} -lt 2 ]]; then
    echo "[$ARM/$pid] SKIP: no moving cycle" >&2; rc=1; continue
  fi

  reg_dir="$OUT/$pid/registered/registered"
  qc_out="$OUT/$pid/qc/registration"
  mkdir -p "$reg_dir" "$qc_out" "$WORK/$pid/tiles"
  tiles=(); regs=()
  ok=1
  for k in "${!imgs[@]}"; do
    t="$WORK/$pid/tiles/${names[$k]}"
    tiles+=("$t"); regs+=("$reg_dir/${names[$k]}_registered.ome.tiff")
    # shellcheck disable=SC2086
    step ASHLAR_RETILE "$pid" --input "${imgs[$k]}" -- \
        $QC_EXEC python3 -m benchmarks.ashlar.retile \
        --image "${imgs[$k]}" --outdir "$t" --cycle "$k" \
        --tile-size "$TILE" --overlap "$OVERLAP" "${px_arg[@]}" \
        --stage-jitter-um "$JITTER_UM" --noise-frac "$NOISE_FRAC" --seed "$SEED" \
        --exact-overlap --canvas-like "${imgs[@]}" \
      || { echo "[$ARM/$pid] FAILED retiling ${names[$k]}" >&2; ok=0; break; }
  done
  [[ "$ok" == "1" ]] || { rc=1; continue; }

  echo "[$ARM/$pid] ashlar ${names[0]} + $(( ${#names[@]} - 1 )) cycle(s) (tile=$TILE shift=${MAXSHIFT}um jitter=${JITTER_UM}um noise=$NOISE_FRAC)"
  # shellcheck disable=SC2086
  if ! step ASHLAR_SOLVE "$pid" $(printf -- '--input %s ' "${tiles[@]}") -- \
      $ASHLAR_EXEC python3 -m benchmarks.ashlar.original \
      --tiles "${tiles[@]}" --names "${names[@]}" --outdir "$WORK/$pid/ashlar" \
      --maximum-shift "$MAXSHIFT" --max-discard-fraction "$MAX_DISCARD" \
      --split "${regs[@]}" --channels "${chans[@]}" \
      --ashlar-args $EXTRA_ARGS; then
    echo "[$ARM/$pid] FAILED ashlar" >&2; rc=1; continue
  fi
  # what ASHLAR was run with and where it put every tile: kept beside its images
  cp "$WORK/$pid/ashlar/positions.json" "$reg_dir/../ashlar_positions.json"

  ref_name="${names[0]}"
  printf '%s,%s,%s,true,%s,%s\n' "$pid" "$ref_name" "${regs[0]}" "${chans[0]}" "$ref_px" >> "$REG_CSV"
  for k in "${!imgs[@]}"; do
    [[ "$k" == "0" ]] && continue
    mov_name="${names[$k]}"
    printf '%s,%s,%s,false,%s,%s\n' "$pid" "${mov_name}_registered" "${regs[$k]}" "${chans[$k]}" "$ref_px" >> "$REG_CSV"
    cp "$WORK/$pid/ashlar/$mov_name/tre.json" "$qc_out/${pid}_${mov_name}_ashlar.json"
    if [[ "$REG_QC" == "1" ]]; then
      # shellcheck disable=SC2086
      step ASHLAR_REG_QC "$pid" --input "${regs[0]}" --input "${regs[$k]}" --input "${imgs[$k]}" -- \
          $REGQC_EXEC python3 "$REPO/bin/generate_registration_qc.py" \
          --reference "${regs[0]}" --registered "${regs[$k]}" --native "${imgs[$k]}" \
          --output "$qc_out" "${px_arg[@]}" \
        || { echo "[$ARM/$pid] FAILED registration QC for $mov_name" >&2; rc=1; }
    fi
    [[ "$SEG_QC" == "1" ]] || continue
    ref_gj="$gj_dir/${ref_name}.geojson"
    [[ -f "$ref_gj" ]] || { ref_gj=$(ls "$gj_dir"/*"${ref_name}"*.geojson 2>/dev/null | head -1) || ref_gj=""; }
    mov_gj="$gj_dir/${mov_name}.geojson"
    [[ -f "$mov_gj" ]] || { mov_gj=$(ls "$gj_dir"/*"${mov_name}"*.geojson 2>/dev/null | head -1) || mov_gj=""; }
    if [[ -z "$ref_gj" || -z "$mov_gj" ]]; then
      echo "[$ARM/$pid] SKIP scoring $mov_name: no geojson for it or the reference in $gj_dir" >&2
      rc=1; continue
    fi
    # shellcheck disable=SC2086
    step ASHLAR_SEG_QC "$pid" --input "$ref_gj" --input "$mov_gj" -- \
        $QC_EXEC python3 "$REPO/bin/warp_seg_qc.py" \
        --method tiled \
        --pickle "$WORK/$pid/ashlar/$mov_name/manifest.json" \
        --ref-slide "$ref_name" \
        --moving-slide "$mov_name" \
        --ref-geojson "$ref_gj" \
        --moving-geojson "$mov_gj" \
        --patient-id "$pid" \
        --output "$qc_out/${pid}_${mov_name}_seg_qc.json" \
        --per-cell-csv "$qc_out/${pid}_${mov_name}_reg_residuals.csv" \
      || { echo "[$ARM/$pid] FAILED scoring $mov_name" >&2; rc=1; }
  done
  if [[ "$KEEP_WORK" != "1" ]]; then
    rm -rf "$WORK/$pid/tiles" "$WORK/$pid/ashlar/ashlar_output.ome.tif"
  fi
done

exit "$rc"
