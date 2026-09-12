#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ashlar_arm.sh — one EXTERNAL-baseline arm of the real-sample benchmark.
#
# ashlar is not a pipeline backend (the registration_method enum is
# ['valis','tiled']), so this arm is not a Nextflow launch. It is the three-step
# external path, run once per patient:
#
#     retile ref  ─┐
#     retile mov  ─┴─> ashlar solve ──> manifest.json ──> warp_seg_qc.py
#
# and its OUTPUT is deliberately indistinguishable from a registration arm's:
#
#     <root>/<arm>/<patient>/qc/registration/<pid>_<mov>_seg_qc.json
#
# That is the whole point. benchmarks/ashlar/solve.py rewrites ashlar's per-tile
# placements into the same M0 + mesh manifest STARE emits, and
# bin/warp_seg_qc.py --method tiled reads that manifest JVM-free — so ashlar is
# scored by the PIPELINE'S OWN reg_qc=2 seg-overlap scorer, on the SAME nuclei,
# into the SAME tree. Same metric family, same columns, one layout. Scored any
# other way it lands in a different metric family that shares no column with the
# arm table, which is what the deleted synthetic ground-truth rung did.
#
# THE NUCLEI ARE REUSED, NOT RE-SEGMENTED. seg_qc.nf segments the NATIVE slides,
# so its geojsons are identical across registration arms sharing a seg_method.
# Re-segmenting here would confound the ashlar-vs-VALIS difference with segmenter
# noise, so this reads <root>/<from_arm>/<patient>/qc/registration/geojson/.
#
# Usage:
#   run_ashlar_arm.sh <root> <arm> <geojson_from_arm> <preprocessed_csv> \
#                     <tile_size> <overlap_fraction> <max_shift_um>
#
# Environment:
#   ASHLAR_EXEC   command prefix for the retile/solve steps (they need the ashlar
#                 package: labsyspharm/ashlar:1.20.0, amd64-only).
#                 e.g. ASHLAR_EXEC="singularity exec ashlar_1.20.0.sif"
#   QC_EXEC       command prefix for warp_seg_qc.py (needs the pipeline's tiled
#                 image: bolt3x/mirage-tiled:1.0.0 — see lib/WarpBackends.groovy).
#                 e.g. QC_EXEC="singularity exec mirage-tiled_1.0.0.sif"
#   Both default to empty, i.e. run bare against whatever is on PATH.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:?results root}"
ARM="${2:?arm name}"
FROM_ARM="${3:?arm whose QC geojsons to reuse}"
PREPROC_CSV="${4:?preprocessed.csv from the shared preprocessing run}"
TILE="${5:?tile size (px)}"
OVERLAP="${6:?overlap fraction}"
MAXSHIFT="${7:?maximum shift (um)}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ASHLAR_EXEC="${ASHLAR_EXEC:-}"
QC_EXEC="${QC_EXEC:-}"

OUT="$ROOT/$ARM"
WORK="$ROOT/.launch/$ARM"
mkdir -p "$OUT" "$WORK"

# RESOURCE CAPTURE. Every other arm is a Nextflow run and gets a trace for free;
# this one runs outside Nextflow and got none, so ASHLAR was absent from
# measurements.csv / run_cost / resource_stats while sitting in the accuracy table.
# Each heavy invocation below goes through benchmarks/trace_step.py, which appends
# one Nextflow-format trace row (wall-clock, peak RSS, CPU time, exit status) to
# the SAME path load_runs reads for every arm -- <root>/<arm>/trace/trace.txt --
# and records the step's input size in size_logs/input_sizes.csv. Process names
# ASHLAR_RETILE / ASHLAR_SOLVE / ASHLAR_SEG_QC, tag = patient. The wrapper exits
# with the child's status, so `set -e` and the `|| { ... }` below behave as before.
TRACE="$OUT/trace/trace.txt"
mkdir -p "$OUT/trace"
step() {                         # step <PROCESS> <patient> [--input PATH ...] -- <command...>
  python3 "$HERE/trace_step.py" --trace "$TRACE" --process "$1" --tag "$2" "${@:3}"
}

# preprocessed.csv columns (lib/Checkpoint.groovy): the header is read rather than
# assumed, because a positional read is exactly how a schema change becomes a silent
# mis-registration instead of an error.
col() {                          # col <name> <csv>
  head -n1 "$2" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1
}
C_PID=$(col patient_id "$PREPROC_CSV")
C_IMG=$(col preprocessed_image "$PREPROC_CSV")
C_REF=$(col is_reference "$PREPROC_CSV")
if [[ -z "$C_PID" || -z "$C_IMG" || -z "$C_REF" ]]; then
  echo "[$ARM] ERROR: $PREPROC_CSV lacks patient_id/preprocessed_image/is_reference" >&2
  exit 1
fi

patients=$(tail -n +2 "$PREPROC_CSV" | tr -d '\r' | cut -d',' -f"$C_PID" | sort -u)
[[ -n "$patients" ]] || { echo "[$ARM] ERROR: no patients in $PREPROC_CSV" >&2; exit 1; }

rc=0
for pid in $patients; do
  rows=$(tail -n +2 "$PREPROC_CSV" | tr -d '\r' | awk -F',' -v p="$pid" -v c="$C_PID" '$c==p')
  ref_img=$(echo "$rows" | awk -F',' -v r="$C_REF" -v i="$C_IMG" '$r=="true"{print $i; exit}')
  if [[ -z "$ref_img" ]]; then
    echo "[$ARM/$pid] SKIP: no is_reference=true row" >&2; rc=1; continue
  fi

  gj_dir="$ROOT/$FROM_ARM/$pid/qc/registration/geojson"
  if [[ ! -d "$gj_dir" ]]; then
    # Not fatal for the other patients: one arm's QC failing should not lose the rest.
    echo "[$ARM/$pid] SKIP: $gj_dir missing — arm '$FROM_ARM' produced no QC nuclei" >&2
    rc=1; continue
  fi

  ref_name=$(basename "$ref_img"); ref_name="${ref_name%%.*}"
  ref_gj="$gj_dir/${ref_name}.geojson"
  if [[ ! -f "$ref_gj" ]]; then
    # `ls` exits 1 on no match and this script runs under `set -o pipefail`, so the
    # no-match case has to be written as the ASSIGNMENT it is. Nothing is swallowed:
    # the emptiness is asserted immediately below.
    ref_gj=$(ls "$gj_dir"/*"${ref_name}"*.geojson 2>/dev/null | head -1) || ref_gj=""
  fi
  if [[ -z "$ref_gj" ]]; then
    echo "[$ARM/$pid] SKIP: no reference geojson matching '$ref_name' in $gj_dir" >&2
    rc=1; continue
  fi

  # The reference is retiled ONCE per patient and reused for every moving slide.
  ref_tiles="$WORK/$pid/ref"
  mkdir -p "$ref_tiles"
  # shellcheck disable=SC2086
  step ASHLAR_RETILE "$pid" --input "$ref_img" -- \
      $ASHLAR_EXEC python3 -m benchmarks.ashlar.retile \
      --image "$ref_img" --outdir "$ref_tiles" --cycle 0 \
      --tile-size "$TILE" --overlap "$OVERLAP"

  while IFS= read -r mov_img; do
    [[ -n "$mov_img" ]] || continue
    # Slide names are DERIVED from the filenames, never hardcoded: warp_seg_qc looks the
    # moving slide up BY NAME in the manifest, and a name that does not match yields an
    # empty transform that scores as a perfect identity — a silent pass, not an error.
    mov_name=$(basename "$mov_img"); mov_name="${mov_name%%.*}"
    mov_gj="$gj_dir/${mov_name}.geojson"
    if [[ ! -f "$mov_gj" ]]; then
      # `ls` exits 1 on no match and this script runs under `set -o pipefail`, so the
      # no-match case has to be written as the ASSIGNMENT it is. Nothing is swallowed:
      # the emptiness is asserted immediately below.
      mov_gj=$(ls "$gj_dir"/*"${mov_name}"*.geojson 2>/dev/null | head -1) || mov_gj=""
    fi
    if [[ -z "$mov_gj" ]]; then
      echo "[$ARM/$pid] SKIP $mov_name: no geojson in $gj_dir" >&2; rc=1; continue
    fi

    d="$WORK/$pid/$mov_name"
    mkdir -p "$d/mov"
    qc_out="$OUT/$pid/qc/registration"
    mkdir -p "$qc_out"

    echo "[$ARM/$pid] $ref_name -> $mov_name (tile=$TILE shift=${MAXSHIFT}um)"
    # shellcheck disable=SC2086
    step ASHLAR_RETILE "$pid" --input "$mov_img" -- \
        $ASHLAR_EXEC python3 -m benchmarks.ashlar.retile \
        --image "$mov_img" --outdir "$d/mov" --cycle 1 \
        --tile-size "$TILE" --overlap "$OVERLAP"
    # shellcheck disable=SC2086
    step ASHLAR_SOLVE "$pid" --input "$ref_tiles" --input "$d/mov" -- \
        $ASHLAR_EXEC python3 -m benchmarks.ashlar.solve \
        --ref-tiles "$ref_tiles" --moving-tiles "$d/mov" \
        --reference-name "$ref_name" --moving-name "$mov_name" \
        --maximum-shift "$MAXSHIFT" \
        --out-manifest "$d/manifest.json" --out-tre "$d/tre.json"

    # --method tiled is the ONLY backend flag the manifest path needs
    # (lib/WarpBackends.groovy's tiled entry) — no --micro-reg, no --checkpoint-dir,
    # no --jvm-heap-gb. ashlar's terminal stage is `refined`, the same name STARE's is,
    # which is what lets the analysis layer's _STAGE_RANK reduce both without a case.
    # shellcheck disable=SC2086
    step ASHLAR_SEG_QC "$pid" --input "$ref_gj" --input "$mov_gj" -- \
        $QC_EXEC python3 "$REPO/bin/warp_seg_qc.py" \
        --method tiled \
        --pickle "$d/manifest.json" \
        --ref-slide "$ref_name" \
        --moving-slide "$mov_name" \
        --ref-geojson "$ref_gj" \
        --moving-geojson "$mov_gj" \
        --patient-id "$pid" \
        --output "$qc_out/${pid}_${mov_name}_seg_qc.json" \
        --per-cell-csv "$qc_out/${pid}_${mov_name}_reg_residuals.csv" \
      || { echo "[$ARM/$pid] FAILED scoring $mov_name" >&2; rc=1; }
  done < <(echo "$rows" | awk -F',' -v r="$C_REF" -v i="$C_IMG" '$r!="true"{print $i}')
done

exit "$rc"
