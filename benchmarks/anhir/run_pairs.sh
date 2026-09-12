#!/usr/bin/env bash
# Register every prepared ANHIR case with ONE of the pipeline's backends.
#
# Usage:
#   benchmarks/anhir/run_pairs.sh <prepared_dir> <results_root> <valis|tiled> [extra nextflow args...]
#
#   <prepared_dir>  prepare.py convert's --work: holds samplesheet.csv + pairs_manifest.csv
#   <results_root>  the run publishes to <results_root>/<method>/ (its --outdir)
#   <method>        the pipeline's registration_method
#   extra args      appended verbatim, e.g. --reg_tiled_mode low, -resume
#
# Environment:
#   ANHIR_PROFILE       -profile value (default: docker)
#   ANHIR_SITE_CONFIG   a site.config to pass with -c (required on a cluster; it
#                       carries max_cpus / max_memory, which the pipeline requires)
#   ANHIR_MAX_CPUS / ANHIR_MAX_MEMORY
#                       passed as --max_cpus / --max_memory when no site config is given
#
# Every case is one patient with the target slide as the reference, so a single
# run registers all of them; --start registration --stop registration skips
# preprocessing (the OME-TIFFs prepare.py wrote ARE the preprocessed slides; the
# pipeline's BaSiC illumination correction is for fluorescence) and everything
# after registration. -with-trace is passed so warp.py can read execution time.
#
# The transform each case's landmarks are warped through is what the run
# PUBLISHES -- registered/transform/*_registrar.pickle (valis) or
# registered/manifest/*_manifest.json (tiled) -- so cleanup_level must stay at
# its default ('none'); do not pass --cleanup_level final.

PREP="${1:?prepared_dir (prepare.py convert --work)}"
RESULTS="${2:?results_root}"
METHOD="${3:?method: valis | tiled}"
shift 3

case "$METHOD" in
  valis|tiled) ;;
  *) echo "method must be valis or tiled, got '$METHOD'" >&2; exit 2 ;;
esac
SHEET="$PREP/samplesheet.csv"
[[ -f "$SHEET" ]] || { echo "no samplesheet at $SHEET -- run prepare.py convert first" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$RESULTS/$METHOD"
mkdir -p "$OUT"

args=(run "$REPO"
  --input "$SHEET"
  --outdir "$OUT"
  --start registration --stop registration
  --registration_method "$METHOD"
  -profile "${ANHIR_PROFILE:-docker}"
  -with-trace "$OUT/trace.txt"
  -work-dir "$RESULTS/work_$METHOD")
if [[ -n "${ANHIR_SITE_CONFIG:-}" ]]; then
  args+=(-c "$ANHIR_SITE_CONFIG")
else
  [[ -n "${ANHIR_MAX_CPUS:-}"   ]] && args+=(--max_cpus "$ANHIR_MAX_CPUS")
  [[ -n "${ANHIR_MAX_MEMORY:-}" ]] && args+=(--max_memory "$ANHIR_MAX_MEMORY")
fi
args+=("$@")

echo "nextflow ${args[*]}"
nextflow "${args[@]}"
status=$?
echo
echo "Next: python -m benchmarks.anhir.warp --method $METHOD --pairs $PREP/pairs_manifest.csv \\"
echo "        --outdir $OUT --trace $OUT/trace.txt --out $RESULTS/${METHOD}_warped"
exit $status
