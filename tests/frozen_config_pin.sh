#!/usr/bin/env bash
# A `-c` pin of a frozen config param is REFUSED at launch; the same pin via
# -params-file or the CLI ARRIVES.
#
# WHY THIS IS A SCRIPT AND NOT A pytest OR AN nf-test
#
# tests/test_frozen_config_params.py pins where nextflow.config's params-derived
# scalars sit and that ParamUtils.validateFrozenConfig names each of them. Neither
# says the check fires, or that the working routes actually change the setting:
# `nextflow config` prints the pinned PARAM even when the scalar it drives kept the
# old value (that is the whole defect), and nf-test delivers its own params through
# -params-file, so it can only ever exercise the route that works. Only a run with
# the `-c` route can show the refusal.
#
# Measured 2026-09-09 on dev @ 3f482063 (Nextflow 25.04.7), before the fix: a
# `params { concurrency = 7 }` pin via `-c` left executor.queueSize at 20 and every
# maxForks at 5 while params.concurrency printed 7; via -params-file and the CLI
# both moved. tests/test_frozen_config_params.py's docstring has the full table.
#
# Three legs:
#   1. `-c pin.config` setting concurrency=7, enable_trace=false -> the run must FAIL
#      at launch, and the error must name the validator and the param.
#   2. `-params-file pin.json`, same values plus trace_dir -> the run must SUCCEED,
#      the local executor's monitor line must show capacity=28 (7 * 4), and no
#      trace.txt may be written at the pinned trace_dir (enable_trace=false).
#   3. `--concurrency 7` on the CLI -> succeed, capacity=28.
#
# The `capacity=` needle is the local executor's own report of executor.queueSize
# (`Creating local task monitor for executor 'local' > ... capacity=N`), read from a
# `-log` file this script owns, so it does not depend on the launch dir's log.
#
# Usage: bash tests/frozen_config_pin.sh
set -uo pipefail

cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

NF="${NEXTFLOW:-nextflow}"

fail() { echo "FAIL: $*"; exit 1; }

cat > "$TMP/pin.config" <<'CFG'
params {
    concurrency  = 7
    enable_trace = false
}
CFG

cat > "$TMP/pin.json" <<JSON
{
  "concurrency": 7,
  "enable_trace": false,
  "trace_dir": "$TMP/trace2",
  "cleanup_work": false
}
JSON

capacity_of() {  # capacity_of <nextflow log>
    grep -o 'capacity=[0-9]*' "$1" | head -1 | cut -d= -f2
}

# --------------------------------------------------------------------------
# 1. The `-c` route is refused, naming the validator and the param.
# --------------------------------------------------------------------------
"$NF" -q -log "$TMP/run1.nflog" run . -profile test -stub \
    -c "$TMP/pin.config" -w "$TMP/w1" --outdir "$TMP/o1" > "$TMP/run1.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "a -c pin of concurrency/enable_trace was accepted: the frozen scalars were silently ignored"
grep -q 'validateFrozenConfig' "$TMP/run1.log" \
    || { cat "$TMP/run1.log"; fail "the -c run failed (exit $rc) but not on validateFrozenConfig"; }
grep -q 'executor.queueSize = 20' "$TMP/run1.log" \
    || { cat "$TMP/run1.log"; fail "the refusal does not report the frozen queueSize (20) against the pinned concurrency"; }
grep -q 'trace.enabled = true' "$TMP/run1.log" \
    || { cat "$TMP/run1.log"; fail "the refusal does not report the frozen trace.enabled against the pinned enable_trace"; }
echo "ok: -c pin refused at launch (exit $rc), naming validateFrozenConfig, queueSize and trace.enabled"

# --------------------------------------------------------------------------
# 2. The -params-file route arrives: capacity=28, no trace written.
# --------------------------------------------------------------------------
"$NF" -q -log "$TMP/run2.nflog" run . -profile test -stub \
    -params-file "$TMP/pin.json" -w "$TMP/w2" --outdir "$TMP/o2" > "$TMP/run2.log" 2>&1 \
    || { cat "$TMP/run2.log"; fail "the -params-file run did not succeed"; }
cap=$(capacity_of "$TMP/run2.nflog")
[ "$cap" = "28" ] || fail "-params-file concurrency=7 should give executor.queueSize 28, local executor reports capacity=${cap:-<none>}"
[ ! -e "$TMP/trace2/trace.txt" ] || fail "-params-file enable_trace=false but trace.txt was written under the pinned trace_dir"
echo "ok: -params-file pin arrived (capacity=$cap, no trace written)"

# --------------------------------------------------------------------------
# 3. The CLI route arrives: capacity=28.
# --------------------------------------------------------------------------
"$NF" -q -log "$TMP/run3.nflog" run . -profile test -stub \
    --concurrency 7 -w "$TMP/w3" --outdir "$TMP/o3" > "$TMP/run3.log" 2>&1 \
    || { cat "$TMP/run3.log"; fail "the CLI run did not succeed"; }
cap=$(capacity_of "$TMP/run3.nflog")
[ "$cap" = "28" ] || fail "--concurrency 7 should give executor.queueSize 28, local executor reports capacity=${cap:-<none>}"
echo "ok: CLI pin arrived (capacity=$cap)"

echo "PASS"
