#!/usr/bin/env bash
# Work-directory cleanup and final-outputs-only publishing, end to end.
#
# A script, not a pytest or nf-test: `cleanup` runs at session teardown, outside
# the DAG and after nf-test's assertion context has gone (which is also why
# conf/test.config pins `cleanup_work = false`). Only a run shows it.
#
# Cleanup must never fire on a failed run (case 4), or a failure is undiagnosable.
# Nextflow empties task directories but leaves the tree, so case 1 counts files,
# not directories.
#
# The test profile pins both cleanup params, so this sets the pair under test
# (cleanup_work=true, cleanup_level=final) via -params-file: Nextflow 26 delivers
# every CLI --param as a String, which the schema rejects for a boolean.
#
# Usage: bash tests/cleanup_work.sh
set -uo pipefail

cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

NF="${NEXTFLOW:-nextflow}"

cat > "$TMP/cleaning_level.json" <<'JSON'
{
  "cleanup_work": true,
  "cleanup_level": "final"
}
JSON

fail() { echo "FAIL: $*"; exit 1; }

# --------------------------------------------------------------------------
# 1. A SUCCESSFUL run empties the work directory.
# --------------------------------------------------------------------------
W="$TMP/w1"; O="$TMP/o1"
"$NF" -q run . -profile test -stub \
    -params-file "$TMP/cleaning_level.json" \
    -w "$W" --outdir "$O" > "$TMP/run1.log" 2>&1 \
    || { cat "$TMP/run1.log"; fail "the baseline run did not succeed"; }

remaining=$(find "$W" -type f 2>/dev/null | wc -l | tr -d ' ')
# Not zero: collect-file scratch and work/tmp survive teardown by design. The
# threshold is "the task outputs are gone", and an uncleaned run leaves hundreds.
[ "$remaining" -lt 50 ] || fail "work dir still holds $remaining files after a successful run"
echo "ok: successful run left $remaining file(s) in work/"

# --------------------------------------------------------------------------
# 2. Final artifacts survive.
# --------------------------------------------------------------------------
for kind in pyramid geojson quantification; do
    find "$O" -type d -name "$kind" | grep -q . \
      || fail "final artifact '$kind' is missing"
done
find "$O" -type d -name 'qc' | grep -q . || fail "the QC tree is missing"
echo "ok: pyramid/ geojson/ quantification/ qc/ all present"

# --------------------------------------------------------------------------
# 3. Intermediates were never written.
# --------------------------------------------------------------------------
# Scan only <outdir>/<patient>/<kind>: an unbounded `-name` also matches VALIS's
# own preprocessed/data/ inside the ungated registrar-pickle transform artifact.
for kind in converted preprocessed split_channels quantify cell_properties segmentation; do
    while read -r d; do
        [ -z "$d" ] && continue
        if [ -n "$(find "$d" -type f 2>/dev/null | head -1)" ]; then
            fail "intermediate '$kind' was published at --cleanup_level=final ($d)"
        fi
    done < <(find "$O" -mindepth 2 -maxdepth 2 -type d -name "$kind" 2>/dev/null)
done
echo "ok: no intermediate was published"

# --------------------------------------------------------------------------
# 3b. And the checkpoint directory says why it is empty.
# --------------------------------------------------------------------------
[ -f "$O/csv/README.txt" ] || fail "csv/README.txt was not written"
grep -q -- '--cleanup_level none' "$O/csv/README.txt" \
    || fail "csv/README.txt does not name the flag that restores the manifests"
[ -z "$(find "$O/csv" -name '*.csv' 2>/dev/null)" ] \
    || fail "a checkpoint manifest was written at --cleanup_level=final"
echo "ok: csv/ holds only the README"

# --------------------------------------------------------------------------
# 3c. And no empty directory is left behind. publishDir creates its target
#     before saveAs decides whether anything lands there; main.nf's onComplete
#     prunes the empties, and only a run can see that.
# --------------------------------------------------------------------------
empties=$(find "$O" -type d -empty 2>/dev/null | wc -l | tr -d ' ')
if [ "$empties" -ne 0 ]; then
    find "$O" -type d -empty
    fail "$empties empty director(ies) left under --outdir at --cleanup_level=final"
fi
echo "ok: no empty directory left behind"

# --------------------------------------------------------------------------
# 4. A FAILED run KEEPS its work directory. The evidence must survive.
# --------------------------------------------------------------------------
W2="$TMP/w2"; O2="$TMP/o2"
mkdir -p "$W2"
cat > "$TMP/fail.config" <<'CFG'
process { withName: 'CONVERT_IMAGE' { beforeScript = 'exit 1' } }
CFG
"$NF" -q run . -profile test -stub \
    -params-file "$TMP/cleaning_level.json" \
    -c "$TMP/fail.config" -w "$W2" --outdir "$O2" > "$TMP/run2.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "the forced-failure run exited 0"
kept=$(find "$W2" -type f 2>/dev/null | wc -l | tr -d ' ')
[ "$kept" -gt 0 ] || fail "work dir was emptied after a FAILED run -- the evidence is gone"
echo "ok: failed run (exit $rc) kept $kept file(s) in work/"

echo "PASS"
