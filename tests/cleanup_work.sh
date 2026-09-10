#!/usr/bin/env bash
# Work-directory cleanup and final-outputs-only publishing, end to end.
#
# WHY THIS IS A SCRIPT AND NOT A pytest OR AN nf-test
#
# `cleanup` is a Nextflow SESSION-TEARDOWN behaviour. It is not in the DAG, so no
# static guard over the sources can see it, and it fires after the pipeline has
# exited, so an nf-test's assertion context cannot observe it either -- nf-test
# hashes output files after teardown, which is exactly why conf/test.config has to
# pin `cleanup_work = false` for the rest of the suite to work at all. The only way
# to check this is to run the pipeline and look at the filesystem afterwards.
#
# Four properties, and the fourth is the one that matters most: cleanup must NEVER
# fire on a failed run, or a real failure becomes undiagnosable.
#
# WHAT "CLEANUP" ACTUALLY DOES -- measured, not assumed. Nextflow empties the task
# directories; it does not remove the tree. On the stub dataset a successful run
# goes from 404 files to 10 (collect-file scratch and work/tmp), with the empty
# two-character shells left behind. So case 1 counts FILES. An assertion on
# directories would fail against correct behaviour, which is the worst kind of
# guard to write.
#
# The test profile pins BOTH cleanup params (see conf/test.config), so this script
# has to set the pair it is about: cleanup_work=true (the shipped default) and
# cleanup_level=final (the CLEANING level -- the shipped default has been 'none'
# since 2026-09-10; 'final' is what cases 2-3c assert on). Via -params-file, never
# on the command line: Nextflow 26 delivers every CLI --param as a String, so a
# boolean passed that way is rejected by the schema as "[string] but should be
# [boolean]". docs/usage.md documents this for every boolean param, and
# tests/test_no_cli_boolean_params_in_docs.py enforces it across the repo.
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
# A published intermediate lives at exactly <outdir>/<patient>/<kind> -- the same
# position tests/checkpoint_manifest.nf.test asserts on. The scan is pinned to that
# depth deliberately: an unbounded `-name` match also hits
# <patient>/registered/transform/preprocessed/data/, which is VALIS's OWN internal
# directory layout inside the registrar-pickle transform artifact
# (conf/modules.config:450-455). That artifact is published UNGATED on purpose -- it
# is the whole transform for the VALIS backend, the analogue of TILED_SOLVE's
# manifest.json -- so matching it here reported a leak that does not exist, on every
# Nextflow version (verified 25.04.7 and 26.04.6, 2026-08-29).
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
# 3c. And no EMPTY directory is left behind. publishDir creates its target
#     directory before saveAs decides whether anything lands there, so a cleaning
#     level used to leave seven empty intermediate directories per patient
#     (converted/, preprocessed/, ... registered/summary/) -- measured 2026-09-09.
#     main.nf's onComplete prunes them; this is the only place that can see it.
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
