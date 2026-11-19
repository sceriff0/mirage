#!/usr/bin/env bash
# `cleanup_level` must reach the publish gates by every route a param can arrive.
#
# A script, not a pytest or nf-test: a gate computed while conf/modules.config is
# parsed freezes before `profiles {}` and any `-c` file merge, yet `nextflow config`
# still prints the pinned value. Only a run shows the difference.
#
# Each leg checks whether the `preprocessed/` intermediate holds a file (no file
# counts; they drift with every added artifact):
#   1. `-profile test` alone: conf/test.config pins 'none', so it is published.
#   2. `-c pin.config` setting 'none': the same, by the site-config route.
#   3. `-c pin.config` setting 'final' over the profile's 'none': not published,
#      because a `-c` file outranks a profile on params.
#
# Usage: bash tests/cleanup_level_pin.sh
set -uo pipefail

cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

NF="${NEXTFLOW:-nextflow}"

fail() { echo "FAIL: $*"; exit 1; }

# Only <outdir>/<patient>/preprocessed/: an unbounded -name also matches VALIS's
# own `preprocessed/data/` inside the ungated transform artifact.
intermediate_files() {
    local out="$1"
    find "$out" -mindepth 2 -maxdepth 2 -type d -name preprocessed 2>/dev/null \
      | while read -r d; do find "$d" -type f; done | wc -l | tr -d ' '
}

run_stub() {  # run_stub <label> <workdir> <outdir> [extra nextflow args...]
    local label="$1" w="$2" o="$3"; shift 3
    "$NF" -q run . -profile test -stub "$@" -w "$w" --outdir "$o" > "$TMP/$label.log" 2>&1 \
      || { cat "$TMP/$label.log"; fail "the '$label' run did not succeed"; }
}

# --------------------------------------------------------------------------
# 1. The test profile's own pin ('none') publishes the intermediate.
# --------------------------------------------------------------------------
run_stub profile "$TMP/w1" "$TMP/o1"
n=$(intermediate_files "$TMP/o1")
[ "$n" -gt 0 ] || fail "-profile test pins cleanup_level='none' but preprocessed/ holds no file: the gate did not see the profile"
echo "ok: -profile test (pins 'none') published $n preprocessed file(s)"

# --------------------------------------------------------------------------
# 2. A -c site-config pin ('none') publishes the intermediate.
# --------------------------------------------------------------------------
printf "params { cleanup_level = 'none' }\n" > "$TMP/pin_none.config"
run_stub c_none "$TMP/w2" "$TMP/o2" -c "$TMP/pin_none.config"
n=$(intermediate_files "$TMP/o2")
[ "$n" -gt 0 ] || fail "-c site.config pins cleanup_level='none' but preprocessed/ holds no file: the gate did not see the -c file"
echo "ok: -c pin ('none') published $n preprocessed file(s)"

# --------------------------------------------------------------------------
# 3. A -c pin ('final') over the profile's 'none' withholds the intermediate.
# --------------------------------------------------------------------------
printf "params { cleanup_level = 'final' }\n" > "$TMP/pin_final.config"
run_stub c_final "$TMP/w3" "$TMP/o3" -c "$TMP/pin_final.config"
n=$(intermediate_files "$TMP/o3")
[ "$n" -eq 0 ] || fail "-c site.config pins cleanup_level='final' over the profile's 'none' but preprocessed/ holds $n file(s): the gate followed the wrong layer"
echo "ok: -c pin ('final') over the profile published no preprocessed file"

echo "PASS"
