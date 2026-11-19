#!/usr/bin/env bash
# Behavioural counterpart to tests/test_resume_determinism.py: does an identical
# rerun actually reuse the cache?
#
# Not in the blocking gate: it measures an outcome, and an arrival-order regression
# can be intermittent, so a pass proves little. tests/test_resume_determinism.py
# checks the ordering mechanism instead. Run this after touching a fan-in, a
# collectFile, or anything a process script reads.
#
# Expected: every resume re-runs only GENERATE_QC_REPORT and AGGREGATE_SIZE_LOGS,
# both `cache = false` because their inputs are collectFile() results, rewritten to
# a fresh work/tmp/<hash>/ every run.
#
# Usage: tests/resume_check.sh [runs]     (default 3)

cd "$(dirname "$0")/.."

RUNS="${1:-3}"
WORK="$(mktemp -d)/w"
OUT="$(mktemp -d)/out"
EXPECTED_UNCACHEABLE=("GENERATE_QC_REPORT" "AGGREGATE_SIZE_LOGS")

cleanup() { rm -rf "$(dirname "$WORK")" "$(dirname "$OUT")"; }
trap cleanup EXIT

echo "baseline run..."
# A failed baseline must stop here: an empty log would make the checks below vacuous.
nextflow -q run . -profile test -stub -w "$WORK" --outdir "$OUT" >/dev/null \
    || { echo "FAIL: baseline run did not complete" >&2; exit 1; }

status=0
for i in $(seq 1 "$RUNS"); do
    log=$(nextflow run . -profile test -stub -w "$WORK" --outdir "$OUT" -resume -ansi-log false 2>&1) \
        || { echo "FAIL: resume $i did not complete" >&2; echo "$log" >&2; exit 1; }
    # `grep -c` exits 1 on zero matches, which is data here, not an error.
    cached=$(grep -c 'Cached process'   <<<"$log") || cached=0
    rerun=$(grep -c 'Submitted process' <<<"$log") || rerun=0
    names=$(grep 'Submitted process' <<<"$log" | sed 's/.*> //' | sed 's/ (.*//' | sed 's/.*://' | sort -u | tr '\n' ' ')
    echo "resume $i: CACHED=$cached RESUBMITTED=$rerun  [$names]"

    # The stub run always has cacheable processes, so CACHED=0 means the cache was
    # never consulted (a fresh -w, or a hash that changes every run).
    if [[ "$cached" -eq 0 ]]; then
        echo "  FAIL: resume $i cached nothing (CACHED=0). A successful resume of" >&2
        echo "        this pipeline can never report CACHED=0 -- either the work" >&2
        echo "        directory was not actually reused, or every task's hash changed." >&2
        status=1
    fi

    for n in $names; do
        if [[ ! " ${EXPECTED_UNCACHEABLE[*]} " =~ [[:space:]]${n}[[:space:]] ]]; then
            echo "  FAIL: $n re-ran but is not one of the declared-uncacheable reports." >&2
            echo "        An identical rerun must reuse it. Something reaching it is ordered by" >&2
            echo "        arrival time, or its script references a value that changes per run." >&2
            echo "        Diagnose with: nextflow run ... -resume -dump-hashes json, twice, and diff" >&2
            echo "        that process's entries." >&2
            status=1
        fi
    done
done

[[ $status -eq 0 ]] && echo "OK: only the declared-uncacheable reports re-ran."
exit $status
