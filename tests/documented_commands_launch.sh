#!/usr/bin/env bash
# Execution counterpart to tests/test_documented_commands_are_runnable.py's static
# token check, which never invokes Nextflow and so cannot see what only launch
# validation catches (an unknown flag, a missing samplesheet, a rejected column).
# A script, not a pytest: each command is a real `nextflow run` with its own JVM.
#
# `dry_run` makes it cheap: validation runs before any process is instantiated, so
# each command exits in seconds with no container pulled, whatever profile it names.
# `dry_run` is a boolean, so it comes from `-params-file params/dry_run.json`
# (Nextflow 26 delivers every CLI --param as a String).
#
# Commands come from `python3 -m tests.test_documented_commands_are_runnable --emit`,
# the same extractor the static guard uses. That module also substitutes fixture
# inputs and the `__OUTDIR__`/`__SITE_CONFIG__` sentinels filled in below. A command
# marked by a `# not-runnable: <reason>` comment on the line above it is skipped.
#
# Usage: bash tests/documented_commands_launch.sh
set -euo pipefail

cd "$(dirname "$0")/.."

NF="${NEXTFLOW:-nextflow}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SITE_CONFIG="$TMP/site.config"
cp conf/site.config.template "$SITE_CONFIG"

start_time=$(date +%s)
total=0
skipped=0
launched=0

while IFS=$'\t' read -r doc_file doc_line reason command; do
    total=$((total + 1))

    # "-" is the emitter's NO_REASON sentinel: bash's read collapses an empty
    # field between two tabs and shifts every field after it.
    if [[ "$reason" != "-" ]]; then
        echo "SKIP  ${doc_file}:${doc_line}  ($reason)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -z "$command" ]]; then
        echo "FAIL: ${doc_file}:${doc_line} produced no executable command -- the emitter is broken" >&2
        exit 1
    fi

    outdir="$TMP/out_${total}"
    workdir="$TMP/work_${total}"
    cmd="${command//__OUTDIR__/$outdir}"
    cmd="${cmd//__SITE_CONFIG__/$SITE_CONFIG}"
    cmd="${cmd/nextflow /$NF }"
    cmd="$cmd -w $workdir"

    log="$TMP/log_${total}.txt"
    if bash -c "$cmd" >"$log" 2>&1; then
        launched=$((launched + 1))
        echo "OK    ${doc_file}:${doc_line}"
    else
        echo "FAIL: ${doc_file}:${doc_line} did not pass launch validation" >&2
        echo "  command: $cmd" >&2
        echo "  --- nextflow output ---" >&2
        cat "$log" >&2
        exit 1
    fi
done < <(python3 -m tests.test_documented_commands_are_runnable --emit)

# If `--emit` printed nothing, the loop never ran and this would pass vacuously.
# Same floor as test_every_documented_command_would_launch.
if [[ "$total" -lt 25 ]]; then
    echo "FAIL: only $total command(s) were extracted -- the emitter has stopped matching and this scan read almost nothing" >&2
    exit 1
fi

elapsed=$(( $(date +%s) - start_time ))
echo "PASS: ${launched} launched, ${skipped} skipped, ${total} total documented commands in ${elapsed}s"
