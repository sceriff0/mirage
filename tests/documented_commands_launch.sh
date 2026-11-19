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
# `_verdict()` in the Python file is a STATIC check: it looks for
# `--outdir`/`-c`/`max_cpus`/`max_memory` substrings and sizing-pinning profile
# names in the command TEXT. It never invokes Nextflow, so it cannot see
# anything that only surfaces at real launch validation -- an unrecognised
# flag, a samplesheet that does not exist, a column the schema rejects.
# Measured gap: `nextflow run . --nonexistent_param 1 --outdir results -c
# site.config` reads as clean to the static check (it has --outdir and -c) but
# nf-schema refuses it outright. Closing that gap needs an actual `nextflow`
# invocation per command, which the Python guard's fast in-process assertions
# deliberately are not -- that is also why this is a script and not a pytest:
# a real `nextflow run`, even under `-stub`, is a subprocess with its own JVM
# startup, not something that belongs in the pytest collection budget.
#
# WHAT MAKES THIS AFFORDABLE: `--dry_run`. Per CLAUDE.md, "everything runs
# before any process is instantiated, and --dry_run exercises exactly that
# surface" -- so every command below validates and exits in a few seconds with
# no container ever pulled and no process ever submitted, regardless of which
# container/executor profile the documented command itself chose (`docker`,
# `slurm,singularity`, ...). Confirmed by inspecting the resulting work
# directory after a passing run: 0 files.
#
# `dry_run` is a BOOLEAN, so it comes from `-params-file params/dry_run.json`,
# never `--dry_run` on the CLI -- Nextflow 26 delivers every CLI --param as a
# String, and a boolean arrives as "true" rather than true and fails schema
# validation for a reason that has nothing to do with what is under test here.
#
# WHERE THE COMMAND LIST COMES FROM: `python3 -m
# tests.test_documented_commands_are_runnable --emit`, i.e. the SAME
# `_commands()` extractor the static guard uses -- no second regex parses the
# docs. That module also owns the substitution table (`_executable()`) that
# turns a documented command into one this repo's own fixtures can satisfy:
# `--input` becomes the fixture samplesheet matching the command's `--start`,
# a placeholder `-profile
# <profile>`/`<site>` becomes `-profile test`, and `--outdir`/`-c` become the
# sentinels `__OUTDIR__`/`__SITE_CONFIG__` this script fills in with tmp paths
# it controls. A command whose doc text marks it illustrative (a `# not-runnable:
# <reason>` comment on the line directly above it -- the one convention this
# repo defines, because none existed before this script) is SKIPPED, not run:
# docs/parameters.md's CSE example ends in a literal `...` eliding more flags,
# and is prose, not a paste-ready command.
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
