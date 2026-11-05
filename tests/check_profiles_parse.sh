#!/usr/bin/env bash
# ============================================================================
# Profile parse guard
# ============================================================================
# Prevents the strict-parser regression that took CI down twice (see Phase 1
# of the hardening campaign): a committed profile must NEVER `includeConfig` a
# gitignored / possibly-missing file (e.g. conf/ieo.config). The latest
# Nextflow parser eagerly evaluates EVERY profile's includes at parse time, so
# one missing include breaks ALL profiles in CI and fresh clones.
#
# This guard temporarily removes conf/ieo.config (the gitignored site overlay)
# and asserts that every committed profile still parses (`nextflow config`
# exits 0). Run it in CI before the stub tests.
#
# Usage: tests/check_profiles_parse.sh [comma-separated-profiles]
# ============================================================================

PROFILES="${1:-test,shipped_defaults_test,ieo,singularity,docker,conda,slurm,local}"

moved=0
if [ -f conf/ieo.config ]; then
    mv conf/ieo.config /tmp/ieo.config.guard.bak
    moved=1
fi
# NOT `[ "$moved" = 1 ] && mv ... || true`. That was the classic `A && B || C`
# trap: the `|| true` existed only to stop the `&&` returning 1 when the file was
# never moved, and it swallowed a FAILED `mv` at the same time -- which would
# leave the developer's conf/ieo.config sitting in /tmp with the guard reporting
# success. The `if` says the same thing and hides nothing.
restore() {
    if [ "$moved" = 1 ]; then
        mv /tmp/ieo.config.guard.bak conf/ieo.config
    fi
}
trap restore EXIT

echo "Profile parse guard (conf/ieo.config temporarily absent)"
echo "Profiles: ${PROFILES}"
echo "------------------------------------------------------------"

fail=0
IFS=',' read -ra PS <<< "$PROFILES"
for p in "${PS[@]}"; do
    if nextflow config -profile "$p" >/dev/null 2>"/tmp/parse_${p}.err"; then
        echo "  ✓ profile '${p}' parses"
    else
        echo "  ✗ profile '${p}' FAILED to parse:"
        sed 's/^/      /' "/tmp/parse_${p}.err" | tail -5
        fail=1
    fi
done

echo "------------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
    echo "FAIL: at least one profile does not parse with conf/ieo.config absent."
    echo "      Do NOT includeConfig a gitignored file inside a committed profile."
    exit 1
fi
echo "PASS: all profiles parse with conf/ieo.config absent."
