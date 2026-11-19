#!/usr/bin/env bash
# Profile parse guard: a committed profile must never `includeConfig` a
# gitignored or possibly-missing file (e.g. conf/ieo.config). Nextflow evaluates
# every profile's includes at parse time, so one missing include breaks all
# profiles on CI and fresh clones. This moves conf/ieo.config aside and asserts
# every committed profile still parses (`nextflow config` exits 0).
#
# Usage: tests/check_profiles_parse.sh [comma-separated-profiles]

PROFILES="${1:-test,shipped_defaults_test,ieo,singularity,docker,slurm,local}"

moved=0
if [ -f conf/ieo.config ]; then
    mv conf/ieo.config /tmp/ieo.config.guard.bak
    moved=1
fi
# An `if`, not `A && mv || true`: the `|| true` would also hide a failed restore.
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
