#!/usr/bin/env bash
# Keep the tables and records of old regbench / arms / sweep results, free everything else.
#
#   benchmarks/regbench/keep_results.sh --to KEEP \
#       [--regbench OUT]... [--arms ROOT] [--sweep ROOT --sweep-plan PLAN] [--cases CASES] \
#       [--apply]
#
# Run it from the mirage checkout on the login node (it sources slurm/env.sh, so the
# REGBENCH_SIF_* you launched with are used; with none set, the `python` on PATH).
#
#   1 collect  every paper table from those results      -> KEEP/tables   (regbench collect)
#   2 plots    draft figures from KEEP/tables             -> KEEP/figures  (regbench plots)
#   3 copy     every small record of each root, tree kept -> KEEP/raw/<name>/
#              (*.csv *.tsv *.json *.txt *.yml *.yaml *.log *.html, Nextflow work/ skipped),
#              then checked: every file there with the same bytes, or it stops here
#   4 slim     each root (regbench slim):
#                regbench roots at level `images`: written images and work go; the maps
#                  (warped.npz), run.json, records and eval planes stay, so it can be
#                  RE-SCORED with the new scorer later
#                arms / sweep roots at level `csv`: only CSVs, run.json, *_seg_qc.json,
#                  traces and size_logs/ stay (nothing re-scores them; KEEP/raw has the rest)
#              LEVEL=images or LEVEL=csv sets one level for every root.
#
# Without --apply NOTHING is deleted: steps 1-3 only write under KEEP, and step 4 prints what
# it would free. Read that, then run the same command again with --apply.
# Every step's output is also in KEEP/keep_results.log; KEEP/README.txt says what was kept.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec 3>&2  # the terminal's stderr, kept: a refusal reaches it even while output goes to the log
LOG=""
die() {
    echo "keep_results: $*" >&3
    if [ -n "$LOG" ]; then echo "keep_results: $*" >> "$LOG"; fi
    exit 1
}

KEEP="" ARMS="" SWEEP="" SWEEP_PLAN="" CASES="" APPLY=0
REGBENCH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --to) KEEP="${2:?--to needs a folder}"; shift 2 ;;
        --regbench) REGBENCH+=("${2:?--regbench needs a folder}"); shift 2 ;;
        --arms) ARMS="${2:?--arms needs a folder}"; shift 2 ;;
        --sweep) SWEEP="${2:?--sweep needs a folder}"; shift 2 ;;
        --sweep-plan) SWEEP_PLAN="${2:?--sweep-plan needs a file}"; shift 2 ;;
        --cases) CASES="${2:?--cases needs a folder}"; shift 2 ;;
        --apply) APPLY=1; shift ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) die "unknown argument '$1' (--help)" ;;
    esac
done
[ -n "$KEEP" ] || die "--to KEEP is required (--help)"
ROOTS=("${REGBENCH[@]+"${REGBENCH[@]}"}")
[ -n "$ARMS" ] && ROOTS+=("$ARMS")
[ -n "$SWEEP" ] && ROOTS+=("$SWEEP")
[ ${#ROOTS[@]} -gt 0 ] || die "name at least one --regbench, --arms or --sweep folder"
[ -z "$SWEEP" ] || [ -n "$SWEEP_PLAN" ] || die "--sweep needs --sweep-plan (the run plan CSV)"
case "${LEVEL:-}" in ""|images|csv) ;; *) die "LEVEL='$LEVEL': images or csv" ;; esac

mkdir -p "$KEEP"
KEEP="$(cd "$KEEP" && pwd)"
for r in "${ROOTS[@]}"; do
    [ -d "$r" ] || die "$r: not a folder"
    abs="$(cd "$r" && pwd)"
    case "$KEEP/" in "$abs"/*) die "--to $KEEP is inside $abs: put it outside every root" ;; esac
done
LOG="$KEEP/keep_results.log"
exec > >(tee -a "$LOG") 2>&1
echo "# keep_results $(date '+%F %T')  apply=$APPLY  to=$KEEP"

# the roots and KEEP must be visible inside the images
binds="$KEEP"
for r in "${ROOTS[@]}" ${CASES:+"$CASES"} ${SWEEP_PLAN:+"$(dirname "$SWEEP_PLAN")"}; do
    binds="$binds,$(cd "$r" 2>/dev/null && pwd || echo "$r")"
done
export REGBENCH_BIND="${REGBENCH_BIND:+$REGBENCH_BIND,}$binds"
# shellcheck source=benchmarks/regbench/slurm/env.sh
source "$HERE/slurm/env.sh"
rb() { regbench_py STARE_REGBENCH "$@"; }

# ── 1 collect ────────────────────────────────────────────────────────────────
echo "== 1/4 collect -> $KEEP/tables"
rm -rf "$KEEP/tables.new"
args=(collect --to "$KEEP/tables.new")
for r in "${REGBENCH[@]+"${REGBENCH[@]}"}"; do args+=(--regbench "$r"); done
[ -n "$ARMS" ] && args+=(--arms "$ARMS")
[ -n "$SWEEP" ] && args+=(--sweep "$SWEEP" --sweep-plan "$SWEEP_PLAN")
[ -n "$CASES" ] && args+=(--cases "$CASES")
rb "${args[@]}" || die "collect failed: nothing was deleted (see $LOG)"
rm -rf "$KEEP/tables" && mv "$KEEP/tables.new" "$KEEP/tables"

# ── 2 plots ──────────────────────────────────────────────────────────────────
echo "== 2/4 draft figures -> $KEEP/figures"
PLOTS=ok
rb plots --tables "$KEEP/tables" --to "$KEEP/figures" || {
    PLOTS=failed
    echo "WARNING: the draft figures failed (the tables are kept; draw them later with" \
         "\`regbench plots --tables $KEEP/tables --to DIR\`). Continuing."
}

# ── 3 copy the small records ─────────────────────────────────────────────────
echo "== 3/4 copy records -> $KEEP/raw/"
list_records() {  # relative paths of every record under $1, Nextflow work/ skipped
    (cd "$1" && find . -type d -name work -prune -o -type f \( \
        -name '*.csv' -o -name '*.tsv' -o -name '*.json' -o -name '*.txt' -o \
        -name '*.yml' -o -name '*.yaml' -o -name '*.log' -o -name '*.html' \) -print \
        | sed 's|^\./||' | LC_ALL=C sort)
}
if stat -c %s / >/dev/null 2>&1; then SIZE=(stat -c %s); else SIZE=(stat -f %z); fi
bytes_of() {  # total bytes, under folder $1, of the relative paths on stdin (missing ones: 0)
    local present
    present="$(cd "$1" && while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done; true)"
    [ -n "$present" ] || { echo 0; return; }
    (cd "$1" && printf '%s\n' "$present" | tr '\n' '\0' | xargs -0 "${SIZE[@]}") \
        | awk '{s+=$1} END {print s+0}'
}
SEEN=" "
RAWS=()
for r in "${ROOTS[@]}"; do
    abs="$(cd "$r" && pwd)"
    name="$(basename "$abs")"; n=2
    while case "$SEEN" in *" $name "*) true ;; *) false ;; esac; do
        name="$(basename "$abs")_$n"; n=$((n + 1))
    done
    SEEN="$SEEN$name "
    dest="$KEEP/raw/$name"; files="$KEEP/raw/$name.files"; mkdir -p "$dest"
    list_records "$abs" > "$files"
    tar -C "$abs" -cf - -T "$files" | tar -C "$dest" -xpf -
    # every source record must be in the copy with the same bytes (a copy from an earlier
    # run may hold more: what an --apply at level csv removed from the root since)
    missing=$(list_records "$dest" | LC_ALL=C comm -23 "$files" - | wc -l | tr -d ' ')
    src_n=$(wc -l < "$files" | tr -d ' ')
    src_b=$(bytes_of "$abs" < "$files")
    dst_b=$(bytes_of "$dest" < "$files")
    echo "$abs -> raw/$name: $src_n files, $src_b bytes in the source, $dst_b in the copy, $missing missing"
    [ "$missing" = 0 ] && [ "$src_b" = "$dst_b" ] \
        || die "the copy of $abs is incomplete: nothing was deleted (see $LOG)"
    RAWS+=("$abs -> raw/$name ($src_n files, $src_b bytes)")
done

# ── 4 slim ───────────────────────────────────────────────────────────────────
echo "== 4/4 slim ($([ "$APPLY" = 1 ] && echo APPLY: deleting || echo dry run: nothing deleted))"
level_of() {
    if [ -n "${LEVEL:-}" ]; then echo "$LEVEL"; return; fi
    for g in "${REGBENCH[@]+"${REGBENCH[@]}"}"; do [ "$g" = "$1" ] && { echo images; return; }; done
    echo csv
}
SLIMS=()
for r in "${ROOTS[@]}"; do
    lv="$(level_of "$r")"
    args=(slim --root "$r" --level "$lv")
    [ "$lv" = csv ] && args+=(--collected "$KEEP/tables")
    [ "$APPLY" = 1 ] && args+=(--apply)
    echo "-- $r (level $lv)"
    if rb "${args[@]}"; then SLIMS+=("$r: level $lv, $([ "$APPLY" = 1 ] && echo "done" || echo "dry run")")
    else SLIMS+=("$r: level $lv, REFUSED (see the message above; nothing deleted there)"); fi
done

{
    echo "keep_results $(date '+%F %T')"
    echo "tables:  tables/  (regbench collect; interim, NA where a measurement is to come)"
    echo "figures: figures/ ($PLOTS)"
    echo "records: raw/<name>/ (every csv tsv json txt yml yaml log html, tree kept)"
    printf '  %s\n' "${RAWS[@]}"
    echo "slim:"
    printf '  %s\n' "${SLIMS[@]}"
    echo "log: keep_results.log"
} > "$KEEP/README.txt"
cat "$KEEP/README.txt"
[ "$APPLY" = 1 ] || echo "Dry run: nothing was deleted. Same command with --apply to free the space."
