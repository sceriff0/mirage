#!/usr/bin/env bash
# Which arm runs did given head jobs launch? Sourced by submit_arms.sh (REDO_LAUNCHED_BY /
# REDO_SINCE). Answered from the RESULTS ROOT, not from the head jobs' logs: a log can be
# deleted (it was, 2026-09-29), while every launch leaves a line in its launch dir's
# .nextflow/history -- `<timestamp>\t<duration>\tarms-<run_id>[-rN]\t<status>...` -- and
# SLURM's accounting keeps each job's start and end.
#
#   job_window <jobid>...                 -> "START\tEND" (earliest start, latest end; a job
#                                            still running ends "now"), history's time format
#   launched_between <root> <since> [until] -> the run ids launched in that window, one per line
#
# A run launched in the window by ANOTHER head that overlapped it is listed too. That is
# the safe side: it ran while the collision did.
# Guarded by benchmarks/tests/test_redo_window.py.

job_window() {
  local out
  out=$(sacct -n -X -P -j "$(IFS=,; echo "$*")" --format=Start,End 2>/dev/null) || return 1
  [[ -n "$out" ]] || return 1
  awk -F'|' -v now="$(date '+%Y-%m-%dT%H:%M:%S')" '
    $1 == "" || $1 == "Unknown" || $1 == "None" { next }
    { s = $1; e = $2
      if (e == "" || e == "Unknown" || e == "None") e = now
      if (min == "" || s < min) min = s
      if (max == "" || e > max) max = e }
    END { if (min == "") exit 1
          sub("T", " ", min); sub("T", " ", max); print min "\t" max }' <<< "$out"
}

launched_between() {
  local root="$1" since="$2" until="${3:-9999-12-31 23:59:59}" h
  for h in "$root"/.launch/*/.nextflow/history; do
    [[ -f "$h" ]] && cat "$h"
  done | awk -F'\t' -v s="$since" -v u="$until" '$1 >= s && $1 <= u && $3 ~ /^arms-/ { print $3 }' \
       | sed -E 's/^arms-//; s/-r[0-9]+$//' | sort -u
}
