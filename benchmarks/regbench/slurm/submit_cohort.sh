#!/bin/bash
# Submit the cohort benchmark: every run of the run list behind its own check, each on a
# four-rung memory ladder, then the scorer. From a login node, in two steps:
#
#   benchmarks/regbench/slurm/submit_cohort.sh --prepare [-- extra sbatch args]
#   (wait for the prepare jobs; look at their logs)
#   benchmarks/regbench/slurm/submit_cohort.sh [--dry-run] [--runs "a b"] \
#       [--datasets "multiplex synthetic"] [-- extra sbatch args]
#
# THE DEFAULT is `multiplex`, then `synthetic`: every array of a run on `synthetic` waits
# for that run's last array on `multiplex`, so the cohort is done first. `semisynth` and
# the HyReCo sets are DORMANT: implemented and tested, never part of the default, and
# submitted only when --datasets names them (the README lists what is known to be wrong
# with each before any use).
#
# --dry-run (or REGBENCH_DRY=1) prints every sbatch command with its dependencies, a job
# name in angle brackets standing for each job id, and submits nothing.
#
# THE RUN LIST is read from `python -m benchmarks.regbench runs` and from nowhere else:
# this script names no method. What it submits, in order:
#
#   1. per run that gates a run: its selfcheck, at 64, then 256, then 512 GB. An attempt
#      above the first runs only when the one before ended for memory. A run whose
#      results another reads (column `after`) hands its selfcheck's work to that one
#      (`--m0-from`), which waits for it.
#   2. per run: a gate job (`regbench gate`). It fails when the selfcheck that gates the
#      run is anything but ok, and then THAT run alone never starts. The rows it says to
#      drop are collected for the scorer.
#   3. per run and dataset: the memory ladder, 16 -> 64 -> 256 -> 512 GB at 4 CPUs, as
#      EIGHT job arrays, two per rung, each waiting for the one before (`afterany`): the
#      attempt at that rung, then `-again`, the same rung once more, which runs a unit
#      only when its attempt was lost to the scheduler (node failure, preemption, a
#      cancelled job, a task that never started). A task of a rung above the first
#      exits at once unless its unit's last run.json names a memory cause. One task is
#      one case for a run that works per pair and one patient for a run that works per
#      patient. A run that reads another's results waits for that run's last array on
#      the same dataset. A unit whose results are all ok already (an earlier
#      submission into the same REGBENCH_OUT) is reused, not run: REGBENCH_FORCE=1.
#   4. for a run whose realcheck is a gate (column `realcheck`): the first unit of the
#      first dataset goes alone, with `--realcheck`; a second gate job reads its
#      realcheck.json; everything else of that run waits for it.
#   5. the tissue-level pieces, then the scorer (`score --drop ...`), both in the
#      scorer's image, after everything.
#
# Nothing waits on the login node: no `sbatch --wait`, no job that submits jobs.
#
# AN EXTERNAL LANDMARK SET (HyReCo) is submitted on its own, never with the cohort:
#
#   submit_cohort.sh --prepare --datasets hyreco_a      (then the same without --prepare)
#   submit_cohort.sh --prepare --datasets hyreco_b      (then the same without --prepare)
#
# `hyreco_a` is the TUNING set and `hyreco_b` the REPORT set: subset B is prepared with
# the settings of the prepared subset-A cases and with nothing else. Such a submission
# reads its own run list (`runs --dataset <name>`: only the runs with an input rule for
# brightfield), prepares one pair per array task, submits no tissue job, sends no first
# unit ahead for a realcheck (both are defined on nuclei), and its scorer writes
# $REGBENCH_OUT/tables/<name>/ and none of the cohort's tables. Use a REGBENCH_OUT of
# its own.
#
# ENVIRONMENT (env.sh has the paths and the images)
#   REGBENCH_CASES, REGBENCH_OUT      prepared cases; results
#   REGBENCH_SIF_<image>              one per image of the run list (column `image`)
#   REGBENCH_SBATCH_GPU               REQUIRED for the runs that need a GPU, no default:
#                                     the site's request, e.g. "--gres=gpu:nvidia_h200:1"
#   REGBENCH_TIME (24:00:00)          per round, every rung. A run that works per patient
#                                     gets it times the rounds of the largest patient,
#                                     never above the site's longest job (7 days)
#   REGBENCH_SELFCHECK_TIME (24:00:00)
#   REGBENCH_ARRAY_LIMIT (20)         tasks of one array that run at once (%N)
#   REGBENCH_FORCE=1                  run every unit again, also those already ok
#   REGBENCH_SUBMISSION               this call's id (default: time and process id).
#                                     Records of another submission are evidence of
#                                     nothing: not for a gate, the ladder or the scorer
#   REGBENCH_ALLOW_UNCHECKED          run names that start whatever their checks say
#   REGBENCH_PIPELINE_TRACE           a Nextflow trace.txt of a pipeline run on this
#   REGBENCH_PIPELINE_TRACE_LABEL     cohort, and what to call it: the scorer reads the
#                                     pipeline's convert and merge steps from it
#   REGBENCH_MANIFEST                 with --prepare: the cohort's rounds.csv
#   REGBENCH_TRUTH_PATIENTS           with --prepare: the patients of the known-truth
#                                     cases, comma-separated (default: all)
#   REGBENCH_PREPARE_ARGS             with --prepare: more `prepare multiplex` options
#   REGBENCH_PREPARE_FORCE=1          with --prepare: redo cases that exist
#   REGBENCH_HYRECO_A                 with --prepare --datasets hyreco_a: the unpacked
#                                     HyReCo/ folder (it holds HE/ and PHH3/)
#   REGBENCH_HYRECO_A_ARGS            more `prepare hyreco --subset a` options: the
#                                     settings being tried (--proxy, --reduce)
#   REGBENCH_HYRECO_B_HE, REGBENCH_HYRECO_B_PHH3
#                                     with --prepare --datasets hyreco_b: the HE/ and
#                                     PHH3/ folders of HyReCo-Additional (two archives)
#   REGBENCH_HYRECO_TUNED             the cases folder holding the prepared hyreco_a
#                                     cases subset B takes its settings from (default:
#                                     REGBENCH_CASES)
#   REGBENCH_HYRECO_B_ARGS            more `prepare hyreco --subset b` options (never a
#                                     setting: --proxy and --reduce are refused there)
#   REGBENCH_HEAD_IMAGE               the image this script itself reads the run list
#                                     with (default STARE_REGBENCH)
# The rungs, 4 CPUs and the 7-day limit are frozen (benchmarks/regbench/ladder.py).
# No thread variable is exported anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=benchmarks/regbench/slurm/env.sh
source "$HERE/env.sh"

RUNGS_GB="16 64 256 512"
SELFCHECK_RUNGS_GB="64 256 512"
CPUS=4
MAX_SECONDS=$((7 * 24 * 3600))
: "${REGBENCH_TIME:=24:00:00}"
: "${REGBENCH_SELFCHECK_TIME:=24:00:00}"
: "${REGBENCH_ARRAY_LIMIT:=20}"
: "${REGBENCH_HEAD_IMAGE:=STARE_REGBENCH}"
DRY="${REGBENCH_DRY:-0}"
# multiplex first, then synthetic. semisynth and hyreco_* are dormant: only when named.
DEFAULT_DATASETS="multiplex synthetic"
DORMANT_DATASETS="semisynth hyreco_a hyreco_b"
prepare=0 only_runs="" datasets="$DEFAULT_DATASETS"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --prepare) prepare=1; shift ;;
        --runs) only_runs="$2"; shift 2 ;;
        --datasets) datasets="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
[ "$DRY" = 0 ] || DRY=1

die() { echo "submit_cohort: $*" >&2; exit 2; }
has() { case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac; }

# one id per call: every record of this submission carries it
: "${REGBENCH_SUBMISSION:=$(date +%Y%m%dT%H%M%S)-$$}"
export REGBENCH_SUBMISSION
for ds in $datasets; do
    has "$DORMANT_DATASETS" "$ds" && echo "# $ds is DORMANT: submitted because it was named. Read \"Dormant datasets: known defects before any use\" in benchmarks/regbench/README.md first."
done

# An external landmark set is submitted alone: its own run list, its own tables.
EXTERNAL="" n_named=0
for ds in $datasets; do
    n_named=$((n_named + 1))
    case "$ds" in hyreco_a|hyreco_b) EXTERNAL="$ds" ;; esac
done
if [ -n "$EXTERNAL" ] && [ "$n_named" -ne 1 ]; then
    die "--datasets '$datasets': an external landmark set (hyreco_a, hyreco_b) is submitted alone, with its own run list and its own tables. hyreco_a is the tuning set, hyreco_b the report set. Nothing was submitted."
fi

# seconds <[D-]HH:MM:SS> ; limit <seconds> -> D-HH:MM:SS, never above the site's maximum
seconds() {
    local t="$1" d=0 h m s
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    IFS=: read -r h m s <<< "$t"
    [ -n "${s:-}" ] || die "a time must be [D-]HH:MM:SS, got '$1'"
    echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s ))
}
limit() {
    local s="$1"
    [ "$s" -gt "$MAX_SECONDS" ] && s="$MAX_SECONDS"
    printf '%d-%02d:%02d:%02d' $((s / 86400)) $((s % 86400 / 3600)) $((s % 3600 / 60)) $((s % 60))
}

# submit <job name> <sbatch arguments...>: sets JOB_ID. Dry: prints the command.
N_JOBS=0
JOB_ID=""
JOBS_FILE=""
submit() {
    local name="$1"; shift
    N_JOBS=$((N_JOBS + 1))
    if [ "$DRY" = 1 ]; then
        printf 'sbatch --parsable --job-name=%s' "$name"
        printf ' %s' "$@"
        printf '\n'
        JOB_ID="<$name>"
        return 0
    fi
    local got
    got="$(sbatch --parsable --job-name="$name" "$@" < /dev/null)"
    JOB_ID="${got%%;*}"
    printf '%s\t%s\n' "$name" "$JOB_ID"
    printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$name" "$JOB_ID" >> "$JOBS_FILE"
}
# every job: the caller's own options, logs in one place, and never pending for ever on
# a dependency that can no longer be met
common() {
    COMMON=(--export=ALL --kill-on-invalid-dep=yes)
    [ ${#extra[@]} -gt 0 ] && COMMON+=("${extra[@]}")
    return 0
}
common
LOG_ONE="--output=$REGBENCH_OUT/logs/%x-%j.log"
LOG_ARRAY="--output=$REGBENCH_OUT/logs/%x-%A_%a.log"
round_s="$(seconds "$REGBENCH_TIME")"
sc_s="$(seconds "$REGBENCH_SELFCHECK_TIME")"
prep_s="$(seconds "${REGBENCH_PREPARE_TIME:-24:00:00}")"
# shellcheck disable=SC2206  # the site's GPU request is several words on purpose
GPU_ARGS=(${REGBENCH_SBATCH_GPU:-})

if [ "$DRY" = 1 ]; then
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
else
    mkdir -p "$REGBENCH_OUT/logs/gates" "$REGBENCH_OUT/logs/selfcheck"
    scratch="$REGBENCH_OUT/logs"
    JOBS_FILE="$REGBENCH_OUT/logs/jobs.tsv"
fi

# ── the run list, from the harness alone ──────────────────────────────────────────────
table="$scratch/runs.tsv"
if [ -n "$EXTERNAL" ]; then
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --dataset "$EXTERNAL" > "$table"
else
    regbench_py "$REGBENCH_HEAD_IMAGE" runs > "$table"
fi
[ -s "$table" ] || die "the run list is empty (python -m benchmarks.regbench runs)"
if [ -n "$EXTERNAL" ]; then
    regbench_py "$REGBENCH_HEAD_IMAGE" runs --dataset "$EXTERNAL" --excluded | \
        while IFS=$'\t' read -r name why; do echo "# $EXTERNAL: $name is not run ($why)"; done
fi
REGBENCH_SCORER_IMAGE="$(regbench_py "$REGBENCH_HEAD_IMAGE" runs --scorer-image)"
export REGBENCH_SCORER_IMAGE

all_names="" selected=""
while IFS=$'\t' read -r name _; do all_names="$all_names $name"; done < "$table"
for r in $only_runs; do has "$all_names" "$r" || die "--runs: no run named '$r' (see: python -m benchmarks.regbench runs)"; done
for r in $all_names; do
    if [ -z "$only_runs" ] || has "$only_runs" "$r"; then selected="$selected $r"; fi
done
for r in ${REGBENCH_ALLOW_UNCHECKED:-}; do has "$all_names" "$r" || die "REGBENCH_ALLOW_UNCHECKED: no run named '$r'"; done

# ── prepare, on its own: array sizes are read from what it writes ─────────────────────
if [ "$prepare" = 1 ]; then
    [ -n "$EXTERNAL" ] || [ -n "${REGBENCH_MANIFEST:-}" ] || die "--prepare needs REGBENCH_MANIFEST (the rounds.csv of python -m benchmarks.regbench.arms_manifest)"
    force=()
    [ "${REGBENCH_PREPARE_FORCE:-0}" = 1 ] && force=(--force)
    alloc=(--cpus-per-task="$CPUS" --mem="${REGBENCH_PREPARE_MEM:-64G}"
           --time="$(limit "$prep_s")")
    export REGBENCH_PREPARE_IMAGE="$REGBENCH_SCORER_IMAGE"
    patients=()
    [ -n "${REGBENCH_TRUTH_PATIENTS:-}" ] && patients=(--patients "$REGBENCH_TRUTH_PATIENTS")
    # shellcheck disable=SC2206  # extra `prepare multiplex` options, several words
    more=(${REGBENCH_PREPARE_ARGS:-})
    for ds in $datasets; do
        array=() log="$LOG_ONE"
        case "$ds" in
            multiplex) what=(multiplex --manifest "$REGBENCH_MANIFEST" ${more[@]+"${more[@]}"}) ;;
            semisynth) what=(semisynth --truth --manifest "$REGBENCH_MANIFEST" ${patients[@]+"${patients[@]}"}) ;;
            synthetic) what=(synthetic --suite truth) ;;
            hyreco_a)
                [ -n "${REGBENCH_HYRECO_A:-}" ] || die "--prepare --datasets hyreco_a needs REGBENCH_HYRECO_A (the unpacked HyReCo/ folder, holding HE/ and PHH3/). Nothing was submitted."
                # shellcheck disable=SC2206  # the settings being tried, several words
                tried=(${REGBENCH_HYRECO_A_ARGS:-})
                what=(hyreco --subset a --data-root "$REGBENCH_HYRECO_A" ${tried[@]+"${tried[@]}"}) ;;
            hyreco_b)
                [ -n "${REGBENCH_HYRECO_B_HE:-}" ] && [ -n "${REGBENCH_HYRECO_B_PHH3:-}" ] || die "--prepare --datasets hyreco_b needs REGBENCH_HYRECO_B_HE and REGBENCH_HYRECO_B_PHH3 (the HE/ and PHH3/ folders of HyReCo-Additional). Nothing was submitted."
                # shellcheck disable=SC2206  # several words; a setting is refused by prepare itself
                given=(${REGBENCH_HYRECO_B_ARGS:-})
                what=(hyreco --subset b --ref-dir "$REGBENCH_HYRECO_B_HE" --mov-dir "$REGBENCH_HYRECO_B_PHH3"
                      --settings-from "${REGBENCH_HYRECO_TUNED:-$REGBENCH_CASES}" ${given[@]+"${given[@]}"}) ;;
            *) die "--datasets: '$ds' is not one of multiplex, semisynth, synthetic, hyreco_a, hyreco_b" ;;
        esac
        if [ -n "$EXTERNAL" ]; then
            # one pair per task: a slide is about 21 gigapixels, and its proxy is written here
            n_pairs="$(regbench_py "$REGBENCH_HEAD_IMAGE" prepare hyreco --cases "$REGBENCH_CASES" \
                --subset "${ds#hyreco_}" --count)" || die "cannot size the $ds prepare array"
            array=(--array="0-$((n_pairs - 1))%$REGBENCH_ARRAY_LIMIT")
            log="$LOG_ARRAY"
        fi
        submit "rb-prepare-$ds" "${COMMON[@]}" "$log" "${alloc[@]}" ${array[@]+"${array[@]}"} \
            "$HERE/prepare.sbatch" "${what[@]}" ${force[@]+"${force[@]}"}
    done
    echo "prepare: $N_JOBS job(s). When they are done, run this script again without --prepare."
    exit 0
fi

# ── refusals, before anything is submitted ────────────────────────────────────────────
gpu_runs="" images=""
while IFS=$'\t' read -r name _ _ _ image gpu _; do
    has "$selected" "$name" || continue
    [ "$gpu" = 1 ] && gpu_runs="$gpu_runs $name"
    has "$images" "$image" || images="$images $image"
done < "$table"
if [ -n "$gpu_runs" ] && [ -z "${REGBENCH_SBATCH_GPU:-}" ]; then
    die "these runs need a GPU and REGBENCH_SBATCH_GPU is not set:$gpu_runs. Export the site's request (e.g. REGBENCH_SBATCH_GPU=\"--gres=gpu:nvidia_h200:1\"), or leave them out with --runs. Nothing was submitted."
fi
missing=""
for image in $images $REGBENCH_SCORER_IMAGE; do
    [ -n "$(_regbench_var SIF "$image")$(_regbench_var CONDA "$image")$(_regbench_var VENV "$image")" ] \
        || has "$missing" "REGBENCH_SIF_$image" || missing="$missing REGBENCH_SIF_$image"
done
if [ -n "$missing" ]; then
    if [ "$DRY" = 1 ] || [ "${REGBENCH_ALLOW_PATH_PYTHON:-0}" = 1 ]; then
        echo "# not set:$missing (those jobs would use the python on PATH)"
    else
        die "not set:$missing. Each image of the run list needs its path (or REGBENCH_ALLOW_PATH_PYTHON=1 to use the python on PATH). Nothing was submitted."
    fi
fi

# ── the cases: every array is sized from what is prepared ─────────────────────────────
sized=""
for ds in $datasets; do
    read -r n_cases n_groups max_rounds < <(regbench_py "$REGBENCH_HEAD_IMAGE" list \
        --cases "$REGBENCH_CASES" --dataset "$ds" --sizes) || die "cannot list the $ds cases under $REGBENCH_CASES"
    if [ "${n_cases:-0}" -eq 0 ]; then
        echo "# $ds: no prepared cases under $REGBENCH_CASES, left out (submit_cohort.sh --prepare)"
        continue
    fi
    sized="$sized $ds"
    printf -v "CASES_$ds" '%s' "$n_cases"
    printf -v "GROUPS_$ds" '%s' "$n_groups"
    printf -v "ROUNDS_$ds" '%s' "$max_rounds"
    echo "# $ds: $n_cases case(s), $n_groups patient(s), at most $max_rounds round(s) per patient"
done
[ -n "$sized" ] || die "no prepared cases under $REGBENCH_CASES for any of: $datasets. Run with --prepare first. Nothing was submitted."
# the cohort first, whatever the order they were named in
ordered=""
has "$sized" multiplex && ordered="multiplex"
for ds in $sized; do [ "$ds" = multiplex ] || ordered="$ordered $ds"; done
sized="$ordered"
first_dataset="${sized# }"; first_dataset="${first_dataset%% *}"

# ── 1. selfchecks: the runs that gate a selected run, and the runs they are anchored on
needed=""
while IFS=$'\t' read -r name _ _ _ _ _ _ after gated_by _; do
    has "$selected" "$name" || continue
    has "$needed" "$gated_by" || needed="$needed $gated_by"
done < "$table"
# a gating run that reads another's results needs that one's selfcheck too
while IFS=$'\t' read -r name _ _ _ _ _ _ after _; do
    has "$needed" "$name" || continue
    [ "$after" = - ] || has "$needed" "$after" || needed="$needed $after"
done < "$table"

sc_time="$(limit "$sc_s")"
while IFS=$'\t' read -r name method label opts image gpu _ after _ mask _; do
    has "$needed" "$name" || continue
    gres=()
    [ "$gpu" = 1 ] && gres=(${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
    base="" anchor=-
    if [ "$after" != - ]; then
        anchor="$after"
        last="SC_LAST_$after"
        base="afterok:${!last}"
    fi
    previous=- attempt=0
    for gb in $SELFCHECK_RUNGS_GB; do
        attempt=$((attempt + 1))
        dep="$base"
        [ "$previous" = - ] || dep="${dep:+$dep,}afterany:$previous"
        deps=()
        [ -n "$dep" ] && deps=(--dependency="$dep")
        submit "rb-sc-$name-${gb}g" "${COMMON[@]}" "$LOG_ONE" ${deps[@]+"${deps[@]}"} \
            --cpus-per-task="$CPUS" --mem="${gb}G" --time="$sc_time" ${gres[@]+"${gres[@]}"} \
            "$HERE/selfcheck.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
            "$mask" "$anchor" "$attempt" "$gb" "$previous"
        previous="$JOB_ID"
    done
    printf -v "SC_LAST_$name" '%s' "$previous"
done < "$table"

# ── 2. one gate per run ───────────────────────────────────────────────────────────────
small=(--cpus-per-task=1 --mem=4G --time=0-01:00:00)
while IFS=$'\t' read -r name _ _ _ _ _ _ _ gated_by _; do
    has "$selected" "$name" || continue
    allowed=0
    has "${REGBENCH_ALLOW_UNCHECKED:-}" "$name" && allowed=1
    # an earlier submission's verdict must not be what a task of this one finds
    [ "$DRY" = 1 ] || rm -f "$REGBENCH_OUT/logs/gates/$name.tsv" "$REGBENCH_OUT/logs/gates/$name.realcheck.tsv"
    last="SC_LAST_$gated_by"
    submit "rb-gate-$name" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:${!last}" "${small[@]}" \
        "$HERE/gate.sbatch" selfcheck "$name" "$gated_by" "$allowed"
    printf -v "GATE_$name" '%s' "$JOB_ID"
done < "$table"

# ── 3 and 4. the ladders ──────────────────────────────────────────────────────────────
# ladder <job prefix> <array range> <realcheck 0|1> <dependencies> <time>: two arrays per
# rung (the attempt, then the same rung again for a unit the scheduler lost), each after
# the one before; sets LADDER_LAST. Reads the current row's variables. Every task is
# given the ids of all the ladder's earlier jobs: it asks the scheduler what became of
# its unit in each.
ladder() {
    local prefix="$1" range="$2" real="$3" base="$4" time="$5"
    local earlier="" attempt=0 gb dep task jobname
    for gb in $RUNGS_GB; do
        attempt=$((attempt + 1))
        for task in attempt again; do
            jobname="$prefix-${gb}g"
            if [ "$task" = again ]; then
                jobname="$jobname-again"
            elif [ "$attempt" = 1 ]; then
                task=first
            else
                task=climb
            fi
            dep="$base"
            [ -z "$earlier" ] || dep="$dep,afterany:${earlier##*,}"
            submit "$jobname" "${COMMON[@]}" "$LOG_ARRAY" --dependency="$dep" \
                --array="$range%$REGBENCH_ARRAY_LIMIT" --cpus-per-task="$CPUS" --mem="${gb}G" \
                --time="$time" ${gres[@]+"${gres[@]}"} \
                "$HERE/run_rung.sbatch" "$name" "$method" "$label" "$opts" "$image" "$gpu" \
                "$ds" "$attempt" "$gb" "$real" "${earlier:--}" "$task" "$time"
            earlier="${earlier:+$earlier,}$JOB_ID"
        done
    done
    LADDER_LAST="${earlier##*,}"
}

finals="" prev_ds=""
for ds in $sized; do
    n_cases="CASES_$ds"; n_groups="GROUPS_$ds"; rounds="ROUNDS_$ds"
    while IFS=$'\t' read -r name method label opts image gpu scope after _ _ _ _ _ _ realcheck; do
        has "$selected" "$name" || continue
        gres=()
        [ "$gpu" = 1 ] && gres=(${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
        if [ "$scope" = group ]; then
            n="${!n_groups}"; time="$(limit $((round_s * ${!rounds})))"
        else
            n="${!n_cases}"; time="$(limit "$round_s")"
        fi
        gate="GATE_$name"
        base="afterok:${!gate}"
        if [ "$after" != - ]; then
            last="LAST_${ds}_$after"
            if [ -n "${!last:-}" ]; then
                base="$base,afterany:${!last}"
            else
                echo "# $name reads the results of $after, which is not submitted here: they must already be under $REGBENCH_OUT"
            fi
        fi
        # one dataset after the other, per run: the cohort (multiplex) is done before
        # the same run starts on the next dataset
        if [ -n "$prev_ds" ]; then
            last="LAST_${prev_ds}_$name"
            [ -n "${!last:-}" ] && base="$base,afterany:${!last}"
        fi
        start=0
        if [ "${realcheck:--}" != - ]; then
            real="REAL_$name"
            if [ "$ds" = "$first_dataset" ]; then
                allowed=0
                has "${REGBENCH_ALLOW_UNCHECKED:-}" "$name" && allowed=1
                ladder "rb-$name-$ds-first" "0-0" 1 "$base" "$time"
                finals="$finals $LADDER_LAST"
                submit "rb-realgate-$name" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$LADDER_LAST" \
                    "${small[@]}" "$HERE/gate.sbatch" realcheck "$name" "$method" "$label" \
                    "$ds" "$realcheck" "$allowed"
                printf -v "$real" '%s' "$JOB_ID"
                start=1
            fi
            base="$base,afterok:${!real}"
        fi
        if [ "$n" -gt "$start" ]; then
            ladder "rb-$name-$ds" "$start-$((n - 1))" 0 "$base" "$time"
            printf -v "LAST_${ds}_$name" '%s' "$LADDER_LAST"
            finals="$finals $LADDER_LAST"
        elif [ "$start" = 1 ]; then
            real="REAL_$name"
            printf -v "LAST_${ds}_$name" '%s' "${!real}"
        fi
    done < "$table"
    prev_ds="$ds"
done

# ── 5. tissue, then the scorer, in the scorer's image ─────────────────────────────────
dep=""
for j in $finals; do dep="${dep:+$dep:}$j"; done
while IFS=$'\t' read -r name _; do
    has "$selected" "$name" || continue
    gate="GATE_$name"
    dep="${dep:+$dep:}${!gate}"
done < "$table"
if [ -n "$EXTERNAL" ]; then
    # no tissue job (it is defined on nuclear channels); the set's own scorer and tables
    submit "rb-score-$EXTERNAL" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$dep" --cpus-per-task="$CPUS" \
        --mem="${REGBENCH_SCORE_MEM:-64G}" --time=1-00:00:00 "$HERE/score_cohort.sbatch" score "$EXTERNAL"
else
    submit "rb-tissue" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$dep" --cpus-per-task="$CPUS" \
        --mem="${REGBENCH_SCORE_MEM:-64G}" --time=1-00:00:00 "$HERE/score_cohort.sbatch" tissue
    submit "rb-score" "${COMMON[@]}" "$LOG_ONE" --dependency="afterany:$JOB_ID" --cpus-per-task="$CPUS" \
        --mem="${REGBENCH_SCORE_MEM:-64G}" --time=1-00:00:00 "$HERE/score_cohort.sbatch" score
fi

echo "# $N_JOBS job(s)$([ "$DRY" = 1 ] && echo ' would be submitted; nothing was' || echo ' submitted')"
echo "# submission  -> $REGBENCH_SUBMISSION (a unit already ok under $REGBENCH_OUT is reused$([ "${REGBENCH_FORCE:-0}" = 1 ] && echo ': NO, REGBENCH_FORCE=1 runs everything again'))"
echo "# gates       -> $REGBENCH_OUT/logs/gates/<run>.tsv (run, status, rows dropped, blocked)"
if [ -n "$EXTERNAL" ]; then
    echo "# results     -> $REGBENCH_OUT/tables/$EXTERNAL/summary.md (this set's own tables; none of the cohort's)"
    [ "$EXTERNAL" = hyreco_a ] && echo "# hyreco_a is the TUNING set: nothing in its tables is a result"
else
    echo "# results     -> $REGBENCH_OUT/tables/summary.md: it names every blocked run and dropped row"
fi
[ -n "${REGBENCH_ALLOW_UNCHECKED:-}" ] && echo "# allowed to start unchecked: $REGBENCH_ALLOW_UNCHECKED"
[ -n "$EXTERNAL" ] || [ -n "${REGBENCH_PIPELINE_TRACE:-}" ] || echo "# no REGBENCH_PIPELINE_TRACE: the pipeline's convert and merge steps will read 'not available'"
exit 0
