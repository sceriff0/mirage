#!/bin/bash
# shellcheck disable=SC2154,SC2153  # repo, launcher, here, ALL_* and the REGBENCH_* settings are launch.sbatch's, which sources this file
# launch.sbatch --all: the whole benchmark from ONE sbatch, as a CHAIN OF SHORT HEAD JOBS.
# Sourced by launch.sbatch (after its SETTINGS, the site and the images), never run on
# its own; chain_run is the whole of it. README, "The whole benchmark in one command".
#
# THE STAGES, in order. Each head runs ONE, submits the next head (launch.sbatch again,
# --all-stage=<next>) with a dependency on that stage's jobs, and ends: no head sleeps
# waiting for another, nothing waits on the login node, nothing uses `sbatch --wait`.
#
#   stage    done when (read from disk, by the FIRST head only)   the head runs        the next head waits for
#   inputs   $REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv exists
#                                                                   launcher --inputs    afterok: the rb-inputs job
#   prepare  the rounds file exists AND the cohort has prepared cases (the launcher's
#            own predicate: submit_cohort.sh --cases-there)       launcher --prepare   afterok: the rb-prepare-<ds> job(s),
#                                                                                       NEVER the nuclei tasks (a slide that
#                                                                                       fails is tolerated: afterany upstream)
#   launch   -                                                      the launch           nothing: the chain ends here
#
# A stage that is done is skipped; prepare done counts inputs as done (nothing after it
# reads the input step). The first head runs the first stage not done; every later head
# is GIVEN its stage, so the chain can never loop or skip. The launch stage takes
# --datasets "multiplex <every public dataset whose cases are there>" (the ones left out
# are said, with why) unless --datasets was given, and --compute-all when REGBENCH_INPUT
# is set (else a line says it is skipped and why). Every other launcher argument of the
# --all call goes to the launch, and the arguments after `--` to every stage.
#
# THE RECORD: $REGBENCH_ROOT/logs/chain/<chain id>.tsv (stage, head_job, ids_file,
# next_head, at), one line per head. Every head prints ONE scancel line that removes
# everything the chain has queued so far: the heads, and every job id the stages' ids
# files hold (the tile submitters add theirs to the launch's file as they feed).
#
# A STAGE THAT QUEUED NOTHING (the launcher refused, or stopped) ends the chain with an
# exit other than 0 and says why: no next head is submitted to wait on nothing.
#
# --all --dry-run submits nothing: the stage plan as read from disk, the next head's
# sbatch line, and the launcher's own --dry-run of the stage this head would run (and
# of the launch too when the inputs and the cohort's cases are there).

chain_say() { echo "regbench-launch --all: $*"; }
chain_fail() { echo "regbench-launch --all: $1" >&2; return "${2:-1}"; }

# the launcher as a probe: no `[regbench] end kind=launch` line of its own
chain_probe() { (unset REGBENCH_LAUNCH_T0; bash "$launcher" "$@"); }

# chain_head_options: the first head's own --partition, --account, --qos and --time,
# READ from the scheduler (scontrol show job): they were given to the first sbatch and
# are visible nowhere else inside the job. The next head is given the same.
chain_head_options() {
    local info pair key flag value out=""
    [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1 || return 0
    info="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)" || return 0
    for pair in Partition:--partition Account:--account QOS:--qos TimeLimit:--time; do
        key="${pair%%:*}" flag="${pair#*:}"
        value="$(grep -oE "(^|[[:space:]])$key=[^[:space:]]+" <<< "$info" | head -n 1 | sed -E "s/^[[:space:]]*$key=//")"
        case "$value" in ''|'(null)'|N/A|UNLIMITED|Partition_Limit|INVALID) continue ;; esac
        out="$out $flag=$value"
    done
    echo "${out# }"
}

# chain_next_head <stage> <afterok ids, `:`-joined>: the next head's sbatch arguments,
# in CHAIN_NEXT. Every flag, and why:
chain_next_head() {
    local opts=()
    # shellcheck disable=SC2206  # flags with no spaces in them, one word each
    opts=(${REGBENCH_CHAIN_SBATCH:-})
    CHAIN_NEXT=(
        # the job id alone on stdout (`<id>[;cluster]`), to record and to wait on
        --parsable
        # the head's environment (the SETTINGS, the images, the chain's id) reaches it
        --export=ALL
        # a stage that FAILED cancels the next head, instead of leaving it pending for
        # ever as DependencyNeverSatisfied
        --kill-on-invalid-dep=yes
        # it starts only when every job of the stage it waits for ended ok
        --dependency="afterok:$2"
        # one name for every head: squeue --me -n regbench-launch
        --job-name=regbench-launch
        # the first head's submit folder: launch.sbatch's $here, and so REGBENCH_ROOT's
        # default, is the same for every head
        --chdir="$REGBENCH_CHAIN_SUBMIT_DIR"
        # an absolute log, beside the chain's record
        --output="$REGBENCH_ROOT/logs/regbench-launch-%j.log"
        # the first head's --partition, --account, --qos and --time (chain_head_options)
        ${opts[@]+"${opts[@]}"}
        # NEVER --wait (a head would sleep) and never --requeue (a requeued head would
        # submit its stage a second time)
        "$repo/benchmarks/regbench/slurm/launch.sbatch" --all-stage="$1"
        ${ALL_LAUNCHER[@]+"${ALL_LAUNCHER[@]}"}
    )
    [ ${#ALL_EXTRA[@]} -eq 0 ] || CHAIN_NEXT+=(-- "${ALL_EXTRA[@]}")
}

# chain_scancel: the ONE line that removes everything the chain has queued so far
chain_scancel() {
    local heads files
    heads="$(awk -F'\t' 'NR > 1 { if ($2 != "-") print $2; if ($4 != "-") print $4 }' "$CHAIN_FILE" \
        | awk '!seen[$0]++' | paste -sd' ' -)"
    files="$(awk -F'\t' 'NR > 1 && $3 != "-" { n = split($3, f, ","); for (i = 1; i <= n; i++) printf " \"%s\"", f[i] }' "$CHAIN_FILE")"
    if [ -n "$files" ]; then
        chain_say "to remove everything this chain has queued so far: scancel ${heads:+$heads }\$(cat${files})"
    else
        chain_say "to remove everything this chain has queued so far: scancel $heads"
    fi
}

# chain_record <stage> <ids files, `,`-joined, or -> <next head or ->
chain_record() {
    mkdir -p "$(dirname "$CHAIN_FILE")" || return 1
    [ -s "$CHAIN_FILE" ] || printf 'stage\thead_job\tids_file\tnext_head\tat\n' > "$CHAIN_FILE" || return 1
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "${SLURM_JOB_ID:--}" "$2" "$3" "$(date +%FT%T)" >> "$CHAIN_FILE"
}

# chain_stage_args <stage>: the launcher's arguments for it, in STAGE_ARGS
chain_stage_args() {
    case "$1" in
        inputs) STAGE_ARGS=(--inputs) ;;
        prepare) STAGE_ARGS=(--prepare) ;;
        launch)
            STAGE_ARGS=(${ALL_LAUNCHER[@]+"${ALL_LAUNCHER[@]}"})
            [ "$ALL_HAS_DATASETS" = 1 ] || STAGE_ARGS+=(--datasets "$LAUNCH_DATASETS")
            [ "$LAUNCH_COMPUTE" != 1 ] || STAGE_ARGS+=(--compute-all) ;;
    esac
    [ ${#ALL_EXTRA[@]} -eq 0 ] || STAGE_ARGS+=(-- "${ALL_EXTRA[@]}")
}

# chain_launch_datasets: LAUNCH_DATASETS and LAUNCH_COMPUTE, said
chain_launch_datasets() {
    local ds ok why left=""
    LAUNCH_DATASETS="multiplex"
    if [ "$ALL_HAS_DATASETS" = 1 ]; then
        chain_say "the launch's datasets: as given (--datasets)"
    else
        while IFS=$'\t' read -r ds ok why; do
            [ -n "$ds" ] && [ "$ds" != multiplex ] || continue
            if [ "$ok" = yes ]; then
                LAUNCH_DATASETS="$LAUNCH_DATASETS $ds"
            else
                left="$left; $ds ($why)"
            fi
        done <<< "$CASES_THERE"
        chain_say "the launch's datasets: --datasets \"$LAUNCH_DATASETS\" (the cohort and every public dataset whose cases are prepared)"
        [ -z "$left" ] || chain_say "left out: ${left#; } (a public dataset is prepared by hand: README, \"State\")"
    fi
    LAUNCH_COMPUTE=0
    case " ${ALL_LAUNCHER[*]:-} " in *" --compute-all "*) LAUNCH_COMPUTE=2 ;; esac
    if [ "$LAUNCH_COMPUTE" = 2 ]; then
        LAUNCH_COMPUTE=0  # given already: it is among the launcher's arguments
        chain_say "compute-all: --compute-all as given"
    elif [ -n "${REGBENCH_INPUT:-}" ]; then
        LAUNCH_COMPUTE=1
        chain_say "compute-all: --compute-all (REGBENCH_INPUT=$REGBENCH_INPUT): the whole pipeline's cost per patient and backend, run last and alone"
    else
        chain_say "compute-all is skipped: REGBENCH_INPUT (the cohort's samplesheet, which compute-all runs the pipeline on) is not set"
    fi
}

chain_run() {
    local stage next="" ids_files=() f n code got next_id wait_name wait_ids existing pre rounds
    local inputs_done=0 cohort=0 prepare_done=0 cohort_said
    # the launcher's own arguments of this call: everything but --dry-run
    ALL_LAUNCHER=() ALL_HAS_DATASETS=0
    for f in ${ALL_ARGS[@]+"${ALL_ARGS[@]}"}; do
        [ "$f" = --dry-run ] && continue
        [ "$f" = --datasets ] && ALL_HAS_DATASETS=1
        ALL_LAUNCHER+=("$f")
    done

    # ── what is on disk ──
    pre="$REGBENCH_INPUTS/preprocess/results/csv/preprocessed.csv"
    rounds="${REGBENCH_MANIFEST:-$REGBENCH_INPUTS/rounds.csv}"
    [ -e "$pre" ] && inputs_done=1
    CASES_THERE="$(chain_probe --cases-there)" \
        || { chain_fail "cannot tell which datasets have prepared cases (submit_cohort.sh --cases-there failed: the lines above). Nothing was submitted." 2; return; }
    cohort_said="$(awk -F'\t' '$1 == "multiplex" { print $3 }' <<< "$CASES_THERE")"
    grep -q $'^multiplex\tyes' <<< "$CASES_THERE" && cohort=1
    [ -s "$rounds" ] && [ "$cohort" = 1 ] && prepare_done=1
    if [ "$inputs_done" = 1 ]; then
        chain_say "stage inputs: done ($pre exists)"
    elif [ "$prepare_done" = 1 ]; then
        chain_say "stage inputs: counted as done: the prepare stage is done ($pre is not there)"
    else
        chain_say "stage inputs: not done ($pre is not there)"
    fi
    if [ "$prepare_done" = 1 ]; then
        chain_say "stage prepare: done (the rounds file $rounds, the cohort: $cohort_said)"
    else
        chain_say "stage prepare: not done (the rounds file $rounds: $([ -s "$rounds" ] && echo there || echo "not there"); the cohort: $cohort_said)"
    fi
    chain_say "stage launch: the last stage, the launcher's launch"

    # ── which stage this head runs ──
    if [ -n "$ALL_STAGE" ]; then
        stage="$ALL_STAGE"
        [ -n "${REGBENCH_CHAIN:-}" ] \
            || { chain_fail "--all-stage=$stage is a later head of a chain, which the head before it hands REGBENCH_CHAIN; start a chain with --all. Nothing was submitted." 2; return; }
    elif [ "$prepare_done" = 1 ]; then
        stage=launch
    elif [ "$inputs_done" = 1 ]; then
        stage=prepare
    else
        stage=inputs
    fi
    case "$stage" in inputs) next=prepare ;; prepare) next=launch ;; esac
    chain_say "this head runs the stage $stage${next:+, then submits the head of the stage $next}"

    # ── the chain's identity: made by the first head, handed on (--export=ALL) ──
    if [ -z "${REGBENCH_CHAIN:-}" ]; then
        REGBENCH_CHAIN="$(date +%Y%m%dT%H%M%S)-${SLURM_JOB_ID:-$$}"
    fi
    : "${REGBENCH_CHAIN_SUBMIT_DIR:=$here}"
    [ -n "${REGBENCH_CHAIN_SBATCH+set}" ] || REGBENCH_CHAIN_SBATCH="$(chain_head_options)"
    : "${REGBENCH_SUBMISSION:=$REGBENCH_CHAIN}"
    export REGBENCH_CHAIN REGBENCH_CHAIN_SUBMIT_DIR REGBENCH_CHAIN_SBATCH REGBENCH_SUBMISSION
    CHAIN_FILE="$REGBENCH_ROOT/logs/chain/$REGBENCH_CHAIN.tsv"
    [ "$stage" != launch ] || chain_launch_datasets
    chain_stage_args "$stage"
    # the launch prints its peak (tasks, CPUs, GPUs, GB) before its first sbatch
    [ "$stage" != launch ] || export REGBENCH_PEAK=1

    case "$stage" in
        inputs) wait_name=rb-inputs ;;
        prepare) wait_name=rb-prepare- ;;
        *) wait_name="" ;;
    esac

    # ── a dry run: the plan, the next head, the launcher's own dry run ──
    if [ "$ALL_DRY" = 1 ]; then
        chain_say "dry run: nothing is submitted, no chain record is written (chain id $REGBENCH_CHAIN)"
        chain_say "the launcher's dry run of the stage $stage:"
        chain_probe --dry-run "${STAGE_ARGS[@]}" || { chain_fail "the launcher refused the stage $stage (the lines above)" 2; return; }
        if [ -n "$next" ]; then
            chain_next_head "$next" "<$wait_name...>"
            chain_say "the next head: sbatch ${CHAIN_NEXT[*]}"
        else
            chain_say "the chain ends with this stage"
        fi
        if [ "$stage" != launch ] && [ "$inputs_done" = 1 ] && [ "$cohort" = 1 ]; then
            chain_launch_datasets
            chain_stage_args launch
            chain_say "the inputs and the cohort's cases are there: the launcher's dry run of the launch:"
            REGBENCH_PEAK=1 chain_probe --dry-run "${STAGE_ARGS[@]}" || { chain_fail "the launcher refused the launch (the lines above)" 2; return; }
        fi
        return 0
    fi

    # ── the stage ──
    mkdir -p "$REGBENCH_ROOT/logs" || { chain_fail "cannot write $REGBENCH_ROOT/logs. Nothing was submitted." 2; return; }
    bash "$launcher" "${STAGE_ARGS[@]}"
    code=$?
    case "$stage" in
        inputs) ids_files=("$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.inputs.ids") ;;
        prepare) ids_files=("$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.prepare.ids") ;;
        launch)
            ids_files=("$REGBENCH_OUT/logs/submitted/$REGBENCH_SUBMISSION.ids")
            # REGBENCH_REPEAT: one results folder, and one ids file, per repeat
            for f in "$REGBENCH_OUT"/rep*/logs/submitted/"$REGBENCH_SUBMISSION.ids"; do
                [ -e "$f" ] && ids_files+=("$f")
            done ;;
    esac
    existing="" n=0
    for f in "${ids_files[@]}"; do
        [ -s "$f" ] || continue
        existing="${existing:+$existing,}$f"
        n=$((n + $(grep -c . "$f")))
    done
    if [ "$code" != 0 ] || [ "$n" = 0 ]; then
        chain_record "$stage" "${existing:--}" - \
            || echo "regbench-launch --all: cannot write the chain record $CHAIN_FILE" >&2
        chain_scancel
        if [ "$n" = 0 ]; then
            chain_fail "the stage $stage queued nothing (the launcher ended with exit $code; the reason is above): no next head is submitted, and the chain ends here. Fix that reason and submit --all again: the stages that are done are skipped." 1
        else
            chain_fail "the stage $stage stopped (exit $code) after queuing $n job(s) (the launcher's lines above): no next head is submitted, and the chain ends here." 1
        fi
        return
    fi
    if [ -z "$next" ]; then
        chain_record "$stage" "$existing" - \
            || echo "regbench-launch --all: cannot write the chain record $CHAIN_FILE" >&2
        chain_say "the stage launch queued $n job(s): the chain ends; the launcher's own jobs run from here"
        chain_scancel
        return 0
    fi
    # the jobs the next head waits for: of THIS stage (its ids files), by name
    wait_ids="$(awk -F'\t' -v pre="$wait_name" 'NR == FNR { mine[$1] = 1; next } index($2, pre) == 1 && ($3 in mine) { print $3 }' \
        <(cat "${ids_files[@]}" 2>/dev/null) "$REGBENCH_OUT/logs/jobs.tsv" 2>/dev/null | paste -sd: -)"
    if [ -z "$wait_ids" ]; then
        chain_record "$stage" "$existing" - || true
        chain_scancel
        chain_fail "the stage $stage queued $n job(s) but none named $wait_name* ($REGBENCH_OUT/logs/jobs.tsv): no next head is submitted, and the chain ends here." 1
        return
    fi
    chain_next_head "$next" "$wait_ids"
    got="$(cd "$REGBENCH_CHAIN_SUBMIT_DIR" && sbatch "${CHAIN_NEXT[@]}" < /dev/null 2>&1)"
    code=$?
    next_id="$(awk 'NF { last = $0 } END { print last }' <<< "$got")"
    next_id="${next_id%%;*}"
    case "$next_id" in ''|*[!0-9]*) code=1 ;; esac
    if [ "$code" != 0 ]; then
        chain_record "$stage" "$existing" - || true
        chain_scancel
        chain_fail "sbatch refused the next head ($got): the stage $stage's jobs are queued; submit the next head yourself: sbatch ${CHAIN_NEXT[*]}" 1
        return
    fi
    chain_record "$stage" "$existing" "$next_id" \
        || echo "regbench-launch --all: cannot write the chain record $CHAIN_FILE" >&2
    chain_say "the stage $stage queued $n job(s); the head of the stage $next is job $next_id (afterok:$wait_ids). Watch: squeue --me -n regbench-launch; the record: $CHAIN_FILE"
    chain_scancel
    return 0
}
