#!/bin/bash
#SBATCH --job-name=regbench-retry
# The retries of ONE chain (a run's memory ladder on one dataset, or one stage of a run
# whose registration is several jobs per round), submitted ONLY AFTER A FAILURE.
#
# submit_cohort.sh submits the chain's FIRST array and this small job after it
# (`afterany`); nothing else of the chain is in the queue. When the first array has
# ended this job reads what became of each unit and, step by step, submits the next
# attempt ONLY for the units that need one, as `--array=<those indices>`, waits for it
# to leave the queue, and goes on. A chain whose units are all ok submits nothing and
# this job ends at once. It runs on a compute node, OUTSIDE any image, and calls
# `sbatch` and `squeue` itself, as the tile submitter (feed_tiles.sbatch) does.
#
#   retry.sbatch <run|-> <plan folder> <first job id> <first job name>
#                <array range a-b[,c,...]|-> [<link>]
#
# A range of `-`: the chain is of SINGLE jobs, not arrays (a selfcheck). Then a step is
# submitted without --array, the end line is <job name>-<job id>.end, and this job
# exits 0 only when the chain's last job ended ok. Nothing waits for it with `afterok`:
# the selfcheck is reported, never a gate. A run of `-`: the chain is a selfcheck's.
#
# THE PLAN is written by submit_cohort.sh: <plan folder>/step-<k>, one per step in the
# order the chain runs, one item per line:
#   line 1    the step's kind: `again` (the same rung once more), `climb` (the next rung),
#             `cut` (a rung the partition cannot grant, below), or anything else
#             (`stage`, `notok`: the next attempt of a stage job or of a selfcheck)
#   line 2    the job name
#   line 3..  the sbatch arguments (no --array, no --dependency), then the job script
#             and its arguments. The item `@EARLIER@` is replaced by the ids of every
#             job of this chain submitted so far (run_rung.sbatch asks the scheduler's
#             accounting about them), and `@PREVIOUS@` by the id of the last one.
#             A `cut` step has one more line before them, line 3: why the rung cannot
#             be granted (submit_cohort.sh read it off `sinfo` before the first sbatch)
#
# A RUNG THAT CANNOT BE GRANTED. A `cut` step is never submitted: the units that would
# climb to it are written down as `capped` (`regbench chain-mark`: the memory failure
# below kept in cause_from, `ladder_cut` = the rung, the error "capped at <the rung
# below> GB"): the method's, imputed, and their rows say that the ladder stopped at the
# largest rung the partition grants (Task 28, B6).
#
# A STEP SBATCH REFUSES is asked again, REGBENCH_RETRY_SUBMIT_TRIES (5) times in all,
# waiting REGBENCH_RETRY_SUBMIT_WAIT_S (60) seconds and twice as long after each refusal;
# before each new try the queue is asked whether the refused call went through after
# all (a slurmctld that timed out can still have queued it), so a step is never queued
# twice. Refused every time, its units are written down as `submit_refused` (not the
# method's, B7) and the chain stops there.
#
# A WATCHER THAT GIVES UP (the queue could not be read REGBENCH_RETRY_SQUEUE_TRIES times
# in a row) first writes the units of the step it was at down as `watcher_gave_up`
# itself (chain-mark, M6), then why in <plan folder>/gave_up before it exits: the settle
# job writes the units it was still waiting for, and those it never submitted, as
# `watcher_gave_up` too (chains.mark_in_flight, S3). A later link that takes the chain over
# removes the file: the chain is not given up while a link goes on with it.
#
# THE WATCHER IS A CHAIN OF LINKS (REGBENCH_RETRY_WATCHERS, submit_cohort.sh): link 1
# waits for the first array, each later link for the link before it (`afterany`), and
# what waits for the chain waits for the LAST link. A link ended at its time limit, or
# lost with its node, leaves its successor to go on: every step it submits is recorded
# in <plan folder>/state before anything else (the step, the job id, the ids so far),
# and a link that finds one waits for that job to leave the queue and goes on with the
# next step. A step whose sbatch may have gone through without the id being recorded
# (<plan folder>/submitting) is looked for in the queue by its name before it is
# submitted again. The link that ends the chain writes <plan folder>/done; a later link
# that finds it ends at once, as the chain ended. A chain no link finished (no `done`)
# is named by the settle job, the scorer's summary and `regbench status`.
#
# WHICH UNITS A STEP IS SUBMITTED FOR. Each task leaves an end line in
# $REGBENCH_OUT/logs/end/<job name>-<job id>_<index>.end (env.sh). The unit's LAST task
# decides:
#   no end line     the task died or never started: every kind of step takes it
#   status=ok       nothing more, ever
#   `again`/`climb` the unit's end line names the step in `wants=` (a run task's: the
#                   ONE predicate, failure_log.chain_wants, read from every record of
#                   the unit; `wants=-` is no step). A run end line without the key
#                   (written before it existed) is taken, as one with no end line is
#   any other kind  anything that is not ok
# This is a FILTER, not the decision: the task that is submitted still decides for
# itself, exactly as before (run_rung.sbatch, stage.sbatch), and exits at once when its
# unit has nothing to do. The filter only keeps a unit that cannot need the step out of
# the queue; a test holds the predicate to never keeping out one the ladder would run.
#
# It gives up (exit 1, the rest of the chain is not submitted) when sbatch refuses a
# step every time it is asked, or the queue cannot be read REGBENCH_RETRY_SQUEUE_TRIES
# (30) times in a row.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" plan="$2" first_id="$3" first_name="$4" range="$5" link="${6:-1}"
: "${REGBENCH_RETRY_POLL_S:=60}"
: "${REGBENCH_RETRY_SQUEUE_TRIES:=30}"
: "${REGBENCH_RETRY_SUBMIT_TRIES:=5}"
: "${REGBENCH_RETRY_SUBMIT_WAIT_S:=60}"
: "${REGBENCH_ARRAY_LIMIT:=20}"
ends="$REGBENCH_OUT/logs/end"
ids_file="${REGBENCH_IDS_FILE:-$REGBENCH_OUT/logs/submitted/${REGBENCH_SUBMISSION:-none}.ids}"
mkdir -p "$(dirname "$ids_file")"
state="$plan/state" done_file="$plan/done" submitting="$plan/submitting"
# the %N of this chain's arrays: the one its plan names (<plan folder>/limit, written by
# submit_cohort.sh: REGBENCH_GPU_ARRAY_LIMIT for a run that asks for a GPU), else, for a
# plan written before that file existed, REGBENCH_ARRAY_LIMIT
array_limit="$REGBENCH_ARRAY_LIMIT"
if [ -s "$plan/limit" ]; then
    read -r array_limit < "$plan/limit"
    case "$array_limit" in ''|*[!0-9]*|0) echo "the plan's array limit ($plan/limit) is '$array_limit', not a whole number of at least 1: REGBENCH_ARRAY_LIMIT=$REGBENCH_ARRAY_LIMIT is used"; array_limit="$REGBENCH_ARRAY_LIMIT" ;; esac
fi

end() {  # end <status> <cause|-> : the last line of the log
    local cause="" whose=launcher
    case "$2" in submit_refused|watcher_gave_up) whose=infrastructure ;; esac
    [ "$2" = - ] || cause=" cause=$2 whose=$whose"
    regbench_say_end "REGBENCH_END kind=retry run=$name chain=$first_name link=$link status=$1$cause log=${REGBENCH_LOG:--}"
}
regbench_start retry "run=$name" "chain=$first_name" "link=$link"
# finish <status> <cause|-> <exit code>: the chain is over; said for every later link
finish() {
    printf '%s %s %s\n' "$1" "$2" "$3" > "$done_file.new" && mv "$done_file.new" "$done_file"
    end "$1" "$2"
    exit "$3"
}

# a later link of a chain an earlier link finished: it ends as the chain did
if [ -s "$done_file" ]; then
    read -r was_status was_cause was_code < "$done_file"
    echo "$first_name: the chain was finished by an earlier link ($was_status); link $link has nothing to do"
    end "$was_status" "$was_cause"
    exit "${was_code:-0}"
fi
# this link goes on with the chain: an earlier link's giving up is no longer the end
rm -f "$plan/gave_up"
# give_up <why> [k]: the watcher stops. It first writes the units of step k (the step
# it was submitting or waiting for) down as watcher_gave_up itself (M6), so a settle job
# that fails does not leave them the method's; the settle job writes the rest.
give_up() {
    if [ -n "${2:-}" ] && [ -s "$plan/due-$2" ]; then
        mark_cut "$2" watcher_gave_up "the watcher gave up: $1"
    fi
    printf '%s %s\n' "$(date +%FT%T)" "$1" > "$plan/gave_up.new" && mv "$plan/gave_up.new" "$plan/gave_up"
    end failed watcher_gave_up
    exit 1
}

# last: "<index> <job name>-<job id>" per unit, the last job that held the unit
last="$plan/last"
single=0
[ "$range" = - ] && single=1
# indices <a-b,c,...>: the first array's tasks, one per line (a list: the nuclei step
# is submitted for the slides that need it only)
indices() {
    local part
    local -a parts
    IFS=, read -r -a parts <<< "$1"
    for part in "${parts[@]}"; do
        case "$part" in *-*) seq "${part%%-*}" "${part#*-}" ;; *) echo "$part" ;; esac
    done
}
if [ ! -s "$last" ]; then
    { if [ "$single" = 1 ]; then echo -; else indices "$range"; fi; } \
        | awk -v j="$first_name-$first_id" '{ print $1, j }' > "$last.new" && mv "$last.new" "$last" \
        || { echo "cannot write $last"; end failed not_submitted; exit 1; }
fi
earlier="$first_id" previous="$first_id"
# end_file <index> <job name>-<job id>: where that task left its end line
end_file() {
    if [ "$single" = 1 ]; then echo "$ends/$2.end"; else echo "$ends/${2}_$1.end"; fi
}

# wanted <kind>: the indices the step is submitted for, one per line
wanted() {
    local kind="$1" idx job file st wants
    [ "$kind" = cut ] && kind=climb  # a cut rung is for the units that would climb
    while read -r idx job; do
        file="$(end_file "$idx" "$job")"
        if [ ! -s "$file" ]; then echo "$idx"; continue; fi
        st="$(sed -nE 's/.* status=([^ ]+).*/\1/p' "$file" | tail -n 1)"
        [ "$st" = ok ] && continue
        case "$kind" in
            again|climb)
                # `wants=` says which steps can still run the unit; none, for a line
                # written before the key existed: then every step takes it
                if ! grep -qE '(^| )wants=' "$file"; then echo "$idx"; continue; fi
                wants="$(sed -nE 's/.* wants=([^ ]+).*/\1/p' "$file" | tail -n 1)"
                case ",$wants," in *",$kind,"*) echo "$idx" ;; esac
                ;;
            *) echo "$idx" ;;
        esac
    done < "$last"
}

# spec: indices on stdin, one per line, as an --array list with runs folded (0-3,7,9-12)
spec() {
    sort -n | awk '
        NR == 1 { a = $1; b = $1; next }
        $1 == b + 1 { b = $1; next }
        { out = out (out == "" ? "" : ",") (a == b ? a : a "-" b); a = $1; b = $1 }
        END { if (NR) print out (out == "" ? "" : ",") (a == b ? a : a "-" b) }'
}

# gone <job id>: wait until the job has left the queue. A queue that cannot be read is
# never taken for an empty one.
gone() {
    local id="$1" out bad=0
    while :; do
        if out="$(squeue -h -j "$id" -o '%i' 2>&1)"; then
            bad=0
            [ -z "$out" ] && return 0
        else
            # a job the scheduler no longer knows has left the queue
            case "$out" in *[Ii]nvalid\ job\ id*) return 0 ;; esac
            bad=$((bad + 1))
            echo "$(date +%FT%T) the queue could not be read (squeue -j $id), look $bad of at most $REGBENCH_RETRY_SQUEUE_TRIES in a row: $out"
            [ "$bad" -ge "$REGBENCH_RETRY_SQUEUE_TRIES" ] && return 1
        fi
        sleep "$REGBENCH_RETRY_POLL_S"
    done
}

# in_queue <job name>: the id of this user's queued job of that name (the first), or
# nothing when the queue has none. The queue is looked at as gone() looks: a failed
# squeue is asked again, up to REGBENCH_RETRY_SQUEUE_TRIES times in a row, and then it
# returns 1. "Not there" is only ever an answer the queue GAVE: a successor that
# resubmits on a failed look would put the step in the queue twice.
in_queue() {
    local name="$1" got bad=0 err="$plan/squeue.err"
    while :; do
        # the id comes from stdout alone: a squeue that answers can still warn on stderr
        if got="$(squeue -h -u "$(id -un)" --name="$name" -o '%i' 2>"$err")"; then
            got="$(grep -m1 -E '^[0-9]+(_|$)' <<< "$got")" || got=""
            echo "${got%%_*}"
            return 0
        fi
        bad=$((bad + 1))
        echo "$(date +%FT%T) the queue could not be read (squeue --name=$name), look $bad of at most $REGBENCH_RETRY_SQUEUE_TRIES in a row: $(cat "$err" 2>/dev/null)" >&2
        [ "$bad" -ge "$REGBENCH_RETRY_SQUEUE_TRIES" ] && return 1
        sleep "$REGBENCH_RETRY_POLL_S"
    done
}

# recorded <k> <job name> <id>: a step's job is in the chain's files, each write made
# so that doing it twice (a successor finishing what a lost link began) changes nothing
recorded() {
    local k="$1" jobname="$2" id="$3"
    grep -qxF -- "$id" "$ids_file" 2>/dev/null || echo "$id" >> "$ids_file"
    grep -qF -- "$(printf '\t%s\t%s' "$jobname" "$id")" "$REGBENCH_OUT/logs/jobs.tsv" 2>/dev/null \
        || printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$jobname" "$id" >> "$REGBENCH_OUT/logs/jobs.tsv"
    # these units' last job is now this one
    awk -v j="$jobname-$id" 'NR == FNR { due[$1] = 1; next } ($1 in due) { print $1, j; next } { print }' \
        "$plan/due-$k" "$last" > "$last.new" && mv "$last.new" "$last"
}
# save_state <k> <job name> <id>: the step this chain is at, for a successor
save_state() {
    printf 'k=%s\njob=%s\nid=%s\nearlier=%s\nprevious=%s\n' "$1" "$2" "$3" "$earlier" "$previous" \
        > "$state.new" && mv "$state.new" "$state" && rm -f "$submitting"
}
# mark_cut <k> <cause> <why>: write the due units of step k down as the launcher's
# (`regbench chain-mark`), in the scorer's image: the step's own job script arguments
# name the run (run_rung.sbatch <name> <method> <label> <opts> <image> <gpu> <dataset>
# <attempt> <GB> ...)
mark_cut() {
    local k="$1" cause="$2" why="$3" i at=-1 item
    local -a items=() run=() marks=()
    while IFS= read -r item; do items+=("$item"); done < "$plan/step-$k"
    for i in "${!items[@]}"; do
        case "${items[$i]}" in */run_rung.sbatch) at="$i" ;; esac
    done
    if [ "$at" -lt 0 ]; then
        # not a step of a run's ladder (a selfcheck's or a stage's next attempt): no
        # record of it takes the cause, so each unit is a row of
        # logs/refused_units.tsv, which the scorer makes a loss row of that cause
        local refused="$REGBENCH_OUT/logs/refused_units.tsv" flat
        flat="$(tr '\t\n' '  ' <<< "$why")"
        mkdir -p "$REGBENCH_OUT/logs"
        [ -s "$refused" ] || printf 'plan\tstep\tjob\tindex\tcause\twhy\n' > "$refused"
        while read -r i; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename "$plan")" "$k" "${items[1]}" "$i" "$cause" "$flat" >> "$refused"
        done < "$plan/due-$k"
        echo "step $k: not a step of a run's ladder; its $(wc -l < "$plan/due-$k" | tr -d ' ') unit(s) are written down as $cause in $refused"
        return 0
    fi
    run=("${items[@]:$((at + 1)):9}")
    while read -r i; do marks+=(--index "$i"); done < "$plan/due-$k"
    # no cases folder, no record: said, and the watcher goes on (as when chain-mark fails)
    local cases
    if ! cases="$(regbench_cases_of "${run[6]}")"; then
        echo "REGBENCH_CASES_<NAME> not set for ${run[6]}" >&2
        echo "step $k: the units could not be written down as $cause (no cases folder for ${run[6]}): their records still say the rung below's cause"
        return 0
    fi
    local args=(chain-mark --cases "$cases" --out "$REGBENCH_OUT" --dataset "${run[6]}"
                --method "${run[1]}" --rung-gb "${run[8]}" --cause "$cause" --why "$why"
                "${marks[@]}")
    [ "${run[2]}" != - ] && args+=(--label "${run[2]}")
    if ! regbench_py "${REGBENCH_SCORER_IMAGE:-STARE_REGBENCH}" "${args[@]}"; then
        echo "step $k: the units could not be written down as $cause (regbench chain-mark failed above): their records still say the rung below's cause"
    fi
}

n_steps=0 n_sent=0
k=1
# A LINK THAT TAKES OVER. What an earlier link of this chain left: the step it had
# submitted, whose job is waited for before the chain goes on.
if [ -s "$submitting" ] && { [ ! -s "$state" ] || [ "$(cut -d' ' -f1 "$submitting")" != "$(sed -n 's/^k=//p' "$state" 2>/dev/null)" ]; }; then
    read -r k_was job_was < "$submitting"
    if ! found="$(in_queue "$job_was")"; then
        echo "link $link: the queue could not be read $REGBENCH_RETRY_SQUEUE_TRIES times in a row while looking for step $k_was ($job_was), which an earlier link was submitting when it ended: it is NOT submitted again (it may be in the queue already). This link stops here; submit again into the same results folder once the queue answers"
        give_up "the queue could not be read while looking for step $k_was ($job_was)" "$k_was"
    fi
    if [ -n "$found" ]; then
        echo "link $link: step $k_was ($job_was) was submitted by an earlier link that ended before it recorded the id; it is in the queue as $found"
        [ -s "$state" ] && while IFS='=' read -r key value; do
            case "$key" in earlier) earlier="$value" ;; esac
        done < "$state"
        earlier="$earlier,$found" previous="$found"
        recorded "$k_was" "$job_was" "$found"
        save_state "$k_was" "$job_was" "$found"
    else
        echo "link $link: step $k_was ($job_was) may not have reached the queue (an earlier link ended while submitting it, and the queue does not have it): it is submitted again"
        rm -f "$submitting"
    fi
fi
if [ -s "$state" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            k) k_was="$value" ;; job) job_was="$value" ;; id) id_was="$value" ;;
            earlier) earlier="$value" ;; previous) previous="$value" ;;
        esac
    done < "$state"
    echo "link $link: TAKING OVER the chain $first_name from an earlier link that did not finish it (ended at its time limit, or lost with its node): step $k_was ($job_was) is job $id_was"
    recorded "$k_was" "$job_was" "$id_was"
    if ! gone "$id_was"; then
        echo "the queue could not be read $REGBENCH_RETRY_SQUEUE_TRIES times in a row: giving up; $job_was ($id_was) is left in the queue, and a later link of this chain (if any) goes on"
        give_up "the queue could not be read while waiting for step $k_was ($job_was, $id_was)" "$k_was"
    fi
    k=$((k_was + 1)) n_steps="$k_was"
fi
while [ -s "$plan/step-$k" ]; do
    items=()
    while IFS= read -r line; do items+=("$line"); done < "$plan/step-$k"
    kind="${items[0]}" jobname="${items[1]}"
    n_steps=$((n_steps + 1))
    due="$(wanted "$kind")"
    if [ -z "$due" ]; then
        echo "step $k ($jobname, $kind): no unit needs it; not submitted"
        k=$((k + 1))
        continue
    fi
    echo "$due" > "$plan/due-$k"
    if [ "$kind" = cut ]; then
        why="${items[2]}"
        echo "step $k ($jobname): NOT SUBMITTED, the partition cannot grant this rung ($why): $(wc -l <<< "$due" | tr -d ' ') unit(s) that would climb to it are written down as capped (the method's, imputed: the ladder stops at the rung below)"
        mark_cut "$k" capped "$why"
        echo "$first_name: the ladder stops at the rung below; $n_sent retry step(s) were submitted"
        finish ok - 0
    fi
    args=() array="-"
    if [ "$single" != 1 ]; then
        array="$(spec <<< "$due")"
        args+=(--array="$array%$array_limit")
    fi
    for item in "${items[@]:2}"; do
        [ "$item" = "@EARLIER@" ] && item="$earlier"
        [ "$item" = "@PREVIOUS@" ] && item="$previous"
        args+=("$item")
    done
    printf '%s %s\n' "$k" "$jobname" > "$submitting"
    try=1 wait_s="$REGBENCH_RETRY_SUBMIT_WAIT_S"
    until got="$(sbatch --parsable --job-name="$jobname" "${args[@]}" < /dev/null 2>&1)"; do
        echo "sbatch REFUSED $jobname (--array=$array), try $try of $REGBENCH_RETRY_SUBMIT_TRIES: $got"
        refused="$got"
        if [ "$try" -ge "$REGBENCH_RETRY_SUBMIT_TRIES" ]; then
            rm -f "$submitting"
            echo "$(wc -l <<< "$due" | tr -d ' ') unit(s) of this step are written down as submit_refused (not the method's): $jobname was refused $try times"
            mark_cut "$k" submit_refused "sbatch refused $jobname $try times: $(tr '\n' ' ' <<< "$refused")"
            finish failed submit_refused 1
        fi
        sleep "$wait_s"
        wait_s=$((wait_s * 2)) try=$((try + 1))
        # a refusal can still have queued the job: never queue it twice
        if ! found="$(in_queue "$jobname")"; then
            give_up "the queue could not be read after sbatch refused $jobname" "$k"
        fi
        if [ -n "$found" ]; then
            echo "sbatch said it refused $jobname, but the queue has it as $found: taken"
            got="$found"
            break
        fi
    done
    id="$(awk 'NF { last = $0 } END { print last }' <<< "$got")"
    [ "$id" = "$got" ] || awk -v id="$id" '$0 != id && NF { print "sbatch said: " $0 }' <<< "$got"
    id="${id%%;*}"
    case "$id" in ''|*[!0-9_]*) rm -f "$submitting"; echo "sbatch printed no job id for $jobname: $got"; finish failed not_submitted 1 ;; esac
    earlier="$earlier,$id" previous="$id"
    # the state first: whatever ends this link after this line, its successor knows
    save_state "$k" "$jobname" "$id"
    recorded "$k" "$jobname" "$id"
    n_sent=$((n_sent + 1))
    echo "step $k ($jobname, $kind): submitted as $id for $(wc -l <<< "$due" | tr -d ' ') unit(s): --array=$array"
    if ! gone "$id"; then
        echo "the queue could not be read $REGBENCH_RETRY_SQUEUE_TRIES times in a row: giving up; $jobname ($id) is left in the queue, and a later link of this chain (if any) goes on"
        give_up "the queue could not be read while waiting for step $k ($jobname, $id)" "$k"
    fi
    k=$((k + 1))
done
echo "$first_name: $n_sent of $n_steps retry step(s) were needed and submitted"
if [ "$single" = 1 ]; then
    # a chain of single jobs: this job ends as the chain's last job did
    read -r idx job < "$last"
    file="$(end_file "$idx" "$job")"
    st=""
    [ -s "$file" ] && st="$(sed -nE 's/.* status=([^ ]+).*/\1/p' "$file" | tail -n 1)"
    if [ "$st" != ok ]; then
        echo "$first_name: the chain's last job ($job) ended ${st:-without an end line}, not ok"
        finish failed - 1
    fi
fi
finish ok - 0
