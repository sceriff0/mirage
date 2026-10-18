#!/bin/bash
#SBATCH --job-name=regbench-feed
# The tile submitter of ONE run on ONE dataset, for a run whose registration is several
# scheduler jobs per round (`runs`, column `stages`). It runs on a compute node, OUTSIDE
# any image, and calls `sbatch` and `squeue` itself, the way a Nextflow head job does:
# the tile jobs reach the queue a few at a time, never all at once. Submitted by
# submit_cohort.sh after COARSE's last attempt, with these arguments:
#
#   feed_tiles.sbatch <name> <method> <label|-> <opts|-> <image> <dataset>
#                     <layout file> <rounds> <tiles> <link>
#                     <cpus:GB:time of a tile job at attempt 1> <... 2> <... 3> <... 4>
#
# WHAT IT SUBMITS, one attempt after the other, each only when the one before has left
# the queue:
#
#   attempt 1       ONE TASK PER TILE that is due (`regbench stages --due 1`: every tile
#                   of every round whose COARSE is ok, less the tiles a reusable record
#                   already covers), in arrays of at most REGBENCH_STAGE_ARRAY_MAX tasks,
#                   each task with the pipeline's allocation for TILED_REG_TILE
#   attempts 2-4    a RETRY JOB is one task per ROUND that has a tile the pipeline would
#                   retry (`stages --due <attempt>`): it runs that round's due tiles one
#                   after another, with the attempt's memory and time (x attempt)
#
# THE CEILING. Before every `sbatch` it counts the tile tasks of this run and dataset
# that are pending or running (`squeue -r`, by job name) and submits only what keeps the
# count at or below REGBENCH_STAGE_QUEUE_RUN (this run's share of the launcher's
# REGBENCH_STAGE_QUEUE_MAX, which is a total over the launch; submit_cohort.sh divides it
# by the number of runs with a submitter). It waits until at least
# REGBENCH_STAGE_FEED_MIN slots are free (or fewer tiles are left), so that it does not
# submit one-task arrays. Between looks it sleeps REGBENCH_STAGE_POLL_S seconds.
#
# IT NEVER WAITS IN SILENCE, AND FOR EVER ONLY ON A QUEUE THAT IS MERELY BUSY.
#   A queue it cannot read (`squeue` fails) is never taken for an empty one: nothing is
#   submitted, every such look is a line of the log, and after
#   REGBENCH_STAGE_SQUEUE_TRIES (30) of them in a row it GIVES UP.
#   While it waits (for room, or for the queue to empty before the next attempt) and no
#   tile task has left the queue for REGBENCH_STAGE_STALL_S (3600) seconds, it prints
#   every tile task still there with its state and the scheduler's reason. A task whose
#   reason is one it never leaves by itself (REGBENCH_STAGE_HELD_REASONS: a hold, a
#   dependency that can never be met, a failed launch) makes it GIVE UP at once; so does
#   REGBENCH_STAGE_STALL_MAX_S (172800: two days) without one task leaving; the time in
#   which EVERY task left is only waiting its turn (REGBENCH_STAGE_WAIT_REASONS:
#   Priority, Resources, ...) is not counted in those two days, and is said each
#   stretch. Otherwise it goes on waiting and says so again after the next stretch.
#   GIVING UP is the safer of the two ways out: going on would count tiles that are
#   still in the queue as done. It writes <layout>.refused with the reason and what is
#   still queued, ends with exit 1, and the rounds whose tiles are not done fail as
#   `not_submitted` (the launcher's). The tile tasks in the queue are left alone; the
#   next link of the chain starts where this one stopped.
#
# RESTART. Everything it has done is in two files beside the layout:
#   <layout>.due.a<attempt>   what is due at that attempt, one index per line, written once
#   <layout>.fed.tsv          attempt, first line of the due file, lines, array base,
#                             job id: one line BEFORE each sbatch (job id `-`) and one after
# A link that starts (the launcher chains REGBENCH_STAGE_FEEDERS of them, `afterany`)
# continues from those files and from the queue: what was submitted is not submitted
# again, and tile tasks an earlier link left in the queue are counted. A line without a
# job id (the link died between the two writes) is looked up by job name, in the queue
# and in the accounting (from the time the first link started, kept in
# <layout>.started: sacct's own window starts at midnight); found nowhere, that one array
# is submitted again, and a tile whose record is already ok exits at once.
# <layout>.done says nothing is left.
#
# REFUSED. An `sbatch` the scheduler refuses is tried REGBENCH_STAGE_SUBMIT_TRIES times.
# After that this job writes <layout>.refused (what was refused, the scheduler's words,
# what is queued and how to cancel it) and ends: the rounds whose tiles never reached the
# queue fail as `not_submitted`, the launcher's failure and not the method's. The next
# link, or a new submission, goes on from the same files.
#
# No job waits on `sbatch --wait`. The exit code is 0 unless an sbatch was refused or
# the submitter gave up.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" method="$2" label="$3" opts="$4" image="$5" dataset="$6" layout="$7"
n_rounds="$8" n_tiles="$9" link="${10}"
ALLOC=("" "${11}" "${12}" "${13}" "${14}")
: "${REGBENCH_STAGE_ARRAY_MAX:=100}"
: "${REGBENCH_STAGE_QUEUE_RUN:=100}"
: "${REGBENCH_STAGE_FEED_MIN:=25}"
: "${REGBENCH_STAGE_POLL_S:=60}"
: "${REGBENCH_STAGE_SUBMIT_TRIES:=5}"
: "${REGBENCH_STAGE_ARRAY_SPAN:=1000}"
: "${REGBENCH_STAGE_SQUEUE_TRIES:=30}"
: "${REGBENCH_STAGE_STALL_S:=3600}"
: "${REGBENCH_STAGE_STALL_MAX_S:=172800}"
# squeue's reasons of a pending job that does not start by itself (SLURM "Job Reason
# Codes", slurm.schedmd.com/job_reason_codes.html `[PARTIAL: read the entries named here
# and Priority, Resources, Dependency, BeginTime, None, the QOS* and Assoc* families;
# missing the rest of the list]`)
: "${REGBENCH_STAGE_HELD_REASONS:=JobHeldAdmin JobHeldUser JobHoldMaxRequeue DependencyNeverSatisfied JobLaunchFailure}"
# squeue's reasons of a job that is only waiting its turn: while EVERY tile task left has
# one of them, the time is not counted towards REGBENCH_STAGE_STALL_MAX_S (a busy cluster
# is no stall). Same page: Priority "One of more higher priority jobs exist for the
# partition", Resources "The resources requested by the job are not available (e.g.,
# already used by other jobs)", QOSMaxJobsPerUserLimit and AssocMaxJobsLimit (the limit
# on jobs running at a given time), BeginTime, Dependency "has not been satisfied", None
# "has not been considered yet in the current backfill cycle" `[PARTIAL: these seven
# entries, as quoted by a summarising fetch of the page; the rest of the list not read]`
: "${REGBENCH_STAGE_WAIT_REASONS=Priority Resources QOSMaxJobsPerUserLimit AssocMaxJobsLimit BeginTime Dependency None}"
HERE="$REGBENCH_REPO/benchmarks/regbench/slurm"
state="${layout%.tsv}"
fed="$state.fed.tsv"
prefix="rb-$name-$dataset-tile-"
me="$(id -un)"
ids_file="${REGBENCH_IDS_FILE:-$REGBENCH_OUT/logs/submitted/${REGBENCH_SUBMISSION:-none}.ids}"
mkdir -p "$(dirname "$ids_file")"

end() {  # end <status> <cause|-> : the last line of the log
    local cause=""
    [ "$2" = - ] || cause=" cause=$2 whose=launcher"
    regbench_say_end "REGBENCH_END kind=submitter run=$name dataset=$dataset link=$link status=$1$cause log=${REGBENCH_LOG:--}"
}
regbench_start submitter "run=$name" "label=$label" "dataset=$dataset" "link=$link"

if [ -e "$state.done" ]; then
    echo "$name on $dataset: every tile job was submitted and has left the queue ($state.done); nothing to do"
    end ok -
    exit 0
fi
[ -s "$layout" ] || { echo "no layout file $layout: the launcher writes it"; end failed not_submitted; exit 1; }

# the caller's own sbatch arguments (account, partition, ...), one per line
extra=()
if [ -s "$state.sbatch_args" ]; then
    while IFS= read -r line; do extra+=("$line"); done < "$state.sbatch_args"
fi

# when this chain of submitters first started: no array of it is older
[ -s "$state.started" ] || date +%Y-%m-%dT%H:%M:%S > "$state.started"
started="$(head -n 1 "$state.started")"

# queued: how many tile tasks of this run and dataset are pending or running; -1 when
# the queue cannot be read
queued() {
    local names
    if ! names="$(squeue -h -r -u "$me" -o '%j' 2>/dev/null)"; then
        echo -1
        return 0
    fi
    # grep -c prints 0 and exits 1 for no match: the count is printed either way
    local n
    n="$(grep -c "^$prefix" <<< "$names")" || n=0
    echo "$n"
}
# gave_up <reason>: the marker a round reads (stages.feed_problem), the end line, exit 1
gave_up() {
    {
        echo "time: $(date +%FT%T)"
        echo "run: $name  dataset: $dataset  submission: ${REGBENCH_SUBMISSION:-}  link: $link"
        echo "gave up: $1"
        echo "tile tasks of this run still queued or running: ${2:-unknown}"
        echo "they are left in the queue; every job id of this submission, one per line: $ids_file"
        echo "cancel them: scancel \$(cat \"$ids_file\")"
        echo "go on: the next link of the chain, or submit again (submit_cohort.sh); what is ok is reused"
    } > "$state.refused"
    cat "$state.refused"
    end refused not_submitted
    exit 1
}
# look: sets Q to the count; 1 when the queue could not be read (said, and counted: after
# REGBENCH_STAGE_SQUEUE_TRIES in a row the submitter gives up)
Q=0 unreadable=0
look() {
    Q="$(queued)"
    if [ "$Q" -lt 0 ]; then
        unreadable=$((unreadable + 1))
        echo "$(date +%FT%T) the queue could not be read (squeue failed), look $unreadable of at most $REGBENCH_STAGE_SQUEUE_TRIES in a row: nothing is submitted on a queue that cannot be read"
        [ "$unreadable" -ge "$REGBENCH_STAGE_SQUEUE_TRIES" ] \
            && gave_up "the queue could not be read $unreadable times in a row, ${REGBENCH_STAGE_POLL_S}s apart (squeue -h -r -u $me failed)"
        return 1
    fi
    unreadable=0
    return 0
}
# moved <count>: a tile task left the queue, or one was just put in: the wait starts anew
# (`turn`: the seconds of the wait in which every task left was only waiting its turn)
low=-1 since=$SECONDS first=$SECONDS turn=0
moved() { low="$1"; since=$SECONDS; first=$SECONDS; turn=0; }
# waiting <what for>: called with Q at every look of a wait. Nothing for
# REGBENCH_STAGE_STALL_S seconds: say what is still in the queue and why, give up on a
# task that does not leave by itself or after REGBENCH_STAGE_STALL_MAX_S, else wait on.
waiting() {
    local what="$1" left held="" reason stretch others
    if [ "$low" -lt 0 ] || [ "$Q" -lt "$low" ]; then
        moved "$Q"
        return 0
    fi
    [ $((SECONDS - since)) -ge "$REGBENCH_STAGE_STALL_S" ] || return 0
    stretch=$((SECONDS - since))
    since=$SECONDS
    if ! left="$(squeue -h -r -u "$me" -o '%i|%j|%T|%r' 2>/dev/null)"; then
        echo "$(date +%FT%T) $what: no tile task has left the queue for $((SECONDS - first))s, and what is left could not be listed (squeue failed)"
        return 0
    fi
    left="$(awk -F'|' -v p="$prefix" 'index($2, p) == 1' <<< "$left")"
    echo "$(date +%FT%T) $what: $Q tile task(s) in the queue and none has left it for $((SECONDS - first))s. Still there (job id|name|state|reason):"
    echo "${left:-  (none listed)}"
    for reason in $REGBENCH_STAGE_HELD_REASONS; do
        grep -q "|$reason\$" <<< "$left" && held="$held $reason"
    done
    [ -n "$held" ] && gave_up "$what: $Q tile task(s) have not left the queue for $((SECONDS - first))s and at least one does not leave by itself (reason:$held); see the log for the list" "$Q"
    # the reasons that are not "waiting its turn", among the tasks listed
    others="$(awk -F'|' -v ok=" $REGBENCH_STAGE_WAIT_REASONS " 'index(ok, " " $4 " ") == 0 { print $4 }' <<< "$left" | sort -u | paste -sd' ' -)"
    if [ -n "$left" ] && [ -z "$others" ]; then
        turn=$((turn + stretch))
        echo "$(date +%FT%T) $what: every one of them is waiting its turn (REGBENCH_STAGE_WAIT_REASONS: $REGBENCH_STAGE_WAIT_REASONS); these ${stretch}s are not counted, waiting on (${turn}s of the $((SECONDS - first))s so far were such)"
        return 0
    fi
    [ $((SECONDS - first - turn)) -ge "$REGBENCH_STAGE_STALL_MAX_S" ] \
        && gave_up "$what: no tile task has left the queue for $((SECONDS - first))s, $((SECONDS - first - turn))s of them not merely waiting its turn (REGBENCH_STAGE_STALL_MAX_S=$REGBENCH_STAGE_STALL_MAX_S; reasons: ${others:-none listed}); see the log for the list" "$Q"
    echo "$(date +%FT%T) $what: none of them is held; waiting on (at most $REGBENCH_STAGE_STALL_MAX_S s without one leaving, the time in which all only wait their turn not counted)"
    return 0
}
# drain <what for>: wait until none is left
drain() {
    low=-1
    while :; do
        if look; then
            [ "$Q" = 0 ] && return 0
            waiting "$1"
        fi
        sleep "$REGBENCH_STAGE_POLL_S"
    done
}

refused() {  # refused <attempt> <what> <the scheduler's words>
    local q
    q="$(queued)"
    {
        echo "time: $(date +%FT%T)"
        echo "run: $name  dataset: $dataset  submission: ${REGBENCH_SUBMISSION:-}  attempt: $1"
        echo "refused: $2"
        echo "scheduler: $3"
        echo "tile tasks of this run still queued or running: $q"
        echo "every job id of this submission, one per line: $ids_file"
        echo "cancel them: scancel \$(cat \"$ids_file\")"
        echo "go on: submit again (submit_cohort.sh); what is ok is reused, and the tiles that were never submitted are"
    } > "$state.refused"
    cat "$state.refused"
    end refused not_submitted
    exit 1
}

# the_due <attempt>: writes $state.due.a<attempt> once (tile indices over every round's
# tiles laid end to end for attempt 1, round indices above it)
the_due() {
    local attempt="$1" file="$state.due.a$1" args items o dropped rc
    [ -e "$file" ] && return 0
    args=(stages --cases "$(regbench_cases_of "$dataset")" --dataset "$dataset" --method "$method"
          --layout "$layout" --out "$REGBENCH_OUT" --due "$attempt")
    [ "$label" != - ] && args+=(--label "$label")
    if [ "$opts" != - ]; then
        IFS=, read -ra items <<< "$opts"
        for o in "${items[@]}"; do args+=(--opt "$o"); done
    fi
    if regbench_py "$image" "${args[@]}" > "$file.tmp"; then
        # an index per line and nothing else: a line that is not one (a warning some
        # library wrote among them) is no tile, and would be an array no scheduler takes
        # exit 1 is "no line is an index" (an empty list); only 2 is grep's own error
        rc=0
        grep -E '^[0-9]+$' "$file.tmp" > "$file.clean" || rc=$?
        if [ "$rc" -ge 2 ]; then
            rm -f "$file.clean" "$file.tmp"
            gave_up "attempt $attempt: the list of what is due could not be read (grep exited $rc); nothing of this attempt was submitted"
        fi
        dropped=$(( $(wc -l < "$file.tmp") - $(wc -l < "$file.clean") ))
        [ "$dropped" = 0 ] || echo "attempt $attempt: $dropped line(s) of the list of what is due were not an index and were dropped (the first: $(grep -vE '^[0-9]+$' "$file.tmp" | head -n 1))" >&2
        mv "$file.clean" "$file"
        rm -f "$file.tmp"
    else
        # the records could not be read: everything is submitted, and a task with
        # nothing to do exits at once
        echo "attempt $attempt: cannot list what is due (python -m benchmarks.regbench stages --due); submitting every $([ "$attempt" = 1 ] && echo tile || echo round)"
        if [ "$attempt" = 1 ]; then seq 0 $((n_tiles - 1)); else seq 0 $((n_rounds - 1)); fi > "$file"
        rm -f "$file.tmp"
    fi
}

# was_submitted <job name>: is a job of that name in the queue or in the accounting
was_submitted() {
    local found
    found="$(squeue -h -u "$me" -n "$1" -o '%A' 2>/dev/null | head -n 1)"
    [ -n "$found" ] && { echo "$found"; return 0; }
    if command -v sacct >/dev/null; then
        found="$(sacct -n -X -P -S "$started" --name="$1" -o JobID 2>/dev/null | head -n 1)"
        [ -n "$found" ] && { echo "${found%%_*}"; return 0; }
    fi
    return 1
}

# feed <attempt>: every line of the attempt's due file, once, under the ceiling
feed() {
    local attempt="$1" file="$state.due.a$1" total next=0 cpus gb time
    local q room left want need size base spec jobname got try a start count b id tail before
    IFS=: read -r cpus gb time <<< "${ALLOC[$attempt]}"
    total="$(wc -l < "$file" | tr -d ' ')"
    # where an earlier link stopped
    if [ -s "$fed" ]; then
        before="$(cat "$fed")"
        while IFS=$'\t' read -r a start count b id; do
            [ "$a" = "$attempt" ] || continue
            if [ "$id" = - ]; then
                # written before an sbatch whose answer was never recorded
                grep -q "^$a	$start	$count	$b	[0-9]" <<< "$before" && continue
                jobname="$prefix""a$attempt-o$start"
                if id="$(was_submitted "$jobname")"; then
                    printf '%s\t%s\t%s\t%s\t%s\n' "$a" "$start" "$count" "$b" "$id" >> "$fed"
                else
                    echo "attempt $attempt: no job named $jobname in the queue or the accounting: that array is submitted again"
                    continue
                fi
            fi
            [ $((start + count)) -gt "$next" ] && next=$((start + count))
        done <<< "$before"
    fi
    low=-1
    while [ "$next" -lt "$total" ]; do
        if ! look; then
            sleep "$REGBENCH_STAGE_POLL_S"
            continue
        fi
        q="$Q"
        room=$((REGBENCH_STAGE_QUEUE_RUN - q))
        left=$((total - next))
        want="$REGBENCH_STAGE_ARRAY_MAX"
        [ "$want" -gt "$left" ] && want="$left"
        need="$REGBENCH_STAGE_FEED_MIN"
        [ "$need" -gt "$want" ] && need="$want"
        [ "$need" -gt "$REGBENCH_STAGE_QUEUE_RUN" ] && need="$REGBENCH_STAGE_QUEUE_RUN"
        if [ "$room" -lt "$need" ]; then
            waiting "attempt $attempt, waiting for $need free slot(s) of $REGBENCH_STAGE_QUEUE_RUN"
            sleep "$REGBENCH_STAGE_POLL_S"
            continue
        fi
        size="$want"
        [ "$size" -gt "$room" ] && size="$room"
        # the array's indices: for tiles, counted from the first one (an array index
        # stays below the site's MaxArraySize, so an array never spans more than
        # REGBENCH_STAGE_ARRAY_SPAN indices); for rounds, the round itself
        base=0
        [ "$attempt" = 1 ] && base="$(sed -n "$((next + 1))p" "$file")"
        read -r size spec < <(sed -n "$((next + 1)),$((next + size))p" "$file" | awk -v b="$base" -v span="$REGBENCH_STAGE_ARRAY_SPAN" '
            !/^[0-9]+$/ { bad = 1; exit }
            { v = $1 - b; if (v < 0) { bad = 1; exit }
              if (v >= span) exit
              n++
              if (n == 1) { s = v; p = v; next }
              if (v == p + 1) { p = v; next }
              out = out (s == p ? s : s "-" p) ","; s = v; p = v }
            END { if (bad) print "none", "none"; else print n, out (s == p ? s : s "-" p) }')
        jobname="$prefix""a$attempt-o$next"
        # an array that is none (a line of the list that is no index) is never
        # submitted, and never tried again: without a size the loop would not advance
        case "$size:$base" in *[!0-9:]*|:*|*:|0:*)
            gave_up "attempt $attempt: line $((next + 1)) of $file gives no array (size '${size:-}', first index '${base:-}'): the list of what is due is not a list of indices" "$q" ;;
        esac
        case "$spec" in ''|*[!0-9,-]*)
            gave_up "attempt $attempt: lines $((next + 1)) to $((next + size)) of $file give the array '--array=${spec:-}', which is none: the list of what is due is not a list of indices" "$q" ;;
        esac
        tail=()
        [ "$attempt" = 1 ] && tail=("$layout" "$base")
        printf '%s\t%s\t%s\t%s\t-\n' "$attempt" "$next" "$size" "$base" >> "$fed"
        try=0 got=""
        while :; do
            try=$((try + 1))
            if got="$(sbatch --parsable --job-name="$jobname" --export=ALL --no-requeue ${extra[@]+"${extra[@]}"} \
                    --output="$REGBENCH_OUT/logs/%x-%A_%a.log" --array="$spec" \
                    --cpus-per-task="$cpus" --mem="${gb}G" --time="$time" \
                    "$HERE/stage.sbatch" "$name" "$method" "$label" "$opts" "$image" "$dataset" \
                    tile "$attempt" "$gb" "$time" ${tail[@]+"${tail[@]}"} < /dev/null 2>&1)"; then
                break
            fi
            echo "sbatch refused $jobname (try $try of $REGBENCH_STAGE_SUBMIT_TRIES): $got"
            [ "$try" -ge "$REGBENCH_STAGE_SUBMIT_TRIES" ] && refused "$attempt" "$jobname, $size task(s), --array=$spec" "$got"
            sleep "$REGBENCH_STAGE_POLL_S"
        done
        # the id is the LAST line: a site's submit plugin may print lines before it
        id="$(awk 'NF { last = $0 } END { print last }' <<< "$got")"
        [ "$id" = "$got" ] || awk -v id="$id" '$0 != id && NF { print "sbatch said: " $0 }' <<< "$got"
        id="${id%%;*}"
        # no job id is no job: nothing of it is written down as submitted
        case "$id" in ''|*[!0-9_]*) refused "$attempt" "$jobname, $size task(s), --array=$spec" "sbatch printed no job id: $got" ;; esac
        printf '%s\t%s\t%s\t%s\t%s\n' "$attempt" "$next" "$size" "$base" "$id" >> "$fed"
        printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$jobname" "$id" >> "$REGBENCH_OUT/logs/jobs.tsv"
        echo "$id" >> "$ids_file"
        echo "attempt $attempt: $jobname -> $id, $size task(s) (--array=$spec), $q in the queue before"
        next=$((next + size))
        moved $((q + size))
    done
}

for attempt in 1 2 3 4; do
    # what is due at an attempt is read off the records once the attempt before has
    # left the queue; a link that finds the list written goes on feeding from it
    [ -e "$state.due.a$attempt" ] || drain "before attempt $attempt is listed"
    the_due "$attempt"
    if [ ! -s "$state.due.a$attempt" ]; then
        echo "attempt $attempt: nothing is due"
        [ "$attempt" -gt 1 ] && break
        continue
    fi
    feed "$attempt"
done
drain "after the last attempt"
rm -f "$state.refused"
date +%FT%T > "$state.done"
echo "$name on $dataset: every tile job was submitted and has left the queue"
end ok -
exit 0
