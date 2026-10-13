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
#   retry.sbatch <run|-> <plan folder> <first job id> <first job name> <array range a-b|->
#
# A range of `-`: the chain is of SINGLE jobs, not arrays (a selfcheck). Then a step is
# submitted without --array, the end line is <job name>-<job id>.end, and this job
# exits 0 only when the chain's last job ended ok (what waits for a selfcheck with
# `afterok` waits for this job). A run of `-`: the chain runs before its run's gate
# (a selfcheck), so there is no verdict to obey yet.
#
# THE PLAN is written by submit_cohort.sh: <plan folder>/step-<k>, one per step in the
# order the chain runs, one item per line:
#   line 1    the step's kind: `again` (the same rung once more), `climb` (the next rung),
#             or anything else (`stage`, `notok`: the next attempt of a stage job or of
#             a selfcheck)
#   line 2    the job name
#   line 3..  the sbatch arguments (no --array, no --dependency), then the job script
#             and its arguments. The item `@EARLIER@` is replaced by the ids of every
#             job of this chain submitted so far (run_rung.sbatch asks the scheduler's
#             accounting about them), and `@PREVIOUS@` by the id of the last one.
#
# WHICH UNITS A STEP IS SUBMITTED FOR. Each task leaves an end line in
# $REGBENCH_OUT/logs/end/<job name>-<job id>_<index>.end (env.sh). The unit's LAST task
# decides:
#   no end line     the task died or never started: every kind of step takes it
#   status=ok       nothing more, ever
#   `again`         cause killed or infrastructure
#   `climb`         a memory cause, killed or infrastructure
#   any other kind  anything that is not ok
# (ladder.py: AGAIN_CAUSES, CLIMB_CAUSES; a test keeps the lists below equal to them.)
# This is a FILTER, not the decision: the task that is submitted still decides for
# itself, exactly as before (run_rung.sbatch, stage.sbatch), and exits at once when its
# unit has nothing to do. The filter only keeps a unit that cannot need the step out of
# the queue.
#
# It gives up (exit 1, the rest of the chain is not submitted) when sbatch refuses a
# step or the queue cannot be read REGBENCH_RETRY_SQUEUE_TRIES (30) times in a row.
set -uo pipefail
source "$REGBENCH_REPO/benchmarks/regbench/slurm/env.sh"
name="$1" plan="$2" first_id="$3" first_name="$4" range="$5"
: "${REGBENCH_RETRY_POLL_S:=60}"
: "${REGBENCH_RETRY_SQUEUE_TRIES:=30}"
: "${REGBENCH_ARRAY_LIMIT:=20}"
AGAIN_CAUSES="killed infrastructure"
CLIMB_CAUSES="oom writer_oom benchmark_oom killed infrastructure"
ends="$REGBENCH_OUT/logs/end"
ids_file="${REGBENCH_IDS_FILE:-$REGBENCH_OUT/logs/submitted/${REGBENCH_SUBMISSION:-none}.ids}"
mkdir -p "$(dirname "$ids_file")"

end() {  # end <status> <cause|-> : the last line of the log
    local cause=""
    [ "$2" = - ] || cause=" cause=$2 whose=launcher"
    echo "REGBENCH_END kind=retry run=$name chain=$first_name status=$1$cause log=${REGBENCH_LOG:--}"
}

# the second lock behind the dependency on the gate job, as in run_rung.sbatch
gate="$REGBENCH_OUT/logs/gates/$name.tsv"
status=not_checked blocked=1
if [ -s "$gate" ]; then
    IFS=$'\t' read -r _ status _ blocked _ < "$gate"
fi
[ "$name" = - ] && blocked=0
if [ "$blocked" != 0 ]; then
    echo "$name: the gate says $status and the run is not allowed to start unchecked; no retry is submitted"
    end blocked gate_blocked
    exit 0
fi

# last: "<index> <job name>-<job id>" per unit, the last job that held the unit
last="$plan/last"
single=0
[ "$range" = - ] && single=1
lo="${range%%-*}" hi="${range#*-}"
if [ ! -s "$last" ]; then
    { if [ "$single" = 1 ]; then echo -; else seq "$lo" "$hi"; fi; } \
        | awk -v j="$first_name-$first_id" '{ print $1, j }' > "$last" \
        || { echo "cannot write $last"; end failed not_submitted; exit 1; }
fi
earlier="$first_id" previous="$first_id"
# end_file <index> <job name>-<job id>: where that task left its end line
end_file() {
    if [ "$single" = 1 ]; then echo "$ends/$2.end"; else echo "$ends/${2}_$1.end"; fi
}

# wanted <kind>: the indices the step is submitted for, one per line
wanted() {
    local kind="$1" idx job file st cause causes
    case "$kind" in again) causes=" $AGAIN_CAUSES " ;; climb) causes=" $CLIMB_CAUSES " ;; *) causes="" ;; esac
    while read -r idx job; do
        file="$(end_file "$idx" "$job")"
        if [ ! -s "$file" ]; then echo "$idx"; continue; fi
        st="$(sed -nE 's/.* status=([^ ]+).*/\1/p' "$file" | tail -n 1)"
        cause="$(sed -nE 's/.* cause=([^ ]+).*/\1/p' "$file" | tail -n 1)"
        [ "$st" = ok ] && continue
        if [ -z "$causes" ]; then echo "$idx"; continue; fi
        case "$causes" in *" $cause "*) echo "$idx" ;; esac
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

n_steps=0 n_sent=0
k=1
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
    args=() array="-"
    if [ "$single" != 1 ]; then
        array="$(spec <<< "$due")"
        args+=(--array="$array%$REGBENCH_ARRAY_LIMIT")
    fi
    for item in "${items[@]:2}"; do
        [ "$item" = "@EARLIER@" ] && item="$earlier"
        [ "$item" = "@PREVIOUS@" ] && item="$previous"
        args+=("$item")
    done
    if ! got="$(sbatch --parsable --job-name="$jobname" "${args[@]}" < /dev/null 2>&1)"; then
        echo "sbatch REFUSED $jobname (--array=$array): $got"
        end failed not_submitted
        exit 1
    fi
    id="$(awk 'NF { last = $0 } END { print last }' <<< "$got")"
    [ "$id" = "$got" ] || awk -v id="$id" '$0 != id && NF { print "sbatch said: " $0 }' <<< "$got"
    id="${id%%;*}"
    case "$id" in ''|*[!0-9_]*) echo "sbatch printed no job id for $jobname: $got"; end failed not_submitted; exit 1 ;; esac
    echo "$id" >> "$ids_file"
    printf '%s\t%s\t%s\n' "$(date +%FT%T)" "$jobname" "$id" >> "$REGBENCH_OUT/logs/jobs.tsv"
    n_sent=$((n_sent + 1))
    echo "step $k ($jobname, $kind): submitted as $id for $(wc -l <<< "$due" | tr -d ' ') unit(s): --array=$array"
    earlier="$earlier,$id" previous="$id"
    # these units' last job is now this one
    awk -v j="$jobname-$id" 'NR == FNR { due[$1] = 1; next } ($1 in due) { print $1, j; next } { print }' \
        <(echo "$due") "$last" > "$last.new" && mv "$last.new" "$last"
    if ! gone "$id"; then
        echo "the queue could not be read $REGBENCH_RETRY_SQUEUE_TRIES times in a row: giving up; $jobname ($id) is left in the queue and the steps after it are not submitted"
        end failed not_submitted
        exit 1
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
        end failed -
        exit 1
    fi
fi
end ok -
exit 0
