# shellcheck shell=bash
# Sourced by every regbench sbatch script. Export what you need before submitting.
#
#   REGBENCH_REPO    the mirage repository (default: where this file is)
#   REGBENCH_CASES   the prepared cases (`python -m benchmarks.regbench prepare ...`)
#   REGBENCH_CASES_<NAME>  the prepared cases of the public dataset <name>
#                    (regbench.PUBLIC; the name in capitals, e.g. REGBENCH_CASES_MCMICRO):
#                    read only when `submit_cohort.sh --datasets` names it
#                    (regbench_cases_of), and bound into the images when set
#   REGBENCH_OUT     results: one folder per result name, plus tables/, logs/ and cache/
#   REGBENCH_WORK    scratch for a method's intermediates (default: $TMPDIR, node-local)
#   REGBENCH_BIND    more paths to bind into the images, comma-separated: wherever the
#                    cohort's slides and nuclei live
#
# THE IMAGES. Each run names the image it runs in (`python -m benchmarks.regbench runs`,
# column `image`); the path of image X is REGBENCH_SIF_X:
#
#   REGBENCH_SIF_STARE_REGBENCH   containers/stare-regbench: STARE, and where `prepare`,
#                                 `verdict`, `tissue` and `score` run (variants.SCORER_IMAGE)
#   REGBENCH_SIF_VALIS_MIRAGE     containers/valis100 (the pipeline's VALIS 1.0.0 image + codecs)
#   REGBENCH_SIF_VALIS120         containers/valis120 (VALIS 1.2.0)
#   REGBENCH_SIF_DEEPERHISTREG    containers/deeperhistreg
#   REGBENCH_SIF_ASHLAR           containers/ashlar120 (labsyspharm/ashlar:1.20.0 + imagecodecs)
#   REGBENCH_SIF_PALOM            containers/palom
#   REGBENCH_SIF_STARDIST         the pipeline's StarDist image (bolt3x/mirage-stardist:1.0.0):
#                                 only for the re-run of the bare nuclei (nuclei.sbatch)
#
# REGBENCH_SITE_CONFIG (default: conf/ieo.config of the checkout, if it is there) is where
# that re-run reads the site's StarDist model and settings when REGBENCH_STARDIST_MODEL
# and REGBENCH_STARDIST_MODEL_DIR are not exported (benchmarks/regbench/site_config.py).
#
# In place of an image, REGBENCH_CONDA_X (a conda environment) or REGBENCH_VENV_X (a
# virtualenv) is taken; with none of the three, the `python` on PATH.
#
#   REGBENCH_WRITABLE_TMPFS_IMAGES   images started with --writable-tmpfs (default:
#                                    VALIS120, whose package rewrites a file of its own
#                                    at every JVM start)
#
#   REGBENCH_STARE_SRC_IMAGES        images that have no `stare` package inside (default:
#                                    ASHLAR VALIS_MIRAGE, the two pulled as their
#                                    projects publish them). The harness reads every
#                                    image through `stare.slide_io`, so for these the
#                                    source of the release pinned in
#                                    requirements/stare.txt is put on PYTHONPATH: pure
#                                    Python only. The image must have what that code
#                                    imports and what is compiled: numpy, tifffile, zarr
#                                    (with numcodecs), imagecodecs; psutil if the
#                                    monitor is to use it. `regbench preflight` checks
#   REGBENCH_STARE_SRC               where that source is (default
#                                    $REGBENCH_OUT/cache/stare-<tag>/src): fetched once
#                                    by submit_cohort.sh on the login node
#                                    (regbench_fetch_stare), never by a job
#
# INSIDE EVERY IMAGE (`--cleanenv` passes nothing of the host's environment; a variable
# is set inside through APPTAINERENV_<name> / SINGULARITYENV_<name>, the mechanism
# Apptainer documents for it: "Environment and Metadata" of the Apptainer user guide
# `[MEMORY]`, not retrieved while this was written):
#   PYTHONNOUSERSITE=1   the host user's ~/.local/lib/pythonX.Y/site-packages (the home
#                        folder is mounted into the image) cannot shadow a pinned package
#   PYTHONPATH           the checkout, and for REGBENCH_STARE_SRC_IMAGES the STARE source:
#                        nothing of the host's own PYTHONPATH
#   TMPDIR               the job's, when it is a folder: bound and passed on, so that a
#                        method's temporary files go to the node's scratch
#   REGBENCH_WORK, REGBENCH_PRIVATE, REGBENCH_PRIVATE_DIR   as exported (the private
#                        plug-in folder is bound when it is named; a plug-in that is
#                        switched on and cannot be read still stops the run:
#                        benchmarks/regbench/plugins.py)
#
# A GPU is passed into an image (--nv) only for a run whose `gpu` column is 1: the
# launcher exports REGBENCH_GPU for each job. VALIS 1.2.0's image has a CUDA torch
# inside and must never see one.
#
# Every method runs at 4 CPUs on a memory ladder (16 -> 64 -> 256 -> 512 -> 700 GB; a unit moves
# up one rung only when it ran out of memory). submit_cohort.sh owns that; it exports
# REGBENCH_ATTEMPT and REGBENCH_MEM_GB to each task. No thread variable is exported: a
# method's own core-count option is set to the allocation by the driver, and what its
# pools saw is recorded.
#
# REGBENCH_SUBMISSION is the id of one call of submit_cohort.sh. Every record a job
# writes carries it (run.json: submissions; selfcheck.json: submission), and the verdict, the
# ladder and the scorer take no record of another submission as evidence. REGBENCH_FORCE=1
# re-runs units whose results are all ok already (the default reuses them).
#
# REGBENCH_CPUS (4) is every ladder task's CPUs (submit_cohort.sh: 4 or 8; 4 is every
# method's allocation, 8 exists for the CPU cross only, user rulings 2026-10-16 and
# 2026-10-18). REGBENCH_MEM (16G) is exported and read by no job.

: "${REGBENCH_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${REGBENCH_CASES:=$HOME/regbench_cases}"
# the commit the jobs run from: the submitting checkout's (--export=ALL carries it to
# every job), named on each job's `[regbench] start` line
: "${REGBENCH_COMMIT:=$(git -C "$REGBENCH_REPO" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}"
export REGBENCH_COMMIT
# whether REGBENCH_OUT is the commit-hash default (submit_cohort.sh warns when an earlier
# default-named folder holds results of the same cases: a new commit is a new folder)
# (read by submit_cohort.sh, which sources this file)
REGBENCH_OUT_DEFAULTED=0
[ -n "${REGBENCH_OUT:-}" ] || REGBENCH_OUT_DEFAULTED=1
export REGBENCH_OUT_DEFAULTED
REGBENCH_RESULTS_HOME="${HOME:-}/regbench_results"
: "${REGBENCH_OUT:=$REGBENCH_RESULTS_HOME/mirage-$(git -C "$REGBENCH_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
: "${REGBENCH_CPUS:=4}"
: "${REGBENCH_MEM:=16G}"
: "${REGBENCH_WRITABLE_TMPFS_IMAGES:=VALIS120}"
: "${REGBENCH_STARE_SRC_IMAGES:=ASHLAR VALIS_MIRAGE}"
export REGBENCH_REPO REGBENCH_CASES REGBENCH_OUT REGBENCH_CPUS REGBENCH_MEM
export REGBENCH_STARE_SRC_IMAGES

# regbench_cases_of <dataset>: the folder the dataset's cases are prepared under. The
# cohort's are REGBENCH_CASES; a public dataset's (regbench.PUBLIC) REGBENCH_CASES_<NAME>,
# by the dataset's base name (`mcmicro@raw` is MCMICRO). Empty, with exit 1, when that is
# not set: the caller says so.
regbench_cases_of() {
    local base="${1%%@*}" var
    if [ "$1" = multiplex ]; then
        printf '%s\n' "$REGBENCH_CASES"
        return 0
    fi
    var="REGBENCH_CASES_$(tr '[:lower:]' '[:upper:]' <<< "$base")"
    [ -n "${!var:-}" ] || return 1
    printf '%s\n' "${!var}"
}
# the case folders of this submission's public datasets (REGBENCH_PUBLIC_SETS, which the
# launcher exports: those --datasets names, and only those), comma-separated, to bind
# into the images. A REGBENCH_CASES_<NAME> of a dataset not named is never bound (a stale
# one in a profile would stop every container job); a named one that is not set or not
# there is said on stderr and left out (the job that reads it stops on it, naming it).
_regbench_public_cases() {
    local ds dir out=""
    for ds in ${REGBENCH_PUBLIC_SETS:-}; do
        if ! dir="$(regbench_cases_of "$ds")"; then
            echo "regbench: REGBENCH_CASES_$(tr '[:lower:]' '[:upper:]' <<< "${ds%%@*}") not set for $ds: its cases are not bound" >&2
        elif [ ! -d "$dir" ]; then
            echo "regbench: the cases folder of $ds, $dir, does not exist: not bound" >&2
        else
            out="${out:+$out,}$dir"
        fi
    done
    printf '%s' "$out"
}

# The harness is imported from the checkout as `benchmarks.regbench`. STARE itself is the
# pinned release installed in the method's environment (requirements/stare.txt).
export PYTHONPATH="$REGBENCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
REGBENCH_MODULE=benchmarks.regbench

# The STARE release the checkout pins (requirements/stare.txt), as source, for the images
# that have no `stare` inside (REGBENCH_STARE_SRC_IMAGES).
regbench_stare_url() {
    grep -oE 'https://github\.com/sceriff0/stare/archive/refs/tags/[^[:space:]]+\.tar\.gz' \
        "$REGBENCH_REPO/requirements/stare.txt" | head -n 1
}
regbench_stare_src() {
    local url
    url="$(regbench_stare_url)"
    echo "${REGBENCH_STARE_SRC:-$REGBENCH_OUT/cache/stare-$(basename "$url" .tar.gz)/src}"
}
# regbench_fetch_stare: download that source once (the login node: a job may have no
# network). Prints the folder; non-zero when it is not there afterwards.
regbench_fetch_stare() {
    local url src tmp
    url="$(regbench_stare_url)"
    src="$(regbench_stare_src)"
    [ -n "$url" ] || { echo "no STARE release URL in $REGBENCH_REPO/requirements/stare.txt" >&2; return 1; }
    if [ ! -f "$src/stare/slide_io.py" ]; then
        mkdir -p "$(dirname "$src")" || return 1
        tmp="$(mktemp -d "$(dirname "$src")/dl.XXXXXX")" || return 1
        if curl -fsSL "$url" | tar -xz -C "$tmp" --strip-components=1 && [ -d "$tmp/src" ]; then
            # losing the race to another job that fetched the same release is fine:
            # it left $src, and that is what is asked next
            mv -n "$tmp/src" "$src" 2>/dev/null || [ -d "$src" ]
        fi
        rm -rf "$tmp"
    fi
    [ -f "$src/stare/slide_io.py" ] || { echo "could not fetch STARE from $url into $src" >&2; return 1; }
    echo "$src"
}

_regbench_var() {  # _regbench_var SIF valis120 -> value of REGBENCH_SIF_VALIS120
    local name
    name="REGBENCH_$1_$(echo "$2" | tr '[:lower:]' '[:upper:]')"
    echo "${!name:-}"
}

# ── the task's log, and its last line ────────────────────────────────────────────────
# REGBENCH_LOG is this task's scheduler log, as submit_cohort.sh names it (%x-%A_%a.log
# for an array task, %x-%j.log otherwise). It goes into run.json and selfcheck.json
# (`log_path`), and from there into logs/verdicts.tsv, so the scorer's failure log can
# point at it.
# REGBENCH_END_FILE is where the Python step leaves its end line; regbench_end prints it
# as the LAST line of the log, or a fallback when the step died before writing one:
#   REGBENCH_END kind=... run=... status=... cause=... whose=... log=...
# (benchmarks/regbench/failure_log.py has the vocabulary; `whose` is method, benchmark,
# infrastructure or launcher.)
regbench_task_log() {
    [ -n "${REGBENCH_OUT:-}" ] || return 0
    local id="${SLURM_JOB_ID:-}" name="${SLURM_JOB_NAME:-}"
    [ -n "$id" ] && [ -n "$name" ] || return 0
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    export REGBENCH_LOG="$REGBENCH_OUT/logs/$name-$id.log"
    export REGBENCH_END_FILE="$REGBENCH_OUT/logs/end/$name-$id.end"
    mkdir -p "$REGBENCH_OUT/logs/end" 2>/dev/null && rm -f "$REGBENCH_END_FILE"
}
regbench_task_log

# ── what the operator greps: one start line and one end line per job ────────────────
# Every regbench job prints, in the same format whatever it is:
#   [regbench] start kind=<kind> job=<name>-<id> run=... label=... dataset=... unit=...
#              rung_gb=... attempt=... out=<REGBENCH_OUT> commit=<sha> host=... at=<time>
#   [regbench] end kind=<kind> job=<name>-<id> outcome=<ok|status[/cause]> wall=<s>s
#              peak_rss=<MB>MB|- record=<path>|- exit=<code>
# (one line each; a field the job does not have is `-`). The end line comes just BEFORE
# the job's REGBENCH_END line, which stays the log's last. A job that ends without one
# (a refusal under `set -e`, a signal) still prints its end line, from the EXIT trap.
#   grep -h '^\[regbench\] end' $REGBENCH_OUT/logs/*.log | grep -v outcome=ok
REGBENCH_JOB_KIND="" REGBENCH_T0="" REGBENCH_SAID_END=0
_regbench_job() {
    local id="${SLURM_JOB_ID:--}"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    echo "${SLURM_JOB_NAME:-local}-$id"
}
# regbench_start <kind> [key=value ...]: the start line; arms the EXIT trap
regbench_start() {
    REGBENCH_JOB_KIND="$1"; shift
    REGBENCH_T0="$(date +%s)"
    local f k v line more=""
    line="[regbench] start kind=$REGBENCH_JOB_KIND job=$(_regbench_job)"
    # the six fields every job has, in this order (`-` when it has none), then its own
    for k in run label dataset unit rung_gb attempt; do
        v=-
        for f in "$@"; do [ "${f%%=*}" = "$k" ] && v="${f#*=}"; done
        line="$line $k=${v:--}"
    done
    for f in "$@"; do
        # shellcheck disable=SC2194  # the fixed fields are the subject; ${f%%=*} is the pattern
        case " run label dataset unit rung_gb attempt " in
            *" ${f%%=*} "*) ;;
            *) more="$more ${f%%=*}=${f#*=}" ;;
        esac
    done
    line="$line$more"
    echo "$line out=${REGBENCH_OUT:--} commit=${REGBENCH_COMMIT:-unknown} host=$(hostname 2>/dev/null || echo -) at=$(date +%FT%T)"
    trap 'regbench_exit_said $?' EXIT
    regbench_telemetry_start "$REGBENCH_JOB_KIND"
}
# the peak memory of this job's cgroup in MB (cgroup v2 memory.peak, v1
# memory.max_usage_in_bytes), or `-` when it cannot be read here
_regbench_peak_mb() {
    local path f
    path="$(awk -F: '$1 == "0" || $2 ~ /memory/ { print $3; exit }' /proc/self/cgroup 2>/dev/null)" || path=""
    for f in "/sys/fs/cgroup$path/memory.peak" "/sys/fs/cgroup/memory$path/memory.max_usage_in_bytes"; do
        if [ -n "$path" ] && [ -r "$f" ]; then
            awk '{ printf "%d\n", $1 / 1048576 }' "$f" && return 0
        fi
    done
    echo -
}
# ── what the node was doing: one file per run task and stage job (R14, R16) ────────
# <out>/logs/telemetry/<launch id>.tsv, key<TAB>value lines (benchmarks/regbench/
# telemetry.py reads them at table time; they are in no result's key): the load and the
# node's other running jobs at the start and the end, the CPU model, the CPUs the shell
# may use, the job cgroup's memory peak at the end, and the largest sum of the cgroup's
# processes' VmHWM, sampled every REGBENCH_HWM_POLL_S (1) seconds by a loop of this
# shell. Only for a scheduler job (SLURM_JOB_ID) of kind run or stage, and never a
# reason to fail: REGBENCH_TELEMETRY=0 turns it off. REGBENCH_PROC (/proc) is where
# the load, the CPU and the cgroup are read (a test points it at a folder of its own).
REGBENCH_TEL_FILE="" REGBENCH_HWM_PID=""
_regbench_launch_id() {
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    else
        echo "${SLURM_JOB_ID:-}"
    fi
}
# the job cgroup's folder, or nothing
_regbench_cgroup_dir() {
    local path
    path="$(awk -F: '$1 == "0" { print $3; exit }' "${REGBENCH_PROC:-/proc}"/self/cgroup 2>/dev/null)" || path=""
    [ -n "$path" ] && [ -d "/sys/fs/cgroup$path" ] && echo "/sys/fs/cgroup$path"
}
# the jobs RUNNING on this node, this one left out (`-` when squeue cannot say)
_regbench_other_jobs() {
    local mine got
    command -v squeue >/dev/null 2>&1 || { echo -; return 0; }
    mine="$(_regbench_launch_id)"
    local wait=()
    command -v timeout >/dev/null 2>&1 && wait=(timeout 20)  # never a hung job's end
    got="$(${wait[@]+"${wait[@]}"} squeue -h -w "$(hostname -s 2>/dev/null || hostname)" -t RUNNING -o '%i' 2>/dev/null)" \
        || { echo -; return 0; }
    awk -v a="$mine" -v b="${SLURM_JOB_ID:-}" 'NF && $1 != a && $1 != b { n++ } END { print n + 0 }' <<< "$got"
}
_regbench_tel() { [ -n "$REGBENCH_TEL_FILE" ] && printf '%s\t%s\n' "$1" "$2" >> "$REGBENCH_TEL_FILE" 2>/dev/null; return 0; }
# the sampling loop: the largest sum of VmHWM (kB) over the cgroup's processes, in MB
_regbench_hwm_loop() {
    local procs="$1" into="$2" peak=0 now n=0
    while :; do
        # a process that ended between the listing and the read is skipped (cat's error),
        # never the reason the whole sample is lost
        # shellcheck disable=SC2016  # $2 is awk's field, not the shell's
        now="$(sed 's|^|/proc/|; s|$|/status|' "$procs" 2>/dev/null \
            | xargs cat 2>/dev/null | awk '/^VmHWM:/ { s += $2 } END { print s + 0 }')" || now=0
        n=$((n + 1))
        case "$now" in ''|*[!0-9]*) now=0 ;; esac
        [ "$now" -gt "$peak" ] && peak="$now"
        printf '%s\t%s\n' "$((peak / 1024))" "$n" > "$into.new" 2>/dev/null && mv "$into.new" "$into" 2>/dev/null
        sleep "${REGBENCH_HWM_POLL_S:-1}"
    done
}
regbench_telemetry_start() {
    local kind="$1" id dir cg
    REGBENCH_TEL_FILE=""
    [ "${REGBENCH_TELEMETRY:-1}" != 0 ] || return 0
    case "$kind" in run|stage) ;; *) return 0 ;; esac
    [ -n "${REGBENCH_OUT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ] && [ -r "${REGBENCH_PROC:-/proc}"/loadavg ] || return 0
    id="$(_regbench_launch_id)"
    dir="$REGBENCH_OUT/logs/telemetry"
    mkdir -p "$dir" 2>/dev/null || return 0
    REGBENCH_TEL_FILE="$dir/$id.tsv"
    : > "$REGBENCH_TEL_FILE" 2>/dev/null || { REGBENCH_TEL_FILE=""; return 0; }
    _regbench_tel kind "$kind"
    _regbench_tel host "$(hostname 2>/dev/null)"
    _regbench_tel loadavg_start "$(cat "${REGBENCH_PROC:-/proc}"/loadavg 2>/dev/null)"
    _regbench_tel other_jobs_start "$(_regbench_other_jobs)"
    _regbench_tel cpu_model "$(awk -F': ' '/^model name/ { print $2; exit }' "${REGBENCH_PROC:-/proc}"/cpuinfo 2>/dev/null)"
    _regbench_tel nproc "$(nproc 2>/dev/null)"
    _regbench_tel nproc_all "$(nproc --all 2>/dev/null)"
    _regbench_tel cpus_per_task "${SLURM_CPUS_PER_TASK:-}"
    cg="$(_regbench_cgroup_dir)"
    if [ -n "$cg" ] && [ -r "$cg/cgroup.procs" ]; then
        # its output goes nowhere: a caller reading this job's output must not wait on it
        _regbench_hwm_loop "$cg/cgroup.procs" "$REGBENCH_TEL_FILE.hwm" > /dev/null 2>&1 < /dev/null &
        REGBENCH_HWM_PID=$!
    fi
    return 0
}
regbench_telemetry_end() {
    [ -n "$REGBENCH_TEL_FILE" ] || return 0
    local peak
    if [ -n "$REGBENCH_HWM_PID" ]; then
        kill "$REGBENCH_HWM_PID" 2>/dev/null
        wait "$REGBENCH_HWM_PID" 2>/dev/null
        REGBENCH_HWM_PID=""
    fi
    if [ -s "$REGBENCH_TEL_FILE.hwm" ]; then
        _regbench_tel hwm_peak_mb "$(cut -f1 "$REGBENCH_TEL_FILE.hwm")"
        _regbench_tel hwm_samples "$(cut -f2 "$REGBENCH_TEL_FILE.hwm")"
    fi
    rm -f "$REGBENCH_TEL_FILE.hwm" "$REGBENCH_TEL_FILE.hwm.new" 2>/dev/null
    peak="$(_regbench_peak_mb)"
    [ "$peak" = - ] || _regbench_tel cgroup_peak_mb "$peak"
    _regbench_tel loadavg_end "$(cat "${REGBENCH_PROC:-/proc}"/loadavg 2>/dev/null)"
    _regbench_tel other_jobs_end "$(_regbench_other_jobs)"
    REGBENCH_TEL_FILE=""
    return 0
}
# regbench_say_end <REGBENCH_END line>: the `[regbench] end` line read off it, then it
regbench_say_end() {
    local line="$1" status cause record peak code wall="-"
    status="$(sed -nE 's/.* status=([^ ]+).*/\1/p' <<< "$line")"
    cause="$(sed -nE 's/.* cause=([^ ]*).*/\1/p' <<< "$line")"
    record="$(sed -nE 's/.* record=([^ ]+).*/\1/p' <<< "$line")"
    peak="$(sed -nE 's/.* peak_rss_mb=([^ ]+).*/\1/p' <<< "$line")"
    code="$(sed -nE 's/.* exit=([^ ]+).*/\1/p' <<< "$line")"
    [ -n "$peak" ] && [ "$peak" != - ] || peak="$(_regbench_peak_mb)"
    [ "$peak" = - ] || peak="${peak}MB"
    [ -n "$REGBENCH_T0" ] && wall="$(( $(date +%s) - REGBENCH_T0 ))s"
    local outcome="${status:-unknown}"
    [ "$outcome" = ok ] || [ -z "$cause" ] || [ "$cause" = - ] || outcome="$outcome/$cause"
    echo "[regbench] end kind=${REGBENCH_JOB_KIND:-?} job=$(_regbench_job) outcome=$outcome wall=$wall peak_rss=$peak record=${record:-${REGBENCH_RECORD:--}} exit=${code:--}"
    REGBENCH_SAID_END=1
    regbench_telemetry_end
    echo "$line"
}
# the EXIT trap: a job that ended without its end line (a refusal under `set -e`, a
# `die`, a signal) still says how it ended
regbench_exit_said() {
    [ "$REGBENCH_SAID_END" = 1 ] && return 0
    [ -n "$REGBENCH_JOB_KIND" ] || return 0
    regbench_telemetry_end
    local wall="-"
    [ -n "$REGBENCH_T0" ] && wall="$(( $(date +%s) - REGBENCH_T0 ))s"
    echo "[regbench] end kind=$REGBENCH_JOB_KIND job=$(_regbench_job) outcome=$([ "$1" = 0 ] && echo ok || echo "exit_$1") wall=$wall peak_rss=$(_regbench_peak_mb | sed -E 's/^([0-9]+)$/\1MB/') record=${REGBENCH_RECORD:--} exit=$1"
}

# regbench_end <kind> <exit code of the Python step> [key=value ...]
regbench_end() {
    local kind="$1" code="$2"; shift 2
    if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
        sed '$d' "$REGBENCH_END_FILE"
        regbench_say_end "$(tail -n 1 "$REGBENCH_END_FILE")"
    else
        # the step wrote no end line: it died (killed for memory or time, a crash of the
        # interpreter, a container that did not start). The scheduler's accounting may
        # later say `infrastructure`; until then a dead process is counted as the method's.
        regbench_say_end "REGBENCH_END kind=$kind $* status=no_end_line cause=killed whose=method exit=$code log=${REGBENCH_LOG:--}"
    fi
}

# regbench_exec <image> <command...>: run any command in that image, from the repository
# root (in a subshell: the caller's directory and environment are left alone).
regbench_exec() (
    local method="$1"; shift
    local sif conda venv upper
    upper="$(echo "$method" | tr '[:lower:]' '[:upper:]')"
    sif="$(_regbench_var SIF "$method")"
    conda="$(_regbench_var CONDA "$method")"
    venv="$(_regbench_var VENV "$method")"
    cd "$REGBENCH_REPO" || return 1
    if [ -n "$sif" ]; then
        local runtime bind v nv=() tmpfs=()
        # no runtime is said, never a bare exit: a caller under `set -e` would stop
        # here without a word (the modules of a login shell are not a job's)
        runtime="$(command -v apptainer || command -v singularity)" || runtime=""
        if [ -z "$runtime" ]; then
            echo "regbench: neither apptainer nor singularity is on PATH here ($(hostname)), and $method runs in the image $sif. Load the site's container module before the call (a job has not the modules of a login shell)." >&2
            return 127
        fi
        # a folder that is bound must exist: the runtime refuses to start the image
        # otherwise ("mount source ... doesn't exist"), also for the first call of all,
        # which is the one that fills the cases folder
        mkdir -p "$REGBENCH_CASES" "$REGBENCH_OUT" ${REGBENCH_WORK:+"$REGBENCH_WORK"} \
            ${REGBENCH_NUCLEI:+"$REGBENCH_NUCLEI"} ${REGBENCH_INPUTS:+"$REGBENCH_INPUTS"} 2>/dev/null \
            || echo "regbench: could not create every folder to bind; the runtime names the one that is missing." >&2
        bind="$REGBENCH_REPO,$REGBENCH_CASES,$REGBENCH_OUT${REGBENCH_WORK:+,$REGBENCH_WORK}${REGBENCH_BIND:+,$REGBENCH_BIND}"
        # the public datasets' cases, those that are named (REGBENCH_CASES_<NAME>)
        v="$(_regbench_public_cases)"
        [ -z "$v" ] || bind="$bind,$v"
        # the input folder: the preprocessed slides, rounds.csv and the outlines the
        # nuclei step traces (launch.sbatch --inputs / --prepare). Not twice: under a
        # folder that is bound already (launch.sbatch binds REGBENCH_ROOT) it is there
        if [ -n "${REGBENCH_INPUTS:-}" ]; then
            local held=0 b
            local -a binds
            IFS=, read -r -a binds <<< "$bind"
            for b in "${binds[@]}"; do
                b="${b%%:*}"
                [ -n "$b" ] || continue
                case "${REGBENCH_INPUTS%/}/" in "${b%/}/"*) held=1 ;; esac
            done
            [ "$held" = 1 ] || bind="$bind,$REGBENCH_INPUTS"
        fi
        # the StarDist re-run of the bare nuclei reads the site's model and writes here
        bind="$bind${REGBENCH_STARDIST_MODEL_DIR:+,$REGBENCH_STARDIST_MODEL_DIR}${REGBENCH_NUCLEI:+,$REGBENCH_NUCLEI}"
        # a site config outside the checkout (the model and settings of that re-run)
        case "${REGBENCH_SITE_CONFIG:-none}" in none|-) ;; *) bind="$bind,$REGBENCH_SITE_CONFIG" ;; esac
        # the private plug-in's folder, when one is named, and the job's TMPDIR
        bind="$bind${REGBENCH_PRIVATE_DIR:+,$REGBENCH_PRIVATE_DIR}"
        [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && bind="$bind,${TMPDIR%/}"
        # Inside the image: the checkout on PYTHONPATH (plus the STARE source for an
        # image without one) and nothing of the host's; no user site-packages.
        local inside="$REGBENCH_REPO"
        case " $REGBENCH_STARE_SRC_IMAGES " in *" $upper "*) inside="$inside:$(regbench_stare_src)" ;; esac
        export "SINGULARITYENV_PYTHONPATH=$inside" "APPTAINERENV_PYTHONPATH=$inside"
        export SINGULARITYENV_PYTHONNOUSERSITE=1 APPTAINERENV_PYTHONNOUSERSITE=1
        # <PREFIX>ENV_X sets X inside --cleanenv on every Singularity 3.x and Apptainer.
        for v in TMPDIR REGBENCH_WORK REGBENCH_PRIVATE REGBENCH_PRIVATE_DIR SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_ID SLURM_ARRAY_JOB_ID \
                 SLURM_ARRAY_TASK_ID REGBENCH_ATTEMPT REGBENCH_MEM_GB \
                 REGBENCH_RETRY REGBENCH_SUBMISSION REGBENCH_TIME_LIMIT REGBENCH_FORCE \
                 REGBENCH_STAGES \
                 REGBENCH_LOG REGBENCH_END_FILE \
                 REGBENCH_STARDIST_MODEL_DIR REGBENCH_STARDIST_MODEL \
                 REGBENCH_STARDIST_PROB_THRESH REGBENCH_STARDIST_CPU REGBENCH_SITE_CONFIG \
                 CUDA_VISIBLE_DEVICES; do
            [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}" "APPTAINERENV_$v=${!v}"
        done
        # --nv only for a run that asked for a GPU. The cohort launcher says so in
        # REGBENCH_GPU; unset, a visible device decides.
        if [ "${REGBENCH_GPU:-}" = 1 ]; then
            nv=(--nv)
        elif [ -z "${REGBENCH_GPU:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            nv=(--nv)
        fi
        case " $REGBENCH_WRITABLE_TMPFS_IMAGES " in *" $upper "*) tmpfs=(--writable-tmpfs) ;; esac
        "$runtime" exec --cleanenv ${nv[@]+"${nv[@]}"} ${tmpfs[@]+"${tmpfs[@]}"} --bind "$bind" \
            "$sif" "$@"
    elif [ -n "$conda" ]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$conda"
        PYTHONNOUSERSITE=1 "$@"
    elif [ -n "$venv" ]; then
        PYTHONNOUSERSITE=1 PATH="$venv/bin:$PATH" "$@"
    else
        "$@"
    fi
)

# regbench_py <image> <regbench args...>: run `python -m benchmarks.regbench` in that image.
# The interpreter is looked for INSIDE the image (or environment), never on the host:
# `python` where there is one, else `python3`. labsyspharm/ashlar:1.20.0 is Ubuntu 22.04
# with python3 only, and a bare `python` there is "executable file not found in $PATH".
REGBENCH_PYTHON='if command -v python >/dev/null 2>&1; then exec python "$@"; else exec python3 "$@"; fi'
regbench_py() {
    local method="$1"; shift
    regbench_exec "$method" sh -c "$REGBENCH_PYTHON" regbench-python -m "$REGBENCH_MODULE" "$@"
}
