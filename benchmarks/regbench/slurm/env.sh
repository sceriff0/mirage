# shellcheck shell=bash
# Sourced by every regbench sbatch script. Export what you need before submitting.
#
#   REGBENCH_REPO    the mirage repository (default: where this file is)
#   REGBENCH_CASES   the prepared cases (`python -m benchmarks.regbench prepare ...`)
#   REGBENCH_OUT     results: one folder per result name, plus tables/, logs/ and cache/
#   REGBENCH_WORK    scratch for a method's intermediates (default: $TMPDIR, node-local)
#   REGBENCH_BIND    more paths to bind into the images, comma-separated: wherever the
#                    cohort's slides and nuclei live
#
# THE IMAGES. Each run names the image it runs in (`python -m benchmarks.regbench runs`,
# column `image`); the path of image X is REGBENCH_SIF_X:
#
#   REGBENCH_SIF_STARE_REGBENCH   containers/stare-regbench: STARE, and where `prepare`,
#                                 `gate`, `tissue` and `score` run (variants.SCORER_IMAGE)
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
# Every method runs at 4 CPUs on a memory ladder (16 -> 64 -> 256 -> 512 GB; a unit moves
# up one rung only when it ran out of memory). submit_cohort.sh owns that; it exports
# REGBENCH_ATTEMPT and REGBENCH_MEM_GB to each task. No thread variable is exported: a
# method's own core-count option is set to the allocation by the driver, and what its
# pools saw is recorded.
#
# REGBENCH_SUBMISSION is the id of one call of submit_cohort.sh. Every record a job
# writes carries it (run.json: submissions; selfcheck.json: submission), and a gate, the
# ladder and the scorer take no record of another submission as evidence. REGBENCH_FORCE=1
# re-runs units whose results are all ok already (the default reuses them).
#
# REGBENCH_CPUS (4) and REGBENCH_MEM (16G) are the defaults of jobs outside the ladder
# (prepare, gate, score).

: "${REGBENCH_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${REGBENCH_CASES:=$HOME/regbench_cases}"
: "${REGBENCH_OUT:=$HOME/regbench_results/mirage-$(git -C "$REGBENCH_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
: "${REGBENCH_CPUS:=4}"
: "${REGBENCH_MEM:=16G}"
: "${REGBENCH_WRITABLE_TMPFS_IMAGES:=VALIS120}"
: "${REGBENCH_STARE_SRC_IMAGES:=ASHLAR VALIS_MIRAGE}"
export REGBENCH_REPO REGBENCH_CASES REGBENCH_OUT REGBENCH_CPUS REGBENCH_MEM
export REGBENCH_STARE_SRC_IMAGES

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
            mv -n "$tmp/src" "$src" 2>/dev/null || true
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
# for an array task, %x-%j.log otherwise). It goes into run.json, selfcheck.json and the
# gate records (`log_path`), so the scorer's failure log can point at it.
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

# regbench_end <kind> <exit code of the Python step> [key=value ...]
regbench_end() {
    local kind="$1" code="$2"; shift 2
    if [ -n "${REGBENCH_END_FILE:-}" ] && [ -s "$REGBENCH_END_FILE" ]; then
        cat "$REGBENCH_END_FILE"
    else
        # the step wrote no end line: it died (killed for memory or time, a crash of the
        # interpreter, a container that did not start). The scheduler's accounting may
        # later say `infrastructure`; until then a dead process is counted as the method's.
        echo "REGBENCH_END kind=$kind $* status=no_end_line cause=killed whose=method exit=$code log=${REGBENCH_LOG:--}"
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
        runtime="$(command -v apptainer || command -v singularity || true)"
        if [ -z "$runtime" ]; then
            echo "regbench: neither apptainer nor singularity is on PATH here ($(hostname)), and $method runs in the image $sif. Load the site's container module before the call (a job has not the modules of a login shell)." >&2
            return 127
        fi
        # a folder that is bound must exist: the runtime refuses to start the image
        # otherwise ("mount source ... doesn't exist"), also for the first call of all,
        # which is the one that fills the cases folder
        mkdir -p "$REGBENCH_CASES" "$REGBENCH_OUT" ${REGBENCH_WORK:+"$REGBENCH_WORK"} \
            ${REGBENCH_NUCLEI:+"$REGBENCH_NUCLEI"} 2>/dev/null || true
        bind="$REGBENCH_REPO,$REGBENCH_CASES,$REGBENCH_OUT${REGBENCH_WORK:+,$REGBENCH_WORK}${REGBENCH_BIND:+,$REGBENCH_BIND}"
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
                 SLURM_ARRAY_TASK_ID REGBENCH_ATTEMPT REGBENCH_MEM_GB REGBENCH_SELFCHECK_STATUS \
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
