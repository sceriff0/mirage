# shellcheck shell=bash
# Sourced by every regbench sbatch script. Export what you need before submitting.
#
#   REGBENCH_REPO    the mirage repository (default: where this file is)
#   REGBENCH_CASES   the prepared cases (`python -m benchmarks.regbench prepare ...`)
#   REGBENCH_OUT     results: one folder per result name, plus tables/, logs/ and cache/
#   REGBENCH_WORK    scratch for a method's intermediates (default: $TMPDIR, node-local)
#   REGBENCH_BIND    more paths to bind into the images, comma-separated: wherever the
#                    cohort's slides and nuclei live, and the pipeline trace if one is read
#
# THE IMAGES. Each run names the image it runs in (`python -m benchmarks.regbench runs`,
# column `image`); the path of image X is REGBENCH_SIF_X:
#
#   REGBENCH_SIF_STARE_REGBENCH   containers/stare-regbench: STARE, and where `prepare`,
#                                 `gate`, `tissue` and `score` run (variants.SCORER_IMAGE)
#   REGBENCH_SIF_VALIS_MIRAGE     the pipeline's VALIS 1.0.0 image
#   REGBENCH_SIF_VALIS120         containers/valis120 (VALIS 1.2.0)
#   REGBENCH_SIF_DEEPERHISTREG    containers/deeperhistreg
#   REGBENCH_SIF_ASHLAR           labsyspharm/ashlar:1.20.0
#   REGBENCH_SIF_PALOM            containers/palom
#
# In place of an image, REGBENCH_CONDA_X (a conda environment) or REGBENCH_VENV_X (a
# virtualenv) is taken; with none of the three, the `python` on PATH.
#
#   REGBENCH_WRITABLE_TMPFS_IMAGES   images started with --writable-tmpfs (default:
#                                    VALIS120, whose package rewrites a file of its own
#                                    at every JVM start)
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
export REGBENCH_REPO REGBENCH_CASES REGBENCH_OUT REGBENCH_CPUS REGBENCH_MEM

# The harness is imported from the checkout as `benchmarks.regbench`. STARE itself is the
# pinned release installed in the method's environment (requirements/stare.txt).
export PYTHONPATH="$REGBENCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
REGBENCH_MODULE=benchmarks.regbench

_regbench_var() {  # _regbench_var SIF valis120 -> value of REGBENCH_SIF_VALIS120
    local name
    name="REGBENCH_$1_$(echo "$2" | tr '[:lower:]' '[:upper:]')"
    echo "${!name:-}"
}

# regbench_py <image> <regbench args...>: run `python -m benchmarks.regbench` in that
# image, from the repository
# root (in a subshell: the caller's directory and environment are left alone).
regbench_py() (
    local method="$1"; shift
    local sif conda venv upper
    upper="$(echo "$method" | tr '[:lower:]' '[:upper:]')"
    sif="$(_regbench_var SIF "$method")"
    conda="$(_regbench_var CONDA "$method")"
    venv="$(_regbench_var VENV "$method")"
    cd "$REGBENCH_REPO" || return 1
    if [ -n "$sif" ]; then
        local runtime bind v nv=() tmpfs=()
        runtime="$(command -v apptainer || command -v singularity)"
        bind="$REGBENCH_REPO,$REGBENCH_CASES,$REGBENCH_OUT${REGBENCH_WORK:+,$REGBENCH_WORK}${REGBENCH_BIND:+,$REGBENCH_BIND}"
        # <PREFIX>ENV_X sets X inside --cleanenv on every Singularity 3.x and Apptainer.
        for v in PYTHONPATH SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_ID SLURM_ARRAY_JOB_ID \
                 SLURM_ARRAY_TASK_ID REGBENCH_ATTEMPT REGBENCH_MEM_GB REGBENCH_SELFCHECK_STATUS \
                 REGBENCH_RETRY REGBENCH_SUBMISSION REGBENCH_TIME_LIMIT REGBENCH_FORCE \
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
            "$sif" python -m "$REGBENCH_MODULE" "$@"
    elif [ -n "$conda" ]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$conda"
        python -m "$REGBENCH_MODULE" "$@"
    elif [ -n "$venv" ]; then
        "$venv/bin/python" -m "$REGBENCH_MODULE" "$@"
    else
        python -m "$REGBENCH_MODULE" "$@"
    fi
)
