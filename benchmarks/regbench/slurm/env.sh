# shellcheck shell=bash
# Sourced by every regbench sbatch script. Export what you need before `submit.sh`.
#
#   REGBENCH_REPO    the mirage repository (default: where this file is)
#   REGBENCH_CASES   the prepared cases (`python -m benchmarks.regbench prepare ...`)
#   REGBENCH_OUT     results: one folder per method variant, plus tables/ and logs/
#   REGBENCH_WORK    scratch for a method's intermediates (default: $TMPDIR, node-local)
#
# Every method runs at 4 CPUs on a memory ladder that starts low (16 -> 64 -> 256 -> 512 GB;
# a run moves up one rung only when it ran out of memory). These three are the first rung:
#   REGBENCH_CPUS (4)   REGBENCH_MEM (16G)   REGBENCH_TIME (04:00:00)
# No thread variable is exported: a method's own core-count option is set to the
# allocation by the driver, and what its pools saw is recorded.
#
# Each method runs in its own software environment. For a method M in
# {STARE, VALIS, DEEPERHISTREG}, set ONE of
#   REGBENCH_SIF_M     an Apptainer/Singularity image
#   REGBENCH_CONDA_M   a conda environment name or prefix
#   REGBENCH_VENV_M    a virtualenv directory
# or neither, to use whatever `python` is on PATH. Also per method:
#   REGBENCH_SBATCH_M  extra sbatch options, e.g. "--gres=gpu:1 --partition=gpu"
#   REGBENCH_OPTS_M    method options, e.g. "micro_rigid=1 max_image_dim_px=2048"
#   REGBENCH_LABEL_M   name the results differently (to compare two configurations)
# The scorer and `prepare` use STARE's environment (NumPy, SciPy, scikit-image, tifffile).

: "${REGBENCH_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${REGBENCH_CASES:=$HOME/regbench_cases}"
: "${REGBENCH_OUT:=$HOME/regbench_results/mirage-$(git -C "$REGBENCH_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
: "${REGBENCH_CPUS:=4}"
: "${REGBENCH_MEM:=16G}"
: "${REGBENCH_TIME:=04:00:00}"
export REGBENCH_REPO REGBENCH_CASES REGBENCH_OUT REGBENCH_CPUS REGBENCH_MEM REGBENCH_TIME

# The harness is imported from the checkout as `benchmarks.regbench`. STARE itself is the
# pinned release installed in the method's environment (requirements/stare.txt).
export PYTHONPATH="$REGBENCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
REGBENCH_MODULE=benchmarks.regbench

_regbench_var() {  # _regbench_var SIF stare -> value of REGBENCH_SIF_STARE
    local name="REGBENCH_$1_$(echo "$2" | tr '[:lower:]' '[:upper:]')"
    echo "${!name:-}"
}

# regbench_py <method> <regbench args...>: run `python -m benchmarks.regbench` in that
# method's environment, from the repository root (in a subshell: the caller's directory
# and environment are left alone).
regbench_py() (
    local method="$1"; shift
    local sif conda venv
    sif="$(_regbench_var SIF "$method")"
    conda="$(_regbench_var CONDA "$method")"
    venv="$(_regbench_var VENV "$method")"
    cd "$REGBENCH_REPO" || return 1
    if [ -n "$sif" ]; then
        local runtime bind v nv=()
        runtime="$(command -v apptainer || command -v singularity)"
        bind="$REGBENCH_REPO,$REGBENCH_CASES,$REGBENCH_OUT${REGBENCH_WORK:+,$REGBENCH_WORK}${REGBENCH_BIND:+,$REGBENCH_BIND}"
        # <PREFIX>ENV_X sets X inside --cleanenv on every Singularity 3.x and Apptainer.
        for v in PYTHONPATH SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_ID SLURM_ARRAY_JOB_ID \
                 SLURM_ARRAY_TASK_ID REGBENCH_ATTEMPT REGBENCH_MEM_GB REGBENCH_SELFCHECK_STATUS; do
            [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}" "APPTAINERENV_$v=${!v}"
        done
        [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nv=(--nv)
        "$runtime" exec --cleanenv "${nv[@]}" --bind "$bind" "$sif" python -m "$REGBENCH_MODULE" "$@"
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
