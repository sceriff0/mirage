# shellcheck shell=bash
# Sourced by every regbench sbatch script. Export what you need before `submit.sh`.
#
#   REGBENCH_REPO    this repository, on the `benchmarking` branch (default: where this file is)
#   REGBENCH_CASES   the prepared cases (`python -m regbench prepare ...`)
#   REGBENCH_OUT     results: one folder per method variant, plus tables/ and logs/
#   REGBENCH_WORK    scratch for a method's intermediates (default: $TMPDIR, node-local)
#
# Every method gets the SAME allocation, so their costs are comparable:
#   REGBENCH_CPUS (8)   REGBENCH_MEM (64G)   REGBENCH_TIME (04:00:00)
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
: "${REGBENCH_OUT:=$HOME/regbench_results/stare-$(git -C "$REGBENCH_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
: "${REGBENCH_CPUS:=8}"
: "${REGBENCH_MEM:=64G}"
: "${REGBENCH_TIME:=04:00:00}"
export REGBENCH_REPO REGBENCH_CASES REGBENCH_OUT REGBENCH_CPUS REGBENCH_MEM REGBENCH_TIME

# The checked-out source wins over any STARE installed in an image or environment, so the
# benchmark always measures this commit.
export PYTHONPATH="$REGBENCH_REPO/src:$REGBENCH_REPO/benchmarks${PYTHONPATH:+:$PYTHONPATH}"

_regbench_var() {  # _regbench_var SIF stare -> value of REGBENCH_SIF_STARE
    local name="REGBENCH_$1_$(echo "$2" | tr '[:lower:]' '[:upper:]')"
    echo "${!name:-}"
}

# regbench_py <method> <regbench args...>: run `python -m regbench` in that method's environment.
regbench_py() {
    local method="$1"; shift
    local sif conda venv
    sif="$(_regbench_var SIF "$method")"
    conda="$(_regbench_var CONDA "$method")"
    venv="$(_regbench_var VENV "$method")"
    if [ -n "$sif" ]; then
        local runtime bind v nv=()
        runtime="$(command -v apptainer || command -v singularity)"
        bind="$REGBENCH_REPO,$REGBENCH_CASES,$REGBENCH_OUT${REGBENCH_WORK:+,$REGBENCH_WORK}${REGBENCH_BIND:+,$REGBENCH_BIND}"
        # <PREFIX>ENV_X sets X inside --cleanenv on every Singularity 3.x and Apptainer.
        for v in PYTHONPATH SLURM_CPUS_PER_TASK SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID; do
            [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}" "APPTAINERENV_$v=${!v}"
        done
        [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nv=(--nv)
        "$runtime" exec --cleanenv "${nv[@]}" --bind "$bind" "$sif" python -m regbench "$@"
    elif [ -n "$conda" ]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$conda"
        python -m regbench "$@"
    elif [ -n "$venv" ]; then
        "$venv/bin/python" -m regbench "$@"
    else
        python -m regbench "$@"
    fi
}
