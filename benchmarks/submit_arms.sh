#!/usr/bin/env bash
#SBATCH --job-name=mirage_arms
#SBATCH --output=/hpcnfs/home/ieo7660/pipelines/logs/arms_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/pipelines/logs/arms_%j.err
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4    # headroom for CONCURRENCY Nextflow heads (they poll SLURM, not compute)
#SBATCH --mem=64G            # ALL heads share this; NXF_OPTS -Xmx caps each head's heap (below)
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE real-sample ARM benchmark — SLURM "head" (orchestrator) job
# ============================================================================
# The sibling of submit_sweep.sh, for the REAL slides rather than the synthetic
# matrix. See docs/benchmarks_real.md for what the arms are and why.
#
# This job is LIGHTWEIGHT. It runs Nextflow orchestrators, which submit ONE SLURM
# job per pipeline process (executor='slurm' in conf/ieo.config). The heavy compute
# — PREPROCESS, REGISTER, SEGMENT, QUANTIFY — runs in those CHILD jobs, sized by
# conf/base.config + modules.config, NOT here. Keep this job small; give it a LONG
# walltime, because it lives for the whole benchmark.
#
# THE CODE may live in $HOME (it is small). THE DATA MUST NOT. BENCH_DIR below is
# the beegfs storage root and holds the run plan, every arm's --outdir and every
# Nextflow work dir -- run_arms.sh puts each run's work under
# <RESULTS>/.launch/<run_id>/work, so nothing large is ever written to $HOME,
# which is small and read-only inside the containers.
#
# The only thing written back into the checkout is benchmarks/_handoff/arms/*.csv
# in step 3 (a few hundred KB of tables and figures); pull_to_ihc_method.sh reads
# them from there. The SWEEP's equivalents land in benchmarks/_handoff/sweep --
# separate roots, because the two experiments write the same filenames.
#
# Prereq (login node, once): the benchmark lives on the `benchmarking` branch, so
# the checkout must be on it --
#   git -C ~/pipelines/mirage fetch origin
#   git -C ~/pipelines/mirage checkout benchmarking
#   git -C ~/pipelines/mirage pull
#
# Submit:  cd /beegfs/scratch/ieo7660/analysis_runs/method_paper/benchmark
#          mkdir -p logs && sbatch ~/pipelines/mirage/benchmarks/submit_arms.sh
# Watch:   squeue -u $USER                 # 1 head job + N child jobs
#          tail -f logs/arms_<jobid>.out
# ============================================================================

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
BENCH_DIR="${BENCH_DIR:-/beegfs/scratch/ieo7660/ihc_method/benchmark}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"   # the checkout. NOTE: unquoted $HOME, never "~/..."
INPUT="${INPUT:-/beegfs/scratch/ieo7660/ihc_method/head_neck/input.csv}"
RESULTS="${RESULTS:-$BENCH_DIR/arm_results}"   # every arm's outdir + work dir lands under here
ARMS_YAML="$SRC_DIR/benchmarks/configs/arms.yaml"
PROFILES="${PROFILES:-singularity,ieo}"        # OVERRIDES run_arms.sh's default -profile docker
SITE_CONFIG="$SRC_DIR/conf/ieo.config"    # gitignored: executor=slurm + cacheDir + paths
CONDA_ENV="nf-env"
CONCURRENCY="${ARMS_CONCURRENCY:-8}"      # arms launched AT ONCE. Each is one Nextflow head.
                                          # 4 heads x 3 GB heap = 12 GB < --mem=32G. RAISE --mem
                                          # BEFORE raising this: N heads x -Xmx must fit, and the
                                          # default NXF_JVM_ARGS people copy from the normal
                                          # launcher (-Xmx32g) would blow a 32 GB job at N=2.
ENABLE_CSE="${ENABLE_CSE:-true}"         # true => score the segmentation arms with CSE.
                                          # Needs bolt3x/mirage-segeval:${segeval_tag} published
                                          # (1.0.1 is live as of 2026-08-21).
# SUBSET RE-RUN after a code change (docs/benchmarks_real.md, "Re-running a subset
# after a code change"). Either builds a SUBSET plan (arm_plan.subset.csv, beside the
# full one, which the analysis keeps reading) instead of the full plan:
#   CHANGED="tiled"            space-separated components, each passed as
#                              build_arm_plan.py --changed (tiled|stare|valis|qc|
#                              preprocess|seg:<method>); the closure adds every cross,
#                              segmentation arm and external arm that depends on them
#   ONLY='^tiled_high_'        a regex on arm/run_id, passed as --only
# and ARMS_REPLACE=1 makes run_arms.sh move those arms' previous results aside to
# $RESULTS/.replaced/<timestamp>/ before launching (never deleted). Typical:
#   CHANGED=tiled ARMS_REPLACE=1 sbatch benchmarks/submit_arms.sh
CHANGED="${CHANGED:-}"
ONLY="${ONLY:-}"
# -------------------------------------------------------------------------------

# NOTE: do NOT write SRC_DIR="~/..." — bash does tilde expansion BEFORE parameter
# expansion, so a tilde arriving via a variable stays a literal "~" and the path
# never resolves. Use an absolute path or "$HOME/...".

cd "$BENCH_DIR"
mkdir -p logs

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
command -v nextflow >/dev/null || { echo "nextflow not on PATH (check CONDA_ENV)"; exit 1; }
# Resolve the interpreter rather than assuming `python` exists. A conda env can
# provide nextflow while exposing python only as `python3` (or leaving python in
# `base`), and with no `set -e` here a bare `python ...` would fail, the plan
# would not be written, and the launch would proceed against a stale one. Observed
# on this cluster: "line 112: python: command not found".
if [ -z "${PYTHON:-}" ]; then
    # Written as a chain of assignments rather than `$(... || true)`: `command -v`
    # exits 1 to mean NOT FOUND, which is data, and the emptiness is asserted on the
    # next line. tests/test_no_swallowed_failures.py forbids the `|| true` form.
    PYTHON=$(command -v python3) || PYTHON=$(command -v python) || PYTHON=""
fi
[ -n "$PYTHON" ] || { echo "no python3/python on PATH (check CONDA_ENV)"; exit 1; }
echo "Using python: $PYTHON ($("$PYTHON" --version 2>&1))"
# The plan builder needs PyYAML. If PYTHON resolved to a system interpreter rather
# than the conda env's, the import fails INSIDE the plan step, which without
# `set -e` just leaves the plan unwritten. Check it here so the message names the
# real problem.
"$PYTHON" -c "import yaml" 2>/dev/null || {
    echo "ERROR: $PYTHON cannot import yaml (PyYAML)." >&2
    echo "       conda activate $CONDA_ENV && conda install pyyaml   (or pip install pyyaml)" >&2
    echo "       or set PYTHON=/path/to/the/right/python before sbatch." >&2
    exit 1
}

# Pin Nextflow to 25.04.x. The pin STAYS, but its stated reasons are now stale in
# both halves and are corrected here rather than left to mislead:
#
#   * "NF 26.x dropped the automatic lib/*.groovy class loading this pipeline
#     relies on" -- it did not, or no longer does. The pipeline is verified
#     end-to-end on 26.04.6, lib/ classes included (Layout, ParamUtils, Checkpoint
#     are all exercised by tests/lib_probe.nf on both engines).
#   * "rejects the CLI boolean forms below" -- true, and BROADER than booleans:
#     NF26 delivers EVERY CLI param as a String, so integer axes are rejected too
#     ("--reg_qc (1): Value is [string] but should be [integer]"). run_arms.sh and
#     run_sweep.sh no longer pass any of them on the command line; they build a
#     -params-file, which carries JSON types and works on both engines.
#
# What is left is that nothing has RUN on 26.x on this cluster. Lifting the pin is
# a cluster experiment, not an edit -- do it deliberately, with one arm, before
# trusting a whole benchmark to it.
export NXF_VER="${NXF_VER:-25.04.7}"
# Singularity image cache off read-only $HOME (matches conf/ieo.config's cacheDir).
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
export NXF_SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR:-$SINGULARITY_CACHEDIR}"
# Cap EACH concurrent head's heap so CONCURRENCY x heap stays under --mem.
# This is NOT the -Xmx32g of a single-run launcher: that sizes ONE head, and here
# there are CONCURRENCY of them sharing one allocation.
export NXF_OPTS="${NXF_OPTS:--Xms512m -Xmx3g}"

# Concurrency is passed on the COMMAND LINE, not via benchmark.config. Every
# per-process cap in conf/modules.config is Math.min(own, params.max_forks), evaluated
# EAGERLY when that file is parsed -- which happens before a -c file is merged, so a -c
# cannot reach the clamps. The two must be raised TOGETHER: the lower of the pair binds,
# and at the shipped defaults queue_size (20) is far below max_forks (100), so raising
# max_forks alone does nothing. See docs/resources.md.
# 20 is the ceiling: conf/modules.config caps every heavy process at
# Math.min(10 or 20, params.max_forks), so anything above 20 is inert. At the old 5,
# max_forks -- not queue_size -- was the binding constraint and the cluster sat idle.
MAX_FORKS="${MAX_FORKS:-20}"
# Lower than the synthetic sweep's 100 ON PURPOSE. These are REAL whole slides: REGISTER has
# been observed at 483 GB and MERGE_AND_PYRAMID at 6.5 h (docs/benchmarks_real.md), so the
# binding resource is node memory, not the SLURM job count. Peak in-flight is
# CONCURRENCY x QUEUE_SIZE = 4 x 50 = 200; SLURM will queue what does not fit, but a much
# larger number just buries your own queue behind jobs that cannot start.
QUEUE_SIZE="${QUEUE_SIZE:-50}"

PEAK_JOBS=$(( CONCURRENCY * QUEUE_SIZE ))
MAXSUBMIT=$(sacctmgr -n show assoc user="$USER" format=maxsubmit 2>/dev/null | tr -d ' \n' | head -c 16)
echo "Concurrency: ${CONCURRENCY} arms x queue_size ${QUEUE_SIZE} = ~${PEAK_JOBS} in-flight SLURM jobs"
echo "             max_forks=${MAX_FORKS} (ceiling: conf/modules.config clamps at 20)"
echo "             SLURM per-user maxsubmit: ${MAXSUBMIT:-<not reported>}"

EXTRA_ARGS=(--max_forks "$MAX_FORKS" --queue_size "$QUEUE_SIZE")
# CSE is enabled by a PROFILE, never `--skip_seg_quality_eval false`: Nextflow 26
# delivers every --param as a String, so that flag sets the param to the truthy
# string "false" and disables the scorer while reading like it enables it.
if [[ "$ENABLE_CSE" == "true" ]]; then
    PROFILES="$PROFILES,seg_quality"
fi

# CellSAM weights are gated. singularity.envWhitelist forwards DEEPCELL_ACCESS_TOKEN
# by REFERENCE, so it must exist in the environment on the COMPUTE node — a site
# launching with --export=NONE silently stops delivering it. Warn now rather than
# let a multi-hour segmentation arm die on the download. Set cellsam_model_path in
# your site config instead if the compute nodes have no internet.
if grep -q "cellsam" "$ARMS_YAML" && [[ -z "${DEEPCELL_ACCESS_TOKEN:-}" ]]; then
    echo "WARNING: arms.yaml uses the cellsam backend but DEEPCELL_ACCESS_TOKEN is empty."
    echo "         Get a token at https://users.deepcell.org, then either"
    echo "           export DEEPCELL_ACCESS_TOKEN=... in ~/.bashrc (sourced above), or"
    echo "           set params.cellsam_model_path to pre-downloaded weights."
fi

# StarDist weights are the OTHER gated backend, and it had no check here while cellsam did
# -- the same failure class, one guarded and one not. nextflow.config ships
# segmentation_model_dir = null and a segmentation_model name that is NOT a StarDist built-in,
# so segment.py raises FileNotFoundError. That surfaces AFTER preprocessing and registration
# have already been paid for, on every patient in the cohort.
if grep -q "stardist" "$ARMS_YAML" \
   && ! grep -qE "^[^/]*segmentation_model_dir" "$SITE_CONFIG" 2>/dev/null; then
    echo "WARNING: arms.yaml uses the stardist backend but segmentation_model_dir is not set"
    echo "         in $SITE_CONFIG. The shipped segmentation_model name is not a StarDist"
    echo "         built-in, so that arm will fail with FileNotFoundError after preprocessing"
    echo "         and registration have already run. Set params.segmentation_model_dir to the"
    echo "         trained model directory, or drop 'stardist' from arms.yaml."
fi

echo "=================================================="
echo "Head job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}"
echo "Start:      $(date)"
echo "Bench dir:  $BENCH_DIR"
echo "Input:      $INPUT"
echo "Results:    $RESULTS"
echo "Profiles:   $PROFILES   Concurrency: $CONCURRENCY   CSE: $ENABLE_CSE"
[ -n "$CHANGED$ONLY" ] && echo "Subset:     CHANGED='$CHANGED' ONLY='$ONLY' ARMS_REPLACE='${ARMS_REPLACE:-}'"
echo "=================================================="

# 1. Expand arms.yaml -> arm_plan.csv + the consumer's arms.csv (seconds, local).
#    --results-root puts arms.csv where registration_arms.R looks for it.
#    A SUBSET (CHANGED/ONLY set) is written to arm_plan.subset.csv so the FULL plan,
#    which `make arm-tables` and pull_to_ihc_method.sh read, is never overwritten by
#    a subset; arms.csv is written from the full plan either way.
PLAN_CSV="$BENCH_DIR/arm_plan.csv"
SUBSET_ARGS=()
for c in $CHANGED; do SUBSET_ARGS+=(--changed "$c"); done
[ -n "$ONLY" ] && SUBSET_ARGS+=(--only "$ONLY")
if [ "${#SUBSET_ARGS[@]}" -gt 0 ]; then
    PLAN_CSV="$BENCH_DIR/arm_plan.subset.csv"
    rm -f "$PLAN_CSV"
    echo "Subset plan: CHANGED='$CHANGED' ONLY='$ONLY' ARMS_REPLACE='${ARMS_REPLACE:-}' -> $PLAN_CSV"
fi
"$PYTHON" "$SRC_DIR/benchmarks/build_arm_plan.py" \
    --arms         "$ARMS_YAML" \
    --input        "$INPUT" \
    --out          "$PLAN_CSV" \
    --results-root "$RESULTS" \
    "${SUBSET_ARGS[@]+"${SUBSET_ARGS[@]}"}"

# Checked explicitly because there is no `set -e` here: without this, a failed or
# skipped plan step falls straight through to run_arms.sh, which would launch days
# of cluster work against a STALE plan from a previous submission.
if [ ! -s "$PLAN_CSV" ]; then
    echo "ERROR: $PLAN_CSV was not written; not launching." >&2
    exit 1
fi

# 2. Launch. run_arms.sh runs the passes in order — preprocess, registration,
#    segmentation, compute — with a barrier between: each pass resumes from a
#    checkpoint the previous one wrote, and the compute arm is timed last and
#    alone so its numbers are not taken under self-inflicted contention.
#    Pass the profile via ARMS_PROFILE, NOT a trailing -profile: Nextflow accepts
#    -profile only once. A trailing -c IS fine (Nextflow merges multiple -c).
#    ARMS_REPLACE (if set in the submitting environment) reaches run_arms.sh through
#    the environment unchanged: sbatch --export=ALL is the default.
export ARMS_CONCURRENCY="$CONCURRENCY"
export ARMS_PROFILE="$PROFILES"
"$SRC_DIR/benchmarks/run_arms.sh" \
    "$PLAN_CSV" \
    "$INPUT" \
    "$RESULTS" \
    -c "$SITE_CONFIG" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "=================================================="
echo "Arms finished: $(date)"
echo
# The arms and sweep experiments write to SEPARATE roots, and BOTH halves matter:
# make_tables and make_figures each write nine filenames the other experiment also
# writes. They shared benchmarks/paper_data (tables) and benchmarks/analysis
# (figures) until 2026-08-25, so whichever analysis ran second silently overwrote
# the first's, and pull_to_ihc_method.sh copied whatever was left into the
# consumer. These are the same roots the Makefile's `arm-tables` target uses.
# Guarded by benchmarks/tests/test_handoff_paths_are_disjoint.py.
echo "Next, on a login node — emit the tables:"
echo "    cd $SRC_DIR"
echo "    python -m benchmarks.analysis.make_tables \\"
echo "        --results-root $RESULTS --run-plan $BENCH_DIR/arm_plan.csv \\"
echo "        --outdir benchmarks/_handoff/arms"
# --reg-eval is REQUIRED by make_figures, with an explicit opt-out: this repo
# ships no ground-truth harness (see benchmarks/README.md section B), so `none`
# is the normal path, not an unusual one -- it writes NO_GROUND_TRUTH.txt into
# the output, so a cost-only result says so in the deliverable instead of
# reading like a complete one. If you have an EXTERNALLY produced landmark TRE
# CSV (one row per pair_id/mode, see load.GROUND_TRUTH_COLS), pass it instead.
echo "    python -m benchmarks.analysis.make_figures \\"
echo "        --results-root $RESULTS --run-plan $BENCH_DIR/arm_plan.csv \\"
echo "        --reg-eval <external landmark TRE csv | none> \\"
echo "        --outdir benchmarks/_handoff/arms"
echo
echo "Then hand off to ihc_method (small QC artifacts only — the images stay here):"
echo "    benchmarks/pull_to_ihc_method.sh $RESULTS <path-to>/ihc_method"
echo "=================================================="
