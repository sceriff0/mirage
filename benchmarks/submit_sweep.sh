#!/usr/bin/env bash
#SBATCH --job-name=mirage_bench
#SBATCH --output=/hpcnfs/home/ieo7660/logs/bench_%j.out
#SBATCH --error=/hpcnfs/home/ieo7660/logs/bench_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8   # headroom for CONCURRENCY Nextflow JVMs (they mostly poll SLURM, not compute)
#SBATCH --mem=64G           # all CONCURRENCY heads share this; NXF_OPTS -Xmx caps each head's heap (below)
# #SBATCH --partition=<your_partition>     # uncomment + set if your site needs it
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@ieo.it
#
# ============================================================================
# MIRAGE benchmark sweep — SLURM "head" (orchestrator) job
# ============================================================================
# This job is LIGHTWEIGHT: it runs the Nextflow orchestrator, which submits ONE
# SLURM job per pipeline process (executor='slurm' in conf/ieo.config). The heavy
# compute — CONVERT_IMAGE, PREPROCESS, REGISTER / REG_*, SEGMENT, QUANTIFY — runs
# in those CHILD jobs, sized by conf/base.config + modules.config, NOT here. So keep
# this job small (2 cpus / 16 GB for the Nextflow JVM) but give it a LONG walltime:
# run_sweep.sh launches the runs sequentially, so the head job lives for the whole sweep.
#
# STORAGE FIRST. The shipped sweep.yaml scales to target_px=90000 x 4 channels and
# needs up to 7 moving panels per cell, so the generated matrix is ~1.3 TB of
# uncompressed uint16 BEFORE a single run starts, and the 135-run plan (405 at the
# default --repeats 3) writes work dirs on top. Both live under BENCH_DIR on beegfs
# for that reason. Trim sweep.yaml's scaling_grid/registration_grid target_px if
# that does not fit -- the regression needs a size RANGE, not the largest cell.
#
# ONE PREREQ, run yourself on a login node: the image matrix. It is the TB-scale,
# hours-long step, so it stays outside this job -- a resubmission must not
# regenerate it. The run plan is NOT a prereq: this script builds it below, the
# way submit_arms.sh does, so it can never drift from sweep.yaml.
#
#   python $SRC_DIR/benchmarks/generate_matrix.py --source <one-real-slide>.ome.tif \
#       --outdir $BENCH_DIR/bench_matrix --sweep $SRC_DIR/benchmarks/configs/sweep.yaml
#
# Submit:  cd /beegfs/scratch/ieo7660/analysis_runs/method_paper/benchmark
#          mkdir -p logs && sbatch ~/pipelines/mirage/benchmarks/submit_sweep.sh
# Watch:   squeue -u $USER        # 1 head job + N child jobs
#          tail -f logs/bench_<jobid>.out
# ============================================================================

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
BENCH_DIR="${BENCH_DIR:-/beegfs/scratch/ieo7660/ihc_method/benchmark}"
SRC_DIR="${SRC_DIR:-$HOME/pipelines/mirage}"   # the checkout. NOTE: unquoted $HOME, never "~/..."
MATRIX_DIR="${MATRIX_DIR:-$BENCH_DIR/bench_matrix}"  # from generate_matrix --outdir (BIG -- see below)
RUN_PLAN="$BENCH_DIR/bench_run_plan.csv"  # written by this script (step 1 below)
RESULTS="${RESULTS:-$BENCH_DIR/bench_results}"  # per-run outputs + work dirs land here
PROFILES="${PROFILES:-singularity,ieo}"        # OVERRIDES run_sweep.sh's default -profile docker
SITE_CONFIG="$SRC_DIR/conf/ieo.config"    # gitignored: executor=slurm + singularity cacheDir + paths
CONDA_ENV="nf-env"                        # env that has nextflow + python
REPEATS="${SWEEP_REPEATS:-3}"             # replicate runs per config. 135 configs -> 405 runs at 3,
                                          # 135 at 1. Timing at n=1 is noisy (cache state, node
                                          # contention), so 3 is the default; drop to 1 for a first pass.
CONCURRENCY="${SWEEP_CONCURRENCY:-16}"   # pipeline runs launched AT ONCE (each = 1 Nextflow head that
                                          # submits its OWN SLURM process jobs). 16 heads x ~3 GB heap
                                          # (NXF_OPTS below) fit in --mem=64G; each head runs up to
                                          # queueSize=100 process jobs (benchmark.config), so peak in-flight
                                          # SLURM jobs ~ CONCURRENCY x 100 (~1600). Raise CONCURRENCY only
                                          # alongside --mem (~4 GB/head incl. overhead) and mind your
                                          # per-user SLURM job cap (sacctmgr ... format=maxsubmit).
# -------------------------------------------------------------------------------

cd "$BENCH_DIR"

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
command -v nextflow >/dev/null || { echo "nextflow not on PATH (check CONDA_ENV / module load)"; exit 1; }
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
# Keep Singularity image cache off read-only $HOME (matches conf/ieo.config's cacheDir).
export NXF_SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR:-/hpcnfs/scratch/P_DIMA_ATTEND/users/vfassi/docker_images}"
# Cap EACH concurrent Nextflow head's JVM heap so CONCURRENCY x heap stays under --mem
# (16 x 3 GB = 48 GB < 64 GB, leaving room for JVM/OS overhead). Raise -Xmx only if a head OOMs.
export NXF_OPTS="${NXF_OPTS:--Xms512m -Xmx3g}"

echo "=================================================="
echo "Head job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)   Profiles: $PROFILES   Results: $RESULTS"
echo "=================================================="

# 1. Expand sweep.yaml -> bench_run_plan.csv (seconds). Built here rather than by
#    hand so the plan and sweep.yaml cannot drift apart between submissions.
"$PYTHON" "$SRC_DIR/benchmarks/build_run_plan.py" \
    --sweep   "$SRC_DIR/benchmarks/configs/sweep.yaml" \
    --out     "$RUN_PLAN" \
    --repeats "$REPEATS"

# Checked explicitly because there is no `set -e` here: without this, a failed plan
# step falls through to run_sweep.sh, which would launch against a STALE plan.
if [ ! -s "$RUN_PLAN" ]; then
    echo "ERROR: $RUN_PLAN was not written; not launching." >&2
    exit 1
fi

# The matrix IS a prereq, and a missing one produces "no matrix cell" warnings for
# every row rather than an error, so say so here instead.
if [ ! -s "$MATRIX_DIR/matrix_manifest.csv" ]; then
    echo "ERROR: $MATRIX_DIR/matrix_manifest.csv not found." >&2
    echo "       Run generate_matrix.py first (see the header of this file)." >&2
    exit 1
fi

# 2. run_sweep.sh: one `nextflow run` per run_plan row; each submits its process jobs to SLURM.
# SWEEP_CONCURRENCY launches several of those runs at once (parallelising the sweep across the cluster).
# SWEEP_PROFILE sets the (single) Nextflow -profile; the site config is added as a trailing -c.
# NB: pass the profile via SWEEP_PROFILE, NOT a trailing -profile — Nextflow allows -profile only once.
# Concurrency is passed on the COMMAND LINE, not via benchmark.config. Every
# per-process cap in conf/modules.config is Math.min(own, params.max_forks), evaluated
# EAGERLY when that file is parsed -- which happens before a -c file is merged, so a -c
# cannot reach the clamps. The two must be raised TOGETHER: the lower of the pair binds,
# and at the shipped defaults queue_size (20) is far below max_forks (100), so raising
# max_forks alone does nothing. See docs/resources.md.
# MAX_FORKS=20 is the CEILING, not a preference. Every heavy process in
# conf/modules.config caps itself at Math.min(10 or 20, params.max_forks) -- the largest
# such clamp is 20 -- so any value above 20 is inert and only makes the number misleading.
# 20 unlocks every clamp fully.
MAX_FORKS="${MAX_FORKS:-20}"
# QUEUE_SIZE is the real throttle: in-flight SLURM jobs PER Nextflow head. Peak cluster load
# is roughly CONCURRENCY x QUEUE_SIZE (16 x 100 = 1600 here). The lower of
# (max_forks, queue_size) binds per process, so these must be raised together -- at the old
# 5/20 pair, max_forks was the binding constraint and the cluster sat idle.
QUEUE_SIZE="${QUEUE_SIZE:-100}"

# Pre-flight: print the SLURM per-user submit cap next to what this run will actually ask for,
# so a mismatch shows up in the log BEFORE 1600 submissions start failing. Not a hard gate --
# not every site exposes maxsubmit, and an unset cap prints blank rather than a number.
PEAK_JOBS=$(( CONCURRENCY * QUEUE_SIZE ))
MAXSUBMIT=$(sacctmgr -n show assoc user="$USER" format=maxsubmit 2>/dev/null | tr -d ' \n' | head -c 16)
echo "Concurrency: ${CONCURRENCY} heads x queue_size ${QUEUE_SIZE} = ~${PEAK_JOBS} in-flight SLURM jobs"
echo "             max_forks=${MAX_FORKS} (ceiling: conf/modules.config clamps at 20)"
echo "             SLURM per-user maxsubmit: ${MAXSUBMIT:-<not reported>}"
if [[ -n "$MAXSUBMIT" && "$MAXSUBMIT" =~ ^[0-9]+$ && "$PEAK_JOBS" -gt "$MAXSUBMIT" ]]; then
  echo "WARNING: peak in-flight (${PEAK_JOBS}) exceeds your maxsubmit (${MAXSUBMIT})."
  echo "         Lower QUEUE_SIZE or SWEEP_CONCURRENCY, or submissions will be rejected."
fi

export SWEEP_CONCURRENCY="$CONCURRENCY"
export SWEEP_PROFILE="$PROFILES"
"$SRC_DIR/benchmarks/run_sweep.sh" \
    "$RUN_PLAN" \
    "$MATRIX_DIR/matrix_manifest.csv" \
    "$RESULTS" \
    -c "$SITE_CONFIG" \
    --max_forks "$MAX_FORKS" --queue_size "$QUEUE_SIZE"

echo "=================================================="
echo "Sweep finished: $(date)"
# The arms and sweep experiments write to SEPARATE roots, and BOTH halves matter:
# make_tables and make_figures each write nine filenames the other experiment also
# writes. They shared benchmarks/paper_data (tables) and benchmarks/analysis
# (figures) until 2026-08-25, so whichever analysis ran second silently overwrote
# the first's. These are the same roots the Makefile's `sweep-tables` target uses.
# Guarded by benchmarks/tests/test_handoff_paths_are_disjoint.py.
echo "Next (login node) — emit the paper DATA tables:"
echo "    python -m benchmarks.analysis.make_tables \\"
echo "        --results-root $RESULTS --run-plan $RUN_PLAN --outdir benchmarks/_handoff/sweep"
# --reg-eval is REQUIRED by make_figures, with an explicit opt-out: this repo
# ships no ground-truth harness (see benchmarks/README.md section B), so `none`
# is the normal path, not an unusual one -- it writes NO_GROUND_TRUTH.txt into
# the output, so a cost-only result says so in the deliverable instead of
# reading like a complete one. If you have an EXTERNALLY produced landmark TRE
# CSV (one row per pair_id/mode, see load.GROUND_TRUTH_COLS), pass it instead.
echo "Figures + derived config (make_figures also emits the accuracy-vs-cost table):"
echo "    python -m benchmarks.analysis.make_figures \\"
echo "        --results-root $RESULTS --run-plan $RUN_PLAN \\"
echo "        --reg-eval <external landmark TRE csv | none> \\"
echo "        --outdir benchmarks/_handoff/sweep"
echo "=================================================="
