#!/usr/bin/env bash
#SBATCH --job-name=mirage_matrix
#SBATCH --output=logs/matrix_%j.out
#SBATCH --error=logs/matrix_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=normal
#
# ============================================================================
# MIRAGE synthetic benchmark matrix — generation job
# ============================================================================
# Wraps benchmarks/generate_matrix.py, which is the one prerequisite of
# submit_sweep.sh. Unlike the two submitters this is NOT a lightweight
# orchestrator: it is the compute, and it does all of it in this one job. It
# reads one real slide, resizes it to each cell of sweep.yaml's grids, and writes
# the OME-TIFFs the resource sweep runs against.
#
# WHY IT IS A JOB AND NOT A LOGIN-NODE COMMAND: hours of CPU and hundreds of GB
# of writes. Both are things login nodes are not for, and a login-node session
# that drops takes the half-written matrix with it.
#
# ---- SIZING ----------------------------------------------------------------
# Peak RAM is one synthesized stack plus the resized source:
#
#     mem  ~=  n_channels * H * W * 2 bytes   (the stack)
#           +  H * W * 2 bytes                (the resized source, held across
#                                              the channel loop)
#
# evaluated at the LARGEST cell in sweep.yaml. For the shipped grid that cell is
# 90000 px x 4 ch = 64.8 GB + 16.2 GB ~= 81 GB, so --mem=128G above clears it. For
# the trimmed grid most people actually want (max 32768 px x 4 ch) the peak is
# ~11 GB and 32G is plenty.
#
# THAT FORMULA ONLY BECAME TRUE ON 2026-08-19. Before it, run_matrix held the
# reference stack bound while it built each moving panel's stack, so the peak was
# TWO stacks -- 145.8 GB at the 90000x4 cell -- and job 6511490 was SIGKILLed
# (exit 137) at --mem=128G after 77 minutes. Each stack is now released before the
# next is allocated, and benchmarks/tests/test_generate_matrix.py counts live
# stacks by weakref so the peak cannot quietly become N+1 again.
#
# DISK is the other axis, and the bigger surprise:
#     shipped grid (to 90000 px)   ~1.33 TB
#     trimmed grid (to 32768 px)   ~170 GB
# Check with `df -h /beegfs/scratch/$USER` BEFORE submitting. Trimming
# scaling_grid/registration_grid target_px in sweep.yaml is the lever; the
# regression needs a size RANGE, not the largest cell.
#
# ---- PREREQ ----------------------------------------------------------------
# A python env with numpy/pandas/tifffile/pyyaml/pyvips, plus bioio + bioio-nd2
# if SOURCE is an ND2/CZI/LIF. `nf-env` does NOT have these -- it is a
# nextflow-only env, and a bare `python3` there falls through to /usr/bin/python3.
#
#   conda create -y -n mirage-bench -c conda-forge \
#       python=3.11 numpy pandas scikit-learn matplotlib tifffile pyyaml pyvips pip
#   conda activate mirage-bench && pip install bioio bioio-nd2
#
# Submit:  cd /beegfs/scratch/ieo7660/ihc_method/benchmark
#          mkdir -p logs && sbatch ~/pipelines/mirage/benchmarks/submit_matrix.sh
# Watch:   tail -f logs/matrix_<jobid>.out
#
# Then:    sbatch ~/pipelines/mirage/benchmarks/submit_sweep.sh
# ============================================================================

# ---- EDIT THESE FOR YOUR SITE --------------------------------------------------
# Overridable and IDENTICAL to submit_arms.sh / submit_sweep.sh: all three read and
# write one benchmark directory (bench_matrix, arm_results, bench_results, logs).
# It was a hardcoded path that no longer exists, so job 6831427 died on `cd` in
# under a second (2026-09-16). Pinned to the other two by
# benchmarks/tests/test_submitters_agree_on_bench_dir.py.
BENCH_DIR="${BENCH_DIR:-/beegfs/scratch/ieo7660/ihc_method/benchmark}"
SRC_DIR="$HOME/pipelines/mirage"          # the checkout. NOTE: unquoted $HOME, never "~/..."
SOURCE="/hpcnfs/techunits/imaging/PublicData/ImagingU/cborriero/nd2_images_2/24086/24086_DAPI_SMA_PANCK.nd2"
MATRIX_DIR="$BENCH_DIR/bench_matrix"      # where the generated cells land
SWEEP_YAML="$SRC_DIR/benchmarks/configs/sweep.yaml"
CONDA_ENV="mirage-bench"                  # env with numpy/tifffile/pyvips/bioio
# -------------------------------------------------------------------------------

cd "$BENCH_DIR" || { echo "cannot cd to $BENCH_DIR"; exit 1; }
mkdir -p logs

# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"

# Resolve the interpreter rather than assuming `python`. Observed on this cluster:
# with nf-env active, `python` does not exist and `python3` silently resolves to
# /usr/bin/python3, which has no numpy.
if [ -z "${PYTHON:-}" ]; then
    # Written as a chain of assignments rather than `$(... || true)`: `command -v`
    # exits 1 to mean NOT FOUND, which is data, and the emptiness is asserted on the
    # next line. tests/test_no_swallowed_failures.py forbids the `|| true` form.
    PYTHON=$(command -v python3) || PYTHON=$(command -v python) || PYTHON=""
fi
[ -n "$PYTHON" ] || { echo "no python3/python on PATH (check CONDA_ENV=$CONDA_ENV)"; exit 1; }
echo "Using python: $PYTHON ($("$PYTHON" --version 2>&1))"

# Preflight the imports HERE, so a missing dependency costs seconds rather than
# failing partway through a multi-hour write. bioio is only needed for the vendor
# formats tifffile cannot open.
"$PYTHON" - <<'PYEOF' || exit 1
import sys
missing = []
for mod in ("numpy", "tifffile", "yaml"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
try:
    import pyvips  # noqa: F401
except Exception:
    try:
        from PIL import Image  # noqa: F401
        print("NOTE: pyvips unavailable; falling back to PIL. Slower, and it "
              "cannot resize every dtype -- install pyvips if this errors.")
    except ImportError:
        missing.append("pyvips (or Pillow)")
if missing:
    print("ERROR: missing python packages: " + ", ".join(missing), file=sys.stderr)
    print("       conda activate mirage-bench  (see this script's header)", file=sys.stderr)
    sys.exit(1)
print("python deps OK")
PYEOF

# `tr` rather than ${SOURCE,,}: that expansion is bash 4+, and it fails at
# RUNTIME ("bad substitution") -- `bash -n` parses it happily, so a syntax
# check does not catch it. run_sweep.sh keeps to bash 3.2 for the same reason.
SOURCE_LC=$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')
case "$SOURCE_LC" in
    *.nd2|*.czi|*.lif)
        "$PYTHON" -c "from bioio import BioImage" 2>/dev/null || {
            echo "ERROR: SOURCE is ${SOURCE##*.}, which tifffile cannot read, and bioio is missing." >&2
            echo "       pip install bioio bioio-nd2   (bioio-czi / bioio-lif as needed)" >&2
            exit 1
        }
        echo "Source format: ${SOURCE##*.} -- will be read via bioio"
        ;;
esac

[ -f "$SOURCE" ] || { echo "ERROR: SOURCE not found: $SOURCE" >&2; exit 1; }

echo "=================================================="
echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}"
echo "Start:   $(date)"
echo "Source:  $SOURCE"
echo "Outdir:  $MATRIX_DIR"
echo "Sweep:   $SWEEP_YAML"
echo "Free on target filesystem:"
df -h "$BENCH_DIR" | tail -1
echo "=================================================="

"$PYTHON" "$SRC_DIR/benchmarks/generate_matrix.py" \
    --source "$SOURCE" \
    --outdir "$MATRIX_DIR" \
    --sweep  "$SWEEP_YAML"
STATUS=$?

echo "=================================================="
echo "End: $(date)   exit=$STATUS"
if [ "$STATUS" -eq 0 ] && [ -s "$MATRIX_DIR/matrix_manifest.csv" ]; then
    echo "Cells generated: $(($(wc -l < "$MATRIX_DIR/matrix_manifest.csv") - 1))"
    echo "Matrix size:     $(du -sh "$MATRIX_DIR" 2>/dev/null | cut -f1)"
    echo
    echo "Next — launch the resource sweep:"
    echo "    cd $BENCH_DIR"
    echo "    sbatch --export=ALL,SWEEP_REPEATS=1 $SRC_DIR/benchmarks/submit_sweep.sh"
else
    echo "FAILED: no usable $MATRIX_DIR/matrix_manifest.csv. Do NOT submit the sweep;"
    echo "        it checks for that file and will refuse anyway."
fi
echo "=================================================="
exit $STATUS
