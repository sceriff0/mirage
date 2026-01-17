#!/bin/bash
#SBATCH --job-name=ateia_test
#SBATCH --output=logs/nftest_%j.out
#SBATCH --error=logs/nftest_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com

# ============================================================================
# ATEIA WSI Processing Pipeline - TEST MODE SLURM Submission Script
# ============================================================================
#
# This script runs nf-test suite on SLURM
#
# Usage:
#   sbatch submit_test.sh
#
# Options:
#   - Stub tests only (fast): modify PROFILE below
#   - Full tests with containers: use singularity profile
# ============================================================================

# Test configuration
PROFILE="test,singularity"  # Use singularity on HPC
# PROFILE="stub"            # Uncomment for fast stub-only tests (no containers)

SRC_DIR="/beegfs/scratch/ieo7660/analysis_runs/ateia"

# Create necessary directories
mkdir -p logs reports .nf-test

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Profile: $PROFILE"
echo "=================================================="

source ~/.bashrc
conda activate nf-env
export SINGULARITY_CACHEDIR="/hpcnfs/scratch/P_DIMA_ATTEND/docker_images"
export NXF_SINGULARITY_CACHEDIR="/hpcnfs/scratch/P_DIMA_ATTEND/docker_images"

cd ${SRC_DIR}

# Generate test data if needed
if [ ! -d "tests/testdata" ] || [ ! -f "tests/testdata/P001_ref.ome.tiff" ]; then
    echo "Generating test data..."
    python tests/testdata/generate_complete_testdata.py
fi

# Run nf-test suite
# Options:
#   - Run all tests: nf-test test --profile $PROFILE
#   - Run only module tests: nf-test test tests/modules/ --profile $PROFILE
#   - Run specific test: nf-test test tests/modules/preprocess.nf.test --profile $PROFILE
echo "Running nf-test with profile: $PROFILE"
nf-test test --profile $PROFILE

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo "=================================================="
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Status: ALL TESTS PASSED"
else
    echo "Status: SOME TESTS FAILED"
fi
echo "=================================================="

# Exit with nf-test's exit status
exit $EXIT_STATUS
