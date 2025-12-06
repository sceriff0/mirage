#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --output=cuda_check_%j.log
#SBATCH --error=cuda_check_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h200:1

# CUDA Compatibility Check Script
# This script checks NVIDIA driver version, pulls the registration container,
# and tests CuPy to verify CUDA runtime compatibility

echo "========================================================================"
echo "CUDA Compatibility Check"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"
echo ""

# ============================================================================
# 1. Check NVIDIA Driver Version
# ============================================================================
echo "1. NVIDIA Driver Information"
echo "------------------------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "Running nvidia-smi..."
    nvidia-smi
    echo ""

    # Extract driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "NVIDIA Driver Version: $DRIVER_VERSION"

    # Extract CUDA version supported by driver
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Maximum CUDA Version Supported by Driver: $CUDA_VERSION"
else
    echo "ERROR: nvidia-smi not found!"
    exit 1
fi
echo ""
echo ""

# ============================================================================
# 2. Locate Singularity Container
# ============================================================================
echo "2. Locating Singularity Container"
echo "------------------------------------------------------------------------"

# Option 1: Use existing .img or .sif file (set this path to your container)
# Default: look for container in common locations
CONTAINER_PATH="${CONTAINER_PATH:-}"

if [ -z "$CONTAINER_PATH" ]; then
    # Try to find the container in common locations
    if [ -f "bolt3x-attend_image_analysis-v2.4.img" ]; then
        CONTAINER_PATH="bolt3x-attend_image_analysis-v2.4.img"
    elif [ -f "$HOME/.singularity/cache/bolt3x-attend_image_analysis-v2.4.sif" ]; then
        CONTAINER_PATH="$HOME/.singularity/cache/bolt3x-attend_image_analysis-v2.4.sif"
    elif [ -f "/tmp/register_gpu.sif" ]; then
        CONTAINER_PATH="/tmp/register_gpu.sif"
    else
        echo "ERROR: Container not found!"
        echo "Please set CONTAINER_PATH environment variable to your container location"
        echo "Example: export CONTAINER_PATH=/path/to/bolt3x-attend_image_analysis-v2.4.img"
        exit 1
    fi
fi

echo "Container path: $CONTAINER_PATH"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "ERROR: Container file not found at: $CONTAINER_PATH"
    exit 1
fi

echo "✓ Container found"
echo "Container size: $(du -h $CONTAINER_PATH | cut -f1)"
echo ""
echo ""

# ============================================================================
# 3. Check Container CUDA Runtime Version
# ============================================================================
echo "3. Container CUDA Runtime Information"
echo "------------------------------------------------------------------------"
echo "Checking CUDA runtime version in container..."

singularity exec --nv "$CONTAINER_PATH" bash -c '
if [ -f /usr/local/cuda/version.txt ]; then
    echo "CUDA Runtime (from version.txt):"
    cat /usr/local/cuda/version.txt
elif command -v nvcc &> /dev/null; then
    echo "CUDA Runtime (from nvcc):"
    nvcc --version | grep "release"
else
    echo "CUDA version file not found, checking with Python..."
fi
'
echo ""
echo ""

# ============================================================================
# 4. Test CuPy and GPU Access
# ============================================================================
echo "4. Testing CuPy and GPU Access"
echo "------------------------------------------------------------------------"
echo "Running CuPy test..."

singularity exec --nv "$CONTAINER_PATH" python3 << 'EOF'
import sys
import os

print("Python version:", sys.version)
print("")

# Test CuPy import and CUDA availability
print("=" * 70)
print("CuPy Import and CUDA Check")
print("=" * 70)

try:
    import cupy as cp
    print("✓ CuPy imported successfully")
    print(f"  CuPy version: {cp.__version__}")
    print("")

    # Get CUDA runtime version
    try:
        runtime_version = cp.cuda.runtime.runtimeGetVersion()
        major = runtime_version // 1000
        minor = (runtime_version % 1000) // 10
        print(f"  CUDA Runtime Version (from CuPy): {major}.{minor}")
    except Exception as e:
        print(f"  Could not get CUDA runtime version: {e}")

    # Get driver version
    try:
        driver_version = cp.cuda.runtime.driverGetVersion()
        major = driver_version // 1000
        minor = (driver_version % 1000) // 10
        print(f"  CUDA Driver Version (from CuPy): {major}.{minor}")
    except Exception as e:
        print(f"  Could not get driver version: {e}")

    print("")

except ImportError as e:
    print(f"✗ CuPy import failed: {e}")
    sys.exit(1)

# Test GPU device access
print("=" * 70)
print("GPU Device Access Test")
print("=" * 70)

try:
    device = cp.cuda.Device(0)
    print(f"✓ GPU device accessible")
    print(f"  Device ID: {device.id}")

    # Get device properties
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"  Device Name: {props['name'].decode('utf-8')}")
    print(f"  Compute Capability: {props['major']}.{props['minor']}")
    print(f"  Total Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
    print("")

except Exception as e:
    print(f"✗ GPU device access failed: {e}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test basic GPU operations
print("=" * 70)
print("GPU Operations Test")
print("=" * 70)

try:
    # Create array on GPU
    print("Creating test array on GPU...")
    x = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    print(f"✓ Created GPU array: {x}")

    # Perform computation
    print("Performing computation (sum)...")
    result = cp.sum(x)
    print(f"✓ Computation result: {result}")

    # Test memory pool
    print("Checking memory pool...")
    mempool = cp.get_default_memory_pool()
    print(f"✓ Memory pool used bytes: {mempool.used_bytes()}")
    print(f"✓ Memory pool total bytes: {mempool.total_bytes()}")

    print("")
    print("✓ All GPU operations successful!")

except Exception as e:
    print(f"✗ GPU operation failed: {e}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dipy imports (used in registration)
print("")
print("=" * 70)
print("Registration Dependencies Test")
print("=" * 70)

try:
    from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from cudipy.align.metrics import CCMetric
    print("✓ cuDIPY imports successful")

    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")

    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")

    print("")
    print("✓ All registration dependencies available!")

except ImportError as e:
    print(f"✗ Dependency import failed: {e}")
    sys.exit(1)

print("")
print("=" * 70)
print("SUCCESS: All compatibility checks passed!")
print("=" * 70)

EOF

PYTHON_EXIT_CODE=$?
echo ""

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✓ CuPy test passed"
else
    echo "✗ CuPy test failed with exit code $PYTHON_EXIT_CODE"
fi
echo ""
echo ""

# ============================================================================
# 5. Summary and Recommendations
# ============================================================================
echo "========================================================================"
echo "Summary and Recommendations"
echo "========================================================================"
echo ""

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✓ COMPATIBILITY CHECK PASSED"
    echo ""
    echo "Your system configuration is compatible:"
    echo "  - NVIDIA Driver: $DRIVER_VERSION (supports CUDA $CUDA_VERSION)"
    echo "  - Container: $CONTAINER_URI"
    echo "  - GPU: nvidia_h200"
    echo ""
    echo "You can proceed with GPU registration jobs."
else
    echo "✗ COMPATIBILITY CHECK FAILED"
    echo ""
    echo "Issue detected:"
    echo "  - NVIDIA Driver: $DRIVER_VERSION (supports CUDA $CUDA_VERSION)"
    echo "  - Container CUDA Runtime: Check error messages above"
    echo ""
    echo "Recommended actions:"
    echo "  1. Contact HPC admin to update NVIDIA drivers to support CUDA 12.x"
    echo "  2. OR use a container with older CUDA runtime (e.g., CUDA 11.8)"
    echo "  3. Check if a different GPU partition has newer drivers"
fi
echo ""

# ============================================================================
# 6. Cleanup
# ============================================================================
# No cleanup needed - using existing container file
echo ""

echo "========================================================================"
echo "Check complete: $(date)"
echo "Log file: cuda_check_${SLURM_JOB_ID}.log"
echo "========================================================================"

exit $PYTHON_EXIT_CODE
