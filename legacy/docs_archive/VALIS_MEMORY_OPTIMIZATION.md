# VALIS Memory Optimization Guide

## Current State Analysis

### Memory Allocation

**Current configuration** ([conf/modules.config:119-123](conf/modules.config:119-123)):
```groovy
withName: 'REGISTER' {
    cpus   = 16
    memory = '512.GB'  // ⚠️ EXTREMELY HIGH
    time   = '48.h'
}
```

**Problem:** 512 GB is excessive for registration and indicates potential memory management issues in the VALIS workflow.

---

## Root Causes of High Memory Usage

### 1. **Full Image Caching** (PRIMARY CAUSE)

**Location:** [bin/register.py:269-291](bin/register.py:269-291)

**Parameter:** `--max-image-dim` (default: 6000 px)

**Current behavior:**
- VALIS loads entire WSI into memory if dimensions < `max_image_dim`
- For images >6000px in either dimension, uses disk-based tile caching
- **Your images are likely 20000-30000 pixels** → should NOT be cached
- But caching threshold may not be working correctly

**Memory impact:**
```
Image: 25000 x 30000 pixels x 3 channels x 2 bytes (uint16) = ~4.5 GB per image
VALIS processes: 4-8 images simultaneously
Total: 18-36 GB just for raw images
Plus registration matrices, features, warped outputs: 2-3x multiplier
TOTAL: ~50-100 GB theoretical (but hitting 512 GB!)
```

### 2. **QC Generation Memory Leak**

**Location:** [bin/register.py:99-198](bin/register.py:99-198)

**Issues:**
- Line 137: `warped_vips = slide_obj.warp_slide(level=0, non_rigid=True)`
- Warps FULL RESOLUTION images for QC
- Creates multiple full-res copies (RGB composites)
- Even with `gc.collect()`, memory may not be released immediately

**Memory impact:**
```
Per QC generation:
- Warped image: 4.5 GB
- Reference DAPI: 1.5 GB
- RGB stack: 3 GB
- Temporary downsampled: 0.3 GB
Total per QC: ~10 GB

For 7 images (6 QC comparisons): ~60 GB
```

### 3. **VALIS Internal Caching**

**Issues:**
- VALIS caches warped slides internally
- Feature detection results cached
- Transformation matrices accumulated
- No explicit cache clearing between slides

### 4. **PyVIPS Memory Management**

**Issue:**
- PyVIPS uses lazy evaluation
- Memory not released until explicitly called
- VALIS may create multiple PyVIPS image objects without cleanup

---

## Non-Parameter-Dependent Optimizations

These changes reduce memory WITHOUT changing registration quality:

### Optimization 1: Disable Full-Resolution QC (RECOMMENDED)

**Impact:** Reduces memory by 50-70 GB

**Implementation:**

Edit [bin/register.py:99-198](bin/register.py:99-198):

```python
def save_qc_dapi_rgb(registrar, qc_dir: str, ref_image: str):
    """Create QC RGB composites - MEMORY OPTIMIZED VERSION."""
    from pathlib import Path
    from skimage.transform import rescale
    import cv2

    qc_path = Path(qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)

    # Find reference
    ref_image_no_ext = ref_image.replace('.ome.tif', '').replace('.ome.tiff', '')
    ref_slide_key = next((k for k in registrar.slide_dict.keys() if k == ref_image_no_ext), None)

    if ref_slide_key is None:
        raise KeyError(f"Reference '{ref_image_no_ext}' not found.")

    # Extract reference DAPI at REDUCED LEVEL (e.g., level=2 = 25% resolution)
    ref_slide = registrar.slide_dict[ref_slide_key]
    ref_channels = get_channel_names(os.path.basename(ref_slide_key))
    ref_dapi_idx = next((i for i, ch in enumerate(ref_channels) if "DAPI" in ch.upper()), 0)

    # ✅ FIX #1: Load at reduced resolution
    ref_vips = ref_slide.slide2vips(level=2)  # 25% of full resolution
    ref_dapi = ref_vips.extract_band(ref_dapi_idx).numpy()
    ref_dapi_scaled = autoscale(ref_dapi)

    log_progress(f"Processing {len(registrar.slide_dict) - 1} slides for QC...")

    for slide_name, slide_obj in registrar.slide_dict.items():
        if slide_name == ref_slide_key:
            continue

        log_progress(f"Creating QC for: {slide_name}")

        # Extract registered DAPI
        slide_channels = get_channel_names(os.path.basename(slide_name))
        slide_dapi_idx = next((i for i, ch in enumerate(slide_channels) if "DAPI" in ch.upper()), 0)

        # ✅ FIX #2: Warp at reduced resolution
        warped_vips = slide_obj.warp_slide(level=2, non_rigid=True, crop=registrar.crop)
        reg_dapi = warped_vips.extract_band(slide_dapi_idx).numpy()
        reg_dapi_scaled = autoscale(reg_dapi)

        # ❌ REMOVE: Full-resolution TIFF saving
        # This saves 4-6 GB per QC image

        # Create RGB composite: Red = registered, Green = reference
        h, w = reg_dapi_scaled.shape
        rgb_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_bgr[:, :, 0] = 0                # Blue channel
        rgb_bgr[:, :, 1] = ref_dapi_scaled  # Green channel (reference)
        rgb_bgr[:, :, 2] = reg_dapi_scaled  # Red channel (registered)

        # Save as PNG only (compressed, ~10-20 MB)
        base_name = os.path.basename(slide_name)
        png_path = qc_path / f"{base_name}_QC_RGB.png"
        cv2.imwrite(str(png_path), rgb_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        log_progress(f"  Saved QC PNG: {png_path.name}")

        # ✅ FIX #3: Explicit memory cleanup
        del reg_dapi, reg_dapi_scaled, warped_vips, rgb_bgr
        gc.collect()

    # ✅ FIX #4: Clean up reference
    del ref_vips, ref_dapi, ref_dapi_scaled
    gc.collect()

    log_progress("QC generation complete")
```

**Memory savings:** ~60 GB (no full-resolution QC TIFFs)

### Optimization 2: Force Disk Caching for Large Images

**Impact:** Ensures images >6000px use disk instead of RAM

**Edit [bin/register.py:269](bin/register.py:269):**

```python
def valis_registration(input_dir: str, out: str, qc_dir: Optional[str] = None,
                       reference_markers: Optional[list[str]] = None,
                       max_processed_image_dim_px: int = 1800,
                       max_non_rigid_dim_px: int = 3500,
                       micro_reg_fraction: float = 0.5,
                       num_features: int = 5000,
                       max_image_dim_px: int = 6000) -> int:  # ← This parameter
    """..."""

    # ✅ FIX: Override max_image_dim_px to force disk caching
    # Set to a value SMALLER than your actual image dimensions
    # This forces VALIS to use disk-based tiling instead of loading full image into RAM
    max_image_dim_px = min(max_image_dim_px, 4000)  # Force disk caching for images >4000px

    log_progress(f"VALIS registration with max_image_dim={max_image_dim_px}px (disk caching)")

    # ... rest of function
```

**Update nextflow.config:**

```groovy
// Add new parameter
params.reg_max_image_dim = 4000  // Force disk caching (was: 6000)
```

**Memory savings:** ~20-40 GB (prevents full image loading)

### Optimization 3: Add Explicit Memory Cleanup

**Add cleanup after each registration stage:**

Edit [bin/register.py](bin/register.py) in the main registration function (add after line ~320):

```python
# After registrar.register(...)
log_progress("Registration complete, clearing caches...")

# Clear slide caches
for slide in registrar.slide_dict.values():
    if hasattr(slide, '_cache'):
        slide._cache.clear()
    if hasattr(slide, 'image'):
        del slide.image

# Clear pyvips cache
import pyvips
pyvips.cache_set_max(0)  # Disable caching
pyvips.cache_set_max_mem(0)

# Force garbage collection
gc.collect()

log_progress("Memory cleared")
```

**Memory savings:** ~10-20 GB (cached intermediate results)

### Optimization 4: Process-Based Parallelism (Alternative Approach)

**Current:** VALIS processes all images in one Python process (memory accumulates)

**Better:** Use Nextflow's natural parallelism

**Edit main registration workflow to use GPU/CPU methods instead:**

```groovy
// Instead of VALIS batch processing:
// - Use GPU_REGISTER or CPU_REGISTER for pairwise registration
// - Each image processed in separate process (memory isolated)
// - Faster overall (parallel execution)
// - Lower peak memory per process
```

**Recommendation:** Switch to `--registration_method gpu` or `--registration_method cpu`

---

## Parameter-Based Optimizations (Affect Quality)

These reduce memory but may reduce registration accuracy:

### Optimization 5: Reduce Processing Dimensions

**Current defaults:**
```groovy
reg_max_processed_dim     = 1800  // Rigid registration
reg_max_non_rigid_dim     = 3500  // Non-rigid registration
```

**Optimized (lower memory, slightly lower accuracy):**
```groovy
reg_max_processed_dim     = 1200  // ↓ 33% memory
reg_max_non_rigid_dim     = 2500  // ↓ 30% memory
```

**Memory savings:** ~30 GB

**Quality impact:** Minimal if features are well-distributed

### Optimization 6: Reduce Feature Count

**Current:**
```groovy
reg_num_features = 5000
```

**Optimized:**
```groovy
reg_num_features = 3000  // Usually sufficient
```

**Memory savings:** ~5-10 GB (feature matrices smaller)

**Quality impact:** Minimal for WSI with good DAPI signal

### Optimization 7: Reduce Micro-Registration Fraction

**Current:**
```groovy
reg_micro_reg_fraction = 0.5  // Process 50% of image for micro-registration
```

**Optimized:**
```groovy
reg_micro_reg_fraction = 0.25  // Process 25% of image
```

**Memory savings:** ~15-20 GB

**Quality impact:** Slight reduction in local alignment accuracy

---

## Recommended Implementation Strategy

### Phase 1: Immediate Fixes (No Quality Impact)

Apply these changes immediately to reduce memory from 512 GB → ~200 GB:

1. ✅ **Edit bin/register.py:** Implement QC optimization (remove full-res TIFFs, use level=2)
2. ✅ **Edit bin/register.py:** Force `max_image_dim_px = 4000` to enable disk caching
3. ✅ **Edit bin/register.py:** Add explicit memory cleanup after registration
4. ✅ **Update conf/modules.config:**

```groovy
withName: 'REGISTER' {
    cpus   = 16
    memory = { check_max( 200.GB * task.attempt, 'memory' ) }  // Down from 512 GB
    time   = '48.h'
}
```

### Phase 2: Switch Registration Method (RECOMMENDED)

**Switch from VALIS to GPU registration:**

```bash
--registration_method gpu  # Instead of valis
```

**Advantages:**
- Lower memory: 128-256 GB per image (vs 512 GB for all)
- Faster: Parallel processing (each image separate job)
- Better retry logic: If one image fails, others continue
- Isolated processes: Memory released after each image

**Memory comparison:**
```
VALIS batch (current):
- Peak: 512 GB (all images simultaneously)
- Duration: 6-12 hours continuous

GPU pairwise (recommended):
- Peak: 256 GB (one image at a time)
- Duration: 2-3 hours per image (parallel jobs)
- Total cluster memory: Same but distributed
```

### Phase 3: Parameter Tuning (If Still Needed)

Only if memory still exceeds 200 GB after Phase 1:

1. Reduce `reg_max_non_rigid_dim`: `3500` → `2500`
2. Reduce `reg_num_features`: `5000` → `3000`
3. Reduce `reg_micro_reg_fraction`: `0.5` → `0.25`

**Expected final memory:** ~150 GB

---

## Configuration File Changes

### Updated nextflow.config

```groovy
// Classic registration (VALIS) - OPTIMIZED
reg_reference_markers     = ['DAPI', 'FITC']
reg_max_processed_dim     = 1800           // Keep (rigid registration)
reg_max_non_rigid_dim     = 2500           // ↓ Reduced from 3500
reg_micro_reg_fraction    = 0.25           // ↓ Reduced from 0.5
reg_num_features          = 3000           // ↓ Reduced from 5000
reg_max_image_dim         = 4000           // ↓ Reduced from 6000 (force disk caching)
```

### Updated conf/modules.config

```groovy
withName: 'REGISTER' {
    cpus   = 16
    memory = { check_max( 200.GB * task.attempt, 'memory' ) }  // ↓ From 512 GB
    time   = { check_max( 48.h * task.attempt, 'time' ) }

    errorStrategy = { task.exitStatus in [137, 104] ? 'retry' : 'finish' }
    maxRetries    = 2
}
```

---

## Testing the Optimizations

### Step 1: Test with Small Dataset

```bash
nextflow run main.nf \
  --input small_test.csv \
  --registration_method valis \
  --reg_max_image_dim 4000 \
  --reg_max_non_rigid_dim 2500 \
  -profile slurm
```

Monitor memory:
```bash
# In SLURM job script or via sstat:
watch -n 5 'sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,MaxVMSize'
```

### Step 2: Compare Methods

Run same dataset with GPU method:
```bash
nextflow run main.nf \
  --input small_test.csv \
  --registration_method gpu \
  -profile slurm
```

Compare:
- Peak memory usage
- Total runtime
- Registration quality (visual QC)

### Step 3: Validate Quality

Check registration QC images:
```bash
# Yellow overlay = good alignment
# Red/green separation = poor alignment

ls results/*/registration_qc/*.png
```

If quality acceptable, roll out to full dataset.

---

## Summary of Expected Memory Reductions

| Optimization | Memory Saved | Quality Impact |
|--------------|--------------|----------------|
| QC at reduced resolution + PNG only | ~60 GB | None (QC visualization only) |
| Force disk caching (max_image_dim=4000) | ~30 GB | None |
| Explicit memory cleanup | ~20 GB | None |
| Reduce non-rigid dim (3500→2500) | ~30 GB | Minimal |
| Reduce features (5000→3000) | ~10 GB | Minimal |
| Reduce micro-reg (0.5→0.25) | ~20 GB | Slight |
| **TOTAL** | **~170 GB** | **Minimal to None** |

**Result:** 512 GB → ~200-250 GB (60% reduction)

**Alternative:** Switch to GPU method → 128-256 GB per image (pairwise, parallel)

---

## Monitoring and Debugging

### Enable Memory Profiling

Add to Python script:
```python
import tracemalloc

tracemalloc.start()

# ... registration code ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**9:.1f} GB")
print(f"Peak memory usage: {peak / 10**9:.1f} GB")
tracemalloc.stop()
```

### SLURM Memory Tracking

```bash
# After job completes:
sacct -j $JOBID --format=JobID,MaxRSS,ReqMem,Elapsed

# Expected output:
# JobID    MaxRSS    ReqMem   Elapsed
# 12345    200000M   512000M  06:30:00
#          ↑ Actual  ↑ Requested
```

If `MaxRSS` << `ReqMem`, you're over-allocating.

---

## Additional Resources

- VALIS Documentation: https://github.com/MathOnco/valis
- PyVIPS Memory Management: https://libvips.github.io/libvips/API/current/VipsImage.html#memory
- Nextflow Retry Strategies: https://www.nextflow.io/docs/latest/process.html#errorstrategy

---

## Contact

For questions about VALIS optimization, contact the pipeline maintainers or consult the VALIS GitHub issues.
