# Script Argument Analysis & Configuration Recommendations

This document analyzes all Python scripts and their arguments, identifying which arguments should be:
1. **Kept as CLI arguments** (vary per run/sample)
2. **Moved to nextflow.config** (pipeline-level settings)
3. **Removed as non-useful** (unused, redundant, or hardcoded)

---

## 1. PREPROCESS.PY

### Current Arguments
```python
--image                    # Input multichannel TIFF (REQUIRED, varies per sample)
--output_dir              # Output directory (REQUIRED, varies per run)
--fov_size                # FOV size for BaSiC tiling (default: 1950)
--n_workers               # Parallel workers for channels (default: 4)
--skip_dapi               # Skip BaSiC correction for DAPI (flag)
--autotune                # Autotune BaSiC parameters (flag)
--n_iter                  # Autotuning iterations (default: 3)
--overlap                 # Overlap between FOV tiles (default: 0)
--no_darkfield            # Disable darkfield estimation (flag)
```

### Analysis
✅ **Keep as CLI (passed from Nextflow)**
- `--image` - Sample-specific input file
- `--output_dir` - Process-specific output directory

✅ **Already in nextflow.config (GOOD)**
- `--fov_size` → `params.preproc_tile_size` (1950)
- `--n_workers` → `params.preproc_pool_workers` (4)
- `--skip_dapi` → `params.preproc_skip_dapi` (true)
- `--autotune` → `params.preproc_autotune` (false)
- `--n_iter` → `params.preproc_n_iter` (3)
- `--overlap` → `params.preproc_overlap` (0)
- `--no_darkfield` → `params.preproc_no_darkfield` (false)

❌ **Non-useful Arguments**
- None - all arguments are used

### Recommendations
✅ **PERFECT** - All parameters are properly managed:
- Variable inputs passed per sample
- Algorithm settings in config
- No hardcoded values in script

---

## 2. REGISTER.PY (VALIS Registration)

### Current Arguments
```python
--input-dir               # Directory with preprocessed files (REQUIRED)
--out                     # Output merged path (REQUIRED)
--qc-dir                  # QC directory for outputs (OPTIONAL)
--reference-markers       # Markers to identify reference (default: ['DAPI', 'SMA'])
--max-processed-dim       # Max dimension for rigid registration (default: 1800)
--max-non-rigid-dim       # Max dimension for non-rigid (default: 3500)
--micro-reg-fraction      # Micro-registration fraction (default: 0.5)
--num-features            # SuperPoint features to detect (default: 5000)
```

### Analysis
✅ **Keep as CLI**
- `--input-dir` - Process-specific directory
- `--out` - Process-specific output
- `--qc-dir` - Process-specific QC directory

✅ **Already in nextflow.config (GOOD)**
- `--reference-markers` → `params.reg_reference_markers` (['DAPI', 'FITC'])
- `--max-processed-dim` → `params.reg_max_processed_dim` (512) ⚠️ **MISMATCH**: script default 1800, config has 512
- `--max-non-rigid-dim` → `params.reg_max_non_rigid_dim` (2048) ⚠️ **MISMATCH**: script default 3500, config has 2048
- `--micro-reg-fraction` → `params.reg_micro_reg_fraction` (0.125) ⚠️ **MISMATCH**: script default 0.5, config has 0.125
- `--num-features` → `params.reg_num_features` (2000) ⚠️ **MISMATCH**: script default 5000, config has 2000

🔧 **Issues Found**
1. **Hardcoded values in script** (lines 295-300):
   ```python
   if reference_markers is None:
       reference_markers = ['DAPI', 'SMA']  # Should use config value

   max_image_dim = 6000  # Hardcoded! Should be configurable
   ```

2. **Memory optimization parameters not exposed**:
   - `max_image_dim_px = 6000` (line 402) - Critical for RAM control, not configurable!
   - `pyvips.cache_set_max(0)` (line 393) - Good but not documented

### Recommendations
⚠️ **NEEDS IMPROVEMENT**
1. Add `--max-image-dim` parameter (for RAM control)
2. Remove hardcoded `reference_markers = ['DAPI', 'SMA']` fallback
3. Consider adding `--disable-pyvips-cache` flag
4. **Align script defaults with config defaults** OR **always pass from config**

**Suggested config additions:**
```groovy
// Registration - VALIS memory optimization
reg_max_image_dim         = 6000    // Limit cached image size (critical for RAM)
reg_disable_pyvips_cache  = true    // Disable pyvips cache to reduce memory
```

---

## 3. SEGMENT.PY (StarDist Segmentation)

### Current Arguments
```python
--image                   # Multichannel OME-TIFF (REQUIRED)
--model-dir              # StarDist model directory (REQUIRED)
--model-name             # StarDist model name (REQUIRED)
--output-dir             # Output directory (default: './segmentation_output')
--dapi-channel           # DAPI channel index (default: 0)
--use-gpu                # Use GPU acceleration (flag, default: False)
--n-tiles                # Number of tiles Y X (default: [16, 16])
--expand-distance        # Distance to expand nuclei (default: 10)
--pmin                   # Lower percentile for normalization (default: 1.0)
--pmax                   # Upper percentile for normalization (default: 99.8)
```

### Analysis
✅ **Keep as CLI**
- `--image` - Sample-specific input
- `--output-dir` - Process-specific output

✅ **Should be in config (SOME MISSING)**
- `--model-dir` → `params.segmentation_model_dir` ✅ (exists)
- `--model-name` → `params.segmentation_model` ✅ (exists)
- `--dapi-channel` → No config param (always 0, could be hardcoded)
- `--use-gpu` → `params.seg_gpu` ✅ (exists)
- `--n-tiles` → `params.seg_n_tiles_y`, `params.seg_n_tiles_x` ⚠️ **MISMATCH**: script default [16,16], module calls with [16,16]
- `--expand-distance` → `params.seg_expand_distance` ⚠️ **MISSING in config**
- `--pmin` → `params.seg_pmin` ✅ (exists: 1.0)
- `--pmax` → `params.seg_pmax` ✅ (exists: 99.8)

### Recommendations
⚠️ **NEEDS IMPROVEMENT**
1. **Add missing config parameter:**
   ```groovy
   seg_expand_distance    = 10    // Distance (pixels) to expand nuclei for whole-cell masks
   seg_n_tiles_y          = 16    // Number of tiles in Y direction (or rename existing)
   seg_n_tiles_x          = 16    // Number of tiles in X direction
   ```

2. **Consider hardcoding** `--dapi-channel` to 0 (always the case in your pipeline)

---

## 4. QUANTIFY.PY (Cell Quantification)

### Current Arguments
```python
--mode                    # Processing mode: 'cpu' or 'gpu' (default: 'cpu')
--channel_tiff           # Single channel TIFF image (REQUIRED)
--mask_file              # Segmentation mask (.npy) (REQUIRED)
--outdir                 # Output directory (default: '.')
--output_file            # Output CSV filename (OPTIONAL)
--min_area               # Minimum cell area (default: 0)
--channel-name           # Explicit channel name (OPTIONAL)
```

### Analysis
✅ **Keep as CLI**
- `--channel_tiff` - Sample/channel-specific input
- `--mask_file` - Sample-specific segmentation mask
- `--outdir` - Process-specific output directory
- `--output_file` - Process-specific output filename
- `--channel-name` - Channel-specific metadata

✅ **Already in config (GOOD)**
- `--min_area` → `params.quant_min_area` (10)

❌ **Non-useful Arguments**
- `--mode` - **UNUSED!** The script has this parameter but then has `if False:` on line 516, meaning GPU mode is **never executed**
  - This is technical debt from refactoring
  - Either remove `--mode` or fix the GPU execution path

🔧 **Issues Found**
1. **Dead code** (lines 516-534):
   ```python
   if False:  # This is ALWAYS False - GPU code never runs!
       logger.info('Running GPU quantification')
       try:
           run_quantification_gpu(...)
       except ImportError:
           run_quantification(...)  # Fallback to CPU
   else:
       logger.info('Running CPU quantification')
       run_quantification(...)
   ```

### Recommendations
⚠️ **NEEDS CLEANUP**
1. **Option A**: Remove `--mode` argument entirely (simplest)
   - GPU code is dead anyway (line 516: `if False:`)
   - Remove `run_quantification_gpu()` function
   - Keep only CPU implementation

2. **Option B**: Fix GPU implementation (if needed)
   - Change `if False:` to `if args.mode == 'gpu':`
   - Add `params.quant_gpu` to config (already exists as `true`)
   - Pass `--mode gpu` from Nextflow module

**Recommended config:**
```groovy
quant_gpu       = true     // Use GPU for quantification (if implemented)
quant_min_area  = 10       // Minimum cell area (pixels) ✅ Already exists
```

---

## 5. SPLIT_MULTICHANNEL.PY (Channel Splitting)

### Current Arguments
```python
input                     # Input multichannel TIFF (REQUIRED, positional)
output_dir                # Output directory (REQUIRED, positional)
--is-reference           # Save DAPI channel flag (default: False)
--channels               # Channel names (OPTIONAL, default: read from OME metadata)
```

### Analysis
✅ **Keep as CLI**
- `input` - Sample-specific input file
- `output_dir` - Process-specific output directory
- `--is-reference` - Sample-specific boolean (varies per sample)
- `--channels` - Sample-specific metadata (from meta map)

❌ **Non-useful Arguments**
- None - all arguments are used appropriately

### Recommendations
✅ **PERFECT** - Simple, clean interface:
- Positional arguments for required inputs
- Metadata from Nextflow meta map
- No configuration needed

---

## 6. GPU/CPU Registration Scripts (register_gpu.py, register_cpu.py, etc.)

### Analysis Summary
These scripts have **many parameters** that are **already properly configured**:

✅ **Already in nextflow.config:**
- Crop sizes (affine, diffeomorphic, micro)
- Overlap percentages
- Number of features
- Optimization tolerances
- Padding modes
- Feature detector settings

⚠️ **Potential Issues:**
1. **Too many similar parameters** - Consider consolidating:
   - `gpu_reg_*` vs `cpu_reg_*` vs `cpu_multires_*` vs `cpu_cdm_*`
   - Total: ~30+ registration parameters in config
   - Many have same values across methods

2. **Defaults in scripts don't match config** (need verification)

### Recommendations
🔧 **Consider consolidation:**
```groovy
// Instead of separate params for each method, use shared params with method-specific overrides
registration {
    feature_detector     = 'superpoint'
    feature_max_dim      = 2048
    n_features           = 5000

    affine {
        crop_size         = 10000
        overlap_percent   = 10.0
        n_features        = 2000  // Override default
    }

    diffeomorphic {
        crop_size         = 2000
        overlap_percent   = 20.0
        opt_tol           = 1e-6
        inv_tol           = 1e-6
        sigma_diff        = 20
        radius            = 20
    }
}
```

---

## Summary of Recommendations

### 🔴 CRITICAL Issues

1. **quantify.py**: Dead GPU code (`if False:`) - **Remove or fix**
2. **register.py**: Hardcoded `max_image_dim = 6000` - **Add to config**
3. **register.py**: Defaults don't match config values - **Align or always pass from Nextflow**

### 🟡 IMPROVEMENTS Needed

1. **segment.py**: Add missing `params.seg_expand_distance` to config
2. **preprocess.py**: Document that all params are already in config ✅
3. **Config consolidation**: Consider grouping related registration params

### ✅ GOOD Practices Found

1. **split_multichannel.py**: Perfect CLI interface
2. **Most scripts**: Proper separation of variable vs configurable params
3. **Module files**: Correctly pass config params to scripts

---

## Proposed nextflow.config Additions

```groovy
params {
    // ============================================================================
    // Segmentation Parameters (additions)
    // ============================================================================
    seg_expand_distance    = 10          // Distance to expand nuclei for whole-cell masks
    seg_n_tiles_y          = 16          // Number of tiles in Y direction
    seg_n_tiles_x          = 16          // Number of tiles in X direction

    // ============================================================================
    // Registration - VALIS Memory Optimization (additions)
    // ============================================================================
    reg_max_image_dim         = 6000     // Limit cached image size (critical for RAM control)
    reg_disable_pyvips_cache  = true     // Disable pyvips cache to reduce memory usage

    // ============================================================================
    // Quantification Parameters (clarification)
    // ============================================================================
    quant_gpu              = true        // Use GPU for quantification (currently unused in code)
    quant_min_area         = 10          // Minimum cell area (pixels) ✅ Already exists
}
```

---

## Script-by-Script Summary Table

| Script | CLI Args | Config Params | Issues | Status |
|--------|----------|---------------|--------|--------|
| preprocess.py | 2 required, 7 configurable | ✅ All in config | None | ✅ Perfect |
| register.py | 3 required, 5 configurable | ⚠️ Some mismatches | Hardcoded values | 🟡 Needs work |
| segment.py | 2 required, 8 configurable | ⚠️ 1 missing | Missing param | 🟡 Minor fix |
| quantify.py | 5 required, 2 configurable | ✅ In config | Dead GPU code | 🔴 Cleanup needed |
| split_multichannel.py | 2 required, 2 optional | N/A | None | ✅ Perfect |
| register_gpu.py | Many | ✅ All in config | Too many params? | 🟡 Consider consolidation |
| register_cpu*.py | Many | ✅ All in config | Too many params? | 🟡 Consider consolidation |

---

## Final Recommendations

### Immediate Actions (High Priority)
1. **Fix quantify.py GPU code** - Remove `if False:` or implement properly
2. **Add to register.py** - Expose `--max-image-dim` parameter
3. **Add to config** - `params.seg_expand_distance = 10`

### Medium Priority
1. **Align defaults** - Ensure script defaults match config values or always pass from Nextflow
2. **Document memory settings** - Add comments about `reg_max_image_dim` importance
3. **Remove unused args** - Clean up `--mode` from quantify.py if not using GPU

### Low Priority (Optional)
1. **Consolidate registration params** - Group related parameters in config
2. **Add validation** - Check that critical params are set (e.g., model paths)
3. **Document presets** - Better document the HIGH_ACCURACY vs FAST presets in config

---

## Files to Modify

1. **bin/quantify.py** - Fix GPU execution logic (line 516)
2. **bin/register.py** - Add `--max-image-dim` parameter, remove hardcoded value
3. **nextflow.config** - Add missing parameters (see "Proposed additions" section)
4. **modules/local/segment.nf** - Pass `params.seg_expand_distance` to script
5. **modules/local/register.nf** - Pass `params.reg_max_image_dim` if added

---

**Generated**: 2025-12-27
**Pipeline Version**: 0.1.0
**Analysis Type**: Comprehensive argument audit
