# Registration QC Decoupling Refactoring

## Overview

This refactoring decouples QC (Quality Control) generation from the specific registration methods (VALIS, GPU, CPU). Previously, each registration method generated its own QC outputs, leading to code duplication and tight coupling.

## Motivation

### Problems with Previous Design

1. **Code Duplication**: QC generation logic was duplicated across 3+ Python scripts:
   - `bin/register.py` (VALIS)
   - `bin/register_gpu.py` (GPU)
   - `bin/register_cpu.py` (CPU)
   - Similar logic in multiple other registration variants

2. **Tight Coupling**: QC generation was embedded within registration processes, making it:
   - Hard to modify QC logic consistently
   - Difficult to skip QC when not needed
   - Impossible to regenerate QC without re-running registration

3. **Inconsistent Outputs**: Each method potentially had slightly different QC implementations

4. **Testing Complexity**: Testing QC required running full registration pipelines

## Solution Architecture

### New Components

```
modules/local/generate_registration_qc.nf    # Nextflow process
bin/generate_registration_qc.py              # Python script for QC generation
```

### Design Principles

1. **Single Responsibility**: QC generation is now a separate, standalone module
2. **Method-Independent**: Works with output from any registration method
3. **Standardized Output**: All methods produce identical QC format
4. **Configurable**: Can be skipped or customized via parameters

### Data Flow

```
Before Refactoring:
┌─────────────────────┐
│ VALIS_ADAPTER       │──┬──> registered images
│  ├─ REGISTER        │  └──> QC outputs (embedded)
│  └─ QC generation   │
└─────────────────────┘

After Refactoring:
┌─────────────────────┐
│ VALIS_ADAPTER       │──┐
│  └─ REGISTER        │  │
└─────────────────────┘  │
                         ├──> registered images ──┐
┌─────────────────────┐  │                        │
│ GPU_ADAPTER         │──┤                        ↓
│  └─ GPU_REGISTER    │  │              ┌──────────────────────┐
└─────────────────────┘  │              │ GENERATE_           │
                         │              │ REGISTRATION_QC     │
┌─────────────────────┐  │              │  (method-           │
│ CPU_ADAPTER         │──┘              │   independent)      │
│  └─ CPU_REGISTER    │                 └──────────────────────┘
└─────────────────────┘                          │
                                                 └──> QC outputs
```

## Implementation Details

### 1. Python Script: `bin/generate_registration_qc.py`

**Purpose**: Standalone script to generate QC comparing registered vs reference images

**Features**:
- Extracts DAPI channels from both registered and reference images
- Creates standardized RGB composite (Red=registered, Green=reference, Blue=empty)
- Generates two outputs:
  - Full-resolution TIFF (compressed, ImageJ-compatible)
  - Downsampled PNG (default 0.25x scale for quick viewing)
- Method-agnostic: works with any registration output
- Batch processing support

**Usage**:
```bash
generate_registration_qc.py \
  --reference patient1_DAPI_CD4_CD8.ome.tif \
  --registered patient1_DAPI_CD4_CD8_registered.ome.tif \
  --output qc/ \
  --scale-factor 0.25
```

### 2. Nextflow Process: `modules/local/generate_registration_qc.nf`

**Purpose**: Wrapper process for the Python QC script

**Inputs**:
- `meta`: Sample metadata
- `registered`: Registered image file
- `reference`: Reference image file

**Outputs**:
- `qc`: QC outputs (PNG + TIFF)
- `versions.yml`: Software versions

**Resource Label**: `process_medium` (QC is computationally lighter than registration)

### 3. Workflow Integration: `subworkflows/local/registration.nf`

**Changes**:

1. **Import the new module**:
   ```groovy
   include { GENERATE_REGISTRATION_QC } from '../../modules/local/generate_registration_qc'
   ```

2. **Remove QC from adapters**: Adapters now only emit `registered` channel

3. **Add decoupled QC step** (STEP 3b):
   ```groovy
   // Branch registered images into reference vs moving
   ch_qc_input = ch_registered.branch {
       reference: it[0].is_reference
       moving: !it[0].is_reference
   }

   // Extract references by patient
   ch_references_for_qc = ch_qc_input.reference
       .map { meta, file -> [meta.patient_id, file] }

   // Join moving images with their patient's reference
   ch_for_qc = ch_qc_input.moving
       .map { meta, file -> [meta.patient_id, meta, file] }
       .join(ch_references_for_qc, by: 0)
       .map { patient_id, meta, registered_file, reference_file ->
           [meta, registered_file, reference_file]
       }

   // Generate QC
   if (!params.skip_registration_qc) {
       GENERATE_REGISTRATION_QC(ch_for_qc)
       ch_qc = GENERATE_REGISTRATION_QC.out.qc
   } else {
       ch_qc = Channel.empty()
   }
   ```

### 4. Adapter Modifications

**Files Modified**:
- `subworkflows/local/adapters/valis_adapter.nf`
- `subworkflows/local/adapters/gpu_adapter.nf`
- `subworkflows/local/adapters/cpu_adapter.nf`

**Changes**:
- Removed `qc` emit from all adapters
- Added comment explaining QC is now decoupled

**Before**:
```groovy
emit:
registered = ch_registered
qc = REGISTER.out.qc
```

**After**:
```groovy
emit:
registered = ch_registered
// QC generation is now decoupled - handled by GENERATE_REGISTRATION_QC module
```

### 5. Configuration: `nextflow.config`

**New Parameters**:
```groovy
// QC Generation (method-independent)
skip_registration_qc = false   // Skip registration QC generation
qc_scale_factor      = 0.25    // Downsampling factor for QC PNG outputs
```

## Benefits of This Refactoring

### 1. **Maintainability**
- Single source of truth for QC logic
- Changes to QC format only need to be made once
- Easier to add new registration methods

### 2. **Flexibility**
- Can skip QC entirely with `--skip_registration_qc`
- Can adjust QC resolution with `--qc_scale_factor`
- Can regenerate QC without re-running registration

### 3. **Consistency**
- All registration methods produce identical QC outputs
- Standardized output naming and format
- Predictable behavior across methods

### 4. **Performance**
- QC can be run in parallel with other post-processing
- Easier to optimize QC generation independently
- Can skip QC during testing/development

### 5. **Testing**
- QC generation can be tested independently
- Mock registered images can be used for QC testing
- No need to run full registration for QC tests

## Migration Guide

### For Users

**No action required!** The QC outputs remain the same:
- Same file naming convention
- Same RGB composite format (Red=registered, Green=reference)
- Same full-res TIFF + downsampled PNG outputs

**New capabilities**:
```bash
# Skip QC generation (faster testing)
nextflow run main.nf --skip_registration_qc

# Adjust QC resolution
nextflow run main.nf --qc_scale_factor 0.5  # Larger PNGs

# Regenerate QC only (future capability)
nextflow run main.nf --step qc --input registered_images.csv
```

### For Developers

**Adding a new registration method**:

1. Create adapter in `subworkflows/local/adapters/`
2. Ensure adapter outputs `[meta, file]` format
3. **Do NOT** implement QC in the adapter
4. Add method to switch statement in `registration.nf`
5. QC will be automatically generated!

**Example**:
```groovy
// subworkflows/local/adapters/new_method_adapter.nf
workflow NEW_METHOD_ADAPTER {
    take:
    ch_images
    ch_grouped_meta

    main:
    // ... registration logic ...

    emit:
    registered = ch_registered
    // QC is handled by GENERATE_REGISTRATION_QC - don't add it here!
}
```

### For Python Script Developers

**QC logic is now centralized in**:
- `bin/generate_registration_qc.py`

**To modify QC outputs**:
1. Edit only this one file
2. Changes apply to all registration methods
3. Test with: `python bin/generate_registration_qc.py --help`

## Output Format Specification

### QC File Naming

For registered image: `patient1_DAPI_CD4_CD8_registered.ome.tif`

**Outputs**:
- `patient1_DAPI_CD4_CD8_registered_QC_RGB_fullres.tif` (full resolution, compressed)
- `patient1_DAPI_CD4_CD8_registered_QC_RGB.png` (downsampled for viewing)

### QC File Contents

**Both outputs contain the same 3-channel RGB composite**:
- **Red channel**: Registered image DAPI (autoscaled 1-99 percentile)
- **Green channel**: Reference image DAPI (autoscaled 1-99 percentile)
- **Blue channel**: Empty (zeros)

**Interpretation**:
- **Red areas**: Only in registered image
- **Green areas**: Only in reference image
- **Yellow areas**: Perfect alignment (red + green = yellow)
- **Dark areas**: Background in both images

### Technical Details

**TIFF Output**:
- Format: ImageJ-compatible multi-channel TIFF
- Compression: zlib
- Metadata: `axes='CYX'`, `mode='composite'`
- BigTIFF: Enabled for large images

**PNG Output**:
- Format: Standard RGB PNG (OpenCV BGR order)
- Resolution: Original × `qc_scale_factor` (default 0.25)
- Compression: PNG standard

## Testing

### Unit Testing QC Script

```bash
# Test with sample data
python bin/generate_registration_qc.py \
  --reference test_data/patient1_ref.ome.tif \
  --registered test_data/patient1_reg.ome.tif \
  --output test_output/ \
  --verbose

# Verify outputs
ls test_output/
# Should contain:
#   patient1_reg_QC_RGB_fullres.tif
#   patient1_reg_QC_RGB.png
```

### Integration Testing

```bash
# Test with full pipeline
nextflow run main.nf \
  -profile test,docker \
  --registration_method gpu \
  --outdir test_results/

# Verify QC outputs
ls test_results/registration/qc/
```

### Regression Testing

Compare QC outputs before and after refactoring:
```bash
# Should produce identical results
diff old_qc/patient1_QC_RGB.png new_qc/patient1_QC_RGB.png
```

## Performance Considerations

### Resource Usage

**QC Generation Process**:
- Label: `process_medium` (6 CPU, 36GB RAM default)
- Time: ~2-5 minutes per image (depending on size)
- Parallelization: Each image processed independently

**Optimization Tips**:
- Use `--skip_registration_qc` during development
- Increase `--qc_scale_factor` for faster QC (lower resolution)
- QC runs in parallel with checkpoint writing

### Memory Usage

QC script uses memory-efficient approach:
- Loads only DAPI channels (not full multi-channel images)
- Downsampling uses scikit-image with anti-aliasing
- No large intermediate arrays kept in memory

## Future Enhancements

### Potential Improvements

1. **Multiple QC Metrics**:
   - Add correlation coefficient calculation
   - Generate difference maps
   - Compute registration error heatmaps

2. **Interactive QC**:
   - Generate HTML reports with image sliders
   - Side-by-side comparison views
   - Zoom and pan capabilities

3. **Batch QC Reports**:
   - MultiQC-style summary for all samples
   - Aggregate registration quality metrics
   - Flag potential registration failures

4. **Configurable Channels**:
   - Allow QC on channels other than DAPI
   - Support multi-channel QC overlays
   - User-specified channel selection

### Implementation Notes

All future enhancements can be added to `bin/generate_registration_qc.py` without modifying:
- Registration methods
- Adapters
- Workflow structure

This demonstrates the power of decoupling!

## Troubleshooting

### QC Generation Fails

**Issue**: QC process fails with dimension mismatch
```
ValueError: Image dimensions do not match!
  Reference: (10000, 10000)
  Registered: (9999, 10000)
```

**Solution**: This indicates registration didn't work correctly. Check:
1. Registration logs for errors
2. Reference image selection is correct
3. Padding is enabled if images have different sizes

### Missing DAPI Channel

**Issue**: QC fails with "DAPI channel not found"

**Solution**:
- Ensure filename includes channel names: `patient1_DAPI_CD4_CD8.ome.tif`
- DAPI channel is identified by "DAPI" in channel name (case-insensitive)
- If no DAPI, channel 0 is used by default

### QC Outputs Not Published

**Issue**: QC files not appearing in output directory

**Check**:
```bash
# Verify QC was run
grep "GENERATE_REGISTRATION_QC" .nextflow.log

# Check if QC was skipped
grep "skip_registration_qc" .nextflow.log
```

**Solution**: Ensure `--skip_registration_qc` is not set to `true`

## Summary

This refactoring achieves:
- ✅ **Decoupling**: QC is independent of registration methods
- ✅ **DRY Principle**: Single implementation, no duplication
- ✅ **Flexibility**: Can skip, customize, or extend QC easily
- ✅ **Maintainability**: Changes in one place affect all methods
- ✅ **Testing**: QC can be tested independently

The registration workflow now follows best practices:
1. **Separation of Concerns**: Registration vs QC
2. **Single Responsibility**: Each module has one job
3. **Open/Closed Principle**: Easy to extend, no need to modify existing code
4. **Interface Segregation**: Clean adapter interfaces
5. **Dependency Inversion**: QC depends on standard [meta, file] format, not specific methods

---

**Author**: AI Refactoring Agent
**Date**: 2025-12-27
**Version**: 1.0
