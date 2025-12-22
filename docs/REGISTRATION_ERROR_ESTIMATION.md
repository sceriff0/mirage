# Registration Error Estimation

## Overview

This pipeline implements **two complementary approaches** to quantify registration quality:

1. **Feature-Based TRE** - Fast, sparse measurements at keypoints
2. **Segmentation-Based Metrics** - Dense, biologically meaningful IoU/Dice scores

Both methods provide independent validation of registration quality and together offer comprehensive error assessment.

## Methods

---

## Method 1: Feature-Based Error Estimation

### 1.1 Pre-Registration Feature Matching (`COMPUTE_FEATURES`)

**Purpose**: Establish baseline feature correspondences before registration.

**Process**:
- Detect features in reference image (DAPI channel)
- Detect features in moving image (unregistered)
- Match features between reference and moving
- Save matched keypoint coordinates

**Output**: `{moving}_features.json`

### 1.2 Registration (`GPU_REGISTER` / `CPU_REGISTER` / `REGISTER`)

**Purpose**: Align moving image to reference.

**Methods**:
- **GPU/CPU**: Crop-based affine + diffeomorphic registration
- **VALIS**: SuperPoint features + optical flow deformation

**Output**: `{moving}_registered.ome.tif`

### 1.3 Feature-Based Error Estimation (`ESTIMATE_REG_ERROR`)

**Purpose**: Measure residual misalignment using direct feature matching.

**Simplified Approach** (No transformation estimation):

#### Step 1: Load Pre-Registration Baseline
```python
pre_distances = ||ref_kp - mov_kp||  # Before registration
baseline_error = mean(pre_distances)
```

#### Step 2: Re-detect Features After Registration
```python
ref_kp_post, ref_desc_post = detect(reference_img)
reg_kp_post, reg_desc_post = detect(registered_img)
```

#### Step 3: Match Features After Registration
```python
matches = match_features(ref_desc_post, reg_desc_post)
matched_ref_kp = ref_kp_post[matches.ref_indices]
matched_reg_kp = reg_kp_post[matches.reg_indices]
```

#### Step 4: Compute Post-Registration TRE (Direct Residual)
```python
tre_distances = ||matched_ref_kp - matched_reg_kp||
mean_tre = mean(tre_distances)  # PRIMARY METRIC
```

#### Step 5: Compute Improvement
```python
improvement = baseline_error - mean_tre
improvement_pct = (improvement / baseline_error) * 100
```

**Output**: `{moving}_registration_error.json`, `{moving}_tre_histogram.png`

**Advantages**:
- ✅ Fast and simple
- ✅ Independent validation (no transform estimation)
- ✅ No circular dependencies
- ✅ Works with any registration method

**Limitations**:
- ⚠️ Sparse measurements (only at keypoints)
- ⚠️ Different features before/after (not true tracking)

---

## Method 2: Segmentation-Based Error Estimation

### 2.1 Nucleus Segmentation (`ESTIMATE_REG_ERROR_SEGMENTATION`)

**Purpose**: Dense, biologically meaningful registration quality metrics.

**Process**:

#### Step 1: Segment DAPI Nuclei
```python
# Otsu threshold + morphological operations
mask_ref = segment_nuclei(reference_img)
mask_reg = segment_nuclei(registered_img)
```

#### Step 2: Compute Overlap Metrics
```python
# Intersection over Union (Jaccard)
IoU = |ref ∩ reg| / |ref ∪ reg|

# Dice Coefficient
Dice = 2|ref ∩ reg| / (|ref| + |reg|)

# Coverage
Coverage = |ref ∩ reg| / |ref|
```

#### Step 3: Cell Count Correlation
```python
n_cells_ref = max_label(mask_ref)
n_cells_reg = max_label(mask_reg)
correlation = min(n_ref, n_reg) / max(n_ref, n_reg)
```

#### Step 4: Spatial Error Distribution
```python
# Divide image into grid, compute IoU per region
# Identifies areas with poor registration
spatial_iou = [iou_grid_cell(i,j) for all grid cells]
```

**Output**: `{moving}_segmentation_error.json`, `{moving}_segmentation_overlay.png`

**Advantages**:
- ✅ Dense measurement (entire image)
- ✅ Biologically interpretable
- ✅ Independent of feature detectors
- ✅ Standard metric (used in medical imaging)
- ✅ Robust to feature detection failures

**Limitations**:
- ⚠️ Requires good segmentation
- ⚠️ Slower than feature-based (segmentation overhead)

---

## Metrics Explained

### Feature-Based Metrics

#### Baseline Distances (Before Registration)
- **Mean/Median**: Average misalignment before registration
- **Purpose**: Establishes pre-registration baseline

#### Target Registration Error (TRE)
- **Mean/Median**: Average residual error after registration
- **95th percentile**: Worst-case errors (excluding outliers)
- **Purpose**: Residual misalignment at matched features
- **Lower is better**: Perfect registration would have TRE ≈ 0

#### Improvement Metrics
- **TRE improvement (pixels)**: `baseline_distance - TRE`
- **TRE improvement (%)**: `(improvement / baseline_distance) × 100`
- **Purpose**: Quantify registration effectiveness

### Segmentation-Based Metrics

#### IoU (Intersection over Union / Jaccard Index)
- **Formula**: `|A ∩ B| / |A ∪ B|`
- **Range**: [0, 1], higher is better
- **Interpretation**:
  - **> 0.75**: Excellent alignment
  - **0.50 - 0.75**: Good alignment
  - **0.25 - 0.50**: Fair alignment
  - **< 0.25**: Poor alignment

#### Dice Coefficient
- **Formula**: `2|A ∩ B| / (|A| + |B|)`
- **Range**: [0, 1], higher is better
- **More sensitive** to small changes than IoU

#### Coverage
- **Formula**: `|A ∩ B| / |A|`
- **Interpretation**: Fraction of reference covered by registered
- **Asymmetric metric** (directional)

#### Cell Count Correlation
- **Formula**: `min(n_ref, n_reg) / max(n_ref, n_reg)`
- **Interpretation**: How well cell counts match

---

## Interpretation

### Good Registration (Feature-Based)
```
Baseline distance:  125.3 pixels
Mean TRE:           2.8 pixels
Improvement:        122.5 pixels (97.8%)
```
✅ Registration successfully aligned images with <3 pixel error.

### Good Registration (Segmentation-Based)
```
IoU:                0.87
Dice:               0.93
Coverage:           0.92
Cell Count Corr:    0.95
```
✅ Excellent overlap, cells align well.

### Poor Registration (Feature-Based)
```
Baseline distance:  125.3 pixels
Mean TRE:           48.2 pixels
Improvement:        77.1 pixels (61.5%)
```
⚠️ Significant residual error remains; registration partially failed.

### Poor Registration (Segmentation-Based)
```
IoU:                0.32
Dice:               0.48
Coverage:           0.45
Cell Count Corr:    0.67
```
⚠️ Low overlap, misalignment visible.

### Failed Registration
```
Feature-Based:
  Baseline:         125.3 pixels
  Mean TRE:         132.7 pixels
  Improvement:      -7.4 pixels (-5.9%)

Segmentation-Based:
  IoU:              0.12
  Dice:             0.21
```
❌ Registration made alignment worse; likely registration failure.

---

## Why These Approaches?

### Feature-Based Advantages
✅ **Fast**: Minutes per image pair
✅ **Simple**: Direct residual measurement
✅ **No estimation errors**: No circular dependencies
✅ **Works with all methods**: VALIS, GPU, CPU registration

### Segmentation-Based Advantages
✅ **Dense measurement**: Entire image evaluated
✅ **Biologically meaningful**: Aligns with cell structures
✅ **Robust**: Independent of feature detector quality
✅ **Standard metric**: Used in medical image registration
✅ **Interpretable**: "90% IoU" is clear and actionable

### Complementary Strengths
| Aspect | Feature-Based | Segmentation-Based |
|--------|---------------|-------------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium |
| **Coverage** | ⭐ Sparse (keypoints) | ⭐⭐⭐ Dense (full image) |
| **Interpretability** | ⭐⭐ Technical | ⭐⭐⭐ Biological |
| **Robustness** | ⭐⭐ Medium | ⭐⭐⭐ High |
| **Dependencies** | Feature detector | Segmentation |

**Recommendation**: Use both for comprehensive validation

---

## File Outputs

### Feature-Based Errors
```
{outdir}/{id}/{registration_method}/registration_errors/
├── slide1_registration_error.json    # Feature-based TRE metrics
├── slide1_tre_histogram.png          # TRE distribution
├── slide2_registration_error.json
├── slide2_tre_histogram.png
└── ...
```

### Segmentation-Based Errors
```
{outdir}/{id}/{registration_method}/registration_errors_segmentation/
├── slide1_segmentation_error.json    # IoU/Dice metrics
├── slide1_segmentation_overlay.png   # Visual overlay
├── slide2_segmentation_error.json
├── slide2_segmentation_overlay.png
└── ...
```

### Pre-Registration Features
```
{outdir}/{id}/{registration_method}/features/
├── slide1_features.json    # Pre-registration matched keypoints
├── slide2_features.json
└── ...
```

---

## Configuration

### Feature-Based Parameters
```groovy
params {
    feature_detector = 'superpoint'    // superpoint, disk, dedode, brisk
    feature_max_dim = 2048            // Downsample for speed
    feature_n_features = 5000         // Number of features to match
}
```

### Segmentation-Based Parameters
```groovy
params {
    enable_segmentation_error = true  // Enable/disable (default: true)
    min_nucleus_size = 100            // Min nucleus size (pixels)
    max_nucleus_size = 5000           // Max nucleus size (pixels)
}
```

### Feature Detectors
- **SuperPoint** (default): Deep learning, robust, requires GPU
- **DISK**: Deep learning, robust, requires GPU
- **DeDoDe**: Deep learning, robust, requires GPU
- **BRISK**: Traditional, CPU-only, faster but less accurate

---

## Usage Examples

### Run with Both Error Metrics (Default)
```bash
nextflow run main.nf \\
  --input "images/*.tif" \\
  --outdir results \\
  --registration_method gpu \\
  --enable_segmentation_error true  # Both metrics
```

### Run Only Feature-Based (Faster)
```bash
nextflow run main.nf \\
  --input "images/*.tif" \\
  --outdir results \\
  --registration_method gpu \\
  --enable_segmentation_error false  # Skip segmentation
```

### Customize Segmentation Parameters
```bash
nextflow run main.nf \\
  --input "images/*.tif" \\
  --outdir results \\
  --min_nucleus_size 50 \\      # Smaller nuclei
  --max_nucleus_size 10000      # Larger max size
```

---

## Summary

This pipeline provides **two complementary error estimation methods**:

1. **Feature-Based TRE**: Fast, sparse measurements using direct residual matching
   - Simple and robust
   - No transformation estimation
   - Independent validation

2. **Segmentation-Based IoU/Dice**: Dense, biologically meaningful overlap metrics
   - Comprehensive coverage
   - Interpretable results
   - Standard in medical imaging

**Together**, they provide thorough registration quality assessment from both technical (feature alignment) and biological (tissue structure alignment) perspectives.
- **DeDoDe**: Deep learning, newer, requires GPU
- **BRISK**: Traditional, fast, CPU-only

## References

1. Gatenbee et al. (2023). "VALIS: Virtual Alignment of pathoLogy Image Series." *Nature Communications*
2. Original VALIS implementation: https://github.com/MathOnco/valis
3. VALIS `measure_error()` method: Tracks same keypoints through registration

## Implementation Details

### Key Functions

**`compute_features.py`**
- Detects and matches features before registration
- Saves matched keypoint coordinates for later TRE calculation

**`estimate_registration_error.py`**
- Loads pre-registration keypoints
- Estimates global transform from registered image
- Applies transform to original keypoints
- Computes TRE as residual distance

### Transformation Estimation

The global transform is estimated by:
1. Detecting features in reference and registered images
2. Matching features using SuperGlue/LightGlue
3. Estimating affine transform using RANSAC
4. This approximates the composite effect of all crop-based registrations

### Coordinate Systems

All coordinates are in the **downsampled image space** (max_dim):
- Features detected at reduced resolution for speed
- Transform estimated at same resolution
- TRE measured in downsampled pixels

To convert to full resolution:
```python
full_res_tre = downsampled_tre * (original_size / max_dim)
```

## Troubleshooting

### Warning: "Could not estimate global transformation"
**Cause**: Too few feature matches between reference and registered
**Solution**: Check if registration completely failed; try different detector

### Warning: "Pre-registration matched keypoints not available"
**Cause**: `compute_features.py` didn't save keypoints
**Solution**: Ensure COMPUTE_FEATURES ran before ESTIMATE_REG_ERROR

### High TRE values
**Cause**: Registration failed or partially failed
**Solution**: Check QC plots; adjust registration parameters

### Negative improvement
**Cause**: Registration made alignment worse
**Solution**: Registration catastrophically failed; check input images

---

**For questions or issues**: Check implementation in `bin/estimate_registration_error.py` and `modules/local/estimate_registration_error.nf`
