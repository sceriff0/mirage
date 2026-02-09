# VALIS Registration Error Metrics

This document explains how registration errors are computed in the VALIS summary CSV file.

## Overview

Registration error measures **how well two images are aligned** by calculating the distance between matched feature points. Lower values indicate better alignment.

---

## Key Concepts

### Matched Feature Points

During registration, VALIS detects distinctive features (keypoints) in each image and finds corresponding matches between image pairs:

- **`xy_matched_to_prev`**: Coordinates of features in the current (moving) image
- **`xy_in_prev`**: Coordinates of the corresponding matched features in the reference (fixed) image

These matched point pairs are the basis for all error calculations.

### Physical Units

All distance metrics (`*_D`) are reported in **physical units** (typically micrometers, µm) rather than pixels:

```
distance_physical = distance_pixels × resolution
```

Where `resolution` = physical size per pixel, extracted from image metadata.

---

## Error Metrics

### 1. Distance Error (D)

**What it measures**: The median Euclidean distance between matched feature points.

**Formula**:
```
D = median( sqrt[(x1 - x2)^2 + (y1 - y2)^2] ) × resolution
```

Where (x1, y1) and (x2, y2) are corresponding matched keypoints.

| Metric | Description |
|--------|-------------|
| `original_D` | Distance **before** any registration (unaligned images) |
| `rigid_D` | Distance **after** rigid registration |
| `non_rigid_D` | Distance **after** non-rigid registration |

**Interpretation**:
- `original_D` of 50 µm means matched points are ~50 µm apart before alignment
- `rigid_D` of 5 µm means rigid registration reduced misalignment to ~5 µm
- Lower is better

---

### 2. Relative Target Registration Error (rTRE)

**What it measures**: Distance error normalized by the reference image diagonal, making it comparable across different image sizes.

**Formula**:
```
rTRE = distance_pixels / reference_diagonal
```

Where:
```
reference_diagonal = sqrt(height^2 + width^2)  of the reference image
```

| Metric | Description |
|--------|-------------|
| `original_rTRE` | Relative error **before** registration |
| `rigid_rTRE` | Relative error **after** rigid registration |
| `non_rigid_rTRE` | Relative error **after** non-rigid registration |

**Interpretation**:
- Unitless ratio (typically 0.0 to 1.0)
- `rTRE = 0.01` means error is 1% of the image diagonal
- Allows comparison between images of different resolutions/sizes

---

### 3. Mean Errors (Weighted Averages)

When registering multiple images, mean errors summarize overall registration quality.

| Metric | Description |
|--------|-------------|
| `mean_original_D` | Weighted average of `original_D` across all images |
| `mean_rigid_D` | Weighted average of `rigid_D` across all images |
| `mean_non_rigid_D` | Weighted average of `non_rigid_D` across all images |

**Weighting**: Each image's error is weighted by its **number of matched features**. Images with more feature matches contribute more to the mean.

```python
mean_D = sum(D_i × n_i) / sum(n_i)
```

Where `n_i` = number of matched features for image i.

---

## Computation Process

### Original Error (Pre-Registration)

```
1. Get matched keypoints from both images
2. Transform both to common coordinate space (using padding matrices only)
3. Calculate Euclidean distance between each matched pair
4. Take median of all distances
5. Multiply by resolution for physical units
6. Divide by diagonal for rTRE
```

### Post-Registration Error

```
1. Get matched keypoints from both images
2. Apply transformation matrix M to warp points to aligned space
3. (For non-rigid: also apply displacement field)
4. Calculate Euclidean distance between each matched pair
5. Take median of all distances
6. Multiply by resolution for physical units
7. Divide by diagonal for rTRE
```

The key difference is step 2-3: post-registration applies the computed transformations before measuring distances.

---

## Summary Table

| Column | Stage | Unit | Description |
|--------|-------|------|-------------|
| `original_D` | Before | µm | Median distance, unaligned |
| `original_rTRE` | Before | ratio | Relative error, unaligned |
| `rigid_D` | After rigid | µm | Median distance, rigid-aligned |
| `rigid_rTRE` | After rigid | ratio | Relative error, rigid-aligned |
| `non_rigid_D` | After non-rigid | µm | Median distance, fully aligned |
| `non_rigid_rTRE` | After non-rigid | ratio | Relative error, fully aligned |
| `mean_original_D` | Before | µm | Weighted avg across all images |
| `mean_rigid_D` | After rigid | µm | Weighted avg across all images |
| `mean_non_rigid_D` | After non-rigid | µm | Weighted avg across all images |

---

## Example Interpretation

| Image | original_D | rigid_D | non_rigid_D |
|-------|------------|---------|-------------|
| sample_A | 45.2 µm | 3.1 µm | 1.2 µm |
| sample_B | 38.7 µm | 4.5 µm | 1.8 µm |

**Reading this**:
- Before registration, matched features were ~40-45 µm apart
- Rigid registration reduced error to ~3-5 µm (>90% improvement)
- Non-rigid registration further reduced to ~1-2 µm

---

## Source Code Reference

- Error computation: `valis_lib/registration.py:3770-3975` (`measure_error()`)
- Distance calculation: `valis_lib/warp_tools.py:1664-1683` (`calc_d()`)
