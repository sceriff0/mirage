# Per-Nucleus Registration Error: In-Depth Explanation

## Overview

Per-nucleus registration error is an **instance-level correspondence metric** that quantifies registration quality by measuring the spatial displacement of individual nuclei between reference and registered images. This approach provides significantly more information than bulk overlap metrics (Dice/IoU) by treating each nucleus as a distinct landmark.

---

## What Is It?

### Definition

Per-nucleus registration error measures the **centroid distance** between corresponding nuclei in the reference and registered images. Each nucleus acts as a biological landmark, and the error is the Euclidean distance between matched pairs.

### Mathematical Formulation

For a reference image with nuclei `R = {r₁, r₂, ..., rₙ}` and registered image with nuclei `S = {s₁, s₂, ..., sₘ}`:

1. **Extract centroids**:
   - `C_R = {c(r₁), c(r₂), ..., c(rₙ)}` where `c(rᵢ) = (yᵢ, xᵢ)`
   - `C_S = {c(s₁), c(s₂), ..., c(sₘ)}` where `c(sⱼ) = (yⱼ, xⱼ)`

2. **Build cost matrix**:
   - `D[i,j] = ||c(rᵢ) - c(sⱼ)||₂` (Euclidean distance)
   - Shape: `n × m`

3. **Solve assignment problem** (Hungarian algorithm):
   - Find optimal one-to-one matching that minimizes total cost
   - Result: pairs `(rᵢ, sⱼ)` where each nucleus matched at most once

4. **Filter by threshold**:
   - Keep only matches where `D[i,j] ≤ max_distance` (default 50 pixels)
   - Prevents spurious long-distance matches

5. **Compute statistics**:
   - Mean error: `E_mean = (1/k) Σ D[matched pairs]`
   - Median, Q95, etc.

---

## Why Is This Better Than Dice/IoU?

### 1. **Disentangles Spatial Error from Segmentation Error**

**Problem with Dice**: A Dice score of 0.85 could mean:
- Perfect registration + 15% segmentation error
- Perfect segmentation + 15% registration error
- 7.5% registration + 7.5% segmentation error
- **You cannot separate these contributions!**

**Per-nucleus error**: By using **centroids** (which are robust to boundary segmentation errors), you measure registration displacement more directly.

#### Example:

```
Scenario 1: Perfect registration, imperfect segmentation
┌─────────────────────────────────────────────────┐
│ Reference nucleus:  ⬤  (centroid at pixel 100) │
│ Registered nucleus: ⬤  (centroid at pixel 100) │
│ Boundary mismatch:  ████ vs ███                │
│                                                  │
│ → Dice: 0.75 (boundary errors)                 │
│ → Centroid distance: 0 pixels ✓ TRUE ERROR     │
└─────────────────────────────────────────────────┘

Scenario 2: Imperfect registration, perfect segmentation
┌─────────────────────────────────────────────────┐
│ Reference nucleus:  ⬤      (centroid at 100)   │
│ Registered nucleus:     ⬤  (centroid at 105)   │
│ Same shape, shifted 5px                         │
│                                                  │
│ → Dice: 0.60 (spatial shift)                   │
│ → Centroid distance: 5 pixels ✓ TRUE ERROR     │
└─────────────────────────────────────────────────┘
```

**Dice conflates both errors**; **centroid distance isolates registration error**.

---

### 2. **Instance-Level Resolution**

**Dice**: Single global number for entire image
```json
{
  "dice": 0.87  // What does this tell you?
}
```

**Per-nucleus**: Distribution of errors across all nuclei
```json
{
  "matched_pairs": 342,
  "mean_centroid_distance": 2.3,
  "median_centroid_distance": 1.8,
  "q95_centroid_distance": 6.4,
  "max_centroid_distance": 15.2,
  "distances": [0.5, 1.2, 2.1, ..., 15.2]  // Full distribution!
}
```

**What you can learn**:
- **Mean/median**: Typical registration accuracy
- **Q95**: Worst 5% of nuclei (critical regions?)
- **Max**: Absolute worst case
- **Distribution shape**: Are errors uniform or are there outliers?

#### Example Use Cases:

**Case 1: Good registration**
```
Mean: 1.2px, Median: 1.0px, Q95: 3.5px
→ Most nuclei well-aligned, few outliers
→ Safe to proceed with analysis
```

**Case 2: Spatially-varying error**
```
Mean: 3.5px, Median: 2.0px, Q95: 15.2px
→ Median is good, but Q95 is terrible!
→ Image edges or specific regions failing
→ Need to investigate spatial distribution
```

**Case 3: Systematic failure**
```
Mean: 12.4px, Median: 11.8px, Q95: 25.6px
→ Entire image misaligned
→ Registration failed, do not use results
```

---

### 3. **Provides Matched vs Unmatched Counts**

**Critical for detecting registration failures:**

```json
{
  "reference_nuclei": 500,
  "registered_nuclei": 485,
  "matched_pairs": 470,
  "unmatched_reference": 30,
  "unmatched_registered": 15
}
```

**Interpretation**:
- **94% match rate** (470/500): Good overall
- **30 unmatched reference nuclei**: Potentially moved out of frame or occluded
- **15 unmatched registered nuclei**: False detections or nuclei that moved into frame

**Red flags**:
- Low match rate (<80%): Registration likely shifted image significantly
- Many unmatched nuclei: Image cropping, rotation, or severe deformation

---

### 4. **Spatially-Resolved Quality Assessment**

You can analyze **where** registration fails by grouping nuclei by location:

```python
# Pseudocode for spatial analysis
center_nuclei = nuclei with centroid in center 50% of image
edge_nuclei = nuclei with centroid in outer 25% of image

center_error = mean(distances[center_nuclei])  # e.g., 1.5px
edge_error = mean(distances[edge_nuclei])      # e.g., 8.3px

# Edge is 5.5× worse! Registration warping inadequate at boundaries
```

**This tells you**:
- Whether to trust analysis in specific regions
- If you need non-rigid vs affine registration
- Where to place manual QC checkpoints

---

## How Does the Hungarian Algorithm Work?

The **Hungarian algorithm** (also called Kuhn-Munkres) solves the **assignment problem**:

> Given n reference nuclei and m registered nuclei with pairwise distances,
> find the one-to-one matching that minimizes total distance.

### Visual Example:

```
Reference nuclei (R):     Registered nuclei (S):
   R1: (10, 20)              S1: (11, 21)   ← 1.4px from R1
   R2: (50, 30)              S2: (52, 31)   ← 2.2px from R2
   R3: (80, 60)              S3: (79, 58)   ← 2.2px from R3

Cost matrix (distances):
        S1    S2    S3
   R1  1.4   45.6  87.2
   R2  44.7   2.2  32.8
   R3  86.3  34.2   2.2

Hungarian algorithm finds optimal matching:
   R1 ↔ S1 (cost: 1.4)
   R2 ↔ S2 (cost: 2.2)
   R3 ↔ S3 (cost: 2.2)

Total cost: 5.8 (minimized across all possible matchings)
```

### Why Not Just Nearest Neighbors?

Nearest neighbor can give **suboptimal** results:

```
Greedy nearest neighbor:
   R1 → closest S1 (1.4px)  ✓
   R2 → closest S2 (2.2px)  ✓
   R3 → closest S3 (2.2px)  ✓
   Total: 5.8px

Problematic case:
   R1: (10, 10)    S1: (11, 11)  ← 1.4px from R1
   R2: (12, 12)    S2: (50, 50)  ← 1.4px from R2 (also close to S1!)

Greedy approach might match R1→S1 and R2→S2 (total: 55px)
Optimal: R1→S2 and R2→S1 (total: 55px... wait, same here)

BUT in complex scenarios with many nuclei, greedy can fail badly.
```

**Hungarian guarantees global optimum** in `O(n³)` time.

---

## Advantages Over Feature-Based Methods

| Aspect | Feature Matching (SuperPoint/SIFT) | Per-Nucleus Matching |
|--------|-------------------------------------|----------------------|
| **Ground truth** | ❌ Abstract keypoints | ✅ Real biological structures |
| **Circular reasoning** | ⚠️ RANSAC filters geometrically | ✅ Independent segmentation |
| **Correspondence** | ⚠️ Different features before/after | ✅ Same nuclei tracked |
| **Interpretability** | ❌ "Feature distance" unclear | ✅ "Nucleus shifted 3px" intuitive |
| **Biological relevance** | ❌ Not tied to analysis | ✅ Directly relevant to cell analysis |
| **Spatial coverage** | ⚠️ Sparse (1000s keypoints) | ✅ Dense (10,000s nuclei) |

---

## Limitations and Caveats

### 1. **Still Confounded with Segmentation Error**

While **centroid** is more robust than boundary overlap, segmentation errors still matter:

```
Perfect segmentation:          Imperfect segmentation:
     ⬤  centroid: (100, 100)       ⬤    centroid: (102, 98)
                                   (segmentation shifted by 2px)

If registered nucleus is also shifted by 2px:
  → Centroid distance = 0px (looks perfect!)
  → But actual registration has 2px error
```

**Mitigation**: Use high-quality segmentation (StarDist better than thresholding)

### 2. **Assumes Nuclei Correspondence Exists**

**Biological reality**:
- Serial tissue sections may have different nuclei in field of view
- Thick tissue sections: nuclei appear/disappear across z-planes
- Cell division, death, movement in time-series

**The method assumes**: Every nucleus in reference has a corresponding nucleus in registered image within `max_distance` pixels.

**When violated**: Unmatched nuclei are reported, but interpretation is ambiguous:
- Registration failure?
- Biological difference?
- Segmentation missed a nucleus?

### 3. **Centroid ≠ Entire Cell**

Centroids only capture **translational error** at the nucleus center. They don't capture:
- Rotational misalignment
- Anisotropic stretching
- Local warping within a cell

**Example**:
```
Reference nucleus:  ⬤ (circular)
Registered:         ⬭ (elliptical, rotated)
Centroids match → distance = 0px

But cell is deformed! This is invisible to centroid-based metrics.
```

**Better**: Compute additional shape similarity metrics (area, eccentricity, orientation)

### 4. **Sensitive to `max_distance` Threshold**

```python
max_distance = 50.0  # pixels
```

**Too low** (e.g., 10px): Reject valid matches in poorly-registered images
**Too high** (e.g., 200px): Include spurious matches between unrelated nuclei

**Recommendation**: Set `max_distance` based on:
- Typical nucleus spacing (avoid matching to wrong nucleus)
- Expected registration error (if you expect 5-10px error, use 50px threshold)

---

## Statistical Interpretation

### Metrics Provided

```python
{
    "matched_pairs": 342,                   # n matched nuclei
    "mean_centroid_distance": 2.34,         # μ
    "median_centroid_distance": 1.82,       # median (robust to outliers)
    "std_centroid_distance": 1.56,          # σ
    "q25_centroid_distance": 1.05,          # 25th percentile
    "q75_centroid_distance": 3.12,          # 75th percentile (IQR = Q75-Q25)
    "q90_centroid_distance": 4.87,          # 90th percentile
    "q95_centroid_distance": 6.43,          # 95th percentile
    "max_centroid_distance": 15.23,         # worst case
    "unmatched_reference": 8,               # missing matches
    "unmatched_registered": 5,
    "distances": [0.5, 0.8, ..., 15.23]    # full distribution
}
```

### What Each Metric Tells You

| Metric | Interpretation | Clinical Relevance |
|--------|---------------|-------------------|
| **Mean** | Average error across all nuclei | Typical registration accuracy |
| **Median** | Middle value (robust to outliers) | "Most nuclei have this error" |
| **Std** | Variability of errors | Consistency of registration |
| **Q95** | 95% of nuclei have error below this | Conservative quality bound |
| **Max** | Absolute worst nucleus | Safety margin for critical analyses |
| **Unmatched** | Failed correspondences | Completeness of registration |

### Example Interpretation:

```json
{
  "mean": 2.3,
  "median": 1.8,
  "std": 1.2,
  "q95": 5.4
}
```

**Reads as**:
- "On average, nuclei are displaced by **2.3 pixels**"
- "Half of nuclei are within **1.8 pixels** of their true position"
- "95% of nuclei are within **5.4 pixels**"
- "Registration is **consistent** (low std=1.2px)"

**Is this good?**
- For 10× imaging (0.65 µm/pixel): 2.3px = 1.5 µm → **Excellent** (subcellular)
- For 40× imaging (0.16 µm/pixel): 2.3px = 0.37 µm → **Outstanding** (near diffraction limit)
- For 2× imaging (5 µm/pixel): 2.3px = 11.5 µm → **Good** (cell-level accuracy)

**Context matters!** Acceptable error depends on:
1. Pixel resolution
2. Downstream analysis requirements (e.g., proximity analysis needs <5µm)
3. Biological tissue type (brain needs higher accuracy than skin)

---

## Comparison to "Before vs After" Feature Distances

The original `estimate_feature_distances.py` computed distances before/after registration. Here's the comparison:

| Aspect | Feature Distances (Before/After) | Per-Nucleus Error |
|--------|----------------------------------|-------------------|
| **"Before" measurement** | ⚠️ RANSAC-filtered (circular reasoning) | N/A (only "after") |
| **Correspondence** | ❌ Different features detected | ✅ Same nuclei matched |
| **Ground truth** | ❌ No true landmarks | ⚠️ Biological structures |
| **Interpretability** | ❌ "Feature" abstract | ✅ "Nucleus" intuitive |
| **Spatial density** | ⚠️ Sparse (1000s features) | ✅ Dense (10,000s nuclei) |
| **Segmentation dependence** | ✅ None | ⚠️ Depends on StarDist quality |

**Key difference**: Feature methods show *improvement* (before vs after) but don't measure absolute error. Per-nucleus shows **absolute registration error** in biological terms.

---

## Best Practices for Interpretation

### 1. **Use in Combination with Other Metrics**

```
Registration Quality = f(Dice, Per-Nucleus Error, Visual Inspection)
```

**Example decision tree**:
```
IF Dice > 0.85 AND mean_nucleus_error < 3px AND q95 < 8px:
    → EXCELLENT registration
ELIF Dice > 0.75 AND mean_nucleus_error < 5px:
    → GOOD registration
ELIF matched_pairs / total_nuclei < 0.8:
    → FAILED registration (too many unmatched)
ELSE:
    → MARGINAL, requires visual QC
```

### 2. **Inspect the Distribution, Not Just the Mean**

```python
# Two scenarios with same mean but different quality:

Scenario A: Uniform errors
distances = [2.1, 2.3, 2.4, 2.2, 2.5, ...]
mean = 2.3px, std = 0.2px, q95 = 2.7px
→ Excellent! Consistent across image

Scenario B: Outliers present
distances = [1.0, 1.2, 1.1, 15.2, 1.3, 14.8, ...]
mean = 2.3px, std = 5.4px, q95 = 15.0px
→ Problem! Some regions failing badly
```

**Always check Q95 and histogram!**

### 3. **Consider Pixel Resolution**

Convert to physical units:
```python
pixel_size_um = 0.65  # µm/pixel at 10× magnification
mean_error_um = mean_centroid_distance * pixel_size_um
print(f"Mean registration error: {mean_error_um:.2f} µm")
```

**Biological context**:
- Typical nucleus diameter: 10-15 µm
- Cell-cell contact: 0-5 µm
- Acceptable error for spatial transcriptomics: <10 µm
- Acceptable error for connectomics: <0.5 µm

### 4. **Report in Methods Sections**

**Example**:
> "Registration quality was assessed using per-nucleus centroid matching (Hungarian algorithm). We computed the Euclidean distance between corresponding nucleus centroids in reference and registered images. The mean centroid distance was 2.3 ± 1.2 pixels (1.5 ± 0.8 µm), with 95% of matched nuclei within 5.4 pixels (3.5 µm). A total of 342/350 (97.7%) nuclei were successfully matched."

---

## Future Enhancements

### 1. **Add Confidence Intervals**

Bootstrap resampling to quantify uncertainty:
```python
mean_error = 2.3px
ci_95 = [2.1, 2.5]  # 95% confidence interval via bootstrap
```

### 2. **Spatial Error Maps**

Compute local error by image region:
```python
grid = divide_image_into_4x4_tiles()
for tile in grid:
    tile.local_error = mean(distances[nuclei_in_tile])

# Visualize as heatmap showing which regions fail
```

### 3. **Shape-Based Validation**

Beyond centroids, compare nucleus properties:
```python
for matched_pair in matches:
    area_ratio = reg_nucleus.area / ref_nucleus.area
    if area_ratio < 0.5 or area_ratio > 2.0:
        flag_as_suspicious()
```

### 4. **Temporal Tracking** (for time-series)

Extend to track nuclei across time:
```python
nucleus_trajectory = match_across_timepoints(t0, t1, t2, ...)
track_velocity = compute_velocity(trajectory)
```

---

## Summary

**Per-nucleus registration error** is a powerful, interpretable metric that:

✅ **Provides instance-level resolution** (not just global Dice)
✅ **Measures biologically meaningful error** (nucleus displacement)
✅ **Separates spatial error from segmentation error** (centroids more robust)
✅ **Gives rich statistics** (mean, median, percentiles, distribution)
✅ **Enables spatial analysis** (where does registration fail?)

⚠️ **But still has limitations**:
- Depends on segmentation quality (use StarDist!)
- Assumes nuclei correspondence exists
- Centroids don't capture rotation/deformation
- Needs careful threshold selection

**Use it alongside** Dice scores, visual inspection, and domain knowledge for comprehensive registration validation.
