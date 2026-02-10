# Feature-Based Registration Error Estimation

This document explains how the `estimate_feature_distances.py` script measures registration quality using feature-based Target Registration Error (TRE) metrics.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Step 1: Image Loading](#3-step-1-image-loading)
4. [Step 2: Feature Detection](#4-step-2-feature-detection)
5. [Step 3: Feature Matching](#5-step-3-feature-matching)
6. [Step 4: Distance Computation](#6-step-4-distance-computation)
7. [Step 5: Improvement Metrics](#7-step-5-improvement-metrics)
8. [Mathematical Foundations](#8-mathematical-foundations)
9. [Output Format](#9-output-format)
10. [Detector Comparison](#10-detector-comparison)

---

## 1. Overview

### Purpose

Registration quality is measured by comparing feature correspondences **before** and **after** registration:

```
BEFORE Registration:
  Reference Image ←→ Moving Image (unaligned)
  Features are spatially misaligned → Large distances

AFTER Registration:
  Reference Image ←→ Registered Image (aligned)
  Features should be aligned → Small distances
```

### Key Metric: Target Registration Error (TRE)

TRE measures the Euclidean distance between corresponding anatomical points after registration:

```
TRE = ||p_reference - p_registered||₂
```

Where:
- `p_reference` = keypoint position in reference image
- `p_registered` = matched keypoint position in registered image

A successful registration reduces the mean TRE significantly.

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ESTIMATE FEATURE DISTANCES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Reference   │  │   Moving     │  │  Registered  │                  │
│  │   Image      │  │   Image      │  │   Image      │                  │
│  │  (fixed)     │  │  (before)    │  │  (after)     │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         ▼                 ▼                 ▼                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              STEP 1: Load as Grayscale                       │      │
│  │              Extract channel 0 (DAPI), scale to uint8        │      │
│  └──────────────────────────────────────────────────────────────┘      │
│         │                 │                 │                           │
│         ▼                 ▼                 ▼                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              STEP 2: Detect Features                         │      │
│  │              SuperPoint/BRISK → keypoints + descriptors      │      │
│  └──────────────────────────────────────────────────────────────┘      │
│         │                 │                 │                           │
│         ▼                 ▼                 ▼                           │
│      ref_kp            mov_kp            reg_kp                        │
│      ref_desc          mov_desc          reg_desc                      │
│         │                 │                 │                           │
│         └────────┬────────┘                 │                           │
│                  │                          │                           │
│                  ▼                          │                           │
│  ┌────────────────────────────┐            │                           │
│  │  STEP 3A: Match BEFORE     │            │                           │
│  │  ref ←→ moving             │            │                           │
│  │  (pre-registration)        │            │                           │
│  └────────────────────────────┘            │                           │
│                  │                          │                           │
│                  ▼                          │                           │
│         matched_before                      │                           │
│                                             │                           │
│         ┌───────────────────────────────────┘                           │
│         │                                                               │
│         ▼                                                               │
│  ┌────────────────────────────┐                                        │
│  │  STEP 3B: Match AFTER      │                                        │
│  │  ref ←→ registered         │                                        │
│  │  (post-registration)       │                                        │
│  └────────────────────────────┘                                        │
│                  │                                                      │
│                  ▼                                                      │
│         matched_after                                                   │
│                  │                                                      │
│                  ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              STEP 4: Compute Distances                       │      │
│  │              Euclidean distance for each keypoint pair       │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                  │                                                      │
│                  ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              STEP 5: Compute Improvement                     │      │
│  │              Compare before vs after metrics                 │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                  │                                                      │
│                  ▼                                                      │
│         JSON metrics + histogram PNG                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: Image Loading

### Function: `load_image_grayscale()`

**Location:** `bin/utils/image_utils.py`

### Process

```python
def load_image_grayscale(path, max_dim=None):
    # 1. Load with pyvips (memory-efficient)
    img = pyvips.Image.new_from_file(path)

    # 2. Optional downsampling for large images
    if max_dim and max(img.width, img.height) > max_dim:
        scale = max_dim / max(img.width, img.height)
        img = img.resize(scale)

    # 3. Extract first channel (assumes DAPI at channel 0)
    if img.bands > 1:
        img = img.extract_band(0)

    # 4. Convert to uint8 with min-max scaling
    min_val, max_val = img.min(), img.max()
    img = ((img - min_val) / (max_val - min_val) * 255).cast("uchar")

    return img.numpy()
```

### Why Channel 0?

In multiplexed imaging, DAPI (nuclear stain) is typically stored as the first channel. Nuclear features provide:
- High contrast boundaries
- Consistent morphology across samples
- Dense, well-distributed keypoints

### Min-Max Normalization

```
I_normalized = (I - I_min) / (I_max - I_min) × 255
```

This ensures consistent intensity range [0, 255] regardless of original bit depth.

---

## 4. Step 2: Feature Detection

### SuperPoint Detector (Default)

**Location:** `valis_lib/feature_detectors.py`, `valis_lib/superglue_models/superpoint.py`

SuperPoint is a self-supervised deep learning model that simultaneously detects keypoints and computes descriptors.

### Architecture

```
Input Image (H × W)
       │
       ▼
┌─────────────────────────────────────────────┐
│           SHARED ENCODER                     │
│  Conv Block 1: 1→64 channels, /2 pool       │
│  Conv Block 2: 64→64 channels, /2 pool      │
│  Conv Block 3: 64→128 channels, /2 pool     │
│  Conv Block 4: 128→128 channels, no pool    │
└─────────────────────────────────────────────┘
       │
       ├────────────────────────────┐
       ▼                            ▼
┌──────────────────┐    ┌──────────────────┐
│ KEYPOINT HEAD    │    │ DESCRIPTOR HEAD  │
│ Conv 256→256     │    │ Conv 256→256     │
│ Conv 256→65      │    │ Conv 256→256     │
│ Softmax + NMS    │    │ L2 Normalize     │
└──────────────────┘    └──────────────────┘
       │                            │
       ▼                            ▼
   Keypoints (N×2)          Descriptors (N×256)
```

### Keypoint Detection

1. **Dense Score Map**: The network outputs a 65-channel tensor (64 keypoint classes + 1 "dustbin" for non-keypoints)

2. **Softmax Activation**:
   ```
   P(keypoint at cell c) = exp(s_c) / Σ exp(s_i)
   ```

3. **Non-Maximum Suppression (NMS)**:
   ```python
   def simple_nms(scores, radius=4):
       # Apply max pooling with kernel size (2r+1)
       max_scores = max_pool2d(scores, kernel=2*radius+1, padding=radius)
       # Keep only local maxima
       return scores * (scores == max_scores)
   ```

4. **Thresholding**: Keep keypoints where `score > keypoint_threshold` (default: 0.005)

### Descriptor Extraction

1. **Dense Descriptor Map**: 256-dimensional descriptor at each pixel

2. **L2 Normalization**:
   ```
   d_normalized = d / ||d||₂
   ```

3. **Bilinear Interpolation**: Sample descriptors at exact keypoint locations
   ```python
   # Normalize coordinates to [-1, 1]
   grid = (keypoints / image_size) * 2 - 1
   # Bilinear sample
   descriptors = grid_sample(dense_descriptors, grid)
   ```

### Alternative: BRISK Detector

For CPU-only environments or when deep learning isn't available:

```
BRISK (Binary Robust Invariant Scalable Keypoints)
├── Multi-scale AGAST corner detection
├── Scale-space keypoint localization
└── Binary descriptor (512 bits)
    └── Pairwise brightness comparisons in sampling pattern
```

---

## 5. Step 3: Feature Matching

### SuperGlue Matcher (Default with SuperPoint)

**Location:** `valis_lib/superglue_models/superglue.py`

SuperGlue uses Graph Neural Networks with attention to learn optimal feature matching.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SUPERGLUE MATCHER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image 1 Keypoints        Image 2 Keypoints                    │
│  (x₁, y₁, score₁)         (x₂, y₂, score₂)                     │
│         │                        │                              │
│         ▼                        ▼                              │
│  ┌──────────────────────────────────────────┐                  │
│  │         KEYPOINT ENCODER                 │                  │
│  │  MLP: [3] → [32, 64, 128, 256]          │                  │
│  │  Encodes position + confidence           │                  │
│  └──────────────────────────────────────────┘                  │
│         │                        │                              │
│         ▼                        ▼                              │
│  ┌──────────────────────────────────────────┐                  │
│  │      ATTENTIONAL GRAPH NEURAL NETWORK    │                  │
│  │                                          │                  │
│  │  For each layer (default: 9 layers):     │                  │
│  │    ├── Self-Attention (within image)     │                  │
│  │    │   Refine features using context     │                  │
│  │    │                                     │                  │
│  │    └── Cross-Attention (between images)  │                  │
│  │        Exchange information across images│                  │
│  │                                          │                  │
│  └──────────────────────────────────────────┘                  │
│         │                        │                              │
│         ▼                        ▼                              │
│      Refined             Refined                                │
│      Features₁           Features₂                              │
│         │                        │                              │
│         └────────┬───────────────┘                              │
│                  ▼                                              │
│  ┌──────────────────────────────────────────┐                  │
│  │         OPTIMAL TRANSPORT                │                  │
│  │                                          │                  │
│  │  Score matrix: S[i,j] = f₁[i]ᵀ · f₂[j]  │                  │
│  │                                          │                  │
│  │  Add dustbin for unmatched points        │                  │
│  │                                          │                  │
│  │  Sinkhorn iterations (100×):             │                  │
│  │    Alternating row/column normalization  │                  │
│  │    Converges to optimal assignment       │                  │
│  │                                          │                  │
│  └──────────────────────────────────────────┘                  │
│                  │                                              │
│                  ▼                                              │
│         Match Indices + Confidence                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Head Attention

For each attention layer:

```
Query:   Q = W_Q · features
Key:     K = W_K · features
Value:   V = W_V · features

Attention Scores:
  A = softmax(Q · Kᵀ / √d)

Output:
  out = A · V
```

Where `d` is the feature dimension (prevents gradient explosion).

### Optimal Transport (Sinkhorn Algorithm)

The matching problem is formulated as optimal transport:

```
Given:
  - Score matrix S ∈ ℝ^(N×M) where S[i,j] = similarity(feat₁[i], feat₂[j])
  - Augmented with dustbin row/column for unmatched points

Find:
  - Assignment matrix P minimizing transport cost

Sinkhorn Iteration (in log-space for stability):
  for iter in range(100):
      # Row normalization
      u = log(a) - logsumexp(S + v, dim=1)
      # Column normalization
      v = log(b) - logsumexp(S + u, dim=0)

  P = exp(S + u + v)
```

### Alternative: Brute-Force Matcher with RANSAC

For non-deep-learning detectors:

```python
# 1. Compute all pairwise distances
distances = cdist(desc1, desc2, metric='euclidean')

# 2. Find nearest neighbors
matches = argmin(distances, axis=1)

# 3. Apply Lowe's ratio test
ratio = dist_1st / dist_2nd
keep = ratio < 0.8

# 4. RANSAC filtering
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold=7.0)
inliers = matches[mask.ravel() == 1]
```

---

## 6. Step 4: Distance Computation

### Function: `compute_feature_distances()`

**Location:** `bin/estimate_feature_distances.py`

### Euclidean Distance

For each matched keypoint pair:

```
distance[i] = ||p_ref[i] - p_mov[i]||₂
            = √[(x_ref - x_mov)² + (y_ref - y_mov)²]
```

### Implementation

```python
def compute_feature_distances(ref_kp, mov_kp):
    """
    Parameters:
        ref_kp: (N, 2) array - matched keypoints in reference
        mov_kp: (N, 2) array - matched keypoints in moving/registered

    Returns:
        distances: (N,) array - per-keypoint distances in pixels
        statistics: dict - summary statistics
    """
    # Euclidean distance for each pair
    distances = np.linalg.norm(ref_kp - mov_kp, axis=1)

    # Compute statistics
    stats = {
        "mean": np.mean(distances),
        "median": np.median(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
        "q25": np.percentile(distances, 25),
        "q75": np.percentile(distances, 75),
        "q90": np.percentile(distances, 90),
        "q95": np.percentile(distances, 95),
        "q99": np.percentile(distances, 99),
        "n_points": len(distances)
    }

    return distances, stats
```

### Interpreting Distance Statistics

| Statistic | Meaning |
|-----------|---------|
| **mean** | Average misalignment in pixels |
| **median** | Typical misalignment (robust to outliers) |
| **std** | Consistency of alignment |
| **q95/q99** | Worst-case alignment (identifies problem areas) |

---

## 7. Step 5: Improvement Metrics

### Computing Improvement

```python
improvement = {
    # Absolute improvement (pixels)
    "distance_reduction_pixels": mean_before - mean_after,

    # Relative improvement (percentage)
    "distance_reduction_percent": (mean_before - mean_after) / mean_before * 100,

    # Match count change (more matches = better features visible)
    "match_count_increase": n_matches_after - n_matches_before,

    # Descriptor similarity improvement
    "descriptor_distance_decrease": desc_dist_before - desc_dist_after
}
```

### Interpretation

| Metric | Good Value | Meaning |
|--------|------------|---------|
| `distance_reduction_percent` | > 50% | Registration significantly reduced misalignment |
| `distance_reduction_pixels` | Depends on resolution | Absolute improvement in pixels |
| `match_count_increase` | > 0 | More features match after registration |
| Mean TRE (after) | < 5 pixels | Excellent alignment |

---

## 8. Mathematical Foundations

### 8.1 Target Registration Error (TRE)

The gold standard for registration accuracy:

```
TRE = (1/N) Σᵢ ||T(pᵢ) - qᵢ||₂
```

Where:
- `T` = transformation function
- `pᵢ` = point in moving image
- `qᵢ` = corresponding point in reference image
- `N` = number of corresponding points

### 8.2 Feature-Based TRE Approximation

Since true correspondences are unknown, we use feature matching as a proxy:

```
TRE_approx ≈ (1/M) Σⱼ ||matched_ref[j] - matched_mov[j]||₂
```

Where `M` = number of successfully matched features.

### 8.3 Descriptor Distance

Measures feature similarity (lower = more similar):

```
For L2-normalized descriptors:
  distance = ||d₁ - d₂||₂ = √(2 - 2·d₁ᵀd₂)

Equivalently:
  cosine_similarity = d₁ᵀd₂
  distance = √(2 - 2·cos_sim)
```

### 8.4 Lowe's Ratio Test

Filters ambiguous matches:

```
ratio = distance_to_best_match / distance_to_second_best_match

if ratio < threshold (typically 0.8):
    keep match  # Distinctive match
else:
    reject match  # Ambiguous
```

### 8.5 RANSAC Homography

Robust estimation of geometric transformation:

```
1. Sample 4 random point correspondences
2. Compute homography H from these 4 points
3. Count inliers: points where ||Hp - q|| < threshold
4. Repeat and keep H with most inliers
5. Refine H using all inliers
```

---

## 9. Output Format

### JSON Output (`*_feature_distances.json`)

```json
{
  "reference_image": "patient1_reference.tif",
  "moving_image": "patient1_moving.tif",
  "registered_image": "patient1_registered.tif",
  "detector_type": "superpoint",
  "max_dim": 2048,
  "n_features": 5000,

  "before_registration": {
    "n_keypoints": 4523,
    "n_matches": 1847,
    "match_ratio": 0.408,
    "mean_descriptor_distance": 0.342,
    "feature_distances": {
      "mean": 127.34,
      "median": 98.21,
      "std": 89.45,
      "min": 2.14,
      "max": 456.78,
      "q25": 45.67,
      "q75": 178.90,
      "q90": 267.34,
      "q95": 312.45,
      "q99": 398.67,
      "n_points": 1847
    }
  },

  "after_registration": {
    "n_keypoints": 4612,
    "n_matches": 2134,
    "match_ratio": 0.463,
    "mean_descriptor_distance": 0.289,
    "feature_distances": {
      "mean": 4.56,
      "median": 3.21,
      "std": 3.89,
      "min": 0.12,
      "max": 28.45,
      "q25": 1.89,
      "q75": 5.67,
      "q90": 9.34,
      "q95": 12.45,
      "q99": 21.23,
      "n_points": 2134
    }
  },

  "improvement": {
    "distance_reduction_pixels": 122.78,
    "distance_reduction_percent": 96.42,
    "match_count_increase": 287,
    "descriptor_distance_decrease": 0.053
  }
}
```

### Histogram Output (`*_distance_histogram.png`)

```
┌────────────────────────────────────────────────────────────────┐
│         Feature Distance Distribution - sample1                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Before Registration          After Registration               │
│  ┌─────────────────┐         ┌─────────────────┐              │
│  │    ▄▄▄▄         │         │▄▄▄▄▄            │              │
│  │   ██████▄       │         │██████           │              │
│  │  ████████▄▄     │         │████████▄        │              │
│  │ ██████████████▄ │         │██████████▄▄     │              │
│  │████████████████ │         │████████████     │              │
│  └─────────────────┘         └─────────────────┘              │
│    0   100  200  300           0   5   10   15                │
│                                                                │
│  Mean: 127.34px               Mean: 4.56px                    │
│  Median: 98.21px              Median: 3.21px                  │
│                               Improvement: 96.4%               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 10. Detector Comparison

| Detector | Type | Descriptors | Speed | Accuracy | Best For |
|----------|------|-------------|-------|----------|----------|
| **SuperPoint** | Deep Learning | 256-dim float | Slow (GPU) | Excellent | High-quality registration |
| **BRISK** | Classical | 512-bit binary | Fast | Good | CPU-only, real-time |
| **ORB** | Classical | 256-bit binary | Very Fast | Moderate | Speed-critical |
| **DISK** | Deep Learning | 128-dim float | Medium | Excellent | Modern alternative |
| **DeDoDe** | Deep Learning | Variable | Medium | Excellent | Dense matching |

### Matcher Compatibility

| Detector | Recommended Matcher | Alternative |
|----------|---------------------|-------------|
| SuperPoint | SuperGlue | Brute-force + RANSAC |
| DISK | LightGlue | Brute-force + RANSAC |
| DeDoDe | LightGlue | Brute-force + RANSAC |
| BRISK/ORB | Brute-force + RANSAC/GMS | N/A |

---

## Usage Example

```bash
python bin/estimate_feature_distances.py \
    --reference /path/to/reference.tif \
    --moving /path/to/moving.tif \
    --registered /path/to/registered.tif \
    --output-prefix sample1 \
    --detector superpoint \
    --max-dim 2048 \
    --n-features 5000
```

### Expected Output

```
======================================================================
FEATURE DISTANCE ESTIMATION (Before vs After Registration)
======================================================================

[1/5] Initializing feature detector: superpoint

[2/5] Processing reference image...
  Reference: reference.tif
  Extracting DAPI channel (channel 0) for feature detection...
  Detecting reference features...
    Detected 4523 keypoints

[3/5] Processing moving image (BEFORE registration)...
  Moving: moving.tif
  Extracting DAPI channel (channel 0) for feature detection...
  Detecting features in moving image...
    Detected 4234 keypoints
  Matching features (BEFORE registration)...
    Matches: 1847
    Mean distance: 127.34 pixels

[4/5] Processing registered image (AFTER registration)...
  Registered: registered.tif
  Extracting DAPI channel (channel 0) for feature detection...
  Detecting features in registered image...
    Detected 4612 keypoints
  Matching features (AFTER registration)...
    Matches: 2134
    Mean TRE: 4.56 pixels

[5/5] Computing improvement metrics...
  Distance reduction: 122.78 pixels (96.4%)
  Match count change: +287

  Saved histogram: sample1_distance_histogram.png
  Saved results: sample1_feature_distances.json
    Mean TRE: 4.56 pixels
    Improvement: 96.4%

======================================================================
✓ Feature distance estimation complete!
======================================================================
```

---

## References

1. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-Supervised Interest Point Detection and Description. CVPR Workshop.

2. Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. CVPR.

3. Leutenegger, S., Chli, M., & Siegwart, R. (2011). BRISK: Binary Robust Invariant Scalable Keypoints. ICCV.

4. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. IJCV.

5. Fitzpatrick, J. M. (2001). The role of registration in accurate surgical guidance. JMIV.
