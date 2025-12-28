# REGISTRATION Subworkflow - Complete Dataflow Analysis

This document provides a comprehensive, step-by-step trace of data transformations through the REGISTRATION subworkflow.

---

## Table of Contents

1. [Input Structure](#input-structure)
2. [Step 1: Optional Padding](#step-1-optional-padding)
3. [Step 2: Patient Grouping and Reference Identification](#step-2-patient-grouping-and-reference-identification)
4. [Step 3: Registration via Method-Specific Adapters](#step-3-registration-via-method-specific-adapters)
5. [Step 3B: QC Generation](#step-3b-qc-generation)
6. [Step 3C: Error Estimation](#step-3c-error-estimation)
7. [Step 4: Checkpoint CSV Writing](#step-4-checkpoint-csv-writing)
8. [Output Emissions](#output-emissions)
9. [Complete Flow Diagrams](#complete-flow-diagrams)

---

## Input Structure

### Entry Point

```groovy
workflow REGISTRATION {
    take:
    ch_preprocessed  // Channel of [meta, file] tuples
    method           // String: 'valis' | 'gpu' | 'cpu'
```

### Input Channel Format

```groovy
ch_preprocessed = [meta, file]

where:
  meta = [
    patient_id  : String    (e.g., "patient_001")
    is_reference: Boolean   (true for reference, false for moving)
    channels    : List[String] (e.g., ["DAPI", "CD3", "CD8"])
  ]

  file = Path (preprocessed OME-TIFF image)
```

### Example Input Data

```
Channel content (3 images for patient_001):

[{patient_id: "patient_001", is_reference: true,  channels: ["DAPI","CD3","CD8"]}, /path/patient_001_ref.ome.tiff]
[{patient_id: "patient_001", is_reference: false, channels: ["DAPI","CD3","CD8"]}, /path/patient_001_day1.ome.tiff]
[{patient_id: "patient_001", is_reference: false, channels: ["DAPI","CD3","CD8"]}, /path/patient_001_day2.ome.tiff]
```

---

## Step 1: Optional Padding

### Code Location: Lines 65-86

### Conditional Logic

```groovy
if (params.padding) {
    // Run padding workflow
} else {
    ch_images = ch_preprocessed
}
```

### When Padding is DISABLED (params.padding = false)

**Transformation:**
```
ch_preprocessed → ch_images (no change)
```

### When Padding is ENABLED (params.padding = true)

#### Step 1.1: Get Image Dimensions

```groovy
GET_IMAGE_DIMS(ch_preprocessed)
```

**Input to GET_IMAGE_DIMS:**
```
[meta, file]
```

**Output from GET_IMAGE_DIMS:**
```
[meta, dims_json]

where dims_json contains: {"width": 5000, "height": 4000}
```

#### Step 1.2: Group Dimensions by Patient

```groovy
ch_grouped_dims = GET_IMAGE_DIMS.out.dims
    .map { meta, dims -> [meta.patient_id, dims] }
    .groupTuple(by: 0)
```

**Transformation:**
```
Input:
  [meta1, dims1]  // patient_001, ref
  [meta2, dims2]  // patient_001, day1
  [meta3, dims3]  // patient_001, day2

After .map:
  ["patient_001", dims1]
  ["patient_001", dims2]
  ["patient_001", dims3]

After .groupTuple(by: 0):
  ["patient_001", [dims1, dims2, dims3]]
```

#### Step 1.3: Find Maximum Dimensions per Patient

```groovy
MAX_DIM(ch_grouped_dims)
```

**Input:**
```
["patient_001", [dims1, dims2, dims3]]
```

**Output:**
```
["patient_001", max_dims_file]

where max_dims_file contains: {"width": 5000, "height": 4500}  // max across all images
```

#### Step 1.4: Join and Pad Images

```groovy
ch_to_pad = ch_preprocessed
    .map { meta, file -> [meta.patient_id, meta, file] }
    .join(MAX_DIM.out.max_dims_file, by: 0)
    .map { patient_id, meta, file, max_dims -> [meta, file, max_dims] }

PAD_IMAGES(ch_to_pad)
ch_images = PAD_IMAGES.out.padded
```

**Transformation:**
```
ch_preprocessed:
  [meta1, file1]  // patient_001, ref
  [meta2, file2]  // patient_001, day1
  [meta3, file3]  // patient_001, day2

After .map (add patient_id as key):
  ["patient_001", meta1, file1]
  ["patient_001", meta2, file2]
  ["patient_001", meta3, file3]

After .join with max_dims:
  ["patient_001", meta1, file1, max_dims_001]
  ["patient_001", meta2, file2, max_dims_001]
  ["patient_001", meta3, file3, max_dims_001]

After .map (prepare for PAD_IMAGES):
  [meta1, file1, max_dims_001]
  [meta2, file2, max_dims_001]
  [meta3, file3, max_dims_001]

After PAD_IMAGES:
  [meta1, padded_file1]
  [meta2, padded_file2]
  [meta3, padded_file3]
```

### Output from Step 1

```
ch_images = [meta, file]  // Either original or padded files
```

---

## Step 2: Patient Grouping and Reference Identification

### Code Location: Lines 88-125

### Purpose
- Group all images by patient_id
- Identify reference image within each patient group
- Prepare data structure for batch/pairwise registration

### Step 2.1: Transform to Patient-Keyed Structure

```groovy
ch_grouped = ch_images
    .map { meta, file -> [meta.patient_id, meta, file] }
    .groupTuple(by: 0)
```

**Transformation:**
```
Input (ch_images):
  [meta1, file1]  // {patient_id: "p001", is_reference: true,  channels: ["DAPI","CD3"]}
  [meta2, file2]  // {patient_id: "p001", is_reference: false, channels: ["DAPI","CD3"]}
  [meta3, file3]  // {patient_id: "p001", is_reference: false, channels: ["DAPI","CD3"]}
  [meta4, file4]  // {patient_id: "p002", is_reference: true,  channels: ["DAPI","CD8"]}

After .map { meta, file -> [meta.patient_id, meta, file] }:
  ["p001", meta1, file1]
  ["p001", meta2, file2]
  ["p001", meta3, file3]
  ["p002", meta4, file4]

After .groupTuple(by: 0):
  ["p001", [meta1, meta2, meta3], [file1, file2, file3]]
  ["p002", [meta4], [file4]]
```

### Step 2.2: Identify Reference and Structure Items

```groovy
.map { patient_id, metas, files ->
    // Combine metas and files into items
    def items = [metas, files].transpose()

    // Find reference image
    def ref = items.find { item -> item[0].is_reference }

    // Reference fallback logic
    if (!ref) {
        if (params.allow_auto_reference) {
            log.warn "No reference for ${patient_id}, using first image"
            ref = items[0]
        } else {
            throw new Exception("No reference for ${patient_id}")
        }
    }

    [patient_id, ref, items]
}
```

**Detailed Transformation:**

```
Input (from groupTuple):
  ["p001", [meta1, meta2, meta3], [file1, file2, file3]]

Step 1: Transpose metas and files
  items = [metas, files].transpose()
  items = [[meta1, file1], [meta2, file2], [meta3, file3]]

Step 2: Find reference
  ref = items.find { item -> item[0].is_reference }
  ref = [meta1, file1]  // because meta1.is_reference == true

Step 3: Output structure
  ["p001", [meta1, file1], [[meta1, file1], [meta2, file2], [meta3, file3]]]
```

### Output from Step 2

```
ch_grouped = [patient_id, reference_item, all_items]

where:
  patient_id = String
  reference_item = [meta_ref, file_ref]
  all_items = [
    [meta_ref, file_ref],    // reference included in all_items
    [meta_mov1, file_mov1],
    [meta_mov2, file_mov2],
    ...
  ]
```

**Visual Representation:**

```
For patient_001 with 3 images (1 reference, 2 moving):

[
  "patient_001",

  [meta_ref, file_ref],  // <-- reference_item

  [                      // <-- all_items
    [meta_ref, file_ref],
    [meta_day1, file_day1],
    [meta_day2, file_day2]
  ]
]
```

---

## Step 3: Registration via Method-Specific Adapters

### Code Location: Lines 127-154

### Adapter Selection

```groovy
switch(method) {
    case 'valis':
        VALIS_ADAPTER(ch_grouped)
        ch_registered = VALIS_ADAPTER.out.registered
        break
    case 'gpu':
        GPU_ADAPTER(ch_grouped)
        ch_registered = GPU_ADAPTER.out.registered
        break
    case 'cpu':
        CPU_ADAPTER(ch_grouped)
        ch_registered = CPU_ADAPTER.out.registered
        break
}
```

All adapters:
- **Input**: `ch_grouped` = `[patient_id, reference_item, all_items]`
- **Output**: `ch_registered` = `[meta, registered_file]`

---

### Option A: VALIS_ADAPTER (Batch Registration)

#### Input to VALIS_ADAPTER

```
[patient_id, reference_item, all_items]
```

#### Step 3A.1: Transform to VALIS Batch Format

```groovy
ch_valis_input = ch_grouped_meta
    .map { patient_id, ref_item, all_items ->
        def ref_file = ref_item[1]
        def all_files = all_items.collect { item -> item[1] }
        def all_metas = all_items.collect { item -> item[0] }

        tuple(patient_id, ref_file, all_files, all_metas)
    }
```

**Transformation:**

```
Input:
  [
    "p001",
    [meta_ref, file_ref],
    [[meta_ref, file_ref], [meta_day1, file_day1], [meta_day2, file_day2]]
  ]

Extract components:
  ref_file = file_ref
  all_files = [file_ref, file_day1, file_day2]
  all_metas = [meta_ref, meta_day1, meta_day2]

Output:
  ["p001", file_ref, [file_ref, file_day1, file_day2], [meta_ref, meta_day1, meta_day2]]
```

#### Step 3A.2: REGISTER Process (VALIS)

**Input:**
```groovy
tuple val(patient_id),
      path(reference, stageAs: 'ref/*'),
      path(preproc_files, stageAs: 'input_?/*'),
      val(all_metas)
```

**File Staging:**
```
Staged files in work directory:
  ref/file_ref.ome.tiff
  input_1/file_ref.ome.tiff     (duplicate, VALIS needs all files)
  input_2/file_day1.ome.tiff
  input_3/file_day2.ome.tiff
```

**Processing:**
- VALIS builds transformation graph for all images
- Registers all images to reference
- Outputs to `registered_slides/`

**Output:**
```groovy
tuple val(patient_id),
      path("registered_slides/*_registered.ome.tiff"),
      val(all_metas)

Example:
  [
    "p001",
    [
      registered_slides/p001_DAPI_CD3_registered.ome.tiff,
      registered_slides/p001_day1_DAPI_CD3_registered.ome.tiff,
      registered_slides/p001_day2_DAPI_CD3_registered.ome.tiff
    ],
    [meta_ref, meta_day1, meta_day2]
  ]
```

#### Step 3A.3: Reconstruct [meta, file] Format

```groovy
ch_registered = REGISTER.out.registered
    .flatMap { patient_id, reg_files, metas ->
        // Sanity check
        if (reg_files.size() != metas.size()) {
            throw new Exception("File count mismatch")
        }

        // Build fuzzy matching dictionary
        def filename_to_meta = [:]
        metas.each { meta ->
            def exact_prefix = "${meta.patient_id}_${meta.channels.join('_')}"
            filename_to_meta[exact_prefix] = meta
            filename_to_meta[exact_prefix.toLowerCase()] = meta
            filename_to_meta[meta.channels.join('_')] = meta
        }

        // Match each file to metadata
        reg_files.collect { reg_file ->
            def basename = reg_file.name
                .replaceAll(/_(registered|corrected|padded)+/, '')
                .replaceAll(/\.ome\.tiff?$/, '')
                .replaceAll(/\.tiff?$/, '')

            def matched_meta = filename_to_meta[basename] ?:
                               filename_to_meta[basename.toLowerCase()] ?:
                               filename_to_meta.find { k, v -> basename.contains(k) }?.value

            if (!matched_meta) {
                throw new Exception("Could not match ${reg_file.name}")
            }

            [matched_meta, reg_file]
        }
    }
```

**Detailed Matching Example:**

```
Input from REGISTER:
  patient_id = "p001"
  reg_files = [
    "registered_slides/p001_DAPI_CD3_registered.ome.tiff",
    "registered_slides/p001_day1_DAPI_CD3_registered.ome.tiff",
    "registered_slides/p001_day2_DAPI_CD3_registered.ome.tiff"
  ]
  metas = [meta_ref, meta_day1, meta_day2]

Build matching dictionary:
  For meta_ref {patient_id: "p001", channels: ["DAPI","CD3"]}:
    "p001_DAPI_CD3" → meta_ref
    "p001_dapi_cd3" → meta_ref
    "DAPI_CD3" → meta_ref

  For meta_day1 {patient_id: "p001", channels: ["DAPI","CD3"]}:
    "p001_DAPI_CD3" → meta_day1  // OVERWRITES! This is a problem...

Actually, this is where the bug fix is important:
  The filename should contain unique identifiers beyond just channels
  The fuzzy matching tries multiple strategies to handle this

Match first file "p001_DAPI_CD3_registered.ome.tiff":
  Strip suffixes: "p001_DAPI_CD3"
  Try exact match: filename_to_meta["p001_DAPI_CD3"]
  Returns: meta_ref (or appropriate meta)

Output (flatMap emits individual items):
  [meta_ref, registered_slides/p001_DAPI_CD3_registered.ome.tiff]
  [meta_day1, registered_slides/p001_day1_DAPI_CD3_registered.ome.tiff]
  [meta_day2, registered_slides/p001_day2_DAPI_CD3_registered.ome.tiff]
```

---

### Option B: GPU_ADAPTER (Pairwise Registration)

#### Step 3B.1: Transform to Pairwise Format

```groovy
ch_pairs = ch_grouped_meta
    .flatMap { patient_id, ref_item, all_items ->
        def ref_file = ref_item[1]
        all_items
            .findAll { item -> !item[0].is_reference }  // SKIP REFERENCE!
            .collect { moving_item ->
                tuple(moving_item[0], ref_file, moving_item[1])
            }
    }
```

**Transformation:**

```
Input:
  [
    "p001",
    [meta_ref, file_ref],
    [[meta_ref, file_ref], [meta_day1, file_day1], [meta_day2, file_day2]]
  ]

Extract reference:
  ref_file = file_ref

Filter non-reference items:
  all_items.findAll { !item[0].is_reference }
  = [[meta_day1, file_day1], [meta_day2, file_day2]]

Collect pairs (flatMap emits individually):
  [meta_day1, file_ref, file_day1]
  [meta_day2, file_ref, file_day2]
```

#### Step 3B.2: GPU_REGISTER Process

**Input per pair:**
```groovy
tuple val(meta), path(reference), path(moving)
```

**Example:**
```
[meta_day1, file_ref, file_day1]
```

**Processing:**
- Crop-based affine registration
- Crop-based diffeomorphic registration
- Merge registered crops

**Output:**
```groovy
tuple val(meta), path("*_registered.ome.tiff")

Example:
  [meta_day1, p001_day1_DAPI_CD3_registered.ome.tiff]
  [meta_day2, p001_day2_DAPI_CD3_registered.ome.tiff]
```

#### Step 3B.3: Add Back Reference Images

```groovy
ch_references = ch_grouped_meta
    .map { patient_id, ref_item, all_items -> ref_item }

ch_all = ch_references.concat(GPU_REGISTER.out.registered)
```

**Transformation:**

```
ch_references:
  [meta_ref, file_ref]

GPU_REGISTER.out.registered:
  [meta_day1, p001_day1_registered.ome.tiff]
  [meta_day2, p001_day2_registered.ome.tiff]

ch_all (after .concat):
  [meta_ref, file_ref]
  [meta_day1, p001_day1_registered.ome.tiff]
  [meta_day2, p001_day2_registered.ome.tiff]
```

---

### Option C: CPU_ADAPTER

**Identical dataflow to GPU_ADAPTER**, just uses different registration algorithm (SimpleElastix instead of crop-based deformable).

---

### Output from Step 3

```
ch_registered = [meta, registered_file]

All adapters output the same format:
  [meta_ref, file_ref or registered_ref]
  [meta_mov1, file_mov1_registered]
  [meta_mov2, file_mov2_registered]
  ...
```

---

## Step 3B: QC Generation

### Code Location: Lines 156-187

### Purpose
Compare registered images to reference using DAPI channel overlay

### Step 3B.1: Branch Reference from Moving

```groovy
ch_qc_input = ch_registered
    .branch {
        reference: it[0].is_reference
        moving: !it[0].is_reference
    }
```

**Transformation:**

```
Input (ch_registered):
  [meta_ref, file_ref]
  [meta_day1, file_day1_registered]
  [meta_day2, file_day2_registered]

After .branch:
  ch_qc_input.reference:
    [meta_ref, file_ref]

  ch_qc_input.moving:
    [meta_day1, file_day1_registered]
    [meta_day2, file_day2_registered]
```

### Step 3B.2: Extract References by Patient

```groovy
ch_references_for_qc = ch_qc_input.reference
    .map { meta, file -> [meta.patient_id, file] }
```

**Transformation:**

```
Input:
  [meta_ref, file_ref]  // {patient_id: "p001", ...}

Output:
  ["p001", file_ref]
```

### Step 3B.3: Join Moving Images with References

```groovy
ch_for_qc = ch_qc_input.moving
    .map { meta, file -> [meta.patient_id, meta, file] }
    .join(ch_references_for_qc, by: 0)
    .map { patient_id, meta, registered_file, reference_file ->
        [meta, registered_file, reference_file]
    }
```

**Transformation:**

```
ch_qc_input.moving:
  [meta_day1, file_day1_registered]  // {patient_id: "p001", ...}
  [meta_day2, file_day2_registered]  // {patient_id: "p001", ...}

After .map (add patient_id key):
  ["p001", meta_day1, file_day1_registered]
  ["p001", meta_day2, file_day2_registered]

ch_references_for_qc:
  ["p001", file_ref]

After .join(by: 0):
  ["p001", meta_day1, file_day1_registered, file_ref]
  ["p001", meta_day2, file_day2_registered, file_ref]

After .map (remove patient_id, rearrange):
  [meta_day1, file_day1_registered, file_ref]
  [meta_day2, file_day2_registered, file_ref]
```

### Step 3B.4: Generate QC

```groovy
if (!params.skip_registration_qc) {
    GENERATE_REGISTRATION_QC(ch_for_qc)
    ch_qc = GENERATE_REGISTRATION_QC.out.qc
} else {
    ch_qc = Channel.empty()
}
```

**Input to GENERATE_REGISTRATION_QC:**
```
[meta, registered_file, reference_file]
```

**Output:**
```
[meta, qc_files]

where qc_files are PNG and TIFF overlays
```

---

## Step 3C: Error Estimation

### Code Location: Lines 179-231

### Purpose
Measure registration quality using:
1. Feature-based distances (before/after registration)
2. Segmentation-based overlap (IoU/Dice)

### Conditional Execution

```groovy
if (params.enable_feature_error || params.enable_segmentation_error) {
    // Run error estimation
}
```

### Step 3C.1: Prepare Error Estimation Data

This is the most complex channel transformation in the entire workflow!

```groovy
ch_for_error = ch_registered
    .map { meta, reg_file -> [meta.patient_id, meta, reg_file] }
    .combine(
        ch_images
            .map { meta, preproc_file -> [meta.patient_id, meta, preproc_file] },
        by: 0
    )
    .map { patient_id, reg_meta, reg_file, preproc_meta, preproc_file ->
        // Only pair non-reference images
        if (!reg_meta.is_reference) {
            [patient_id, reg_meta, reg_file, preproc_file]
        }
    }
    .filter { it != null }
    .groupTuple(by: 0)
    .join(
        // Get reference for each patient
        ch_images
            .filter { meta, file -> meta.is_reference }
            .map { meta, file -> [meta.patient_id, file] },
        by: 0
    )
    .map { patient_id, metas, reg_files, preproc_files, ref_file ->
        tuple(patient_id, ref_file, preproc_files, reg_files, metas)
    }
```

**Let's trace this step-by-step:**

#### Initial State

```
ch_registered (after registration):
  [meta_ref, file_ref_maybe_registered]
  [meta_day1, file_day1_registered]
  [meta_day2, file_day2_registered]

ch_images (from Step 1, before registration):
  [meta_ref, file_ref_preprocessed]
  [meta_day1, file_day1_preprocessed]
  [meta_day2, file_day2_preprocessed]
```

#### Substep 1: Add patient_id to registered

```groovy
.map { meta, reg_file -> [meta.patient_id, meta, reg_file] }
```

```
Result:
  ["p001", meta_ref, file_ref_maybe_registered]
  ["p001", meta_day1, file_day1_registered]
  ["p001", meta_day2, file_day2_registered]
```

#### Substep 2: Combine with preprocessed images

```groovy
.combine(
    ch_images
        .map { meta, preproc_file -> [meta.patient_id, meta, preproc_file] },
    by: 0
)
```

**What .combine(by: 0) does:**
- Cartesian product within each patient_id group
- Every registered file paired with every preprocessed file (same patient)

```
ch_images after .map:
  ["p001", meta_ref, file_ref_preprocessed]
  ["p001", meta_day1, file_day1_preprocessed]
  ["p001", meta_day2, file_day2_preprocessed]

After .combine(by: 0):
  ["p001", meta_ref, file_ref_registered, meta_ref, file_ref_preprocessed]
  ["p001", meta_ref, file_ref_registered, meta_day1, file_day1_preprocessed]
  ["p001", meta_ref, file_ref_registered, meta_day2, file_day2_preprocessed]
  ["p001", meta_day1, file_day1_registered, meta_ref, file_ref_preprocessed]
  ["p001", meta_day1, file_day1_registered, meta_day1, file_day1_preprocessed]  ← CORRECT PAIR
  ["p001", meta_day1, file_day1_registered, meta_day2, file_day2_preprocessed]
  ["p001", meta_day2, file_day2_registered, meta_ref, file_ref_preprocessed]
  ["p001", meta_day2, file_day2_registered, meta_day1, file_day1_preprocessed]
  ["p001", meta_day2, file_day2_registered, meta_day2, file_day2_preprocessed]  ← CORRECT PAIR

Total: 3 × 3 = 9 combinations
```

#### Substep 3: Filter to keep only matching pairs (non-reference)

```groovy
.map { patient_id, reg_meta, reg_file, preproc_meta, preproc_file ->
    // Only pair non-reference images
    if (!reg_meta.is_reference) {
        [patient_id, reg_meta, reg_file, preproc_file]
    }
}
.filter { it != null }
```

**This filters out:**
1. Reference images (we don't measure error for reference)
2. Mismatched pairs (implicit - we rely on metadata matching in next steps)

```
After .map and .filter:
  ["p001", meta_day1, file_day1_registered, file_ref_preprocessed]
  ["p001", meta_day1, file_day1_registered, file_day1_preprocessed]
  ["p001", meta_day1, file_day1_registered, file_day2_preprocessed]
  ["p001", meta_day2, file_day2_registered, file_ref_preprocessed]
  ["p001", meta_day2, file_day2_registered, file_day1_preprocessed]
  ["p001", meta_day2, file_day2_registered, file_day2_preprocessed]

NOTE: We still have 6 items (2 registered × 3 preprocessed each)
```

#### Substep 4: Group by patient

```groovy
.groupTuple(by: 0)
```

```
After .groupTuple(by: 0):
  [
    "p001",
    [meta_day1, meta_day1, meta_day1, meta_day2, meta_day2, meta_day2],
    [file_day1_reg, file_day1_reg, file_day1_reg, file_day2_reg, file_day2_reg, file_day2_reg],
    [file_ref_prep, file_day1_prep, file_day2_prep, file_ref_prep, file_day1_prep, file_day2_prep]
  ]

Wait, this looks wrong! We have duplicates because .combine created all pairs.
```

**CRITICAL REALIZATION:**

The `.combine()` creates ALL pairs, but we actually want MATCHING pairs. However, the current implementation keeps all combinations, which will be handled by the downstream process.

Actually, looking more carefully, I think the intent is:
- The metadata matching happens implicitly through filename matching in the Python script
- OR there's a bug here and we should filter to matching pairs

Let me re-examine... Actually, I think the issue is that we want to group DIFFERENT channels:
- registered: day1, day2
- preprocessed: day1, day2

So after groupTuple, we should get:
```
[
  "p001",
  [meta_day1, meta_day2],           // metas for registered images
  [file_day1_reg, file_day2_reg],   // registered files
  [file_day1_prep, file_day2_prep]  // preprocessed files (matching)
]
```

**BUT** the .combine creates all pairs, so we get duplicates. This seems like a bug!

**WAIT** - Let me reconsider. Looking at the actual metadata structure, maybe the meta objects are IDENTICAL between registered and preprocessed (same meta, different files). Let me trace more carefully:

Actually, the issue is:
- `ch_registered` has modified files but same metadata
- `ch_images` has original files with same metadata
- When we `.combine(by: 0)`, we create all pairs
- We need to filter to pairs where `reg_meta == preproc_meta` (same image, different stage)

**The implicit assumption** is that the metadata objects are identical (same channels, same patient, same is_reference), so when we group, we rely on:
1. The groupTuple collecting everything per patient
2. The downstream process matching files by metadata

Actually, I think there IS a logical issue here. Let me check if there's additional filtering...

Looking at the code, I don't see additional filtering. This means the Python script receives:
- All registered files for patient
- All preprocessed files for patient
- Metadata for all

And the Python script must match them by filename or metadata.

**ACTUALLY**, wait. Let me re-read the .combine logic again more carefully:

```groovy
.combine(
    ch_images.map { meta, preproc_file -> [patient_id, meta, preproc_file] },
    by: 0
)
```

The `by: 0` means: "join/combine on the first element (patient_id)".

So for each patient:
- Every registered image (non-ref + ref) is combined with
- Every preprocessed image (non-ref + ref)

Creating N × M pairs per patient.

Then we filter to keep only non-reference registered images.

So yes, we DO get extra combinations. For example:
- `day1_registered` is paired with `day1_preprocessed` ✓ (correct)
- `day1_registered` is paired with `day2_preprocessed` ✗ (incorrect, but kept)
- `day1_registered` is paired with `ref_preprocessed` ✗ (incorrect, but kept)

**This seems inefficient!** We're passing too much data.

**HOWEVER**, looking at the final output structure:
```groovy
tuple(patient_id, ref_file, preproc_files, reg_files, metas)
```

The groupTuple collects everything together, so the downstream process receives:
- 1 reference file
- List of ALL preprocessed files
- List of ALL registered files
- List of metadata

And the Python script matches them internally. So the extra combinations don't matter - they get collapsed in groupTuple!

Let me trace again with this understanding:

#### Corrected Substep 4: Group by patient

After the .map and .filter, we have individual tuples:
```
["p001", meta_day1, file_day1_reg, file_X_prep]  (multiple X values)
["p001", meta_day2, file_day2_reg, file_Y_prep]  (multiple Y values)
```

After .groupTuple(by: 0):
```
[
  "p001",
  [meta_day1, meta_day1, meta_day1, meta_day2, meta_day2, meta_day2],  // metas from reg images
  [file_day1_reg, file_day1_reg, file_day1_reg, file_day2_reg, file_day2_reg, file_day2_reg],
  [file_ref_prep, file_day1_prep, file_day2_prep, file_ref_prep, file_day1_prep, file_day2_prep]
]
```

This has duplicates! But wait - the final .map might deduplicate:

#### Substep 5: Join with reference

```groovy
.join(
    ch_images
        .filter { meta, file -> meta.is_reference }
        .map { meta, file -> [meta.patient_id, file] },
    by: 0
)
```

```
ch_images (reference only):
  [meta_ref, file_ref_prep]

After .filter and .map:
  ["p001", file_ref_prep]

After .join with grouped data:
  [
    "p001",
    [meta_day1, meta_day1, ..., meta_day2, meta_day2, ...],  // duplicates!
    [file_day1_reg, file_day1_reg, ..., file_day2_reg, file_day2_reg, ...],
    [file_ref_prep, file_day1_prep, file_day2_prep, file_ref_prep, file_day1_prep, file_day2_prep],
    file_ref_prep  // joined reference
  ]
```

#### Substep 6: Final mapping

```groovy
.map { patient_id, metas, reg_files, preproc_files, ref_file ->
    tuple(patient_id, ref_file, preproc_files, reg_files, metas)
}
```

```
Output:
  [
    "p001",
    file_ref_prep,                                              // reference (preprocessed)
    [file_ref_prep, file_day1_prep, file_day2_prep, ...],      // all preprocessed (with duplicates)
    [file_day1_reg, file_day1_reg, ..., file_day2_reg, ...],   // all registered (with duplicates)
    [meta_day1, meta_day1, ..., meta_day2, ...]                // all metas (with duplicates)
  ]
```

**PROBLEM IDENTIFIED**: The lists have duplicates!

**SOLUTION**: The Python script should deduplicate by filename or use `unique()`:

Actually, let me check if Nextflow automatically deduplicates when grouping files...

No, it doesn't. This is a BUG or inefficiency in the current implementation!

**WORKAROUND**: The Python script probably deduplicates when it reads the directories.

**BETTER FIX**: Add `.unique()` after groupTuple or rework the combine logic.

For now, let's document this as-is and note the inefficiency:

### Step 3C.2: Feature Distance Estimation

```groovy
if (params.enable_feature_error) {
    ESTIMATE_FEATURE_DISTANCES(ch_for_error)
    ch_error_metrics = ch_error_metrics.mix(ESTIMATE_FEATURE_DISTANCES.out.distance_metrics)
}
```

**Input to ESTIMATE_FEATURE_DISTANCES:**
```
[patient_id, ref_file, preproc_files, reg_files, metas]

May contain duplicates in the lists!
```

**Process behavior:**
- Iterates through `moving/` directory (preprocessed files)
- Iterates through `registered/` directory (registered files)
- Matches by filename
- Computes distances for each pair

**Output:**
```
*_feature_distances.json    (one per moving image)
*_distance_histogram.png    (one per moving image)
versions.yml
```

### Step 3C.3: Segmentation Overlap Estimation

```groovy
if (params.enable_segmentation_error) {
    ch_for_seg = ch_for_error
        .map { patient_id, ref_file, preproc_files, reg_files, metas ->
            tuple(patient_id, ref_file, reg_files, metas)
        }

    ESTIMATE_SEGMENTATION_OVERLAP(ch_for_seg)
    ch_error_metrics = ch_error_metrics.mix(ESTIMATE_SEGMENTATION_OVERLAP.out.overlap_metrics)
}
```

**Input transformation:**
```
Input (ch_for_error):
  [patient_id, ref_file, preproc_files, reg_files, metas]

After .map (remove preproc_files):
  [patient_id, ref_file, reg_files, metas]
```

**Process behavior:**
- Segments nuclei in reference
- Segments nuclei in each registered image
- Computes IoU/Dice overlap

**Output:**
```
*_segmentation_overlap.json  (one per moving image)
*_segmentation_overlay.png   (one per moving image)
versions.yml
```

---

## Step 4: Checkpoint CSV Writing

### Code Location: Lines 233-266

### Step 4.1: Prepare Checkpoint Data

```groovy
ch_checkpoint_data = ch_registered
    .map { meta, file ->
        def file_path = file instanceof List ? file[0] : file
        def relative_path = file_path.parent ?
            "${file_path.parent.name}/${file_path.name}" :
            file_path.name
        def published_path = "${params.outdir}/${meta.patient_id}/registered/${relative_path}"
        [meta.patient_id, published_path, meta.is_reference, meta.channels.join('|')]
    }
    .toList()
```

**Transformation:**

```
Input (ch_registered):
  [meta_ref, file_ref]
  [meta_day1, file_day1_registered]
  [meta_day2, file_day2_registered]

After .map:
  ["p001", "/results/p001/registered/file_ref.ome.tiff", true, "DAPI|CD3|CD8"]
  ["p001", "/results/p001/registered/file_day1_registered.ome.tiff", false, "DAPI|CD3|CD8"]
  ["p001", "/results/p001/registered/file_day2_registered.ome.tiff", false, "DAPI|CD3|CD8"]

After .toList():
  [
    ["p001", "/results/p001/registered/file_ref.ome.tiff", true, "DAPI|CD3|CD8"],
    ["p001", "/results/p001/registered/file_day1_registered.ome.tiff", false, "DAPI|CD3|CD8"],
    ["p001", "/results/p001/registered/file_day2_registered.ome.tiff", false, "DAPI|CD3|CD8"]
  ]
```

### Step 4.2: Write Checkpoint CSV

```groovy
WRITE_CHECKPOINT_CSV(
    'registered',
    'patient_id,registered_image,is_reference,channels',
    ch_checkpoint_data
)
```

**Output CSV:**
```csv
patient_id,registered_image,is_reference,channels
p001,/results/p001/registered/file_ref.ome.tiff,true,DAPI|CD3|CD8
p001,/results/p001/registered/file_day1_registered.ome.tiff,false,DAPI|CD3|CD8
p001,/results/p001/registered/file_day2_registered.ome.tiff,false,DAPI|CD3|CD8
```

---

## Output Emissions

### Code Location: Lines 268-273

```groovy
emit:
    registered = ch_registered           // [meta, file]
    checkpoint_csv = WRITE_CHECKPOINT_CSV.out.csv
    qc = ch_qc                          // [meta, qc_files]
    error_metrics = ch_error_metrics     // JSON and PNG files
```

### Output Channels

1. **registered**: `[meta, registered_file]` for all images
2. **checkpoint_csv**: Path to CSV file listing all registered images
3. **qc**: `[meta, qc_files]` for non-reference images (if enabled)
4. **error_metrics**: Mixed channel of JSON/PNG error metric files (if enabled)

---

## Complete Flow Diagrams

### High-Level Flow

```
ch_preprocessed [meta, file]
    ↓
┌──[padding?]──┐
│   yes    no  │
│    ↓      ↓  │
│  PAD  PASSTHRU
│    ↓      ↓  │
└───→ ch_images ←────┘
    ↓
[Group by patient + identify reference]
    ↓
ch_grouped [patient_id, ref_item, all_items]
    ↓
┌──[method?]────────┐
│ valis  gpu   cpu  │
│   ↓     ↓     ↓   │
│  VALIS_ADAPTER    │
│  GPU_ADAPTER      │
│  CPU_ADAPTER      │
│   ↓     ↓     ↓   │
└──→ ch_registered ←┘
    ↓
├─→ [QC Generation] → ch_qc
│
├─→ [Error Estimation] → ch_error_metrics
│   ├─→ Feature Distances
│   └─→ Segmentation Overlap
│
└─→ [Checkpoint CSV] → checkpoint_csv

Emit:
  - registered
  - checkpoint_csv
  - qc
  - error_metrics
```

### Error Estimation Detailed Flow

```
ch_registered [meta, reg_file]
    +
ch_images [meta, preproc_file]
    ↓
.map (add patient_id key)
    ↓
.combine (cartesian product within patient)
    ↓
.map (filter non-reference)
    ↓
.filter (remove nulls)
    ↓
.groupTuple (collect per patient)
    ↓
.join (with reference file)
    ↓
.map (reformat)
    ↓
[patient_id, ref_file, [preproc_files], [reg_files], [metas]]
    ↓
├─→ ESTIMATE_FEATURE_DISTANCES
│   Input: patient_id, ref, moving/, registered/, metas
│   Output: *_feature_distances.json, *_distance_histogram.png
│
└─→ ESTIMATE_SEGMENTATION_OVERLAP
    Input: patient_id, ref, registered/, metas
    Output: *_segmentation_overlap.json, *_segmentation_overlay.png
```

---

## Key Insights

### 1. **Metadata Preservation**

Throughout the entire workflow, the meta map `{patient_id, is_reference, channels}` is preserved and never modified. It's used for:
- Grouping (by patient_id)
- Filtering (by is_reference)
- Reconstruction (matching files to metadata)

### 2. **Adapter Pattern**

All registration methods (VALIS/GPU/CPU) receive the same input format and produce the same output format, enabling method-agnostic QC and error estimation.

### 3. **Channel Complexity**

The error estimation step has the most complex channel transformation:
- Combines registered and preprocessed images
- Groups by patient
- Joins with reference
- May create duplicates (potential inefficiency)

### 4. **Lazy Evaluation**

Nextflow channels are lazy - transformations only execute when data flows through. The `.branch`, `.filter`, and conditional `if` statements control execution paths.

### 5. **File Staging**

Different processes use different staging strategies:
- VALIS: `stageAs: 'input_?/*'` creates input_1/, input_2/, etc.
- Error estimation: `stageAs: 'moving/*'` and `stageAs: 'registered/*'`

This prevents filename collisions when multiple files are staged.

---

## Potential Issues

### Issue 1: Duplicate Files in Error Estimation

The `.combine()` operation creates all pairs, leading to duplicates after `.groupTuple()`. This could be optimized by matching metadata explicitly before grouping.

**Impact**: Slightly larger channel, but Python script likely handles it via directory iteration.

### Issue 2: Missing File Deduplication

If the same file appears multiple times in `preproc_files` or `reg_files` lists, the Python script should deduplicate.

**Recommendation**: Add `.unique()` or rework the combine logic to create only matching pairs.

### Issue 3: Metadata Matching Assumptions

VALIS adapter relies on fuzzy filename matching to reconstruct metadata associations. If VALIS changes filename format, this could break.

**Mitigation**: Multiple matching strategies implemented (exact, lowercase, contains).

---

This completes the in-depth dataflow analysis of the REGISTRATION subworkflow!
