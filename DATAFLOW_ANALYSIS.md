# Critical Dataflow Analysis: SPLIT_CHANNELS → QUANTIFY

## Input CSV Analysis

Example row from the failing run:
```csv
patient_id,registered_image,is_reference,channels
175029E,./results/175029E/registered/registered_slides/175029E_DAPI_SMA_corrected_registered.ome.tiff,true,DAPI|SMA
```

### Metadata Parsing (CsvUtils.parseMetadata)
```groovy
meta = [
    patient_id: '175029E',
    is_reference: true,
    channels: ['DAPI', 'SMA']  // ← List of 2 strings
]
```

---

## Dataflow Trace

### Step 1: SPLIT_CHANNELS Input
```groovy
// From postprocess.nf:55-57
SPLIT_CHANNELS(
    ch_registered.map { meta, file -> [meta, file, meta.is_reference] }
)

// Input tuple:
[
    meta: [patient_id: '175029E', is_reference: true, channels: ['DAPI', 'SMA']],
    file: Path('175029E_DAPI_SMA_corrected_registered.ome.tiff'),
    is_reference: true
]
```

### Step 2: SPLIT_CHANNELS Execution

**Process:** [split_channels.nf:3-67](modules/local/split_channels.nf#L3-L67)

The Python script `split_multichannel.py` will:
1. Read the 2-channel OME-TIFF (DAPI + SMA)
2. Because `is_reference=true`, save BOTH channels:
   - `DAPI.tiff`
   - `SMA.tiff`

**Output declaration:**
```groovy
output:
tuple val(meta), path("*.tiff", includeInputs: false), emit: channels
```

### Step 3: SPLIT_CHANNELS Output

**CRITICAL QUESTION: What does `path("*.tiff")` emit?**

According to Nextflow documentation:
- `path("*.tiff")` with glob pattern returns:
  - **Single Path object** if only ONE file matches
  - **List of Path objects** if MULTIPLE files match

For the reference slide with DAPI + SMA:
```groovy
// Expected output:
tuple(
    meta: [patient_id: '175029E', is_reference: true, channels: ['DAPI', 'SMA']],
    tiffs: [Path('DAPI.tiff'), Path('SMA.tiff')]  // ← List of 2 Path objects
)
```

For a non-reference slide with 3 markers (e.g., CD3|DAPI|P53):
```groovy
// Python script saves only CD3 and P53 (skips DAPI for non-reference)
tuple(
    meta: [patient_id: '175029E', is_reference: false, channels: ['CD3', 'DAPI', 'P53']],
    tiffs: [Path('CD3.tiff'), Path('P53.tiff')]  // ← List of 2 Path objects (DAPI skipped)
)
```

---

## Problem Analysis: The Original Bug

### What Was Happening BEFORE the Fix

**Location:** [postprocess.nf:65-74](subworkflows/local/postprocess.nf#L65-L74) (OLD CODE)

```groovy
ch_flatmapped = ch_split_output
    .flatMap { meta, tiffs ->
        // BUG: Assuming tiffs is always a List
        tiffs.collect { tiff ->  // ← PROBLEM HERE
            def channel_meta = meta.clone()
            channel_meta.id = "${meta.patient_id}_${tiff.baseName}"
            channel_meta.channel_name = tiff.baseName
            [channel_meta, tiff]
        }
    }
```

**Scenario A: When multiple files match (normal case)**
```groovy
tiffs = [Path('DAPI.tiff'), Path('SMA.tiff')]  // List
tiffs.collect { tiff -> ... }  // ✓ Works - iterates over 2 Path objects
// Result: [
//   [meta(id='175029E_DAPI'), Path('DAPI.tiff')],
//   [meta(id='175029E_SMA'), Path('SMA.tiff')]
// ]
```

**Scenario B: When single file matches (edge case that triggered the bug)**

This could happen if:
1. The Python script only outputs ONE file (e.g., single-channel image)
2. Nextflow's glob returns a single Path object instead of a List

```groovy
tiffs = Path('/beegfs/scratch/ieo7660/analysis_runs/valis/run_batch_1/work/a1/d2ac97e1aaa1fe3b5877858b120704/DAPI.tiff')
// ^ Single Path object, NOT a List!

// When you call .collect on a Path object:
tiffs.collect { segment -> ... }
// It iterates over PATH SEGMENTS (like calling path.split('/'))
// Results in: [
//   [meta(id='175029E_beegfs'), 'beegfs'],
//   [meta(id='175029E_scratch'), 'scratch'],
//   [meta(id='175029E_ieo7660'), 'ieo7660'],
//   [meta(id='175029E_analysis_runs'), 'analysis_runs'],
//   ...
// ]
```

---

## Evidence from Failed Tasks

Failed task names from the original error:
```
POSTPROCESSING:QUANTIFY (175029E - beegfs)
POSTPROCESSING:QUANTIFY (175029E - ieo7660)
POSTPROCESSING:QUANTIFY (175029E - scratch)
POSTPROCESSING:QUANTIFY (175029E - analysis_runs)
POSTPROCESSING:QUANTIFY (175029E - run_batch_1)
POSTPROCESSING:QUANTIFY (175029E - valis)
POSTPROCESSING:QUANTIFY (175029E - work)
POSTPROCESSING:QUANTIFY (175029E - a1)
POSTPROCESSING:QUANTIFY (175029E - L1CAM)  ← One valid marker!
POSTPROCESSING:QUANTIFY (175029E - d2ac97e1aaa1fe3b5877858b120704)
```

**Analysis:**
- Most tasks are path segments: `beegfs`, `scratch`, `ieo7660`, etc.
- ONE task shows a valid marker: `L1CAM`
- This suggests the slide `175029E_DAPI_L1CAM_corrected_registered.ome.tiff` was processed
- **Hypothesis:** This slide only had L1CAM channel saved (DAPI was skipped for non-reference)
- The glob `*.tiff` returned a **single Path object** for `L1CAM.tiff`
- But this Path object was the FULL ABSOLUTE PATH from the work directory
- When `.collect` was called on it, it iterated over the path segments

---

## The Fix: Type-Safe Handling

### Current Implementation (AFTER FIX)

**Location:** [postprocess.nf:65-78](subworkflows/local/postprocess.nf#L65-L78)

```groovy
ch_flatmapped = ch_split_output
    .flatMap { meta, tiffs ->
        // FIX: Normalize to List regardless of input type
        def tiff_list = tiffs instanceof List ? tiffs : [tiffs]

        // Now safe to iterate
        tiff_list.collect { tiff ->
            def channel_meta = meta.clone()
            channel_meta.id = "${meta.patient_id}_${tiff.baseName}"
            channel_meta.channel_name = tiff.baseName
            [channel_meta, tiff]
        }
    }
    .view { meta, tiff -> "After flatMap: meta.id=${meta.id}, channel=${meta.channel_name}, tiff=${tiff.name}" }
```

**Behavior:**
```groovy
// Case 1: Multiple files
tiffs = [Path('DAPI.tiff'), Path('SMA.tiff')]
tiff_list = [Path('DAPI.tiff'), Path('SMA.tiff')]  // Already a List
// ✓ Works correctly

// Case 2: Single file
tiffs = Path('L1CAM.tiff')
tiff_list = [Path('L1CAM.tiff')]  // Wrapped in List
// ✓ Now works correctly!
```

---

## Remaining Questions and Issues

### Question 1: Why did the glob return a single Path object?

**Possible reasons:**
1. **Non-reference slides with only 1 non-DAPI marker**
   - Example: `175029E_DAPI_L1CAM_corrected_registered.ome.tiff`
   - Channels: `['DAPI', 'L1CAM']`
   - Python script skips DAPI (non-reference)
   - Only saves `L1CAM.tiff`
   - Glob `*.tiff` matches 1 file → returns single Path

2. **Nextflow version-specific behavior**
   - Some Nextflow versions always return List
   - Others return single object for single match
   - Current fix handles both cases

### Question 2: Is the current fix sufficient?

**YES**, the fix handles both scenarios:
- ✓ Multiple files → already a List → works
- ✓ Single file → wrapped in List → works

### Question 3: Are there other potential issues?

**YES - CRITICAL ISSUE FOUND:**

Looking at the input CSV again:
```csv
175029E,./results/175029E/.../175029E_DAPI_L1CAM_corrected_registered.ome.tiff,false,DAPI|L1CAM
```

The metadata says `channels: ['DAPI', 'L1CAM']`, but the Python script will:
- Skip DAPI (because `is_reference=false`)
- Save only `L1CAM.tiff`

**Then in the MERGE process (postprocess.nf:124-136):**
```groovy
ch_split_grouped = SPLIT_CHANNELS.out.channels
    .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
    .groupTuple(by: 0)
    .map { patient_id, _metas, tiff_lists ->
        def all_tiffs = tiff_lists.flatten()  // ← PROBLEM HERE
        ...
    }
```

**The issue:** `tiff_lists` is a List of tiff collections. Each collection could be:
- A List: `[Path('DAPI.tiff'), Path('SMA.tiff')]`
- A single Path: `Path('L1CAM.tiff')`

When you call `.flatten()` on this:
- Lists are flattened correctly
- **Single Path objects are interpreted as path segments!**

---

## Required Additional Fix

### Location: [postprocess.nf:124-136](subworkflows/local/postprocess.nf#L124-L136)

**Current code:**
```groovy
ch_split_grouped = SPLIT_CHANNELS.out.channels
    .map { meta, tiffs -> [meta.patient_id, meta, tiffs] }
    .groupTuple(by: 0)
    .map { patient_id, _metas, tiff_lists ->
        def all_tiffs = tiff_lists.flatten()  // ← UNSAFE
        ...
    }
```

**Required fix:**
```groovy
ch_split_grouped = SPLIT_CHANNELS.out.channels
    .map { meta, tiffs ->
        // Normalize to List before grouping
        def tiff_list = tiffs instanceof List ? tiffs : [tiffs]
        [meta.patient_id, meta, tiff_list]
    }
    .groupTuple(by: 0)
    .map { patient_id, _metas, tiff_lists ->
        // Now safe to flatten (all elements are guaranteed to be Lists)
        def all_tiffs = tiff_lists.flatten()
        ...
    }
```

---

## Conclusion

### Issues Identified:
1. ✅ **FIXED:** `flatMap` operator not handling single Path objects
2. ❌ **NOT FIXED:** `groupTuple` + `flatten()` has the same issue in MERGE section

### Recommendation:
Apply the same type-safe pattern everywhere `SPLIT_CHANNELS.out.channels` is consumed.

**Pattern to use:**
```groovy
.map { meta, tiffs ->
    def tiff_list = tiffs instanceof List ? tiffs : [tiffs]
    // Now use tiff_list instead of tiffs
}
```
