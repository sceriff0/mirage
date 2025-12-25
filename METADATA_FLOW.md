# Metadata Flow Through Pipeline

This document illustrates how metadata (`patient_id`, `is_reference`, `channels`) flows through the entire ATEIA pipeline.

## Overview

```
Input CSV → Preprocessing → Registration → Postprocessing → Results
   ↓             ↓              ↓               ↓              ↓
[meta]      [meta, file]   [meta, file]    [meta, file]   [files]
```

## Detailed Flow

### 1. Input CSV Parsing
```
INPUT: samples.csv
┌─────────────┬──────────────┬──────────────┬───────────┬───────────┬──────────────┐
│ patient_id  │ path_to_file │ is_reference │ channel_1 │ channel_2 │ channel_3    │
├─────────────┼──────────────┼──────────────┼───────────┼───────────┼──────────────┤
│ P001        │ /path/P001   │ true         │ DAPI      │ FITC      │ Texas Red    │
│ P002        │ /path/P002   │ false        │ FITC      │ DAPI      │ Texas Red    │
└─────────────┴──────────────┴──────────────┴───────────┴───────────┴──────────────┘

PARSED TO: Channel<[meta, file]>
┌──────────────────────────────────────────────┬──────────────┐
│ meta                                         │ file         │
├──────────────────────────────────────────────┼──────────────┤
│ {patient_id: P001,                           │ /path/P001   │
│  is_reference: true,                         │              │
│  channels: [DAPI, FITC, Texas Red]}          │              │
│                                              │              │
│ {patient_id: P002,                           │ /path/P002   │
│  is_reference: false,                        │              │
│  channels: [FITC, DAPI, Texas Red]}          │              │
└──────────────────────────────────────────────┴──────────────┘
```

### 2. Preprocessing Step
```
INPUT: Channel<[meta, file]>

CONVERT_IMAGE:
  - Reads channels from meta: [FITC, DAPI, Texas Red]
  - Reorders to put DAPI first: [DAPI, FITC, Texas Red]
  - Updates meta.channels with new order
  - Outputs: [meta_updated, ome_file]

PREPROCESS, PAD_IMAGES:
  - Process files
  - Metadata preserved alongside files
  - Outputs: [meta, padded_file]

CHECKPOINT CSV: preprocessed.csv
patient_id,padded_image,is_reference,channels
P001,/results/P001/...padded.ome.tif,true,DAPI|FITC|Texas Red
P002,/results/P002/...padded.ome.tif,false,DAPI|FITC|Texas Red
```

### 3. Registration Step
```
INPUT: Channel<[meta, file]> (from preprocessing OR checkpoint)

STEP 1: Find reference using is_reference
┌──────────────────────────────────────┐
│ Collect all [meta, file] tuples     │
│ Find item where meta.is_reference=true │
│ → Reference: P001                    │
└──────────────────────────────────────┘

STEP 2: Create registration pairs
┌──────────────────────────────────────┐
│ Reference file: P001_padded.ome.tif  │
│ Moving files:   P002_padded.ome.tif  │
│                 (where is_reference=false) │
└──────────────────────────────────────┘

STEP 3: Register moving images
  - Input: (ref_file, moving_file)
  - Output: registered_file

STEP 4: Reconstruct metadata
  - Match registered files back to original metadata by filename
  - Output: [meta, registered_file]

CHECKPOINT CSV: registered.csv
patient_id,registered_image,is_reference,channels
P001,/results/P001/...registered.ome.tif,true,DAPI|FITC|Texas Red
P002,/results/P002/...registered.ome.tif,false,DAPI|FITC|Texas Red
```

### 4. Postprocessing Step
```
INPUT: Channel<[meta, file]> (from registration OR checkpoint)

STEP 1: Extract reference using is_reference
┌──────────────────────────────────────┐
│ Filter: meta.is_reference == true    │
│ → Reference file for segmentation    │
└──────────────────────────────────────┘

STEP 2: Segment reference image
  - SEGMENT(reference_file)
  - Outputs: cell_mask

STEP 3: Split all channels
  - For each [meta, file]:
      - Extract is_reference from meta
      - SPLIT_CHANNELS(file, is_reference)

STEP 4: Quantify, Merge, Phenotype
  - Process all images together
  - Output: phenotype results for entire cohort

CHECKPOINT CSV: postprocessed.csv
patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,/results/phenotype.csv,/results/phenotype_mask.tif,/results/phenotype_mapping.csv,/results/merged.csv,/results/cell_mask.tif
```

## Key Improvements

### ✅ Before (Problems)
- ❌ Metadata lost during registration
- ❌ Reference selected by filename matching
- ❌ patient_id extracted from filenames using fallback
- ❌ Postprocessing checkpoint missing patient_id

### ✅ After (Solutions)
- ✅ Metadata preserved through entire pipeline
- ✅ Reference selected using `is_reference` boolean
- ✅ patient_id carried in metadata throughout
- ✅ All checkpoints include full metadata

## Reference Selection Comparison

### Old Method (Filename Matching)
```groovy
// Registration
def reference = sorted_files.find { f ->
    def filename = f.name.toUpperCase()
    reference_markers.every { marker ->
        filename.contains(marker.toUpperCase())
    }
}
// Problem: Depends on filename conventions
// Problem: Fails if filename doesn't match pattern
```

### New Method (Metadata)
```groovy
// Registration
def reference_item = sorted_items.find { it[0].is_reference }
// ✅ Explicit, reliable
// ✅ Works with any filename
// ✅ Consistent with CSV design
```

## Metadata Schema

Throughout the pipeline, the `meta` map contains:

```groovy
meta = [
    patient_id: String,      // e.g., "P001"
    is_reference: Boolean,   // true or false
    channels: List<String>   // e.g., ["DAPI", "FITC", "Texas Red"]
]
```

## Channel Reordering During Conversion

DAPI can be specified in any position in the input CSV, and will be automatically moved to channel 0:

```
Input CSV:
┌──────────────┬───────────┬───────────┬──────────────┐
│ channel_1    │ channel_2 │ channel_3 │              │
├──────────────┼───────────┼───────────┼──────────────┤
│ FITC         │ Texas Red │ DAPI      │ → [FITC, Texas Red, DAPI] │
└──────────────┴───────────┴───────────┴──────────────┘

After CONVERT_IMAGE:
┌──────────────┬───────────┬──────────────┐
│ channel_0    │ channel_1 │ channel_2    │
├──────────────┼───────────┼──────────────┤
│ DAPI         │ FITC      │ Texas Red    │  ← DAPI moved to first
└──────────────┴───────────┴──────────────┘

Updated meta.channels: [DAPI, FITC, Texas Red]
```

## Checkpoint CSV Loading

All checkpoint CSVs can be loaded to restart the pipeline from any step:

### Loading `preprocessed.csv` (step=registration)
```groovy
ch_for_registration = channel.fromPath(params.input)
    .splitCsv(header: true)
    .map { row ->
        def channels = row.channels.split('\\|')  // Parse pipe-separated
        def meta = [
            patient_id: row.patient_id,
            is_reference: row.is_reference.toBoolean(),
            channels: channels
        ]
        return [meta, file(row.padded_image)]
    }
```

### Loading `registered.csv` (step=postprocessing)
```groovy
ch_for_postprocessing = channel.fromPath(params.input)
    .splitCsv(header: true)
    .map { row ->
        def channels = row.channels.split('\\|')
        def meta = [
            patient_id: row.patient_id,
            is_reference: row.is_reference.toBoolean(),
            channels: channels
        ]
        return [meta, file(row.registered_image)]
    }
```

### Loading `postprocessed.csv` (step=results)
```groovy
ch_checkpoint = channel.fromPath(params.input)
    .splitCsv(header: true)
    .first()

// Extract files (no metadata needed for results step)
ch_phenotype_csv = ch_checkpoint.map { row -> file(row.phenotype_csv) }
ch_phenotype_mask = ch_checkpoint.map { row -> file(row.phenotype_mask) }
// ... etc
```

## Summary

The pipeline now has a **consistent metadata model** throughout:
1. ✅ All data flows as `[meta, file]` tuples
2. ✅ Reference selection uses explicit `is_reference` boolean
3. ✅ patient_id is always available for traceability
4. ✅ Channel information preserved for downstream analysis
5. ✅ Checkpoints contain complete metadata for restartability
