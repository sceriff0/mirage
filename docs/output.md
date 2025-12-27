# ATEIA Pipeline Output Structure

## Table of Contents

1. [Output Directory Structure](#output-directory-structure)
2. [Preprocessing Outputs](#preprocessing-outputs)
3. [Registration Outputs](#registration-outputs)
4. [Postprocessing Outputs](#postprocessing-outputs)
5. [Results/Final Outputs](#resultsfinal-outputs)
6. [Checkpoint CSVs](#checkpoint-csvs)
7. [File Formats](#file-formats)

---

## Output Directory Structure

```
results/
├── csv/                                    # Checkpoint CSVs for restart
│   ├── preprocessed.csv
│   ├── registered.csv
│   └── postprocessed.csv
│
├── <patient_id>/
│   ├── preprocess/
│   │   ├── <patient>_corrected.ome.tif    # BaSiC-corrected image
│   │   └── <patient>_dims.txt             # Image dimensions
│   │
│   ├── registration/
│   │   ├── <patient>_registered.ome.tif   # Registered image
│   │   └── qc/
│   │       ├── <patient>_QC_RGB.png      # QC overlay (downsampled)
│   │       └── <patient>_QC_RGB.tif      # QC overlay (full res)
│   │
│   ├── segmentation/
│   │   ├── <patient>_nuclei_mask.tif     # Nuclear segmentation
│   │   └── <patient>_cell_mask.tif       # Whole-cell segmentation
│   │
│   ├── quantification/
│   │   ├── by_marker/
│   │   │   ├── DAPI_quant.csv
│   │   │   ├── FITC_quant.csv
│   │   │   └── TexasRed_quant.csv
│   │   └── merged_quant.csv              # All markers merged
│   │
│   ├── phenotyping/
│   │   ├── phenotypes_data.csv           # Cell phenotypes
│   │   ├── phenotypes_mask.tiff          # Phenotype visualization
│   │   └── phenotype_mapping.json        # Phenotype definitions
│   │
│   └── merged/
│       └── <patient>_pyramidal.ome.tif   # Final multi-resolution output
│
├── pipeline_info/
│   ├── software_versions.yml             # Software version tracking
│   ├── execution_report.html             # Nextflow execution report
│   └── timeline.html                     # Execution timeline
│
└── registration_qc/                       # Optional: Registration QC
    ├── <patient>_QC_RGB.png
    └── <patient>_QC_RGB.tif
```

---

## Preprocessing Outputs

### Corrected Images

**File:** `<patient_id>/preprocess/<patient>_corrected.ome.tif`

**Description:** Multi-channel OME-TIFF with BaSiC illumination correction applied

**Details:**
- DAPI channel repositioned to channel 0
- Flat-field and dark-field corrections applied
- Maintains original bit depth (typically 16-bit)
- OME-XML metadata includes channel names

**Size:** Similar to input (typically 2-10 GB per image)

### Dimension Files

**File:** `<patient_id>/preprocess/<patient>_dims.txt`

**Format:**
```
width,height,channels
25000,30000,3
```

**Purpose:** Used for optional padding to uniform dimensions

---

## Registration Outputs

### Registered Images

**File:** `<patient_id>/registration/<patient>_registered.ome.tif`

**Description:** Registered multi-channel OME-TIFF aligned to reference

**Coordinate System:**
- All images aligned to reference coordinate space
- DAPI channel used for registration (channel 0)
- All other channels transformed with same parameters

**Transformation Applied:**
- **VALIS:** Affine + non-rigid (optical flow)
- **GPU:** Affine + diffeomorphic (symmetric)
- **CPU:** Affine + diffeomorphic (symmetric)

### Registration QC

**Files:**
- `registration/qc/<patient>_QC_RGB.png` - Downsampled for quick review
- `registration/qc/<patient>_QC_RGB.tif` - Full resolution

**Visualization:**
- **Red channel:** Registered DAPI
- **Green channel:** Reference DAPI
- **Overlap (Yellow):** Good alignment
- **Red/Green separation:** Misalignment

**How to Read:**
- Perfect alignment: Uniform yellow
- Slight misalignment: Red and green edges
- Significant misalignment: Distinct red and green regions

---

## Postprocessing Outputs

### Segmentation Masks

#### Nuclei Mask

**File:** `<patient_id>/segmentation/<patient>_nuclei_mask.tif`

**Format:** 32-bit integer TIFF
- Background: 0
- Cell 1: 1
- Cell 2: 2
- ... (sequential labels)

**Method:** StarDist or DeepCell nuclear segmentation on DAPI channel

#### Cell Mask

**File:** `<patient_id>/segmentation/<patient>_cell_mask.tif`

**Format:** 32-bit integer TIFF (same labeling as nuclei)

**Method:** Nuclei expanded by `seg_expand_distance` pixels (default: 10)

**Purpose:** Whole-cell quantification (includes cytoplasm)

### Quantification Data

#### Per-Marker CSVs

**Location:** `<patient_id>/quantification/by_marker/`

**Format:** CSV files with columns:

```csv
label,y,x,area,eccentricity,perimeter,convex_area,axis_major_length,axis_minor_length,<MARKER_NAME>
1,1000,1500,250,0.65,85,255,25.3,12.1,150.5
2,1020,1550,180,0.58,70,185,22.1,10.8,125.3
...
```

**Columns:**
- **Morphology:** `label`, `y`, `x`, `area`, `eccentricity`, `perimeter`, `convex_area`, `axis_major_length`, `axis_minor_length`
- **Intensity:** `<MARKER_NAME>` - mean intensity for this marker

#### Merged Quantification CSV

**File:** `<patient_id>/quantification/merged_quant.csv`

**Description:** All markers merged into single CSV

**Format:**
```csv
label,y,x,area,...,DAPI,FITC,TexasRed,PANCK,CD3,CD8
1,1000,1500,250,...,150.5,200.3,125.8,180.2,50.1,25.3
```

**Uses:**
- Input for phenotyping
- Single-cell analysis
- Spatial analysis
- Machine learning

### Phenotyping Outputs

#### Phenotype Data CSV

**File:** `<patient_id>/phenotyping/phenotypes_data.csv`

**Format:**
```csv
label,phenotype,phenotype_id
1,CD3+ CD8+ (Cytotoxic T cell),1
2,CD3+ CD8- (Helper T cell),2
3,CD3- PANCK+ (Epithelial),3
...
```

**Columns:**
- `label`: Cell ID (matches quantification CSVs)
- `phenotype`: Human-readable phenotype name
- `phenotype_id`: Numeric phenotype identifier

#### Phenotype Mask

**File:** `<patient_id>/phenotyping/phenotypes_mask.tiff`

**Format:** 8-bit TIFF with phenotype IDs as pixel values

**Visualization:**
- Load in ImageJ/QuPath with LUT for cell type visualization
- Overlay on original image for spatial distribution

#### Phenotype Mapping JSON

**File:** `<patient_id>/phenotyping/phenotype_mapping.json`

**Format:**
```json
{
  "1": {
    "name": "CD3+ CD8+ (Cytotoxic T cell)",
    "markers_positive": ["CD3", "CD8"],
    "markers_negative": [],
    "count": 1523
  },
  "2": {
    "name": "CD3+ CD8- (Helper T cell)",
    "markers_positive": ["CD3"],
    "markers_negative": ["CD8"],
    "count": 892
  }
}
```

**Purpose:** Phenotype definitions and cell counts

---

## Results/Final Outputs

### Pyramidal OME-TIFF

**File:** `<patient_id>/merged/<patient>_pyramidal.ome.tif`

**Description:** Multi-resolution OME-TIFF with registered images + masks

**Channels:**
1. DAPI (nuclear stain)
2. Marker 1 (e.g., FITC)
3. Marker 2 (e.g., Texas Red)
4. ... additional markers
5. Nuclei segmentation mask
6. Cell segmentation mask
7. Phenotype mask

**Resolution Levels:**
- Level 0: Full resolution
- Level 1: 50% (scale factor 2)
- Level 2: 25%
- Level 3: 12.5%
- Level 4: 6.25%

**Tile Size:** 512x512 pixels (configurable)

**Uses:**
- Visualization in QuPath, OMERO, or web viewers
- Region-based analysis
- Export to other formats

### Final Archive

**Location:** `<savedir>/<id>_<method>/`

**Contents:** All pipeline outputs copied to long-term storage

---

## Checkpoint CSVs

### Purpose

Enable pipeline restart from any step without re-running completed work.

### preprocessed.csv

**Location:** `results/csv/preprocessed.csv`

**Columns:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/results/P001/preprocess/P001_corrected.ome.tif,true,DAPI|FITC|TexasRed
```

**Use:** Restart from registration step

### registered.csv

**Location:** `results/csv/registered.csv`

**Columns:**
```csv
patient_id,registered_image,is_reference,channels
P001,/results/P001/registration/P001_registered.ome.tif,true,DAPI|FITC|TexasRed
```

**Use:** Restart from postprocessing step

### postprocessed.csv

**Location:** `results/csv/postprocessed.csv`

**Columns:**
```csv
patient_id,is_reference,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,/results/P001/phenotyping/phenotypes_data.csv,...
```

**Use:** Restart from results step

---

## File Formats

### OME-TIFF

**Extension:** `.ome.tif` or `.ome.tiff`

**Specification:** Open Microscopy Environment TIFF

**Advantages:**
- Self-contained metadata (OME-XML)
- Multi-channel support
- Multi-resolution (pyramidal)
- Widely supported (ImageJ, QuPath, OMERO, bioformats)

**Reading in Python:**
```python
import tifffile
img = tifffile.imread('image.ome.tif')  # Returns (C, Y, X) array
```

**Reading in R:**
```r
library(tiff)
img <- readTIFF('image.ome.tif', all=TRUE)
```

### Quantification CSVs

**Format:** Standard CSV (RFC 4180)

**Encoding:** UTF-8

**Delimiter:** Comma (`,`)

**Headers:** First row

**Reading in Python:**
```python
import pandas as pd
df = pd.read_csv('merged_quant.csv')
```

**Reading in R:**
```r
df <- read.csv('merged_quant.csv')
```

### Segmentation Masks

**Format:** 32-bit integer TIFF

**Values:**
- 0: Background
- 1-N: Cell labels

**Reading in Python:**
```python
import tifffile
mask = tifffile.imread('cell_mask.tif')  # dtype: int32
```

**Overlaying in ImageJ:**
1. Open image: `File > Open > image.ome.tif`
2. Open mask: `File > Open > cell_mask.tif`
3. Convert mask to ROIs: `Plugins > MorphoLibJ > Label Images > Labels to ROIs`

---

## Disk Space Estimates

Per patient (typical multiplex IF image: 25000x30000 pixels, 5 channels):

| Step | Output Size | Cumulative |
|------|-------------|------------|
| Preprocessing | 3-5 GB | 3-5 GB |
| Registration | 3-5 GB | 6-10 GB |
| Segmentation | 200-500 MB | 6.5-10.5 GB |
| Quantification | 10-50 MB | 6.5-10.6 GB |
| Phenotyping | 10-50 MB | 6.5-10.7 GB |
| Pyramidal TIFF | 5-8 GB | 11.5-18.7 GB |

**Total per patient:** ~12-20 GB (with all intermediates)

**Final archived:** ~8-12 GB (if intermediates removed)

---

## Quality Control Checklist

After pipeline completion, verify:

1. **Registration QC:**
   - Check `registration_qc/*.png` for alignment quality
   - Yellow overlay indicates good registration
   - Red/green separation indicates misalignment

2. **Segmentation QC:**
   - Load masks in ImageJ with image overlay
   - Verify nuclei are properly segmented
   - Check for over/under-segmentation

3. **Quantification QC:**
   - Check `merged_quant.csv` cell count
   - Verify intensity distributions make sense
   - Look for outliers or failed channels

4. **Phenotyping QC:**
   - Review `phenotype_mapping.json` for expected cell types
   - Check cell counts are reasonable
   - Verify phenotype definitions match biology

5. **Final Output:**
   - Open pyramidal TIFF in QuPath or ImageJ
   - Verify all channels present
   - Check masks overlay correctly

---

## Additional Resources

- [Usage Guide](usage.md)
- [Input Format](INPUT_FORMAT.md)
- [Troubleshooting](usage.md#troubleshooting)
