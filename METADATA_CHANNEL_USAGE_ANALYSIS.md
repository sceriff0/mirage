# Metadata vs Filename Convention Analysis

## Executive Summary

The pipeline **properly maintains channel metadata** (`meta.channels`) through Nextflow processes, but several Python scripts still rely on **filename parsing** instead of using this metadata. This creates potential inconsistencies and fragility.

## Current State

### ✅ What's Working Well

1. **Nextflow Metadata Flow**
   - Channel names properly parsed from input CSV (`channels: "PANCK|SMA|DAPI"`)
   - Stored in `meta.channels` as list: `['PANCK', 'SMA', 'DAPI']`
   - Propagated through all processes
   - Written to checkpoint CSVs

2. **OME-TIFF Metadata**
   - Most scripts write OME metadata with channel names
   - Many scripts read OME metadata as primary source
   - Good fallback mechanism when OME metadata exists

3. **Scripts with Good Practices**
   - `classify.py`: Reads OME first, filename fallback
   - `merge_registered.py`: Reads OME first
   - `preprocess.py`: Writes OME metadata
   - All `register_*.py` scripts: Try OME first

### ❌ Problem Areas

#### **Critical: quantify.py**

**Issue:** Completely relies on filename parsing with no metadata reading

**Location:** `bin/quantify.py:264-265`
```python
# Extract channel name from filename
channel_name = Path(channel_path).stem.split('_')[-1]
```

**Problem:**
- Assumes filename format: `sample_registered_CHANNELNAME.tiff`
- Takes last underscore-delimited segment
- No validation against metadata
- Breaks if filename format changes

**Called by:** `modules/local/quantify.nf:30-35`
```groovy
quantify.py \
    --channel_tiff ${channel_tiff} \
    --mask_file ${seg_mask} \
    --outdir . \
    --output_file ${channel_tiff.simpleName}_quant.csv
```

**Impact:** Medium-High
- Affects downstream quantification results
- Channel mismatch could assign wrong marker to measurements

---

#### **Moderate: SPLIT_CHANNELS Process**

**Issue:** Script has capability but Nextflow doesn't use it

**Script has argument:** `bin/split_multichannel.py:175`
```python
parser.add_argument('--channels', nargs='+', default=None,
                   help='Channel names (optional, will try to read from OME metadata)')
```

**But process doesn't pass it:** `modules/local/split_channels.nf:30-34`
```groovy
split_multichannel.py \
    ${registered_image} \
    . \
    ${ref_flag} \
    ${args}  # Missing: --channels argument
```

**Current behavior:**
1. Reads OME metadata (usually works ✓)
2. Falls back to `Channel_0, Channel_1, ...` if OME missing
3. Could be more explicit by passing from Nextflow metadata

**Impact:** Low-Medium
- Usually works due to OME metadata from registration
- Fallback names are generic but functional

---

#### **Minor: generate_registration_qc.py**

**Issue:** Parses channel names from filename

**Location:** `bin/generate_registration_qc.py:34-48`
```python
def extract_markers_from_filename(filename: str) -> List[str]:
    """Extract marker names from filename."""
    filename = filename.replace('.ome.tiff', '').replace('.ome.tif', '')
    parts = filename.split('_')
    # Assuming format: SampleID_Marker1_Marker2_..._MarkerN
    if len(parts) > 1:
        return parts[1:]  # Skip sample ID
    return []
```

**Impact:** Low
- Only used for QC visualization
- Not critical to analysis pipeline

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT CSV                                                    │
│ channels: "PANCK|SMA|DAPI"                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ main.nf parseMetadata()                                      │
│ meta.channels = ['PANCK', 'SMA', 'DAPI']                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESS                                                   │
│ • convert_image.py → Writes OME-TIFF with channel metadata  │
│ • pad_image.py → Preserves OME metadata                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ REGISTRATION                                                 │
│ • register_*.py → Reads OME metadata ✓                      │
│                   (fallback: filename parsing)               │
│ • Writes OME metadata to output ✓                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ POSTPROCESSING                                               │
│                                                              │
│ ┌────────────────────────────────────────┐                  │
│ │ SEGMENT (segment.py)                   │                  │
│ │ • Hardcoded: DAPI = channel 0          │                  │
│ └────────────────────────────────────────┘                  │
│                                                              │
│ ┌────────────────────────────────────────┐                  │
│ │ SPLIT_CHANNELS (split_multichannel.py) │                  │
│ │ • Reads OME metadata ✓                 │                  │
│ │ • Could receive --channels from NF     │                  │
│ └──────────────┬─────────────────────────┘                  │
│                │                                             │
│                ▼                                             │
│         Individual channel TIFFs                             │
│         (filename = channel name)                            │
│                │                                             │
│                ▼                                             │
│ ┌────────────────────────────────────────┐                  │
│ │ QUANTIFY (quantify.py)                 │                  │
│ │ • Parses channel from FILENAME ✗       │                  │
│ │ • NO metadata reading!                 │                  │
│ └────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Fix Recommendations

### 1. Fix quantify.py (HIGH PRIORITY)

#### A. Update Python Script

**File:** `bin/quantify.py`

**Add argument:**
```python
parser.add_argument('--channel-name', type=str, required=True,
                   help='Name of the channel being quantified')
```

**Replace line 265:**
```python
# OLD:
channel_name = Path(channel_path).stem.split('_')[-1]

# NEW:
channel_name = args.channel_name
```

#### B. Update Nextflow Process

**File:** `modules/local/quantify.nf`

**Modify script section (lines 30-40):**
```groovy
script:
def args = task.ext.args ?: ''
// Extract channel name from metadata
def channel_index = channel_tiff.name.tokenize('_').last().replaceAll('.tiff', '')
def channel_name = meta.channels?.find { it.toLowerCase() == channel_index.toLowerCase() } ?: channel_index

"""
quantify.py \\
    --channel_tiff ${channel_tiff} \\
    --channel-name ${channel_name} \\
    --mask_file ${seg_mask} \\
    --outdir . \\
    --output_file ${channel_tiff.simpleName}_quant.csv
"""
```

**Better alternative (if split_channels emits metadata):**
```groovy
// If SPLIT_CHANNELS can emit channel name with each file:
input:
tuple val(meta), path(channel_tiff), val(channel_name), path(seg_mask)

script:
"""
quantify.py \\
    --channel_tiff ${channel_tiff} \\
    --channel-name ${channel_name} \\
    --mask_file ${seg_mask} \\
    --outdir . \\
    --output_file ${channel_tiff.simpleName}_quant.csv
"""
```

---

### 2. Enhance SPLIT_CHANNELS (MEDIUM PRIORITY)

#### Update Process to Pass Metadata

**File:** `modules/local/split_channels.nf`

**Modify script section (lines 25-40):**
```groovy
script:
def args = task.ext.args ?: ''
def ref_flag = meta.is_reference ? '--reference' : ''
// Pass channel names from metadata
def channel_args = meta.channels ? "--channels ${meta.channels.join(' ')}" : ""

"""
split_multichannel.py \\
    ${registered_image} \\
    . \\
    ${ref_flag} \\
    ${channel_args} \\
    ${args}
"""
```

#### Enhance Output to Include Channel Names

**Add to output section:**
```groovy
output:
tuple val(meta), path("*_ch*.tiff"), emit: channels
path("channel_mapping.txt"), emit: mapping  // File with channel name -> filename mapping

script:
// ...
"""
split_multichannel.py \\
    ${registered_image} \\
    . \\
    ${ref_flag} \\
    ${channel_args} \\
    ${args}

# Create mapping file for downstream processes
ls *_ch*.tiff | while read f; do
    ch_num=\$(echo \$f | grep -oP '(?<=ch)\\d+')
    ch_name=\$(echo "${meta.channels.join(' ')}" | cut -d' ' -f\$((ch_num+1)))
    echo "\$f\t\$ch_name"
done > channel_mapping.txt
"""
```

---

### 3. Make OME Metadata Mandatory (LONG-TERM)

#### A. Add Validation to Scripts

**Create utility function in `bin/_common.py`:**
```python
def get_channel_names_strict(filepath: str) -> List[str]:
    """
    Extract channel names from OME-TIFF metadata.
    Raises ValueError if metadata is missing or invalid.

    Args:
        filepath: Path to OME-TIFF file

    Returns:
        List of channel names

    Raises:
        ValueError: If OME metadata is missing or doesn't contain channel names
    """
    import tifffile
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(filepath) as tif:
        if not hasattr(tif, 'ome_metadata') or not tif.ome_metadata:
            raise ValueError(f"File {filepath} is missing OME metadata")

        try:
            root = ET.fromstring(tif.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            channels = root.findall('.//ome:Channel', ns)

            if not channels:
                raise ValueError(f"OME metadata in {filepath} contains no channel information")

            channel_names = [ch.get('Name') for ch in channels]

            if not all(channel_names):
                raise ValueError(f"Some channels in {filepath} have no names")

            return channel_names

        except ET.ParseError as e:
            raise ValueError(f"Invalid OME XML in {filepath}: {e}")
```

#### B. Update Scripts to Use Strict Mode

**Add parameter to enable strict mode:**
```python
parser.add_argument('--strict-metadata', action='store_true',
                   help='Require OME metadata, fail if missing (recommended)')
```

**Use in scripts:**
```python
if args.strict_metadata:
    channel_names = get_channel_names_strict(input_path)
else:
    # Existing fallback logic
    channel_names = get_ome_channel_names(input_path)
    if not channel_names:
        channel_names = parse_from_filename(input_path)
```

---

### 4. Add Pipeline Validation

**Create validation script:** `bin/validate_ome_metadata.py`

```python
#!/usr/bin/env python3
"""
Validate that OME-TIFF files contain proper channel metadata.
"""
import sys
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

def validate_ome_tiff(filepath: Path, expected_channels: List[str] = None) -> Tuple[bool, str]:
    """
    Validate OME-TIFF file has channel metadata.

    Args:
        filepath: Path to OME-TIFF file
        expected_channels: Optional list of expected channel names

    Returns:
        (is_valid, message)
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            # Check for OME metadata
            if not hasattr(tif, 'ome_metadata') or not tif.ome_metadata:
                return False, "Missing OME metadata"

            # Parse OME XML
            root = ET.fromstring(tif.ome_metadata)
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            channels = root.findall('.//ome:Channel', ns)

            if not channels:
                return False, "OME metadata contains no channels"

            # Get channel names
            channel_names = [ch.get('Name') for ch in channels]

            if not all(channel_names):
                return False, f"Some channels missing names: {channel_names}"

            # Check against expected channels if provided
            if expected_channels:
                if len(channel_names) != len(expected_channels):
                    return False, f"Expected {len(expected_channels)} channels, found {len(channel_names)}"

                if set(channel_names) != set(expected_channels):
                    return False, f"Channel mismatch. Expected: {expected_channels}, Found: {channel_names}"

            return True, f"Valid OME-TIFF with {len(channel_names)} channels: {', '.join(channel_names)}"

    except Exception as e:
        return False, f"Error reading file: {e}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate OME-TIFF metadata')
    parser.add_argument('image', type=Path, help='OME-TIFF file to validate')
    parser.add_argument('--expected-channels', nargs='+', help='Expected channel names')
    args = parser.parse_args()

    is_valid, message = validate_ome_tiff(args.image, args.expected_channels)

    print(f"{'✓' if is_valid else '✗'} {args.image.name}: {message}")

    sys.exit(0 if is_valid else 1)
```

**Add validation process:**

```groovy
// modules/local/validate_ome.nf
process VALIDATE_OME {
    tag "$meta.id"
    label 'process_single'

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path(image), emit: validated
    path("*.validation.txt"), emit: report

    script:
    def expected = meta.channels ? "--expected-channels ${meta.channels.join(' ')}" : ""
    """
    validate_ome_metadata.py ${image} ${expected} | tee ${meta.id}.validation.txt
    """
}
```

---

## Implementation Plan

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix `quantify.py` to accept `--channel-name` argument
- [ ] Update `QUANTIFY` process to pass channel name from metadata
- [ ] Test with sample dataset
- [ ] Verify quantification results match expected channels

### Phase 2: Enhancements (Week 2)
- [ ] Update `SPLIT_CHANNELS` to pass `--channels` from metadata
- [ ] Create channel mapping output from `SPLIT_CHANNELS`
- [ ] Update downstream processes to use mapping
- [ ] Add validation tests

### Phase 3: Validation Infrastructure (Week 3)
- [ ] Create `validate_ome_metadata.py` script
- [ ] Add `VALIDATE_OME` process after each OME-writing step
- [ ] Create utility functions in `_common.py`
- [ ] Add `--strict-metadata` mode to all scripts

### Phase 4: Migration to Strict Mode (Week 4)
- [ ] Enable strict metadata validation in test profile
- [ ] Fix any failures in test datasets
- [ ] Document metadata requirements
- [ ] Make strict mode default for production

---

## Testing Checklist

### Unit Tests
- [ ] Test `quantify.py` with explicit channel names
- [ ] Test `split_multichannel.py` with `--channels` argument
- [ ] Test validation script with valid/invalid OME-TIFFs

### Integration Tests
- [ ] Run full pipeline with metadata validation enabled
- [ ] Verify channel names in final quantification CSVs
- [ ] Check all intermediate OME-TIFFs have proper metadata
- [ ] Test with edge cases (single channel, many channels, special characters)

### Regression Tests
- [ ] Verify quantification results unchanged after fix
- [ ] Check that existing workflows still work
- [ ] Test backward compatibility with old data

---

## Expected Benefits

1. **Robustness**: Pipeline no longer relies on filename conventions
2. **Flexibility**: Can rename files without breaking analysis
3. **Validation**: Early detection of metadata issues
4. **Reproducibility**: Explicit channel tracking throughout pipeline
5. **Debugging**: Easier to trace channel identity issues

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing workflows | High | Maintain backward compatibility with fallback mode |
| OME metadata missing in old data | Medium | Add conversion script for legacy data |
| Performance overhead from validation | Low | Make validation optional, only in test/dev modes |
| Complex channel name handling | Medium | Create comprehensive test suite with edge cases |

---

## Files Requiring Changes

### Python Scripts
- `bin/quantify.py` - Add `--channel-name` argument (CRITICAL)
- `bin/_common.py` - Add metadata utility functions
- `bin/validate_ome_metadata.py` - NEW: Validation script

### Nextflow Modules
- `modules/local/quantify.nf` - Pass channel name from metadata
- `modules/local/split_channels.nf` - Pass `--channels` argument
- `modules/local/validate_ome.nf` - NEW: Validation process

### Workflows
- `subworkflows/local/postprocess.nf` - Wire channel metadata through QUANTIFY

### Documentation
- Update README with metadata requirements
- Document channel naming conventions
- Add troubleshooting guide for metadata issues
