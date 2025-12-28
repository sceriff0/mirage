# Pipeline Validation Test Cases

This document outlines comprehensive test cases for the logical correctness fixes applied to the ATEIA pipeline.

## Test Suite Organization

```
tests/
├── test_validation.md          # This file
├── fixtures/                   # Test input files
│   ├── valid_input.csv
│   ├── invalid_multi_ref.csv
│   ├── invalid_no_ref.csv
│   ├── invalid_dapi_position.csv
│   └── valid_checkpoint_*.csv
└── expected/                   # Expected outputs
    └── error_messages/
```

---

## Issue #1: Multi-Reference Validation

### Test Case 1.1: Valid Input - One Reference Per Patient
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_ref.nd2,true,DAPI|PANCK|SMA
P001,/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
P001,/data/P001_mov2.nd2,false,DAPI|PANCK|SMA
P002,/data/P002_ref.nd2,true,DAPI|CD3|CD8
P002,/data/P002_mov1.nd2,false,DAPI|CD3|CD8
```

**Expected Result:** ✅ Pass validation
**Test Command:**
```bash
nextflow run main.nf \
  --input tests/fixtures/valid_input.csv \
  --step preprocessing \
  --outdir test_output
```

---

### Test Case 1.2: Invalid Input - Multiple References Per Patient
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_ref1.nd2,true,DAPI|PANCK|SMA
P001,/data/P001_ref2.nd2,true,DAPI|PANCK|SMA  # ❌ Second reference!
P001,/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
```

**Expected Result:** ❌ Fail with error message:
```
❌ CRITICAL: Reference image validation failed!

Each patient must have EXACTLY ONE reference image.

Violations found:
  • Patient 'P001': Multiple reference images found (2 references out of 3 images)

💡 Hint: Check your input CSV's 'is_reference' column. For each patient_id:
  - Exactly one row should have is_reference=true
  - All other rows should have is_reference=false
```

**Test Command:**
```bash
nextflow run main.nf \
  --input tests/fixtures/invalid_multi_ref.csv \
  --step preprocessing \
  --outdir test_output
# Should fail immediately after parsing CSV
```

---

### Test Case 1.3: Invalid Input - No Reference Per Patient
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
P001,/data/P001_mov2.nd2,false,DAPI|PANCK|SMA
```

**Expected Result:** ❌ Fail with error message:
```
❌ CRITICAL: Reference image validation failed!

Each patient must have EXACTLY ONE reference image.

Violations found:
  • Patient 'P001': No reference image found (has 2 images)
```

---

## Issue #2: DAPI Position Validation

### Test Case 2.1: Valid - DAPI in Channel 0 (Conversion Enabled)
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001.nd2,true,PANCK|DAPI|SMA  # DAPI not first, but will be moved by converter
```

**Expected Result:** ✅ Pass - Converter places DAPI first
**Validation:** After conversion, channels.txt should contain `DAPI,PANCK,SMA`

---

### Test Case 2.2: Invalid - DAPI Not First (Conversion Disabled)
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001.ome.tiff,true,PANCK|DAPI|SMA  # ❌ DAPI not first!
```

**Config:** `skip_nd2_conversion = true`

**Expected Result:** ❌ Fail with error message:
```
❌ CRITICAL: DAPI must be in channel 0 for P001!

You are using pre-converted images (skip_nd2_conversion=true).
The channels in your input CSV are: [PANCK, DAPI, SMA]
DAPI is in position: 1

💡 Fix: Update your input CSV so DAPI is listed first in the 'channels' column.
   Example: 'DAPI|PANCK|SMA' instead of 'PANCK|DAPI|SMA'

💡 Note: The actual OME-TIFF file must also have DAPI in channel 0!
```

**Test Command:**
```bash
nextflow run main.nf \
  --input tests/fixtures/invalid_dapi_position.csv \
  --step preprocessing \
  --skip_nd2_conversion true \
  --outdir test_output
```

---

### Test Case 2.3: Valid - DAPI First (Conversion Disabled)
**Input CSV:**
```csv
patient_id,path_to_file,is_reference,channels
P001,/data/P001.ome.tiff,true,DAPI|PANCK|SMA  # ✅ DAPI first!
```

**Config:** `skip_nd2_conversion = true`

**Expected Result:** ✅ Pass validation

---

## Issue #3: Metadata Loss in Results Step

### Test Case 3.1: Resume from Results Step - Channels Preserved
**Checkpoint CSV (from postprocessing):**
```csv
patient_id,is_reference,channels,phenotype_csv,phenotype_mask,phenotype_mapping,merged_csv,cell_mask
P001,true,DAPI|PANCK|SMA,/path/pheno.csv,/path/mask.tiff,/path/map.json,/path/merged.csv,/path/cell.tiff
```

**Test Command:**
```bash
nextflow run main.nf \
  --input postprocessing_checkpoint.csv \
  --step results \
  --outdir test_output
```

**Validation:**
```groovy
// Check that metadata includes channels field
assert meta.channels == ['DAPI', 'PANCK', 'SMA']
```

---

## Issue #4: MERGE Reference Marker Conflict

### Test Case 4.1: MERGE Uses DAPI as Reference
**Setup:** Patient with multiple slides, reference has DAPI

**Expected Behavior:**
- MERGE script receives `--reference-markers DAPI`
- Script identifies reference slide by presence of DAPI channel
- Only DAPI from reference slide is retained in final merged image

**Validation:**
```bash
# Check merge_registered.py command line
grep "reference-markers DAPI" .nextflow/log
```

---

## Issue #6: Checkpoint CSV Validation

### Test Case 6.1: Valid Checkpoint CSV (Registration)
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/data/P001_preprocessed.ome.tiff,true,DAPI|PANCK|SMA
P001,/data/P001_mov1_preprocessed.ome.tiff,false,DAPI|PANCK|SMA
```

**Test Command:**
```bash
nextflow run main.nf \
  --input tests/fixtures/valid_checkpoint_registration.csv \
  --step registration \
  --outdir test_output
```

**Expected Result:** ✅ Pass validation

---

### Test Case 6.2: Invalid - Missing Required Column
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference
P001,/data/P001.tiff,true
```

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Checkpoint CSV validation failed for registration!

File: invalid_checkpoint.csv
Found columns: [patient_id, preprocessed_image, is_reference]

Errors:
  • Missing required column: 'channels'

💡 Expected columns: patient_id, preprocessed_image, is_reference, channels
```

---

### Test Case 6.3: Invalid - Malformed is_reference Value
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/data/P001.tiff,yes,DAPI|PANCK  # ❌ Should be 'true' or 'false'
```

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Checkpoint CSV data validation failed for registration!

File: invalid_checkpoint.csv
Total rows: 1 (excluding header)

Errors found:
  • Row 2: Invalid is_reference value 'yes' (must be 'true' or 'false')
```

---

### Test Case 6.4: Invalid - File Does Not Exist
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/nonexistent/file.tiff,true,DAPI|PANCK
```

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Checkpoint CSV data validation failed for registration!

Errors found:
  • Row 2: File not found: /nonexistent/file.tiff
```

---

### Test Case 6.5: Invalid - Missing DAPI Channel
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/data/P001.tiff,true,PANCK|SMA  # ❌ No DAPI!
```

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Checkpoint CSV data validation failed for registration!

Errors found:
  • Row 2: DAPI channel not found in 'PANCK|SMA'
```

---

### Test Case 6.6: Invalid - Multiple References Per Patient
**Checkpoint CSV:**
```csv
patient_id,preprocessed_image,is_reference,channels
P001,/data/P001_ref1.tiff,true,DAPI|PANCK
P001,/data/P001_ref2.tiff,true,DAPI|PANCK  # ❌ Second reference!
```

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Checkpoint CSV data validation failed for registration!

Errors found:
  • Patient 'P001': Multiple references (2 rows with is_reference=true)
```

---

## Issue #11: SPLIT_CHANNELS Output Cardinality

### Test Case 11.1: Valid - Correct Number of Channels
**Setup:**
- Input image has 3 channels: DAPI, PANCK, SMA
- Metadata: `meta.channels = ['DAPI', 'PANCK', 'SMA']`
- SPLIT_CHANNELS outputs 3 files

**Expected Result:** ✅ Pass - No errors

---

### Test Case 11.2: Invalid - Missing Channel File
**Setup:**
- Input image has 3 channels
- Metadata: `meta.channels = ['DAPI', 'PANCK', 'SMA']`
- SPLIT_CHANNELS outputs only 2 files (corrupted output)

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Channel count mismatch for P001!

Expected 3 channels (from metadata): [DAPI, PANCK, SMA]
Got 2 channel files from SPLIT_CHANNELS

💡 This indicates SPLIT_CHANNELS may have failed or produced corrupted output.
   Check SPLIT_CHANNELS logs for patient P001
```

---

### Test Case 11.3: Invalid - Empty Channel File
**Setup:**
- SPLIT_CHANNELS outputs 3 files, but one is empty (0 bytes)

**Expected Result:** ❌ Fail with error:
```
❌ CRITICAL: Empty channel file detected: P001_PANCK.tiff for patient P001
```

---

## Automated Test Script

```bash
#!/bin/bash
# tests/run_validation_tests.sh

set -e

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES="$TESTS_DIR/fixtures"
OUTPUT="$TESTS_DIR/output"

echo "=========================================="
echo "Running Pipeline Validation Test Suite"
echo "=========================================="

# Clean up previous test runs
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

# Test Case 1.1: Valid input
echo "[TEST 1.1] Valid input - one reference per patient"
nextflow run main.nf \
  --input "$FIXTURES/valid_input.csv" \
  --step preprocessing \
  --outdir "$OUTPUT/test_1_1" \
  --dry_run true
if [ $? -eq 0 ]; then
  echo "✅ PASS"
else
  echo "❌ FAIL"
  exit 1
fi

# Test Case 1.2: Multiple references
echo "[TEST 1.2] Invalid - multiple references per patient"
if nextflow run main.nf \
  --input "$FIXTURES/invalid_multi_ref.csv" \
  --step preprocessing \
  --outdir "$OUTPUT/test_1_2" 2>&1 | grep -q "Multiple reference images found"; then
  echo "✅ PASS - Correctly detected multiple references"
else
  echo "❌ FAIL - Did not detect multiple references"
  exit 1
fi

# Test Case 1.3: No references
echo "[TEST 1.3] Invalid - no reference per patient"
if nextflow run main.nf \
  --input "$FIXTURES/invalid_no_ref.csv" \
  --step preprocessing \
  --outdir "$OUTPUT/test_1_3" 2>&1 | grep -q "No reference image found"; then
  echo "✅ PASS - Correctly detected missing reference"
else
  echo "❌ FAIL - Did not detect missing reference"
  exit 1
fi

# Test Case 2.2: DAPI not first (skip conversion)
echo "[TEST 2.2] Invalid - DAPI not in channel 0 (skip_nd2_conversion=true)"
if nextflow run main.nf \
  --input "$FIXTURES/invalid_dapi_position.csv" \
  --step preprocessing \
  --skip_nd2_conversion true \
  --outdir "$OUTPUT/test_2_2" 2>&1 | grep -q "DAPI must be in channel 0"; then
  echo "✅ PASS - Correctly detected DAPI position error"
else
  echo "❌ FAIL - Did not detect DAPI position error"
  exit 1
fi

# Test Case 6.2: Missing column
echo "[TEST 6.2] Invalid checkpoint - missing required column"
if nextflow run main.nf \
  --input "$FIXTURES/invalid_checkpoint_missing_col.csv" \
  --step registration \
  --outdir "$OUTPUT/test_6_2" 2>&1 | grep -q "Missing required column"; then
  echo "✅ PASS - Correctly detected missing column"
else
  echo "❌ FAIL - Did not detect missing column"
  exit 1
fi

# Test Case 6.3: Invalid is_reference value
echo "[TEST 6.3] Invalid checkpoint - malformed is_reference"
if nextflow run main.nf \
  --input "$FIXTURES/invalid_checkpoint_bad_ref.csv" \
  --step registration \
  --outdir "$OUTPUT/test_6_3" 2>&1 | grep -q "Invalid is_reference value"; then
  echo "✅ PASS - Correctly detected invalid is_reference"
else
  echo "❌ FAIL - Did not detect invalid is_reference"
  exit 1
fi

echo "=========================================="
echo "All validation tests passed!"
echo "=========================================="
```

---

## Test Fixtures Creation

### Create valid_input.csv
```bash
cat > tests/fixtures/valid_input.csv << 'EOF'
patient_id,path_to_file,is_reference,channels
P001,tests/data/P001_ref.nd2,true,DAPI|PANCK|SMA
P001,tests/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
EOF
```

### Create invalid_multi_ref.csv
```bash
cat > tests/fixtures/invalid_multi_ref.csv << 'EOF'
patient_id,path_to_file,is_reference,channels
P001,tests/data/P001_ref1.nd2,true,DAPI|PANCK|SMA
P001,tests/data/P001_ref2.nd2,true,DAPI|PANCK|SMA
P001,tests/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
EOF
```

### Create invalid_no_ref.csv
```bash
cat > tests/fixtures/invalid_no_ref.csv << 'EOF'
patient_id,path_to_file,is_reference,channels
P001,tests/data/P001_mov1.nd2,false,DAPI|PANCK|SMA
P001,tests/data/P001_mov2.nd2,false,DAPI|PANCK|SMA
EOF
```

### Create invalid_dapi_position.csv
```bash
cat > tests/fixtures/invalid_dapi_position.csv << 'EOF'
patient_id,path_to_file,is_reference,channels
P001,tests/data/P001.ome.tiff,true,PANCK|DAPI|SMA
EOF
```

---

## Running the Tests

### Quick Validation (Dry Run)
```bash
# Uses --dry_run mode to validate inputs without executing
bash tests/run_validation_tests.sh
```

### Full Integration Tests
```bash
# Requires test data files
nextflow run main.nf \
  -profile test,docker \
  --outdir test_results
```

### Continuous Integration
```yaml
# .github/workflows/validation_tests.yml
name: Validation Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: nf-core/setup-nextflow@v2
      - name: Run validation tests
        run: bash tests/run_validation_tests.sh
```

---

## Expected Behavior Summary

| Test Case | Input | Expected Outcome |
|-----------|-------|------------------|
| 1.1 | Valid: 1 ref/patient | ✅ Pass |
| 1.2 | Invalid: 2+ refs/patient | ❌ Fail with clear error |
| 1.3 | Invalid: 0 refs/patient | ❌ Fail with clear error |
| 2.1 | DAPI anywhere (conversion on) | ✅ Pass (DAPI moved to ch0) |
| 2.2 | DAPI not ch0 (conversion off) | ❌ Fail with clear error |
| 2.3 | DAPI in ch0 (conversion off) | ✅ Pass |
| 3.1 | Resume from results | ✅ Channels preserved |
| 4.1 | MERGE with DAPI ref | ✅ Uses DAPI as marker |
| 6.1 | Valid checkpoint CSV | ✅ Pass |
| 6.2 | Missing column | ❌ Fail with clear error |
| 6.3 | Invalid is_reference | ❌ Fail with clear error |
| 6.4 | File not found | ❌ Fail with clear error |
| 6.5 | Missing DAPI | ❌ Fail with clear error |
| 6.6 | Multiple refs in CSV | ❌ Fail with clear error |
| 11.1 | Correct # channels | ✅ Pass |
| 11.2 | Missing channel file | ❌ Fail with clear error |
| 11.3 | Empty channel file | ❌ Fail with clear error |

---

## Maintenance

These test cases should be updated whenever:
1. New validation rules are added
2. Error messages are modified
3. New input formats are supported
4. Edge cases are discovered in production

Last Updated: 2025-12-27
