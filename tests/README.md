# ATEIA Pipeline Testing Infrastructure

Comprehensive testing suite for the ATEIA (Automated Tissue Exploration and Image Analysis) pipeline following nf-core best practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Types](#test-types)
3. [Quick Start](#quick-start)
4. [Test Data](#test-data)
5. [Running Tests](#running-tests)
6. [CI/CD Integration](#cicd-integration)
7. [Writing New Tests](#writing-new-tests)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The ATEIA testing infrastructure includes:

- ✅ **Python Unit Tests** - Test individual Python scripts (pytest)
- ✅ **Nextflow Process Tests** - Test individual processes (nf-test)
- ✅ **Pipeline Integration Tests** - Test full pipeline workflows (nf-test)
- ✅ **Validation Tests** - Test input validation and error handling
- ✅ **Stub Mode Tests** - Fast testing without running actual tools
- ✅ **CI/CD Integration** - Automated testing on GitHub Actions

### Test Coverage

| Component | Test Type | Coverage | Location |
|-----------|-----------|----------|----------|
| Python Scripts | Unit tests (pytest) | ✅ High | `tests/test_*.py` |
| Nextflow Processes | Process tests (nf-test) | ✅ Critical processes | `tests/modules/*.nf.test` |
| Full Pipeline | Integration tests (nf-test) | ✅ Main workflows | `tests/main.nf.test` |
| Input Validation | Validation tests (bash) | ✅ Edge cases | `tests/run_validation_tests.sh` |
| Error Handling | Validation tests | ✅ All error paths | `tests/test_validation.md` |

---

## Test Types

### 1. Python Unit Tests

**Purpose**: Test individual Python script functions in isolation

**Technology**: pytest

**Location**: `tests/test_*.py`

**Example**:
```python
# tests/test_quantify.py
def test_compute_cell_intensities_simple():
    mask = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    channel = np.array([[0, 5, 7], [0, 3, 9]], dtype=np.float32)
    df = compute_cell_intensities(mask, channel, "TEST", min_area=0)
    assert "TEST" in df.columns
```

**Run**:
```bash
pytest -v tests/
```

### 2. Nextflow Process Tests (nf-test)

**Purpose**: Test individual Nextflow processes with realistic inputs

**Technology**: nf-test

**Location**: `tests/modules/*.nf.test`

**Example**:
```groovy
// tests/modules/segment.nf.test
test("Should segment cells from merged image - stub") {
    options "-stub"
    when {
        process {
            """
            input[0] = [
                [patient_id: 'P001', channels: ['DAPI', 'PANCK', 'SMA']],
                file('tests/testdata/P001_ref.ome.tiff')
            ]
            """
        }
    }
    then {
        assert process.success
        assert process.out.nuclei_mask
        assert process.out.cell_mask
    }
}
```

**Run**:
```bash
nf-test test tests/modules/segment.nf.test
```

### 3. Pipeline Integration Tests

**Purpose**: Test complete pipeline workflows end-to-end

**Technology**: nf-test

**Location**: `tests/main.nf.test`

**Run**:
```bash
nf-test test tests/main.nf.test
```

### 4. Input Validation Tests

**Purpose**: Test input validation, error messages, and edge cases

**Technology**: Bash script with Nextflow dry-run mode

**Location**: `tests/run_validation_tests.sh`

**Run**:
```bash
bash tests/run_validation_tests.sh
```

**Tests Covered**:
- Multiple references per patient ❌
- No reference per patient ❌
- DAPI not in channel 0 ❌
- Missing DAPI channel ❌
- Invalid checkpoint CSVs ❌
- File not found ❌
- Malformed parameters ❌

---

## Quick Start

### Prerequisites

```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# Install nf-test
curl -fsSL https://code.askimed.com/install/nf-test | bash
sudo mv nf-test /usr/local/bin/

# Install Python dependencies
pip install pytest numpy tifffile pandas scikit-image
```

### Generate Test Data

```bash
python tests/testdata/generate_complete_testdata.py
```

This creates:
- Multi-channel OME-TIFF images (128x128, 3 channels)
- Segmentation masks
- Valid and invalid CSV files for testing

### Run All Tests

```bash
# Python unit tests
pytest -v tests/

# Nextflow process tests
nf-test test --profile test,docker

# Validation tests
bash tests/run_validation_tests.sh

# Full pipeline (stub mode - fast)
nextflow run . -profile test,docker -stub
```

---

## Test Data

### Location
`tests/testdata/`

### Files Generated

#### Valid Test Data
```
tests/testdata/
├── P001_ref.ome.tiff              # Reference image (3 channels, 128x128)
├── P001_mov1.ome.tiff             # Moving image 1
├── P001_mov2.ome.tiff             # Moving image 2
├── P002_ref.ome.tiff              # Single-slide patient
├── P001_cell_mask.npy             # Segmentation mask (20 cells)
├── P002_cell_mask.npy             # Segmentation mask (15 cells)
├── valid_preprocessing.csv        # Valid input for preprocessing
├── valid_checkpoint_registration.csv  # Valid checkpoint for registration
└── test_input.csv                 # Test profile input
```

#### Invalid Test Data (for validation testing)
```
tests/testdata/
├── invalid_multi_ref.csv          # Multiple references per patient
├── invalid_no_ref.csv             # No reference per patient
├── invalid_dapi_position.csv      # DAPI not in channel 0
├── invalid_no_dapi.csv            # Missing DAPI channel
├── invalid_checkpoint_missing_col.csv  # Missing required column
├── invalid_checkpoint_bad_ref.csv # Malformed is_reference value
└── invalid_file_not_found.csv     # File does not exist
```

### Regenerating Test Data

Test data is automatically generated but can be regenerated:

```bash
python tests/testdata/generate_complete_testdata.py
```

---

## Running Tests

### Local Development

#### Run Python Unit Tests
```bash
# All tests
pytest -v tests/

# Specific test file
pytest -v tests/test_quantify.py

# With coverage
pytest -v tests/ --cov=bin --cov-report=html
```

#### Run nf-test (Process Tests)
```bash
# All nf-tests
nf-test test --profile test,docker

# Specific process
nf-test test tests/modules/segment.nf.test

# In stub mode (faster)
nf-test test --profile test,docker -stub
```

#### Run Validation Tests
```bash
# All validation tests
bash tests/run_validation_tests.sh

# Output logs to tests/output/validation/
```

#### Run Full Pipeline
```bash
# Stub mode (fast, creates empty output files)
nextflow run . -profile test,docker -stub --outdir test_results

# Full run with test data (requires all dependencies)
nextflow run . -profile test,docker --outdir test_results
```

### Test Profiles

Available in `conf/test.config`:

```groovy
params {
    max_cpus   = 2
    max_memory = '6.GB'
    max_time   = '1.h'

    seg_gpu    = false

    input = 'tests/testdata/test_input.csv'
}
```

### Stub Mode

All processes have `stub:` blocks for fast testing:

```bash
# Run with -stub flag to use stub implementations
nextflow run . -profile test,docker -stub
```

Stub mode:
- Creates empty output files
- Skips actual computations
- Validates workflow logic
- Completes in seconds instead of hours

---

## CI/CD Integration

### GitHub Actions Workflow

Location: `.github/workflows/ci.yml`

### Test Jobs

1. **python-tests** (Matrix: Python 3.9, 3.10, 3.11)
   - Runs pytest unit tests
   - Uploads coverage to Codecov

2. **nextflow-tests** (Matrix: Nextflow 23.04.0, latest)
   - Generates test data
   - Runs validation tests
   - Runs pipeline in stub mode
   - Runs nf-test suite
   - Uploads test artifacts

3. **lint**
   - Runs nf-core lint
   - Checks code style

### Triggers

- Push to `main` branch
- Pull requests to `main`

### View Results

- ✅ Green checkmark: All tests passed
- ❌ Red X: Tests failed (click for details)
- 📊 Artifacts: Download test outputs

---

## Writing New Tests

### Adding a Python Unit Test

1. Create or edit `tests/test_<module>.py`:

```python
import numpy as np
from scripts.my_module import my_function

def test_my_function():
    """Test my_function with valid input"""
    result = my_function(input_data)
    assert result == expected_output
```

2. Run: `pytest -v tests/test_<module>.py`

### Adding an nf-test for a Process

1. Create `tests/modules/<process_name>.nf.test`:

```groovy
nextflow_process {
    name "Test MY_PROCESS"
    script "modules/local/my_process.nf"
    process "MY_PROCESS"

    test("Should process input correctly - stub") {
        options "-stub"

        when {
            process {
                """
                input[0] = [
                    [id: 'test'],
                    file('tests/testdata/input.tiff')
                ]
                """
            }
        }

        then {
            assert process.success
            assert process.out.output_file
        }
    }
}
```

2. Run: `nf-test test tests/modules/<process_name>.nf.test`

### Adding a Validation Test

1. Edit `tests/run_validation_tests.sh`:

```bash
run_test \
    "My new validation test" \
    "fail" \
    "$TESTDATA_DIR/my_invalid_input.csv" \
    --step preprocessing \
    "Expected error message"
```

2. Create test fixture: `tests/testdata/my_invalid_input.csv`

3. Run: `bash tests/run_validation_tests.sh`

---

## Troubleshooting

### Common Issues

#### 1. "Test data not found"

**Solution**: Generate test data
```bash
python tests/testdata/generate_complete_testdata.py
```

#### 2. "nf-test: command not found"

**Solution**: Install nf-test
```bash
curl -fsSL https://code.askimed.com/install/nf-test | bash
sudo mv nf-test /usr/local/bin/
```

#### 3. "Docker daemon not running"

**Solution**: Start Docker or use `-profile test` without docker
```bash
nextflow run . -profile test -stub
```

#### 4. "Process failed in nf-test"

**Solution**: Check if running in stub mode
```bash
nf-test test --profile test,docker -stub
```

#### 5. "Import errors in Python tests"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
pip install pytest numpy tifffile pandas scikit-image
```

### Debug Mode

#### Nextflow
```bash
nextflow run . -profile test,docker -stub --debug
```

#### nf-test
```bash
nf-test test --verbose --debug
```

#### Pytest
```bash
pytest -v -s tests/  # -s shows print statements
```

### View Test Logs

- **Nextflow**: `.nextflow.log`
- **nf-test**: `.nf-test/tests/*/output.log`
- **Validation**: `tests/output/validation/test_*.log`

---

## Best Practices

### When Writing Tests

1. **Use stub mode first** - Validate workflow logic before full runs
2. **Test edge cases** - Empty inputs, invalid formats, missing files
3. **Keep tests fast** - Use small test data (128x128 images)
4. **Make tests independent** - Each test should run in isolation
5. **Use descriptive names** - `test_should_fail_with_invalid_dapi_position`
6. **Assert outputs exist** - Check files were created
7. **Validate metadata** - Ensure meta maps are preserved

### Test Data Guidelines

- Images: 128x128 pixels maximum
- Channels: 1-3 channels
- Cells: 10-20 cells per mask
- File formats: OME-TIFF (real) or TIFF (simple)

### CI/CD Guidelines

- Tests must complete in < 10 minutes
- Use `-stub` mode for full pipeline tests
- Upload artifacts for debugging failed tests
- Don't fail on lint warnings (informational only)

---

## Testing Checklist

Before submitting a PR:

- [ ] Python unit tests pass: `pytest -v tests/`
- [ ] nf-tests pass: `nf-test test --profile test,docker -stub`
- [ ] Validation tests pass: `bash tests/run_validation_tests.sh`
- [ ] Pipeline runs in stub mode: `nextflow run . -profile test,docker -stub`
- [ ] New code has tests
- [ ] Test data regenerated if schemas changed
- [ ] CI/CD passes on GitHub

---

## Additional Resources

- [nf-test Documentation](https://code.askimed.com/nf-test/)
- [nf-core Best Practices](https://nf-co.re/docs/contributing/guidelines)
- [Pytest Documentation](https://docs.pytest.org/)
- [Nextflow Testing](https://www.nextflow.io/docs/latest/testing.html)

---

## Support

For issues with tests:

1. Check this README
2. Review [test_validation.md](test_validation.md) for test case details
3. Check logs in `tests/output/` and `.nf-test/`
4. Open an issue on GitHub with:
   - Test command run
   - Error message
   - Relevant logs

---

**Last Updated**: 2025-12-28

**Test Infrastructure Version**: 1.0.0
