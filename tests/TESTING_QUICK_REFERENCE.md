# Testing Quick Reference

Fast reference for common testing tasks.

---

## Setup (One-Time)

```bash
# Install dependencies
pip install pytest numpy tifffile pandas scikit-image
curl -s https://get.nextflow.io | bash && sudo mv nextflow /usr/local/bin/
curl -fsSL https://code.askimed.com/install/nf-test | bash && sudo mv nf-test /usr/local/bin/

# Generate test data
python tests/testdata/generate_complete_testdata.py
```

---

## Run All Tests

```bash
# Complete test suite (recommended before PR)
pytest -v tests/                                    # Python tests
nf-test test --profile test,docker -stub            # Process tests
bash tests/run_validation_tests.sh                  # Validation tests
nextflow run . -profile test,docker -stub            # Full pipeline
```

---

## Quick Tests (Development)

```bash
# Fast iteration during development
pytest -v tests/test_<module>.py                    # Single Python test
nf-test test tests/modules/<process>.nf.test -stub  # Single process test
nextflow run . -profile test,docker -stub            # Pipeline stub run
```

---

## Test Commands by Type

### Python Unit Tests
```bash
pytest -v tests/                      # All tests
pytest -v tests/test_quantify.py      # Single file
pytest -v -k "test_compute"           # Tests matching pattern
pytest -v --cov=bin                   # With coverage
```

### Nextflow Process Tests (nf-test)
```bash
nf-test test                                        # All tests
nf-test test tests/modules/segment.nf.test         # Single process
nf-test test --profile test,docker                 # With profile
nf-test test -stub                                  # Stub mode (fast)
```

### Pipeline Integration Tests
```bash
nextflow run . -profile test,docker -stub           # Stub mode
nextflow run . -profile test,docker                 # Full run
nextflow run . -profile test,docker --step registration  # Specific step
```

### Validation Tests
```bash
bash tests/run_validation_tests.sh                 # All validation tests
# Logs: tests/output/validation/test_*.log
```

---

## Common Workflows

### Testing a New Process

1. Add stub block to process:
```groovy
stub:
def prefix = task.ext.prefix ?: "${meta.id}"
"""
touch ${prefix}_output.tiff
echo 'versions' > versions.yml
"""
```

2. Create nf-test:
```groovy
// tests/modules/my_process.nf.test
nextflow_process {
    name "Test MY_PROCESS"
    script "modules/local/my_process.nf"
    process "MY_PROCESS"

    test("Should work - stub") {
        options "-stub"
        when {
            process {
                """
                input[0] = [[id: 'test'], file('tests/testdata/input.tiff')]
                """
            }
        }
        then {
            assert process.success
        }
    }
}
```

3. Run test:
```bash
nf-test test tests/modules/my_process.nf.test -stub
```

### Testing Input Validation

1. Create invalid CSV:
```csv
# tests/testdata/invalid_my_case.csv
patient_id,path_to_file,is_reference,channels
P001,file.tiff,INVALID,DAPI|PANCK
```

2. Add to validation script:
```bash
# tests/run_validation_tests.sh
run_test \
    "Invalid my case" \
    "fail" \
    "$TESTDATA_DIR/invalid_my_case.csv" \
    --step preprocessing \
    "Expected error text"
```

3. Run:
```bash
bash tests/run_validation_tests.sh
```

---

## Debugging Failed Tests

### Python Tests
```bash
pytest -v -s tests/test_<module>.py     # Show print() output
pytest -v --pdb tests/                  # Drop to debugger on failure
```

### nf-test
```bash
nf-test test --verbose --debug tests/modules/<process>.nf.test
# Check: .nf-test/tests/*/output.log
```

### Pipeline
```bash
nextflow run . -profile test,docker -stub --debug
# Check: .nextflow.log
```

### Validation Tests
```bash
bash tests/run_validation_tests.sh
# Check: tests/output/validation/test_*.log
```

---

## CI/CD Status

### View in GitHub
- Go to repository → Actions tab
- See test results for each commit/PR

### Locally Simulate CI
```bash
# Python tests (what CI runs)
python tests/testdata/generate_complete_testdata.py
pytest -v tests/ --ignore=tests/testdata

# Nextflow tests (what CI runs)
bash tests/run_validation_tests.sh
nextflow run . -profile test,docker -stub --outdir test_results_stub
nf-test test --profile test,docker
```

---

## Test Data

### Location
`tests/testdata/`

### Regenerate
```bash
python tests/testdata/generate_complete_testdata.py
```

### Files Created
- **Valid**: P001_ref.ome.tiff, P001_mov1.ome.tiff, valid_preprocessing.csv
- **Invalid**: invalid_multi_ref.csv, invalid_no_ref.csv, etc.

---

## Checklist Before Commit

```bash
# 1. Generate test data
python tests/testdata/generate_complete_testdata.py

# 2. Run tests
pytest -v tests/
nf-test test -stub
bash tests/run_validation_tests.sh

# 3. Quick pipeline check
nextflow run . -profile test,docker -stub

# 4. Commit if all pass
git add .
git commit -m "Your message"
git push
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "Test data not found" | `python tests/testdata/generate_complete_testdata.py` |
| "nf-test: command not found" | Install nf-test (see Setup) |
| "Docker not running" | Start Docker or use `-profile test` |
| "Process failed" | Add `-stub` flag for fast testing |
| "Import error" | `pip install -r requirements.txt` |

---

## Performance Tips

- Use `-stub` during development (10x faster)
- Test single processes before full pipeline
- Use small test data (128x128 images)
- Run validation tests in dry_run mode
- Cache Docker images

---

## Key Files

```
tests/
├── README.md                           # Full documentation
├── TESTING_QUICK_REFERENCE.md          # This file
├── test_validation.md                  # Test case specifications
├── run_validation_tests.sh             # Validation test runner
├── main.nf.test                        # Pipeline integration tests
├── modules/                            # Process-level tests
│   ├── segment.nf.test
│   ├── quantify.nf.test
│   └── split_channels.nf.test
├── testdata/                           # Test data
│   └── generate_complete_testdata.py
└── test_*.py                           # Python unit tests
```

---

**For detailed information**, see [tests/README.md](README.md)
