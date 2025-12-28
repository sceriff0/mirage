# ATEIA Pipeline Testing

This document provides an overview of the testing infrastructure for the ATEIA pipeline.

---

## Quick Links

- **Full Testing Guide**: [../tests/README.md](../tests/README.md)
- **Quick Reference**: [../tests/TESTING_QUICK_REFERENCE.md](../tests/TESTING_QUICK_REFERENCE.md)
- **Test Case Specifications**: [../tests/test_validation.md](../tests/test_validation.md)

---

## Test Coverage Summary

The ATEIA pipeline has comprehensive test coverage across multiple levels:

### ✅ Python Unit Tests (pytest)

**Coverage**: Script logic and utility functions

**Tests**:
- Preprocessing: FOV splitting, image reconstruction
- Segmentation: Label remapping, normalization
- Quantification: Cell intensity computation
- Registration: Alignment algorithms
- Phenotyping: Classification logic

**Run**: `pytest -v tests/`

### ✅ Nextflow Process Tests (nf-test)

**Coverage**: Individual process execution and outputs

**Tested Processes**:
- `SEGMENT`: Cell segmentation with DAPI validation
- `QUANTIFY`: Single-channel quantification
- `MERGE_QUANT_CSVS`: Multi-channel merging
- `SPLIT_CHANNELS`: Channel separation

**Run**: `nf-test test --profile test,docker`

### ✅ Pipeline Integration Tests (nf-test)

**Coverage**: End-to-end workflow execution

**Tested Workflows**:
- Preprocessing workflow (with test data)
- Registration workflow (from checkpoint)
- Parameter validation
- Checkpoint CSV handling

**Run**: `nextflow run . -profile test,docker -stub`

### ✅ Input Validation Tests (Bash)

**Coverage**: Error handling and edge cases

**Tested Scenarios**:
- Multiple references per patient ❌
- No reference per patient ❌
- DAPI not in channel 0 ❌
- Missing DAPI channel ❌
- Invalid checkpoint CSVs ❌
- File not found errors ❌
- Malformed parameters ❌

**Run**: `bash tests/run_validation_tests.sh`

---

## Test Infrastructure

### Test Data Generation

Automated test data generation creates:
- Multi-channel OME-TIFF images (3 channels, 128x128)
- Segmentation masks (10-20 labeled cells)
- Valid CSV files for all pipeline entry points
- Invalid CSV files for validation testing

**Generate**: `python tests/testdata/generate_complete_testdata.py`

### Stub Mode

All processes include stub implementations for fast testing:
- Creates empty output files
- Validates workflow structure
- Completes in seconds vs hours
- Ideal for development iteration

**Use**: Add `-stub` flag to any test command

### CI/CD Integration

Automated testing on GitHub Actions:
- **Python Tests**: Matrix testing across Python 3.9, 3.10, 3.11
- **Nextflow Tests**: Matrix testing across Nextflow 23.04.0 and latest
- **Validation Tests**: All edge cases tested
- **Linting**: nf-core standards compliance

**Status**: Check Actions tab on GitHub

---

## Running Tests

### Quick Start

```bash
# One-time setup
pip install pytest numpy tifffile pandas scikit-image
python tests/testdata/generate_complete_testdata.py

# Run all tests
pytest -v tests/
nf-test test --profile test,docker -stub
bash tests/run_validation_tests.sh
nextflow run . -profile test,docker -stub
```

### Development Workflow

```bash
# During development (fast iteration)
nf-test test tests/modules/my_process.nf.test -stub

# Before commit (full validation)
pytest -v tests/
bash tests/run_validation_tests.sh
nextflow run . -profile test,docker -stub
```

---

## Test Organization

```
tests/
├── README.md                      # Comprehensive testing guide
├── TESTING_QUICK_REFERENCE.md     # Quick command reference
├── test_validation.md             # Test case specifications
│
├── run_validation_tests.sh        # Automated validation testing
├── main.nf.test                   # Pipeline integration tests
│
├── modules/                       # Process-level nf-tests
│   ├── segment.nf.test
│   ├── quantify.nf.test
│   └── split_channels.nf.test
│
├── testdata/                      # Test data
│   ├── generate_complete_testdata.py
│   ├── *.ome.tiff                 # Test images
│   ├── *.npy                      # Test masks
│   ├── valid_*.csv                # Valid inputs
│   └── invalid_*.csv              # Invalid inputs
│
└── test_*.py                      # Python unit tests
    ├── test_preprocess.py
    ├── test_segment.py
    ├── test_quantify.py
    ├── test_register.py
    └── test_phenotype.py
```

---

## Test Quality Metrics

| Metric | Status | Details |
|--------|--------|---------|
| Python Unit Tests | ✅ High Coverage | All core functions tested |
| Process Tests | ✅ Critical Paths | Main processes covered |
| Integration Tests | ✅ End-to-End | Full workflows tested |
| Validation Tests | ✅ Comprehensive | All error paths covered |
| Stub Implementations | ✅ 100% | All processes have stubs |
| CI/CD Integration | ✅ Automated | Tests run on every commit |
| Test Data | ✅ Generated | Reproducible synthetic data |
| Documentation | ✅ Complete | Multiple guides available |

---

## Best Practices

### Writing Tests

1. **Start with stub mode** - Validate logic before full execution
2. **Test edge cases** - Invalid inputs, missing files, edge conditions
3. **Keep tests fast** - Use minimal test data (128x128 images)
4. **Make tests independent** - No shared state between tests
5. **Use descriptive names** - Clearly indicate what is being tested
6. **Verify outputs** - Check files exist and contain expected data

### Test Data

- **Size**: Keep images small (128x128 pixels maximum)
- **Realism**: Use realistic cell-like structures
- **Variety**: Cover single/multi-channel, single/multi-slide cases
- **Invalid Cases**: Create comprehensive invalid inputs for validation

### CI/CD

- **Speed**: Tests complete in < 10 minutes
- **Reliability**: Use stub mode to avoid flaky tests
- **Coverage**: Test on multiple Python/Nextflow versions
- **Artifacts**: Upload test outputs for debugging

---

## Troubleshooting

### Common Issues

```bash
# Test data not found
python tests/testdata/generate_complete_testdata.py

# nf-test not installed
curl -fsSL https://code.askimed.com/install/nf-test | bash
sudo mv nf-test /usr/local/bin/

# Docker not running
# Start Docker or use: -profile test (without docker)

# Process failed in test
# Use stub mode: nf-test test -stub
```

### Getting Help

1. Check [tests/README.md](../tests/README.md) for detailed information
2. Review test logs in `tests/output/` or `.nf-test/`
3. Consult [test_validation.md](../tests/test_validation.md) for test specifications
4. Open an issue with error logs and test command

---

## Continuous Improvement

The testing infrastructure is continuously improved:

- ✅ Added nf-test framework
- ✅ Created comprehensive test data
- ✅ Implemented validation testing
- ✅ Integrated with CI/CD
- ✅ Documented all test types

### Future Enhancements

- [ ] Add performance benchmarking tests
- [ ] Expand process test coverage to all modules
- [ ] Add integration tests for all checkpoint entry points
- [ ] Implement snapshot testing for outputs
- [ ] Add test coverage reporting

---

## Contributing

When contributing to ATEIA:

1. **Write tests** for new features
2. **Update test data** if input schemas change
3. **Run all tests** before submitting PR
4. **Check CI/CD** passes on GitHub
5. **Update documentation** if test procedures change

### Pre-Commit Checklist

```bash
# Generate test data
python tests/testdata/generate_complete_testdata.py

# Run tests
pytest -v tests/
nf-test test -stub
bash tests/run_validation_tests.sh
nextflow run . -profile test,docker -stub

# All pass? Commit!
git add .
git commit -m "Your changes"
```

---

## Resources

- [nf-test Documentation](https://code.askimed.com/nf-test/)
- [nf-core Testing Guidelines](https://nf-co.re/docs/contributing/guidelines)
- [Pytest Documentation](https://docs.pytest.org/)
- [Nextflow Testing Docs](https://www.nextflow.io/docs/latest/testing.html)

---

**For detailed testing instructions**, see the full [Testing Guide](../tests/README.md).

**For quick command reference**, see the [Quick Reference](../tests/TESTING_QUICK_REFERENCE.md).
