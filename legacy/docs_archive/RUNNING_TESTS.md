# Running Tests in MIRAGE Pipeline

This guide explains how to run tests for the MIRAGE WSI processing pipeline using both the Nextflow test profile and the nf-test framework, following [nf-core best practices](https://nf-co.re/docs/guidelines/pipelines/recommendations/testing).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Framework Overview](#test-framework-overview)
3. [Running Pipeline Tests](#running-pipeline-tests)
4. [Running nf-test Suites](#running-nf-test-suites)
5. [Test Profiles](#test-profiles)
6. [Generating Snapshots](#generating-snapshots)
7. [CI/CD Integration](#cicd-integration)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Install Nextflow** (≥23.04.0):
   ```bash
   curl -s https://get.nextflow.io | bash
   sudo mv nextflow /usr/local/bin/
   ```

2. **Install nf-test**:
   ```bash
   curl -fsSL https://get.nf-test.com | bash
   sudo mv nf-test /usr/local/bin/
   ```

3. **Install Docker** (recommended) or Singularity:
   - [Docker installation guide](https://docs.docker.com/get-docker/)
   - [Singularity installation guide](https://sylabs.io/guides/latest/user-guide/)

### Run Quick Test

```bash
# Run the full pipeline with test profile (fastest)
nextflow run main.nf -profile test,docker

# Or run nf-test suite (comprehensive)
nf-test test
```

---

## Test Framework Overview

The MIRAGE pipeline uses a two-tiered testing approach following nf-core guidelines:

### 1. **Nextflow Test Profile** (`-profile test`)
- **Purpose**: Quick end-to-end validation with minimal data
- **Duration**: ~5-10 minutes
- **Use case**: Fast feedback during development, CI/CD

### 2. **nf-test Framework**
- **Purpose**: Comprehensive testing of modules, subworkflows, and pipeline
- **Duration**: ~10-30 minutes (full suite)
- **Use case**: Detailed validation, snapshot testing, regression prevention

**Reference**:
- [nf-test documentation](https://www.nf-test.com/)
- [nf-core testing guidelines](https://nf-co.re/docs/contributing/nf-test/assertions)

---

## Running Pipeline Tests

### Method 1: Using Test Profile (Recommended for Quick Validation)

#### Basic Test Run

```bash
# Run with Docker
nextflow run main.nf -profile test,docker

# Run with Singularity
nextflow run main.nf -profile test,singularity

# Run with Conda
nextflow run main.nf -profile test,conda
```

#### Full Test (Larger Dataset)

```bash
# Comprehensive testing with more data
nextflow run main.nf -profile test_full,docker
```

#### Test Specific Steps

```bash
# Test preprocessing only
nextflow run main.nf -profile test,docker --step preprocessing

# Test registration from checkpoint
nextflow run main.nf -profile test,docker \
    --step registration \
    --input tests/testdata/valid_checkpoint_registration.csv

# Test postprocessing
nextflow run main.nf -profile test,docker \
    --step postprocessing \
    --input tests/testdata/valid_checkpoint_postprocessing.csv
```

#### Test Different Registration Methods

```bash
# Test VALIS registration
nextflow run main.nf -profile test,docker \
    --step registration \
    --registration_method valis

# Test GPU registration (requires GPU)
nextflow run main.nf -profile test,docker \
    --step registration \
    --registration_method gpu

# Test CPU registration
nextflow run main.nf -profile test,docker \
    --step registration \
    --registration_method cpu
```

---

## Running nf-test Suites

nf-test provides fine-grained testing capabilities for modules, subworkflows, and pipelines with snapshot support.

### Basic Usage

```bash
# Run all tests
nf-test test

# Run with verbose output
nf-test test --verbose

# Run specific test file
nf-test test tests/modules/segment.nf.test

# Run tests matching a pattern
nf-test test --tag modules
nf-test test --tag subworkflow
nf-test test --tag pipeline
```

### Test by Component Type

```bash
# Test all modules
nf-test test tests/modules/

# Test all subworkflows
nf-test test tests/subworkflows/

# Test main pipeline
nf-test test tests/main.nf.test
```

### Run Specific Tests

```bash
# Test specific module
nf-test test tests/modules/segment.nf.test
nf-test test tests/modules/quantify.nf.test
nf-test test tests/modules/preprocess.nf.test

# Test specific subworkflow
nf-test test tests/subworkflows/local/preprocessing.nf.test
nf-test test tests/subworkflows/local/registration.nf.test
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
nf-test test --parallel

# Specify number of threads
nf-test test --parallel --threads 8
```

### Debug Mode

```bash
# Run with debug output
nf-test test --profile debug --verbose

# Keep work directories for inspection
nf-test test --no-cleanup
```

---

## Test Profiles

Test profiles are defined in `conf/test.config` and `conf/test_full.config`.

### Test Profile (Minimal - Fast)

**File**: `conf/test.config`

**Characteristics**:
- Resource limits: 2 CPUs, 6 GB RAM, 1 hour
- Minimal test dataset (2 patients, 3 images)
- Reduced computational parameters
- GPU disabled
- Estimated time: ~5-10 minutes

**Use cases**:
- Quick validation during development
- CI/CD pipelines
- Smoke testing

```bash
nextflow run main.nf -profile test,docker
```

### Test Full Profile (Comprehensive)

**File**: `conf/test_full.config`

**Characteristics**:
- Resource limits: 8 CPUs, 32 GB RAM, 6 hours
- Full test dataset (multi-patient)
- Realistic parameters
- Optional GPU support
- Padding enabled
- Error estimation enabled
- Estimated time: ~1-2 hours

**Use cases**:
- Comprehensive validation before release
- Performance testing
- Integration testing

```bash
nextflow run main.nf -profile test_full,docker
```

### Comparison Table

| Feature | test | test_full |
|---------|------|-----------|
| CPUs | 2 | 8 |
| Memory | 6 GB | 32 GB |
| Time limit | 1 hour | 6 hours |
| Dataset size | Minimal | Full |
| Patients | 1 | 2+ |
| Images per patient | 2 | 3+ |
| Padding | Disabled | Enabled |
| QC generation | Minimal (0.1x) | Full (0.25x) |
| Error estimation | Disabled | Enabled |
| Duration | ~5-10 min | ~1-2 hours |

---

## Generating Snapshots

Snapshots capture expected outputs for regression testing. See [nf-test snapshot documentation](https://www.nf-test.com/docs/assertions/snapshots/).

### Initial Snapshot Creation

```bash
# Generate snapshots for all tests
nf-test test --update-snapshot

# Generate for specific test
nf-test test tests/modules/segment.nf.test --update-snapshot
```

### Updating Snapshots

When module output changes (e.g., after version updates):

```bash
# Review changes first
nf-test test tests/modules/preprocess.nf.test

# If changes are expected, update snapshot
nf-test test tests/modules/preprocess.nf.test --update-snapshot
```

### Snapshot Best Practices

1. **Always inspect snapshots manually** after generation:
   ```bash
   cat tests/modules/segment.nf.test.snap
   ```

2. **Check for empty files** (md5sum: d41d8cd98f00b204e9800998ecf8427e):
   - Empty files indicate potential issues
   - Exception: stub tests may have empty outputs

3. **Version control**: Commit `.nf.test.snap` files to Git

4. **Review changes**: Use `git diff` to review snapshot changes before committing

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/ci.yml
name: nf-core CI

on:
  push:
    branches: [main, dev]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        NXF_VER:
          - '23.04.0'
          - 'latest-everything'
        profile:
          - 'test,docker'
    steps:
      - uses: actions/checkout@v4

      - uses: nf-core/setup-nextflow@v2
        with:
          version: ${{ matrix.NXF_VER }}

      - name: Install nf-test
        run: |
          curl -fsSL https://get.nf-test.com | bash
          sudo mv nf-test /usr/local/bin/

      - name: Run nf-test
        run: nf-test test --profile ${{ matrix.profile }}

      - name: Run pipeline test
        run: nextflow run main.nf -profile ${{ matrix.profile }}
```

### GitLab CI Example

```yaml
# .gitlab-ci.yml
test:
  image: nextflow/nextflow:23.04.0
  script:
    - curl -fsSL https://get.nf-test.com | bash
    - ./nf-test test --profile test,docker
    - nextflow run main.nf -profile test,docker
  only:
    - branches
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Permission Denied

**Error**: `permission denied while trying to connect to Docker daemon`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for CI)
sudo nextflow run main.nf -profile test,docker
```

#### 2. Nextflow Version Too Old

**Error**: `nextflow.enable.dsl requires version >=23.04.0`

**Solution**:
```bash
# Update Nextflow
nextflow self-update

# Or install specific version
export NXF_VER=23.04.0
curl -s https://get.nextflow.io | bash
```

#### 3. Test Data Not Found

**Error**: `No such file or directory: tests/testdata/test_input.csv`

**Solution**:
```bash
# Generate test data
cd tests/testdata
python3 generate_complete_testdata.py
cd ../..
```

#### 4. Out of Memory

**Error**: `Process exceeded available memory`

**Solution**:
```bash
# Reduce resource requirements
nextflow run main.nf -profile test,docker \
    --max_memory 4.GB \
    --preproc_tile_size 128
```

#### 5. nf-test Not Found

**Error**: `command not found: nf-test`

**Solution**:
```bash
# Install nf-test
curl -fsSL https://get.nf-test.com | bash
sudo mv nf-test /usr/local/bin/

# Verify installation
nf-test version
```

#### 6. Snapshot Mismatch

**Error**: `Snapshot mismatch detected`

**Cause**: Output changed since snapshot was created

**Solution**:
```bash
# Review the changes
nf-test test tests/modules/segment.nf.test --verbose

# If changes are expected (e.g., after module update)
nf-test test tests/modules/segment.nf.test --update-snapshot

# Always inspect the updated snapshot
cat tests/modules/segment.nf.test.snap
```

---

## Advanced Testing

### Test with Specific Parameters

```bash
# Test with custom preprocessing parameters
nextflow run main.nf -profile test,docker \
    --preproc_tile_size 512 \
    --preproc_n_iter 3

# Test with different segmentation settings
nextflow run main.nf -profile test,docker \
    --seg_n_tiles_y 4 \
    --seg_n_tiles_x 4
```

### Test with Work Directory Preservation

```bash
# Keep work directory for debugging
nextflow run main.nf -profile test,docker --outdir test_results -work-dir work_test

# Inspect failed task
ls -lh work_test/*/*/
cat work_test/*/.command.sh
cat work_test/*/.command.log
```

### Test Cleanup

```bash
# Remove test outputs
rm -rf test_results test_results_full .nextflow* work .nf-test

# Clean Docker containers/images
docker system prune -af
```

---

## Best Practices

1. **Run tests frequently**: Before committing, after merging, before releases

2. **Use appropriate profile**:
   - Development: `-profile test,docker` (fast)
   - Pre-release: `-profile test_full,docker` (comprehensive)

3. **Check test coverage**: Aim for >80% module coverage

4. **Update snapshots carefully**: Always inspect changes

5. **Use CI/CD**: Automate testing on every push/PR

6. **Document test failures**: Include logs when reporting issues

7. **Test on multiple platforms**: Docker, Singularity, Conda

---

## Resources

- [nf-test Documentation](https://www.nf-test.com/)
- [nf-core Testing Guidelines](https://nf-co.re/docs/guidelines/pipelines/recommendations/testing)
- [nf-test Assertions Examples](https://nf-co.re/docs/contributing/nf-test/assertions)
- [Nextflow Testing](https://training.nextflow.io/2.1.3/side_quests/nf-test/)
- [nf-core fetchngs (best practice example)](https://github.com/nf-core/fetchngs)

---

## Quick Reference

```bash
# Test the pipeline
nextflow run main.nf -profile test,docker                    # Quick test
nextflow run main.nf -profile test_full,docker               # Full test

# Run nf-test
nf-test test                                                 # All tests
nf-test test --tag modules                                   # Module tests only
nf-test test tests/modules/segment.nf.test                   # Specific module
nf-test test --update-snapshot                               # Update snapshots

# Generate test data
python3 tests/testdata/generate_complete_testdata.py         # Create test fixtures

# Cleanup
rm -rf test_results work .nextflow* .nf-test                # Remove test artifacts
```

---

**For questions or issues**, please consult the [main documentation](../README.md) or open an issue on GitHub.
