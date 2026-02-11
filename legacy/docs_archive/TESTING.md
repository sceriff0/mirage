# MIRAGE Pipeline Testing Guide

This document explains the testing infrastructure for the MIRAGE pipeline, how tests work, and how to run them.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Framework: nf-test](#test-framework-nf-test)
3. [Test Structure](#test-structure)
4. [Running Tests](#running-tests)
5. [Test Types](#test-types)
6. [Writing New Tests](#writing-new-tests)
7. [Debugging Failed Tests](#debugging-failed-tests)
8. [Continuous Integration](#continuous-integration)

---

## Overview

The MIRAGE pipeline uses **nf-test** as its testing framework. nf-test is the official testing framework for Nextflow pipelines, developed by the nf-core community.

### Why Testing Matters

- **Catch bugs early** before they reach production
- **Validate refactoring** - ensure changes don't break functionality
- **Document expected behavior** - tests serve as executable specifications
- **Enable confident development** - make changes knowing tests will catch issues

### Test Coverage

Current test coverage includes:

- ✅ **Process tests**: Individual module validation (SEGMENT, SPLIT_CHANNELS, QUANTIFY, MERGE_QUANT_CSVS)
- ✅ **Pipeline tests**: End-to-end workflow validation
- ✅ **Parameter validation**: Input validation and error handling
- ✅ **Stub mode tests**: Fast structural validation
- ✅ **Full execution tests**: Real data processing validation

---

## Test Framework: nf-test

### What is nf-test?

nf-test is a testing framework specifically designed for Nextflow pipelines. It provides:

- **Process testing**: Test individual processes in isolation
- **Workflow testing**: Test subworkflows and complete pipelines
- **Snapshot testing**: Capture and compare process outputs
- **Stub mode**: Run tests without executing real commands (fast)
- **Parallel execution**: Run tests concurrently for speed

### Installation

nf-test should already be available if you're running the pipeline. To verify:

```bash
nf-test version
```

If not installed, follow the [official installation guide](https://www.nf-test.com/docs/getting-started/installation/).

---

## Test Structure

### Directory Layout

```
mirage/
├── tests/
│   ├── main.nf.test              # Pipeline-level tests
│   ├── modules/
│   │   ├── segment.nf.test       # SEGMENT process tests
│   │   ├── split_channels.nf.test # SPLIT_CHANNELS tests
│   │   └── quantify.nf.test      # QUANTIFY + MERGE tests
│   └── testdata/
│       ├── test_input.csv        # Main test input
│       ├── P001_ref.ome.tiff     # Reference image
│       ├── P001_mov1.ome.tiff    # Moving image
│       ├── sample_*.tif          # Sample channel images
│       ├── sample_*_quant.csv    # Sample quantification CSVs
│       └── valid_*.csv           # Checkpoint CSVs for testing
```

