#!/usr/bin/env bash
#
# Automated Validation Test Suite
# Tests input validation, error handling, and edge cases
#
# Based on test cases documented in tests/test_validation.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDATA_DIR="$SCRIPT_DIR/testdata"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output/validation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

echo "=========================================="
echo "ATEIA Pipeline Validation Test Suite"
echo "=========================================="
echo ""

# Clean up previous test runs
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Helper function to run a test
run_test() {
    local test_name="$1"
    local expected_result="$2"  # "pass" or "fail"
    local input_csv="$3"
    shift 3
    local extra_args=("$@")

    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "${YELLOW}[TEST $TESTS_TOTAL]${NC} $test_name"

    local output_file="$OUTPUT_DIR/test_${TESTS_TOTAL}.log"
    local exit_code=0

    # Run pipeline with dry_run for fast validation
    cd "$PROJECT_ROOT"
    nextflow run main.nf \
        --input "$input_csv" \
        --outdir "$OUTPUT_DIR/test_${TESTS_TOTAL}" \
        --dry_run true \
        "${extra_args[@]}" \
        > "$output_file" 2>&1 || exit_code=$?

    if [ "$expected_result" = "pass" ]; then
        # Should succeed
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}✓ PASS${NC} - Validation succeeded as expected"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ FAIL${NC} - Expected success but got failure"
            echo "  Exit code: $exit_code"
            echo "  Log: $output_file"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    else
        # Should fail with expected error message
        local expected_error="${extra_args[-1]}"  # Last arg is expected error
        if [ $exit_code -ne 0 ]; then
            # Check for expected error message
            if grep -q "$expected_error" "$output_file"; then
                echo -e "${GREEN}✓ PASS${NC} - Failed with expected error: '$expected_error'"
                TESTS_PASSED=$((TESTS_PASSED + 1))
                return 0
            else
                echo -e "${RED}✗ FAIL${NC} - Failed but error message not found"
                echo "  Expected: '$expected_error'"
                echo "  Log: $output_file"
                TESTS_FAILED=$((TESTS_FAILED + 1))
                return 1
            fi
        else
            echo -e "${RED}✗ FAIL${NC} - Expected failure but validation passed"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    fi
}

echo "Generating test data..."
python3 "$TESTDATA_DIR/generate_complete_testdata.py" > /dev/null 2>&1

echo ""
echo "=========================================="
echo "Test Suite: Input Validation"
echo "=========================================="
echo ""

# Test 1.1: Valid input - one reference per patient
run_test \
    "Valid input - one reference per patient" \
    "pass" \
    "$TESTDATA_DIR/valid_preprocessing.csv" \
    --step preprocessing

# Test 1.2: Invalid - multiple references per patient
run_test \
    "Invalid - multiple references per patient" \
    "fail" \
    "$TESTDATA_DIR/invalid_multi_ref.csv" \
    --step preprocessing \
    "Multiple reference images found"

# Test 1.3: Invalid - no reference per patient
run_test \
    "Invalid - no reference per patient" \
    "fail" \
    "$TESTDATA_DIR/invalid_no_ref.csv" \
    --step preprocessing \
    "No reference image found"

# Test 2.1: Valid - DAPI in any position
echo -e "${YELLOW}[INFO]${NC} Test 2.1 skipped - requires ND2 conversion which needs ND2 files"

# Test 2.2: Invalid - DAPI not in channel 0
run_test \
    "Invalid - DAPI not in channel 0" \
    "fail" \
    "$TESTDATA_DIR/invalid_dapi_position.csv" \
    --step preprocessing \
    "DAPI must be in channel 0"

# Test 2.3: Valid - DAPI in channel 0
run_test \
    "Valid - DAPI in channel 0" \
    "pass" \
    "$TESTDATA_DIR/valid_preprocessing.csv" \
    --step preprocessing

echo ""
echo "=========================================="
echo "Test Suite: Checkpoint CSV Validation"
echo "=========================================="
echo ""

# Test 6.1: Valid checkpoint CSV (registration)
run_test \
    "Valid checkpoint CSV for registration step" \
    "pass" \
    "$TESTDATA_DIR/valid_checkpoint_registration.csv" \
    --step registration \
    --registration_method gpu

# Test 6.2: Invalid - missing required column
run_test \
    "Invalid checkpoint - missing required column" \
    "fail" \
    "$TESTDATA_DIR/invalid_checkpoint_missing_col.csv" \
    --step registration \
    "Missing required column"

# Test 6.3: Invalid - malformed is_reference value
run_test \
    "Invalid checkpoint - malformed is_reference" \
    "fail" \
    "$TESTDATA_DIR/invalid_checkpoint_bad_ref.csv" \
    --step registration \
    "Invalid is_reference value"

# Test 6.4: Invalid - file does not exist
run_test \
    "Invalid - file not found" \
    "fail" \
    "$TESTDATA_DIR/invalid_file_not_found.csv" \
    --step preprocessing \
    "does not exist"

# Test 6.5: Invalid - missing DAPI channel
run_test \
    "Invalid - missing DAPI channel" \
    "fail" \
    "$TESTDATA_DIR/invalid_no_dapi.csv" \
    --step preprocessing \
    "DAPI channel not found"

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""
echo "Total tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "All validation tests passed! ✓"
    echo -e "==========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "Some tests failed! ✗"
    echo -e "==========================================${NC}"
    echo ""
    echo "Check logs in: $OUTPUT_DIR"
    exit 1
fi
