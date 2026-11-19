#!/usr/bin/env bash
#
# Automated Validation Test Suite
# Tests input validation, error handling, and edge cases
#
# Based on test cases documented in tests/test_validation.md


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
echo "MIRAGE Pipeline Validation Test Suite"
echo "=========================================="
echo ""

# Clean up previous test runs
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Params go in a file, not on the command line: Nextflow 26 hands every CLI param
# to nf-schema as a String, so `--dry_run true` (or a bare `--dry_run`) is rejected
# as "[string] but should be [boolean]". A -params-file carries a real JSON boolean.
# Any boolean this script needs goes here.
PARAMS_FILE="$OUTPUT_DIR/dry_run_params.json"
# dry_run exercises the whole validation surface without instantiating a process.
# pixel_size and a resource ceiling are required at launch; without them every case
# would fail on the wrong error. The values are arbitrary but legal.
printf '{ "dry_run": true, "pixel_size": 0.325, "max_cpus": 2, "max_memory": "6.GB" }\n' \
    > "$PARAMS_FILE"

# Helper function to run a test.
#
# Usage: run_test <name> <pass|fail> <input_csv> <expected_error> [extra nextflow args...]
#   <expected_error> is the substring required in the log for a "fail" test, ""
#   for a "pass" test. A positional argument so it never leaks into the command.
run_test() {
    local test_name="$1"
    local expected_result="$2"  # "pass" or "fail"
    local input_csv="$3"
    local expected_error="$4"
    shift 4
    local extra_args=("$@")

    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "${YELLOW}[TEST $TESTS_TOTAL]${NC} $test_name"

    local output_file="$OUTPUT_DIR/test_${TESTS_TOTAL}.log"
    local exit_code=0

    # ${arr[@]+"${arr[@]}"} expands a possibly-empty array under `set -u`
    # (bash 3.2 on macOS errors on a bare "${arr[@]}" when empty).
    cd "$PROJECT_ROOT"
    nextflow run main.nf \
        -params-file "$PARAMS_FILE" \
        --input "$input_csv" \
        --outdir "$OUTPUT_DIR/test_${TESTS_TOTAL}" \
        ${extra_args[@]+"${extra_args[@]}"} \
        > "$output_file" 2>&1 || exit_code=$?

    if [ "$expected_result" = "pass" ]; then
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
        if [ $exit_code -ne 0 ]; then
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
    "" \
    --start preprocessing

# Test 1.2: Invalid - multiple references per patient
run_test \
    "Invalid - multiple references per patient" \
    "fail" \
    "$TESTDATA_DIR/invalid_multi_ref.csv" \
    "Multiple reference images found" \
    --start preprocessing

# Test 1.3: Invalid - no reference per patient
run_test \
    "Invalid - no reference per patient" \
    "fail" \
    "$TESTDATA_DIR/invalid_no_ref.csv" \
    "No reference image found" \
    --start preprocessing

# Test 2.1: Valid - DAPI in a non-zero position (segmentation locates DAPI by
# name, so any position is accepted as long as DAPI is present).
run_test \
    "Valid - DAPI in non-zero position" \
    "pass" \
    "$TESTDATA_DIR/invalid_dapi_position.csv" \
    "" \
    --start preprocessing

# Test 2.2: Valid - DAPI in channel 0
run_test \
    "Valid - DAPI in channel 0" \
    "pass" \
    "$TESTDATA_DIR/valid_preprocessing.csv" \
    "" \
    --start preprocessing

# Test 2.3: Invalid - no nuclear channel at all. The expectation is a regex, so the
# marker list in the parentheses can change; it still asserts the reason and patient.
run_test \
    "Invalid - no nuclear channel" \
    "fail" \
    "$TESTDATA_DIR/invalid_no_dapi.csv" \
    "No nuclear channel (.*) found for patient P001" \
    --start preprocessing

# Test 2.4: Invalid - input file does not exist
run_test \
    "Invalid - input file not found" \
    "fail" \
    "$TESTDATA_DIR/invalid_file_not_found.csv" \
    "does not exist" \
    --start preprocessing

# Test 2.5: Invalid - empty samplesheet (header only)
run_test \
    "Invalid - empty samplesheet" \
    "fail" \
    "$TESTDATA_DIR/empty_samplesheet.csv" \
    "no data rows" \
    --start preprocessing

echo ""
echo "=========================================="
echo "Test Suite: Samplesheet shapes"
echo "=========================================="
echo ""

# Test 3.1: paths relative to the launch directory. Only possible here: relative
# paths resolve against the JVM's working directory, and under nf-test that is a
# per-test hash directory unknown when the fixture is generated.
run_test \
    "Valid - paths relative to the launch directory" \
    "pass" \
    "$TESTDATA_DIR/relative_paths_input.csv" \
    "" \
    --start preprocessing

# Test 3.2: CRLF line endings (Excel on Windows). splitCsv strips the trailing \r;
# this guards against a reader that splits on "\n" itself.
run_test \
    "Valid - CRLF line endings" \
    "pass" \
    "$TESTDATA_DIR/crlf_input.csv" \
    "" \
    --start preprocessing

# Test 3.3: UTF-8 BOM on the header. Without CsvUtils.readCsvLines' strip the
# first column reads as "<BOM>patient_id" and every lookup returns -1 -- which
# is a hang in the sized groupTuples, not an error.
run_test \
    "Valid - UTF-8 BOM on the header row" \
    "pass" \
    "$TESTDATA_DIR/bom_input.csv" \
    "" \
    --start preprocessing

# Test 3.4: a quoted path containing a space.
run_test \
    "Valid - quoted path containing a space" \
    "pass" \
    "$TESTDATA_DIR/space_path_input.csv" \
    "" \
    --start preprocessing

# Test 3.5: a non-ASCII filename.
run_test \
    "Valid - non-ASCII filename" \
    "pass" \
    "$TESTDATA_DIR/unicode_input.csv" \
    "" \
    --start preprocessing

# Test 3.6: extra, unknown columns are ACCEPTED. Rejecting them would be wrong:
# a checkpoint CSV legitimately carries columns the entry step does not read.
# The run must still warn -- that half is asserted in Test 3.7.
run_test \
    "Valid - unknown extra columns are ignored" \
    "pass" \
    "$TESTDATA_DIR/extra_column_input.csv" \
    "" \
    --start preprocessing

# Test 3.7: the warning half of Test 3.6: the log must name the ignored columns.
# run_test greps only failure cases, so this greps the log directly.
TESTS_TOTAL=$((TESTS_TOTAL + 1))
echo -e "${YELLOW}[TEST $TESTS_TOTAL]${NC} Unknown extra columns are NAMED in a warning"
warn_log="$OUTPUT_DIR/test_${TESTS_TOTAL}.log"
cd "$PROJECT_ROOT"
# The exit status is not the assertion; it is captured, not discarded.
nextflow run main.nf \
    -params-file "$PARAMS_FILE" \
    --input "$TESTDATA_DIR/extra_column_input.csv" \
    --outdir "$OUTPUT_DIR/test_${TESTS_TOTAL}" \
    --start preprocessing \
    > "$warn_log" 2>&1 || warn_run_exit=$?
if grep -q "operator, scan_date" "$warn_log"; then
    echo -e "${GREEN}✓ PASS${NC} - Unknown columns named in the log"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ FAIL${NC} - The run did not name the ignored columns"
    echo "  Log: $warn_log (nextflow exit ${warn_run_exit:-0})"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

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
    "" \
    --start registration

# Test 6.2: Invalid - missing required column
run_test \
    "Invalid checkpoint - missing required column" \
    "fail" \
    "$TESTDATA_DIR/invalid_checkpoint_missing_col.csv" \
    "Missing required column" \
    --start registration

# Test 6.3: Invalid - malformed is_reference value
run_test \
    "Invalid checkpoint - malformed is_reference" \
    "fail" \
    "$TESTDATA_DIR/invalid_checkpoint_bad_ref.csv" \
    "Invalid is_reference value" \
    --start registration

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
