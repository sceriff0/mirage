# Testing

## Python and Consistency Checks

```bash
python3 tests/check_param_consistency.py
pytest --collect-only -q
pytest -q
```

## Pipeline-Oriented Tests

```bash
bash tests/run_tests.sh
bash tests/run_validation_tests.sh
```

## nf-test

```bash
nf-test test tests/main.nf.test
nf-test test tests/subworkflows/local/registration.nf.test
nf-test test tests/modules/copy_results.nf.test
```

## What `check_param_consistency.py` Validates

- `nextflow.config` `params` keys match `nextflow_schema.json`
- `params.*` references used across `.nf`, `.groovy`, and `conf/*.config` are synchronized

## Optional/Heavy Dependencies

Some unit tests depend on optional runtime packages (for example `valis`, `csbdeep`, `stardist`, `basicpy`).
Tests for those paths are designed to skip cleanly when dependencies are unavailable.

