#!/usr/bin/env bash
set -euo pipefail

echo "Generating test fixtures..."
python3 tests/generate_testdata.py

echo "Running pytest unit tests..."
pytest -q

echo "Running nf-core lint (if installed)..."
if command -v nf-core >/dev/null 2>&1; then
  nf-core lint . || true
else
  echo "nf-core not installed; skipping lint"
fi

echo "Tests complete"
#!/bin/bash
# Minimal smoke test
echo 'Running one-patient test pipeline'