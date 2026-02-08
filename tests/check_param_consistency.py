#!/usr/bin/env python3
"""Validate consistency across Nextflow params, schema, and code references."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "nextflow.config"
SCHEMA_PATH = ROOT / "nextflow_schema.json"

PARAM_REF_RE = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)")

# Internal references that are intentionally not user-facing params.
ALLOWED_REFERENCE_ONLY = {
    "container",
    "test",
}


def extract_params_block_keys(config_text: str) -> set[str]:
    start = config_text.find("params {")
    if start == -1:
        raise ValueError("Could not locate `params { ... }` block in nextflow.config")

    brace_start = config_text.find("{", start)
    depth = 0
    end = None
    for i in range(brace_start, len(config_text)):
        ch = config_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("Unclosed `params { ... }` block in nextflow.config")

    block = config_text[brace_start + 1 : end]
    keys: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        if m:
            keys.add(m.group(1))

    return keys


def extract_param_references() -> set[str]:
    refs: set[str] = set()
    patterns = [
        "main.nf",
        "subworkflows/**/*.nf",
        "modules/**/*.nf",
        "lib/**/*.groovy",
        "conf/**/*.config",
    ]
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            text = path.read_text()
            refs.update(PARAM_REF_RE.findall(text))
    return refs


def main() -> int:
    config_text = CONFIG_PATH.read_text()
    config_keys = extract_params_block_keys(config_text)

    schema = json.loads(SCHEMA_PATH.read_text())
    schema_keys = set(schema.get("properties", {}).keys())

    ref_keys = extract_param_references()

    ref_missing_in_config = sorted(ref_keys - config_keys - ALLOWED_REFERENCE_ONLY)
    ref_missing_in_schema = sorted(ref_keys - schema_keys - ALLOWED_REFERENCE_ONLY)
    config_missing_in_schema = sorted(config_keys - schema_keys)
    schema_missing_in_config = sorted(schema_keys - config_keys)

    issues: list[str] = []
    if ref_missing_in_config:
        issues.append("Referenced in code/config but missing from params block:")
        issues.extend(f"  - {key}" for key in ref_missing_in_config)
    if ref_missing_in_schema:
        issues.append("Referenced in code/config but missing from schema:")
        issues.extend(f"  - {key}" for key in ref_missing_in_schema)
    if config_missing_in_schema:
        issues.append("Defined in params block but missing from schema:")
        issues.extend(f"  - {key}" for key in config_missing_in_schema)
    if schema_missing_in_config:
        issues.append("Defined in schema but missing from params block:")
        issues.extend(f"  - {key}" for key in schema_missing_in_config)

    if issues:
        print("Parameter surface drift detected:")
        print("\n".join(issues))
        return 1

    print(
        "Parameter surface is consistent: "
        f"{len(config_keys)} params, {len(ref_keys)} references, {len(schema_keys)} schema entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
