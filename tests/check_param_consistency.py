#!/usr/bin/env python3
"""Validate consistency across Nextflow params, schema, and code references.

Two independent surfaces are compared:

1. **Key sets** -- every param must appear in `nextflow.config`'s `params {}`
   block, in `nextflow_schema.json`, and (where referenced) in the code.
2. **Default values** -- a param declared in both places must carry the *same*
   default in both. `nextflow.config` is the source of truth; the schema only
   documents it. The schema default is what `--help` prints, so a stale entry
   advertises a default the pipeline does not use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from _code_view import code_view

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "nextflow.config"
SCHEMA_PATH = ROOT / "nextflow_schema.json"

PARAM_REF_RE = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)")
PARAM_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
QUOTED_STR_RE = re.compile(r"^'([^']*)'$|^\"([^\"]*)\"$")
LIST_ITEM_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")

# Internal references that are intentionally not user-facing params.
ALLOWED_REFERENCE_ONLY = {
    "container",
    "test",
}


def params_block(config_text: str) -> str:
    """The body of nextflow.config's top-level `params { ... }` block."""
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

    return config_text[brace_start + 1 : end]


def extract_params_block_keys(config_text: str) -> set[str]:
    keys: set[str] = set()
    for line in params_block(config_text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        if m:
            keys.add(m.group(1))

    return keys


def strip_line_comment(value: str) -> str:
    """Drop a trailing `// ...` comment, ignoring `//` inside a quoted string."""
    quote = None
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "/" and value[i + 1 : i + 2] == "/":
            return value[:i]
    return value


class Unparsed(str):
    """A Groovy literal this checker cannot evaluate; compared to nothing."""


def parse_groovy_literal(raw: str):
    """Groovy literal -> Python value. Returns `Unparsed` when not representable."""
    s = strip_line_comment(raw).strip()
    if s == "null":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d*\.\d+", s):
        return float(s)
    m = QUOTED_STR_RE.match(s)
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    if s.startswith("[") and s.endswith("]"):
        if s == "[]":
            return []
        items = [a or b for a, b in LIST_ITEM_RE.findall(s)]
        if items:
            return items
    return Unparsed(s)


def extract_config_defaults(config_text: str) -> dict[str, object]:
    """param name -> default value, as declared in nextflow.config."""
    defaults: dict[str, object] = {}
    for line in params_block(config_text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = PARAM_ASSIGN_RE.match(stripped)
        if m:
            defaults[m.group(1)] = parse_groovy_literal(m.group(2))
    return defaults


def extract_param_references() -> set[str]:
    """Every `params.<name>` READ across pipeline code.

    Scans `code_view(path)`, which blanks comments and strings, so a comment
    mentioning `params.subMap(...)` or a filename like `params.py` is not a read.
    A match followed by `(` is a `Map` method call (`params.subMap(...)`,
    `params.containsKey(...)`), not a param read.
    """
    refs: set[str] = set()
    patterns = [
        "main.nf",
        "workflows/**/*.nf",  # a typo'd params.foo here read null and no guard saw it
        "subworkflows/**/*.nf",
        "modules/**/*.nf",
        "lib/**/*.groovy",
        "conf/**/*.config",
    ]
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            refs |= _refs_from_text(code_view(path))
    return refs


def _refs_from_text(text: str) -> set[str]:
    """The `params.<name>` READ rule, applied to already-masked text.

    `tests/nfmodel.param_refs` implements the same rule over caller-supplied
    text; `tests/test_param_consistency_rule.py` checks the two agree.
    """
    refs: set[str] = set()
    for m in PARAM_REF_RE.finditer(text):
        tail = text[m.end() :].lstrip()
        if tail.startswith("("):
            continue  # a Map method call (params.subMap(...)), not a param read
        refs.add(m.group(1))
    return refs


def values_equal(config_value, schema_value) -> bool:
    """Equality that does not let Python's `True == 1` mask a real type drift."""
    if isinstance(config_value, bool) != isinstance(schema_value, bool):
        return False
    return config_value == schema_value


def check_default_agreement(
    config_text: str, schema_props: dict[str, dict]
) -> list[str]:
    """Report params whose schema default contradicts their nextflow.config default."""
    config_defaults = extract_config_defaults(config_text)
    drift: list[str] = []

    for name in sorted(schema_props):
        if name not in config_defaults:
            continue  # already reported by the key-set comparison
        config_value = config_defaults[name]
        if isinstance(config_value, Unparsed):
            continue  # literal this checker cannot evaluate; not a drift signal
        has_schema_default = "default" in schema_props[name]
        schema_value = schema_props[name].get("default")

        if config_value is None:
            # `x = null` means "derive at runtime"; the schema states no default.
            if has_schema_default and schema_value is not None:
                drift.append(f"{name}: schema={schema_value!r}, nextflow.config=null")
        elif not has_schema_default:
            drift.append(
                f"{name}: schema has no default, nextflow.config={config_value!r}"
            )
        elif not values_equal(config_value, schema_value):
            drift.append(
                f"{name}: schema={schema_value!r}, nextflow.config={config_value!r}"
            )

    return drift


def main() -> int:
    config_text = CONFIG_PATH.read_text()
    config_keys = extract_params_block_keys(config_text)

    schema = json.loads(SCHEMA_PATH.read_text())
    # Params live under the nf-core grouped structure (definitions/$defs + allOf),
    # and/or a flat top-level "properties". Flatten all of them so the check works
    # regardless of which layout the schema uses.
    schema_props: dict[str, dict] = dict(schema.get("properties", {}))
    for group in {**schema.get("definitions", {}), **schema.get("$defs", {})}.values():
        schema_props.update(group.get("properties", {}))
    schema_keys = set(schema_props)

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

    default_drift = check_default_agreement(config_text, schema_props)
    if default_drift:
        issues.append(
            "Schema default disagrees with nextflow.config "
            "(nextflow.config is the source of truth -- fix the schema):"
        )
        issues.extend(f"  - {line}" for line in default_drift)

    if issues:
        print("Parameter surface drift detected:")
        print("\n".join(issues))
        return 1

    print(
        "Parameter surface is consistent: "
        f"{len(config_keys)} params, {len(ref_keys)} references, {len(schema_keys)} schema entries, "
        "schema defaults agree with nextflow.config."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
