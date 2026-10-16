#!/usr/bin/env python3
"""Turn a run-plan row into a Nextflow ``-params-file`` with real JSON types.

WHY THIS EXISTS. ``run_sweep.sh`` and ``run_arms.sh`` (the arm and sweep launchers
regbench replaces) used to pass every swept axis as ``--<name> <value>``. **Nextflow 26 delivers every CLI ``--param`` as a
String**, and ``nextflow_schema.json`` declares real types, so the run dies at
``validateParameters()`` before a single task::

    * --reg_qc (1): Value is [string] but should be [integer]
    * --reg_qc (1): Expected any of [[0, 1, 2]]
    * --pyramid_resolutions (4): Value is [string] but should be [integer]

Reproduced 2026-08-25: NXF_VER=25.04.7 accepts the CLI form, 26.04.6 rejects it.
The sweep passes ~40 such axes, so the whole experiment was unlaunchable on the
engine this pipeline is verified on -- and it would have failed at LAUNCH, on the
cluster, after the matrix had already been generated.

A ``-params-file`` carries JSON types, so it survives both engines. It is also
the mechanism ``docs/usage.md`` already prescribes for boolean params, for the
same underlying reason.

WHY IT COERCES FROM THE SCHEMA RATHER THAN GUESSING. "Looks like a number" is
the wrong rule: a param declared ``"type": "string"`` whose value happens to be
``2048`` must stay a string, or validation fails in the other direction. The
schema is the thing being validated against, so it is the thing consulted.
A run-plan column naming no schema param is passed through unchanged and
reported on stderr rather than silently dropped.

regbench's own pipeline runs build their ``-params-file`` with it (``slurm/inputs.sbatch``,
the input step).

Usage::

    python3 -m benchmarks.regbench.params_json --out params.json start=preprocessing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = REPO_ROOT / "nextflow_schema.json"

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


class CoercionError(ValueError):
    """A value that cannot be represented as the type the schema declares."""


def schema_types(schema_path: Path | None = None) -> dict:
    """param name -> the set of JSON types the schema allows for it."""
    doc = json.loads((schema_path or SCHEMA).read_text())
    found: dict[str, set] = {}

    def walk(node):
        if isinstance(node, dict):
            for name, entry in (node.get("properties") or {}).items():
                if isinstance(entry, dict) and "type" in entry:
                    t = entry["type"]
                    found[name] = {t} if isinstance(t, str) else set(t)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    return found


def coerce(name: str, raw: str, types: dict):
    """One CSV cell -> the JSON value the schema declares for ``name``.

    ``types`` comes from :func:`schema_types`. Order matters: boolean before
    integer, because ``0``/``1`` are legal spellings of both and a param
    declared boolean must not become an int.
    """
    allowed = types.get(name)
    if allowed is None:
        print(
            f"params_json: '{name}' is not declared in nextflow_schema.json; "
            f"passing it through as a string",
            file=sys.stderr,
        )
        return raw

    if "null" in allowed and raw.strip().lower() in ("", "null", "none"):
        return None
    if "boolean" in allowed:
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise CoercionError(f"{name}={raw!r} is not a boolean")
    if "integer" in allowed:
        try:
            return int(raw)
        except ValueError as exc:
            raise CoercionError(f"{name}={raw!r} is not an integer") from exc
    if "number" in allowed:
        try:
            return float(raw)
        except ValueError as exc:
            raise CoercionError(f"{name}={raw!r} is not a number") from exc
    return raw


def build(pairs, types: dict | None = None) -> dict:
    """``["k=v", ...]`` -> the params mapping, typed."""
    types = schema_types() if types is None else types
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise CoercionError(f"expected key=value, got {pair!r}")
        name, raw = pair.split("=", 1)
        out[name] = coerce(name, raw, types)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.regbench.params_json",
        description=__doc__.split("\n", 1)[0],
    )
    ap.add_argument("--out", required=True, help="path to write the JSON params file")
    ap.add_argument("pairs", nargs="*", help="key=value, one per swept param")
    a = ap.parse_args(argv)
    Path(a.out).write_text(json.dumps(build(a.pairs), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
