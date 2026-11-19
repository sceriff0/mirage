"""Groovy's `?:` cannot express "unset" for a numeric parameter.

`params.x ?: fallback` substitutes the fallback whenever `params.x` is *falsy*, and in Groovy
`0` is falsy. For a parameter whose schema declares `"type": ["number", "null"]` that is wrong
by construction: `null` means unset, `0` is a legal value with a meaning, and Elvis silently
turns the second into the first.

Example: `params.reg_tiled_max_disp ?: params.reg_tiled_halo` turns a requested `0` into the
halo value.

String parameters are out of scope: for a string, `''` and `null` both mean "not supplied",
so `?:` is idiomatic there (`params.stop`, `instanseg_model_dir`, `cellsam_model_path`).

The fix in every case is an explicit null test: `params.x != null ? params.x : fallback`.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMA = REPO / "nextflow_schema.json"
SCOPE = ("conf", "workflows", "subworkflows", "modules", "lib")
SUFFIXES = (".nf", ".config", ".groovy")

_ELVIS = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)\s*\?:")


def _numeric_nullable_params():
    """Schema params that accept a number AND null -- the ones where 0 and unset differ."""
    schema = json.loads(SCHEMA.read_text())
    found = {}

    def walk(node):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            for name, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                types = spec.get("type")
                types = types if isinstance(types, list) else [types]
                if "null" in types and ({"number", "integer"} & set(types)):
                    found[name] = spec
        for v in node.values():
            walk(v)

    walk(schema)
    return found


def _sources():
    files = []
    for d in SCOPE:
        for suf in SUFFIXES:
            files.extend(sorted((REPO / d).rglob(f"*{suf}")))
    assert files, "found no sources to scan -- the glob is wrong, not the repo"
    return files


def _offenders():
    numeric_nullable = _numeric_nullable_params()
    out = []
    for f in _sources():
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            for name in _ELVIS.findall(line):
                if name in numeric_nullable:
                    out.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    return out


def test_no_numeric_nullable_param_is_read_with_elvis():
    offenders = _offenders()

    assert not offenders, (
        "Groovy's `?:` treats 0 as unset, so these silently substitute the fallback for a "
        "legal 0. Use an explicit null test -- `params.x != null ? params.x : fallback`:\n  "
        + "\n  ".join(offenders)
    )


def test_the_schema_scan_finds_the_parameter_this_guard_exists_for():
    """If the schema walk returns nothing, the guard passes vacuously."""
    numeric_nullable = _numeric_nullable_params()

    assert "reg_tiled_max_disp" in numeric_nullable
    assert numeric_nullable["reg_tiled_max_disp"].get("minimum") == 0


def test_string_nullable_params_are_not_flagged():
    """The idiomatic string use must stay legal, or the guard becomes noise people disable."""
    numeric_nullable = _numeric_nullable_params()

    for name in ("stop", "instanseg_model_dir", "cellsam_model_path"):
        assert name not in numeric_nullable


def test_the_detector_recognises_the_shape_it_is_meant_to_catch():
    """Pin the regex against the exact removed line, so it cannot rot into a no-op."""
    removed = (
        "            def maxDisp = params.reg_tiled_max_disp ?: params.reg_tiled_halo"
    )

    assert _ELVIS.findall(removed) == ["reg_tiled_max_disp"]


def test_the_detector_accepts_the_prescribed_fix():
    fixed = "def maxDisp = params.reg_tiled_max_disp != null ? params.reg_tiled_max_disp : params.reg_tiled_halo"

    assert not _ELVIS.findall(fixed)
