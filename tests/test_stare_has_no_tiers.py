"""STARE has no cost tiers: its four geometry knobs are plain defaults, read directly.

STARE's `reg_tiled_mode` tiers (`high | medium | low | custom`) were removed on 2026-11-04,
and VALIS has had none since 2026-11-05, so `lib/RegPresets.groovy` held nothing and is gone.
This file replaces `tests/test_reg_presets_inlined_in_config.py`, which pinned the tier table
to the copies `conf/modules.config` had to inline: with no table there is nothing to pin, and
what is left to guard is that the removal is complete and that it changed no behaviour:

* the four knobs default to the former `high` row (read from `ParamUtils.RETIRED_STARE_TIERS`,
  the table the launch refusal names), in nextflow.config AND in the schema;
* `reg_tiled_mode` is declared nowhere, but nf-schema lets it through so that
  `ParamUtils.checkRetiredParams` is the one that answers it;
* every TILED_* resource/args closure in conf/modules.config and every tiled module reads the
  PLAIN params -- no inlined `_p` table, no `reg_tiled_mode`, no null fallback to a tier value.

The refusal's behaviour (medium/low/custom throw, high warns) is exercised by
tests/lib_probe.nf (checkRetiredStareTier). Nextflow source is read through tests.nfmodel.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.nfmodel import (
    REPO_ROOT,
    block_extent,
    processes,
    strip_comments,
    strip_comments_and_strings,
    with_name_blocks,
)

PARAM_UTILS = (REPO_ROOT / "lib" / "ParamUtils.groovy").read_text()
NEXTFLOW_CONFIG = (REPO_ROOT / "nextflow.config").read_text()
TIERS = ("high", "medium", "low")
STARE_KNOBS = (
    "reg_tiled_tile",
    "reg_tiled_halo",
    "reg_tiled_out_tile",
    "reg_tiled_coarse_max_dim",
)

# What each TILED_* config closure must read, by (process, closure).
CLOSURE_READS = {
    ("TILED_REG_TILE", "memory"): ("reg_tiled_tile", "reg_tiled_halo", "reg_tiled_stride"),
    ("TILED_COARSE", "memory"): ("reg_tiled_coarse_max_dim",),
    ("TILED_STITCH", "memory"): ("reg_tiled_out_tile",),
    ("TILED_SOLVE", "ext.args"): ("reg_tiled_halo", "reg_tiled_max_disp"),
}
# What each tiled module's script: block must read.
MODULE_READS = {
    "TILED_COARSE": ("reg_tiled_tile", "reg_tiled_halo", "reg_tiled_coarse_max_dim"),
    "TILED_STITCH": ("reg_tiled_out_tile",),
}


def _parse_row(block: str, mode: str) -> dict[str, int]:
    row = re.search(rf"\b{mode}\s*:\s*\[([^\]]*)\]", block)
    assert row, f"no '{mode}' row in:\n{block}"
    return {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row.group(1))}


def retired_stare_tiers() -> dict[str, dict[str, int]]:
    body = re.search(r"RETIRED_STARE_TIERS\s*=\s*\[(.*?)\n    \]", PARAM_UTILS, re.S)
    assert body, "could not locate RETIRED_STARE_TIERS in lib/ParamUtils.groovy"
    return {mode: _parse_row(body.group(1), mode) for mode in TIERS}


def _config_default(param: str) -> str:
    m = re.search(rf"^\s*{param}\s*=\s*(\S+)", NEXTFLOW_CONFIG, re.M)
    assert m, f"{param} is not declared in nextflow.config"
    return m.group(1)


def _schema_registration() -> dict:
    schema = json.loads((REPO_ROOT / "nextflow_schema.json").read_text())
    groups = schema.get("$defs") or schema["definitions"]
    return groups["registration_options"]["properties"]


def _closure(process: str, key: str) -> str:
    """The comment-stripped (strings kept) body of `key = { ... }` in `process`'s block."""
    blocks = [b for b in with_name_blocks() if process in b.names]
    assert len(blocks) == 1, f"expected one withName block for {process}, found {len(blocks)}"
    raw, clean = blocks[0].raw_body, blocks[0].body
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*\{{", clean)
    assert m, f"{process}: no `{key} = {{ ... }}` closure"
    end = block_extent(clean, m.end())
    return strip_comments(raw[m.end() : end])


def test_parsers_actually_parse():
    """If these come back empty every other test in this file passes vacuously."""
    rows = retired_stare_tiers()
    assert set(rows) == set(TIERS)
    for mode, row in rows.items():
        assert set(row) == set(STARE_KNOBS), f"RETIRED_STARE_TIERS['{mode}'] parsed {row}"
    for (process, key), reads in CLOSURE_READS.items():
        assert _closure(process, key).strip(), f"{process}.{key} parsed empty"


@pytest.mark.parametrize("param", STARE_KNOBS)
def test_stare_knob_defaults_are_the_former_high_row(param):
    """A default run must behave exactly as the old `reg_tiled_mode = 'high'` did."""
    want = retired_stare_tiers()["high"][param]
    assert _config_default(param) == str(want), (
        f"nextflow.config declares {param} = {_config_default(param)}; the former 'high' row "
        f"(ParamUtils.RETIRED_STARE_TIERS) says {want}. Changing a STARE default is a behaviour "
        "change of every default run, not part of the tier removal."
    )
    entry = _schema_registration()[param]
    assert entry.get("type") == "integer", f"schema {param} must be a plain integer, not nullable"
    assert entry.get("default") == want, f"schema {param}.default={entry.get('default')} != {want}"


def test_reg_tiled_mode_is_gone_but_reaches_the_refusal():
    params_block = re.search(r"^params\s*\{(.*?)^\}", NEXTFLOW_CONFIG, re.S | re.M)
    assert params_block, "could not locate the params {} block of nextflow.config"
    assert not re.search(r"^\s*reg_tiled_mode\s*=", params_block.group(1), re.M)
    assert "reg_tiled_mode" not in _schema_registration()
    # ...but nf-schema must let it through, or its generic error pre-empts the refusal
    ignore = re.search(r"ignoreParams\s*=\s*\[(.*?)\]", NEXTFLOW_CONFIG, re.S)
    assert ignore and "'reg_tiled_mode'" in ignore.group(1), (
        "reg_tiled_mode must be in validation.ignoreParams so ParamUtils.checkRetiredParams, "
        "not nf-schema's 'unrecognised parameter', answers a stale value"
    )
    mirage = strip_comments((REPO_ROOT / "workflows" / "mirage.nf").read_text())
    check, validate = mirage.find("checkRetiredParams(params)"), mirage.find("validateParameters()")
    assert 0 <= check < validate, "checkRetiredParams must run before validateParameters()"


def test_no_tier_machinery_is_left():
    """Removed, not dormant: no RegPresets class, and nothing that names it."""
    assert not (REPO_ROOT / "lib" / "RegPresets.groovy").exists()
    live = [
        *(REPO_ROOT / "lib").glob("*.groovy"),
        *(REPO_ROOT / "modules" / "local").glob("*.nf"),
        *(REPO_ROOT / "subworkflows").rglob("*.nf"),
        *(REPO_ROOT / "workflows").glob("*.nf"),
        REPO_ROOT / "conf" / "modules.config",
        REPO_ROOT / "nextflow.config",
    ]
    for path in live:
        code = strip_comments_and_strings(path.read_text())
        rel = path.relative_to(REPO_ROOT)
        assert "RegPresets" not in code, f"{rel} still uses RegPresets"
        assert "reg_tiled_mode" not in code, f"{rel} still reads reg_tiled_mode"


@pytest.mark.parametrize("process,key", sorted(CLOSURE_READS))
def test_tiled_config_closures_read_the_plain_params(process, key):
    """The STARE memory formulas (and SOLVE's range gate) read each knob straight from
    params: no inlined tier table, no null fallback that would silently pick a tier value."""
    body = _closure(process, key)
    assert not re.search(r"\bdef\s+_p\b|\b_row\b", body), f"{process}.{key} inlines a tier table"
    for param in CLOSURE_READS[(process, key)]:
        assert f"params.{param}" in body, f"{process}.{key} does not read params.{param}"
    for param in STARE_KNOBS:
        assert not re.search(rf"params\.{param}\s*!=\s*null", body), (
            f"{process}.{key} null-tests params.{param}: it has a plain default now, so a "
            "null fallback can only be a leftover tier lookup"
        )


@pytest.mark.parametrize("process", sorted(MODULE_READS))
def test_tiled_modules_read_the_plain_params(process):
    script = strip_comments(processes()[process].script_body)
    for param in MODULE_READS[process]:
        assert re.search(rf"params\.{param}\s+as\s+int", script), (
            f"{process}'s script: does not read params.{param} directly (an itemised read "
            "keeps only that value in the task hash)"
        )


def test_the_refusal_names_each_retired_rows_values():
    body = re.search(
        r"static String checkRetiredParams\(Map params\)\s*\{(.*?)\n    \}", PARAM_UTILS, re.S
    )
    assert body, "ParamUtils.checkRetiredParams is missing"
    assert "RETIRED_STARE_TIERS[mode]" in body.group(1)
    assert "throw new IllegalArgumentException" in body.group(1)
    rows = retired_stare_tiers()
    assert rows["medium"] == dict(zip(STARE_KNOBS, (1024, 192, 768, 1024)))
    assert rows["low"] == dict(zip(STARE_KNOBS, (512, 128, 512, 512)))
