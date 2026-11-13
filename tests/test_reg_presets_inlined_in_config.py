"""Pin the registration tier vocabulary, and STARE's tier-free defaults.

VALIS keeps its `high | medium | low | custom` tiers (`memory_mode`). The vocabulary exists in
more than one place, and not by choice:

* `lib/RegPresets.groovy` -- MODES / DEFAULT_MODE.
* `bin/utils/valis_config.py` -- the VALIS tier table, which cannot move to Groovy because its rows
  hold Python objects (a SuperPointFD class, a SuperGlueMatcher instance).
* `nextflow_schema.json` -- the enum the schema validates `memory_mode` against.

STARE's tiers (`reg_tiled_mode`) were REMOVED on 2026-11-04: STARE runs one default
configuration, the former `high` row, which is now the plain nextflow.config default of its four
knobs. The tests below pin that the removal is complete (no tier table left in lib/, no inlined
copy left in conf/modules.config, no `reg_tiled_mode` in the schema or the params block) and that
the defaults are byte-for-byte the former `high` row, read from
`ParamUtils.RETIRED_STARE_TIERS` (the table the launch refusal names).

Everything below parses the real files.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin" / "utils"))

GROOVY = (ROOT / "lib" / "RegPresets.groovy").read_text()
CONFIG = (ROOT / "conf" / "modules.config").read_text()
NEXTFLOW_CONFIG = (ROOT / "nextflow.config").read_text()

TIERS = ("high", "medium", "low")


@pytest.fixture
def valis_config(monkeypatch):
    """Import bin/utils/valis_config with the valis package faked out.

    valis is not installed in CI and valis_config imports it at module scope. Same approach as
    tests/test_micro_reg_gating.py and tests/test_jvm_cache_guard.py.
    """
    micro_mod = types.ModuleType("valis.micro_rigid_registrar")
    micro_mod.MicroRigidRegistrar = type("MicroRigidRegistrar", (), {})
    nonrigid_mod = types.ModuleType("valis.non_rigid_registrars")
    nonrigid_mod.OpticalFlowWarper = type("OpticalFlowWarper", (), {})
    fd_mod = MagicMock(name="valis.feature_detectors")
    fm_mod = MagicMock(name="valis.feature_matcher")

    valis_pkg = types.ModuleType("valis")
    valis_pkg.feature_detectors = fd_mod
    valis_pkg.feature_matcher = fm_mod
    valis_pkg.micro_rigid_registrar = micro_mod
    valis_pkg.non_rigid_registrars = nonrigid_mod

    for name, mod in {
        "valis": valis_pkg,
        "valis.feature_detectors": fd_mod,
        "valis.feature_matcher": fm_mod,
        "valis.micro_rigid_registrar": micro_mod,
        "valis.non_rigid_registrars": nonrigid_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.delitem(sys.modules, "valis_config", raising=False)
    import valis_config as vc

    return vc


# ---------------------------------------------------------------------------
# Parsers. Each has a companion test asserting it found something, so a regex
# that stops matching fails loudly instead of turning its callers into no-ops.
# ---------------------------------------------------------------------------


def _parse_row(block: str, mode: str) -> dict[str, int]:
    # NOT line-anchored, so a row written on one line is still found.
    row = re.search(rf"\b{mode}\s*:\s*\[([^\]]*)\]", block)
    assert row, f"no '{mode}' row in:\n{block}"
    return {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row.group(1))}


PARAM_UTILS = (ROOT / "lib" / "ParamUtils.groovy").read_text()
STARE_KNOBS = ("reg_tiled_tile", "reg_tiled_halo", "reg_tiled_out_tile", "reg_tiled_coarse_max_dim")


def retired_stare_tiers() -> dict[str, dict[str, int]]:
    """ParamUtils.RETIRED_STARE_TIERS: the rows the launch refusal names."""
    body = re.search(r"RETIRED_STARE_TIERS\s*=\s*\[(.*?)\n    \]", PARAM_UTILS, re.S)
    assert body, "could not locate RETIRED_STARE_TIERS in lib/ParamUtils.groovy"
    return {mode: _parse_row(body.group(1), mode) for mode in TIERS}


def groovy_modes() -> list[str]:
    m = re.search(r"MODES\s*=\s*\[([^\]]*)\]", GROOVY)
    assert m, "could not locate RegPresets.MODES"
    return re.findall(r"'([^']+)'", m.group(1))


def _config_default(param: str) -> str:
    m = re.search(rf"^\s*{param}\s*=\s*(\S+)", NEXTFLOW_CONFIG, re.M)
    assert m, f"{param} is not declared in nextflow.config"
    return m.group(1)


def _schema_registration() -> dict:
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    groups = schema.get("$defs") or schema["definitions"]
    return groups["registration_options"]["properties"]


def test_parsers_actually_parse():
    """If these come back empty every other test in this file passes vacuously."""
    rows = retired_stare_tiers()
    assert set(rows) == set(TIERS), f"parsed tiers {sorted(rows)}"
    for mode, row in rows.items():
        assert set(row) == set(STARE_KNOBS), f"RETIRED_STARE_TIERS['{mode}'] parsed {row}"
    assert groovy_modes(), "RegPresets.MODES parsed empty"


# ---------------------------------------------------------------------------
# VALIS: the tier vocabulary
# ---------------------------------------------------------------------------


def test_the_valis_tier_vocabulary_is_shared(valis_config):
    """RegPresets.MODES (Groovy) and valis_config.MEMORY_MODES (Python) must be the same list."""
    MEMORY_MODES = valis_config.MEMORY_MODES

    assert groovy_modes() == MEMORY_MODES, (
        f"lib/RegPresets.groovy MODES={groovy_modes()} but "
        f"bin/utils/valis_config.py MEMORY_MODES={MEMORY_MODES}. They advertise one "
        "vocabulary to the user; they must not diverge."
    )


def test_schema_enum_matches_the_tier_vocabulary():
    """nextflow_schema.json validates the user's input, so its enum is the one that must be right."""
    entry = _schema_registration()["memory_mode"]
    assert entry["enum"] == groovy_modes(), (
        f"nextflow_schema.json memory_mode.enum={entry['enum']} but the tier vocabulary is "
        f"{groovy_modes()}. validateParameters() would reject a tier the pipeline supports "
        "(or accept one it does not)."
    )


def test_custom_is_not_a_preset_row(valis_config):
    """'custom' must resolve to 'high', not be its own row."""
    assert "custom" in groovy_modes(), "'custom' left the tier vocabulary"
    assert "custom" not in valis_config.MEMORY_PRESETS, (
        "MEMORY_PRESETS grew a 'custom' row. 'custom' is defined as 'start from high and apply "
        "the overrides given' -- a row of its own contradicts that."
    )
    assert valis_config.resolve_memory_mode("custom") == "high"


@pytest.mark.parametrize("param", ["reg_valis_max_processed_dim", "reg_valis_max_non_rigid_dim"])
def test_valis_tier_owned_params_are_null_declared(param):
    """A tier-owned knob carrying a literal default would make the tier unreachable.

    Nextflow merges CLI args AFTER evaluating the params block, so a concrete default there is
    indistinguishable from a user-supplied value and `--memory_mode low` would silently do nothing.
    """
    value = _config_default(param)
    assert value == "null", (
        f"nextflow.config declares {param} = {value}, but it is tier-owned and must be null."
    )


# ---------------------------------------------------------------------------
# STARE: no tiers, and the defaults are the former high row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", STARE_KNOBS)
def test_stare_knob_defaults_are_the_former_high_row(param):
    """A default run must behave byte-for-byte as the old `reg_tiled_mode = 'high'` did."""
    want = retired_stare_tiers()["high"][param]
    assert _config_default(param) == str(want), (
        f"nextflow.config declares {param} = {_config_default(param)}; the former 'high' row "
        f"(ParamUtils.RETIRED_STARE_TIERS) says {want}. Changing a STARE default is a behaviour "
        "change of every default run, not part of the tier removal."
    )
    entry = _schema_registration()[param]
    assert entry.get("type") == "integer", f"schema {param} must be a plain integer, not nullable"
    assert entry.get("default") == want, f"schema {param}.default={entry.get('default')} != {want}"


def test_reg_tiled_mode_is_gone_from_the_params_block_and_the_schema():
    """Declaring it again would make the launch refusal unreachable for the CLI value."""
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


def test_no_stare_tier_table_is_left_in_lib_or_config():
    """The tier machinery is removed, not dormant: no table, no resolver, no inlined copy."""
    for name in ("STARE_KEYS", "STARE_PARAM_OF", "static int stare(", "STARE ="):
        assert name not in GROOVY, f"lib/RegPresets.groovy still carries {name!r}"
    assert not re.search(r"def _p\s*=", CONFIG), "conf/modules.config still inlines a tier table"
    assert "reg_tiled_mode" not in CONFIG, "conf/modules.config still reads reg_tiled_mode"
    for module in ("tiled_coarse.nf", "tiled_stitch.nf", "tiled_reg_tile.nf", "tiled_solve.nf"):
        text = (ROOT / "modules" / "local" / module).read_text()
        assert "RegPresets" not in text and "reg_tiled_mode" not in text, module


def test_the_refusal_names_each_retired_rows_values():
    """The message is built from RETIRED_STARE_TIERS, so medium/low name their own numbers;
    the behaviour itself is exercised by tests/lib_probe.nf (checkRetiredStareTier)."""
    body = re.search(r"static String checkRetiredParams\(Map params\)\s*\{(.*?)\n    \}",
                     PARAM_UTILS, re.S)
    assert body, "ParamUtils.checkRetiredParams is missing"
    assert "RETIRED_STARE_TIERS[mode]" in body.group(1)
    assert "throw new IllegalArgumentException" in body.group(1)
    rows = retired_stare_tiers()
    assert rows["medium"] == dict(zip(STARE_KNOBS, (1024, 192, 768, 1024)))
    assert rows["low"] == dict(zip(STARE_KNOBS, (512, 128, 512, 512)))
