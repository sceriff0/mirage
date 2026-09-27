"""Pin the registration cost/accuracy tier tables to their forced duplicates.

The `high | medium | low | custom` vocabulary and the DRAPE tier values exist in more than one
place, and not by choice:

* `lib/RegPresets.groovy` -- the single source for the pipeline and the tiled modules.
* `conf/modules.config` -- TILED_COARSE's, TILED_REG_TILE's and TILED_STITCH's `memory = {}`
  closures and TILED_SOLVE's `ext.args = {}` closure INLINE the table. They have to: `conf/*.config` cannot see
  `lib/*.groovy` at all (the class name resolves silently against ConfigObject and fails only when
  the closure runs), and a shared helper is not expressible in a config file under Nextflow 26's
  strict parser. See the long comment at the top of conf/modules.config.
* `bin/utils/valis_config.py` -- the VALIS tier table, which cannot move to Groovy because its rows
  hold Python objects (a SuperPointFD class, a SuperGlueMatcher instance).
* `nextflow_schema.json` -- the enums the schema validates `memory_mode` / `reg_tiled_mode` against.

Duplication that cannot be removed has to be pinned instead, or it drifts. The specific failure
this prevents is silent and expensive: an inlined copy that still says `tile: 2048` after the
table moved would size TILED_REG_TILE's memory request for a tier the run is not using, and the
run would look completely normal -- `conf/modules.config`'s errorStrategy has an 'ignore' branch
that can drop the resulting OOM without failing the pipeline.

Everything below parses the real files. Nothing is mirrored as a literal here, because a mirrored
literal is just one more copy to go stale.
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
    # NOT line-anchored: TILED_SOLVE's inlined table is written on a single line, so a `^`
    # anchor would silently skip its 'medium' and 'low' rows.
    row = re.search(rf"\b{mode}\s*:\s*\[([^\]]*)\]", block)
    assert row, f"no '{mode}' row in:\n{block}"
    return {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row.group(1))}


def reg_presets_drape() -> dict[str, dict[str, int]]:
    """The authoritative table: RegPresets.DRAPE from lib/RegPresets.groovy."""
    body = re.search(r"DRAPE\s*=\s*\[(.*?)\n    \]", GROOVY, re.S)
    assert body, "could not locate the DRAPE table in lib/RegPresets.groovy"
    return {mode: _parse_row(body.group(1), mode) for mode in TIERS}


def inlined_tables() -> list[dict[str, dict[str, int]]]:
    """Every inlined `_p = [high: [...], medium: [...], low: [...]]` in conf/modules.config."""
    blocks = re.findall(r"def _p\s*=\s*\[(.*?)\]\s*\n\s*def _row", CONFIG, re.S)
    return [{mode: _parse_row(b, mode) for mode in TIERS} for b in blocks]


def groovy_modes() -> list[str]:
    m = re.search(r"MODES\s*=\s*\[([^\]]*)\]", GROOVY)
    assert m, "could not locate RegPresets.MODES"
    return re.findall(r"'([^']+)'", m.group(1))


def test_parsers_actually_parse():
    """If these come back empty every other test in this file passes vacuously."""
    drape = reg_presets_drape()
    assert set(drape) == set(TIERS), f"parsed tiers {sorted(drape)}"
    for mode, row in drape.items():
        assert row, f"RegPresets.DRAPE['{mode}'] parsed empty"

    tables = inlined_tables()
    assert len(tables) == 4, (
        f"expected 4 inlined tier tables in conf/modules.config, parsed {len(tables)}. "
        "TILED_COARSE (memory), TILED_REG_TILE (memory), TILED_STITCH (memory) and "
        "TILED_SOLVE (ext.args) each carry one."
    )
    assert groovy_modes(), "RegPresets.MODES parsed empty"


# ---------------------------------------------------------------------------
# The pins
# ---------------------------------------------------------------------------


def test_inlined_config_tables_match_reg_presets():
    """Each inlined copy must agree with RegPresets for every key it carries.

    Checked per-key rather than as whole rows: a closure only inlines the knobs it needs
    (TILED_SOLVE carries halo alone), and demanding full rows would force dead keys into the
    config just to satisfy a test.
    """
    authoritative = reg_presets_drape()
    for idx, table in enumerate(inlined_tables()):
        for mode, row in table.items():
            for key, value in row.items():
                assert key in authoritative[mode], (
                    f"inlined table #{idx + 1} in conf/modules.config has "
                    f"{mode}.{key}, which RegPresets.DRAPE['{mode}'] does not define"
                )
                assert value == authoritative[mode][key], (
                    f"inlined table #{idx + 1} in conf/modules.config says "
                    f"{mode}.{key}={value}, but lib/RegPresets.groovy says "
                    f"{authoritative[mode][key]}. The config copy is forced (config files cannot "
                    f"see lib/*.groovy) -- update BOTH, or the memory request is sized for a tier "
                    f"the run is not using."
                )


def test_the_two_backends_share_one_tier_vocabulary(valis_config):
    """RegPresets.MODES (Groovy) and valis_config.MEMORY_MODES (Python) must be the same list."""
    MEMORY_MODES = valis_config.MEMORY_MODES

    assert groovy_modes() == MEMORY_MODES, (
        f"lib/RegPresets.groovy MODES={groovy_modes()} but "
        f"bin/utils/valis_config.py MEMORY_MODES={MEMORY_MODES}. The two backends advertise one "
        "shared vocabulary to the user; they must not diverge."
    )


@pytest.mark.parametrize("param", ["memory_mode", "reg_tiled_mode"])
def test_schema_enum_matches_the_tier_vocabulary(param):
    """nextflow_schema.json validates the user's input, so its enum is the one that must be right."""
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    groups = schema.get("$defs") or schema["definitions"]
    entry = groups["registration_options"]["properties"][param]
    assert entry["enum"] == groovy_modes(), (
        f"nextflow_schema.json {param}.enum={entry['enum']} but the tier vocabulary is "
        f"{groovy_modes()}. validateParameters() would reject a tier the pipeline supports "
        "(or accept one it does not)."
    )


def test_custom_is_not_a_preset_row(valis_config):
    """'custom' must resolve to 'high', not be its own row.

    If someone adds a `custom:` row, "unset knobs stay at the high value" quietly stops being
    true and `custom` becomes a fourth tier with its own numbers.
    """
    assert "custom" in groovy_modes(), "'custom' left the tier vocabulary"
    assert "custom" not in reg_presets_drape(), (
        "RegPresets.DRAPE grew a 'custom' row. 'custom' is defined as 'start from high and apply "
        "the overrides given' -- a row of its own contradicts that."
    )

    assert "custom" not in valis_config.MEMORY_PRESETS, (
        "MEMORY_PRESETS grew a 'custom' row -- same problem"
    )
    assert valis_config.resolve_memory_mode("custom") == "high"


@pytest.mark.parametrize(
    "param",
    [
        "reg_tiled_tile",
        "reg_tiled_halo",
        "reg_tiled_out_tile",
        "reg_tiled_coarse_max_dim",
        "reg_valis_max_processed_dim",
        "reg_valis_max_non_rigid_dim",
    ],
)
def test_tier_owned_params_are_null_declared(param):
    """A tier-owned knob carrying a literal default would make the tier unreachable.

    Nextflow merges CLI args AFTER evaluating the params block, so resolution happens downstream
    of the config. A concrete default there is indistinguishable from a user-supplied value, so
    the preset could never win and `--reg_tiled_mode low` would silently do nothing.
    """
    m = re.search(rf"^\s*{param}\s*=\s*(\S+)", NEXTFLOW_CONFIG, re.M)
    assert m, f"{param} is not declared in nextflow.config"
    assert m.group(1) == "null", (
        f"nextflow.config declares {param} = {m.group(1)}, but it is tier-owned and must be null. "
        "A literal here cannot be distinguished from a user override, so the preset would never "
        "apply."
    )


def test_every_drape_tier_key_maps_to_a_real_param():
    """DRAPE_PARAM_OF must not name a param that does not exist -- the lookup fails silently."""
    mapping = dict(re.findall(r"(\w+)\s*:\s*'(reg_tiled_\w+)'", GROOVY))
    assert mapping, "could not parse RegPresets.DRAPE_PARAM_OF"
    drape_keys = set(reg_presets_drape()["high"])
    assert set(mapping) == drape_keys, (
        f"DRAPE_PARAM_OF covers {sorted(mapping)} but the tier rows define {sorted(drape_keys)}"
    )
    for key, param in mapping.items():
        assert re.search(rf"^\s*{param}\s*=", NEXTFLOW_CONFIG, re.M), (
            f"RegPresets.DRAPE_PARAM_OF maps '{key}' to '{param}', which nextflow.config does "
            "not declare"
        )
