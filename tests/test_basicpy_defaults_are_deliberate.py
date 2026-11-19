"""The vendored nf-core BASICPY module, and the decision to run it at UPSTREAM DEFAULTS.

Two separate things are pinned here, both of which are decisions a maintainer could undo
by accident:

1. **`modules/nf-core/basicpy/` is vendored unmodified.** Its container tag, its hardcoded
   version literal and its conda refusal are upstream's, not mirage's. If someone patches
   the module, the copy stops being "the nf-core module" and its MIRAGE-NOTES.md stops
   being true.

2. **`ext.args` is empty on purpose**: BaSiC runs at upstream's defaults. An empty
   string in a config file looks like an oversight, so it is pinned.

   Two flags in particular must not come back:

   * `--darkfield`. Adding it restores an additive darkfield model that the shipped
     configuration does not have. That is a change to the correction, not a
     tuning knob -- AND it disarms a safety check: see
     `test_the_darkfield_flag_has_not_come_back` below.
   * `--no_autotune`. Its name is INVERTED. Upstream declares it
     `action="store_false"`, `default=True`, and gates autotune with
     `if not args.no_autotune:` -- so NOT passing it SKIPS autotune (the intended
     behaviour) and PASSING it ENABLES autotune. A maintainer reading the flag name and
     "helpfully" adding it to turn autotune off would turn it ON.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "modules" / "nf-core" / "basicpy" / "main.nf"
NOTES = ROOT / "modules" / "nf-core" / "basicpy" / "MIRAGE-NOTES.md"
MODULES_CONFIG = ROOT / "conf" / "modules.config"


def _basicpy_block() -> str:
    """The body of `withName: 'BASICPY' { ... }`, found by brace-matching.

    Brace-matched rather than regexed to the first `}`: the block contains a
    `memory = { ... }` closure and a `publishDir = [ ... ]`, and a naive scan would stop
    inside the closure and read a truncated body -- i.e. would stop seeing the very
    `ext.args` line these tests exist to check.
    """
    text = MODULES_CONFIG.read_text()
    match = re.search(r"withName:\s*'BASICPY'\s*\{", text)
    assert match, "no `withName: 'BASICPY'` block in conf/modules.config"
    depth, pos = 1, match.end()
    while depth > 0 and pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[match.end() : pos - 1]


def _ext_args_value() -> str:
    block = _basicpy_block()
    match = re.search(r"^\s*ext\.args\s*=\s*(.+?)\s*$", block, re.M)
    assert match, (
        "BASICPY's withName block sets no ext.args at all. It must set it EXPLICITLY, "
        "even to '', so that running at upstream defaults reads as a decision rather "
        "than as a line nobody got round to writing."
    )
    return match.group(1)


# ---------------------------------------------------------------------------
# The decision: upstream defaults, nothing overridden
# ---------------------------------------------------------------------------


def test_basicpy_ext_args_carries_no_argument_at_all():
    """Empty, and explicitly so. Any flag here is an override of an upstream default."""
    value = _ext_args_value()
    assert value in ("''", '""'), (
        "conf/modules.config's `withName: 'BASICPY'` sets "
        f"ext.args = {value}. The ruling is to run nf-core's BASICPY at UPSTREAM "
        "DEFAULTS and not to reproduce mirage's previous in-process BaSiC parameters "
        "(get_darkfield=True, smoothness_flatfield=1). If that decision is being "
        "reversed, reverse it here and in this test together, with the reason -- do "
        "not let a flag appear on its own."
    )


def test_the_darkfield_flag_has_not_come_back():
    """`--darkfield` would restore a correction model the shipped configuration lacks.

    At upstream defaults `get_darkfield` is False: only the multiplicative flatfield is
    divided out and the additive offset stays in the data. Adding the flag changes what
    the pipeline computes, not how well it computes it.

    It also disarms the swap guard. `bin/apply_basic_profiles.py:_read_profile_stack`
    refuses a non-positive flatfield; at upstream defaults the darkfield is all zeros, so
    a dfp/ffp swap presents an all-zero flatfield and is rejected. With darkfield on, a
    swapped pair can be two plausible positive profiles and the output is silently
    wrong. Enabling `--darkfield` needs a replacement guard.
    """
    assert "--darkfield" not in _basicpy_block()
    assert "-df" not in _ext_args_value()


def test_the_inverted_autotune_flag_has_not_come_back():
    """`--no_autotune` ENABLES autotune. Read the docstring at the top of this file."""
    assert "--no_autotune" not in _basicpy_block()
    assert "-na" not in _ext_args_value()


def test_the_config_states_that_the_empty_args_are_deliberate():
    """An empty `ext.args` with no explanation is indistinguishable from an omission."""
    block_start = MODULES_CONFIG.read_text().index("withName: 'BASICPY'")
    preamble = MODULES_CONFIG.read_text()[max(0, block_start - 4000) : block_start]
    assert "UPSTREAM DEFAULTS" in preamble, (
        "the comment block above `withName: 'BASICPY'` no longer records that running "
        "at upstream defaults is a decision. Without it the empty ext.args reads as an "
        "oversight and will be 'fixed'."
    )


# ---------------------------------------------------------------------------
# The vendored module is upstream's
# ---------------------------------------------------------------------------


def test_the_vendored_module_pins_the_upstream_container():
    assert (
        'container "docker.io/labsyspharm/basicpy-docker-mcmicro:1.2.0-patch5"'
        in MODULE.read_text()
    )


def test_the_vendored_module_still_refuses_conda_in_both_blocks():
    """Upstream errors under -profile conda/mamba by design, in script: AND stub:.

    Two occurrences, not one: `-stub` does not exempt the guard, which is worth knowing
    before someone adds a conda profile and wonders why the stub suite fails.
    """
    text = MODULE.read_text()
    assert text.count("Basicpy module does not support Conda") == 2


def test_the_vendored_modules_version_emit_is_still_the_hardcoded_literal():
    """It reports `1.2.0` while the container is `1.2.0-patch5`, and it is a topic emit.

    Both facts are why BASICPY is absent from mirage's versions.yml aggregation. If
    upstream ever fixes either, this test fails and MIRAGE-NOTES.md needs revisiting --
    which is the point.
    """
    text = MODULE.read_text()
    assert 'val("1.2.0"), emit: versions_basicpy, topic: versions' in text
    assert "WARN: Version information not provided by tool on CLI" in text


@pytest.mark.parametrize(
    "phrase",
    [
        "does not support Conda",
        "versions.yml",
        "1.2.0-patch5",
        "--no_autotune",
    ],
)
def test_mirage_notes_records_each_divergence(phrase):
    """The notes file is the only place these facts live; an empty one is worse than none."""
    assert phrase in NOTES.read_text()
