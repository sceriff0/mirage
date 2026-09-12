"""Every process whose script can take a scale must be handed the configured one.

`params.pixel_size` is the single owner of every µm conversion in the pipeline. That
only holds if each process that accepts a scale is actually given it: a `bin/` script
whose `--pixel-size` is never passed falls back to its own argparse default, which is a
second, silent owner — exactly the failure this file exists to prevent.

Static rather than behavioural on purpose. `-stub` never evaluates a `script:` block, so
a stub run cannot see a rendered command at all, and a real nf-test per process would
need that process's whole dependency stack (a STARE manifest for
TILED_STITCH) to be installed just to read a string.
`tests/modules/split_channels.nf.test`'s rendered-command case is the behavioural
counterpart for the one process that can be rendered cheaply.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.stare_shims import source_of

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
MODULES = ROOT / "modules" / "local"
MODULES_CONFIG = ROOT / "conf" / "modules.config"

FLAG_RE = re.compile(r'add_argument\(\s*["\'](--pixel[-_]size)["\']')
PASSED_RE = re.compile(r"--pixel[-_]size\s+\$\{([^}]+)\}")

# Scripts that are operator tools rather than pipeline processes: no module invokes
# them, so there is no call site to check. Each must stay unreferenced by modules/ for
# this exemption to remain honest, which is asserted below.
STANDALONE = {"join_flowpath.py"}


def _scripts_accepting_a_scale() -> dict[str, str]:
    """{bin script name: the --pixel-size flag it declares}.

    Read through ``tests.stare_shims.source_of``: ``bin/tiled_stitch.py`` is a shim over
    ``stare.stages.stitch``, whose argparse is where the flag now lives. Keyed by the
    SHIM's name, because that is what ``modules/local/tiled_stitch.nf`` invokes.
    """
    out = {}
    for path in sorted(BIN.glob("*.py")):
        m = FLAG_RE.search(source_of(path).read_text())
        if m:
            out[path.name] = m.group(1)
    return out


# SEGMENT does not name its backend script: it renders `${backend.entrypoint}` from
# lib/SegBackends.groovy, so no .nf file contains the string "segment_instantseg.py".
# Resolved through the same table the module uses, rather than hardcoded here, so a
# renamed entrypoint surfaces as a lookup failure instead of a silently skipped script.
SEG_BACKENDS = ROOT / "lib" / "SegBackends.groovy"


def _backend_dispatched() -> set[str]:
    return set(re.findall(r"entrypoint:\s*'([^']+)'", SEG_BACKENDS.read_text()))


def _call_sites(script: str) -> list[Path]:
    """Module files whose script block invokes this bin script."""
    direct = [
        nf for nf in sorted(MODULES.glob("*.nf")) if f"{script} " in nf.read_text()
    ]
    if direct or script not in _backend_dispatched():
        return direct
    return [MODULES / "segment.nf"]


def _haystack(nf: Path) -> str:
    """The module's own text plus the `withName:` block that supplies its ext.args.

    A flag can legitimately live in either place: `conf/modules.config` is where tool
    arguments belong by convention, and several processes pass the scale from there.
    """
    text = nf.read_text()
    m = re.search(r"^process\s+([A-Z0-9_]+)\s*\{", text, re.M)
    if not m:
        return text
    conf = MODULES_CONFIG.read_text()
    block = re.search(rf"withName:\s*'{m.group(1)}'\s*\{{(.*?)\n    \}}", conf, re.S)
    return text + (block.group(1) if block else "")


def _resolves_to_the_param(expr: str, nf_text: str) -> bool:
    """True if the rendered value is params.pixel_size, directly or via a local def.

    `meta.pixel_size` is also accepted: it is INPUT_CHECK's per-slide value, already
    resolved from params.pixel_size by PREFLIGHT_SCALE (subworkflows/local/input_check.nf),
    and it is the only correct choice for a process with no image of its own to read a
    scale from -- passing params.pixel_size there would hand the tool the literal string
    'auto'. That is not a second owner of the scale: PREFLIGHT_SCALE is still the one
    place params.pixel_size gets resolved to a number.
    """
    expr = expr.strip()
    if expr in ("params.pixel_size", "meta.pixel_size"):
        return True
    return bool(
        re.search(rf"def\s+{re.escape(expr)}\s*=\s*params\.pixel_size\b", nf_text)
    )


SCRIPTS = _scripts_accepting_a_scale()


def test_the_scan_found_the_scripts_it_is_meant_to_cover():
    """A scope glob that matches nothing passes vacuously; this is the tripwire."""
    assert {
        "apply_basic_profiles.py",
        "convert_image.py",
        "split_multichannel.py",
        "tiled_stitch.py",
        "export_geojson.py",
    } <= set(SCRIPTS), sorted(SCRIPTS)


@pytest.mark.parametrize("script", sorted(_scripts_accepting_a_scale()))
def test_every_scale_accepting_script_is_handed_the_configured_scale(script):
    sites = _call_sites(script)
    if script in STANDALONE:
        assert not sites, (
            f"{script} is listed as standalone but {[p.name for p in sites]} invokes "
            f"it — remove it from STANDALONE and check the call site instead"
        )
        return
    assert sites, f"no module invokes {script}; add it to STANDALONE with a reason"
    for nf in sites:
        found = PASSED_RE.search(_haystack(nf))
        assert found, (
            f"{nf.name} invokes {script}, which accepts {SCRIPTS[script]}, but never "
            f"passes it — the script would silently use its own argparse default"
        )
        assert _resolves_to_the_param(found.group(1), nf.read_text()), (
            f"{nf.name} passes {SCRIPTS[script]} as '{found.group(1)}', which does not "
            f"resolve to params.pixel_size — a second owner of the scale"
        )
