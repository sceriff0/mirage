#!/usr/bin/env python3
"""Static guard: nothing reads `params.nuclear_markers` raw.

`params.nuclear_markers` arrives in three shapes and two of them break SILENTLY, so
`lib/MarkerUtils.groovy`'s `markerList()` is the only sanctioned way to read it:

  ['DAPI','CELLTOX']   nextflow.config / a params file -- the intended shape
  'CELLTOX'            `--nuclear_markers CELLTOX` (nextflow_schema.json permits a
                       string, because an array cannot be given on a command line)
  ['DAPI,CELLTOX']     a params file with the list written as one comma-joined string

The trap: on the String shape, `params.nuclear_markers.join(' ')` does not throw.
Groovy dispatches `"CELLTOX".join(' ')` to Java's static `String.join` with zero
varargs, which returns an empty string, so the rendered command carries a bare
`--nuclear-markers`. `-stub` never evaluates a `script:` block, so the guard is static.

Two rules, both mechanical:

  A. A raw read may never be followed directly by `.` or `[`. That is the defect shape:
     any method call or index on the un-normalised parameter.
  B. Every read must be routed -- passed to `MarkerUtils.`/`CsvUtils.` (whose entry
     points all funnel into `markerList`), on the same line or as an argument line of a
     multi-line call to one of them.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every place Nextflow/Groovy code can live. `nextflow.config` is excluded: it DECLARES
# the parameter (that is its job and the only place a default may live) rather than
# consuming it.
SCANNED_GLOBS = [
    "main.nf",
    "modules/local/*.nf",
    "subworkflows/**/*.nf",
    "workflows/*.nf",
    "lib/*.groovy",
    "conf/*.config",
]

# The router itself: its mentions of the parameter are documentation and an exception
# message. Excluded by name; `test_scan_actually_finds_the_consumers` keeps the
# exclusion from widening.
ROUTER_FILE = "MarkerUtils.groovy"

READ_RE = re.compile(r"(?:\w+\.)?params\.nuclear_markers")
# Rule A: the raw read with a method call or an index hung directly off it.
RAW_USE_RE = re.compile(r"(?:\w+\.)?params\.nuclear_markers\s*(?:\.\w|\[)")
ROUTERS = ("MarkerUtils.", "CsvUtils.")
# Rule B fallback: a bare argument line inside a multi-line call.
BARE_ARG_RE = re.compile(r"^\(?\s*(?:\w+\.)?params\.nuclear_markers\s*[,)]?\s*$")
# How far back a bare argument line may look for its callee.
CALL_LOOKBACK = 10


def _is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith(("//", "*", "/*", "#"))


def _read_sites() -> list[tuple[Path, int, str, list[str]]]:
    """(file, 1-based line no, line text, preceding lines) for every read."""
    sites = []
    for pattern in SCANNED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.name == ROUTER_FILE:
                continue
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if _is_comment(line) or not READ_RE.search(line):
                    continue
                start = max(0, i - CALL_LOOKBACK)
                sites.append((path, i + 1, line, lines[start:i]))
    return sites


def test_scan_actually_finds_the_consumers():
    """A scan that matches nothing would pass both rules vacuously."""
    sites = _read_sites()
    files = {path.name for path, _no, _line, _prev in sites}
    assert len(sites) >= 6, f"only {len(sites)} read sites found -- globs may be stale"
    # The known consumers. If one disappears, that is a real change worth noticing
    # rather than a scan silently going quiet.
    for expected in (
        "convert_image.nf",
        "SegBackends.groovy",
        "input_check.nf",
    ):
        assert expected in files, f"{expected} no longer reads params.nuclear_markers"


def test_no_method_call_on_the_raw_parameter():
    """Rule A -- the shape of the convert_image.nf defect."""
    offenders = [
        f"{path.relative_to(ROOT)}:{no}: {line.strip()}"
        for path, no, line, _prev in _read_sites()
        if RAW_USE_RE.search(line)
    ]
    assert not offenders, (
        f"{len(offenders)} site(s) call a method on (or index) params.nuclear_markers "
        "directly. The parameter may be a List, a bare String, or a one-element "
        "comma-joined String -- `.join(' ')` on the String shape silently returns the "
        "EMPTY string. Wrap it: MarkerUtils.markerList(params.nuclear_markers).\n"
        + "\n".join(offenders)
    )


def test_every_read_is_routed_through_markerutils():
    """Rule B -- no sixth consumer can appear without going through the one rule."""
    offenders = []
    for path, no, line, prev in _read_sites():
        if any(r in line for r in ROUTERS):
            continue
        if BARE_ARG_RE.match(line.strip()) and any(
            r in p for p in prev for r in ROUTERS
        ):
            continue  # argument line of a multi-line CsvUtils/MarkerUtils call
        offenders.append(f"{path.relative_to(ROOT)}:{no}: {line.strip()}")

    assert not offenders, (
        f"{len(offenders)} site(s) read params.nuclear_markers without routing it "
        "through MarkerUtils (directly, or via a CsvUtils entry point that does). "
        "markerList() is the single place the parameter's three possible shapes are "
        "normalised; a site that bypasses it is a silent divergence, not an error.\n"
        + "\n".join(offenders)
    )
