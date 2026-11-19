#!/usr/bin/env python3
"""Static guard: nothing reads `params.quantify_compartments` /
`params.expanded_quantification` / `params.embed_masks` raw except the resolver
(`lib/ParamUtils.groovy`'s `compartmentMode()`) and an explicit allowlist.

All three params are watched because `compartmentMode()` resolves them together
into one map; `workflows/mirage.nf` resolves it once and passes it down to the
subworkflows as an argument, the way `--registration_method` is read once in
`registration.nf`.

Two kinds of exception are allowed, for different reasons:

  1. `modules/local/*.nf` `script:`/`stub:` blocks that read a param to build
     their own CLI flag. Each is a leaf: it renders a command string and never
     passes the decision on, so there is nothing to thread.

  2. The one `ext.args` line in `conf/modules.config` that reads
     `params.expanded_quantification`: `conf/*.config` closures cannot see
     `lib/*.groovy` classes, so `ext.args` must read params raw. The exemption
     names that exact line, not the file, so a new raw read of any of the three
     params elsewhere in the config still fails.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every place production Nextflow/Groovy code lives (same as
# test_nuclear_marker_routing.py). `nextflow.config` declares the params rather
# than consuming them, and `tests/` may reference them freely.
SCANNED_GLOBS = [
    "main.nf",
    "modules/local/*.nf",
    "subworkflows/**/*.nf",
    "workflows/*.nf",
    "lib/*.groovy",
    "conf/*.config",
]

# The resolver itself. Its body is what "routed" means; it is not a consumer.
ROUTER_FILE = "ParamUtils.groovy"

# The three fields ParamUtils.compartmentMode() resolves together.
READ_RE = re.compile(
    r"(?:\w+\.)?params\.(?:quantify_compartments|expanded_quantification|embed_masks)"
)

# path-relative-to-ROOT -> reason. A whole-file exemption: every raw read in
# these files is a script:/stub: leaf building its own CLI flag.
ALLOWED_FILES = {
    "modules/local/quantify.nf": (
        "script: block -- builds its own --nuclei_mask_file flag from "
        "params.quantify_compartments, the same way every other "
        "ext.args-shaped flag this process builds is assembled inside its "
        "own script: block."
    ),
    "modules/local/export_geojson.nf": (
        "script:/stub: blocks -- script: builds its own "
        "--nucleus_contours_json flag from params.quantify_compartments; "
        "stub: mirrors that same branch (only touches "
        "cells_wholecell.geojson when the real run would have produced it) "
        "so the -stub output tree matches the real one."
    ),
    "modules/local/export_spatialdata.nf": (
        "script: block -- builds its own --nucleus-contours-json flag from "
        "params.quantify_compartments, the same leaf pattern as "
        "export_geojson.nf."
    ),
}

# path-relative-to-ROOT -> {normalised (stripped) code line: reason}. An
# exact-line exemption keyed on the line's content, not its number, so edits
# elsewhere in the file do not move it. Each line must occur exactly once
# (test_scan_actually_finds_the_consumers), and any other raw read in the file,
# including a second copy of the same line, still fails.
ALLOWED_LINES = {
    "conf/modules.config": {
        "ext.args = { params.expanded_quantification ? '--expanded' : '' }": (
            "conf/*.config closures cannot see lib/*.groovy classes, "
            "so ext.args must read params raw here."
        ),
    },
}


def _is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith(("//", "*", "/*", "#"))


def _read_sites() -> list[tuple[Path, int, str]]:
    """(file, 1-based line no, line text) for every raw read outside the resolver."""
    sites = []
    for pattern in SCANNED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.name == ROUTER_FILE:
                continue
            for i, line in enumerate(path.read_text().splitlines()):
                if _is_comment(line) or not READ_RE.search(line):
                    continue
                sites.append((path, i + 1, line))
    return sites


def _is_allowed(rel_path: str, line: str, sites: list[tuple[Path, int, str]]) -> bool:
    """True for a whole-file exemption, or an allowed line that occurs exactly once."""
    if rel_path in ALLOWED_FILES:
        return True
    text = line.strip()
    if text not in ALLOWED_LINES.get(rel_path, {}):
        return False
    copies = [
        1
        for p, _no, other in sites
        if str(p.relative_to(ROOT)) == rel_path and other.strip() == text
    ]
    return len(copies) == 1


def test_scan_actually_finds_the_consumers():
    """A scan matching nothing would pass the allowlist check vacuously."""
    sites = _read_sites()
    files = {str(path.relative_to(ROOT)) for path, _no, _line in sites}
    lines = [(str(path.relative_to(ROOT)), line.strip()) for path, _no, line in sites]
    assert len(sites) >= 4, (
        f"only {len(sites)} read site(s) found -- globs may be stale"
    )
    for expected in ALLOWED_FILES:
        assert expected in files, (
            f"{expected} no longer reads any of the three compartment params "
            "raw. If it was routed through ParamUtils.compartmentMode() "
            "instead, remove its ALLOWED_FILES entry too -- an allowlist "
            "entry for a file that no longer needs it lets this test go "
            "quiet on the next regression."
        )
    for expected_file, expected_lines in ALLOWED_LINES.items():
        for expected_line in expected_lines:
            count = lines.count((expected_file, expected_line))
            assert count == 1, (
                f"{expected_file}: allowed line {expected_line!r} occurs {count} "
                "time(s), expected exactly 1. If it is gone, remove its "
                "ALLOWED_LINES entry too, so a stale exemption cannot hide a "
                "regression; if it is duplicated, the copy is a new raw read."
            )


def test_no_file_outside_the_resolver_and_allowlist_reads_it_raw():
    """The seam: every other consumer must hold ParamUtils.compartmentMode()'s
    map, not read params.quantify_compartments / params.expanded_quantification
    / params.embed_masks itself."""
    sites = _read_sites()
    offenders = [
        f"{path.relative_to(ROOT)}:{no}: {line.strip()}"
        for path, no, line in sites
        if not _is_allowed(str(path.relative_to(ROOT)), line, sites)
    ]
    assert not offenders, (
        f"{len(offenders)} site(s) read a compartment-quantification param raw "
        "outside lib/ParamUtils.groovy (the resolver) and the documented "
        "allowlist. Resolve it once via ParamUtils.compartmentMode(params) and "
        "thread the result down as an argument instead -- the same seam "
        "--registration_method has (subworkflows/local/registration.nf) -- "
        "rather than re-reading the raw param here.\n" + "\n".join(offenders)
    )
