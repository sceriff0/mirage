#!/usr/bin/env python3
"""An aliased process inherits the base process' `publishDir`, and must override it.

Nextflow matches a config selector against an aliased include's OWN name *and* the
process' original declared name. `include { SPLIT_CHANNELS as SPLIT_PRIOR_PYRAMID }`
is therefore governed by `withName: 'SPLIT_CHANNELS'`, confirmed in this repo from a
real run's debug log (`Config settings withName:SPLIT_CHANNELS matches process
SPLIT_PRIOR_PYRAMID`) and written up beside that block in conf/modules.config.

For resources that is exactly what you want -- the alias is the same process and should
cost the same. For `publishDir` it is a silent data-loss bug, and it has fired here:
both instances of SPLIT_CHANNELS published `*.tiff` by bare channel name into the same
`<pid>/split_channels/`, and whichever task finished last overwrote the other's file for
every marker the two shared. Nothing failed; a file was just wrong.

So: if a base process publishes, every alias of it must say where IT publishes (or that
it does not), in its own `withName:` block, placed textually AFTER the base's -- Nextflow
applies later matching selectors on top of earlier ones, per field, so an override that
comes first is silently won back by the block it meant to override.

This test does not check resources. `tests/test_resource_label_coverage.py` owns those,
and the inheritance it relies on is the half of alias matching that is correct.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES_CONFIG = ROOT / "conf" / "modules.config"
NF_DIRS = (ROOT / "workflows", ROOT / "subworkflows")

# `include { BASE as ALIAS } from '...'` -- the only form that creates a second name
# for one process. A plain `include { NAME }` needs nothing from this test.
ALIAS_INCLUDE_RE = re.compile(r"include\s*\{\s*(\w+)\s+as\s+(\w+)\s*\}", re.MULTILINE)

WITH_NAME_OPEN_RE = re.compile(r"withName:\s*(?:'([^']+)'|\"([^\"]+)\")\s*\{")

# A real `publishDir = ...` ASSIGNMENT, anchored to the start of a line so a `//`
# comment merely discussing publishDir does not count. Every block touched by this
# rule has a comment explaining its publishing decision, so a bare `"publishDir" in
# body` substring test passes on a block that sets nothing -- verified by deleting
# the assignment and watching that version of this guard stay green.
PUBLISH_DIR_ASSIGN_RE = re.compile(r"^\s*publishDir\s*=", re.MULTILINE)


def _with_name_blocks(text: str):
    """Yield (names, start_offset, body) for each `withName:` block, in file order.

    `names` is the selector split on `|` -- conf/modules.config uses alternation to
    cover several processes with one block. The body is found by brace-matching from
    the selector's opening `{`, because a body contains closures and list literals
    that no regex can balance.
    """
    for m in WITH_NAME_OPEN_RE.finditer(text):
        selector = m.group(1) or m.group(2)
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        yield [n.strip() for n in selector.split("|")], m.start(), text[m.end() : i - 1]


def _aliases():
    """Every `{BASE as ALIAS}` include across workflows/ and subworkflows/."""
    found = []
    for d in NF_DIRS:
        for nf in sorted(d.rglob("*.nf")):
            for base, alias in ALIAS_INCLUDE_RE.findall(nf.read_text()):
                found.append((base, alias, nf.relative_to(ROOT)))
    return found


def test_scan_finds_the_known_aliases():
    """A scan that matched nothing would pass every rule below vacuously."""
    aliases = {alias for _base, alias, _src in _aliases()}
    for expected in ("SPLIT_PRIOR_PYRAMID", "SEG_QC_SEGMENT"):
        assert expected in aliases, (
            f"{expected} is an aliased include in this repo but the scan missed it -- "
            f"ALIAS_INCLUDE_RE has gone stale. Found: {sorted(aliases)}"
        )


def test_aliases_override_an_inherited_publishdir():
    """Every alias of a publishing process sets its own publishDir, after the base's."""
    text = MODULES_CONFIG.read_text()
    blocks = list(_with_name_blocks(text))

    def publishing_blocks(name):
        return [
            (offset, body)
            for names, offset, body in blocks
            if name in names and PUBLISH_DIR_ASSIGN_RE.search(body)
        ]

    failures = []
    for base, alias, src in _aliases():
        base_pub = publishing_blocks(base)
        if not base_pub:
            continue  # nothing inherited, nothing to override
        alias_pub = publishing_blocks(alias)
        if not alias_pub:
            failures.append(
                f"{src}: `{base} as {alias}` inherits {base}'s publishDir "
                f"(Nextflow matches the selector on the ORIGINAL name too), but no "
                f"`withName: '{alias}'` block sets publishDir. Both would publish into "
                f"the same directory."
            )
            continue
        last_base = max(offset for offset, _ in base_pub)
        first_alias = min(offset for offset, _ in alias_pub)
        if first_alias < last_base:
            failures.append(
                f"conf/modules.config: `withName: '{alias}'` sets publishDir at offset "
                f"{first_alias}, BEFORE `withName: '{base}'` at {last_base}. Nextflow "
                f"applies later selectors on top, so {base}'s publishDir wins back for "
                f"{alias}. Move the {alias} block after it."
            )

    assert not failures, "\n".join(failures)
