"""lib/RegBackends.groovy is the one place a registration backend is named.

Five places used to decide it independently: nextflow_schema.json's
registration_method enum, workflows/mirage.nf's add_cycle allowlist (dev branch),
subworkflows/local/register_patient.nf's adapter dispatch,
subworkflows/local/seg_qc.nf's join-shape branch, and lib/WarpBackends.groovy's
own two-key map. lib_probe.nf covers the Groovy side (the table's own accessors,
and that RegBackends and WarpBackends declare the same set). This file covers
the two things a Nextflow script cannot reach:

1. The SCHEMA agreement. nextflow_schema.json's enum is not generated from
   anything -- there is no codegen step in this repo -- so nothing structurally
   stops it drifting from the table. A drift in one direction makes a valid
   backend unlaunchable; in the other it lets --registration_method through to a
   dispatch that has no arm for it. tests/test_step_vocabulary_consistency.py
   solves the identical problem for ParamUtils.STEPS and this is that pattern,
   applied to the backend table.

2. That the three MIGRATED call sites hold no method-name literal any more. A
   surviving `method == 'tiled'` is not merely untidy: it is a fourth answer to
   "which backend is this?", free to disagree with the table, and it is exactly
   the shape the seg_qc.nf branch had before it moved.

WHY THE GROOVY TABLE IS PARSED FROM SOURCE. There is no way to call a Groovy
static from pytest, and adding a JVM to the Python suite for two lists would be
out of all proportion. tests/test_step_vocabulary_consistency.py parses
ParamUtils.STEPS the same way and for the same reason; the parse goes through
tests.nfmodel.strip_comments (view B), which blanks explanations while keeping
every quoted literal intact -- so a comment naming a method cannot be read as a
table row, and a table row inside a string stays visible.

WATCHED FAILING. See the break-it steps in the plan that introduced this file:
the enum and the table were pulled apart in both directions, and a literal
comparison was reintroduced into each of the three call sites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.nfmodel import strip_comments

ROOT = Path(__file__).resolve().parent.parent
REG_BACKENDS_PATH = ROOT / "lib" / "RegBackends.groovy"
WARP_BACKENDS_PATH = ROOT / "lib" / "WarpBackends.groovy"
SCHEMA_PATH = ROOT / "nextflow_schema.json"

#: Call sites that must not name a backend method themselves.
MIGRATED_CALL_SITES = (
    "subworkflows/local/register_patient.nf",
    "subworkflows/local/seg_qc.nf",
    # workflows/mirage.nf's add_cycle allowlist is the third site on the dev branch.
)

#: The exact call each site must make, one needle per site. A bare "RegBackends."
#: substring is also satisfied by a doc line that only names the class.
CALL_SITE_NEEDLES = {
    "subworkflows/local/register_patient.nf": "RegBackends.of(method).adapter",
    "subworkflows/local/seg_qc.nf": "RegBackends.segQcJoin(method)",
}

_BACKENDS_BLOCK = re.compile(
    r"static\s+final\s+Map<String,\s*Map>\s+BACKENDS\s*=\s*\[(.*?)\]\.asImmutable\(\)",
    re.S,
)
_WARP_BLOCK = re.compile(
    r"private\s+static\s+final\s+Map<String,\s*Map>\s+BACKENDS\s*=\s*\[(.*?)\]\.asImmutable\(\)",
    re.S,
)
#: A top-level `name: [` key in one of those blocks. Anchored to the line start so a
#: nested `valis: [...]` value or a `warp: 'valis'` field is not counted as a backend.
_TOP_LEVEL_KEY = re.compile(r"^\s{4,8}([a-z][a-z0-9_]*)\s*:\s*\[", re.M)

#: A comparison against a backend-name literal: `method == 'tiled'`, `'valis' != x`, or
#: the membership form `method in ['tiled']`.
_LITERAL_COMPARISON = re.compile(
    r"(?:==|!=)\s*'(valis|tiled)'|'(valis|tiled)'\s*(?:==|!=)"
    r"|\bin\s*\[\s*'(valis|tiled)'"
)


def _block(path: Path, pattern: re.Pattern) -> str:
    text = strip_comments(path.read_text())
    match = pattern.search(text)
    assert match, (
        f"could not locate the BACKENDS table in {path.relative_to(ROOT)}. Did it get "
        "renamed or reshaped? This parse is the only thing tying the table to the schema."
    )
    return match.group(1)


def reg_backend_methods() -> list[str]:
    """The ordered backend names in lib/RegBackends.groovy's BACKENDS table.

    Also used by tests/test_ashlar_backend_removed.py, so the list is not restated.
    """
    return _TOP_LEVEL_KEY.findall(_block(REG_BACKENDS_PATH, _BACKENDS_BLOCK))


def warp_backend_methods() -> list[str]:
    """The ordered backend names in lib/WarpBackends.groovy's BACKENDS table."""
    return _TOP_LEVEL_KEY.findall(_block(WARP_BACKENDS_PATH, _WARP_BLOCK))


def _schema_registration_method_enums() -> list[list]:
    """Every registration_method enum in the schema, wherever it lives."""
    schema = json.loads(SCHEMA_PATH.read_text())
    found = []

    def walk(node):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict) and "registration_method" in props:
            enum = props["registration_method"].get("enum")
            if enum is not None:
                found.append(enum)
        for value in node.values():
            walk(value)

    walk(schema)
    return found


def test_the_parse_is_not_vacuous():
    """Tripwire: the tables parse to names, the schema has an enum, and the call-site
    files exist."""
    assert len(reg_backend_methods()) >= 2, (
        f"RegBackends parsed to {reg_backend_methods()} -- the regex or the table broke"
    )
    assert len(warp_backend_methods()) >= 2, (
        f"WarpBackends parsed to {warp_backend_methods()} -- the regex or the table broke"
    )
    assert _schema_registration_method_enums(), (
        "registration_method has no enum in nextflow_schema.json at all"
    )
    for rel in MIGRATED_CALL_SITES:
        assert (ROOT / rel).is_file(), f"{rel} does not exist -- the list is stale"
    assert set(CALL_SITE_NEEDLES) == set(MIGRATED_CALL_SITES), (
        "CALL_SITE_NEEDLES and MIGRATED_CALL_SITES have drifted apart -- every "
        "migrated call site needs exactly one needle"
    )


def test_the_literal_comparison_regex_actually_matches_one():
    """The literal-comparison needle fires on equality, inequality and membership
    forms (`method in ['tiled']`)."""
    assert _LITERAL_COMPARISON.search("if (method == 'tiled') {")
    assert _LITERAL_COMPARISON.search("if (params.registration_method != 'valis') {")
    assert _LITERAL_COMPARISON.search(
        "def adapter = (method in ['tiled']) ? 'TILED_ADAPTER' : 'VALIS_ADAPTER'"
    )
    # ... and it must not fire on the table's own field values, which are legitimate.
    assert not _LITERAL_COMPARISON.search("warp: 'valis'],")
    assert not _LITERAL_COMPARISON.search("adapter: 'TILED_ADAPTER',")
    assert not _LITERAL_COMPARISON.search("supportedModes: ['linear'],")


def test_schema_enum_matches_the_backend_table():
    methods = reg_backend_methods()
    for enum in _schema_registration_method_enums():
        assert enum == methods, (
            "nextflow_schema.json's registration_method enum has drifted from "
            "lib/RegBackends.groovy's BACKENDS table.\n"
            f"  schema:      {enum}\n"
            f"  RegBackends: {methods}\n"
            "The table is the source of truth and the schema is NOT generated from it. "
            "A name in the schema but not the table reaches a dispatch with no arm for "
            "it; a name in the table but not the schema cannot be launched at all."
        )


def test_the_two_backend_tables_declare_the_same_set():
    """WarpBackends keys the reg_qc=2 warp; RegBackends keys registration itself.
    A method in one and not the other is a run that registers and then cannot be
    scored, or a warp configured for a backend that cannot run.

    tests/lib_probe.nf checks the runtime values agree; this checks the source tables."""
    assert sorted(reg_backend_methods()) == sorted(warp_backend_methods()), (
        f"RegBackends declares {reg_backend_methods()} and WarpBackends declares "
        f"{warp_backend_methods()}"
    )


def test_no_migrated_call_site_compares_against_a_method_literal():
    """Each call site makes its exact RegBackends call (CALL_SITE_NEEDLES).

    The negative rule alone is satisfied by a site that never mentions RegBackends,
    e.g. a ternary on `method.startsWith('t')`; the per-site needle catches that.
    """
    offenders = []
    silent = []
    for rel in MIGRATED_CALL_SITES:
        body = strip_comments((ROOT / rel).read_text())

        needle = CALL_SITE_NEEDLES[rel]
        if needle not in body:
            silent.append(f"{rel} (expected {needle!r})")

        for match in _LITERAL_COMPARISON.finditer(body):
            line = body[: match.start()].count("\n") + 1
            offenders.append(f"{rel}:{line} -> {match.group(0).strip()}")

    assert not silent, (
        "these migrated call sites no longer make the expected RegBackends call:\n  "
        + "\n  ".join(silent)
        + "\nA call site that never names RegBackends can decide the backend any way "
        "it likes -- including reintroducing the exact literal dispatch this table was "
        "built to replace -- without tripping the comparison-shape check below."
    )

    assert not offenders, (
        "these files compare against a registration-method name directly:\n  "
        + "\n  ".join(offenders)
        + "\nThat is a fourth answer to 'which backend is this?', free to disagree "
        "with lib/RegBackends.groovy. Ask the table instead: RegBackends.of(method) "
        "for the adapter, RegBackends.segQcJoin(method) for the seg-QC join shape, "
        "RegBackends.supportsMode(method, mode) for the run-mode allowlist."
    )
