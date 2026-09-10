"""The cleanup vocabulary has one owner, like every other path decision.

Layout is THE owner of published-path layout -- tests/test_layout.py forbids a
hand-written published path anywhere else, and checks PUBLISHED_KINDS against
conf/modules.config in both directions. Which of those leaves survives a cleaned
run is the same kind of decision, so it lives in the same place rather than as a
literal list repeated across conf/modules.config, ParamUtils and the docs.

The gate at each publishDir still has to be an inlined boolean, because
conf/*.config cannot see lib/*.groovy (CLAUDE.md). What Layout owns is the
DECISION -- which kinds are final -- and this file is what stops the inlined
copies drifting from it.
"""

import json
import re
from pathlib import Path

from tests.nfmodel import strip_comments

ROOT = Path(__file__).resolve().parent.parent

# The comments-blanked, strings-INTACT view. Both matter here: a `//`-commented-out
# FINAL_KINDS or a default sitting inside a comment must not satisfy these
# assertions, and the values being asserted ARE string literals, so the
# fully-stripped view would blank exactly what is under test.
LAYOUT = strip_comments((ROOT / "lib" / "Layout.groovy").read_text())


def _groovy_list(name: str) -> list:
    """Entries of a `static final List<String> NAME = [ ... ]` declaration."""
    m = re.search(rf"{name}\s*=\s*\[(.*?)\]", LAYOUT, re.S)
    assert m, f"{name} is not declared in lib/Layout.groovy"
    return re.findall(r"'([^']+)'", m.group(1))


def test_final_kinds_are_declared_and_are_a_subset_of_published_kinds():
    final = _groovy_list("FINAL_KINDS")
    published = _groovy_list("PUBLISHED_KINDS")
    assert final, "FINAL_KINDS is empty"
    # PUBLISHED_KINDS names three of its entries via constants (PREPROCESSED,
    # REGISTERED, POSTPROCESSED) rather than literals, so this compares only the
    # literals -- the constants are all intermediates by construction, and a
    # FINAL_KINDS entry spelled as one of them would fail here, correctly.
    assert set(final) <= set(published), (
        f"FINAL_KINDS entries that are not published leaves: "
        f"{sorted(set(final) - set(published))}"
    )


def test_intermediate_kinds_is_derived_not_restated():
    """A second hand-written list is how the keep-set map and the analysis
    identity set both drifted. INTERMEDIATE_KINDS must be computed."""
    m = re.search(r"INTERMEDIATE_KINDS\s*=\s*(.+)", LAYOUT)
    assert m, "INTERMEDIATE_KINDS is not declared"
    assert "PUBLISHED_KINDS" in m.group(1) and "FINAL_KINDS" in m.group(1), (
        "INTERMEDIATE_KINDS must be derived from PUBLISHED_KINDS - FINAL_KINDS, "
        f"not restated: {m.group(1).strip()}"
    )


def _schema_entry(name):
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    found = []

    def walk(node):
        if isinstance(node, dict):
            if name in node.get("properties", {}):
                found.append(node["properties"][name])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return found


def test_cleanup_level_enum_matches_layout():
    """The schema enum is not generated -- the same situation
    test_step_vocabulary_consistency.py covers for start/stop. Assert they
    agree, or --cleanup_level accepts a value the pipeline cannot honour."""
    levels = _groovy_list("CLEANUP_LEVELS")
    found = _schema_entry("cleanup_level")
    assert found, "cleanup_level is absent from nextflow_schema.json"
    assert sorted(found[0].get("enum") or []) == sorted(levels), (
        f"schema enum {found[0].get('enum')} != Layout.CLEANUP_LEVELS {levels}"
    )


def test_cleanup_work_is_in_the_schema_too():
    """failUnrecognisedParams is on, so a param missing from the schema is a
    hard launch failure -- for cleanup_work as much as for cleanup_level."""
    found = _schema_entry("cleanup_work")
    assert found, "cleanup_work is absent from nextflow_schema.json"
    assert found[0].get("type") == "boolean", (
        f"cleanup_work should be a boolean in the schema, got {found[0].get('type')!r}"
    )


def test_defaults_are_the_requested_ones():
    """cleanup_level: 'none' is the 2026-09-10 user decision (a run's output
    is re-enterable by default; 'final' is the opt-in cleaning level). It was
    'final' from 2026-08-25. cleanup_work: a successful run deletes its work
    directory (2026-08-25, unchanged)."""
    cfg = strip_comments((ROOT / "nextflow.config").read_text())
    assert re.search(r"^\s*cleanup_level\s*=\s*'none'", cfg, re.M), (
        "cleanup_level no longer defaults to 'none'"
    )
    assert re.search(r"^\s*cleanup_work\s*=\s*true", cfg, re.M), (
        "cleanup_work no longer defaults to true"
    )


def test_the_escape_hatch_survives():
    """'none' is load-bearing, not a convenience: mode='add_cycle' reads the
    prior run's registered/ AND its segmentation masks, so without a level that
    publishes intermediates, cyclic-IF becomes structurally impossible. It is
    also what `--start segmentation` needs."""
    assert "none" in _groovy_list("CLEANUP_LEVELS"), (
        "--cleanup_level=none is gone. add_cycle and --start <step> both re-enter "
        "a prior run's output tree; there is no other level that publishes it."
    )
