"""`docs/parameters.md` must document exactly the schema's parameter surface.

The page is hand-maintained and rots silently in two ways: a new parameter is
never documented, or a removed parameter's row survives and advertises a flag
that is rejected at launch.

A parameter counts as documented only when it is the first cell of a Markdown
table row, in backticks. Prose mentioning a name does not count.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "parameters.md"

# Parameters a RUN never sets, each with its reason (none at present). Verified below
# to still exist in the schema, so a stale entry cannot hide an undocumented parameter.
NOT_RUN_FACING: dict[str, str] = {}

_ROW = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|", re.M)
# Same row, plus its second (default) cell -- for the literal-default check below.
_ROW_WITH_DEFAULT = re.compile(
    r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|\s*([^|]*)\|", re.M
)
_LITERAL_CELL = re.compile(r"^`([^`]*)`$")


def _check_param_consistency():
    """Load the reader that already owns nextflow.config's params{} block --
    tests/check_param_consistency.py, imported the way
    tests/test_figures_match_the_pipeline.py already does."""
    spec = importlib.util.spec_from_file_location(
        "check_param_consistency", REPO / "tests" / "check_param_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    spec.loader.exec_module(module)
    return module


def _schema_params():
    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    names = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(k for k, v in props.items() if isinstance(v, dict))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert len(names) > 50, (
        f"only {len(names)} schema params found -- the walk is broken"
    )
    return names


def _documented():
    rows = set(_ROW.findall(DOC.read_text()))
    assert len(rows) > 50, (
        f"only {len(rows)} parameter rows parsed out of docs/parameters.md -- the row "
        "shape changed and every check below would pass vacuously"
    )
    return rows


def test_every_schema_parameter_has_a_row():
    missing = sorted(_schema_params() - _documented() - set(NOT_RUN_FACING))
    assert not missing, (
        "nextflow_schema.json declares parameters docs/parameters.md does not "
        "document: " + ", ".join(missing) + ". The page is hand-maintained; add a "
        "row in the group the parameter belongs to."
    )


def test_no_row_documents_a_parameter_that_no_longer_exists():
    stale = sorted(_documented() - _schema_params())
    assert not stale, (
        "docs/parameters.md documents parameters the schema no longer declares: "
        + ", ".join(stale)
        + ". A run passing one of these is rejected at launch, so the page is "
        "advertising a flag that does not work."
    )


def test_the_exemptions_are_not_stale():
    schema = _schema_params()
    for name, reason in NOT_RUN_FACING.items():
        assert reason.strip(), f"{name} has an empty reason"
        assert name in schema, (
            f"NOT_RUN_FACING names {name}, which the schema no longer declares -- "
            "the exemption is dead and would hide the next undocumented parameter"
        )


def test_every_literal_doc_default_matches_nextflow_config():
    """`nextflow.config` is the single source of truth for defaults (its own
    comment, and this page's standfirst, both say so). A row whose default cell
    is a clean backtick literal -- most of them -- must parse to the SAME value
    nextflow.config declares. A narrative cell (a per-tier default, or
    "required, no default") documents a null config default in prose instead of
    restating `null`; a literal parser cannot judge prose, so those are exempt.
    """
    module = _check_param_consistency()
    config_defaults = module.extract_config_defaults(
        (REPO / "nextflow.config").read_text()
    )
    compared, mismatches = 0, []
    for name, cell in _ROW_WITH_DEFAULT.findall(DOC.read_text()):
        literal = _LITERAL_CELL.match(cell.strip())
        if not literal or name not in config_defaults:
            continue
        doc_value = module.parse_groovy_literal(literal.group(1))
        if isinstance(doc_value, module.Unparsed):
            continue
        compared += 1
        config_value = config_defaults[name]
        # A bare doc cell (`1`, no quotes) that looks numeric parses as an int/
        # float even when nextflow.config quotes it as a String (gpu_type='1'):
        # the doc's convention is to show string defaults unquoted throughout,
        # so accept a numeric doc value against its own string form too.
        same_stringified = (
            not isinstance(doc_value, bool)
            and isinstance(doc_value, (int, float))
            and isinstance(config_value, str)
            and str(doc_value) == config_value
        )
        if doc_value != config_value and not same_stringified:
            mismatches.append(
                f"{name}: doc says {doc_value!r}, nextflow.config says {config_value!r}"
            )
    assert compared > 50, (
        f"only {compared} doc defaults were comparable -- the literal-cell parse "
        "changed shape and this check would pass vacuously"
    )
    assert not mismatches, (
        "docs/parameters.md's default disagrees with nextflow.config for:\n  "
        + "\n  ".join(mismatches)
    )


def test_the_page_still_says_it_is_hand_maintained():
    """Literal default cells are checked against nextflow.config (see
    test_every_literal_doc_default_matches_nextflow_config), but a NARRATIVE
    default cell and every description are still a human's job, and the page
    must keep telling the reader that."""
    text = DOC.read_text()
    assert "not</em> machine-checked" in text or "not* machine-checked" in text, (
        "docs/parameters.md no longer warns that narrative defaults and "
        "descriptions are hand-maintained, not machine-checked."
    )
