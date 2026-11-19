#!/usr/bin/env python3
"""Guard: bin/register.py's importable surface is what the tests claim it is.

Runs without valis installed: bin/register.py's module-level
`from valis import registration` makes it unimportable in CI, so its surface is
checked from source.

The real behavioural coverage for register.py's pure helpers lives in
tests/test_register_helpers.py, which AST-extracts them.
"""

from __future__ import annotations

import ast
from pathlib import Path

REGISTER = Path(__file__).resolve().parent.parent / "bin" / "register.py"

# The functions tests/test_register_helpers.py AST-extracts. If one is renamed or
# removed, that file's extraction silently yields nothing and its tests stop
# covering anything; naming them here makes a rename fail loudly.
EXTRACTED_HELPERS = {"validate_input_slides", "find_reference_image"}


def _top_level_functions() -> set[str]:
    tree = ast.parse(REGISTER.read_text())
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_helpers_extracted_by_test_register_helpers_still_exist():
    """A renamed helper must fail here, not silently empty another file's coverage."""
    missing = EXTRACTED_HELPERS - _top_level_functions()
    assert not missing, (
        f"bin/register.py no longer defines {sorted(missing)}. "
        "tests/test_register_helpers.py AST-extracts these by name; a rename makes "
        "its extraction yield nothing and its assertions cover nothing."
    )


def test_no_test_imports_a_function_register_does_not_define():
    """`merge_first_file` (a name tests once imported) stays undefined, and every
    extracted helper is defined."""
    defined = _top_level_functions()
    assert "merge_first_file" not in defined, (
        "bin/register.py now defines merge_first_file — restore its real test."
    )
    for name in EXTRACTED_HELPERS:
        assert name in defined
