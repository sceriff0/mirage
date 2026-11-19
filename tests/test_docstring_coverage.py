"""Every public definition under bin/ carries a docstring.

Public means ``not name.startswith("_")``. Scope is ``bin/**/*.py`` minus
``bin/utils/cse/``, the vendored CellSegmentationEvaluator subset that
``ruff.toml`` excludes for the same reason: it is held to its upstream style,
not ours.

Nested functions are exempt structurally: the walk descends into module and class
bodies only, so a small closure inside a function is never visited.

Style is numpy (Parameters / Returns); only presence is checked, not style.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "bin"
VENDORED = BIN / "utils" / "cse"


def _files() -> list[Path]:
    """Every non-vendored Python file under bin/, sorted."""
    return [p for p in sorted(BIN.rglob("*.py")) if VENDORED not in p.parents]


def _misses() -> list[str]:
    """`path:line name` for every public module/class/function without a docstring."""
    out: list[str] = []

    def visit(body, path: Path, prefix: str) -> None:
        for node in body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                rel = path.relative_to(REPO).as_posix()
                out.append(f"{rel}:{node.lineno} {prefix}{node.name}")
            if isinstance(node, ast.ClassDef):
                visit(node.body, path, f"{prefix}{node.name}.")

    for path in _files():
        tree = ast.parse(path.read_text(), filename=str(path))
        if not ast.get_docstring(tree):
            out.append(f"{path.relative_to(REPO).as_posix()}:1 <module>")
        visit(tree.body, path, "")
    return out


def test_the_walk_actually_visited_the_tree():
    """A scope glob that matches nothing reports zero misses and passes vacuously."""
    files = {p.relative_to(REPO).as_posix() for p in _files()}
    assert len(files) >= 40, sorted(files)
    assert {
        "bin/convert_image.py",
        "bin/register.py",
        "bin/segment.py",
        "bin/utils/logger.py",
        "bin/utils/cell_pairs.py",
    } <= files, sorted(files)
    assert not any(f.startswith("bin/utils/cse/") for f in files), (
        "vendored code in scope"
    )


def test_no_public_definition_lacks_a_docstring():
    """The coverage assertion itself."""
    misses = _misses()
    assert not misses, (
        f"{len(misses)} public definition(s) under bin/ have no docstring:\n  "
        + "\n  ".join(misses)
    )
