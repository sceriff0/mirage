"""Resolve a ``bin/`` STARE shim to the package source it stands for.

STARE is an installed package (github.com/sceriff0/stare, pinned in requirements/stare.txt;
it lived in ``packages/stare`` until 2026-10-01). Four scripts under ``bin/``
(``tiled_coarse.py``, ``tiled_reg_tile.py``, ``tiled_solve.py``, ``tiled_stitch.py``)
and nine modules under ``bin/utils/`` are SHIMS: a docstring, an import of the package
module, and ``sys.modules[__name__] = _impl``. A guard that reads a bin script's source
to check its argparse flags, its pixel-writing calls or its tifffile keywords would, on
a shim, read a file that contains none of those and pass vacuously -- or fail its own
non-vacuity tripwire. Both happened on the first run after the move.

This module is the ONE definition of "what source does this bin file stand for", derived
by parsing the shim (``from stare.stages import coarse as _impl``) rather than from a
hand-written table, so a shim that starts pointing somewhere else is followed, not
silently mis-attributed. Guards call ``source_of(path)`` on every ``bin/**/*.py`` they
scan, and get back either the same path (an ordinary script) or the package file.

Plain module, not a test; imported by guards under ``tests/``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "bin"


def _installed_package_src() -> Path:
    """The directory holding the INSTALLED `stare` package (site-packages, or an editable
    checkout's src/). Guards read STARE's source from what is installed, which is exactly
    what the pinned release put there -- never from a copy in this repository."""
    spec = importlib.util.find_spec("stare")
    if spec is None or not spec.origin:
        raise ImportError(
            "the `stare` package is not installed: pip install -r requirements/ci.txt "
            "(it pins stare-registration from github.com/sceriff0/stare)"
        )
    return Path(spec.origin).resolve().parent.parent


PACKAGE_SRC = _installed_package_src()


def package_module_file(dotted: str) -> Path | None:
    """The installed file a dotted ``stare.*`` module name resolves to, or None."""
    if not dotted or dotted.split(".")[0] != "stare":
        return None
    base = PACKAGE_SRC.joinpath(*dotted.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def shim_target(path: Path) -> Path | None:
    """The package file a shim re-exports, or None if ``path`` is not a shim.

    A shim is recognised STRUCTURALLY: its module body is a docstring, imports, an
    ``if __name__`` guard at most, and the assignment ``sys.modules[__name__] = <name>``
    where ``<name>`` was bound by ``from stare[.x] import y as <name>``. Anything else --
    a function, a class, a second statement -- makes it a real module and it is returned
    as itself by ``source_of``.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None
    bound: dict[str, str] = {}
    alias_target: str | None = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the docstring
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                dotted = f"{node.module}.{a.name}"
                if package_module_file(dotted):
                    bound[a.asname or a.name] = dotted
            continue
        if isinstance(node, ast.Import):
            if all(a.name == "sys" for a in node.names):
                continue
            return None
        if isinstance(node, ast.If):
            continue  # `if __name__ != "__main__": sys.modules[...] = _impl` / the entrypoint
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.value, ast.Name)
        ):
            alias_target = node.value.id
            continue
        return None
    # the assignment may sit inside the `if __name__ != "__main__"` guard
    if alias_target is None:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.value, ast.Name)
            ):
                alias_target = node.value.id
    if alias_target is None or alias_target not in bound:
        return None
    return package_module_file(bound[alias_target])


def source_of(path: Path) -> Path:
    """``path`` itself for an ordinary bin file; the package file it stands for if a shim."""
    return shim_target(path) or path


def shims() -> dict[Path, Path]:
    """Every shim under ``bin/`` -> the package file it re-exports."""
    out = {}
    for p in sorted(BIN.rglob("*.py")):
        target = shim_target(p)
        if target is not None:
            out[p] = target
    return out
