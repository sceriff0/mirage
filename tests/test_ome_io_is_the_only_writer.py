"""bin/utils/ome_io.py is the only module under bin/ that may write a TIFF.

A closed rule, not an allowlist: the OME metadata (axes, channel names,
PhysicalSizeX/Y and unit) has one owner, so copies cannot drift into headers a
downstream reader misinterprets.

An AST walk, not grep: `tifffile.imwrite` named in a docstring (as in
bin/convert_image.py) is not a call.

A write is any call to `imwrite` or `TiffWriter` whose receiver is bound to the
`tifffile` module: `import tifffile`, a module alias (`import tifffile as tf`),
or a bare name from `from tifffile import imwrite`. `cv2.imwrite` in
bin/utils/qc.py writes a display PNG and is not matched.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"

#: The one module allowed to name tifffile's writers.
THE_OWNER = BIN_DIR / "utils" / "ome_io.py"

#: Vendored third-party code under bin/, kept byte-identical to upstream (the same
#: exclusion ruff.toml carries).
VENDORED = (BIN_DIR / "utils" / "cse",)

WRITER_NAMES = ("imwrite", "TiffWriter")


def _candidate_files():
    out = []
    for path in sorted(BIN_DIR.rglob("*.py")):
        if any(vendor in path.parents for vendor in VENDORED):
            continue
        out.append(path)
    return out


def _tifffile_writer_calls(path: Path):
    """(lineno, rendered_call) for every tifffile write call in `path`.

    Matches three spellings: `tifffile.imwrite(...)` / `tifffile.TiffWriter(...)`; the
    same through a MODULE alias (`import tifffile as tf` -> `tf.imwrite(...)`); and a
    bare `imwrite(...)` / `TiffWriter(...)` that a `from tifffile import ...` brought
    into scope. Only names actually bound by those imports count, so a same-named
    function or a variable called `tf` for another module is not swept up.
    """
    tree = ast.parse(path.read_text())

    bare_names = set()
    mod_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tifffile":
            for alias in node.names:
                if alias.name in WRITER_NAMES:
                    bare_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tifffile":
                    mod_aliases.add(alias.asname or alias.name)

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            receiver = fn.value
            is_tifffile = isinstance(receiver, ast.Name) and receiver.id in mod_aliases
            if is_tifffile and fn.attr in WRITER_NAMES:
                found.append((node.lineno, f"{receiver.id}.{fn.attr}(...)"))
        elif isinstance(fn, ast.Name) and fn.id in bare_names:
            found.append((node.lineno, f"{fn.id}(...)  [from tifffile import ...]"))
    return found


def test_the_scan_actually_reads_files():
    """Three ways this guard could pass while checking nothing: an empty candidate
    list, a bin/ that is not this repo's, or an owner that does not write."""
    candidates = _candidate_files()
    assert len(candidates) >= 20, (
        f"only {len(candidates)} Python file(s) under {BIN_DIR} -- the scan is "
        "mis-rooted, not the tree"
    )
    assert THE_OWNER in candidates, f"{THE_OWNER} is not in the scanned set"


def test_the_owner_really_is_a_writer():
    """A guard whose sole exemption writes nothing is exempting nothing, and every
    'no writer here' answer below would then be vacuous."""
    calls = _tifffile_writer_calls(THE_OWNER)
    assert calls, (
        f"{THE_OWNER.relative_to(REPO_ROOT)} contains no tifffile write call at all. "
        "It is the module every other writer delegates to; if the write moved, this "
        "guard is now pointing at the wrong owner."
    )


@pytest.mark.parametrize(
    "path", [p for p in _candidate_files() if p != THE_OWNER], ids=lambda p: p.name
)
def test_no_file_under_bin_writes_a_tiff_except_ome_io(path):
    calls = _tifffile_writer_calls(path)
    rel = path.relative_to(REPO_ROOT)
    assert not calls, (
        f"{rel} calls tifffile's writer directly at "
        + ", ".join(f"line {ln} ({what})" for ln, what in calls)
        + ". bin/utils/ome_io.py is the only module allowed to: an OME slide goes "
        "through ome_io.write_ome_tiff (which owns the metadata dict, the tiling and "
        "the BigTIFF threshold), a multi-write through ome_io.ome_tiff_writer, and "
        "anything non-OME through ome_io.write_tiff, which forwards every keyword "
        "verbatim so a migrated call keeps its exact behaviour."
    )


def test_a_docstring_mentioning_the_writer_is_not_a_call():
    """A docstring naming `tifffile.imwrite` is not a call.

    bin/convert_image.py's docstrings name it; a text-matching guard would need a
    file exception for that.
    """
    source = "\n".join(
        [
            "def f():",
            '    """Docstring naming tifffile.imwrite and TiffWriter in prose."""',
            "    # tifffile.imwrite(path, data) -- the call this replaced",
            "    return 1",
        ]
    )
    tree = ast.parse(source)
    found = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in WRITER_NAMES
    ]
    assert not found, "the AST walk must not see prose as a call"


def test_cv2_imwrite_is_not_swept_up():
    """bin/utils/qc.py writes a display PNG with cv2.imwrite, not a TIFF; the
    receiver check excludes it without a special case."""
    tree = ast.parse("import cv2\ncv2.imwrite('x.png', d)\n")
    found = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "tifffile"
    ]
    assert not found


def test_a_module_alias_of_tifffile_is_recognized_as_the_receiver(tmp_path):
    """`import tifffile as tf; tf.imwrite(...)` is caught: the receiver is checked
    against names bound to the tifffile module, not the literal string "tifffile"."""
    probe = tmp_path / "probe.py"
    probe.write_text("import tifffile as tf\ntf.imwrite('x.tif', data)\n")
    calls = _tifffile_writer_calls(probe)
    assert calls == [(2, "tf.imwrite(...)")], (
        f"expected the aliased call to be caught at line 2, got {calls}"
    )


def test_an_unrelated_name_that_looks_like_an_alias_is_not_swept_up(tmp_path):
    """The alias check binds to a real `import tifffile as ...` statement; a local
    variable called `tf` for an unrelated object is not flagged."""
    probe = tmp_path / "probe.py"
    probe.write_text("tf = SomeUnrelatedWriter()\ntf.imwrite('x.tif', data)\n")
    assert _tifffile_writer_calls(probe) == []
