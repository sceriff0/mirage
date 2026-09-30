"""The `stare` package's copies of pipeline helpers cannot drift from the originals.

``packages/stare`` must import nothing from the pipeline (it is installed on its own,
in containers/tiled and, eventually, from its own repository), so it carries COPIES of
the few helpers its stages reached into ``bin/utils`` for:

  * ``stare/log.py``      <- ``bin/utils/logger.py``     (``configure_logging``, ``get_logger``)
  * ``stare/ome.py``      <- ``bin/utils/ome_io.py``     (``ome_metadata``, ``ome_tiff_writer``)
                          <- ``bin/utils/pixel_size.py`` (``resolve_pixel_size`` and the
                             readers behind it)
  * ``stare/slide_io.py`` <- ``bin/utils/tiled_io.py``   (the whole module: the lazy
                             zarr-region reader is pipeline-wide infrastructure -- eight
                             images' scripts read through it -- so the pipeline keeps its
                             own copy rather than importing the package everywhere)

A copy is the thing this repo's guards exist to distrust: two definitions of one rule,
each free to drift, with the drift surfacing as a header a reader misinterprets rather
than as a failure. This file makes each copy CHECKED rather than trusted:

  1. STATICALLY -- every function/class both files define under the same name has the
     same AST, docstrings excluded (the package documents itself in its own words; the
     code must be the same code). ``slide_io`` must be the whole file, byte for byte.
  2. BEHAVIOURALLY -- for the OME copy, the same array written through both writers
     with both metadata builders produces the same parsed header (channel names,
     PhysicalSize) and the same pixels, and both pixel-size resolvers agree on a header
     they wrote and on the three error cases.

Static equality catches an edit to one side; the behavioural check is what proves the
static check is comparing the functions that matter, and would still fail if a shared
constant (``_MICRON``) diverged, which an AST comparison of functions cannot see.
"""

from __future__ import annotations

import ast
import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIN_UTILS = REPO / "bin" / "utils"
STARE = REPO / "packages" / "stare" / "src" / "stare"

sys.path.insert(0, str(REPO / "bin"))
sys.path.insert(0, str(BIN_UTILS))

# (package copy, original, names that must be identical)
COPIES = {
    "log": (
        STARE / "log.py",
        BIN_UTILS / "logger.py",
        ("configure_logging", "get_logger"),
    ),
    "ome<-ome_io": (
        STARE / "ome.py",
        BIN_UTILS / "ome_io.py",
        ("ome_metadata", "ome_tiff_writer"),
    ),
    "ome<-pixel_size": (
        STARE / "ome.py",
        BIN_UTILS / "pixel_size.py",
        (
            "unit_to_um",
            "_to_um",
            "read_ome_pixel_size",
            "PixelSizeError",
            "resolve_pixel_size",
        ),
    ),
}


def _defs(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _without_docstring(node: ast.AST) -> ast.AST:
    """The definition with its docstring statement removed and line numbers zeroed."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        body = [ast.Pass()]
    clone = type(node)(**{**node.__dict__, "body": body})
    for n in ast.walk(clone):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, 0)
    return clone


@pytest.mark.parametrize("key", sorted(COPIES))
def test_every_copied_definition_is_identical_to_the_original(key):
    copy_path, orig_path, names = COPIES[key]
    copy, orig = _defs(copy_path), _defs(orig_path)
    for name in names:
        assert name in copy, f"{copy_path.name} no longer defines {name}"
        assert name in orig, f"{orig_path.name} no longer defines {name}"
        a = ast.dump(_without_docstring(copy[name]))
        b = ast.dump(_without_docstring(orig[name]))
        assert a == b, (
            f"{name} differs between {copy_path.relative_to(REPO)} and "
            f"{orig_path.relative_to(REPO)}. The package copy exists so `stare` imports "
            "nothing from the pipeline; it may not drift. Change both or neither."
        )


def test_the_copied_names_are_the_ones_the_stitch_stage_uses():
    """Non-vacuity: COPIES must name what ``stare.stages.stitch`` actually imports from
    ``stare.ome``/``stare.log``, or the guard pins functions nothing runs."""
    tree = ast.parse((STARE / "stages" / "stitch.py").read_text())
    used = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module in ("stare.ome", "stare.log")
        for alias in node.names
    }
    guarded = set().union(*(names for _, _, names in COPIES.values()))
    assert used, "stitch.py imports nothing from stare.ome/stare.log -- scope is wrong"
    assert used <= guarded, (
        f"stitch.py uses {sorted(used - guarded)}, which COPIES does not pin"
    )


def test_slide_io_is_the_whole_tiled_io_file():
    a = (STARE / "slide_io.py").read_text()
    b = (BIN_UTILS / "tiled_io.py").read_text()
    assert a == b, (
        "packages/stare/src/stare/slide_io.py and bin/utils/tiled_io.py differ. They are "
        "one module kept in two places (the package must not import the pipeline, and the "
        "pipeline's other images must not need the package); copy the change to the other."
    )


def test_the_ome_copies_agree_on_a_written_header(tmp_path):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    ome_io = importlib.import_module("ome_io")
    pixel_size = importlib.import_module("pixel_size")
    stare_ome = importlib.import_module("stare.ome")

    data = (np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8) * 7) % 65535
    names = ["DAPI", "CD3", "CD20"]
    written = {}
    for label, mod in (("pipeline", ome_io), ("stare", stare_ome)):
        out = tmp_path / f"{label}.ome.tiff"
        with mod.ome_tiff_writer(str(out), bigtiff=True, ome=True) as tw:
            tw.write(
                data,
                photometric="minisblack",
                metadata=mod.ome_metadata(names, 0.325),
            )
        with tifffile.TiffFile(str(out)) as tif:
            root = ET.fromstring(tif.ome_metadata)
            pixels = root.find(".//{*}Pixels")
            written[label] = (
                [c.get("Name") for c in pixels.findall("{*}Channel")],
                pixels.get("PhysicalSizeX"),
                pixels.get("PhysicalSizeXUnit"),
                pixels.get("PhysicalSizeY"),
                pixels.get("PhysicalSizeYUnit"),
                tif.asarray().tobytes(),
            )
        # both resolvers read back what they wrote, and agree
        assert pixel_size.resolve_pixel_size("auto", out) == pytest.approx(0.325)
        assert stare_ome.resolve_pixel_size("auto", out) == pytest.approx(0.325)
    assert written["pipeline"] == written["stare"]
    assert written["stare"][0] == names and written["stare"][2] == "µm"


@pytest.mark.parametrize("configured", [None, "", "not-a-number", -1.0])
def test_the_pixel_size_copies_raise_the_same_way(configured):
    pixel_size = importlib.import_module("pixel_size")
    stare_ome = importlib.import_module("stare.ome")
    with pytest.raises(pixel_size.PixelSizeError) as a:
        pixel_size.resolve_pixel_size(configured, "x.ome.tiff")
    with pytest.raises(stare_ome.PixelSizeError) as b:
        stare_ome.resolve_pixel_size(configured, "x.ome.tiff")
    assert str(a.value) == str(b.value)


def test_the_guard_would_notice_a_one_token_change(tmp_path):
    """Watched failing: the AST comparison is not satisfied by a comment or a docstring
    edit alone, and IS failed by a one-token code change."""
    original = ast.parse("def f(x):\n    '''doc'''\n    return x + 1\n").body[0]
    docstring_only = ast.parse(
        "def f(x):\n    '''other doc'''\n    return x + 1\n"
    ).body[0]
    code_changed = ast.parse("def f(x):\n    '''doc'''\n    return x + 2\n").body[0]
    same = ast.dump(_without_docstring(original))
    assert ast.dump(_without_docstring(docstring_only)) == same
    assert ast.dump(_without_docstring(code_changed)) != same
