"""The pixel-writer seam: every writer is declared here, and none may appear without a decision.

The review's Step F opens with "twelve pixel writers with no owner, each re-deciding codec, tile
size, bit depth and where pixel size is recorded, while every reader re-derives what it hopes was
written". Moving them behind one module is a multi-session refactor. What can be done first, and
is what this file does, is make the seam **enforceable where it already is**: every call that
writes pixels is declared below with the decisions it makes, so a new one cannot appear silently
and a changed one shows up as a failing test rather than as a surprise at gigapixel scale.

The measured inventory, at the time of writing:

    writer                          multi-ch  photometric  compression  tile      bigtiff
    convert_image.py                  yes     minisblack   none         2048      yes
    tile_for_basic.py                 yes     minisblack   none         none      yes
    apply_basic_profiles.py           yes     minisblack   zlib         2048      yes
    merge_channels_pyramid.py         yes     minisblack   zstd(param)  tile_size yes
    tiled_stitch.py                   yes     minisblack   none         out_tile  yes
    split_multichannel.py             no      -            zlib         2048      yes
    segment.py / _cellsam / _instanseg no     -            zlib         none      NO
    extract_mask_series.py            no      -            zlib         none      yes
    utils/image_utils.py              generic passed through by the caller

Two inconsistencies that fall straight out of it, recorded rather than fixed here because both
change published bytes:

  * the six segmentation mask writers omit ``bigtiff`` while ``extract_mask_series`` -- writing
    THE SAME masks, read back out of the pyramid -- now sets it;
  * the two largest intermediates, ``convert_image`` and ``tiled_stitch``, are written
    uncompressed -- compression stays out of scope for both. ``convert_image`` USED to be
    untiled too; PERF-PLAN.md measured that an untiled canonical intermediate forecloses
    every windowed read downstream (BaSiC, STARE registration, SPLIT_CHANNELS, QC) at 2%
    wall-clock cost to fix, so it is now tiled at ``CONVERT_TIFF_TILE`` (2048px) -- see
    ``tests/test_convert_streaming_write.py::test_the_write_is_tiled``.

The one rule asserted as a rule, rather than merely recorded, is the multi-channel
``photometric="minisblack"`` precondition -- see below.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# CLAUDE.md's "Verification reality" item 7: a POSITIVE text-matching guard ("does this
# file still write pixels?") is satisfied by a `write_tiff(`/`tifffile.imwrite(` sitting
# in a comment, keeping a dead PIXEL_WRITERS entry alive. `ci_actions.strip_line_comment`
# is the one quote-aware `#`-stripper this repo shares for exactly that, rather than a
# private `split("#")` that would truncate a quoted `#` (e.g. inside a path or f-string).
# `tests/test_layout.py`'s own stripper (the nfmodel helper `strip_comments_and_strings`)
# is Groovy/Nextflow-shaped (`//` and `/* */`) and wrong for this file's Python `#` comments.
ci_actions = importlib.import_module("ci_actions")


def _strip_comments(text: str) -> str:
    """`text` with every `#` line/trailing comment removed, quote-aware."""
    return "\n".join(ci_actions.strip_line_comment(line) for line in text.splitlines())


# file -> (number of pixel-writing call sites, what it writes)
#
# Keyed by file and COUNT rather than by line number: line numbers drift under unrelated edits
# (see tests/test_compartment_mode_routing.py's re-pinning history), while a count still fails
# the moment a writer is added or removed.
PIXEL_WRITERS = {
    "bin/apply_basic_profiles.py": (
        1,
        "the illumination-corrected multi-channel slide",
    ),
    "bin/convert_image.py": (1, "the converted multi-channel slide"),
    "bin/extract_mask_series.py": (
        1,
        "cell/nuclei masks recovered from a prior pyramid",
    ),
    "bin/merge_channels_pyramid.py": (1, "the published QuPath pyramid"),
    "bin/segment.py": (2, "StarDist cell + nuclei masks"),
    "bin/segment_cellsam.py": (2, "CellSAM cell + nuclei masks"),
    "bin/segment_instantseg.py": (2, "InstanSeg cell + nuclei masks"),
    "bin/split_multichannel.py": (1, "one single-channel plane per marker"),
    "bin/tile_for_basic.py": (
        1,
        "the multi-site CZYX pseudo-FOV stack BASICPY fits on",
    ),
    "bin/tiled_stitch.py": (1, "the STARE registered slide"),
    # The seam itself. The regex counts BOTH a def line and a call line whenever they share
    # a name -- `def ome_tiff_writer(` and `def write_ome_tiff(` each match their own pattern
    # too, not just their call sites -- so this is 3 def lines (ome_tiff_writer, write_ome_tiff,
    # write_tiff) plus the 3 real tifffile calls they wrap (TiffWriter(...), the
    # ome_tiff_writer(...) call inside write_ome_tiff, and the tifffile.imwrite(...) call
    # inside write_tiff): 6, measured with _writer_sites("bin/utils/ome_io.py") rather than
    # assumed. This is the ONLY file allowed to name tifffile's writers at all; see
    # tests/test_ome_io_is_the_only_writer.py.
    "bin/utils/ome_io.py": (
        6,
        "the seam: every TIFF this pipeline writes goes through here",
    ),
    "bin/utils/image_utils.py": (
        1,
        "generic helper; the caller supplies the decisions",
    ),
    "bin/utils/qc.py": (2, "QC raster output, not a pipeline artifact"),
}

# Writers that emit a (C, H, W) stack and pass photometric THEMSELVES. Every one of these
# still constructs its own tw.write(...) call, so the flag is theirs to set.
MULTI_CHANNEL_WRITERS = (
    "bin/apply_basic_profiles.py",
    "bin/tile_for_basic.py",
    "bin/merge_channels_pyramid.py",
    "bin/tiled_stitch.py",
)

# ... and the one that DELEGATES the flag. bin/convert_image.py hands its stack to
# ome_io.write_ome_tiff, which sets photometric="minisblack" itself. Naming the delegation
# rather than dropping the file is what keeps the property covered: without the second
# assertion below, moving the flag into the seam would have silently removed
# convert_image.py from this check with nothing put in its place.
DELEGATED_MULTI_CHANNEL_WRITERS = {
    "bin/convert_image.py": "bin/utils/ome_io.py",
}

#: Every route by which a file under bin/ writes pixels. Before bin/utils/ome_io.py
#: existed this was `tifffile.imwrite(` and `TiffWriter(` alone, because those were the
#: only routes there were; ten scripts called them directly. They are now reachable from
#: exactly one file, and the other twelve writers call one of ome_io's three entry
#: points. Both halves are matched here so that the INVENTORY -- which file writes what,
#: and how many times -- survives the seam rather than collapsing to a single row.
#:
#: `TiffWriter(` and the ome_io entry points are receiver-agnostic substrings on purpose
#: -- they never required a `tifffile.` prefix. `imwrite(` alone would be too loose (it
#: would sweep up cv2.imwrite and any other `.imwrite` method), so the ALIAS a file's own
#: `import tifffile [as X]` binds is resolved per file with `_tifffile_module_aliases`
#: below and substituted in -- not a hardcoded `tifffile.` or `tf.` -- so
#: `import tifffile as tf; tf.imwrite(...)` is counted the same as
#: `tifffile.imwrite(...)`. Missed until a reviewer's probe file proved the untampered
#: `tifffile\.imwrite\(` alternative let an aliased call through uncounted.
_WRITE_CALL_STATIC = r"TiffWriter\(|write_tiff\(|write_ome_tiff\(|ome_tiff_writer\("


def _tifffile_module_aliases(path):
    """Names bound to the tifffile module in `path`: always "tifffile" itself, plus
    whatever `import tifffile as X` adds. AST, not a hardcoded alias, because the point
    is to catch whichever name a file actually chose."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {"tifffile"}
    aliases = {"tifffile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tifffile":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _write_call_pattern(path):
    """The _WRITE_CALL alternation, widened with this file's OWN tifffile module
    alias(es) so an aliased `X.imwrite(...)` counts the same as `tifffile.imwrite(...)`."""
    alias_alt = "|".join(re.escape(a) for a in sorted(_tifffile_module_aliases(path)))
    return re.compile(rf"(?:{alias_alt})\.imwrite\(|{_WRITE_CALL_STATIC}")


def _writer_sites(rel):
    path = REPO / rel
    return len(_write_call_pattern(path).findall(_strip_comments(path.read_text())))


def _all_writer_files():
    found = {}
    for path in sorted((REPO / "bin").rglob("*.py")):
        n = len(_write_call_pattern(path).findall(_strip_comments(path.read_text())))
        if n:
            found[path.relative_to(REPO).as_posix()] = n
    return found


def test_no_undeclared_pixel_writer_exists():
    """A new writer must be declared with its decisions, not appear silently."""
    found = _all_writer_files()
    undeclared = sorted(set(found) - set(PIXEL_WRITERS))

    assert not undeclared, (
        "these files write pixels but are not declared in PIXEL_WRITERS. Add them with the "
        "decisions they make (codec, tile, bigtiff, photometric), or route them through an "
        "existing writer:\n  " + "\n  ".join(undeclared)
    )


def test_every_declared_writer_still_exists_and_still_writes():
    """A stale declaration is worse than none: it makes the count check pass vacuously."""
    found = _all_writer_files()
    stale = sorted(set(PIXEL_WRITERS) - set(found))

    assert not stale, (
        "these are declared as pixel writers but no longer write pixels; remove the entry:\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.parametrize("rel", sorted(PIXEL_WRITERS))
def test_the_number_of_writers_per_file_is_unchanged(rel):
    expected, what = PIXEL_WRITERS[rel]

    assert _writer_sites(rel) == expected, (
        f"{rel} ({what}) now has {_writer_sites(rel)} pixel-writing call sites, not {expected}. "
        "If that is intended, update PIXEL_WRITERS and state what the new writer decides."
    )


@pytest.mark.parametrize("rel", MULTI_CHANNEL_WRITERS)
def test_every_multi_channel_writer_sets_photometric_minisblack(rel):
    """PERF-PLAN.md Wave 0.3: the precondition the whole per-page read strategy rests on.

    Without `photometric="minisblack"`, tifffile stores a (C, H, W) array as ONE page with C
    samples, and both `key=` and `pages[i]` then raise IndexError. Every "read the page, not the
    plane" optimisation downstream depends on this silently.
    """
    text = (REPO / rel).read_text()

    assert 'photometric="minisblack"' in text, (
        f'{rel} writes a multi-channel stack without photometric="minisblack". tifffile will '
        "store it as one page with C samples and pages[i] will raise IndexError."
    )


@pytest.mark.parametrize("rel,owner", sorted(DELEGATED_MULTI_CHANNEL_WRITERS.items()))
def test_a_delegating_multi_channel_writer_has_an_owner_that_sets_the_flag(rel, owner):
    """The delegation must be real in both directions: the caller must not set the flag
    (or it is not delegating), and the owner must.

    The owner-side check matches the KWARG shape (a trailing comma; both real call sites
    in ome_io.py have another keyword after ``photometric=`` so this is structural, not a
    style guess) rather than the bare ``photometric="minisblack"`` substring. ome_io.py's
    own module docstring names that bare string in prose while explaining the rule, and a
    text-matching guard checked against the bare substring is satisfied by that prose even
    when the real kwarg is changed to something else -- watched failing this way before
    being narrowed.
    """
    assert 'photometric="minisblack"' not in (REPO / rel).read_text(), (
        f"{rel} sets photometric itself, so it is not delegating -- move it back into "
        "MULTI_CHANNEL_WRITERS"
    )
    assert 'photometric="minisblack",' in (REPO / owner).read_text(), (
        f"{rel} delegates its photometric flag to {owner}, which does not set it. Every "
        "per-page read downstream of the converted slide breaks silently."
    )


def test_minisblack_really_is_what_makes_pages_addressable(tmp_path):
    """Pin the REASON for the rule, so it cannot decay into a style preference.

    This is PERF-PLAN's blocker #2, reproduced: the rule is not cosmetic, and a reader that
    assumes one page per channel is broken by its absence.
    """
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")

    data = np.zeros((3, 32, 32), dtype=np.uint16)

    with_flag = tmp_path / "with.tiff"
    without_flag = tmp_path / "without.tiff"
    tifffile.imwrite(str(with_flag), data, photometric="minisblack")
    tifffile.imwrite(str(without_flag), data)

    with tifffile.TiffFile(str(with_flag)) as tif:
        assert len(tif.series[0].pages) == 3, (
            "one page per channel is the property we rely on"
        )
        assert tif.series[0].pages[2].asarray().shape == (32, 32)

    with tifffile.TiffFile(str(without_flag)) as tif:
        n_pages = len(tif.series[0].pages)

    assert n_pages != 3, (
        'writing without photometric="minisblack" produced one page per channel anyway on this '
        "tifffile version, so this test no longer demonstrates why the rule exists. Re-check "
        "against the pinned container version before relaxing anything."
    )


# ---------------------------------------------------------------------------
# The inconsistency the inventory exposed
# ---------------------------------------------------------------------------

MASK_WRITERS = (
    "bin/segment.py",
    "bin/segment_cellsam.py",
    "bin/segment_instantseg.py",
    "bin/extract_mask_series.py",
)


@pytest.mark.parametrize("rel", MASK_WRITERS)
def test_every_mask_writer_compresses(rel):
    """Label masks are long runs of identical integers; compression is close to free.

    PERF-PLAN measures an uncompressed uint32 label mask against zstd-3 at 6.7x the write time,
    8.4x the read time, and vastly larger. There is no trade-off to weigh.
    """
    assert 'compression="zlib"' in (REPO / rel).read_text(), (
        f"{rel} writes a label mask without compression"
    )


@pytest.mark.parametrize("rel", MASK_WRITERS)
def test_every_mask_writer_sets_bigtiff(rel):
    """The same masks, written by four files, must not disagree about the 4 GB ceiling.

    A 40000x40000 uint32 mask is 6.4 GB before compression. `tiled_stitch.py:118` states the
    rule: classic TIFF's 32-bit offsets overflow past 4 GB. Compression usually keeps a label
    mask under it -- usually is not a contract, and a mask with many labels and little run
    structure is exactly the case that compresses worst AND is largest.
    """
    assert "bigtiff=True" in (REPO / rel).read_text(), (
        f"{rel} writes a full-resolution label mask without bigtiff, while its sibling "
        "extract_mask_series.py -- writing the same masks back out of the pyramid -- sets it"
    )


# ---------------------------------------------------------------------------
# The tile/iterator contract — the one this seam did not know about
# ---------------------------------------------------------------------------
#
# tifffile's iterator mode is TILE-wise, not plane-wise, whenever `tile=` is set: it
# walks `numtiles` items per page and raises ValueError('tile is too large') the moment
# one exceeds a single tile. So a writer that passes BOTH `tile=` AND a generator must
# feed that generator TILES.
#
# convert_image.py fed it whole planes. Nothing caught it, in three separate places:
# this seam checks photometric/compression/bigtiff but not the data argument;
# test_convert_streaming_write.py's fixtures are 24x20, smaller than one 2048 tile, so
# tifffile PADDED them and the tiled-write test passed; and `-stub` never runs the
# script at all. It failed on the first real slide (30552 x 32072), at CONVERT_IMAGE,
# after the scheduler had granted the task its memory.
#
# Every generator fed to a tiled write is declared here with what it yields. A new one
# has to be added deliberately, which is the moment to ask whether it yields tiles.
TILE_FED_GENERATORS = {
    "_iter_tiles": "bin/utils/ome_io.py -- wraps _iter_planes and re-slices each plane",
    "_tiles": "bin/apply_basic_profiles.py -- channel-major, tile-major",
    "_plane_tiles": "bin/merge_channels_pyramid.py -- per-plane tile walk",
    "stream_tiles": "bin/tiled_stitch.py -- warps and emits one out_tile at a time",
}


def _tiled_writes_with_a_generator():
    """(file, lineno, callee) for every write that sets tile= and is fed a call."""
    out = []
    for rel in sorted(set(PIXEL_WRITERS)):
        path = REPO / rel
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("write", "imwrite", "write_tiff"):
                continue
            if not any(k.arg == "tile" for k in node.keywords):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Call):
                callee = first.func
                out.append(
                    (
                        rel,
                        node.lineno,
                        callee.attr
                        if isinstance(callee, ast.Attribute)
                        else getattr(callee, "id", "<expr>"),
                    )
                )
    return out


def test_every_generator_fed_to_a_tiled_write_yields_tiles():
    """A tiled write fed a PLANE generator fails only on an image bigger than one
    tile -- i.e. never in this suite, and always in production."""
    undeclared = [
        (f, ln, c)
        for f, ln, c in _tiled_writes_with_a_generator()
        if c not in TILE_FED_GENERATORS
    ]
    assert not undeclared, (
        "tiled write fed an undeclared generator: "
        + ", ".join(f"{f}:{ln} -> {c}()" for f, ln, c in undeclared)
        + ". tifffile requires TILES here, not planes; a plane generator raises "
        "'tile is too large' on any image larger than one tile. Declare it in "
        "TILE_FED_GENERATORS once you have checked what it yields."
    )


def test_the_declaration_is_not_carrying_dead_entries():
    """A name here that no tiled write uses is an excuse for a call that is gone."""
    live = {c for _, _, c in _tiled_writes_with_a_generator()}
    stale = sorted(set(TILE_FED_GENERATORS) - live)
    assert not stale, (
        f"TILE_FED_GENERATORS names generators no tiled write feeds: {stale}"
    )
