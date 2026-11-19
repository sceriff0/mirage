"""bin/utils/valis_preflight.py: the VALIS 1.0.0 (through 1.2.0) pyramid-level -1 defect,
and the clamp that closes it without modifying the VALIS image.

The defect: `prep_images_for_large_non_rigid_registration` picks each slide's pyramid
level as `np.where(level_max < needed_src_dim)[0][0] - 1`, which is `-1` when even level 0
is smaller than the non-rigid stage needs. `slide2vips(-1)` then sizes the tile grid from
the smallest level and every tile thread calls Bio-Formats `setResolution(-1)`, which
throws; the reader swallows it, `tile` stays unbound, and `Valis.register()`'s catch-all
kills the JVM and returns `(None, None, None)`.

The needed source dimension is `max_non_rigid_dim / (tissue-mask extent)`, so it depends
on image content and no size check can predict it; hence a clamp, not a refusal.

`clamp_negative_levels` wraps `BioFormatsSlideReader.slide2vips` / `slide2image` so a
negative level reaches Bio-Formats as 0 (the upstream fix
`max(closest_img_levels[0] - 1, 0)`, applied at the reader). `setResolution()` rejects
every negative level, so only calls that would have crashed change. VALIS's next line,
`resize_img(vips_level_img, src_img_shape_rc)`, upsamples level 0 to the size it asked for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import tifffile
from utils.valis_preflight import (
    clamp_negative_levels,
    level0_max_dims,
    slides_too_small_for_non_rigid,
    upsample_notice,
)

ROOT = Path(__file__).resolve().parent.parent


def _write_ome(path: Path, height: int, width: int, channels: int = 3) -> Path:
    data = np.zeros((channels, height, width), dtype=np.uint16)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CYX"})
    return path


@pytest.fixture
def two_slides(tmp_path):
    small = _write_ome(tmp_path / "001_003_small.ome.tif", 2380, 2720)
    big = _write_ome(tmp_path / "001_000_big.ome.tif", 4600, 5200)
    return small, big


def test_level0_max_dims_reads_the_largest_spatial_axis_per_slide(two_slides):
    small, big = two_slides
    dims = level0_max_dims([str(small), str(big)])
    assert dims == {str(small): 2720, str(big): 5200}


def test_level0_max_dims_ignores_the_channel_axis(tmp_path):
    """A 40-channel cyclic-IF stack must not report 40, and a YXS RGB page must not
    report 3."""
    tall = _write_ome(tmp_path / "stack.ome.tif", 300, 200, channels=40)
    rgb = tmp_path / "rgb.tif"
    tifffile.imwrite(rgb, np.zeros((300, 200, 3), dtype=np.uint8), photometric="rgb")
    dims = level0_max_dims([str(tall), str(rgb)])
    assert dims == {str(tall): 300, str(rgb): 300}


def test_a_slide_no_larger_than_the_non_rigid_size_is_flagged():
    dims = {"small": 2720, "big": 5200}
    assert slides_too_small_for_non_rigid(dims, 4096) == [("small", 2720)]
    # equal is flagged too: VALIS's ceil makes the needed source 1 px larger than the slide
    assert slides_too_small_for_non_rigid(dims, 2720) == [("small", 2720)]
    assert slides_too_small_for_non_rigid(dims, 2719) == []


def test_flagged_slides_come_smallest_first():
    dims = {"b": 3000, "a": 2720, "c": 5200}
    assert slides_too_small_for_non_rigid(dims, 4096) == [("a", 2720), ("b", 3000)]


def test_the_notice_names_the_slide_the_size_and_what_happens_to_it():
    msg = upsample_notice([("/x/001_003_small.ome.tif", 2720)], non_rigid_dim=4096)
    assert "001_003_small.ome.tif" in msg
    assert "2720" in msg and "4096" in msg
    # level 0 is read and upsampled, no crash
    assert "full resolution" in msg and "upsampl" in msg
    # and it must not read as a refusal or a remedy list
    assert "Refusing" not in msg and "Remedies" not in msg


class _FakeBioFormatsReader:
    """The two methods of VALIS 1.0.0's BioFormatsSlideReader that take a pyramid level,
    with Bio-Formats' own rule: `setResolution(no)` throws for `no < 0`
    (loci.formats.FormatReader). The level is the first positional parameter in both
    (valis_lib/slide_io.py:909 and :973), and `slide2vips` reaches `slide2image`
    positionally from its tile threads (:885)."""

    def __init__(self):
        self.seen = []

    def _set_resolution(self, level):
        if level < 0:
            raise ValueError(f"Invalid resolution: {level}")
        self.seen.append(level)

    def slide2vips(
        self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs
    ):
        return self.slide2image(level, series, xywh=xywh, z=z, t=t)

    def slide2image(self, level, series=None, xywh=None, z=0, t=0, *args, **kwargs):
        self._set_resolution(level)
        return level


@pytest.fixture
def reader_cls():
    # A fresh class per test: the clamp patches the class, and tests must not share it.
    return type("BioFormatsSlideReader", (_FakeBioFormatsReader,), {})


def test_the_unpatched_reader_crashes_on_level_minus_one(reader_cls):
    """The fake has teeth: without the clamp it fails exactly where VALIS fails."""
    with pytest.raises(ValueError, match="Invalid resolution: -1"):
        reader_cls().slide2vips(-1)


@pytest.mark.parametrize("method", ["slide2vips", "slide2image"])
@pytest.mark.parametrize("as_keyword", [False, True])
def test_a_negative_level_reaches_bio_formats_as_zero(reader_cls, method, as_keyword):
    patched = clamp_negative_levels(reader_cls)
    assert set(patched) == {"slide2vips", "slide2image"}
    reader = reader_cls()
    call = getattr(reader, method)
    got = call(level=-1) if as_keyword else call(-1)
    assert got == 0 and reader.seen == [0]


@pytest.mark.parametrize("level", [0, 1, 3])
def test_a_valid_level_passes_through_unchanged(reader_cls, level):
    """The clamp may only change calls that would have crashed."""
    clamp_negative_levels(reader_cls)
    reader = reader_cls()
    assert reader.slide2vips(level) == level
    assert reader.slide2image(level, 0, xywh=(0, 0, 5, 5)) == level
    assert reader.seen == [level, level]


def test_a_numpy_integer_level_is_clamped_too(reader_cls):
    """VALIS computes the level with np.where, so it arrives as np.int64."""
    clamp_negative_levels(reader_cls)
    assert reader_cls().slide2vips(np.int64(-1)) == 0


def test_the_clamp_is_idempotent(reader_cls):
    clamp_negative_levels(reader_cls)
    once = reader_cls.slide2vips
    assert clamp_negative_levels(reader_cls) == []
    assert reader_cls.slide2vips is once


def test_a_reader_whose_first_parameter_is_not_level_is_left_alone():
    """A VALIS upgrade that reorders the signature must not have a clamp rewriting the
    wrong argument; it is skipped and the caller is told (an empty list)."""

    class Other:
        def slide2vips(self, xywh=None, *args, **kwargs):
            return xywh

    assert clamp_negative_levels(Other) == []
    assert Other().slide2vips(-1) == -1


def _valis_registration_body() -> str:
    """The function body only, unparsed: the module docstring quotes the VALIS call, so a
    raw-text index would find the documentation, not the code."""
    tree = ast.parse((ROOT / "bin" / "register.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    )
    return ast.unparse(fn)


def test_register_py_clamps_the_bio_formats_reader_before_it_builds_the_registrar():
    src = _valis_registration_body()
    clamp = src.index("clamp_negative_levels(slide_io.BioFormatsSlideReader)")
    build = src.index("registration.Valis(")
    assert clamp < build, "the clamp is applied AFTER the registrar is built"


def test_register_py_no_longer_refuses_small_slides():
    """A slide no larger than the non-rigid size registers with the clamp (level 0,
    upsampled), so the preflight logs a notice instead of refusing."""
    tree = ast.parse((ROOT / "bin" / "register.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    )
    body = ast.unparse(fn)
    assert "refusal_message" not in body
    at = body.index("slides_too_small_for_non_rigid(")
    notice = body.index("upsample_notice(", at)
    assert "raise" not in body[at:notice], "the size check still raises"
    assert "logger.warning(upsample_notice(" in body


def test_register_py_runs_the_preflight_before_it_builds_the_registrar():
    """The check runs before `registration.Valis(...)`, which starts the JVM and
    reads every slide."""
    assert "valis_preflight" in (ROOT / "bin" / "register.py").read_text(), (
        "bin/register.py does not import the preflight"
    )
    src = _valis_registration_body()
    call = src.index("slides_too_small_for_non_rigid(")
    build = src.index("registration.Valis(")
    assert call < build, "the preflight runs AFTER the registrar is built"


def test_register_py_treats_a_none_error_df_as_the_failure_it_is():
    """`Valis.register()` never raises: on any exception it prints the traceback,
    kills the JVM and returns (None, None, None). bin/register.py must fail there,
    not log success and die later on the dead JVM."""
    body = _valis_registration_body()
    call = body.index("registrar.register(")
    check = body.find("error_df is None", call)
    assert check != -1, "no `error_df is None` check after registrar.register()"
    raise_pos = body.find("raise RuntimeError", check)
    completed = body.find("Registration completed", call)
    assert raise_pos != -1 and raise_pos < completed, (
        "the None check must raise BEFORE anything is logged as completed"
    )


def test_the_micro_reg_advice_is_gone():
    """No 'try --micro-reg 0' advice for a dead JVM: the micro pass is consumed only
    by register_micro(), which has not run yet when register() fails."""
    src = (ROOT / "bin" / "register.py").read_text()
    assert "Try --micro-reg 0" not in src
