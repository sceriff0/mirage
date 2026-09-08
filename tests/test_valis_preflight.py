"""bin/utils/valis_preflight.py: refuse, BEFORE the JVM starts, the input VALIS 1.0.0
(through 1.2.0) cannot register.

The defect, reproduced 2026-09-08 with VALIS's own geometry code on the dimensions from
a failed TMA run: `prep_images_for_large_non_rigid_registration` picks the pyramid level
to read for each slide as `np.where(level_max < needed_src_dim)[0][0] - 1`. When even
level 0 (full resolution) is smaller than the source dimension the non-rigid stage
needs, that is `-1`. `slide2vips(-1)` then sizes the tile grid from
`slide_dimensions[-1]` (Python negative index: the SMALLEST pyramid level, silently) and
every tile thread calls Bio-Formats `setResolution(-1)`, which throws
IllegalArgumentException. The reader swallows it (`print(e); pass`), leaves `tile`
unbound, and `UnboundLocalError: local variable 'tile' referenced before assignment`
propagates to `Valis.register()`'s catch-all, which kills the JVM and returns
`(None, None, None)`. The pipeline then fails 158 s later with "JVM is not running"
and advice about micro-registration, which is unrelated.

VALIS's own clamp ("Requested size ... was 4096. However, not all images are this
large. Setting max_non_rigid_registration_dim_px to 2720") does NOT prevent it: the
source dimension it then needs is `processed_max * s`, and `s` is chosen so that the
reference's processed frame (or, with `create_masks=True`, the tissue-mask bounding
box, which is smaller) reaches the clamped value -- so the smallest slide always comes
out at least one pixel short (2721 vs 2720, ceil) and with a mask much more (4096 vs
2720 for a mask covering 60 % of the frame). The clamp firing is therefore a
deterministic predictor of the crash, and it fires exactly when some slide's level-0
largest dimension is below the requested non-rigid size. The `<=` here also covers the
equal case, which the ceil makes fail in practice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import tifffile
from utils.valis_preflight import (
    level0_max_dims,
    refusal_message,
    slides_too_small_for_non_rigid,
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


def test_the_refusal_names_the_slide_the_size_and_the_two_remedies():
    msg = refusal_message([("/x/001_003_small.ome.tif", 2720)], non_rigid_dim=4096)
    assert "001_003_small.ome.tif" in msg
    assert "2720" in msg and "4096" in msg
    # remedy 1: a smaller non-rigid size, below the smallest slide
    assert "--max-non-rigid-dim" in msg and "2719" in msg
    # and the quantified margin, not "leave some margin": half the smallest slide
    assert "1360" in msg and "tissue-mask extent" in msg
    assert "memory_mode" in msg and "custom" in msg
    # remedy 2: the other backend
    assert "registration_method" in msg and "tiled" in msg
    # and the truth about why, so nobody reads it as a memory problem
    assert "setResolution(-1)" in msg
    assert "micro" not in msg.lower()


def test_register_py_runs_the_preflight_before_it_builds_the_registrar():
    """The check has to run before `registration.Valis(...)`, which starts the JVM
    and reads every slide -- a check after that point has already paid for the
    rigid stage the crash then throws away."""
    src = (ROOT / "bin" / "register.py").read_text()
    assert "from utils.valis_preflight import" in src or "valis_preflight" in src, (
        "bin/register.py does not import the preflight"
    )
    call = src.index("slides_too_small_for_non_rigid(")
    build = src.index("registration.Valis(")
    assert call < build, "the preflight runs AFTER the registrar is built"


def test_register_py_treats_a_none_error_df_as_the_failure_it_is():
    """`Valis.register()` never raises: on any exception it prints the traceback,
    kills the JVM and returns (None, None, None). bin/register.py used to log
    'Initial registration completed' on that and fail 158 s later on the dead JVM
    with advice about --micro-reg 0."""
    tree = ast.parse((ROOT / "bin" / "register.py").read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    )
    body = ast.unparse(fn)
    call = body.index("registrar.register()")
    check = body.find("error_df is None", call)
    assert check != -1, "no `error_df is None` check after registrar.register()"
    raise_pos = body.find("raise RuntimeError", check)
    completed = body.find("Initial registration completed", call)
    assert raise_pos != -1 and raise_pos < completed, (
        "the None check must raise BEFORE anything is logged as completed"
    )


def test_the_micro_reg_advice_is_gone():
    """'Try --micro-reg 0' was the diagnosis for every dead JVM. The micro pass is
    consumed only by register_micro(), which had not run yet in the failing case."""
    src = (ROOT / "bin" / "register.py").read_text()
    assert "Try --micro-reg 0" not in src
