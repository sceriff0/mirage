"""Tests for bin/utils/tiled_io.py's own primitives, independent of any particular caller.

``open_lazy`` is the shared entry point every lazy-zarr-view reader in this repo goes through
(the STARE tiled fan-out, the illumination-correction tiling/apply pair, the two QC sites, the
segmentation readers). Its callers rely on different guarantees it makes --
``tests/test_tiled_coarse_thumbnail.py`` and ``tests/test_tiled_reg_tile_lazy.py`` already pin
the region-read (bounded-memory) side. This file pins the OTHER guarantee: every call opens a
genuinely fresh store, never one shared with a prior call on the same path -- so a caller that
opens, reads and closes in a loop cannot be handed a handle a previous iteration already closed.

That guarantee was originally pinned here because ``bin/preprocess.py`` read its channels from a
``ThreadPoolExecutor`` and gave each worker its own handle. That module is deleted -- illumination
correction now runs as three Nextflow processes around nf-core's BASICPY -- and NO script under
``bin/`` uses a thread pool any more, so nothing in-process depends on it for thread safety
today. It is kept because it is a real, stated property of ``open_lazy``'s contract that every
remaining open/read/close caller depends on, and because the next threaded or forked caller would
otherwise reintroduce the hazard with nothing pinning the property it needs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

from tiled_io import band_rows_for, open_lazy, read_decimated  # noqa: E402


def _write_small_ome_tiff(tmp_path, n_channels=2, h=32, w=32):
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 4000, size=(n_channels, h, w), dtype=np.uint16)
    path = tmp_path / "image.ome.tiff"
    tifffile.imwrite(
        str(path), stack, ome=True, photometric="minisblack", metadata={"axes": "CYX"}
    )
    return path


def test_open_lazy_returns_a_fresh_store_per_call(tmp_path):
    """Two ``open_lazy`` calls on the SAME path must return distinct objects, never the same
    underlying store -- the property ``open_lazy``'s docstring states explicitly. A shared store
    would mean one caller's ``close`` invalidating another's array, and (for any future
    concurrent caller) two workers racing on the same mutable I/O state.
    """
    path = _write_small_ome_tiff(tmp_path)

    arr1, _dtype1, close1 = open_lazy(path)
    arr2, _dtype2, close2 = open_lazy(path)
    try:
        assert arr1 is not arr2, (
            "open_lazy must build a fresh array/store on every call"
        )
        assert close1 is not close2, (
            "each call's close callable must belong to its own store"
        )
        # Not just distinct wrapper objects -- distinct underlying zarr stores, so closing one
        # cannot invalidate reads through the other.
        assert arr1._z is not arr2._z
    finally:
        close1()
        close2()


def _write_single_channel(tmp_path, dtype, h, w, seed=0, name="plane"):
    """A single-channel plain (non-OME) TIFF of ``dtype``, values spanning its usable range."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        hi = min(
            info.max, 60000
        )  # keep uint16 fixtures away from tifffile edge cases at 65535
        arr = rng.integers(0, hi + 1, size=(1, h, w)).astype(dtype)
    else:
        arr = rng.random((1, h, w)).astype(dtype)
    path = tmp_path / f"{name}_{np.dtype(dtype).name}.tiff"
    tifffile.imwrite(str(path), arr, photometric="minisblack")
    return path


@pytest.mark.parametrize("factor", [1, 2, 3, 5])
@pytest.mark.parametrize("source_dtype", [np.uint8, np.uint16, np.float32])
def test_read_decimated_matches_independent_direct_slice(
    tmp_path, factor, source_dtype
):
    """``read_decimated`` must equal ``src[index, ::factor, ::factor]`` computed a DIFFERENT way
    than the function under test: via a direct step-sliced getitem on the lazy view (which goes
    through zarr's own indexing engine), not by re-running read_decimated's own banded loop. A
    height not a multiple of the (deliberately tiny) band size forces a ragged final band -- the
    case the function's own docstring flags as the easiest to get wrong for a pre-allocated
    destination.

    ``read_decimated`` always returns float32, unconditionally (it takes no ``dtype=``
    argument -- see its docstring for why a narrow-dtype variant was tried and reverted).
    ``source_dtype`` here varies only the SOURCE file's on-disk dtype, to prove the pre-
    allocated streaming/striding logic is correct regardless of what native dtype it is
    reading from, not to exercise any caller-selectable output dtype.
    """
    h, w = 37, 21  # h % (any band size forced below) != 0 -> ragged final band
    path = _write_single_channel(tmp_path, source_dtype, h, w)

    tiny_budget = 8 * w * np.dtype(source_dtype).itemsize
    band = band_rows_for(w, factor, tiny_budget)
    assert band < h, "budget must force more than one band so the final band is ragged"
    assert h % band != 0, "fixture must actually exercise a ragged final band"

    src, src_dtype, close = open_lazy(path)
    try:
        assert src_dtype == source_dtype
        got = read_decimated(src, 0, factor, band_bytes=tiny_budget)
        expected = np.asarray(src[0, ::factor, ::factor]).astype(np.float32)
    finally:
        close()

    assert got.dtype == np.float32
    assert expected.dtype == np.float32
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_autoscale_uint16_vs_float32_disagree_at_a_real_triple():
    """Regression pin for the counter-example that overturned this task's original precision
    assumption: ``autoscale_for_display`` does NOT always return the same uint8 output for a
    uint16 input as for the same values widened to float32.

    An earlier version of this test asserted the two were identical over the full uint16
    domain at ``min=0, max=65535`` and concluded (wrongly) that this generalised to every
    ``(min, max)`` pair -- it does not, because the divergence depends on the ``(min, max,
    value)`` triple jointly, and ``min=0, max=65535`` happens to be a clean case. At
    ``min=13286, max=62449, value=52520`` (all reachable real uint16 pixel data) the two
    dtypes disagree by 1:

    - float64 path (uint16 input, numpy true-divide promotes ``(img - img_min) / range_val``
      to float64): quotient is ``39234 / 49163 = 0.7980391758029413``, scaled by 255 gives
      ``203.49998982975`` -- clearly below the ``.5`` boundary, rounds DOWN to 203.
    - float32 path (float32 input, the arithmetic stays float32 throughout): the same
      division rounds, in float32's coarser representable-value spacing, to an EXACT tie at
      ``203.5``. ``np.round``'s round-half-to-even then resolves that spurious tie UP, to 204.

    This is why ``bin/utils/qc.py``'s ``create_registration_qc`` reads its nuclear channel at
    ``read_decimated``'s float32 default and does NOT ask for a narrower uint8/uint16 dtype,
    even though the only consumer of that plane is ``autoscale_for_display`` on the way to
    uint8: a narrow read changes the arithmetic autoscale_for_display runs internally, and
    this pixel is proof it is not merely "different precision" but a real, reachable
    disagreement. This was tried (narrow read + a widen-back cast to correct it) and
    reverted: the memory saving the narrow read appeared to offer does not survive the
    widen-back that correctness requires -- measured (docs/perf/2026-08-26-rss.md) at statistically
    indistinguishable peak RSS from never narrowing at all, on top of the added complexity of
    the cast and its branching. So `qc.py` simply never narrows the read; this test's job is
    to keep it that way by documenting exactly what breaks if someone "optimises" it back in
    without re-deriving (or re-measuring) why it doesn't help.
    """
    import os
    import sys

    # qc.py imports cv2 and skimage unconditionally at module scope, even though this test
    # only needs autoscale_for_display -- skip (not error) if either is unavailable, matching
    # tests/test_registration_qc_lazy_read.py's guard for the same module.
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
        ),
    )
    from qc import autoscale_for_display

    lo, hi, value = 13286, 62449, 52520
    sub_uint16 = np.arange(lo, hi + 1, dtype=np.uint16)
    sub_float32 = sub_uint16.astype(np.float32)
    idx = value - lo
    assert sub_uint16[idx] == value

    # Pin the exact quotient this hinges on, independent of autoscale_for_display, so a
    # future numpy/platform change that happens to close the gap is caught precisely.
    exact_quotient = (value - lo) / (hi - lo)  # python float division == float64
    float32_quotient = np.float32(sub_float32[idx] - sub_float32.min()) / np.float32(
        sub_float32.max() - sub_float32.min()
    )
    assert exact_quotient == pytest.approx(203.49998982975 / 255, abs=1e-12)
    assert float32_quotient * np.float32(255) == np.float32(203.5), (
        "the float32 path must land on an exact .5 tie for this test to mean anything"
    )

    got_uint16 = autoscale_for_display(sub_uint16, method="minmax")
    got_float32 = autoscale_for_display(sub_float32, method="minmax")
    assert got_uint16.dtype == np.uint8
    assert got_float32.dtype == np.uint8
    assert got_uint16[idx] == 203, (
        "float64 (uint16-input) path must round down, away from the tie"
    )
    assert got_float32[idx] == 204, (
        "float32-input path must round the spurious tie up (round-half-to-even, 204 is even)"
    )
    assert got_uint16[idx] != got_float32[idx], (
        "the whole point of this test: uint16 vs float32 input disagree at this real triple"
    )
