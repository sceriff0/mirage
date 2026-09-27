"""Memory-shape + correctness tests for the thumbnail coarse anchor in tiled_coarse.py.

TILED_COARSE was the last caller of the eager ``nuclear_channel(load_channels(path))`` whole-slide
decode that test_tiled_reg_tile_lazy.py already retired from the per-tile step. On top of the
full decode it matched at *native* resolution, which is how patient 052 exhausted all four
attempts with exit 137. The design always specified this step as a THUMBNAIL feature-align to a
global rigid M0, and the thumbnail is what these tests pin.

The bound still matters: the anchor (NCC rotation sweep + refine, ORB fallback) warps and FFTs
the whole plane it is handed ~50 times, and TILED_COARSE's memory request is derived from
``--max-dim``. An unbounded plane is not a slow run, it is an unschedulable one.

Properties pinned here:
  1. The anchor never sees a full-resolution plane: the arrays handed to estimate_anchor have
     their longest side bounded by --max-dim. This is the test that fails against the
     pre-change code.
  2. No whole-slide array is ever materialised: every region read off either slide is a bounded
     row band, not the whole plane.
  3. M0 is written in FULL-RESOLUTION coordinates -- estimating on a decimated pair and
     forgetting to lift the translation column is the obvious way to get this wrong.
  4. ref_h/ref_w stay full-resolution, so the tile plan still covers the whole frame.
  5. Reference and moving slides of *different* sizes share ONE decimation factor; a per-image
     factor would put the two thumbnails at different scales and silently corrupt M0.
"""

from __future__ import annotations

import csv
import json
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
pytest.importorskip("skimage")
pytest.importorskip("scipy")
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

import tiled_coarse  # noqa: E402


def _textured(seed, h, w, sigma=2.0):
    """Blurred noise: structure that survives decimation, so the matcher still finds keypoints."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(h, w)), sigma)
    return (img - img.min()) / (np.ptp(img) + 1e-9)


def _write_pair(tmp_path, n=1024, shift=(16, -12), ref_n=None):
    """A 2-channel reference/moving pair related by a known full-resolution translation.

    ``shift`` is ``(tx, ty)`` in full-resolution pixels, applied to the moving slide, and is
    chosen by callers to be a multiple of the decimation factor so the decimated pair is related
    by exactly ``shift / factor`` and the recovered M0 has no aliasing floor.
    """
    ref_n = ref_n or n
    dapi = _textured(0, ref_n, ref_n)
    marker = _textured(1, ref_n, ref_n)
    ref = (np.stack([marker, dapi]) * 60000).astype(np.uint16)

    # moving = reference content translated by `shift`; np.roll keeps it exact (no interpolation)
    tx, ty = shift
    mov_nuc = np.roll(_textured(0, n, n), (ty, tx), axis=(0, 1))
    mov_marker = np.roll(_textured(1, n, n), (ty, tx), axis=(0, 1))
    mov = (np.stack([mov_marker, mov_nuc]) * 60000).astype(np.uint16)

    ref_f = tmp_path / "ref.ome.tiff"
    mov_f = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref_f), ref, photometric="minisblack")
    tifffile.imwrite(str(mov_f), mov, photometric="minisblack")
    return ref_f, mov_f


def _run(tmp_path, ref_f, mov_f, max_dim, nuclear_index=1, tile=256, halo=32):
    m0_f = tmp_path / "m0.json"
    tiles_f = tmp_path / "tiles.csv"
    tiled_coarse.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--nuclear-index",
            str(nuclear_index),
            "--tile",
            str(tile),
            "--halo",
            str(halo),
            "--max-dim",
            str(max_dim),
            "--out-m0",
            str(m0_f),
            "--out-tiles",
            str(tiles_f),
        ]
    )
    return json.loads(m0_f.read_text()), tiles_f


def test_the_matcher_never_sees_a_full_resolution_plane(tmp_path, monkeypatch):
    """The arrays handed to estimate_anchor must be bounded by --max-dim.

    Fails against the pre-change code, which passed the native-resolution DAPI plane straight
    into the matcher (1024 px here, 8x over the 128 px bound).
    """
    n, max_dim = 1024, 128
    ref_f, mov_f = _write_pair(tmp_path, n=n, shift=(16, -8))

    seen = []
    orig = tiled_coarse.estimate_anchor

    def spy(ref, mov, **kw):
        seen.append((ref.shape, mov.shape))
        return orig(ref, mov, **kw)

    monkeypatch.setattr(tiled_coarse, "estimate_anchor", spy)
    _run(tmp_path, ref_f, mov_f, max_dim=max_dim)

    assert seen, "estimate_anchor was never called"
    for ref_shape, mov_shape in seen:
        assert max(ref_shape) <= max_dim, (
            f"the matcher got a {ref_shape} reference plane, over the {max_dim}px "
            f"thumbnail bound"
        )
        assert max(mov_shape) <= max_dim, (
            f"the matcher got a {mov_shape} moving plane, over the {max_dim}px "
            f"thumbnail bound"
        )
        assert ref_shape == mov_shape, "thumbnails must share one scale for matching"


def test_slides_are_read_lazily_in_bounded_bands(tmp_path, monkeypatch):
    """Both slides must be reached through open_lazy region reads, each within the band budget.

    Fails against the pre-change code in the most direct way available: that code called
    ``tifffile.imread`` via ``load_channels`` and never touched ``open_lazy`` at all, so no
    region read is recorded and ``reads`` is empty.
    """
    from tiled_io import DEFAULT_BAND_BYTES

    n, max_dim = 1024, 128
    ref_f, mov_f = _write_pair(tmp_path, n=n, shift=(16, -8))

    import tiled_io

    orig_open_lazy = tiled_io.open_lazy
    reads = []

    def spying_open_lazy(path):
        arr, dtype, close = orig_open_lazy(path)

        class _Spy:
            def __init__(self, inner):
                self._inner = inner
                self.shape = inner.shape
                self.dtype = inner.dtype

            def __getitem__(self, key):
                _c, ys, xs = key
                reads.append((ys.stop - ys.start) * (xs.stop - xs.start))
                return self._inner[key]

        return _Spy(arr), dtype, close

    monkeypatch.setattr(tiled_coarse, "open_lazy", spying_open_lazy)
    _run(tmp_path, ref_f, mov_f, max_dim=max_dim)

    assert reads, (
        "no region read was recorded -- the coarse step is not going through open_lazy, "
        "i.e. it is still eagerly decoding whole slides via tifffile.imread"
    )
    assert max(reads) * 4 <= DEFAULT_BAND_BYTES, (
        f"a single read materialised {max(reads) * 4} bytes as float32, over the "
        f"{DEFAULT_BAND_BYTES} band budget"
    )


def test_banded_read_is_numerically_identical_to_full_decimation(tmp_path):
    """Multi-band streaming must equal a whole-plane stride exactly.

    Band height is snapped to a multiple of the factor precisely so the sampled rows stay on the
    global 0, f, 2f grid; an unsnapped band restarts the stride at each band boundary and
    silently resamples different rows.
    """
    from tiled_io import band_rows_for, open_lazy, read_decimated

    n, factor = 512, 4
    ref_f, _mov_f = _write_pair(tmp_path, n=n, shift=(0, 0))
    full = tifffile.imread(str(ref_f))[1].astype(np.float32)

    tiny_budget = 8 * n * 4  # ~8 rows per band -> many bands, several per factor stride
    assert band_rows_for(n, factor, tiny_budget) < n, (
        "budget must force more than one band"
    )

    src, _dtype, close = open_lazy(ref_f)
    try:
        got = read_decimated(src, 1, factor, band_bytes=tiny_budget)
    finally:
        close()

    assert np.array_equal(got, full[::factor, ::factor])


def test_band_height_is_snapped_to_the_decimation_factor():
    """Rows per band must be a multiple of the factor, and never zero for a very wide slide."""
    from tiled_io import band_rows_for

    for width in (1024, 30000, 120000):
        for factor in (1, 3, 8, 17):
            rows = band_rows_for(width, factor, 64 * 1024 * 1024)
            assert rows >= factor and rows % factor == 0, (width, factor, rows)
    # a slide so wide that one row blows the budget still yields at least one stride of rows
    assert band_rows_for(10**9, 8, 1024) == 8


def test_m0_is_written_in_full_resolution_coordinates(tmp_path):
    """M0 must map moving -> reference in FULL-RES px, not thumbnail px.

    Estimating on a 4x-decimated pair and returning that matrix verbatim would report a
    translation 4x too small. The linear 2x2 block is scale-invariant; only the translation
    column scales.
    """
    n, max_dim = 1024, 256  # factor 4
    tx, ty = (
        16,
        -12,
    )  # multiples of 4, so the decimated pair is related by exactly (4, -3)
    ref_f, mov_f = _write_pair(tmp_path, n=n, shift=(tx, ty))

    m0_doc, _ = _run(tmp_path, ref_f, mov_f, max_dim=max_dim)
    m0 = np.asarray(m0_doc["M0"], dtype=float)

    # moving pixel p maps to reference pixel M0 @ p; content was shifted by (tx, ty), so the
    # recovered forward map must translate by (-tx, -ty).
    assert m0[0, 2] == pytest.approx(-tx, abs=2.0), (
        f"M0 x-translation {m0[0, 2]} != {-tx}"
    )
    assert m0[1, 2] == pytest.approx(-ty, abs=2.0), (
        f"M0 y-translation {m0[1, 2]} != {-ty}"
    )
    # near-identity rotation/scale
    assert np.allclose(m0[:2, :2], np.eye(2), atol=0.02)
    # how the anchor was found is recorded, and it is the sweep on a clean pair
    assert m0_doc["coarse_method"] == "ncc_sweep"
    assert m0_doc["coarse_peak_ncc"] >= 0.3
    assert abs(m0_doc["coarse_angle_deg"]) < 0.5


def test_ref_dims_and_tile_plan_stay_full_resolution(tmp_path):
    """ref_h/ref_w feed tile_grid -- if they were thumbnail dims the plan would miss the slide."""
    n, max_dim, tile, halo = 1024, 128, 256, 32
    ref_f, mov_f = _write_pair(tmp_path, n=n, shift=(8, -8))

    m0_doc, tiles_f = _run(
        tmp_path, ref_f, mov_f, max_dim=max_dim, tile=tile, halo=halo
    )

    assert m0_doc["ref_h"] == n and m0_doc["ref_w"] == n

    with open(tiles_f) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (n // tile) ** 2, (
        f"tile plan has {len(rows)} tiles, expected full frame"
    )
    assert max(int(r["x1"]) for r in rows) == n
    assert max(int(r["y1"]) for r in rows) == n


def test_reference_and_moving_share_one_decimation_factor(tmp_path, monkeypatch):
    """Differently-sized slides must be decimated by the SAME factor.

    A per-image factor (each independently squeezed under --max-dim) puts the two thumbnails at
    different scales; the matcher would then match across a scale change and M0 would carry a
    bogus scale term that no per-tile refinement can undo.
    """
    ref_f, mov_f = _write_pair(tmp_path, n=512, ref_n=1024, shift=(0, 0))

    seen = []

    # NOT a call-through spy: the two slides here are different random fields (a 512 px draw
    # is not a crop of the 1024 px one), so the real anchor rightly REFUSES them. This case is
    # about the shapes the anchor is handed, so it answers with an identity anchor.
    def spy(ref, mov, **kw):
        seen.append((ref.shape, mov.shape))
        from drape.coarse_align import Anchor

        return Anchor(np.eye(3), 1.0, 0, "ncc_sweep", 1.0, 2.0, 0.0)

    monkeypatch.setattr(tiled_coarse, "estimate_anchor", spy)
    _run(tmp_path, ref_f, mov_f, max_dim=128)

    ref_shape, mov_shape = seen[0]
    # ref is 1024 and mov is 512; one shared factor of 8 gives 128 and 64 respectively.
    assert ref_shape == (128, 128), f"reference thumbnail {ref_shape} != (128, 128)"
    assert mov_shape == (64, 64), (
        f"moving thumbnail {mov_shape} != (64, 64) -- the two slides were decimated by "
        "different factors, so the thumbnails are not at a common scale"
    )


def test_small_slide_is_not_decimated(tmp_path, monkeypatch):
    """A slide already under --max-dim must pass through at factor 1 (no behaviour change)."""
    n = 256
    ref_f, mov_f = _write_pair(tmp_path, n=n, shift=(6, -4))

    seen = []
    orig = tiled_coarse.estimate_anchor

    def spy(ref, mov, **kw):
        seen.append((ref.shape, mov.shape))
        return orig(ref, mov, **kw)

    monkeypatch.setattr(tiled_coarse, "estimate_anchor", spy)
    m0_doc, _ = _run(tmp_path, ref_f, mov_f, max_dim=4096, tile=128, halo=16)

    assert seen[0] == ((n, n), (n, n)), "small slide must not be decimated"
    m0 = np.asarray(m0_doc["M0"], dtype=float)
    assert m0[0, 2] == pytest.approx(-6, abs=1.0)
    assert m0[1, 2] == pytest.approx(4, abs=1.0)


def test_an_unanchorable_pair_fails_the_task_naming_both_slides(tmp_path):
    """Two unrelated slides must not produce an M0 at all: a wrong anchor fails nothing
    downstream (tiles are simply read from the wrong place), so COARSE refuses, loudly, with
    the slide names and the scores in the error, and writes no M0 JSON."""
    from drape.coarse_align import CoarseRefused

    # 512 vs 1024 draws of the textured field are different random fields, not a crop.
    ref_f, mov_f = _write_pair(tmp_path, n=512, ref_n=1024, shift=(0, 0))
    with pytest.raises(CoarseRefused) as ei:
        _run(tmp_path, ref_f, mov_f, max_dim=128)
    msg = str(ei.value)
    assert "REFUSED" in msg and "mov.ome.tiff" in msg and "ref.ome.tiff" in msg, msg
    assert "peak" in msg and "ORB" in msg, msg
    assert not (tmp_path / "m0.json").exists()
