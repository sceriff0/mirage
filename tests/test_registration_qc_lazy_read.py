"""Registration QC must stop eagerly decoding the whole multi-channel slide.

``bin/utils/qc.py:create_registration_qc`` used to call ``tifffile.imread`` on the
*entire* reference and registered stacks (every marker channel, full resolution) just
to slice out the one nuclear/fiducial channel the overlay is built from. On a
gigapixel cyclic-IF slide with a dozen-plus marker channels, that is an order of
magnitude more data decoded than the QC step ever uses -- paid twice, once per slide.

``bin/utils/tiled_io.py`` already has the machinery to read one channel through a
lazy, region-readable zarr view (``open_lazy``) at a chosen decimation
(``decimation_factor`` + ``read_decimated``), proven correct by the STARE coarse-align
path (``tests/test_tiled_coarse_thumbnail.py::test_banded_read_is_numerically_identical_to_full_decimation``).
This file pins two properties for routing ``create_registration_qc`` through it:

1. The whole-stack eager read is gone -- both slides are opened through ``open_lazy``.
2. The result is EXACTLY what the old whole-stack read produced, not merely similar:
   same pixel values, same shape, same dtype, through the untouched
   ``create_nuclear_overlay``/``autoscale_for_display`` pipeline.

Property 2 does not fail against the pre-change code by construction: the pre-change
code already reads the correct channel via ``tifffile.imread``, so an "old vs. new"
pixel comparison built from the SAME resolved index is old-vs-old before the fix
lands, and only exercises the new code path's arithmetic once the fix is in. It is
included anyway as a regression guard against the real failure mode here -- silently
different pixels, not a crash -- and was falsified during development by temporarily
asserting the arrays were NOT equal (see task-1-report.md for the transcript), which
failed until the assertion was reverted, proving the comparison is not vacuous.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

UTILS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

pytest.importorskip("skimage")
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
cv2 = pytest.importorskip("cv2")

import qc  # noqa: E402
from metadata import extract_channel_names_from_ome, pick_nuclear_index  # noqa: E402


def _write_pair(tmp_path, h=96, w=80, channel_names=("SMA", "DAPI")):
    """A small 2-channel OME-TIFF pair with the nuclear marker deliberately NOT at
    index 0, so the test also exercises ``pick_nuclear_index`` rather than the
    channel-0 fallback CLAUDE.md warns is "correct only by accident".
    """
    rng = np.random.default_rng(1234)

    def _plane(seed):
        r = np.random.default_rng(seed)
        return r.integers(0, 65536, size=(h, w), dtype=np.uint16)

    ref = np.stack([_plane(1), _plane(2)]).astype(np.uint16)
    # registered image: same shape (post-registration slides share dimensions), different content
    reg = np.stack([_plane(3), _plane(4)]).astype(np.uint16)
    del rng

    ref_path = tmp_path / "reference.ome.tiff"
    reg_path = tmp_path / "registered.ome.tiff"
    tifffile.imwrite(
        str(ref_path),
        ref,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(channel_names)}},
    )
    tifffile.imwrite(
        str(reg_path),
        reg,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(channel_names)}},
    )
    return ref_path, reg_path


def test_registration_qc_never_eagerly_decodes_the_whole_stack(tmp_path, monkeypatch):
    """Both slides must be reached through ``open_lazy``, not a whole-array ``tifffile.imread``.

    Fails against the pre-change code, which has no ``open_lazy`` attribute on the ``qc``
    module at all -- this assertion cannot even be wired up until the routing lands.
    """
    ref_path, reg_path = _write_pair(tmp_path)
    out_path = tmp_path / "out" / "qc.tif"
    out_path.parent.mkdir()

    seen_paths = []
    orig_open_lazy = qc.open_lazy

    def spying_open_lazy(path):
        seen_paths.append(str(path))
        return orig_open_lazy(path)

    monkeypatch.setattr(qc, "open_lazy", spying_open_lazy)

    imread_full_calls = []
    orig_imread = tifffile.imread

    def spying_imread(path, *args, **kwargs):
        # A lazy zarr open still goes through tifffile.imread(..., aszarr=True); only a
        # plain call (no aszarr) actually materialises pixel data eagerly.
        if not kwargs.get("aszarr", False):
            imread_full_calls.append(str(path))
        return orig_imread(path, *args, **kwargs)

    # qc.py no longer binds the name `tifffile` at all (Task 7 routed its two writes
    # through ome_io.write_tiff and its reads always went through open_lazy/read_decimated),
    # so the spy patches the shared tifffile module object directly -- the same object any
    # `import tifffile; tifffile.imread(...)` in qc.py's call chain would resolve to, were
    # one ever reintroduced.
    monkeypatch.setattr(tifffile, "imread", spying_imread)

    qc.create_registration_qc(
        reference_path=ref_path,
        registered_path=reg_path,
        output_path=out_path,
        scale_factor=0.25,
        save_fullres=True,
        save_png=True,
        save_tiff=True,
    )

    assert str(ref_path) in seen_paths, "reference slide never went through open_lazy"
    assert str(reg_path) in seen_paths, "registered slide never went through open_lazy"
    assert str(ref_path) not in imread_full_calls, (
        "reference slide was still eagerly decoded in full via tifffile.imread"
    )
    assert str(reg_path) not in imread_full_calls, (
        "registered slide was still eagerly decoded in full via tifffile.imread"
    )


def test_qc_overlay_pixels_match_the_old_whole_stack_read(tmp_path):
    """The decimated-read overlay must equal the old full-read overlay bit-for-bit.

    Builds the "old" reference the way ``create_registration_qc`` used to: a plain
    ``tifffile.imread`` of the whole stack, sliced at the ``pick_nuclear_index``-resolved
    channel, pushed through the untouched ``create_nuclear_overlay``. Compares that against
    what the real (patched) ``create_registration_qc`` actually wrote to disk.
    """
    ref_path, reg_path = _write_pair(tmp_path)

    ref_channels = extract_channel_names_from_ome(ref_path)
    reg_channels = extract_channel_names_from_ome(reg_path)
    ref_idx = pick_nuclear_index(ref_channels, None)
    reg_idx = pick_nuclear_index(reg_channels, None)
    assert ref_idx == 1 and reg_idx == 1, (
        "fixture must exercise a non-zero nuclear index"
    )

    # HAZARD: raw-uint16 (float64 path) vs. production's float32 path -- green here only
    # because _write_pair avoids the adversarial triple; see
    # test_tiled_io.py::test_autoscale_uint16_vs_float32_disagree_at_a_real_triple before
    # blaming a real change if this ever goes red.
    old_ref_nuc = tifffile.imread(str(ref_path))[ref_idx]
    old_reg_nuc = tifffile.imread(str(reg_path))[reg_idx]
    expected_bgr, expected_cyx = qc.create_nuclear_overlay(
        old_ref_nuc, old_reg_nuc, scale_factor=0.25
    )

    out_path = tmp_path / "out" / "qc.tif"
    out_path.parent.mkdir()
    qc.create_registration_qc(
        reference_path=ref_path,
        registered_path=reg_path,
        output_path=out_path,
        scale_factor=0.25,
        save_fullres=False,
        save_png=True,
        save_tiff=True,
    )

    got_bgr = cv2.imread(str(out_path.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
    got_cyx = tifffile.imread(str(out_path.with_suffix(".tif")))

    assert got_bgr.dtype == expected_bgr.dtype
    assert got_bgr.shape == expected_bgr.shape
    assert np.array_equal(got_bgr, expected_bgr)

    assert got_cyx.dtype == expected_cyx.dtype
    assert got_cyx.shape == expected_cyx.shape
    assert np.array_equal(got_cyx, expected_cyx)


def test_qc_fullres_output_matches_old_whole_stack_read(tmp_path):
    """The full-resolution TIFF (always produced in production) must also stay bit-exact."""
    ref_path, reg_path = _write_pair(tmp_path)

    ref_channels = extract_channel_names_from_ome(ref_path)
    reg_channels = extract_channel_names_from_ome(reg_path)
    ref_idx = pick_nuclear_index(ref_channels, None)
    reg_idx = pick_nuclear_index(reg_channels, None)

    # HAZARD: raw-uint16 (float64 path) vs. production's float32 path -- green here only
    # because _write_pair avoids the adversarial triple; see
    # test_tiled_io.py::test_autoscale_uint16_vs_float32_disagree_at_a_real_triple before
    # blaming a real change if this ever goes red.
    old_ref_nuc = tifffile.imread(str(ref_path))[ref_idx]
    old_reg_nuc = tifffile.imread(str(reg_path))[reg_idx]
    ref_scaled = qc.autoscale_for_display(old_ref_nuc, method="minmax")
    reg_scaled = qc.autoscale_for_display(old_reg_nuc, method="minmax")
    expected_fullres = np.stack(
        [reg_scaled, ref_scaled, np.zeros_like(ref_scaled)], axis=0
    )

    out_path = tmp_path / "out" / "qc.tif"
    out_path.parent.mkdir()
    qc.create_registration_qc(
        reference_path=ref_path,
        registered_path=reg_path,
        output_path=out_path,
        scale_factor=0.25,
        save_fullres=True,
        save_png=False,
        save_tiff=False,
    )

    fullres_path = out_path.with_name(out_path.stem + "_fullres.tif")
    got_fullres = tifffile.imread(str(fullres_path))

    assert got_fullres.dtype == expected_fullres.dtype
    assert got_fullres.shape == expected_fullres.shape
    assert np.array_equal(got_fullres, expected_fullres)


def test_qc_png_and_fullres_tiff_are_byte_identical_to_the_old_float32_widened_read(
    tmp_path,
):
    """``create_registration_qc``'s own read+render plumbing must match a direct call to the
    same functions -- byte-for-byte, for both the published QC PNG and the full-resolution
    TIFF. (A narrower uint8/uint16 read was also tried at this call site and reverted -- see
    ``bin/utils/qc.py``'s comment above the read and
    ``test_autoscale_uint16_vs_float32_disagree_at_a_real_triple`` in test_tiled_io.py for
    why.)

    NOTE on what this test does and does NOT prove: it builds its "expectation" by calling
    ``read_decimated`` itself (same function under test), so it cannot catch a bug INSIDE
    ``read_decimated``'s pre-allocated-destination streaming loop -- a wrong banded/ragged-band
    computation would corrupt both sides identically and this test would still pass. That
    proof lives elsewhere, built against a genuinely independent code path:
    ``tests/test_tiled_io.py::test_read_decimated_matches_independent_direct_slice`` (expected
    built via a direct step-sliced ``src[...]`` getitem, through zarr's own indexing engine)
    and
    ``tests/test_tiled_coarse_thumbnail.py::test_banded_read_is_numerically_identical_to_full_decimation``
    (expected built via a plain ``tifffile.imread``). What THIS test covers, and is worth
    having independently, is that ``create_registration_qc``'s own orchestration --
    channel-index resolution, ``open_lazy``/``read_decimated`` wiring, and the
    ``create_nuclear_overlay``/``autoscale_for_display``/PNG-TIFF-write chain -- matches
    calling those same pieces directly, with nothing lost or reordered in between.

    Builds the "before" expectation independently, by calling ``read_decimated`` directly
    (its only supported dtype, float32, matching exactly what ``create_registration_qc``
    gets) and pushing it through the SAME untouched ``create_nuclear_overlay`` /
    ``autoscale_for_display`` pipeline the real function uses. Compares that against what the
    real ``create_registration_qc`` actually writes to disk.
    """
    from tiled_io import open_lazy, read_decimated

    ref_path, reg_path = _write_pair(tmp_path)

    ref_channels = extract_channel_names_from_ome(ref_path)
    reg_channels = extract_channel_names_from_ome(reg_path)
    ref_idx = pick_nuclear_index(ref_channels, None)
    reg_idx = pick_nuclear_index(reg_channels, None)

    ref_src, ref_dtype, ref_close = open_lazy(ref_path)
    reg_src, reg_dtype, reg_close = open_lazy(reg_path)
    try:
        assert ref_dtype == np.uint16 and reg_dtype == np.uint16, (
            "fixture must be uint16 -- the source dtype create_registration_qc actually reads"
        )
        old_ref_nuc = read_decimated(ref_src, ref_idx, factor=1)
        old_reg_nuc = read_decimated(reg_src, reg_idx, factor=1)
    finally:
        ref_close()
        reg_close()

    expected_bgr, expected_cyx = qc.create_nuclear_overlay(
        old_ref_nuc, old_reg_nuc, scale_factor=0.25
    )
    ref_scaled = qc.autoscale_for_display(old_ref_nuc, method="minmax")
    reg_scaled = qc.autoscale_for_display(old_reg_nuc, method="minmax")
    expected_fullres = np.stack(
        [reg_scaled, ref_scaled, np.zeros_like(ref_scaled)], axis=0
    )

    out_path = tmp_path / "out" / "qc.tif"
    out_path.parent.mkdir()
    qc.create_registration_qc(
        reference_path=ref_path,
        registered_path=reg_path,
        output_path=out_path,
        scale_factor=0.25,
        save_fullres=True,
        save_png=True,
        save_tiff=False,
    )

    got_bgr = cv2.imread(str(out_path.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
    fullres_path = out_path.with_name(out_path.stem + "_fullres.tif")
    got_fullres = tifffile.imread(str(fullres_path))

    assert got_bgr.dtype == expected_bgr.dtype
    assert got_bgr.shape == expected_bgr.shape
    assert np.array_equal(got_bgr, expected_bgr)

    assert got_fullres.dtype == expected_fullres.dtype
    assert got_fullres.shape == expected_fullres.shape
    assert np.array_equal(got_fullres, expected_fullres)


def test_qc_output_is_exact_at_the_known_adversarial_pixel_triple(tmp_path):
    """Regression guard against reintroducing a narrow (uint8/uint16) read in
    ``create_registration_qc``. See
    ``tests/test_tiled_io.py::test_autoscale_uint16_vs_float32_disagree_at_a_real_triple``
    for the underlying counter-example this fixture is built on.

    ``create_registration_qc`` reads its nuclear channel through ``read_decimated``, which
    always returns float32 (it has no ``dtype=`` parameter -- a narrow-read variant was
    tried and reverted; see ``bin/utils/qc.py``'s comment above the read for why). What this
    function has therefore ALWAYS emitted, for every slide, is the float32-arithmetic path
    (204 at this triple). The float64/uint16-native path (203) is more precise in isolation
    but is NOT what this function has ever produced, and is not the correctness bar here.

    The reference slide's nuclear channel is engineered so its min, max, and one interior
    pixel are EXACTLY the adversarial triple (13286, 62449, 52520) that makes a naive
    uint16-input read diverge from the real float32-input read by one uint8 level. This test
    is what would fail the moment someone "optimises" ``create_registration_qc`` back into
    reading a narrow dtype without an accompanying widen-back: the real pipeline's output
    would silently drop from 204 to 203 at this pixel while the independently-built
    (always-float32) expectation stayed at 204.
    """
    from tiled_io import open_lazy, read_decimated

    h, w = 64, 64
    rng = np.random.default_rng(7)
    lo, hi, adversarial_value = 13286, 62449, 52520

    dapi = rng.integers(lo + 1, hi, size=(h, w), dtype=np.uint32).astype(np.uint16)
    dapi[0, 0] = lo
    dapi[0, 1] = hi
    dapi[1, 0] = adversarial_value
    assert dapi.min() == lo and dapi.max() == hi
    assert dapi[1, 0] == adversarial_value

    marker = rng.integers(0, 4000, size=(h, w), dtype=np.uint16)
    ref = np.stack([marker, dapi]).astype(np.uint16)
    # registered image: independent content, same shape
    reg = np.stack(
        [
            rng.integers(0, 4000, size=(h, w), dtype=np.uint16),
            rng.integers(0, 65536, size=(h, w), dtype=np.uint16),
        ]
    ).astype(np.uint16)

    ref_path = tmp_path / "reference.ome.tiff"
    reg_path = tmp_path / "registered.ome.tiff"
    tifffile.imwrite(
        str(ref_path),
        ref,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["SMA", "DAPI"]}},
    )
    tifffile.imwrite(
        str(reg_path),
        reg,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["SMA", "DAPI"]}},
    )

    ref_channels = extract_channel_names_from_ome(ref_path)
    reg_channels = extract_channel_names_from_ome(reg_path)
    ref_idx = pick_nuclear_index(ref_channels, None)
    reg_idx = pick_nuclear_index(reg_channels, None)
    assert ref_idx == 1

    # Sanity: confirm this fixture actually reproduces the known divergence, and pin which
    # value is the historical/G1 baseline (float32 -> 204) versus which would be a silent
    # regression if a narrow read were ever reintroduced without a widen-back).
    naive_narrow = qc.autoscale_for_display(dapi, method="minmax")
    historical_float32 = qc.autoscale_for_display(
        dapi.astype(np.float32), method="minmax"
    )
    assert naive_narrow[1, 0] == 203, "a bare uint16 read would land on the lower value"
    assert historical_float32[1, 0] == 204, (
        "the float32 path is what qc.py has always emitted"
    )
    assert naive_narrow[1, 0] != historical_float32[1, 0], (
        "fixture must reproduce the known uint16-vs-float32 divergence to be meaningful"
    )

    # Independently-built expectation: read_decimated's only supported dtype is float32,
    # exactly what create_registration_qc actually gets.
    ref_src, ref_dtype, ref_close = open_lazy(ref_path)
    reg_src, reg_dtype, reg_close = open_lazy(reg_path)
    try:
        assert ref_dtype == np.uint16 and reg_dtype == np.uint16
        old_ref_nuc = read_decimated(ref_src, ref_idx, factor=1)
        old_reg_nuc = read_decimated(reg_src, reg_idx, factor=1)
    finally:
        ref_close()
        reg_close()

    expected_bgr, _expected_cyx = qc.create_nuclear_overlay(
        old_ref_nuc, old_reg_nuc, scale_factor=1.0
    )
    ref_scaled = qc.autoscale_for_display(old_ref_nuc, method="minmax")
    assert ref_scaled[1, 0] == 204, (
        "the G1 baseline itself must be the historical float32 value"
    )

    out_path = tmp_path / "out" / "qc.tif"
    out_path.parent.mkdir()
    qc.create_registration_qc(
        reference_path=ref_path,
        registered_path=reg_path,
        output_path=out_path,
        scale_factor=1.0,
        save_fullres=False,
        save_png=True,
        save_tiff=False,
    )

    got_bgr = cv2.imread(str(out_path.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(got_bgr, expected_bgr)
    # The adversarial pixel specifically, in the green (reference) channel.
    assert got_bgr[1, 0, 1] == 204, (
        "the real pipeline must reproduce the historical float32 value (204), not the naive "
        "narrow-read value (203), at the known adversarial pixel"
    )
