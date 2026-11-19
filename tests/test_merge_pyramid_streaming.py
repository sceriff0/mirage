"""MERGE_AND_PYRAMID must build the published pyramid without ever holding the slide.

``bin/merge_channels_pyramid.py`` streams: the base resolution is written from a generator
of tiles fed by ``open_lazy`` row-band reads, each pyramid level is derived one channel at a
time and tiled only on the way out, and the mask series streams the same way.

What this file pins:

1. Every level is pixel-identical to the whole-array derivation. The reference below is a
   whole-array writer with its own inline ``cv2.resize(..., INTER_AREA)``, so a bug in
   ``_downsample_plane_f32`` cannot cancel out on both sides. Files are also compared byte
   for byte with only tifffile's per-file UUID masked, which catches metadata and IFD drift.
2. An odd-dimensioned fixture, where band-wise level derivation diverges (see
   ``test_band_wise_downsampling_is_not_whole_plane_downsampling``).
3. Bands are forced small (``READ_BAND_BYTES`` monkeypatched) at more than one size: at the
   shipped 64 MiB budget every test image is one band. A three-tile-row band adds a partial
   final band whose boundary is not a plane boundary.
4. The peak grows only at the documented stash slope (``tracemalloc``): ``_PyramidLevels``
   keeps levels 2..K of every channel, ``H*W/6`` bytes per channel, rather than re-reading
   each source plane per level. ``conf/modules.config``'s memory closure states this term.
5. The per-channel validation log is unchanged, with the wrapped-value percentage computed
   against the whole channel, not a band.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import tracemalloc
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("zarr")
cv2 = pytest.importorskip("cv2")

_SPEC = importlib.util.spec_from_file_location(
    "merge_channels_pyramid",
    Path(__file__).resolve().parent.parent / "bin" / "merge_channels_pyramid.py",
)
mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcp)


# ---------------------------------------------------------------------------
# the reference: a whole-array writer, written out independently
# ---------------------------------------------------------------------------


def _reference_downsample(stack, factor):
    """``downsample_image``'s 3-D integer branch, whole-array.

    Written out here rather than imported so the two sides of the equality test do not
    share an implementation.
    """
    c, h, w = stack.shape
    new_h, new_w = h // factor, w // factor
    result = np.zeros((c, new_h, new_w), dtype=stack.dtype)
    for i in range(c):
        resized = cv2.resize(
            stack[i].astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        result[i] = np.round(resized).astype(stack.dtype)
    return result


def _reference_stack(channel_files, dtype):
    """Whole-plane channel read: read whole, clip negatives, cast float->uint16."""
    planes = []
    for path in channel_files:
        plane = np.asarray(tifffile.imread(str(path)))
        if plane.ndim > 2:
            plane = plane.squeeze()
        if plane.min() < 0:
            plane = np.clip(plane, 0, None)
        if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            plane = mcp.to_uint16(plane)
        planes.append(plane)
    return np.stack(planes)


def _reference_write(
    stack,
    path,
    channel_names,
    *,
    pyramid_resolutions,
    pyramid_scale=2,
    tile_size=512,
    compression="zstd",
    compressionargs=None,
    mask_stack=None,
    mask_names=None,
    physical_size_x=0.325,
    physical_size_y=0.325,
):
    """A whole-array ``write_pyramidal_ome_tiff``: one whole-array write per level."""
    levels = mcp.calculate_pyramid_levels(
        stack.shape[1],
        stack.shape[2],
        min_size=64,
        max_levels=pyramid_resolutions,
        scale_factor=pyramid_scale,
    )
    metadata = {
        "axes": "CYX",
        "Channel": {"Name": list(channel_names)},
        "PhysicalSizeX": physical_size_x,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": physical_size_y,
        "PhysicalSizeYUnit": "µm",
    }
    options = dict(
        tile=(tile_size, tile_size),
        compression=compression,
        photometric="minisblack",
        resolutionunit="CENTIMETER",
    )
    if compressionargs:
        options["compressionargs"] = compressionargs

    expected = [stack]
    with tifffile.TiffWriter(str(path), bigtiff=True, ome=True) as tif:
        tif.write(
            stack,
            subifds=len(levels) - 1,
            resolution=(1e4 / physical_size_x, 1e4 / physical_size_y),
            metadata=metadata,
            **options,
        )
        current = stack
        for level_idx in range(1, len(levels)):
            current = _reference_downsample(current, pyramid_scale)
            expected.append(current)
            tif.write(
                current,
                subfiletype=1,
                resolution=(
                    1e4 / (physical_size_x * (pyramid_scale**level_idx)),
                    1e4 / (physical_size_y * (pyramid_scale**level_idx)),
                ),
                **options,
            )
        if mask_stack is not None:
            if mask_stack.dtype != np.uint32:
                mask_stack = mask_stack.astype(np.uint32)
            tif.write(
                mask_stack,
                metadata={
                    "axes": "CYX",
                    "Channel": {"Name": mask_names or ["cell_mask", "nuclei_mask"]},
                },
                tile=(tile_size, tile_size),
                compression=compression,
                photometric="minisblack",
                resolutionunit="CENTIMETER",
            )
    return expected


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# (id, channel names, height, width, dtype, with_masks)
#
# `odd_both_axes` is the fixture that matters: 1023 and 777 are both indivisible by 2, so
# INTER_AREA's effective ratio is 2.00196 rather than 2, and any band- or tile-wise level
# derivation de-aligns progressively down the plane (up to ~29642 counts on uint16).
# `odd_height_only` covers the one-axis case.
SHAPES = [
    ("even", ["DAPI", "CD8"], 512, 512, np.uint16, False),
    ("odd_both_axes", ["DAPI", "CD8", "PANCK"], 1023, 777, np.uint16, True),
    ("odd_height_only", ["DAPI", "SMA"], 513, 640, np.uint16, False),
    ("odd_float_input", ["DAPI", "CD3"], 385, 259, np.float32, False),
]


def _make_inputs(tmp_path, names, h, w, dtype, with_masks, seed=20260819):
    """Write the per-channel TIFFs SPLIT_CHANNELS would have written, plus masks."""
    rng = np.random.default_rng(seed)
    channels_dir = tmp_path / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        if dtype == np.float32:
            # A float input exercises the clip-then-round-then-cast path, negatives and all.
            plane = rng.uniform(-50.0, 4000.0, size=(h, w)).astype(np.float32)
        else:
            plane = rng.integers(0, 65000, size=(h, w), dtype=np.uint16)
        # Tiled, as SPLIT_CHANNELS writes them, so the region reads are real.
        tifffile.imwrite(
            str(channels_dir / f"{name}.tif"),
            plane,
            tile=(256, 256),
            compression="zlib",
        )
        del plane

    masks_dir = None
    if with_masks:
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("cell_mask", "nuclei_mask"):
            tifffile.imwrite(
                str(masks_dir / f"P001_{suffix}.tif"),
                rng.integers(0, 200000, size=(h, w), dtype=np.uint32),
                compression="zlib",
            )
    # sorted() is what _scan_channel_metadata uses to fix channel order.
    channel_files = sorted(
        list(channels_dir.glob("*.tif")) + list(channels_dir.glob("*.tiff"))
    )
    return channels_dir, channel_files, masks_dir


_UUID_RE = re.compile(rb"urn:uuid:[0-9a-fA-F-]{36}")


def _mask_uuid(path):
    """File bytes with tifffile's random per-file OME UUID neutralised.

    The UUID is a fixed-length field inside the ImageDescription, so replacing it changes
    no offset and the rest of the comparison stays meaningful.
    """
    return _UUID_RE.sub(b"urn:uuid:" + b"0" * 36, Path(path).read_bytes())


def _levels_from_file(path):
    with tifffile.TiffFile(str(path)) as tif:
        return [np.asarray(level.asarray()) for level in tif.series[0].levels]


# ---------------------------------------------------------------------------
# 1. exactness
# ---------------------------------------------------------------------------


#: Band budgets the level-identity test runs at, in bytes, against the tile size those
#: fixtures write at (128). ``1`` floors to one tile row. The second budget resolves to two
#: or three tile rows depending on the fixture's width, and leaves a PARTIAL final band on
#: every one of the four -- see the test's docstring for why that is not redundant.
IDENTITY_BAND_BYTES = [
    pytest.param(1, id="band1tile"),
    pytest.param(3 * 128 * 640 * 2, id="band3tiles"),
]


@pytest.mark.parametrize("band_bytes", IDENTITY_BAND_BYTES)
@pytest.mark.parametrize(
    "shape_id,names,h,w,dtype,with_masks", SHAPES, ids=[s[0] for s in SHAPES]
)
def test_every_pyramid_level_is_pixel_identical_to_the_whole_array_derivation(
    tmp_path, monkeypatch, shape_id, names, h, w, dtype, with_masks, band_bytes
):
    """Build the pyramid both ways and compare every level with ``np.array_equal``.

    Bands are forced small so the streamed base genuinely crosses band boundaries; at the
    shipped 64 MiB budget these fixtures would be a single band and the test would not
    exercise the seam it exists to check. Two band sizes, because they fail differently:
    ``1`` collapses to exactly one tile row (every band boundary is a tile boundary), while
    a three-tile-row budget leaves a partial final band on fixtures whose height is not a
    multiple of it -- the shape production actually runs, and the one where an off-by-one
    in the band-to-tile arithmetic survives the single-tile-row case.
    """
    monkeypatch.setattr(mcp, "READ_BAND_BYTES", band_bytes)

    channels_dir, channel_files, masks_dir = _make_inputs(
        tmp_path, names, h, w, dtype, with_masks
    )
    mask_stack = None
    if masks_dir is not None:
        mask_stack = np.stack(
            [
                np.asarray(tifffile.imread(str(sorted(masks_dir.glob(f"*{s}.tif"))[0])))
                for s in ("cell_mask", "nuclei_mask")
            ]
        ).astype(np.uint32)

    stack = _reference_stack(channel_files, np.dtype(dtype))
    # `_scan_channel_metadata` fixes channel order by SORTING the directory, so the OME
    # header's channel names follow the filenames, not the order they were created in.
    ordered_names = [f.stem for f in channel_files]
    expected_levels = _reference_write(
        stack,
        tmp_path / "reference.ome.tiff",
        ordered_names,
        pyramid_resolutions=4,
        tile_size=128,
        compressionargs={"level": 2},
        mask_stack=mask_stack,
    )

    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "streamed.ome.tiff"),
        pyramid_resolutions=4,
        tile_size=128,
        compressionargs={"level": 2},
        masks_dir=None if masks_dir is None else str(masks_dir),
    )

    written = _levels_from_file(tmp_path / "streamed.ome.tiff")
    assert len(written) == len(expected_levels), (
        f"{shape_id}: wrote {len(written)} levels, the whole-array writer wrote "
        f"{len(expected_levels)}"
    )
    for i, (got, want) in enumerate(zip(written, expected_levels)):
        assert got.shape == want.shape, (
            f"{shape_id}: level {i} shape {got.shape} != {want.shape}"
        )
        if not np.array_equal(got, want):
            diff = np.abs(got.astype(np.int64) - want.astype(np.int64))
            pytest.fail(
                f"{shape_id}: pyramid level {i} differs from the whole-array derivation: "
                f"{int(np.count_nonzero(diff))} of {diff.size} pixels, max difference "
                f"{int(diff.max())} counts. A band- or tile-wise INTER_AREA is not a "
                f"whole-plane INTER_AREA whenever the factor does not divide the dimension."
            )

    if mask_stack is not None:
        with tifffile.TiffFile(str(tmp_path / "streamed.ome.tiff")) as tif:
            assert len(tif.series) == 2
            assert np.array_equal(np.asarray(tif.series[1].asarray()), mask_stack), (
                f"{shape_id}: the streamed mask series is not the labels that went in"
            )


@pytest.mark.parametrize(
    "shape_id,names,h,w,dtype,with_masks", SHAPES, ids=[s[0] for s in SHAPES]
)
def test_the_streamed_file_matches_the_whole_array_writer_byte_for_byte(
    tmp_path, monkeypatch, shape_id, names, h, w, dtype, with_masks
):
    """Stronger than the pixel comparison: same tags, same IFD layout, same OME-XML.

    Only tifffile's random per-file UUID is masked. A pixel-equal file with a different tile
    size, a lost PhysicalSize or a missing SubIFD chain would pass the level comparison
    above and fail here, and it is the OME-XML and the SubIFD chain that decide whether
    QuPath sees a pyramid at all.
    """
    monkeypatch.setattr(mcp, "READ_BAND_BYTES", 1)

    channels_dir, channel_files, masks_dir = _make_inputs(
        tmp_path, names, h, w, dtype, with_masks
    )
    mask_stack = None
    if masks_dir is not None:
        mask_stack = np.stack(
            [
                np.asarray(tifffile.imread(str(sorted(masks_dir.glob(f"*{s}.tif"))[0])))
                for s in ("cell_mask", "nuclei_mask")
            ]
        ).astype(np.uint32)

    stack = _reference_stack(channel_files, np.dtype(dtype))
    _reference_write(
        stack,
        tmp_path / "reference.ome.tiff",
        [f.stem for f in channel_files],
        pyramid_resolutions=4,
        tile_size=128,
        compressionargs={"level": 2},
        mask_stack=mask_stack,
    )
    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "streamed.ome.tiff"),
        pyramid_resolutions=4,
        tile_size=128,
        compressionargs={"level": 2},
        masks_dir=None if masks_dir is None else str(masks_dir),
    )

    assert _mask_uuid(tmp_path / "streamed.ome.tiff") == _mask_uuid(
        tmp_path / "reference.ome.tiff"
    ), (
        f"{shape_id}: the streamed file is not byte-identical to the whole-array writer's"
    )


def test_band_wise_downsampling_is_not_whole_plane_downsampling():
    """The trap, pinned as a measurement so nobody "optimises" the levels into the tile loop.

    ``cv2.INTER_AREA``'s effective vertical ratio is ``h / (h // factor)``, which is the
    requested factor only when the factor DIVIDES the dimension. This test asserts the
    positive result on a dimension that divides and the negative one on a dimension that
    does not -- both, because a test that only showed the exact case would license exactly
    the band-wise derivation this design rejects.
    """
    rng = np.random.default_rng(7)

    def banded(plane, factor, band):
        assert band % factor == 0
        parts = [
            np.round(
                cv2.resize(
                    plane[y : y + band].astype(np.float32),
                    (plane.shape[1] // factor, min(band, plane.shape[0] - y) // factor),
                    interpolation=cv2.INTER_AREA,
                )
            ).astype(plane.dtype)
            for y in range(0, plane.shape[0] - plane.shape[0] % band, band)
        ]
        return np.concatenate(parts, axis=0)

    def whole(plane, factor):
        return np.round(
            cv2.resize(
                plane.astype(np.float32),
                (plane.shape[1] // factor, plane.shape[0] // factor),
                interpolation=cv2.INTER_AREA,
            )
        ).astype(plane.dtype)

    divides = rng.integers(0, 65000, size=(1024, 512), dtype=np.uint16)
    assert np.array_equal(banded(divides, 2, 256), whole(divides, 2)), (
        "at h=1024 the factor divides the height, so banding IS exact -- if this fails the "
        "measurement instrument itself is wrong"
    )

    does_not = rng.integers(0, 65000, size=(1023, 512), dtype=np.uint16)
    banded_1023 = banded(does_not, 2, 256)
    whole_1023 = whole(does_not, 2)
    diff = np.abs(
        banded_1023[: whole_1023.shape[0]].astype(np.int64)
        - whole_1023[: banded_1023.shape[0]].astype(np.int64)
    )
    assert diff.max() > 1000, (
        f"at h=1023 band-wise INTER_AREA came within {diff.max()} counts of the whole-plane "
        "result on this OpenCV. The design's premise -- that bands are catastrophically, "
        "not marginally, wrong on an indivisible dimension -- no longer holds; re-derive it "
        "before relying on the whole-plane level derivation for anything but exactness."
    )


# ---------------------------------------------------------------------------
# 2. the peak
# ---------------------------------------------------------------------------


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _run_merge(tmp_path, n_channels, h=1024, w=1024):
    names = [f"CH{i:02d}" for i in range(n_channels)]
    channels_dir, _files, _masks = _make_inputs(
        tmp_path, names, h, w, np.uint16, False, seed=11 + n_channels
    )
    out = tmp_path / "pyr.ome.tiff"
    return lambda: mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(out),
        pyramid_resolutions=4,
        tile_size=256,
    )


def test_peak_memory_grows_only_at_the_documented_stash_slope(tmp_path):
    """The O(C) slope the memory closure in conf/modules.config is derived from.

    ``_PyramidLevels`` stashes levels 2..K of every channel's chain, ``H*W/6`` bytes per
    channel (one twelfth of a base plane), so the peak is not flat in C. A whole-stack
    implementation would grow by about one plane per channel. A flat cap would fail at
    larger C with nothing regressed, so the slope is asserted instead.
    """
    plane_bytes = 1024 * 1024 * 2
    peak_2 = _peak_bytes(_run_merge(tmp_path / "c2", 2))
    peak_16 = _peak_bytes(_run_merge(tmp_path / "c16", 16))

    delta_c = 16 - 2
    growth = peak_16 - peak_2
    # Model term: plane_bytes/12 per channel. 3x slack covers measurement noise; a
    # ~1-plane-per-channel slope (twelve times steeper) still fails by a wide margin.
    budget = delta_c * (plane_bytes / 12) * 3
    assert growth <= budget, (
        f"peak grew {growth / 1e6:.2f} MB over {delta_c} extra channels "
        f"({growth / delta_c / 1e6:.4f} MB/channel) -- more than 3x the documented "
        f"H*W/6-bytes-per-channel stash slope ({budget / 1e6:.2f} MB budget for this "
        "fixture)"
    )


def test_peak_memory_is_a_small_multiple_of_one_plane(tmp_path):
    """No ``(C, H, W)`` array is resident at any point.

    The bound is stated in planes rather than in bytes because that is what the memory
    closure derives from: one float32 plane for the level derivation (4 bytes/px), its
    resize output and level chain (about 1.5 more), one read band and one write tile. Six
    planes is generous for that and far below a twelve-plane whole stack.
    """
    plane_bytes = 1024 * 1024 * 2
    peak = _peak_bytes(_run_merge(tmp_path / "bound", 12))

    assert peak < 6 * plane_bytes, (
        f"peak was {peak / 1e6:.1f} MB, above {6 * plane_bytes / 1e6:.1f} MB "
        f"(6 x one {plane_bytes / 1e6:.1f} MB plane) on a 12-channel slide"
    )


# ---------------------------------------------------------------------------
# 3. the shape of the reads and the writes
# ---------------------------------------------------------------------------


def test_every_write_is_fed_from_an_iterator(tmp_path, monkeypatch):
    """The write side, which no read-side spy can see.

    Handing ``tifffile`` an ndarray is the single observable signature of the regression --
    it means the base, or a whole level, was assembled first. ``shape=`` and ``dtype=`` must
    accompany the iterator or tifffile cannot size the output and would have to buffer.
    """
    # The fixtures are written FIRST: `_make_inputs` goes through `tifffile.imwrite`, which
    # is itself a TiffWriter.write of an ndarray, and would trip the assertion below.
    channels_dir, _files, masks_dir = _make_inputs(
        tmp_path, ["DAPI", "CD8"], 512, 512, np.uint16, True
    )

    seen = []
    original_write = tifffile.TiffWriter.write

    def spying_write(self, data=None, **kwargs):
        seen.append((type(data).__name__, isinstance(data, np.ndarray), set(kwargs)))
        return original_write(self, data, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", spying_write)
    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=3,
        tile_size=128,
        masks_dir=str(masks_dir),
    )

    assert seen, (
        "TiffWriter.write was never called -- the output was written some other way"
    )
    for type_name, is_ndarray, kwargs in seen:
        assert not is_ndarray, (
            f"a whole array ({type_name}) was handed to tifffile; the base, every pyramid "
            "level and the mask series must each be fed by an iterator of tiles"
        )
        assert {"shape", "dtype", "tile"} <= kwargs, (
            f"an iterator-fed write needs an explicit shape/dtype/tile; got {kwargs}"
        )
    # base + 2 sublevels + mask series
    assert len(seen) == 4, f"expected 4 iterator-fed writes, got {len(seen)}"


def test_every_write_pins_one_compressor_thread(tmp_path, monkeypatch):
    """`maxworkers=1` on every write, and it is a memory bound, not a tuning preference.

    tifffile defaults `maxworkers` to the machine's CPU count for a compressed tiled write
    and dispatches encoded tiles through a thread pool that runs ahead of the writer. The
    queued tiles are then a term in the peak that scales with the channel count. Peak on
    a 1024^2 slide with four pyramid levels, in units of one plane:

        C=                    2      8     12     16
        tifffile 2023.4.12  8.24   4.78   5.10   5.41
        tifffile 2025.5.10  7.44   9.41  13.54  17.54

    containers/merge/Dockerfile pins 2025.5.10, where the slope is about one plane per
    channel. With `maxworkers=1` both versions give 4.46 / 4.77 / 5.09 / 5.41.

    Asserted on the argument rather than measured: with an older tifffile the default pool
    is shallow enough that the peak tests would not catch its removal.
    """
    channels_dir, _files, masks_dir = _make_inputs(
        tmp_path, ["DAPI", "CD8"], 512, 512, np.uint16, True
    )

    seen = []
    original_write = tifffile.TiffWriter.write

    def spying_write(self, data=None, **kwargs):
        seen.append(kwargs.get("maxworkers", "<default>"))
        return original_write(self, data, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", spying_write)

    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=3,
        tile_size=128,
        masks_dir=str(masks_dir),
    )

    assert seen == [1, 1, 1, 1], (
        f"maxworkers per write was {seen}, not 1 on all four (base, two levels, masks). "
        "tifffile's default is the machine's CPU count, and its encode queue is an "
        "unbounded, channel-count-scaled term in the peak at the container's pinned version."
    )


def test_no_channel_file_is_read_whole_through_imread(tmp_path, monkeypatch):
    """No channel or mask file is decoded whole with ``tifffile.imread``.

    The only whole-file reads are the metadata scan's ``TiffFile`` header reads, which
    decode no pixels.
    """
    eager = []
    original_imread = tifffile.imread

    def spying_imread(path, *args, **kwargs):
        if not kwargs.get("aszarr", False):
            eager.append(str(path))
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(mcp.tifffile, "imread", spying_imread)

    channels_dir, _files, masks_dir = _make_inputs(
        tmp_path, ["DAPI", "CD8"], 512, 512, np.uint16, True
    )
    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=3,
        tile_size=128,
        masks_dir=str(masks_dir),
    )

    assert not eager, (
        f"these files were read whole with tifffile.imread: {eager}. Channels and masks "
        "must be reached through open_lazy region reads."
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.float32], ids=["uint16", "float32"])
def test_the_base_pass_reads_bands_not_planes(tmp_path, monkeypatch, dtype):
    """Every base-pass read is a single-channel Y/X REGION bounded by the band budget.

    The level pass legitimately reads a whole plane -- INTER_AREA leaves no choice -- so the
    bound asserted here is per read and sized from ``READ_BAND_BYTES``, not "smaller than a
    plane" unconditionally.

    The float32 case is the one that matters: the band is budgeted from the source
    itemsize, not the uint16 output's, so a float32 read materialises 4 bytes/px. On a
    uint16 fixture the two are equal and the distinction is invisible.
    """
    # Three tile rows of a 512px-wide uint16 plane, so the budget binds rather than
    # `_band_rows`' one-tile-row floor (at the floor a wrong itemsize hides).
    monkeypatch.setattr(mcp, "READ_BAND_BYTES", 3 * 128 * 512 * 2)

    reads = []
    original_open_lazy = mcp.open_lazy

    class _Spy:
        def __init__(self, arr):
            self._arr = arr
            self.shape = arr.shape
            self.dtype = arr.dtype

        def __getitem__(self, key):
            out = np.asarray(self._arr[key])
            reads.append((key, int(out.nbytes)))
            return out

    def spying_open_lazy(path):
        arr, dtype, close = original_open_lazy(path)
        return _Spy(arr), dtype, close

    monkeypatch.setattr(mcp, "open_lazy", spying_open_lazy)

    channels_dir, _files, _masks = _make_inputs(
        tmp_path, ["DAPI", "CD8"], 512, 512, dtype, False
    )
    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=3,
        tile_size=128,
    )

    assert reads, "the lazy view was opened but never read from"
    itemsize = np.dtype(dtype).itemsize
    plane_bytes = 512 * 512 * itemsize
    budget_bytes = mcp.READ_BAND_BYTES
    for key, nbytes in reads:
        assert isinstance(key, tuple) and isinstance(key[0], (int, np.integer)), (
            f"read {key!r} is not a single-channel index -- a channel slab was materialised"
        )
        assert isinstance(key[1], slice) and isinstance(key[2], slice), (
            f"read {key!r} does not index a Y/X REGION"
        )
        assert nbytes < plane_bytes, (
            f"read {key!r} materialised {nbytes} bytes, a whole {plane_bytes}-byte plane, "
            f"with the band budget set to {budget_bytes} bytes"
        )

    # Every base-pass read (i.e. every read except the one whole plane per channel the
    # level pass takes) sits inside the budget, or inside `_band_rows`' one-tile-row floor
    # where that is larger. Sized from the SOURCE itemsize: budgeting a float32 channel by
    # the uint16 OUTPUT would give it three tile rows where the budget allows one, and the
    # read would be 786432 bytes against a 393216-byte budget.
    one_tile_row = 128 * 512 * itemsize
    ceiling = max(budget_bytes, one_tile_row)
    band_reads = [n for _key, n in reads if n < plane_bytes]
    assert max(band_reads) <= ceiling, (
        f"the largest base-pass read was {max(band_reads)} bytes against a ceiling of "
        f"{ceiling} (budget {budget_bytes}, one tile row {one_tile_row}). The band is "
        f"budgeted from the SOURCE itemsize ({itemsize} here); using the output dtype's "
        "oversizes a float32 channel's band by the ratio between them."
    )


# ---------------------------------------------------------------------------
# 4. the per-channel validation log
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def captured_log():
    handler = _Capture()
    for name in ("merge_channels_pyramid", "validation"):
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.DEBUG)
    mcp.logger.addHandler(handler)
    mcp.logger.setLevel(logging.DEBUG)
    yield handler
    mcp.logger.removeHandler(handler)
    for name in ("merge_channels_pyramid", "validation"):
        logging.getLogger(name).removeHandler(handler)


def test_the_negative_clip_still_logs_one_line_per_channel_with_the_whole_channel_min(
    tmp_path, monkeypatch, captured_log
):
    """Same two lines, same numbers -- and the min is the CHANNEL's, not a band's.

    Streaming sees the plane in pieces, so the decision to log and the number logged are
    deferred to the last band. A per-band implementation would print one pair of lines per
    band, each with that band's min.
    """
    monkeypatch.setattr(mcp, "READ_BAND_BYTES", 1)  # one tile row per band

    rng = np.random.default_rng(3)
    channels_dir = tmp_path / "channels"
    channels_dir.mkdir()
    plane = rng.uniform(5.0, 4000.0, size=(384, 256)).astype(np.float32)
    # A single, deep negative in the last band.
    plane[380, 7] = -1234.5
    tifffile.imwrite(str(channels_dir / "DAPI.tif"), plane)
    tifffile.imwrite(
        str(channels_dir / "CD8.tif"),
        rng.uniform(5.0, 4000.0, size=(384, 256)).astype(np.float32),
    )

    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=2,
        tile_size=128,
    )

    negatives = [
        line for line in captured_log.lines if "Negative values detected in" in line
    ]
    clipped = [line for line in captured_log.lines if "Clipped to 0. New min=" in line]
    assert negatives == [
        f"    WARNING: Negative values detected in DAPI: min={np.float32(-1234.5)}"
    ], f"got {negatives}"
    assert clipped == ["    Clipped to 0. New min=0.0"], f"got {clipped}"


def test_the_wrapped_value_percentage_is_against_the_whole_channel(
    tmp_path, monkeypatch, captured_log
):
    """The denominator is the whole channel, not the band holding the pixels.

    The suspicious pixels are all parked in one band. Reported against the whole channel
    they are a small fraction; reported against the band that holds them they are ~24x that,
    and the 0.01%% decision threshold makes the difference visible rather than cosmetic.
    ``detect_wrapped_values`` on the whole plane is the oracle.
    """
    monkeypatch.setattr(mcp, "READ_BAND_BYTES", 1)

    validation = importlib.import_module("validation")

    rng = np.random.default_rng(5)
    h, w = 384, 256
    channels_dir = tmp_path / "channels"
    channels_dir.mkdir()
    plane = rng.integers(0, 50000, size=(h, w), dtype=np.uint16)
    plane[3:9, :] = 65000  # 6 rows of "wrapped" pixels, all inside one 16-row band
    tifffile.imwrite(str(channels_dir / "DAPI.tif"), plane)

    # A logger of its own: the oracle call must not emit into the capture it is the
    # oracle FOR, or the streamed run's single line looks like two.
    has_wrapped, count, pct = validation.detect_wrapped_values(
        plane, logger=logging.getLogger("merge-streaming-test-oracle")
    )
    assert has_wrapped, "fixture does not trip the whole-array detector"

    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(tmp_path / "pyr.ome.tiff"),
        pyramid_resolutions=2,
        tile_size=128,
    )

    wrapped = [line for line in captured_log.lines if "potential wrapped" in line]
    assert wrapped == [
        f"    WARNING: {count} potential wrapped negative pixels ({pct:.4f}%) in DAPI"
    ], f"got {wrapped}; the whole-array call reports {count} pixels ({pct:.4f}%)"

    # validation.py's own line, emitted once, with the same aggregates.
    shared = [
        line for line in captured_log.lines if "potentially wrapped pixels" in line
    ]
    assert shared == [
        f"Detected {count} potentially wrapped pixels ({pct:.4f}%), max value: 65000"
    ], f"got {shared}"


# ---------------------------------------------------------------------------
# 5. the retained whole-array entry point
# ---------------------------------------------------------------------------


def test_the_array_entry_point_does_not_clip_signed_input(tmp_path):
    """`write_pyramidal_ome_tiff` must round-trip a signed-integer stack unchanged.

    The negative clip belongs to the pipeline's read of the channel files
    (`_ChannelFileSource`), not to `write_pyramidal_ome_tiff`, which only casts float
    input. `_ArraySource` must not clip an int16 stack's negatives to 0.

    int16 makes the difference observable: for uint16 the clip is a no-op, and float is
    clipped by `to_uint16` regardless (asserted below).
    """
    data = np.full((2, 64, 64), -100, dtype=np.int16)
    data[:, 32:, :] = 500
    out = tmp_path / "signed.ome.tiff"

    mcp.write_pyramidal_ome_tiff(
        data,
        str(out),
        channel_names=["DAPI", "CD8"],
        channel_colors=[(0, 0, 255), (255, 0, 0)],
        pyramid_resolutions=1,
    )

    written = np.asarray(tifffile.imread(str(out)))
    assert written.dtype == np.int16, f"dtype changed to {written.dtype}"
    assert int(written.min()) == -100, (
        f"the array entry point clipped signed input: min is {int(written.min())}, not -100. "
        "The negative clip belongs to _ChannelFileSource (the pipeline's read), not here."
    )
    assert np.array_equal(written, data)


def test_the_array_entry_point_still_clips_and_rounds_float_input(tmp_path):
    """The other half of the same rule: float input IS clipped, by `to_uint16`.

    Without this the test above could be satisfied by removing the cast altogether, which
    would change what the QuPath-facing pyramid does with the float stacks it is documented
    to accept. -5.0 must become 0 and 100.7 must become 101, not 100.
    """
    data = np.array([[[-5.0, 100.7], [70000.0, 3.0]]], dtype=np.float32)
    out = tmp_path / "floaty.ome.tiff"

    mcp.write_pyramidal_ome_tiff(
        data,
        str(out),
        channel_names=["DAPI"],
        channel_colors=[(0, 0, 255)],
        pyramid_resolutions=1,
    )

    written = np.asarray(tifffile.imread(str(out)))
    assert written.dtype == np.uint16
    assert written.tolist() == [[0, 101], [65535, 3]]
