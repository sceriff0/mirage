"""The two BASICPY-path readers must never materialise more than one tile.

``bin/tile_for_basic.py`` and ``bin/apply_basic_profiles.py`` are the two halves of the
three-process illumination path (TILE_FOR_BASIC -> BASICPY -> APPLY_PROFILES). Both open
the slide through ``bin/utils/tiled_io.py:open_lazy`` and read it **one tile at a time**;
neither ever holds a channel, let alone the ``(C, H, W)`` stack.

A read-side spy cannot see a write-side accumulation (stacking every corrected channel
before writing would peak at several times the slide), so
``test_apply_basic_profiles_streams_its_write`` and
``test_apply_basic_profiles_does_not_return_the_assembled_slide`` pin the shape of the
write separately.

Four properties, per script:

1. **The slide is reached through ``open_lazy``**, and never through a whole-array
   ``tifffile.imread`` of the slide path. (``imread`` on the small BASICPY *profile*
   files is legitimate and is not flagged -- those are tile-sized, not slide-sized.)
2. **Every materialisation is at most one tile.** The spy below records the size of every
   array handed back through the lazy view, so a bigger read is caught by its *bytes*, not
   merely by the shape of the index expression.
3. **The peak does not grow with the channel count**, and
4. **the peak does not grow with the slide size** -- the O(C) -> O(1) and O(H*W) -> O(1)
   claims stated as assertions. (4) is what the ``conf/modules.config`` memory formulas
   are derived from.

Not covered: peak process memory also depends on how much the TIFF reader decodes to
satisfy a region read. On an untiled file tifffile's zarr view reports
``chunks=(1, H, W)``, so a region read decodes a whole plane; that is why every producer
ahead of a lazy reader writes tiled
(``tests/test_convert_streaming_write.py::test_the_write_is_tiled``). The spy measures
what the code asked for and received.

The spy calls ``np.asarray`` on what it passes through; both scripts do the same on the
next line, so it adds no materialisation.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("zarr")

import apply_basic_profiles  # noqa: E402
import tile_for_basic  # noqa: E402

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_slide(path, stack, channel_names):
    tifffile.imwrite(
        str(path),
        stack,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(channel_names)}},
        ome=True,
    )


def _write_profile(path, planes):
    tifffile.imwrite(
        str(path),
        np.asarray(planes, dtype=np.float32),
        photometric="minisblack",
        ome=True,
    )


def _slide(tmp_path, n_channels, h=300, w=300, seed=3):
    """A ``(C, h, w)`` uint16 slide whose channel 0 is the nuclear/fiducial marker."""
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 4000, size=(n_channels, h, w), dtype=np.uint16)
    names = ["DAPI"] + [f"MARKER_{i}" for i in range(1, n_channels)]
    path = tmp_path / "slide.ome.tif"
    _write_slide(path, stack, names)
    return path, names, stack


def _tiled(tmp_path, n_channels, fov=100, h=300, w=300):
    """Run the real TILE_FOR_BASIC step and return everything APPLY_PROFILES needs."""
    slide, names, stack = _slide(tmp_path, n_channels, h=h, w=w)
    sidecar = tmp_path / "slide_tiles.json"
    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "slide_tiles.ome.tif"),
        str(sidecar),
        channel_names=list(names),
        fov_size=(fov, fov),
        skip_nuclear=True,
        nuclear_markers=["DAPI"],
    )
    th, tw = manifest["tile_shape"]
    n_fitted = len(manifest["profile_channels"])
    ffp = tmp_path / "slide_tiles-ffp.ome.tif"
    dfp = tmp_path / "slide_tiles-dfp.ome.tif"
    _write_profile(ffp, [np.full((th, tw), 2.0)] * n_fitted)
    _write_profile(dfp, [np.zeros((th, tw))] * n_fitted)
    return slide, sidecar, ffp, dfp, names, stack, manifest


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


class _MaterialisationSpy:
    """A pass-through wrapper over a lazy ``(C, H, W)`` view that records what it hands
    back: the index expression, and the number of BYTES that expression materialised."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.keys = []
        self.nbytes = []

    def __getitem__(self, key):
        out = np.asarray(self._arr[key])
        self.keys.append(key)
        self.nbytes.append(int(out.nbytes))
        return out


def _install_spies(monkeypatch, module, slide_path):
    """Wrap ``module.open_lazy`` in a spy and flag any whole-array read of the slide.

    Returns ``(spies, eager_imreads)``: ``spies`` maps opened path -> spy.
    """
    spies = {}
    original_open_lazy = module.open_lazy

    def spying_open_lazy(path):
        arr, dtype, close = original_open_lazy(path)
        spy = _MaterialisationSpy(arr)
        spies[str(path)] = spy
        return spy, dtype, close

    monkeypatch.setattr(module, "open_lazy", spying_open_lazy)

    eager = []
    original_imread = tifffile.imread

    def spying_imread(path, *args, **kwargs):
        # aszarr=True is open_lazy's own call and decodes nothing. Only the SLIDE is
        # flagged: the BASICPY profile files are tile-sized and read whole by design.
        if str(path) == str(slide_path) and not kwargs.get("aszarr", False):
            eager.append(str(path))
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(module.tifffile, "imread", spying_imread)
    return spies, eager


def _assert_region_reads(spy, slide_shape, max_bytes, expected_reads, where):
    """Every read is a single-channel REGION of at most ``max_bytes``, and there are
    exactly ``expected_reads`` of them."""
    n_channels, h, w = slide_shape
    plane_bytes = h * w * np.dtype(np.uint16).itemsize

    assert spy.keys, f"{where}: the lazy view was opened but never read from"
    for key in spy.keys:
        assert isinstance(key, tuple) and isinstance(key[0], (int, np.integer)), (
            f"{where}: read {key!r} is not a single-channel index -- the whole stack "
            "(or a channel slab) was materialised"
        )
        assert isinstance(key[1], slice) and isinstance(key[2], slice), (
            f"{where}: read {key!r} does not index a Y/X REGION -- a whole plane was "
            "requested where a tile was expected"
        )
    assert max(spy.nbytes) <= max_bytes, (
        f"{where}: the largest single materialisation was {max(spy.nbytes)} bytes, above "
        f"the {max_bytes}-byte tile bound (one plane is {plane_bytes} bytes and the whole "
        f"{n_channels}-channel stack is {plane_bytes * n_channels})"
    )
    assert len(spy.keys) == expected_reads, (
        f"{where}: expected {expected_reads} region reads, got {len(spy.keys)}"
    )


# ---------------------------------------------------------------------------
# TILE_FOR_BASIC
# ---------------------------------------------------------------------------


def test_tile_for_basic_reads_one_fov_at_a_time(tmp_path, monkeypatch):
    """TILE_FOR_BASIC reads one pseudo-FOV at a time through ``open_lazy``.

    Exactly ``len(profile_channels) * n_tiles`` reads: one per tile written.
    ``fov_tiling.fov_positions`` returns the grid without touching the image.
    """
    slide, names, stack = _slide(tmp_path, 4)
    spies, eager = _install_spies(monkeypatch, tile_for_basic, slide)

    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "tiles.ome.tif"),
        str(tmp_path / "tiles.json"),
        channel_names=list(names),
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["DAPI"],
    )

    assert str(slide) in spies, "the slide never went through open_lazy"
    assert not eager, (
        f"the slide was eagerly decoded in full via tifffile.imread: {eager}"
    )
    n_tiles = manifest["n_fovs_y"] * manifest["n_fovs_x"]
    th, tw = manifest["tile_shape"]
    _assert_region_reads(
        spies[str(slide)],
        stack.shape,
        max_bytes=th * tw * 2,
        expected_reads=len(manifest["profile_channels"]) * n_tiles,
        where="tile_for_basic",
    )


def test_tile_for_basic_peak_read_is_independent_of_channel_count(
    tmp_path, monkeypatch
):
    """The largest single materialisation is one FOV at 2 channels and at 6 -- O(1), not
    O(C). A whole-stack read passes the shape assertions of a sloppier test but triples
    here."""
    peaks = {}
    for n_channels in (2, 6):
        work = tmp_path / f"c{n_channels}"
        work.mkdir()
        slide, names, stack = _slide(work, n_channels)
        with monkeypatch.context() as ctx:
            spies, eager = _install_spies(ctx, tile_for_basic, slide)
            tile_for_basic.tile_for_basic(
                str(slide),
                str(work / "tiles.ome.tif"),
                str(work / "tiles.json"),
                channel_names=list(names),
                fov_size=(100, 100),
                skip_nuclear=True,
                nuclear_markers=["DAPI"],
            )
        assert not eager
        peaks[n_channels] = max(spies[str(slide)].nbytes)

    assert peaks[2] == peaks[6] == 100 * 100 * 2, (
        f"peak materialisation scales with the channel count: {peaks}"
    )


def test_tile_for_basic_peak_read_is_independent_of_slide_size(tmp_path, monkeypatch):
    """The claim conf/modules.config's memory formula rests on: at a fixed FOV size, a 4x
    larger slide reads the same number of bytes at a time -- it just reads more times.

    Both sizes are exact multiples of the FOV, so the grid's tile shape is the same 100x100
    in each and the peaks are directly comparable.
    """
    peaks, reads = {}, {}
    for side in (300, 600):
        work = tmp_path / f"s{side}"
        work.mkdir()
        slide, names, stack = _slide(work, 3, h=side, w=side)
        with monkeypatch.context() as ctx:
            spies, eager = _install_spies(ctx, tile_for_basic, slide)
            tile_for_basic.tile_for_basic(
                str(slide),
                str(work / "tiles.ome.tif"),
                str(work / "tiles.json"),
                channel_names=list(names),
                fov_size=(100, 100),
                skip_nuclear=True,
                nuclear_markers=["DAPI"],
            )
        assert not eager
        peaks[side] = max(spies[str(slide)].nbytes)
        reads[side] = len(spies[str(slide)].keys)

    assert peaks[300] == peaks[600] == 100 * 100 * 2, (
        f"peak materialisation scales with the slide size: {peaks}"
    )
    assert reads[600] == reads[300] * 4, (
        f"a 4x larger slide should take 4x as many same-sized reads, got {reads}"
    )


# ---------------------------------------------------------------------------
# APPLY_PROFILES
# ---------------------------------------------------------------------------


def test_apply_basic_profiles_reads_one_write_tile_at_a_time(tmp_path, monkeypatch):
    """APPLY_PROFILES reads one output write-tile's region at a time -- including for the
    nuclear/fiducial channel it does not correct, which is carried through untouched.

    ``WRITE_TILE`` is patched below the fixture's size: at its shipped 2048 a 300px test
    slide is one tile, and the tiling would not be exercised.
    """
    monkeypatch.setattr(apply_basic_profiles, "WRITE_TILE", 64)
    slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(tmp_path, 4)
    spies, eager = _install_spies(monkeypatch, apply_basic_profiles, slide)

    apply_basic_profiles.apply_basic_profiles(
        str(slide),
        str(sidecar),
        str(ffp),
        str(dfp),
        str(tmp_path / "slide_corrected.ome.tif"),
    )

    assert str(slide) in spies, "the slide never went through open_lazy"
    assert not eager, (
        f"the slide was eagerly decoded in full via tifffile.imread: {eager}"
    )
    spy = spies[str(slide)]
    n_channels, h, w = stack.shape
    tiles_per_channel = -(-h // 64) * -(-w // 64)
    _assert_region_reads(
        spy,
        stack.shape,
        max_bytes=64 * 64 * 2,
        expected_reads=n_channels * tiles_per_channel,
        where="apply_basic_profiles",
    )
    assert [int(k[0]) for k in spy.keys] == [
        c for c in range(n_channels) for _ in range(tiles_per_channel)
    ], "channels should be visited once each, in order, tile by tile"


def test_apply_basic_profiles_peak_read_is_independent_of_channel_count(
    tmp_path, monkeypatch
):
    """Same O(1)-in-C claim for the apply half."""
    peaks = {}
    for n_channels in (2, 6):
        work = tmp_path / f"c{n_channels}"
        work.mkdir()
        slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(work, n_channels)
        with monkeypatch.context() as ctx:
            ctx.setattr(apply_basic_profiles, "WRITE_TILE", 64)
            spies, eager = _install_spies(ctx, apply_basic_profiles, slide)
            apply_basic_profiles.apply_basic_profiles(
                str(slide),
                str(sidecar),
                str(ffp),
                str(dfp),
                str(work / "slide_corrected.ome.tif"),
            )
        assert not eager
        peaks[n_channels] = max(spies[str(slide)].nbytes)

    assert peaks[2] == peaks[6] == 64 * 64 * 2, (
        f"peak materialisation scales with the channel count: {peaks}"
    )


def test_apply_basic_profiles_peak_read_is_independent_of_slide_size(
    tmp_path, monkeypatch
):
    """The O(1)-in-slide-size claim conf/modules.config's APPLY_PROFILES formula rests on."""
    peaks, reads = {}, {}
    for side in (256, 512):
        work = tmp_path / f"s{side}"
        work.mkdir()
        slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(
            work, 3, fov=64, h=side, w=side
        )
        with monkeypatch.context() as ctx:
            ctx.setattr(apply_basic_profiles, "WRITE_TILE", 64)
            spies, eager = _install_spies(ctx, apply_basic_profiles, slide)
            apply_basic_profiles.apply_basic_profiles(
                str(slide),
                str(sidecar),
                str(ffp),
                str(dfp),
                str(work / "slide_corrected.ome.tif"),
            )
        assert not eager
        peaks[side] = max(spies[str(slide)].nbytes)
        reads[side] = len(spies[str(slide)].keys)

    assert peaks[256] == peaks[512] == 64 * 64 * 2, (
        f"peak materialisation scales with the slide size: {peaks}"
    )
    assert reads[512] == reads[256] * 4, (
        f"a 4x larger slide should take 4x as many same-sized reads, got {reads}"
    )


def test_apply_basic_profiles_streams_its_write(tmp_path, monkeypatch):
    """The write side, which no read-side spy can see.

    Stacking every corrected channel and handing ``tifffile`` one slide-sized array shows up
    as an ndarray instead of an iterator, so that is asserted directly. ``shape=`` and
    ``dtype=`` must accompany the iterator, or ``tifffile`` would have to buffer.
    """
    seen = {}
    original_write = tifffile.TiffWriter.write

    def spying_write(self, data=None, **kwargs):
        seen["type"] = type(data).__name__
        seen["is_ndarray"] = isinstance(data, np.ndarray)
        seen["kwargs"] = set(kwargs)
        return original_write(self, data, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", spying_write)

    slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(tmp_path, 3)
    apply_basic_profiles.apply_basic_profiles(
        str(slide),
        str(sidecar),
        str(ffp),
        str(dfp),
        str(tmp_path / "out_corrected.ome.tif"),
    )

    assert seen, (
        "TiffWriter.write was never called -- the output was written some other way"
    )
    assert not seen["is_ndarray"], (
        f"the whole image was handed to tifffile as a {seen['type']}; the write must be "
        "fed by an iterator of tiles, or the assembled slide is resident again"
    )
    assert {"shape", "dtype", "tile"} <= seen["kwargs"], (
        f"an iterator-fed write needs an explicit shape/dtype/tile; got {seen['kwargs']}"
    )


def test_apply_basic_profiles_does_not_return_the_assembled_slide(tmp_path):
    """The function returns the output PATH, not the pixels.

    Returning the assembled array would make every caller hold a slide-sized object
    however the write was streamed.
    """
    slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(tmp_path, 3)
    result = apply_basic_profiles.apply_basic_profiles(
        str(slide),
        str(sidecar),
        str(ffp),
        str(dfp),
        str(tmp_path / "r_corrected.ome.tif"),
    )
    assert not isinstance(result, np.ndarray), (
        "apply_basic_profiles returned an ndarray -- the assembled slide is resident again"
    )


def test_the_sidecar_the_two_halves_share_is_what_pins_them_to_the_same_slide(tmp_path):
    """Not a laziness assertion -- a guard on the fixtures above.

    Both tests read the slide through the sidecar TILE_FOR_BASIC wrote. If that sidecar
    ever stopped describing the same channel set, the per-channel read counts asserted
    above would drift for a reason unrelated to laziness.
    """
    slide, sidecar, ffp, dfp, names, stack, manifest = _tiled(tmp_path, 4)
    written = json.loads(sidecar.read_text())
    assert written["channel_names"] == names
    # DAPI (index 0) is the fiducial: fitted on nothing, corrected on nothing.
    assert written["profile_channels"] == [1, 2, 3]
    assert written["corrected_channels"] == [1, 2, 3]
    assert written["skipped_channels"] == [0]
