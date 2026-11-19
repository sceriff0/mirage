"""SPLIT_CHANNELS reads one channel at a time instead of decoding the whole stack.

``bin/split_multichannel.py:split_multichannel_tiff`` writes one single-channel
TIFF per marker, and nothing needs every channel at once, so the whole
``(C, H, W)`` slide is never decoded together.

This file pins three properties:

1. The whole-stack eager read is gone: the image is opened through
   ``tiled_io.open_lazy`` and each channel is read on its own, one at a time
   (never all channels materialised together).
2. Every output TIFF has the same decoded pixels, dtype, and
   resolution/resolutionunit tags as what the old whole-stack-read path
   produced, with the same compression/bigtiff settings.
3. ``clip_negative_values`` (``bin/utils/validation.py``) computes GLOBAL
   whole-image statistics (negative-pixel count, percentage of the WHOLE
   image, min before/after) and emits ONE log line. Reading one channel at a
   time must not fragment that into one line per channel with per-channel
   (therefore wrong) percentages -- the code accumulates across channels and
   emits a single line with the numbers a whole-image call would produce.

The output is written tiled (``split_multichannel.SPLIT_TIFF_TILE``), while
``_old_split_multichannel_tiff`` is a striped whole-stack reference, so
equivalence is checked on decoded pixels, dtype and resolution tags, not bytes.
``test_the_write_is_tiled`` pins the tiled layout itself.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pytest

BIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"
)
UTILS_DIR = os.path.join(BIN_DIR, "utils")
for _dir in (BIN_DIR, UTILS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

import split_multichannel as sm  # noqa: E402


def _write_stack(
    tmp_path,
    channel_names,
    h=48,
    w=40,
    seed=7,
    dtype=np.uint16,
    axes="CYX",
    neg_channel_pixels=None,
    filename="P001_registered.ome.tiff",
):
    """A small multichannel OME-TIFF, planar (C, H, W) by default.

    ``neg_channel_pixels``, if given, is a dict of ``{channel_index: count}``:
    that many pixels in that channel are set to a negative value, to exercise
    ``clip_negative_values``. Only meaningful for a signed/float ``dtype``.
    """
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo = max(info.min, 0)
        planes = [
            rng.integers(lo, info.max, size=(h, w), dtype=dtype) for _ in channel_names
        ]
    else:
        planes = [rng.random(size=(h, w)).astype(dtype) * 1000.0 for _ in channel_names]
    stack = np.stack(planes, axis=0)  # (C, H, W)

    if neg_channel_pixels:
        flat_hw = h * w
        for ci, count in neg_channel_pixels.items():
            flat = stack[ci].reshape(-1)
            idx = rng.choice(flat_hw, size=count, replace=False)
            flat[idx] = -rng.integers(1, 100, size=count).astype(dtype)
            stack[ci] = flat.reshape(h, w)

    path = tmp_path / filename
    if axes == "YXC":
        stack = np.transpose(stack, (1, 2, 0))  # (H, W, C)
        tifffile.imwrite(str(path), stack, photometric="minisblack")
    else:
        tifffile.imwrite(
            str(path),
            stack,
            ome=True,
            photometric="minisblack",
            metadata={"axes": axes, "Channel": {"Name": list(channel_names)}},
        )
    return path


def _old_split_multichannel_tiff(
    input_path, output_dir, is_reference, channel_names, nuclear_markers, pixel_size
):
    """Whole-stack read path with a striped write, kept as the independent
    reference for the equivalence tests below.
    """
    from image_utils import ensure_dir, normalize_image_dimensions
    from metadata import extract_channel_names_from_ome, is_nuclear
    from pixel_size import read_ome_pixel_size, warn_on_pixel_size_mismatch
    from validation import clip_negative_values

    logger = logging.getLogger("old_split_multichannel")

    img = tifffile.imread(input_path)
    img = clip_negative_values(img, logger, stage_name="split_multichannel")
    data = normalize_image_dimensions(img)
    n_channels = data.shape[0]

    if channel_names is None:
        channel_names = extract_channel_names_from_ome(input_path) or None
    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(n_channels)]
    else:
        channel_names = list(channel_names)

    if len(channel_names) != n_channels:
        if len(channel_names) < n_channels:
            channel_names.extend(
                [f"Channel_{i}" for i in range(len(channel_names), n_channels)]
            )
        else:
            channel_names = channel_names[:n_channels]

    ensure_dir(output_dir)

    detected = read_ome_pixel_size(input_path)
    warn_on_pixel_size_mismatch(
        detected, pixel_size, source=str(input_path), logger=logger
    )
    resolution = (1e4 / pixel_size, 1e4 / pixel_size)

    saved_paths = []
    for i, name in enumerate(channel_names):
        is_nuclear_channel = is_nuclear(name, nuclear_markers)
        if is_nuclear_channel and not is_reference:
            continue

        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        candidate_path = os.path.join(output_dir, f"{clean_name}.tiff")
        if os.path.exists(candidate_path):
            suffix = 2
            while os.path.exists(
                os.path.join(output_dir, f"{clean_name}_{suffix}.tiff")
            ):
                suffix += 1
            clean_name = f"{clean_name}_{suffix}"
        output_path = os.path.join(output_dir, f"{clean_name}.tiff")

        channel_data = data[i]
        tifffile.imwrite(
            output_path,
            channel_data,
            bigtiff=True,
            compression="zlib",
            resolution=resolution,
            resolutionunit="CENTIMETER",
        )
        saved_paths.append(output_path)

    return saved_paths


def _assert_pixels_and_tags_equivalent(old_path, new_path):
    """Decoded-pixel equivalence, used in place of a byte comparison.

    The writer tiles its output (``split_multichannel.SPLIT_TIFF_TILE``) while
    ``_old_split_multichannel_tiff`` above does not, so the two files' raw bytes
    differ (different TIFF tags, different tile/strip layout).
    What must still match is the published PIXELS, dtype, and the resolution tags
    ``split_multichannel_tiff`` stamps on for MERGE_AND_PYRAMID to read.
    """
    old_arr = tifffile.imread(str(old_path))
    new_arr = tifffile.imread(str(new_path))
    np.testing.assert_array_equal(old_arr, new_arr)
    assert old_arr.dtype == new_arr.dtype

    with (
        tifffile.TiffFile(str(old_path)) as old_tif,
        tifffile.TiffFile(str(new_path)) as new_tif,
    ):
        old_tags, new_tags = old_tif.pages[0].tags, new_tif.pages[0].tags
        assert new_tags["XResolution"].value == old_tags["XResolution"].value
        assert new_tags["YResolution"].value == old_tags["YResolution"].value
        assert new_tags["ResolutionUnit"].value == old_tags["ResolutionUnit"].value


def test_split_multichannel_reads_one_channel_at_a_time(tmp_path, monkeypatch):
    """The slide must be reached through ``open_lazy`` and read one channel at a
    time, never via a whole-array ``tifffile.imread``.

    """
    channel_names = ["DAPI", "CD4", "CD8"]
    image_path = _write_stack(tmp_path, channel_names)
    out_dir = tmp_path / "out"

    seen_open_paths = []
    orig_open_lazy = sm.open_lazy

    seen_indices = []

    def spying_open_lazy(path):
        seen_open_paths.append(str(path))
        arr, dtype, close = orig_open_lazy(path)

        class _Spy:
            shape = arr.shape
            dtype = arr.dtype

            def __getitem__(self, key):
                seen_indices.append(key[0])
                return arr[key]

        return _Spy(), dtype, close

    monkeypatch.setattr(sm, "open_lazy", spying_open_lazy)

    imread_full_calls = []
    orig_imread = tifffile.imread

    def spying_imread(path, *args, **kwargs):
        if not kwargs.get("aszarr", False):
            imread_full_calls.append(str(path))
        return orig_imread(path, *args, **kwargs)

    monkeypatch.setattr(sm.tifffile, "imread", spying_imread)

    sm.split_multichannel_tiff(
        str(image_path),
        str(out_dir),
        is_reference=True,
        channel_names=list(channel_names),
        nuclear_markers=["DAPI"],
        pixel_size=0.325,
    )

    assert str(image_path) in seen_open_paths, "image never went through open_lazy"
    assert str(image_path) not in imread_full_calls, (
        "image was still eagerly decoded in full via tifffile.imread"
    )
    assert seen_indices == [0, 1, 2], (
        f"expected one read per channel, in order, once each; got {seen_indices}"
    )


@pytest.mark.parametrize("is_reference", [True, False])
@pytest.mark.parametrize("dtype", [np.uint16, np.int16])
def test_split_multichannel_outputs_pixel_identical_to_old_whole_stack_read(
    tmp_path, is_reference, dtype
):
    """Every output TIFF from the per-channel-read path must have the same
    decoded pixels, dtype, and resolution tags as the whole-stack reference,
    including a case with real negative pixels (int16) that exercises
    clip_negative_values.

    Not a byte comparison: the writer tiles its output, so its bytes differ
    from the striped reference.
    """
    channel_names = ["DAPI", "CD4", "CD8"]
    neg = {1: 25, 2: 5} if dtype == np.int16 else None
    image_path = _write_stack(
        tmp_path, channel_names, dtype=dtype, neg_channel_pixels=neg
    )

    old_dir = tmp_path / "old_out"
    old_paths = _old_split_multichannel_tiff(
        str(image_path),
        str(old_dir),
        is_reference,
        list(channel_names),
        ["DAPI"],
        0.325,
    )

    new_dir = tmp_path / "new_out"
    new_paths = sm.split_multichannel_tiff(
        str(image_path),
        str(new_dir),
        is_reference=is_reference,
        channel_names=list(channel_names),
        nuclear_markers=["DAPI"],
        pixel_size=0.325,
    )

    assert [os.path.basename(p) for p in new_paths] == [
        os.path.basename(p) for p in old_paths
    ]
    assert len(new_paths) > 0

    for old_path, new_path in zip(old_paths, new_paths):
        _assert_pixels_and_tags_equivalent(old_path, new_path)


def test_split_multichannel_channel_last_layout_still_matches_old_read(tmp_path):
    """A hypothetical (Y, X, C) input (never produced by this pipeline's own
    producers, but a defensive branch handles it) must fall back to a
    whole-stack read and produce output equivalent to the reference
    (pixels/dtype/resolution tags).
    """
    channel_names = ["DAPI", "CD4"]
    image_path = _write_stack(tmp_path, channel_names, axes="YXC")

    old_dir = tmp_path / "old_out"
    old_paths = _old_split_multichannel_tiff(
        str(image_path), str(old_dir), True, list(channel_names), ["DAPI"], 0.325
    )

    new_dir = tmp_path / "new_out"
    new_paths = sm.split_multichannel_tiff(
        str(image_path),
        str(new_dir),
        is_reference=True,
        channel_names=list(channel_names),
        nuclear_markers=["DAPI"],
        pixel_size=0.325,
    )

    assert len(new_paths) == len(old_paths) == len(channel_names)
    for old_path, new_path in zip(old_paths, new_paths):
        _assert_pixels_and_tags_equivalent(old_path, new_path)


def test_split_multichannel_negative_clip_stats_logged_once_for_whole_image(
    tmp_path, caplog
):
    """clip_negative_values' whole-image stats (count, percentage of the WHOLE
    image, min before/after) must be logged exactly ONCE, with numbers matching
    a real whole-array computation -- not fragmented into one line per channel
    with per-channel (and therefore wrong) percentages.

    Channel 0 (DAPI, the nuclear channel) is also given negative pixels
    too, and the run is is_reference=False (so DAPI is READ, for stats purposes,
    but never WRITTEN) -- pinning that a channel skipped from output still has
    to count toward the aggregate.
    """
    channel_names = ["DAPI", "CD4", "CD8"]
    neg_by_channel = {0: 7, 1: 13, 2: 3}
    image_path = _write_stack(
        tmp_path,
        channel_names,
        dtype=np.int16,
        neg_channel_pixels=neg_by_channel,
    )

    # Independent expected aggregate, computed directly from the whole array.
    whole = tifffile.imread(str(image_path))
    expected_count = int(np.sum(whole < 0))
    expected_pct = expected_count / whole.size * 100
    expected_min_before = int(whole.min())

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.DEBUG, logger=sm.logger.name):
        sm.split_multichannel_tiff(
            str(image_path),
            str(out_dir),
            is_reference=False,
            channel_names=list(channel_names),
            nuclear_markers=["DAPI"],
            pixel_size=0.325,
        )

    clipped_lines = [
        r
        for r in caplog.records
        if "Clipped" in r.message and "negative pixels" in r.message
    ]
    detected_lines = [
        r
        for r in caplog.records
        if r.message.startswith("Detected") and "negative pixels" in r.message
    ]

    assert len(clipped_lines) == 1, (
        f"expected exactly one aggregate 'Clipped ...' line, got {len(clipped_lines)}: "
        f"{[r.message for r in clipped_lines]}"
    )
    assert len(detected_lines) == 1, (
        f"expected exactly one aggregate 'Detected ...' line, got {len(detected_lines)}: "
        f"{[r.message for r in detected_lines]}"
    )

    clipped_msg = clipped_lines[0].message
    assert f"Clipped {expected_count} negative pixels" in clipped_msg
    assert f"({expected_pct:.4f}%)" in clipped_msg
    assert f"Min before: {expected_min_before}" in clipped_msg

    detected_msg = detected_lines[0].message
    assert f"Detected {expected_count} negative pixels" in detected_msg
    assert f"({expected_pct:.4f}%)" in detected_msg


# ---------------------------------------------------------------------------
# the tiled layout
# ---------------------------------------------------------------------------


def test_the_write_is_tiled(tmp_path):
    """Each single-channel TIFF is written tiled.

    A striped TIFF (tifffile's default) is one zarr chunk per plane, so every
    region read downstream (MERGE_AND_PYRAMID, quantification) decodes the
    whole plane. Checked two ways:

    1. the TIFF tags tifffile itself writes (``TileWidth``/``TileLength``),
       against ``sm.SPLIT_TIFF_TILE`` rather than a bare ``2048``, so the two
       stay in sync if the constant ever changes;
    2. the zarr chunk shape the pipeline's own region reader actually sees
       (``bin/utils/tiled_io.py:open_lazy`` uses this ``aszarr=True`` +
       ``zarr.open`` pattern), since the tags could be present without the zarr
       view honouring them.

    The striped reference writer (``_old_split_multichannel_tiff``) is asserted
    to be striped, as the contrast.
    """
    import zarr

    channel_names = ["DAPI", "CD4"]
    image_path = _write_stack(tmp_path, channel_names)

    old_dir = tmp_path / "old_out"
    old_paths = _old_split_multichannel_tiff(
        str(image_path), str(old_dir), True, list(channel_names), ["DAPI"], 0.325
    )

    new_dir = tmp_path / "new_out"
    new_paths = sm.split_multichannel_tiff(
        str(image_path),
        str(new_dir),
        is_reference=True,
        channel_names=list(channel_names),
        nuclear_markers=["DAPI"],
        pixel_size=0.325,
    )

    new_path = new_paths[0]
    old_path = old_paths[0]

    with tifffile.TiffFile(str(new_path)) as new_tif:
        new_tags = new_tif.pages[0].tags
        assert new_tags["TileWidth"].value == sm.SPLIT_TIFF_TILE
        assert new_tags["TileLength"].value == sm.SPLIT_TIFF_TILE

    with tifffile.TiffFile(str(old_path)) as old_tif:
        old_tag_names = {t.name for t in old_tif.pages[0].tags}
        assert "TileWidth" not in old_tag_names, (
            "the reference writer is supposed to be striped, not tiled -- if this fires, "
            "tifffile's default changed and the contrast this test relies on is gone"
        )

    def _chunks(path):
        store = tifffile.imread(str(path), aszarr=True)
        try:
            grp = zarr.open(store, mode="r")
            arr = grp[0] if isinstance(grp, zarr.Group) else grp
            return arr.chunks
        finally:
            store.close()

    new_shape = tifffile.imread(str(new_path)).shape
    assert _chunks(new_path) == (sm.SPLIT_TIFF_TILE, sm.SPLIT_TIFF_TILE)
    assert _chunks(old_path) == new_shape
