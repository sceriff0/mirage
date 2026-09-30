"""Preprocess QC must stop eagerly decoding the whole multi-channel stack.

``bin/generate_preprocess_qc.py:generate_preprocess_qc`` used to call
``tifffile.imread`` on the *entire* ``(C, H, W)`` stack just to slice out one
channel at a time, normalize it, and downsample it for a PNG preview -- an image
with a dozen-plus marker channels means every one of them sits fully decoded in
memory while only one is ever used at a time.

This is a peak-MEMORY fix, not an I/O-size fix: ``downsample_image`` uses
``skimage.transform.resize(order=1, anti_aliasing=True)``, an anti-aliased
(area-averaged) resize that needs every source pixel by definition, so there is no
read-size saving available here (unlike the naive ``[::factor, ::factor]``
decimation ``tiled_io.read_decimated`` performs at nonzero ``factor``, which is
NOT used at this call site -- see ``bin/generate_preprocess_qc.py``'s comment).
``bin/utils/tiled_io.py`` already has the machinery (proven correct by the STARE
coarse-align path and by Task 1's registration-QC change) to read one channel
through a lazy, region-readable zarr view (``open_lazy``) banded via
``read_decimated(..., factor=1)`` -- full resolution, one channel, one plane (or
band) of memory at a time.

This file pins two properties for routing ``generate_preprocess_qc`` through it:

1. The whole-stack eager read is gone -- the image is opened through ``open_lazy``,
   and channels are read one at a time via ``read_decimated``.
2. The result is EXACTLY what the old whole-stack read produced, not merely
   similar: same PNG bytes, same shape, same dtype.

Property 2 does not fail against the pre-change code by construction: the
pre-change code already reads the correct channel via ``tifffile.imread``, so an
"old vs. new" pixel comparison built from the same channel index is old-vs-old
before the fix lands, and only exercises the new code path's arithmetic once the
fix is in. It is included anyway as a regression guard against the real failure
mode here -- silently different pixels, not a crash -- and was falsified during
development by temporarily comparing against the WRONG channel (see
task-2-report.md for the transcript), which failed until the mutation was
reverted, proving the comparison is not vacuous.
"""

from __future__ import annotations

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

pytest.importorskip("skimage")
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

import generate_preprocess_qc as gpq  # noqa: E402


def _write_stack(tmp_path, channel_names, h=64, w=48, seed=1234, axes="CYX"):
    """A small multichannel uint16 OME-TIFF, planar (C, H, W) by default.

    The channel-last (Y, X, C) case is written as a plain (non-OME) TIFF: tifffile's
    OME writer rejects a contiguous interleaved "YXC" layout for a non-RGB channel
    count under `photometric="minisblack"` ("shape does not match stored shape").
    `generate_preprocess_qc`'s channel-last detection is purely shape-based (it
    never reads OME channel metadata), so a plain TIFF exercises it identically.
    """
    rng = np.random.default_rng(seed)
    planes = [
        rng.integers(0, 65536, size=(h, w), dtype=np.uint16) for _ in channel_names
    ]
    stack = np.stack(planes, axis=0)  # (C, H, W)

    path = tmp_path / "P001_corrected.ome.tiff"
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


def _old_generate_preprocess_qc(image_path, output_dir, channel_names, scale_factor):
    """Reimplements the PRE-CHANGE whole-stack read path, using the untouched
    ``normalize_image``/``downsample_image`` functions, to build an independent
    "old" reference for the equivalence tests below.
    """
    img_stack = tifffile.imread(image_path)
    if img_stack.ndim == 2:
        img_stack = np.expand_dims(img_stack, axis=0)
    if (
        img_stack.ndim == 3
        and img_stack.shape[2] == len(channel_names)
        and img_stack.shape[0] != len(channel_names)
    ):
        img_stack = np.transpose(img_stack, (2, 0, 1))

    n_channels = img_stack.shape[0]
    names = channel_names + [
        f"Channel_{i}" for i in range(len(channel_names), n_channels)
    ]
    names = names[:n_channels]

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for i, name in enumerate(names):
        normalized = gpq.normalize_image(img_stack[i])
        downsampled = gpq.downsample_image(normalized, scale_factor)
        outputs[name] = downsampled
    return outputs


def test_preprocess_qc_never_eagerly_decodes_the_whole_stack(tmp_path, monkeypatch):
    """The image must be reached through ``open_lazy`` and per-channel ``read_decimated``
    calls, not a whole-array ``tifffile.imread``.

    Fails against the pre-change code, which has no ``open_lazy``/``read_decimated``
    attribute on the ``generate_preprocess_qc`` module at all -- this assertion cannot
    even be wired up until the routing lands.
    """
    channel_names = ["DAPI", "CD4", "CD8"]
    image_path = _write_stack(tmp_path, channel_names)
    out_dir = tmp_path / "qc"

    seen_paths = []
    orig_open_lazy = gpq.open_lazy

    def spying_open_lazy(path):
        seen_paths.append(str(path))
        return orig_open_lazy(path)

    monkeypatch.setattr(gpq, "open_lazy", spying_open_lazy)

    seen_indices = []
    orig_read_decimated = gpq.read_decimated

    def spying_read_decimated(src, index, factor, *args, **kwargs):
        seen_indices.append(index)
        return orig_read_decimated(src, index, factor, *args, **kwargs)

    monkeypatch.setattr(gpq, "read_decimated", spying_read_decimated)

    imread_full_calls = []
    orig_imread = tifffile.imread

    def spying_imread(path, *args, **kwargs):
        # A lazy zarr open still goes through tifffile.imread(..., aszarr=True); only a
        # plain call (no aszarr) actually materialises pixel data eagerly.
        if not kwargs.get("aszarr", False):
            imread_full_calls.append(str(path))
        return orig_imread(path, *args, **kwargs)

    monkeypatch.setattr(gpq.tifffile, "imread", spying_imread)

    gpq.generate_preprocess_qc(
        image_path=image_path,
        output_dir=out_dir,
        channel_names=list(channel_names),
        scale_factor=0.5,
    )

    assert str(image_path) in seen_paths, "image never went through open_lazy"
    assert str(image_path) not in imread_full_calls, (
        "image was still eagerly decoded in full via tifffile.imread"
    )
    assert seen_indices == [0, 1, 2], (
        f"expected one read_decimated call per channel, in order; got {seen_indices}"
    )


def test_preprocess_qc_png_pixels_match_old_whole_stack_read(tmp_path):
    """The per-channel-read PNGs must equal the old whole-stack-read PNGs bit-for-bit."""
    channel_names = ["DAPI", "CD4", "CD8"]
    image_path = _write_stack(tmp_path, channel_names)

    expected = _old_generate_preprocess_qc(
        image_path, tmp_path / "old_out", channel_names, scale_factor=0.5
    )

    out_dir = tmp_path / "new_out"
    output_files = gpq.generate_preprocess_qc(
        image_path=image_path,
        output_dir=out_dir,
        channel_names=list(channel_names),
        scale_factor=0.5,
    )

    assert len(output_files) == len(channel_names)
    for name, out_path in zip(channel_names, output_files):
        got = _read_png(out_path)
        exp = expected[name]
        assert got.dtype == exp.dtype
        assert got.shape == exp.shape
        assert np.array_equal(got, exp), (
            f"channel {name} PNG differs from old read path"
        )


def _read_png(path):
    from skimage.io import imread as skio_imread

    return skio_imread(str(path))


def test_preprocess_qc_channel_last_layout_still_matches_old_read(tmp_path):
    """A hypothetical (Y, X, C) input (never produced by this pipeline's own writers, but
    the original defensive branch handled it) must still fall back to a whole-stack read
    and produce output identical to the pre-change code.
    """
    channel_names = ["DAPI", "CD4"]
    image_path = _write_stack(tmp_path, channel_names, axes="YXC")

    expected = _old_generate_preprocess_qc(
        image_path, tmp_path / "old_out", channel_names, scale_factor=0.5
    )

    out_dir = tmp_path / "new_out"
    output_files = gpq.generate_preprocess_qc(
        image_path=image_path,
        output_dir=out_dir,
        channel_names=list(channel_names),
        scale_factor=0.5,
    )

    assert len(output_files) == len(channel_names)
    for name, out_path in zip(channel_names, output_files):
        got = _read_png(out_path)
        exp = expected[name]
        assert got.dtype == exp.dtype
        assert got.shape == exp.shape
        assert np.array_equal(got, exp)
