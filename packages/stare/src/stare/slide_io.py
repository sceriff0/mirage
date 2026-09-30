"""Small OME-TIFF I/O helpers shared by the tiled ('STARE') fan-out CLIs.

Keeps channel handling (promote 2-D to ``(C, H, W)``, pick the nuclear/fiducial channel, clamp to non-negative
and restore the source dtype on write) in one place so tiled_coarse / tiled_reg_tile / tiled_stitch
stay thin wrappers over the tested cores.
"""

from __future__ import annotations

import numpy as np


def load_channels(path):
    """Read an OME-TIFF as ``(C, H, W)`` (a 2-D image is promoted to ``C=1``)."""
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"{path}: expected 2-D or 3-D (C,H,W), got shape {arr.shape}")
    return arr


def nuclear_channel(arr_chw, index):
    """Return the ``index`` channel of a ``(C, H, W)`` array as float32."""
    if not 0 <= index < arr_chw.shape[0]:
        raise ValueError(
            f"--nuclear-index {index} out of range for C={arr_chw.shape[0]}"
        )
    return arr_chw[index].astype(np.float32)


def open_lazy(path):
    """Open an OME-TIFF as a lazy ``(C, H, W)`` array + a close callable, without loading it.

    Uses tifffile's zarr view so region reads (``arr[c, y0:y1, x0:x1]``) fetch only the tiles they
    touch — the property the streaming gigapixel stitch relies on. A 2-D image is presented as
    ``C=1``; a pyramidal OME-TIFF resolves to its base (full-resolution) level.

    EVERY CALL BUILDS A FRESH STORE -- ``tifffile.imread(path, aszarr=True)`` is invoked here,
    unconditionally, on every invocation, so two calls on the same ``path`` never share any
    mutable I/O state. This is a guarantee, not an implementation accident, and it is what
    makes the function safe to call once per worker: a caller that fans a single file out
    across threads gives each worker its OWN handle rather than sharing one, which sidesteps
    whether tifffile's zarr view is safe for concurrent region reads through a SHARED store
    (not documented either way) by construction. Pinned by
    ``tests/test_tiled_io.py::test_open_lazy_returns_a_fresh_store_per_call``.

    NO CALLER IN ``bin/`` IS CURRENTLY THREADED. The one that was -- the deleted in-process
    ``bin/preprocess.py``, one worker per channel -- went with the nf-core BASICPY swap, and
    the concurrency tests that evidenced its per-thread handles went with it. Every present
    consumer reads sequentially, so the guarantee above is currently unexercised rather than
    load-bearing. Do not remove it on that basis: it is a precondition for ever threading one
    of these loops again, and it costs nothing.

    Returns ``(arr, dtype, close)`` where ``arr`` is a zarr array with a NumPy-like getitem.
    """
    import tifffile
    import zarr

    store = tifffile.imread(str(path), aszarr=True)
    grp = zarr.open(store, mode="r")
    # base level if pyramidal. `zarr.Group` -- not `zarr.hierarchy.Group`, which zarr 3 removed;
    # the top-level name resolves under both majors, so this does not pin the container's zarr.
    arr = grp[0] if isinstance(grp, zarr.Group) else grp

    class _CHW:
        """Adapt a 2-D or (C,H,W) zarr array to a uniform (C,H,W) region-readable view."""

        def __init__(self, z):
            self._z = z
            self._2d = z.ndim == 2
            if z.ndim not in (2, 3):
                raise ValueError(
                    f"{path}: expected 2-D or 3-D (C,H,W), got shape {z.shape}"
                )
            self.shape = (1, *z.shape) if self._2d else tuple(z.shape)
            self.dtype = z.dtype

        def __getitem__(self, key):
            # Positional unpack, not by name: ome_io.py's read_plane is the caller that
            # matters here, and it builds `key` as a plain (channel, :, :)-shaped 3-tuple.
            c, ys, xs = key
            return self._z[ys, xs] if self._2d else self._z[c, ys, xs]

    return _CHW(arr), arr.dtype, store.close


def decimation_factor(shapes, max_dim):
    """Return the single integer decimation factor that puts every shape under ``max_dim``.

    ``shapes`` is an iterable of ``(H, W)``. One *shared* factor is deliberate: the coarse anchor
    matches learned descriptors between the reference and moving thumbnails, so both must sit at
    a common scale. Squeezing each slide independently under ``max_dim`` would introduce a scale
    change between the two thumbnails and bake a bogus scale term into M0.
    """
    if max_dim is None or max_dim <= 0:
        return 1
    longest = max(max(h, w) for h, w in shapes)
    return max(1, -(-longest // int(max_dim)))  # ceil division


# Byte budget for one streamed row band. Peak memory of a decimated read is this plus the
# (small) thumbnail, independent of slide width -- a fixed *row* count would not be, since a
# band of a 30k-wide slide is ~7x the same band of a 4k-wide one. ``band_rows_for`` sizes the
# band as if every source were float32 (4 bytes/px); the actual streamed band in
# ``read_decimated`` is read at the SOURCE's own dtype, so for a uint8/uint16 source this
# budget is conservative (bands come out smaller than it could afford -- free headroom, not a
# bug) and for a float64 source it would under-reserve by 2x (not exercised by any caller
# today, but worth knowing if one is ever added).
DEFAULT_BAND_BYTES = 64 * 1024 * 1024


def band_rows_for(width, factor, band_bytes=DEFAULT_BAND_BYTES):
    """Rows per streamed band: the most that fit in ``band_bytes`` AS IF THE SOURCE WERE
    FLOAT32 (this is a budget, not a measurement of what ``read_decimated`` actually streams
    -- it reads each band at the source's own dtype, so a narrower source uses less than this
    per band), snapped down to a multiple of ``factor`` (so the sampled rows stay globally
    aligned) and never below ``factor``.
    """
    factor = max(1, int(factor))
    per_row = (
        max(1, int(width)) * 4
    )  # budgeted as float32; see DEFAULT_BAND_BYTES's comment
    rows = max(factor, int(band_bytes) // per_row)
    return max(factor, (rows // factor) * factor)


def read_decimated(src, index, factor, band_bytes=DEFAULT_BAND_BYTES):
    """Read channel ``index`` of a lazy ``(C, H, W)`` view as a ``factor``-decimated float32
    plane.

    Always returns float32, unconditionally, for all three of this function's callers, which
    need it for three different reasons:

    - ``bin/tiled_coarse.py``'s anchor path (see ``normalize_intensity`` in
      ``bin/utils/coarse_align.py``): the learned matcher was trained on images in ``[0, 1]``,
      and the percentile rescale that puts them there needs a float buffer.
    - ``bin/utils/qc.py``'s ``create_registration_qc``: its only consumer of the result is
      ``autoscale_for_display`` on the way to uint8, and that function's min-max arithmetic
      runs directly on whatever dtype it is handed -- no upfront cast of its own. A narrower
      uint8/uint16 read changes that arithmetic (a uint16 true-divide promotes to float64; a
      float32 one stays float32) and demonstrably rounds ``np.round(normalized * 255)``
      differently for real data -- see
      ``tests/test_tiled_io.py::test_autoscale_uint16_vs_float32_disagree_at_a_real_triple``
      for a reproduced counter-example. A prior version of this function took an optional
      ``dtype=`` for exactly that narrow read; it was reverted (task 3, second pass) once
      measurement showed the memory it appeared to save at the read step does not survive the
      widen-back-to-float32 that correctness requires downstream -- see
      ``docs/perf/2026-08-26-rss.md`` for the measured numbers.
    - ``bin/generate_preprocess_qc.py:211`` (``read_decimated(lazy_arr, i, factor=1)``, one
      call per channel) feeds the result to ``normalize_image`` -> ``downsample_image`` ->
      ``imsave`` -- a display-only-to-PNG chain that LOOKS like the same trap as qc.py's, and
      is worth naming here for exactly that resemblance. It is actually different:
      ``normalize_image`` (percentile-based contrast, not min-max) does
      ``image.astype(np.float32)`` as its OWN first line, unconditionally, before touching the
      input at all. So this caller does not depend on ``read_decimated`` handing it float32
      already -- it would re-derive an equivalent float32 array safely even from a narrower
      read, the same way qc.py's reverted widen-back did. Its safety is real but incidental:
      it depends on ``normalize_image`` keeping that upfront cast, not on any contract
      ``read_decimated`` makes. Do not use this caller as precedent that narrowing is free
      elsewhere, and do not reintroduce a narrow-dtype parameter here without checking all
      three callers' actual arithmetic (not just their apparent shape) and re-measuring
      whether it is worth the complexity.

    Streams the plane in row bands and strides each band as it's read. Bands are written
    directly into a pre-allocated float32 destination array (never accumulated as a list and
    ``concatenate``d), so peak memory is one band's full-resolution buffer plus the single
    decimated destination -- at every factor, including ``factor == 1``, where a
    band-list-plus-concatenate would otherwise hold two full-resolution-sized planes at once.
    Band height is snapped to a multiple of ``factor`` so the rows sampled are globally
    ``0, factor, 2*factor, ...``; the destination is therefore filled with exactly
    ``src[index, ::factor, ::factor]`` at any factor, including a ragged final band (a height
    not a multiple of the band size).
    """
    import numpy as _np

    c_n, h, w = src.shape
    if not 0 <= index < c_n:
        raise ValueError(f"--nuclear-index {index} out of range for C={c_n}")
    factor = max(1, int(factor))
    band = band_rows_for(w, factor, band_bytes)

    out_h = -(
        -h // factor
    )  # ceil division: number of sampled rows 0, factor, 2*factor, ...
    out_w = -(-w // factor)
    dest = _np.empty((out_h, out_w), dtype=_np.float32)

    out_row = 0
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        band_arr = _np.asarray(src[index, slice(y0, y1), slice(0, w)])[
            ::factor, ::factor
        ]
        n_rows = band_arr.shape[0]
        dest[out_row : out_row + n_rows] = band_arr
        out_row += n_rows

    return dest
