"""A stand-in for the ``ashlar`` package, for the adapter's tests where ASHLAR is not
installed (it is pinned at 1.20.0 and lives only in its own image).

It does three things and nothing else:

* **records** every call the adapter's wrapper makes into it (``CALLS``: the command line,
  each aligner's constructor keywords, the writer), so a test can assert that the command
  is ASHLAR's own and that nothing but the documented flags reaches it;
* keeps ASHLAR's **geometry conventions**, transcribed from labsyspharm/ashlar at the tag
  v1.20.0 (``reg.py`` ``[PARTIAL: lines 1-119, 436-810, 836-1375]``, ``utils.py``,
  ``scripts/ashlar.py``, ``filepattern.py`` ``[FULL]``): the ``reader|path|key=value``
  argument, the row-major tile order and ``[row, col] * tile_size * (1 - overlap)``
  positions of the file-pattern reader (``filepattern.py:90-100``), the reference cycle's
  origin moved to zero (``reg.py:713-716``), ``LayerAligner.calculate_positions`` and
  ``constrain_positions`` (``reg.py:900-946``), the mosaic's shape (``reg.py:791-795``);
* **draws** the mosaic the way ``Mosaic.assemble_channel`` and ``utils.paste`` are written
  to (``reg.py:1137-1163``, ``utils.py:149-216``): tile after tile at its position, the
  whole part by slicing, the fractional part by ``scipy.ndimage.shift``, blended by the
  distance to unfilled space. That is an emulation of the writer written from the source,
  never the real one, and it writes one resolution level, not a pyramid.

It does NOT register anything: the reference tiles get the shifts ``CONFIG["edge_shifts"]``
names and each later cycle ``CONFIG["layer_shifts"]``. The selfcheck does not need a good
registration, only a writer and a point path that must agree.
"""

from __future__ import annotations

import argparse
import re
import sys
import types
import warnings as _warnings
from pathlib import Path

import numpy as np

VERSION = "1.20.0"
CALLS = []
# What a test may set before the command runs.
CONFIG = {
    "version": VERSION,
    # f(n_rows, n_cols) -> (n, 2) (dy, dx): where the reference stitch moves each tile
    "edge_shifts": None,
    # f(cycle, n_rows, n_cols) -> (n, 2) (dy, dx): each later cycle's per-tile shifts
    "layer_shifts": None,
    # f(cycle) -> (dy, dx): the thumbnail offset of a later cycle
    "cycle_offset": None,
    # f(cycle, n) -> (n,) errors of a later cycle's tiles (inf = no registration)
    "layer_errors": None,
    "raise_in": None,  # "edge", or a cycle number: raise ASHLAR's clean-tile RuntimeError
    "exit_code": 0,  # non-zero: the command prints an error and returns it, no mosaic
    "warn": None,  # a DataWarning text raised by the reference stitch
    # BioformatsReader: {"positions": (n, 2) (row, col) in PIXELS, "pixel_size": um,
    # "planes": f(series, c) -> 2-D array, "num_channels": int}
    "bioformats": None,
    # f(metadata) -> (n, 2) (row, col) px: what EdgeAligner.run() finds for a reader that
    # is not the file-pattern one (a ``reg.Reader`` subclass, e.g. regbench's front-end);
    # run() then reads every tile of its channel through ``self.reader`` twice, as the real
    # one does (``make_thumbnail``, then the pairs), so a cache would hold them all
    "edge_positions": None,
}
DEFAULTS = dict(CONFIG)
RAISE_TEXT = "correlation > total_amplitude (diff=3.0517578125e-05)"
_STATE = {"cycle": 0}


def record(name, **kw):
    CALLS.append({"call": name, **kw})


def calls(name):
    return [c for c in CALLS if c["call"] == name]


def default_layer_shifts(cycle, n_rows, n_cols):
    """A different shift for every tile, to a tenth of a pixel, up to 12 px either way."""
    rng = np.random.default_rng([7, int(cycle)])
    return np.round(rng.uniform(-12.0, 12.0, (n_rows * n_cols, 2)), 1)


# ── the file-pattern reader ───────────────────────────────────────────────────
class _Metadata:
    """``FilePatternMetadata`` (``filepattern.py:16-100``)."""

    def __init__(self, path, pattern, overlap, pixel_size):
        import tifffile

        self.path, self.pattern = Path(path), pattern
        self.overlap, self.pixel_size = float(overlap), float(pixel_size)
        regex = re.sub(
            r"{([^:}]+)(?:[^}]*)}", r"(?P<\1>.*?)", pattern.replace(".", r"\.")
        )
        rows, cols = set(), set()
        for p in self.path.iterdir():
            m = re.match(regex, p.name)
            if m:
                rows.add(int(m["row"]))
                cols.add(int(m["col"]))
        self.height, self.width = len(rows), len(cols)
        self.num_images = self.height * self.width
        first = tifffile.imread(self.path / pattern.format(row=0, col=0))
        self.size = np.array(first.shape[-2:])
        self.pixel_dtype = first.dtype
        self.num_channels = first.shape[0] if first.ndim == 3 else 1
        self.positions = np.vstack(
            [
                np.array([i // self.width, i % self.width])
                * self.size
                * (1 - self.overlap)
                for i in range(self.num_images)
            ]
        )

    @property
    def origin(self):
        return self.positions.min(axis=0)


class FilePatternReader:
    def __init__(self, path, pattern, overlap, pixel_size=1.0):
        self.metadata = _Metadata(path, pattern, overlap, pixel_size)

    def read(self, series, c):
        import tifffile

        m = self.metadata
        name = m.pattern.format(row=series // m.width, col=series % m.width)
        img = tifffile.imread(m.path / name)
        return img[c] if img.ndim == 3 else img


def build_reader(spec):
    """``scripts/ashlar.py:356-394``: ``reader|path|key=value|...``, values parsed as
    floats where they can be."""
    name, path, rest = spec.split("|", 2)
    if name != "filepattern":
        raise ProcessingError(f"Unknown reader: {name}")
    kwargs = {}
    for piece in rest.split("|"):
        key, value = piece.split("=")
        try:
            value = float(value)
        except ValueError:
            pass
        kwargs[key] = value
    return FilePatternReader(path, **kwargs)


# ── the Bio-Formats reader ────────────────────────────────────────────────────
def stage_to_positions(stage_yx_um, pixel_size, format_name="Zeiss"):
    """``BioformatsMetadata.tile_position`` (``reg.py:349-389``, ashlar 1.20.0), for the
    stage coordinates Bio-Formats hands over as (Y, X) micrometres::

        position_microns *= [-1, 1]            # the y-axis flipped (not for InCell 1000/2000)
        position_pixels = position_microns / self.pixel_size

    so ``ashlar.reg``'s ``metadata.positions`` are in PIXELS of the image, ordered
    (row, col) = (Y, X)."""
    pos = np.array(stage_yx_um, dtype=float)
    if format_name != "InCell 1000/2000":
        pos = pos * [-1, 1]
    return pos / pixel_size


class _BioformatsMetadata:
    def __init__(self, cfg):
        self.positions = np.asarray(cfg["positions"], dtype=float)
        self.pixel_size = float(cfg["pixel_size"])
        self.num_images = len(self.positions)
        self.num_channels = int(cfg.get("num_channels", 1))


class BioformatsReader:
    """``ashlar.reg.BioformatsReader(path)``: the geometry and the planes come from
    ``CONFIG["bioformats"]`` (nothing is opened)."""

    def __init__(self, path, plate=None, well=None):
        cfg = CONFIG["bioformats"]
        if cfg is None:
            raise RuntimeError("fake_ashlar: CONFIG['bioformats'] is not set")
        record("BioformatsReader", path=str(path))
        self.path = path
        self.metadata = _BioformatsMetadata(cfg)
        self._planes = cfg["planes"]

    def read(self, series, c):
        record("BioformatsReader.read", series=series, c=c)
        return self._planes(series, c)


# ── ASHLAR's reader interface (reg.py:58-119, 186-189, 436-462) ──────────────
class Metadata:
    """``reg.Metadata``: a subclass gives ``_num_images``, ``num_channels``, ``pixel_size``,
    ``pixel_dtype``, ``tile_position(i)`` (row, col px) and ``tile_size(i)``; the rest is
    derived here exactly as ``reg.py:83-119`` derives it."""

    @property
    def _num_images(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def pixel_size(self):
        raise NotImplementedError

    @property
    def pixel_dtype(self):
        raise NotImplementedError

    def tile_position(self, i):
        raise NotImplementedError

    def tile_size(self, i):
        raise NotImplementedError

    @property
    def num_images(self):
        return self._num_images

    @property
    def positions(self):
        if not hasattr(self, "_positions"):
            self._positions = np.vstack(
                [self.tile_position(i) for i in range(self._num_images)]
            )
        return self._positions

    @property
    def size(self):
        if not hasattr(self, "_size"):
            s0 = self.tile_size(0)
            image_ids = range(1, self._num_images)
            if any(any(self.tile_size(i) != s0) for i in image_ids):
                raise ValueError("Image series must all have the same dimensions")
            self._size = s0
        return self._size

    @property
    def centers(self):
        return self.positions + self.size / 2

    @property
    def origin(self):
        return self.positions.min(axis=0)


class Reader:
    """``reg.Reader``: ``read(series, c)`` and a ``metadata`` attribute."""

    def read(self, series, c):
        raise NotImplementedError


class CachingReader(Reader):
    """``reg.py:436-462``: keeps every tile of ``channel`` it has read, forever."""

    def __init__(self, reader, channel):
        self.reader = reader
        self.channel = channel
        self._cache = {}

    @property
    def metadata(self):
        return self.reader.metadata

    def read(self, series, c):
        if c == self.channel and series in self._cache:
            img = self._cache[series]
        else:
            img = self.reader.read(series, c)
        if c == self.channel and series not in self._cache:
            self._cache[series] = img
        return img


# ── the aligners ──────────────────────────────────────────────────────────────
class Warning(UserWarning):  # noqa: A001 - ASHLAR's own name (reg.py:1372)
    pass


class DataWarning(Warning):
    pass


class _Graph:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


class _Model:
    """The fitted linear model: here the identity plus an intercept."""

    def __init__(self, intercept):
        self.coef_ = np.eye(2)
        self.intercept_ = np.asarray(intercept, dtype=float)

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.coef_.T + self.intercept_


class EdgeAligner:
    def __init__(self, reader, **kwargs):
        record("EdgeAligner", **kwargs)
        self.channel = kwargs.get("channel", 0)
        # reg.py:502: the aligner wraps whatever reader it is given in a CachingReader
        self.reader = CachingReader(reader, self.channel)
        # reg.py:505-506: "Unit is micrometers."
        self.max_shift = kwargs.get("max_shift", 15)
        self.max_shift_pixels = self.max_shift / self.metadata.pixel_size

    @property
    def metadata(self):
        return self.reader.metadata

    def run(self):
        if CONFIG["raise_in"] == "edge":
            raise RuntimeError(RAISE_TEXT)
        if CONFIG["warn"]:
            _warnings.warn(CONFIG["warn"], DataWarning, stacklevel=1)
        m = self.metadata
        if CONFIG["edge_positions"] is not None:
            reader = self.reader
            record("EdgeAligner.run", reader=type(reader).__name__)
            for _pass in range(2):  # make_thumbnail, then register_all (reg.py:517-524)
                for i in range(m.num_images):
                    reader.read(series=i, c=self.channel)
            self.positions = np.array(CONFIG["edge_positions"](m), dtype=float)
            # reg.py:713-716
            self.origin = self.positions.min(axis=0)
            self.positions = self.positions - self.origin
            self.lr = _Model(-self.origin)
            self.centers = self.positions + m.size / 2
            return
        shifts = np.zeros((m.num_images, 2))
        if CONFIG["edge_shifts"] is not None:
            shifts = np.asarray(CONFIG["edge_shifts"](m.height, m.width), dtype=float)
        self.shifts = shifts
        self.positions = m.positions + shifts
        # reg.py:713-716: positions and the model's intercept moved so the origin is 0, 0
        self.origin = self.positions.min(axis=0)
        self.positions = self.positions - self.origin
        self.lr = _Model(-self.origin)
        self.max_error = 1.0
        edges = m.height * (m.width - 1) + m.width * (m.height - 1)
        self.neighbors_graph = _Graph(edges)
        self.spanning_tree = _Graph(m.num_images - 1)
        _STATE["cycle"] = 0

    @property
    def mosaic_shape(self):
        upper = self.positions + self.metadata.size
        return tuple(int(v) for v in np.ceil(upper.max(axis=0)))


class LayerAligner:
    def __init__(self, reader, reference_aligner, **kwargs):
        record("LayerAligner", **kwargs)
        self.reader, self.reference_aligner = reader, reference_aligner
        self.max_shift = kwargs.get("max_shift", 15)
        self.max_shift_pixels = self.max_shift / reader.metadata.pixel_size

    @property
    def metadata(self):
        return self.reader.metadata

    def run(self):
        from scipy.spatial.distance import cdist

        _STATE["cycle"] += 1
        cycle = _STATE["cycle"]
        if CONFIG["raise_in"] == cycle:
            raise RuntimeError(RAISE_TEXT)
        m, ref = self.metadata, self.reference_aligner
        offset = CONFIG["cycle_offset"](cycle) if CONFIG["cycle_offset"] else (0.0, 0.0)
        self.cycle_offset = np.asarray(offset, dtype=float)
        # reg.py:872-884
        self.corrected_nominal_positions = m.positions + self.cycle_offset
        reference_positions = ref.metadata.positions
        dist = cdist(reference_positions, self.corrected_nominal_positions)
        self.reference_idx = np.argmin(dist, 0)
        self.reference_positions = reference_positions[self.reference_idx]
        self.reference_aligner_positions = ref.positions[self.reference_idx]
        shifts = (CONFIG["layer_shifts"] or default_layer_shifts)(
            cycle, m.height, m.width
        )
        self.shifts = np.asarray(shifts, dtype=float)
        self.errors = np.full(m.num_images, 0.5)
        if CONFIG["layer_errors"] is not None:
            self.errors = np.asarray(
                CONFIG["layer_errors"](cycle, m.num_images), dtype=float
            )
        # reg.py:900-906
        self.positions = (
            self.corrected_nominal_positions
            + self.shifts
            + self.reference_aligner_positions
            - self.reference_positions
        )
        self.constrain_positions()

    def constrain_positions(self):
        """``reg.py:910-946``, statement for statement."""
        position_diffs = np.absolute(self.positions - self.reference_aligner_positions)
        position_diffs = np.rint(position_diffs * 10) / 10
        discard = (position_diffs == 0).all(axis=1)
        discard |= np.isinf(self.errors)
        if discard.all():
            offset = 0
        else:
            offset = np.nan_to_num(np.median(self.shifts[~discard], axis=0))
        predictions = self.reference_aligner.lr.predict(
            self.corrected_nominal_positions
        )
        distance = np.linalg.norm(self.positions - predictions - offset, axis=1)
        extremes = distance > self.max_shift_pixels
        discard |= extremes
        self.discard = discard
        if discard.all():
            self.offset = 0
        else:
            self.offset = np.nan_to_num(np.mean(self.shifts[~discard], axis=0))
        self.positions[discard] = predictions[discard] + self.offset


# ── the writer ────────────────────────────────────────────────────────────────
def pastefunc_blend(target, img):
    """``utils.py:197-216``: linear blend by the distance to unfilled space in ``target``;
    the target is kept wherever ``img`` is zero."""
    import scipy.ndimage
    import skimage.morphology

    hole_threshold = np.mean(target.shape)
    mask = skimage.morphology.remove_small_holes(target != 0, hole_threshold)
    dist = scipy.ndimage.distance_transform_cdt(mask)
    dmax = dist.max()
    if dmax == 0:
        alpha = 0
    else:
        alpha = dist / dmax
        alpha[skimage.morphology.binary_dilation(img == 0)] = 1
    return target * alpha + img * (1 - alpha)


def paste(target, img, pos):
    """``utils.py:149-194`` for 2-D integer images: tile pixel ``l`` lands on mosaic pixel
    ``pos + l``."""
    import scipy.ndimage

    pos = np.array(pos, dtype=float)
    if np.any(pos >= target.shape[:2]) or np.any(pos + img.shape[:2] < 0):
        return
    pos_f, pos_i = np.modf(pos)
    yi, xi = pos_i.astype("i8")
    if yi < 0:
        img = img[-yi:]
        yi = 0
    if xi < 0:
        img = img[:, -xi:]
        xi = 0
    target_slice = target[yi : yi + img.shape[0], xi : xi + img.shape[1]]
    img = img[: target_slice.shape[0], : target_slice.shape[1]]
    if pos_f.any():
        img = scipy.ndimage.shift(img, pos_f)
        y1 = None if pos_f[0] <= 0 else 1
        y2 = None if pos_f[0] >= 0 else -1
        x1 = None if pos_f[1] <= 0 else 1
        x2 = None if pos_f[1] >= 0 else -1
        img = img[y1:y2, x1:x2]
        target_slice = target_slice[y1:y2, x1:x2]
        if not np.all(img.shape):
            return
    target_slice[:] = pastefunc_blend(target_slice, img)


class Mosaic:
    def __init__(self, aligner, shape, **kwargs):
        record("Mosaic", **kwargs)
        self.aligner, self.shape = aligner, tuple(shape)
        self.channels = range(aligner.metadata.num_channels)
        self.dtype = aligner.metadata.pixel_dtype

    def assemble_channel(self, channel):
        out = np.zeros(self.shape, self.dtype)
        for si, position in enumerate(self.aligner.positions):
            paste(out, self.aligner.reader.read(c=channel, series=si), position)
        return out


class PyramidWriter:
    def __init__(self, mosaics, path, **kwargs):
        record("PyramidWriter", path=str(path), **kwargs)
        self.mosaics, self.path = mosaics, path

    def run(self):
        import tifffile

        record("PyramidWriter.run", path=str(self.path))
        planes = [m.assemble_channel(c) for m in self.mosaics for c in m.channels]
        px = self.mosaics[0].aligner.metadata.pixel_size
        with tifffile.TiffWriter(self.path, ome=True, bigtiff=True) as tiff:
            tiff.write(
                np.stack(planes),
                tile=(256, 256),
                photometric="minisblack",
                compression="adobe_deflate",
                metadata={
                    "Creator": f"Ashlar v{CONFIG['version']} (fake)",
                    "Pixels": {
                        "PhysicalSizeX": px,
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeY": px,
                        "PhysicalSizeYUnit": "µm",
                    },
                },
            )


# ── the command ───────────────────────────────────────────────────────────────
class ProcessingError(RuntimeError):
    pass


def main(argv=None):
    """``scripts/ashlar.py:15-298``: the flags the adapter may pass, the reference cycle
    stitched first, every later cycle aligned, THEN the writer."""
    argv = list(sys.argv if argv is None else argv)
    record("main", argv=argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="+")
    parser.add_argument("-o", "--output", default="ashlar_output.ome.tif")
    parser.add_argument("-c", "--align-channel", type=int, default=0)
    parser.add_argument("-m", "--maximum-shift", type=float, default=15)
    parser.add_argument("--filter-sigma", type=float, default=0)
    args = parser.parse_args(argv[1:])
    reg = sys.modules["ashlar.reg"]
    if not Path(args.output).parent.is_dir():
        print("ERROR: Output location does not exist or is not a directory")
        return 1
    if CONFIG["exit_code"]:
        print("ERROR: the fake was told to fail")
        return int(CONFIG["exit_code"])
    aligner_args = {
        "channel": args.align_channel,
        "verbose": True,
        "max_shift": args.maximum_shift,
        "filter_sigma": args.filter_sigma,
    }
    edge = reg.EdgeAligner(
        build_reader(args.filepaths[0]), alpha=0.01, max_error=None, **aligner_args
    )
    edge.run()
    mosaics = [reg.Mosaic(edge, edge.mosaic_shape, verbose=True)]
    for spec in args.filepaths[1:]:
        layer = reg.LayerAligner(build_reader(spec), edge, **aligner_args)
        layer.run()
        mosaics.append(reg.Mosaic(layer, edge.mosaic_shape, verbose=True))
    reg.PyramidWriter(mosaics, args.output, verbose=True, tile_size=1024).run()
    return 0


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def modules():
    reg = _module(
        "ashlar.reg",
        Metadata=Metadata,
        Reader=Reader,
        CachingReader=CachingReader,
        EdgeAligner=EdgeAligner,
        LayerAligner=LayerAligner,
        Mosaic=Mosaic,
        BioformatsReader=BioformatsReader,
        PyramidWriter=PyramidWriter,
        Warning=Warning,
        DataWarning=DataWarning,
    )
    cli = _module("ashlar.scripts.ashlar", main=main, ProcessingError=ProcessingError)
    scripts = _module("ashlar.scripts", ashlar=cli)
    pkg = _module("ashlar", __version__=CONFIG["version"], reg=reg, scripts=scripts)
    pkg.__file__ = __file__
    return {
        "ashlar": pkg,
        "ashlar.reg": reg,
        "ashlar.scripts": scripts,
        "ashlar.scripts.ashlar": cli,
    }


def configure(**config):
    """Reset the fake to its defaults, then apply ``config``; forget recorded calls."""
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"fake_ashlar has no setting {sorted(unknown)}")
    CONFIG.update(DEFAULTS)
    CONFIG.update(config)
    CALLS.clear()
    _STATE["cycle"] = 0


def install(setitem=None, **config):
    """Configure the fake and put it in ``sys.modules`` (through ``monkeypatch.setitem``
    when given, so a test leaves nothing behind)."""
    configure(**config)
    put = setitem or (lambda mapping, key, value: mapping.__setitem__(key, value))
    for name, mod in modules().items():
        put(sys.modules, name, mod)
