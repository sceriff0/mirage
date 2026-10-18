"""Benchmark-side one-channel proxies of an RGB slide, for STARE.

Restored from commit 5a46f778 (``benchmarks/regbench/prep.py``) and reduced to its proxy
writers. What was removed: the closed candidate list of the ``stare_prep`` row, its
selection on a tuning set and the frozen file of that selection, and the CLAHE operation
(STARE 1.3.0 has CLAHE of its own: the ``sprep`` levels, ``regimes.SPREPS``). What is kept
is what turns an RGB slide into the ONE channel STARE reads, the frame check that flags
a proxy which moves image content (it refused one until B10), and the readers that never
hold a whole slide.

STARE reads one channel per slide, so a brightfield (RGB) case is registered on a
one-channel proxy of each slide (``methods.stare``, ``regimes.PROXIES``). The case's own
proxy (``case.ref_nuclear`` / ``case.mov_nuclear``, written by its dataset module) is
``grey``; the others are written here (``write_inputs``) before STARE's first stage, and
the run lists that as benchmark-side work (``info.implementation.benchmark_side``).

**The proxies** (``PROXIES``: a *base*, how an RGB image becomes one channel, and an
*operation* on that one channel).

``grey``         inverted luminance and nothing else: what a dataset module writes as the
                 case's own proxy (the baseline).
``haem``         the haematoxylin channel of a colour deconvolution.
``grey_smooth``  a Gaussian (sigma 2 px) on ``grey``.

No proxy resamples, crops, pads or shifts: every output has the input's shape and pixel
grid (``frame_check`` measures it on dots; when it fails ``write_inputs`` writes the
proxy all the same and records ``frame_check: failed``, a ``frame_check_failed`` row of
losses.csv: B10, user ruling "no blocking").

**Definitions** (as written at 5a46f778; the tags are that text's, not re-read here).

``grey``    ``255 - (0.299 R + 0.587 G + 0.114 B)``, rounded, uint8 (``GREY_LUMA``: the
            ITU-R BT.601 luma weights, the ones ``PIL.Image.convert("L")`` documents
            ``[MEMORY]``). Tissue bright on a dark background, as a fluorescence channel.
``haem``    channel 0 of ``skimage.color.rgb2hed`` (Ruifrok and Johnston's stain matrix as
            shipped by scikit-image; the function's docstring read in the installed
            0.24.0 ``[PARTIAL: the docstrings of rgb2hed and separate_stains]``), clipped
            to ``[0, 0.255]`` and stored as uint8 at 1000 counts per unit. The range is a
            storage range: the 99.9th percentile of that channel was at most 0.19 on the
            seven images it was looked at on (one image each of seven sets, some on the
            report side; nothing was registered to choose it).
smooth      ``scipy.ndimage.gaussian_filter(sigma=2)`` in float32, rounded back to the
            input's type.

The image is processed in blocks (the written tile, 1024 px) with a 512 px halo and the
image mirrored beyond its edges, so the result does not depend on the block size (a test
compares two block sizes).

**Memory.** Nothing is ever read or held whole but a JPEG / PNG (ANHIR's images, which
PIL decodes whole): a whole-slide RGB TIFF (HyReCo, an RGB OME-TIFF) is read in
rectangles, each block is computed with its halo and written as one tile, the bases work
a few rows at a time, and the writer encodes on one thread (tifffile's default pool queues
tiles in proportion to the slide). A test measures the peak at two slide sizes 8x apart:
it does not grow.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

PROXIES = {
    "grey": {"base": "grey", "op": None},
    "haem": {"base": "haem", "op": None},
    "grey_smooth": {"base": "grey", "op": "smooth"},
}
GREY_LUMA = (0.299, 0.587, 0.114)
HAEM_MAX, HAEM_SCALE = 0.255, 1000.0
SMOOTH_SIGMA = 2.0
BLOCK, HALO, TILE = 2048, 512, 1024
FRAME_GATE_PX = 0.02  # the mean displacement of the dots through a proxy
# ... and the largest displacement of any ONE dot: a rotation or a scaling about the
# image centre moves every dot and leaves the mean at zero. The centroid of a single
# dot after 8-bit rounding is good to a few hundredths of a pixel.
FRAME_GATE_MAX_PX = 0.15


class Refused(ValueError):
    """An input this module will not write a proxy of, or a proxy that is not what it
    must be (off the slide's frame or shape), with the reason. Writing the proxy is the
    benchmark's own step, so its cause is the benchmark's (``resources.classify_error``
    reads ``regbench_cause``), never the method's."""

    regbench_cause = "benchmark"


def proxy(name):
    if name not in PROXIES:
        raise ValueError(f"proxy={name!r}: expected one of {', '.join(PROXIES)}")
    return PROXIES[name]


# ── the operations ────────────────────────────────────────────────────────────
# the bases work in floats: a few rows at a time bound their temporaries (per pixel:
# the rows change nothing)
BASE_ROWS = 64


def grey_plane(rgb):
    """Inverted luminance of ``(h, w, 3)`` uint8 as uint8: the case's own proxy."""
    rgb = np.asarray(rgb)
    luma = np.asarray(GREY_LUMA, np.float32)
    out = np.empty(rgb.shape[:2], np.uint8)
    for y0 in range(0, rgb.shape[0], BASE_ROWS):
        lum = np.asarray(rgb[y0 : y0 + BASE_ROWS], np.float32)[..., :3] @ luma
        out[y0 : y0 + BASE_ROWS] = (255.0 - lum).round().clip(0, 255)
    return out


def haem_plane(rgb):
    from skimage.color import rgb2hed

    rgb = np.asarray(rgb)[..., :3]
    out = np.empty(rgb.shape[:2], np.uint8)
    for y0 in range(0, rgb.shape[0], BASE_ROWS):  # rgb2hed works in float64
        hed = rgb2hed(rgb[y0 : y0 + BASE_ROWS])[..., 0]
        out[y0 : y0 + BASE_ROWS] = (np.clip(hed, 0.0, HAEM_MAX) * HAEM_SCALE).round()
    return out


BASES = {"grey": grey_plane, "haem": haem_plane}


class Reader:
    """One RGB image, read in rectangles: ``read(y0, y1, x0, x1)`` gives ``(h, w, 3)``."""

    def __init__(self, hw, read, close=None):
        self.hw = tuple(int(v) for v in hw)
        self.read = read
        self.close = close or (lambda: None)


def array_reader(a):
    a = np.asarray(a)
    if a.ndim != 3 or a.shape[2] < 3:
        raise Refused(f"expected an (H, W, 3) RGB image, found the shape {a.shape}")
    return Reader(a.shape[:2], lambda y0, y1, x0, x1: a[y0:y1, x0:x1])


def _smooth(block):
    from scipy.ndimage import gaussian_filter

    out = gaussian_filter(block.astype(np.float32), SMOOTH_SIGMA, mode="reflect")
    info = np.iinfo(block.dtype)  # both bases are uint8
    np.round(out, out=out)
    np.clip(out, info.min, info.max, out=out)
    return out.astype(block.dtype)


def blocks(reader, name, block=BLOCK):
    """The proxy as ``(y0, x0, core)`` in row-major order of ``block`` x ``block`` cores
    (smaller at the right and bottom edges). One core and its halo are in memory at a
    time: with ``block`` equal to the written tile, nothing larger than a tile's working
    set ever is (the memory contract), whatever the slide's size. The result does not
    depend on ``block``."""
    c = proxy(name)
    base, op = BASES[c["base"]], c["op"]
    h, w = reader.hw
    full = block + 2 * HALO
    for y0 in range(0, h, block):
        y1 = min(y0 + block, h)
        for x0 in range(0, w, block):
            x1 = min(x0 + block, w)
            if op is None:
                yield y0, x0, np.asarray(base(reader.read(y0, y1, x0, x1)), np.uint8)
                continue
            ry0, rx0 = max(y0 - HALO, 0), max(x0 - HALO, 0)
            ry1, rx1 = min(y0 + block + HALO, h), min(x0 + block + HALO, w)
            part = base(reader.read(ry0, ry1, rx0, rx1))
            before = (ry0 - (y0 - HALO), rx0 - (x0 - HALO))
            pad = [
                (before[0], full - before[0] - part.shape[0]),
                (before[1], full - before[1] - part.shape[1]),
            ]
            # mirrored beyond the image; numpy mirrors again where a pad is larger
            # than the image, so a small image is as well defined as a large one
            part = np.pad(part, pad, mode="reflect") if any(map(any, pad)) else part
            done = _smooth(part)
            core = done[HALO : HALO + y1 - y0, HALO : HALO + x1 - x0]
            yield y0, x0, np.ascontiguousarray(core, np.uint8)


def to_array(reader, name, block=BLOCK):
    """The whole proxy (tests and small images)."""
    out = np.empty(reader.hw, np.uint8)
    for y0, x0, core in blocks(reader, name, block):
        out[y0 : y0 + core.shape[0], x0 : x0 + core.shape[1]] = core
    return out


def write(reader, name, out, pixel_size_um=None):
    """One proxy as a tiled, compressed, one-channel OME-TIFF. Returns its ``(H, W)``,
    checked equal to the input's."""
    from . import imageio

    h, w = reader.hw

    def tiles():  # one core per written tile, in the writer's row-major order
        for _, _, core in blocks(reader, name, TILE):
            yield core

    imageio.write_nuclear_tiles(out, (h, w), np.uint8, tiles(), TILE, pixel_size_um)
    got = imageio.shape_of(out)
    if tuple(got[1:]) != (h, w):
        raise Refused(
            f"{out}: the proxy is {got[1:]}, its input {(h, w)}: the frame of the "
            "transform would not be the slide's"
        )
    return h, w


# ── the frame check ───────────────────────────────────────────────────────────
def _dots(hw, seed, n=120, margin=24):
    rng = np.random.default_rng(seed)
    h, w = hw
    return np.column_stack(
        [rng.uniform(margin, w - margin, n), rng.uniform(margin, h - margin, n)]
    )


def _render(hw, xy, sigma=2.5):
    h, w = hw
    out = np.zeros(hw, np.float64)
    r = int(4 * sigma) + 1
    for x, y in xy:
        ix, iy = int(round(x)), int(round(y))
        ys, xs = np.mgrid[iy - r : iy + r + 1, ix - r : ix + r + 1]
        keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
        np.maximum.at(out, (ys[keep], xs[keep]), g[keep])
    return out


def _centroids(image, xy, r=9):
    a = np.asarray(image, np.float64)
    a = a - np.median(a)
    out = np.full((len(xy), 2), np.nan)
    for i, (x, y) in enumerate(xy):
        ix, iy = int(round(x)), int(round(y))
        win = np.clip(a[iy - r : iy + r + 1, ix - r : ix + r + 1], 0, None)
        if win.shape != (2 * r + 1, 2 * r + 1) or win.sum() <= 0:
            continue
        ys, xs = np.mgrid[iy - r : iy + r + 1, ix - r : ix + r + 1]
        out[i] = (xs * win).sum() / win.sum(), (ys * win).sum() / win.sum()
    return out


def frame_check(name, hw=(700, 900), seed=0, process=None):
    """How far a proxy moves image content: the mean displacement ``(dx, dy)`` px of
    isolated dark dots drawn at known positions on a white RGB slide, through the SAME
    code that writes a slide's proxy. ``{"dx", "dy", "max", "n", "ok"}``; ``ok`` when
    both means are under ``FRAME_GATE_PX``, no single dot moved by ``FRAME_GATE_MAX_PX``
    or more, every dot was found and the output has the input's shape. ``process``
    replaces ``to_array`` (tests)."""
    xy = _dots(hw, seed)
    # dots further apart than the centroid window, so each window holds one dot
    keep = [0]
    for i in range(1, len(xy)):
        if np.min(np.hypot(*(xy[keep] - xy[i]).T)) > 30:
            keep.append(i)
    xy = xy[keep]
    g = _render(hw, xy)
    # dark bluish dots on a white slide
    dot, paper = np.array([70.0, 60.0, 140.0]), np.array([240.0, 240.0, 240.0])
    image = (paper + g[..., None] * (dot - paper)).round().astype(np.uint8)
    reader = array_reader(image)
    out = (process or (lambda r, n: to_array(r, n, TILE)))(reader, name)
    nan = {"dx": math.nan, "dy": math.nan, "max": math.nan}
    if out.shape != tuple(hw):
        return {**nan, "n": 0, "ok": False, "shape": out.shape}
    found = _centroids(out, xy)
    good = np.isfinite(found).all(axis=1)
    if good.sum() < len(xy) // 2:
        return {**nan, "n": int(good.sum()), "ok": False}
    moved = found[good] - xy[good]
    dx, dy = (float(v) for v in moved.mean(axis=0))
    worst = float(np.hypot(*moved.T).max())
    ok = (
        abs(dx) < FRAME_GATE_PX
        and abs(dy) < FRAME_GATE_PX
        and worst < FRAME_GATE_MAX_PX
        and good.sum() == len(xy)  # a dot that is gone was moved out of its window
    )
    return {"dx": dx, "dy": dy, "max": worst, "n": int(good.sum()), "ok": bool(ok)}


def _peak_rss_mb():
    """The process's peak resident memory so far, MB (``ru_maxrss``: bytes on macOS,
    kilobytes on Linux)."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1024.0**2 if sys.platform == "darwin" else 1024.0)


# ── a case ────────────────────────────────────────────────────────────────────
RECORD = "record.json"


def code_sha256():
    """sha256 of this module's source: the code that wrote a proxy."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def input_paths(dest, name):
    """``(reference, moving)``: where ``write_inputs`` puts the two proxies."""
    dest = Path(dest)
    return dest / f"reference.{name}.ome.tif", dest / f"moving.{name}.ome.tif"


def identity(case, name):
    """What the two proxies of a case are made FROM: the proxy, this module's code and
    the bytes of the two slides (``cases.file_fingerprint``). Equal identity, equal
    files."""
    from .cases import file_fingerprint

    return {
        "proxy": name,
        "code": code_sha256(),
        "inputs": [file_fingerprint(case.ref_image), file_fingerprint(case.mov_image)],
    }


def current(case, name, dest):
    """The record of the proxies under ``dest`` when both files are there and were
    written from what ``identity`` names now, else None."""
    paths = input_paths(dest, name)
    try:
        rec = json.loads((Path(dest) / RECORD).read_text())
    except (OSError, ValueError):
        return None
    if not all(p.exists() for p in paths) or rec.get("identity") != identity(
        case, name
    ):
        return None
    return rec


def _reader(src):
    """One RGB image as a ``Reader`` (``imageio.open_rgb``: a TIFF in rectangles, never
    read whole; a JPEG / PNG decoded whole). Refused: not 8-bit RGB, or no lazy reader
    (zarr) installed, which would read the slide whole."""
    from .imageio import open_rgb

    tiff = str(src).lower().endswith((".tif", ".tiff"))
    try:
        read, hw, close = open_rgb(src, lazy_only=tiff)
    except ImportError as exc:
        raise Refused(
            f"{src}: no lazy TIFF reader ({exc}); reading the slide whole is refused"
        ) from None
    except (ValueError, IndexError) as exc:
        if isinstance(exc, Refused):
            raise
        raise Refused(f"{src}: {exc}") from None
    return Reader(hw, read, close)


def write_inputs(case, name, dest):
    """Both proxies ``name`` of a brightfield case under ``dest``; returns ``(reference,
    moving)``, the two files STARE is given (channel 0 of each). Read from the case's
    RGB slides (``ref_image`` / ``mov_image``): a whole-slide TIFF by tiles
    (``imageio.open_rgb``), a JPEG or PNG whole. What was done is ``dest/record.json``.
    Every refusal is ``Refused``: the benchmark's cause."""
    proxy(name)
    if case.modality != "brightfield":
        raise Refused(
            f"{case.case_id}: a proxy is written of an RGB slide; this case is "
            f"{case.modality}, whose proxy is the channel itself"
        )
    check = frame_check(name)
    if not check["ok"]:
        # B10 (user ruling: no blocking): written, flagged in the record, and a
        # `frame_check_failed` row of losses.csv
        print(
            f"prep: WARNING proxy={name}: the proxy moves image content ({check}): the "
            "transform STARE estimates may not be in the slide's frame; written and "
            "flagged (record.json frame_check=failed)"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / RECORD).unlink(missing_ok=True)  # no record of the files about to change
    paths, t0, shapes = input_paths(dest, name), time.perf_counter(), {}
    for side, path in zip(("ref", "mov"), paths):
        reader = _reader(getattr(case, f"{side}_image"))
        try:
            shapes[side] = write(reader, name, path, case.pixel_size_um)
        finally:
            reader.close()
        want = getattr(case, f"{side}_hw")
        if want and tuple(int(v) for v in want) != tuple(shapes[side]):
            raise Refused(
                f"{case.case_id}: the {side} image is {shapes[side]}, the case says "
                f"{tuple(want)}"
            )
    c = PROXIES[name]
    record = {
        "identity": identity(case, name),
        "proxy": name,
        "base": c["base"],
        "operation": c["op"] or "none",
        "frame_check": "ok" if check["ok"] else "failed",
        "frame_check_px": {k: check[k] for k in ("dx", "dy", "max", "n")},
        "prep_s": time.perf_counter() - t0,
        "peak_rss_mb": _peak_rss_mb(),
        "peak_rss_is": "the writing process's peak when the proxies were written (the "
        "registration's own peak is measured apart)",
        "hw": {k: list(v) for k, v in shapes.items()},
        "reference": str(paths[0]),
        "moving": str(paths[1]),
    }
    tmp = dest / (RECORD + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(dest / RECORD)
    return str(paths[0]), str(paths[1])
