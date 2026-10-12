"""Benchmark-side preprocessing of the one image STARE registers on (the ``stare_prep`` row).

STARE itself is not changed: the package, its version and its parameters are those of the
``stare`` row. The ``stare_prep`` row differs in ONE thing, done by the benchmark before
STARE sees the pair: both images are rewritten as one-channel files by one of the
candidates below, and STARE registers those. The transform it estimates is then applied
to the ORIGINAL moving slide (never to the preprocessed pixels), so the written image and
every score are defined exactly as for ``stare``.

**The candidates** (``CANDIDATES``, declared before any run; the list is closed). A
candidate is a *base* (how an RGB image becomes one channel) followed by an *operation* on
that one channel. A fluorescence case has no RGB image: there the base is the nuclear
channel itself, untouched, and only the operation applies.

``grey``         brightfield: inverted luminance and nothing else, exactly what the plain
                 ``stare`` row is given. One channel: IDENTITY (the channel as it is,
                 the same input as the ``stare`` row).
``haem``         brightfield: the haematoxylin channel of a colour deconvolution. One
                 channel: IDENTITY (there is no colour to separate).
``grey_clahe``   brightfield: CLAHE on ``grey``. One channel: CLAHE on the channel.
``haem_clahe``   brightfield: CLAHE on ``haem``. One channel: CLAHE on the channel (the
                 same as ``grey_clahe`` there).
``grey_smooth``  brightfield: a Gaussian (sigma 2 px) on ``grey``. One channel: the same
                 Gaussian on the channel.

Why these five. ``grey`` is the baseline, so the selection can end on "no preprocessing
helps". ``haem`` is the one signal two differently stained sections are expected to share
(the counterstain); whether every stain of the tuning sets has one is not established by
any source read, which is what the tuning comparison is for. CLAHE evens out local
contrast between a pale and a dark section, on either base. ``grey_smooth`` removes detail
below about two pixels, which consecutive sections do not share. No candidate resamples,
crops, pads or shifts: every output has the input's shape and pixel grid (``frame_check``
measures it on dots; the adapter refuses a run when it fails).

**Definitions.**

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
CLAHE       OpenCV's ``createCLAHE(clipLimit=2.0)`` with square contextual regions of
            256 px, on an 8-bit image (OpenCV 4.10.0, called, its source not read
            ``[MEMORY]`` for the algorithm). A 16-bit channel is first mapped linearly to
            0..255 between 0 and its own 99.9th percentile (every 16th pixel each way);
            an 8-bit channel is used as it is. The output is uint8.
smooth      ``scipy.ndimage.gaussian_filter(sigma=2)`` in float32, rounded back to the
            input's type.

The image is processed in blocks of 2048 px with a 512 px halo, on a grid of contextual
regions fixed to the image origin and with the image mirrored beyond its edges, so the
result does not depend on the block size (a test compares two block sizes).

**The frozen choice** (``prep_frozen.json``, tracked, hashed). ``selected`` is the
candidate the ``stare_prep`` row runs with, on every dataset. It is chosen on ANHIR's
tuning side only (the ``anhir_tune`` dataset: ``datasets/anhir.py``) by ``CRITERION`` and
written by ``python -m benchmarks.regbench prep-select --freeze`` together with the table
it was chosen from. Until that has run the file says ``"status": "provisional"``: the
value in it was chosen by nobody's evidence, and every table that shows ``stare_prep``
says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

CANDIDATES = {
    "grey": {"base": "grey", "op": None},
    "haem": {"base": "haem", "op": None},
    "grey_clahe": {"base": "grey", "op": "clahe"},
    "haem_clahe": {"base": "haem", "op": "clahe"},
    "grey_smooth": {"base": "grey", "op": "smooth"},
}
# what each candidate is on a one-channel (fluorescence) image
SINGLE_CHANNEL = {
    "grey": "identity (the channel as it is: the same input as the `stare` row)",
    "haem": "identity (no colour to separate: the same input as the `stare` row)",
    "grey_clahe": "CLAHE on the channel",
    "haem_clahe": "CLAHE on the channel (the same as grey_clahe)",
    "grey_smooth": "Gaussian, sigma 2 px, on the channel",
}
ROW = "stare_prep"  # the run and its final result name
GREY_LUMA = (0.299, 0.587, 0.114)
HAEM_MAX, HAEM_SCALE = 0.255, 1000.0
CLAHE_CLIP, CLAHE_REGION = 2.0, 256
CLAHE_WINDOW_PERCENTILE, CLAHE_WINDOW_STEP = 99.9, 16
SMOOTH_SIGMA = 2.0
BLOCK, HALO, TILE = 2048, 512, 1024
FRAME_GATE_PX = 0.02  # the mean displacement of the dots through a candidate
# ... and the largest displacement of any ONE dot: a rotation or a scaling about the
# image centre moves every dot and leaves the mean at zero. The centroid of a single
# dot after CLAHE or 8-bit rounding is good to a few hundredths of a pixel.
FRAME_GATE_MAX_PX = 0.15

FROZEN_PATH = Path(__file__).with_name("prep_frozen.json")
STATUSES = ("provisional", "selected")
CRITERION = (
    "on the tuning-side pairs only (dataset anhir_tune): the lowest median over pairs of "
    "the per-pair median rTRE, a pair that did not register counting at its unregistered "
    "landmarks (the official rule); ties by the higher mean robustness, then by the "
    "order of the candidate list"
)
PROVISIONAL_NOTE = (
    "PROVISIONAL: the selection has not been run. This candidate was put here without "
    "evidence, only because it is not the identity on either modality, so the row can "
    "be told from `stare` everywhere. Run the candidates on anhir_tune and "
    "`python -m benchmarks.regbench prep-select --freeze` before any number of the "
    "stare_prep row is used."
)


class Refused(ValueError):
    """An input this module will not preprocess, with the reason."""


def candidate(name):
    if name not in CANDIDATES:
        raise ValueError(f"prep={name!r}: expected one of {', '.join(CANDIDATES)}")
    return CANDIDATES[name]


def is_identity(name, brightfield):
    """True when a candidate leaves the input of the plain row unchanged."""
    c = candidate(name)
    return c["op"] is None and (not brightfield or c["base"] == "grey")


# ── the frozen choice ─────────────────────────────────────────────────────────
def record_hash(record):
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_frozen(path=None):
    """The frozen record, checked: its hash, its candidate list against this module's,
    its status. SystemExit on any mismatch (an edited file is not a frozen one)."""
    path = Path(path or FROZEN_PATH)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: the frozen preprocessing is unreadable ({exc})")
    if record.get("sha256") != record_hash(record):
        raise SystemExit(
            f"{path}: its content does not match its sha256. It is written by "
            "`python -m benchmarks.regbench prep-select --freeze`, never by hand."
        )
    if list(record.get("candidates") or []) != list(CANDIDATES):
        raise SystemExit(
            f"{path}: it lists the candidates {record.get('candidates')}, this code "
            f"has {list(CANDIDATES)}: a selection made over another list is void"
        )
    if record.get("selected") not in CANDIDATES or record.get("status") not in STATUSES:
        raise SystemExit(f"{path}: no valid `selected` / `status`")
    return record


def selected(path=None):
    """The candidate the ``stare_prep`` row runs with."""
    return load_frozen(path)["selected"]


def frozen_summary(path=None):
    """One line for a table's header."""
    r = load_frozen(path)
    if r["status"] == "provisional":
        return (
            f"`{ROW}` preprocessing: `{r['selected']}`, **PROVISIONAL: the selection "
            "on ANHIR's tuning side has not been run; this candidate was chosen by no "
            f"evidence** (prep_frozen.json {r['sha256'][:12]})"
        )
    return (
        f"`{ROW}` preprocessing: `{r['selected']}`, selected on ANHIR's tuning side "
        f"({r.get('selected_on', '?')}; prep_frozen.json {r['sha256'][:12]})"
    )


def make_record(selected_name, status, evidence=(), **more):
    candidate(selected_name)
    if status not in STATUSES:
        raise ValueError(f"status {status!r}: expected one of {STATUSES}")
    record = {
        "version": 1,
        "candidates": list(CANDIDATES),
        "single_channel": dict(SINGLE_CHANNEL),
        "criterion": CRITERION,
        "selected": selected_name,
        "status": status,
        "note": PROVISIONAL_NOTE if status == "provisional" else "",
        "evidence": [dict(r) for r in evidence],
        **more,
    }
    record["sha256"] = record_hash(record)
    return record


def write_frozen(record, path=None):
    path = Path(path or FROZEN_PATH)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def rank(rows):
    """The evidence rows in the criterion's order, the winner first. A row is ``{"candidate",
    "median_of_median_rtre", "mean_robustness", ...}``."""
    order = list(CANDIDATES)
    for r in rows:
        candidate(r["candidate"])
        if not math.isfinite(float(r["median_of_median_rtre"])):
            raise ValueError(f"{r['candidate']}: no finite median rTRE to rank by")
    return sorted(
        rows,
        key=lambda r: (
            float(r["median_of_median_rtre"]),
            -float(r["mean_robustness"]),
            order.index(r["candidate"]),
        ),
    )


# ── the operations ────────────────────────────────────────────────────────────
def grey_plane(rgb):
    """Inverted luminance of ``(h, w, 3)`` uint8 as uint8: the plain row's input."""
    lum = np.asarray(rgb, np.float32)[..., :3] @ np.asarray(GREY_LUMA, np.float32)
    return (255.0 - lum).round().clip(0, 255).astype(np.uint8)


def haem_plane(rgb):
    from skimage.color import rgb2hed

    hed = rgb2hed(np.asarray(rgb)[..., :3])[..., 0]
    return (np.clip(hed, 0.0, HAEM_MAX) * HAEM_SCALE).round().astype(np.uint8)


BASES = {"grey": grey_plane, "haem": haem_plane}


class Reader:
    """One image, read in rectangles: ``read(y0, y1, x0, x1)`` gives ``(h, w)`` (one
    channel) or ``(h, w, 3)`` (RGB)."""

    def __init__(self, hw, dtype, read, rgb, close=None):
        self.hw = tuple(int(v) for v in hw)
        self.dtype = np.dtype(dtype)
        self.read, self.rgb = read, bool(rgb)
        self.close = close or (lambda: None)


def array_reader(a):
    a = np.asarray(a)
    rgb = a.ndim == 3
    return Reader(a.shape[:2], a.dtype, lambda y0, y1, x0, x1: a[y0:y1, x0:x1], rgb)


def read_rgb(path):
    """A JPEG / PNG (ANHIR) whole, as ``(H, W, 3)`` uint8."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # these are whole sections, not an attack
    with Image.open(str(path)) as im:
        return np.asarray(im.convert("RGB"))


def channel_reader(path, channel):
    from stare.slide_io import open_lazy

    arr, dtype, close = open_lazy(str(path))
    c = int(channel)
    if not 0 <= c < int(arr.shape[0]):
        close()
        raise Refused(f"{path}: channel {c} out of range for C={arr.shape[0]}")
    return Reader(
        arr.shape[1:],
        dtype,
        lambda y0, y1, x0, x1: np.asarray(arr[c, y0:y1, x0:x1]),
        False,
        close,
    )


def _window_top(reader):
    """The value mapped to 255 before CLAHE: 255 for an 8-bit base, else the 99.9th
    percentile of every 16th pixel of the whole image (at least 1)."""
    if reader.rgb or reader.dtype == np.uint8:  # both bases of an RGB image are 8-bit
        return 255.0
    h, w = reader.hw
    s, parts = CLAHE_WINDOW_STEP, []
    for y0 in range(0, h, BLOCK):
        band = reader.read(y0, min(y0 + BLOCK, h), 0, w)
        first = (-y0) % s  # the rows y with y % s == 0, whatever the band
        parts.append(np.asarray(band)[first::s, ::s].ravel())
    values = np.concatenate(parts) if parts else np.zeros(1)
    return max(float(np.percentile(values, CLAHE_WINDOW_PERCENTILE)), 1.0)


def _to8(block, top):
    if block.dtype == np.uint8 and top == 255.0:
        return block
    return (
        (block.astype(np.float32) * (255.0 / top)).round().clip(0, 255).astype(np.uint8)
    )


def _clahe(block8):
    import cv2

    grid = (block8.shape[1] // CLAHE_REGION, block8.shape[0] // CLAHE_REGION)
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=grid).apply(
        np.ascontiguousarray(block8)
    )


def _smooth(block):
    from scipy.ndimage import gaussian_filter

    out = gaussian_filter(block.astype(np.float32), SMOOTH_SIGMA, mode="reflect")
    if block.dtype.kind in "ui":
        info = np.iinfo(block.dtype)
        return out.round().clip(info.min, info.max).astype(block.dtype)
    return out.astype(block.dtype)


def out_dtype(name, reader):
    c = candidate(name)
    if c["op"] == "clahe" or reader.rgb:
        return np.dtype(np.uint8)
    return reader.dtype


def bands(reader, name, block=BLOCK):
    """The preprocessed image as ``(y0, band)`` in order, ``band`` of shape ``(rows, W)``.

    ``block`` is the core size (a multiple of the contextual region and of the halo); the
    result does not depend on it."""
    c = candidate(name)
    if block % CLAHE_REGION or HALO % CLAHE_REGION:
        raise ValueError("the block and the halo are multiples of the CLAHE region")
    base = BASES[c["base"]] if reader.rgb else (lambda a: np.asarray(a))
    op = c["op"]
    h, w = reader.hw
    top = _window_top(reader) if op == "clahe" else None
    full = block + 2 * HALO
    for y0 in range(0, h, block):
        y1 = min(y0 + block, h)
        band = np.empty((y1 - y0, w), out_dtype(name, reader))
        for x0 in range(0, w, block):
            x1 = min(x0 + block, w)
            if op is None:
                band[:, x0:x1] = base(reader.read(y0, y1, x0, x1))
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
            done = _clahe(_to8(part, top)) if op == "clahe" else _smooth(part)
            band[:, x0:x1] = done[HALO : HALO + y1 - y0, HALO : HALO + x1 - x0]
        yield y0, band


def to_array(reader, name, block=BLOCK):
    """The whole preprocessed image (tests and small images)."""
    out = np.empty(reader.hw, out_dtype(name, reader))
    for y0, band in bands(reader, name, block):
        out[y0 : y0 + band.shape[0]] = band
    return out


def write(reader, name, out, pixel_size_um=None):
    """One preprocessed image as a tiled, compressed, one-channel OME-TIFF. Returns its
    ``(H, W)``, checked equal to the input's."""
    from . import imageio

    h, w = reader.hw
    dtype = out_dtype(name, reader)

    def tiles():
        for _, band in bands(reader, name):
            for ty in range(0, band.shape[0], TILE):
                for x0 in range(0, w, TILE):
                    yield band[ty : ty + TILE, x0 : x0 + TILE]

    imageio.write_nuclear_tiles(out, (h, w), dtype, tiles(), TILE, pixel_size_um)
    got = imageio.shape_of(out)
    if tuple(got[1:]) != (h, w):
        raise RuntimeError(
            f"{out}: the preprocessed image is {got[1:]}, its input {(h, w)}: the "
            "frame of the transform would not be the slide's"
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


def frame_check(name, brightfield, hw=(700, 900), seed=0, process=None):
    """How far a candidate moves image content: the mean displacement ``(dx, dy)`` px of
    isolated dots drawn at known positions, through the SAME code that preprocesses a
    slide. ``{"dx", "dy", "max", "n", "ok"}``; ``ok`` when both means are under
    ``FRAME_GATE_PX``, no single dot moved by ``FRAME_GATE_MAX_PX`` or more, every dot
    was found and the output has the input's shape. ``process`` replaces ``to_array`` (tests)."""
    xy = _dots(hw, seed)
    # dots further apart than the centroid window, so each window holds one dot
    keep = [0]
    for i in range(1, len(xy)):
        if np.min(np.hypot(*(xy[keep] - xy[i]).T)) > 30:
            keep.append(i)
    xy = xy[keep]
    g = _render(hw, xy)
    if brightfield:  # dark bluish dots on a white slide
        dot, paper = np.array([70.0, 60.0, 140.0]), np.array([240.0, 240.0, 240.0])
        image = (paper + g[..., None] * (dot - paper)).round().astype(np.uint8)
    else:
        image = (g * 20000).round().astype(np.uint16)
    reader = array_reader(image)
    out = (process or to_array)(reader, name)
    if out.shape != tuple(hw):
        return {
            "dx": math.nan,
            "dy": math.nan,
            "max": math.nan,
            "n": 0,
            "ok": False,
            "shape": out.shape,
        }
    found = _centroids(out, xy)
    good = np.isfinite(found).all(axis=1)
    if good.sum() < len(xy) // 2:
        return {
            "dx": math.nan,
            "dy": math.nan,
            "max": math.nan,
            "n": int(good.sum()),
            "ok": False,
        }
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
def code_sha256():
    """sha256 of this module's source: the preprocessing code a `prep` job ran."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def input_paths(dest, name):
    """``(reference, moving)``: where ``write_inputs`` puts the two files."""
    dest = Path(dest)
    return dest / f"reference.{name}.ome.tif", dest / f"moving.{name}.ome.tif"


def write_inputs(case, dest, name, channel=0):
    """Both images of a case, preprocessed, under ``dest``. Returns ``{"reference",
    "moving", "record"}``: the two files STARE is given (channel 0 of each) and what was
    done. A brightfield case is read from its RGB images (``ref_image`` /
    ``mov_image``, JPEG or PNG); a fluorescence case from channel ``channel`` of its
    original slides."""
    c = candidate(name)
    brightfield = case.modality == "brightfield"
    check = frame_check(name, brightfield)
    if not check["ok"]:
        raise RuntimeError(
            f"prep={name}: the preprocessing moves image content ({check}): the "
            "transform STARE estimates would not be in the slide's frame"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out, t0, shapes = {}, time.perf_counter(), {}
    for side in ("ref", "mov"):
        src = getattr(case, f"{side}_image")
        if brightfield:
            if str(src).lower().endswith((".tif", ".tiff")):
                raise Refused(
                    f"{case.case_id}: the preprocessing row reads a brightfield case's "
                    "JPEG / PNG image whole; a whole-slide RGB TIFF has no reader here"
                )
            reader = array_reader(read_rgb(src))
        else:
            reader = channel_reader(src, channel)
        path = input_paths(dest, name)[side == "mov"]
        try:
            shapes[side] = write(reader, name, path, case.pixel_size_um)
        finally:
            reader.close()
        want = getattr(case, f"{side}_hw")
        if want and tuple(int(v) for v in want) != tuple(shapes[side]):
            raise RuntimeError(
                f"{case.case_id}: the {side} image is {shapes[side]}, the case says "
                f"{tuple(want)}"
            )
        out["reference" if side == "ref" else "moving"] = str(path)
    out["record"] = {
        "peak_rss_mb": _peak_rss_mb(),
        "peak_rss_is": "the process's peak when the preprocessing ended (it is the "
        "first thing the run does; the registration's own peak is measured apart)",
        "candidate": name,
        "base": c["base"] if brightfield else "the channel itself",
        "operation": c["op"] or "none",
        "single_channel_rule": SINGLE_CHANNEL[name],
        "identity_here": is_identity(name, brightfield),
        "frame_check_px": {
            "dx": check["dx"],
            "dy": check["dy"],
            "max": check["max"],
            "n": check["n"],
        },
        "prep_s": time.perf_counter() - t0,
        "hw": {k: list(v) for k, v in shapes.items()},
    }
    return out
