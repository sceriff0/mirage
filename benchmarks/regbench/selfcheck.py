"""selfcheck and realcheck: do a method's points follow the image its own writer produced?

A method is scored through a point mapping (``mapping.PointMap``). That is only fair if the
mapping IS the transform of the image the method wrote. These two checks measure exactly
that and nothing else: registration accuracy does not enter either of them.

**selfcheck** (a gate; ``run``, ``python -m benchmarks.regbench selfcheck``). A synthetic
pair whose moving slide carries a second channel of dots on a jittered lattice
(``make_pair``). The method runs natively through the normal driver, writer included. The
dots are found blind in the method's own output (``find_dots``), and compared with where
the point mapping says they are (``compare``):

``mov_to_ref``  ``fn(dots)`` is where each dot should be drawn; the nearest detection
                within ``MATCH_PX`` is its image. Residual = ``fn(dot) - detection``.
``ref_to_mov``  every detection is sent back, ``fn(detection)``, and matched to the nearest
                original dot. Residual = ``fn(detection) - dot``.

In both directions the residual is *the point mapping minus the image*, in the frame the
mapping leads to. A method whose writer is known to depart from its mapping declares that
departure (``SELFCHECK_OFFSET``, below) and it is subtracted point by point.

The gate (frozen): 95th percentile of the residual at most ``GATE_P95`` px and mean signed
residual at most ``GATE_MEAN`` px per axis, with at least ``GATE_FOUND`` of the dots found,
on the whole image AND in every ``BLOCK`` px block that holds at least ``MIN_BLOCK_DOTS``
dots. The lattice is jittered so that a mapping wrong by a whole lattice period cannot pass.

Statuses, per variant and overall (the worst over the checked variants):

``ok``              the gate holds
``inconsistent``    the gate fails: the points do not follow the image. BLOCKS the method
``crashed``         the method raised. BLOCKS the method
``not_checked``     nothing could be measured (the dots were not detected, the method wrote
                    no image, the run ran out of memory or time). BLOCKS the method, with
                    ``cause`` saying why; it is never reported as ``inconsistent``
``no_stage_image``  a stage the method's own writer cannot draw (``stare_rigid``): reported,
                    never counted as checked, does not block

Two blind spots travel with the result: a defect the writer and the point path SHARE passes
this check, and a stage with no image of its own is not checked.

**What a method provides** (all optional, used here only):

``write_native(state, case_id, dest_dir)``   as for a cohort run: the final variant's image.
``info["native"]`` at a yield                an image the method wrote at that yield. With
        ``opts["stage_images"]`` set (the selfcheck sets it) a method writes its earlier
        stages this way too.
``write_stage(state, case_id, variant, dest_dir)``  an earlier stage's image, written after
        the generator ends; returns the same dict as ``write_native``, or None.
``warp_extra(state, case_id, variant, image_path, dest_dir)``  the method's own writer
        applied to another image on the moving slide's grid (the one-channel dots image),
        for a method whose native output does not carry the second channel.
``SELFCHECK_OFFSET(info)``  ``{variant: f}`` with ``f(xy (N, 2)) -> (N, 2)``: the declared
        point-mapping-minus-image offset at the points handed to ``fn`` (the dots for
        ``mov_to_ref``, the detections for ``ref_to_mov``), in the frame ``fn`` leads to.
``SELFCHECK_MASK(info, mask)``  ``g(xy (N, 2)) -> bool (N,)``, True where a point is left
        out (``--mask overlap``: ASHLAR's tile overlap bands), at the same points.
``SELFCHECK_SUBPROCESS = True``  each pair is run in its own Python process (a method that
        ends its process state on return, such as a JVM that cannot restart).
``SELFCHECK_REF_DOTS = True``  the REFERENCE slide of the pair carries its own dots too
        (another lattice), for a method that redraws the reference (ASHLAR re-stitches it).
``SELFCHECK_REFERENCE(info, native, pair)``  for such a method: ``{"point_map", "file",
        "channel", "drop", "expect_origin_xy"}``, a second comparison of the same kind on
        the reference's dots (``pair["dots_ref_xy"]``) in the image the method wrote of
        the reference. ``expect_origin_xy`` is the ``origin_xy`` the method reports for
        its image; it must agree, within the gate's mean, with the origin read off the
        image (the median of matched dot minus the pixel it was found at).
        Its result is stored under ``reference`` and the variant's status is the worse of
        the two.
``SELFCHECK_NOT_CHECKED(info, variant)``  a reason (text) why an ``ok`` comparison of this
        variant proves too little on this run, or None. With a reason the variant is
        ``not_checked`` and the numbers are kept (PALOM: every block shift is zero, so a
        wrong sign or block order could not have shown). It never softens a failure.
``SELFCHECK_PAIR_TILE(opts)``  the TIFF tile of the pair's files, for a method whose
        behaviour depends on it (PALOM's blocks are the reference file's tiles); None
        or absent: the generator's own.

A native dict may say where its dots are: ``dots_file``, ``dots_channel`` (default: the
first of ``files``, at the moving slide's dots channel INDEX; names are not used, because
a writer may leave channels anonymous) and ``origin_xy`` (default ``(0, 0)``: where the
image's first pixel sits in the reference frame).

**realcheck** (one real case per method; ``realcheck``, ``run --realcheck``). The synthetic
pair does not exercise whole-slide code paths. On a real case, nuclei are detected blind in
five crops of the method's native nuclear output and matched to the point-mapped native
nuclei; reported per crop are the mean signed offset per axis and the matched fraction
against a baseline (the same detector on the unwarped slide of the frame the mapping leads
to). The median distance is shown and not used: it measures segmentation noise.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path

import numpy as np

from . import cells, imageio, pairs, resources
from .cases import (
    Case,
    load_case,
    load_points,
    polys,
    result_dir,
    write_case,
    write_json,
)
from .datasets import synthetic
from .mapping import MOV_TO_REF

# ── the synthetic pair ────────────────────────────────────────────────────────
DOT_SIGMA = 1.5  # px
DOT_PEAK = 20000.0  # counts, on a zero background
LATTICE = 96  # px between dots
JITTER = 24.0  # px, uniform, each axis: an offset of one lattice period cannot match
DOT_MARGIN = 32  # every dot at least this far inside the moving frame
PIXEL_SIZE_UM = synthetic.PIXEL_SIZE_UM
NUCLEAR_CHANNEL, DOTS_CHANNEL = 0, 1
CHANNEL_NAMES = ("DAPI", "dots")
DEFAULT_HW = (2048, 2560)
DATASET = "selfcheck"
# (reference (H, W), moving (H, W) or None for the same canvas), by method or label. The
# longest key that the label (else the method) equals or starts with applies.
SIZES = {
    # two REG_TILE tiles and one seam, then a pair with a four-tile junction
    "stare": [(DEFAULT_HW, None), ((4096, 5120), None)],
    # Confirmed from VALIS's source by Task 7 (sdists 1.0.0 and 1.2.0 [PARTIAL: see
    # methods/_valis_points.py]). On the default pair the stages run at or near full
    # resolution (rigid 512 or 2048 px, first field 2048 px on a 2560 px slide), and two
    # micro passes do not run as on the cohort: 1.2.0's `register_micro` returns early
    # unless its size exceeds the first pass's 2048 px (`registration.py:4862-4871`), and
    # the hires size is 0.25 x the smallest slide's long side (640 px here); the
    # pipeline's is 0.125 x (320 px), which 1.0.0 runs at a COARSER grid than the first
    # pass. With the smaller slide at 29000 px the three micro sizes are 4096, 7250 and
    # 3625 px, all above 2048, and the first field is applied at 14x (29x on a 60000 px
    # cohort slide). Not reached by either pair: the tiler (stored fields), which needs
    # more than 10 GB of images at registration size (`TILER_THRESH_GB`).
    "valis": [(DEFAULT_HW, None), ((24000, 30000), (23000, 29000))],
    "valis_mirage": [(DEFAULT_HW, None), ((24000, 30000), (23000, 29000))],
    "valis120": [(DEFAULT_HW, None), ((24000, 30000), (23000, 29000))],
    # DeeperHistReg, one pair for all eight labels (confirmed by Task 9 from the source:
    # MWod/DeeperHistReg main @ 42e7c9d `registration_params.py:484-565, 652-733`,
    # `dhr_utils/utils.py:530-540`, `60-64`, `435-464` [PARTIAL: see
    # methods/deeperhistreg.py]). It loads at 0.2 and resamples the SHORT side down to
    # `initial_resolution`, never up. With 40000 px on the short side the 4096 px set
    # works at k = 1.953125, S = 9.765625 (the cohort's class), and the 8192 px set at
    # k = 1, S = 5.0 (loaded 8000 px is below 8192). Every division is exact here (loaded
    # 8000 x 10000, a field of 4096 x 5120 or 8000 x 10000, pads 200 and 1000), which
    # the declared offset of the estimated-field row needs. The coarsest pyramid level
    # is 32 px (eight levels) and 31 px (nine). A smaller pair for the 8192 px set was
    # dropped: (8200, 10240) also gives S = 5.0 and adds only a 6 px coarsest level,
    # nothing the cohort has (the package fails outright once a level reaches 1 px,
    # its issue #22). NOT reached by this pair: a non-integer second resampling at the
    # 8192 px set (short side above 40960 px: S between 5 and about 6.1), and sizes that
    # leave remainders; the realcheck is the check there.
    "deeperhistreg": [((40000, 50000), (40000, 48000))],
    # ASHLAR at 1024 px tiles and a 922 px pitch: 3 x 3 tiles, then 5 x 6 with interior
    # tiles and four-tile corners. The same two pairs as `stare`, whose rigid matrix the
    # anchored rows read (`--m0-from`).
    "ashlar": [(DEFAULT_HW, None), ((4096, 5120), None)],
    "affine_ashlar": [(DEFAULT_HW, None), ((4096, 5120), None)],
    # PALOM (both labels): 8 x 10 blocks of 1024 px. Its filter thresholds the
    # distribution of the block shifts and fits a regression on them
    # (labsyspharm/palom v2026.4.1 `align.py:54-80` [FULL]), which the default pair's
    # 2 x 3 blocks cannot feed; the maintainer says small tiles break it too (issue #12
    # [PARTIAL: the maintainer's comments]). The second pair gives the two slides
    # different sizes, hence different thumbnail down-factors (`reader.py:70-79`): on
    # the one-level files written here PALOM builds its own levels by 2 x 2 means with
    # the excess trimmed, so 7782 rows give 1945 at level 2 and a factor of 4.0005.
    # `palom` reads files with 1024 px tiles; `palom_b1024` files with 2048 px tiles
    # (the cohort's layout) rechunked to 1024 (`methods/palom.py::SELFCHECK_PAIR_TILE`).
    # NOT reached: the cohort's 2048 px blocks for `palom` (the realcheck is the check).
    "palom": [((8192, 10240), None), ((8192, 10240), (7782, 9728))],
}

# ── the detector ──────────────────────────────────────────────────────────────
MEDIAN_PX = 15  # background: a median of this side ...
MEDIAN_STEP = 4  # ... evaluated on a lattice of this step and interpolated
WINDOW = 7  # px: second-moment width and centroid
WIDTH_PX = (1.2, 1.8)  # a dot; a sigma 5 px nucleus is wider
NOISE_FACTOR = 8.0  # x 1.4826 x MAD of the block
MAX_FRACTION = 0.25  # x the brightest peak of the whole image
DETECT_BLOCK = 2048
_HALO = 16

# ── the gate (frozen: changing one after a cohort run is a post hoc change) ───
EDGE_PX = 16  # dots closer than this to the output's edge are not counted
MATCH_PX = 3.0
BLOCK = 512
MIN_BLOCK_DOTS = 8
GATE_P95 = 0.40
GATE_MEAN = 0.15
GATE_FOUND = 0.99
MIN_DETECTED = 0.5  # of the dots drawn; below it nothing was measured

OK, INCONSISTENT, CRASHED = "ok", "inconsistent", "crashed"
NOT_CHECKED, NO_STAGE_IMAGE = "not_checked", "no_stage_image"
WORST_FIRST = (INCONSISTENT, CRASHED, NOT_CHECKED, OK)
BLOCKING = (INCONSISTENT, CRASHED, NOT_CHECKED)
EXIT_CODE = {OK: 0, INCONSISTENT: 1, NOT_CHECKED: 2, CRASHED: 3}
# run.json causes that say the run could not be measured, not that the method is wrong
_RESOURCE_CAUSES = ("oom", "gpu_oom", "killed", "timeout")

# ── realcheck (frozen) ────────────────────────────────────────────────────────
REAL_CROP = 2048
REAL_OFFSET_PX = 0.3
REAL_MATCHED = 0.7  # of the baseline's matched fraction
REAL_MIN_NUCLEI = 20  # a crop with fewer detections is reported and not judged
REALCHECK_EXIT = 3  # `run --realcheck` when a realcheck is flagged


def sizes_for(name):
    """The pairs a method (or a label) is checked on."""
    keys = [k for k in SIZES if name == k or str(name).startswith(k + "_")]
    return list(SIZES[max(keys, key=len)]) if keys else [(DEFAULT_HW, None)]


def pair_spec(ref_hw=DEFAULT_HW, mov_hw=None, seed=0):
    """The generator spec of the selfcheck pair: 1.5 degrees, a shift and a smooth 10 px
    field, on a canvas that is not square and whose tissue does not start at the origin."""
    h, w = (int(v) for v in ref_hw)
    return {
        **synthetic._BASE,
        "family": "selfcheck",
        "n": max(h, w),
        "ref_hw": (h, w),
        "mov_hw": tuple(int(v) for v in mov_hw) if mov_hw else (h, w),
        "theta": 1.5,
        "shift": (18.0, -12.0),
        "amp": 10.0,
        "wavelength": 1500.0,
        "seed": int(seed),
        "tissue_box": (int(0.08 * w), int(0.06 * h), int(0.97 * w), int(0.96 * h)),
    }


def lattice_dots(mov_hw, seed=0):
    """``(K, 2)`` dot centres in the moving frame: a jittered lattice, every dot at least
    ``DOT_MARGIN`` px inside."""
    h, w = mov_hw
    gx, gy = np.meshgrid(
        np.arange(LATTICE / 2.0, w, LATTICE), np.arange(LATTICE / 2.0, h, LATTICE)
    )
    rng = np.random.default_rng([int(seed), 96])
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy = xy + rng.uniform(-JITTER, JITTER, xy.shape)
    lo, hi = DOT_MARGIN, np.array([w - 1.0 - DOT_MARGIN, h - 1.0 - DOT_MARGIN])
    return xy[((xy >= lo) & (xy <= hi)).all(axis=1)]


def render_dots(dots_xy, x0, y0, x1, y1, sigma=DOT_SIGMA, peak=DOT_PEAK):
    """The dots image on ``[y0, y1) x [x0, x1)``, float64, zero background."""
    out = np.zeros((y1 - y0, x1 - x0))
    r = int(math.ceil(6 * sigma))
    xy = np.asarray(dots_xy, dtype=float).reshape(-1, 2)
    near = (
        (xy[:, 0] >= x0 - r)
        & (xy[:, 0] < x1 + r)
        & (xy[:, 1] >= y0 - r)
        & (xy[:, 1] < y1 + r)
    )
    for x, y in xy[near]:
        ix, iy = int(round(x)), int(round(y))
        ax0, ax1 = max(ix - r, x0), min(ix + r + 1, x1)
        ay0, ay1 = max(iy - r, y0), min(iy + r + 1, y1)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        ex = np.exp(-((np.arange(ax0, ax1) - x) ** 2) / (2.0 * sigma * sigma))
        ey = np.exp(-((np.arange(ay0, ay1) - y) ** 2) / (2.0 * sigma * sigma))
        out[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] += peak * ey[:, None] * ex[None]
    return out


def inverse_of(spec, tol=1e-4, max_iter=60):
    """reference -> moving: the fixed-point inverse of the GENERATOR's forward map, to
    ``tol`` px. This is the benchmark's own truth, used to build test renders; no method's
    transform is ever inverted."""
    g = synthetic.forward_map(spec)
    th = math.radians(spec["theta"])
    rinv = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])

    def inv(xy):
        q = np.asarray(xy, dtype=float).reshape(-1, 2)
        p = q.copy()
        for _ in range(max_iter):
            step = (q - g(p)) @ rinv.T
            p = p + step
            if not len(p) or float(np.abs(step).max()) < tol:
                break
        return p

    return inv


def make_pair(
    work,
    ref_hw=DEFAULT_HW,
    mov_hw=None,
    seed=0,
    workers=1,
    dots_image=False,
    ref_dots=False,
    tile=None,
):
    """Write the selfcheck pair under ``work`` and describe it.

    ``reference`` and ``moving`` are two-channel 16-bit OME-TIFFs at 0.325 um per pixel:
    channel 0 the nuclei (``DAPI``), channel 1 ``dots`` (empty in the reference). Returns
    them with ``dots_mov_xy (K, 2)``, ``truth`` (moving -> reference, exact) and
    ``truth_inverse`` (reference -> moving, the generator's map inverted to 1e-4 px).
    ``dots_image`` also writes the dots as a one-channel image on the moving grid, for a
    method whose writer is applied to it (``warp_extra``). ``ref_dots`` draws another
    lattice of dots in the reference's second channel (``dots_ref_xy``, reference frame),
    for a method that redraws the reference; no method registers on that channel.
    ``tile`` is the TIFF tile of every file written (default: the generator's).
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    spec = pair_spec(ref_hw, mov_hw, seed)
    ref_hw, mov_hw = spec["ref_hw"], spec["mov_hw"]
    dots = lattice_dots(mov_hw, seed)
    dots_ref = lattice_dots(ref_hw, seed + 1) if ref_dots else np.empty((0, 2))
    tile = int(tile or synthetic.TILE)

    def dot_tiles(hw, xy):
        for x0, y0, x1, y1 in synthetic.tile_boxes(hw, tile):
            yield np.rint(render_dots(xy, x0, y0, x1, y1)).astype(np.uint16)

    def both(which, hw, xy):
        yield from synthetic._tiles(spec, which, workers, tile)
        yield from dot_tiles(hw, xy)

    paths = {}
    for which, name, hw, xy in (
        ("ref", "reference", ref_hw, dots_ref),
        ("mov", "moving", mov_hw, dots),
    ):
        paths[name] = work / f"{name}.ome.tif"
        imageio.write_channel_tiles(
            paths[name],
            hw,
            np.uint16,
            both(which, hw, xy),
            tile,
            CHANNEL_NAMES,
            PIXEL_SIZE_UM,
        )
    dots_path = None
    if dots_image:
        dots_path = work / "dots.ome.tif"
        imageio.write_channel_tiles(
            dots_path,
            mov_hw,
            np.uint16,
            dot_tiles(mov_hw, dots),
            tile,
            (CHANNEL_NAMES[DOTS_CHANNEL],),
            PIXEL_SIZE_UM,
        )
    return {
        "reference": str(paths["reference"]),
        "moving": str(paths["moving"]),
        "dots_image": str(dots_path) if dots_path else None,
        "dots_mov_xy": dots,
        "dots_ref_xy": dots_ref,
        "truth": synthetic.forward_map(spec),
        "truth_inverse": inverse_of(spec),
        "spec": spec,
        "ref_hw": tuple(ref_hw),
        "mov_hw": tuple(mov_hw),
        "seed": int(seed),
        "pixel_size_um": PIXEL_SIZE_UM,
        "nuclear_channel": NUCLEAR_CHANNEL,
        "dots_channel": DOTS_CHANNEL,
        "tile": tile,
    }


def pair_case(pair, cases_root):
    """The pair as a case of the dataset ``selfcheck`` (no nuclei: nothing is scored)."""
    h, w = pair["ref_hw"]
    case = Case(
        dataset=DATASET,
        case_id=f"pair_{h}x{w}",
        group=f"pair_{h}x{w}",
        ref_nuclear=pair["reference"],
        mov_nuclear=pair["moving"],
        ref_image=pair["reference"],
        mov_image=pair["moving"],
        modality="fluorescence",
        diagonal=math.hypot(h, w),
        pixel_size_um=pair["pixel_size_um"],
        ref_hw=list(pair["ref_hw"]),
        mov_hw=list(pair["mov_hw"]),
        extra={
            "ref_channel": pair["nuclear_channel"],
            "mov_channel": pair["nuclear_channel"],
            "dots_channel": pair["dots_channel"],
            "channel_names": list(CHANNEL_NAMES),
        },
    )
    return load_case(write_case(cases_root, case))


# ── blind detection ───────────────────────────────────────────────────────────
class _Plane:
    """One 2-D plane, read by region: an array, or a channel of a lazy ``(C, H, W)``."""

    def __init__(self, image, channel=None):
        self.image, self.channel = image, channel
        shape = image.shape
        self.shape = tuple(int(v) for v in (shape if channel is None else shape[1:]))
        if len(self.shape) != 2:
            raise ValueError(f"expected one 2-D plane, got shape {shape}")

    def read(self, y0, y1, x0, x1):
        sl = (slice(y0, y1), slice(x0, x1))
        if self.channel is not None:
            sl = (self.channel, *sl)
        return np.asarray(self.image[sl], dtype=np.float32)


def _as_plane(image):
    return image if isinstance(image, _Plane) else _Plane(np.asarray(image))


def _background(a):
    """A ``MEDIAN_PX`` median, evaluated every ``MEDIAN_STEP`` px and interpolated."""
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.ndimage import map_coordinates

    pad = MEDIAN_PX // 2
    p = np.pad(a, pad, mode="reflect")
    win = sliding_window_view(p, (MEDIAN_PX, MEDIAN_PX))[::MEDIAN_STEP, ::MEDIAN_STEP]
    med = np.empty(win.shape[:2], np.float32)
    rows = max(1, 4_000_000 // (win.shape[1] * MEDIAN_PX * MEDIAN_PX))
    for i in range(0, win.shape[0], rows):
        part = win[i : i + rows]
        med[i : i + rows] = np.median(part.reshape(*part.shape[:2], -1), axis=-1)
    yy = np.arange(a.shape[0], dtype=np.float32)[:, None] / MEDIAN_STEP
    xx = np.arange(a.shape[1], dtype=np.float32)[None, :] / MEDIAN_STEP
    at = np.broadcast_arrays(yy, xx)
    return map_coordinates(med, at, order=1, mode="nearest", prefilter=False)


def _candidates(a, x0, y0, core):
    """Peaks of one block: ``(xy (K, 2), peak (K,), width (K,))`` of the 3 x 3 local maxima
    more than ``NOISE_FACTOR`` robust s.d. above the background, refined in a window."""
    from scipy.ndimage import maximum_filter

    empty = np.empty((0, 2)), np.empty(0), np.empty(0)
    if min(a.shape) < WINDOW:
        return empty
    sub = a - _background(a)
    noise = 1.4826 * float(np.median(np.abs(sub - np.median(sub))))
    peak = (sub == maximum_filter(sub, size=3)) & (sub > NOISE_FACTOR * noise)
    peak &= sub > 0
    r = WINDOW // 2
    peak[:r], peak[-r:], peak[:, :r], peak[:, -r:] = False, False, False, False
    ys, xs = np.nonzero(peak)
    keep = (
        (xs + x0 >= core[0])
        & (xs + x0 < core[2])
        & (ys + y0 >= core[1])
        & (ys + y0 < core[3])
    )
    ys, xs = ys[keep], xs[keep]
    if not len(ys):
        return empty
    off = np.arange(-r, r + 1)
    win = np.clip(
        sub[ys[:, None, None] + off[None, :, None], xs[:, None, None] + off], 0, None
    ).astype(float)
    total = win.sum(axis=(1, 2))
    cx = (win * off[None, None, :]).sum(axis=(1, 2)) / total
    cy = (win * off[None, :, None]).sum(axis=(1, 2)) / total
    vx = (win * off[None, None, :] ** 2).sum(axis=(1, 2)) / total - cx * cx
    vy = (win * off[None, :, None] ** 2).sum(axis=(1, 2)) / total - cy * cy
    width = np.sqrt(np.clip((vx + vy) / 2.0, 0, None))
    xy = np.stack([xs + x0 + cx, ys + y0 + cy], axis=1)
    return xy, sub[ys, xs].astype(float), width


def find_dots(image):
    """Dot centres ``(K, 2)`` in X, Y, found blind in one plane.

    Per ``DETECT_BLOCK`` px block: subtract the background (a 15 px median); take the 3 x 3
    local maxima above ``8 x 1.4826 x MAD`` of the block. Over the whole image: keep those
    above ``0.25 x`` the brightest peak, whose second-moment width in a 7 px window is 1.2
    to 1.8 px, and place each at the intensity-weighted centroid of that window.
    """
    plane = _as_plane(image)
    h, w = plane.shape
    parts = []
    for by in range(0, h, DETECT_BLOCK):
        for bx in range(0, w, DETECT_BLOCK):
            x0, y0 = max(bx - _HALO, 0), max(by - _HALO, 0)
            x1 = min(bx + DETECT_BLOCK + _HALO, w)
            y1 = min(by + DETECT_BLOCK + _HALO, h)
            core = (bx, by, bx + DETECT_BLOCK, by + DETECT_BLOCK)
            parts.append(_candidates(plane.read(y0, y1, x0, x1), x0, y0, core))
    xy = np.concatenate([p[0] for p in parts]) if parts else np.empty((0, 2))
    if not len(xy):
        return np.empty((0, 2))
    peak = np.concatenate([p[1] for p in parts])
    width = np.concatenate([p[2] for p in parts])
    keep = (peak > MAX_FRACTION * float(peak.max())) & (width >= WIDTH_PX[0])
    keep &= width <= WIDTH_PX[1]
    xy, peak = xy[keep], peak[keep]
    if (
        len(xy) > 1
    ):  # a flat-topped dot (an 8-bit output) has two equal maxima: keep one
        from scipy.spatial import cKDTree

        order = np.argsort(-peak, kind="stable")
        xy = xy[order]
        drop = np.zeros(len(xy), bool)
        for i, j in sorted(cKDTree(xy).query_pairs(2.0)):
            if not drop[i]:
                drop[j] = True
        xy = xy[~drop]
    return xy


# ── comparison ────────────────────────────────────────────────────────────────
def _offset_fn(expected_offset):
    """None, a constant ``(dx, dy)`` or ``f(xy) -> (N, 2)``, as a function of position."""
    if expected_offset is None:
        return lambda xy: np.zeros((len(xy), 2))
    if callable(expected_offset):
        return lambda xy: np.asarray(expected_offset(xy), dtype=float).reshape(
            len(xy), 2
        )
    const = np.asarray(expected_offset, dtype=float).reshape(2)
    return lambda xy: np.broadcast_to(const, (len(xy), 2)).copy()


def _inside(xy, hw, margin):
    h, w = hw
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return (
            (xy[:, 0] >= margin)
            & (xy[:, 0] <= w - 1 - margin)
            & (xy[:, 1] >= margin)
            & (xy[:, 1] <= h - 1 - margin)
        )


def _nearest(points, targets, radius):
    """``(found (N,), index (N,))``: the nearest target within ``radius`` of each point."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    found, idx = np.zeros(len(points), bool), np.zeros(len(points), np.int64)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    usable = np.flatnonzero(np.isfinite(targets).all(axis=1))  # a NaN is nowhere
    finite = np.isfinite(points).all(axis=1)
    if len(usable) and finite.any():
        tree = cKDTree(targets[usable])
        d, j = tree.query(points[finite], distance_upper_bound=radius)
        found[finite] = np.isfinite(d)
        idx[finite] = usable[np.where(np.isfinite(d), j, 0)]
    return found, idx


def _stats(resid, n_dots):
    """The gate's numbers for one set of residuals (whole image or one block)."""
    n_found = len(resid)
    out = {"n_dots": int(n_dots), "n_found": int(n_found)}
    if n_found:
        err = np.hypot(resid[:, 0], resid[:, 1])
        out.update(
            err_px_p50=float(np.percentile(err, 50)),
            err_px_p95=float(np.percentile(err, 95)),
            offset_px_mean_x=float(resid[:, 0].mean()),
            offset_px_mean_y=float(resid[:, 1].mean()),
        )
    else:
        nan = float("nan")
        out.update(
            err_px_p50=nan, err_px_p95=nan, offset_px_mean_x=nan, offset_px_mean_y=nan
        )
    out["passes"] = bool(
        n_dots > 0
        and n_found >= GATE_FOUND * n_dots
        and out["err_px_p95"] <= GATE_P95
        and abs(out["offset_px_mean_x"]) <= GATE_MEAN
        and abs(out["offset_px_mean_y"]) <= GATE_MEAN
    )
    return out


def compare(
    point_map,
    dots_mov_xy,
    native_dots_image,
    expected_offset=None,
    origin_xy=(0.0, 0.0),
    drop=None,
):
    """Does ``point_map`` follow the dots drawn in ``native_dots_image``?

    ``native_dots_image`` is the dots channel of the method's own output, in the reference
    frame, its first pixel at ``origin_xy``. ``expected_offset`` is the declared
    point-mapping-minus-image offset: a constant ``(dx, dy)`` or a function of the points
    handed to ``fn``. ``drop`` marks points to leave out (True), at the same points.

    Returns the whole-image numbers (``n_drawn``, ``n_detected``, ``n_dots``, ``n_found``,
    ``err_px_p50``, ``err_px_p95``, ``offset_px_mean_x``, ``offset_px_mean_y``: the residual
    after the declared offset), ``n_unexplained`` (``mov_to_ref``: detections the mapping
    puts no dot on), ``n_masked``, ``declared_px_mean_x/y``, ``blocks`` (the same numbers
    per ``BLOCK`` px block of the output), ``failed`` (what the gate objected to) and
    ``status``.
    """
    plane = _as_plane(native_dots_image)
    hw = plane.shape
    origin = np.asarray(origin_xy, dtype=float).reshape(2)
    dots = np.asarray(dots_mov_xy, dtype=float).reshape(-1, 2)
    det_px = find_dots(plane)  # the output's own pixels
    det = det_px + origin  # the reference frame
    declared = _offset_fn(expected_offset)
    forward = point_map.direction == MOV_TO_REF
    out = {
        "direction": point_map.direction,
        "n_drawn": int(len(dots)),
        "n_detected": int(len(det)),
        "n_masked": 0,
        "n_unexplained": 0,
    }

    def masked(xy):
        if drop is None or not len(xy):
            return np.zeros(len(xy), bool)
        return np.asarray(drop(xy), dtype=bool).reshape(len(xy))

    if forward:
        # where the image should show each dot, if the points follow the image
        src = dots[~masked(dots)]
        out["n_masked"] = int(len(dots) - len(src))
        off = declared(src)
        pred = np.asarray(point_map.fn(src), dtype=float).reshape(-1, 2) - off
        keep = _inside(pred - origin, hw, EDGE_PX)
        pred, off = pred[keep], off[keep]
        found, j = _nearest(pred, det, MATCH_PX)
        resid = pred[found] - det[j[found]]
        where = pred - origin
        # the other way round: a dot the image shows and the mapping puts nothing on
        deep = det[_inside(det_px, hw, EDGE_PX + MATCH_PX)]
        if drop is None:
            every = np.asarray(point_map.fn(dots), dtype=float).reshape(-1, 2)
            every = every - declared(dots)
            out["n_unexplained"] = int((~_nearest(deep, every, MATCH_PX)[0]).sum())
        n_deep = len(deep)
    else:
        src = det[_inside(det_px, hw, EDGE_PX)]
        hide = masked(src)
        out["n_masked"] = int(hide.sum())
        src = src[~hide]
        off = declared(src)
        back = np.asarray(point_map.fn(src), dtype=float).reshape(-1, 2) - off
        found, j = _nearest(back, dots, MATCH_PX)
        resid = back[found] - dots[j[found]]
        where = src - origin
        n_deep = 0
        if found.any():
            # where the image's first pixel sits in the frame the mapping leads to, read
            # off the image: matched dot minus the pixel it was found at. A constant only
            # for a mapping that is a translation there (a mosaic, not a warp).
            at = np.median(dots[j[found]] - (src[found] - origin), axis=0)
            out["image_origin_px_median_x"] = float(at[0])
            out["image_origin_px_median_y"] = float(at[1])
    out.update(_stats(resid, len(where)))
    whole_passes = out.pop("passes")
    out["declared_px_mean_x"] = float(off[:, 0].mean()) if len(off) else 0.0
    out["declared_px_mean_y"] = float(off[:, 1].mean()) if len(off) else 0.0

    blocks, failed = [], []
    if len(where):
        key = np.floor(where / BLOCK).astype(np.int64)
        idx_found = np.cumsum(found) - 1  # row of `resid` for each found point
        order = np.lexsort((key[:, 1], key[:, 0]))  # one pass, whatever the slide size
        cuts = np.flatnonzero(np.any(np.diff(key[order], axis=0) != 0, axis=1)) + 1
        for rows in np.split(order, cuts):
            bx, by = (int(v) for v in key[rows[0]])
            st = _stats(resid[idx_found[rows[found[rows]]]], len(rows))
            judged = st["n_dots"] >= MIN_BLOCK_DOTS
            st.update(block_x=bx, block_y=by, judged=judged)
            if judged and not st["passes"]:
                failed.append(f"block ({bx}, {by})")
            blocks.append(st)
    out["blocks"] = blocks
    out["n_blocks_judged"] = sum(b["judged"] for b in blocks)
    if not whole_passes:
        failed.insert(0, "whole image")
    if out["n_unexplained"] > (1.0 - GATE_FOUND) * max(n_deep, 1):
        failed.append("detections the mapping puts no dot on")
    out["failed"] = failed
    if out["n_detected"] < MIN_DETECTED * out["n_drawn"]:
        out["status"] = NOT_CHECKED
        out["cause"] = (
            f"{out['n_detected']} of {out['n_drawn']} dots were detected in the output"
        )
    else:
        out["status"] = INCONSISTENT if failed else OK
    return out


# ── running a method on the pair ──────────────────────────────────────────────
def worst(statuses):
    """The overall status of the checked variants; with none checked, ``not_checked``."""
    checked = [s for s in statuses if s != NO_STAGE_IMAGE]
    for s in WORST_FIRST:
        if s in checked:
            return s
    return NOT_CHECKED


def _files(native):
    return (native or {}).get("files") if isinstance(native, dict) else None


def _dots_plane(native, default_channel):
    """``(_Plane, close, origin_xy)`` of the dots in a native dict, or None when the image
    does not carry them."""
    from stare.slide_io import open_lazy

    path = native.get("dots_file") or _files(native)[0]
    arr, _, close = open_lazy(path)
    channel = native.get("dots_channel")
    if channel is None:
        channel = default_channel
    if not 0 <= int(channel) < arr.shape[0]:
        close()
        return None
    origin = native.get("origin_xy") or (0.0, 0.0)
    return _Plane(arr, int(channel)), close, origin


def _proxy(mod, name, pair, extras):
    """The method, with a writer that also asks for what only the selfcheck needs."""
    real = getattr(mod, "write_native", None)
    write_stage = getattr(mod, "write_stage", None)
    warp_extra = getattr(mod, "warp_extra", None)

    def write_native(state, case_id, dest_dir):
        native = real(state, case_id, dest_dir) if real else None
        dest = Path(dest_dir)
        for i, variant in enumerate(mod.VARIANTS):
            final = i == len(mod.VARIANTS) - 1
            for hook, key in ((write_stage, "stage"), (warp_extra, "dots")):
                if hook is None or (final and key == "stage"):
                    continue
                d = dest.parent / f"selfcheck_{key}" / variant
                d.mkdir(parents=True, exist_ok=True)
                try:
                    if key == "stage":
                        got = hook(state, case_id, variant, d)
                    else:
                        got = hook(state, case_id, variant, pair["dots_image"], d)
                        if got is not None:
                            got = {"dots_channel": 0, **got}
                except Exception as exc:
                    got = {"error": f"{type(exc).__name__}: {exc}"}
                if got is not None:
                    extras[(key, variant)] = got
        return native

    ns = types.SimpleNamespace(
        NAME=name, SCOPE=getattr(mod, "SCOPE", "pair"), VARIANTS=mod.VARIANTS
    )
    for attr in ("register", "register_group", "VARIANT_LABELS"):
        if hasattr(mod, attr):
            setattr(ns, attr, getattr(mod, attr))
    if real or write_stage or warp_extra:
        ns.write_native = write_native
    return ns


def _check_variant(mod, variant, final, rec, got, extras, pair, mask):
    """One variant's selfcheck record."""
    if rec is None or not rec.get("ok"):
        rec = rec or {}
        cause = rec.get("cause") or "exception"
        status = NOT_CHECKED if cause in _RESOURCE_CAUSES else CRASHED
        return {"status": status, "cause": cause, "error": rec.get("error", "")}
    if got is None:
        return {"status": CRASHED, "cause": "the driver never handed this variant over"}
    pmap, info = got
    sources = [
        extras.get(("dots", variant)),
        rec.get("native"),
        extras.get(("stage", variant)),
    ]
    native = next((s for s in sources if _files(s)), None)
    if native is None:
        problem = next(
            (s["error"] for s in sources if isinstance(s, dict) and s.get("error")),
            None,
        )
        if not final and not problem:
            return {
                "status": NO_STAGE_IMAGE,
                "cause": "the method's own writer draws no image of this stage",
            }
        return {
            "status": NOT_CHECKED,
            "cause": problem or "the method wrote no native image",
        }
    opened = _dots_plane(native, pair["dots_channel"])
    if opened is None:
        return {
            "status": NOT_CHECKED,
            "cause": "the native image carries no dots channel and the method has no "
            "warp_extra",
            "native": native,
        }
    plane, close, origin = opened
    offsets = getattr(mod, "SELFCHECK_OFFSET", None)
    offset = (offsets(info) or {}).get(variant) if offsets else None
    drop = mod.SELFCHECK_MASK(info, mask) if mask else None
    try:
        out = compare(pmap, pair["dots_mov_xy"], plane, offset, origin, drop)
    except MemoryError as exc:
        return {"status": NOT_CHECKED, "cause": "oom", "error": str(exc)}
    except Exception as exc:
        return {
            "status": CRASHED,
            "cause": "the comparison raised",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        close()
    out["native"] = {k: v for k, v in native.items() if k != "error"}
    out["declared_offset"] = offset is not None
    too_little = getattr(mod, "SELFCHECK_NOT_CHECKED", None)
    if too_little is not None and out["status"] == OK:
        why = too_little(info, variant)
        if why:
            out["status"], out["cause"] = NOT_CHECKED, str(why)
    reference = getattr(mod, "SELFCHECK_REFERENCE", None)
    if reference is not None:
        out["reference"] = _check_reference(reference, info, native, pair)
        if out["reference"]["status"] != OK:
            out["failed"] = [*out.get("failed", []), "the reference as redrawn"]
            out["status"] = worst([out["status"], out["reference"]["status"]])
    return out


def _check_reference(hook, info, native, pair):
    """The second comparison of a method that redraws the reference: its own dots."""
    from stare.slide_io import open_lazy

    try:
        spec = hook(info, native, pair)
        if not spec:
            return {"status": NOT_CHECKED, "cause": "the method named no image"}
        arr, _, close = open_lazy(spec["file"])
    except Exception as exc:
        return {
            "status": CRASHED,
            "cause": "the reference comparison could not start",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        plane = _Plane(arr, int(spec["channel"]))
        res = compare(
            spec["point_map"],
            pair["dots_ref_xy"],
            plane,
            None,
            spec.get("origin_xy") or (0.0, 0.0),
            spec.get("drop"),
        )
        res.pop("blocks", None)
        res["file"], res["channel"] = str(spec["file"]), int(spec["channel"])
        want = spec.get("expect_origin_xy")
        if want is not None and res["status"] == OK:
            # the origin the method reports for its image, against the image itself
            got = [
                res.get("image_origin_px_median_x", math.nan),
                res.get("image_origin_px_median_y", math.nan),
            ]
            res["origin_xy_reported"] = [float(v) for v in want]
            off = np.abs(np.asarray(got) - np.asarray(want, dtype=float))
            if not (off <= GATE_MEAN).all():
                res["failed"] = [*res["failed"], "origin_xy"]
                res["status"] = INCONSISTENT
        return res
    except MemoryError as exc:
        return {"status": NOT_CHECKED, "cause": "oom", "error": str(exc)}
    except Exception as exc:
        return {
            "status": CRASHED,
            "cause": "the reference comparison raised",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        close()


def run_pair(
    method,
    work,
    ref_hw=DEFAULT_HW,
    mov_hw=None,
    opts=None,
    label=None,
    mask=None,
    m0_from=None,
    seed=0,
    keep_work=False,
):
    """The selfcheck of one method on one pair, in this process: a dict with ``status`` and
    every variant's numbers under ``variants``."""
    from . import methods

    mod, name = methods.load(method), methods.method_name(method)
    work = Path(work)
    opts = dict(opts or {})
    workers = int(opts.get("workers") or resources.n_cpus())
    wants_dots = hasattr(mod, "warp_extra")
    ref_dots = bool(getattr(mod, "SELFCHECK_REF_DOTS", False))
    pair_tile = getattr(mod, "SELFCHECK_PAIR_TILE", None)
    tile = pair_tile({**opts, "label": label or ""}) if pair_tile else None
    pair = make_pair(
        work / "pair", ref_hw, mov_hw, seed, workers, wants_dots, ref_dots, tile
    )
    case = pair_case(pair, work / "cases")
    out = work / "out"
    opts["stage_images"] = "1"
    if m0_from:
        opts["m0_from"] = str(m0_from)
    got, extras = {}, {}
    labelled = dict(zip(mod.VARIANTS, methods.variant_names(method, label)))
    back = {v: k for k, v in labelled.items()}

    def on_variant(case_, variant, pmap, info):
        got[back[variant]] = (pmap, info)

    proxy = _proxy(mod, name, pair, extras)
    args = (out, opts, label, work / "work", True, on_variant)
    error = ""
    try:
        if methods.scope_of(proxy) == "pair":
            methods.run_case(proxy, case, *args)
        else:
            methods.run_group(proxy, [case], *args)
    except MemoryError as exc:
        error = f"oom: {exc}"
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"

    variants = {}
    for i, variant in enumerate(mod.VARIANTS):
        try:
            rec = json.loads(
                (result_dir(out, labelled[variant], case) / "run.json").read_text()
            )
        except (OSError, ValueError):
            rec = None
        if error and (rec is None or rec.get("cause") == "killed"):
            oom = error.startswith("oom")
            res = {"status": NOT_CHECKED if oom else CRASHED, "cause": error}
        else:
            final = i == len(mod.VARIANTS) - 1
            res = _check_variant(
                mod, variant, final, rec, got.get(variant), extras, pair, mask
            )
        if rec:
            res["register_s"] = rec.get("register_s")
            res["version"] = (rec.get("info") or {}).get("version")
        variants[labelled[variant]] = res
    got.clear()
    if not keep_work:
        shutil.rmtree(work / "work", ignore_errors=True)
    return {
        "ref_hw": list(pair["ref_hw"]),
        "mov_hw": list(pair["mov_hw"]),
        "seed": int(seed),
        "n_drawn": int(len(pair["dots_mov_xy"])),
        "tile": int(pair["tile"]),
        "status": worst(v["status"] for v in variants.values()),
        "variants": variants,
    }


def _size_text(ref_hw, mov_hw):
    text = f"{ref_hw[0]}x{ref_hw[1]}"
    return text + (f":{mov_hw[0]}x{mov_hw[1]}" if mov_hw else "")


def parse_size(text):
    """``HxW`` or ``HxW:HxW`` (reference, moving) as ``(ref_hw, mov_hw | None)``."""
    try:
        parts = [tuple(int(v) for v in p.lower().split("x")) for p in text.split(":")]
        if not 1 <= len(parts) <= 2 or any(len(p) != 2 for p in parts):
            raise ValueError
    except ValueError:
        raise SystemExit(f"--size wants HxW or HxW:HxW, got {text!r}") from None
    return parts[0], (parts[1] if len(parts) == 2 else None)


def _in_subprocess(name, sub, size, opts, label, mask, m0_from, seed):
    """One pair in its own process: the pair's record, whatever happened to the process."""
    argv = [sys.executable, "-m", "benchmarks.regbench", "selfcheck", "--in-process"]
    argv += ["--method", name, "--work", str(sub), "--size", _size_text(*size)]
    argv += ["--seed", str(seed)]
    for k, v in (opts or {}).items():
        argv += ["--opt", f"{k}={v}"]
    for flag, value in (("--label", label), ("--mask", mask), ("--m0-from", m0_from)):
        if value:
            argv += [flag, str(value)]
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(argv, cwd=root, check=False)
    try:
        if done.returncode not in EXIT_CODE.values():
            raise ValueError
        return json.loads((Path(sub) / "selfcheck.json").read_text())["pairs"][0]
    except (OSError, ValueError, KeyError, IndexError):
        killed = done.returncode < 0 or done.returncode == 137
        return {
            "ref_hw": list(size[0]),
            "mov_hw": list(size[1] or size[0]),
            "seed": int(seed),
            "status": NOT_CHECKED if killed else CRASHED,
            "cause": f"the selfcheck process exited {done.returncode}"
            + (" (killed: out of memory?)" if killed else ""),
            "variants": {},
        }


def run(
    method,
    work,
    opts=None,
    label=None,
    mask=None,
    m0_from=None,
    sizes=None,
    seed=0,
    in_process=False,
    keep_work=False,
):
    """Run a method's selfcheck on every pair it is checked on and write
    ``<work>/selfcheck.json``. Returns that record: ``status`` (the worst over pairs),
    ``variants`` (each variant's worst status) and ``pairs`` (the numbers)."""
    from . import methods

    mod, name = methods.load(method), methods.method_name(method)
    if mask and not hasattr(mod, "SELFCHECK_MASK"):
        raise SystemExit(f"{name} has no SELFCHECK_MASK: --mask {mask} means nothing")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sizes = list(sizes or sizes_for(label or name))
    apart = getattr(mod, "SELFCHECK_SUBPROCESS", False) and not in_process
    if apart and not isinstance(method, str):
        raise SystemExit("a method run in a subprocess must be named in METHODS")
    host = resources.host_info()
    try:
        earlier = list(
            json.loads((work / "selfcheck.json").read_text()).get("attempts") or []
        )
    except (OSError, ValueError, AttributeError):
        earlier = []
    results = []
    for ref_hw, mov_hw in sizes:
        sub = work / f"{ref_hw[0]}x{ref_hw[1]}" if len(sizes) > 1 or apart else work
        if apart:
            size = (ref_hw, mov_hw)
            res = _in_subprocess(name, sub, size, opts, label, mask, m0_from, seed)
        else:
            res = run_pair(
                method, sub, ref_hw, mov_hw, opts, label, mask, m0_from, seed, keep_work
            )
        results.append(res)
    by_variant = {}
    for res in results:
        for variant, v in res["variants"].items():
            by_variant.setdefault(variant, []).append(v["status"])
    record = {
        "method": name,
        "label": label or "",
        "opts": {k: str(v) for k, v in (opts or {}).items()},
        "mask": mask or "",
        "status": worst(r["status"] for r in results),
        "blocks_the_method": worst(r["status"] for r in results) in BLOCKING,
        "variants": {
            k: (NO_STAGE_IMAGE if set(v) == {NO_STAGE_IMAGE} else worst(v))
            for k, v in by_variant.items()
        },
        "gate": {
            "err_px_p95_max": GATE_P95,
            "offset_px_mean_max": GATE_MEAN,
            "found_min": GATE_FOUND,
            "block_px": BLOCK,
            "min_block_dots": MIN_BLOCK_DOTS,
            "match_px": MATCH_PX,
            "edge_px": EDGE_PX,
        },
        "host": {k: host.get(k) for k in ("host", "launch_id")},
        "pairs": results,
    }
    # The selfcheck climbs in memory like a run (64 -> 256 -> 512 GB): the allocation
    # and the outcome of every attempt made in this folder, this one last.
    record["attempts"] = [
        *earlier,
        {
            "attempt": host.get("attempt"),
            "mem_requested_gb": host.get("mem_requested_gb"),
            "cpus_requested": host.get("cpus_requested"),
            "launch_id": host.get("launch_id"),
            "status": record["status"],
        },
    ]
    write_json(work / "selfcheck.json", record)
    return record


# ── realcheck ─────────────────────────────────────────────────────────────────
def _nuclear_of(case, side):
    """``(path, channel)`` of one native slide's nuclear channel."""
    path = getattr(case, f"{side}_nuclear")
    c = imageio.shape_of(path)[0]
    return path, (0 if c == 1 else int(case.extra.get(f"{side}_channel", 0)))


def _crop_boxes(xy, hw, crop):
    """Five ``(x0, y0, x1, y1)`` crops: the four corners of the points' bounding box and
    its centre, each inside the image."""
    h, w = hw
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return []
    cw, ch = min(crop, w), min(crop, h)
    bx0, by0 = np.clip(np.floor(xy.min(axis=0)), 0, [w - cw, h - ch]).astype(int)
    bx1 = int(np.clip(math.ceil(xy[:, 0].max()) + 1 - cw, bx0, w - cw))
    by1 = int(np.clip(math.ceil(xy[:, 1].max()) + 1 - ch, by0, h - ch))
    mid = ((bx0 + bx1) // 2, (by0 + by1) // 2)
    corners = [(bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1), mid]
    return [(int(x), int(y), int(x) + cw, int(y) + ch) for x, y in corners]


def _detect_nuclei(arr, channel, box, threshold, diameter):
    """Centroids of the nuclei the built-in segmenter finds in one crop, whole nuclei only."""
    x0, y0, x1, y1 = box
    tile = np.asarray(arr[channel, slice(y0, y1), slice(x0, x1)])
    labels = cells.segment_tile(tile, threshold, diameter)
    halo = int(math.ceil(diameter))
    core = (halo, halo, x1 - x0 - halo, y1 - y0 - halo)
    ps = cells.label_polys(labels, (x0, y0), core, math.pi * (diameter / 4.0) ** 2)
    return cells.area_centroid(ps)[1]


def realcheck(
    case, variant_dir, point_map, native, offset=None, crop=None, points=None
):
    """Check one real case: does the mapping follow the nuclei in the method's own output?

    ``native`` is the dict the method's writer returned (``files``; optionally
    ``nuclear_channel``, ``origin_xy``). ``offset`` is the variant's declared
    point-mapping-minus-image offset (``SELFCHECK_OFFSET``), subtracted before judging.
    Writes ``<variant_dir>/realcheck.json`` and returns it: per crop ``n``,
    ``matched_fraction``, ``baseline_fraction``, ``offset_px_mean_x/y``, ``err_px_p50``;
    ``flag`` is true when a crop's mean offset exceeds ``REAL_OFFSET_PX`` on an axis or its
    matched fraction is under ``REAL_MATCHED`` of the baseline's; ``status`` is ``ok``,
    ``flagged`` or ``not_checked`` (no crop held enough nuclei).
    """
    from stare.slide_io import open_lazy

    crop = int(crop or REAL_CROP)
    pts = points if points is not None else load_points(case)
    if polys(pts, "mov") is None or polys(pts, "ref") is None:
        record = {
            "status": NOT_CHECKED,
            "flag": False,
            "cause": "the case has no nuclei",
            "crops": [],
        }
        write_json(Path(variant_dir) / "realcheck.json", record)
        return record
    area_m, cent_m = pairs._area_cent(polys(pts, "mov"))
    area_r, cent_r = pairs._area_cent(polys(pts, "ref"))
    radius = pairs.cp.median_equivalent_radius(np.concatenate([area_m, area_r]))
    diameter = 2.0 * radius
    forward = point_map.direction == MOV_TO_REF
    declared = _offset_fn(offset)
    origin = np.asarray(native.get("origin_xy") or (0.0, 0.0), dtype=float)
    path = native.get("nuclear_file") or native["files"][0]
    arr, _, close = open_lazy(path)
    record = {
        "direction": point_map.direction,
        "native": str(path),
        "radius_px": float(radius),
        "crop_px": int(crop),
        "declared_offset": offset is not None,
        "limits": {"offset_px": REAL_OFFSET_PX, "matched_of_baseline": REAL_MATCHED},
    }
    try:
        channel = native.get("nuclear_channel")
        if channel is None:
            channel = 0 if arr.shape[0] == 1 else case.extra.get("mov_channel", 0)
        channel = int(channel)
        hw = tuple(int(v) for v in arr.shape[1:])
        if forward:  # where the image should show each moving nucleus
            mapped = np.asarray(point_map.fn(cent_m), dtype=float).reshape(-1, 2)
            mapped = mapped - declared(cent_m)
            tissue = mapped
        else:
            tissue = cent_r
        base_side = "ref" if forward else "mov"
        base_path, base_channel = _nuclear_of(case, base_side)
        base_cent = cent_r if forward else cent_m
        base_arr, _, base_close = open_lazy(base_path)
        try:
            base_hw = tuple(int(v) for v in base_arr.shape[1:])
            threshold = cells.slide_threshold(path, channel)
            base_threshold = cells.slide_threshold(base_path, base_channel)
            crops = []
            for box in dict.fromkeys(_crop_boxes(tissue - origin, hw, crop)):
                det = _detect_nuclei(arr, channel, box, threshold, diameter) + origin
                size = np.array([box[2] - box[0], box[3] - box[1]])
                if forward:
                    found, j = _nearest(det, mapped, radius)
                    resid = mapped[j[found]] - det[found]
                    lo = np.array(box[:2]) + np.rint(origin).astype(int)
                else:
                    back = np.asarray(point_map.fn(det), dtype=float).reshape(-1, 2)
                    back = back - declared(det)
                    found, j = _nearest(back, cent_m, radius)
                    resid = back[found] - cent_m[j[found]]
                    # the same crop, carried to the moving frame by the mapping itself
                    mid = np.array([box[:2]], dtype=float) + size / 2.0 + origin
                    c = np.asarray(point_map.fn(mid), dtype=float).reshape(2)
                    lo = np.rint(np.nan_to_num(c - size / 2.0)).astype(int)
                lo = np.clip(lo, 0, np.maximum(np.array(base_hw[::-1]) - size, 0))
                hi = np.minimum(lo + size, base_hw[::-1])
                base_box = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
                base_det = np.empty((0, 2))
                if min(hi - lo) > 4 * diameter:
                    base_det = _detect_nuclei(
                        base_arr, base_channel, base_box, base_threshold, diameter
                    )
                base_found = _nearest(base_det, base_cent, radius)[0]
                nan = float("nan")
                rec = {
                    "box": [int(v) for v in box],
                    "baseline_box": list(base_box),
                    "n": int(len(det)),
                    "n_matched": int(found.sum()),
                    "baseline_n": int(len(base_det)),
                    "matched_fraction": float(found.mean()) if len(det) else nan,
                    "baseline_fraction": (
                        float(base_found.mean()) if len(base_det) else nan
                    ),
                    "offset_px_mean_x": float(resid[:, 0].mean())
                    if len(resid)
                    else nan,
                    "offset_px_mean_y": float(resid[:, 1].mean())
                    if len(resid)
                    else nan,
                    "err_px_p50": (
                        float(np.median(np.hypot(resid[:, 0], resid[:, 1])))
                        if len(resid)
                        else nan
                    ),
                }
                judged = len(det) >= REAL_MIN_NUCLEI
                why = []
                if judged:
                    if (
                        max(abs(rec["offset_px_mean_x"]), abs(rec["offset_px_mean_y"]))
                        > REAL_OFFSET_PX
                    ):
                        why.append("offset")
                    # written so that a NaN (nothing to compare with) is flagged
                    if not (
                        rec["matched_fraction"]
                        >= REAL_MATCHED * rec["baseline_fraction"]
                    ):
                        why.append("matched_fraction")
                rec.update(judged=bool(judged), flag=bool(why), why=why)
                crops.append(rec)
        finally:
            base_close()
    finally:
        close()
    judged = [c for c in crops if c["judged"]]
    record["crops"] = crops
    record["flag"] = any(c["flag"] for c in judged)
    record["status"] = (
        "flagged" if record["flag"] else ("ok" if judged else NOT_CHECKED)
    )
    write_json(Path(variant_dir) / "realcheck.json", record)
    return record
