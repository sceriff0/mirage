"""DeeperHistReg 1.0.1 (MWod/DeeperHistReg, ``pip install deeperhistreg==1.0.1``) in its
native roles, with a shipped parameter set and nothing changed but the device.

**The software is DeeperHistReg; the method is RegWSI** (Wodzinski et al. 2024,
doi:10.1016/j.cmpb.2024.108187). There is no separate RegWSI code: the package is where the
method ships. Two of its parameter factories are run:

``default_initial_nonrigid``                  4096 px on the short side (the default rows)
``default_initial_nonrigid_high_resolution``  8192 px on the short side (the ``_hr`` rows)

Rows (``--label`` with the options that go with it; a label and its options must agree):

==============================  =====================================================
``deeperhistreg_gpu``           ``--opt device=cuda:0``
``deeperhistreg_cpu``           ``--opt device=cpu``
``deeperhistreg_hr_gpu|cpu``    ``+ --opt config=default_initial_nonrigid_high_resolution``
``deeperhistreg_fluo_gpu|cpu``  ``+ --opt fluo=1`` (configured by the benchmark, two keys)
``deeperhistreg_hr_fluo_...``   both
==============================  =====================================================

Each run yields two results, both reference -> moving (``REF_TO_MOV``):

``deeperhistreg``          the field the method ESTIMATED: reference pixel -> moving pixel
                           through the package's own ``warp_landmarks`` and the coordinate
                           chain of issue #12 (a user's function the maintainer endorsed;
                           the package itself never calls ``warp_landmarks``). Secondary.
``deeperhistreg_asdrawn``  the image AS WRITTEN by the package's own writer
                           (``apply_deformation_pyvips``): for every pixel of the file on
                           disk, the moving-slide position it was sampled from. This is
                           the row whose points are exactly the transform of the image,
                           and the one that can be primary.

**Why they differ** (read from the source, never executed here). The field lives on the
padded, resampled grid: ``warped(x) = source(x + u(x))`` (``dhr_utils/warping.py:55-82``).
The writer turns ``grid + u`` into a libvips index image, enlarges it to the full-resolution
padded canvas with ``resize(W_pad / w_field, kernel='linear', vscale=H_pad / h_field)``,
multiplies by the same ratios and calls ``mapim`` (``warping.py:224-239``). libvips
enlarges through ``vips_affine`` with an input displacement of 0.5 and edge copying, so
output pixel ``X`` reads the field at ``X / S - 0.5`` (libvips v8.15.2 ``resize.c:233-277``,
``affine.c:445-480`` (the output area is fixed BEFORE the displacement is applied, so it
starts at 0), ``:320-349``, ``:529``; ``mapim.c:589-593`` bilinear, background 0). The
written image therefore samples ``X - S/2 + S u(X/S - 0.5)``, half a field pixel up and
left of the chain's ``X + S u(X/S)``: its content sits about ``S/2`` px right and down of
the registration, 4.9 px at S = 9.77 and 2.5 px at S = 5. On slides whose sizes do not
divide evenly there is more: the writer pads at full resolution and stretches by
``W_pad / w_field``, the registration padded at the loaded resolution and resampled by
``initial_resample_ratio``; the two agree only when every division is exact. The as-drawn
map uses the writer's own numbers (``asdrawn_map``); how far the two maps are apart on a
case, and how much of that the half-pixel term explains, is in ``info["gap"]``.

**The written image** has the reference slide's size and is on its grid: the writer crops
the padded canvas back by the reference's own pads (``pair_full_saver.py:76-80``,
``utils.py:299-311``), so ``origin_xy`` is ``(0, 0)`` and the pads enter the MAP, not the
image's origin. Pixels from outside the moving slide are the pad value (255 counts at the
shipped parameters, 0 for the ``_fluo`` rows) inside the padded canvas and 0 beyond it;
``valid_box`` bounds the pixels that come from the moving slide.

**Inputs.** The package does not read OME-TIFF (issue #4) and its TIFF loader counts
top-level pages as pyramid levels (``tiff_loader.py:34-38``, ``:57-80``). Each slide's
nuclear channel is therefore written as a tiled TIFF with ONE top-level page: 16-bit,
values untouched, uncompressed (``single_page_tiff``). The clock is paused while this is
done; ``convert_s`` is recorded.

**Options** (``--opt key=value``): ``config`` (one of ``CONFIGS``), ``device`` (``cpu``
or ``cuda:N``; default ``cpu``; a CUDA request without CUDA is refused, because the
package would fall back silently, ``full_resolution.py:58``), ``fluo`` (``pad_value = 0``
and ``flip_intensity = False``).

**Failures.** ``run_registration`` prints an exception and carries on (``run.py:40-46``);
what the caller sees is a missing ``displacement_field.mha`` (``run.py:59-60``). stdout is
captured around the call and the error raised with that text, so the driver can tell a
CUDA or host out-of-memory from another failure (``resources.classify_error``).

Sources, with what was read for this file. MWod/DeeperHistReg main @ ``42e7c9d`` (the
PyPI 1.0.1 wheel differs in two data paths only, per the port's RegWSI report) ``[PARTIAL:
read in full run.py, dhr_pipeline/full_resolution.py, dhr_deformation/apply_deformation.py,
pair_full_loader.py, tiff_loader.py, tiff_saver.py, pair_full_saver.py, results_saver.py,
displacement_loader.py, displacement_saver.py, general_preprocessing.py, multi_feature.py;
registration_params.py:1-75, 484-565, 652-761; dhr_utils/utils.py:25-92, 126-392, 435-464,
530-744; dhr_utils/warping.py:1-252; io_nonrigid.py:28-133; cost_functions.py:55-89;
missing: the networks, the feature matchers, instance_optimization.py, the other loaders
and savers]``; ``README.md:10-32``; issues #4, #6, #8, #12, #15 ``[FULL]``, #22 ``[PARTIAL: the first
2500 characters]``. libvips v8.15.2 ``[PARTIAL: resize.c:150-300, affine.c:225-262, 300-400,
436-532, transform.c:158-252, mapim.c:225-300, 455-480, 589-594; not executed]``. The
RegWSI paper was NOT read for this file: what is said about it comes from the port's RegWSI
report, which read arXiv:2404.13108v2 ``[FULL]`` (TeX, tables, captions) and the journal
version ``[ABSTRACT]``. Nothing here was run with the package: it is not installed where
this was written.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from ..mapping import REF_TO_MOV, PointMap

SCOPE = "pair"
VARIANTS = ("deeperhistreg", "deeperhistreg_asdrawn")
VERSION = "1.0.1"
DIRECTION = REF_TO_MOV
METHOD = "RegWSI (Wodzinski et al. 2024, doi:10.1016/j.cmpb.2024.108187)"
# factory -> the resolution (short side, px) of the RegWSI paper's ablation it stands at
CONFIGS = {
    "default_initial_nonrigid": 4096,
    "default_initial_nonrigid_high_resolution": 8192,
}
DEFAULT_CONFIG = "default_initial_nonrigid"
HR_CONFIG = "default_initial_nonrigid_high_resolution"
# Where the shipped sets visibly depart from the paper's text (the port's RegWSI report,
# B2; code lines read for this file, the paper's statements taken from the report).
DEPARTURES = (
    "a scale-invariant feature transform + RANSAC candidate also runs in the initial "
    "alignment (the shipped sets switch it on; multi_feature.py:56-61); the paper's "
    "method is SuperPoint/SuperGlue only",
    "transform_type is 'rigid', a four-parameter similarity (dhr_utils/utils.py:583-603); "
    "the paper says affine",
    "the 8192 px set has nine initial-alignment sizes (150 to 600; "
    "registration_params.py:697); the paper's tables stop at eight",
    "the writer enlarges the field with libvips kernel='linear' "
    "(dhr_utils/warping.py:233); the paper says B-spline upsampling",
    "flip_intensity (1 - x after the grey conversion; general_preprocessing.py:54-56) "
    "is not mentioned in the paper's text",
)
WRITER = (
    "deeperhistreg apply_deformation_pyvips (dhr_deformation/apply_deformation.py:71-92) "
    "with the arguments save_final passes (full_resolution.py:271-281), once per channel "
    "of the moving slide"
)
WRITER_SHIFT = 0.5  # libvips' input displacement when it enlarges (resize.c:244-246)
TILE = 512
STAGE_LINES = {
    "loading_s": "Loading time",
    "preprocessing_s": "Preprocessing time",
    "initial_registration_s": "Initial registration time",
    "nonrigid_registration_s": "Nonrigid registration time",
    "total_registration_s": "Total registration time",
}
# The live maps of the run in progress, by ``info["run_token"]``: a declared offset is a
# function of position and cannot travel through run.json. One entry at a time.
_LIVE = {}


class _NullMonitor:
    def pause(self):
        pass

    def resume(self):
        pass


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ── settings ──────────────────────────────────────────────────────────────────
def settings(opts):
    """``(config, device, fluo)`` of this run, checked against its label.

    A label of the form ``deeperhistreg[_hr][_fluo]_<gpu|cpu>`` names its own settings;
    a run whose options say otherwise is refused, so a row cannot carry another row's name.
    """
    config = str(opts.get("config", DEFAULT_CONFIG))
    if config not in CONFIGS:
        raise ValueError(f"config={config!r}: expected one of {sorted(CONFIGS)}")
    device = str(opts.get("device", "cpu")).strip()
    if device != "cpu" and not re.fullmatch(r"cuda(:\d+)?", device):
        raise ValueError(f"device={device!r}: expected cpu or cuda:N")
    fluo = _truthy(opts.get("fluo", 0))
    label = str(opts.get("label") or "")
    if label.startswith("deeperhistreg_"):
        tokens = set(label.split("_")[1:])
        if tokens <= {"hr", "fluo", "gpu", "cpu"}:
            want_config = HR_CONFIG if "hr" in tokens else DEFAULT_CONFIG
            wrong = config != want_config or fluo != ("fluo" in tokens)
            wrong |= "gpu" in tokens and device == "cpu"
            wrong |= "cpu" in tokens and device != "cpu"
            if wrong:
                kind = (
                    "cuda" if "gpu" in tokens else "cpu" if "cpu" in tokens else "any"
                )
                raise ValueError(
                    f"the label {label!r} names a run with config={want_config}, "
                    f"fluo={int('fluo' in tokens)}, device {kind}; this one has "
                    f"config={config}, fluo={int(fluo)}, device={device}. A label and "
                    "its options come together"
                )
    return config, device, fluo


def changed_keys(params, device, fluo):
    """Edit a shipped parameter set IN PLACE and return the dotted keys touched, sorted.

    Every ``device`` and ``cuda`` key at any depth (the factories bake ``cuda:0`` into
    nested blocks, ``registration_params.py:536-537, 550``); ``save_final_images = False``
    (the package's own example does the same, ``deeperhistreg_examples/apply_deformation.py``:
    the writer is run afterwards, outside the clock); with ``fluo``,
    ``loading_params.pad_value = 0.0`` and ``preprocessing_params.flip_intensity = False``.
    Nothing else.
    """
    touched = []

    def walk(d, prefix):
        for key, value in d.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            elif key == "device":
                d[key] = device
                touched.append(prefix + key)
            elif key == "cuda":
                d[key] = device != "cpu"
                touched.append(prefix + key)

    walk(params, "")
    params["save_final_images"] = False
    touched.append("save_final_images")
    if fluo:
        params["loading_params"]["pad_value"] = 0.0
        params["preprocessing_params"]["flip_intensity"] = False
        touched += ["loading_params.pad_value", "preprocessing_params.flip_intensity"]
    return sorted(touched)


def working_scale(short_side_px, initial_resolution, load_ratio=0.2):
    """``(S, k)`` the package works at for a pair whose padded short side is
    ``short_side_px``: ``k`` is its ``initial_resample_ratio`` (1 when the loaded image
    is already below ``initial_resolution``; it never enlarges,
    ``dhr_utils/utils.py:530-540``) and ``S = k / load_ratio`` the full-resolution pixels
    per working pixel. The short side is ``min(max of the widths, max of the heights)``."""
    loaded = float(short_side_px) * float(load_ratio)
    k = 1.0 if initial_resolution > loaded else loaded / float(initial_resolution)
    return k / float(load_ratio), k


# ── inputs ────────────────────────────────────────────────────────────────────
def single_page_tiff(src, channel, dst):
    """Channel ``channel`` of ``src`` as a tiled TIFF with ONE top-level page, no SubIFDs,
    the source's dtype and values, uncompressed. Returns ``{"hw", "dtype", "levels"}``:
    ``levels`` is the number of occupied 8-bit grey levels the channel would keep under
    the package's min-max rule (``dhr_utils/utils.py:25-37``), counted on every fourth
    row and column of the full-resolution channel (None for a non-integer dtype)."""
    import tifffile
    from stare.slide_io import open_lazy

    arr, dtype, close = open_lazy(str(src))
    try:
        c, h, w = (int(v) for v in arr.shape)
        if not 0 <= int(channel) < c:
            raise ValueError(f"{src}: channel {channel} out of range for C={c}")
        dtype = np.dtype(dtype)
        integer = dtype.kind in "ui" and dtype.itemsize <= 2
        counts = np.zeros(1 << (8 * dtype.itemsize), np.int64) if integer else None
        lo = int(np.iinfo(dtype).min) if integer else 0

        def tiles():
            for y0 in range(0, h, TILE):
                band = np.asarray(
                    arr[int(channel), slice(y0, min(y0 + TILE, h)), slice(0, w)]
                )
                if counts is not None:
                    sub = band[::4, ::4].astype(np.int64).ravel() - lo
                    counts[: sub.max() + 1] += np.bincount(sub)
                for x0 in range(0, w, TILE):
                    t = band[:, x0 : x0 + TILE]
                    if t.shape != (
                        TILE,
                        TILE,
                    ):  # edge tiles padded, as TIFF stores them
                        full = np.zeros((TILE, TILE), dtype)
                        full[: t.shape[0], : t.shape[1]] = t
                        t = full
                    yield np.ascontiguousarray(t, dtype=dtype)

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(dst),
            tiles(),
            shape=(h, w),
            dtype=dtype,
            bigtiff=True,
            tile=(TILE, TILE),
            photometric="minisblack",
            metadata=None,
        )
    finally:
        close()
    levels = None
    if counts is not None and counts.any():
        seen = np.flatnonzero(counts).astype(float)
        span = seen[-1] - seen[0]
        levels = (
            1
            if span == 0
            else int(len(np.unique(((seen - seen[0]) / span * 255).astype(np.uint8))))
        )
    return {"hw": [h, w], "dtype": str(dtype), "levels": levels}


def _nuclear(case, side):
    """``(path, channel index)`` of a slide's nuclear channel, in its original file."""
    path = getattr(case, f"{side}_image") or getattr(case, f"{side}_nuclear")
    return str(path), int(case.extra.get(f"{side}_channel", 0))


def _channel_names(case, n):
    names = case.extra.get("channel_names")
    if names is None:
        from .. import imageio

        try:
            names = imageio.channel_names(case.mov_image)
        except Exception:
            names = []
    names = [str(v) for v in names]
    return names if len(names) == n else [f"channel_{i}" for i in range(n)]


# ── geometry ──────────────────────────────────────────────────────────────────
def centre_pads(size_a, size_b):
    """``(top, left)`` padding of each image when both are centre-padded to a common shape
    (``dhr_utils/utils.py:152-180``, ``calculate_pad_value``: the leading pad is
    ``floor(difference / 2)``; libvips' ``gravity("centre")`` at ``:220-221`` does the
    same)."""
    pads = []
    for mine, other in ((size_a, size_b), (size_b, size_a)):
        pads.append(tuple(max(int(o) - int(m), 0) // 2 for m, o in zip(mine, other)))
    return pads


def scale_xy(canvas_hw, field_hw):
    """``(S_x, S_y)``: the ratios the writer enlarges the field by, ``image_width /
    df_width`` and ``image_height / df_height`` (``dhr_utils/warping.py:228-235``)."""
    return canvas_hw[1] / field_hw[1], canvas_hw[0] / field_hw[0]


class Geometry:
    """What the two maps need: the saved field and the numbers around it.

    ``field`` is ``(2, h, w)``: X then Y displacement in FIELD pixels, as the package saves
    it (``displacement_saver.py:28-38`` after ``tc_df_to_np_df``, ``utils.py:673-702``).
    ``post`` is its ``postprocessing_params.json`` (``full_resolution.py:82-83, 104-106,
    262``): ``pad_1`` (the source, here the MOVING slide) and ``pad_2`` (the target, the
    REFERENCE) as ``[[top, bottom], [left, right]]`` at the loaded resolution, the two
    load ratios, ``initial_resampling`` and ``initial_resample_ratio``.
    """

    def __init__(self, field, post, ref_hw, mov_hw):
        self.field = np.asarray(field, dtype=np.float32)
        if self.field.ndim != 3 or self.field.shape[0] != 2:
            raise RuntimeError(
                f"unexpected displacement field shape {self.field.shape}"
            )
        self.field_hw = tuple(int(v) for v in self.field.shape[1:])
        self.ref_hw = tuple(int(v) for v in ref_hw)
        self.mov_hw = tuple(int(v) for v in mov_hw)
        self.r_mov = float(post["source_resample_ratio"])
        self.r_ref = float(post["target_resample_ratio"])
        self.resampled = bool(post.get("initial_resampling"))
        self.k = float(post["initial_resample_ratio"]) if self.resampled else 1.0
        p_mov, p_ref = post["pad_1"], post["pad_2"]
        self.pad_mov_xy = np.array([p_mov[1][0], p_mov[0][0]], dtype=float)
        self.pad_ref_xy = np.array([p_ref[1][0], p_ref[0][0]], dtype=float)
        # the writer's own pads and canvas, at full resolution (pair_full_loader.py:80-94)
        (mt, ml), (rt, rl) = centre_pads(self.mov_hw, self.ref_hw)
        self.full_pad_mov_xy = np.array([ml, mt], dtype=float)
        self.full_pad_ref_xy = np.array([rl, rt], dtype=float)
        self.canvas_hw = (
            max(self.ref_hw[0], self.mov_hw[0]),
            max(self.ref_hw[1], self.mov_hw[1]),
        )
        self.scale_xy = np.array(scale_xy(self.canvas_hw, self.field_hw), dtype=float)


def registration_map(geo, warp_landmarks):
    """reference px -> moving px: the field the method estimated.

    The chain is the function of issue #12, step for step, around the package's own
    ``warp_landmarks(points, field)`` (``dhr_utils/warping.py:212-221``: a cubic-spline
    lookup of the field at the points, ``mode='nearest'``). It leaves out the half-pixel
    terms of the two resamplings, so it is off from the registration by about
    ``grad(u) (S - 1) / 2``.
    """

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        pts = xy * geo.r_ref + geo.pad_ref_xy
        if geo.resampled:
            pts = pts / geo.k
        moved = np.asarray(warp_landmarks(pts, geo.field), dtype=float).reshape(-1, 2)
        if geo.resampled:
            moved = moved * geo.k
        return (moved - geo.pad_mov_xy) / geo.r_mov

    return fn


def asdrawn_map(geo):
    """reference px -> moving px: the image as the package's writer draws it.

    Pixel ``X`` of the written file is pixel ``X + pad_ref`` of the padded canvas. There
    the writer's index image holds ``S * m(Xp / S - 0.5)``, with ``m = grid + u``
    interpolated linearly and its edge copied; the padded moving slide is sampled at that
    position, which is ``pad_mov`` away from the moving slide's own pixel. Benchmark-side
    arithmetic: the writer's, re-stated.
    """
    from scipy.ndimage import map_coordinates

    h, w = geo.field_hw

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        p = xy + geo.full_pad_ref_xy
        qx = np.clip(p[:, 0] / geo.scale_xy[0] - WRITER_SHIFT, 0, w - 1)
        qy = np.clip(p[:, 1] / geo.scale_xy[1] - WRITER_SHIFT, 0, h - 1)
        at = [qy, qx]
        ux = map_coordinates(geo.field[0], at, order=1, mode="nearest", prefilter=False)
        uy = map_coordinates(geo.field[1], at, order=1, mode="nearest", prefilter=False)
        out = np.stack([qx + ux, qy + uy], axis=1) * geo.scale_xy
        return out - geo.full_pad_mov_xy

    return fn


def declared_offset(registration, geo):
    """The estimated-field map minus the written image, as the reading predicts it:
    ``R(X) - R(X - S/2)``, the registration differenced over the half field pixel the
    writer is off by. To first order it is ``(S/2) x Jacobian x (1, 1)``. It is built from
    the registration map alone, so a wrong chain cannot cancel out of the check."""
    half = geo.scale_xy * WRITER_SHIFT

    def offset(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        both = np.asarray(registration(np.concatenate([xy, xy - half])), dtype=float)
        return both[: len(xy)] - both[len(xy) :]

    return offset


def gap_stats(registration, asdrawn, geo, max_side=96):
    """How far the two maps are apart on a lattice over the reference slide, in moving px.

    ``mean_px``: the mean of estimated-field minus as-drawn, per axis (about ``S/2``).
    ``beyond_declared_px_p95``: the 95th percentile of what ``declared_offset`` does NOT
    explain: zero where every division of the package's resamplings is exact, more on
    slides where the writer's pads and stretch differ from the registration's.
    ``chain_halfpixel_px_p95``: the size of the half-pixel terms the issue-#12 chain
    leaves out, ``|R(X) - (R(X - c) + c)|`` with ``c = (S - 1) / 2``.
    """
    h, w = geo.ref_hw
    step = max(1.0, max(h, w) / max_side)
    gx, gy = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    half, c = geo.scale_xy * WRITER_SHIFT, (geo.scale_xy - 1.0) / 2.0
    n = len(xy)
    r = np.asarray(registration(np.concatenate([xy, xy - half, xy - c])), dtype=float)
    gap = r[:n] - np.asarray(asdrawn(xy), dtype=float)
    beyond = gap - (r[:n] - r[n : 2 * n])
    chain = r[:n] - (r[2 * n :] + c)
    return {
        "n": int(n),
        "mean_px": [float(v) for v in gap.mean(axis=0)],
        "declared_px_mean": [float(v) for v in (r[:n] - r[n : 2 * n]).mean(axis=0)],
        "half_field_px": [float(v) for v in half],
        "beyond_declared_px_p95": float(np.percentile(np.hypot(*beyond.T), 95)),
        "chain_halfpixel_px_p95": float(np.percentile(np.hypot(*chain.T), 95)),
    }


def footprint_box(fn, ref_hw, mov_hw):
    """``[x0, y0, x1, y1]`` on the written image outside which every pixel is fill."""
    from ._valis_points import footprint_box as box

    return box(fn, ref_hw, mov_hw)


# ── the package ───────────────────────────────────────────────────────────────
def installed_version(package):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("deeperhistreg")
    except PackageNotFoundError:
        return str(getattr(package, "__version__", "?"))


def _concurrency(torch):
    seen = {}
    try:
        seen["torch_num_threads"] = int(torch.get_num_threads())
    except Exception as exc:
        seen["torch_num_threads"] = f"{type(exc).__name__}: {exc}"
    try:
        import pyvips

        seen["vips_concurrency"] = int(pyvips.vips_lib.vips_concurrency_get())
    except Exception as exc:
        seen["vips_concurrency"] = f"{type(exc).__name__}: {exc}"
    return seen


def stage_times(log_text):
    """The package's own per-stage seconds, from its ``logs.txt``
    (``full_resolution.py:91, 111, 153, 194, 302``)."""
    out = {}
    for key, label in STAGE_LINES.items():
        hits = re.findall(rf"{re.escape(label)}: ([0-9.eE+-]+) seconds", log_text)
        if hits:
            out[key] = float(hits[-1])
    return out


def run_package(package, source, target, out, params, token, tmp):
    """``deeperhistreg.run_registration`` with its stdout captured: ``(printed, error)``.

    The package swallows the pipeline's exception and prints it; with final saving off the
    only thing it then raises is the ``FileNotFoundError`` of the missing field."""
    printed, error = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(printed):
            package.run_registration(
                source_path=str(source),
                target_path=str(target),
                output_path=str(out),
                registration_parameters=params,
                case_name=token,
                save_displacement_field=True,
                copy_target=False,
                delete_temporary_results=False,
                temporary_path=str(tmp),
            )
    except FileNotFoundError as exc:
        error = exc
    text = printed.getvalue()
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    return text, error


def register(case, work, opts):
    import deeperhistreg
    import SimpleITK as sitk
    import torch

    config, device, fluo = settings(opts)
    if case.modality != "fluorescence":
        raise ValueError(
            f"{case.case_id}: this adapter hands DeeperHistReg one fluorescence channel "
            f"per slide; a {case.modality} case needs its own input rule"
        )
    version = installed_version(deeperhistreg)
    if version != VERSION:
        raise RuntimeError(
            f"deeperhistreg {VERSION} is the pinned release; found {version}"
        )
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device} was asked for and torch sees no CUDA device. DeeperHistReg "
            "would fall back to the CPU without saying so (full_resolution.py:58); a GPU "
            "row that ran on the CPU is refused"
        )
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    mon = opts.get("monitor") or _NullMonitor()
    pipeline = deeperhistreg.direct_registration  # dhr_pipeline.full_resolution

    # Native roles: the moving slide is the SOURCE, the reference the TARGET.
    mon.pause()
    t0 = time.perf_counter()
    try:
        ref_p, mov_p = work / "reference.tif", work / "moving.tif"
        ref_meta = single_page_tiff(*_nuclear(case, "ref"), ref_p)
        mov_meta = single_page_tiff(*_nuclear(case, "mov"), mov_p)
    finally:
        convert_s = time.perf_counter() - t0
        mon.resume()

    params = copy.deepcopy(getattr(deeperhistreg.configs, config)())
    changed = changed_keys(params, device, fluo)
    recorded = copy.deepcopy(params)  # the call adds a log path and a case name
    token = f"regbench_{uuid.uuid4().hex[:12]}"  # its logger adds a handler per name
    out, tmp = work / "out", work / "tmp"
    field_p = out / "displacement_field.mha"
    field_p.unlink(missing_ok=True)  # never a field left by an earlier attempt
    t0 = time.perf_counter()
    printed, error = run_package(deeperhistreg, mov_p, ref_p, out, params, token, tmp)
    call_s = time.perf_counter() - t0
    log = tmp / "logs.txt"
    log_text = log.read_text(errors="replace") if log.exists() else ""
    if error is not None or not field_p.exists():
        raise RuntimeError(
            "DeeperHistReg wrote no displacement field. What it printed: "
            f"{printed.strip()[-2000:] or '(nothing)'} | logs.txt: "
            f"{log_text.strip()[-600:] or '(empty)'}"
        )
    # as its own loader reads it (displacement_loader.py:33)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_p))).astype(np.float32)
    post = json.loads((out / "postprocessing_params.json").read_text())
    geo = Geometry(field, post, ref_meta["hw"], mov_meta["hw"])
    registration = registration_map(geo, pipeline.w.warp_landmarks)
    asdrawn = asdrawn_map(geo)

    mon.pause()  # the benchmark's own description of the two maps, not the method's work
    try:
        gap = gap_stats(registration, asdrawn, geo)
    finally:
        mon.resume()
    _LIVE.clear()
    _LIVE[token] = {"offset": declared_offset(registration, geo)}

    px = float(case.pixel_size_um) if case.pixel_size_um else None
    s_x, s_y = (float(v) for v in geo.scale_xy)
    seen = _concurrency(torch)
    info = {
        "version": version,
        "torch": str(torch.__version__),
        "device": device,
        "config": config,
        "fluo": bool(fluo),
        "run_token": token,
        "workers_seen": seen.get("torch_num_threads"),
        **seen,
        "S_x": s_x,
        "S_y": s_y,
        "field_shape": list(field.shape),
        "canvas_hw": list(geo.canvas_hw),
        "postprocessing_params": post,
        "convert_s": convert_s,
        "register_call_s": call_s,
        "stage_s": stage_times(log_text),
        "grey_levels": {"reference": ref_meta["levels"], "moving": mov_meta["levels"]},
        "gap": gap,
        "implementation": {
            "package": "deeperhistreg",
            "source": str(Path(getattr(deeperhistreg, "__file__", "?")).parent),
            "method": METHOD,
            "config_factory": config,
            "paper_resolution_px": CONFIGS[config],
            "initial_resample_ratio": geo.k,
            "working_um_per_px": None if px is None else px * (s_x + s_y) / 2.0,
            "departures_from_paper_text": list(DEPARTURES),
            "registration": "deeperhistreg.run_registration",
            "roles": "source = the moving slide, target = the reference (native)",
            "parameters": recorded,
            "changed_keys": changed,
            "point_warp": "dhr_utils.warping.warp_landmarks (deeperhistreg): a cubic-"
            "spline lookup of the saved field",
            "writer": WRITER,
            "benchmark_side": [
                "each slide's nuclear channel written as a tiled TIFF with one top-level "
                "page, same dtype and values, uncompressed: the package does not read "
                "OME-TIFF (issue #4) and its TIFF loader counts top-level pages as "
                f"pyramid levels. Clock paused; convert_s = {convert_s:.1f}",
                "parameters: the device keys and save_final_images (changed_keys)"
                + (
                    "; pad_value = 0 and flip_intensity = False, configured by the "
                    "benchmark for fluorescence"
                    if fluo
                    else ""
                ),
                "deeperhistreg: the full-resolution coordinate chain around the "
                "package's warp_landmarks (issue #12: a user's function the maintainer "
                "endorsed; the package has no caller for warp_landmarks). The chain "
                "leaves out the half-pixel terms of the two resamplings: "
                f"chain_halfpixel_px_p95 = {gap['chain_halfpixel_px_p95']:.3f}",
                "deeperhistreg_asdrawn: the writer's arithmetic re-stated so the image "
                "on disk can be scored: full-resolution centre pads, the field's index "
                "image read at X / S - 0.5 with linear interpolation and edge copying, "
                "times S = canvas / field size",
                "the writer run once per channel of the moving slide; its per-channel "
                "files stacked into one OME-TIFF, values unchanged",
                "stdout captured around run_registration (the package prints its "
                "exception there and continues); a new case_name per call",
            ],
        },
    }
    state = {
        "case_id": case.case_id,
        "case": case,
        "work": work,
        "params": recorded,
        "field": field_p,
        "target": ref_p,
        "source": mov_p,
        "source_channel": _nuclear(case, "mov")[1],
        "asdrawn": asdrawn,
        "ref_hw": geo.ref_hw,
        "mov_hw": geo.mov_hw,
        "pipeline": pipeline,
        "fluo": bool(fluo),
        "native": None,
    }
    yield VARIANTS[0], PointMap(registration, DIRECTION), dict(info)
    yield VARIANTS[1], PointMap(asdrawn, DIRECTION), {**info, "_state": state}


# ── the writer ────────────────────────────────────────────────────────────────
def _first_page(path):
    """``(array-like (H, W), dtype, close)``: the first top-level page of a TIFF, read
    lazily. libvips writes a pyramid's reduced layers as further pages, never read."""
    import tifffile
    import zarr

    tif = tifffile.TiffFile(str(path))
    try:
        page = tif.pages[0]
        arr = zarr.open(page.aszarr(), mode="r")
        if arr.ndim != 2:
            raise RuntimeError(f"{path}: expected one 2-D plane, got {arr.shape}")
    except Exception:
        tif.close()
        raise
    return arr, np.dtype(page.dtype), tif.close


def write_native(state, case_id, dest_dir):
    """The package's own writer on every channel of the moving slide, stacked.

    ``apply_deformation_pyvips`` with what ``save_final`` passes
    (``full_resolution.py:271-281``): the loader and saver classes its parameters name,
    ``level = final_level``, ``pad_value``, ``save_source_only=True``,
    ``to_template_shape=True``. Its files are single-channel pyramidal TIFFs; their first
    pages are copied, value for value, into ``registered.ome.tif``.
    """
    from stare.slide_io import open_lazy

    from .. import imageio

    case, work, params = state["case"], Path(state["work"]), state["params"]
    pipeline = state["pipeline"]
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    loading, saving = params["loading_params"], params["saving_params"]
    loader = pipeline.loader_mapper[loading["loader"]]
    saver = pipeline.saver_mapper[saving["final_saver"]]
    src_path, nuclear = _nuclear(case, "mov")
    arr, _, close = open_lazy(src_path)
    try:
        n_channels = int(arr.shape[0])
    finally:
        close()
    warped = []
    for c in range(n_channels):
        if c == nuclear:
            source = Path(state["source"])
        else:
            source = work / f"moving_c{c}.tif"
            single_page_tiff(src_path, c, source)
        target = dest / f"warped_c{c}{saving['final_extension']}"
        pipeline.adf.apply_deformation_pyvips(
            str(source),
            str(state["target"]),
            str(target),
            str(state["field"]),
            loader=loader,
            saver=saver,
            level=loading["final_level"],
            pad_value=loading["pad_value"],
            save_source_only=True,
            to_template_shape=True,
        )
        if c != nuclear:
            source.unlink(missing_ok=True)
        warped.append(target)

    h, w = state["ref_hw"]
    opened = [_first_page(p) for p in warped]
    try:
        dtypes = {str(d) for _, d, _ in opened}
        shapes = {tuple(int(v) for v in a.shape) for a, _, _ in opened}
        if shapes != {(h, w)}:
            raise RuntimeError(
                f"the writer's images are {sorted(shapes)}, the reference slide is "
                f"{(h, w)}: the declared origin (0, 0) would not hold"
            )
        if len(dtypes) != 1:
            raise RuntimeError(
                f"the writer's channels differ in dtype: {sorted(dtypes)}"
            )
        dtype = opened[0][1]

        def tiles():
            for a, _, _ in opened:
                for y0 in range(0, h, TILE):
                    for x0 in range(0, w, TILE):
                        yield np.asarray(a[y0 : y0 + TILE, x0 : x0 + TILE], dtype=dtype)

        names = _channel_names(case, n_channels)
        out = dest / "registered.ome.tif"
        imageio.write_channel_tiles(
            out, (h, w), dtype, tiles(), TILE, names, case.pixel_size_um
        )
    finally:
        for _, _, done in opened:
            done()
    for p in warped:
        p.unlink(missing_ok=True)
    pad_value = loading["pad_value"]
    native = {
        "files": [str(out)],
        "format": "OME-TIFF: the first page of each of the writer's single-channel "
        "pyramidal TIFFs, stacked, values unchanged",
        "writer": WRITER,
        "fill": f"{pad_value:g} counts inside the padded canvas (the writer's pad_value), "
        "0 beyond it",
        "origin_xy": [0.0, 0.0],
        "valid_box": footprint_box(state["asdrawn"], (h, w), state["mov_hw"]),
        "channels": names,
        "dtype": str(dtype),
        "nuclear_channel": int(nuclear),
        "frame": "the reference slide's grid: the writer crops the padded canvas back "
        "by the reference's own pads",
    }
    state["native"] = native
    return native


def write_stage(state, case_id, variant, dest_dir):
    """The estimated-field row has no image of its own: the package writes one image, and
    that row is checked against it with its declared offset."""
    return dict(state["native"]) if state.get("native") else None


def SELFCHECK_OFFSET(info):
    """``{variant: f(xy)}``: the estimated-field row's map minus the written image, as
    declared (``declared_offset``). The as-drawn row declares nothing: it must follow the
    image as it is."""
    live = _LIVE.get((info or {}).get("run_token"))
    return {VARIANTS[0]: live["offset"]} if live else {}
