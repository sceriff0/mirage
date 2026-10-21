"""VALIS's transform as a point mapping: the map of the image its own writer draws.

Shared by ``valis_mirage`` (VALIS 1.0.0 through the pipeline) and ``valis`` (1.2.0). Nothing
here imports VALIS: every function is handed VALIS's own ``warp_tools`` module and its
``Slide`` objects, so the arithmetic that moves a point is VALIS's.

Source read: the PyPI sdists ``valis-wsi==1.0.0`` and ``==1.2.0``, never executed
``[PARTIAL: registration.py -- Slide (crop, displacement getters, warp_img, warp_slide,
warp_and_save_slide, warp_xy, warp_xy_from_to), Valis.__init__, pad_displacement,
non_rigid_register, register, register_micro, get_aligned_slide_shape,
warp_and_save_slides; warp_tools.py -- warp_img, crop_img, get_warp_map, warp_xy_rigid,
get_warp_scaling_factors, _warp_pt_vips, _warp_xy_numpy, warp_xy, warp_xy_inv,
warp_xy_from_to; slide_tools.warp_slide. Not read: serial_rigid, serial_non_rigid, the
registrars, slide_io's readers and writers, libvips itself]``. Line numbers below are
``1.0.0 / 1.2.0``.

**What the writer draws.** ``Slide.warp_slide`` (``registration.py:832-913 / 895-1001``)
hands ``warp_tools.warp_img`` (``warp_tools.py:897-1102 / 1021-1230``) the slide's ``M``, its
backward field ``bk_dxdy`` and a crop box. For output pixel ``P`` of a slide written with
``crop="reference"``:

1. ``X = P + c``: the pixel of the uncropped registered canvas, ``c`` the crop box's
   origin as ``extract_area`` used it;
2. ``a = X + k * bk(X / k)``: the field, scaled up by ``k`` = canvas shape / field shape
   (``get_warp_scaling_factors``, ``warp_tools.py:1836-1939 / 1995-2098``) with libvips'
   interpolator, added to the pixel index;
3. the moving slide is sampled at ``s * M(a * reg / canvas)``, ``s`` = slide shape /
   processed shape.

``warp_tools.warp_xy_inv`` (``:2235-2307 / 2377-2449``, identical in both versions) does
steps 2 and 3 to a point, in the field's own grid, and applies ``M`` (it inverts nothing:
its own ``np.linalg.inv(M)`` is undone by ``warp_xy_rigid``'s). So ``image_map`` is
``P -> warp_xy_inv(P + c)``: reference-grid pixel of the written image -> moving slide
pixel, ``REF_TO_MOV``. Two things differ from the writer and are not corrected: points
read the field through VALIS's cubic spline, the image through libvips' bicubic; and the
writer's rigid step goes through a four-corner projective fit of the same chain
(``warp_tools.py:1150-1158`` at 1.2.0). Both are what the selfcheck measures.

**The crop origin is the writer's, not the reference slide's ``M``.** The box comes from
``Slide.get_crop_xywh`` (``:533-566 / 574-607``): ``-ref.M[:2, 2] * (slide / processed)``,
a real number. 1.2.0 takes ``np.ceil`` of it (``warp_tools.py:1118``); 1.0.0 passes it to
pyvips as it is (``:1097``), which converts it. ``writer_rounding`` asks the
installed ``warp_tools.warp_img`` itself, on an 8 px probe, what it does with a fractional
origin, so no rounding rule is assumed here.

**Why not ``Slide.warp_xy_from_to(reference -> moving)``.** That function
(``:1117-1199 / 1200-1282``; ``ref_to_mov`` below calls ``warp_tools.warp_xy_from_to``
exactly as it does) first places the reference point on the canvas with the reference
slide's own ``M``: ``(P / s - t) * canvas / reg``. The writer places it at ``P + c``. The
two differ by the rounding of ``c`` (up to 1 px) and by ``canvas = ceil(reg * s)``
(``get_aligned_slide_shape``, ``:4414-4446 / 5065-5097``; up to 1 px at the far edge).
In 1.2.0 the reference slide is written unwarped (``:955-974``), so ``image_map`` is the
relation between the two files on disk. In 1.0.0 the reference is itself resampled
through its ``M`` and the same crop, so ``warp_xy_from_to`` relates the two WRITTEN files
and ``image_map`` relates the written moving file to the NATIVE reference grid, which is
where the benchmark's reference nuclei are. ``from_to_minus_writer`` reports the
difference per run; ``--opt points=from_to`` scores VALIS's own function instead.

**Fields are taken at the yield.** In 1.2.0 ``Slide.bk_dxdy`` is a property that pads the
stored array with registrar attributes ``register_micro`` rewrites (``:663-682``,
``:5025-5026``); in 1.0.0 a stored field is a file ``register_micro`` replaces
(``:4352-4361``). ``numpy_field`` materialises what the property returns at that moment.
A stored (pyvips) field is loaded whole: VALIS's pyvips branch warps one point per Python
call (``_warp_pt_vips``) and its NumPy branch is vectorised; the field is the same.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

import numpy as np

from ..mapping import REF_TO_MOV, PointMap

CROP = "reference"
ROUNDINGS = ("probe", "ceil", "trunc", "round")
POINTS = ("image", "from_to")
SLIDE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


class _NullMonitor:
    """Stands in for the driver's monitor when an adapter is called without the driver."""

    def pause(self):
        return self

    def resume(self):
        return self

    def snapshot(self):
        return {}


def monitor_of(opts):
    return opts.get("monitor") or _NullMonitor()


@contextlib.contextmanager
def swallowed_errors(valtils):
    """What VALIS caught, printed and carried on from, for the duration: a list of texts.

    ``Valis.register()`` wraps its whole body in ``except Exception``, hands the
    exception to ``valtils.print_warning`` and returns ``(None, None, None)`` (1.0.0
    ``registration.py:4149-4153``: ``print_warning(e)`` then ``print(traceback...)``;
    1.2.0 ``:4773-4779``: ``print_warning(e, rgb=Fore.RED, traceback_msg=...)``; PyPI
    sdists `[PARTIAL: those lines, and valtils.print_warning]`). So the real cause, a
    Java ``OutOfMemoryError`` or a NumPy memory error included, never reaches the
    caller. ``valtils.print_warning`` is replaced by a wrapper that notes every
    exception (or message with a traceback) it is given and then calls the original
    with the same arguments: nothing VALIS does changes."""
    caught = []
    original = valtils.print_warning

    def print_warning(msg, *args, **kwargs):
        try:
            trace = kwargs.get("traceback_msg")
            if isinstance(msg, BaseException):
                caught.append(f"{type(msg).__name__}: {msg}")
            elif trace:
                caught.append(str(msg))
            if trace:
                caught.append(str(trace)[-2000:])
        except Exception:  # noting it must never be the method's problem
            pass
        return original(msg, *args, **kwargs)

    valtils.print_warning = print_warning
    try:
        yield caught
    finally:
        valtils.print_warning = original


class _Tee:
    """``sys.stdout`` for the duration of ``printed_errors``: everything goes on to the
    real stream; a chunk that names a memory failure is also kept."""

    def __init__(self, stream, kept, most):
        self._stream, self._kept, self._most = stream, kept, most

    def write(self, text):
        try:
            if len(self._kept) < self._most and _names_memory(text):
                self._kept.append(str(text).strip()[-400:])
        except Exception:  # noting it must never be the method's problem
            pass
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _names_memory(text):
    from .. import resources

    return resources.classify_text(text) in ("oom", "gpu_oom")


@contextlib.contextmanager
def printed_errors(most=50):
    """What VALIS PRINTS that names a memory failure, for the duration: a list of texts.

    VALIS 1.0.0 reads BioFormats tiles in threads whose worker does ``try: tile =
    self.slide2image(...) except Exception as e: print(e)`` and then uses ``tile``
    (``slide_io.py:885-900``, as read in the pre-freeze review `[not re-read this
    session]`): a Java ``OutOfMemoryError`` there goes to stdout and what is raised is
    an ``UnboundLocalError``, which ``swallowed_errors`` alone would classify as the
    method's ``exception``. ``sys.stdout`` is replaced by a tee that passes everything
    on and keeps (at most ``most``) chunks that ``resources.classify_text`` reads as a
    memory failure; nothing VALIS does changes."""
    import sys

    kept = []
    original = sys.stdout
    sys.stdout = _Tee(original, kept, most)
    try:
        yield kept
    finally:
        sys.stdout = original


def reported(caught):
    """The texts VALIS printed, short enough for ``run.json: error``."""
    text = " | ".join(str(c).strip() for c in caught if str(c).strip())
    return text[-3000:]


# VALIS's own heap: registration.init_jvm(jar=None, mem_gb=10) passes the literal 10 on
# to slide_io.init_jvm (1.2.0 registration.py:132-135, 1.0.0 registration.py:115-118
# `[PARTIAL: those lines and slide_io.init_jvm]`). The benchmark keeps it at every rung
# (user ruling 2026-10-21): it is how VALIS ships, and a heap scaled with the rung is a
# departure no other method gets. A JVM heap failure is therefore ``jvm_heap``, final.
VALIS_JVM_HEAP_GB = 10


def jvm_heap_gb(opts, host=None):
    """The JVM heap VALIS runs with: ``VALIS_JVM_HEAP_GB`` (10 GiB) at every rung, never
    scaled with the memory. ``--opt jvm_heap_gb=N`` overrides it, a recorded departure.
    ``host`` is accepted and ignored (the heap does not depend on it)."""
    del host
    if opts.get("jvm_heap_gb"):
        return int(opts["jvm_heap_gb"])
    return VALIS_JVM_HEAP_GB


def mark_jvm_heap(exc, *texts):
    """``exc`` with the cause ``jvm_heap`` when it, or what VALIS printed (``texts``),
    says the JVM ran out of its fixed heap and it declares no cause of its own; else
    ``exc`` as it is. ``resources.classify_error`` reads the declared cause first. A
    JPype Java exception refuses new attributes (its setter raises AttributeError for
    any name without a leading underscore), so it comes back wrapped: a RuntimeError
    with its text, the cause, and the Java exception as ``__cause__``."""
    from .. import resources

    text = f"{type(exc).__name__}: {exc}"
    if getattr(exc, "regbench_cause", None) is not None or not resources.names_jvm_heap(
        text, *texts
    ):
        return exc
    try:
        exc.regbench_cause = "jvm_heap"
        return exc
    except (AttributeError, TypeError):
        wrapped = RuntimeError(text)
        wrapped.regbench_cause = "jvm_heap"
        wrapped.__cause__ = exc
        return wrapped


def jvm_max_heap_gb():
    """The live JVM's own ``Runtime.maxMemory()`` in GiB: the heap it REALLY has, which
    is the only way to know that a requested ``-Xmx`` arrived. None without a JVM."""
    try:
        import jpype

        if not jpype.isJVMStarted():
            return None
        runtime = jpype.JClass("java.lang.Runtime").getRuntime()
        return float(runtime.maxMemory()) / 2**30
    except Exception:
        return None


def truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def slide_name(path):
    """A slide's name as VALIS and the pipeline derive it: the basename without its
    extension (``valtils.get_name``; ``bin/register.py:906-916``)."""
    name = os.path.basename(str(path))
    for suffix in SLIDE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def placed_name(path, prefixed):
    """The file name a slide is placed under: its own, or with its parent folder's name
    in front (``HE_29.tif``) where the patient's slides share a basename."""
    p = Path(str(path))
    return f"{p.parent.name}_{p.name}" if prefixed else p.name


def placed_names(paths):
    """``{source: placed file name}`` of a patient's slides. Where two DIFFERENT files
    share a basename (``a/29.tif`` and ``b/29.tif``) EVERY slide
    is placed under its parent folder's name in front, so the names follow one rule.
    Raises ValueError when the placed names still clash (the same folder name over
    different grandparents), because VALIS names a slide by its file name."""
    paths = [str(p) for p in dict.fromkeys(str(p) for p in paths)]
    base = [Path(p).name for p in paths]
    prefixed = len(set(base)) != len(base)
    out = {p: placed_name(p, prefixed) for p in paths}
    seen = {}
    for src, name in out.items():
        other = seen.setdefault(slide_name(name), src)
        if other != src:
            raise ValueError(
                f"{other} and {src} would both be the VALIS slide {slide_name(name)!r}; "
                "VALIS names a slide by its file name"
            )
    return out


def slides_source(placed):
    """``{placed file name: source path}``: where each slide VALIS was given came from
    (recorded in ``run.json``, so a point finds its slide)."""
    return {Path(dst).name: str(src) for src, dst in placed.items()}


def patient_slides(cases):
    """``(reference path, {case_id: moving path})`` of one patient's cases: the ORIGINAL
    multichannel slides. Refuses two reference slides, a moving slide that IS the
    reference (the same file), and slides VALIS would give the same name even after
    ``placed_names``. Two different files of one basename are not refused: they are
    placed apart."""
    refs = {str(c.ref_image) for c in cases}
    if len(refs) != 1:
        raise ValueError(
            f"the cases of patient {cases[0].group!r} name {len(refs)} reference slides "
            f"({sorted(refs)}); VALIS registers one patient to ONE reference"
        )
    ref = refs.pop()
    moving = {c.case_id: str(c.mov_image) for c in cases}
    again = sorted(
        {str(m) for m in moving.values() if list(moving.values()).count(m) > 1}
    )
    if again:
        raise ValueError(
            f"patient {cases[0].group!r}: two cases name the same moving slide "
            f"({', '.join(again)}); VALIS registers each slide once"
        )
    if any(Path(m).resolve() == Path(ref).resolve() for m in moving.values()):
        raise ValueError(
            f"patient {cases[0].group!r}: a moving slide is the reference slide"
        )
    placed_names([ref, *moving.values()])
    return ref, moving


# ── a brightfield (RGB) patient ───────────────────────────────────────────────
# What VALIS does with an RGB slide, read in the PyPI sdists ``valis-wsi==1.0.0`` and
# ``==1.2.0`` ``[PARTIAL: registration.py -- Valis.register's signature (1.0.0 :3977-3982,
# 1.2.0 :4593-4598), its call of create_img_processor_dict (:4102 / :4725),
# create_img_processor_dict (:2428-2503 / :2679-2754), Slide.__init__'s is_rgb and
# img_type (:343-344 / :364-365); slide_io.py -- guess_image_type (:666-684 /
# :1017-1035), BioFormatsSlideReader._check_rgb (:1161-1171 / :1513-1523), the vips
# reader's _check_rgb (1.2.0 :1858-1881), the OME is_rgb rule (1.2.0 :785-800); missing:
# the brightfield processors themselves (1.0.0 ColorfulStandardizer, 1.2.0 OD) and every
# other reader]``. ``register()`` takes ``brightfield_processing_cls`` /
# ``if_processing_cls`` with defaults and ``processor_dict=None``; given no processor
# dict, it assigns each slide the BRIGHTFIELD class when ``slide.img_type`` is
# ``"brightfield"``, which ``guess_image_type`` sets exactly when the reader's
# ``metadata.is_rgb`` is true. For Bio-Formats that is ``rdr.isRGB()``, which Bio-Formats
# defines as more than one sample per pixel (Bio-Formats' own source not read:
# ``[MEMORY]``). A slide written as three separate channels is therefore expected NOT to
# be read as RGB, and to be given the fluorescence processing (its ``dapi`` channel,
# else channel 0). The adapters
# pass no processor (``bin/register.py:516``: ``registrar.register()``), so what VALIS
# chose is recorded per slide (``valis_img_type``), and a brightfield case whose slide
# VALIS read otherwise carries a ``problem``.
RGB_INPUT = (
    "the patient's original RGB slides as they are; VALIS chooses each slide's "
    "processing itself from its reader's is_rgb (an RGB slide: its brightfield class), "
    "no processor is passed"
)


def brightfield(cases):
    """A patient of brightfield (RGB) cases."""
    return any(c.modality == "brightfield" for c in cases)


def modality_record(cases, slides):
    """``(info, problem)`` of a brightfield patient: ``info`` adds ``modality``, sets
    ``nuclear_channel`` to None (no channel is looked for on an RGB slide) and records
    ``valis_img_type`` (VALIS's own reading of each slide); ``problem`` names the slides
    VALIS did not read as brightfield, or is empty. ``({}, "")`` for a fluorescence
    patient: its record is what it was. ``slides`` is ``{name: Slide}``."""
    if not brightfield(cases):
        return {}, ""
    types = {str(n): str(getattr(s, "img_type", "unknown")) for n, s in slides.items()}
    other = sorted(n for n, t in types.items() if t != "brightfield")
    problem = ""
    if other:
        problem = (
            f"VALIS read {', '.join(f'{n} as {types[n]}' for n in other)}, not "
            "brightfield: its reader does not see three samples per pixel, so those "
            "slides were given the fluorescence processing"
        )
    info = {"modality": "brightfield", "nuclear_channel": None, "valis_img_type": types}
    return info, problem


def place_slides(paths, dst_dir):
    """Put the slides in ``dst_dir``: a hard link, or a copy where a link is not
    possible. Never a symlink (the pipeline copies for that reason,
    ``modules/local/register.nf``). Each keeps its own basename, except where the
    slides share one: ``placed_names``. Returns ``({source: placed path}, {name: how})``."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = placed_names(paths)
    placed, how = {}, {}
    for src in paths:
        real = Path(src).resolve()
        dst = dst_dir / names[str(src)]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(real, dst)
            how[dst.name] = "hard link"
        except OSError:
            shutil.copyfile(real, dst)
            how[dst.name] = "copy"
        placed[str(src)] = str(dst)
    return placed, how


def place_slides_unbilled(paths, dst_dir, opts):
    """``place_slides`` with the run's clock and peak paused: putting the slides where
    VALIS reads a folder from is the benchmark's arrangement (a hard link, or a full
    copy of every slide where the folders are on two file systems), not VALIS's work.
    Returns ``(placed, how, place_s)``; ``place_s`` is recorded as ``info["place_s"]``
    and is in no cost."""
    import time

    mon = monitor_of(opts)
    mon.pause()
    t0 = time.perf_counter()
    try:
        placed, how = place_slides(paths, dst_dir)
    finally:
        took = time.perf_counter() - t0
        mon.resume()
    return placed, how, took


# ── fields ────────────────────────────────────────────────────────────────────
def numpy_field(field, warp_tools, copy=True):
    """A VALIS displacement field as a ``(2, H, W)`` array, read NOW.

    ``field`` is what ``Slide.bk_dxdy`` / ``.fwd_dxdy`` returned: None, an array, a list
    of two planes, or a pyvips image of two bands (a stored field), which is loaded with
    VALIS's ``vips2numpy`` and split the way VALIS itself does (``np.array([a[..., 0],
    a[..., 1]])``, ``registration.py:3296-3297 / 3755-3756``)."""
    if field is None:
        return None
    if isinstance(field, (np.ndarray, list, tuple)):
        arr = np.array(field, copy=copy)
    else:
        a = np.asarray(warp_tools.vips2numpy(field))
        arr = np.array([a[..., 0], a[..., 1]])
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(
            f"expected a (2, H, W) displacement field, got shape {arr.shape}"
        )
    return arr


def reference_field(ref_slide, warp_tools):
    """The reference slide's forward field as an array, which must be all zero.

    ``Slide.warp_xy_from_to`` passes it as ``from_fwd_dxdy``. After a tiled run it is a
    stored pyvips image like every other slide's (``registration.py:3709-3721`` at 1.0.0),
    which would send every point through VALIS's per-point loop; as an array it takes the
    vectorised branch. A reference with a non-zero field is not a reference: refused."""
    fwd = numpy_field(getattr(ref_slide, "fwd_dxdy", None), warp_tools)
    if fwd is not None and (float(np.abs(fwd).max()) if fwd.size else 0.0) != 0.0:
        raise ValueError(
            f"the reference slide {getattr(ref_slide, 'name', '?')!r} carries a non-zero "
            f"displacement field (max |d| = {float(np.abs(fwd).max()):.4g} px)"
        )
    return fwd


# ── the writer's crop ─────────────────────────────────────────────────────────
def writer_rounding(raw_xy, warp_tools, how="probe"):
    """The crop origin as VALIS's writer uses it: ``raw_xy`` after whatever the installed
    ``warp_tools.warp_img`` does to a fractional ``bbox_xywh`` origin.

    ``probe`` (the default) measures it: an 8 px ``pyvips.Image.xyz`` (pixel value = its
    own index) goes through ``warp_img`` with no transform and a box whose origin has the
    same fractional part, and the first pixel that comes back says where the crop began.
    ``ceil`` / ``trunc`` / ``round`` state the rule instead (``--opt crop_round=``)."""
    raw = np.asarray(raw_xy, dtype=float).reshape(2)
    if how not in ROUNDINGS:
        raise ValueError(f"crop_round={how!r}: expected one of {ROUNDINGS}")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError(
            f"the reference crop box starts at {raw.tolist()}: outside the registered canvas"
        )
    if how == "ceil":
        return np.ceil(raw)
    if how == "trunc":
        return np.trunc(raw)
    if how == "round":
        return np.rint(raw)
    import pyvips

    base = np.floor(raw)
    frac = raw - base
    pad = 2.0
    probe = pyvips.Image.xyz(8, 8)
    got = warp_tools.warp_img(probe, bbox_xywh=(frac[0] + pad, frac[1] + pad, 2, 2))
    first = np.asarray(got(0, 0), dtype=float).reshape(-1)[:2]
    origin = base + (first - pad)
    if not np.isfinite(origin).all() or np.abs(origin - raw).max() > 1.0:
        raise RuntimeError(
            f"the crop probe put the origin of {raw.tolist()} at {origin.tolist()}"
        )
    return origin


def crop_origin(mov_slide, ref_slide, warp_tools, how="probe"):
    """``(origin_xy, raw_xy)`` of ``crop="reference"`` at level 0, from the slide's own
    ``get_crop_xywh`` called the way ``Slide.warp_slide`` calls it."""
    ref_shape_rc = np.asarray(ref_slide.slide_dimensions_wh[0])[::-1]
    bbox, _ = mov_slide.get_crop_xywh(crop=CROP, out_shape_rc=ref_shape_rc)
    raw = np.asarray(bbox[0:2], dtype=float)
    return writer_rounding(raw, warp_tools, how), raw


# ── one stage of one moving slide ─────────────────────────────────────────────
def _shape(v):
    return tuple(int(x) for x in np.asarray(v).reshape(-1)[:2])


def _geometry(slide):
    """The numbers ``Slide.warp_slide`` and ``Slide.warp_xy_from_to`` read off a slide."""
    return {
        "name": str(getattr(slide, "name", "")),
        "M": np.array(slide.M, dtype=float, copy=True),
        "processed_rc": _shape(slide.processed_img_shape_rc),
        "reg_rc": _shape(slide.reg_img_shape_rc),
        "src_rc": _shape(np.asarray(slide.slide_dimensions_wh[0])[::-1]),
        "aligned_rc": _shape(slide.aligned_slide_shape_rc),
    }


def snapshot(
    mov_slide,
    ref_slide,
    warp_tools,
    non_rigid=True,
    crop_round="probe",
    with_reference=False,
):
    """Everything one stage's map needs, copied out of the live registrar NOW.

    ``non_rigid=False`` is the rigid stage (no field), exactly what the writer's own
    ``non_rigid=False`` draws. ``with_reference`` also loads the reference slide's forward
    field, which only ``from_to_map`` passes on (the writer of a moving slide never reads
    it, and after a tiled run it is as large as a moving slide's)."""
    origin, raw = crop_origin(mov_slide, ref_slide, warp_tools, crop_round)
    snap = _geometry(mov_slide)
    snap.update(
        bk_dxdy=numpy_field(mov_slide.bk_dxdy, warp_tools) if non_rigid else None,
        stored=bool(getattr(mov_slide, "stored_dxdy", False)),
        origin_xy=origin,
        origin_raw_xy=raw,
        crop_round=crop_round,
        ref=_geometry(ref_slide),
    )
    snap["ref"]["fwd_dxdy"] = (
        reference_field(ref_slide, warp_tools) if non_rigid and with_reference else None
    )
    return snap


def image_map(snap, warp_tools):
    """The map of the image VALIS's writer draws for this stage (module docstring):
    reference-grid pixel -> moving slide pixel."""
    origin = np.asarray(snap["origin_xy"], dtype=float)
    kwargs = {
        "M": snap["M"],
        "transformation_src_shape_rc": snap["processed_rc"],
        "transformation_dst_shape_rc": snap["reg_rc"],
        "src_shape_rc": snap["src_rc"],
        "dst_shape_rc": snap["aligned_rc"],
        "bk_dxdy": snap["bk_dxdy"],
    }

    def fn(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not len(xy):
            return xy.copy()
        out = warp_tools.warp_xy_inv(xy + origin, **kwargs)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    return PointMap(fn, REF_TO_MOV)


def ref_to_mov(ref_slide, mov_slide, bk_dxdy, warp_tools, mov_M=None):
    """VALIS's own reference -> moving point function, called exactly as
    ``Slide.warp_xy_from_to`` calls ``warp_tools.warp_xy_from_to`` (twelve keywords and the
    points; ``registration.py:1182-1196 / 1266-1280``), with the stage's field handed in.

    ``bk_dxdy=None`` is the rigid stage (``non_rigid=False`` there passes neither field).
    This is NOT the map of the written image (module docstring); it is kept as the
    ``points=from_to`` option and as the record of what VALIS's own function gives."""
    ref, mov = _geometry(ref_slide), _geometry(mov_slide)
    kwargs = {
        "from_M": ref["M"],
        "from_transformation_src_shape_rc": ref["processed_rc"],
        "from_transformation_dst_shape_rc": ref["reg_rc"],
        "from_src_shape_rc": ref["src_rc"],
        "from_dst_shape_rc": ref["aligned_rc"],
        "from_fwd_dxdy": (
            None if bk_dxdy is None else reference_field(ref_slide, warp_tools)
        ),
        "to_M": mov["M"] if mov_M is None else np.array(mov_M, dtype=float, copy=True),
        "to_transformation_src_shape_rc": mov["processed_rc"],
        "to_transformation_dst_shape_rc": mov["reg_rc"],
        "to_src_shape_rc": mov["src_rc"],
        "to_dst_shape_rc": ref["aligned_rc"],
        "to_bk_dxdy": bk_dxdy,
    }
    return _from_to(kwargs, warp_tools)


def _from_to(kwargs, warp_tools):
    def fn(xy):
        xy = np.array(xy, dtype=float, copy=True).reshape(-1, 2)
        if not len(xy):
            return xy
        out = warp_tools.warp_xy_from_to(xy, **kwargs)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    return PointMap(fn, REF_TO_MOV)


def from_to_map(snap, warp_tools):
    """``ref_to_mov`` from a snapshot taken ``with_reference`` (the ``points=from_to``
    option)."""
    ref = snap["ref"]
    return _from_to(
        {
            "from_M": ref["M"],
            "from_transformation_src_shape_rc": ref["processed_rc"],
            "from_transformation_dst_shape_rc": ref["reg_rc"],
            "from_src_shape_rc": ref["src_rc"],
            "from_dst_shape_rc": ref["aligned_rc"],
            "from_fwd_dxdy": None if snap["bk_dxdy"] is None else ref["fwd_dxdy"],
            "to_M": snap["M"],
            "to_transformation_src_shape_rc": snap["processed_rc"],
            "to_transformation_dst_shape_rc": snap["reg_rc"],
            "to_src_shape_rc": snap["src_rc"],
            "to_dst_shape_rc": ref["aligned_rc"],
            "to_bk_dxdy": snap["bk_dxdy"],
        },
        warp_tools,
    )


def point_map(snap, warp_tools, points="image"):
    if points not in POINTS:
        raise ValueError(f"points={points!r}: expected one of {POINTS}")
    return (image_map if points == "image" else from_to_map)(snap, warp_tools)


def from_to_minus_writer(snap, warp_tools):
    """Where VALIS's own point function puts a reference pixel on the registered canvas,
    minus where its writer crops that pixel from: ``{at_origin, at_far_corner}`` in canvas
    pixels (x, y). Zero would mean the two agree; they differ by the rounding of the crop
    origin and by the ceiling in the canvas size (module docstring)."""
    ref = snap["ref"]
    h, w = ref["src_rc"]
    corners = np.array([[0.0, 0.0], [w - 1.0, h - 1.0]])
    on_canvas = warp_tools.warp_xy(
        corners.copy(),
        M=ref["M"],
        transformation_src_shape_rc=ref["processed_rc"],
        transformation_dst_shape_rc=ref["reg_rc"],
        src_shape_rc=ref["src_rc"],
        dst_shape_rc=ref["aligned_rc"],
    )
    diff = np.asarray(on_canvas, dtype=float).reshape(2, 2) - (
        corners + np.asarray(snap["origin_xy"], dtype=float)
    )
    return {
        "at_origin": [float(v) for v in diff[0]],
        "at_far_corner": [float(v) for v in diff[1]],
    }


# ── what a stage says about itself ────────────────────────────────────────────
def footprint_box(fn, ref_hw, mov_hw, max_side=256):
    """``[x0, y0, x1, y1]`` on the reference grid outside which the written image is fill:
    the bounding box of the lattice points the map sends INSIDE the moving slide, grown by
    one lattice step. Only the map's own direction is used. None when no point lands
    inside."""
    h, w = (int(v) for v in ref_hw)
    step = max(1, -(-max(h, w) // max_side))
    ys = np.unique(np.append(np.arange(0, h, step), h - 1)).astype(float)
    xs = np.unique(np.append(np.arange(0, w, step), w - 1)).astype(float)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    out = np.asarray(fn(grid), dtype=float).reshape(-1, 2)
    mh, mw = (int(v) for v in mov_hw)
    with np.errstate(invalid="ignore"):
        inside = (
            (out[:, 0] >= 0)
            & (out[:, 0] <= mw - 1)
            & (out[:, 1] >= 0)
            & (out[:, 1] <= mh - 1)
        )
    if not inside.any():
        return None
    pts = grid[inside]
    x0, y0 = np.maximum(pts.min(axis=0) - step, 0)
    x1, y1 = np.minimum(pts.max(axis=0) + step + 1, [w, h])
    return [int(x0), int(y0), int(x1), int(y1)]


def um_per_px(snap, pixel_size_um):
    """The size of one pixel of the image each stage was computed on, in um: the
    processed image (rigid) and the field's own grid (non-rigid), from the shapes."""
    if not pixel_size_um:
        return {"rigid": None, "field": None}
    px = float(pixel_size_um)
    ref = snap["ref"]
    rigid = px * max(ref["src_rc"]) / max(ref["processed_rc"])
    field = None
    if snap["bk_dxdy"] is not None:
        field = px * max(snap["aligned_rc"]) / max(snap["bk_dxdy"].shape[1:])
    return {"rigid": float(rigid), "field": None if field is None else float(field)}


def stage_info(snap, warp_tools, pixel_size_um=None):
    """The JSON-able record of one stage's map."""
    field = snap["bk_dxdy"]
    return {
        "crop": CROP,
        "crop_origin_xy": [float(v) for v in snap["origin_xy"]],
        "crop_origin_raw_xy": [float(v) for v in snap["origin_raw_xy"]],
        "crop_round": snap["crop_round"],
        "field_shape": None if field is None else [int(v) for v in field.shape[1:]],
        "field_stored": bool(snap["stored"]),
        "field_max_abs_px": None if field is None else float(np.abs(field).max()),
        "um_per_px": um_per_px(snap, pixel_size_um),
        "valis_from_to_minus_writer_px": from_to_minus_writer(snap, warp_tools),
    }


def describe_file(path):
    """``channels`` and ``dtype`` of a written OME-TIFF, or nothing when it cannot be
    read here (the row does not depend on it)."""
    try:
        from stare.slide_io import open_lazy

        arr, dtype, close = open_lazy(path)
        try:
            return {"channels": int(arr.shape[0]), "dtype": str(np.dtype(dtype))}
        finally:
            close()
    except Exception as exc:
        return {"unread": f"{type(exc).__name__}: {exc}"}
