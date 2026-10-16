"""VALIS 1.2.0 (MathOnco/valis, ``pip install valis-wsi==1.2.0``) as shipped, one registrar
per patient. Run it with ``--label valis120``.

One registration yields three results per round:

``valis_rigid``   the rigid stage
``valis``         ``Valis.register()`` with its defaults: rigid 512 px, non-rigid 2048 px
                  (``registration.py:76-79``)
``valis_micro``   after ``Valis.register_micro()`` with NO argument (4096 px, ``:80``)

**The call** is upstream's default one and nothing else:
``registration.Valis(src_dir, dst_dir, img_list=<the patient's slides>,
reference_img_f=<the reference>)``. ``align_to_reference`` stays at its default ``False``
(``:1776``); the reference slide is the one argument the benchmark sets. With a reference
given the crop defaults to ``"reference"`` (``:2142-2146``).

**``hires=1``, label ``valis120_hires``** (the primary row of the family ``valis120``,
``variants.PRIMARY_VALIS120``): upstream's high-resolution example, MathOnco/valis
v1.2.0 ``docs/examples.rst:122-149`` ``[PARTIAL: lines 100-175 read]``: ``micro_rigid_registrar_cls=MicroRigidRegistrar`` in the constructor, then
``register_micro(max_non_rigid_registration_dim_px=micro_reg_size)`` with
``micro_reg_size = np.floor(min over slides of the longest side * 0.25).astype(int)``
computed as lines 141-146 do. Its three results are named ``valis120_hires_rigid``,
``valis120_hires_nonrigid`` and ``valis120_hires`` (``VARIANT_LABELS``); the label and the
option must come together, or the run is refused.

**Inputs.** The patient's ORIGINAL multichannel slides (``case.ref_image``, each
``case.mov_image``), hard-linked under their own names into one folder and read by VALIS's
own readers. VALIS picks the channel it registers on by name (``"dapi"``, a fuzzy match;
``slide_io.py:1074-1099``); the index it picked is recorded per slide.

**Points.** ``REF_TO_MOV``: reference-grid pixel of the written image -> moving slide
pixel, through ``valis.warp_tools.warp_xy_inv`` with the slide's own ``M`` and ``bk_dxdy``
(``methods/_valis_points.py`` has the derivation and why it is not
``Slide.warp_xy_from_to``). ``--opt points=from_to`` scores VALIS's own point function
instead; ``--opt crop_round=<ceil|trunc|round>`` states the writer's rounding of the crop
origin instead of measuring it.

**Images, and when.** ``register_micro`` rewrites every field (``:4989-4990``,
``:5025-5026``), so each result's image is written at ITS yield, with the clock and the
memory peak paused, by ``registrar.warp_and_save_slides(dst, crop="reference")``
(``:5107-5201``): ``valis`` before the micro pass runs, ``valis_micro`` after it.
``valis_rigid`` gets an image (the same call with ``non_rigid=False``) only for the
selfcheck (``opts["stage_images"]``). One call writes the whole patient, so a round's
``write_s`` is the call's time divided by the slides written. The image has the reference
slide's size and is on its grid (``:942-956``): no ``origin_xy``. Pixels the moving slide
does not cover are filled with the slide's background colour, the value of its darkest
pixel, not zero (``:637-643``, ``:984-999``; ``warp_tools.py:1139-1144``), so the native
record carries ``valid_box``.

**Known defects, reported as measured.** On its default non-tiled path 1.2.0 prepares
every slide's image for the micro pass with the LAST slide's field: ``dxdy`` is set inside
the per-slide loop (``:3944-3968``) and used by the later normalisation loop
(``:4045-4062``). Its field getter compares ``shape[1:2]``, i.e. ``(H,)``, with ``(H, W)``
(``:675``), so on a non-square canvas an already full-size field is padded again. The
writer and the point path read the same field, so the selfcheck cannot see either;
``info`` records the last slide of ``slide_dict``, whether the tiler engaged and the
non-rigid bounding box.

**CPUs.** VALIS 1.2.0 has no core-count option: its pools take the affinity mask minus one
(``valtils.py:219-227``, used at ``serial_rigid.py:622`` and four other places). The
harness sets nothing; ``info["workers_seen"]`` is what VALIS saw.

Source: PyPI sdist ``valis-wsi==1.2.0``, read, never executed here ``[PARTIAL: see
methods/_valis_points.py]``. A failure inside ``register()`` is swallowed by VALIS, which
prints it, kills its JVM and returns ``(None, None, None)`` (``:4773-4779``): one registrar
per Python process.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import _valis_points as vp
from . import require_fluorescence

SCOPE = "group"
VARIANTS = ("valis_rigid", "valis", "valis_micro")
HIRES_LABEL = "valis120_hires"
VARIANT_LABELS = {
    HIRES_LABEL: (
        "valis120_hires_rigid",
        "valis120_hires_nonrigid",
        "valis120_hires",
    )
}
SELFCHECK_SUBPROCESS = True  # VALIS kills its JVM, which cannot restart in a process
VERSION = "1.2.0"
# the `--opt` keys this adapter reads (methods.known_options): any other is refused
OPTIONS = ("hires", "points", "crop_round", "any_version", "jvm_heap_gb")
MICRO_REG_FRACTION = 0.25  # docs/examples.rst:134 at v1.2.0
CHANNEL = "dapi"  # registration.DEFAULT_FLOURESCENCE_PROCESSING_ARGS
WRITER = "valis.registration.Valis.warp_and_save_slides"
FILL = "Slide.bg_color (the value of the slide's darkest pixel), not zero"


def _error(exc):
    return f"{type(exc).__name__}: {exc}"


def _jsonable(v):
    if isinstance(v, type):
        return f"{v.__module__}.{v.__qualname__}"
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _channel_used(slide):
    """The channel VALIS's default fluorescence processor reads on this slide."""
    try:
        idx = int(slide.reader.get_channel_index(CHANNEL))
        names = list(slide.reader.metadata.channel_names or [])
        return {"index": idx, "name": str(names[idx]) if idx < len(names) else None}
    except Exception as exc:
        return {"error": _error(exc)}


def _jvm_cache():
    """scyjava's cache moved off a read-only $HOME, with the pipeline's own helper."""
    utils = Path(__file__).resolve().parents[3] / "bin" / "utils"
    try:
        if str(utils) not in sys.path:
            sys.path.insert(0, str(utils))
        from jvm_cache import point_jvm_cache_off_readonly_home

        return point_jvm_cache_off_readonly_home()
    except Exception as exc:
        return _error(exc)


def _write(reg, slides, variant, opts, non_rigid, staging):
    """VALIS's own writer for one result: ``{case_id: native record}``.

    A failure here does not undo the registration: the record says what went wrong."""
    dst = Path(staging) / variant
    kwargs = {"crop": vp.CROP} if non_rigid else {"non_rigid": False, "crop": vp.CROP}
    t0 = time.perf_counter()
    try:
        reg.warp_and_save_slides(str(dst), **kwargs)
    except Exception as exc:
        return {cid: {"error": _error(exc)} for cid in slides}
    elapsed = time.perf_counter() - t0
    written = max(1, len(list(dst.glob("*.ome.tiff"))))
    native_dir = opts.get("native_dir")
    out = {}
    for cid, slide in slides.items():
        src = dst / f"{slide.name}.ome.tiff"
        if not src.exists():
            out[cid] = {"error": f"{WRITER} wrote no {src.name}"}
            continue
        path = src
        if native_dir is not None:
            home = Path(native_dir(cid, variant))
            home.mkdir(parents=True, exist_ok=True)
            path = home / src.name
            shutil.move(str(src), str(path))
        call = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        out[cid] = {
            "files": [str(path)],
            "format": "ome-tiff",
            "writer": f"{WRITER}(dst, {call})",
            "write_s": elapsed / written,
            "write_s_patient": elapsed,
            "fill": FILL,
            **vp.describe_file(path),
        }
    return out


def _stage_items(cases, variant, snaps, natives, warp_tools, info, points):
    """One result for every round, ready to yield: ``(case_id, variant, PointMap | None,
    info)``. Built while the clock is paused: the valid box goes through the point path."""
    items = []
    for case in cases:
        cid = case.case_id
        snap = snaps.get(cid)
        if isinstance(snap, str) or snap is None:
            items.append((cid, variant, None, {**info, "error": snap or "no slide"}))
            continue
        pmap = vp.point_map(snap, warp_tools, points)
        own = {**info, "stage": vp.stage_info(snap, warp_tools, case.pixel_size_um)}
        native = (natives or {}).get(cid)
        if native is not None:
            native = dict(native)
            own["write_s"] = native.pop("write_s", 0.0)
            if native.get("files"):
                # only the map's own direction: where it lands inside the moving slide
                native["valid_box"] = vp.footprint_box(
                    vp.image_map(snap, warp_tools).fn,
                    snap["ref"]["src_rc"],
                    snap["src_rc"],
                )
            own["native"] = native
        items.append((cid, variant, pmap, own))
    return items


def register_group(cases, work, opts):
    work = Path(work)
    hires = vp.truthy(opts.get("hires", 0))
    if "label" in opts and (opts["label"] == HIRES_LABEL) != hires:
        raise ValueError(
            f"the label {HIRES_LABEL} and the option hires=1 name the same run and must "
            f"come together (label {opts['label']!r}, hires={int(hires)})"
        )
    points = str(opts.get("points", "image"))
    crop_round = str(opts.get("crop_round", "probe"))
    if points not in vp.POINTS or crop_round not in vp.ROUNDINGS:
        raise ValueError(
            f"points={points!r} (one of {vp.POINTS}), "
            f"crop_round={crop_round!r} (one of {vp.ROUNDINGS})"
        )
    stage_images = vp.truthy(opts.get("stage_images", 0))
    mon = vp.monitor_of(opts)
    for case in cases:
        require_fluorescence(case, "valis")  # before valis is imported

    # matplotlib (imported by valis) builds a font cache under $HOME, read-only on a node
    os.environ.setdefault("MPLCONFIGDIR", str(work / "mplconfig"))
    import valis
    from valis import registration, valtils, warp_tools

    version = str(getattr(valis, "__version__", "?"))
    if version != VERSION and not vp.truthy(opts.get("any_version", 0)):
        raise RuntimeError(
            f"this adapter is VALIS {VERSION} as shipped; the installed valis is {version}"
        )
    jvm_cache = _jvm_cache()
    # The heap. VALIS starts its JVM with 10 GB whatever it is asked:
    # registration.init_jvm(jar=None, mem_gb=10) calls slide_io.init_jvm(jar=None,
    # mem_gb=10) with the literal (1.2.0 registration.py:132-135 `[PARTIAL: those
    # lines and slide_io.py:231-260]`). A Java heap error would then repeat on every
    # rung of the ladder. So the JVM is started here, through VALIS's own
    # slide_io.init_jvm, with the heap the pipeline's module asks for at this rung;
    # VALIS's later init_jvm calls find it started and do nothing.
    heap = vp.jvm_heap_gb(opts)
    if heap:
        from valis import slide_io

        slide_io.init_jvm(mem_gb=heap)

    ref_path, moving = vp.patient_slides(cases)
    placed, how, place_s = vp.place_slides_unbilled(
        [ref_path, *moving.values()], work / "slides", opts
    )
    kwargs = {
        "img_list": [placed[p] for p in (ref_path, *moving.values())],
        "reference_img_f": placed[ref_path],
    }
    if hires:
        from valis.micro_rigid_registrar import MicroRigidRegistrar

        kwargs["micro_rigid_registrar_cls"] = MicroRigidRegistrar
    info = {
        "version": version,
        "source": str(Path(valis.__file__).parent),
        "hires": hires,
        "points": points,
        "workers_seen": int(valtils.get_ncpus_available()),
        "constructor": {k: _jsonable(v) for k, v in kwargs.items()},
        "slides_placed": how,
        "slides_source": vp.slides_source(placed),
        "place_s": place_s,  # the benchmark's own, clock paused: in no cost
        "jvm_cache": jvm_cache,
        "jvm_heap_gb": heap,
        "jvm_heap": f"-Xmx{heap}G, set by the benchmark through valis.slide_io."
        "init_jvm before VALIS starts the JVM (min(32 + 16 x rung, GiB - 4), the "
        "pipeline module's formula)"
        if heap
        else "VALIS's own 10 GB (no memory request is known: not on the ladder)",
        "implementation": {
            "package": "valis-wsi",
            "source": str(Path(valis.__file__).parent),
            "registration": "valis.registration.Valis(src, dst, img_list=..., "
            "reference_img_f=...).register(), then .register_micro()",
            "parameters": "upstream defaults"
            + (", upstream's high-resolution example" if hires else ""),
            "changed_from_defaults": {
                "reference_img_f": "the benchmark's reference slide",
                **(
                    {
                        "JVM heap": f"-Xmx{heap}G instead of VALIS's fixed 10 GB "
                        "(slide_io.init_jvm called first; not a registration "
                        "parameter)"
                    }
                    if heap
                    else {}
                ),
                **(
                    {
                        "micro_rigid_registrar_cls": "MicroRigidRegistrar",
                        "register_micro.max_non_rigid_registration_dim_px": "floor("
                        f"{MICRO_REG_FRACTION} x the smallest slide's longest side)",
                    }
                    if hires
                    else {}
                ),
            },
            "point_warp": "valis.warp_tools.warp_xy_inv on the writer's crop"
            if points == "image"
            else "valis.warp_tools.warp_xy_from_to as Slide.warp_xy_from_to calls it",
            "benchmark_side": [
                "the patient's original multichannel slides hard-linked (or copied) "
                "into one folder under their own names (under their parent folder's name in "
                "front where two slides share a basename, `info.slides_source`) and "
                "listed through img_list",
                "the reference slide fixed (reference_img_f)",
                "points: the reference-grid pixel plus the writer's crop origin, then "
                "valis.warp_tools.warp_xy_inv with the slide's own M, shapes and "
                "bk_dxdy, all copied at the yield; the rounding of the crop origin is "
                "measured on the installed warp_tools.warp_img",
                "a stored (pyvips) field is loaded whole as NumPy (VALIS's vectorised "
                "branch instead of its per-point loop; the field is the same)",
                "each result's image written by warp_and_save_slides(crop='reference') "
                "at its yield, clock and memory peak paused",
                "MPLCONFIGDIR defaulted to the work directory; scyjava's cache moved "
                "off a read-only $HOME by bin/utils/jvm_cache.py",
            ],
        },
    }

    reg = registration.Valis(str(work / "slides"), str(work / "valis"), **kwargs)
    errors = contextlib.ExitStack()
    caught = errors.enter_context(vp.swallowed_errors(valtils))
    try:
        _, _, error_df = reg.register()
        if error_df is None:
            # the text VALIS printed is in the error, so that a memory failure is
            # classified as one (resources.MEMORY_ERRORS) and goes up the ladder
            raise RuntimeError(
                "Valis.register() returned None: it caught an exception, printed it "
                "with its traceback to stdout, and killed the JVM. It reported: "
                + (vp.reported(caught) or "nothing through valtils.print_warning")
            )
        info["jvm_max_heap_gb"] = vp.jvm_max_heap_gb()

        def stage(non_rigid):
            ref = reg.get_ref_slide()
            snaps, slides = {}, {}
            for case in cases:
                cid = case.case_id
                try:
                    slide = reg.get_slide(placed[moving[cid]])
                    if slide is None or slide.M is None:
                        raise RuntimeError("VALIS has no transform for this slide")
                    args = (slide, ref, warp_tools, non_rigid, crop_round)
                    snaps[cid] = vp.snapshot(*args, points == "from_to")
                    slides[cid] = slide
                except Exception as exc:
                    snaps[cid] = _error(exc)
            return snaps, slides

        mon.pause()
        try:
            rigid, slides = stage(False)
            nonrigid, _ = stage(True)
            ref = reg.get_ref_slide()
            info.update(
                reference=str(ref.name),
                last_slide_in_slide_dict=str(list(reg.slide_dict)[-1]),
                nuclear_channel={
                    str(s.name): _channel_used(s) for s in reg.slide_dict.values()
                },
            )
            staging = work / "native"
            wrote = _write(reg, slides, "valis", opts, True, staging)
            wrote_rigid = (
                _write(reg, slides, "valis_rigid", opts, False, staging)
                if stage_images
                else None
            )
            nr = {
                "tiler_engaged": any(
                    s["stored"] for s in nonrigid.values() if isinstance(s, dict)
                ),
                "non_rigid_bbox": _jsonable(getattr(reg, "_non_rigid_bbox", None)),
            }
            args = (warp_tools, {**info, **nr}, points)
            ready = [
                *_stage_items(cases, "valis_rigid", rigid, wrote_rigid, *args),
                *_stage_items(cases, "valis", nonrigid, wrote, *args),
            ]
        finally:
            mon.resume()
        yield from ready

        micro_kwargs = {}
        if hires:  # docs/examples.rst:141-146 at v1.2.0, statement for statement
            img_dims = np.array(
                [s.slide_dimensions_wh[0] for s in reg.slide_dict.values()]
            )
            min_max_size = np.min([np.max(d) for d in img_dims])
            micro_reg_size = np.floor(min_max_size * MICRO_REG_FRACTION).astype(int)
            micro_kwargs["max_non_rigid_registration_dim_px"] = micro_reg_size
        micro_reg, _ = reg.register_micro(**micro_kwargs)
        # an early return hands back (None, error_df): nothing was registered
        micro_ran = micro_reg is not None

        mon.pause()
        try:
            final, slides = stage(True)
            wrote = _write(reg, slides, "valis_micro", opts, True, staging)
            micro = {
                "micro_ran": micro_ran,
                "register_micro": {k: _jsonable(v) for k, v in micro_kwargs.items()},
                "tiler_engaged": any(
                    s["stored"] for s in final.values() if isinstance(s, dict)
                ),
                "non_rigid_bbox": _jsonable(getattr(reg, "_non_rigid_bbox", None)),
            }
            if not micro_ran:
                micro["problem"] = (
                    "register_micro returned without registering (the first non-rigid "
                    "pass already ran at this size or larger): this row is the "
                    "non-rigid result"
                )
            args = (warp_tools, {**info, **micro}, points)
            ready = _stage_items(cases, "valis_micro", final, wrote, *args)
        finally:
            mon.resume()
        yield from ready
    finally:
        errors.close()
        try:  # a live BioFormats JVM keeps the interpreter from exiting
            registration.kill_jvm()
        except Exception:
            pass
