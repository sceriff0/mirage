"""VALIS as the pipeline runs it: ``bin/register.py::valis_registration``, one call per
patient, on VALIS 1.0.0.

``valis_mirage_rigid``     the rigid stage, AFTER the pipeline's micro-rigid refinement
                           (``MicroRigidRegistrar`` runs inside ``Valis.register()`` and
                           rewrites ``Slide.M``)
``valis_mirage_nonrigid``  rigid + the first non-rigid pass, as it stood when the pipeline
                           called ``register_micro``
``valis_mirage``           the final transform: what the pipeline's writer drew

**The call** is the shipped one: ``ARGS`` below, every value the default of
``nextflow.config`` (``memory_mode`` ``high``, ``reg_micro_reg_fraction`` 0.125,
``reg_max_image_dim`` 4000, ``reg_micro_reg`` 2) as ``modules/local/register.nf`` passes
them, plus the five that belong to a run (``RUN_ARGS``). The pipeline's own settings are
inside that function and are not repeated here: ``align_to_reference=True``,
``crop="reference"``, SuperPoint + SuperGlue, ``OpticalFlowWarper``, the micro-rigid
registrar (``bin/utils/valis_config.py:273-286``), and
``register_micro(max_non_rigid_registration_dim_px=floor(0.125 x the smallest slide's
longest side), reference_img_f=<reference>, align_to_reference=True, tile_wh=2048)``
(``bin/register.py:785-799``).

``n_cpus`` is the allocation (``opts["workers"]``); the pipeline's ``bound_cpu_count``
rebinds ``multiprocessing.cpu_count`` to it, so VALIS 1.0.0's pools, each sized
``cpu_count() - 1`` (``serial_rigid.py:578, 643`` and four other places in the 1.0.0
sdist), run on one thread fewer. ``jvm_heap_gb`` follows the module's own formula,
``min(32 + 16 * attempt, memory in GiB - 4)``, with the ladder's rung as the attempt;
outside a scheduler it is left to the pipeline's estimate. ``stage_checkpoint_dir=None``
differs from the arm, which writes checkpoints at ``reg_qc=2``: it only skips files.

**Inputs.** The patient's ORIGINAL multichannel slides, hard-linked into ``preprocessed/``
under their own names, as the module stages them. VALIS writes its results into that same
folder (``dst_dir/<name of src_dir>``), as it does in the pipeline.

**Points.** ``REF_TO_MOV``, the map of the image the pipeline's writer drew
(``methods/_valis_points.py``). The three stages are captured inside the call by wrappers
that change no argument and no result:

- ``Valis.register_micro``: before it runs, each moving slide's ``M`` and ``bk_dxdy`` are
  copied (the rigid and non-rigid stages), and for the selfcheck their images are written
  with the unwrapped ``Slide.warp_and_save_slide`` while the JVM is alive. The capture is
  in its own ``try``; the original is ALWAYS called, because the pipeline calls it inside
  a ``try/except Exception`` that would otherwise swallow a failed capture and silently
  skip the micro pass.
- ``Slide.warp_and_save_slide``: the clock and the memory peak are paused, the slide's
  final ``M`` and ``bk_dxdy`` are copied at the moment its writer reads them, the original
  runs, the path and the time are recorded.
- ``Valis.register``: after it returns, the slides whose forward field is missing are
  listed (the condition of the pipeline's repair, ``bin/register.py:540-562``) and the
  cost so far is noted.

The function kills the JVM before it returns (``bin/register.py:1071-1074``), so one call
per Python process; the point path needs no JVM.

**Images.** The final image is the file the pipeline's function wrote
(``<slide>_registered.ome.tiff``, ``crop="reference"``, bicubic), moved to the result's
``native/`` folder. It has the reference slide's size and sits on its grid: no
``origin_xy``. Pixels the moving slide does not cover are filled with the slide's
background colour when the writer is given the slide's own path, else with zero (VALIS
1.0.0 ``registration.py:900-903``; ``warp_tools.py`` ``warp_img``), so the record carries
``valid_box``. A slide the function did not write fails its round, as the pipeline's
module fails the task.

Every result of one call carries the cost of the WHOLE call (the function is one piece);
``info["cost_after_register"]`` is the bill when ``Valis.register()`` returned, before the
micro pass.

VALIS source: PyPI sdist ``valis-wsi==1.0.0``, read, never executed here ``[PARTIAL: see
methods/_valis_points.py]``. Pipeline source: ``bin/register.py`` ``[FULL]``,
``bin/utils/valis_config.py`` ``[FULL]``, ``modules/local/register.nf`` ``[PARTIAL: the
script block]``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import multiprocessing
import shutil
import sys
import time
import traceback
from pathlib import Path

from .. import resources
from . import _valis_points as vp

SCOPE = "group"
VARIANTS = ("valis_mirage_rigid", "valis_mirage_nonrigid", "valis_mirage")
RIGID, NONRIGID, FINAL = VARIANTS
SELFCHECK_SUBPROCESS = True  # valis_registration kills the JVM, which cannot restart
VERSION = "1.0.0"
# The shipped call (module docstring). tests/regbench/test_valis_adapters.py checks that
# ARGS and RUN_ARGS together are exactly the parameters of valis_registration.
ARGS = {
    "reference_markers": None,
    "memory_mode": "high",
    "max_processed_dim": None,
    "max_non_rigid_dim": None,
    "micro_reg_fraction": 0.125,
    "max_image_dim_px": 4000,
    "micro_reg": 2,
    "interp_method": "bicubic",
    "stage_checkpoint_dir": None,
    "max_keypoints": None,
}
RUN_ARGS = ("input_dir", "out", "reference", "jvm_heap_gb", "n_cpus")
INPUT_DIR, OUT_DIR = "preprocessed", "registered_slides"
WRITER = "valis.registration.Slide.warp_and_save_slide, called by bin/register.py"
FILL = "Slide.bg_color (the value of the slide's darkest pixel) or zero"
BIN = Path(__file__).resolve().parents[3] / "bin"
_PIPELINE = None


def pipeline():
    """The pipeline's ``register`` module, loaded from ``bin/register.py``. Its import is
    what applies the pipeline's ``no_grad`` wrappers and environment settings, so it comes
    BEFORE any other import of VALIS."""
    global _PIPELINE
    if _PIPELINE is None:
        for path in (BIN / "utils", BIN):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        spec = importlib.util.spec_from_file_location(
            "mirage_bin_register", BIN / "register.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _PIPELINE = module
    return _PIPELINE


def jvm_heap_gb(opts, host=None):
    """``modules/local/register.nf``'s formula, the ladder's rung as the attempt:
    12 GiB at 16 GB, 60 at 64 GB, 80 at 256 GB, 96 at 512 GB. None (the pipeline's own
    estimate from the input size) when the memory request is unknown."""
    if opts.get("jvm_heap_gb"):
        return int(opts["jvm_heap_gb"])
    host = host or resources.host_info()
    mem = host.get("mem_requested_gb")
    if not mem:
        return None
    return max(1, int(min(32 + 16 * int(host.get("attempt") or 1), float(mem) - 4)))


def _error(exc):
    return f"{type(exc).__name__}: {exc}"


@contextlib.contextmanager
def _replaced(owner, name, make):
    """``owner.name`` replaced by ``make(original)`` for the duration."""
    original = getattr(owner, name)
    setattr(owner, name, functools.wraps(original)(make(original)))
    try:
        yield
    finally:
        setattr(owner, name, original)


def register_micro_wrapper(original, capture, state):
    """``Valis.register_micro`` with a capture in front. The original always runs."""

    def register_micro(self, *args, **kwargs):
        try:
            capture(self, kwargs)
        except Exception as exc:  # never the pipeline's problem: see the docstring
            state["capture_error"] = _error(exc)
            state["capture_traceback"] = traceback.format_exc()
        state["micro_called"] = True
        try:
            out = original(self, *args, **kwargs)
        except BaseException as exc:
            state["micro_error"] = _error(exc)
            raise
        state["micro_ran"] = True
        return out

    return register_micro


def _capture_before_micro(state, mon, warp_tools, opts, unwrapped_save, work):
    """What ``register_micro``'s wrapper does first: the rigid and non-rigid stages of
    every moving slide, and their images when the selfcheck asked for them."""
    how = (str(opts.get("crop_round", "probe")), opts.get("points") == "from_to")
    stage_images = vp.truthy(opts.get("stage_images", 0))

    def capture(registrar, kwargs):
        mon.pause()
        try:
            state["registrar"] = registrar
            state["register_micro"] = {k: str(v) for k, v in kwargs.items()}
            ref = registrar.get_ref_slide()
            for name, slide in registrar.slide_dict.items():
                if slide is ref:
                    continue
                try:
                    snaps = {
                        RIGID: vp.snapshot(slide, ref, warp_tools, False, *how),
                        NONRIGID: vp.snapshot(slide, ref, warp_tools, True, *how),
                    }
                    for variant, snap in snaps.items():
                        state["stages"].setdefault(variant, {})[name] = snap
                    if not stage_images:
                        continue
                    for variant in (RIGID, NONRIGID):
                        dst = Path(work) / "stage" / variant / f"{name}.ome.tiff"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        t0 = time.perf_counter()
                        unwrapped_save(
                            slide,
                            src_f=slide.src_f,
                            dst_f=str(dst),
                            level=0,
                            non_rigid=variant == NONRIGID,
                            crop=vp.CROP,
                            interp_method=ARGS["interp_method"],
                        )
                        state["stage_files"].setdefault(variant, {})[name] = {
                            "path": str(dst),
                            "write_s": time.perf_counter() - t0,
                        }
                except Exception as exc:
                    state["errors"].setdefault(name, []).append(_error(exc))
        finally:
            mon.resume()

    return capture


def save_wrapper(original, state, mon, warp_tools, opts):
    """``Slide.warp_and_save_slide`` with the clock paused and the final stage copied at
    the moment the writer reads it."""
    how = (str(opts.get("crop_round", "probe")), opts.get("points") == "from_to")

    def warp_and_save_slide(self, *args, **kwargs):
        mon.pause()
        try:
            name = str(self.name)
            try:
                registrar = self.val_obj
                state["registrar"] = registrar
                ref = registrar.get_ref_slide()
                if self is not ref:
                    non_rigid = bool(kwargs.get("non_rigid", True))
                    state["stages"].setdefault(FINAL, {})[name] = vp.snapshot(
                        self, ref, warp_tools, non_rigid, *how
                    )
                    state["final_non_rigid"][name] = non_rigid
            except Exception as exc:
                state["errors"].setdefault(name, []).append(_error(exc))
            t0 = time.perf_counter()
            out = original(self, *args, **kwargs)
            state["written"][name] = {
                "path": str(kwargs.get("dst_f") or (args[0] if args else "")),
                "write_s": time.perf_counter() - t0,
                "kwargs": {
                    k: str(v) for k, v in kwargs.items() if k not in ("src_f", "dst_f")
                },
            }
            return out
        finally:
            mon.resume()

    return warp_and_save_slide


def register_wrapper(original, state, mon):
    """``Valis.register`` unchanged; afterwards, the pipeline's repair condition and the
    bill so far are noted."""

    def register(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        mon.pause()
        try:
            state["registrar"] = self
            state["cost_after_register"] = mon.snapshot()
            state["repair_expected"] = [
                str(name)
                for name, slide in self.slide_dict.items()
                if slide.bk_dxdy is not None and slide.fwd_dxdy is None
            ]
        except Exception as exc:
            state["register_note_error"] = _error(exc)
        finally:
            mon.resume()
        return out

    return register


def _channel_used(slide):
    """The channel VALIS's default fluorescence processor reads on this slide: a regular
    expression for ``dapi`` on the channel names, channel 0 on no match or on several
    (VALIS 1.0.0 ``slide_io.py:723-747``)."""
    try:
        idx = int(slide.reader.get_channel_index("dapi"))
        names = list(slide.reader.metadata.channel_names or [])
        return {"index": idx, "name": str(names[idx]) if idx < len(names) else None}
    except Exception as exc:
        return {"error": _error(exc)}


def _move(src, opts, case_id, variant):
    native_dir = opts.get("native_dir")
    src = Path(src)
    if native_dir is None:
        return src
    home = Path(native_dir(case_id, variant))
    home.mkdir(parents=True, exist_ok=True)
    dst = home / src.name
    shutil.move(str(src), str(dst))
    return dst


def register_group(cases, work, opts):
    work = Path(work)
    points = str(opts.get("points", "image"))
    crop_round = str(opts.get("crop_round", "probe"))
    if points not in vp.POINTS or crop_round not in vp.ROUNDINGS:
        raise ValueError(
            f"points={points!r} (one of {vp.POINTS}), "
            f"crop_round={crop_round!r} (one of {vp.ROUNDINGS})"
        )
    mon = vp.monitor_of(opts)
    ref_path, moving = vp.patient_slides(cases)

    reg_module = pipeline()  # first: see pipeline()
    import valis
    from valis import registration, warp_tools

    version = str(getattr(valis, "__version__", "?"))
    if version != VERSION and not vp.truthy(opts.get("any_version", 0)):
        raise RuntimeError(
            f"this adapter is the pipeline's VALIS, {VERSION}; the installed valis is "
            f"{version}"
        )

    input_dir, out_dir = work / INPUT_DIR, work / OUT_DIR
    placed, how = vp.place_slides([ref_path, *moving.values()], input_dir)
    host = resources.host_info()
    call = {
        **ARGS,
        "input_dir": str(input_dir),
        "out": str(out_dir),
        "reference": Path(placed[ref_path]).name,
        "jvm_heap_gb": jvm_heap_gb(opts, host),
        "n_cpus": int(opts.get("workers") or resources.n_cpus()),
    }
    state = {
        "stages": {},
        "stage_files": {},
        "written": {},
        "errors": {},
        "final_non_rigid": {},
        "micro_called": False,
        "micro_ran": False,
    }
    slide_cls, valis_cls = registration.Slide, registration.Valis
    unwrapped_save = slide_cls.warp_and_save_slide
    capture = _capture_before_micro(state, mon, warp_tools, opts, unwrapped_save, work)
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _replaced(valis_cls, "register", lambda f: register_wrapper(f, state, mon))
        )
        stack.enter_context(
            _replaced(
                valis_cls,
                "register_micro",
                lambda f: register_micro_wrapper(f, capture, state),
            )
        )
        stack.enter_context(
            _replaced(
                slide_cls,
                "warp_and_save_slide",
                lambda f: save_wrapper(f, state, mon, warp_tools, opts),
            )
        )
        rc = reg_module.valis_registration(**call)
    # Everything below is the benchmark's own work (the point path, moving files): the
    # clock stays paused until every result is ready, so each carries the call's cost.
    mon.pause()
    try:
        ready = list(_results(cases, moving, placed, how, call, state, opts, rc, host))
    finally:
        mon.resume()
    yield from ready


def _results(cases, moving, placed, how, call, state, opts, rc, host):
    """Every result of one call: ``(case_id, variant, PointMap | None, info)``."""
    import valis
    from valis import warp_tools

    version = str(getattr(valis, "__version__", "?"))
    points = str(opts.get("points", "image"))
    if rc:
        raise RuntimeError(
            f"valis_registration returned {rc}: no slide was warped (its log is on stdout)"
        )

    registrar = state.get("registrar")
    stages = state["stages"]
    # a slide's name is VALIS's own; the file name is only the fallback
    names = {}
    for cid, path in moving.items():
        slide = None
        with contextlib.suppress(Exception):
            slide = registrar.get_slide(placed[path])
        names[cid] = str(slide.name) if slide is not None else vp.slide_name(path)
    seen = multiprocessing.cpu_count()  # what bound_cpu_count left for VALIS's pools
    torch = sys.modules.get("torch")
    info = {
        "version": version,
        "source": str(Path(valis.__file__).parent),
        "pipeline": str(BIN / "register.py"),
        "ARGS": {k: v for k, v in call.items() if k in ARGS},
        "run_args": {k: v for k, v in call.items() if k in RUN_ARGS},
        "points": points,
        "workers_seen": int(seen),
        "cpus": {
            "multiprocessing_cpu_count": int(seen),
            "valis_pool_threads": int(seen) - 1,
            "torch_threads": int(torch.get_num_threads())
            if hasattr(torch, "get_num_threads")
            else None,
        },
        "attempt": host.get("attempt"),
        "mem_requested_gb": host.get("mem_requested_gb"),
        "micro_called": bool(state["micro_called"]),
        "micro_ran": bool(state["micro_ran"]),
        "micro_error": state.get("micro_error", ""),
        "register_micro": state.get("register_micro", {}),
        "repair_fired": bool(state.get("repair_expected")),
        "repair_slides": state.get("repair_expected", []),
        "cost_after_register": state.get("cost_after_register"),
        "capture_error": state.get("capture_error", ""),
        "slides_placed": how,
        "slides_written": sorted(state["written"]),
        "slides_expected": sorted(Path(p).name for p in placed.values()),
        "implementation": {
            "package": "valis-wsi, through the pipeline",
            "source": str(Path(valis.__file__).parent),
            "registration": "bin/register.py::valis_registration",
            "parameters": "the pipeline's shipped call (ARGS; nextflow.config defaults)",
            "changed_from_defaults": {
                "stage_checkpoint_dir": "None (the arm writes checkpoints: files only)",
                "n_cpus": "the allocation (4 on every rung; the pipeline's REGISTER "
                "asks for 8)",
            },
            "point_warp": "valis.warp_tools.warp_xy_inv on the writer's crop"
            if points == "image"
            else "valis.warp_tools.warp_xy_from_to as Slide.warp_xy_from_to calls it",
            "benchmark_side": [
                "the patient's original multichannel slides hard-linked (or copied) "
                "into preprocessed/ under their own names",
                "Valis.register, Valis.register_micro and Slide.warp_and_save_slide "
                "wrapped for the duration of the call: fields and matrices are copied "
                "out, the writer runs with the clock paused; no argument or result is "
                "changed",
                "points: the reference-grid pixel plus the writer's crop origin, then "
                "valis.warp_tools.warp_xy_inv with the slide's own M, shapes and "
                "bk_dxdy; the rounding of the crop origin is measured on the installed "
                "warp_tools.warp_img",
                "a stored (pyvips) field is loaded whole as NumPy (VALIS's vectorised "
                "branch instead of its per-point loop; the field is the same)",
                "the written image moved from registered_slides/ to the result's "
                "native/ folder",
            ],
        },
    }
    if registrar is not None:
        with contextlib.suppress(Exception):
            ref = registrar.get_ref_slide()
            info["reference"] = str(ref.name)
            info["non_rigid_bbox"] = str(getattr(registrar, "_non_rigid_bbox", None))
            info["nuclear_channel"] = {
                str(n): _channel_used(s) for n, s in registrar.slide_dict.items()
            }

    for variant in VARIANTS:
        for case in cases:
            cid, name = case.case_id, names[case.case_id]
            written = state["written"].get(name)
            final = stages.get(FINAL, {}).get(name)
            if written is None or final is None:
                why = "; ".join(state["errors"].get(name, []))
                msg = (
                    f"valis_registration wrote no registered image of {name}"
                    if written is None
                    else f"the final transform of {name} was not captured"
                )
                yield cid, variant, None, {**info, "error": f"{msg}. {why}".strip()}
                continue
            # M is not touched after register() returns, and without a captured
            # pre-micro stage nothing ran after the first non-rigid pass
            snap = stages.get(variant, {}).get(name)
            if snap is None:
                snap = {**final, "bk_dxdy": None} if variant == RIGID else final
            own = {
                **info,
                "stage": vp.stage_info(snap, warp_tools, case.pixel_size_um),
                "final_non_rigid": bool(state["final_non_rigid"].get(name, True)),
            }
            problems = list(state["errors"].get(name, []))
            if variant == FINAL and not state["micro_ran"]:
                problems.append(
                    "the micro pass did not run"
                    + (f" ({state['micro_error']})" if state.get("micro_error") else "")
                    + ": this row is the non-rigid result"
                )
            if variant != RIGID and not own["final_non_rigid"]:
                problems.append("no displacement field: the writer drew rigid only")
            if state.get("capture_error"):
                problems.append(f"stage capture failed: {state['capture_error']}")
            if problems:
                own["problem"] = "; ".join(problems)
            source = (
                written
                if variant == FINAL
                else state["stage_files"].get(variant, {}).get(name)
            )
            if source is not None:
                path = _move(source["path"], opts, cid, variant)
                own["write_s"] = float(source["write_s"])
                own["native"] = {
                    "files": [str(path)],
                    "format": "ome-tiff",
                    "writer": WRITER
                    if variant == FINAL
                    else "valis.registration.Slide.warp_and_save_slide (stage image)",
                    "writer_kwargs": source.get("kwargs", {}),
                    "fill": FILL,
                    "valid_box": vp.footprint_box(
                        vp.image_map(snap, warp_tools).fn,
                        snap["ref"]["src_rc"],
                        snap["src_rc"],
                    ),
                    **vp.describe_file(path),
                }
            yield cid, variant, vp.point_map(snap, warp_tools, points), own
