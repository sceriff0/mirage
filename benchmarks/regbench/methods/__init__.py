"""The methods, and the one driver that runs any of them on a case or on a patient.

A method is a module with

``SCOPE``      ``"pair"`` (one reference and one moving slide per run) or ``"group"`` (every
               round of one patient in one run)
``VARIANTS``   the result names it produces, in the order it produces them

and one generator:

``register(case, work, opts)``         (pair) yields ``(variant, PointMap, info)``
``register_group(cases, work, opts)``  (group) yields ``(case_id, variant, PointMap | None,
                                       info)``. ``cases`` is every case of one patient in
                                       ``list_cases`` order. A failed round is yielded as
                                       ``(case_id, variant, None, {"error": msg})``; a
                                       degraded one carries ``info["problem"]``.

A ``PointMap`` (``regbench.mapping``) is the method's transform in its OWN direction, in
full-resolution pixels of the native slides. Optional hooks:

``write_native(state, case_id, dest_dir)``  the method's own writer, called after the
        generator ends with the clock paused; returns ``{"files", "format", "channels",
        "dtype"}``. ``state`` is whatever the method put in ``info["_state"]`` of its last
        yield. A method may instead write an image at a yield (inside ``opts["monitor"]``
        ``.pause()`` / ``.resume()``) and hand the same dict over as ``info["native"]``.
``write_stage`` / ``warp_extra`` / ``SELFCHECK_OFFSET`` / ``SELFCHECK_MASK`` /
``SELFCHECK_SUBPROCESS``  used by the selfcheck and the realcheck only; their signatures are
        in ``regbench.selfcheck``. A point mapping must stay valid until the run ends: the
        realcheck calls it after the writer.
``VARIANT_LABELS``  ``{label: variant names}`` for a label whose names do not follow the
        replace rule of ``variant_names`` (``valis120_hires``).

The driver hands a method three things through ``opts``, none of them recorded as an
option: ``monitor`` (the live ``resources.Monitor``), ``native_dir`` (``f(case_id,
variant) -> Path``, where an image written at a yield belongs:
``<out>/<variant>/<dataset>/<case>/native/``; the work directory is deleted after the run)
and ``label`` (the run's label, ``""`` without one).

**At each yield, before the generator is resumed and with the clock paused,** the driver
moves the nuclei through the map (the moving nuclei for ``mov_to_ref``, the reference
nuclei for ``ref_to_mov``), writes ``warped.npz`` and ``run.json``, and calls
``on_variant``. A method can therefore hand over live objects that a later stage
overwrites.

**What is billed.** Everything from the call to the generator until the variant is yielded:
reading the images, any conversion the method needs, the registration itself. The clock
and the memory peak are per variant: each ``run.json`` holds the cumulative time and the
process-tree peak up to ITS yield, so a variant yielded before a later stage has its own
cost. Warping the benchmark's points and a method's writer are not billed.

**How a run ends.** ``run.json`` is written as ``ok: false, cause: killed`` before the
method starts, so a task the scheduler kills leaves a failed row, not a gap. A method that
raises (``SystemExit`` included: STARE's argument errors raise it) is ``exception``, or
``oom`` / ``gpu_oom`` / ``timeout`` when the error says so (``resources.classify_error``).
A watcher polls the task's cgroup out-of-memory counter and ends the run as ``oom`` when
it rises, because the kernel may kill one worker and leave the parent waiting.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .. import METHODS, resources
from ..cases import load_points, result_dir, write_json, write_npz
from ..mapping import MOV_TO_REF, PointMap

SCOPES = ("pair", "group")
UNIT = {"pair": "pair", "group": "patient"}
KILLED = (
    "killed before finishing (out of memory or time limit? see the SLURM log and "
    "slurm_accounting.psv)"
)
OOM_POLL_S = 1.0  # how often the cgroup out-of-memory counter is read
OOM_GRACE_S = 60.0  # how long an interrupted method has to unwind before a hard exit
OOM_EXIT_CODE = 137
_LAUNCHES = itertools.count(1)


def load(method):
    """The method's module. A module-like object (tests) is returned as it is."""
    if not isinstance(method, str):
        return method
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; expected one of {METHODS}")
    return importlib.import_module(f"{__name__}.{method}")


def method_name(method):
    if isinstance(method, str):
        return method
    return getattr(method, "NAME", None) or method.__name__.rsplit(".", 1)[-1]


def scope_of(method):
    scope = getattr(load(method), "SCOPE", "pair")
    if scope not in SCOPES:
        raise SystemExit(f"{method_name(method)}: SCOPE must be one of {SCOPES}")
    return scope


def variant_names(method, label=None):
    """``--label`` renames a method's variants, so two configurations of it can sit side by side."""
    mod, name = load(method), method_name(method)
    names = mod.VARIANTS
    declared = (getattr(mod, "VARIANT_LABELS", None) or {}).get(label)
    if declared is not None:
        if len(declared) != len(names):
            raise SystemExit(
                f"{name}: VARIANT_LABELS[{label!r}] names {len(declared)} variants, "
                f"the method produces {len(names)}"
            )
        return list(declared)
    return [v.replace(name, label, 1) for v in names] if label else list(names)


def parse_opts(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--opt wants key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def selfcheck_status():
    """The status the launcher hands a cohort run (``REGBENCH_SELFCHECK_STATUS``)."""
    return os.environ.get("REGBENCH_SELFCHECK_STATUS") or "not_checked"


def _warp_chunked(fn, xy, chunk=2_000_000):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return xy
    out = [
        np.asarray(fn(xy[i : i + chunk]), dtype=float).reshape(-1, 2)
        for i in range(0, len(xy), chunk)
    ]
    out = np.concatenate(out)
    if out.shape != xy.shape:
        raise RuntimeError(f"point map returned {out.shape} for {xy.shape} points")
    return out


def jacobian_grid(hw, max_side=512):
    """``(xy (N, 2), (rows, cols), step)``: a regular lattice over a slide, at most
    ``max_side`` points a side. The scorer differentiates each method's map on it."""
    h, w = hw
    step = max(8, -(-max(h, w) // max_side))
    ys, xs = np.arange(step / 2.0, h, step), np.arange(step / 2.0, w, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1), (len(ys), len(xs)), step


def _jsonable(obj):
    """``info`` as JSON: arrays become lists, anything else unknown its ``str``."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


class _OomWatch:
    """Polls the cgroup ``oom_kill`` counter; on a rise, interrupts the main thread."""

    def __init__(self, on_stuck):
        self.path = resources.oom_events_path()
        self.start_count = resources.read_oom_kills(self.path)
        self.fired = False
        self._acknowledged = threading.Event()
        self._stop = threading.Event()
        self._on_stuck = on_stuck
        self._thread = None

    def rose(self):
        if self.start_count is None:
            return False
        now = resources.read_oom_kills(self.path)
        return now is not None and now > self.start_count

    def _loop(self):
        while not self._stop.wait(OOM_POLL_S):
            if not self.rose():
                continue
            self.fired = True
            try:  # a real signal: it also wakes a main thread blocked in a system call
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            except (AttributeError, OSError, ValueError):
                pass
            if not self._acknowledged.wait(OOM_GRACE_S):
                # The method never came back to the interpreter (native code waiting on a
                # dead worker). Record the run, then leave: nothing else can end it.
                self._on_stuck()
                os._exit(OOM_EXIT_CODE)
            return

    def start(self):
        if self.start_count is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def acknowledge(self):
        self._acknowledged.set()

    def stop(self):
        self._stop.set()
        self._acknowledged.set()


def _launch_id():
    """The scheduler's job id; without a scheduler, this process and a count of its launches."""
    launch = resources.host_info()["launch_id"]
    return f"{launch}.{next(_LAUNCHES)}" if launch.startswith("pid") else launch


def _run(
    method,
    cases,
    out,
    opts,
    label,
    work,
    keep_work,
    on_variant,
    launch_id,
    realcheck=False,
):
    mod, name = load(method), method_name(method)
    scope = scope_of(method)
    opts = dict(opts or {})
    opts.setdefault("workers", resources.n_cpus())
    names = dict(zip(mod.VARIANTS, variant_names(method, label)))
    by_id = {c.case_id: c for c in cases}
    points = {c.case_id: load_points(c) for c in cases}
    final = mod.VARIANTS[-1]
    work = Path(work) if work else result_dir(out, names[final], cases[0]) / "work"
    work.mkdir(parents=True, exist_ok=True)

    host = resources.host_info()
    base = {
        "method": name,
        "label": label or "",
        "opts": {k: str(v) for k, v in opts.items()},
        "workers": int(opts["workers"]),
        "unit": UNIT[scope],
        "group": cases[0].group,
        "group_cases": [c.case_id for c in cases],
        "selfcheck_status": selfcheck_status(),
        "native": None,
        "write_s": 0.0,
        **host,
        "launch_id": launch_id,
    }
    lock = threading.Lock()
    done, failed, final_maps = {}, {}, {}  # keyed by (case_id, variant)
    infos = {}  # the live info of each yield, for a declared offset (realcheck)
    earlier = {}

    def head(case, variant):
        return {
            **base,
            "dataset": case.dataset,
            "case_id": case.case_id,
            "variant": names[variant],
            "previous_attempts": earlier.get((case.case_id, variant), []),
        }

    def fail(case, variant, error, cause, cost=None, tb="", info=None):
        d = result_dir(out, names[variant], case)
        (d / "warped.npz").unlink(missing_ok=True)
        rec = {**head(case, variant), "ok": False, "error": error, "cause": cause}
        if cost is not None:
            rec.update(register_s=cost["wall_s"], resources=cost)
        if tb:
            rec["traceback"] = tb
        if info:
            rec["info"] = _jsonable(info)
        write_json(d / "run.json", rec)
        return rec

    # Written first: if the scheduler kills the task (out of memory, time limit) nothing
    # below runs, and this is what the scorer finds instead of silence. A failed earlier
    # attempt of the ladder is carried along, so its cost is not lost when it is overwritten.
    for case in cases:
        for variant in mod.VARIANTS:
            d = result_dir(out, names[variant], case)
            try:
                old = json.loads((d / "run.json").read_text())
            except (OSError, ValueError):
                old = None
            if old and not old.get("ok") and old.get("attempt", 1) < host["attempt"]:
                res = old.get("resources") or {}
                earlier[(case.case_id, variant)] = [
                    *old.get("previous_attempts", []),
                    {
                        "attempt": old.get("attempt", 1),
                        "cause": old.get("cause", ""),
                        "mem_requested_gb": old.get("mem_requested_gb"),
                        "launch_id": old.get("launch_id"),
                        "wall_s": res.get("wall_s"),
                        "cpu_s": res.get("cpu_s"),
                    },
                ]
            fail(case, variant, KILLED, "killed")

    mon = resources.Monitor()
    # for a method whose own writer runs at a yield: the clock to pause, and where the
    # image belongs (the work directory does not outlive the run)
    opts["monitor"] = mon
    opts["native_dir"] = lambda case_id, variant: (
        result_dir(out, names[variant], by_id[case_id]) / "native"
    )
    opts["label"] = label or ""
    for key in ("monitor", "native_dir", "label"):
        base["opts"].pop(key, None)

    def not_done(error, cause, cost, tb=""):
        with lock:
            for case in cases:
                for variant in mod.VARIANTS:
                    key = (case.case_id, variant)
                    if key not in done and key not in failed:
                        failed[key] = fail(case, variant, error, cause, cost, tb)

    watch = _OomWatch(
        lambda: not_done(
            "the cgroup out-of-memory counter rose and the method did not return",
            "oom",
            None,
        )
    )
    state = None
    error = cause = tb = ""
    mon.start()
    watch.start()
    try:
        gen = (
            mod.register(cases[0], work, opts)
            if scope == "pair"
            else mod.register_group(list(cases), work, opts)
        )
        for item in gen:
            mon.pause()
            cost = mon.snapshot()
            if scope == "pair":
                case_id = cases[0].case_id
                variant, pmap, info = item
            else:
                case_id, variant, pmap, info = item
            case, key = by_id[case_id], (case_id, variant)
            info = dict(info or {})
            if "_state" in info:
                state = info.pop("_state")
            band = info.pop("band_mov", None)
            native = info.pop("native", None)
            try:
                if watch.rose():
                    raise MemoryError(
                        "the cgroup out-of-memory counter rose during this run"
                    )
                if pmap is None:
                    failed[key] = fail(
                        case,
                        variant,
                        str(info.get("error") or "the method reported a failed round"),
                        "exception",
                        cost,
                        info=info,
                    )
                    continue
                if not isinstance(pmap, PointMap):
                    raise TypeError(
                        f"{name} yielded {type(pmap).__name__} for {variant}; a method "
                        "returns a regbench.mapping.PointMap"
                    )
                pts = points[case_id]
                forward = pmap.direction == MOV_TO_REF
                side, hw = ("mov", case.mov_hw) if forward else ("ref", case.ref_hw)
                grid, grid_shape, grid_step = (
                    jacobian_grid(hw) if hw else (np.empty((0, 2)), (0, 0), 1)
                )
                t0 = time.perf_counter()
                arrays = {
                    "direction": np.array(pmap.direction),
                    "cells_xy": _warp_chunked(
                        pmap.fn, pts.get(f"cells_{side}_xy", np.empty((0, 2)))
                    ).astype(np.float32),
                    # the known-truth probes of the side the map starts from, at full
                    # precision: fn(truth_mov) or fn(truth_ref), never the other way
                    "truth_xy": _warp_chunked(
                        pmap.fn, pts.get(f"truth_{side}", np.empty((0, 2)))
                    ),
                    "grid": _warp_chunked(pmap.fn, grid).astype(np.float32),
                    "grid_shape": np.array(grid_shape),
                    "grid_step": np.array(grid_step),
                }
                if band is not None:
                    arrays["band"] = np.asarray(band, dtype=bool).ravel()
                warp_s = time.perf_counter() - t0
                d = result_dir(out, names[variant], case)
                write_npz(d / "warped.npz", **arrays)
                rec = {
                    **head(case, variant),
                    "ok": True,
                    "error": "",
                    "cause": "",
                    "problem": str(info.get("problem") or ""),
                    "direction": pmap.direction,
                    "register_s": cost["wall_s"],
                    "warp_s": warp_s,
                    "resources": cost,
                    "native": native,
                    "write_s": float(info.get("write_s") or 0.0),
                    "info": _jsonable(info),
                }
                write_json(d / "run.json", rec)
                done[key] = rec
                final_maps[key] = pmap
                infos[key] = info
            except (MemoryError, KeyboardInterrupt):
                raise
            except Exception as exc:  # the benchmark's own step failed for this variant
                failed[key] = fail(
                    case,
                    variant,
                    f"{type(exc).__name__}: {exc}",
                    "exception",
                    cost,
                    traceback.format_exc(),
                )
            else:
                # outside the handler above: an error in the caller's hook is the caller's
                if on_variant is not None:
                    on_variant(case, names[variant], pmap, info)
            finally:
                mon.resume()
    except KeyboardInterrupt:
        if not watch.fired:
            raise
        watch.acknowledge()
        error = (
            "the cgroup out-of-memory counter rose: a process of this run was killed"
        )
        cause, tb = "oom", traceback.format_exc()
    except (
        Exception,
        SystemExit,
    ) as exc:  # a failed case is scored at the initial pose
        error = f"{type(exc).__name__}: {exc}"
        cause, tb = resources.classify_error(exc), traceback.format_exc()
        if watch.rose():
            cause = "oom"
    finally:
        watch.stop()
        mon.pause()
        cost = mon.snapshot()
    if watch.fired and cause != "oom":
        try:  # let a pending interrupt land here, not inside the writer below
            time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    if error:
        ids = "+".join(c.case_id for c in cases)
        print(f"{cases[0].dataset}/{ids} [{name}]: {error}", file=sys.stderr)

    # The method's own writer, clock and peak paused. Its failure does not undo the
    # registration: the row stays ok and `native` says what went wrong.
    write_native = getattr(mod, "write_native", None)
    natives, write_s = {}, {}
    if write_native is not None and state is not None:
        for case in cases:
            if (case.case_id, final) not in done:
                continue
            dest = result_dir(out, names[final], case) / "native"
            t0 = time.perf_counter()
            try:
                dest.mkdir(parents=True, exist_ok=True)
                natives[case.case_id] = write_native(state, case.case_id, dest)
            except (Exception, SystemExit) as exc:
                natives[case.case_id] = {"error": f"{type(exc).__name__}: {exc}"}
            write_s[case.case_id] = time.perf_counter() - t0
    end = mon.stop()
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    gpu = resources.gpu_info()
    for case in cases:
        for variant in mod.VARIANTS:
            key = (case.case_id, variant)
            if key in done:
                rec = done[key]
                # the scheduler's peak is only known once the task is over
                rec["resources"] = {**rec["resources"], **end}
                rec.update(gpu)
                if variant == final and case.case_id in natives:
                    rec["native"] = natives[case.case_id]
                    rec["write_s"] = write_s[case.case_id]
                write_json(result_dir(out, names[variant], case) / "run.json", rec)
            elif key not in failed:
                failed[key] = fail(
                    case,
                    variant,
                    error or "the method did not produce this variant",
                    cause or "exception",
                    {**cost, **end},
                    tb,
                )
    if realcheck:
        # Clock stopped, writers done: does each mapping follow the image written for it?
        from .. import selfcheck

        declare = getattr(mod, "SELFCHECK_OFFSET", None)
        for (case_id, variant), rec in done.items():
            native = rec.get("native")
            if not (isinstance(native, dict) and native.get("files")):
                continue
            d = result_dir(out, names[variant], by_id[case_id])
            try:
                offsets = declare(infos[(case_id, variant)]) if declare else None
                res = selfcheck.realcheck(
                    by_id[case_id],
                    d,
                    final_maps[(case_id, variant)],
                    native,
                    offset=(offsets or {}).get(variant),
                    points=points[case_id],
                )
                rec["realcheck"] = {"status": res["status"], "flag": res["flag"]}
            except (Exception, SystemExit) as exc:
                rec["realcheck"] = {
                    "status": "crashed",
                    "flag": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            write_json(d / "run.json", rec)
    final_maps.clear()  # kept until the writer has run (the realcheck reads them)
    infos.clear()
    return {
        c.case_id: [names[v] for v in mod.VARIANTS if (c.case_id, v) in done]
        for c in cases
    }


def run_case(
    method,
    case,
    out,
    opts=None,
    label=None,
    work=None,
    keep_work=False,
    on_variant=None,
    realcheck=False,
):
    """Register one case with a pair method; returns the variants that succeeded.

    ``realcheck`` also checks, after the method's writer, that each mapping follows the
    image written for it (``regbench.selfcheck.realcheck``; ``run.json: realcheck``)."""
    if scope_of(method) != "pair":
        raise SystemExit(
            f"{method_name(method)} runs per patient: use run_group (`run --group`)"
        )
    args = (opts, label, work, keep_work, on_variant, _launch_id(), realcheck)
    done = _run(method, [case], out, *args)
    return done[case.case_id]


def run_group(
    method,
    cases,
    out,
    opts=None,
    label=None,
    work=None,
    keep_work=False,
    on_variant=None,
    realcheck=False,
):
    """Register one patient: ``{case_id: [variants that succeeded]}``.

    A per-patient method gets every case in one run. A pair method is run on each case in
    turn, so a launcher can address both kinds by patient.
    """
    cases = list(cases)
    if not cases:
        return {}
    if len({c.group for c in cases}) != 1:
        raise SystemExit("run_group takes the cases of ONE patient")
    launch = _launch_id()  # one launch, however many rounds it holds
    if scope_of(method) == "group":
        args = (opts, label, work, keep_work, on_variant, launch, realcheck)
        return _run(method, cases, out, *args)
    done = {}
    for case in cases:
        sub = Path(work) / str(case.case_id) if work else None
        args = (opts, label, sub, keep_work, on_variant, launch, realcheck)
        done.update(_run(method, [case], out, *args))
    return done
