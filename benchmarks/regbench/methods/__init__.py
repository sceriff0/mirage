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
                                       ``(case_id, variant, None, {"error": msg})``
                                       (``"cause"`` may name why: ``oom`` climbs); a
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
``info["band_mov_nuc"]`` / ``info["band_ref_nuc"]``  the same seam rule applied to the
        BARE nuclei of that side (``cases.polys(points, side, "nuclei")``). Used only
        where the case does not say which outline each nucleus belongs to
        (``_nuclei_band``); stored as ``band_nuc`` / ``band_ref_nuc``.
``info["band_mov"]`` / ``info["band_ref"]``  a seam mask: one bool per moving (reference)
        nucleus, True where the nucleus straddles one of the method's own joins (ASHLAR's
        tile overlaps, PALOM's block edges). Stored in ``warped.npz`` as ``band`` /
        ``band_ref``; the scorer scores EVERY variant of the case a second time without
        those nuclei (``<variant>:non_<owner>``). A known-truth probe is a moving
        nucleus's centroid, so a moving-side mask drops its probes by itself; a method
        with a reference-side mask says which probes are on its seams with
        ``info["band_truth"]`` (one bool per probe, stored as ``band_truth``).
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
A watcher polls the task's cgroup out-of-memory counter from before the method starts
until its writer has finished, and ends the run when it rises, because the kernel may
kill one worker and leave the parent waiting. ``run.json: oom_watch`` says whether the
counter was readable and the watcher started.

**Three stretches, three memory causes** (all three go up the ladder):

``oom``            the method ran out of memory while registering
``writer_oom``     its own writer did: ``write_native``, a writer that reported a memory
                   error at a yield (``native: {"error": ...}``), or a kill the watcher
                   saw while the method itself had the clock paused
``benchmark_oom``  the benchmark's own step did, with the clock stopped (moving the
                   nuclei through the map): not the method's failure

**A row is ok only when its image exists.** A method's writer that fails, at a yield or
in ``write_native``, fails the row it was writing for: ``writer_oom``, or ``writer`` for
any other error. While ``write_native`` runs, that row's ``run.json`` says ``ok: false,
cause: killed, phase: writer``, so a kill there leaves a failed row too. The rule is the
same for every method.

**A unit that is already done is not run again** (``resume=True``, what ``run`` passes):
when every result of the unit is ``ok`` for the same method, label, options and cases,
nothing runs and no placeholder is written; the records gain this submission's id
(``submissions``, ``reused_by``). ``REGBENCH_FORCE=1`` (``run --force``) runs it anyway.
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
WRITER_KILLED = (
    "killed while the method's own writer ran: the image was not finished (out of "
    "memory or time limit? see the SLURM log)"
)
# the stretch a memory death happened in -> its cause (module docstring)
MEMORY_CAUSE = {"method": "oom", "writer": "writer_oom", "benchmark": "benchmark_oom"}


class _OutOfMemory(MemoryError):
    """A memory death whose stretch is known."""

    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase


def writer_cause(error):
    """``writer_oom`` when a writer's error text names memory, else ``writer``."""
    return "writer_oom" if resources.classify_text(error) == "oom" else "writer"


def patient_channels(cases):
    """How many channels the patient's slides hold together (the reference once, every
    moving round of this launch), or None where a case does not list its channels. A
    written file with fewer cannot be the patient's merged image."""
    ref = len(cases[0].extra.get("channels_ref") or [])
    mov = [len(c.extra.get("channels_mov") or []) for c in cases]
    if not ref or not all(mov):
        return None
    return ref + sum(mov)


_LAUNCHES = itertools.count(1)

# Methods with NO input rule for a brightfield case (an RGB image pair: HyReCo, ANHIR),
# and why. Their adapters refuse such a case with this text (`require_fluorescence`), and
# the run list leaves their runs out of a brightfield dataset (`variants.runs_for`). A
# method not named here has a rule: STARE reads the case's one-channel proxy
# (methods/stare.py), VALIS 1.2.0 reads the original RGB images and picks its own
# brightfield processing, DeeperHistReg reads the RGB image with its shipped parameters
# (methods/deeperhistreg.py). A run of such a method can still be left out of ONE
# dataset (`variants.NOT_ON`).
NO_BRIGHTFIELD = {
    "valis_mirage": "it is the pipeline's own VALIS call, and the pipeline registers "
    "multichannel fluorescence rounds only: there is no shipped brightfield "
    "configuration of it to run",
    "ashlar": "ASHLAR stitches unstitched tiles by one translation per tile; a pair of "
    "whole brightfield images of two sections (or of one section re-mounted between "
    "the scans) is not its input (rulings R3.8 and R3.20: ASHLAR is run on neither "
    "HyReCo nor ANHIR)",
    "palom": "this adapter follows PALOM's README recipe on one channel by index; on an "
    "RGB slide that would be the red channel, chosen by nobody. PALOM has colour "
    "modes of its own for an RGB slide (palom/color.py HaxProcessor: grayscale, "
    "hematoxylin, aec, dab); which one is the input rule is not decided, and PALOM "
    "is described by its README as made for re-imaged slides of the same section",
}


def require_fluorescence(case, method):
    """Refuse a case of another modality for a method without an input rule for it."""
    if case.modality != "fluorescence":
        raise ValueError(
            f"{case.case_id}: {method} has no input rule for a {case.modality} case: "
            f"{NO_BRIGHTFIELD[method]}"
        )


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


def _nuclei_band(points, side, outline_band, own):
    """``(seam flag per bare nucleus of ``side``, how it was got)``, or ``(None, "")``.

    ``inherited``: the case says which outline each nucleus belongs to
    (``nuclei_<side>_outline``, written only where it is known by construction: both sets
    cut from one segmentation), so a nucleus takes its outline's flag and the two sets
    drop the same objects. Otherwise ``geometry``: the method applied its own seam rule
    to the nuclei (``info["band_mov_nuc"]`` / ``info["band_ref_nuc"]``). With neither
    there is no flag, and the scorer makes no off-seam row for the nuclei."""
    if f"nuclei_{side}_n" not in points:
        return None, ""
    n = int(points[f"nuclei_{side}_n"])
    parent = points.get(f"nuclei_{side}_outline")
    if outline_band is not None and parent is not None and len(parent) == n:
        flags = np.asarray(outline_band, dtype=bool).ravel()
        parent = np.asarray(parent, dtype=np.int64)
        if not n or (parent.min() >= 0 and parent.max() < len(flags)):
            return flags[parent], "inherited"
    if own is not None and len(np.ravel(own)) == n:
        return np.asarray(own, dtype=bool).ravel(), "geometry"
    return None, ""


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

    def __init__(self, on_stuck, where=None):
        self.path = resources.oom_events_path()
        self.start_count = resources.read_oom_kills(self.path)
        self.fired = False
        self.phase = None  # the stretch the run was in when the counter rose
        self.started = False
        self._where = where or (lambda: "method")
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
            self.phase = self._where()
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
            self.started = True
        return self

    def record(self):
        """``run.json: oom_watch``: without a readable counter nothing watches, and a
        killed worker can leave the run waiting until the time limit."""
        return {
            "readable": self.start_count is not None,
            "started": bool(self.started),
            "path": str(self.path) if self.path else "",
        }

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
    resume=False,
):
    from ..ladder import mine

    mod, name = load(method), method_name(method)
    scope = scope_of(method)
    opts = dict(opts or {})
    opts.setdefault("workers", resources.n_cpus())
    names = dict(zip(mod.VARIANTS, variant_names(method, label)))
    host = resources.host_info()
    submission = host["submission"]
    recorded = {
        k: str(v)
        for k, v in opts.items()
        if k not in ("monitor", "native_dir", "label")
    }
    if resume and os.environ.get("REGBENCH_FORCE") != "1":
        reused = _reuse(
            out, cases, names, name, label, recorded, realcheck, host, launch_id
        )
        if reused is not None:
            return reused
    by_id = {c.case_id: c for c in cases}
    points = {c.case_id: load_points(c) for c in cases}
    final = mod.VARIANTS[-1]
    work = Path(work) if work else result_dir(out, names[final], cases[0]) / "work"
    work.mkdir(parents=True, exist_ok=True)

    phase = {"now": "method"}
    mon = resources.Monitor()

    def where():
        if phase["now"] != "method":
            return phase["now"]
        # a pause the method took itself is its own writer
        return "writer" if mon.paused else "method"

    watch = _OomWatch(lambda: stuck(), where)
    base = {
        "method": name,
        "label": label or "",
        "opts": recorded,
        "submissions": [submission] if submission else [],
        "oom_watch": watch.record(),
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
        # nothing an earlier attempt left may answer for this one (a gate reads it)
        (d / "realcheck.json").unlink(missing_ok=True)
        rec = {**head(case, variant), "ok": False, "error": error, "cause": cause}
        if cost is not None:
            rec.update(register_s=cost.get("wall_s"), resources=cost)
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
            before = (old or {}).get("attempt", 1), (old or {}).get("retry", 0)
            if (
                old
                and not old.get("ok")
                and mine(old, submission)
                and before < (host["attempt"], host["retry"])
            ):
                res = old.get("resources") or {}
                sched = old.get("scheduler") or {}
                earlier[(case.case_id, variant)] = [
                    *old.get("previous_attempts", []),
                    {
                        "attempt": old.get("attempt", 1),
                        "cause": old.get("cause", ""),
                        "mem_requested_gb": old.get("mem_requested_gb"),
                        "launch_id": old.get("launch_id"),
                        "retry": old.get("retry", 0),
                        "wall_s": res.get("wall_s"),
                        "cpu_s": res.get("cpu_s"),
                        # the scheduler's figures for the whole task, where the
                        # process could record none (ladder.settle)
                        "scheduler_state": sched.get("state"),
                        "scheduler_elapsed_s": sched.get("elapsed_s"),
                        "scheduler_cpu_s": sched.get("cpu_s"),
                        # for the failure log: what that attempt said of itself
                        "error": str(old.get("error") or "")[:400],
                        "cause_from": old.get("cause_from") or "",
                        "phase": old.get("phase") or "",
                        "peak_rss_mb": res.get("peak_rss_mb"),
                        "log_path": old.get("log_path") or "",
                    },
                ]
            fail(case, variant, KILLED, "killed")

    # for a method whose own writer runs at a yield: the clock to pause, and where the
    # image belongs (the work directory does not outlive the run)
    opts["monitor"] = mon
    opts["native_dir"] = lambda case_id, variant: (
        result_dir(out, names[variant], by_id[case_id]) / "native"
    )
    opts["label"] = label or ""

    def not_done(error, cause, cost, tb=""):
        with lock:
            for case in cases:
                for variant in mod.VARIANTS:
                    key = (case.case_id, variant)
                    if key not in done and key not in failed:
                        failed[key] = fail(case, variant, error, cause, cost, tb)

    natives, write_s = {}, {}

    def stuck():
        """The counter rose and the main thread never came back: record, then leave."""
        cause = MEMORY_CAUSE[watch.phase or "method"]
        if phase["now"] == "writer":  # inside write_native
            for case in cases:
                key = (case.case_id, final)
                if key in done and case.case_id not in natives:
                    failed[key] = fail(
                        case,
                        final,
                        "the cgroup out-of-memory counter rose and the method's "
                        "writer did not return",
                        cause,
                        done.pop(key).get("resources"),
                    )
        not_done(
            "the cgroup out-of-memory counter rose and the method did not return",
            cause,
            None,
        )

    state = None
    error = cause = tb = ""
    mon.start()
    watch.start()
    base["oom_watch"] = watch.record()
    try:
        gen = (
            mod.register(cases[0], work, opts)
            if scope == "pair"
            else mod.register_group(list(cases), work, opts)
        )
        for item in gen:
            mon.pause()
            phase["now"] = "benchmark"
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
            band_ref = info.pop("band_ref", None)
            band_truth = info.pop("band_truth", None)
            band_nuc_own = info.pop("band_mov_nuc", None)
            band_ref_nuc_own = info.pop("band_ref_nuc", None)
            native = info.pop("native", None)
            try:
                if watch.rose():
                    # it rose before this yield: in the method's stretch, or in a
                    # writer the method ran with the clock paused
                    raise _OutOfMemory(
                        watch.phase or "method",
                        "the cgroup out-of-memory counter rose during this run",
                    )
                if (
                    isinstance(native, dict)
                    and native.get("error")
                    and not native.get("files")
                ):
                    # the method's own writer failed at this yield: no image, so the
                    # row is not ok, whichever method it is
                    failed[key] = fail(
                        case,
                        variant,
                        f"the method's own writer failed: {native['error']}",
                        writer_cause(native["error"]),
                        cost,
                        info=info,
                    )
                    continue
                if pmap is None:
                    failed[key] = fail(
                        case,
                        variant,
                        str(info.get("error") or "the method reported a failed round"),
                        # a method may say why (a memory cause goes up the ladder)
                        info.pop("cause", None)
                        if info.get("cause") in resources.CAUSES
                        else "exception",
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
                # a method's own seam mask, by nucleus: on the moving side ("band", from
                # info["band_mov"]) or on the reference side ("band_ref"). The scorer
                # applies an owner's mask to every variant of the case (score.SEAM_OWNERS)
                if band is not None:
                    arrays["band"] = np.asarray(band, dtype=bool).ravel()
                if band_ref is not None:
                    arrays["band_ref"] = np.asarray(band_ref, dtype=bool).ravel()
                if band_truth is not None:  # the same seams, by known-truth probe
                    arrays["band_truth"] = np.asarray(band_truth, dtype=bool).ravel()
                # the SECOND polygon set, the bare nuclei, through the same map in the
                # same direction; and the owner's seam mask on them (_nuclei_band)
                nuclei_band = {}
                if f"nuclei_{side}_xy" in pts:
                    arrays["nuclei_xy"] = _warp_chunked(
                        pmap.fn, pts[f"nuclei_{side}_xy"]
                    ).astype(np.float32)
                for name_nuc, where, flags, own in (
                    ("band_nuc", "mov", band, band_nuc_own),
                    ("band_ref_nuc", "ref", band_ref, band_ref_nuc_own),
                ):
                    got, how = _nuclei_band(pts, where, flags, own)
                    if got is not None:
                        arrays[name_nuc], nuclei_band[where] = got, how
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
                if nuclei_band:  # how each side's nuclei got their seam flag
                    rec["nuclei_band"] = nuclei_band
                write_json(d / "run.json", rec)
                done[key] = rec
                final_maps[key] = pmap
                infos[key] = info
            except (_OutOfMemory, KeyboardInterrupt):
                raise
            except MemoryError as exc:
                # the benchmark's own step, clock stopped: more memory is the remedy
                # (the unit goes up), and it is not billed as the method's oom
                raise _OutOfMemory(
                    "benchmark",
                    f"the benchmark's own step ran out of memory while it moved the "
                    f"nuclei of {variant} through the map: {exc}",
                ) from exc
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
                phase["now"] = "method"
                mon.resume()
    except KeyboardInterrupt:
        if not watch.fired:
            raise
        watch.acknowledge()
        error = (
            "the cgroup out-of-memory counter rose: a process of this run was killed"
        )
        cause, tb = MEMORY_CAUSE[watch.phase or "method"], traceback.format_exc()
    except (
        Exception,
        SystemExit,
    ) as exc:  # a failed case is scored at the initial pose
        error = f"{type(exc).__name__}: {exc}"
        cause, tb = resources.classify_error(exc), traceback.format_exc()
        if isinstance(exc, _OutOfMemory):
            cause = MEMORY_CAUSE[exc.phase]
        elif watch.rose():
            cause = MEMORY_CAUSE[watch.phase or "method"]
    finally:
        phase["now"] = "method"
        mon.pause()
        cost = mon.snapshot()
    if watch.fired and cause not in resources.MEMORY_CAUSES:
        try:  # let a pending interrupt land here, not inside the writer below
            time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    if error:
        ids = "+".join(c.case_id for c in cases)
        print(f"{cases[0].dataset}/{ids} [{name}]: {error}", file=sys.stderr)

    # The method's own writer, clock and peak paused, the watcher still running. The row
    # it writes for is ok only once the image exists: until then its run.json says
    # killed, and a writer that fails (or is killed) leaves a failed row, for every
    # method alike. A memory death there goes up the ladder as `writer_oom`.
    write_native = getattr(mod, "write_native", None)
    if write_native is not None and state is not None:
        phase["now"] = "writer"
        dead = ""  # a memory death: the rounds not yet written share it

        def lose(case, why, how, trace=""):
            key = (case.case_id, final)
            rec = done.pop(key)
            final_maps.pop(key, None)
            failed[key] = fail(
                case, final, why, how, rec.get("resources"), trace, rec.get("info")
            )

        for case in cases:
            key = (case.case_id, final)
            if key not in done:
                continue
            d = result_dir(out, names[final], case)
            if dead:
                lose(case, f"the method's own writer was not run: {dead}", "writer_oom")
                continue
            holding = {
                **done[key],
                "ok": False,
                "error": WRITER_KILLED,
                "cause": "killed",
                "phase": "writer",
            }
            write_json(d / "run.json", holding)
            t0 = time.perf_counter()
            try:
                (d / "native").mkdir(parents=True, exist_ok=True)
                got = write_native(state, case.case_id, d / "native")
                if watch.rose():
                    raise _OutOfMemory(
                        "writer",
                        "the cgroup out-of-memory counter rose while the method's "
                        "own writer ran",
                    )
                if isinstance(got, dict) and got.get("error") and not got.get("files"):
                    raise RuntimeError(str(got["error"]))
                natives[case.case_id] = got
                write_s[case.case_id] = time.perf_counter() - t0
            except KeyboardInterrupt:
                if not watch.fired:
                    raise
                watch.acknowledge()
                dead = "the cgroup out-of-memory counter rose while it ran"
                lose(case, f"the method's own writer was killed: {dead}", "writer_oom")
            except (Exception, SystemExit) as exc:
                text = f"{type(exc).__name__}: {exc}"
                how = (
                    "writer_oom"
                    if resources.classify_error(exc) == "oom" or watch.rose()
                    else "writer"
                )
                if how == "writer_oom":
                    dead = text
                lose(
                    case,
                    f"the method's own writer failed: {text}",
                    how,
                    traceback.format_exc(),
                )
        phase["now"] = "method"
    watch.stop()
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
    # Clock stopped, writers done: what each written image IS (channels, pyramid or not,
    # compression, size), read from the file. A method whose own output is not the file
    # it hands over (ASHLAR's mosaic, DeeperHistReg's per-channel files) says so itself
    # in `native["written"]`, and that is kept.
    from .. import imageio

    total = patient_channels(cases)
    for (case_id, variant), rec in done.items():
        native = rec.get("native")
        if not (isinstance(native, dict) and native.get("files")):
            continue
        if "written" not in native:
            native["written"] = imageio.describe_written(native["files"][0])
        if isinstance(native["written"], dict):
            # what the patient's slides hold together: the end-to-end cost reads from
            # it whether this file is already the patient's merged image (cost.py)
            native["written"] = {**native["written"], "patient_channels": total}
        write_json(result_dir(out, names[variant], by_id[case_id]) / "run.json", rec)
    if not opts.get("stage_images"):  # a selfcheck's pair carries dots, not tissue
        # Clock stopped, writers done: the nuclear channel of each written image, reduced
        # to the evaluation size of the tissue-level block (regbench.tissue). One channel
        # in row bands; a failure here is recorded and changes nothing else.
        from .. import tissue

        for (case_id, variant), rec in done.items():
            native = rec.get("native")
            if not (isinstance(native, dict) and native.get("files")):
                continue
            d = result_dir(out, names[variant], by_id[case_id])
            if by_id[case_id].modality != "fluorescence":
                # the block compares NUCLEAR channels; a brightfield image has none
                rec["tissue"] = {"skipped": tissue.NOT_BRIGHTFIELD}
                write_json(d / "run.json", rec)
                continue
            try:
                rec["tissue"] = tissue.reduce_native(by_id[case_id], native, d)
            except (Exception, SystemExit) as exc:
                rec["tissue"] = {"error": f"{type(exc).__name__}: {exc}"}
            write_json(d / "run.json", rec)
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


def _reuse(out, cases, names, name, label, recorded, realcheck, host, launch_id):
    """``{case_id: [variants]}`` when every result of this unit is already ``ok`` for the
    same configuration (method, label, options, cases), else None. Nothing is run and no
    placeholder is written; each record gains this submission and a ``reused_by`` entry.
    A unit asked for a realcheck is reused only when it has one."""
    found = {}
    ids = [c.case_id for c in cases]
    for case in cases:
        for variant in names.values():
            path = result_dir(out, variant, case) / "run.json"
            try:
                rec = json.loads(path.read_text())
            except (OSError, ValueError):
                return None
            same = (
                rec.get("ok")
                and rec.get("method") == name
                and rec.get("label", "") == (label or "")
                and rec.get("opts") == recorded
                and rec.get("group_cases") == ids
                and (result_dir(out, variant, case) / "warped.npz").exists()
            )
            if not same:
                return None
            found[path] = rec
    with_image = [
        r for r in found.values() if ((r.get("native") or {}).get("files") or [])
    ]
    if realcheck and with_image and not any(r.get("realcheck") for r in with_image):
        return None
    submission = host["submission"]
    for path, rec in found.items():
        known = list(rec.get("submissions") or [])
        if submission and submission not in known:
            known.append(submission)
        rec["submissions"] = known
        rec["reused_by"] = [
            *(rec.get("reused_by") or []),
            {
                "submission": submission,
                "launch_id": launch_id,
                "attempt": host["attempt"],
            },
        ]
        write_json(path, rec)
    print(
        f"{cases[0].dataset}/{'+'.join(ids)} [{label or name}]: every result is "
        "already ok for this configuration: reused, nothing run (REGBENCH_FORCE=1 or "
        "`run --force` runs it again)"
    )
    return {c.case_id: list(names.values()) for c in cases}


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
    resume=False,
):
    """Register one case with a pair method; returns the variants that succeeded.

    ``realcheck`` also checks, after the method's writer, that each mapping follows the
    image written for it (``regbench.selfcheck.realcheck``; ``run.json: realcheck``)."""
    if scope_of(method) != "pair":
        raise SystemExit(
            f"{method_name(method)} runs per patient: use run_group (`run --group`)"
        )
    args = (opts, label, work, keep_work, on_variant, _launch_id(), realcheck, resume)
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
    resume=False,
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
        args = (opts, label, work, keep_work, on_variant, launch, realcheck, resume)
        return _run(method, cases, out, *args)
    done = {}
    for case in cases:
        sub = Path(work) / str(case.case_id) if work else None
        args = (opts, label, sub, keep_work, on_variant, launch, realcheck, resume)
        done.update(_run(method, [case], out, *args))
    return done
