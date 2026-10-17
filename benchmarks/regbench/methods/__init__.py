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
``info["_cost"]``  for a registration that ran in scheduler jobs of its own: the
        ``wall_s`` / ``cpu_s`` / ``peak_rss_mb`` those jobs recorded, which replace this
        process's own measurement in ``run.json`` (``register_s``, ``resources``).
``info["band_mov_nuc"]`` / ``info["band_ref_nuc"]``  the same seam rule applied to the
        BARE nuclei of that side (``cases.polys(points, side, "nuclei")``). Used only
        where the case does not say which outline each nucleus belongs to
        (``_nuclei_band``); stored as ``band_nuc`` / ``band_ref_nuc``.
``info["band_mov"]`` / ``info["band_ref"]``  a seam mask: one bool per moving (reference)
        nucleus, True where the nucleus straddles one of the method's own joins (ASHLAR's
        tile overlaps, PALOM's block edges). Stored in ``warped.npz`` as ``band`` /
        ``band_ref``; the scorer scores EVERY variant of the case a second time without
        those nuclei (``<variant>:non_<owner>``).
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
``oom`` / ``gpu_oom`` / ``timeout`` when the error says so (``resources.classify_error``). An error that carries ``regbench_no_rerun`` (a stage
job that failed after its own retries, ``regbench.stages``) is recorded with
``no_rerun: true``: the ladder runs such a unit at no further rung.
A watcher polls the task's cgroup out-of-memory counter from before the method starts
until its writer has finished, and ends the run when it rises, because the kernel may
kill one worker and leave the parent waiting. ``run.json: oom_watch`` says whether the
counter was readable and the watcher started.

**Three stretches, three memory causes** (all three go up the ladder):

``oom``            the method ran out of memory while registering
``writer_oom``     its own writer did: ``write_native``, a writer that reported a memory
                   error at a yield (``native: {"error": ...}``), or a kill the watcher
                   saw while the method itself had the clock paused
``benchmark_oom``  the benchmark's own step did, with the clock stopped (its own lines
                   around the map: the lattice, the arrays, the files): not the
                   method's failure. A kill the watcher saw while the method's own
                   point mapping was running is ``oom``

**Whose a failure at a yield is.** The method's point mapping is the method's code: an
error it raises, a memory error inside it, or a result that is not one X, Y pair per
point fails the row under the method's own cause (``MappingFailed``: ``exception``,
``oom``, ``gpu_oom``, ``timeout``). Only an error of the benchmark's own lines around
that call is ``benchmark`` / ``benchmark_oom``.

**A row is ok only when its image exists.** A method's writer that fails, at a yield or
in ``write_native``, fails the row it was writing for: ``writer_oom``, or ``writer`` for
any other error. While ``write_native`` runs, that row's ``run.json`` says ``ok: false,
cause: killed, phase: writer``, so a kill there leaves a failed row too. The rule is the
same for every method.

**A unit that is already done is not run again** (``resume=True``, what ``run`` passes):
when every result of the unit is ``ok`` for the same method, label, options, cases and
points, AND was computed the same way from the same bytes (``reuse_key``, kept in
``run.json: reuse_key``), nothing runs and no placeholder is written; the records gain
this submission's id (``submissions``, ``reused_by``). ``REGBENCH_FORCE=1``
(``run --force``) runs it anyway. ``reuse_key`` holds, for every method: ``execution``
(``one process`` for a method that registers in one; for a method that declares
``STAGES``, STARE, who starts its stage processes as ``REGBENCH_STAGES`` states it:
``jobs`` or ``local``), ``code`` (a sha256 of the
driver, the adapter module, the sibling modules it imports and the ``regbench`` modules
it imports directly: ``code_files``), ``inputs`` (``cases.inputs_fingerprint`` of every
case: size, time and sampled bytes of its image files) and ``method`` (what the adapter
adds through ``REUSE_KEY(cases, out, opts, label)``: STARE's version and the identity
of its SOLVE job). The version of a package inside an image is NOT in the key unless
the adapter puts it there: a rebuilt image is a reason for ``REGBENCH_FORCE=1``.
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

from .. import METHODS, billing, plugins, resources
from ..cases import (
    load_points,
    points_fingerprint,
    result_dir,
    stale_points,
    write_json,
    write_npz,
)
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
# ("mapping": the clock is stopped and the method's own point mapping is running)
MEMORY_CAUSE = {
    "method": "oom",
    "mapping": "oom",
    "writer": "writer_oom",
    "benchmark": "benchmark_oom",
}


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


# Methods with NO input rule for a brightfield case (an RGB image pair), and why, word for
# word as they were written at 5a46f778 (methods/__init__.py). Their adapters refuse such a
# case with this text (``require_fluorescence``). Every other method has a rule: both
# VALIS adapters read the original RGB slides and VALIS picks its own brightfield
# processing (methods/valis_mirage.py, methods/valis.py), DeeperHistReg reads the RGB
# image with its shipped parameters (methods/deeperhistreg.py), and STARE reads a
# one-channel proxy of each RGB slide (methods/stare.py).
NO_BRIGHTFIELD = {
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


def input_rule(method, modality):
    """None when ``method`` has an input rule for a case of ``modality`` (it is
    accepted), else the reason it has none (``NO_BRIGHTFIELD``). Every method takes a
    fluorescence case."""
    if modality == "fluorescence":
        return None
    return NO_BRIGHTFIELD.get(method)


def require_fluorescence(case, method):
    """Refuse a case of another modality for a method without an input rule for it
    (``NO_BRIGHTFIELD``); every other case passes. The name is older than the rule (it
    refused every brightfield case once) and is kept because two adapters that are not
    edited for the rule call it (``methods/ashlar.py``, ``methods/palom.py``)."""
    reason = input_rule(method, case.modality)
    if reason is not None:
        raise ValueError(
            f"{case.case_id}: {method} has no input rule for a {case.modality} case: "
            f"{reason}"
        )


def check_inputs(cases):
    """Refuse a case whose image files cannot be read from here, before the method
    starts. A case names its ORIGINAL slides where they were when it was prepared (the
    cohort's own folders); when such a folder is not bound
    into the method's container, every unit of every method that reads them would fail
    there and be counted as the method's failure. It is the launch's fault, not a
    method's: ``resources.InputUnreadable``, cause ``infrastructure``."""
    for case in cases:
        for path in (
            case.ref_image,
            case.mov_image,
            case.ref_nuclear,
            case.mov_nuclear,
        ):
            if path and not os.access(str(path), os.R_OK):
                raise resources.InputUnreadable(
                    f"{case.dataset}/{case.case_id}: {path} cannot be read from where "
                    "this run is. Its folder is not bound into the image, or the cases "
                    "were prepared on another machine: name the slides' folder in "
                    "REGBENCH_BIND, with the path "
                    "the cases were prepared with."
                )


def load(method):
    """The method's module. A module-like object (tests) is returned as it is."""
    if not isinstance(method, str):
        return method
    private = plugins.method(method)  # a private method plug-in (regbench.plugins)
    if private is not None:
        return private
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; expected one of {METHODS}")
    return importlib.import_module(f"{__name__}.{method}")


def code_files(method):
    """The source files whose bytes are the ``code`` of a method's ``reuse_key``: this
    driver, the adapter module, the modules of ``methods/`` it imports (followed
    through), and the ``regbench`` modules the adapter imports directly. ``[]`` beyond
    the driver for a module-like object without a file (a test's stand-in)."""
    import ast

    here = Path(__file__).resolve()
    package, root = here.parent, here.parent.parent
    mod = load(method)
    start = getattr(mod, "__file__", None)
    files, todo = [here], [Path(start).resolve()] if start else []
    while todo:
        path = todo.pop()
        if path in files or not path.is_file():
            continue
        files.append(path)
        if path.parent != package:
            continue  # a regbench module: its bytes, not what it imports in turn
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level not in (1, 2):
                continue
            base = package if node.level == 1 else root
            names = [node.module] if node.module else [a.name for a in node.names]
            if node.module:  # `from .x import y`: y may be a module of the package x
                names += [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                target = base.joinpath(*name.split("."))
                for found in (target.with_suffix(".py"), target / "__init__.py"):
                    if found.is_file():
                        todo.append(found)
    return files


def code_fingerprint(method):
    """sha256 over ``code_files``: names (relative to ``regbench``) and bytes."""
    import hashlib

    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for path in sorted(code_files(method)):
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = path.name  # a private plug-in outside the package
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def reuse_key(method, cases, out, opts, label):
    """What an ``ok`` result must have been computed WITH to stand for this run
    (module docstring, "A unit that is already done"). JSON-able."""
    from ..cases import inputs_fingerprint

    mod = load(method)
    hook = getattr(mod, "REUSE_KEY", None)
    return _jsonable(
        {
            # who starts the stage processes, for a method that has stages (it
            # declares STAGES); any other method is one process whatever a stray
            # REGBENCH_STAGES of the caller's shell says
            "execution": (
                os.environ.get("REGBENCH_STAGES") or "one process"
                if getattr(mod, "STAGES", None)
                else "one process"
            ),
            "code": code_fingerprint(method),
            "inputs": {c.case_id: inputs_fingerprint(c) for c in cases},
            "method": hook(cases, out, opts, label) if hook else None,
        }
    )


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


# `--opt` keys every method takes: the driver's own
COMMON_OPTIONS = ("workers", "stage_images")


def known_options(method):
    """The ``--opt`` keys ``method`` reads: the driver's own and the adapter's
    ``OPTIONS`` (a module constant; a public adapter without one has no option of its
    own). None for a private plug-in that declares none: its keys are not known here,
    and are not checked."""
    mod = load(method)
    own = getattr(mod, "OPTIONS", None)
    if own is None and isinstance(method, str) and method not in METHODS:
        return None
    return tuple(dict.fromkeys((*COMMON_OPTIONS, *(own or ()))))


def parse_opts(pairs, method=None):
    """``{key: value}`` of ``--opt key=value`` items. With ``method``: a key that the
    method does not read is refused here, before any work (it would otherwise be
    ignored, and the run would be the default one under another name)."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--opt wants key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    known = known_options(method) if method is not None else None
    unknown = sorted(set(out) - set(known)) if known is not None else []
    if unknown:
        raise SystemExit(
            f"--opt {', '.join(unknown)}: not an option of {method_name(method)}, "
            f"which reads: {', '.join(known)}. Nothing was run."
        )
    return out


def selfcheck_status():
    """The selfcheck status a run was handed (``REGBENCH_SELFCHECK_STATUS``), kept in its
    ``run.json``. The cohort launcher hands none any more (a run never waits for its
    selfcheck), so it reads ``not_checked`` there; the scorer sets the tables' column
    ``selfcheck_status`` from ``logs/verdicts.tsv`` instead (``score.set_selfcheck_status``)."""
    return os.environ.get("REGBENCH_SELFCHECK_STATUS") or "not_checked"


class MappingFailed(RuntimeError):
    """The method's own point mapping failed: it raised, or it returned something that
    is not one X, Y pair per point. The method's failure, with the cause its error
    names (``resources.classify_error``: ``exception``, ``oom``, ``gpu_oom``,
    ``timeout``)."""

    def __init__(self, message, cause="exception"):
        super().__init__(message)
        self.cause = cause


def _warp_chunked(fn, xy, chunk=2_000_000, inside=None):
    """``fn`` over ``xy`` in chunks. Whatever goes wrong INSIDE ``fn``, or with what it
    returns, is raised as :class:`MappingFailed`; an error of this function's own lines
    is raised as it is (the benchmark's). ``inside(True)`` / ``inside(False)`` is
    called around each call of ``fn``: the driver's note of whose code is running."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return xy
    out = []
    for i in range(0, len(xy), chunk):
        part = xy[i : i + chunk]
        try:
            if inside is not None:
                inside(True)
            try:
                got = fn(part)
            finally:
                if inside is not None:
                    inside(False)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as exc:
            raise MappingFailed(
                f"the method's point mapping raised {type(exc).__name__}: {exc}",
                resources.classify_error(exc),
            ) from exc
        try:
            got = np.asarray(got, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MappingFailed(
                f"the method's point mapping returned something that is not numbers "
                f"for {len(part)} points: {type(exc).__name__}: {exc}"
            ) from exc
        if got.size != part.size:
            raise MappingFailed(
                f"the method's point mapping returned an array of shape {got.shape} "
                f"for {len(part)} points (one X, Y pair per point is {part.shape})"
            )
        out.append(got.reshape(-1, 2))
    return np.concatenate(out)


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
    try:
        key = reuse_key(method, cases, out, recorded, label)
    except (Exception, SystemExit) as exc:
        # an adapter that cannot say what its result rests on (a refused option, an
        # unreadable file): nothing is reused, and the run below says why it fails
        key = {"error": f"{type(exc).__name__}: {exc}"}
    if resume and os.environ.get("REGBENCH_FORCE") != "1" and "error" not in key:
        reused = _reuse(
            out, cases, names, name, label, recorded, realcheck, host, launch_id, key
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

    def mapping(now):
        # the method's own point mapping, inside the benchmark's stretch
        phase["now"] = "mapping" if now else "benchmark"

    watch = _OomWatch(lambda: stuck(), where)
    base = {
        "method": name,
        "label": label or "",
        "opts": recorded,
        "reuse_key": key,
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
            # the points this result is made for: a case prepared again with others
            # makes it stale (never reused, and scored as not run)
            "points_sha256": points_fingerprint(case),
            "previous_attempts": earlier.get((case.case_id, variant), []),
        }

    def fail(case, variant, error, cause, cost=None, tb="", info=None, no_rerun=False):
        d = result_dir(out, names[variant], case)
        (d / "warped.npz").unlink(missing_ok=True)
        # nothing an earlier attempt left may answer for this one (the verdict reads it)
        (d / "realcheck.json").unlink(missing_ok=True)
        rec = {**head(case, variant), "ok": False, "error": error, "cause": cause}
        if cost is not None:
            rec.update(register_s=cost.get("wall_s"), resources=cost)
        if tb:
            rec["traceback"] = tb
        if info:
            rec["info"] = _jsonable(info)
        if no_rerun:
            rec["no_rerun"] = True
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
                        "wall_s": res.get("driver_wall_s", res.get("wall_s")),
                        "cpu_s": res.get("driver_cpu_s", res.get("cpu_s")),
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

    natives, write_s, write_cpu = {}, {}, {}
    # CPU of the stretches the METHOD took off the clock itself (a writer it ran at a
    # yield, and whatever its adapter does beside it with the clock stopped): the
    # monitor's paused CPU less the driver's own stretches. `at_yield` holds the rows
    # whose image was written in such a stretch
    own_paused = {"cpu_s": 0.0, "seen": 0.0}
    at_yield = set()

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
    no_rerun = False
    failure_info = None
    mon.start()
    watch.start()
    base["oom_watch"] = watch.record()
    try:
        check_inputs(cases)
        gen = (
            mod.register(cases[0], work, opts)
            if scope == "pair"
            else mod.register_group(list(cases), work, opts)
        )
        for item in gen:
            mon.pause()
            phase["now"] = "benchmark"
            cost = mon.snapshot()
            own_paused["cpu_s"] += cost["paused_cpu_s"] - own_paused["seen"]
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
            band_nuc_own = info.pop("band_mov_nuc", None)
            band_ref_nuc_own = info.pop("band_ref_nuc", None)
            native = info.pop("native", None)
            # a registration that ran in jobs of its own bills what those jobs
            # recorded: this process's clock saw none of it (regbench.stages)
            billed = info.pop("_cost", None)
            if billed:
                # what THIS process cost is kept beside it: a later attempt of this
                # job wastes that, not the registration the other jobs ran
                cost = {
                    **cost,
                    "cpu_instrument": resources.STAGE_JOBS,
                    **billed,
                    "driver_wall_s": cost.get("wall_s"),
                    "driver_cpu_s": cost.get("cpu_s"),
                    "driver_cpu_instrument": cost.get("cpu_instrument"),
                }
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
                t0, cpu0 = time.perf_counter(), mon.cpu_clock()
                arrays = {
                    "direction": np.array(pmap.direction),
                    "cells_xy": _warp_chunked(
                        pmap.fn,
                        pts.get(f"cells_{side}_xy", np.empty((0, 2))),
                        inside=mapping,
                    ).astype(np.float32),
                    "grid": _warp_chunked(pmap.fn, grid, inside=mapping).astype(
                        np.float32
                    ),
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
                # the SECOND polygon set, the bare nuclei, through the same map in the
                # same direction; and the owner's seam mask on them (_nuclei_band)
                nuclei_band = {}
                if f"nuclei_{side}_xy" in pts:
                    arrays["nuclei_xy"] = _warp_chunked(
                        pmap.fn, pts[f"nuclei_{side}_xy"], inside=mapping
                    ).astype(np.float32)
                for name_nuc, where, flags, own in (
                    ("band_nuc", "mov", band, band_nuc_own),
                    ("band_ref_nuc", "ref", band_ref, band_ref_nuc_own),
                ):
                    got, how = _nuclei_band(pts, where, flags, own)
                    if got is not None:
                        arrays[name_nuc], nuclei_band[where] = got, how
                warp_s, warp_cpu_s = time.perf_counter() - t0, mon.cpu_clock() - cpu0
                if isinstance(native, dict) and native.get("files"):
                    at_yield.add(key)
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
                    # the benchmark's own CPU for it, in no bill of the method
                    "warp_cpu_s": warp_cpu_s,
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
            except MappingFailed as exc:
                # the method's own mapping function: its failure, under its own cause.
                # A memory cause ends the run as the method's `oom` (the unit goes up);
                # any other fails this variant and the run goes on
                if exc.cause in resources.MEMORY_CAUSES:
                    raise _OutOfMemory(
                        "method",
                        f"the point mapping of {variant} ran out of memory: {exc}",
                    ) from exc
                failed[key] = fail(
                    case,
                    variant,
                    f"{variant}: {exc}",
                    exc.cause,
                    cost,
                    traceback.format_exc(),
                )
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
                    "the benchmark's own step failed while it moved the points of "
                    f"{variant} through the map: {type(exc).__name__}: {exc}",
                    "benchmark",
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
                own_paused["seen"] = mon.paused_cpu_s
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
        # already retried where it happened (a stage job): no rung above runs it again
        no_rerun = bool(getattr(exc, "regbench_no_rerun", False))
        # what the failed run cost in jobs of its own, where the method recorded it
        failure_info = getattr(exc, "regbench_info", None)
        if isinstance(exc, _OutOfMemory):
            cause = MEMORY_CAUSE[exc.phase]
        elif watch.rose():
            cause = MEMORY_CAUSE[watch.phase or "method"]
    finally:
        phase["now"] = "method"
        mon.pause()
        cost = mon.snapshot()
        own_paused["cpu_s"] += cost["paused_cpu_s"] - own_paused["seen"]
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

        def lose(case, why, how, trace="", no_rerun=False):
            key = (case.case_id, final)
            rec = done.pop(key)
            final_maps.pop(key, None)
            failed[key] = fail(
                case,
                final,
                why,
                how,
                rec.get("resources"),
                trace,
                rec.get("info"),
                no_rerun,
            )
            # the transform exists: what failed is the writer or the job around it
            failed[key]["registered"] = True
            write_json(result_dir(out, names[final], case) / "run.json", failed[key])

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
            t0, cpu0 = time.perf_counter(), mon.cpu_clock()
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
                own = got.pop("write_s", None) if isinstance(got, dict) else None
                # the writer's own figure where it gives one (a process it timed)
                write_s[case.case_id] = (
                    float(own) if own is not None else time.perf_counter() - t0
                )
                write_cpu[case.case_id] = mon.cpu_clock() - cpu0
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
                # a writer that names its own cause (STARE's STITCH inside the
                # round's last job: the writer's, or the job's memory)
                named = getattr(exc, "regbench_cause", None)
                if named in ("writer", "writer_oom", "benchmark_oom", "benchmark"):
                    how = named
                if how == "writer_oom":
                    dead = text
                lose(
                    case,
                    (
                        "the benchmark's job ran out of memory around the method's "
                        "own writer: "
                        if how == "benchmark_oom"
                        else "the method's own writer failed: "
                    )
                    + text,
                    how,
                    traceback.format_exc(),
                    bool(getattr(exc, "regbench_no_rerun", False)),
                )
        phase["now"] = "method"
    watch.stop()
    end = mon.stop()
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    gpu = resources.gpu_info()
    # a writer that ran at a yield: the CPU of the method's own paused stretches of the
    # whole launch, shared between the rows written there as their `write_s` is
    shared = sum(float(done[k].get("write_s") or 0.0) for k in at_yield if k in done)
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
                    rec["write_cpu_s"] = write_cpu[case.case_id]
                    rec["write_cpu_from"] = billing.WRITE_CPU_CALL
                elif key in at_yield and shared > 0:
                    rec["write_cpu_s"] = (
                        own_paused["cpu_s"] * float(rec.get("write_s") or 0.0) / shared
                    )
                    rec["write_cpu_from"] = billing.WRITE_CPU_STRETCH
                write_json(result_dir(out, names[variant], case) / "run.json", rec)
            elif key not in failed:
                failed[key] = fail(
                    case,
                    variant,
                    error or "the method did not produce this variant",
                    cause or "exception",
                    {**cost, **end},
                    tb,
                    info=failure_info,
                    no_rerun=no_rerun,
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


KEY_PARTS = ("execution", "code", "inputs", "method")


def record_gaps(rec, case, result_path, key):
    """What keeps ONE finished result from standing for this run, as a list: ``record``
    (not an ``ok`` record, or its ``warped.npz`` is gone), the parts of ``reuse_key``
    that differ (``KEY_PARTS``; ``key`` if only the whole differs), ``points`` (made for
    points the case no longer has: a forced re-prepare changes them). ``[]`` = reusable.
    The one predicate of ``_reuse`` and of ``regbench reuse`` (``reuse.py``)."""
    if not isinstance(rec, dict) or not rec.get("ok"):
        return ["record"]
    gaps = []
    was = rec.get("reuse_key")
    if was != key:
        if isinstance(key, dict):
            old = was if isinstance(was, dict) else {}
            gaps += [c for c in KEY_PARTS if key.get(c) != old.get(c)]
        gaps = gaps or ["key"]
    made_for = rec.get("points_sha256")
    if not isinstance(made_for, dict) or stale_points(made_for, case):
        gaps.append("points")
    if not (Path(result_path) / "warped.npz").exists():
        gaps.append("record")
    return gaps


def _reuse(out, cases, names, name, label, recorded, realcheck, host, launch_id, key):
    """``{case_id: [variants]}`` when every result of this unit is already ``ok`` for the
    same configuration (method, label, options, cases, points) and was computed the
    same way from the same bytes (``key``: ``reuse_key``), else None. Nothing is run
    and no placeholder is written; each record gains this submission and a
    ``reused_by`` entry. A unit asked for a realcheck is reused only when it has one."""
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
                isinstance(rec, dict)
                and rec.get("method") == name
                and rec.get("label", "") == (label or "")
                and rec.get("opts") == recorded
                and rec.get("group_cases") == ids
                and not record_gaps(rec, case, path.parent, key)
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
