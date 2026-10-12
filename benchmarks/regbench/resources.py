"""What a registration cost: wall time, CPU time and peak memory of the WHOLE process tree.

A method's work is not confined to one process. STARE fans its tiles out over a
``multiprocessing`` pool, VALIS runs a JVM and native thread pools, DeeperHistReg runs torch
threads and optionally a GPU. ``time.perf_counter`` and ``ru_maxrss`` of the parent alone
would therefore charge STARE a fraction of its real CPU and memory and the others nearly
all of theirs. So a sampler thread walks the tree (this process and every descendant)
several times a second and keeps:

``wall_s``         elapsed time
``peak_rss_mb``    the largest *simultaneous* sum of resident set sizes over the tree
``gpu_peak_mb``    torch's peak allocated CUDA memory, when torch is loaded and has a GPU

**CPU time** (``cpu_s``, user + system) is not sampled: it is a counter the kernel keeps,
read at ``start``, at every ``pause`` / ``resume`` and at every ``snapshot``, and billed as
differences. ``cpu_instrument`` names the counter, the first of these that can be read:

``cgroup_v2``      ``usage_usec`` of ``cpu.stat`` in this task's own cgroup, which
                   "account[s] for all the processes in the cgroup (including those in
                   descendant cgroups)" (kernel.org, Documentation/admin-guide/
                   cgroup-v2.rst `[PARTIAL: read the "CPU Interface Files" entry of
                   cpu.stat; the rest of the document not read]`). Every process of the
                   task is in it, however short its life.
``cgroup_v1``      ``cpuacct.usage`` of the task's cpuacct group, "the CPU time (in
                   nanoseconds) obtained by this group", children's included
                   (kernel.org, Documentation/admin-guide/cgroup-v1/cpuacct.rst
                   `[FULL]`).
``rusage+live``    no such counter (a laptop, macOS, a site that does not put a job in a
                   cgroup of its own): ``os.times()`` of this process, which is its own
                   threads plus "all children of the calling process that have
                   terminated and been waited for", grandchildren included "if all of
                   the intervening descendants waited on their terminated children"
                   (getrusage(2), man7.org `[PARTIAL: the RUSAGE_SELF / RUSAGE_CHILDREN /
                   ru_utime / ru_stime entries and the SIGCHLD history note, as quoted by
                   a summarising fetch; the rest of the page not read]`; times(2),
                   man7.org `[PARTIAL: DESCRIPTION and the Linux notes read; the rest
                   not]`). To that is added, for every descendant still alive, its own
                   time and that of the children IT has waited for (``cutime`` /
                   ``cstime`` of ``/proc/<pid>/stat``, proc_pid_stat(5), man7.org
                   `[PARTIAL: fields 14-17 read]`). A process is in exactly one of the
                   two sums: waited for, or alive. The tree is read parents first, so a
                   child that ends between two reads is missed until the next reading
                   and never counted twice. What it cannot see: a child nobody waits
                   for, and (macOS only, where a live process does not report its
                   waited children) the finished children of a child that is still
                   running, until that child ends. And a child's time enters the
                   waited-for sum when it is REAPED, not when it was spent: a child
                   that exits before a pause and is reaped during it has what the
                   last reading of it alive had not seen billed to the paused stretch
                   (the writer's, or the benchmark's), not to the registration.
``rusage``         the same without a way to read live descendants (no psutil, no
                   ``/proc``): exact once every child has been waited for. Here a
                   child that exits before a pause and is reaped during it has ALL
                   its time billed to the paused stretch. The cgroup counters count
                   time when it is spent and have neither effect.
``polled (may undercount)``  no ``os.times`` children figures (Windows): each
                   process's last sampled total, summed. CPU a child spends between its
                   last sample and its exit is lost, and a child that lives between two
                   samples is not seen at all.

The cgroup counter is used only when the cgroup is a scheduler step's (a ``step_*``
component in its path: slurm.schedmd.com/cgroup_v2.html `[PARTIAL: "Slurm cgroup v2
hierarchy" and "Working at the task level" read]`, slurm.schedmd.com/cgroups.html
`[PARTIAL: the hierarchy examples read]`); a desktop session's cgroup holds processes
that are not this run's. It also holds the few processes of the task that are not
this driver's (the batch script's shell).

and, read once at the end, the scheduler's own number where there is one:

``cgroup_peak_mb`` the job cgroup's peak memory (what SLURM enforces ``--mem`` against). It
                   is the WHOLE job's high-water mark, writers included, for every method:
                   it cannot be paused.

**Pausing.** Between :meth:`Monitor.pause` and :meth:`Monitor.resume` nothing is billed: the
wall and CPU time of that stretch are taken out (the counter is read at both ends; what
it advanced by is ``paused_cpu_s``) and the memory peak is not updated. The
driver pauses while it warps the benchmark's points and while a method's own writer runs,
so nothing a paused stretch allocates AND frees enters ``peak_rss_mb``. Memory that a
paused stretch still holds after ``resume`` is seen again from then on: where a method's
writer runs between two results (VALIS, ASHLAR), what that writer leaves resident is in
the peak of every LATER result of the same launch. The first result written, and every
method whose writer runs after the generator has ended (STARE, PALOM, DeeperHistReg),
have a peak without any writer.

RSS summed over processes counts pages shared between them once per process, so
``peak_rss_mb`` is an upper bound for a forked pool; STARE's pool is spawned, which shares
little. ``rss_method`` records how the tree was read: ``psutil``, ``proc`` (Linux
``/proc``, no dependency), or ``rusage`` (neither available: the parent's own high-water
mark plus the largest single child -- a LOWER bound that cannot be paused, flagged so it is
not compared).

**Out of memory.** The kernel may kill one worker of a job and leave its parent waiting
until the time limit, so the driver polls the task's own cgroup counter
(:func:`oom_events_path`, :func:`read_oom_kills`) and ends the run when it rises.
:data:`MEMORY_ERRORS` is the fixed list of memory errors a method reports itself.

**Threads.** No thread variable is exported. :func:`n_cpus` is the allocation, and it is
what a method's own core-count option is set to.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from pathlib import Path

MB = 1024.0 * 1024.0


def _tree_psutil(psutil):
    """``{pid: (rss bytes, cpu seconds, cpu seconds of its waited-for children)}`` for
    this process and every descendant, each parent read before its children."""
    me = psutil.Process()
    out = {}
    for p in [me, *me.children(recursive=True)]:
        try:
            t = p.cpu_times()
            waited = getattr(t, "children_user", 0.0) + getattr(
                t, "children_system", 0.0
            )
            out[p.pid] = (p.memory_info().rss, t.user + t.system, waited)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return out


def _tree_proc():
    """The same from Linux ``/proc``. The tree is found in one pass; each member is then
    read again, a parent before its children."""
    tick, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")

    def stat(pid):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        # the command name is parenthesised and may itself contain spaces or parentheses
        f = raw[raw.rindex(")") + 2 :].split()
        return (
            int(f[1]),
            int(f[21]) * page,
            (int(f[11]) + int(f[12])) / tick,
            (int(f[13]) + int(f[14])) / tick,
        )

    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            got = stat(d)
            if got is not None:
                kids.setdefault(got[0], []).append(int(d))
    out, todo = {}, [os.getpid()]
    while todo:
        pid = todo.pop()
        if pid in out:
            continue
        got = stat(pid)
        if got is not None:
            out[pid] = got[1:]
            todo += kids.get(pid, [])
    return out


def _reader():
    try:
        import psutil

        return "psutil", lambda: _tree_psutil(psutil)
    except ImportError:
        pass
    if Path("/proc/self/stat").exists():
        return "proc", _tree_proc
    return "rusage", None


# where this process's cgroup is named, and where the cgroup file systems are mounted
PROC_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_file(v2_name, controller, v1_name):
    """``(file, version, cgroup path)`` of one interface file of this process's own
    cgroup, or None: ``v2_name`` in the unified hierarchy, ``v1_name`` in the v1
    hierarchy of ``controller``. The cgroup is the one ``/proc/self/cgroup`` names."""
    try:
        lines = PROC_CGROUP.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            _, controllers, path = line.split(":", 2)
        except ValueError:
            continue
        rel = path.strip().lstrip("/")
        if controllers == "":  # cgroup v2
            found = [(CGROUP_ROOT / rel / v2_name, 2)]
        elif controller in controllers.split(","):
            # a v1 hierarchy is mounted under its controller's name, or under the names
            # of the controllers mounted together ("cpu,cpuacct")
            found = [
                (CGROUP_ROOT / name / rel / v1_name, 1)
                for name in dict.fromkeys((controller, controllers))
            ]
        else:
            continue
        for f, version in found:
            if f.exists():
                return f, version, path.strip()
    return None


def _cgroup_peak_mb():
    found = _cgroup_file("memory.peak", "memory", "memory.max_usage_in_bytes")
    try:
        return int(found[0].read_text()) / MB if found else None
    except (OSError, ValueError):
        return None


# ── CPU time: the counter, and which one ──────────────────────────────────────
CGROUP_V2, CGROUP_V1 = "cgroup_v2", "cgroup_v1"
RUSAGE_LIVE, RUSAGE = "rusage+live", "rusage"
POLLED = "polled (may undercount)"
CPU_INSTRUMENTS = (CGROUP_V2, CGROUP_V1, RUSAGE_LIVE, RUSAGE, POLLED)
# a registration that ran as scheduler jobs of its own: each stage command's rusage
STAGE_JOBS = "wait4 of each stage command"


def cgroup_cpu_counter():
    """``(file, version)`` of the CPU counter of this task's own cgroup, or None when
    there is none or the cgroup is not a scheduler step's (module docstring)."""
    found = _cgroup_file("cpu.stat", "cpuacct", "cpuacct.usage")
    if found is None:
        return None
    f, version, path = found
    if not any(part.startswith("step_") for part in path.split("/")):
        return None
    return f, version


def read_cgroup_cpu_s(path, version):
    """CPU seconds a cgroup has used so far: ``usage_usec`` of ``cpu.stat`` (v2,
    microseconds) or ``cpuacct.usage`` (v1, nanoseconds). OSError or ValueError when it
    cannot be read."""
    text = Path(path).read_text()
    if version == 1:
        return int(text.split()[0]) / 1e9
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec":
            return int(parts[1]) / 1e6
    raise ValueError(f"no usage_usec in {path}")


def _waited_children_counted():
    """Does ``os.times()`` carry the CPU time of waited-for children here?"""
    return sys.platform != "win32" and hasattr(os, "times")


def _gpu_peak_mb():
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / MB
    except Exception:
        pass
    return None


def _affinity():
    """``(count, how)``: the cores this process may run on, and how that was read."""
    try:
        return len(os.sched_getaffinity(0)), "sched_getaffinity"
    except AttributeError:  # macOS: no affinity mask
        return os.cpu_count() or 1, "cpu_count"


def cpu_counts():
    """Both numbers behind :func:`n_cpus`, as recorded in ``run.json``.

    ``cpus_requested`` is ``SLURM_CPUS_PER_TASK`` (None outside SLURM); ``cpus_affinity`` is
    the affinity mask. Whether a site confines a job to its cores is the site's setting, so
    the two can differ.
    """
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        requested = int(raw) if raw else None
    except ValueError:
        requested = None
    affinity, how = _affinity()
    return {
        "cpus_requested": requested,
        "cpus_affinity": affinity,
        "cpus_affinity_from": how,
        "cpus_node": os.cpu_count(),
    }


def n_cpus():
    """The allocation: ``min(SLURM_CPUS_PER_TASK, affinity)``, or the affinity outside SLURM."""
    c = cpu_counts()
    if c["cpus_requested"]:
        return max(1, min(c["cpus_requested"], c["cpus_affinity"]))
    return max(1, c["cpus_affinity"])


def _blas():
    """The BLAS NumPy was built against (ASHLAR's clean rows depend on its rounding)."""
    try:
        import numpy as np

        dep = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{dep.get('name', '?')} {dep.get('version', '?')}"
    except Exception:
        return ""


def _mem_requested_gb():
    """The memory request of this attempt: the launcher's rung, else SLURM's own figure."""
    raw = os.environ.get("REGBENCH_MEM_GB")
    try:
        if raw:
            return float(raw)
        per_node = os.environ.get("SLURM_MEM_PER_NODE")  # megabytes
        return float(per_node) / 1024.0 if per_node else None
    except ValueError:
        return None


def _gpu():
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_name(0), int(torch.cuda.device_count())
    except Exception:
        pass
    return "", 0


def gpu_info():
    """``{"gpu_model", "gpus"}`` once torch is loaded; empty and 0 otherwise."""
    model, count = _gpu()
    return {"gpu_model": model, "gpus": count}


def host_info():
    """What the run had: host, CPU, BLAS, the allocation and the ladder's rung."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        attempt = int(os.environ.get("REGBENCH_ATTEMPT") or 1)
    except ValueError:
        attempt = 1
    array_job, task = (
        os.environ.get("SLURM_ARRAY_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    job = (
        f"{array_job}_{task}" if array_job and task else os.environ.get("SLURM_JOB_ID")
    )
    counts = cpu_counts()
    try:
        retry = int(os.environ.get("REGBENCH_RETRY") or 0)
    except ValueError:
        retry = 0
    return {
        "host": platform.node(),
        "cpu_model": cpu,
        "blas": _blas(),
        "n_cpus": n_cpus(),
        "cpus_requested": counts["cpus_requested"],
        "cpus_affinity": counts["cpus_affinity"],
        "cpus_node": counts["cpus_node"],
        "attempt": attempt,
        # 1 for the second try of a rung whose first was lost to the scheduler
        "retry": retry,
        # the submission this launch belongs to (the cohort launcher's id); "" outside
        "submission": os.environ.get("REGBENCH_SUBMISSION") or "",
        "time_limit": os.environ.get("REGBENCH_TIME_LIMIT") or "",
        # the scheduler's log of this task (set by the batch scripts): where a human
        # reads what the failure log's row is about; "" outside the launcher
        "log_path": os.environ.get("REGBENCH_LOG") or "",
        "mem_requested_gb": _mem_requested_gb(),
        # the scheduler's job id, or this process when there is no scheduler
        "launch_id": job or f"pid{os.getpid()}",
        # recorded, never set: the benchmark exports no thread variable
        "threads_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


# ── out of memory ─────────────────────────────────────────────────────────────
# Substrings of the memory errors a method reports itself; any of them makes the run's cause
# ``oom``. Frozen: changing the list after a cohort run is a post hoc change.
#   "Java heap space"       java.lang.OutOfMemoryError from the JVM VALIS starts; it comes
#                           back as a plain exit 1 (conf/modules.config:460-464, plan A8)
#   "MemoryError"           Python's and NumPy's (numpy.core._exceptions._ArrayMemoryError),
#                           and every java.lang.OutOfMemoryError whatever its message
#                           ("GC overhead limit exceeded"): the class name ends in it.
#                           A CUDA one is looked for first.
#   "Unable to allocate"    NumPy's message for the same, where only the text survives
#                           (a package that prints its exception and returns) `[MEMORY]`
#   "DefaultCPUAllocator"   PyTorch's CPU allocation failure (c10 alloc_cpu.cpp `[MEMORY]`:
#                           the wording was not re-read from the torch source this session)
#   "Cannot allocate memory"  strerror(ENOMEM): a failed fork or mmap (`errno(3)`
#                           `[MEMORY]`)
#   "std::bad_alloc"        a C++ allocation failure surfacing through a binding
MEMORY_ERRORS = (
    "Java heap space",
    "MemoryError",
    "Unable to allocate",
    "DefaultCPUAllocator",
    "Cannot allocate memory",
    "std::bad_alloc",
)
GPU_MEMORY_ERRORS = ("CUDA out of memory",)
# Every cause a run.json can end with (the scorer adds ``not_run``):
#   exception      the method raised, or reported a failed round
#   oom            the method ran out of host memory while registering
#   writer_oom     the method's own writer ran out of memory (no image: a failed row)
#   writer         the method's own writer failed for another reason (no image)
#   gpu_oom, timeout
#   killed         the process died and nobody could say why (the placeholder)
#   benchmark_oom  the BENCHMARK's own step ran out of memory with the clock stopped
#                  (moving the nuclei through the map): not the method's failure
#   infrastructure the scheduler lost the attempt (node failure, preemption, a cancelled
#                  job, a task that never started): not the method's failure
CAUSES = (
    "exception",
    "killed",
    "not_run",
    "oom",
    "timeout",
    "gpu_oom",
    "writer",
    "writer_oom",
    "benchmark_oom",
    "infrastructure",
    # a stage job of the registration never reached the queue (the launcher's tile
    # submitter was refused or did not finish): the launcher's failure
    "not_submitted",
    # the benchmark's own step failed or was ended, and not for memory: moving the
    # points, or the last job of a round whose registration ran in other jobs
    "benchmark",
)
# more memory is the remedy: the unit goes up a rung (regbench.ladder)
MEMORY_CAUSES = ("oom", "writer_oom", "benchmark_oom")
# counted apart from a method's failures in every table that counts failures
NOT_THE_METHOD = ("infrastructure", "benchmark_oom", "not_submitted", "benchmark")


def classify_text(text):
    """The cause a captured error TEXT names: ``gpu_oom``, ``oom`` or ``exception``.
    For a package that prints its exception and carries on (VALIS)."""
    text = str(text or "")
    if any(s in text for s in GPU_MEMORY_ERRORS):
        return "gpu_oom"
    if any(s in text for s in MEMORY_ERRORS):
        return "oom"
    return "exception"


class InputUnreadable(OSError):
    """A file the case names cannot be read from where the run is (a folder that is
    not bound into the container, a path of another machine). The method never saw
    its input: the failure is not its own (``infrastructure``)."""

    regbench_cause = "infrastructure"


def classify_error(exc):
    """The ``cause`` of a method's own failure: ``gpu_oom``, ``oom``, ``timeout`` or ``exception``.

    A GPU out-of-memory is looked for first: ``torch.cuda.OutOfMemoryError`` contains the
    text ``MemoryError`` and must not be read as a host out-of-memory, which would move the
    run up the ladder. Only ``oom`` escalates.
    """
    import subprocess

    declared = getattr(exc, "regbench_cause", None)
    if declared in CAUSES:  # an error that says whose it is (`InputUnreadable`)
        return declared
    text = f"{type(exc).__name__}: {exc}"
    if any(s in text for s in GPU_MEMORY_ERRORS):
        return "gpu_oom"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(exc, MemoryError) or any(s in text for s in MEMORY_ERRORS):
        return "oom"
    return "exception"


def oom_events_path():
    """The file holding this task's cgroup ``oom_kill`` counter, or None.

    Read from the path in ``/proc/self/cgroup``: ``memory.events`` on cgroup v2,
    ``memory.oom_control`` on v1 (Linux kernel cgroup documentation `[MEMORY]`: file names
    and the ``oom_kill`` key as remembered, not re-read this session; untested on a cluster).
    """
    found = _cgroup_file("memory.events", "memory", "memory.oom_control")
    return found[0] if found else None


def read_oom_kills(path):
    """The ``oom_kill`` count in a cgroup events file; None when it cannot be read."""
    if path is None:
        return None
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "oom_kill":
                return int(parts[1])
    except (OSError, ValueError):
        pass
    return None


class Monitor:
    """Sample the process tree from ``start()`` until ``stop()``; ``snapshot()`` at any point.

    ``cpu_instrument`` forces a CPU counter (a name in :data:`CPU_INSTRUMENTS`); left
    out, the first that can be read is taken at ``start`` (module docstring)."""

    def __init__(self, interval=0.2, cpu_instrument=None):
        self.interval = interval
        self.method, self._read = _reader()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.cpu_instrument = cpu_instrument
        self._counter = None  # (file, version) of the cgroup's CPU counter
        self._cpu = {}  # pid -> last sampled cpu seconds (the polled instrument only)
        self._cpu_last = 0.0  # the counter's last reading: it never runs backwards
        self.cpu_read_failures = 0
        self._cpu0 = 0.0
        self._peak = 0.0
        self._paused_wall = self._paused_cpu = 0.0
        self._pause_at = None  # (wall, cpu) clocks when the current pause began

    def _choose(self):
        """The CPU counter of this run: the forced one, else the first readable."""
        asked = self.cpu_instrument
        if asked in (None, CGROUP_V2, CGROUP_V1):
            found = cgroup_cpu_counter()
            try:
                if found is not None:
                    read_cgroup_cpu_s(*found)
                    self._counter = found
                    return CGROUP_V2 if found[1] == 2 else CGROUP_V1
            except (OSError, ValueError):
                pass
        if asked in (None, RUSAGE_LIVE, RUSAGE) and _waited_children_counted():
            return RUSAGE if self._read is None or asked == RUSAGE else RUSAGE_LIVE
        if asked in (None, POLLED) and self._read is not None:
            return POLLED
        return RUSAGE if _waited_children_counted() else POLLED

    def _sample(self):
        if self._read is None:
            return
        try:
            tree = self._read()
        except Exception:
            return
        with self._lock:
            if self._pause_at is None:
                self._peak = max(self._peak, sum(v[0] for v in tree.values()))
            if self.cpu_instrument == POLLED:
                for pid, v in tree.items():
                    self._cpu[pid] = max(self._cpu.get(pid, 0.0), v[1])

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _counter_now(self):
        how = self.cpu_instrument
        if how in (CGROUP_V2, CGROUP_V1):
            return read_cgroup_cpu_s(*self._counter)
        if how == POLLED:
            self._sample()
            with self._lock:
                return sum(self._cpu.values())
        # this process and the children it has waited for, read BEFORE the tree: a
        # child that is waited for in between is then in neither sum, never in both
        t = os.times()
        total = t.user + t.system + t.children_user + t.children_system
        if how == RUSAGE_LIVE:
            me = os.getpid()
            total += sum(v[1] + v[2] for pid, v in self._read().items() if pid != me)
        return total

    def cpu_clock(self):
        """The CPU counter now, in seconds: differences of it are CPU time used by the
        processes it covers, whether the monitor is paused or not."""
        try:
            now = self._counter_now()
        except Exception:
            self.cpu_read_failures += 1
            now = self._cpu_last
        self._cpu_last = max(self._cpu_last, now)
        return self._cpu_last

    def start(self):
        self.cpu_instrument = self._choose()
        if self._read is not None:
            self._sample()
            self._peak = 0.0
        # CPU spent before the method started is not its own
        self._cpu0 = self.cpu_clock()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    @property
    def paused(self):
        return self._pause_at is not None

    def pause(self):
        """Stop billing: the peak so far is kept, and nothing after it counts until ``resume``."""
        if self._pause_at is not None:
            return self
        self._sample()  # the peak up to this instant belongs to the method
        clocks = (time.perf_counter(), self.cpu_clock())
        with self._lock:
            self._pause_at = clocks
        return self

    def resume(self):
        if self._pause_at is None:
            return self
        wall0, cpu0 = self._pause_at
        self._paused_wall += time.perf_counter() - wall0
        self._paused_cpu += self.cpu_clock() - cpu0
        with self._lock:
            self._pause_at = None
        return self

    @property
    def paused_cpu_s(self):
        """CPU of every paused stretch so far, the current one up to now included."""
        if self._pause_at is None:
            return self._paused_cpu
        return self._paused_cpu + self.cpu_clock() - self._pause_at[1]

    def snapshot(self):
        """The bill so far. Taken during a pause, it is the bill at the moment of the pause."""
        self._sample()
        paused_cpu = self.paused_cpu_s
        if self._pause_at is not None:
            wall_now, cpu_now = self._pause_at
        else:
            wall_now, cpu_now = time.perf_counter(), self.cpu_clock()
        wall = wall_now - self._t0 - self._paused_wall
        cpu = cpu_now - self._cpu0 - self._paused_cpu
        if self._read is None:
            import resource

            # bytes on macOS, KiB on Linux
            scale = 1.0 if sys.platform == "darwin" else 1024.0
            peak = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            ) * scale
        else:
            with self._lock:
                peak = self._peak
        return {
            "wall_s": wall,
            "cpu_s": cpu,
            # how cpu_s was read, and the CPU of the stretches taken off it (a writer
            # the method ran with the clock stopped, the benchmark's own steps)
            "cpu_instrument": self.cpu_instrument,
            "paused_cpu_s": paused_cpu,
            "cpu_read_failures": self.cpu_read_failures,
            "peak_rss_mb": peak / MB,
            "gpu_peak_mb": _gpu_peak_mb(),
            "rss_method": self.method,
        }

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return {
            "cgroup_peak_mb": _cgroup_peak_mb(),
            # the whole launch as the same counter read it, paused stretches included:
            # what the scheduler's own figure of the task is compared with
            "launch_cpu_s": self.cpu_clock() - self._cpu0,
        }
