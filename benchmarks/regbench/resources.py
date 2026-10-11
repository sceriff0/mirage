"""What a registration cost: wall time, CPU time and peak memory of the WHOLE process tree.

A method's work is not confined to one process. STARE fans its tiles out over a
``multiprocessing`` pool, VALIS runs a JVM and native thread pools, DeeperHistReg runs torch
threads and optionally a GPU. ``time.perf_counter`` and ``ru_maxrss`` of the parent alone
would therefore charge STARE a fraction of its real CPU and memory and the others nearly
all of theirs. So a sampler thread walks the tree (this process and every descendant)
several times a second and keeps:

``wall_s``         elapsed time
``cpu_s``          user + system CPU seconds summed over the tree, including children that
                   have already exited (each child's last observed total is kept)
``peak_rss_mb``    the largest *simultaneous* sum of resident set sizes over the tree
``gpu_peak_mb``    torch's peak allocated CUDA memory, when torch is loaded and has a GPU

and, read once at the end, the scheduler's own number where there is one:

``cgroup_peak_mb`` the job cgroup's peak memory (what SLURM enforces ``--mem`` against). It
                   is the WHOLE job's high-water mark, writers included, for every method:
                   it cannot be paused.

**Pausing.** Between :meth:`Monitor.pause` and :meth:`Monitor.resume` nothing is billed: the
wall and CPU time of that stretch are taken out and the memory peak is not updated. The
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
    me = psutil.Process()
    out = {}
    for p in [me, *me.children(recursive=True)]:
        try:
            t = p.cpu_times()
            out[p.pid] = (p.memory_info().rss, t.user + t.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return out


def _tree_proc():
    """The same from Linux ``/proc``: ``{pid: (rss bytes, cpu seconds)}`` for self + descendants."""
    tick, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    stats = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            raw = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        # the command name is parenthesised and may itself contain spaces or parentheses
        f = raw[raw.rindex(")") + 2 :].split()
        stats[int(d)] = (int(f[1]), int(f[21]) * page, (int(f[11]) + int(f[12])) / tick)
    kids = {}
    for pid, (ppid, _, _) in stats.items():
        kids.setdefault(ppid, []).append(pid)
    out, todo = {}, [os.getpid()]
    while todo:
        pid = todo.pop()
        if pid in stats and pid not in out:
            out[pid] = stats[pid][1:]
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


def _cgroup_peak_mb():
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if controllers == "":  # cgroup v2
                f = Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.peak"
            elif "memory" in controllers.split(","):
                f = (
                    Path("/sys/fs/cgroup/memory")
                    / path.lstrip("/")
                    / "memory.max_usage_in_bytes"
                )
            else:
                continue
            if f.exists():
                return int(f.read_text()) / MB
    except (OSError, ValueError):
        pass
    return None


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
)
# more memory is the remedy: the unit goes up a rung (regbench.ladder)
MEMORY_CAUSES = ("oom", "writer_oom", "benchmark_oom")
# counted apart from a method's failures in every table that counts failures
NOT_THE_METHOD = ("infrastructure", "benchmark_oom")


def classify_text(text):
    """The cause a captured error TEXT names: ``gpu_oom``, ``oom`` or ``exception``.
    For a package that prints its exception and carries on (VALIS)."""
    text = str(text or "")
    if any(s in text for s in GPU_MEMORY_ERRORS):
        return "gpu_oom"
    if any(s in text for s in MEMORY_ERRORS):
        return "oom"
    return "exception"


def classify_error(exc):
    """The ``cause`` of a method's own failure: ``gpu_oom``, ``oom``, ``timeout`` or ``exception``.

    A GPU out-of-memory is looked for first: ``torch.cuda.OutOfMemoryError`` contains the
    text ``MemoryError`` and must not be read as a host out-of-memory, which would move the
    run up the ladder. Only ``oom`` escalates.
    """
    import subprocess

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
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            _, controllers, path = line.split(":", 2)
        except ValueError:
            continue
        if controllers == "":  # cgroup v2
            f = Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.events"
        elif "memory" in controllers.split(","):
            f = Path("/sys/fs/cgroup/memory") / path.lstrip("/") / "memory.oom_control"
        else:
            continue
        if f.exists():
            return f
    return None


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
    """Sample the process tree from ``start()`` until ``stop()``; ``snapshot()`` at any point."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.method, self._read = _reader()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cpu = {}  # pid -> last observed cpu seconds
        self._cpu0 = 0.0
        self._peak = 0.0
        self._paused_wall = self._paused_cpu = 0.0
        self._pause_at = None  # (wall, cpu) clocks when the current pause began

    def _sample(self):
        if self._read is None:
            return
        try:
            tree = self._read()
        except Exception:
            return
        with self._lock:
            if self._pause_at is None:
                self._peak = max(self._peak, sum(r for r, _ in tree.values()))
            for pid, (_, cpu) in tree.items():
                self._cpu[pid] = max(self._cpu.get(pid, 0.0), cpu)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _cpu_now(self):
        if self._read is None:
            t = os.times()
            return t.user + t.system + t.children_user + t.children_system
        with self._lock:
            return sum(self._cpu.values())

    def start(self):
        if self._read is not None:
            # CPU this process (and any children it already has) spent before the method started
            self._sample()
            self._peak = 0.0
        self._cpu0 = self._cpu_now()
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
        self._sample()  # the peak and the CPU up to this instant belong to the method
        clocks = (time.perf_counter(), self._cpu_now())
        with self._lock:
            self._pause_at = clocks
        return self

    def resume(self):
        if self._pause_at is None:
            return self
        self._sample()  # CPU spent while paused is observed, then taken out below
        wall0, cpu0 = self._pause_at
        self._paused_wall += time.perf_counter() - wall0
        self._paused_cpu += self._cpu_now() - cpu0
        with self._lock:
            self._pause_at = None
        return self

    def snapshot(self):
        """The bill so far. Taken during a pause, it is the bill at the moment of the pause."""
        self._sample()
        if self._pause_at is not None:
            wall_now, cpu_now = self._pause_at
        else:
            wall_now, cpu_now = time.perf_counter(), self._cpu_now()
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
            "peak_rss_mb": peak / MB,
            "gpu_peak_mb": _gpu_peak_mb(),
            "rss_method": self.method,
        }

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return {"cgroup_peak_mb": _cgroup_peak_mb()}
