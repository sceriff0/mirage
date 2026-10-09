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
                   covers the whole task, interpreter start-up and point warping included,
                   and counts shared pages once, so it is the figure to size a job by.

RSS summed over processes counts pages shared between them once per process, so
``peak_rss_mb`` is an upper bound for a forked pool; STARE's pool is spawned, which shares
little. ``rss_method`` records how the tree was read: ``psutil``, ``proc`` (Linux
``/proc``, no dependency), or ``rusage`` (neither available: the parent's own high-water
mark plus the largest single child -- a LOWER bound, flagged so it is not compared).
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
                f = Path("/sys/fs/cgroup/memory") / path.lstrip("/") / "memory.max_usage_in_bytes"
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


def n_cpus():
    """CPUs this process may run on (the SLURM allocation), not the node's core count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def host_info():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"host": platform.node(), "cpu_model": cpu, "n_cpus": n_cpus(),
            "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
            "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "MKL_NUM_THREADS")}}


def set_thread_env(method, workers):
    """One CPU budget for every method, spent the way each method parallelises.

    STARE's parallelism is its tile pool, so its BLAS threads are pinned to 1 (``workers``
    processes x 1 thread). The others parallelise through native thread pools, which get
    ``workers`` threads. Must run before NumPy / torch are imported; a value already in the
    environment wins.
    """
    n = "1" if method in ("stare", "initial") else str(max(1, int(workers)))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, n)


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

    def _sample(self):
        if self._read is None:
            return
        try:
            tree = self._read()
        except Exception:
            return
        with self._lock:
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

    def exclude(self, wall_s, cpu_s):
        """Take a stretch (the benchmark's own point warping) out of the method's bill."""
        self._paused_wall += wall_s
        self._paused_cpu += cpu_s

    def snapshot(self):
        self._sample()
        wall = time.perf_counter() - self._t0 - self._paused_wall
        cpu = self._cpu_now() - self._cpu0 - self._paused_cpu
        if self._read is None:
            import resource

            scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
            peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * scale
        else:
            with self._lock:
                peak = self._peak
        return {"wall_s": wall, "cpu_s": cpu, "peak_rss_mb": peak / MB,
                "gpu_peak_mb": _gpu_peak_mb(), "rss_method": self.method}

    def raw(self):
        """``(wall, cpu)`` clocks without exclusions, for timing a stretch to exclude."""
        self._sample()
        return time.perf_counter(), self._cpu_now()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return {"cgroup_peak_mb": _cgroup_peak_mb()}
