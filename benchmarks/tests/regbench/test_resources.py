"""The resource monitor bills a method for its whole process tree, and the allocation is
what a method is told to use."""

import os
import subprocess
import sys
import time

import pytest

from benchmarks.regbench import resources
from benchmarks.regbench.resources import Monitor

CHILD = """
import time
block = bytearray(300 * 1024 * 1024)
for i in range(0, len(block), 4096):
    block[i] = 1                      # touch every page so it is resident
t = time.process_time()
while time.process_time() - t < 1.0:  # one CPU-second of work
    pass
"""


def test_a_child_process_is_billed_for_its_cpu_and_memory():
    mon = Monitor(interval=0.05)
    if mon.method == "rusage":
        pytest.skip("no psutil and no /proc: only a lower bound is available here")
    mon.start()
    before = mon.snapshot()
    subprocess.run([sys.executable, "-c", CHILD], check=True)
    after = mon.snapshot()
    mon.stop()
    # the child's CPU-second, not the idle parent's
    assert after["cpu_s"] - before["cpu_s"] > 0.8
    assert after["peak_rss_mb"] - before["peak_rss_mb"] > 250
    assert after["wall_s"] >= 1.0


def test_paused_time_is_taken_off_the_bill():
    mon = Monitor(interval=0.05).start()
    mon.pause()
    time.sleep(0.3)
    during = mon.snapshot()[
        "wall_s"
    ]  # a snapshot taken while paused is the bill at the pause
    mon.resume()
    assert during < 0.15
    assert mon.snapshot()["wall_s"] < 0.15
    mon.stop()


def test_a_paused_child_is_billed_for_neither_cpu_nor_memory():
    mon = Monitor(interval=0.05)
    if mon.method == "rusage":
        pytest.skip("no psutil and no /proc: a high-water mark cannot be paused")
    mon.start()
    before = mon.snapshot()
    mon.pause()
    subprocess.run([sys.executable, "-c", CHILD], check=True)
    mon.resume()
    after = mon.snapshot()
    mon.stop()
    assert after["cpu_s"] - before["cpu_s"] < 0.5
    assert after["peak_rss_mb"] - before["peak_rss_mb"] < 50


def test_the_allocation_is_the_smaller_of_the_request_and_the_affinity(monkeypatch):
    affinity = resources.cpu_counts()["cpus_affinity"]
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert resources.n_cpus() == 1
    assert resources.cpu_counts()["cpus_requested"] == 1
    monkeypatch.setenv(
        "SLURM_CPUS_PER_TASK", str(affinity + 60)
    )  # cores are not confined
    assert resources.n_cpus() == affinity
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert resources.n_cpus() == affinity
    assert resources.cpu_counts()["cpus_requested"] is None
    info = resources.host_info()
    for key in (
        "cpus_requested",
        "cpus_affinity",
        "blas",
        "cpu_model",
        "attempt",
        "mem_requested_gb",
        "launch_id",
    ):
        assert key in info, key


def test_no_thread_variable_is_exported(monkeypatch):
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for k in names:
        monkeypatch.delenv(k, raising=False)
    assert not hasattr(resources, "set_thread_env")
    resources.host_info()
    resources.n_cpus()
    assert not [k for k in names if k in os.environ]


def test_the_oom_counter_is_read_from_a_cgroup_events_file(tmp_path):
    v2 = tmp_path / "memory.events"
    v2.write_text("low 0\nhigh 12\nmax 4\noom 2\noom_kill 3\noom_group_kill 0\n")
    assert resources.read_oom_kills(v2) == 3
    v1 = tmp_path / "memory.oom_control"
    v1.write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 1\n")
    assert resources.read_oom_kills(v1) == 1
    assert resources.read_oom_kills(tmp_path / "absent") is None
    assert resources.read_oom_kills(None) is None


def test_a_gpu_out_of_memory_is_not_a_host_out_of_memory():
    class OutOfMemoryError(RuntimeError):  # torch.cuda.OutOfMemoryError by name
        pass

    gpu = OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert resources.classify_error(gpu) == "gpu_oom"
    assert resources.classify_error(MemoryError()) == "oom"
    assert resources.classify_error(subprocess.TimeoutExpired("valis", 5)) == "timeout"
    assert resources.classify_error(ValueError("bad stride")) == "exception"


# ── CPU time is a counter, not a sample ───────────────────────────────────────
SHORT = (
    "import time\n"
    "t = time.process_time()\n"
    "while time.process_time() - t < {cpu}: pass\n"
    "print(time.process_time())\n"
)


def short_children(n, cpu):
    """Run ``n`` children one after another, each busy for ``cpu`` seconds and gone;
    the CPU time they report of themselves, summed."""
    total = 0.0
    for _ in range(n):
        done = subprocess.run(
            [sys.executable, "-S", "-c", SHORT.format(cpu=cpu)],
            check=True,
            capture_output=True,
            text=True,
        )
        total += float(done.stdout)
    return total


def test_children_that_live_between_two_samples_are_billed_in_full():
    """Forty children of 0.05 s CPU each, one after another: a sampler sees a fraction
    of them; the counter has every one once it has been waited for."""
    mon = Monitor(interval=0.2).start()
    assert mon.cpu_instrument in (resources.RUSAGE_LIVE, resources.RUSAGE)
    own = time.process_time()
    theirs = short_children(40, 0.05)
    own = time.process_time() - own
    got = mon.snapshot()
    mon.stop()
    assert got["cpu_instrument"] == mon.cpu_instrument
    assert got["cpu_s"] >= 0.99 * theirs
    # and nothing that was not used: the children, this process, and what a child
    # spends between its own last reading and its exit
    assert got["cpu_s"] <= theirs + own + 40 * 0.02


def test_a_paused_stretch_is_taken_off_the_bill_and_kept_beside_it():
    mon = Monitor(interval=0.2).start()
    mon.pause()
    theirs = short_children(10, 0.05)
    during = mon.snapshot()
    mon.resume()
    after = mon.snapshot()
    mon.stop()
    assert during["cpu_s"] < 0.1 and after["cpu_s"] < 0.1  # the method did nothing
    assert during["paused_cpu_s"] >= 0.99 * theirs
    assert after["paused_cpu_s"] >= during["paused_cpu_s"]
    assert mon.paused_cpu_s == after["paused_cpu_s"]


WORKER = (
    "import sys, time\n"
    "t = time.process_time()\n"
    "while time.process_time() - t < 0.6: pass\n"
    "print('busy done', flush=True)\n"
    "sys.stdin.read()\n"
)


def test_a_child_is_counted_alive_or_waited_for_never_both():
    mon = Monitor(interval=0.05)
    if mon.method == "rusage":
        pytest.skip("no psutil and no /proc: a live child cannot be read")
    mon.start()
    worker = subprocess.Popen(
        [sys.executable, "-S", "-c", WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert worker.stdout.readline().strip() == "busy done"
    own = time.process_time()
    theirs = short_children(5, 0.05)  # waited for while the worker is still alive
    alive = mon.snapshot()["cpu_s"]
    worker.stdin.close()
    worker.wait()  # the worker's time moves from the live sum to the waited-for one
    waited = mon.snapshot()["cpu_s"]
    own = time.process_time() - own
    mon.stop()
    assert alive >= 0.99 * (0.6 + theirs)
    assert alive <= 0.6 + theirs + own + 0.35
    assert waited - alive < 0.2  # counted twice, it would rise by the worker's 0.6 s


class _Times:
    """``os.times()`` of a process that has used ``own`` and waited for ``waited``."""

    def __init__(self, own, waited):
        self.user, self.system = own, 0.0
        self.children_user, self.children_system = waited, 0.0
        self.elapsed = 0.0


def test_the_live_sum_and_the_waited_sum_never_hold_the_same_process(monkeypatch):
    """Fed by hand: a child read alive, then waited for; a live child's own waited-for
    children; and a child that ends between the two reads of one reading."""
    me = os.getpid()
    state = {"times": _Times(1.0, 0.0), "tree": {me: (0, 1.0, 0.0)}}
    monkeypatch.setattr(resources.os, "times", lambda: state["times"])
    mon = Monitor(cpu_instrument=resources.RUSAGE_LIVE)
    mon._read = lambda: state["tree"]
    assert mon._choose() == resources.RUSAGE_LIVE
    base = mon.cpu_clock()
    assert base == 1.0  # this process is read once, from os.times
    # a child alive with 2 s of its own, and 3 s of children IT has waited for
    state["tree"] = {me: (0, 1.0, 0.0), 77: (0, 2.0, 3.0)}
    assert mon.cpu_clock() - base == 5.0
    # it ends and is waited for: all 5 s move into this process's waited-for sum
    state["times"], state["tree"] = _Times(1.0, 5.0), {me: (0, 1.0, 5.0)}
    assert mon.cpu_clock() - base == 5.0
    # a child that ended after os.times was read and before the tree was: in neither
    # sum for one reading, and the counter does not run backwards meanwhile
    state["tree"] = {me: (0, 1.0, 5.0), 78: (0, 4.0, 0.0)}
    assert mon.cpu_clock() - base == 9.0
    state["tree"] = {me: (0, 1.0, 5.0)}  # gone from the tree, not yet in os.times
    assert mon.cpu_clock() - base == 9.0
    state["times"] = _Times(1.0, 9.5)
    assert mon.cpu_clock() - base == 9.5


def _cgroup(monkeypatch, tmp_path, proc_line, counter):
    """A fake ``/proc/self/cgroup`` and cgroup mount; the counter file's path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(proc_line)
    monkeypatch.setattr(resources, "PROC_CGROUP", proc)
    monkeypatch.setattr(resources, "CGROUP_ROOT", tmp_path / "sys")
    path = tmp_path / "sys" / counter
    path.parent.mkdir(parents=True)
    return path


V2 = "0::/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0\n"
V1 = (
    "11:memory:/slurm/uid_5/job_7/step_batch/task_0\n"
    "4:cpu,cpuacct:/slurm/uid_5/job_7/step_batch/task_0\n"
)


@pytest.mark.parametrize(
    "proc_line, counter, name, write",
    [
        (
            V2,
            "system.slice/slurmstepd.scope/job_7/step_batch/user/task_0/cpu.stat",
            "cgroup_v2",
            lambda s: f"usage_usec {int(s * 1e6)}\nuser_usec 1\nsystem_usec 2\n",
        ),
        (
            V1,
            "cpu,cpuacct/slurm/uid_5/job_7/step_batch/task_0/cpuacct.usage",
            "cgroup_v1",
            lambda s: f"{int(s * 1e9)}\n",
        ),
        (
            V1,  # the hierarchy mounted under the controller's own name
            "cpuacct/slurm/uid_5/job_7/step_batch/task_0/cpuacct.usage",
            "cgroup_v1",
            lambda s: f"{int(s * 1e9)}\n",
        ),
    ],
)
def test_the_step_cgroup_counter_is_billed_as_differences(
    monkeypatch, tmp_path, proc_line, counter, name, write
):
    path = _cgroup(monkeypatch, tmp_path, proc_line, counter)
    path.write_text(write(100.0))  # what the task had used before the method started
    mon = Monitor(interval=0.05).start()
    assert mon.cpu_instrument == name
    path.write_text(write(102.0))  # the method: 2 s, in processes nobody sampled
    mon.pause()
    path.write_text(write(105.0))  # a paused stretch: 3 s
    during = mon.snapshot()
    mon.resume()
    path.write_text(write(106.0))  # the method again: 1 s
    after = mon.snapshot()
    path.write_text(write(110.0))
    mon.pause()
    path.unlink()  # a counter that stops answering: the last reading stands
    end = mon.snapshot()
    launch = mon.stop()["launch_cpu_s"]
    assert during["cpu_s"] == pytest.approx(2.0)
    assert during["paused_cpu_s"] == pytest.approx(3.0)
    assert after["cpu_s"] == pytest.approx(3.0)
    assert after["paused_cpu_s"] == pytest.approx(3.0)
    assert after["cpu_instrument"] == name and after["cpu_read_failures"] == 0
    assert end["cpu_s"] == pytest.approx(7.0) and end["cpu_read_failures"] > 0
    assert launch == pytest.approx(10.0)


def test_a_cgroup_that_is_no_scheduler_step_is_not_this_runs_counter(
    monkeypatch, tmp_path
):
    """A desktop session's cgroup holds other people's processes: its counter is not
    read, and the run falls back to its own rusage."""
    path = _cgroup(
        monkeypatch,
        tmp_path,
        "0::/user.slice/user-1000.slice/session-3.scope\n",
        "user.slice/user-1000.slice/session-3.scope/cpu.stat",
    )
    path.write_text("usage_usec 5000000\n")
    assert resources.cgroup_cpu_counter() is None
    mon = Monitor().start()
    mon.stop()
    assert mon.cpu_instrument in (resources.RUSAGE_LIVE, resources.RUSAGE)


def test_the_memory_files_are_found_in_the_same_cgroup(monkeypatch, tmp_path):
    path = _cgroup(
        monkeypatch,
        tmp_path,
        V2,
        "system.slice/slurmstepd.scope/job_7/step_batch/user/task_0/memory.events",
    )
    path.write_text("oom 0\noom_kill 2\n")
    (path.parent / "memory.peak").write_text(str(3 * 1024 * 1024))
    assert resources.oom_events_path() == path
    assert resources.read_oom_kills(resources.oom_events_path()) == 2
    assert resources._cgroup_peak_mb() == 3.0
    old = _cgroup(
        monkeypatch,
        tmp_path / "v1",
        V1,
        "memory/slurm/uid_5/job_7/step_batch/task_0/memory.oom_control",
    )
    old.write_text("oom_kill 1\n")
    (old.parent / "memory.max_usage_in_bytes").write_text(str(5 * 1024 * 1024))
    assert resources.oom_events_path() == old
    assert resources._cgroup_peak_mb() == 5.0


def test_the_sampler_alone_is_named_as_what_it_is():
    mon = Monitor(interval=0.05, cpu_instrument=resources.POLLED)
    if mon.method == "rusage":
        pytest.skip("no psutil and no /proc: there is nothing to sample")
    mon.start()
    subprocess.run([sys.executable, "-c", CHILD], check=True)
    got = mon.snapshot()
    mon.stop()
    assert got["cpu_instrument"] == "polled (may undercount)"
    assert got["cpu_s"] > 0.5  # a child that lives for seconds is seen
