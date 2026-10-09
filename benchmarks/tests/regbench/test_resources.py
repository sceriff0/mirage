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
