"""The resource monitor bills a method for its whole process tree, not just the parent."""

import subprocess
import sys
import time

import pytest
from regbench.resources import Monitor, set_thread_env

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
    assert after["cpu_s"] - before["cpu_s"] > 0.8  # the child's CPU-second, not the idle parent's
    assert after["peak_rss_mb"] - before["peak_rss_mb"] > 250
    assert after["wall_s"] >= 1.0


def test_excluded_time_is_taken_off_the_bill():
    mon = Monitor(interval=0.05).start()
    w0, c0 = mon.raw()
    time.sleep(0.3)
    w1, c1 = mon.raw()
    mon.exclude(w1 - w0, c1 - c0)
    assert mon.snapshot()["wall_s"] < 0.15
    mon.stop()


def test_thread_budget_follows_how_the_method_parallelises(monkeypatch):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    import os

    set_thread_env("stare", 8)  # a pool of 8 processes, one thread each
    assert os.environ["OMP_NUM_THREADS"] == "1"
    monkeypatch.delenv("OMP_NUM_THREADS")
    set_thread_env("valis", 8)  # one process, 8 threads
    assert os.environ["OMP_NUM_THREADS"] == "8"
    set_thread_env("stare", 8)  # an explicit value in the environment wins
    assert os.environ["OMP_NUM_THREADS"] == "8"
