"""JVM heap: VALIS's own fixed 10 GB, and a diagnosable failure.

User ruling 2026-10-21: the heap REGISTER gives VALIS's Bio-Formats JVM is VALIS's own
shipped 10 GB (``registration.init_jvm`` passes the literal 10 on to
``slide_io.init_jvm``), at every attempt. It is no longer scaled with the retry ramp
(it was ``32 + 16 * task.attempt``): the benchmark runs VALIS with the heap VALIS ships,
and the pipeline runs it the same way. ``reg_jvm_heap_gb`` still overrides it (the
``tma`` profile pins 16), and the task's memory still clamps it:

    def heap_request = params.reg_jvm_heap_gb != null ? params.reg_jvm_heap_gb : 10
    def jvm_heap_gb  = Math.min(heap_request, task.memory.toGiga() - 4)

REGISTER's MEMORY still climbs with the attempt; only the heap is pinned. A heap-bound
JVM OOM retried is therefore handed the same heap: retrying it cannot help, which the
log message below must make diagnosable.

``heap_for_attempt()`` reads the constants live out of the script via tests.nfmodel
(never a private regex over the raw .nf file -- see
tests/test_nfmodel.py::test_no_guard_parses_nextflow_source_privately).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import processes

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTER_PY = REPO_ROOT / "bin" / "register.py"

# Narrow, purpose-built patterns matched against the ALREADY-EXTRACTED
# script_body text tests.nfmodel hands back (never against the raw .nf file
# directly -- see tests/test_nfmodel.py::test_no_guard_parses_nextflow_source_privately,
# which forbids a private parse of Nextflow source but not further processing
# of text the model already extracted; tests/test_error_strategy_policy.py
# does the same thing against with_name_blocks() bodies).
_HEAP_REQUEST_RE = re.compile(
    r"def heap_request = params\.reg_jvm_heap_gb != null \? params\.reg_jvm_heap_gb : "
    r"(\d+)\s*$",
    re.MULTILINE,
)
# VALIS's own heap: registration.init_jvm(jar=None, mem_gb=10) (user ruling 2026-10-21)
VALIS_HEAP_GB = 10
_JVM_HEAP_RE = re.compile(
    r"def jvm_heap_gb = Math\.min\(heap_request, task\.memory\.toGiga\(\) - (\d+)\)"
)


def _register_script_body() -> str:
    return processes()["REGISTER"].script_body


def _default_heap_gb() -> int:
    """The default heap read live out of register.nf's
    `def heap_request = ... : 10` line. If the shape of that line changes (a ramp
    comes back), this match fails loudly rather than mirroring a stale formula."""
    m = _HEAP_REQUEST_RE.search(_register_script_body())
    assert m, (
        "modules/local/register.nf's heap_request is no longer a fixed default "
        "(`... : <GiB>`) -- read the new derivation and update this test before "
        "trusting it (user ruling 2026-10-21: VALIS's own 10 GB, never scaled)."
    )
    return int(m.group(1))


def _mem_headroom_gb() -> int:
    """The GB subtracted from task.memory.toGiga() in
    `def jvm_heap_gb = Math.min(heap_request, task.memory.toGiga() - 4)`."""
    m = _JVM_HEAP_RE.search(_register_script_body())
    assert m, (
        "modules/local/register.nf's jvm_heap_gb derivation no longer matches "
        "the expected `Math.min(heap_request, task.memory.toGiga() - N)` shape -- "
        "read the new derivation and update this test before trusting it."
    )
    return int(m.group(1))


def heap_for_attempt(task_memory_gb: int, attempt: int, reg_jvm_heap_gb=None) -> int:
    """Mirrors register.nf's derivation with its constants read live from the script.
    ``attempt`` is taken and ignored: that the heap does not depend on it is the rule."""
    del attempt
    heap_request = reg_jvm_heap_gb if reg_jvm_heap_gb is not None else _default_heap_gb()
    return min(heap_request, task_memory_gb - _mem_headroom_gb())


def test_derivation_matches_register_nf():
    """The default heap is VALIS's own 10 GB and the headroom 4 GB, read via
    tests.nfmodel.processes(), never a private regex over the raw register.nf file."""
    assert _default_heap_gb() == VALIS_HEAP_GB
    assert _mem_headroom_gb() == 4
    assert "task.attempt" not in _HEAP_REQUEST_RE.search(_register_script_body())[0]


def test_the_heap_is_flat_across_attempts_at_every_memory():
    """REGISTER's memory climbs (64 GB x 2^(attempt-1)); the heap does not."""
    for mem in (64, 128, 256, 512):
        by_attempt = [heap_for_attempt(mem, a) for a in (1, 2, 3)]
        assert by_attempt == [VALIS_HEAP_GB] * 3, (mem, by_attempt)


def test_the_heap_is_clamped_at_small_memory():
    """At 6 GB -- the test profile's task.memory -- the heap is min(10, 6 - 4) = 2."""
    assert [heap_for_attempt(6, a) for a in (1, 2, 3)] == [2, 2, 2]


def test_an_explicit_heap_still_overrides():
    """`reg_jvm_heap_gb` (the tma profile pins 16) wins over the default."""
    assert heap_for_attempt(64, 1, reg_jvm_heap_gb=16) == 16
    assert heap_for_attempt(64, 3, reg_jvm_heap_gb=16) == 16


def test_jvm_death_message_names_the_jvm_and_a_remedy():
    """bin/register.py's JVM-death path must produce a message that (a)
    names the JVM as the cause and (b) suggests a remedy, so an operator
    reading the log can tell a JVM OOM apart from a tile-read failure even
    though REGISTER's retry-exit1-then-fail policy can't distinguish them by
    exit code alone."""
    text = REGISTER_PY.read_text()
    jvm_death_msg = "JVM was killed during registration. Warping cannot proceed. "
    assert jvm_death_msg in text, (
        "expected JVM-death RuntimeError message not found in bin/register.py"
    )
    assert "JVM" in jvm_death_msg
    # The surrounding log block is the "remedy" half, printed immediately before
    # this RuntimeError is raised. The remedy named must be a real one. Until
    # 2026-09-08 this asserted `--micro-reg 0`, which was folklore: the micro pass is
    # consumed only by register_micro(), which had not run in any recorded JVM death.
    # Until 2026-09-23 it asserted `--max-non-rigid-dim`, which was no remedy either:
    # the level -1 crash depends on the tissue-mask extent, so no size was safe. It is
    # clamped at the reader now, and the block points at the clamp's own log line.
    block = text[: text.index(jvm_death_msg)]
    block = block[block.rindex("JVM is not running") :]
    assert "clamp was NOT applied" in block
    assert "registration_method = 'tiled'" in block
    assert "Try --micro-reg 0" not in text


def test_jvm_death_message_is_distinct_from_tile_read_failure_message():
    """The tile-read/memory failure path raises a DIFFERENT message. The two
    must not collide -- an operator (or a log-scraping alert) needs to be
    able to tell them apart by text even though the process exit code (1) is
    identical for both under retry-exit1-then-fail."""
    text = REGISTER_PY.read_text()
    jvm_death_msg = "JVM was killed during registration. Warping cannot proceed. "
    tile_read_msg = "VALIS registration failed due to memory/TIFF issue: "

    assert jvm_death_msg in text
    assert tile_read_msg in text
    assert jvm_death_msg != tile_read_msg
    assert jvm_death_msg not in tile_read_msg
    assert tile_read_msg not in jvm_death_msg
    # The tile-read message must not itself mention the JVM -- if it did,
    # naming "JVM" in a message would no longer distinguish the two causes.
    assert "JVM" not in tile_read_msg
