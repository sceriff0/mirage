"""JVM heap: a diagnosable failure and a real retry ramp.

`conf/modules.config` gives REGISTER the `retry-exit1-then-fail` policy on the
premise that "a VALIS tile read failure and a JVM OOM inside the wrapper both
come back as a plain exit 1, so the cause selector cannot be narrowed" (see
tests/test_error_strategy_policy.py's CANONICAL table). Two things follow from
that premise and neither was tested until now:

  1. The heap actually has to SCALE with task.attempt, or retrying an
     OOM'd attempt just repeats it with the identical heap.
  2. When the JVM dies, the log has to say so in words distinguishable from
     a tile-read failure, since the exit code cannot do that job.

The derivation lives at modules/local/register.nf:78-79:

    def heap_request = params.reg_jvm_heap_gb != null ? params.reg_jvm_heap_gb : (32 + 16 * task.attempt)
    def jvm_heap_gb  = Math.min(heap_request, task.memory.toGiga() - 4)

`heap_for_attempt()` below reads the base/per-attempt/headroom constants live
out of that text via tests.nfmodel (never a private regex over the raw .nf
file -- see tests/test_nfmodel.py::test_no_guard_parses_nextflow_source_privately),
so a change to the ramp itself -- not just this file's memory of it -- shows
up directly in the ramp assertions below.

Measured 2026-08-30 (params.reg_jvm_heap_gb unset, the default path):

    task.memory=  6 GB -> heap by attempt 1/2/3: [2, 2, 2]      INERT
    task.memory= 32 GB -> heap by attempt 1/2/3: [28, 28, 28]   INERT
    task.memory= 64 GB -> heap by attempt 1/2/3: [48, 60, 60]
    task.memory=128 GB -> heap by attempt 1/2/3: [48, 64, 80]

Attempt 1 already requests min(48, mem-4), so the ramp only moves once
task.memory > 52 GB. A test asserting "strictly increasing" unconditionally
would fail against correct current behaviour -- the two heap tests below
assert the increasing case AND the flat-clamp case, deliberately, as two
separate properties.
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
    r"\((\d+) \+ (\d+) \* task\.attempt\)"
)
_JVM_HEAP_RE = re.compile(
    r"def jvm_heap_gb = Math\.min\(heap_request, task\.memory\.toGiga\(\) - (\d+)\)"
)


def _register_script_body() -> str:
    return processes()["REGISTER"].script_body


def _ramp_params() -> tuple[int, int]:
    """(base_gb, per_attempt_gb) read live out of register.nf's
    `def heap_request = ... : (32 + 16 * task.attempt)` line. If the shape of
    that line changes (not just its two numbers), this match fails loudly
    rather than silently mirroring a stale formula."""
    m = _HEAP_REQUEST_RE.search(_register_script_body())
    assert m, (
        "modules/local/register.nf's heap_request derivation no longer matches "
        "the expected `base + per_attempt * task.attempt` shape -- read the new "
        "derivation and update this test's regex/expectations before trusting it."
    )
    return int(m.group(1)), int(m.group(2))


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
    """Mirrors modules/local/register.nf:78-79's derivation, but reads its two
    tunable constants (base, per-attempt increment, memory headroom) live out
    of the .nf source via tests.nfmodel rather than hardcoding them -- so a
    change to the ramp itself (not just this test's memory of it) is visible
    in the assertions below, not just in a separate anchor check."""
    base, per_attempt = _ramp_params()
    headroom = _mem_headroom_gb()
    heap_request = (
        reg_jvm_heap_gb
        if reg_jvm_heap_gb is not None
        else (base + per_attempt * attempt)
    )
    return min(heap_request, task_memory_gb - headroom)


def test_derivation_matches_register_nf():
    """The exact two-line derivation this file's arithmetic mirrors is
    readable, in the expected shape, straight out of REGISTER's script: body
    -- and with today's actual constants (base=32, per_attempt=16,
    headroom=4), matching modules/local/register.nf:78-79 as read on
    2026-08-30. Read via tests.nfmodel.processes(), never a private regex
    over the raw register.nf file."""
    assert _ramp_params() == (32, 16)
    assert _mem_headroom_gb() == 4


def test_heap_ramp_strictly_increases_at_large_memory():
    """At 128 GB (well above the ~52 GB point where attempt 1's min(48, ...)
    stops being the binding constraint), the heap must strictly increase
    across attempts 1->2->3, and never exceed task.memory - 4 -- the ramp
    that fixed the REG_WARP_REF OOM is only real if this holds."""
    mem = 128
    by_attempt = [heap_for_attempt(mem, a) for a in (1, 2, 3)]
    assert by_attempt == [48, 64, 80], by_attempt
    assert by_attempt[0] < by_attempt[1] < by_attempt[2], (
        f"heap ramp did not strictly increase at {mem} GB: {by_attempt}"
    )
    for heap in by_attempt:
        assert heap <= mem - 4, f"heap {heap} GB exceeded task.memory-4 ({mem - 4} GB)"


def test_heap_ramp_clamped_flat_at_small_memory():
    """At 6 GB -- the test profile's task.memory -- attempt 1 already
    requests min(32+16, mem-4) = min(48, 2) = 2, and every later attempt's
    larger heap_request is clamped to the exact same task.memory-4 ceiling.
    The ramp is INERT below ~52 GB.

    This is deliberately pinned, not incidental: REGISTER's errorStrategy is
    retry-exit1-then-fail on the premise that retrying helps. Below ~52 GB, a
    heap-driven OOM that gets retried is handed EXACTLY the heap it just died
    on -- the retry cannot possibly change the outcome. That is a real
    property of the current pipeline; pin it so a future ramp change (e.g.
    lowering the base or the per-attempt increment) is a conscious decision,
    not an accidental one discovered in production."""
    mem = 6
    by_attempt = [heap_for_attempt(mem, a) for a in (1, 2, 3)]
    assert by_attempt == [2, 2, 2], by_attempt
    assert len(set(by_attempt)) == 1, (
        f"expected the ramp clamped flat at {mem} GB, got {by_attempt}"
    )


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
    # The surrounding log block is the "remedy" half -- suggested workarounds
    # printed immediately before this RuntimeError is raised.
    assert "Suggested workarounds" in text
    # The remedy named must be a real one. Until 2026-09-08 this asserted
    # `--micro-reg 0`, which was folklore: the micro pass is consumed only by
    # register_micro(), which had not run in any recorded JVM death -- the JVM
    # dies inside Valis.register() (bin/utils/valis_preflight.py has the chain).
    assert "--max-non-rigid-dim" in text
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
