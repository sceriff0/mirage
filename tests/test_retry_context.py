"""RetryContext is a for-loop iterator. Exhaustion must end the loop, not
escape it.

If __next__ raised on exhaustion, the exception would escape the
`for attempt in ctx:` loop, past the caller's post-loop partial-failure
handling (bin/register.py's `all_attempts_failed` check), and kill the whole
task instead of dropping one slide.

`all_attempts_failed` is a `@property`, not a method.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = str(REPO_ROOT / "bin")
UTILS_DIR = str(REPO_ROOT / "bin" / "utils")
for _dir in (BIN_DIR, UTILS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from retry import RetryContext  # noqa: E402


def test_exhaustion_ends_the_loop_rather_than_raising():
    ctx = RetryContext(max_attempts=3)
    attempts = []
    for attempt in ctx:
        attempts.append(attempt)
        ctx.failed(RuntimeError(f"boom {attempt}"))
    assert attempts == [1, 2, 3]


def test_all_attempts_failed_is_reachable_and_true_after_exhaustion():
    ctx = RetryContext(max_attempts=2)
    for attempt in ctx:
        ctx.failed(RuntimeError("boom"))
    assert ctx.all_attempts_failed is True


def test_success_stops_iteration_and_reports_not_failed():
    ctx = RetryContext(max_attempts=3)
    seen = []
    for attempt in ctx:
        seen.append(attempt)
        ctx.succeeded()
    assert seen == [1]
    assert ctx.all_attempts_failed is False


def test_last_exception_is_preserved_for_the_caller_to_re_raise():
    """The exception is not swallowed -- it is handed to the caller, whose job
    it is to decide between 'drop this slide' and 'fail the task'."""
    ctx = RetryContext(max_attempts=2)
    boom = RuntimeError("the real cause")
    for attempt in ctx:
        ctx.failed(boom)
    assert ctx.all_attempts_failed
    assert ctx.last_exception is boom
