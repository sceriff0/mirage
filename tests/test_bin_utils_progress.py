"""bin/utils/progress.py — the ETA arithmetic and the phase bookkeeping.

Both classes are pure bookkeeping over a clock: a division by zero on the first
step, or a phase timer that never resets, produces only a wrong log line.
"""

from __future__ import annotations

import sys
from pathlib import Path

# progress.py does a flat `from logger import log_progress`, so bin/utils must be on
# sys.path before `from utils import progress` runs its module body.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin" / "utils"))

from utils import progress  # noqa: E402


def test_the_eta_says_calculating_before_any_step_has_finished():
    """The first call divides by the number of completed steps. Reporting
    'calculating...' is the guard against that being zero."""
    tracker = progress.ProgressTracker(total_steps=4, operation_name="test")
    tracker.start()
    assert tracker._estimate_remaining() == "calculating..."


def test_completed_steps_are_counted_and_timed():
    tracker = progress.ProgressTracker(total_steps=3, operation_name="test")
    tracker.start()
    tracker.step_complete("first")
    tracker.step_complete("second", details="with details")
    assert tracker.current_step == 2
    assert len(tracker.step_times) == 2
    assert all(t >= 0 for t in tracker.step_times)


def test_the_eta_is_formatted_in_the_right_unit():
    """Seconds, minutes and hours have three different suffixes, and the
    boundaries are exact -- a report reading '3600s' is unreadable.

    ``current_step = total_steps - 1`` (one step remaining) matters:
    ``_estimate_remaining`` multiplies the average step time by the remaining
    step count, so with one step left each step_times value maps onto exactly
    one boundary."""
    tracker = progress.ProgressTracker(total_steps=100, operation_name="test")
    tracker.start()
    tracker.current_step = 99
    tracker.step_times = [0.1]
    assert tracker._estimate_remaining().endswith("s")
    tracker.step_times = [60.0]
    assert tracker._estimate_remaining().endswith("m")
    tracker.step_times = [3600.0]
    assert tracker._estimate_remaining().endswith("h")


def test_finish_reports_without_a_start(capsys):
    """finish() on a tracker that was never started must not raise: it is called
    from an except: branch, where start() may be exactly what failed."""
    tracker = progress.ProgressTracker(total_steps=2, operation_name="test")
    tracker.finish(success=False)
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_a_phase_reporter_announces_every_phase_it_knows(capsys):
    reporter = progress.PhaseReporter()
    for phase_id, display in progress.PhaseReporter.PHASES:
        reporter.enter_phase(phase_id)
        assert reporter.current_phase == phase_id
    out = capsys.readouterr().out
    for _phase_id, display in progress.PhaseReporter.PHASES:
        assert display in out


def test_an_unknown_phase_is_titled_rather_than_dropped(capsys):
    """A phase name the table does not carry must still be announced -- the
    alternative is a silent gap in the log exactly where something new is
    happening."""
    reporter = progress.PhaseReporter()
    reporter.enter_phase("brand_new_stage")
    assert reporter.current_phase == "brand_new_stage"
    assert "Brand_New_Stage" in capsys.readouterr().out
