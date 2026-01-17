"""Progress reporting utilities for long-running operations.

This module provides structured progress tracking with ETA estimation
for multi-step registration operations.

Examples
--------
>>> tracker = ProgressTracker(total_steps=10, operation_name="Slide Warping")
>>> tracker.start()
>>> for i, slide in enumerate(slides):
...     process_slide(slide)
...     tracker.step_complete(slide.name)
>>> tracker.finish()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from logger import log_progress

__all__ = [
    "ProgressTracker",
    "PhaseReporter",
]


@dataclass
class ProgressTracker:
    """Track progress of multi-step operations with ETA estimation.

    Provides formatted progress output with elapsed time tracking
    and remaining time estimation based on average step duration.

    Attributes
    ----------
    total_steps : int
        Total number of steps to complete
    operation_name : str
        Name of the operation being tracked
    current_step : int
        Current step number (0-indexed)
    start_time : datetime or None
        When tracking started
    step_times : list of float
        Duration of each completed step in seconds
    """

    total_steps: int
    operation_name: str
    current_step: int = 0
    start_time: Optional[datetime] = None
    step_times: List[float] = field(default_factory=list)
    _last_step_time: Optional[float] = field(default=None, repr=False)

    def start(self) -> None:
        """Start progress tracking."""
        self.start_time = datetime.now()
        self._last_step_time = time.time()

        log_progress(f"\n{'='*70}")
        log_progress(f"Starting: {self.operation_name}")
        log_progress(f"Total steps: {self.total_steps}")
        log_progress(f"{'='*70}")

    def step_complete(self, step_name: str, details: str = "") -> None:
        """Mark a step as complete and log progress.

        Parameters
        ----------
        step_name : str
            Name of the completed step
        details : str, optional
            Additional details to log
        """
        self.current_step += 1
        now = time.time()

        # Calculate step duration
        if self._last_step_time is not None:
            step_duration = now - self._last_step_time
            self.step_times.append(step_duration)
        self._last_step_time = now

        # Calculate progress
        progress_pct = (self.current_step / self.total_steps) * 100
        eta = self._estimate_remaining()

        log_progress(
            f"[{self.current_step}/{self.total_steps}] {step_name} "
            f"({progress_pct:.0f}% complete, ETA: {eta})"
        )
        if details:
            log_progress(f"    {details}")

    def _estimate_remaining(self) -> str:
        """Estimate remaining time based on average step duration.

        Returns
        -------
        str
            Formatted time remaining estimate
        """
        if not self.step_times:
            return "calculating..."

        avg_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        remaining_secs = avg_time * remaining_steps

        if remaining_secs < 60:
            return f"{remaining_secs:.0f}s"
        elif remaining_secs < 3600:
            return f"{remaining_secs/60:.1f}m"
        else:
            return f"{remaining_secs/3600:.1f}h"

    def finish(self, success: bool = True) -> None:
        """Mark operation as complete.

        Parameters
        ----------
        success : bool, default=True
            Whether the operation completed successfully
        """
        if self.start_time:
            duration = datetime.now() - self.start_time
        else:
            duration = timedelta(seconds=0)

        status = "completed successfully" if success else "FAILED"

        log_progress(f"\n{'='*70}")
        log_progress(f"{self.operation_name} {status}")
        log_progress(f"Total time: {duration}")
        log_progress(f"Steps completed: {self.current_step}/{self.total_steps}")
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            log_progress(f"Average step time: {avg_time:.1f}s")
        log_progress(f"{'='*70}")


class PhaseReporter:
    """Report progress through distinct registration phases.

    Provides clear phase markers with timing information for
    the main stages of registration.

    Attributes
    ----------
    PHASES : list of tuple
        Available phases with (phase_id, display_name) pairs
    current_phase : str or None
        Currently active phase
    phase_start : float or None
        Start time of current phase
    """

    PHASES = [
        ("init", "Initializing"),
        ("validate", "Validating inputs"),
        ("rigid", "Rigid registration"),
        ("micro", "Micro-registration"),
        ("warp", "Warping slides"),
        ("cleanup", "Cleanup"),
    ]

    def __init__(self):
        """Initialize phase reporter."""
        self.current_phase: Optional[str] = None
        self.phase_start: Optional[float] = None
        self._phase_durations: dict = {}

    def enter_phase(self, phase_name: str) -> None:
        """Enter a new registration phase.

        Logs the phase exit timing for the previous phase and
        announces the new phase.

        Parameters
        ----------
        phase_name : str
            ID of the phase to enter (e.g., 'init', 'rigid', 'warp')
        """
        # Exit current phase if any
        if self.current_phase:
            self._exit_current()

        # Get display name
        phase_desc = dict(self.PHASES).get(phase_name, phase_name.title())

        self.current_phase = phase_name
        self.phase_start = time.time()

        log_progress(f"\n>>> Phase: {phase_desc}")

    def _exit_current(self) -> None:
        """Exit current phase with timing information."""
        if self.phase_start is not None and self.current_phase:
            duration = time.time() - self.phase_start
            self._phase_durations[self.current_phase] = duration
            log_progress(f"    Phase completed in {duration:.1f}s")

    def finish(self) -> None:
        """Finish all phases and log summary."""
        if self.current_phase:
            self._exit_current()

        if self._phase_durations:
            log_progress("\n>>> Phase Summary:")
            total = 0
            for phase_id, duration in self._phase_durations.items():
                phase_name = dict(self.PHASES).get(phase_id, phase_id)
                log_progress(f"    {phase_name}: {duration:.1f}s")
                total += duration
            log_progress(f"    Total: {total:.1f}s")

    def get_phase_duration(self, phase_name: str) -> Optional[float]:
        """Get duration of a completed phase.

        Parameters
        ----------
        phase_name : str
            Phase ID to query

        Returns
        -------
        float or None
            Duration in seconds, or None if phase not completed
        """
        return self._phase_durations.get(phase_name)
