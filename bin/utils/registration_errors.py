"""Registration-specific error handling utilities.

This module provides structured error context classes for tracking
registration failures with detailed information for debugging and reporting.

Examples
--------
>>> from registration_errors import RegistrationErrorContext, ErrorSeverity
>>> error = RegistrationErrorContext(
...     slide_name="panel1",
...     phase="warp",
...     error_type="WarpingError",
...     message="Memory exhausted during warping",
...     severity=ErrorSeverity.ERROR,
...     suggestions=["Reduce max_image_dim", "Free system memory"]
... )
>>> error.to_dict()
{'slide': 'panel1', 'phase': 'warp', ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


__all__ = [
    "ErrorSeverity",
    "RegistrationErrorContext",
    "RegistrationResult",
]


class ErrorSeverity(Enum):
    """Severity levels for registration errors."""

    WARNING = "warning"  # Non-blocking issue, processing continues
    ERROR = "error"      # Blocking for this slide, other slides may continue
    FATAL = "fatal"      # Blocks entire registration process


@dataclass
class RegistrationErrorContext:
    """Structured error context for registration failures.

    Provides detailed information about registration errors including
    the affected slide, phase of failure, and actionable suggestions.

    Attributes
    ----------
    slide_name : str
        Name of the slide that encountered the error
    phase : str
        Registration phase where error occurred ('rigid', 'non_rigid', 'micro', 'warp')
    error_type : str
        Type/class name of the exception
    message : str
        Human-readable error message
    severity : ErrorSeverity
        How severe the error is (WARNING, ERROR, FATAL)
    recoverable : bool
        Whether the error can potentially be recovered from
    suggestions : list of str
        Actionable suggestions for resolving the error
    details : dict
        Additional context-specific details
    """

    slide_name: str
    phase: str
    error_type: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recoverable: bool = True
    suggestions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of the error context
        """
        return {
            "slide": self.slide_name,
            "phase": self.phase,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "suggestions": self.suggestions,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"[{self.severity.value.upper()}] {self.slide_name} ({self.phase}): {self.message}"


@dataclass
class RegistrationResult:
    """Structured result for registration operations.

    Tracks overall success/failure and collects all errors and warnings
    encountered during registration.

    Attributes
    ----------
    success : bool
        Whether registration completed successfully
    slides_processed : int
        Number of slides successfully processed
    slides_failed : int
        Number of slides that failed
    errors : list of RegistrationErrorContext
        All errors encountered during registration
    warnings : list of str
        Non-blocking warnings
    """

    success: bool = True
    slides_processed: int = 0
    slides_failed: int = 0
    errors: List[RegistrationErrorContext] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, ctx: RegistrationErrorContext) -> None:
        """Add an error to the result.

        If the error is FATAL, marks the entire result as failed.

        Parameters
        ----------
        ctx : RegistrationErrorContext
            Error context to add
        """
        self.errors.append(ctx)
        if ctx.severity == ErrorSeverity.FATAL:
            self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message.

        Parameters
        ----------
        message : str
            Warning message
        """
        self.warnings.append(message)

    def has_errors(self) -> bool:
        """Check if any errors were recorded.

        Returns
        -------
        bool
            True if there are errors
        """
        return len(self.errors) > 0

    def get_errors_by_phase(self, phase: str) -> List[RegistrationErrorContext]:
        """Get all errors from a specific phase.

        Parameters
        ----------
        phase : str
            Phase name to filter by

        Returns
        -------
        list of RegistrationErrorContext
            Errors from the specified phase
        """
        return [e for e in self.errors if e.phase == phase]

    def summary(self) -> str:
        """Generate a summary of the registration result.

        Returns
        -------
        str
            Multi-line summary string
        """
        lines = [
            f"Registration {'SUCCEEDED' if self.success else 'FAILED'}",
            f"  Processed: {self.slides_processed}",
            f"  Failed: {self.slides_failed}",
        ]

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5
                lines.append(f"    - {error}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")

        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")

        return "\n".join(lines)
