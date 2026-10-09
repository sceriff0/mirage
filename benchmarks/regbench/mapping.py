"""The point mapping a method hands back, with the direction it was estimated in.

A non-rigid transform has one exact direction; going the other way needs an inverse, which
is an approximation. So a method returns its map in its OWN direction and says which one it
is, and the benchmark moves the nuclei that direction only:

``MOV_TO_REF``  ``fn`` takes moving-slide pixels to reference-slide pixels. The moving nuclei
                are warped and scored in the reference frame.
``REF_TO_MOV``  ``fn`` takes reference-slide pixels to moving-slide pixels. The reference
                nuclei are warped and scored in the moving frame.

Coordinates are X, Y in full-resolution pixels of the NATIVE slides, pixel centres on the
integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MOV_TO_REF = "mov_to_ref"
REF_TO_MOV = "ref_to_mov"
DIRECTIONS = (MOV_TO_REF, REF_TO_MOV)
FRAME = {MOV_TO_REF: "reference", REF_TO_MOV: "moving"}


@dataclass(frozen=True)
class PointMap:
    """``fn(xy (N, 2)) -> (N, 2)`` and the direction it maps in."""

    fn: Callable
    direction: str

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"direction must be one of {DIRECTIONS}, got {self.direction!r}"
            )
        if not callable(self.fn):
            raise TypeError("fn must be callable")

    def __call__(self, xy):
        return self.fn(xy)

    @property
    def frame(self):
        """The frame the nuclei are compared in: ``reference`` or ``moving``."""
        return FRAME[self.direction]
