"""No registration: the pose every method must beat, and the one a failed case is scored at."""

from __future__ import annotations

import numpy as np

from ..mapping import MOV_TO_REF, PointMap

SCOPE = "pair"
VARIANTS = ("initial",)


def register(case, work, opts):
    yield "initial", PointMap(lambda xy: np.array(xy, dtype=float), MOV_TO_REF), {}
