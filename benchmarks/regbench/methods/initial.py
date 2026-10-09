"""No registration: the pose every method must beat, and the one a failed case is scored at."""

from __future__ import annotations

import numpy as np

VARIANTS = ("initial",)


def register(case, work, opts):
    yield "initial", lambda xy: np.array(xy, dtype=float), {}
