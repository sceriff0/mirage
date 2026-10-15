"""What PhenoCycler and Keyence CODEX share: the snake grid and best focus."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def snake_rowcol(index: int, n_cols: int) -> tuple[int, int]:
    """(row, col) of the ``index``-th (0-based) tile of a snake: row r runs left to right
    when r is even and right to left when odd."""
    row, k = divmod(index, n_cols)
    return row, (k if row % 2 == 0 else n_cols - 1 - k)


def focus_score(plane: np.ndarray) -> float:
    """Variance of the Laplacian of one plane."""
    lap = ndimage.laplace(np.asarray(plane, dtype=np.float32))
    return float(np.var(lap, dtype=np.float64))


class BestFocus:
    """Keeps the sharpest plane seen so far (and its score), never the stack. A tie keeps
    the first (lowest z)."""

    def __init__(self):
        self.z = None
        self.score = -np.inf
        self.plane = None

    def offer(self, z: int, plane: np.ndarray) -> None:
        score = focus_score(plane)
        if self.plane is None or score > self.score:
            self.z, self.score, self.plane = z, score, plane
