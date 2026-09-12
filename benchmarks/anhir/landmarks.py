"""Landmark I/O and the ANHIR error definitions, numpy only.

From https://anhir.grand-challenge.org/Performance_Metrics/ :

    TRE  = d_e(x^T, x^W)                Euclidean distance, target vs warped
    rTRE = TRE / sqrt(w^2 + h^2)        normalised by the image diagonal
    R    = 1/|L| * sum_j [ rTRE_j(regist) < rTRE_j(init) ]     robustness

Per case the challenge reports the median, mean and max of rTRE over the
landmarks, and robustness. ``case_stats`` computes exactly those.

Landmark files are ``,X,Y`` CSVs (an unnamed index column, then X and Y in
pixels of the image at that scale, origin at the top-left pixel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LANDMARK_COLUMNS = ("X", "Y")


def read_landmarks(path) -> np.ndarray:
    """``(N, 2)`` float array of X, Y from an ANHIR landmark CSV."""
    df = pd.read_csv(path)
    missing = [c for c in LANDMARK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing landmark column(s) {missing}")
    xy = df[list(LANDMARK_COLUMNS)].to_numpy(dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{path}: expected an (N, 2) table, got {xy.shape}")
    return xy


def write_landmarks(path, xy) -> Path:
    """Write ``xy`` in the challenge's ``,X,Y`` format (index column included)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1]}).to_csv(path, index=True)
    return path


def read_imagej_points(path) -> np.ndarray:
    """ImageJ ``point`` text format (``point\\nN\\nx y\\n...``), as ``(N, 2)``.

    This is what the challenge's bUnwarpJ baseline writes as
    ``warped_source_landmarks.txt``.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "point":
        raise ValueError(f"{path}: not an ImageJ point file (first line {lines[:1]!r})")
    n = int(lines[1])
    rows = [tuple(float(v) for v in ln.split()[:2]) for ln in lines[2 : 2 + n]]
    if len(rows) != n:
        raise ValueError(f"{path}: declares {n} points but carries {len(rows)}")
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def tre(warped_xy, target_xy) -> np.ndarray:
    """Per-landmark Euclidean distance in pixels, over the index-paired prefix."""
    a = np.asarray(warped_xy, dtype=float).reshape(-1, 2)
    b = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    n = min(len(a), len(b))
    return np.sqrt(((a[:n] - b[:n]) ** 2).sum(axis=1))


def rtre(tre_px, diagonal: float) -> np.ndarray:
    if not diagonal or diagonal <= 0:
        raise ValueError(f"image diagonal must be positive, got {diagonal!r}")
    return np.asarray(tre_px, dtype=float) / float(diagonal)


def robustness(rtre_regist, rtre_init) -> float:
    """Fraction of landmarks the registration brought CLOSER than the initial pose."""
    a = np.asarray(rtre_regist, dtype=float)
    b = np.asarray(rtre_init, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] < b[:n]))


def case_stats(warped_xy, target_xy, source_xy, diagonal: float) -> dict:
    """The challenge's per-case numbers.

    ``source_xy`` is the unwarped source landmark set; the initial error the
    robustness term compares against is its distance to the target landmarks
    (both images of a case share one scale, so the identity is the initial pose).
    """
    warped = np.asarray(warped_xy, dtype=float).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=float).reshape(-1, 2)
    source = np.asarray(source_xy, dtype=float).reshape(-1, 2)
    n = min(len(warped), len(target))
    if n == 0:
        return {
            "n_landmarks": 0,
            "complete": False,
            "rtre_median": float("nan"),
            "rtre_mean": float("nan"),
            "rtre_max": float("nan"),
            "tre_median_px": float("nan"),
            "robustness": float("nan"),
        }
    t_px = tre(warped, target)
    r = rtre(t_px, diagonal)
    r_init = rtre(tre(source, target), diagonal)
    return {
        "n_landmarks": int(n),
        # the challenge needs a warped point for every source landmark
        "complete": bool(len(warped) >= len(source)),
        "rtre_median": float(np.median(r)),
        "rtre_mean": float(np.mean(r)),
        "rtre_max": float(np.max(r)),
        "tre_median_px": float(np.median(t_px)),
        "robustness": robustness(r, r_init),
    }
