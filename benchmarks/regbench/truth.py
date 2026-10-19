"""Landmark error: how far a method's map leaves a landmark from its partner.

Restored from commit 5a46f778 (``benchmarks/regbench/truth.py``) with ``error_stats`` and
``jacobian_stats`` only, the two the landmark scorers call (``regbench.landmarks``,
``regbench.challenge``). The known-truth spec, its families and its probes are not
restored: nothing in this harness generates a case any more.

**Direction.** A method is asked only for what it estimated. For a ``mov_to_ref`` map the
driver stores ``fn(moving landmarks)`` and the error is its distance to the reference
landmarks; for a ``ref_to_mov`` map it stores ``fn(reference landmarks)`` and the error is
its distance to the moving ones (``warped.npz: truth_xy`` / ``truth_side``,
``regbench.landmark_io``). Neither map is ever turned around.
"""

from __future__ import annotations

import math

import numpy as np

ERROR_KEYS = (
    "n_probes",
    "n_nonfinite",
    "tre_um_p50",
    "tre_um_p90",
    "tre_um_max",
    "frac_under_1um",
    "frac_under_2um",
    "initial_tre_um_p50",
    "frac_improved",
)
JACOBIAN_KEYS = ("fold_pct", "sd_log_jac")


def _percentile(sorted_values, q):
    """Linear percentile of an ascending array that may end in ``inf``."""
    n = len(sorted_values)
    pos = q / 100.0 * (n - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    a, b = float(sorted_values[lo]), float(sorted_values[hi])
    if lo == hi or a == b:
        return a
    if math.isinf(b):
        return math.inf
    return a + (b - a) * (pos - lo)


def error_stats(mapped, target, source, pixel_size_um):
    """Error of ``mapped = fn(source)`` against ``target``, in micrometres.

    A point the method returned as NaN or infinite counts AGAINST it: its error is
    infinite, so it is under no threshold, it is not improved, and it pulls the percentiles
    up. Without a pixel size the micrometre numbers are NaN, never zero.
    """
    out = dict.fromkeys(ERROR_KEYS, math.nan)
    mapped = np.asarray(mapped, dtype=float).reshape(-1, 2)
    target = np.asarray(target, dtype=float).reshape(-1, 2)
    source = np.asarray(source, dtype=float).reshape(-1, 2)
    out["n_probes"], out["n_nonfinite"] = len(target), 0
    if not len(target) or len(mapped) != len(target):
        return out
    bad = ~np.isfinite(mapped).all(axis=1)
    err = np.full(len(target), math.inf)
    err[~bad] = np.hypot(*(mapped[~bad] - target[~bad]).T)
    init = np.hypot(*(source - target).T)
    out["n_nonfinite"] = int(bad.sum())
    out["frac_improved"] = float(np.mean(err < init))
    if not pixel_size_um:
        return out
    px = float(pixel_size_um)
    err_um, init_um = np.sort(err * px), np.sort(init * px)
    out["tre_um_p50"] = _percentile(err_um, 50)
    out["tre_um_p90"] = _percentile(err_um, 90)
    out["tre_um_max"] = float(err_um[-1])
    out["frac_under_1um"] = float(np.mean(err_um < 1.0))
    out["frac_under_2um"] = float(np.mean(err_um < 2.0))
    out["initial_tre_um_p50"] = _percentile(init_um, 50)
    return out


def jacobian_stats(warped):
    """``fold_pct`` and ``sd_log_jac`` of a method's map, from the lattice the driver
    pushed through it (``warped.npz``: ``grid``, ``grid_shape``, ``grid_step``): the
    cohort scorer's own definition (``score.jacobian_stats``), one definition for both."""
    from .score import jacobian_stats as of_score

    return of_score(warped)
