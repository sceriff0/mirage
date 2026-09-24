"""Keep the low-resolution rigid transform when VALIS's micro-rigid pass finds no matches.

VALIS-free on purpose, like ``valis_preflight``: this runs (and is tested) where
``valis`` is not installed. ``bin/register.py`` hands it
``valis.micro_rigid_registrar.MicroRigidRegistrar`` to patch.

THE DEFECT THIS CLOSES -- a TMA run on 2026-09-24 (patient 005, all four attempts).
``MicroRigidRegistrar`` runs inside ``Valis.register()`` whenever ``micro_reg >= 1``
(``bin/utils/valis_config.py``). For each moving/fixed pair, ``align_slides`` cuts the
tissue ROI into tiles, matches each tile with SuperPoint/SuperGlue, and drops tiles
that yield fewer than three matches (``_match_tile`` returns ``None`` and prints the
exception). Then it stacks what is left::

    high_rez_moving_match_xy = np.vstack(high_rez_moving_match_xy_list)   # :304 (1.0.0)

A TMA core's ROI is tiny (``ROI width, height is [78.85 160.53] pixels``), so its tiles
are around 64x80 px. SuperPoint's pooling takes that to zero height
(``Given input size: (64x80x1). Calculated output size: (64x40x0). Output size is too
small``), every tile returns ``None``, and ``np.vstack([])`` raises ``need at least one
array to concatenate``. ``Valis.register()``'s catch-all prints the exception as a
``UserWarning``, kills the JVM and returns ``(None, None, None)``. The whole patient
fails on one pair, deterministically, so a retry with more memory cannot help.

THE FIX wraps ``align_slides`` so that an exception there keeps the pair's
low-resolution registration. That is the same outcome VALIS already chooses when the
pass succeeds but ``did not improve alignments. Keeping low rez registration
parameters``. It is safe by construction: ``align_slides`` writes to the moving slide
(``M``, ``xy_matched_to_prev``, ``xy_in_prev`` and the ``*_in_bbox`` pair) only in its
last block, after every call that can raise. An exception therefore leaves the slide
exactly as the rigid stage left it. ``register()`` loops over pairs, so the other pairs
still get their micro-rigid refinement.

``MemoryError`` is re-raised. An out-of-memory error is a resource problem that the
retry ramp exists for, and quietly downgrading the registration instead would hide it.
"""

from __future__ import annotations

import functools
import inspect
import types
from typing import List, Tuple

EXPECTED_PARAMS = ("self", "moving_slide", "fixed_slide")

# One entry per pair that fell back: (moving slide, fixed slide, exception text).
# bin/register.py reads it after register() to put a summary in the task log.
FALLBACKS: List[Tuple[str, str, str]] = []


def _guarded(fn):
    """Wrap ``align_slides`` so an exception keeps the low-resolution transform."""

    @functools.wraps(fn)
    def wrapped(self, moving_slide, fixed_slide, *args, **kwargs):
        try:
            return fn(self, moving_slide, fixed_slide, *args, **kwargs)
        except MemoryError:
            raise
        except Exception as e:
            moving = getattr(moving_slide, "name", str(moving_slide))
            fixed = getattr(fixed_slide, "name", str(fixed_slide))
            FALLBACKS.append((moving, fixed, f"{type(e).__name__}: {e}"))
            print(
                f"[mirage] micro-rigid registration failed for {moving} -> {fixed} "
                f"({type(e).__name__}: {e}). Keeping low rez registration parameters "
                f"(bin/utils/valis_micro_rigid.py)."
            )
            return None

    wrapped._mirage_micro_rigid_guard = True
    return wrapped


def guard_micro_rigid(registrar_cls) -> bool:
    """Patch ``registrar_cls.align_slides`` so a failed pair keeps its rigid transform.

    Returns True only when THIS call installed the guard. It returns False when the
    guard is already installed, and also when ``align_slides`` is missing or its
    signature no longer starts ``(self, moving_slide, fixed_slide)`` (a VALIS release
    that changed the method). The caller logs the second case, because it means the
    defect is open again.
    """
    fn = getattr(registrar_cls, "align_slides", None)
    if not isinstance(fn, types.FunctionType) or is_guarded(registrar_cls):
        return False
    params = tuple(inspect.signature(fn).parameters)[: len(EXPECTED_PARAMS)]
    if params != EXPECTED_PARAMS:
        return False
    registrar_cls.align_slides = _guarded(fn)
    return True


def is_guarded(registrar_cls) -> bool:
    """True when ``registrar_cls.align_slides`` already carries the guard."""
    return getattr(
        getattr(registrar_cls, "align_slides", None), "_mirage_micro_rigid_guard", False
    )
