"""Ground-truth landmarks of a case: a SIDECAR ``truth.npz`` beside ``case.json``.

``points.npz`` and ``cases.py`` are not touched: a case without landmarks (the cohort) has
no sidecar, and adding one changes nothing a result was made for except the landmarks.

The file holds ``mov`` (N x 2) and, when the dataset has them, ``ref`` (N x 2), X, Y in
full-resolution pixels of the NATIVE slides, paired by row. A ONE-SIDED truth (only the
moving slide's points, no ``ref``: ACROBAT's IHC points) is allowed; row pairing is
checked only when both sides exist. Every point must be finite.

The driver (``methods``) moves, at each yield, the side the map takes as input: the
moving landmarks under ``mov_to_ref``, the reference landmarks under ``ref_to_mov``,
into ``warped.npz: truth_xy`` (``truth_side`` says which side was moved). In each
``run.json`` it records ``truth_sha256`` (:func:`truth_sha256`, None without a sidecar:
a result is reused only for the same value) and ``truth_note``: ``""`` when
``truth_xy`` was written, else why not (:data:`NO_TRUTH`,
:data:`ONE_SIDED_REF_TO_MOV`). A failure record (no map, so nothing moved) says
:data:`NO_TRUTH` for a case without a sidecar and :data:`RUN_FAILED` for one with
(:func:`failure_note`): whether a one-sided truth would have been moved is not known
without the map's direction.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

FILE = "truth.npz"
NO_TRUTH = "no truth.npz beside case.json"
ONE_SIDED_REF_TO_MOV = (
    "the truth has moving points only and this map takes reference points "
    "(ref_to_mov): nothing to move"
)
RUN_FAILED = "the run failed: no landmarks were moved"


def _path(case_or_dir):
    d = getattr(case_or_dir, "dir", case_or_dir)
    return Path(d) / FILE


def _checked(path, ref, mov):
    """``(ref | None, mov)`` as float64 N x 2, or ValueError naming ``path``."""
    out = []
    for side, xy in (("ref", ref), ("mov", mov)):
        if xy is None:
            if side == "mov":
                raise ValueError(f"{path}: the moving landmarks are required")
            out.append(None)
            continue
        a = np.asarray(xy, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"{path}: {side} landmarks must be N x 2, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{path}: {side} landmarks are not all finite")
        out.append(a)
    ref, mov = out
    if ref is not None and len(ref) != len(mov):
        raise ValueError(
            f"{path}: {len(ref)} reference and {len(mov)} moving landmarks; "
            "they are paired by row"
        )
    return ref, mov


def _umask_mode():
    """The mode a new file gets here (``0o666 & ~umask``). The umask is only readable by
    setting it, so it is set back at once."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def write_truth(case_dir, truth_ref, truth_mov):
    """Write the sidecar (``truth_ref`` may be None: a one-sided truth). Atomic: a
    temporary name in the same folder, then ``os.replace``. ``mkstemp`` creates the file
    0600 and ``os.replace`` keeps that, so it is given the umask default first."""
    path = _path(case_dir)
    ref, mov = _checked(path, truth_ref, truth_mov)
    arrays = {"mov": mov} if ref is None else {"ref": ref, "mov": mov}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_truth(case):
    """``{"ref": array | None, "mov": array}``, or None when the case has no sidecar."""
    path = _path(case)
    if not path.is_file():
        return None
    with np.load(path) as z:
        if "mov" not in z.files:
            raise ValueError(f"{path}: no 'mov' array (found {sorted(z.files)})")
        ref, mov = _checked(path, z["ref"] if "ref" in z.files else None, z["mov"])
    return {"ref": ref, "mov": mov}


def failure_note(case):
    """``truth_note`` of a failure record: :data:`NO_TRUTH` without a sidecar, else
    :data:`RUN_FAILED`."""
    return NO_TRUTH if not _path(case).is_file() else RUN_FAILED


def truth_sha256(case):
    """sha256 of the sidecar's bytes, or None when the case has none."""
    path = _path(case)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
