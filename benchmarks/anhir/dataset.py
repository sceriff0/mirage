"""The ANHIR cover table (``dataset_medium.csv`` and friends) as typed cases.

The challenge ships one CSV listing every registration case: source and target
image, their landmark files, image size and diagonal, and whether the case is
``training`` (target landmarks public) or ``evaluation`` (target landmarks held
server-side). Every path in it is relative to the image archive root and the
landmark archive root, which ANHIR distributes separately with the same tree
(``<tissue>_<n>/scale-<k>pc/<name>``).
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

COL_ID = "case_id"
COL_DIAGONAL = "Image diagonal [pixels]"
COL_SIZE = "Image size [pixels]"
COL_SOURCE = "Source image"
COL_SOURCE_LND = "Source landmarks"
COL_TARGET = "Target image"
COL_TARGET_LND = "Target landmarks"
COL_STATUS = "status"
COL_WARPED_SOURCE = "Warped source landmarks"
COL_WARPED_TARGET = "Warped target landmarks"
COL_TIME = "Execution time [minutes]"

COVER_COLUMNS = (
    COL_DIAGONAL,
    COL_SIZE,
    COL_SOURCE,
    COL_SOURCE_LND,
    COL_TARGET,
    COL_TARGET_LND,
    COL_STATUS,
)
STATUS_TRAINING = "training"
STATUS_EVALUATION = "evaluation"
STATUSES = (STATUS_TRAINING, STATUS_EVALUATION)

_TISSUE_SUFFIX = re.compile(r"_\d+$")


@dataclass(frozen=True)
class Case:
    """One registration case: warp ``source`` onto ``target``."""

    case_id: int
    tissue: str
    scale: str
    status: str
    source_image: str
    source_landmarks: str
    target_image: str
    target_landmarks: str
    size: tuple  # as the cover table states it, (dim0, dim1)
    diagonal: float

    @property
    def source_stem(self) -> str:
        return Path(self.source_image).stem

    @property
    def target_stem(self) -> str:
        return Path(self.target_image).stem

    @property
    def scored_locally(self) -> bool:
        return self.status == STATUS_TRAINING


def tissue_of(rel_path: str) -> str:
    """``COAD_01/scale-25pc/S1.jpg`` -> ``COAD``; ``lung-lesion_1/...`` -> ``lung-lesion``."""
    head = Path(rel_path).parts[0]
    return _TISSUE_SUFFIX.sub("", head)


def scale_of(rel_path: str) -> str:
    """``COAD_01/scale-25pc/S1.jpg`` -> ``scale-25pc``."""
    parts = Path(rel_path).parts
    return parts[1] if len(parts) > 1 else ""


def _parse_size(value) -> tuple:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return tuple(int(v) for v in parsed)
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    raise ValueError(f"unreadable image size {value!r}")


def _diagonal(row) -> float:
    d = row.get(COL_DIAGONAL)
    if d is not None and not (isinstance(d, float) and math.isnan(d)):
        return float(d)
    a, b = _parse_size(row[COL_SIZE])
    return float(math.hypot(a, b))


def load_cases(csv_path) -> list:
    """Read the cover table. The unnamed first column is the case id."""
    df = pd.read_csv(csv_path, index_col=0)
    missing = [c for c in COVER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing cover column(s) {missing}")
    cases = []
    for idx, row in df.iterrows():
        status = str(row[COL_STATUS]).strip()
        if status not in STATUSES:
            raise ValueError(
                f"{csv_path}: case {idx} has status {status!r}, expected one of {STATUSES}"
            )
        cases.append(
            Case(
                case_id=int(idx),
                tissue=tissue_of(row[COL_SOURCE]),
                scale=scale_of(row[COL_SOURCE]),
                status=status,
                source_image=str(row[COL_SOURCE]),
                source_landmarks=str(row[COL_SOURCE_LND]),
                target_image=str(row[COL_TARGET]),
                target_landmarks=str(row[COL_TARGET_LND]),
                size=_parse_size(row[COL_SIZE]),
                diagonal=_diagonal(row),
            )
        )
    return cases


def load_cover(csv_path) -> pd.DataFrame:
    """The cover table verbatim (index = case id), for writing a submission back out."""
    return pd.read_csv(csv_path, index_col=0)


def select_cases(
    cases: Iterable[Case],
    status: Optional[str] = None,
    tissues: Optional[Iterable[str]] = None,
    case_ids: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
) -> list:
    """Filter cases; ``status='all'`` or None keeps both statuses."""
    tissues = set(tissues) if tissues else None
    ids = set(case_ids) if case_ids else None
    out = []
    for c in cases:
        if status and status != "all" and c.status != status:
            continue
        if tissues and c.tissue not in tissues:
            continue
        if ids and c.case_id not in ids:
            continue
        out.append(c)
    return out[:limit] if limit else out


def resolve_image(case: Case, images_root, which: str) -> Path:
    rel = case.source_image if which == "source" else case.target_image
    return Path(images_root) / rel


def resolve_landmarks(case: Case, landmarks_root, which: str) -> Path:
    rel = case.source_landmarks if which == "source" else case.target_landmarks
    return Path(landmarks_root) / rel
