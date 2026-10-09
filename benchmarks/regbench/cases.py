"""The on-disk contract between datasets, methods and the scorer.

A dataset's ``prepare`` writes one folder per case::

    <cases>/<dataset>/<case_id>/case.json    the images, the pixel size, the group
    <cases>/<dataset>/<case_id>/points.npz   the points to warp and the truth to score against

``points.npz`` holds, all in X, Y full-resolution pixels (pixel centres on the integers):

``landmarks_mov`` / ``landmarks_ref``   paired points: moving-frame positions and where a
                                         perfect registration puts them in the reference
``cells_mov_xy`` / ``cells_mov_off``    nuclear outlines segmented on the NATIVE moving image
``cells_ref_xy`` / ``cells_ref_off``    nuclear outlines segmented on the reference image
``cells_mov_xy_truth``                  (known-truth cases) the moving outlines under the true map

A method reads the two images and warps ``landmarks_mov`` and ``cells_mov_xy`` into the
reference frame, nothing else. It writes::

    <out>/<variant>/<dataset>/<case_id>/warped.npz   ``landmarks``, ``cells_xy``
    <out>/<variant>/<dataset>/<case_id>/run.json     ok, seconds, error, versions

so the scorer needs neither the method's software nor its transform format.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import DATASETS, cells  # noqa: F401

POINT_KEYS = ("landmarks_mov", "landmarks_ref", "cells_mov_xy", "cells_mov_off",
              "cells_ref_xy", "cells_ref_off")


@dataclass
class Case:
    dataset: str
    case_id: str
    group: str  # tissue (ANHIR), deformation family (synthetic), patient (multiplex)
    ref_nuclear: str  # one-channel OME-TIFF, nuclei bright
    mov_nuclear: str
    ref_image: str  # what a method that reads the original modality should use
    mov_image: str
    modality: str  # "fluorescence" | "brightfield"
    diagonal: float  # of the reference image, px: the rTRE denominator
    pixel_size_um: float | None = None
    ref_hw: list | None = None  # full-resolution (H, W) of each slide
    mov_hw: list | None = None
    extra: dict = field(default_factory=dict)
    dir: str = ""

    @property
    def points_path(self):
        return Path(self.dir) / "points.npz"


def case_dir(cases_root, dataset, case_id):
    return Path(cases_root) / dataset / str(case_id)


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    _atomic(path, lambda tmp: Path(tmp).write_text(json.dumps(obj, indent=1, default=str)))


def write_npz(path, **arrays):
    def write(tmp):
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    _atomic(path, write)


def write_case(cases_root, case, landmarks_mov=None, landmarks_ref=None, cells_mov=None,
               cells_ref=None, **extra_arrays):
    """Write points first and ``case.json`` last: its presence means the case is complete."""
    d = case_dir(cases_root, case.dataset, case.case_id)
    case.dir = str(d)
    arrays = {}
    if landmarks_mov is not None:
        arrays["landmarks_mov"] = np.asarray(landmarks_mov, dtype=float).reshape(-1, 2)
    if landmarks_ref is not None:
        arrays["landmarks_ref"] = np.asarray(landmarks_ref, dtype=float).reshape(-1, 2)
    for key, ps in (("cells_mov", cells_mov), ("cells_ref", cells_ref)):
        if ps is not None:
            arrays[f"{key}_xy"], arrays[f"{key}_off"] = ps.xy.astype(np.float32), ps.off
    arrays.update({k: np.asarray(v) for k, v in extra_arrays.items() if v is not None})
    write_npz(d / "points.npz", **arrays)
    write_json(d / "case.json", asdict(case))
    return d


def load_case(path):
    path = Path(path)
    rec = json.loads((path / "case.json").read_text())
    rec["dir"] = str(path)
    return Case(**rec)


def list_cases(cases_root, dataset):
    """Complete cases of a dataset, in a stable order (array task ``i`` is always the same case)."""
    root = Path(cases_root) / dataset
    if not root.exists():
        return []
    dirs = sorted((d for d in root.iterdir() if (d / "case.json").exists()),
                  key=lambda d: (0, int(d.name), "") if d.name.isdigit() else (1, 0, d.name))
    return [load_case(d) for d in dirs]


def load_points(case):
    with np.load(case.points_path) as z:
        return {k: z[k] for k in z.files}


def polys(points, which):
    """``cells_ref`` or ``cells_mov`` of a points dict, or None when the case has no cells."""
    if f"{which}_xy" not in points:
        return None
    return cells.Polys(points[f"{which}_xy"].astype(float), points[f"{which}_off"])


def result_dir(out, variant, case):
    return Path(out) / variant / case.dataset / str(case.case_id)
