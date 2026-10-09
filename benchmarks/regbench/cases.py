"""The on-disk contract between datasets, methods and the scorer.

A dataset's ``prepare`` writes one folder per case::

    <cases>/<dataset>/<case_id>/case.json    the images, the pixel size, the patient
    <cases>/<dataset>/<case_id>/points.npz   the nuclei to warp and the truth to score against

``points.npz`` holds, all in X, Y full-resolution pixels (pixel centres on the integers):

``cells_<side>_xy``          every vertex of every ring of the nuclei segmented on the NATIVE
``cells_<side>_ring_off``    slide of that side (``ref`` or ``mov``): a full
``cells_<side>_ring_feat``   ``cell_pairs.PolySet``, holes and multi-part cells included
``cells_<side>_ring_hole``
``cells_<side>_n``
``truth_mov`` / ``truth_ref``           (known-truth cases) paired probes: centroids of
                                        moving nuclei and where the true map puts them;
                                        ``truth_cell`` is each probe's moving nucleus
``cells_mov_xy_truth``                  (known-truth cases) the moving outlines under the true map

A method reads the two images and returns a point mapping in its own direction
(``mapping.PointMap``). The driver moves the nuclei that direction and writes::

    <out>/<variant>/<dataset>/<case_id>/warped.npz   ``cells_xy``, ``direction``, ``truth_xy``,
                                                     ``grid`` (+ ``band``)
    <out>/<variant>/<dataset>/<case_id>/run.json     ok, cause, seconds, resources, info

so the scorer needs neither the method's software nor its transform format.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import pairs

SIDES = ("ref", "mov")
_SET_KEYS = ("xy", "ring_off", "ring_feat", "ring_hole", "n")


@dataclass
class Case:
    dataset: str
    case_id: str
    group: str  # the PATIENT, in every dataset: the unit a per-patient method registers
    ref_nuclear: str  # one-channel OME-TIFF, nuclei bright
    mov_nuclear: str
    ref_image: str  # what a method that reads the original modality should use
    mov_image: str
    modality: str  # "fluorescence" | "brightfield"
    diagonal: float  # of the reference image, px
    pixel_size_um: float | None = None
    ref_hw: list | None = None  # full-resolution (H, W) of each slide
    mov_hw: list | None = None
    extra: dict = field(default_factory=dict)
    dir: str = ""
    family: str = ""  # deformation family of a known-truth case; empty on real slides
    pose: str = ""  # rigid pose of a known-truth case; empty on real slides

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
    _atomic(
        path, lambda tmp: Path(tmp).write_text(json.dumps(obj, indent=1, default=str))
    )


def write_npz(path, **arrays):
    def write(tmp):
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    _atomic(path, write)


def polyset_arrays(ps, side):
    """The ``cells_<side>_*`` arrays of a polygon set (a ``Polys`` or a ``PolySet``)."""
    ps = pairs.as_polyset(ps)
    return {
        f"cells_{side}_xy": np.asarray(ps.xy, dtype=np.float32).reshape(-1, 2),
        f"cells_{side}_ring_off": np.asarray(ps.ring_off, dtype=np.int64),
        f"cells_{side}_ring_feat": np.asarray(ps.ring_feat, dtype=np.int64),
        f"cells_{side}_ring_hole": np.asarray(ps.ring_hole, dtype=bool),
        f"cells_{side}_n": np.array(int(ps.n_features), dtype=np.int64),
    }


def write_case(
    cases_root,
    case,
    truth_mov=None,
    truth_ref=None,
    cells_mov=None,
    cells_ref=None,
    **extra_arrays,
):
    """Write points first and ``case.json`` last: its presence means the case is complete."""
    d = case_dir(cases_root, case.dataset, case.case_id)
    case.dir = str(d)
    arrays = {}
    if truth_mov is not None:
        arrays["truth_mov"] = np.asarray(truth_mov, dtype=float).reshape(-1, 2)
    if truth_ref is not None:
        arrays["truth_ref"] = np.asarray(truth_ref, dtype=float).reshape(-1, 2)
    for side, ps in (("mov", cells_mov), ("ref", cells_ref)):
        if ps is not None:
            arrays.update(polyset_arrays(ps, side))
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
    dirs = sorted(
        (d for d in root.iterdir() if (d / "case.json").exists()),
        key=lambda d: (0, int(d.name), "") if d.name.isdigit() else (1, 0, d.name),
    )
    return [load_case(d) for d in dirs]


def load_points(case):
    with np.load(case.points_path) as z:
        return {k: z[k] for k in z.files}


def polys(points, side):
    """The nuclei of one side (``"ref"`` or ``"mov"``) as a ``cell_pairs.PolySet``, or None
    when the case has none."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if f"cells_{side}_xy" not in points:
        return None
    return pairs.cp.PolySet(
        xy=np.asarray(points[f"cells_{side}_xy"], dtype=float).reshape(-1, 2),
        ring_off=np.asarray(points[f"cells_{side}_ring_off"], dtype=np.int64),
        ring_feat=np.asarray(points[f"cells_{side}_ring_feat"], dtype=np.int64),
        ring_hole=np.asarray(points[f"cells_{side}_ring_hole"], dtype=bool),
        n_features=int(points[f"cells_{side}_n"]),
    )


def groups(cases):
    """``{patient: [cases]}`` in ``list_cases`` order: what a per-patient method is handed."""
    out = {}
    for c in cases:
        out.setdefault(c.group, []).append(c)
    return out


def result_dir(out, variant, case):
    return Path(out) / variant / case.dataset / str(case.case_id)
