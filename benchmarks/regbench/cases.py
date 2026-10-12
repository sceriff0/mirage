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
``nuclei_<side>_xy`` ...     (since 2026-10-11, optional) a SECOND polygon set per side with
                             the same five arrays: the BARE nuclei. ``cells_<side>_*`` are
                             the OUTLINES (on the cohort: each nucleus grown by 10 px, what
                             the pipeline's QC publishes). ``SETS`` names the two.
``nuclei_<side>_outline``    (optional) for each nucleus, the index of the outline it
                             belongs to; written only where that is known by construction
                             (a nucleus then inherits its outline's seam flag)
``truth_mov`` / ``truth_ref``           (known-truth cases) paired probes: centroids of
                                        moving nuclei and where the true map puts them;
                                        ``truth_cell`` is each probe's moving nucleus
``cells_mov_xy_truth``                  (known-truth cases) the moving outlines under the true map
``nuclei_mov_xy_truth``                 the same for the moving nuclei

``case.json: extra.cell_sets`` says where each set of each side came from:
``{"ref": {"outlines": {"source": ...}, "nuclei": {"source": ..., "exact": bool, ...}},
"mov": {...}}``; a side without nuclei has ``"nuclei": {"source": "", "reason": why}``.
``source`` is one of ``SOURCES`` (``label_image:<path>`` carries its file). A case written
before the second set existed has no ``nuclei_*`` array and no ``cell_sets``: it loads and
is scored on the outlines alone (``nuclei_reason``).

A method reads the two images and returns a point mapping in its own direction
(``mapping.PointMap``). The driver moves the nuclei that direction and writes::

    <out>/<variant>/<dataset>/<case_id>/warped.npz   ``cells_xy``, ``nuclei_xy``, ``direction``,
                                                     ``truth_xy``, ``grid`` (+ ``band``,
                                                     ``band_nuc``)
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
# The two polygon sets of a side, and the array prefix of each in ``points.npz``. The
# outlines keep the prefix they always had, so a case and a result written before the
# nuclei existed read as before.
OUTLINES, NUCLEI = "outlines", "nuclei"
SETS = (OUTLINES, NUCLEI)
PREFIX = {OUTLINES: "cells", NUCLEI: "nuclei"}
# where a set can come from (``case.json: extra.cell_sets``)
SOURCES = (
    "arm_geojson",  # the outlines an arm of the pipeline published
    "label_image",  # an existing label image of that slide (``label_image:<path>``)
    "stardist_rerun",  # the pipeline's bin/segment.py run again on the native slide
    "shrunk_outlines",  # the outline shrunk by the expansion distance: NOT a nucleus
    "builtin",  # the benchmark's own threshold + watershed segmenter
)
NO_NUCLEI_OLD_CASE = (
    "the case was prepared before the bare-nuclei set existed (no nuclei in points.npz)"
)
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


def _prefix(which):
    if which not in SETS:
        raise ValueError(f"set must be one of {SETS}, got {which!r}")
    return PREFIX[which]


def polyset_arrays(ps, side, which=OUTLINES):
    """The ``<prefix>_<side>_*`` arrays of a polygon set (a ``Polys`` or a ``PolySet``);
    ``which`` is the set, ``"outlines"`` (prefix ``cells``) or ``"nuclei"``."""
    ps = pairs.as_polyset(ps)
    p = _prefix(which)
    return {
        f"{p}_{side}_xy": np.asarray(ps.xy, dtype=np.float32).reshape(-1, 2),
        f"{p}_{side}_ring_off": np.asarray(ps.ring_off, dtype=np.int64),
        f"{p}_{side}_ring_feat": np.asarray(ps.ring_feat, dtype=np.int64),
        f"{p}_{side}_ring_hole": np.asarray(ps.ring_hole, dtype=bool),
        f"{p}_{side}_n": np.array(int(ps.n_features), dtype=np.int64),
    }


def write_case(
    cases_root,
    case,
    truth_mov=None,
    truth_ref=None,
    cells_mov=None,
    cells_ref=None,
    nuclei_mov=None,
    nuclei_ref=None,
    **extra_arrays,
):
    """Write points first and ``case.json`` last: its presence means the case is complete.

    ``cells_*`` are the outlines, ``nuclei_*`` the bare nuclei (either may be absent)."""
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
    for side, ps in (("mov", nuclei_mov), ("ref", nuclei_ref)):
        if ps is not None:
            arrays.update(polyset_arrays(ps, side, NUCLEI))
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


# ── what a file held when something was computed from it ──────────────────────
# A whole-slide file is tens of gigabytes and a tile job reads a few megabytes of it:
# hashing it whole in every job would cost more than the job. The fingerprint is the
# file's size, its modification time, and a sha256 over SAMPLE_BLOCKS blocks of
# SAMPLE_BYTES spread evenly from its first byte to its last (the first and last blocks
# included: a TIFF's header and, for most writers, its directories). Any rewrite
# changes the time; a file put back with the same size AND time (`cp -p`, `rsync -t`)
# is caught where it differs inside a sampled block. NOT caught: a file of the same
# size and time that differs only between the samples.
SAMPLE_BLOCKS = 16
SAMPLE_BYTES = 1 << 16
_FILE_FINGERPRINTS = {}


def file_fingerprint(path):
    """``{"size", "mtime_ns", "sample_sha256"}`` of a file as it is now, or
    ``{"missing": True}``. A file of at most ``SAMPLE_BLOCKS * SAMPLE_BYTES`` bytes is
    hashed whole. Cached per file state."""
    import hashlib

    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return {"missing": True}
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _FILE_FINGERPRINTS:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            if st.st_size <= SAMPLE_BLOCKS * SAMPLE_BYTES:
                h.update(fh.read())
            else:
                last = st.st_size - SAMPLE_BYTES
                for i in range(SAMPLE_BLOCKS):
                    fh.seek(last * i // (SAMPLE_BLOCKS - 1))
                    h.update(fh.read(SAMPLE_BYTES))
        _FILE_FINGERPRINTS[key] = {
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "sample_sha256": h.hexdigest(),
        }
    return dict(_FILE_FINGERPRINTS[key])


def inputs_fingerprint(case):
    """``{field: file_fingerprint}`` of the image files a case names (the slides and
    their one-channel copies): what a result of the case was computed from."""
    out = {}
    for name in ("ref_image", "mov_image", "ref_nuclear", "mov_nuclear"):
        value = getattr(case, name, "")
        if value:
            out[name] = file_fingerprint(value)
    return out


_FINGERPRINTS = {}
POINT_GROUPS = {"cells_": "outlines", "nuclei_": "nuclei", "truth_": "truth"}


def points_fingerprint(case):
    """``{"outlines", "nuclei", "truth"}`` -> sha256 of that group of arrays of the
    case's ``points.npz`` as it is on disk now (``""`` for a group it does not have).

    A result is made FOR these points: it moved them. A case prepared again with other
    points makes every earlier result of it stale, never a failure of the method
    (:func:`stale_points`). The groups are apart so that ADDING one (bare nuclei to a
    case that had outlines only) leaves the results of the others valid. Cached per
    file state (path, size, modification time)."""
    import hashlib

    path = Path(case.points_path)
    out = dict.fromkeys(POINT_GROUPS.values(), "")
    try:
        st = path.stat()
    except OSError:
        return out
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _FINGERPRINTS:
        hashes = {}
        with np.load(path) as z:
            for name in sorted(z.files):
                group = next(
                    (g for p, g in POINT_GROUPS.items() if name.startswith(p)), None
                )
                if group is None:
                    continue
                h = hashes.setdefault(group, hashlib.sha256())
                a = np.ascontiguousarray(z[name])
                h.update(f"{name}:{a.dtype}:{a.shape}:".encode())
                h.update(a.tobytes())
        _FINGERPRINTS[key] = {**out, **{g: h.hexdigest() for g, h in hashes.items()}}
    return dict(_FINGERPRINTS[key])


def stale_points(made_for, case):
    """The groups of points a result was made for that the case no longer has in that
    form (``made_for`` is the result's recorded fingerprint; a record without one, or a
    group the result never had, is not stale)."""
    if not isinstance(made_for, dict):
        return []
    now = points_fingerprint(case)
    return [g for g, h in made_for.items() if h and h != now.get(g, "")]


def load_points(case):
    with np.load(case.points_path) as z:
        return {k: z[k] for k in z.files}


def polys(points, side, which=OUTLINES):
    """One polygon set of one side (``"ref"`` or ``"mov"``) as a ``cell_pairs.PolySet``, or
    None when the case has none: the outlines by default, the bare nuclei with
    ``which="nuclei"``."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    p = _prefix(which)
    if f"{p}_{side}_xy" not in points:
        return None
    return pairs.cp.PolySet(
        xy=np.asarray(points[f"{p}_{side}_xy"], dtype=float).reshape(-1, 2),
        ring_off=np.asarray(points[f"{p}_{side}_ring_off"], dtype=np.int64),
        ring_feat=np.asarray(points[f"{p}_{side}_ring_feat"], dtype=np.int64),
        ring_hole=np.asarray(points[f"{p}_{side}_ring_hole"], dtype=bool),
        n_features=int(points[f"{p}_{side}_n"]),
    )


def set_info(case, side, which):
    """What ``case.json`` records of one set of one side (``extra.cell_sets``), ``{}``
    for a case that records nothing."""
    return dict(((case.extra.get("cell_sets") or {}).get(side) or {}).get(which) or {})


def nuclei_reason(case, points):
    """``""`` when both sides have bare nuclei, else why not, in words: the prepared
    reason of the first side without them, or that the case predates the set."""
    why = []
    for side in SIDES:
        ps = polys(points, side, NUCLEI)
        if ps is not None and ps.n_features:
            continue
        said = set_info(case, side, NUCLEI).get("reason")
        if not said and not case.extra.get("cell_sets"):
            said = NO_NUCLEI_OLD_CASE
        why.append(f"{side}: {said or 'no nucleus in the set'}")
    return "; ".join(why)


def groups(cases):
    """``{patient: [cases]}`` in ``list_cases`` order: what a per-patient method is handed."""
    out = {}
    for c in cases:
        out.setdefault(c.group, []).append(c)
    return out


def result_dir(out, variant, case):
    return Path(out) / variant / case.dataset / str(case.case_id)
