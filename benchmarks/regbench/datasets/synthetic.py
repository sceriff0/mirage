"""Synthetic nuclear slides with a KNOWN moving -> reference map.

The reference is a field of Gaussian nuclei inside a blobby tissue silhouette. The moving
slide is the same tissue seen through a forward map

    g(p) = c + R(theta) (p - c) + t + w(p)            moving pixel p  ->  reference position

with ``w`` a displacement field: ``mov(p) = base(g(p))``. ``w`` is a smooth two-wave field of
peak amplitude ``amp`` px and wavelength ``wavelength`` px, plus, by ``field``:

``multiscale``  three random fields of correlation length ``scales`` px and RMS ``rms`` px
                each, summed (energy at several scales at once)
``grid``        a ``grid_nodes`` x ``grid_nodes`` (9 x 9) grid of i.i.d. Gaussian
                control-point moves of SD ``grid_sd`` px, cubic-spline interpolated
                (Casamitjana et al., arXiv:2104.14873)
``bumps``       ``bump_count`` local Gaussian bumps of width ``bump_sigma`` px, each peaking at
                0.2-0.6 sigma (at most 50 px), so no single bump folds the map. A bump is
                summed out to ``BUMP_REACH`` (8) sigma and is zero beyond, point by point
``seams``       an independent shift of SD ``seam_sd`` px per ``seam_tile`` px tile: the jump a
                stitching error leaves at a tile boundary. Not continuous, on purpose.
 Because ``g`` is written in the direction a
registration is scored in, the truth for any moving point is one evaluation of ``g``, with no
inversion and no interpolation.

Two things besides geometry differ between the slides, as between two staining rounds:
independent noise, and (``dropout``) a fraction of nuclei missing from the moving slide.
``mov_hw`` gives the moving slide a smaller canvas than the reference (its top-left
``(H, W)`` pixels), because real pairs rarely share a shape and methods that pad or rescale
must get the bookkeeping right. ``ref_hw`` gives the reference a canvas that is not square
(the top-left ``(H, W)`` of the ``n`` x ``n`` scene) and ``tissue_box`` keeps the nuclei
inside a box, so the tissue need not start at the origin; the selfcheck's pair uses both.

**Patients.** A spec with ``patient`` is one ROUND of that patient: ``scene_seed`` fixes the
tissue and the reference slide, which every round of the patient shares (one file), and
``seed`` fixes that round's deformation and noise. ``truth_suite`` builds the frozen
known-truth cases of ``truth_spec.json`` this way: one patient per deformation family, two
rounds each (``regbench.truth``). The ``seed`` of such a plan is the seed ASKED for:
``prepare_one`` first settles it (``settle``), skipping a seed whose true map folds, and
``case.json`` holds the seed used (``extra.spec.seed``) and the record (``extra.reseed``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .. import cells, truth
from ..cases import Case, case_dir, write_case

# The scene generator is the selfcheck's own module now (``regbench.scene``). Its names are
# read back here, and re-exported, for the parts of this dataset that remain until it is
# deleted.
from ..scene import *  # noqa: F403
from ..scene import (
    _BASE,
    _POSE,
    NUCLEUS_DIAMETER,
    PIXEL_SIZE_UM,
    SCALE_SIZES,
    _write_slide,
    forward_map,
)


def _suite(rows):
    return {
        name: {**_BASE, "family": name.rsplit("_", 1)[0].rstrip("0123456789_"), **kw}
        for name, kw in rows
    }


SUITES = {
    # One deformation at every size from 1024 to 65536 px a side (1 Mpx to 4.3 Gpx per
    # slide): the same rotation, shift and smooth warp, so what changes down the suite is
    # the slide, and accuracy and cost can be read against size. Names sort by size.
    "scale": _suite(
        [
            (
                f"scale_{n:05d}",
                dict(
                    n=n,
                    theta=1.0,
                    shift=(30.0, -20.0),
                    amp=10.0,
                    wavelength=1500.0,
                    seed=100,
                    max_cells=200_000,
                ),
            )
            for n in SCALE_SIZES
        ]
    ),
    # What the test-suite registers: seconds per case, one with a smaller moving slide, and
    # two that pin the conditions under which STARE 1.2.1's SOLVE picks a flat field.
    "test": _suite(
        [
            ("rigid", dict(n=1024, theta=2.0, shift=(14.0, -9.0), seed=1)),
            ("smooth", dict(n=1024, amp=5.0, wavelength=600.0, seed=4)),
            (
                "crop",
                dict(
                    n=1024,
                    theta=1.0,
                    shift=(10.0, 6.0),
                    amp=4.0,
                    wavelength=600.0,
                    mov_hw=(832, 928),
                    seed=4,
                ),
            ),
            ("smooth_border", dict(n=1024, amp=5.0, wavelength=600.0, seed=2)),
        ]
    ),
    # Small enough for a CI runner: one rigid, one smooth, one both.
    "ci": _suite(
        [
            ("rigid_0", dict(n=1536, theta=2.0, shift=(18.0, -11.0), seed=1)),
            ("smooth_0", dict(n=1536, amp=6.0, wavelength=700.0, seed=2)),
            (
                "mixed_0",
                dict(
                    n=1536,
                    theta=-1.5,
                    shift=(-9.0, 14.0),
                    amp=5.0,
                    wavelength=700.0,
                    dropout=0.1,
                    seed=3,
                ),
            ),
            (
                "crop_0",
                dict(
                    n=1536,
                    theta=1.0,
                    shift=(12.0, 8.0),
                    amp=4.0,
                    wavelength=700.0,
                    mov_hw=(1280, 1408),
                    seed=4,
                ),
            ),
        ]
    ),
    "full": _suite(
        [
            (f"rigid_{i}", dict(theta=th, shift=sh, seed=10 + i))
            for i, (th, sh) in enumerate(
                [
                    (0.0, (40.0, -25.0)),
                    (3.0, (60.0, 30.0)),
                    (15.0, (-80.0, 50.0)),
                    (90.0, (20.0, 20.0)),
                    (180.0, (-30.0, 10.0)),
                ]
            )
        ]
        + [
            (f"smooth_{i}", dict(amp=a, wavelength=wl, seed=20 + i))
            for i, (a, wl) in enumerate(
                [
                    (3.0, 2000.0),
                    (8.0, 1500.0),
                    (15.0, 1500.0),
                    (15.0, 600.0),
                    (30.0, 1500.0),
                    (60.0, 2500.0),
                ]
            )
        ]
        + [
            (f"mixed_{i}", dict(theta=th, shift=sh, amp=a, wavelength=wl, seed=30 + i))
            for i, (th, sh, a, wl) in enumerate(
                [
                    (2.0, (30.0, -20.0), 8.0, 1500.0),
                    (5.0, (-50.0, 40.0), 15.0, 1000.0),
                    (30.0, (60.0, 60.0), 15.0, 1500.0),
                    (180.0, (10.0, -10.0), 30.0, 2000.0),
                ]
            )
        ]
        + [
            (
                f"noisy_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, noise=nz, seed=40 + i),
            )
            for i, nz in enumerate([0.05, 0.15, 0.4])
        ]
        + [
            (
                f"dropout_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, dropout=d, seed=50 + i),
            )
            for i, d in enumerate([0.1, 0.3, 0.6])
        ]
        + [
            (
                f"sparse_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, keep=k, seed=60 + i),
            )
            for i, k in enumerate([0.4, 0.15])
        ]
        + [
            (
                f"dim_{i}",
                dict(theta=2.0, shift=(30.0, -20.0), amp=10.0, gain=g, seed=70 + i),
            )
            for i, g in enumerate([0.4, 0.1])
        ]
        + [
            (f"multiscale_{i}", dict(**_POSE, field="multiscale", rms=r, seed=80 + i))
            for i, r in enumerate([(4.0, 1.5, 0.5), (8.0, 3.0, 1.0), (16.0, 6.0, 2.0)])
        ]
        + [
            (f"grid_{i}", dict(**_POSE, field="grid", grid_sd=sd, seed=90 + i))
            for i, sd in enumerate([3.0, 5.0, 7.0])
        ]
        # sigma 25 px is finer than STARE's 64 px vector lattice: a case it cannot represent
        + [
            (
                f"bumps_{i}",
                dict(**_POSE, field="bumps", bump_sigma=sg, bump_count=k, seed=110 + i),
            )
            for i, (sg, k) in enumerate([(25.0, 60), (100.0, 20), (400.0, 4)])
        ]
        + [
            (f"seams_{i}", dict(**_POSE, field="seams", seam_sd=sd, seed=120 + i))
            for i, sd in enumerate([1.5, 3.0, 6.0])
        ]
    ),
    # Development cases: what method options may be chosen on. Their
    # seeds are used nowhere else and the scorer leaves `dev_*` cases out of every table.
    "dev": _suite(
        [
            ("dev_wave", dict(**_POSE, amp=10.0, seed=900)),
            ("dev_multiscale", dict(**_POSE, field="multiscale", seed=901)),
            ("dev_grid", dict(**_POSE, field="grid", seed=902)),
            ("dev_bumps", dict(**_POSE, field="bumps", seed=903)),
            ("dev_seams", dict(**_POSE, field="seams", seed=904)),
            ("dev_dropout", dict(**_POSE, amp=10.0, dropout=0.3, seed=905)),
        ]
    ),
}
TRUTH_SUITE = "truth"
TRUTH_WINDOW = "blob"  # the synthetic scene's name in a patient id


def truth_suite(spec=None, n=None, families=None):
    """``{case_id: generator spec}`` of the frozen known-truth cases: one patient per
    family (its own blob scene), one round per pose.

    ``n`` and ``families`` make a reduced copy for a test; a reduced copy is not the frozen
    suite and its cases carry no spec hash.
    """
    spec = spec or truth.load_spec()
    names = truth.family_names(spec)
    frozen = n is None and families is None
    out = {}
    for family in names:
        if families is not None and family not in families:
            continue
        # the family's own index, not its position: see truth.family_index
        j = truth.family_index(spec, family)
        scene_seed = spec["synthetic"]["scene_seed"] + 100 * j
        patient = f"{TRUTH_WINDOW}_{family}"
        for k, pose in enumerate(truth.pose_names(spec)):
            out[f"{patient}_{pose}"] = {
                **_BASE,
                "n": int(n or spec["n_px"]),
                "noise": spec["synthetic"]["noise"],
                "family": family,
                "pose": pose,
                "patient": patient,
                "scene_seed": scene_seed,
                "seed": scene_seed + 1 + k,
                **truth.pose_kwargs(spec, pose),
                **truth.family_kwargs(spec, family, spec["pixel_size_um"], n),
                **truth.case_extra(spec, family, frozen),
            }
    return out


def suite_specs(name):
    """The cases of a suite by name; ``truth`` is read from the frozen spec."""
    return truth_suite() if name == TRUTH_SUITE else SUITES[name]


def field_stats(spec, pixel_size_um, step=8, threshold_um=1.0):
    """What a case's non-rigid field does to its window, on a lattice every ``step`` px:

    ``moved_frac_over_1um``  share of the window the field ``w`` moves by more than
                             ``threshold_um`` (the pose is not counted)
    ``min_jacobian``         smallest determinant of the Jacobian of the whole map ``g``,
                             by central differences; a value <= 0 is a fold
    ``fold_pct``             share of the lattice where it is <= 0
    """
    n = spec["n"]
    at = np.arange(step / 2.0, n, step)
    xx, yy = np.meshgrid(at, at)
    q = forward_map(spec)(np.stack([xx, yy], axis=-1))
    rigid = forward_map({**spec, "amp": 0.0, "field": "wave"})(
        np.stack([xx, yy], axis=-1)
    )
    moved = np.hypot(*np.moveaxis(q - rigid, -1, 0)) * float(pixel_size_um)
    det = np.gradient(q[..., 0], step, axis=1) * np.gradient(
        q[..., 1], step, axis=0
    ) - np.gradient(q[..., 0], step, axis=0) * np.gradient(q[..., 1], step, axis=1)
    return {
        "moved_frac_over_1um": float(np.mean(moved > threshold_um)),
        "min_jacobian": float(det.min()),
        "fold_pct": float(100.0 * np.mean(det <= 0)),
    }


def settle(spec, pixel_size_um):
    """The spec a known-truth case is really made with, and the record of how its seed
    was settled: ``(spec, record)``.

    A spec carrying a re-seed rule (``spec["reseed"]``, from ``truth_spec.json`` through
    ``truth.case_extra``) is measured before anything is drawn: the smallest Jacobian
    determinant of its true map on a lattice every ``lattice_px`` px. Above
    ``min_jacobian`` the seed is used. Otherwise the map folds, or comes closer to
    folding than a lattice can tell apart from it, and the seed is skipped for
    ``seed + step``, up to ``max_tries`` seeds. The answer depends on the spec alone, so
    every process that prepares the case settles on the same seed.

    The record: ``seed_asked``, ``seed`` (the one used), ``skipped`` (each skipped seed
    with its determinant, in order), ``step``, ``lattice_px``, ``min_jacobian`` (of the
    map used) and ``min_jacobian_required``. A spec without the rule comes back as it
    is, with None.
    """
    rule = spec.get("reseed")
    if not rule:
        return spec, None
    asked, skipped = int(spec["seed"]), []
    for attempt in range(int(rule["max_tries"])):
        seed = asked + attempt * int(rule["step"])
        trial = {**spec, "seed": seed}
        low = field_stats(trial, pixel_size_um, step=rule["lattice_px"])["min_jacobian"]
        if low > rule["min_jacobian"]:
            return trial, {
                "seed_asked": asked,
                "seed": seed,
                "skipped": skipped,
                "step": int(rule["step"]),
                "lattice_px": rule["lattice_px"],
                "min_jacobian": low,
                "min_jacobian_required": rule["min_jacobian"],
            }
        skipped.append({"seed": seed, "min_jacobian": low})
    raise SystemExit(
        f"{spec.get('family')} case with seed {asked}: the true map folds (smallest "
        f"Jacobian determinant not above {rule['min_jacobian']} on a "
        f"{rule['lattice_px']} px lattice) for all {rule['max_tries']} seeds tried: "
        + ", ".join(f"{s['seed']} ({s['min_jacobian']:.3f})" for s in skipped)
        + ". That is not chance: the family's amplitude or count makes it fold."
    )


def known_truth_field(spec, pixel_size_um):
    """``{"field": field_stats}`` for a known-truth case (one with a pose), else nothing:
    each such case says in its own ``case.json`` how much of it moves and that its true
    map does not fold, measured on the map it was really made with."""
    if not spec.get("pose"):
        return {}
    return {"field": {"lattice_px": 8, **field_stats(spec, pixel_size_um, step=8)}}


def shared_reference(cases_root, dataset, patient):
    """Where the one reference slide of a patient with several rounds is kept."""
    return Path(cases_root) / "_cache" / dataset / str(patient) / "ref.ome.tif"


def write_once(path, write):
    """``write(tmp_path)`` then rename, unless ``path`` exists: two array tasks preparing
    two rounds of one patient both find a complete file."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"tmp{os.getpid()}_{path.name}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# What ``case.json: extra.cell_sets`` says of a known-truth case, either side. The
# expansion distance is the pipeline's (``cells.EXPAND_PX``); it is recorded here, per
# case, and NOT in ``truth_spec.json``: the spec freezes the maps, the probes and the
# seeds, none of which the outlines enter (the probes are the nuclei's centroids), so the
# second set changes no frozen quantity and the spec's hash stands.
_BUILTIN_SIDE = {
    "outlines": {"source": "builtin", "expand_px": cells.EXPAND_PX},
    "nuclei": {"source": "builtin", "exact": True},
}
BUILTIN_SETS = {"ref": _BUILTIN_SIDE, "mov": _BUILTIN_SIDE}


def two_sets(mov_nuclei, mov_outlines, ref_nuclei, ref_outlines, g):
    """The ``write_case`` arguments of both polygon sets of a known-truth case.

    The nuclei are the built-in segmenter's objects; outline ``k`` is nucleus ``k`` grown
    (``cells.segment_slide_sets``), which ``nuclei_<side>_outline`` records. ``g`` is the
    true map: where a perfect registration puts each moving set, which gives the cell
    metric's ceiling for this pair (noise, resampling and dropout, no registration error).
    """
    return dict(
        cells_mov=mov_outlines,
        cells_ref=ref_outlines,
        nuclei_mov=mov_nuclei,
        nuclei_ref=ref_nuclei,
        nuclei_mov_outline=np.arange(len(mov_nuclei), dtype=np.int64),
        nuclei_ref_outline=np.arange(len(ref_nuclei), dtype=np.int64),
        cells_mov_xy_truth=g(mov_outlines.xy).astype(np.float32),
        nuclei_mov_xy_truth=g(mov_nuclei.xy).astype(np.float32),
    )


def prepare_one(cases_root, name, spec, workers=1, force=False):
    d = case_dir(cases_root, "synthetic", name)
    if (d / "case.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    # a known-truth case whose true map folds is made with the next seed that does not
    spec, reseed = settle(spec, PIXEL_SIZE_UM)
    patient = spec.get("patient")
    ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
    if patient:
        ref_p = write_once(
            shared_reference(cases_root, "synthetic", patient),
            lambda tmp: _write_slide(spec, "ref", tmp, workers),
        )
    else:
        _write_slide(spec, "ref", ref_p, workers)
    scene = _write_slide(spec, "mov", mov_p, workers)
    # `mov_cells` / `ref_cells` are the segmented objects, the BARE nuclei; the outlines
    # are the same labels grown as the pipeline grows its cell mask (two_sets)
    seg = dict(diameter=NUCLEUS_DIAMETER, workers=workers)
    (mov_cells, mov_outlines), (ref_cells, ref_outlines) = (
        cells.segment_slide_sets(mov_p, **seg),
        cells.segment_slide_sets(ref_p, **seg),
    )
    cap = spec.get("max_cells")
    if cap and len(mov_cells) > cap:
        # only the moving side is thinned: each sampled cell still meets ALL reference cells
        keep = np.sort(
            np.random.default_rng(0).choice(len(mov_cells), cap, replace=False)
        )
        mov_cells, mov_outlines = mov_cells.take(keep), mov_outlines.take(keep)
    hw = [spec["n"], spec["n"]]
    case = Case(
        dataset="synthetic",
        case_id=name,
        group=patient or name,
        family=spec["family"],
        pose=spec.get("pose", ""),
        ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p),
        ref_image=str(ref_p),
        mov_image=str(mov_p),
        modality="fluorescence",
        diagonal=math.hypot(spec["n"], spec["n"]),
        pixel_size_um=PIXEL_SIZE_UM,
        ref_hw=hw,
        mov_hw=list(scene.mov_hw),
        extra={
            "spec": {k: v for k, v in spec.items()},
            "in_stare_model": spec.get("in_stare_model", ""),
            "truth_spec_sha256": spec.get("truth_spec_sha256", ""),
            **known_truth_field(spec, PIXEL_SIZE_UM),
            **({"reseed": reseed} if reseed else {}),
            "cell_sets": BUILTIN_SETS,
        },
    )
    g = forward_map(spec)
    # Paired probes for the known-truth error: moving nuclei and where the map puts them.
    # They are the centroids of the NUCLEI, as they were before the outlines were grown.
    _, centroids = cells.area_centroid(mov_cells)
    probes = truth.probe_arrays(centroids, g, hw, spec["seed"])
    return write_case(
        cases_root,
        case,
        probes["truth_mov"],
        probes["truth_ref"],
        truth_cell=probes["truth_cell"],
        **two_sets(mov_cells, mov_outlines, ref_cells, ref_outlines, g),
    )


def refuse_other_spec(cases_root, nothing="Nothing was prepared."):
    """Stop when a synthetic case that is already prepared carries the hash of a
    ``truth_spec.json`` other than the current one. Such a case is neither kept (its
    maps, probes and seeds are another spec's) nor rewritten in place (the results
    registered on it would then rest on images that are gone). ``nothing`` is the
    message's last sentence: what the caller had not done yet (``prepare``, and the
    launcher before its first job)."""
    current = truth.load_spec()[truth.HASH_KEY]
    for path in sorted((Path(cases_root) / "synthetic").glob("*/case.json")):
        try:
            extra = json.loads(path.read_text()).get("extra") or {}
        except (OSError, ValueError):
            continue
        have = str(extra.get("truth_spec_sha256") or "")
        if have and have != current:
            raise SystemExit(
                f"{path.parent}: this case was prepared from truth_spec.json "
                f"{have[:12]}, and the current one is {current[:12]}. Cases of two "
                "specs are never in one folder, and an existing case is not replaced. "
                "Prepare into a new, empty cases folder: export REGBENCH_CASES=<a "
                "folder that does not exist yet> (or --cases) and prepare again. "
                f"{nothing}"
            )


def prepare(a):
    refuse_other_spec(a.cases)
    suite = suite_specs(a.suite)
    names = sorted(suite)
    if a.index is not None:
        names = names[a.index : a.index + 1]
    for name in names:
        d = prepare_one(a.cases, name, suite[name], workers=a.workers, force=a.force)
        print(f"synthetic/{name} -> {d}")
    return 0
