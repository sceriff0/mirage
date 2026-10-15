"""The test tree's toy cases: four small generated pairs, written as cases of the one
dataset regbench scores (``multiplex``). Test-fixture code, not a dataset.

Each toy case is drawn with the selfcheck's scene generator (``regbench.scene``), segmented
on both sides with the built-in segmenter (nuclei and outlines grown as the pipeline grows
them, ``cells.segment_slide_sets``), and written as its own patient (``group = name``).
Their known map stays in ``case.json: extra.spec``: :func:`true_error_px` measures a
method's error against it. No known-truth probes are written: those belonged to the
synthetic dataset.

``python -m benchmarks.tests.regbench.toy_cases DIR`` prepares ``SPECS`` into ``DIR``, for
``REGBENCH_TEST_CASES`` (``conftest.py``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from benchmarks.regbench import cells, scene
from benchmarks.regbench.cases import Case, case_dir, write_case

DATASET = "multiplex"


def _spec(name, **kw):
    """A toy spec as the dataset's suites made it: the generator's base, the family
    taken from the name (``smooth_border`` -> ``smooth``), then the case's own values."""
    return {
        **scene._BASE,
        "family": name.rsplit("_", 1)[0].rstrip("0123456789_"),
        **kw,
    }


# Seconds per case, one with a smaller moving slide, and two that pin the conditions under
# which STARE 1.2.1's SOLVE picks a flat field.
SPECS = {
    "rigid": _spec("rigid", n=1024, theta=2.0, shift=(14.0, -9.0), seed=1),
    "smooth": _spec("smooth", n=1024, amp=5.0, wavelength=600.0, seed=4),
    "crop": _spec(
        "crop",
        n=1024,
        theta=1.0,
        shift=(10.0, 6.0),
        amp=4.0,
        wavelength=600.0,
        mov_hw=(832, 928),
        seed=4,
    ),
    "smooth_border": _spec("smooth_border", n=1024, amp=5.0, wavelength=600.0, seed=2),
}

# What ``case.json: extra.cell_sets`` says of a toy case, either side: the built-in
# segmenter's nuclei, and its outlines grown by the pipeline's distance.
_BUILTIN_SIDE = {
    "outlines": {"source": "builtin", "expand_px": cells.EXPAND_PX},
    "nuclei": {"source": "builtin", "exact": True},
}
BUILTIN_SETS = {"ref": _BUILTIN_SIDE, "mov": _BUILTIN_SIDE}

# A point whose true position lies less than this far inside the reference is not
# measured (the rule the known-truth probes of the removed synthetic dataset had).
MARGIN_PX = 32.0


def two_sets(mov_nuclei, mov_outlines, ref_nuclei, ref_outlines, g):
    """The ``write_case`` arguments of both polygon sets of a toy case.

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


def prepare_toy(root, name, spec, workers=1):
    """Write the toy case ``name`` under ``root/multiplex/name``; return its folder. A case
    already there (its ``case.json`` is written last) is kept."""
    d = case_dir(root, DATASET, name)
    if (d / "case.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
    sc = scene.write_pair(spec, ref_p, mov_p, workers=workers)
    # the BARE nuclei, and the same labels grown as the pipeline grows its cell mask
    seg = dict(diameter=scene.NUCLEUS_DIAMETER, workers=workers)
    (mov_cells, mov_outlines), (ref_cells, ref_outlines) = (
        cells.segment_slide_sets(mov_p, **seg),
        cells.segment_slide_sets(ref_p, **seg),
    )
    case = Case(
        dataset=DATASET,
        case_id=name,
        group=name,
        family=spec["family"],
        ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p),
        ref_image=str(ref_p),
        mov_image=str(mov_p),
        modality="fluorescence",
        diagonal=math.hypot(spec["n"], spec["n"]),
        pixel_size_um=scene.PIXEL_SIZE_UM,
        ref_hw=[spec["n"], spec["n"]],
        mov_hw=list(sc.mov_hw),
        extra={"spec": dict(spec), "cell_sets": BUILTIN_SETS},
    )
    g = scene.forward_map(spec)
    return write_case(
        root,
        case,
        **two_sets(mov_cells, mov_outlines, ref_cells, ref_outlines, g),
    )


def true_error_px(case, mapped_xy, from_xy):
    """Per point, the distance in px between where a method put a moving point
    (``mapped_xy``) and where the case's true map puts it (``scene.forward_map`` of
    ``case.extra["spec"]`` at ``from_xy``, the same points before the method moved them).

    A point whose true position lies less than ``MARGIN_PX`` inside the reference is NaN:
    nothing on the reference anchors it. Take ``np.nanmedian`` of the result."""
    mapped = np.asarray(mapped_xy, dtype=float).reshape(-1, 2)
    src = np.asarray(from_xy, dtype=float).reshape(-1, 2)
    if len(mapped) != len(src):
        raise ValueError(f"{len(mapped)} mapped points for {len(src)} moving points")
    target = np.asarray(scene.forward_map(case.extra["spec"])(src), dtype=float)
    h, w = case.ref_hw
    inside = (
        (target >= MARGIN_PX).all(axis=1)
        & (target[:, 0] <= w - 1 - MARGIN_PX)
        & (target[:, 1] <= h - 1 - MARGIN_PX)
    )
    err = np.hypot(*(mapped - target).T)
    return np.where(inside, err, np.nan)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "usage: python -m benchmarks.tests.regbench.toy_cases DIR", file=sys.stderr
        )
        return 2
    root = Path(argv[0])
    for name, spec in SPECS.items():
        print(f"{DATASET}/{name} -> {prepare_toy(root, name, spec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
