"""A small cohort with every kind of row the scorer makes, for the tests of the per-case
score records (``test_score_records.py``).

Four ``multiplex`` rounds of three patients (one prepared before the bare nuclei
existed). Results: a map in each direction, two seam owners (a mask on moving nuclei, a
mask on reference nuclei), a method that raises, a method that never ran, a dropped row, and a
case one method was not run on.

Nothing here reads the clock: the same call builds the same cases. The results carry
their run's timings, so two BUILDS differ in ``resources.csv``; every comparison in the
tests is between scorings of ONE build.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from benchmarks.regbench import cells, methods
from benchmarks.regbench.cases import Case, load_case, write_case
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap

GRID = np.array(
    [(x, y) for x in range(40, 400, 40) for y in range(40, 400, 40)], dtype=float
)
SHIFT = np.array([12.0, -9.0])
OUTLINE_HALF, NUCLEUS_HALF = 6.0, 3.0
OWNERS = ("tiles", "blocks")
SCORE_KW = dict(
    expected=["ghost"],
    primary=["stare", "rtm"],
    seam_owners=OWNERS,
    dropped=["dropme"],
    row_checks={"stare": "ok", "rtm": "no_stage_image", "tiles": "ok"},
)


def squares(centres, half):
    return cells.from_rings(
        [
            np.array(
                [
                    [x - half, y - half],
                    [x + half, y - half],
                    [x + half, y + half],
                    [x - half, y + half],
                ]
            )
            for x, y in centres
        ]
    )


def _case(dataset, case_id, group, **kw):
    return Case(
        dataset=dataset,
        case_id=case_id,
        group=group,
        ref_nuclear="",
        mov_nuclear="",
        ref_image="",
        mov_image="",
        modality="fluorescence",
        diagonal=622.0,
        pixel_size_um=0.5,
        ref_hw=[440, 440],
        mov_hw=[440, 440],
        **kw,
    )


def _centres(seed):
    """The reference centres of one case, and its moving ones: displaced by SHIFT and by
    a little of their own, so that no two cases score alike."""
    rng = np.random.default_rng(seed)
    ref = GRID + rng.uniform(-3.0, 3.0, GRID.shape)
    mov = ref - SHIFT + rng.normal(0.0, 0.6, GRID.shape)
    return ref, mov


def multiplex_case(root, case_id, group, seed, nuclei=True, parents=False):
    ref, mov = _centres(seed)
    more = {}
    if nuclei:
        more = dict(
            nuclei_mov=squares(mov, NUCLEUS_HALF), nuclei_ref=squares(ref, NUCLEUS_HALF)
        )
        if parents:
            more["nuclei_mov_outline"] = np.arange(len(GRID))
            more["nuclei_ref_outline"] = np.arange(len(GRID))
    write_case(
        root,
        _case("multiplex", case_id, group),
        cells_mov=squares(mov, OUTLINE_HALF),
        cells_ref=squares(ref, OUTLINE_HALF),
        **more,
    )
    return load_case(Path(root) / "multiplex" / case_id)


def fake(name, register):
    return types.SimpleNamespace(
        NAME=name, SCOPE="pair", VARIANTS=(name,), register=register
    )


def _wobble(xy, amplitude):
    xy = np.asarray(xy, dtype=float)
    return amplitude * np.sin(xy[:, ::-1] / 37.0)


def forward(name, amplitude, info=None):
    """A moving -> reference map: the truth plus a smooth error of ``amplitude`` px."""

    def register(case, work, opts):
        def fn(xy):
            xy = np.asarray(xy, dtype=float)
            return xy + SHIFT + _wobble(xy, amplitude)

        yield name, PointMap(fn, MOV_TO_REF), dict(info(case) if info else {})

    return fake(name, register)


def backward(name, amplitude):
    """A reference -> moving map (scored in the moving frame)."""

    def register(case, work, opts):
        def fn(xy):
            xy = np.asarray(xy, dtype=float)
            return xy - SHIFT + _wobble(xy, amplitude)

        yield name, PointMap(fn, REF_TO_MOV), {}

    return fake(name, register)


def crashing(name):
    def register(case, work, opts):
        raise RuntimeError("it fell over")
        yield  # pragma: no cover

    return fake(name, register)


def _band(n, every):
    band = np.zeros(n, dtype=bool)
    band[::every] = True
    return band


def _moving_mask(case):
    n = len(GRID)
    return {"band_mov": _band(n, 4), "band_mov_nuc": _band(n, 5)}


def _reference_mask(case):
    n = len(GRID)
    return {"band_ref": _band(n, 3), "band_ref_nuc": _band(n, 6)}


def build(tmp):
    """``(cases root, results folder, the cases)`` of the cohort, results written."""
    tmp = Path(tmp)
    root, out = tmp / "cases", tmp / "out"
    made = [
        multiplex_case(root, "P1_r1", "P1", 1, parents=True),
        multiplex_case(root, "P1_r2", "P1", 2, parents=True),
        multiplex_case(root, "P2_r1", "P2", 3),
        multiplex_case(root, "P3_r1", "P3", 4, nuclei=False),
    ]
    for case in made:
        methods.run_case(forward("stare", 0.4), case, out)
        if case.case_id != "P2_r1":  # one case this method was never run on
            methods.run_case(backward("rtm", 0.9), case, out)
        if case.case_id != "P3_r1":  # an owner that left no mask on one case
            methods.run_case(forward("tiles", 1.5, _moving_mask), case, out)
        methods.run_case(forward("blocks", 2.5, _reference_mask), case, out)
        methods.run_case(crashing("crash"), case, out)
        methods.run_case(forward("dropme", 0.1), case, out)
    return root, out, made
