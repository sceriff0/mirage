"""Two polygon sets per slide: the outlines and the bare nuclei (ruling R3.18).

The outlines are what the cohort's arms publish (a nucleus grown by 10 px); the nuclei are
the objects themselves. A case holds both, the driver moves both, the scorer reports every
overlap number on both, and a case or a result made before the second set existed still
loads and scores on the outlines, with the reason the nuclei are absent.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import cases as cases_mod
from benchmarks.regbench import cells, methods, pairs, truth
from benchmarks.regbench.cases import (
    Case,
    load_case,
    load_points,
    polys,
    result_dir,
    write_case,
)
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.score import score

from .test_pipeline import GRID, SHIFT, fake, good, squares

REPO = Path(__file__).resolve().parents[3]
OUTLINE_HALF, NUCLEUS_HALF = 6.0, 3.0


def two_set_case(root, case_id="P1_r1", group="P1", nuclei=True, parents=False, **kw):
    """Outlines 12 px a side and, inside each, a nucleus 6 px a side."""
    case = Case(
        dataset="multiplex",
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
        extra=kw.pop("extra", {}),
    )
    more = {}
    if nuclei:
        more = dict(
            nuclei_mov=squares(GRID - SHIFT, NUCLEUS_HALF),
            nuclei_ref=squares(GRID, NUCLEUS_HALF),
        )
        if parents:
            more["nuclei_mov_outline"] = np.arange(len(GRID))
            more["nuclei_ref_outline"] = np.arange(len(GRID))
    write_case(
        root,
        case,
        cells_mov=squares(GRID - SHIFT, OUTLINE_HALF),
        cells_ref=squares(GRID, OUTLINE_HALF),
        **more,
        **kw,
    )
    return load_case(root / "multiplex" / case_id)


def off_by(name, by, direction=MOV_TO_REF):
    sign = 1.0 if direction == MOV_TO_REF else -1.0

    def register(case, work, opts):
        yield (
            name,
            PointMap(
                lambda xy: np.asarray(xy, dtype=float) + sign * (SHIFT + by), direction
            ),
            {},
        )

    return fake(name, [name], register)


def row(rows, method, case_id="P1_r1"):
    (r,) = [r for r in rows if r["method"] == method and r["case_id"] == case_id]
    return r


def table(out, name):
    with open(out / "tables" / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


# ── the contract ──────────────────────────────────────────────────────────────
def test_a_case_holds_the_outlines_and_the_nuclei_of_each_side(tmp_path):
    case = two_set_case(tmp_path)
    points = load_points(case)
    for side in ("ref", "mov"):
        outl, nuc = polys(points, side), polys(points, side, "nuclei")
        assert outl.n_features == nuc.n_features == len(GRID)
        a_out, c_out = pairs.cp.feature_area_centroid(outl)
        a_nuc, c_nuc = pairs.cp.feature_area_centroid(nuc)
        assert np.allclose(a_out, (2 * OUTLINE_HALF) ** 2)
        assert np.allclose(a_nuc, (2 * NUCLEUS_HALF) ** 2)
        assert np.allclose(c_out, c_nuc)
    assert cases_mod.nuclei_reason(case, points) == ""
    with pytest.raises(ValueError, match="set must be one of"):
        polys(points, "ref", "membranes")


def test_a_case_written_before_the_second_set_still_loads_and_scores(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root, nuclei=False)
    points = load_points(case)
    assert polys(points, "mov", "nuclei") is None
    assert cases_mod.nuclei_reason(case, points).startswith(
        "ref: " + cases_mod.NO_NUCLEI_OLD_CASE
    )
    methods.run_case(good(), case, out)
    assert "nuclei_xy" not in np.load(result_dir(out, "good", case) / "warped.npz")
    res = score(root, out, ["multiplex"])
    assert row(res["cells"], "good")["pair_fraction"] == 1.0  # the outlines, as before
    assert res["cells_nuclei"] == []
    (missing,) = res["nuclei_missing"]
    assert missing["case_id"] == "P1_r1" and missing["method"] == ""
    assert cases_mod.NO_NUCLEI_OLD_CASE in missing["reason"]
    assert table(out, "nuclei_missing")[0]["reason"] == missing["reason"]
    assert not (out / "tables" / "cells_nuclei.csv").exists()
    assert "no bare nuclei" in (out / "tables" / "summary.md").read_text()


def test_the_reason_a_side_has_no_nuclei_is_the_prepared_one(tmp_path):
    why = "no nuclear label image and no StarDist re-run for slide P1.r1"
    sets = {"mov": {"nuclei": {"source": "", "reason": why}}}
    root = tmp_path / "cases"
    case = two_set_case(root, nuclei=False, extra={"cell_sets": sets})
    assert cases_mod.set_info(case, "mov", "nuclei")["reason"] == why
    reason = cases_mod.nuclei_reason(case, load_points(case))
    assert f"mov: {why}" in reason and "ref: no nucleus in the set" in reason


# ── the driver ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("direction", [MOV_TO_REF, REF_TO_MOV])
def test_the_driver_moves_both_sets_through_the_map(tmp_path, direction):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)
    methods.run_case(off_by("m", np.zeros(2), direction), case, out)
    warped = np.load(result_dir(out, "m", case) / "warped.npz")
    points = load_points(case)
    side, sign = ("mov", 1.0) if direction == MOV_TO_REF else ("ref", -1.0)
    for key, prefix in (("cells_xy", "cells"), ("nuclei_xy", "nuclei")):
        assert np.allclose(warped[key], points[f"{prefix}_{side}_xy"] + sign * SHIFT)


def _seamed(name, info):
    def register(case, work, opts):
        yield name, PointMap(lambda xy: np.asarray(xy, float) + SHIFT, MOV_TO_REF), info

    return fake(name, [name], register)


def _band(n=20):
    band = np.zeros(len(GRID), dtype=bool)
    band[:n] = True
    return band


def test_a_nucleus_inherits_its_outlines_seam_flag_when_the_sets_are_aligned(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root, parents=True)
    # the method's own nuclei mask is NOT used when the outline of each nucleus is known
    other = ~_band()
    methods.run_case(
        _seamed("tiles", {"band_mov": _band(), "band_mov_nuc": other}), case, out
    )
    warped = np.load(result_dir(out, "tiles", case) / "warped.npz")
    assert (warped["band_nuc"] == _band()).all()
    run = methods.json.loads((result_dir(out, "tiles", case) / "run.json").read_text())
    assert run["nuclei_band"] == {"mov": "inherited"}
    res = score(root, out, ["multiplex"], seam_owners=("tiles",))
    non = row(res["nonband_cells_nuclei"], "tiles:non_tiles")
    assert (non["n_mov_kept"], non["n_mov_all"]) == (len(GRID) - 20, len(GRID))
    assert non["set"] == "nuclei"


def test_without_alignment_the_nuclei_mask_is_the_methods_own_geometry(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)
    mine = _band(7)
    methods.run_case(
        _seamed("tiles", {"band_mov": _band(), "band_mov_nuc": mine}), case, out
    )
    warped = np.load(result_dir(out, "tiles", case) / "warped.npz")
    assert (warped["band_nuc"] == mine).all()
    run = methods.json.loads((result_dir(out, "tiles", case) / "run.json").read_text())
    assert run["nuclei_band"] == {"mov": "geometry"}


def test_no_alignment_and_no_geometry_gives_no_off_seam_nuclei_row(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)
    methods.run_case(_seamed("tiles", {"band_mov": _band()}), case, out)
    assert "band_nuc" not in np.load(result_dir(out, "tiles", case) / "warped.npz")
    res = score(root, out, ["multiplex"], seam_owners=("tiles",))
    assert res["nonband_cells"] and res["nonband_cells_nuclei"] == []
    assert row(res["cells_nuclei"], "tiles")["pair_fraction"] == 1.0
    # ... and the reason is on record (review item 8): the missing rows are named,
    # with the owner, and the summary says so
    (why,) = [r for r in res["nuclei_missing"] if r.get("owner")]
    assert why["method"] == "*:non_tiles" and why["case_id"] == case.case_id
    assert "left no mask on this case's nuclei" in why["reason"]
    text = (out / "tables" / "summary.md").read_text()
    assert "have no off-seam rows on the bare nuclei" in text and "`tiles`" in text
    assert "hold no bare nuclei" not in text  # not mistaken for a result without nuclei


# ── the scorer ────────────────────────────────────────────────────────────────
def test_the_nuclei_have_their_own_gate_and_a_stricter_dice(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)
    methods.run_case(off_by("near", np.array([2.0, 0.0])), case, out)
    res = score(root, out, ["multiplex"])
    outl, nuc = row(res["cells"], "near"), row(res["cells_nuclei"], "near")
    assert outl["set"] == "outlines" and nuc["set"] == "nuclei"
    for r, half in ((outl, OUTLINE_HALF), (nuc, NUCLEUS_HALF)):
        radius = math.sqrt((2 * half) ** 2 / math.pi)
        assert r["median_cell_radius_px"] == pytest.approx(radius)
        assert r["match_radius_px"] == pytest.approx(pairs.GATE_FACTOR * radius)
    # the same 2 px error: 10/12 of an outline overlaps, 4/6 of a nucleus (the
    # pipeline's IoU is rasterised, hence the tolerance)
    assert outl["dice_matched"] == pytest.approx(10 / 12, abs=0.03)
    assert nuc["dice_matched"] == pytest.approx(4 / 6, abs=0.03)
    assert outl["displacement_px_p50"] == nuc["displacement_px_p50"] == 2.0
    for key in pairs.RECORD_KEYS:  # every overlap quantity, on both sets
        assert key in nuc and key in outl
    assert row(res["cells_nuclei"], "initial")["pair_fraction"] < 0.5
    files = {p.name for p in (out / "tables").iterdir()}
    assert {
        "cells_nuclei.csv",
        "summary_cells_nuclei.csv",
        "common_block_nuclei.csv",
    } <= files
    assert {r["set"] for r in table(out, "summary_cells_nuclei")} == {"nuclei"}
    assert {r["set"] for r in table(out, "cells")} == {"outlines"}


def test_a_failed_run_is_scored_at_the_unregistered_pose_on_the_nuclei_too(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)

    def register(case, work, opts):
        raise RuntimeError("boom")
        yield

    methods.run_case(fake("bad", ["bad"], register), case, out)
    res = score(root, out, ["multiplex"])
    nuc = row(res["cells_nuclei"], "bad")
    assert nuc["imputed_initial"] and nuc["cause"] == "exception"
    assert nuc["pair_fraction"] == row(res["cells_nuclei"], "initial")["pair_fraction"]


def test_a_result_made_before_the_case_had_nuclei_is_listed_not_imputed(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    old = two_set_case(root, nuclei=False)
    methods.run_case(good("early"), old, out)
    case = two_set_case(root)  # the case is prepared again, now with nuclei
    methods.run_case(good("late"), case, out)
    res = score(root, out, ["multiplex"])
    assert {r["method"] for r in res["cells_nuclei"]} == {"initial", "late"}
    (missing,) = res["nuclei_missing"]
    assert missing["method"] == "early" and "Run it again" in missing["reason"]
    assert row(res["cells"], "early")["pair_fraction"] == 1.0


def test_paired_differences_and_shared_numbers_exist_on_the_nuclei(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    for i in range(2):
        case = two_set_case(root, f"P{i}_r1", f"P{i}")
        methods.run_case(off_by("stare", np.zeros(2)), case, out)
        methods.run_case(off_by("other", np.array([2.0, 0.0])), case, out)
    res = score(root, out, ["multiplex"], primary=["stare", "other"])
    (s,) = [
        r
        for r in res["summary_cells_nuclei"]
        if r["method"] == "other" and r["subset"] == "all"
    ]
    assert s["dice_matched_shared_med"] == pytest.approx(4 / 6, abs=0.03)
    assert s["n_dice_shared"] == 2 and s["set"] == "nuclei"
    (d,) = [r for r in res["paired_summary_nuclei"] if r["method"] == "other"]
    assert d["d_dice_matched_med"] == pytest.approx(4 / 6 - 1.0, abs=0.03)
    files = {p.name for p in (out / "tables").iterdir()}
    assert {"paired_vs_primary_nuclei.csv", "paired_summary_nuclei.csv"} <= files
    (o,) = [r for r in res["paired_summary"] if r["method"] == "other"]
    assert o["d_dice_matched_med"] == pytest.approx(10 / 12 - 1.0, abs=0.03)


def test_the_summary_prints_the_nuclei_beside_the_outlines(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = two_set_case(root)
    methods.run_case(off_by("stare", np.array([2.0, 0.0])), case, out)
    res = score(root, out, ["multiplex"], primary=["stare"])
    text = (out / "tables" / "summary.md").read_text()
    head = "#### Outlines and bare nuclei, side by side"
    assert text.count(head) == 2  # all cases, and the common block
    block = text.split(head)[1].split("###")[0]
    (line,) = [ln for ln in block.splitlines() if ln.startswith("| stare ")]
    outl = row(res["cells"], "stare")["dice_matched"]
    nuc = row(res["cells_nuclei"], "stare")["dice_matched"]
    assert outl > nuc
    assert f"{outl:.4g} / {nuc:.4g}" in line  # matched Dice: outlines / nuclei
    assert "dice_matched: outlines / nuclei" in block


# ── known-truth datasets: both sets from one segmentation ────────────────────
def test_a_toy_case_has_nuclei_and_outlines_grown_as_the_pipeline_does(
    toy_cases,
):
    case = next(iter(toy_cases.values()))
    points = load_points(case)
    for side in ("ref", "mov"):
        nuc, outl = polys(points, side, "nuclei"), polys(points, side)
        assert nuc.n_features == outl.n_features > 50
        assert (points[f"nuclei_{side}_outline"] == np.arange(nuc.n_features)).all()
        a_nuc, c_nuc = pairs.cp.feature_area_centroid(nuc)
        a_out, _ = pairs.cp.feature_area_centroid(outl)
        assert (a_out > a_nuc).all()
        # an isolated nucleus gains a 10 px rim; a crowded one less, never more
        r_nuc, r_out = np.sqrt(a_nuc / np.pi), np.sqrt(a_out / np.pi)
        assert (r_out - r_nuc).max() <= cells.EXPAND_PX + 1.0
        assert np.median(r_out - r_nuc) > 0.5 * cells.EXPAND_PX
        info = cases_mod.set_info(case, side, "nuclei")
        assert info == {"source": "builtin", "exact": True}
        assert cases_mod.set_info(case, side, "outlines") == {
            "source": "builtin",
            "expand_px": cells.EXPAND_PX,
        }
    assert len(points["nuclei_mov_xy_truth"]) == len(points["nuclei_mov_xy"])


def test_growing_the_outlines_does_not_touch_the_nuclei_or_the_spec(real_like):
    alone = cells.segment_slide(real_like, 0, 12.0, tile=512)
    nuclei, outlines = cells.segment_slide_sets(real_like, 0, 12.0, tile=512)
    assert np.array_equal(alone.xy, nuclei.xy) and np.array_equal(alone.off, nuclei.off)
    assert len(outlines) == len(nuclei)
    # the expansion distance is not a frozen parameter: the spec and its hash stand
    spec = truth.load_spec()
    assert spec["sha256"] == (
        "1bade785322e5472314c8d2f05934c41a93e07b0b86521b5cdceff7def16dcbf"
    )
    assert "expand" not in truth.SPEC_PATH.read_text()


def test_grown_labels_never_overlap_and_match_one_whole_image_call():
    from skimage.segmentation import expand_labels

    rng = np.random.default_rng(3)
    labels = np.zeros((120, 160), dtype=np.int32)
    for k, (y, x) in enumerate(rng.integers(8, 110, size=(25, 2)), start=1):
        labels[y : y + 5, x : x + 5] = k
    grown = cells.grow(labels)
    assert np.array_equal(grown, expand_labels(labels, distance=10))
    assert (grown[labels > 0] == labels[labels > 0]).all()


def test_the_expansion_distance_is_the_pipelines_default():
    """A guard: ``cells.EXPAND_PX`` is ``seg_expand_distance`` of nextflow.config, and
    the growth is the skimage call ``bin/utils/segment_io.py`` makes."""
    config = (REPO / "nextflow.config").read_text()
    (value,) = re.findall(r"^\s*seg_expand_distance\s*=\s*(\d+)", config, re.M)
    assert int(value) == cells.EXPAND_PX
    pipeline = (REPO / "bin" / "utils" / "segment_io.py").read_text()
    assert "segmentation.expand_labels(label_image, distance=distance)" in pipeline
    source = (REPO / "benchmarks" / "regbench" / "cells.py").read_text()
    assert "expand_labels(np.asarray(labels), distance=int(distance))" in source
