"""The driver and the scorer, end to end.

On the toy cases (generated slides with a known map): the truth is self-consistent and STARE is really run
and scored. On hand-made cases and fake methods: a failed, killed or missing run is a row,
a map is scored in its own direction, each variant has its own cost, and an out-of-memory
ends the run as ``oom``.
"""

import csv
import json
import os
import shutil
import time
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import cells, methods, pairs, resources, scene
from benchmarks.regbench.cases import (
    Case,
    load_case,
    load_points,
    polys,
    result_dir,
    write_case,
    write_json,
    write_npz,
)
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.score import DESCRIPTIVE, score

from .toy_cases import true_error_px

pytestmark = pytest.mark.usefixtures("local_stages")  # STARE runs in-process

WORKERS = {"workers": 2}
GRID = np.array(
    [(x, y) for x in range(40, 400, 40) for y in range(40, 400, 40)], dtype=float
)
# larger than the gate (8.5 px): unregistered, nothing pairs on the 40 px grid
SHIFT = np.array([12.0, -9.0])


def squares(centres, half=5.0):
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


def toy_case(root, case_id, group):
    """A case with nuclei and no images: the moving nuclei are the reference's minus SHIFT."""
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
    )
    write_case(root, case, cells_mov=squares(GRID - SHIFT), cells_ref=squares(GRID))
    return load_case(root / "multiplex" / case_id)


def true_map():
    return PointMap(lambda xy: np.asarray(xy, dtype=float) + SHIFT, MOV_TO_REF)


def fake(name, variants, register=None, register_group=None, **extra):
    """A method module, without a module."""
    return types.SimpleNamespace(
        NAME=name,
        SCOPE="group" if register_group else "pair",
        VARIANTS=tuple(variants),
        register=register,
        register_group=register_group,
        **extra,
    )


def good(name="good"):
    def register(case, work, opts):
        yield name, true_map(), {}

    return fake(name, [name], register)


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def rows_of(out, table):
    with open(out / "tables" / f"{table}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _row(rows, case_id, method):
    (r,) = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
    return r


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "cases"
    return root, toy_case(root, "P1_r1", "P1"), tmp_path / "out"


# ── the toy cases, with STARE really run ─────────────────────────────────────
def test_the_true_map_puts_moving_nuclei_on_reference_nuclei(toy_cases):
    """Guards the truth's DIRECTION: nuclei segmented on the moving slide, pushed through the
    generator's forward map, land on the nuclei segmented on the reference."""
    for name, case in toy_cases.items():
        pts = load_points(case)
        g = scene.forward_map(case.extra["spec"])
        ref, mov = polys(pts, "ref"), polys(pts, "mov")
        radius, _ = pairs.gate_px(ref, mov)
        before = pairs.score(ref, mov, radius, None)
        after = pairs.score(ref, pairs.with_xy(mov, g(mov.xy)), radius, None)
        assert after["displacement_px_p50"] < 0.5, name
        assert after["dice_matched"] > 0.9, name
        assert after["displacement_px_p50"] < before["displacement_px_p50"], name
        assert np.allclose(
            g(pts["nuclei_mov_xy"]), pts["nuclei_mov_xy_truth"], atol=1e-2
        )
        assert case.pixel_size_um == scene.PIXEL_SIZE_UM
        assert case.group == name and case.family  # the patient is not the family


@pytest.fixture(scope="module")
def scored(cases_root, toy_cases, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    for case in toy_cases.values():
        done = methods.run_case("stare", case, out, opts=WORKERS)
        assert done == ["stare_rigid", "stare"]
    return out, score(cases_root, out, ["multiplex"])


def _tre(out, variant, case):
    """Median distance, px, of the method's warped moving nuclei to where the case's true
    map puts them (``toy_cases.true_error_px``)."""
    pts = load_points(case)
    with np.load(result_dir(out, variant, case) / "warped.npz") as z:
        assert str(z["direction"]) == MOV_TO_REF
        err = true_error_px(case, z["nuclei_xy"], pts["nuclei_mov_xy"])
    return float(np.nanmedian(err))


def _initial_tre(case):
    """The same distance before any registration (the moving nuclei where they are)."""
    mov = load_points(case)["nuclei_mov_xy"]
    return float(np.nanmedian(true_error_px(case, mov, mov)))


def test_stare_registers_rigid_and_smooth_cases_to_subpixel_error(scored, toy_cases):
    out, res = scored
    for name in ("rigid", "smooth"):
        case = toy_cases[name]
        assert _initial_tre(case) > 3.0
        assert _tre(out, "stare", case) < 1.0, name
        assert not _row(res["cells"], name, "stare")["imputed_initial"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STARE 1.2.1: REG-TILE emits full-weight vectors for lattice windows the moving "
        "slide does not cover (here the bottom three rows: residuals 7-35 px against ~3 px "
        "elsewhere). SOLVE selects s by block CV before any outlier rejection; no s "
        "predicts those vectors, a flat field predicts them least badly, so s = 1e6 is "
        "chosen for the whole slide and the mesh adds nothing (~2.5 px, the rigid anchor). "
        "4 of 6 seeds at 1024 px, 2 of 6 at 2048 px. Remove the xfail when uncovered "
        "windows are rejected."
    ),
)
def test_stare_refines_a_moving_slide_that_does_not_cover_the_reference(
    scored, toy_cases
):
    out, _ = scored
    assert _tre(out, "stare", toy_cases["crop"]) < 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STARE 1.2.1, the same selection on a full-size 1024 px pair: held-out 3x3 patches "
        "on the lattice border must be extrapolated, a flexible fit extrapolates worse "
        "than a flat one, and on a 15 x 15 lattice 44 % of the nodes are border, so block "
        "CV can prefer s = 1e6 although the interior is predicted twice as well by a small "
        "s. 2 of 6 seeds at 1024 px, 0 of 14 at 2048 px and above. Remove the xfail when "
        "the selection is fixed."
    ),
)
def test_stare_refines_a_small_slide_whatever_the_seed(scored, toy_cases):
    out, _ = scored
    assert _tre(out, "stare", toy_cases["smooth_border"]) < 1.0


def test_stare_still_anchors_a_moving_slide_of_another_shape(scored, toy_cases):
    out, _ = scored
    case = toy_cases["crop"]
    assert _tre(out, "stare_rigid", case) < 3.0 < _initial_tre(case)


def test_the_mesh_is_what_fixes_a_smooth_deformation(scored, toy_cases):
    out, _ = scored
    case = toy_cases["smooth"]
    assert _tre(out, "stare_rigid", case) > 2.0
    assert _tre(out, "stare", case) < 1.0


def test_resources_are_measured_for_the_whole_run(scored, toy_cases):
    out, res = scored
    r = _row(res["resources"], "smooth", "stare")
    assert r["ok"] and r["register_s"] > 0 and r["cpu_h"] > 0
    assert r["unit"] == "pair" and r["attempt"] == 1 and r["wasted_attempts"] == 0
    assert r["total_to_image_s"] == pytest.approx(r["register_s"] + r["write_s"])
    assert r["cpu_efficiency"] > 0
    if r["rss_method"] != "rusage":
        assert r["peak_rss_gb"] > 0.05  # at least the interpreter and one slide
    # the anchor is yielded first: its own, smaller bill
    rigid = _row(res["resources"], "smooth", "stare_rigid")
    assert rigid["register_s"] <= r["register_s"]
    run = run_json(out, "stare", toy_cases["smooth"])
    for key in (
        "attempt",
        "mem_requested_gb",
        "cpus_requested",
        "cpus_affinity",
        "blas",
        "cpu_model",
        "launch_id",
        "unit",
        "group_cases",
        "native",
        "selfcheck_status",
        "register_s",
        "write_s",
        "cause",
    ):
        assert key in run, key
    for name in (
        "cells",
        "summary_cells",
        "paired_vs_primary",
        "failures",
        "resources",
        "summary_resources",
    ):
        assert rows_of(out, name), name
    text = (out / "tables" / "summary.md").read_text()
    assert "## Resources" in text and "whole job, writers included" in text


def test_the_image_stare_wrote_has_a_tissue_row_beside_initial(scored):
    """The tissue-level block on a REAL method's written image (STARE's own STITCH)."""
    pytest.importorskip("cv2")
    _, tables = scored
    rows = {(r["case_id"], r["variant"]): r for r in tables["tissue"]}
    assert {v for _, v in rows} == {"initial", "stare"}
    assert {(m["variant"], m["reason"]) for m in tables["tissue_missing"]} == {
        ("stare_rigid", "the method writes no image for this variant")
    }
    stare, initial = (rows[("rigid", v)] for v in ("stare", "initial"))
    assert stare["source"] == "method" and initial["source"] == "benchmark"
    for key in ("tissue_dice", "boundary_dice"):
        assert stare[key] > initial[key], key
    assert stare["qcs"] >= initial["qcs"]
    for name in ("cells", "summary_cells", "paired_vs_primary"):
        assert not any("qcs" in k or "tissue" in k for k in tables[name][0]), name


def test_paired_differences_are_against_stare(scored):
    _, res = scored
    paired = res["paired_vs_primary"]
    assert {r["against"] for r in paired} == {"stare"}
    (d,) = [r for r in paired if r["group"] == "rigid" and r["method"] == "initial"]
    assert d["d_pair_fraction_excess"] < 0  # unregistered is below STARE


@pytest.mark.slow
def test_a_failed_method_is_scored_at_the_initial_pose(
    cases_root, toy_cases, tmp_path, monkeypatch
):
    def boom(case, work, opts):
        raise RuntimeError("no transform")
        yield  # pragma: no cover

    monkeypatch.setattr(methods.load("stare"), "register", boom)
    case = toy_cases["rigid"]
    assert methods.run_case("stare", case, tmp_path) == []
    run = run_json(tmp_path, "stare", case)
    assert run["ok"] is False and "no transform" in run["error"]
    assert run["cause"] == "exception"

    res = score(cases_root, tmp_path, ["multiplex"])
    init = _row(res["cells"], "rigid", "initial")
    failed = _row(res["cells"], "rigid", "stare")
    assert failed["imputed_initial"] and failed["cause"] == "exception"
    assert failed["pair_fraction"] == init["pair_fraction"]
    # the three cases it was never run on are rows too
    never = _row(res["cells"], "smooth", "stare")
    assert never["problem"] == "not run" and never["cause"] == "not_run"
    # the toy cases are cohort (multiplex) cases: one summary row per stratum, no family
    agg = [
        a
        for a in res["summary_cells"]
        if a["method"] == "stare" and a["subset"] == "all"
    ]
    assert sum(a["n_cases"] for a in agg) == 4 and not any(a["n_ok"] for a in agg)
    assert len({a["family"] for a in agg}) == len(agg)
    assert not [a for a in res["summary_cells"] if a["subset"] == "common"]
    (f,) = [f for f in res["failures"] if f["method"] == "stare"]
    assert f["launches"] == 1 and f["launches_failed_exception"] == 1
    assert f["rounds_lost"] == 4 and f["rounds_not_run"] == 3


# ── a failed, killed or missing run is a row ─────────────────────────────────
def test_failed_method_is_a_row_not_a_gap(toy):
    root, case, out = toy
    write_json(
        result_dir(out, "ghost", case) / "run.json",
        # settled: the scheduler ended it (Task 28, S1: a kill nobody settled is
        # killed_unsettled, not the method's)
        {"ok": False, "error": "node failure", "cause": "killed", "launch_id": "77",
         "scheduler": {"state": "FAILED"}},
    )  # fmt: skip
    score(root, out, ["multiplex"])
    rows = rows_of(out, "cells")
    init, failed = _row(rows, "P1_r1", "initial"), _row(rows, "P1_r1", "ghost")
    assert failed["imputed_initial"] == "True"
    assert failed["problem"] == "node failure" and failed["cause"] == "killed"
    assert failed["pair_fraction"] == init["pair_fraction"]
    (f,) = rows_of(out, "failures")
    assert f["launches_failed_killed"] == "1" and f["rounds_lost"] == "1"


def test_a_variant_that_never_ran_is_a_row(toy):
    root, case, out = toy
    toy_case(root, "P1_r2", "P1")
    res = score(root, out, ["multiplex"], expected=["ghost"])
    ghosts = [r for r in res["cells"] if r["method"] == "ghost"]
    assert [r["case_id"] for r in ghosts] == ["P1_r1", "P1_r2"]
    assert all(r["problem"] == "not run" for r in ghosts)
    assert all(r["cause"] == "not_run" and r["imputed_initial"] for r in ghosts)


def test_the_run_is_recorded_as_killed_before_the_method_starts(toy):
    _, case, out = toy
    seen = {}

    def register(case, work, opts):
        seen.update(run_json(out, "fake", case))
        yield "fake", true_map(), {}

    assert methods.run_case(fake("fake", ["fake"], register), case, out) == ["fake"]
    assert seen["ok"] is False and seen["cause"] == "killed"
    assert run_json(out, "fake", case)["ok"] is True


def test_a_system_exit_is_an_exception_row_not_killed(toy):
    _, case, out = toy

    def register(case, work, opts):
        raise SystemExit("stare register: error: argument --stride: invalid int")
        yield  # pragma: no cover

    assert methods.run_case(fake("fake", ["fake"], register), case, out) == []
    run = run_json(out, "fake", case)
    assert run["ok"] is False and run["cause"] == "exception"
    assert "SystemExit" in run["error"] and "--stride" in run["error"]


def test_a_reported_memory_error_is_oom(toy):
    _, case, out = toy
    for k, exc in enumerate(
        [
            MemoryError("Unable to allocate 512. GiB for an array"),
            RuntimeError("java.lang.OutOfMemoryError: Java heap space"),
            RuntimeError("DefaultCPUAllocator: not enough memory"),
        ]
    ):

        def register(case, work, opts, exc=exc):
            raise exc
            yield  # pragma: no cover

        name = f"fake{k}"
        assert methods.run_case(fake(name, [name], register), case, out) == []
        assert run_json(out, name, case)["cause"] == "oom", exc

    def gpu(case, work, opts):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        yield  # pragma: no cover

    methods.run_case(fake("gpu", ["gpu"], gpu), case, out)
    assert run_json(out, "gpu", case)["cause"] == "gpu_oom"  # reported, never escalated


def test_a_timeout_is_not_oom(toy):
    _, case, out = toy

    def register(case, work, opts):
        raise TimeoutError("the registration did not finish in 14400 s")
        yield  # pragma: no cover

    assert methods.run_case(fake("slow", ["slow"], register), case, out) == []
    assert run_json(out, "slow", case)["cause"] == "timeout"
    assert "timeout" in resources.CAUSES and "oom" in resources.CAUSES


def test_an_oom_counter_rise_ends_the_run_as_oom(toy, tmp_path, monkeypatch):
    """The kernel kills one worker and the parent waits for it: the counter is what ends it."""
    _, case, out = toy
    counter = tmp_path / "memory.events"
    counter.write_text("low 0\nhigh 0\nmax 3\noom 0\noom_kill 0\n")
    monkeypatch.setattr(resources, "oom_events_path", lambda: counter)
    monkeypatch.setattr(methods, "OOM_POLL_S", 0.05)

    def register(case, work, opts):
        yield "fake_a", true_map(), {}
        counter.write_text("low 0\nhigh 0\nmax 9\noom 1\noom_kill 1\n")
        time.sleep(60)  # a parent waiting for a worker that is gone
        yield "fake_b", true_map(), {}

    t0 = time.perf_counter()
    done = methods.run_case(fake("fake", ["fake_a", "fake_b"], register), case, out)
    assert time.perf_counter() - t0 < 20
    assert done == ["fake_a"]  # what existed before the kill is kept
    run = run_json(out, "fake_b", case)
    assert run["ok"] is False and run["cause"] == "oom"
    assert resources.read_oom_kills(counter) == 1


def test_an_earlier_attempt_of_the_ladder_is_kept_as_wasted(toy, monkeypatch):
    root, case, out = toy

    def register(case, work, opts):
        if resources.host_info()["attempt"] == 1:
            raise MemoryError("first rung")
        yield "fake", true_map(), {}

    method = fake("fake", ["fake"], register)
    monkeypatch.setenv("REGBENCH_ATTEMPT", "1")
    monkeypatch.setenv("REGBENCH_MEM_GB", "16")
    assert methods.run_case(method, case, out) == []
    monkeypatch.setenv("REGBENCH_ATTEMPT", "2")
    monkeypatch.setenv("REGBENCH_MEM_GB", "64")
    assert methods.run_case(method, case, out) == ["fake"]
    run = run_json(out, "fake", case)
    assert run["attempt"] == 2 and run["mem_requested_gb"] == 64.0
    (prev,) = run["previous_attempts"]
    assert prev["attempt"] == 1 and prev["cause"] == "oom"
    assert prev["mem_requested_gb"] == 16.0
    res = score(root, out, ["multiplex"])
    r = _row(res["resources"], "P1_r1", "fake")
    assert r["ok"] and r["wasted_attempts"] == 1 and r["mem_requested_gb"] == 64.0


# ── direction, and what is warped when ───────────────────────────────────────
def test_a_ref_to_mov_method_is_scored_in_the_moving_frame(toy_cases, tmp_path):
    """A backward method with the TRUE map, by exact lookup: no inverse is approximated.

    The reference nuclei of this case are the moving nuclei under the generator's map, so
    the moving partner of every reference vertex is known exactly. Anywhere else the map is
    undefined and says so (NaN)."""
    from scipy.spatial import cKDTree

    src = toy_cases["rigid"]
    root = tmp_path / "cases"
    dst = root / "multiplex" / "rigid"
    dst.mkdir(parents=True)
    shutil.copy(src.points_path.parent / "case.json", dst / "case.json")
    pts = load_points(src)
    for key in ("ring_off", "ring_feat", "ring_hole", "n"):
        pts[f"cells_ref_{key}"] = pts[f"cells_mov_{key}"]
    pts["cells_ref_xy"] = pts["cells_mov_xy_truth"]
    write_npz(dst / "points.npz", **pts)
    case = load_case(dst)

    known_ref = np.asarray(pts["cells_ref_xy"], dtype=float)
    known_mov = np.asarray(pts["cells_mov_xy"], dtype=float)
    tree = cKDTree(known_ref)

    def lookup(xy):
        dist, idx = tree.query(np.asarray(xy, dtype=float))
        back = known_mov[idx].copy()
        back[dist > 0] = np.nan  # not a vertex: the truth is not known here
        return back

    def register(case, work, opts):
        yield "backward", PointMap(lookup, REF_TO_MOV), {}

    out = tmp_path / "out"
    done = methods.run_case(fake("backward", ["backward"], register), case, out)
    assert done == ["backward"]
    with np.load(result_dir(out, "backward", case) / "warped.npz") as z:
        assert str(z["direction"]) == REF_TO_MOV
        assert len(z["cells_xy"]) == len(
            pts["cells_ref_xy"]
        )  # the REFERENCE nuclei moved
    res = score(root, out, ["multiplex"])
    row, init = (
        _row(res["cells"], "rigid", "backward"),
        _row(res["cells"], "rigid", "initial"),
    )
    assert row["frame"] == "moving" and init["frame"] == "reference"
    assert row["pair_fraction"] > init["pair_fraction"]
    assert row["within_2um"] > 0.95  # the exact map: every nucleus is on its partner
    assert not row["imputed_initial"]


def test_points_are_warped_at_the_yield(toy):
    _, case, out = toy
    live = {"shift": np.array([1.0, 0.0])}

    def register(case, work, opts):
        fn = PointMap(
            lambda xy: np.asarray(xy, dtype=float) + live["shift"], MOV_TO_REF
        )
        yield "fake_a", fn, {}
        live["shift"] = np.array([0.0, 50.0])  # a later stage overwrites the object
        yield "fake_b", fn, {}

    methods.run_case(fake("fake", ["fake_a", "fake_b"], register), case, out)
    mov = load_points(case)["cells_mov_xy"]
    with np.load(result_dir(out, "fake_a", case) / "warped.npz") as z:
        assert np.allclose(z["cells_xy"], mov + [1.0, 0.0])
    with np.load(result_dir(out, "fake_b", case) / "warped.npz") as z:
        assert np.allclose(z["cells_xy"], mov + [0.0, 50.0])


def test_on_variant_is_called_at_each_yield_with_the_map(toy):
    _, case, out = toy
    seen = []

    def register(case, work, opts):
        yield "fake", true_map(), {"m0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    methods.run_case(
        fake("fake", ["fake"], register),
        case,
        out,
        label="relabelled",
        on_variant=lambda c, v, pm, info: seen.append((c.case_id, v, pm.direction)),
    )
    assert seen == [("P1_r1", "relabelled", MOV_TO_REF)]


def test_a_method_must_return_a_point_map(toy):
    _, case, out = toy

    def register(case, work, opts):
        yield "fake", lambda xy: xy, {}

    assert methods.run_case(fake("fake", ["fake"], register), case, out) == []
    assert "PointMap" in run_json(out, "fake", case)["error"]
    with pytest.raises(ValueError):
        PointMap(lambda xy: xy, "sideways")


# ── off the seams: one method's mask, every method scored on what it leaves ──
def _tiles(band, name="tiles", key="band_mov"):
    """A method with a seam mask: right off the seams, 30 px off on them."""

    def register(case, work, opts):
        def fn(xy):
            xy = np.asarray(xy, dtype=float) + SHIFT
            if key == "band_mov" and len(xy) == 4 * len(GRID):
                xy[np.repeat(band, 4)] += [30.0, 0.0]
            return xy

        yield name, PointMap(fn, MOV_TO_REF), {key: band}

    return fake(name, [name], register)


def _shifted(name, by):
    def register(case, work, opts):
        yield (
            name,
            PointMap(lambda xy: np.asarray(xy, dtype=float) + SHIFT + by, MOV_TO_REF),
            {},
        )

    return fake(name, [name], register)


def _band(n=20):
    band = np.zeros(len(GRID), dtype=bool)
    band[:n] = True
    return band


def test_every_variant_is_scored_again_on_the_nuclei_off_the_owners_seams(toy):
    root, case, out = toy
    methods.run_case(_tiles(_band()), case, out)
    methods.run_case(good("plain"), case, out)  # a method with no seams of its own
    res = score(root, out, ["multiplex"], seam_owners=("tiles",))
    non = res["nonband_cells"]
    names = {r["method"] for r in non}
    assert names == {"initial:non_tiles", "tiles:non_tiles", "plain:non_tiles"}
    # identical nuclei for every row, and their number is in the row
    assert {(r["n_mov_kept"], r["n_mov_all"]) for r in non} == {
        (len(GRID) - 20, len(GRID))
    }
    assert {r["n_moving"] for r in non} == {len(GRID) - 20}
    assert {r["owner"] for r in non} == {"tiles"}
    full = _row(res["cells"], "P1_r1", "tiles")
    off = _row(non, "P1_r1", "tiles:non_tiles")
    assert off["pair_fraction"] == 1.0 > full["pair_fraction"]
    assert off["match_radius_px"] == full["match_radius_px"]  # one gate per case
    assert _row(non, "P1_r1", "plain:non_tiles")["pair_fraction"] == 1.0
    assert _row(non, "P1_r1", "initial:non_tiles")["pair_fraction"] < 0.5
    assert res["nonband_missing"] == [
        {
            "dataset": "multiplex",
            "owner": "tiles",
            "n_cases": 1,
            "n_cases_without_mask": 0,
        }
    ]
    assert rows_of(out, "nonband_cells") and rows_of(out, "nonband_summary")


def test_the_off_seam_block_never_enters_the_primary_tables(toy):
    root, case, out = toy
    methods.run_case(_tiles(_band()), case, out)
    methods.run_case(good("stare"), case, out)
    res = score(root, out, ["multiplex"], seam_owners=("tiles",), primary=["stare"])
    for table in ("cells", "summary_cells", "paired_vs_primary", "resources"):
        assert not [r for r in res[table] if ":non_" in r["method"]], table
        assert ":non_" not in (out / "tables" / f"{table}.csv").read_text()
    text = (out / "tables" / "summary.md").read_text()
    head, block = text.split("## Off the seams")
    assert ":non_" not in head
    assert "tiles:non_tiles" in block and "stare:non_tiles" in block
    assert "not ranked" in text and "61 of 81 moving" in block
    assert "0 of 1 case(s) have no mask" in block


def test_a_failed_owner_gives_no_off_seam_row_to_anyone(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    c1, c2 = toy_case(root, "P1_r1", "P1"), toy_case(root, "P2_r1", "P2")
    methods.run_case(_tiles(_band()), c1, out)

    def broken(case, work, opts):
        raise RuntimeError("the owner failed here")
        yield

    methods.run_case(fake("tiles", ["tiles"], broken), c2, out)
    for case in (c1, c2):
        methods.run_case(good("plain"), case, out)
    res = score(root, out, ["multiplex"], seam_owners=("tiles",))
    assert {r["case_id"] for r in res["nonband_cells"]} == {"P1_r1"}
    assert res["nonband_missing"][0]["n_cases_without_mask"] == 1
    assert res["nonband_missing"][0]["n_cases"] == 2
    text = (out / "tables" / "summary.md").read_text()
    assert "1 of 2 case(s) have no mask" in text
    # the primary table still has both cases, the failed one at the unregistered pose
    assert _row(res["cells"], "P2_r1", "tiles")["imputed_initial"]


def test_the_mask_does_not_move_when_another_methods_scores_change(toy):
    root, case, out = toy
    methods.run_case(_tiles(_band()), case, out)
    kept = []
    for by in ([0.0, 0.0], [3.0, 0.0], [19.0, 0.0]):
        methods.run_case(_shifted("other", np.array(by)), case, out)
        res = score(root, out, ["multiplex"], seam_owners=("tiles",))
        own = _row(res["nonband_cells"], "P1_r1", "tiles:non_tiles")
        other = _row(res["nonband_cells"], "P1_r1", "other:non_tiles")
        kept.append((own["n_mov_kept"], own["pair_fraction"], other["n_mov_kept"]))
    assert len(set(kept)) == 1 and kept[0][0] == len(GRID) - 20
    assert other["pair_fraction"] < 0.5  # the last map is 19 px off: its score moved


def test_a_failed_variant_keeps_its_off_seam_row_at_the_unregistered_pose(toy):
    root, case, out = toy
    methods.run_case(_tiles(_band()), case, out)
    res = score(root, out, ["multiplex"], expected=["ghost"], seam_owners=("tiles",))
    ghost = _row(res["nonband_cells"], "P1_r1", "ghost:non_tiles")
    initial = _row(res["nonband_cells"], "P1_r1", "initial:non_tiles")
    assert ghost["imputed_initial"] and ghost["cause"] == "not_run"
    assert ghost["pair_fraction"] == initial["pair_fraction"]


def test_a_reference_side_mask_is_stored_and_drops_reference_nuclei(toy):
    root, case, out = toy
    band = _band(30)
    methods.run_case(_tiles(band, name="blocks", key="band_ref"), case, out)
    methods.run_case(good("plain"), case, out)
    with np.load(result_dir(out, "blocks", case) / "warped.npz") as z:
        assert z["band_ref"].sum() == 30 and "band" not in z.files
    res = score(root, out, ["multiplex"], seam_owners=("blocks",))
    for name in ("blocks:non_blocks", "plain:non_blocks"):
        r = _row(res["nonband_cells"], "P1_r1", name)
        assert (r["n_ref_kept"], r["n_mov_kept"]) == (len(GRID) - 30, len(GRID))


def test_an_owner_that_was_not_run_makes_no_block(toy):
    root, case, out = toy
    methods.run_case(good("plain"), case, out)
    res = score(root, out, ["multiplex"])  # the default owners, none of them here
    assert res["nonband_cells"] == [] and res["nonband_missing"] == []
    assert "Off the seams" not in (out / "tables" / "summary.md").read_text()
    assert not (out / "tables" / "nonband_cells.csv").exists()


# ── per-patient methods ──────────────────────────────────────────────────────
def test_a_failed_round_of_a_group_method_fails_only_that_round(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    c1, c2 = toy_case(root, "P1_r1", "P1"), toy_case(root, "P1_r2", "P1")

    def register_group(cases, work, opts):
        assert [c.case_id for c in cases] == ["P1_r1", "P1_r2"]
        yield "P1_r1", "flaky", None, {"error": "x"}
        yield "P1_r2", "flaky", true_map(), {}

    method = fake("flaky", ["flaky"], register_group=register_group)
    done = methods.run_group(method, [c1, c2], out)
    assert done == {"P1_r1": [], "P1_r2": ["flaky"]}
    bad, ok = run_json(out, "flaky", c1), run_json(out, "flaky", c2)
    assert bad["ok"] is False and bad["error"] == "x"
    assert ok["ok"] is True
    assert ok["unit"] == "patient" and ok["group_cases"] == ["P1_r1", "P1_r2"]
    assert ok["launch_id"] == bad["launch_id"]
    with pytest.raises(SystemExit):
        methods.run_case(method, c1, out)


def test_the_methods_own_writer_runs_after_the_clock_stops(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    c1, c2 = toy_case(root, "P1_r1", "P1"), toy_case(root, "P1_r2", "P1")
    calls = []

    def register_group(cases, work, opts):
        for c in cases:
            yield c.case_id, "pp", true_map(), {"_state": {"registrar": object()}}

    def write_native(state, case_id, dest_dir):
        calls.append((sorted(state), case_id, dest_dir.name))
        time.sleep(0.4)
        (dest_dir / "registered.ome.tif").write_bytes(b"")
        return {
            "files": [str(dest_dir / "registered.ome.tif")],
            "format": "ome-tiff",
            "channels": 1,
            "dtype": "uint16",
        }

    method = fake(
        "pp", ["pp"], register_group=register_group, write_native=write_native
    )
    methods.run_group(method, [c1, c2], out)
    assert calls == [
        (["registrar"], "P1_r1", "native"),
        (["registrar"], "P1_r2", "native"),
    ]
    run = run_json(out, "pp", c2)
    assert run["native"]["format"] == "ome-tiff" and run["write_s"] >= 0.4
    assert run["register_s"] < 0.3  # the writer is not billed
    assert "_state" not in json.dumps(run)
    assert (result_dir(out, "pp", c2) / "native" / "registered.ome.tif").exists()


def test_summary_counts_failures_and_has_a_common_block(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    by_patient = {
        p: [toy_case(root, f"{p}_r{k}", p) for k in (1, 2)] for p in ("P1", "P2")
    }

    def register_group(cases, work, opts):
        for c in cases:
            if c.case_id == "P2_r2":
                yield c.case_id, "flaky", None, {"error": "anchor unavailable"}
            else:
                yield c.case_id, "flaky", true_map(), {}

    flaky = fake("flaky", ["flaky"], register_group=register_group)
    for cases in by_patient.values():
        methods.run_group(flaky, cases, out)
        methods.run_group(good(), cases, out)  # a pair method, addressed by patient
    res = score(root, out, ["multiplex"])
    summary = {(r["method"], r["subset"]): r for r in res["summary_cells"]}
    assert summary[("flaky", "all")]["n_ok"] == 3
    assert summary[("flaky", "all")]["n_cases"] == 4
    assert summary[("flaky", "all")]["n_patients"] == 2
    assert summary[("good", "all")]["n_ok"] == 4
    for method in ("initial", "good", "flaky"):
        assert summary[(method, "common")]["n_cases"] == 3, method
    # the median hides the failed round; the minimum over cases and n_ok do not
    assert summary[("flaky", "all")]["pair_fraction_min_case"] == 0.0
    assert summary[("good", "all")]["pair_fraction_min_case"] == 1.0
    assert summary[("flaky", "common")]["pair_fraction_min_case"] == 1.0
    (f,) = [f for f in res["failures"] if f["method"] == "flaky"]
    assert f["launches"] == 2 and f["launches_failed"] == 1 and f["rounds_lost"] == 1
    (g,) = [f for f in res["failures"] if f["method"] == "good"]
    assert g["launches_failed"] == 0 and g["rounds_lost"] == 0
    per_patient = [r for r in res["summary_resources"] if r["method"] == "flaky"]
    assert sorted((r["group"], r["n_rounds"], r["n_ok"]) for r in per_patient) == [
        ("P1", 2, 2),
        ("P2", 2, 1),
    ]
    # a primary list defines the common block: `good` registered everything
    res = score(root, out, ["multiplex"], primary=["good"])
    summary = {(r["method"], r["subset"]): r for r in res["summary_cells"]}
    assert summary[("flaky", "common")]["n_cases"] == 4


def test_summary_says_descriptive_at_every_number_of_patients(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    for k in range(7):
        methods.run_case(good(), toy_case(root, f"P{k}_r1", f"P{k}"), out)
    for left in (7, 6, 5):
        score(root, out, ["multiplex"])
        text = (out / "tables" / "summary.md").read_text()
        assert (
            f"The independent unit is the patient. At most {left} unit(s): "
            f"{DESCRIPTIVE}." in text
        )
        shutil.rmtree(root / "multiplex" / f"P{left - 1}_r1")


# ── cost, per variant ────────────────────────────────────────────────────────
# A worker that holds 600 MB and KEEPS it resident. Two things made the earlier version
# (one byte written per page, then sleep) depend on what else the machine was doing: an
# operating system short of memory compresses or pages out idle pages, and pages that are
# all zeros but one byte compress to nothing, so the worker's resident size could fall
# far below 600 MB before the driver read it. Here every page is filled with random bytes
# (nothing to compress) and written again five times a second (never idle).
HOLD = """
import os, sys, time
MB = 1024 * 1024
block = bytearray(600 * MB)
chunk = os.urandom(MB)
for i in range(600):
    block[i * MB:(i + 1) * MB] = chunk
print("ready", flush=True)
end = time.time() + 120
while time.time() < end:
    for i in range(0, len(block), 4096):
        block[i] ^= 1
    time.sleep(0.2)
"""


def _resident_mb(pid):
    """A process's resident size by an instrument that is NOT the driver's monitor, or
    None where there is none (no psutil and no /proc)."""
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / 2**20
    except ImportError:
        pass
    try:
        fields = Path(f"/proc/{pid}/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None


def test_each_variant_has_its_own_clock_and_peak(toy):
    """A variant yielded before a later stage is not billed for that stage.

    What is proved is unchanged: the first variant's bill stops at its yield (it has
    neither the later stage's 0.4 s nor its 600 MB), and the second variant's bill, read
    by the DRIVER's own monitor, has both. The test only makes sure the 600 MB are there
    to be seen: it waits, with an independent reading, until the worker really is
    resident, instead of trusting that it still is after a fixed sleep.

    Each bill is compared with this process's OWN resident size at the same yield, not
    with the other bill: a peak is a sum over the process tree, and a large test process
    is pushed out of memory by the very 600 MB the worker takes (measured with a 12 GB
    test process on a 16 GB machine: 460 MB at the first yield, 661 MB at the second,
    a difference of 200 MB with all 600 MB resident in the worker)."""
    import subprocess
    import sys

    _, case, out = toy
    own = {}

    def register(case, work, opts):
        own["a"] = _resident_mb(os.getpid())
        yield "fake_a", true_map(), {}
        # the later stage: 600 MB in a worker, and some time
        worker = subprocess.Popen([sys.executable, "-c", HOLD], stdout=subprocess.PIPE)
        try:
            assert worker.stdout.readline().strip() == b"ready"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                seen = _resident_mb(worker.pid)
                if seen is None or seen > 550:
                    break
                time.sleep(0.05)
            else:
                pytest.fail(
                    f"the worker holds 600 MB and only {seen:.0f} MB are resident "
                    "after 60 s: this machine cannot keep it in memory"
                )
            time.sleep(0.4)
            own["b"] = _resident_mb(os.getpid())
            yield "fake_b", true_map(), {}
        finally:
            worker.kill()
            worker.wait()

    methods.run_case(fake("fake", ["fake_a", "fake_b"], register), case, out)
    a, b = run_json(out, "fake_a", case), run_json(out, "fake_b", case)
    assert b["register_s"] - a["register_s"] >= 0.4
    if a["resources"]["rss_method"] != "rusage":
        # the first bill is this process alone, the second has the worker on top
        assert a["resources"]["peak_rss_mb"] - own["a"] < 300
        assert b["resources"]["peak_rss_mb"] - own["b"] > 300


def test_pause_freezes_the_memory_peak():
    import subprocess
    import sys

    child = (
        "b = bytearray(200 * 1024 * 1024)\n"
        "for i in range(0, len(b), 4096):\n"
        "    b[i] = 1\n"
        "import time; time.sleep(0.5)\n"
    )
    mon = resources.Monitor(interval=0.05)
    if mon.method == "rusage":
        pytest.skip("no psutil and no /proc: a high-water mark cannot be paused")
    mon.start()
    before = mon.snapshot()
    mon.pause()
    subprocess.run([sys.executable, "-c", child], check=True)  # 200 MB, while paused
    mon.resume()
    after = mon.snapshot()
    mon.stop()
    assert after["peak_rss_mb"] - before["peak_rss_mb"] < 50
    assert after["wall_s"] - before["wall_s"] < 0.3  # nor is its time billed


# ── run.json ─────────────────────────────────────────────────────────────────
def test_info_arrays_round_trip(toy):
    _, case, out = toy
    m0 = [[1.0, 0.0, 12.0], [0.0, 1.0, -9.0], [0.0, 0.0, 1.0]]

    def register(case, work, opts):
        yield "fake_a", true_map(), {"m0": m0}
        yield "fake_b", true_map(), {"m0": np.array(m0), "n": np.int64(3)}

    methods.run_case(fake("fake", ["fake_a", "fake_b"], register), case, out)
    for variant in ("fake_a", "fake_b"):
        got = run_json(out, variant, case)["info"]["m0"]
        assert got == m0 and np.array(got).shape == (3, 3)
    assert run_json(out, "fake_b", case)["info"]["n"] == 3


def test_label_renames_every_variant(toy):
    _, case, out = toy
    assert methods.variant_names("valis", "valis120") == [
        "valis120_rigid",
        "valis120",
        "valis120_micro",
    ]
    done = methods.run_case("initial", case, out, label="baseline")
    assert done == ["baseline"]
    assert (result_dir(out, "baseline", case) / "warped.npz").exists()


# ── a case prepared again: its earlier results are stale, not failures ────────────────
def test_a_result_for_an_earlier_version_of_the_case_is_not_reused_and_not_a_failure(
    tmp_path,
):
    """Review item 9: after a forced re-prepare the polygons of a case change (other
    vertices, another count). A result made for the old ones must not be reused as
    `ok`, and the scorer must not call it a failure of the method (`exception`)."""
    from benchmarks.regbench import cases as cases_mod
    from benchmarks.regbench import score as scorer

    root, out = tmp_path / "cases", tmp_path / "out"
    case = toy_case(root, "P1_r1", "P1")
    calls = []

    def register(case, work, opts):
        calls.append(case.case_id)
        yield "m", true_map(), {}

    method = fake("m", ["m"], register)
    assert methods.run_case(method, case, out, resume=True) == ["m"]
    made_for = run_json(out, "m", case)["points_sha256"]
    assert made_for["outlines"] and made_for["nuclei"] == ""
    assert methods.run_case(method, case, out, resume=True) == ["m"]
    assert calls == ["P1_r1"]  # the same case: reused, nothing run
    assert scorer._load_run(out, "m", case)[3] == ""

    # ADDING a group (bare nuclei) leaves the result of the outlines valid
    write_case(
        root,
        case,
        cells_mov=squares(GRID - SHIFT),
        cells_ref=squares(GRID),
        nuclei_mov=squares(GRID - SHIFT, 3.0),
        nuclei_ref=squares(GRID, 3.0),
    )
    case = load_case(root / "multiplex" / "P1_r1")
    assert cases_mod.stale_points(made_for, case) == []
    assert scorer._load_run(out, "m", case)[3] == ""

    # the outlines themselves change (one fewer, as a re-prepare with other settings)
    write_case(
        root, case, cells_mov=squares(GRID[:-1] - SHIFT), cells_ref=squares(GRID[:-1])
    )
    case = load_case(root / "multiplex" / "P1_r1")
    assert cases_mod.stale_points(made_for, case) == ["outlines"]
    run, warped, problem, cause = scorer._load_run(out, "m", case)
    assert warped is None and cause == "not_run" and "prepared again" in problem
    assert "outlines" in problem
    res = score(root, out, ["multiplex"], with_tissue=False)
    row = next(r for r in res["cells"] if r["method"] == "m")
    assert row["cause"] == "not_run" != "exception"
    # and the launcher's reuse does not take it: the unit runs again
    assert methods.run_case(method, case, out, resume=True) == ["m"]
    assert calls == ["P1_r1", "P1_r1"]
    assert scorer._load_run(out, "m", case)[3] == ""
