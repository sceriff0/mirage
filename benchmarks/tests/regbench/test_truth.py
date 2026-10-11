"""Known-truth error: exact probe pairs, both directions, per family, a frozen spec."""

import io
import json
import math
import shutil
import tokenize
import types

import numpy as np
import pytest

from benchmarks.regbench import cells, methods, truth
from benchmarks.regbench.cases import (
    groups,
    list_cases,
    load_case,
    load_points,
    polys,
    result_dir,
    write_json,
)
from benchmarks.regbench.datasets import semisynth, synthetic
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.score import NON, score

from .conftest import semisynth_args

SPEC = truth.load_spec()
REDUCED = ("rigid", "bumps25", "seams")  # one in-model control and the two outside it
PX = synthetic.PIXEL_SIZE_UM


def fake(name, register):
    return types.SimpleNamespace(
        NAME=name, SCOPE="pair", VARIANTS=(name,), register=register
    )


def exact():
    """A forward method that returns the generating map itself."""

    def register(case, work, opts):
        yield (
            "exact",
            PointMap(synthetic.forward_map(case.extra["spec"]), MOV_TO_REF),
            {},
        )

    return fake("exact", register)


def backward():
    """A backward method that knows the truth at the probes, by LOOKUP: a reference probe
    goes to its moving partner. Any other point takes the shift of the nearest probe, so
    the nuclei and the lattice still get finite coordinates."""
    from scipy.spatial import cKDTree

    def register(case, work, opts):
        pts = load_points(case)
        t_ref, t_mov = pts["truth_ref"], pts["truth_mov"]
        tree = cKDTree(t_ref)

        def lookup(xy):
            xy = np.asarray(xy, dtype=float)
            _, idx = tree.query(xy)
            return xy + (t_mov[idx] - t_ref[idx])

        yield "backward", PointMap(lookup, REF_TO_MOV), {}

    return fake("backward", register)


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    """Three families of the truth suite at 1024 px (not the frozen cases: no hash)."""
    root = tmp_path_factory.mktemp("truth_cases")
    for name, spec in synthetic.truth_suite(n=1024, families=REDUCED).items():
        synthetic.prepare_one(root, name, spec)
    return root, list_cases(root, "synthetic")


@pytest.fixture(scope="module")
def scored(reduced, tmp_path_factory):
    root, cases = reduced
    out = tmp_path_factory.mktemp("truth_out")
    for case in cases:
        assert methods.run_case(exact(), case, out) == ["exact"]
        assert methods.run_case(backward(), case, out) == ["backward"]
    # a run that failed, written by hand, on one case; `ghost` never ran on the others
    failed = result_dir(out, "ghost", cases[0])
    write_json(failed / "run.json", {"ok": False, "error": "boom", "cause": "oom"})
    return out, score(root, out, ["synthetic"], expected=["ghost"])


def _row(rows, case_id, method):
    (r,) = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
    return r


# ── the probes ────────────────────────────────────────────────────────────────
def test_probes_are_exact_pairs(reduced):
    _, cases = reduced
    for case in cases:
        pts = load_points(case)
        g = synthetic.forward_map(case.extra["spec"])
        assert len(pts["truth_mov"]) > 100
        np.testing.assert_allclose(g(pts["truth_mov"]), pts["truth_ref"], atol=1e-9)
        # a probe is the centroid of a nucleus segmented on the native moving slide
        mov = polys(pts, "mov")
        _, cent = cells.area_centroid(cells.Polys(mov.xy, mov.ring_off))
        np.testing.assert_allclose(cent[pts["truth_cell"]], pts["truth_mov"], atol=1e-3)
        # the case says what its own true map does: how much moves, and no fold
        field = case.extra["field"]
        assert field["fold_pct"] == 0.0 and field["min_jacobian"] > 0.05
        # and how its seed was settled: here the seed asked for did not fold
        if case.family == "seams":  # discontinuous on purpose: the rule does not apply
            assert "reseed" not in case.extra
        else:
            settled = case.extra["reseed"]
            assert settled["skipped"] == []
            assert (
                settled["seed"] == settled["seed_asked"] == case.extra["spec"]["seed"]
            )
            assert settled["min_jacobian"] > settled["min_jacobian_required"] == 0.05
            assert settled["lattice_px"] == 4
        if case.family == "rigid":
            assert field["moved_frac_over_1um"] == 0.0
        if case.family == "bumps25":  # the reduced copy keeps the frozen density
            assert 0.15 < field["moved_frac_over_1um"] < 0.35
        # and its true position lies well inside the reference
        margin = SPEC["probes"]["margin_px"]
        assert pts["truth_ref"].min() >= margin
        assert pts["truth_ref"].max() <= 1024 - 1 - margin


def test_probes_are_a_seeded_sample_of_at_most_the_cap():
    cent = np.random.default_rng(0).uniform(100, 8000, (50_000, 2))
    a = truth.probe_arrays(cent, lambda p: p + 3.0, (8192, 8192), seed=5)
    b = truth.probe_arrays(cent, lambda p: p + 3.0, (8192, 8192), seed=5)
    c = truth.probe_arrays(cent, lambda p: p + 3.0, (8192, 8192), seed=6)
    assert len(a["truth_mov"]) == SPEC["probes"]["max"] == 20000
    assert np.array_equal(a["truth_cell"], b["truth_cell"])
    assert not np.array_equal(a["truth_cell"], c["truth_cell"])
    np.testing.assert_array_equal(a["truth_ref"], a["truth_mov"] + 3.0)
    # a probe whose image leaves the reference (less the margin) is dropped
    edge = truth.probe_arrays(
        np.array([[10.0, 500.0], [500.0, 500.0], [np.nan, 1.0]]),
        lambda p: p,
        (1024, 1024),
        seed=0,
    )
    assert edge["truth_cell"].tolist() == [1]


# ── both directions ───────────────────────────────────────────────────────────
def test_forward_method_with_the_true_map_has_zero_error(scored, reduced):
    _, res = scored
    for case in reduced[1]:
        row = _row(res["truth"], case.case_id, "exact")
        assert row["frame"] == "reference" and not row["imputed_initial"]
        assert row["tre_um_max"] < 1e-6
        assert row["frac_under_1um"] == 1.0 and row["n_nonfinite"] == 0
        init = _row(res["truth"], case.case_id, "initial")
        assert init["tre_um_p50"] > 5.0  # a 36 px shift at 0.325 um
        assert row["initial_tre_um_p50"] == pytest.approx(init["tre_um_p50"])
        assert row["frac_improved"] == 1.0 and init["frac_improved"] == 0.0
        assert _row(res["truth"], case.case_id, "truth")["tre_um_max"] == 0.0


def _code_without_comments(path):
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    return " ".join(t.string for t in tokens if t.type != tokenize.COMMENT)


def test_backward_method_is_scored_without_any_inverse(scored, reduced):
    out, res = scored
    for case in reduced[1]:
        row = _row(res["truth"], case.case_id, "backward")
        assert row["frame"] == "moving"
        assert row["tre_um_max"] < 1e-6  # fn(truth_ref) against truth_mov, as stored
        with np.load(result_dir(out, "backward", case) / "warped.npz") as z:
            pts = load_points(case)
            assert str(z["direction"]) == REF_TO_MOV
            np.testing.assert_allclose(z["truth_xy"], pts["truth_mov"], atol=1e-9)
    code = _code_without_comments(truth.SPEC_PATH.with_name("truth.py"))
    assert "error_stats" in code  # the scan read the file
    for banned in ("invert", "inverse", "fixed_point", "RBFInterpolator"):
        assert banned not in code, banned


def test_a_known_shift_is_reported_in_um():
    source = np.random.default_rng(0).uniform(0, 500, (200, 2))
    target = source + [40.0, 0.0]
    got = truth.error_stats(target + [3.0, 4.0], target, source, PX)
    assert got["tre_um_p50"] == pytest.approx(1.625)
    assert got["tre_um_p90"] == pytest.approx(1.625)
    assert got["tre_um_max"] == pytest.approx(1.625)
    assert got["frac_under_2um"] == 1.0 and got["frac_under_1um"] == 0.0
    assert got["initial_tre_um_p50"] == pytest.approx(40 * PX)
    assert got["n_probes"] == 200 and got["frac_improved"] == 1.0
    assert SPEC["thresholds_um"] == [1.0, 2.0]  # the two columns' thresholds


def test_nonfinite_points_count_against_the_method():
    source = np.zeros((10, 2))
    target = source + [10.0, 0.0]
    mapped = target.copy()
    mapped[:2] = np.nan  # the method could not map two of ten probes
    got = truth.error_stats(mapped, target, source, PX)
    assert got["n_nonfinite"] == 2
    assert got["frac_under_1um"] == 0.8 and got["frac_improved"] == 0.8
    assert got["tre_um_p50"] == 0.0
    assert got["tre_um_p90"] == math.inf and got["tre_um_max"] == math.inf
    lost = truth.error_stats(np.full((10, 2), np.inf), target, source, PX)
    assert lost["tre_um_p50"] == math.inf and lost["frac_under_2um"] == 0.0
    # and the family table keeps the infinity instead of dropping it
    rows = [
        {
            "dataset": "d",
            "family": "f",
            "pose": "p",
            "method": "m",
            "in_stare_model": True,
            "imputed_initial": False,
            "fold_pct": 0.0,
            **stats,
        }
        for stats in (got, got, lost)
    ]
    (agg,) = truth.by_family(rows)
    assert agg["tre_um_max_max"] == math.inf and agg["tre_um_p50_max"] == math.inf
    assert agg["tre_um_p50_med"] == 0.0 and agg["n_cases"] == 3


def test_no_pixel_size_gives_nan():
    source = np.zeros((5, 2))
    got = truth.error_stats(source + 1.0, source + 2.0, source, None)
    for key in ("tre_um_p50", "tre_um_p90", "tre_um_max", "initial_tre_um_p50"):
        assert math.isnan(got[key]), key
    assert math.isnan(got["frac_under_1um"]) and math.isnan(got["frac_under_2um"])
    assert got["n_probes"] == 5 and got["frac_improved"] == 1.0


def test_failed_run_is_scored_at_the_unregistered_pose(scored, reduced):
    _, res = scored
    first, second = reduced[1][0], reduced[1][1]
    row = _row(res["truth"], first.case_id, "ghost")
    init = _row(res["truth"], first.case_id, "initial")
    assert row["imputed_initial"] and row["problem"] == "boom" and row["cause"] == "oom"
    assert row["tre_um_p50"] == init["tre_um_p50"] > 0
    assert row["tre_um_max"] == init["tre_um_max"]
    assert row["frac_improved"] == 0.0
    never = _row(res["truth"], second.case_id, "ghost")
    assert never["imputed_initial"] and never["problem"] == "not run"
    assert (
        never["tre_um_p50"]
        == _row(res["truth"], second.case_id, "initial")["tre_um_p50"]
    )
    agg = [r for r in res["truth_by_family"] if r["method"] == "ghost"]
    assert agg and all(r["n_ok"] == 0 and r["n_cases"] == 1 for r in agg)


def test_nuclei_in_bands_get_a_second_truth_row(reduced, tmp_path):
    root, cases = reduced
    case = cases[0]
    n_mov = polys(load_points(case), "mov").n_features

    def register(case, work, opts):
        fn = synthetic.forward_map(case.extra["spec"])

        def off_in_band(xy):  # wrong by 10 px for the first half of the probes only
            out = fn(xy)
            if len(out) == len(load_points(case)["truth_mov"]):
                out[: len(out) // 2] += 10.0
            return out

        band = np.zeros(n_mov, dtype=bool)
        cell = load_points(case)["truth_cell"]
        band[cell[: len(cell) // 2]] = True
        yield "banded", PointMap(off_in_band, MOV_TO_REF), {"band_mov": band}

    one = tmp_path / "cases"
    shutil.copytree(case.dir, one / "synthetic" / case.case_id)
    out = tmp_path / "out"
    methods.run_case(
        fake("banded", register), load_case(one / "synthetic" / case.case_id), out
    )
    res = score(one, out, ["synthetic"], seam_owners=("banded",))
    whole = _row(res["truth"], case.case_id, "banded")
    rest = _row(res["nonband_truth"], case.case_id, "banded" + NON + "banded")
    assert whole["tre_um_max"] > 4.0 and rest["tre_um_max"] < 1e-6
    assert rest["n_probes"] < whole["n_probes"] == rest["n_probes_all"]
    # every variant gets the row, on the same probes; none of them is in truth.csv
    start = _row(res["nonband_truth"], case.case_id, "initial" + NON + "banded")
    assert start["n_probes"] == rest["n_probes"] and start["owner"] == "banded"
    assert not [r for r in res["truth"] if NON in r["method"]]
    assert not [r for r in res["truth_by_family"] if NON in r["method"]]


@pytest.mark.slow
def test_a_reference_side_owner_drops_the_probes_its_own_run_marked(reduced, tmp_path):
    """A mask on reference nuclei has no probe index (a probe is a MOVING nucleus's
    centroid and its true reference position), so the owner says which probes are on its
    seams (``band_truth``); the scorer takes that and nothing else."""
    root, cases = reduced
    case = cases[0]
    pts = load_points(case)
    n_ref, n_probes = polys(pts, "ref").n_features, len(pts["truth_mov"])
    on_seam = np.zeros(n_probes, dtype=bool)
    on_seam[: n_probes // 3] = True
    band_ref = np.zeros(n_ref, dtype=bool)
    band_ref[::7] = True

    def blocks(marks):
        def register(case, work, opts):
            fn = synthetic.forward_map(case.extra["spec"])

            def off_on_seams(xy):  # 10 px wrong on the marked probes only
                out = fn(xy)
                if len(out) == n_probes:
                    out[on_seam] += 10.0
                return out

            yield "blocks", PointMap(off_on_seams, MOV_TO_REF), marks

        return fake("blocks", register)

    one = tmp_path / "cases"
    shutil.copytree(case.dir, one / "synthetic" / case.case_id)
    copy = load_case(one / "synthetic" / case.case_id)
    out = tmp_path / "out"
    methods.run_case(blocks({"band_ref": band_ref, "band_truth": on_seam}), copy, out)
    res = score(one, out, ["synthetic"], seam_owners=("blocks",))
    whole = _row(res["truth"], case.case_id, "blocks")
    rest = _row(res["nonband_truth"], case.case_id, "blocks" + NON + "blocks")
    assert whole["tre_um_max"] > 4.0 and rest["tre_um_max"] < 1e-6
    assert rest["n_probes"] == n_probes - n_probes // 3
    assert rest["n_probes_all"] == whole["n_probes"] == n_probes
    start = _row(res["nonband_truth"], case.case_id, "initial" + NON + "blocks")
    assert start["n_probes"] == rest["n_probes"] and start["owner"] == "blocks"
    cells_row = _row(res["nonband_cells"], case.case_id, "blocks" + NON + "blocks")
    assert cells_row["n_ref_kept"] == n_ref - int(band_ref.sum())
    assert not [r for r in res["truth"] if NON in r["method"]]

    # a reference-side owner that marks no probes gets cell rows and NO truth rows: the
    # scorer does not guess which probes a reference nucleus stands for
    out2 = tmp_path / "out2"
    methods.run_case(blocks({"band_ref": band_ref}), copy, out2)
    res2 = score(one, out2, ["synthetic"], seam_owners=("blocks",))
    assert res2["nonband_truth"] == []
    assert _row(res2["nonband_cells"], case.case_id, "blocks" + NON + "blocks")
    # a probe mask of the wrong length is not applied; with a moving-side mask too, a
    # probe must survive both
    from benchmarks.regbench.score import _truth_keep

    assert _truth_keep(pts, None, on_seam[:-1]) is None
    assert _truth_keep({}, None, on_seam) is None
    assert np.array_equal(_truth_keep(pts, None, on_seam), ~on_seam)
    keep_mov = np.ones(polys(pts, "mov").n_features, dtype=bool)
    keep_mov[pts["truth_cell"][-1]] = False
    both = _truth_keep(pts, keep_mov, on_seam)
    assert not both[-1] and not both[0] and both.sum() == (~on_seam).sum() - 1


# ── per family ────────────────────────────────────────────────────────────────
def test_families_are_never_pooled(scored):
    out, res = scored
    agg = res["truth_by_family"]
    by_method = {}
    for r in agg:
        by_method.setdefault(r["method"], []).append((r["family"], r["pose"]))
        assert r["n_cases"] == 1  # one case per (family, pose): nothing was merged
    want = sorted((f, p) for f in REDUCED for p in truth.pose_names(SPEC))
    for method in ("initial", "exact", "backward", "ghost", "truth"):
        assert sorted(by_method[method]) == want, method
    assert {r["family"] for r in agg} == set(REDUCED)
    text = (out / "tables" / "summary.md").read_text()
    assert "## Known-truth error, per deformation family" in text
    assert "never pooled" in text
    assert (out / "tables" / "truth.csv").exists()
    assert (out / "tables" / "truth_by_family.csv").exists()


def test_out_of_model_families_are_present_and_marked(scored):
    names = truth.family_names(SPEC)
    assert names == [
        "rigid",
        "wave",
        "grid",
        "multiscale",
        "bumps100",
        "bumps25",
        "seams",
    ]
    outside = [n for n in names if not truth.in_stare_model(SPEC, n)]
    assert outside == ["bumps25", "seams"]
    _, res = scored
    for r in res["truth"] + res["truth_by_family"]:
        assert r["in_stare_model"] is (r["family"] == "rigid"), r["family"]
    # the frozen amplitudes and scales
    fam = {n: truth.family(SPEC, n) for n in names}
    assert (fam["wave"]["amp_px"], fam["wave"]["wavelength_px"]) == (10.0, 1500.0)
    assert (fam["grid"]["sd_px"], fam["grid"]["nodes"]) == (5.0, 9)
    assert fam["multiscale"]["rms_um"] == 2.0
    assert fam["multiscale"]["scales_um"] == [333.0, 166.0, 83.0]
    assert (fam["bumps100"]["sigma_px"], fam["bumps100"]["count"]) == (100.0, 120)
    assert (fam["bumps25"]["sigma_px"], fam["bumps25"]["count"]) == (25.0, 4000)
    assert (fam["seams"]["sd_px"], fam["seams"]["block_px"]) == (3.0, 1950)
    poses = {p["name"]: (p["theta_deg"], p["shift_px"]) for p in SPEC["poses"]}
    assert poses == {"shift": (0.0, [30.0, -20.0]), "rot": (2.0, [30.0, -20.0])}
    assert SPEC["n_px"] == 8192 and SPEC["probes"] == {
        "max": 20000,
        "margin_px": 32,
        "seed_offset": 104729,
    }


def test_bumps25_is_finer_than_stares_lattice():
    lattice = SPEC["method_grids_px"]["stare_lattice"]
    assert truth.family(SPEC, "bumps25")["sigma_px"] < lattice
    assert truth.family(SPEC, "bumps100")["sigma_px"] > lattice
    cli = pytest.importorskip("stare.cli")
    register = cli.build_parser()._subparsers._group_actions[0].choices["register"]
    assert register.get_default("stride") == lattice
    assert register.get_default("tile") == SPEC["method_grids_px"]["stare_tile"]
    # and the generator really makes bumps of that width
    spec = synthetic.truth_suite()["blob_bumps25_shift"]
    assert spec["bump_sigma"] == 25.0 and spec["bump_count"] == 4000
    # a reduced copy keeps the bumps' DENSITY, not their number
    small = synthetic.truth_suite(n=1024)
    assert small["blob_bumps25_shift"]["bump_count"] == round(4000 / 64)
    assert small["blob_bumps100_shift"]["bump_count"] == round(120 / 64)


def test_a_bump_is_the_same_in_any_batch():
    """The image is made a tile at a time and a probe is asked for alone: both must see
    the same field, to the last bit, and it must be the plain sum of the Gaussians."""
    spec = synthetic.truth_suite()["blob_bumps25_rot"]
    w = synthetic.extra_field(spec)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, spec["n"], (2, 400))
    whole = w(x, y)
    for i in (0, 17, 399):
        one = w(x[i : i + 1], y[i : i + 1])
        assert one[0][0] == whole[0][i] and one[1][0] == whole[1][i]
    tile = w(*np.meshgrid(np.arange(3000.0, 3040), np.arange(5000.0, 5030)))
    assert tile[0].shape == (30, 40)
    np.testing.assert_array_equal(
        tile[0][7, 9], w(np.array(3009.0), np.array(5007.0))[0]
    )
    # against the sum over every bump, written out
    seed = np.random.default_rng(spec["seed"] + 15485863)
    k, sg = spec["bump_count"], spec["bump_sigma"]
    cen = seed.uniform(0, spec["n"], (k, 2))
    peak = np.minimum(seed.uniform(0.2, 0.6, k) * sg, 50.0)
    ang = seed.uniform(0, 2 * math.pi, k)
    e = np.exp(
        -((x[:, None] - cen[:, 0]) ** 2 + (y[:, None] - cen[:, 1]) ** 2) / (2 * sg * sg)
    )
    np.testing.assert_allclose(whole[0], e @ (peak * np.cos(ang)), atol=1e-11)
    np.testing.assert_allclose(whole[1], e @ (peak * np.sin(ang)), atol=1e-11)
    assert np.isnan(w(np.array([np.nan, 5.0]), np.array([1.0, 5.0]))[0]).tolist() == [
        True,
        False,
    ]


def test_bumps_move_about_a_quarter_of_the_window_without_folding():
    """At the FROZEN size (8192 px, a lattice every 8 px): the share of the window a bump
    family moves by more than 1 um is the one the spec records, 20 to 30 %, and the map
    does not fold."""
    suite = synthetic.truth_suite()
    for name in ("bumps25", "bumps100"):
        fam = truth.family(SPEC, name)
        recorded = fam["measured_moved_frac_over_1um"]
        assert 0.20 <= recorded <= 0.30
        assert f"{100 * recorded:.0f} %" in fam["note"]
        for pose in truth.pose_names(SPEC):
            spec = suite[f"blob_{name}_{pose}"]
            got = synthetic.field_stats(spec, PX, step=8)
            assert abs(got["moved_frac_over_1um"] - recorded) < 0.02, (name, pose, got)
            assert got["min_jacobian"] > 0.05, (name, pose, got)
            # and the number the scorer itself would report for the true map
            xy, shape, step = methods.jacobian_grid((spec["n"], spec["n"]))
            jac = truth.jacobian_stats(
                {
                    "grid": synthetic.forward_map(spec)(xy),
                    "grid_shape": np.array(shape),
                    "grid_step": np.array(step),
                }
            )
            assert jac["fold_pct"] == 0.0
    # the old counts moved almost nothing: the test can tell
    sparse = {**suite["blob_bumps25_shift"], "bump_count": 60}
    assert synthetic.field_stats(sparse, PX, step=8)["moved_frac_over_1um"] < 0.01


def _jump_at(spec, x):
    g = synthetic.forward_map(spec)
    ys = np.arange(100.0, spec["n"], 700.0)
    left = g(np.stack([np.full_like(ys, x - 1e-6), ys], axis=1))
    right = g(np.stack([np.full_like(ys, x + 1e-6), ys], axis=1))
    return float(np.abs(right - left).max())


def test_seams_are_discontinuous():
    suite = synthetic.truth_suite()
    block = truth.family(SPEC, "seams")["block_px"]
    assert _jump_at(suite["blob_seams_rot"], block) > 0.5  # a jump at a block edge
    assert _jump_at(suite["blob_seams_rot"], block / 2) < 1e-4  # none inside a block
    assert _jump_at(suite["blob_grid_rot"], block) < 1e-4  # none in a smooth family


def test_no_seam_falls_on_a_method_grid():
    """Every seam POSITION inside the frozen window is checked against every grid a method
    works on: STARE's vector lattice and tile, ASHLAR's tile pitch, PALOM's blocks. A seam
    on a method's own grid line would be a jump that method can absorb for free."""
    block = truth.family(SPEC, "seams")["block_px"]
    grids = truth.method_grids(SPEC)
    assert grids == [
        ("stare_tile", 2048),
        ("stare_lattice", 64),
        ("ashlar_pitch", 922),
        ("palom_block", 1024),
        ("palom_block", 2048),
    ]
    seams = truth.seam_positions(SPEC)
    assert seams == [1950, 3900, 5850, 7800]
    assert truth.seams_on_a_grid(SPEC) == []
    for at in seams:
        for name, size in grids:
            assert at % size, (at, name, size)
    # the generator really jumps there
    spec = synthetic.truth_suite()["blob_seams_shift"]
    assert spec["seam_tile"] == block and all(_jump_at(spec, at) > 0 for at in seams)
    # the block this replaced was a multiple of STARE's lattice: the check can tell
    old = {**SPEC, "families": [dict(f) for f in SPEC["families"]]}
    truth.family(old, "seams")["block_px"] = 1536
    hits = truth.seams_on_a_grid(old)
    assert (1536, "stare_lattice", 64) in hits and (6144, "stare_tile", 2048) in hits
    assert (6144, "palom_block", 1024) in hits


def test_multiscale_scales_do_not_depend_on_window_size():
    want = tuple(s / PX for s in (333.0, 166.0, 83.0))
    assert truth.family_kwargs(SPEC, "multiscale", PX)["scales"] == pytest.approx(want)
    full = synthetic.truth_suite()["blob_multiscale_shift"]
    small = synthetic.truth_suite(n=1024)["blob_multiscale_shift"]
    assert full["n"] == 8192 and small["n"] == 1024
    assert full["scales"] == small["scales"] and full["rms"] == small["rms"]
    # at a pixel twice the size the same micrometres are half the pixels
    assert truth.family_kwargs(SPEC, "multiscale", 2 * PX)["scales"] == pytest.approx(
        tuple(s / 2 for s in want)
    )
    # the field's RMS displacement is the frozen 2 um
    xs = np.linspace(0, full["n"] - 1, 300)
    dx, dy = synthetic.extra_field(full)(*np.meshgrid(xs, xs))
    assert abs(math.sqrt(float(np.mean(dx**2 + dy**2))) * PX - 2.0) < 0.3


def test_a_synthetic_patient_is_one_family(reduced, real_like, tmp_path):
    suite = synthetic.truth_suite()
    assert len(suite) == 14
    patients = {}
    for name, spec in suite.items():
        assert spec["n"] == 8192 and name == f"{spec['patient']}_{spec['pose']}"
        patients.setdefault(spec["patient"], []).append(spec)
    assert len(patients) == 7
    for rounds in patients.values():
        assert len({s["family"] for s in rounds}) == 1
        assert [s["pose"] for s in rounds] == ["shift", "rot"]
        assert len({s["scene_seed"] for s in rounds}) == 1  # one tissue, one reference
        assert len({s["seed"] for s in rounds}) == 2  # its own map and noise per round
        assert [s["theta"] for s in rounds] == [0.0, 2.0]
    assert len({s["scene_seed"] for s in suite.values()}) == 7

    # as prepared: a patient is one family, its rounds share ONE reference file
    by_patient = groups(reduced[1])
    assert len(by_patient) == len(REDUCED)
    for patient, rounds in by_patient.items():
        assert [c.pose for c in rounds] == ["rot", "shift"]  # list_cases order
        assert {c.family for c in rounds} == {patient.split("_", 1)[1]}
        assert len({c.ref_nuclear for c in rounds}) == 1
        assert len({c.mov_nuclear for c in rounds}) == 2

    # semisynth: (window, family) is the patient; 2 windows x 7 families x 2 poses for
    # each cohort patient, whose id the spec no longer names
    assert "patients" not in SPEC["semisynth"]
    a = semisynth_args(real_like, tmp_path, name="046", truth=True)
    jobs, noise = semisynth.plan_truth(a, n=512)
    assert noise > 0 and len(jobs) == 28
    assert len({spec["patient"] for _, spec, _ in jobs}) == 14
    assert len({origin for _, _, origin in jobs}) == 2
    assert len({spec["seed"] for _, spec, _ in jobs}) == 28
    for name, spec, _ in jobs:
        assert name == f"{spec['patient']}_{spec['pose']}"
        assert spec["patient"].endswith(spec["family"])
        assert spec["truth_spec_sha256"] == ""  # a reduced copy is not the frozen suite
    assert SPEC["semisynth"]["windows_per_patient"] == 2
    assert semisynth.cases_per_patient(SPEC) == 28
    # another patient: other seeds, the same windows of the same image
    b = semisynth_args(real_like, tmp_path, name="052", truth=True)
    other, _ = semisynth.plan_truth(b, n=512)
    assert not {s["seed"] for _, s, _ in jobs} & {s["seed"] for _, s, _ in other}
    with pytest.raises(SystemExit, match="--name"):
        semisynth.plan_truth(semisynth_args(real_like, tmp_path, name=None), n=512)


def test_semisynth_rounds_share_the_untouched_window(real_like, tmp_path):
    import tifffile

    a = semisynth_args(real_like, tmp_path / "cases", name="052", truth=True)
    jobs, noise = semisynth.plan_truth(a, n=512)
    pair = [j for j in jobs if j[1]["patient"] == "052_w0_seams"]
    for name, spec, origin in pair:
        semisynth.prepare_one(a.cases, name, spec, a, origin, noise, 0.5)
    rounds = list_cases(a.cases, "semisynth")
    assert [c.pose for c in rounds] == ["rot", "shift"]
    assert {c.group for c in rounds} == {"052_w0_seams"}
    assert len({c.ref_nuclear for c in rounds}) == 1
    x, y = pair[0][2]
    whole = np.squeeze(tifffile.imread(real_like))
    got = np.squeeze(tifffile.imread(rounds[0].ref_nuclear))
    np.testing.assert_array_equal(got, whole[y : y + 512, x : x + 512])
    for case in rounds:
        pts = load_points(case)
        g = synthetic.forward_map(case.extra["spec"])
        np.testing.assert_allclose(g(pts["truth_mov"]), pts["truth_ref"], atol=1e-9)
        assert case.extra["variant"] == "same" and case.extra["in_stare_model"] is False


# ── a case whose true map folds is re-seeded (R3.11 e) ─────────────────────────
CONTINUOUS = ("rigid", "wave", "grid", "multiscale", "bumps100", "bumps25")
# Found by measuring 240 seeds of the frozen bumps25 field (8192 px, 4000 bumps): at this
# one the Jacobian determinant of the true map is negative at lattice points.
FOLDING_SEED = 3000044


def test_the_spec_says_which_cases_are_reseeded_and_how():
    rule = SPEC["reseed"]
    assert SPEC["version"] == 3
    assert rule["lattice_px"] == 4 and rule["min_jacobian"] == 0.05
    assert rule["max_tries"] >= 4
    # every family whose field is continuous; `seams` jumps on purpose and is left alone
    assert tuple(rule["families"]) == CONTINUOUS
    assert [f["name"] for f in SPEC["families"] if f["field"] != "seams"] == list(
        CONTINUOUS
    )
    for name in CONTINUOUS:
        assert truth.reseed_rule(SPEC, name) == {
            k: rule[k] for k in ("lattice_px", "min_jacobian", "step", "max_tries")
        }
        assert truth.case_extra(SPEC, name, True)["reseed"] == truth.reseed_rule(
            SPEC, name
        )
    assert truth.reseed_rule(SPEC, "seams") is None
    assert truth.case_extra(SPEC, "seams", True)["reseed"] is None
    # "the next seed" must be no case's asked seed: the step is above every seed a seed
    # rule can ask for, so seed + t * step lies outside them all, and two re-seeded cases
    # meet only if they asked for the same seed.
    semi = SPEC["semisynth"]
    assert "crc32(patient) % 100000" in semi["seed_rule"]
    largest = max(
        semi["seed"] + 1000 * 99_999 + 100 * 9 + 10 * 9 + 9,
        max(c["seed"] for c in synthetic.truth_suite().values()),
    )
    assert largest < 1.1e8 < rule["step"]


@pytest.mark.slow
def test_a_seed_that_folds_is_skipped_for_the_next_that_does_not():
    """The real thing, at the frozen size: a seed measured to fold is not used."""
    asked = {**synthetic.truth_suite()["blob_bumps25_shift"], "seed": FOLDING_SEED}
    assert synthetic.field_stats(asked, PX, step=4)["min_jacobian"] < 0  # it folds
    used, settled = synthetic.settle(asked, PX)
    step = SPEC["reseed"]["step"]
    assert used["seed"] == FOLDING_SEED + step == settled["seed"]
    assert settled["seed_asked"] == FOLDING_SEED
    (skipped,) = settled["skipped"]
    assert skipped["seed"] == FOLDING_SEED and skipped["min_jacobian"] < 0
    assert settled["min_jacobian"] > 0.05
    assert synthetic.field_stats(used, PX, step=4)["min_jacobian"] == pytest.approx(
        settled["min_jacobian"]
    )
    # nothing but the seed changed, and the asked spec was not edited in place
    assert {k for k in used if used[k] != asked[k]} == {"seed"}
    assert asked["seed"] == FOLDING_SEED
    # deterministic: the same question has the same answer
    assert synthetic.settle(asked, PX) == (used, settled)


def _fold_for(monkeypatch, bad_seeds):
    """Make the given seeds measure as folding (and every other one as fine)."""
    seen = []

    def stats(spec, pixel_size_um, step=8, threshold_um=1.0):
        seen.append((spec["seed"], step))
        low = -0.2 if spec["seed"] in bad_seeds else 0.4
        return {
            "moved_frac_over_1um": 0.2,
            "min_jacobian": low,
            "fold_pct": 1.0 if low <= 0 else 0.0,
        }

    monkeypatch.setattr(synthetic, "field_stats", stats)
    return seen


def test_the_reseed_rule_step_by_step(monkeypatch):
    rule = truth.reseed_rule(SPEC, "bumps25")
    step = rule["step"]
    job = {**synthetic.truth_suite(n=1024)["blob_bumps25_rot"]}
    seed = job["seed"]
    # two folding seeds in a row: both are skipped, in order, and recorded
    seen = _fold_for(monkeypatch, {seed, seed + step})
    used, settled = synthetic.settle(job, PX)
    assert used["seed"] == seed + 2 * step
    assert [s["seed"] for s in settled["skipped"]] == [seed, seed + step]
    assert seen == [(seed, 4), (seed + step, 4), (seed + 2 * step, 4)]
    assert settled == {
        "seed_asked": seed,
        "seed": seed + 2 * step,
        "skipped": [
            {"seed": seed, "min_jacobian": -0.2},
            {"seed": seed + step, "min_jacobian": -0.2},
        ],
        "step": step,
        "lattice_px": 4,
        "min_jacobian": 0.4,
        "min_jacobian_required": 0.05,
    }
    # a seed that does not fold is used as asked, after ONE measurement
    seen.clear()
    used, settled = synthetic.settle({**job, "seed": seed + 7}, PX)
    assert used["seed"] == seed + 7 and settled["skipped"] == [] and len(seen) == 1
    # a determinant that is positive but under the margin counts as folding too
    monkeypatch.setattr(
        synthetic,
        "field_stats",
        lambda spec, px, step=8, threshold_um=1.0: {
            "moved_frac_over_1um": 0.2,
            "min_jacobian": 0.03 if spec["seed"] == seed else 0.4,
            "fold_pct": 0.0,
        },
    )
    assert synthetic.settle(job, PX)[0]["seed"] == seed + step
    # every try folds: refused, never a silently folded case
    _fold_for(monkeypatch, {seed + t * step for t in range(rule["max_tries"])})
    with pytest.raises(SystemExit, match="fold"):
        synthetic.settle(job, PX)
    # a spec without the rule (the copied suites, `seams`) is not measured or changed
    seen = _fold_for(monkeypatch, {5})
    plain = {**synthetic._BASE, "family": "t", "seed": 5}
    assert synthetic.settle(plain, PX) == (plain, None)
    seams = synthetic.truth_suite(n=1024)["blob_seams_rot"]
    assert synthetic.settle(seams, PX) == (seams, None)
    assert seen == []


@pytest.mark.slow
def test_no_frozen_synthetic_case_folds():
    """THE guard: every frozen synthetic known-truth case, at the frozen size, is made
    with a seed whose true map does not fold. `SKIPPED` is written out so that a spec
    change which starts skipping a seed is seen here."""
    SKIPPED = {}  # case id -> the seeds skipped; none in the shipped spec
    suite = synthetic.truth_suite()
    assert len(suite) == 14
    got = {}
    for name, job in suite.items():
        used, settled = synthetic.settle(job, PX)
        if job["family"] == "seams":
            assert settled is None and used == job
            continue
        assert settled["min_jacobian"] > SPEC["reseed"]["min_jacobian"], name
        assert synthetic.field_stats(used, PX, step=8)["fold_pct"] == 0.0, name
        if settled["skipped"]:
            got[name] = [s["seed"] for s in settled["skipped"]]
    assert got == SKIPPED


def test_a_prepared_case_is_made_with_the_settled_seed(monkeypatch, tmp_path):
    """End to end, both datasets: the slide, the probes and case.json all come from the
    seed that does not fold, and the skipped seed is in the case's metadata."""
    step = SPEC["reseed"]["step"]
    name, job = (
        "blob_bumps25_shift",
        synthetic.truth_suite(n=1024)["blob_bumps25_shift"],
    )
    asked = job["seed"]
    real = synthetic.field_stats

    def stats(spec, pixel_size_um, step=8, threshold_um=1.0):
        got = real(spec, pixel_size_um, step=step, threshold_um=threshold_um)
        if spec["seed"] == asked:
            got = {**got, "min_jacobian": -0.01, "fold_pct": 0.1}
        return got

    monkeypatch.setattr(synthetic, "field_stats", stats)
    case = load_case(synthetic.prepare_one(tmp_path / "cases", name, job))
    settled = case.extra["reseed"]
    assert settled["seed_asked"] == asked and settled["seed"] == asked + step
    assert [s["seed"] for s in settled["skipped"]] == [asked]
    assert case.extra["spec"]["seed"] == asked + step
    assert case.extra["field"]["fold_pct"] == 0.0  # measured on the map really used
    pts = load_points(case)
    g = synthetic.forward_map(case.extra["spec"])
    np.testing.assert_allclose(g(pts["truth_mov"]), pts["truth_ref"], atol=1e-9)
    wrong = synthetic.forward_map(job)(pts["truth_mov"])
    assert np.abs(wrong - pts["truth_ref"]).max() > 1.0  # not the seed asked for
    assert job["seed"] == asked  # the plan is not edited


def test_a_semisynthetic_case_is_reseeded_too(monkeypatch, real_like, tmp_path):
    step = SPEC["reseed"]["step"]
    a = semisynth_args(real_like, tmp_path / "cases", name="052", truth=True)
    jobs, noise = semisynth.plan_truth(a, n=512)
    (name, job, origin), (other, job2, origin2) = [
        j for j in jobs if j[1]["patient"] == "052_w0_bumps25"
    ]
    assert job["reseed"] == truth.reseed_rule(SPEC, "bumps25")
    _fold_for(monkeypatch, {job["seed"]})
    case = load_case(semisynth.prepare_one(a.cases, name, job, a, origin, noise, 0.5))
    assert case.extra["reseed"]["seed"] == job["seed"] + step
    assert [s["seed"] for s in case.extra["reseed"]["skipped"]] == [job["seed"]]
    assert case.extra["spec"]["seed"] == job["seed"] + step
    pts = load_points(case)
    g = synthetic.forward_map(case.extra["spec"])
    np.testing.assert_allclose(g(pts["truth_mov"]), pts["truth_ref"], atol=1e-9)
    # the patient's other round keeps the seed it asked for
    kept = load_case(
        semisynth.prepare_one(a.cases, other, job2, a, origin2, noise, 0.5)
    )
    assert kept.extra["reseed"]["skipped"] == []
    assert kept.extra["spec"]["seed"] == job2["seed"]
    assert case.ref_nuclear == kept.ref_nuclear  # still one reference for the patient
    # `seams` is left alone
    (sname, sjob, sorigin) = [j for j in jobs if j[1]["patient"] == "052_w0_seams"][0]
    assert sjob["reseed"] is None
    seams = load_case(
        semisynth.prepare_one(a.cases, sname, sjob, a, sorigin, noise, 0.5)
    )
    assert "reseed" not in seams.extra


@pytest.mark.slow
def test_reseeded_cases_are_named_in_the_summary(reduced, tmp_path):
    src = [c for c in reduced[1] if c.family == "bumps25"][0]
    root, out = tmp_path / "cases", tmp_path / "out"
    _hashed_copy(src, root, "asked", "a" * 64)
    score(root, out, ["synthetic"])
    assert "re-seeded" not in (out / "tables" / "summary.md").read_text()
    dst = root / "synthetic" / "moved"
    shutil.copytree(src.dir, dst)
    rec = json.loads((dst / "case.json").read_text())
    rec["case_id"] = "moved"
    rec["extra"]["truth_spec_sha256"] = "a" * 64
    rec["extra"]["reseed"]["skipped"] = [{"seed": 1, "min_jacobian": -0.1}]
    (dst / "case.json").write_text(json.dumps(rec))
    score(root, out, ["synthetic"])
    text = (out / "tables" / "summary.md").read_text()
    assert "1 case(s) were re-seeded" in text and "synthetic/moved (1 seed)" in text


# ── the freeze ────────────────────────────────────────────────────────────────
def _hashed_copy(src, root, case_id, sha):
    dst = root / "synthetic" / case_id
    shutil.copytree(src.dir, dst)
    rec = json.loads((dst / "case.json").read_text())
    rec["case_id"] = case_id
    rec["extra"]["truth_spec_sha256"] = sha
    (dst / "case.json").write_text(json.dumps(rec))
    return load_case(dst)


@pytest.mark.slow
def test_spec_hash_is_enforced(reduced, tmp_path):
    assert SPEC["sha256"] == truth.spec_hash(SPEC)
    changed = json.loads(truth.SPEC_PATH.read_text())
    changed["families"][2]["sd_px"] = 6.0  # an amplitude edited after the freeze
    bad = tmp_path / "truth_spec.json"
    bad.write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="frozen"):
        truth.load_spec(bad)
    changed["sha256"] = truth.spec_hash(changed)
    bad.write_text(json.dumps(changed))
    assert truth.load_spec(bad)["sha256"] != SPEC["sha256"]

    # the frozen suite's cases carry the hash; a reduced copy does not
    assert {s["truth_spec_sha256"] for s in synthetic.truth_suite().values()} == {
        SPEC["sha256"]
    }
    assert {s["truth_spec_sha256"] for s in synthetic.truth_suite(n=1024).values()} == {
        ""
    }

    # prepare writes it into case.json (a 1024 px spec, hashed as its own, stands in)
    small = {**SPEC, "n_px": 1024}
    small["sha256"] = truth.spec_hash(small)
    name, spec = next(iter(synthetic.truth_suite(spec=small).items()))
    made = load_case(synthetic.prepare_one(tmp_path / "made", name, spec))
    assert made.extra["truth_spec_sha256"] == small["sha256"] != SPEC["sha256"]

    # score refuses cases of two hashes, and takes cases of one
    src = reduced[1][0]
    root = tmp_path / "cases"
    _hashed_copy(src, root, "a", "a" * 64)
    b = _hashed_copy(src, root, "b", "a" * 64)
    out = tmp_path / "out"
    methods.run_case(exact(), b, out)
    res = score(root, out, ["synthetic"])
    assert {r["truth_spec"] for r in res["truth"]} == {"a" * 64}
    # frozen known-truth cases are left out of the cost tables, not out of the failures
    assert res["resources"] == [] and res["summary_resources"] == []
    assert not (out / "tables" / "resources.csv").exists()
    (fails,) = res["failures"]
    assert fails["method"] == "exact" and fails["rounds_not_run"] == 1
    _hashed_copy(src, root, "c", "b" * 64)
    with pytest.raises(SystemExit, match="more than one truth_spec.json"):
        score(root, out, ["synthetic"])


@pytest.mark.slow
def test_a_case_whose_true_map_folds_is_named_in_the_summary(reduced, tmp_path):
    src = reduced[1][0]
    root, out = tmp_path / "cases", tmp_path / "out"
    _hashed_copy(src, root, "fine", "a" * 64)
    score(root, out, ["synthetic"])
    text = (out / "tables" / "summary.md").read_text()
    assert "sha256 `" + "a" * 64 in text and "folds somewhere" not in text
    dst = root / "synthetic" / "bent"
    shutil.copytree(src.dir, dst)
    rec = json.loads((dst / "case.json").read_text())
    rec["case_id"] = "bent"
    rec["extra"]["truth_spec_sha256"] = "a" * 64
    rec["extra"]["field"]["fold_pct"] = 0.001
    (dst / "case.json").write_text(json.dumps(rec))
    score(root, out, ["synthetic"])
    text = (out / "tables" / "summary.md").read_text()
    assert "1 case(s) were made by a true map that folds somewhere" in text
    assert "synthetic/bent." in text


def test_folding_is_reported():
    xy, shape, step = methods.jacobian_grid((300, 400))

    def pack(warped):
        return {
            "grid": warped,
            "grid_shape": np.array(shape),
            "grid_step": np.array(step),
        }

    smooth = truth.jacobian_stats(pack(xy * 1.1 + 3.0))
    assert smooth["fold_pct"] == 0.0 and smooth["sd_log_jac"] < 1e-6
    folded = xy.copy()
    left = xy[:, 0] < 200
    folded[left, 0] = 200 - xy[left, 0]  # the left half is mirrored: it folds
    assert 40 < truth.jacobian_stats(pack(folded))["fold_pct"] < 60
    assert math.isnan(truth.jacobian_stats(None)["fold_pct"])


# ── a row dropped by its own failed selfcheck (R3.12 b) ────────────────────────
@pytest.mark.slow
def test_a_dropped_variant_is_in_no_table_and_is_named(scored, reduced, tmp_path):
    out, _ = scored
    root, _cases = reduced
    kept = score(root, out, ["synthetic"], expected=["ghost", "backward"])
    assert {"exact", "backward", "ghost"} <= {r["method"] for r in kept["cells"]}
    assert kept["dropped"] == []
    res = score(
        root, out, ["synthetic"], expected=["ghost", "backward"], dropped=["backward"]
    )
    for table in ("cells", "truth", "summary_cells", "failures", "truth_by_family"):
        methods_in = {r["method"].split(":")[0] for r in res[table]}
        assert "backward" not in methods_in, table
        assert "exact" in methods_in or table == "failures", table
    # not resurrected as a "not run" row by the expected list, and reported by name
    assert res["dropped"] == [
        {
            "variant": "backward",
            "reason": "its selfcheck failed and it does not gate its method",
            "had_results": True,
        }
    ]
    tables = out / "tables"
    assert "backward" in (tables / "dropped.csv").read_text()
    text = (tables / "summary.md").read_text()
    assert "## Rows left out" in text and "`backward`" in text
    # a dropped name that never ran is still reported, as having no results
    res = score(root, out, ["synthetic"], dropped=["never_ran"])
    assert res["dropped"][0]["had_results"] is False
    # without the option there is no such list (and the fixture's tables are restored)
    score(root, out, ["synthetic"], expected=["ghost"])
    assert not (tables / "dropped.csv").exists()
    assert "## Rows left out" not in (tables / "summary.md").read_text()


def test_score_drop_on_the_command_line(monkeypatch):
    from benchmarks.regbench import cli
    from benchmarks.regbench import score as score_module

    seen = {}
    monkeypatch.setattr(
        score_module, "score", lambda cases, out, *a, **kw: seen.update(kw) or {}
    )
    argv = ["score", "--cases", "c", "--out", "o", "--drop", "deeperhistreg_hr_cpu"]
    assert cli.main([*argv, "--drop", "deeperhistreg_cpu"]) == 0
    assert seen["dropped"] == ["deeperhistreg_hr_cpu", "deeperhistreg_cpu"]
    assert "deeperhistreg_hr_cpu" in seen["expected"]  # the scorer filters, not the CLI
    # a name that is no result name of the run list is a typo: refused
    with pytest.raises(SystemExit, match="no_such_row"):
        cli.main([*argv, "--drop", "no_such_row"])
    # and a gating row cannot be dropped: its failure blocks the method instead
    with pytest.raises(SystemExit, match="gates"):
        cli.main(["score", "--cases", "c", "--out", "o", "--drop", "palom"])
    seen.clear()
    assert cli.main(["score", "--cases", "c", "--out", "o"]) == 0
    assert seen["dropped"] == []
