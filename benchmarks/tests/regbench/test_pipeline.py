"""End to end on synthetic slides with a known map: the truth is self-consistent, STARE is
really run and scored, a failed method is imputed, and resources are measured."""

import csv
import json

import numpy as np
import pytest
from regbench import cells, methods
from regbench.cases import load_points, polys, result_dir
from regbench.datasets import synthetic
from regbench.score import score


def test_the_true_map_puts_moving_nuclei_on_reference_nuclei(synthetic_cases):
    """Guards the truth's DIRECTION: nuclei segmented on the moving slide, pushed through the
    generator's forward map, land on the nuclei segmented on the reference."""
    for name, case in synthetic_cases.items():
        pts = load_points(case)
        g = synthetic.forward_map(case.extra["spec"])
        ref, mov = polys(pts, "cells_ref"), polys(pts, "cells_mov")
        before = cells.score_cells(ref, mov)
        after = cells.score_cells(ref, mov.with_xy(g(mov.xy)))
        assert after["displacement_px_p50"] < 0.5, name
        assert after["dice_matched"] > 0.9, name
        assert after["displacement_px_p50"] < before["displacement_px_p50"], name
        assert np.allclose(g(pts["landmarks_mov"]), pts["landmarks_ref"])


@pytest.fixture(scope="module")
def scored(cases_root, synthetic_cases, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    for case in synthetic_cases.values():
        assert methods.run_case("stare", case, out, workers=2) == ["stare_rigid", "stare"]
    return out, score(cases_root, out, ["synthetic"])


def _row(rows, case_id, method):
    (r,) = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
    return r


def test_stare_registers_rigid_and_smooth_cases_to_subpixel_landmark_error(scored):
    _, res = scored
    for name in ("rigid", "smooth"):
        init, full = _row(res["landmarks"], name, "initial"), _row(res["landmarks"], name, "stare")
        assert init["tre_px_median"] > 3.0
        assert full["tre_px_median"] < 1.0, name
        assert full["robustness"] > 0.95
        assert full["rank"] == 1.0
        assert not full["imputed_initial"]


@pytest.mark.xfail(strict=True, reason=(
    "STARE 1.2.1: REG-TILE emits full-weight vectors for lattice windows the moving slide "
    "does not cover (here the bottom three rows: residuals 7-35 px against ~3 px elsewhere). "
    "SOLVE selects s by block CV before any outlier rejection; no s predicts those vectors, "
    "a flat field predicts them least badly, so s = 1e6 is chosen for the whole slide and "
    "the mesh adds nothing (~2.5 px, the rigid anchor). 4 of 6 seeds at 1024 px, 2 of 6 at "
    "2048 px. Remove the xfail when uncovered windows are rejected."))
def test_stare_refines_a_moving_slide_that_does_not_cover_the_reference(scored):
    _, res = scored
    assert _row(res["landmarks"], "crop", "stare")["tre_px_median"] < 1.0


@pytest.mark.xfail(strict=True, reason=(
    "STARE 1.2.1, the same selection on a full-size 1024 px pair: held-out 3x3 patches on "
    "the lattice border must be extrapolated, a flexible fit extrapolates worse than a flat "
    "one, and on a 15 x 15 lattice 44 % of the nodes are border, so block CV can prefer "
    "s = 1e6 although the interior is predicted twice as well by a small s. 2 of 6 seeds at "
    "1024 px, 0 of 14 at 2048 px and above. Remove the xfail when the selection is fixed."))
def test_stare_refines_a_small_slide_whatever_the_seed(scored):
    _, res = scored
    assert _row(res["landmarks"], "smooth_border", "stare")["tre_px_median"] < 1.0


def test_stare_still_anchors_a_moving_slide_of_another_shape(scored):
    _, res = scored
    init, rigid = _row(res["landmarks"], "crop", "initial"), _row(res["landmarks"], "crop", "stare_rigid")
    assert rigid["tre_px_median"] < 3.0 < init["tre_px_median"]
    assert rigid["robustness"] > 0.95


def test_the_mesh_is_what_fixes_a_smooth_deformation(scored):
    _, res = scored
    rigid, full = _row(res["landmarks"], "smooth", "stare_rigid"), _row(res["landmarks"], "smooth", "stare")
    assert rigid["tre_px_median"] > 2.0
    assert full["tre_px_median"] < 1.0


def test_cell_metrics_agree_with_the_landmarks(scored):
    _, res = scored
    for name in ("rigid", "smooth"):
        init, full = _row(res["cells"], name, "initial"), _row(res["cells"], name, "stare")
        assert full["dice_matched"] > 0.85 > init["dice_matched"]
        assert full["displacement_px_p50"] < 1.0
        assert full["pair_fraction"] > 0.9


def test_resources_are_measured_for_the_whole_run(scored):
    out, res = scored
    r = _row(res["resources"], "smooth", "stare")
    assert r["ok"] and r["wall_min"] > 0 and r["cpu_min"] > 0
    assert r["megapixels"] == pytest.approx(2 * 1024 * 1024 / 1e6)
    assert r["wall_s_per_mpx"] == pytest.approx(r["wall_min"] * 60 / r["megapixels"])
    if r["rss_method"] != "rusage":
        assert r["peak_rss_gb"] > 0.05  # at least the interpreter and one slide
    (agg,) = [a for a in res["resources_agg"] if a["method"] == "stare"]
    assert agg["n_runs"] == 4 and agg["n_failed"] == 0
    tables = out / "tables"
    for name in ("landmarks_cases", "landmarks_aggregates", "cells_cases", "cells_aggregates",
                 "resources_cases", "resources_aggregates"):
        with open(tables / f"{name}.csv", newline="") as fh:
            assert list(csv.DictReader(fh))
    assert "## Resources" in (tables / "summary.md").read_text()


def test_a_failed_method_is_scored_at_the_initial_pose(cases_root, synthetic_cases, tmp_path,
                                                       monkeypatch):
    def boom(case, work, opts):
        raise RuntimeError("no transform")
        yield  # pragma: no cover

    monkeypatch.setattr(methods.load("stare"), "register", boom)
    case = synthetic_cases["rigid"]
    assert methods.run_case("stare", case, tmp_path) == []
    run = json.loads((result_dir(tmp_path, "stare", case) / "run.json").read_text())
    assert run["ok"] is False and "no transform" in run["error"]

    res = score(cases_root, tmp_path, ["synthetic"])
    init, failed = _row(res["landmarks"], "rigid", "initial"), _row(res["landmarks"], "rigid", "stare")
    assert failed["imputed_initial"] and failed["tre_px_median"] == init["tre_px_median"]
    assert failed["robustness"] == 0.0
    (agg,) = [a for a in res["landmarks_agg"] if a["method"] == "stare" and a["subset"] == "all"]
    # the three cases it was never run on count at the initial pose too, as a missing
    # submission does in the challenge; with nothing registered there is no common subset
    assert agg["n_imputed"] == 4
    assert not [a for a in res["landmarks_agg"] if a["subset"] == "common"]


def test_label_renames_a_configuration(synthetic_cases, tmp_path):
    case = synthetic_cases["rigid"]
    assert methods.variant_names("valis", "valis_mr") == ["valis_mr_rigid", "valis_mr", "valis_mr_micro"]
    done = methods.run_case("initial", case, tmp_path, label="baseline")
    assert done == ["baseline"] and (result_dir(tmp_path, "baseline", case) / "warped.npz").exists()
