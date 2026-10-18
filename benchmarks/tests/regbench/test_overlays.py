# benchmarks/tests/regbench/test_overlays.py
"""The overlay zooms (``benchmarks.regbench.paper.overlays``): the field rule, the
residuals in each variant's own frame, the rigid variant read from the run list, the
lattice map and its inverse, and the three figures drawn on a toy case with crops read by
region only."""
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from benchmarks.regbench import methods, pairs  # noqa: E402
from benchmarks.regbench.cases import load_case, result_dir, write_case  # noqa: E402
from benchmarks.regbench.imageio import write_nuclear  # noqa: E402
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap  # noqa: E402
from benchmarks.regbench.paper import overlays, schema  # noqa: E402

from . import scored_cohort  # noqa: E402


def test_pick_roi_finds_the_field_where_each_step_is_needed():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1000, size=(4000, 2))
    d = {"initial": np.full(4000, 9.0), "rigid": np.full(4000, 1.0), "final": np.full(4000, 0.5)}
    hot = (xy[:, 0] > 600) & (xy[:, 0] < 800) & (xy[:, 1] > 200) & (xy[:, 1] < 400)
    d["rigid"][hot] = 6.0  # rigid leaves a large residual here; non-rigid removes it
    roi = overlays.pick_roi(xy, d, roi_px=200, step_px=100)
    assert roi["monotone"] and 500 <= roi["roi_x"] <= 700 and 100 <= roi["roi_y"] <= 300
    assert roi["d_rigid_um"] - roi["d_final_um"] > 4
    assert "d_initial > d_rigid > d_final" in roi["rule"]


def test_no_monotone_window_is_flagged():
    xy = np.random.default_rng(2).uniform(0, 500, size=(2000, 2))
    d = {"initial": np.ones(2000), "rigid": np.ones(2000) * 2, "final": np.ones(2000) * 3}
    roi = overlays.pick_roi(xy, d, roi_px=100, step_px=50)
    assert roi["monotone"] is False and "no window is monotone" in roi["rule"]


def test_overlay_residuals_use_each_variants_frame(tmp_path):
    """A ref_to_mov row moves REFERENCE nuclei: its residual is measured against native
    moving centroids, and equals zero when its map is the true one."""
    root, out, made = scored_cohort.build(tmp_path)
    case = made[0]
    d = overlays.residuals(case, out, "rtm", {"initial": "initial", "rigid": "rtm", "final": "rtm"})
    assert set(d) == {"initial", "rigid", "final"} and len(d["final"]) > 0


def _exact_case(root):
    """A scored_cohort-shaped case with NO noise: the moving centres are the reference
    centres minus SHIFT exactly, so ``forward(.., 0)`` / ``backward(.., 0)`` are the truth."""
    ref = scored_cohort.GRID.astype(float)
    mov = ref - scored_cohort.SHIFT
    write_case(
        root,
        scored_cohort._case("multiplex", "X1_r1", "X1"),
        cells_mov=scored_cohort.squares(mov, scored_cohort.OUTLINE_HALF),
        cells_ref=scored_cohort.squares(ref, scored_cohort.OUTLINE_HALF),
    )
    return load_case(Path(root) / "multiplex" / "X1_r1")


def test_overlay_residuals_are_zero_for_the_true_map_in_both_directions(tmp_path):
    case = _exact_case(tmp_path / "cases")
    out = tmp_path / "out"
    methods.run_case(scored_cohort.forward("truth_mtr", 0.0), case, out)
    methods.run_case(scored_cohort.backward("truth_rtm", 0.0), case, out)
    methods.run_case(scored_cohort.forward("off_mtr", 4.0), case, out)
    n_ref = len(scored_cohort.GRID)
    gate_um = 1.5 * math.sqrt((2 * scored_cohort.OUTLINE_HALF) ** 2 / math.pi) * 0.5
    assert abs(gate_um - pairs.GATE_FACTOR * 6.770275 * 0.5) < 1e-3
    for row in ("truth_mtr", "truth_rtm"):
        d = overlays.residuals(case, out, row, {"initial": "initial", "rigid": row, "final": row})
        assert all(len(v) == n_ref for v in d.values())
        # the true map: every reference nucleus has its partner on top of it
        assert np.nanmax(d["final"]) < 1e-3 and np.nanmax(d["rigid"]) < 1e-3, row
        # before registration the shift (15 px) is past the gate: capped at the gate
        assert np.allclose(d["initial"], gate_um, atol=1e-6)
    # a wrong map is not zero: the residual sees the error
    d = overlays.residuals(case, out, "off_mtr", {"initial": "initial", "rigid": None,
                                                  "final": "off_mtr"})
    assert np.nanmedian(d["final"]) > 0.1 and np.isnan(d["rigid"]).all()


def test_rigid_variant_is_read_from_the_run_list():
    assert overlays.rigid_of("stare") == "stare_rigid"
    assert overlays.rigid_of("stare_medium") == "stare_medium_rigid"
    assert overlays.rigid_of("valis_mirage") == "valis_mirage_rigid"
    assert overlays.rigid_of("palom") == "palom_affine"
    assert overlays.rigid_of("ashlar") is None
    assert overlays.rigid_of("deeperhistreg_hr_gpu") is None
    assert overlays.rigid_title("valis_mirage") == "after rigid (+ micro-rigid)"
    assert overlays.rigid_title("valis_mirage_high_m1") == "after rigid (+ micro-rigid)"
    assert overlays.rigid_title("valis_mirage_high_m0") == "after rigid"
    assert overlays.rigid_title("palom") == "after rigid"
    assert overlays.rigid_title("ashlar") == "no rigid stage"
    assert overlays.rigid_title("deeperhistreg_hr_gpu") == "no rigid stage"


def _affine_lattice(hw, a, t):
    xy, shape, step = methods.jacobian_grid(hw)
    grid = xy @ np.asarray(a).T + np.asarray(t)
    return {"grid": grid.astype(np.float32), "grid_shape": np.array(shape),
            "grid_step": np.array(step)}


def test_the_lattice_is_exact_for_an_affine_map_and_inverts_to_a_thousandth_px():
    th = math.radians(3.0)
    a = 1.02 * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.array([17.0, -11.0])
    lat = overlays.Lattice(_affine_lattice((900, 1000), a, t))
    pts = np.random.default_rng(3).uniform(-50, 1050, size=(500, 2))
    assert np.abs(lat(pts) - (pts @ a.T + t)).max() < 1e-2  # float32 lattice values
    back = lat.invert(lat(pts))
    assert np.abs(back - pts).max() < 1e-3 * 5


# ── rendering on a toy case ──────────────────────────────────────────────────
def _plane(path):
    import tifffile

    return np.squeeze(tifffile.imread(str(path)))


def _invert(g, xy, n=60):
    xy = np.asarray(xy, dtype=float)
    m = xy.copy()
    for _ in range(n):
        m = m + (xy - g(m))
    return m


def _stand_in(name, names, direction, writes=True, origin=None):
    """A method of result names ``names`` whose every map is the case's TRUE map in
    ``direction``; at the last name it writes an image (the reference plane, placed at
    ``origin``) as its own writer would."""
    from benchmarks.regbench import scene

    def register(case, work, opts):
        g = scene.forward_map(case.extra["spec"])  # moving -> reference
        fn = g if direction == MOV_TO_REF else (lambda xy: _invert(g, xy))
        for v in names:
            info = {}
            if v == names[-1] and writes:
                dest = Path(opts["native_dir"](case.case_id, v))
                dest.mkdir(parents=True, exist_ok=True)
                path = write_nuclear(dest / "written.ome.tif", _plane(case.ref_nuclear),
                                     case.pixel_size_um)
                info["native"] = {"files": [str(path)], "channels": 1}
                if origin is not None:
                    info["native"]["origin_xy"] = list(origin)
            yield v, PointMap(fn, direction), info

    return types.SimpleNamespace(NAME=name, SCOPE="pair", VARIANTS=tuple(names),
                                 register=register)


def test_the_three_figures_are_drawn_and_crops_are_read_by_region(
    tmp_path, toy_cases, cases_root, monkeypatch
):
    case = toy_cases["rigid"]
    out = tmp_path / "out"
    rows = {
        "stare": _stand_in("stare", ("stare_rigid", "stare"), MOV_TO_REF),
        "valis_mirage": _stand_in(
            "valis_mirage",
            ("valis_mirage_rigid", "valis_mirage_nonrigid", "valis_mirage"),
            REF_TO_MOV,
        ),
        # PALOM's written image is missing: its panel must say so, never vanish
        "palom": _stand_in("palom", ("palom_affine", "palom"), REF_TO_MOV, writes=False),
        "deeperhistreg_hr_gpu": _stand_in(
            "deeperhistreg_hr_gpu", ("deeperhistreg_hr_gpu",), REF_TO_MOV
        ),
        "ashlar": _stand_in("ashlar", ("ashlar",), MOV_TO_REF, origin=(6.0, -4.0)),
    }
    for mod in rows.values():
        methods.run_case(mod, case, out)
    paper = out / "paper"
    paper.mkdir(parents=True)
    schema.write(paper / "method_best.csv", "method_best", [
        schema.empty_row("method_best", source="regbench", dataset="multiplex",
                         family=fam, best_row=row)
        for fam, row in (("stare", "stare"), ("valis_mirage", "valis_mirage"),
                         ("palom", "palom"), ("deeperhistreg", "deeperhistreg_hr_gpu"),
                         ("ashlar", "ashlar"))
    ])
    boxes = []
    real = overlays.read_region

    def recording(path, channel, y0, y1, x0, x1):
        boxes.append((str(path), y1 - y0, x1 - x0))
        return real(path, channel, y0, y1, x0, x1)

    monkeypatch.setattr(overlays, "read_region", recording)
    # only the toy case: the session's other toy cases are not candidates
    got = overlays.build(cases_root, out, roi_um=120.0, case_id="rigid")
    roi_px = int(round(120.0 / case.pixel_size_um))
    fig_dir = paper / "figures"
    for name in overlays.OVERLAYS:
        for ext in ("png", "pdf", "svg"):
            assert (fig_dir / f"{name}.{ext}").is_file(), (name, ext)
        assert (fig_dir / f"{name}_source_data.csv").is_file()
        assert not got[name].get("placeholder")
    assert boxes, "no crop was read"
    limit = overlays.read_limit(roi_px)
    assert limit <= 1.5 * roi_px + 2 * overlays.READ_PAD_PX + 1
    for path, h, w in boxes:
        assert h <= limit and w <= limit, (path, h, w, limit)
    # every row has its own three medians in the one field; PALOM's missing image is named
    rows_csv = schema.read(paper / "overlay_rois.csv")
    drawn = {r["row"]: r for r in rows_csv if r["case_id"] == "rigid"}
    assert set(drawn) >= {"stare", "valis_mirage", "palom", "deeperhistreg_hr_gpu", "ashlar"}
    assert len({(r["roi_x"], r["roi_y"], r["roi_px"]) for r in drawn.values()}) == 1
    assert drawn["ashlar"]["d_rigid_um"] is None  # no rigid stage: NA, never a number
    assert float(drawn["stare"]["d_final_um"]) < float(drawn["stare"]["d_initial_um"])
    src = (fig_dir / "fig3a_overlays_source_data.csv").read_text()
    assert "no image written" in src and "no rigid stage" in src
    run = json.loads((result_dir(out, "palom", case) / "run.json").read_text())
    assert not (run.get("native") or {}).get("files")


def test_without_the_best_rows_every_figure_is_a_placeholder(tmp_path, cases_root):
    got = overlays.build(cases_root, tmp_path / "out", roi_um=60.0)
    for name in overlays.OVERLAYS:
        assert got[name]["placeholder"]
        assert (tmp_path / "out" / "paper" / "figures" / f"{name}.png").is_file()
