"""Real images under a known map: the copied semi-synthetic suite (ported from
sceriff0/stare @ f5ce62e, ``benchmarks/regbench/tests/test_extensions.py``)."""

import numpy as np

from benchmarks.regbench import score
from benchmarks.regbench.cases import load_case, load_points
from benchmarks.regbench.datasets import semisynth, synthetic

from .conftest import semisynth_args


def test_semisynth_cases_carry_the_truth(real_like, tmp_path):
    a = semisynth_args(real_like, tmp_path / "cases")
    jobs, noise = semisynth.plan(a)
    assert noise > 0
    names = [j[0] for j in jobs]
    assert sum(n.startswith("dev_") for n in names) == len(semisynth.FAMILIES)
    assert len(names) == 3 * len(semisynth.FAMILIES)
    dev_origin = {j[2] for j in jobs if j[0].startswith("dev_")}
    assert not dev_origin & {j[2] for j in jobs if not j[0].startswith("dev_")}
    assert len({j[1]["seed"] for j in jobs}) == len(jobs)

    name, spec, origin = next(j for j in jobs if j[0].endswith("w0_multiscale"))
    semisynth.prepare_one(a.cases, name, spec, a, origin, noise, 0.5)
    case = load_case(tmp_path / "cases" / "semisynth" / name)
    assert case.pixel_size_um == 0.5 and case.extra["variant"] == "same"
    assert case.group == name and case.family == "multiscale" and case.pose == ""
    pts = load_points(case)
    assert len(pts["truth_mov"]) > 50
    np.testing.assert_allclose(
        synthetic.forward_map(spec)(pts["truth_mov"]), pts["truth_ref"]
    )
    done = score.score(a.cases, tmp_path / "out", ["semisynth"])
    by = {r["method"]: r for r in done["cells"]}
    assert by["truth"]["dice_matched"] > 0.8 > by["initial"]["dice_matched"]
    assert by["truth"]["displacement_px_p50"] < 1.0
    # a real-image case under a literature-derived map is in the headline tier
    assert {r["tier"] for r in done["cells"]} == {"headline"}
    # and it has a known-truth row: the unregistered pose is tens of pixels off
    known = {r["method"]: r for r in done["truth"]}
    assert known["initial"]["tre_um_p50"] > 10.0 and known["truth"]["tre_um_max"] == 0.0
    assert known["initial"]["truth_spec"] == ""  # not one of the frozen cases


def test_multiscale_amplitude_is_the_measured_residual():
    """RMS displacement of the copied multiscale field is --residual-um, in pixels."""
    n, px = 4096, 0.325
    kw = semisynth.family_spec("multiscale", n, px, 2.0, seed=1)
    assert kw["scales"] == (512.0, 256.0, 128.0)
    spec = {**synthetic._BASE, "n": n, "seed": 1, **kw}
    xs = np.linspace(0, n - 1, 200)
    dx, dy = synthetic.extra_field(spec)(*np.meshgrid(xs, xs))
    assert abs(np.sqrt(np.mean(dx**2 + dy**2)) * px - 2.0) < 0.3
    sd = semisynth.family_spec("grid", n, px, 2.0, seed=1)["grid_sd"]
    assert 3.0 <= sd <= 7.0


def test_semisynth_reference_is_the_untouched_window(real_like, tmp_path):
    import tifffile

    a = semisynth_args(real_like, tmp_path / "cases")
    name, spec, origin = semisynth.plan(a)[0][0]
    semisynth.prepare_one(a.cases, name, spec, a, origin, 0.0, 0.5)
    x, y = origin
    whole = np.squeeze(tifffile.imread(real_like))
    got = np.squeeze(
        tifffile.imread(tmp_path / "cases" / "semisynth" / name / "ref.ome.tif")
    )
    np.testing.assert_array_equal(got, whole[y : y + 512, x : x + 512])
