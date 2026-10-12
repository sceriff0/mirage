"""The STARE adapter: the pipeline's ``high`` tier, the original slides, STARE's own STITCH.

The command-line tests read the namespace the package's own parser makes of the round's
``stare register`` command line: every stage command is built from it
(``test_stages.py`` holds the stage commands and the cohort's separate jobs). The others
run STARE for real, the four stage commands inside one run.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import imageio, methods, stages
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import load_points, result_dir
from benchmarks.regbench.methods import stare as adapter

stare = pytest.importorskip("stare")

REPO = Path(__file__).resolve().parents[3]
NAMES = ["CD3", "DAPI_R2", "PANCK"]


def run_json(out, variant, case):
    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


def multichannel(case, **extra):
    """The case as a cohort case describes its slides: other files, a channel index, names."""
    base = {"ref_channel": 1, "mov_channel": 1, "channel_names": list(NAMES)}
    return dataclasses.replace(
        case,
        ref_image="/cohort/P1_R1.ome.tif",
        mov_image="/cohort/P1_R2.ome.tif",
        extra={**case.extra, **base, **extra},
    )


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """``ns = recorded(case, opts)``: the parameters the round's stages are run with."""

    def record(case, opts=None):
        return adapter.namespace(case, tmp_path, dict(opts or {}))

    return record


@pytest.fixture
def no_stage(monkeypatch):
    """Every stage command that would be run, recorded and not run."""
    called = []
    monkeypatch.setattr(
        stages, "run_command", lambda *a, **k: called.append(a) or {"ok": False}
    )
    return called


# ── the command ───────────────────────────────────────────────────────────────
def high_row():
    """The ``high`` row of the STARE table in ``lib/RegPresets.groovy``, read from the file."""
    text = (REPO / "lib" / "RegPresets.groovy").read_text()
    table = re.search(r"\bSTARE\s*=\s*\[(.*?)\n\s*\]", text, re.S).group(1)
    row = re.search(r"\bhigh\s*:\s*\[([^\]]*)\]", table).group(1)
    return {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row)}


def test_every_tier_owned_value_is_passed(synthetic_cases, recorded, tmp_path):
    row = high_row()
    assert set(row) == {"tile", "halo", "out_tile", "coarse_max_dim"}
    want = {("max_dim" if k == "coarse_max_dim" else k): v for k, v in row.items()}
    assert adapter.TIER_HIGH_STRIDE64 == {**want, "stride": 64}
    assert len(adapter.TIER_HIGH_STRIDE64) == len(row) + 1 == 5
    case = synthetic_cases["rigid"]
    # every value is on the command line, not left to the command's default
    argv = adapter.command(case, tmp_path, {})
    for key, value in adapter.TIER_HIGH_STRIDE64.items():
        flag = "--" + key.replace("_", "-")
        assert argv[argv.index(flag) + 1] == str(value), flag
    ns = recorded(case)
    got = {k: getattr(ns, k) for k in adapter.TIER_HIGH_STRIDE64}
    assert got == adapter.TIER_HIGH_STRIDE64


def test_installed_stare_is_the_pinned_one():
    text = (REPO / "requirements" / "stare.txt").read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith("stare-registration")]
    (pin,) = lines
    assert stare.__version__ == re.search(r"/tags/v([\d.]+)\.tar\.gz", pin).group(1)


@pytest.mark.parametrize("opts, stride", [({}, 64), ({"stride": "64"}, 64)])
def test_an_unset_stride_is_64(synthetic_cases, recorded, opts, stride):
    assert recorded(synthetic_cases["rigid"], opts).stride == stride


@pytest.mark.parametrize("stride", [128, 256])
def test_the_stride_option_reaches_the_command(synthetic_cases, recorded, stride):
    case = synthetic_cases["rigid"]
    ns = recorded(case, {"stride": str(stride)})  # as `--opt stride=N` hands it over
    assert ns.stride == stride
    # nothing else moves with it
    base = vars(recorded(case))
    assert {k for k, v in vars(ns).items() if v != base[k]} == {"stride"}


@pytest.mark.parametrize("bad", ["32", "96", "512", "64.0", "abc", "", 0])
def test_only_three_strides_are_accepted(synthetic_cases, tmp_path, no_stage, bad):
    called = no_stage
    case = synthetic_cases["rigid"]
    assert methods.run_case("stare", case, tmp_path, opts={"stride": bad}) == []
    assert not called
    for variant in ("stare_rigid", "stare"):
        run = run_json(tmp_path, variant, case)
        assert run["ok"] is False and run["cause"] == "exception"
        assert "stride" in run["error"] and "64, 128, 256" in run["error"]
    assert adapter.STRIDES == (64, 128, 256)


def test_a_label_and_a_stride_name_the_stride_runs():
    assert methods.variant_names("stare", "stare_s128") == [
        "stare_s128_rigid",
        "stare_s128",
    ]
    # each stride run is checked on the pairs `stare` is checked on
    assert sc.sizes_for("stare_s256") == sc.sizes_for("stare")


def test_inputs_are_the_original_slides(synthetic_cases, recorded):
    case = multichannel(synthetic_cases["rigid"])
    ns = recorded(case)
    assert (ns.reference, ns.moving) == (case.ref_image, case.mov_image)
    assert ns.reference != case.ref_nuclear and ns.moving != case.mov_nuclear
    assert ns.nuclear_index == 1


def test_the_nuclear_index_defaults_to_zero(synthetic_cases, recorded):
    case = synthetic_cases["rigid"]
    assert "mov_channel" not in case.extra
    assert recorded(case).nuclear_index == 0


def test_a_reference_with_another_nuclear_index_fails_the_case(
    synthetic_cases, tmp_path, no_stage
):
    called = no_stage
    case = multichannel(synthetic_cases["rigid"], ref_channel=0, mov_channel=2)
    assert methods.run_case("stare", case, tmp_path) == []
    run = run_json(tmp_path, "stare", case)
    assert run["ok"] is False and run["cause"] == "exception"
    assert "one --nuclear-index" in run["error"] and not called


def test_pixel_size_and_channel_names_are_passed(synthetic_cases, recorded):
    case = multichannel(synthetic_cases["rigid"])
    ns = recorded(case)
    assert ns.pixel_size == "0.325" and float(ns.pixel_size) == case.pixel_size_um
    assert ns.channel_names == NAMES
    # a case that names no channels: the names in the moving slide's own header
    plain = synthetic_cases["rigid"]
    assert "channel_names" not in plain.extra
    assert recorded(plain).channel_names == imageio.channel_names(plain.mov_image)
    assert recorded(plain).channel_names == ["NUCLEAR"]
    # no pixel size in the case: the command's own default, nothing invented
    unknown = dataclasses.replace(plain, pixel_size_um=None)
    assert recorded(unknown).pixel_size == "auto"


def test_an_argument_error_is_reported_as_an_exception(synthetic_cases, tmp_path):
    case = synthetic_cases["rigid"]
    assert methods.run_case("stare", case, tmp_path, opts={"tile": "big"}) == []
    run = run_json(tmp_path, "stare", case)
    assert run["ok"] is False and run["cause"] == "exception"
    # argparse's own message, not a bare exit code
    assert "SystemExit" in run["error"] and "--tile" in run["error"]
    assert "invalid int value: 'big'" in run["error"]


# ── real runs ─────────────────────────────────────────────────────────────────
OPTS = {"workers": 2}


@pytest.fixture(scope="module")
def ran(synthetic_cases, tmp_path_factory):
    """`crop` (a moving slide of another shape) registered at stride 64 and 256."""
    case = synthetic_cases["crop"]
    out = tmp_path_factory.mktemp("stare")
    assert methods.run_case("stare", case, out, opts=OPTS, keep_work=True) == [
        "stare_rigid",
        "stare",
    ]
    assert methods.run_case(
        "stare", case, out, opts={**OPTS, "stride": 256}, label="stare_s256"
    ) == ["stare_s256_rigid", "stare_s256"]
    return case, out


def test_run_json_records_the_anchor_and_the_allocation(ran):
    case, out = ran
    run = run_json(out, "stare", case)
    info = run["info"]
    m0 = np.asarray(info["m0"], dtype=float)
    assert m0.shape == (3, 3) and np.allclose(m0[2], [0, 0, 1])
    # moving px -> reference px: it is the rigid row's own map
    pts = load_points(case)["truth_mov"]
    with np.load(result_dir(out, "stare_rigid", case) / "warped.npz") as z:
        rigid = z["truth_xy"]
    assert np.allclose(pts @ m0[:2, :2].T + m0[:2, 2], rigid, atol=1e-6)
    assert run_json(out, "stare_rigid", case)["info"]["m0"] == info["m0"]
    assert info["version"] == stare.__version__
    assert info["workers"] == run["workers"] == 2
    assert info["cpu_count"] >= 1
    assert info["stride"] == 64 and info["model"] == "euclidean"
    assert info["max_disp"] == adapter.TIER_HIGH_STRIDE64["halo"]
    assert info["nuclear_index"] == 0 and info["pixel_size"] == "0.325"
    assert isinstance(info["coarse_trusted"], bool) and "solver" in info["solve"]
    impl = info["implementation"]
    assert impl["changed_from_tier"] == {}
    assert len(impl["benchmark_side"]) == 2
    assert "_state" not in info
    # the stage commands ran inside this one run: its cost is this process's own clock
    jobs = info["jobs"]
    assert (
        jobs["execution"] == "local" and "inside ONE job" in impl["benchmark_side"][1]
    )
    assert (jobs["n_jobs"], jobs["n_tiles"]) == (3, 1)
    assert run["register_s"] >= jobs["wall_unlimited_parallel_s"] > 0


def test_the_rigid_stage_does_not_depend_on_the_stride(ran):
    case, out = ran
    a, b = run_json(out, "stare", case), run_json(out, "stare_s256", case)
    assert b["info"]["stride"] == 256 and b["opts"]["stride"] == "256"
    assert b["info"]["implementation"]["changed_from_tier"] == {"stride": "256"}
    assert a["info"]["m0"] == b["info"]["m0"]  # to the last bit
    with np.load(result_dir(out, "stare_rigid", case) / "warped.npz") as z:
        rigid_64 = z["cells_xy"]
    with np.load(result_dir(out, "stare_s256_rigid", case) / "warped.npz") as z:
        assert np.array_equal(z["cells_xy"], rigid_64)
    # ... while the mesh does
    with np.load(result_dir(out, "stare", case) / "warped.npz") as z:
        full_64 = z["cells_xy"]
    with np.load(result_dir(out, "stare_s256", case) / "warped.npz") as z:
        assert not np.array_equal(z["cells_xy"], full_64)


def test_native_output_is_what_stare_register_writes(ran, tmp_path):
    import stare.cli
    import tifffile

    case, out = ran
    run = run_json(out, "stare", case)
    native = run["native"]
    (written,) = native["files"]
    assert native["writer"] == "stare.stages.stitch.main" and run["write_s"] > 0
    # undeferred: the same arguments through the command itself, STITCH included
    argv = adapter.command(case, tmp_path, OPTS)
    assert stare.cli.main(argv) == 0
    direct = tifffile.imread(tmp_path / "registered.ome.tif")
    ours = tifffile.imread(written)
    assert ours.dtype == direct.dtype and np.array_equal(ours, direct)
    assert ours.any()
    # on the reference grid, although the moving slide has another shape: no origin
    assert list(ours.shape[-2:]) == case.ref_hw != case.mov_hw
    assert "origin_xy" not in native and "valid_box" not in native
    # the two STITCH flags arrived: names and scale are in the written header
    assert imageio.channel_names(written) == ["NUCLEAR"]
    assert imageio.pixel_size_um(written) == pytest.approx(case.pixel_size_um)
    assert run_json(out, "stare_rigid", case)["native"] is None


@pytest.mark.slow
def test_selfcheck_is_ok(tmp_path, monkeypatch):
    """STARE for real on its own two pairs. The cluster's list also holds ASHLAR's pair
    and patient on 8192 x 10240 (five registrations, 5.5 minutes on a laptop): their
    place in the list and the lookup of their matrices are tested without running them
    (`test_ashlar_adapter.py::test_stares_selfcheck_holds_the_anchors_of_ashlars_own_cases`)."""
    own = [(sc.DEFAULT_HW, None), (sc.LARGE_HW, None)]
    assert sc.sizes_for("stare")[:2] == own == sc.sizes_for("stare_prep")
    # the gate checks the code path that makes the cohort's results: the four stage
    # commands, each a process of its own, and never `stare register`
    import stare.cli

    def never(ns):
        raise AssertionError("the selfcheck ran `stare register`")

    monkeypatch.setattr(stare.cli, "register", never)
    ran, real = [], stages.run_command

    def spy(stage_dir, stage, command, *a, **k):
        ran.append((stage, command[1:4]))
        return real(stage_dir, stage, command, *a, **k)

    monkeypatch.setattr(stages, "run_command", spy)
    rec = sc.run("stare", tmp_path, sizes=own)
    assert rec["status"] == "ok"
    assert {(s, c[2]) for s, c in ran} == {
        ("coarse", "coarse"),
        ("tile", "reg-tile"),
        ("solve", "solve"),
        ("stitch", "stitch"),
    }
    assert all(c[:2] == ["-m", "stare.cli"] for _, c in ran)
    assert rec["variants"] == {"stare_rigid": sc.NO_STAGE_IMAGE, "stare": sc.OK}
    assert [p["case_id"] for p in rec["pairs"]] == ["pair_2048x2560", "pair_4096x5120"]
    for pair in rec["pairs"]:
        final = pair["variants"]["stare"]
        assert final["err_px_p95"] < 0.25 and final["n_found"] == final["n_dots"]
        assert final["n_narrow"] == 0  # no dot drawn under 0.8 of its size here
        assert final["native"]["channels"] == 2  # the slide's own dots channel, warped
        case = pair["case_id"]
        path = tmp_path / case[5:] / "out" / "stare" / "selfcheck" / case / "run.json"
        m0 = np.array(json.loads(path.read_text())["info"]["m0"])
        assert (
            m0.shape == (3, 3) and np.isfinite(m0).all()
        )  # what an anchored row reads
