"""The STARE adapter: the pipeline's ``high`` tier, the original slides, STARE's own STITCH.

The parameter tests read the namespace the package's own parser of ``stare register``
makes of the round's arguments: every stage command is built from it (``test_stages.py``
holds the stage commands and the cohort's separate jobs). The others run STARE for real,
its stage processes started by the run itself (the local launcher).
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

pytestmark = pytest.mark.usefixtures("local_stages")  # STARE runs in-process

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
    plain = synthetic_cases["rigid"]
    # the slides are read before the method starts, so the case keeps files that exist
    case = dataclasses.replace(
        multichannel(plain, ref_channel=0, mov_channel=2),
        ref_image=plain.ref_image,
        mov_image=plain.mov_image,
    )
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


# ── run options: proxy (benchmark-side) and sprep (STARE's own --prep) ────────
def rgb_case(case, tmp_path, **extra):
    """The case as an RGB one: PNG slides, with the case's own one-channel files standing
    in for the dataset module's grey proxy."""
    from PIL import Image

    paths = {}
    for side in ("ref", "mov"):
        plane = _plane(getattr(case, f"{side}_image"))
        top = float(plane.max()) or 1.0
        # haematoxylin-stained: optical density along the stain's own colour vector, so
        # the colour deconvolution of the `haem` proxy finds the nuclei
        od = 1.5 * plane.astype(float) / top
        rgb = 255 * np.exp(-od[..., None] * np.array([0.65, 0.70, 0.29]))
        paths[side] = tmp_path / f"{side}.png"
        Image.fromarray(rgb.astype("uint8")).save(paths[side])
    return dataclasses.replace(
        case,
        modality="brightfield",
        ref_image=str(paths["ref"]),
        mov_image=str(paths["mov"]),
        ref_nuclear=case.ref_image,
        mov_nuclear=case.mov_image,
        extra={**case.extra, "settings": {"proxy": "grey", "reduce": 1}, **extra},
    )


def _plane(path):
    from stare.slide_io import open_lazy

    arr, _, close = open_lazy(str(path))
    try:
        return np.asarray(arr[0, :, :])
    finally:
        close()


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_sprep_reaches_coarse_and_reg_tile_argv(synthetic_cases, tmp_path):
    argv = adapter.stage_argv(synthetic_cases["rigid"], {"sprep": "percentile"}, tmp_path)
    for stage in ("coarse", "reg-tile"):
        assert flag(argv[stage], "--prep") == "percentile", stage


def test_default_run_passes_prep_none_and_the_pixel_size(synthetic_cases, tmp_path):
    case = multichannel(synthetic_cases["rigid"])
    argv = adapter.stage_argv(case, {}, tmp_path)
    for stage in ("coarse", "reg-tile"):
        assert flag(argv[stage], "--prep") == "none"
        assert flag(argv[stage], "--pixel-size") == str(case.pixel_size_um)
    unknown = dataclasses.replace(synthetic_cases["rigid"], pixel_size_um=None)
    assert flag(adapter.stage_argv(unknown, {}, tmp_path)["coarse"], "--pixel-size") == "auto"


def test_a_run_with_no_option_is_the_run_with_the_defaults_spelled(
    synthetic_cases, tmp_path
):
    case = synthetic_cases["rigid"]
    assert adapter.stage_argv(case, {}, tmp_path) == adapter.stage_argv(
        case, {"sprep": "none", "proxy": "channel"}, tmp_path
    )


@pytest.mark.parametrize(
    "opts, match",
    [
        ({"proxy": "haem"}, "proxy"),
        ({"proxy": "nope"}, "proxy"),
        ({"sprep": "nope"}, "sprep"),
        ({"prep": "grey", "sprep": "clahe"}, "prep"),
        ({"prep": "grey", "proxy": "channel"}, "prep"),
    ],
)
def test_refused_option_combinations_on_fluorescence(synthetic_cases, opts, match):
    with pytest.raises(ValueError, match=match):
        adapter.settings(opts, synthetic_cases["rigid"])


def test_proxy_on_an_rgb_case(synthetic_cases, tmp_path):
    case = rgb_case(synthetic_cases["rigid"], tmp_path)
    assert adapter.settings({}, case) == ("grey", "none")
    assert adapter.settings({"proxy": "haem", "sprep": "clahe"}, case) == (
        "haem",
        "clahe",
    )
    with pytest.raises(ValueError, match="proxy=channel"):
        adapter.settings({"proxy": "channel"}, case)
    unknown = dataclasses.replace(case, pixel_size_um=None)
    with pytest.raises(ValueError, match="pixel size"):
        adapter.settings({"sprep": "clahe"}, unknown)
    assert adapter.settings({"sprep": "percentile"}, unknown) == ("grey", "percentile")


def test_default_grey_proxy_is_not_written_again_but_haem_is(synthetic_cases, tmp_path):
    case = rgb_case(synthetic_cases["rigid"], tmp_path)
    grey = adapter.Round(case, tmp_path / "g", {})
    assert grey.candidate is None and grey.inputs is None
    assert adapter.slides(case)[0] == case.ref_nuclear
    haem = adapter.Round(case, tmp_path / "h", {"proxy": "haem"})
    assert haem.candidate == "haem" and haem.row is None
    assert str(haem.inputs[0]).endswith("reference.haem.ome.tif")
    assert flag(haem.argv["coarse"], "--reference") == str(haem.inputs[0])


def test_proxy_recorded_as_benchmark_side(synthetic_cases, tmp_path):
    case = rgb_case(synthetic_cases["crop"], tmp_path)
    info = {}
    for opts in ({"proxy": "haem", "sprep": "clahe"}, {}):
        out = tmp_path / ("o_" + str(len(opts)))
        methods.run_case("stare", case, out, opts={**OPTS, **opts}, keep_work=True)
        info[len(opts)] = run_json(out, "stare", case)["info"]
    haem, plain = info[2], info[0]
    assert any("proxy `haem`" in x for x in haem["implementation"]["benchmark_side"])
    assert haem["input"]["sprep"] == "clahe" and haem["input"]["proxy_option"] == "haem"
    # sprep is STARE's own option: not a departure; the default proxy is not either
    assert not any("clahe" in x for x in haem["implementation"]["benchmark_side"])
    assert not any("proxy `" in x for x in plain["implementation"]["benchmark_side"])
    assert plain["input"]["sprep"] == "none" and plain["input"]["proxy_option"] == "grey"


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
    # the run started its stage processes itself: its cost is its own process tree's
    jobs = info["jobs"]
    assert jobs["execution"] == "local" and "_cost" not in info
    assert "started by this run itself" in impl["benchmark_side"][1]
    assert "its own process tree's" in impl["benchmark_side"][1]
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


def test_native_output_is_stitch_s_image_on_the_reference_grid(ran):
    import tifffile

    case, out = ran
    run = run_json(out, "stare", case)
    native = run["native"]
    (written,) = native["files"]
    assert native["writer"] == "stare.stages.stitch.main" and run["write_s"] > 0
    ours = tifffile.imread(written)
    assert ours.any() and str(ours.dtype) == native["dtype"]
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
    # ... and the transform is obtained as the cohort's last job obtains it
    collected, collect = [], adapter._collect
    monkeypatch.setattr(
        adapter,
        "_collect",
        lambda rnd, jobs: (
            collected.append((rnd.case.case_id, jobs)) or collect(rnd, jobs)
        ),
    )
    rec = sc.run("stare", tmp_path, sizes=own)
    assert rec["status"] == "ok"
    assert collected == [("pair_2048x2560", False), ("pair_4096x5120", False)]
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
