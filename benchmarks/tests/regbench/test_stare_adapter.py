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


def test_every_tier_owned_value_is_passed(toy_cases, recorded, tmp_path):
    row = high_row()
    assert set(row) == {"tile", "halo", "out_tile", "coarse_max_dim"}
    want = {("max_dim" if k == "coarse_max_dim" else k): v for k, v in row.items()}
    assert adapter.TIER_HIGH_STRIDE64 == {**want, "stride": 64}
    assert len(adapter.TIER_HIGH_STRIDE64) == len(row) + 1 == 5
    case = toy_cases["rigid"]
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
def test_an_unset_stride_is_64(toy_cases, recorded, opts, stride):
    assert recorded(toy_cases["rigid"], opts).stride == stride


@pytest.mark.parametrize("stride", [128, 256])
def test_the_stride_option_reaches_the_command(toy_cases, recorded, stride):
    case = toy_cases["rigid"]
    ns = recorded(case, {"stride": str(stride)})  # as `--opt stride=N` hands it over
    assert ns.stride == stride
    # nothing else moves with it
    base = vars(recorded(case))
    assert {k for k, v in vars(ns).items() if v != base[k]} == {"stride"}


@pytest.mark.parametrize("bad", ["32", "96", "512", "64.0", "abc", "", 0])
def test_only_three_strides_are_accepted(toy_cases, tmp_path, no_stage, bad):
    called = no_stage
    case = toy_cases["rigid"]
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


def test_inputs_are_the_original_slides(toy_cases, recorded):
    case = multichannel(toy_cases["rigid"])
    ns = recorded(case)
    assert (ns.reference, ns.moving) == (case.ref_image, case.mov_image)
    assert ns.reference != case.ref_nuclear and ns.moving != case.mov_nuclear
    assert ns.nuclear_index == 1


def test_the_nuclear_index_defaults_to_zero(toy_cases, recorded):
    case = toy_cases["rigid"]
    assert "mov_channel" not in case.extra
    assert recorded(case).nuclear_index == 0


def test_a_reference_with_another_nuclear_index_fails_the_case(
    toy_cases, tmp_path, no_stage
):
    called = no_stage
    plain = toy_cases["rigid"]
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


def test_pixel_size_and_channel_names_are_passed(toy_cases, recorded):
    case = multichannel(toy_cases["rigid"])
    ns = recorded(case)
    assert ns.pixel_size == "0.325" and float(ns.pixel_size) == case.pixel_size_um
    assert ns.channel_names == NAMES
    # a case that names no channels: the names in the moving slide's own header
    plain = toy_cases["rigid"]
    assert "channel_names" not in plain.extra
    assert recorded(plain).channel_names == imageio.channel_names(plain.mov_image)
    assert recorded(plain).channel_names == ["NUCLEAR"]
    # no pixel size in the case: the command's own default, nothing invented
    unknown = dataclasses.replace(plain, pixel_size_um=None)
    assert recorded(unknown).pixel_size == "auto"


def test_an_argument_error_is_reported_as_an_exception(toy_cases, tmp_path):
    case = toy_cases["rigid"]
    assert methods.run_case("stare", case, tmp_path, opts={"tile": "big"}) == []
    run = run_json(tmp_path, "stare", case)
    assert run["ok"] is False and run["cause"] == "exception"
    # argparse's own message, not a bare exit code
    assert "SystemExit" in run["error"] and "--tile" in run["error"]
    assert "invalid int value: 'big'" in run["error"]


# ── run options: proxy (benchmark-side) and sprep (STARE's own --prep) ────────
def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_sprep_reaches_coarse_and_reg_tile_argv(toy_cases, tmp_path):
    argv = adapter.stage_argv(toy_cases["rigid"], {"sprep": "percentile"}, tmp_path)
    for stage in ("coarse", "reg-tile"):
        assert flag(argv[stage], "--prep") == "percentile", stage


def test_default_run_passes_sprep_none_and_the_pixel_size(toy_cases, tmp_path):
    case = multichannel(toy_cases["rigid"])
    argv = adapter.stage_argv(case, {}, tmp_path)
    for stage in ("coarse", "reg-tile"):
        assert flag(argv[stage], "--prep") == "none"
        assert flag(argv[stage], "--pixel-size") == str(case.pixel_size_um)
    unknown = dataclasses.replace(toy_cases["rigid"], pixel_size_um=None)
    assert (
        flag(adapter.stage_argv(unknown, {}, tmp_path)["coarse"], "--pixel-size")
        == "auto"
    )


def test_a_run_with_no_option_is_the_run_with_the_defaults_spelled(toy_cases, tmp_path):
    case = toy_cases["rigid"]
    assert adapter.stage_argv(case, {}, tmp_path) == adapter.stage_argv(
        case, {"sprep": "none", "proxy": "channel"}, tmp_path
    )


@pytest.mark.parametrize(
    "opts, match",
    [
        ({"proxy": "haem"}, "proxy"),
        ({"proxy": "nope"}, "proxy"),
        ({"sprep": "nope"}, "sprep"),
    ],
)
def test_refused_option_combinations_on_fluorescence(toy_cases, opts, match):
    with pytest.raises(ValueError, match=match):
        adapter.settings(opts, toy_cases["rigid"])


def test_the_fluorescence_command_is_the_one_before_the_brightfield_rule(
    toy_cases, tmp_path
):
    """The cohort's command, every argument, as it was before an RGB case had a rule
    (c532c1ba): the brightfield branches change nothing of a fluorescence round."""
    case = multichannel(toy_cases["rigid"])
    s = str(tmp_path)
    assert adapter.command(case, tmp_path, {}) == [
        "register", "--reference", "/cohort/P1_R1.ome.tif",
        "--moving", "/cohort/P1_R2.ome.tif", "--out", f"{s}/registered.ome.tif",
        "--manifest", f"{s}/manifest.json", "--tre", f"{s}/tre.json",
        "--workdir", s, "--workers", "1", "--reference-name", "reference",
        "--moving-name", "moving", "--nuclear-index", "1", "--tile", "2048",
        "--halo", "256", "--out-tile", "1024", "--max-dim", "1024", "--stride", "64",
        "--prep", "none", "--pixel-size", "0.325", "--channel-names", *NAMES,
    ]  # fmt: skip
    rigid = toy_cases["rigid"]
    assert adapter.stage_argv(rigid, {}, tmp_path)["coarse"] == [
        "--reference", rigid.ref_image, "--moving", rigid.mov_image,
        "--nuclear-index", "0", "--tile", "2048", "--halo", "256", "--max-dim", "1024",
        "--model", "euclidean", "--out-m0", f"{s}/moving_m0.json",
        "--out-tiles", f"{s}/moving_tiles.csv", "--prep", "none",
        "--pixel-size", "0.325",
    ]  # fmt: skip
    rnd = adapter.Round(rigid, tmp_path, {})
    assert rnd.candidate is None and rnd.inputs is None


def test_the_cohort_stare_list_is_unchanged():
    from benchmarks.regbench import variants

    cohort = [
        (variants.run_name(r), r.opts)
        for r in variants.runs_for("multiplex")
        if r.method == "stare"
    ]
    assert ("stare", {}) in cohort
    assert ("stare_clahe", {"proxy": "channel", "sprep": "clahe"}) in cohort
    assert all(o.get("proxy", "channel") == "channel" for _, o in cohort)
    assert cohort == [
        (variants.run_name(r), r.opts)
        for r in variants.runs_for(None)
        if r.method == "stare"
    ]


# ── a brightfield case: a one-channel proxy and STARE's own --prep ────────────
@pytest.fixture
def rgb_case(toy_cases, tmp_path):
    """``rigid`` as an RGB case: interleaved RGB OME-TIFFs and their grey proxies
    (``rgb_cases``), as a dataset module writes them."""
    from . import rgb_cases

    return rgb_cases.rgb_case(toy_cases["rigid"], tmp_path / "bf")


def _reduced(case, f, **settings):
    extra = {**case.extra, "settings": {"proxy": "grey", "reduce": f, **settings}}
    return dataclasses.replace(case, extra=extra)


def test_an_rgb_case_registers_on_its_grey_proxy(rgb_case, tmp_path):
    assert adapter.settings({}, rgb_case) == ("grey", "none")
    argv = adapter.command(rgb_case, tmp_path, {})
    assert argv[argv.index("--reference") + 1] == rgb_case.ref_nuclear
    assert argv[argv.index("--moving") + 1] == rgb_case.mov_nuclear
    assert flag(argv, "--nuclear-index") == "0"
    assert flag(argv, "--pixel-size") == repr(float(rgb_case.pixel_size_um))
    rnd = adapter.Round(rgb_case, tmp_path / "g", {})
    assert rnd.candidate is None and rnd.inputs is None  # the case's own proxy
    # a proxy reduced f-fold: STARE's pixel is f of the slide's
    four = adapter.command(_reduced(rgb_case, 4), tmp_path, {})
    assert flag(four, "--pixel-size") == repr(float(rgb_case.pixel_size_um) * 4)
    # no established pixel size: STITCH's header gets the placeholder
    none = dataclasses.replace(rgb_case, pixel_size_um=None)
    assert flag(adapter.command(none, tmp_path, {}), "--pixel-size") == repr(
        adapter.NOMINAL_PIXEL_SIZE
    )


def test_a_channel_proxy_on_an_rgb_case_is_refused(rgb_case, tmp_path):
    with pytest.raises(ValueError, match="RGB"):
        adapter.settings({"proxy": "channel"}, rgb_case)


def test_the_proxy_options_of_an_rgb_case(rgb_case):
    """Ported from 5a46f778 (``test_proxy_on_an_rgb_case``)."""
    assert adapter.settings({"proxy": "haem", "sprep": "clahe"}, rgb_case) == (
        "haem",
        "clahe",
    )
    assert adapter.settings({"proxy": "grey_smooth"}, rgb_case) == (
        "grey_smooth",
        "none",
    )
    unknown = dataclasses.replace(rgb_case, pixel_size_um=None)
    with pytest.raises(ValueError, match="pixel size"):
        adapter.settings({"sprep": "clahe"}, unknown)
    assert adapter.settings({"sprep": "percentile"}, unknown) == ("grey", "percentile")


def test_the_default_grey_proxy_is_not_written_but_haem_is(rgb_case, tmp_path):
    """Ported from 5a46f778: ``grey`` is the case's own proxy, read as it is; ``haem``
    is written by the benchmark under the round's ``stages/prep`` and is what COARSE
    reads."""
    assert adapter.slides(rgb_case)[:3] == (
        rgb_case.ref_nuclear,
        rgb_case.mov_nuclear,
        0,
    )
    haem = adapter.Round(rgb_case, tmp_path / "h", {"proxy": "haem"})
    assert haem.candidate == "haem" and haem.inputs is not None
    assert haem.inputs[0] == tmp_path / "h" / "prep" / "reference.haem.ome.tif"
    assert flag(haem.argv["coarse"], "--reference") == str(haem.inputs[0])
    assert flag(haem.argv["coarse"], "--nuclear-index") == "0"
    assert adapter.Round(rgb_case, tmp_path / "s", {"proxy": "grey_smooth"}).candidate


def test_the_proxy_is_written_before_coarse_once_and_again_when_a_slide_changes(
    rgb_case, tmp_path, no_stage, monkeypatch
):
    """The benchmark writes it in COARSE's job, outside the stage command; a second
    COARSE of the same slides finds it there and writes nothing; a slide replaced is
    another proxy and another COARSE."""
    import os

    from benchmarks.regbench import prep

    written = []
    real = prep.write_inputs
    monkeypatch.setattr(
        prep, "write_inputs", lambda *a: written.append(a[1]) or real(*a)
    )
    rnd = adapter.Round(rgb_case, tmp_path / "r", {"proxy": "haem"})
    adapter.run_stage(rgb_case, rnd.dir, rnd.opts, stages.COARSE, rnd=rnd)
    assert written == ["haem"] and all(p.exists() for p in rnd.inputs)
    record = json.loads((rnd.prep_dir / "record.json").read_text())
    assert record["proxy"] == "haem" and record["base"] == "haem"
    assert record["hw"] == {"ref": rgb_case.ref_hw, "mov": rgb_case.mov_hw}
    # COARSE ran on the written files
    (coarse,) = no_stage
    assert flag(coarse[2], "--reference") == str(rnd.inputs[0])
    again = adapter.Round(rgb_case, tmp_path / "r", {"proxy": "haem"})
    adapter.run_stage(rgb_case, again.dir, again.opts, stages.COARSE, rnd=again)
    assert written == ["haem"]
    st = os.stat(rgb_case.mov_image)
    os.utime(rgb_case.mov_image, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    third = adapter.Round(rgb_case, tmp_path / "r", {"proxy": "haem"})
    adapter.run_stage(rgb_case, third.dir, third.opts, stages.COARSE, rnd=third)
    assert written == ["haem", "haem"]


@pytest.mark.parametrize(
    "exc, cause",
    [(MemoryError(), "benchmark_oom"), (RuntimeError("a bug"), "benchmark")],
)
def test_a_failed_proxy_is_the_benchmarks_cause(
    rgb_case, tmp_path, no_stage, monkeypatch, exc, cause
):
    """Ported from 5a46f778 (``test_a_failed_preprocessing_job_is_the_benchmarks_cause``):
    whatever ends the benchmark's proxy, the round's cause is the benchmark's, never
    STARE's ``exception`` / ``oom``; COARSE is not run."""
    from benchmarks.regbench import failure_log, prep

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(prep, "write_inputs", boom)
    rnd = adapter.Round(rgb_case, tmp_path / "r", {"proxy": "haem"})
    ((stage, _, ran, why, rec),) = adapter.run_stage(
        rgb_case, rnd.dir, rnd.opts, stages.COARSE, rnd=rnd
    )
    assert stage == stages.COARSE and ran and "proxy" in why and no_stage == []
    assert rec["ok"] is False and rec["cause"] == cause
    assert rnd.record(stages.COARSE)["cause"] == cause
    assert failure_log.WHOSE[cause] == failure_log.BENCHMARK
    err = adapter._failed(rnd, stages.COARSE, None, False)
    assert err.regbench_cause == cause and "was not written" in str(err)


@pytest.mark.parametrize(
    "make, opts, match",
    [
        # HyReCo prepared with `--reduce 2`: a proxy written at the slide's size
        (lambda c: _reduced(c, 2), {"proxy": "haem"}, "reduced proxy"),
        # ANHIR's mammary glands: CLAHE's region is in micrometres
        (
            lambda c: dataclasses.replace(c, pixel_size_um=None),
            {"sprep": "clahe"},
            "no established pixel size",
        ),
        # HyReCo prepared with `--proxy haematoxylin`: `grey` is not its own proxy
        (
            lambda c: _reduced(c, 1, proxy="haematoxylin"),
            {"proxy": "grey", "sprep": "clahe"},
            "haematoxylin",
        ),
    ],
)
def test_a_cell_a_dataset_cannot_have_is_the_benchmarks_refusal(
    rgb_case, tmp_path, make, opts, match
):
    """Ported from 5a46f778 (its three typed refusals), each mapped onto the existing
    cause ``benchmark`` (not the archive's ``reduced_proxy`` / ``no_pixel_size`` /
    ``proxy_mismatch``: score writes a column per cause). The run records the refusal;
    the tile feeder lists no tile of it."""
    from benchmarks.regbench import failure_log, resources

    case = make(rgb_case)
    with pytest.raises(ValueError, match=match) as err:
        adapter.Round(case, tmp_path / "r", opts)
    assert resources.classify_error(err.value) == "benchmark"
    assert failure_log.WHOSE["benchmark"] == failure_log.BENCHMARK
    assert adapter.tiles_due(case, tmp_path / "r", opts, 1) == []
    out = tmp_path / "out"
    methods.run_group("stare", [case], out, opts, label="stare_x")
    runs = [json.loads(p.read_text()) for p in out.rglob("run.json")]
    assert runs and all(r["ok"] is False for r in runs)
    assert {r["cause"] for r in runs} == {"benchmark"}


def test_the_baseline_runs_on_a_haematoxylin_prepared_case(rgb_case):
    haem = _reduced(rgb_case, 1, proxy="haematoxylin")
    assert adapter.settings({}, haem) == ("grey", "none")  # no proxy option: no refusal
    lum = _reduced(rgb_case, 1, proxy="luminance")
    assert adapter.settings({"proxy": "grey", "sprep": "clahe"}, lum) == (
        "grey",
        "clahe",
    )


def test_a_reduced_proxys_map_is_taken_back_to_full_resolution():
    """Block i of an f-fold block average is centred on pixel i f + (f - 1) / 2."""
    f = 4
    shift = np.array([3.0, -2.0])  # a translation in the reduced frame
    full = adapter.reduced_map(lambda xy: np.asarray(xy) + shift, f)
    xy = np.array([[1.5, 1.5], [10.0, 7.0]])  # the centre of block 0 is (1.5, 1.5)
    assert np.allclose(full(xy), xy + shift * f)
    assert adapter.reduced_map(len, 1) is len


def test_the_tile_count_of_a_brightfield_case_is_its_proxys(rgb_case):
    """A brightfield case's tiles are those of the proxy STARE reads (a reduced one is
    smaller than the slide), never of the slide's ``ref_hw``."""
    small = {"tile": "256", "halo": "64"}
    big = dataclasses.replace(rgb_case, ref_hw=[9000, 9000])  # not what STARE reads
    assert adapter.tile_count(big, small) == adapter.tile_count(rgb_case, small) > 1
    fluo = dataclasses.replace(big, modality="fluorescence")
    assert adapter.tile_count(fluo, small) > adapter.tile_count(big, small)


@pytest.fixture(scope="module")
def ran_rgb(toy_cases, tmp_path_factory):
    """``crop`` as an RGB case, registered on its own grey proxy and on a written
    ``haem`` proxy with STARE's CLAHE."""
    from . import rgb_cases

    root = tmp_path_factory.mktemp("stare_rgb")
    case = rgb_cases.rgb_case(toy_cases["crop"], root / "bf")
    for key, opts in (("plain", {}), ("haem", {"proxy": "haem", "sprep": "clahe"})):
        label = None if key == "plain" else "stare_haem_clahe"
        got = methods.run_case(
            "stare",
            case,
            root / key,
            opts={**OPTS, **opts},
            label=label,
            keep_work=True,
        )
        assert got == ([*adapter.VARIANTS] if key == "plain" else [
            "stare_haem_clahe_rigid", "stare_haem_clahe"
        ])  # fmt: skip
    return case, root


def test_a_proxy_is_recorded_as_benchmark_side(ran_rgb):
    """Ported from 5a46f778 (``test_proxy_recorded_as_benchmark_side``)."""
    case, root = ran_rgb
    plain = run_json(root / "plain", "stare", case)["info"]
    haem = run_json(root / "haem", "stare_haem_clahe", case)["info"]
    side = haem["implementation"]["benchmark_side"]
    assert any("proxy `haem`" in x and "unbilled" in x for x in side)
    assert haem["input"]["sprep"] == "clahe" and haem["input"]["proxy_option"] == "haem"
    assert haem["input"]["moving"].endswith("moving.haem.ome.tif")
    assert haem["proxy_prep"]["proxy"] == "haem" and haem["proxy_prep"]["prep_s"] > 0
    # sprep is STARE's own option: not a departure; the default proxy is not either
    assert not any("clahe" in x for x in side)
    plain_side = plain["implementation"]["benchmark_side"]
    assert not any("proxy `" in x for x in plain_side)
    assert any(x.startswith("brightfield: STARE is given") for x in plain_side)
    assert plain["input"] == {
        "reference": case.ref_nuclear,
        "moving": case.mov_nuclear,
        "kind": "one-channel proxy of an RGB slide",
        "proxy_option": "grey",
        "sprep": "none",
        "proxy": "grey",
        "reduced_by": 1,
    }
    assert "proxy_prep" not in plain and plain["nuclear_index"] == 0


def test_the_written_image_is_the_cases_own_proxy_registered(ran_rgb):
    """5a46f778's rule: STITCH draws the one-channel proxy STARE was given, the case's
    own (never a proxy the benchmark wrote), on the reference grid."""
    import tifffile

    case, root = ran_rgb
    for out, name in ((root / "plain", "stare"), (root / "haem", "stare_haem_clahe")):
        native = run_json(out, name, case)["native"]
        assert native["content"] == "the one-channel proxy STARE was given, registered"
        assert native["reduced_by"] == 1 and native["channels"] == 1
        (written,) = native["files"]
        image = tifffile.imread(written)
        assert image.dtype == np.uint8 and list(image.shape[-2:]) == case.ref_hw
        assert image.any()
        (stitch,) = out.rglob("stages/jobs/stitch.json")  # the kept work directory
        command = json.loads(stitch.read_text())["command"]
        assert flag(command, "--moving") == case.mov_nuclear


def test_clahe_needs_the_pixel_size_and_percentile_does_not(toy_cases):
    unknown = dataclasses.replace(toy_cases["rigid"], pixel_size_um=None)
    with pytest.raises(ValueError, match="pixel size"):
        adapter.settings({"sprep": "clahe"}, unknown)
    assert adapter.settings({"sprep": "percentile"}, unknown) == (
        "channel",
        "percentile",
    )


# ── real runs ─────────────────────────────────────────────────────────────────
OPTS = {"workers": 2}


@pytest.fixture(scope="module")
def ran(toy_cases, tmp_path_factory):
    """`crop` (a moving slide of another shape) registered at stride 64 and 256, and at
    the low tier."""
    case = toy_cases["crop"]
    out = tmp_path_factory.mktemp("stare")
    assert methods.run_case("stare", case, out, opts=OPTS, keep_work=True) == [
        "stare_rigid",
        "stare",
    ]
    assert methods.run_case(
        "stare", case, out, opts={**OPTS, "stride": 256}, label="stare_s256"
    ) == ["stare_s256_rigid", "stare_s256"]
    assert methods.run_case(
        "stare", case, out, opts={**OPTS, "tier": "low"}, label="stare_low"
    ) == ["stare_low_rigid", "stare_low"]
    return case, out


def test_run_json_records_the_anchor_and_the_allocation(ran):
    case, out = ran
    run = run_json(out, "stare", case)
    info = run["info"]
    m0 = np.asarray(info["m0"], dtype=float)
    assert m0.shape == (3, 3) and np.allclose(m0[2], [0, 0, 1])
    # moving px -> reference px: it is the rigid row's own map
    pts = load_points(case)["nuclei_mov_xy"]
    with np.load(result_dir(out, "stare_rigid", case) / "warped.npz") as z:
        rigid = z["nuclei_xy"]
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
    assert sc.sizes_for("stare")[:2] == own
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


def test_the_due_tiles_of_every_case_are_listed(toy_cases, tmp_path, monkeypatch):
    """`stages --due` is what the tile feeder submits: the plan rows of every case whose
    COARSE is ok and whose tiles have not run; an error of a round is raised, not
    swallowed."""
    import contextlib
    import io

    from benchmarks.regbench import cli
    from benchmarks.regbench.cases import write_case

    cases = tmp_path / "cases"
    base = toy_cases["rigid"]
    ids = ["a_ok", "c_ok"]
    for cid in ids:
        write_case(cases, dataclasses.replace(base, case_id=cid, ref_hw=[4096, 5120]))
    real = adapter.Round

    class Fake:
        """A round whose COARSE is ok, none of whose tiles has run."""

        def __init__(self, case, stage_dir, opts):
            pass

        def done(self, stage, row=None):
            return {"ok": True}

        def plan(self):
            return [{}] * n

        def record(self, stage, row=None):
            return None

        def command(self, stage, row=None):
            return ["x", str(row)]

        def identity(self, stage, row=None):
            return "i"

    layout = tmp_path / "layout.tsv"
    sized = ["stages", "--cases", str(cases), "--dataset", "multiplex"]
    sized += ["--method", "stare", "--layout", str(layout)]
    assert cli.main(sized) == 0
    n = adapter.tile_count(load_case_by_id(cases, "a_ok"), {"label": ""})
    assert n > 1
    monkeypatch.setattr(adapter, "Round", Fake)
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        code = cli.main([*sized, "--out", str(tmp_path / "out"), "--due", "1"])
    monkeypatch.setattr(adapter, "Round", real)
    assert code == 0
    rows = [int(v) for v in said.getvalue().split()]
    assert rows == [*range(n), *range(n, 2 * n)]

    def broken(*a, **k):
        raise RuntimeError("a bug is not a refusal")

    monkeypatch.setattr(adapter, "Round", broken)
    with pytest.raises(RuntimeError):
        adapter.tiles_due(base, tmp_path / "r", {}, 1)


def load_case_by_id(cases, cid):
    from benchmarks.regbench.cases import list_cases

    return next(c for c in list_cases(cases, "multiplex") if c.case_id == cid)


# ── the tiers: high, medium and low, each the pipeline's row ─────────────────
def tier_row(tier):
    """One row of the STARE table in ``lib/RegPresets.groovy``, in the adapter's names
    (``coarse_max_dim`` is ``--max-dim``), read from the file."""
    text = (REPO / "lib" / "RegPresets.groovy").read_text()
    table = re.search(r"\bSTARE\s*=\s*\[(.*?)\n\s*\]", text, re.S).group(1)
    row = re.search(rf"\b{tier}\s*:\s*\[([^\]]*)\]", table).group(1)
    return {
        ("max_dim" if k == "coarse_max_dim" else k): int(v)
        for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", row)
    }


def test_every_tier_is_the_pipelines_row():
    assert tuple(adapter.TIERS) == ("high", "medium", "low")
    assert adapter.TIERS == {t: tier_row(t) for t in adapter.TIERS}
    # the high row is the one every earlier result was registered with, keys in order
    assert adapter.TIERS["high"] == {
        ("max_dim" if k == "coarse_max_dim" else k): v for k, v in high_row().items()
    }
    assert list(adapter.TIER_HIGH_STRIDE64) == [*adapter.TIERS["high"], "stride"]
    assert adapter.TIER_HIGH_STRIDE64 == {**adapter.TIERS["high"], "stride": 64}
    assert "tier" in adapter.OPTIONS


def test_the_high_tier_command_is_unchanged(toy_cases, tmp_path):
    """Unset is high, and the high command is the one every earlier result was made
    with: the tier's four values from RegPresets' high row, then the stride, in that
    order, between the nuclear index and ``--prep``."""
    case = toy_cases["rigid"]
    row = high_row()
    argv = adapter.command(case, tmp_path, {})
    at = argv.index("--nuclear-index") + 2
    assert argv[at : at + 11] == [
        "--tile",
        str(row["tile"]),
        "--halo",
        str(row["halo"]),
        "--out-tile",
        str(row["out_tile"]),
        "--max-dim",
        str(row["coarse_max_dim"]),
        "--stride",
        "64",
        "--prep",
    ]
    assert row == {"tile": 2048, "halo": 256, "out_tile": 1024, "coarse_max_dim": 1024}
    assert argv == adapter.command(case, tmp_path, {"tier": "high"})
    assert adapter.stage_argv(case, {}, tmp_path) == adapter.stage_argv(
        case, {"tier": "high"}, tmp_path
    )


@pytest.mark.parametrize("tier", ["high", "medium", "low"])
def test_a_tier_sets_its_four_values_and_solve_bound(toy_cases, tmp_path, tier):
    case = toy_cases["rigid"]
    row = tier_row(tier)
    argv = adapter.command(case, tmp_path, {"tier": tier})
    flags = ("--tile", "--halo", "--out-tile", "--max-dim", "--stride")
    got = {f: argv[argv.index(f) + 1] for f in flags}
    assert got == {
        "--tile": str(row["tile"]),
        "--halo": str(row["halo"]),
        "--out-tile": str(row["out_tile"]),
        "--max-dim": str(row["max_dim"]),
        "--stride": "64",
    }
    # the stage commands carry them: COARSE the tile, halo and size; SOLVE's range gate
    # is the tier's halo, passed explicitly as the pipeline's TILED_SOLVE does
    stage = adapter.stage_argv(case, {"tier": tier}, tmp_path)
    assert flag(stage["coarse"], "--tile") == str(row["tile"])
    assert flag(stage["coarse"], "--halo") == str(row["halo"])
    assert flag(stage["coarse"], "--max-dim") == str(row["max_dim"])
    assert flag(stage["solve"], "--max-disp") == str(row["halo"])


def test_a_tier_and_a_stride_cross(toy_cases, recorded):
    ns = recorded(toy_cases["rigid"], {"tier": "medium", "stride": "128"})
    assert (ns.tile, ns.halo, ns.stride) == (1024, 192, 128)


@pytest.mark.parametrize("bad", ["huge", "High", "", "custom"])
def test_an_unknown_tier_is_refused(toy_cases, tmp_path, bad):
    with pytest.raises(ValueError, match="tier"):
        adapter.command(toy_cases["rigid"], tmp_path, {"tier": bad})
    with pytest.raises(ValueError, match="high, medium, low"):
        adapter.tier_of({"tier": bad})
    assert adapter.tier_of({}) == "high"


def test_the_tile_count_follows_the_tier(toy_cases):
    case = toy_cases["rigid"]
    got = [adapter.tile_count(case, {"tier": t}) for t in ("high", "medium", "low")]
    assert got == [1, 1, 4]  # STARE's tile plan of the toy slide at each tier
    assert adapter.tile_count(case, {}) == got[0]


def test_run_json_records_the_tier(ran):
    case, out = ran
    high = run_json(out, "stare", case)["info"]
    low = run_json(out, "stare_low", case)["info"]
    assert high["implementation"]["tier"] == "high"
    assert low["implementation"]["tier"] == "low"
    # a tier's own values are not changes from it; the bound is that tier's halo
    assert low["implementation"]["changed_from_tier"] == {}
    assert low["max_disp"] == adapter.TIERS["low"]["halo"] == 128
    assert low["stride"] == 64
    assert "`low` STARE tier" in low["implementation"]["parameters"]
    assert "`high` STARE tier" in high["implementation"]["parameters"]
