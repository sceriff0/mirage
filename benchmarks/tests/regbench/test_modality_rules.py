"""The brightfield input rules: which methods take an RGB case, and that they run one.

Both VALIS adapters, DeeperHistReg and STARE (through a one-channel proxy) have a rule;
ASHLAR and PALOM have none and refuse such a case with the reason
(``methods.NO_BRIGHTFIELD``).
"""

import dataclasses
import json

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import methods
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import result_dir

from . import rgb_cases

SMALL = (512, 640)


def _bf(case):
    return dataclasses.replace(case, modality="brightfield")


@pytest.mark.parametrize("method", ["valis_mirage", "valis", "deeperhistreg", "stare"])
def test_brightfield_is_accepted(method, toy_cases):
    case = _bf(next(iter(toy_cases.values())))
    assert methods.input_rule(method, "brightfield") is None
    methods.require_fluorescence(case, method)  # does not raise


@pytest.mark.parametrize("method", ["ashlar", "palom"])
def test_brightfield_is_refused_with_its_reason(method, toy_cases):
    case = _bf(next(iter(toy_cases.values())))
    with pytest.raises(ValueError, match="no input rule for a brightfield case"):
        methods.require_fluorescence(case, method)
    assert methods.input_rule(method, "brightfield") == methods.NO_BRIGHTFIELD[method]


def test_the_two_reasons_are_the_archived_ones_word_for_word():
    """`git show 5a46f778:benchmarks/regbench/methods/__init__.py`, lines 202-215."""
    assert set(methods.NO_BRIGHTFIELD) == {"ashlar", "palom"}
    assert methods.NO_BRIGHTFIELD["ashlar"].startswith(
        "ASHLAR stitches unstitched tiles by one translation per tile;"
    )
    assert methods.NO_BRIGHTFIELD["ashlar"].endswith(
        "(rulings R3.8 and R3.20: ASHLAR is run on neither HyReCo nor ANHIR)"
    )
    assert "palom/color.py HaxProcessor" in methods.NO_BRIGHTFIELD["palom"]
    assert methods.NO_BRIGHTFIELD["palom"].endswith(
        "made for re-imaged slides of the same section"
    )


def test_every_method_takes_a_fluorescence_case():
    from benchmarks.regbench import METHODS

    for method in METHODS:
        assert methods.input_rule(method, "fluorescence") is None


def test_the_refusal_names_the_case_the_method_and_the_reason(toy_cases):
    case = _bf(toy_cases["rigid"])
    with pytest.raises(ValueError) as err:
        methods.require_fluorescence(case, "ashlar")
    assert str(err.value) == (
        f"{case.case_id}: ashlar has no input rule for a brightfield case: "
        f"{methods.NO_BRIGHTFIELD['ashlar']}"
    )


# ── the RGB stand-in patient ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rgb_pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("rgb_pair"), SMALL)


@pytest.fixture
def rgb_patient(tmp_path, rgb_pair):
    """Two brightfield rounds of one patient: 3-channel uint8 OME-TIFFs (R, G, B) and
    their inverted-luminance proxies."""
    base = sc.pair_case(rgb_pair, tmp_path / "fluo")
    return [
        rgb_cases.rgb_case(base, tmp_path / "bf", case_id=f"BF1_r{i}", group="BF1")
        for i in (1, 2)
    ]


def test_the_stand_in_slides_are_rgb_with_a_grey_proxy(rgb_patient):
    case = rgb_patient[0]
    with tifffile.TiffFile(case.mov_image) as tif:
        page = tif.pages[0]
        assert page.samplesperpixel == 3 and page.photometric.name == "RGB"
        assert page.dtype == np.uint8
    slide = rgb_cases.read_rgb(case.mov_image)
    proxy = tifffile.imread(case.mov_nuclear)
    assert proxy.ndim == 2 and proxy.dtype == np.uint8
    assert np.array_equal(proxy, rgb_cases.grey(np.moveaxis(slide, -1, 0)))
    assert case.modality == "brightfield" and len({c.group for c in rgb_patient}) == 1


@pytest.mark.parametrize(
    "method, names",
    [
        (
            "valis_mirage",
            ["valis_mirage_rigid", "valis_mirage_nonrigid", "valis_mirage"],
        ),
        ("valis", ["valis_rigid", "valis", "valis_micro"]),
    ],
)
def test_valis_runs_an_rgb_patient_with_the_stand_in(
    tmp_path, rgb_patient, method, names
):
    """The adapter's call (for ``valis_mirage``: the pipeline's own ``bin/register.py``)
    reaches the stand-in for an RGB patient and yields its three rows per round; nothing
    in the adapter looks for a DAPI channel on an RGB slide."""
    from benchmarks.tests.regbench import fake_valis

    rows = fake_valis.run_group(method, rgb_patient, tmp_path)
    got = [v for _, v, pm, _ in rows if pm is not None]
    # the generator yields stage by stage, every round of a stage together
    assert sorted(got) == sorted(names * len(rgb_patient))
    for _, _, pm, info in rows:
        assert pm is not None, info.get("error")
        assert info["modality"] == "brightfield"
        assert info["nuclear_channel"] is None
        assert set(info["valis_img_type"].values()) == {"brightfield"}
        assert "RGB" in info["implementation"]["input"]
        assert "not brightfield" not in info.get("problem", "")


def test_a_slide_of_three_separate_channels_is_not_rgb_to_valis(tmp_path, rgb_pair):
    """VALIS picks its brightfield processing from its reader's ``is_rgb``; three
    separate channels R, G, B are not RGB to Bio-Formats, so VALIS processes them as
    fluorescence. The rows still run, and say so (``problem``)."""
    from benchmarks.tests.regbench import fake_valis

    base = sc.pair_case(rgb_pair, tmp_path / "fluo")
    planar = [
        rgb_cases.rgb_case(base, tmp_path / "bf", case_id="BF2_r1", interleaved=False)
    ]
    rows = fake_valis.run_group("valis_mirage", planar, tmp_path / "v")
    assert rows and all(pm is not None for _, _, pm, _ in rows)
    for *_, info in rows:
        assert set(info["valis_img_type"].values()) == {"fluorescence"}
        assert "not brightfield" in info["problem"]


def test_a_fluorescence_patient_records_no_modality_key(tmp_path, rgb_pair):
    """The fluorescence rows are what they were: no new key in their record."""
    from benchmarks.tests.regbench import fake_valis

    base = sc.pair_case(rgb_pair, tmp_path / "fluo")
    rows = fake_valis.run_group("valis_mirage", [base], tmp_path / "f")
    assert rows and all(pm is not None for _, _, pm, _ in rows)
    assert all("modality" not in info for *_, info in rows)
    assert all(info["nuclear_channel"] is not None for *_, info in rows)


# ── DeeperHistReg on an RGB slide ─────────────────────────────────────────────
@pytest.fixture
def dhr_fake(monkeypatch):
    from . import fake_dhr

    fake_dhr.install(monkeypatch.setitem, initial_resolution=104)
    yield fake_dhr
    fake_dhr.configure()
    from benchmarks.regbench.methods import deeperhistreg

    deeperhistreg._LIVE.clear()


@pytest.fixture(scope="module")
def dhr_pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("dhr_rgb"), (1040, 1280), (1000, 1200))


def test_deeperhistreg_runs_an_rgb_ome_tiff_with_the_shipped_parameters(
    tmp_path, dhr_pair, dhr_fake
):
    base = sc.pair_case(dhr_pair, tmp_path / "fluo")
    case = rgb_cases.rgb_case(base, tmp_path / "bf")
    out = tmp_path / "out"
    done = methods.run_case("deeperhistreg", case, out, keep_work=True)
    assert done == ["deeperhistreg", "deeperhistreg_asdrawn"]
    work = result_dir(out, "deeperhistreg_asdrawn", case) / "work"
    with tifffile.TiffFile(work / "moving.tif") as tif:
        assert len(tif.pages) == 1 and tif.pages[0].photometric.name == "RGB"
        assert tif.pages[0].compression.name == "NONE" and tif.pages[0].is_tiled
        assert np.array_equal(
            tif.pages[0].asarray(), rgb_cases.read_rgb(case.mov_image)
        )
    (call,) = dhr_fake.calls("run_registration")
    params = call["parameters"]
    assert params["loading_params"]["pad_value"] == 255.0
    assert params["preprocessing_params"]["flip_intensity"] is True
    run = json.loads(
        (result_dir(out, "deeperhistreg_asdrawn", case) / "run.json").read_text()
    )
    info = run["info"]
    assert info["fluo"] is False and "RGB image" in info["implementation"]["input"]
    (one,) = dhr_fake.calls("apply_deformation_pyvips")
    assert one["source"] == str(work / "moving.tif") and one["pad_value"] == 255.0
    native = run["native"]
    assert native["channels"] == ["R", "G", "B"] and native["origin_xy"] == [0.0, 0.0]
    with tifffile.TiffFile(native["files"][0]) as tif:
        assert tif.pages[0].shape == (*case.ref_hw, 3)


def test_deeperhistreg_refuses_fluo_on_a_brightfield_case(tmp_path, dhr_pair, dhr_fake):
    from benchmarks.regbench.methods import deeperhistreg as dhr

    case = rgb_cases.rgb_case(sc.pair_case(dhr_pair, tmp_path / "f"), tmp_path / "bf")
    with pytest.raises(ValueError, match="fluo=1 configures DeeperHistReg"):
        next(dhr.register(case, tmp_path / "w", {"fluo": 1}))


@pytest.mark.parametrize("kind", ["jpeg", "png", "interleaved_tiff", "planar_tiff"])
def test_the_rgb_container_keeps_the_decoded_values(tmp_path, kind):
    """``rgb_single_page_tiff``: one top-level page, three 8-bit samples, the decoded
    values untouched, whatever the image's own container (a JPEG or PNG as at
    5a46f778, or an RGB TIFF read in rectangles: a slide wider and taller than a tile)."""
    from PIL import Image

    from benchmarks.regbench.methods import deeperhistreg as dhr

    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (700 // 4, 1100 // 4, 3), dtype=np.uint8)
    a = np.kron(a, np.ones((4, 4, 1), np.uint8))  # blocks survive JPEG
    src = tmp_path / {"jpeg": "s.jpg", "png": "s.png"}.get(kind, "s.ome.tif")
    if kind in ("jpeg", "png"):
        Image.fromarray(a).save(src, **({"quality": 95} if kind == "jpeg" else {}))
        want = np.asarray(Image.open(src).convert("RGB"))
    else:
        rgb_cases.write_rgb(src, np.moveaxis(a, -1, 0), kind == "interleaved_tiff")
        want = a
    meta = dhr.rgb_single_page_tiff(src, tmp_path / "out.tif")
    assert meta == {"hw": [700, 1100], "dtype": "uint8", "levels": None}
    with tifffile.TiffFile(tmp_path / "out.tif") as tif:
        assert len(tif.pages) == 1 and tif.pages[0].photometric.name == "RGB"
        assert tif.pages[0].is_tiled and tif.pages[0].compression.name == "NONE"
        assert np.array_equal(tif.pages[0].asarray(), want)
