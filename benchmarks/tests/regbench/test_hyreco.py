"""HyReCo as an external landmark set: the dataset module, its scorer, the run list.

The data are not on this machine (506 GB behind a login). Every file here is a stand-in
written by the tests in the layout of the dataset page: ``<stain>/<id>.tif`` (RGB, tiled)
and ``<stain>/<id>.csv`` (one ``x,y,z`` point per line, millimetres). What these tests
pin is the module's own rules; what the real files look like is in the module's list of
assumptions, each of which refuses a case here when it does not hold.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import (
    ALL_DATASETS,
    DATASETS,
    EXTERNAL_DATASETS,
    HYRECO_DATASETS,
    cli,
    landmarks,
    methods,
    variants,
)
from benchmarks.regbench import score as cohort
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.cases import (
    Case,
    list_cases,
    load_case,
    load_points,
    result_dir,
    write_case,
)
from benchmarks.regbench.datasets import hyreco
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.methods import stare as stare_adapter

pytestmark = pytest.mark.usefixtures("local_stages")  # STARE runs in-process

PX = 0.2425  # what the stand-in files state: within 2 % of the paper's 0.24, not equal
HW = (96, 128)


# ── stand-in files ────────────────────────────────────────────────────────────
def write_slide(path, rgb, px_um=PX):
    """An RGB TIFF as the page describes one, with resolution tags (None: without)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if px_um is not None:
        extra = {"resolution": (1e4 / px_um, 1e4 / px_um), "resolutionunit": 3}
    tifffile.imwrite(str(path), rgb, photometric="rgb", tile=(64, 64), **extra)


def write_points(path, xy_px, px_um=PX, header=False, z=True):
    lines = ["x,y,z"] if header else []
    for x, y in np.asarray(xy_px, float) * px_um / 1000.0:
        lines.append(f"{x!r},{y!r}" + (",0.0" if z else ""))
    path.write_text("\n".join(lines) + "\n")


def slide_rgb(seed, hw=HW):
    rng = np.random.default_rng(seed)
    return rng.integers(120, 256, (*hw, 3), dtype=np.uint8)


def landmarks_of(cid, n=6, hw=HW, shift=(3.0, -2.0)):
    """``(reference xy, moving xy)``: n points, the moving ones shifted."""
    rng = np.random.default_rng(1000 + int(cid))
    ref = np.column_stack(
        [rng.uniform(10, hw[1] - 10, n), rng.uniform(10, hw[0] - 10, n)]
    )
    return ref, ref - np.asarray(shift)


def write_pair(root, cid, n=6, px_um=PX, **kw):
    ref_xy, mov_xy = landmarks_of(cid, n)
    for stain, xy, seed in (("HE", ref_xy, 1), ("PHH3", mov_xy, 2)):
        write_slide(root / stain / f"{cid}.tif", slide_rgb(seed + int(cid)), px_um)
        write_points(root / stain / f"{cid}.csv", xy, px_um or 0.24, **kw)
    return ref_xy, mov_xy


def write_subset(root, subset, **kw):
    for cid in hyreco.SUBSET_IDS[subset]:
        write_pair(root, cid, **kw)
    return root


def args(cases, subset, root=None, **kw):
    base = dict(
        cases=str(cases),
        subset=subset,
        data_root=str(root) if root else None,
        ref_dir=None,
        mov_dir=None,
        proxy=None,
        reduce=None,
        settings_from=None,
        pixel_size_um=None,
        allow_incomplete=False,
        keep_shared_ids=False,
        index=None,
        workers=1,
        force=False,
    )
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Both subsets as the archives unpack: subset B in two separate trees."""
    tmp = tmp_path_factory.mktemp("hyreco_data")
    a = write_subset(tmp / "HyReCo", "a")
    b = write_subset(tmp / "HyReCo-Additional", "b")
    return types.SimpleNamespace(a=a, b=b)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, data):
    """``hyreco_a`` and ``hyreco_b`` prepared the documented way."""
    cases = tmp_path_factory.mktemp("hyreco_cases")
    assert hyreco.prepare(args(cases, "a", data.a)) == 0
    b = args(
        cases,
        "b",
        ref_dir=str(data.b / "HE"),
        mov_dir=str(data.b / "PHH3"),
        settings_from=str(cases),
    )
    assert hyreco.prepare(b) == 0
    return cases


# ── the two id lists ──────────────────────────────────────────────────────────
def test_the_id_lists_are_the_dataset_pages():
    assert len(hyreco.SUBSET_A_IDS) == len(set(hyreco.SUBSET_A_IDS)) == 9
    assert len(hyreco.SUBSET_B_IDS) == len(set(hyreco.SUBSET_B_IDS)) == 54
    assert hyreco.SHARED_IDS == ("29", "108", "361", "464")
    assert EXTERNAL_DATASETS == ("hyreco_a", "hyreco_b", "anhir_tune", "anhir")
    assert hyreco.DATASET == {"a": "hyreco_a", "b": "hyreco_b"}
    assert not set(EXTERNAL_DATASETS) & set(DATASETS)
    assert ALL_DATASETS == DATASETS + EXTERNAL_DATASETS


# ── landmark files ────────────────────────────────────────────────────────────
def test_points_are_read_in_file_order_with_or_without_z_and_header(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y,z\n1.5,2.5,0\n3.0;4.0;0\n5 6\n\n")
    xy, header = hyreco.read_points_mm(p)
    assert header is True
    assert xy.tolist() == [[1.5, 2.5], [3.0, 4.0], [5.0, 6.0]]
    p.write_text("1.5,2.5\n")
    assert hyreco.read_points_mm(p)[1] is False


@pytest.mark.parametrize(
    "text, why",
    [
        ("1,2,0\nabc,def,0\n3,4,0\n", "not numbers"),
        ("1,2,0\n\n3,4,0\n", "a blank line"),
        ("1,2,0\n7,3,4,0\n", "4 columns"),
        ("1,2,0\n3\n", "1 columns"),
        ("1,2,0\nnan,4,0\n", "not numbers"),
        ("x,y,z\n", "no landmarks"),
        ("", "no landmarks"),
    ],
)
def test_a_malformed_landmark_file_is_refused_not_skipped(tmp_path, text, why):
    p = tmp_path / "a.csv"
    p.write_text(text)
    with pytest.raises(hyreco.Refused, match=why):
        hyreco.read_points_mm(p)


def test_a_missing_landmark_file_is_refused(tmp_path):
    with pytest.raises(hyreco.Refused, match="not readable"):
        hyreco.read_points_mm(tmp_path / "nope.csv")


def one_pair(tmp_path, cid="29", **kw):
    root = tmp_path / "HyReCo"
    xy = write_pair(root, cid, **kw)
    return root, xy


def prepare_pair(tmp_path, root, cid="29", **kw):
    settings = hyreco.make_settings(kw.pop("proxy", None), kw.pop("reduce", None))
    d = hyreco.prepare_one(
        tmp_path / "cases", "a", cid, root / "HE", root / "PHH3", settings, **kw
    )
    return load_case(d)


def test_a_missing_csv_refuses_the_case(tmp_path):
    root, _ = one_pair(tmp_path)
    (root / "PHH3" / "29.csv").unlink()
    with pytest.raises(hyreco.Refused, match="29.csv: missing"):
        prepare_pair(tmp_path, root)


def test_unequal_landmark_files_are_refused_not_truncated(tmp_path):
    root, (ref_xy, mov_xy) = one_pair(tmp_path)
    write_points(root / "PHH3" / "29.csv", mov_xy[:-1])
    with pytest.raises(hyreco.Refused, match="6 landmarks .* and 5 in"):
        prepare_pair(tmp_path, root)
    assert not list_cases(tmp_path / "cases", "hyreco_a")


def test_landmarks_are_paired_by_line_and_converted_with_the_files_pixel_size(
    tmp_path,
):
    """Millimetres to pixels at the FILE's pixel size (0.2425), not the paper's 0.24:
    the i-th line of one file is the partner of the i-th line of the other."""
    root, (ref_xy, mov_xy) = one_pair(tmp_path, header=True)
    case = prepare_pair(tmp_path, root)
    pts = load_points(case)
    np.testing.assert_allclose(pts["truth_ref"], ref_xy, atol=1e-3)
    np.testing.assert_allclose(pts["truth_mov"], mov_xy, atol=1e-3)
    # each landmark keeps its own partner: the pair differs by the shift, line by line
    np.testing.assert_allclose(
        pts["truth_ref"] - pts["truth_mov"], [[3.0, -2.0]] * 6, atol=1e-3
    )
    assert case.pixel_size_um == pytest.approx(PX, rel=1e-5)
    assert case.extra["pixel_size_source"] == "file"
    assert case.extra["landmark_header"] == {"ref": True, "mov": True}
    assert case.extra["n_landmarks"] == 6
    # the paper's two digits would have put every landmark about 1 % further out
    wrong = ref_xy * PX / hyreco.PIXEL_SIZE_UM
    assert np.abs(wrong - pts["truth_ref"]).max() > 0.5


def test_the_case_is_a_brightfield_pair_with_the_he_slide_as_reference(tmp_path):
    root, _ = one_pair(tmp_path)
    case = prepare_pair(tmp_path, root)
    assert case.dataset == "hyreco_a" and case.modality == "brightfield"
    assert case.group == case.case_id == "29"  # each pair is its own patient
    assert case.ref_image == str(root / "HE" / "29.tif")
    assert case.mov_image == str(root / "PHH3" / "29.tif")
    assert case.ref_hw == list(HW) and case.mov_hw == list(HW)
    assert case.ref_nuclear.endswith("HE/29.luminance.r1.ome.tif")
    pts = load_points(case)
    assert set(pts) == {"truth_mov", "truth_ref"}  # no nuclei: landmarks only
    assert case.extra["role"] == "tuning" and "CC BY-SA" in case.extra["licence"]


def test_a_file_without_a_pixel_size_falls_back_to_the_paper_and_says_so(tmp_path):
    root, (ref_xy, _) = one_pair(tmp_path, px_um=None)
    case = prepare_pair(tmp_path, root)
    assert case.pixel_size_um == 0.24
    assert case.extra["pixel_size_source"].startswith("paper, two digits")
    assert case.extra["pixel_size_file_um"] == {"ref": None, "mov": None}
    np.testing.assert_allclose(load_points(case)["truth_ref"], ref_xy, atol=1e-3)


def test_a_pixel_size_far_from_the_papers_is_refused(tmp_path):
    root, _ = one_pair(tmp_path, px_um=0.5)
    with pytest.raises(hyreco.Refused, match="not within 2% of the published 0.24"):
        prepare_pair(tmp_path, root)
    # a stated value is checked the same way, and wins over the file when it passes
    with pytest.raises(hyreco.Refused, match="command line"):
        prepare_pair(tmp_path, root, pixel_size_um=0.3)


def test_a_given_pixel_size_overrides_the_file(tmp_path):
    root, (ref_xy, _) = one_pair(tmp_path)
    case = prepare_pair(tmp_path, root, pixel_size_um=0.24)
    assert case.pixel_size_um == 0.24
    assert case.extra["pixel_size_source"].startswith("command line")
    assert case.extra["pixel_size_file_um"]["ref"] == pytest.approx(PX, rel=1e-5)
    np.testing.assert_allclose(
        load_points(case)["truth_ref"], ref_xy * PX / 0.24, atol=1e-3
    )


def test_the_two_slides_of_a_pair_must_share_a_pixel_size(tmp_path):
    root, _ = one_pair(tmp_path)
    write_slide(root / "PHH3" / "29.tif", slide_rgb(5), 0.2400)
    with pytest.raises(hyreco.Refused, match="one pixel size per pair"):
        prepare_pair(tmp_path, root)


def test_landmarks_outside_the_image_refuse_the_case(tmp_path):
    root, (ref_xy, _) = one_pair(tmp_path)
    far = ref_xy.copy()
    far[2] = [HW[1] + 40.0, 10.0]
    write_points(root / "HE" / "29.csv", far)
    with pytest.raises(hyreco.Refused, match="pixel size or the origin"):
        prepare_pair(tmp_path, root)


def test_a_slide_that_is_not_8_bit_rgb_is_refused(tmp_path):
    root, _ = one_pair(tmp_path)
    grey = np.zeros(HW, np.uint8)
    tifffile.imwrite(str(root / "HE" / "29.tif"), grey, tile=(64, 64))
    with pytest.raises(hyreco.Refused, match="not an RGB image"):
        prepare_pair(tmp_path, root)
    deep = np.zeros((*HW, 3), np.uint16)
    tifffile.imwrite(str(root / "HE" / "29.tif"), deep, photometric="rgb")
    with pytest.raises(hyreco.Refused, match="expected 8-bit RGB"):
        prepare_pair(tmp_path, root)


def test_the_check_by_eye_shows_each_slide_around_its_own_landmark(tmp_path, capsys):
    root, (ref_xy, mov_xy) = one_pair(tmp_path)
    case = prepare_pair(tmp_path, root)
    argv = ["--cases", str(tmp_path / "cases"), "--dataset", "hyreco_a"]
    argv += ["--case-id", "29", "--out", str(tmp_path / "look"), "--n", "2"]
    assert hyreco.main([*argv, "--half", "8"]) == 0
    paths = capsys.readouterr().out.split()
    assert [Path(p).name for p in paths] == [
        "hyreco_a_29_landmark0.tif",
        "hyreco_a_29_landmark1.tif",
    ]
    both = tifffile.imread(paths[0])
    assert both.shape == (16, 32, 3)  # H&E left, PHH3 right
    for side, slide, xy in (
        (0, case.ref_image, ref_xy[0]),
        (1, case.mov_image, mov_xy[0]),
    ):
        x, y = (int(round(v)) for v in xy)
        crop = both[:, side * 16 : side * 16 + 16]
        # the crop's centre pixel is the slide's pixel at the landmark, uncovered
        np.testing.assert_array_equal(crop[8, 8], tifffile.imread(slide)[y, x])
        np.testing.assert_array_equal(crop[8 + 3, 8], (255, 220, 0))  # the cross


# ── the proxy ─────────────────────────────────────────────────────────────────
def test_the_luminance_proxy_is_inverted_rec601(tmp_path):
    rgb = np.full((*HW, 3), 255, np.uint8)
    rgb[10:20, 30:40] = (90, 40, 140)  # a haematoxylin-like blob
    write_slide(tmp_path / "HE" / "1.tif", rgb)
    out = tmp_path / "p.ome.tif"
    hw, proxy_hw = hyreco.write_proxy(
        tmp_path / "HE" / "1.tif", out, hyreco.make_settings(), PX
    )
    plane = tifffile.imread(str(out)).squeeze()
    assert hw == proxy_hw == HW and plane.dtype == np.uint8
    assert plane[0, 0] == 0  # white glass is black
    want = 255 - round(0.299 * 90 + 0.587 * 40 + 0.114 * 140)
    assert plane[15, 35] == want and want > 150  # nuclei bright
    with tifffile.TiffFile(str(out)) as tif:
        assert tif.pages[0].compression != 1  # compressed: most of a slide is glass
        assert tif.pages[0].is_tiled


def test_a_reduced_proxy_is_the_block_average(tmp_path):
    rgb = np.full((*HW, 3), 255, np.uint8)
    rgb[0, 0] = (0, 0, 0)  # one black pixel in the first 2 x 2 block
    write_slide(tmp_path / "HE" / "1.tif", rgb)
    out = tmp_path / "p.ome.tif"
    settings = hyreco.make_settings(reduce=2)
    hw, proxy_hw = hyreco.write_proxy(tmp_path / "HE" / "1.tif", out, settings, PX)
    plane = tifffile.imread(str(out)).squeeze()
    assert hw == HW and proxy_hw == (48, 64) == plane.shape
    assert plane[0, 0] == round(255 / 4) and plane[0, 1] == 0
    from benchmarks.regbench.imageio import pixel_size_um

    assert pixel_size_um(out) == pytest.approx(2 * PX, rel=1e-5)


def test_the_haematoxylin_proxy_is_16_bit_and_bright_on_the_stain(tmp_path):
    rgb = np.full((*HW, 3), 255, np.uint8)
    rgb[10:20, 30:40] = (90, 40, 140)
    write_slide(tmp_path / "HE" / "1.tif", rgb)
    out = tmp_path / "p.ome.tif"
    settings = hyreco.make_settings("haematoxylin")
    hyreco.write_proxy(tmp_path / "HE" / "1.tif", out, settings, PX)
    plane = tifffile.imread(str(out)).squeeze()
    assert plane.dtype == np.uint16
    assert plane[15, 35] > plane[0, 0] + 100


def test_unknown_settings_are_refused():
    with pytest.raises(SystemExit, match="--proxy"):
        hyreco.make_settings("eosin")
    with pytest.raises(SystemExit, match="--reduce"):
        hyreco.make_settings(reduce=3)
    a, b = hyreco.make_settings(), hyreco.make_settings(reduce=2)
    assert hyreco.settings_hash(a) != hyreco.settings_hash(b)


# ── the split: tuned on A, reported on B ──────────────────────────────────────
def test_both_subsets_prepare_under_their_own_names(prepared):
    a, b = list_cases(prepared, "hyreco_a"), list_cases(prepared, "hyreco_b")
    assert sorted(c.case_id for c in a) == sorted(hyreco.SUBSET_A_IDS)
    assert sorted(c.case_id for c in b) == sorted(hyreco.SUBSET_B_IDS)
    assert {c.extra["role"] for c in a} == {"tuning"}
    assert {c.extra["role"] for c in b} == {"report"}
    assert all(c.extra["subset_complete"] for c in a + b)
    # the four ids subset A also has are prepared and left out of the report set
    out = sorted(c.case_id for c in b if not c.extra["report_set"])
    assert out == sorted(hyreco.SHARED_IDS)
    assert sum(c.extra["report_set"] for c in b) == 50
    assert all(c.extra["shared_id_with_subset_a"] != c.extra["report_set"] for c in b)
    # subset B carries subset A's settings and says where they were chosen
    assert {c.extra["settings_sha256"] for c in a + b} == {
        hyreco.settings_hash(hyreco.make_settings())
    }
    assert {c.extra["settings_from"] for c in b} == {str(Path(prepared).resolve())}
    assert {c.extra["settings_from"] for c in a} == {""}


def test_subset_bs_folder_cannot_be_prepared_as_the_tuning_set(tmp_path, data):
    b = args(tmp_path, "a", ref_dir=str(data.b / "HE"), mov_dir=str(data.b / "PHH3"))
    with pytest.raises(SystemExit, match="this is not subset A"):
        hyreco.prepare(b)
    assert not list_cases(tmp_path, "hyreco_a")


def test_subset_as_folder_cannot_be_prepared_as_the_report_set(
    tmp_path, data, prepared
):
    """Subset A's ids are not all in subset B's list, so its folder is refused under
    the report set's name even with every other argument in order."""
    a = args(tmp_path, "b", data.a, settings_from=str(prepared))
    with pytest.raises(SystemExit, match="this is not subset B"):
        hyreco.prepare(a)
    assert not list_cases(tmp_path, "hyreco_b")


@pytest.mark.parametrize("extra", [{"proxy": "haematoxylin"}, {"reduce": 2}])
def test_subset_b_takes_no_setting_of_its_own(tmp_path, data, prepared, extra):
    b = args(tmp_path, "b", data.b, settings_from=str(prepared), **extra)
    with pytest.raises(SystemExit, match="nothing is chosen on subset B"):
        hyreco.prepare(b)
    assert not list_cases(tmp_path, "hyreco_b")


def test_subset_b_needs_the_prepared_tuning_set(tmp_path, data):
    with pytest.raises(SystemExit, match="needs --settings-from"):
        hyreco.prepare(args(tmp_path, "b", data.b))
    # a tuning set that is not all nine pairs has not chosen anything yet
    part = tmp_path / "part"
    assert hyreco.prepare(args(part, "a", data.a, index=0)) == 0
    with pytest.raises(SystemExit, match="not fully prepared"):
        hyreco.prepare(args(tmp_path, "b", data.b, settings_from=str(part)))
    assert not list_cases(tmp_path, "hyreco_b")


def test_subset_b_is_prepared_with_the_tuning_sets_settings(tmp_path, data):
    tuned = tmp_path / "tuned"
    assert hyreco.prepare(args(tuned, "a", data.a, proxy="haematoxylin", reduce=2)) == 0
    b = args(tmp_path / "cases", "b", data.b, settings_from=str(tuned), index=0)
    assert hyreco.prepare(b) == 0
    (case,) = list_cases(tmp_path / "cases", "hyreco_b")
    assert case.case_id == hyreco.SUBSET_B_IDS[0]
    assert case.extra["settings"] == {
        "proxy": "haematoxylin",
        "reduce": 2,
        "version": hyreco.SETTINGS_VERSION,
        "pixel_size_um": None,
        "keep_shared_ids": False,
    }
    assert case.ref_nuclear.endswith("HE/8.haematoxylin.r2.ome.tif")
    assert case.extra["proxy_hw"]["ref"] == [48, 64] and case.ref_hw == list(HW)


def test_a_second_setting_on_subset_b_is_refused(tmp_path, data, prepared):
    """Subset B prepared once with the tuning set's settings cannot be prepared again
    with another tuning set's: that would be choosing on the report set."""
    cases = tmp_path / "cases"
    b = args(cases, "b", data.b, settings_from=str(prepared), index=0)
    assert hyreco.prepare(b) == 0
    other = tmp_path / "other"
    assert hyreco.prepare(args(other, "a", data.a, reduce=2)) == 0
    again = args(cases, "b", data.b, settings_from=str(other), index=1, force=True)
    with pytest.raises(SystemExit, match="A second setting on subset B is tuning"):
        hyreco.prepare(again)
    assert len(list_cases(cases, "hyreco_b")) == 1


def test_a_tuning_set_of_two_settings_chooses_nothing(tmp_path, data):
    mixed = tmp_path / "mixed"
    assert hyreco.prepare(args(mixed, "a", data.a, index=0)) == 0
    for i in range(1, 9):
        assert hyreco.prepare(args(mixed, "a", data.a, index=i, reduce=2)) == 0
    with pytest.raises(SystemExit, match="2 different settings"):
        hyreco.tuned_settings(mixed)


def test_an_incomplete_subset_is_refused_unless_allowed_and_then_marked(tmp_path):
    root = tmp_path / "HyReCo"
    write_pair(root, "29")
    write_pair(root, "108")
    with pytest.raises(SystemExit, match="7 are not in both"):
        hyreco.prepare(args(tmp_path / "c", "a", root))
    assert hyreco.prepare(args(tmp_path / "c", "a", root, allow_incomplete=True)) == 0
    cases = list_cases(tmp_path / "c", "hyreco_a")
    assert [c.case_id for c in cases] == ["29", "108"]
    assert not any(c.extra["subset_complete"] for c in cases)


def test_only_the_he_and_phh3_folders_are_a_pair(tmp_path, data):
    """Subset A also has CD8, CD45 and KI67: consecutive sections, not this module's."""
    cd8 = tmp_path / "CD8"
    cd8.mkdir()
    bad = args(tmp_path, "a", ref_dir=str(data.a / "HE"), mov_dir=str(cd8))
    with pytest.raises(SystemExit, match="expected the dataset's `PHH3` folder"):
        hyreco.prepare(bad)
    with pytest.raises(SystemExit, match="give --data-root"):
        hyreco.prepare(args(tmp_path, "a", ref_dir=str(data.a / "HE")))


def test_keep_shared_ids_puts_the_four_back(tmp_path, data, prepared):
    b = args(
        tmp_path,
        "b",
        data.b,
        settings_from=str(prepared),
        index=hyreco.SUBSET_B_IDS.index("29"),
        keep_shared_ids=True,
    )
    # the flag is a setting of the A/B pair: B may not use it unless A carries it
    with pytest.raises(SystemExit, match="keep_shared_ids"):
        hyreco.prepare(b)
    tuned = tmp_path / "tuned"
    assert hyreco.prepare(args(tuned, "a", data.a, keep_shared_ids=True)) == 0
    b.settings_from = str(tuned)
    assert hyreco.prepare(b) == 0
    (case,) = list_cases(tmp_path, "hyreco_b")
    assert case.case_id == "29" and case.extra["report_set"] is True
    assert case.extra["shared_id_with_subset_a"] is True
    assert case.extra["settings"]["keep_shared_ids"] is True


def test_a_refused_case_makes_prepare_fail_and_the_others_are_kept(tmp_path, capsys):
    root = write_subset(tmp_path / "HyReCo", "a")
    (root / "PHH3" / "361.csv").write_text("1,2,0\noops\n")
    assert hyreco.prepare(args(tmp_path / "c", "a", root)) == 1
    assert "hyreco_a/361: REFUSED" in capsys.readouterr().out
    assert len(list_cases(tmp_path / "c", "hyreco_a")) == 8


# ── the scorer ────────────────────────────────────────────────────────────────
def shifter(dx, direction=MOV_TO_REF, name="m"):
    """A method whose map adds ``dx`` px in x: a fixed, known error."""

    def register(case, work, opts):
        yield (
            name,
            PointMap(lambda xy: np.asarray(xy, float) + [dx, 0.0], direction),
            {},
        )

    return types.SimpleNamespace(
        NAME=name, SCOPE="pair", VARIANTS=(name,), register=register
    )


def crafted(tmp_path, dataset="hyreco_a", spec=(("29", 2, 10.0), ("108", 6, 1.0))):
    """Cases whose landmarks coincide (initial error 0), so a method that shifts by
    ``d`` px has exactly ``d`` px of error on every landmark of that pair."""
    cases = []
    settings = hyreco.make_settings()
    for cid, n, _ in spec:
        xy = np.column_stack([np.arange(n) * 5.0 + 20, np.full(n, 30.0)])
        case = Case(
            dataset=dataset,
            case_id=cid,
            group=cid,
            ref_nuclear="",
            mov_nuclear="",
            ref_image="",
            mov_image="",
            modality="brightfield",
            diagonal=1000.0,
            pixel_size_um=0.25,
            ref_hw=[600, 800],
            mov_hw=[600, 800],
            extra={
                "report_set": cid not in hyreco.SHARED_IDS,
                "shared_id_with_subset_a": cid in hyreco.SHARED_IDS,
                "settings": settings,
                "settings_sha256": hyreco.settings_hash(settings),
                "settings_from": "",
                "n_landmarks": n,
                "pixel_size_source": "file",
            },
        )
        cases.append(load_case(write_case(tmp_path / "cases", case, xy, xy)))
    return cases, {cid: d for cid, _, d in spec}


def run_crafted(tmp_path, cases, shifts, name="m", direction=MOV_TO_REF):
    out = tmp_path / "out"
    for case in cases:
        got = methods.run_case(
            shifter(shifts[case.case_id], direction, name), case, out
        )
        assert got == [name]
    return out


def row(rows, **want):
    (got,) = [r for r in rows if all(r[k] == v for k, v in want.items())]
    return got


def test_error_is_in_micrometres_through_the_methods_own_map(tmp_path):
    cases, shifts = crafted(tmp_path)
    out = run_crafted(tmp_path, cases, shifts)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    pair = row(got["pairs"], method="m", case_id="29")
    assert pair["tre_um_p50"] == pytest.approx(10.0 * 0.25)  # 10 px at 0.25 um
    assert pair["n_probes"] == 2 and pair["frame"] == "reference"
    assert pair["imputed_initial"] is False
    assert row(got["pairs"], method="m", case_id="108")["tre_um_p50"] == pytest.approx(
        0.25
    )
    # the unregistered pose is a row of its own: these landmarks coincide
    assert row(got["pairs"], method="initial", case_id="29")["tre_um_p50"] == 0.0
    pts = [p for p in got["points"] if p["method"] == "m" and p["case_id"] == "29"]
    assert [p["tre_um"] for p in pts] == pytest.approx([2.5, 2.5])


def test_a_ref_to_mov_map_is_scored_in_its_own_direction(tmp_path):
    cases, shifts = crafted(tmp_path)
    out = run_crafted(tmp_path, cases, shifts, direction=REF_TO_MOV)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    pair = row(got["pairs"], method="m", case_id="29")
    assert pair["frame"] == "moving" and pair["tre_um_p50"] == pytest.approx(2.5)


def test_the_pooled_median_is_not_the_median_of_pair_medians(tmp_path):
    """Two landmarks at 2.5 um in one pair, six at 0.25 um in the other: the median
    over pairs of each pair's median is 1.375 um, the median over all eight landmarks
    is 0.25 um. Both are reported, under different names."""
    cases, shifts = crafted(tmp_path)
    out = run_crafted(tmp_path, cases, shifts)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    s = row(got["summary"], method="m", set="tuning")
    assert s["n_pairs"] == 2 and s["n_landmarks"] == 8 and s["n_ok"] == 2
    assert s["per_pair_median_tre_um_med"] == pytest.approx((2.5 + 0.25) / 2)
    assert s["per_pair_median_tre_um_min"] == pytest.approx(0.25)
    assert s["per_pair_median_tre_um_max"] == pytest.approx(2.5)
    assert s["pooled_median_tre_um"] == pytest.approx(0.25)
    assert s["pooled_p90_tre_um"] == pytest.approx(2.5)
    assert s["pooled_frac_under_1um"] == pytest.approx(6 / 8)
    assert s["pooled_frac_improved"] == 0.0  # the landmarks coincided before
    assert s["initial_pooled_median_tre_um"] == 0.0


def test_for_one_pair_the_two_aggregations_agree(tmp_path):
    cases, shifts = crafted(tmp_path, spec=(("29", 5, 3.0),))
    out = run_crafted(tmp_path, cases, shifts)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    s = row(got["summary"], method="m")
    assert s["pooled_median_tre_um"] == s["per_pair_median_tre_um_med"] == 0.75


def test_a_run_that_failed_or_never_ran_is_a_row_at_the_unregistered_pose(tmp_path):
    cases, shifts = crafted(tmp_path)
    out = tmp_path / "out"
    assert methods.run_case(shifter(4.0), cases[0], out) == ["m"]
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a", expected=["m", "ghost"])
    # never started: the pair is listed apart and is in no median of the method
    missing = row(got["pairs"], method="m", case_id="108")
    assert missing["not_run"] is True and missing["cause"] == "not_run"
    s = row(got["summary"], method="m")
    assert (s["n_ok"], s["n_pairs"], s["n_not_run"], s["n_landmarks"]) == (1, 1, 1, 2)
    assert not [r for r in got["summary"] if r["method"] == "ghost"]
    assert row(got["failures"], method="ghost")["rounds_not_run"] == 2


def test_a_point_a_method_could_not_map_counts_against_it(tmp_path):
    cases, _ = crafted(tmp_path, spec=(("29", 3, 0.0),))

    def register(case, work, opts):
        def fn(xy):
            out = np.asarray(xy, float).copy()
            out[1:] = np.nan if len(out) == 3 else out[1:]
            return out

        yield "m", PointMap(fn, MOV_TO_REF), {}

    lost = types.SimpleNamespace(
        NAME="m", SCOPE="pair", VARIANTS=("m",), register=register
    )
    out = tmp_path / "out"
    assert methods.run_case(lost, cases[0], out) == ["m"]
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    s = row(got["summary"], method="m")
    assert s["n_nonfinite"] == 2 and math.isinf(s["pooled_median_tre_um"])


def test_the_report_set_leaves_out_the_shared_ids_and_all_pairs_keeps_them(tmp_path):
    spec = (("29", 2, 10.0), ("8", 6, 1.0), ("31", 6, 1.0))
    cases, shifts = crafted(tmp_path, "hyreco_b", spec)
    tuned = tmp_path / "tuned"  # the prepared tuning set the report set names
    crafted(tuned, "hyreco_a", tuple((i, 2, 1.0) for i in hyreco.SUBSET_A_IDS))
    for case in cases:  # as `prepare --subset b` records it
        case.extra["settings_from"] = str(tuned / "cases")
        write_case(tmp_path / "cases", case, *[load_points(case)["truth_mov"]] * 2)
    out = run_crafted(tmp_path, cases, shifts)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_b")
    report = row(got["summary"], method="m", set="report")
    every = row(got["summary"], method="m", set="all_pairs")
    assert (report["n_pairs"], every["n_pairs"]) == (2, 3)
    assert (report["n_landmarks"], every["n_landmarks"]) == (12, 14)
    assert report["per_pair_median_tre_um_max"] == pytest.approx(0.25)
    assert every["per_pair_median_tre_um_max"] == pytest.approx(2.5)
    text = (Path(got["tables"]) / "summary.md").read_text()
    assert "NOT like for like: 3 of 54 pairs" in text
    assert "REPORT SET. Nothing was chosen on it." in text


def test_the_tables_are_the_sets_own(tmp_path):
    cases, shifts = crafted(tmp_path)
    out = run_crafted(tmp_path, cases, shifts)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    tables = Path(got["tables"])
    assert tables == out / "tables" / "hyreco_a"
    names = sorted(p.name for p in tables.iterdir())
    assert names == sorted(
        [
            "summary.md",
            "hyreco_pairs.csv",
            "hyreco_points.csv",
            "hyreco_summary.csv",
            "hyreco_published.csv",
            "hyreco_failures.csv",
            "hyreco_resources.csv",
            "hyreco_summary_resources.csv",
            "hyreco_excluded.csv",
        ]  # hyreco_not_run.csv only when there is something in it
    )
    assert not list((out / "tables").glob("*.csv"))  # none of the cohort's tables
    text = (tables / "summary.md").read_text()
    assert "TUNING SET. Nothing on this page is a result." in text
    assert "0.57 +- 0.36 um" in text and "no ranking is claimed" in text
    assert "tissue-level block" in text and "off-seam rows" in text
    left = text.split("that are not run here")[1].split("## Not computed")[0]
    for name in ("valis_mirage", "deeperhistreg_hr_gpu", "ashlar", "palom"):
        assert f"`{name}`" in left
    # one line per reason: a method without a rule, or a run HyReCo does not take
    # (a NOT_ON rule that matches no ACTIVE run, as the paused preprocessing row's, says nothing)
    live = [
        rule
        for rule in variants.NOT_ON["hyreco_a"]
        if any(rule[0](r) for r in variants.active_runs())
    ]
    assert left.count("\n- ") == len(methods.NO_BRIGHTFIELD) + len(live)


def test_the_published_number_is_quoted_with_its_source_and_core_is_not():
    b = [r for r in landmarks.PUBLISHED if r["dataset"] == "hyreco_b"]
    best = row(b, method="HistokatFusion, deformable (Fraunhofer MEVIS)")
    assert best["pooled_median_tre_um"] == 0.9 and best["n_pairs"] == 54
    assert best["n_landmarks"] == 2357
    assert best["status"] == "published, not reproduced here"
    assert (
        "10.1117/1.JMI.10.6.067501" in best["source"] and "[PARTIAL" in best["source"]
    )
    everything = json.dumps(landmarks.PUBLISHED).lower()
    assert "core" not in everything.replace("score", "")
    assert all(r["pooled_median_tre_um"] != 0.41 for r in landmarks.PUBLISHED)
    assert all("[" in r["source"] for r in landmarks.PUBLISHED)  # a reading-depth tag
    assert landmarks.ANNOTATION_FLOOR["inter_observer_um"] == 0.57
    assert "[PARTIAL" in landmarks.ANNOTATION_FLOOR["source"]


def test_with_all_54_pairs_the_like_for_like_line_stands_beside_the_quote(
    tmp_path, prepared
):
    out = tmp_path / "out"
    m = shifter(2.0, name="stare")
    for case in list_cases(prepared, "hyreco_b"):
        assert methods.run_case(m, case, out) == ["stare"]
    got = landmarks.score(prepared, out, "hyreco_b", primary=["stare"])
    every = row(got["summary"], method="stare", set="all_pairs")
    report = row(got["summary"], method="stare", set="report")
    assert (every["n_pairs"], report["n_pairs"]) == (54, 50)
    assert every["n_landmarks"] == 54 * 6
    like = landmarks.like_for_like(got["summary"], "hyreco_b")
    assert [r["method"] for r in like] == [
        "initial",
        "stare",
        "HistokatFusion, deformable (Fraunhofer MEVIS)",
        "HistokatFusion, affine only (Fraunhofer MEVIS)",
    ]
    assert like[1]["n_pairs"] == 54 and like[1]["status"].startswith("run here")
    assert like[2]["status"] == "published, not reproduced here"
    text = (Path(got["tables"]) / "summary.md").read_text()
    assert "NOT like for like" not in text
    assert "## Like for like: pooled median over all landmarks (MTRE)" in text
    assert "324 landmark pairs found in the files (published counts" in text


def test_cases_of_two_settings_are_not_scored_together(tmp_path):
    cases, shifts = crafted(tmp_path)
    other = hyreco.make_settings(reduce=2)
    cases[1].extra.update(settings=other, settings_sha256=hyreco.settings_hash(other))
    xy = load_points(cases[1])["truth_mov"]
    write_case(tmp_path / "cases", cases[1], xy, xy)
    with pytest.raises(SystemExit, match="2 different settings"):
        landmarks.score(tmp_path / "cases", tmp_path / "out", "hyreco_a")


def test_a_report_set_that_does_not_follow_its_tuning_set_is_refused(tmp_path, data):
    """`hyreco_b` cases name the tuning set their settings came from; when that set
    is prepared again with other settings, the report set no longer follows it."""
    tuned, cases = tmp_path / "tuned", tmp_path / "cases"
    assert hyreco.prepare(args(tuned, "a", data.a)) == 0
    b = args(cases, "b", data.b, settings_from=str(tuned), index=0)
    assert hyreco.prepare(b) == 0
    assert landmarks.score(cases, tmp_path / "out", "hyreco_b")["summary"]
    assert hyreco.prepare(args(tuned, "a", data.a, reduce=2, force=True)) == 0
    with pytest.raises(SystemExit, match="does not follow the tuning set"):
        landmarks.score(cases, tmp_path / "out", "hyreco_b")
    # and cases that were not written by `prepare hyreco --subset b` name none
    crafted_cases, _ = crafted(tmp_path / "x", "hyreco_b", (("8", 2, 1.0),))
    with pytest.raises(SystemExit, match="do not name one prepared tuning set"):
        landmarks.score(tmp_path / "x" / "cases", tmp_path / "o", "hyreco_b")


# ── never in the cohort's tables ──────────────────────────────────────────────
def test_the_cohorts_score_never_reads_an_external_set(tmp_path, cases_root, toy_cases):
    """With HyReCo cases and results beside the cohort's, the cohort's default score
    has no row of them, and writes nothing of theirs."""
    import shutil

    root = tmp_path / "cases"
    shutil.copytree(cases_root / "multiplex" / "rigid", root / "multiplex" / "rigid")
    crafted(tmp_path)  # writes <tmp>/cases/hyreco_a/{29,108}
    out = tmp_path / "out"
    m = shifter(1.0, name="m")
    for ds in ("multiplex", "hyreco_a"):
        for case in list_cases(root, ds):
            assert methods.run_case(m, case, out) == ["m"]
    got = cohort.score(root, out, with_tissue=False)
    for key in ("cells", "resources", "failures", "summary_cells"):
        assert {r["dataset"] for r in got[key]} == {"multiplex"}, key
    assert not (out / "tables" / "hyreco_a").exists()
    assert "hyreco" not in (out / "tables" / "summary.md").read_text().lower()


def test_score_takes_an_external_set_alone(tmp_path):
    assert cli.split_datasets(None) == (None, None)
    assert cli.split_datasets(["multiplex"]) == (["multiplex"], None)
    assert cli.split_datasets(["hyreco_b"]) == (None, "hyreco_b")
    for mixed in (["hyreco_b", "multiplex"], ["hyreco_a", "hyreco_b"]):
        with pytest.raises(SystemExit, match="scored alone"):
            cli.split_datasets(mixed)
    with pytest.raises(SystemExit, match="scored alone"):
        cli.main(
            ["score", "--cases", str(tmp_path), "--out", str(tmp_path)]
            + ["--dataset", "hyreco_b", "multiplex"]
        )


def test_the_cli_scores_an_external_set_into_its_own_tables(tmp_path, capsys):
    cases, shifts = crafted(tmp_path)
    out = run_crafted(tmp_path, cases, shifts, name="stare")
    argv = ["--cases", str(tmp_path / "cases"), "--out", str(out)]
    assert cli.main(["score", *argv, "--dataset", "hyreco_a"]) == 0
    text = (out / "tables" / "hyreco_a" / "summary.md").read_text()
    # the set's own expected rows: VALIS 1.2.0's are "not run", ASHLAR's are not rows
    assert "| valis120_hires | not_run |" in text and "| ashlar |" not in text
    assert not (out / "tables" / "summary.md").exists()
    with pytest.raises(SystemExit, match="tissue: not computed for hyreco_a"):
        cli.main(["tissue", *argv, "--dataset", "hyreco_a"])
    capsys.readouterr()
    assert cli.main(["list", "--cases", argv[1], "--dataset", "hyreco_a"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_prepare_through_the_command_line(tmp_path, data, capsys):
    cases = tmp_path / "cases"
    base = ["prepare", "hyreco", "--cases", str(cases)]
    assert cli.main([*base, "--subset", "a", "--data-root", str(data.a)]) == 0
    assert "hyreco_a (tuning set): 9/9" in capsys.readouterr().out
    b = [*base, "--subset", "b", "--ref-dir", str(data.b / "HE")]
    b += ["--mov-dir", str(data.b / "PHH3"), "--index", "3"]
    with pytest.raises(SystemExit, match="needs --settings-from"):
        cli.main(b)
    assert cli.main([*b, "--settings-from", str(cases)]) == 0
    assert "hyreco_b (report set): 1/1" in capsys.readouterr().out


# ── which methods run on it ───────────────────────────────────────────────────
def _stare_of(ds):
    """The STARE runs of a dataset's regime, from the tables (stride x level), in RUNS order."""
    from benchmarks.regbench import regimes

    reg = regimes.regime_of(ds)
    here = {
        variants.stare_cell(s, k, lv)[0]
        for k, lv in regimes.stare_levels(reg)
        for s in variants.STARE_STRIDES
    }
    return [
        n for n in (variants.run_name(r) for r in variants.active_runs()) if n in here
    ]


def _left_out_of_hyreco():
    """The active runs HyReCo does not take: a method without a brightfield rule,
    DeeperHistReg (named), and not the regimes' own filtering."""
    return {
        variants.run_name(r)
        for r in variants.active_runs()
        if r.method in methods.NO_BRIGHTFIELD or r.method == "deeperhistreg"
    }


def test_the_run_list_of_an_external_set():
    for ds in HYRECO_DATASETS:
        names = [variants.run_name(r) for r in variants.runs_for(ds)]
        stare = _stare_of(ds)
        assert stare and "stare" in stare
        assert names == [*stare, "valis120", "valis120_hires"]
        assert variants.primary_for(ds) == ["stare", "valis120_hires"]
        assert variants.expected_variants(ds) == [
            *[v for n in stare for v in (f"{n}_rigid", n)],
            *methods.variant_names("valis", "valis120"),
            *methods.variant_names("valis", "valis120_hires"),
        ]
        left = dict(variants.excluded_for(ds))
        assert set(left) == _left_out_of_hyreco() and not set(left) & set(names)
        assert all(len(why) > 40 for why in left.values())
        table = variants.runs_table(dataset=ds)
        assert [r[0] for r in table] == names
        assert all(len(r) == len(variants.RUNS_COLUMNS) for r in table)
    # the cohort's list is the regime's: a fluorescence dataset takes no RGB-proxy STARE run
    assert variants.excluded_for("multiplex") == []
    assert variants.runs_for("multiplex") == tuple(
        r
        for r in variants.active_runs()
        if r.regimes is None or "cyclic_if" in r.regimes
    )
    assert len(variants.runs_table()) == len(variants.runs_for(None))
    assert len(variants.expected_variants()) == sum(
        len(variants.variants_of(r)) for r in variants.runs_for(None)
    )
    assert variants.primary_for() == list(variants.PRIMARY)


def test_the_run_list_names_every_method_once():
    """A method is run on brightfield or named with a reason, never neither."""
    methods_of = {r.method for r in variants.active_runs()}
    with_rule = {r.method for r in variants.runs_for("hyreco_b")}
    assert with_rule == {"stare", "valis"}
    # DeeperHistReg has a rule (ANHIR) and is left out of HyReCo by name
    assert with_rule | set(methods.NO_BRIGHTFIELD) | {"deeperhistreg"} == methods_of
    assert "deeperhistreg_hr_gpu" in dict(variants.excluded_for("hyreco_b"))
    assert not with_rule & set(methods.NO_BRIGHTFIELD)


def test_runs_prints_an_external_sets_list(capsys):
    assert cli.main(["runs", "--dataset", "hyreco_b"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split("\t")[0] for ln in lines] == [
        *_stare_of("hyreco_b"),
        "valis120",
        "valis120_hires",
    ]
    assert cli.main(["runs", "--dataset", "hyreco_b", "--excluded"]) == 0
    left = capsys.readouterr().out.splitlines()
    assert {ln.split("\t")[0] for ln in left} == _left_out_of_hyreco()
    assert cli.main(["runs"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(variants.runs_for(None))


def brightfield(tmp_path, **kw):
    cases, _ = crafted(tmp_path, spec=(("29", 2, 0.0),))
    return dataclasses.replace(cases[0], **kw)


@pytest.mark.parametrize("method", sorted(methods.NO_BRIGHTFIELD))
def test_a_method_without_a_brightfield_rule_refuses_the_case(tmp_path, method):
    """Refused by the adapter itself, before it imports its package: a brightfield case
    handed to it by hand fails with the reason, and never runs on a wrong channel."""
    case = brightfield(tmp_path)
    mod = methods.load(method)
    gen = (
        mod.register(case, tmp_path / "w", {})
        if methods.scope_of(method) == "pair"
        else mod.register_group([case], tmp_path / "w", {})
    )
    with pytest.raises(ValueError, match="has no input rule for a brightfield case"):
        next(gen)
    out = tmp_path / "out"
    run = methods.run_case if methods.scope_of(method) == "pair" else None
    done = run(method, case, out) if run else methods.run_group(method, [case], out)
    assert done in ([], {"29": []})
    name = methods.variant_names(method)[-1]
    rec = json.loads((result_dir(out, name, case) / "run.json").read_text())
    assert rec["ok"] is False and methods.NO_BRIGHTFIELD[method][:40] in rec["error"]


# ── STARE: the proxy ──────────────────────────────────────────────────────────
def test_stare_is_given_the_proxies_of_a_brightfield_case(tmp_path):
    case = brightfield(
        tmp_path,
        ref_image="/data/HE/29.tif",
        mov_image="/data/PHH3/29.tif",
        ref_nuclear="/cache/HE/29.luminance.r1.ome.tif",
        mov_nuclear="/cache/PHH3/29.luminance.r1.ome.tif",
    )
    argv = stare_adapter.command(case, tmp_path, {})
    assert argv[argv.index("--reference") + 1] == case.ref_nuclear
    assert argv[argv.index("--moving") + 1] == case.mov_nuclear
    assert argv[argv.index("--nuclear-index") + 1] == "0"
    assert argv[argv.index("--pixel-size") + 1] == "0.25"
    assert "/data/HE/29.tif" not in argv and "/data/PHH3/29.tif" not in argv
    assert stare_adapter.slides(case)[3] == 1


def test_stare_on_a_reduced_proxy_works_in_the_reduced_frame(tmp_path):
    case = brightfield(tmp_path, ref_nuclear="/c/r.tif", mov_nuclear="/c/m.tif")
    case.extra["settings"] = hyreco.make_settings(reduce=2)
    argv = stare_adapter.command(case, tmp_path, {})
    assert argv[argv.index("--pixel-size") + 1] == "0.5"
    assert stare_adapter.slides(case) == ("/c/r.tif", "/c/m.tif", 0, 2)


def test_a_fluorescence_case_is_still_given_its_original_slides(toy_cases, tmp_path):
    case = toy_cases["rigid"]
    assert stare_adapter.slides(case) == (case.ref_image, case.mov_image, 0, 1)
    argv = stare_adapter.command(case, tmp_path, {})
    assert argv[argv.index("--reference") + 1] == case.ref_image


def test_a_reduced_map_goes_through_block_centres():
    """Reduced pixel i of a 2 x 2 average is centred on full-resolution 2 i + 0.5."""
    seen = []

    def fn(xy):
        seen.append(np.asarray(xy).copy())
        return np.asarray(xy) + [1.0, 0.0]  # one REDUCED pixel to the right

    full = stare_adapter.reduced_map(fn, 2)
    out = full(np.array([[0.5, 0.5], [10.5, 4.5]]))
    np.testing.assert_allclose(seen[0], [[0.0, 0.0], [5.0, 2.0]])
    np.testing.assert_allclose(out, [[2.5, 0.5], [12.5, 4.5]])  # two full pixels
    assert stare_adapter.reduced_map(fn, 1) is fn


def test_a_brightfield_case_without_proxies_is_refused_by_stare(tmp_path):
    case = brightfield(tmp_path)
    with pytest.raises(ValueError, match="needs a one-channel proxy"):
        stare_adapter.command(case, tmp_path, {})


# ── end to end, with the installed STARE ──────────────────────────────────────
def stained(nuclear):
    """A nuclear plane as a brightfield stand-in: white glass, purple nuclei."""
    v = nuclear.astype(np.float32)
    v = v / max(float(np.percentile(v, 99.9)), 1.0)
    v = np.clip(v, 0.0, 1.0)[..., None]
    return (255.0 - v * np.array([150.0, 200.0, 100.0])).round().astype(np.uint8)


def toy_probes(case, pts):
    """``{"truth_mov", "truth_ref"}`` of a toy case: the centres of its moving nuclei and
    where its true map puts them, those at least ``toy_cases.MARGIN_PX`` inside the
    reference, in segmentation order (what the known-truth probes were)."""
    from benchmarks.regbench import scene
    from benchmarks.regbench.cases import polys

    from .toy_cases import MARGIN_PX

    ps = polys(pts, "mov", "nuclei")
    ring = np.searchsorted(ps.ring_off, np.arange(len(ps.xy)), side="right") - 1
    feat = np.asarray(ps.ring_feat)[ring]
    n = np.bincount(feat, minlength=ps.n_features)
    mov = (
        np.stack(
            [np.bincount(feat, ps.xy[:, i], ps.n_features) for i in (0, 1)], axis=1
        )
        / np.maximum(n, 1)[:, None]
    )
    ref = scene.forward_map(case.extra["spec"])(mov)
    h, w = case.ref_hw
    ok = (
        (n > 0)
        & (ref >= MARGIN_PX).all(axis=1)
        & (ref[:, 0] <= w - 1 - MARGIN_PX)
        & (ref[:, 1] <= h - 1 - MARGIN_PX)
    )
    return {"truth_mov": mov[ok], "truth_ref": ref[ok]}


@pytest.fixture(scope="module")
def stare_ran(tmp_path_factory, toy_cases):
    """The toy `rigid` pair re-drawn as an RGB brightfield pair with its true nucleus
    centres as millimetre landmark files, prepared as HyReCo case 29 and registered by
    the installed STARE at full resolution and on a 2 x reduced proxy."""
    pytest.importorskip("stare")
    src = toy_cases["rigid"]
    tmp = tmp_path_factory.mktemp("hyreco_stare")
    root = tmp / "HyReCo"
    pts = toy_probes(src, load_points(src))
    keep = slice(0, 40)
    for stain, image, key in (
        ("HE", src.ref_nuclear, "truth_ref"),
        ("PHH3", src.mov_nuclear, "truth_mov"),
    ):
        write_slide(root / stain / "29.tif", stained(tifffile.imread(image).squeeze()))
        write_points(root / stain / "29.csv", pts[key][keep])
    ran = {}
    for reduce in (1, 2):
        cases = tmp / f"cases_r{reduce}"
        a = args(cases, "a", root, allow_incomplete=True, reduce=reduce)
        assert hyreco.prepare(a) == 0
        (case,) = list_cases(cases, "hyreco_a")
        out = tmp / f"out_r{reduce}"
        got = methods.run_case("stare", case, out, opts={"workers": 2})
        ran[reduce] = (case, out, cases, got)
    return ran, pts


@pytest.mark.parametrize("reduce", [1, 2])
def test_stare_registers_the_proxy_and_is_scored_on_the_landmarks(stare_ran, reduce):
    ran, pts = stare_ran
    case, out, cases, got = ran[reduce]
    assert got == ["stare_rigid", "stare"]
    scored = landmarks.score(cases, out, "hyreco_a", expected=["stare"])
    final = row(scored["pairs"], method="stare")
    initial = row(scored["pairs"], method="initial")
    assert final["n_probes"] == 40 and final["imputed_initial"] is False
    # the pair is rotated 2 degrees and shifted: far apart before, on the nuclei after
    assert initial["tre_um_p50"] > 2.0
    assert final["tre_um_p50"] < 0.25 * initial["tre_um_p50"]
    assert final["tre_um_p50"] < 4 * PX * reduce  # within a few working pixels
    assert final["frac_improved"] == 1.0
    run = json.loads((result_dir(out, "stare", case) / "run.json").read_text())
    assert run["info"]["input"]["kind"] == "one-channel proxy of an RGB slide"
    assert run["info"]["input"]["reduced_by"] == reduce
    assert run["info"]["input"]["reference"] == case.ref_nuclear
    assert run["info"]["nuclear_index"] == 0
    assert float(run["info"]["pixel_size"]) == pytest.approx(PX * reduce, rel=1e-5)
    assert any("proxy" in s for s in run["info"]["implementation"]["benchmark_side"])
    # what STARE wrote is the registered proxy, and there is no tissue-level row
    assert run["native"]["content"].startswith("the one-channel proxy")
    assert run["native"]["reduced_by"] == reduce
    assert "nuclear channel" in run["tissue"]["skipped"]
    assert not (result_dir(out, "stare", case) / "tissue").exists()


# ── VALIS 1.2.0: the original slides (against the recording stand-in) ─────────
def test_valis120_is_given_the_original_slides_of_a_brightfield_case(
    tmp_path, monkeypatch
):
    """The adapter has no modality rule to add: it hands VALIS the case's own slides
    (never the proxy) and VALIS picks its processing. Against the stand-in, so what is
    pinned is the adapter, not VALIS."""
    from . import fake_valis as fv

    monkeypatch.setenv("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
    fv.install(monkeypatch.setitem)
    try:
        pair = sc.make_pair(tmp_path / "pair", (512, 640))
        h, w = pair["ref_hw"]
        settings = hyreco.make_settings()
        # the proxies exist (the driver refuses a case whose files cannot be read) and
        # are not images: an adapter that opened one would fail
        proxies = [tmp_path / f"{side}.proxy.ome.tif" for side in ("ref", "mov")]
        for proxy in proxies:
            proxy.write_bytes(b"never read")
        case = Case(
            dataset="hyreco_a",
            case_id="29",
            group="29",
            ref_nuclear=str(proxies[0]),
            mov_nuclear=str(proxies[1]),
            ref_image=pair["reference"],
            mov_image=pair["moving"],
            modality="brightfield",
            diagonal=math.hypot(h, w),
            pixel_size_um=pair["pixel_size_um"],
            ref_hw=list(pair["ref_hw"]),
            mov_hw=list(pair["mov_hw"]),
            extra={
                "settings": settings,
                "settings_sha256": hyreco.settings_hash(settings),
            },
        )
        xy = np.array([[100.0, 120.0], [300.0, 200.0]])
        case = load_case(write_case(tmp_path / "cases", case, xy, xy))
        out = tmp_path / "out"
        done = methods.run_group("valis", [case], out, {"workers": 1}, label="valis120")
        assert done == {"29": ["valis120_rigid", "valis120", "valis120_micro"]}
        (made,) = fv.calls("Valis")
        assert "proxy" not in json.dumps(made, default=str)
        run = json.loads((result_dir(out, "valis120", case) / "run.json").read_text())
        assert run["direction"] == REF_TO_MOV and run["native"]["files"]
        assert "nuclear channel" in run["tissue"]["skipped"]
        scored = landmarks.score(tmp_path / "cases", out, "hyreco_a")
        assert row(scored["pairs"], method="valis120")["frame"] == "moving"
    finally:
        fv.configure()


# ── defect 1: HE/29.tif and PHH3/29.tif share a basename ──────────────────────
def _same_basename_case(tmp_path):
    root = tmp_path / "HyReCo"
    write_pair(root, "29")
    assert (
        hyreco.prepare(args(tmp_path / "cases", "a", root, allow_incomplete=True)) == 0
    )
    (case,) = list_cases(tmp_path / "cases", "hyreco_a")
    assert Path(case.ref_image).name == Path(case.mov_image).name == "29.tif"
    return case


def test_two_slides_of_one_basename_are_placed_apart(tmp_path):
    from benchmarks.regbench.methods import _valis_points as vp

    case = _same_basename_case(tmp_path)
    ref, moving = vp.patient_slides([case])  # no refusal: they are different files
    assert (ref, moving) == (case.ref_image, {"29": case.mov_image})
    placed, how = vp.place_slides([ref, *moving.values()], tmp_path / "in")
    a, b = Path(placed[ref]), Path(placed[case.mov_image])
    assert a != b and a.name != b.name and a.is_file() and b.is_file()
    assert vp.slide_name(a) != vp.slide_name(b)  # what VALIS will call them
    assert os.stat(a).st_ino == os.stat(ref).st_ino
    assert os.stat(b).st_ino == os.stat(case.mov_image).st_ino
    assert set(how) == {a.name, b.name}
    assert vp.slides_source(placed) == {a.name: ref, b.name: case.mov_image}
    # a TRUE duplicate (one file twice) is still refused, and so is a moving slide
    # that is the reference
    twin = dataclasses.replace(case, case_id="30", group="30")
    twin = dataclasses.replace(twin, mov_image=case.ref_image)
    with pytest.raises(ValueError, match="is the reference slide"):
        vp.patient_slides([twin])


def test_valis_is_given_both_same_basename_slides_and_records_where_they_came_from(
    tmp_path, monkeypatch
):
    from . import fake_valis as fv

    monkeypatch.setenv("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
    fv.install(monkeypatch.setitem)
    try:
        case = _same_basename_case(tmp_path)
        out = tmp_path / "out"
        done = methods.run_group("valis", [case], out, {"workers": 1}, label="valis120")
        # (the stand-in's writer cannot write a 3-band slide: the rigid result is the
        # one that registers; the placement is the same record on all three)
        assert "valis120_rigid" in done["29"]
        run = json.loads(
            (result_dir(out, "valis120_rigid", case) / "run.json").read_text()
        )
        source = run["info"]["slides_source"]
        assert sorted(source.values()) == sorted([case.ref_image, case.mov_image])
        assert len(source) == 2 and set(run["info"]["slides_placed"]) == set(source)
    finally:
        fv.configure()


# ── defect 2: the A/B split covers every setting ──────────────────────────────
def test_the_settings_hash_covers_the_pixel_size_and_the_shared_ids_flag():
    base = hyreco.make_settings()
    assert set(base) == {
        "proxy",
        "reduce",
        "version",
        "pixel_size_um",
        "keep_shared_ids",
    }
    other = [
        hyreco.make_settings(pixel_size_um=0.243),
        hyreco.make_settings(keep_shared_ids=True),
    ]
    hashes = {hyreco.settings_hash(s) for s in [base, *other]}
    assert len(hashes) == 3


def test_subset_b_may_not_choose_a_pixel_size_or_the_shared_ids_alone(
    tmp_path, data, prepared
):
    b = dict(settings_from=str(prepared), index=0)
    with pytest.raises(SystemExit, match="pixel_size_um"):
        hyreco.prepare(args(tmp_path / "x", "b", data.b, pixel_size_um=0.243, **b))
    with pytest.raises(SystemExit, match="keep_shared_ids"):
        hyreco.prepare(args(tmp_path / "x", "b", data.b, keep_shared_ids=True, **b))
    # the same values on both sides are accepted
    tuned = tmp_path / "tuned"
    assert hyreco.prepare(args(tuned, "a", data.a, pixel_size_um=0.243)) == 0
    same = args(
        tmp_path / "y",
        "b",
        data.b,
        pixel_size_um=0.243,
        settings_from=str(tuned),
        index=0,
    )
    assert hyreco.prepare(same) == 0
    (case,) = list_cases(tmp_path / "y", "hyreco_b")
    assert case.extra["settings"]["pixel_size_um"] == 0.243
    assert case.pixel_size_um == 0.243


def test_scoring_a_report_set_without_its_tuning_folder_is_refused(tmp_path, data):
    tuned, cases = tmp_path / "tuned", tmp_path / "cases"
    assert hyreco.prepare(args(tuned, "a", data.a)) == 0
    b = args(cases, "b", data.b, settings_from=str(tuned), index=0)
    assert hyreco.prepare(b) == 0
    import shutil

    shutil.rmtree(tuned)
    with pytest.raises(SystemExit, match="tuning set .* is not there"):
        landmarks.score(cases, tmp_path / "out", "hyreco_b")


# ── defect 3: a run that never started is not the method's failure ────────────
class Reverse:
    """A reference -> moving pair method that fails on the named cases."""

    SCOPE = "pair"

    def __init__(self, name, fail=()):
        self.NAME, self.VARIANTS, self.fail = name, (name,), set(fail)

    def register(self, case, work, opts):
        if case.case_id in self.fail:
            raise RuntimeError("no registration for this pair")
        yield self.NAME, PointMap(lambda xy: np.asarray(xy, float), REF_TO_MOV), {}


def test_a_gate_blocked_or_never_started_pair_is_listed_apart(tmp_path):
    cases, _ = crafted(tmp_path)
    out = tmp_path / "out"
    for case in cases:
        methods.run_case(shifter(0.0, name="stare"), case, out)
        methods.run_case(shifter(0.0, name="valis120_hires"), case, out)
    methods.run_case(shifter(0.0, name="stare_x"), cases[0], out)  # 108 never ran
    gates = out / "logs" / "gates"
    gates.mkdir(parents=True)
    (gates / "valis120_hires.tsv").write_text(
        "valis120_hires\tinconsistent\t-\t1\t-\t-\n"
    )
    got = landmarks.score(
        tmp_path / "cases",
        out,
        "hyreco_a",
        expected=["stare", "valis120_hires", "stare_x"],
        gates=gates,
    )
    lost = {(r["method"], r["cause"]): r for r in got["not_run"]}
    assert lost[("valis120_hires", "gate_blocked")]["n_pairs"] == 2
    assert lost[("stare_x", "not_run")]["n_pairs"] == 1
    assert (
        "blocked by its selfcheck gate"
        in lost[("valis120_hires", "gate_blocked")]["why"]
    )
    # in no median, and not an imputed failure
    assert not [r for r in got["summary"] if r["method"] == "valis120_hires"]
    x = row(got["summary"], method="stare_x", set="tuning")
    assert (x["n_pairs"], x["n_not_run"], x["n_ok"]) == (1, 1, 1)
    assert row(got["summary"], method="stare", set="tuning")["n_not_run"] == 0
    assert all(p["method"] != "valis120_hires" or p["not_run"] for p in got["pairs"])
    # an older result on disk (here: exact, 0 um) is not this pair's: its statistics go
    blocked = [p for p in got["pairs"] if p["method"] == "valis120_hires"]
    assert blocked and all(
        math.isnan(p["tre_um_p50"]) and p["imputed_initial"] for p in blocked
    )
    assert all(p["pixel_size_um"] == 0.25 for p in blocked)
    kept = {(p["method"], p["case_id"]) for p in got["points"]}
    assert not {m for m, _ in kept} & {"valis120_hires"}
    assert ("stare_x", "108") not in kept
    tables = Path(got["tables"])
    assert "valis120_hires" in (tables / "hyreco_not_run.csv").read_text()
    text = (tables / "summary.md").read_text()
    assert "## Not run, for a reason that is not the method's" in text
    assert "| valis120_hires | gate_blocked |" in text


def test_a_failed_pairs_frame_is_never_defaulted_to_the_reference(tmp_path):
    cases, _ = crafted(tmp_path)
    out = tmp_path / "out"
    for case in cases:
        methods.run_case(Reverse("reverse", fail={"108"}), case, out)
        methods.run_case(Reverse("dead", fail={"29", "108"}), case, out)
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    failed = row(got["pairs"], method="reverse", case_id="108")
    assert failed["imputed_initial"] is True and failed["not_run"] is False
    assert failed["frame"] == "moving"  # the row's other pair registered the other way
    dead = {r["frame"] for r in got["pairs"] if r["method"] == "dead"}
    assert dead == {landmarks.UNKNOWN_FRAME}
    # a method's own failure counts, at the unregistered pose
    s = row(got["summary"], method="dead", set="tuning")
    assert (s["n_pairs"], s["n_ok"], s["n_not_run"]) == (2, 0, 0)


def test_the_not_run_causes_agree_with_whose():
    from benchmarks.regbench import failure_log, notrun, resources

    attempts = (*resources.CAUSES, "not_run", "gate_blocked")
    for cause in attempts:
        owner = failure_log.WHOSE[cause]
        if owner in (failure_log.LAUNCHER, failure_log.INFRASTRUCTURE):
            assert cause in notrun.NOT_RUN_CAUSES, cause
    assert landmarks.NOT_RUN_CAUSES is notrun.NOT_RUN_CAUSES  # one definition


def test_a_failed_pairs_declared_direction_names_its_frame(tmp_path):
    cases, _ = crafted(tmp_path)
    out = tmp_path / "out"
    for case in cases:
        methods.run_case(Reverse("dead", fail={"29", "108"}), case, out)
        path = result_dir(out, "dead", case) / "run.json"
        run = json.loads(path.read_text())
        assert "direction" not in run or not run["direction"]
        path.write_text(json.dumps({**run, "direction": REF_TO_MOV}))
    got = landmarks.score(tmp_path / "cases", out, "hyreco_a")
    assert {r["frame"] for r in got["pairs"] if r["method"] == "dead"} == {"moving"}


def test_an_older_settings_version_is_refused_for_both_sets(tmp_path):
    cases, _ = crafted(tmp_path)
    old = {**cases[0].extra["settings"], "version": 1}
    for case in cases:
        case.extra.update(settings=old, settings_sha256=hyreco.settings_hash(old))
        xy = load_points(case)["truth_mov"]
        write_case(tmp_path / "cases", case, xy, xy)
    with pytest.raises(SystemExit, match="settings version 1.*prepare them again"):
        landmarks.score(tmp_path / "cases", tmp_path / "out", "hyreco_a")
