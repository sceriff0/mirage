"""HyReCo (``hyreco``): ONE dataset of subset A and, when downloaded, subset B (ruling
2026-10-19 (hyreco)): the landmark files, the pixel size, ``prepare hyreco`` on the
committed fixture (``fixtures/hyreco/``, synthetic, in the dataset's layout) and the
scorer (``regbench.landmarks``): ``a_`` / ``b_`` case ids, the ``shared_id`` flag, the
three sets, the sanity flags and the published numbers beside them.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import (
    PUBLIC,
    cli,
    landmark_io,
    landmarks,
    losses,
    methods,
    regimes,
)
from benchmarks.regbench.cases import list_cases
from benchmarks.regbench.datasets import hyreco
from benchmarks.regbench.mapping import MOV_TO_REF, PointMap
from benchmarks.tests.regbench.scored_cohort import fake

FIX = Path(__file__).resolve().parent / "fixtures" / "hyreco"
SHIFT = np.array([4.0, -3.0])  # make_landmark_fixtures.SHIFT_PX
PX = 0.24


def _prepare(cases, root=FIX, *extra):
    return cli.main(
        [
            "prepare",
            "hyreco",
            "--data-root",
            str(root),
            "--cases",
            str(cases),
            "--allow-incomplete",
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    cases = tmp_path_factory.mktemp("hyreco_cases")
    assert _prepare(cases) == 0
    return cases


def test_one_dataset_with_one_regime():
    assert "hyreco" in PUBLIC and PUBLIC["hyreco"]["kind"] == "landmarks"
    assert not {"hyreco_a", "hyreco_b"} & set(PUBLIC)
    assert regimes.DATASET_REGIME["hyreco"] == "bf_same"
    assert not {"hyreco_a", "hyreco_b"} & set(regimes.DATASET_REGIME)
    assert landmarks.HYRECO_DATASETS == ("hyreco",)


# ── the files ────────────────────────────────────────────────────────────────
def test_the_landmark_file_is_mm_xyz_without_header(tmp_path):
    xy, header = hyreco.read_points_mm(FIX / "HyReCo/HE/29.csv")
    assert xy.shape == (5, 2) and header is False
    p = tmp_path / "h.csv"
    p.write_text("x,y,z\n1.0,2.0,0\n")
    assert hyreco.read_points_mm(p)[1] is True
    for text, why in (
        ("1,2\n\n3,4\n", "blank"),
        ("1,2,3,4\n", "columns"),
        ("1,2\na,b\n", "numbers"),
    ):
        p.write_text(text)
        with pytest.raises(hyreco.Refused, match=why):
            hyreco.read_points_mm(p)


@pytest.mark.parametrize(
    "path,stated",
    [
        ("HyReCo/HE/29.tif", True),
        ("HyReCo/HE/533.tif", True),
        ("HyReCo-Additional/HE/8.tif", False),
    ],
)
def test_the_pixel_size_the_file_states_else_the_papers(path, stated):
    value, source, got, flag = hyreco.pixel_size(FIX / path)
    assert value == pytest.approx(PX, rel=1e-4) and flag == ""
    assert (got is not None) is stated
    assert source == ("file" if stated else "paper, two digits (the file states none)")


def test_a_pixel_size_far_from_the_papers_is_a_flag_not_a_refusal():
    value, source, _, flag = hyreco.pixel_size(FIX / "HyReCo/HE/29.tif", given=0.5)
    assert value == 0.5 and "--pixel-size-um" in source and "not within" in flag


def test_subsets_there_requires_a_and_takes_b_when_present(tmp_path):
    assert [s for s, _, _ in hyreco.subsets_there(FIX)] == ["a", "b"]
    shutil.copytree(FIX / "HyReCo", tmp_path / "HyReCo")
    assert [s for s, _, _ in hyreco.subsets_there(tmp_path)] == ["a"]
    with pytest.raises(SystemExit, match="HyReCo/"):
        hyreco.subsets_there(tmp_path / "nothing")


def test_one_subsets_files_are_never_prepared_under_the_others_name(tmp_path):
    for s in ("HE", "PHH3"):
        shutil.copytree(FIX / "HyReCo" / s, tmp_path / "HyReCo-Additional" / s)
    with pytest.raises(SystemExit, match="not subset B"):
        hyreco.subset_ids(
            "b", tmp_path / "HyReCo-Additional/HE", tmp_path / "HyReCo-Additional/PHH3"
        )


# ── prepare ──────────────────────────────────────────────────────────────────
def test_prepare_reads_both_subsets_into_one_folder(prepared):
    cases = list_cases(prepared, "hyreco")
    assert sorted(c.case_id for c in cases) == ["a_29", "a_533", "b_29", "b_8"]
    for c in cases:
        assert c.modality == "brightfield" and c.group == c.case_id
        assert c.extra["subset"] == c.case_id[0].upper() and c.extra["flags"] == []
        assert c.pixel_size_um == pytest.approx(PX, rel=1e-4)
        t = landmark_io.read_truth(c)
        np.testing.assert_allclose(t["mov"], t["ref"] + SHIFT, atol=0.01)
    shared = {c.case_id for c in cases if c.extra["shared_id"]}
    assert shared == {"a_29", "b_29"} and "29" in hyreco.SHARED_IDS


def test_count_and_the_missing_pairs_are_losses(prepared, tmp_path, capsys):
    assert _prepare(tmp_path, FIX, "--count") == 0
    n = len(hyreco.SUBSET_A_IDS) + len(hyreco.SUBSET_B_IDS)
    assert capsys.readouterr().out.strip() == str(n)
    rows, _ = losses.read_prepare_dropped(prepared)
    assert len(rows) == n - 4 and {r["status"] for r in rows} == {"DROPPED"}
    with pytest.raises(SystemExit, match="allow-incomplete"):
        cli.main(
            [
                "prepare",
                "hyreco",
                "--data-root",
                str(FIX),
                "--cases",
                str(tmp_path / "c"),
            ]
        )


def test_unequal_landmark_files_are_a_flag_and_the_common_prefix(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIX, root)
    p = root / "HyReCo/PHH3/533.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    assert _prepare(tmp_path / "c", root) == 0
    (c,) = [c for c in list_cases(tmp_path / "c", "hyreco") if c.case_id == "a_533"]
    assert c.extra["n_landmarks"] == 4 and len(c.extra["flags"]) == 1
    assert "common prefix" in c.extra["flags"][0]


# ── the scores ───────────────────────────────────────────────────────────────
def _method(name, err_px=0.0, fail=()):
    def register(case, work, opts):
        if case.case_id in fail:
            raise RuntimeError("no registration for this pair")
        yield (
            name,
            PointMap(lambda xy: np.asarray(xy, float) - SHIFT + err_px, MOV_TO_REF),
            {},
        )

    return fake(name, register)


@pytest.fixture(scope="module")
def scored(prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("hyreco_out")
    for case in list_cases(prepared, "hyreco"):
        for m in (_method("exact"), _method("off", err_px=1.0, fail={"b_8"})):
            methods.run_case(m, case, out)
    return landmarks.score(prepared, out, "hyreco")


def _line(result, set_name, method):
    (row,) = [
        r for r in result["summary"] if (r["set"], r["method"]) == (set_name, method)
    ]
    return row


def test_three_sets_and_the_shared_ids_counted(scored):
    assert {r["set"] for r in scored["summary"]} == {
        "all_pairs",
        "subset_a",
        "subset_b",
    }
    # all_pairs counts a shared id once: subset A's copy (ruling 2026-10-21); each
    # subset's own set keeps its copy, as the published numbers were computed on it
    for name, n, shared in (
        ("all_pairs", 3, 1),
        ("subset_a", 2, 1),
        ("subset_b", 2, 1),
    ):
        line = _line(scored, name, "exact")
        assert (line["n_pairs"], line["n_shared_id"]) == (n, shared)


def test_micrometre_errors(scored):
    assert _line(scored, "all_pairs", "exact")["pooled_median_tre_um"] == pytest.approx(
        0, abs=1e-3
    )  # the files round to 1 nm
    off = _line(scored, "subset_a", "off")
    assert off["pooled_median_tre_um"] == pytest.approx(math.sqrt(2) * PX, rel=1e-3)
    init = _line(scored, "all_pairs", "initial")
    assert init["pooled_median_tre_um"] == pytest.approx(5 * PX, rel=1e-3)


def test_a_failed_pair_counts_at_its_unregistered_landmarks(scored):
    line = _line(scored, "subset_b", "off")
    assert line["n_pairs"] == 2 and line["n_ok"] == 1
    assert any(f["method"] == "off" for f in scored["failures"])


def test_the_published_numbers_stand_beside_never_inside(scored):
    assert {r["set"] for r in scored["published"]} == {"subset_a", "subset_b"}
    assert all(r["status"] == landmarks.NOT_REPRODUCED for r in scored["published"])
    assert not any("HistokatFusion" in r["method"] for r in scored["summary"])
    tables = Path(scored["tables"])
    assert {"hyreco_pairs", "hyreco_points", "hyreco_summary", "hyreco_published"} <= {
        p.stem for p in tables.glob("*.csv")
    }


def test_a_flagged_pair_is_scored_and_a_loss_row(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIX, root)
    p = root / "HyReCo/PHH3/533.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    assert _prepare(tmp_path / "c", root) == 0
    out = tmp_path / "out"
    for case in list_cases(tmp_path / "c", "hyreco"):
        methods.run_case(_method("exact"), case, out)
    result = landmarks.score(tmp_path / "c", out, "hyreco")
    assert _line(result, "all_pairs", "exact")["n_flagged"] == 1
    flagged = [r for r in result["losses"] if r["cause"] == "landmark_flagged"]
    assert [r["case_id"] for r in flagged] == ["a_533"]


def test_all_pairs_keeps_subset_a_copy_of_a_shared_id():
    """2026-10-21: an id both subsets list is one section pair in ``all_pairs``."""
    from benchmarks.regbench.paper import landmarks as paper_landmarks

    keep = dict(landmarks.sets_of("hyreco"))["all_pairs"]
    assert keep({"subset": "A", "shared_id": True})
    assert not keep({"subset": "B", "shared_id": True})
    assert keep({"subset": "B", "shared_id": False})
    row_b = {"subset": "B", "shared_id": "True"}
    assert paper_landmarks._sets("hyreco", row_b) == ["subset_b"]
    row_a = {"subset": "A", "shared_id": "True"}
    assert paper_landmarks._sets("hyreco", row_a) == ["all_pairs", "subset_a"]


def test_the_error_is_scaled_on_the_side_it_is_measured_on():
    """rTRE over the TARGET image's diagonal and TRE in the target's micrometres: the
    reference under mov_to_ref, the moving slide under ref_to_mov (2026-10-21)."""
    from types import SimpleNamespace

    from benchmarks.regbench.mapping import REF_TO_MOV

    case = SimpleNamespace(
        diagonal=math.hypot(300.0, 400.0),
        pixel_size_um=0.25,
        ref_hw=[300, 400],
        mov_hw=[600, 800],
        extra={"pixel_size_mov_um": 0.5},
    )
    assert landmarks.target_scale(case, MOV_TO_REF) == (500.0, 0.25)
    assert landmarks.target_scale(case, REF_TO_MOV) == (1000.0, 0.5)
    # an unregistered pose is measured mov -> ref
    assert landmarks.target_scale(case, "") == (500.0, 0.25)
