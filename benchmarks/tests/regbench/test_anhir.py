"""ANHIR (``anhir``): the cover table, the landmarks, ``prepare anhir`` on the committed
fixture (``fixtures/anhir/``, synthetic, in the archive's layout), and its scorer
(``regbench.challenge``): the challenge's per-pair numbers, the rank among the methods
present, all training pairs scored together (ruling 2026-10-19: no tune / report split),
the sanity flags and the not-run pairs.
"""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import challenge, cli, landmark_io, losses, methods
from benchmarks.regbench.cases import list_cases
from benchmarks.regbench.datasets import anhir
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.regbench.notrun import STALE_TRUTH
from benchmarks.tests.regbench.scored_cohort import fake

FIX = Path(__file__).resolve().parent / "fixtures" / "anhir"
SHIFT = np.array(
    [4.0, -3.0]
)  # make_landmark_fixtures.SHIFT_PX: source = target + SHIFT
HW = (60, 80)


def _prepare(cases, source=FIX, *extra):
    return cli.main(
        ["prepare", "anhir", "--source", str(source), "--cases", str(cases), *extra]
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    cases = tmp_path_factory.mktemp("anhir_cases")
    assert _prepare(cases) == 0
    return cases


# ── the cover table and the landmark files ───────────────────────────────────
def test_the_cover_table_and_its_training_pairs():
    rows = anhir.read_table(FIX / anhir.TABLE)
    assert [r["index"] for r in rows] == [0, 4, 12, 370]
    train = anhir.training(rows)
    assert [anhir.case_id(r) for r in train] == ["004", "012", "370"]
    assert train[0]["source"] == "COAD_01/scale-25pc/S3.jpg"
    assert train[0]["table_diagonal"] == pytest.approx(100.0)
    assert anhir.tissue_of(train[2]["source"]) == "lung-lobes"
    assert anhir.scale_of(train[2]["source"]) == 100


def test_another_table_is_refused(tmp_path):
    bad = tmp_path / "t.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="not an ANHIR cover table"):
        anhir.read_table(bad)


def test_the_landmark_file_format(tmp_path):
    xy = anhir.read_landmarks(FIX / "landmarks/COAD_01/scale-25pc/S1.csv")
    assert xy.shape == (6, 2)
    for text, why in (
        ("A,B\n1,2\n", "header"),
        (",X,Y\n0,1,nan\n", "finite"),
        (",X,Y\n0,1\n", "index,X,Y"),
        (",X,Y\n", "no landmarks"),
    ):
        p = tmp_path / "l.csv"
        p.write_text(text)
        with pytest.raises(anhir.Refused, match=why):
            anhir.read_landmarks(p)


def test_pixel_size_is_the_pages_resolution_over_the_scale_or_none():
    px, source, ok = anhir.pixel_size("COAD", 25)
    assert (
        ok
        and px == pytest.approx(anhir.PIXEL_UM["COAD"][0] * 4)
        and "derived" in source
    )
    assert anhir.pixel_size("mammary-gland", 25)[:3:2] == (None, False)
    assert anhir.pixel_size("no-such-tissue", 25)[0] is None


def test_sanity_checks_are_flags_never_refusals():
    ok = anhir.flags_of({"ref": 6, "mov": 6}, {"ref": 0, "mov": 0}, 100.0, 100.0)
    assert ok == []
    flags = anhir.flags_of({"ref": 6, "mov": 5}, {"ref": 1, "mov": 0}, 90.0, 100.0)
    assert len(flags) == 3
    assert (
        "common prefix" in flags[0] and "outside" in flags[1] and "diagonal" in flags[2]
    )


# ── prepare ──────────────────────────────────────────────────────────────────
def test_prepare_writes_every_training_pair_with_its_truth(prepared):
    cases = list_cases(prepared, anhir.ANHIR)
    assert [c.case_id for c in cases] == [
        "004",
        "012",
        "370",
    ]  # not the evaluation pair
    for c in cases:
        assert c.modality == "brightfield" and c.group == c.case_id
        assert c.diagonal == pytest.approx(math.hypot(*HW))
        t = landmark_io.read_truth(c)
        # roles: the source is the moving image, the target the reference
        np.testing.assert_allclose(t["mov"], t["ref"] + SHIFT, atol=0.051)
        assert c.extra["flags"] == [] and c.extra["subset_complete"] is True
        assert Path(c.ref_nuclear).is_file() and Path(c.mov_nuclear).is_file()
    assert (
        cases[2].extra["tissue"] == "lung-lobes" and cases[2].extra["scale_pc"] == 100
    )


def test_count_says_how_many_training_pairs(tmp_path, capsys):
    assert _prepare(tmp_path, FIX, "--count") == 0
    assert capsys.readouterr().out.strip() == "3"


def _copy(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(FIX, src)
    return src


def test_a_flagged_pair_is_prepared_kept_and_says_why(tmp_path):
    src = _copy(tmp_path)
    p = src / "landmarks/COAD_02/scale-25pc/S5.csv"  # the source of pair 012
    p.write_text("\n".join(p.read_text().splitlines()[:-2]) + "\n")
    assert _prepare(tmp_path / "c", src) == 0
    (c,) = [c for c in list_cases(tmp_path / "c", "anhir") if c.case_id == "012"]
    assert len(c.extra["flags"]) == 1 and "common prefix" in c.extra["flags"][0]
    assert c.extra["n_landmarks"] == 4


def test_missing_images_refuse_unless_allowed_and_every_loss_is_recorded(tmp_path):
    src = _copy(tmp_path)
    (src / "images/COAD_02/scale-25pc/S5.jpg").unlink()
    with pytest.raises(SystemExit, match="allow-incomplete"):
        _prepare(tmp_path / "c", src)
    assert _prepare(tmp_path / "c", src, "--allow-incomplete") == 0
    got = list_cases(tmp_path / "c", "anhir")
    assert [c.case_id for c in got] == ["004", "370"]
    assert all(c.extra["subset_complete"] is False for c in got)
    rows, _ = losses.read_prepare_dropped(tmp_path / "c")
    assert [(r["case_id"], r["status"]) for r in rows] == [("012", "DROPPED")]


def test_a_landmark_file_that_cannot_be_read_drops_the_pair_and_records_it(tmp_path):
    src = _copy(tmp_path)
    (src / "landmarks/COAD_01/scale-25pc/S3.csv").write_text("not,a\nlandmark,file\n")
    assert _prepare(tmp_path / "c", src) == 0
    assert [c.case_id for c in list_cases(tmp_path / "c", "anhir")] == ["012", "370"]
    rows, _ = losses.read_prepare_dropped(tmp_path / "c")
    assert [(r["case_id"], r["status"]) for r in rows] == [("004", "DROPPED")]


# ── the scores ───────────────────────────────────────────────────────────────
def birl_tre(a, b):
    """``compute_tre``, transcribed (BIRL evaluate.py:17-35)."""
    n = min(len(a), len(b))
    return np.sqrt(np.sum((np.asarray(a)[:n] - np.asarray(b)[:n]) ** 2, axis=1))


def _shift_method(name, direction, shift, spoil=0, fail=()):
    def register(case, work, opts):
        if case.case_id in fail:
            raise RuntimeError("no registration for this pair")

        def fn(xy):
            out = np.asarray(xy, float) + shift
            if spoil and len(out) == len(landmark_io.read_truth(case)["mov"]):
                out[:spoil] += 20.0
            return out

        yield name, PointMap(fn, direction), {}

    return fake(name, register)


@pytest.fixture(scope="module")
def scored(prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("anhir_out")
    cases = list_cases(prepared, anhir.ANHIR)
    runs = [
        _shift_method("exact", MOV_TO_REF, -SHIFT),
        _shift_method("rough", MOV_TO_REF, -SHIFT, spoil=4, fail={"004"}),
        _shift_method("inverse", REF_TO_MOV, SHIFT + 1.0),
    ]
    for case in cases:
        for m in runs:
            methods.run_case(m, case, out)
    return challenge.score(prepared, out, anhir.ANHIR), out, cases


def _pair(result, method, cid):
    (row,) = [
        r for r in result["pairs"] if r["method"] == method and r["case_id"] == cid
    ]
    return row


def test_per_pair_numbers_are_the_challenges(scored):
    result, _, cases = scored
    diag = math.hypot(*HW)
    t = landmark_io.read_truth(cases[1])
    assert _pair(result, "exact", "012")["rtre_median"] == pytest.approx(0, abs=1e-9)
    assert _pair(result, "exact", "012")["robustness"] == 1.0
    warp = t["mov"] - SHIFT
    warp[:4] += 20.0
    tre = birl_tre(warp, t["ref"])
    row = _pair(result, "rough", "012")
    assert row["rtre_median"] == pytest.approx(np.median(tre) / diag)
    assert row["rtre_max"] == pytest.approx(np.max(tre) / diag)
    assert row["robustness"] == pytest.approx(2 / 6)
    init = _pair(result, "initial", "012")
    assert init["rtre_median"] == pytest.approx(
        np.median(birl_tre(t["mov"], t["ref"])) / diag
    )
    assert init["robustness"] == 0.0
    # the other direction is exact arithmetic in the source frame, and says so
    inv = _pair(result, "inverse", "012")
    assert inv["rtre_median"] == pytest.approx(math.sqrt(2) / diag, rel=1e-6)
    assert inv["frame"] != _pair(result, "exact", "012")["frame"]


def test_a_failed_pair_counts_at_its_unregistered_landmarks(scored):
    result, _, _ = scored
    row, init = _pair(result, "rough", "004"), _pair(result, "initial", "004")
    assert row["failed"] is True and row["rtre_median"] == init["rtre_median"]
    assert row["robustness"] == 0.0
    assert any(f["method"] == "rough" for f in result["failures"])


def test_all_training_pairs_are_scored_together(scored):
    result, _, _ = scored
    (line,) = [
        r for r in result["summary"] if r["tissue"] == "all" and r["method"] == "rough"
    ]
    assert line["n_pairs"] == 3 and line["n_failed"] == 1
    assert "side" not in line
    assert not any("tune" in str(v) for r in result["summary"] for v in r.values())


def test_the_rank_is_among_the_methods_present_with_ties_averaged(scored):
    result, _, _ = scored
    assert challenge._avg_rank([0.1, 0.1, 0.3]) == [1.5, 1.5, 3.0]
    assert _pair(result, "exact", "012")["rank_median_rtre"] == 1.0
    assert _pair(result, "inverse", "012")["rank_median_rtre"] == 2.0
    assert _pair(result, "rough", "012")["rank_median_rtre"] == 3.0
    assert not math.isfinite(
        _pair(result, "initial", "012").get("rank_median_rtre", math.nan)
    )
    tables = Path(result["tables"])
    assert "among the METHODS PRESENT here" in (tables / "summary.md").read_text()


def test_every_table_is_written(scored):
    result, _, _ = scored
    names = {p.stem for p in Path(result["tables"]).glob("*.csv")}
    assert set(challenge.TABLES) - {"anhir_not_run"} <= names


def test_a_result_made_for_other_landmarks_is_not_run(prepared, tmp_path):
    cases_root = tmp_path / "cases"
    shutil.copytree(prepared, cases_root)
    out = tmp_path / "out"
    (case,) = [c for c in list_cases(cases_root, "anhir") if c.case_id == "370"]
    methods.run_case(_shift_method("exact", MOV_TO_REF, -SHIFT), case, out)
    t = landmark_io.read_truth(case)
    landmark_io.write_truth(
        Path(cases_root) / "anhir" / "370", t["ref"], t["mov"] + 0.5
    )
    result = challenge.score(cases_root, out, "anhir")
    row = _pair(result, "exact", "370")
    assert row["not_run"] is True and row["problem"] == STALE_TRUTH
    assert [r["method"] for r in result["not_run"]] == ["exact"]
    with open(Path(result["tables"]) / "anhir_not_run.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 1
    assert any(r["cause"] == "stale_result" for r in result["losses"])


# ── end to end: prepare -> run -> score -> paper tables ──────────────────────
def test_end_to_end_to_the_paper_tables(tmp_path):
    from benchmarks.regbench.paper import landmarks as paper_landmarks
    from benchmarks.regbench.paper import plots
    from benchmarks.regbench.paper import tables as paper_tables

    cases_root, out = tmp_path / "cases", tmp_path / "out"
    assert _prepare(cases_root) == 0
    # stand-ins under real run names: two rows of STARE's cross and one VALIS row
    for case in list_cases(cases_root, "anhir"):
        methods.run_case(_shift_method("stare", MOV_TO_REF, -SHIFT), case, out)
        methods.run_case(
            _shift_method("stare_medium", MOV_TO_REF, -SHIFT + 0.5), case, out
        )
        methods.run_case(
            _shift_method("valis120_hires", MOV_TO_REF, -SHIFT + 2.0), case, out
        )
    assert (
        cli.main(
            [
                "score",
                "--cases",
                str(cases_root),
                "--out",
                str(out),
                "--dataset",
                "anhir",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    paths = paper_tables.write_all(out, "anhir")
    assert set(paths) == set(paper_landmarks.LANDMARK_WRITTEN)

    def rows(name):
        with open(paths[name]) as fh:
            return list(csv.DictReader(fh))

    summary = {r["row"]: r for r in rows("public_landmarks")}
    assert summary["stare"]["n_pairs"] == "3" and summary["stare"]["set"] == "all"
    assert float(summary["stare"]["rtre_median_med"]) == pytest.approx(0, abs=1e-9)
    assert float(summary["initial"]["rtre_median_med"]) > 0
    assert summary["initial"]["family"] == "NA"
    best = {r["family"]: r for r in rows("method_best")}
    # every terminal row of the family is a candidate, chosen on these pairs (in-sample)
    assert (
        best["stare"]["best_row"] == "stare"
        and best["stare"]["in_sample_selection"] == "True"
    )
    assert int(best["stare"]["n_candidates"]) == 36
    assert best["valis120"]["best_row"] == "valis120_hires"
    # a family with no row scored here has no best row, and says NA
    assert best["deeperhistreg"]["best_row"] == "NA"
    assert {
        r["row"] for r in rows("public_landmarks") if r["best_of_family"] == "True"
    } == {"stare", "valis120_hires"}
    pairs = rows("public_landmark_pairs")
    assert {r["case_id"] for r in pairs} == {"004", "012", "370"}
    got = plots.public_landmarks_best(out / "paper", tmp_path / "figs")
    assert not got.get("placeholder") and got["n"] == {"anhir": 2}
