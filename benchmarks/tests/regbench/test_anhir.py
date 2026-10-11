"""ANHIR's training pairs as an external landmark set: the cases, the split, the scores.

Everything runs on a tiny stand-in for the challenge's folder, built here (a cover table,
a few small JPEG / PNG images, landmark files). One test reads the real folder when it is
on this machine and skips otherwise. The scores are checked against a transcription of the
official evaluation's own functions (Borda/BIRL ``birl/utilities/evaluate.py:17-35,
96-123``), written out below.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import (
    ALL_DATASETS,
    ANHIR,
    ANHIR_DATASETS,
    ANHIR_TUNE,
    DATASETS,
    challenge,
    cli,
    methods,
    prep,
    variants,
)
from benchmarks.regbench.cases import list_cases, load_points
from benchmarks.regbench.datasets import anhir
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap

REAL = Path(
    os.environ.get(
        "REGBENCH_ANHIR", "/Users/valer/Desktop/Github/packages/stare/challenge/anhir"
    )
)
HEADER = [
    "",
    "Image diagonal [pixels]",
    "Image size [pixels]",
    "Source image",
    "Source landmarks",
    "Target image",
    "Target landmarks",
    "status",
    "Warped target landmarks",
    "Warped source landmarks",
    "Execution time [minutes]",
]
HW = (240, 320)
# (set, scale folder, extension, [(source, target, status)])
SETS = [
    *[
        (f"COAD_0{i}", "scale-25pc", "jpg", [("S1", "HE", "training")])
        for i in range(1, 6)
    ],
    ("COAD_01", "scale-25pc", "jpg", [("S2", "HE", "evaluation")]),
    ("mice-kidney_1", "scale-25pc", "jpg", [("a", "b", "training")]),
    ("mammary-gland_1", "scale-25pc", "jpg", [("a", "b", "training")]),
    ("mammary-gland_2", "scale-25pc", "jpg", [("a", "b", "training")]),
    (
        "lung-lobes_1",
        "scale-100pc",
        "png",
        [("a", "b", "training"), ("c", "b", "training")],
    ),
    ("lung-lesion_1", "scale-50pc", "jpg", [("a", "b", "training")]),
]


def _image(path, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    a = np.full((*HW, 3), 235, np.uint8)
    for _ in range(40):
        y, x = rng.integers(8, HW[0] - 8), rng.integers(8, HW[1] - 8)
        a[y - 4 : y + 4, x - 4 : x + 4] = rng.integers(40, 160, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def _landmarks(path, xy, first=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "X", "Y"])
        for i, (x, y) in enumerate(xy, first):
            w.writerow([i, f"{x:.1f}", f"{y:.1f}"])


def build(root, sets=SETS):
    """The stand-in folder; returns its path. The source landmarks are the target's
    shifted by (6, -4) px; `lung-lobes_1/c` has two more landmarks than its target."""
    rng = np.random.default_rng(0)
    rows, seen = [], set()
    for name, scale, ext, pairs in sets:
        for src, tgt, status in pairs:
            for stem in (src, tgt):
                rel = f"{name}/{scale}/{stem}"
                if rel in seen:
                    continue
                seen.add(rel)
                _image(root / "images" / f"{rel}.{ext}", len(seen))
                n = 12 + (2 if stem == "c" else 0)
                xy = rng.uniform(20, 200, (n, 2)).round(1)
                _landmarks(root / "landmarks" / f"{rel}.csv", xy)
            # paired by row: the source's are the target's, moved
            target = anhir.read_landmarks(
                root / "landmarks" / f"{name}/{scale}/{tgt}.csv"
            )
            extra = 2 if src == "c" else 0
            moved = np.vstack([target + (6.0, -4.0), rng.uniform(20, 200, (extra, 2))])
            _landmarks(root / "landmarks" / f"{name}/{scale}/{src}.csv", moved, first=1)
            rows.append(
                [
                    len(rows),
                    f"{math.hypot(*HW):.1f}",
                    str(HW),
                    f"{name}/{scale}/{src}.{ext}",
                    f"{name}/{scale}/{src}.csv",
                    f"{name}/{scale}/{tgt}.{ext}",
                    f"{name}/{scale}/{tgt}.csv",
                    status,
                    "",
                    "",
                    "",
                ]
            )
    with open(root / anhir.TABLE, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    return root


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """``(source folder, frozen split of the stand-in table, its path)``."""
    tmp = tmp_path_factory.mktemp("anhir")
    source = build(tmp / "source")
    split = anhir.make_split(anhir.read_table(source / anhir.TABLE))
    path = tmp / "anhir_split.json"
    path.write_text(json.dumps(split))
    return source, split, path


@pytest.fixture
def frozen(site, monkeypatch):
    monkeypatch.setattr(anhir, "SPLIT_PATH", site[2])
    return site


@pytest.fixture(scope="module")
def prepared(site, tmp_path_factory):
    source, split, path = site
    cases = tmp_path_factory.mktemp("anhir_cases")
    mp = pytest.MonkeyPatch()
    mp.setattr(anhir, "SPLIT_PATH", path)
    try:
        assert (
            cli.main(
                ["prepare", "anhir", "--source", str(source), "--cases", str(cases)]
            )
            == 0
        )
    finally:
        mp.undo()
    return cases


# ── the frozen split ──────────────────────────────────────────────────────────
def test_the_tracked_split_is_by_set_hashed_and_covers_the_230_pairs():
    split = anhir.load_split()
    tune, report = set(split["tune"]), set(split["report"])
    assert not tune & report and len(tune | report) == 49
    assert split["pairs"]["tune"] + split["pairs"]["report"] == anhir.N_TRAINING == 230
    assert split["pairs"] == {"tune": 82, "report": 148}
    assert sum(split["pairs_per_set"].values()) == 230
    assert split["sha256"] == anhir.split_hash(split)
    assert split["rule"] == anhir.SPLIT_RULE and split["salt"] == anhir.SPLIT_SALT
    # every tissue is on the report side; the single-set tissue is wholly there
    assert {anhir.tissue_of(s + "/x") for s in report} == set(anhir.PIXEL_UM)
    assert "mice-kidney_1" in report
    # the rule, recomputed from the names alone, gives the frozen tuning side
    by_tissue = {}
    for name in tune | report:
        by_tissue.setdefault(anhir.tissue_of(name + "/x"), []).append(name)
    again = []
    for names in by_tissue.values():
        names = sorted(names, key=anhir._order)
        again += names[: 0 if len(names) == 1 else math.ceil(len(names) / 4)]
    assert sorted(again) == split["tune"]


def test_the_split_file_cannot_be_edited_by_hand(tmp_path):
    split = anhir.load_split()
    moved = {
        **split,
        "tune": split["tune"][1:],
        "report": [*split["report"], split["tune"][0]],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(moved))
    with pytest.raises(SystemExit, match="does not match its sha256"):
        anhir.load_split(path)
    both = {**split, "report": [*split["report"], split["tune"][0]]}
    both["sha256"] = anhir.split_hash(both)
    path.write_text(json.dumps(both))
    with pytest.raises(SystemExit, match="on both sides"):
        anhir.load_split(path)


def test_the_real_cover_table_gives_the_frozen_split():
    table = REAL / anhir.TABLE
    if not table.is_file():
        pytest.skip(f"the challenge's cover table is not at {table}")
    rows = anhir.read_table(table)
    assert len(rows) == 481 and len(anhir.training(rows)) == 230
    anhir.check_split(rows, anhir.load_split())
    assert anhir.make_split(rows) == anhir.load_split()


def test_the_rule_on_the_stand_in(site):
    _, split, _ = site
    # five COAD sets: two on the tuning side; two mammary sets: one; single sets: none
    assert len([s for s in split["tune"] if s.startswith("COAD")]) == 2
    assert len([s for s in split["tune"] if s.startswith("mammary")]) == 1
    for single in ("mice-kidney_1", "lung-lobes_1", "lung-lesion_1"):
        assert single in split["report"]
    assert split["pairs"]["tune"] + split["pairs"]["report"] == 11  # no evaluation pair


def test_another_release_of_the_table_is_refused(site, tmp_path):
    """The stand-in's table against the TRACKED split: nothing is prepared."""
    source, _, _ = site
    with pytest.raises(SystemExit, match="another release of the table"):
        cli.main(
            ["prepare", "anhir", "--source", str(source), "--cases", str(tmp_path)]
        )
    assert not (tmp_path / ANHIR).exists()


# ── the cases ─────────────────────────────────────────────────────────────────
def test_anhir_is_external_and_never_a_default():
    assert ANHIR_DATASETS == (ANHIR_TUNE, ANHIR) == ("anhir_tune", "anhir")
    assert not set(ANHIR_DATASETS) & set(DATASETS)
    assert set(ANHIR_DATASETS) < set(ALL_DATASETS)
    with pytest.raises(SystemExit, match="scored alone"):
        cli.split_datasets(["multiplex", "anhir"])
    assert cli.split_datasets(["anhir"]) == (None, "anhir")
    assert cli.split_datasets(None) == (None, None)


def test_prepare_writes_the_training_pairs_and_the_tuning_side_again(prepared, site):
    _, split, _ = site
    cases = {c.case_id: c for c in list_cases(prepared, ANHIR)}
    tune = {c.case_id: c for c in list_cases(prepared, ANHIR_TUNE)}
    assert len(cases) == 11 and "005" not in cases  # the evaluation pair is not there
    assert set(tune) == {i for i, c in cases.items() if c.extra["side"] == "tune"}
    assert len(tune) == split["pairs"]["tune"] == 3
    for cid, c in cases.items():
        assert c.group == cid and c.modality == "brightfield"
        assert c.extra["split_sha256"] == split["sha256"]
        assert c.extra["side"] == anhir.side_of(split, c.extra["set"])
        assert c.extra["status"] == "training" and c.ref_hw == list(HW)
        assert c.diagonal == pytest.approx(math.hypot(*HW))
        assert c.extra["table_diagonal_agrees"] is True
    for cid, c in tune.items():  # the same pair, the same files, another dataset
        assert c.dataset == ANHIR_TUNE and c.ref_image == cases[cid].ref_image
        assert c.ref_nuclear == cases[cid].ref_nuclear


def test_roles_landmarks_and_the_common_prefix(prepared, site):
    source = site[0]
    cases = {
        c.extra["stain_mov"] + "@" + c.extra["set"]: c
        for c in list_cases(prepared, ANHIR)
    }
    c = cases["S1@COAD_01"]
    # the Source image is the moving one, the Target the reference
    assert c.mov_image.endswith("COAD_01/scale-25pc/S1.jpg")
    assert c.ref_image.endswith("COAD_01/scale-25pc/HE.jpg")
    pts = load_points(c)
    want = anhir.read_landmarks(source / "landmarks/COAD_01/scale-25pc/HE.csv")
    assert np.array_equal(pts["truth_ref"], want)
    assert np.allclose(pts["truth_mov"], want + (6.0, -4.0))
    # two landmark files of different length: the common prefix, both counts recorded
    c = cases["c@lung-lobes_1"]
    pts = load_points(c)
    assert (
        len(pts["truth_mov"]) == len(pts["truth_ref"]) == 12 == c.extra["n_landmarks"]
    )
    assert c.extra["n_landmarks_file"] == {"ref": 12, "mov": 14}
    assert c.mov_image.endswith(".png")


def test_pixel_size_is_the_pages_resolution_over_the_scale_or_nothing(prepared):
    by_set = {c.extra["set"]: c for c in list_cases(prepared, ANHIR)}
    assert by_set["COAD_01"].pixel_size_um == pytest.approx(1.872)
    assert by_set["lung-lobes_1"].pixel_size_um == pytest.approx(1.274)
    assert by_set["lung-lesion_1"].pixel_size_um == pytest.approx(0.348)
    assert by_set["mice-kidney_1"].pixel_size_um == pytest.approx(0.908)
    assert "derived" in by_set["COAD_01"].extra["pixel_size_source"]
    gland = by_set["mammary-gland_1"]
    assert (
        gland.pixel_size_um is None and gland.extra["pixel_size_established"] is False
    )
    assert "pixels only" in gland.extra["pixel_size_source"]
    assert anhir.pixel_size("gastric", 15)[0] == pytest.approx(0.2528 / 0.15)
    assert anhir.pixel_size("breast", 20)[0] == pytest.approx(1.264)
    assert anhir.pixel_size("unknown-tissue", 25)[::2] == (None, False)
    assert set(anhir.PIXEL_UM) == {
        "lung-lesion",
        "lung-lobes",
        "mammary-gland",
        "mice-kidney",
        "COAD",
        "gastric",
        "breast",
        "kidney",
    }


def test_the_plain_rows_grey_image_is_the_stated_conversion(prepared):
    import tifffile

    c = next(c for c in list_cases(prepared, ANHIR) if c.extra["set"] == "lung-lobes_1")
    rgb = prep.read_rgb(c.ref_image).astype(float)
    want = 255.0 - (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    grey = tifffile.imread(c.ref_nuclear)
    assert grey.dtype == np.uint8 and grey.shape[-2:] == HW
    assert np.abs(grey.reshape(HW) - want).max() <= 0.5 + 1e-3
    assert "0.299" in c.extra["grey"] and c.extra["settings"]["reduce"] == 1
    # one file per IMAGE: the two pairs with the target `b` share it
    lobes = [c for c in list_cases(prepared, ANHIR) if c.extra["set"] == "lung-lobes_1"]
    assert len(lobes) == 2 and len({c.ref_nuclear for c in lobes}) == 1


def test_a_missing_image_refuses_the_whole_prepare_unless_allowed(
    frozen, tmp_path, capsys
):
    source = frozen[0]
    partial = tmp_path / "source"
    partial.mkdir()
    (partial / anhir.TABLE).write_bytes((source / anhir.TABLE).read_bytes())
    (partial / "landmarks").symlink_to(source / "landmarks")
    (partial / "images").mkdir()
    for name in ("COAD_01", "lung-lobes_1"):
        (partial / "images" / name).symlink_to(source / "images" / name)
    cases = tmp_path / "cases"
    base = ["prepare", "anhir", "--source", str(partial), "--cases", str(cases)]
    with pytest.raises(SystemExit, match="8 of 11 training pairs are missing"):
        cli.main(base)
    assert cli.main([*base, "--allow-incomplete"]) == 0
    got = list_cases(cases, ANHIR)
    assert len(got) == 3 and all(c.extra["subset_complete"] is False for c in got)
    assert cli.main([*base, "--count"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "11"


def test_a_landmark_file_of_another_shape_is_refused(tmp_path):
    bad = tmp_path / "a.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(anhir.Refused, match="expected the header"):
        anhir.read_landmarks(bad)
    bad.write_text(",X,Y\n0,1.0,nan\n")
    with pytest.raises(anhir.Refused, match="finite"):
        anhir.read_landmarks(bad)
    bad.write_text(",X,Y\n0,1.0,2.0\n\n1,3.0,4.5\n")
    assert anhir.read_landmarks(bad).tolist() == [[1.0, 2.0], [3.0, 4.5]]


# ── which runs ────────────────────────────────────────────────────────────────
def test_the_run_list_of_anhir_and_the_reasons_of_the_rest():
    names = [variants.run_name(r) for r in variants.runs_for(ANHIR)]
    assert names == [
        "stare",
        "stare_prep",
        "valis120",
        "valis120_hires",
        "deeperhistreg_gpu",
        "deeperhistreg_cpu",
        "deeperhistreg_hr_gpu",
        "deeperhistreg_hr_cpu",
    ]
    assert len(variants.expected_variants(ANHIR)) == 18
    assert variants.primary_for(ANHIR) == [
        "stare",
        "valis120_hires",
        "deeperhistreg_hr_gpu_asdrawn",
    ]
    left = dict(variants.excluded_for(ANHIR))
    assert set(left) | set(names) == {
        variants.run_name(r) for r in variants.active_runs()
    }
    assert not set(left) & set(names)
    for name in (
        "ashlar",
        "affine_ashlar_clean",
        "palom",
        "palom_b1024",
        "valis_mirage",
    ):
        assert len(left[name]) > 40
    assert "fluorescence" in left["deeperhistreg_hr_fluo_gpu"]
    # the tuning side: the candidates, and nothing of the cohort's list
    tune = [variants.run_name(r) for r in variants.runs_for(ANHIR_TUNE)]
    assert tune == [f"stare_prep_{c}" for c in prep.CANDIDATES]
    assert len(variants.expected_variants(ANHIR_TUNE)) == 2 * len(prep.CANDIDATES)
    assert (
        variants.excluded_for(ANHIR_TUNE) == []
        and variants.primary_for(ANHIR_TUNE) == []
    )
    assert all(r[14] == "-" for r in variants.runs_table(dataset=ANHIR))  # no realcheck


def test_runs_prints_the_lists_and_the_frozen_status(capsys):
    assert cli.main(["runs", "--dataset", "anhir_tune"]) == 0
    lines = [ln.split("\t") for ln in capsys.readouterr().out.splitlines()]
    assert [ln[0] for ln in lines] == [f"stare_prep_{c}" for c in prep.CANDIDATES]
    assert [ln[3] for ln in lines] == [f"prep={c}" for c in prep.CANDIDATES]
    assert all(ln[8] == ln[0] for ln in lines)  # each behind its own selfcheck
    assert cli.main(["runs", "--prep-status"]) == 0
    record = prep.load_frozen()
    assert capsys.readouterr().out.split() == [
        record["status"],
        record["selected"],
        prep.ROW,
    ]


# ── the scores ────────────────────────────────────────────────────────────────
def birl_tre(points_1, points_2):
    """``compute_tre``, transcribed (evaluate.py:17-35)."""
    n = min(len(points_1), len(points_2))
    a, b = np.asarray(points_1)[:n], np.asarray(points_2)[:n]
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def birl_robustness(target, init, warp):
    """``compute_tre_robustness``, transcribed (evaluate.py:96-123)."""
    n = min(len(target), len(init), len(warp))
    tre_init = birl_tre(init[:n], target[:n])
    tre_final = birl_tre(warp[:n], target[:n])
    return np.sum(tre_final < tre_init) / float(len(tre_final))


class Fake:
    """A pair method whose map is a shift, with part of the points wrong or lost."""

    SCOPE = "pair"

    def __init__(self, name, direction, shift, spoil=0, lose=0, fail=()):
        self.NAME, self.VARIANTS = name, (name,)
        self.direction, self.shift = direction, np.asarray(shift, float)
        self.spoil, self.lose, self.fail = spoil, lose, set(fail)

    def register(self, case, work, opts):
        if case.case_id in self.fail:
            raise RuntimeError("no registration for this pair")

        def fn(xy):
            out = np.asarray(xy, float) + self.shift
            n = len(load_points(case)["truth_mov"])
            if len(out) == n:  # the landmarks themselves, not the lattice
                out[: self.spoil] += 40.0
                if self.lose:
                    out[-self.lose :] = np.nan
            return out

        yield self.NAME, PointMap(fn, self.direction), {}


@pytest.fixture(scope="module")
def scored(prepared, site, tmp_path_factory):
    out = tmp_path_factory.mktemp("anhir_out")
    cases = list_cases(prepared, ANHIR)
    fakes = [
        # exact: the source landmarks are the target's + (6, -4)
        Fake("exact", MOV_TO_REF, (-6.0, 4.0)),
        # three landmarks made worse, two lost; one pair fails outright
        Fake(
            "rough", MOV_TO_REF, (-5.0, 4.0), spoil=3, lose=2, fail={cases[0].case_id}
        ),
        # the other direction: target landmarks into the source image
        Fake("inverse", REF_TO_MOV, (6.0, -3.0)),
    ]
    for case in cases:
        for fake in fakes:
            methods.run_case(fake, case, out)
    mp = pytest.MonkeyPatch()
    mp.setattr(anhir, "SPLIT_PATH", site[2])
    try:
        result = challenge.score(prepared, out, ANHIR)
    finally:
        mp.undo()
    return result, out, cases


def pair_row(result, method, case_id):
    (row,) = [
        r for r in result["pairs"] if r["method"] == method and r["case_id"] == case_id
    ]
    return row


def test_per_pair_numbers_are_the_official_definitions(scored):
    result, _, cases = scored
    case = cases[1]
    pts = load_points(case)
    src, tgt = pts["truth_mov"], pts["truth_ref"]
    diag = math.hypot(*HW)
    # exact
    row = pair_row(result, "exact", case.case_id)
    assert row["rtre_median"] == pytest.approx(0.0, abs=1e-12)
    assert row["robustness"] == 1.0 and row["challenge_quantity"] is True
    # rough: against BIRL's functions on the warped landmarks a submission would hold
    warp = src + (-5.0, 4.0)
    warp[:3] += 40.0
    warp[-2:] = src[-2:]  # a lost landmark stays where it was
    tre = birl_tre(warp, tgt)
    row = pair_row(result, "rough", case.case_id)
    assert row["rtre_median"] == pytest.approx(np.median(tre) / diag)
    assert row["rtre_mean"] == pytest.approx(np.mean(tre) / diag)
    assert row["rtre_max"] == pytest.approx(np.max(tre) / diag)
    assert row["robustness"] == pytest.approx(birl_robustness(tgt, src, warp))
    assert row["robustness"] == pytest.approx((len(src) - 5) / len(src))
    assert row["n_nonfinite"] == 2 and row["failed"] is False
    assert row["tre_px_p50"] == pytest.approx(np.median(tre))
    # initial: the unregistered pose improves nothing
    row = pair_row(result, "initial", case.case_id)
    assert row["rtre_median"] == pytest.approx(math.hypot(6, 4) / diag)
    assert row["robustness"] == 0.0
    # the other direction: exact arithmetic in the source frame, labelled
    row = pair_row(result, "inverse", case.case_id)
    assert row["rtre_median"] == pytest.approx(1.0 / diag)
    assert (
        row["challenge_quantity"] is False
        and row["frame"] != pair_row(result, "exact", case.case_id)["frame"]
    )


def test_a_failed_pair_counts_at_its_unregistered_landmarks(scored):
    result, _, cases = scored
    row = pair_row(result, "rough", cases[0].case_id)
    init = pair_row(result, "initial", cases[0].case_id)
    assert row["failed"] is True and row["cause"] == "exception"
    assert row["rtre_median"] == init["rtre_median"] and row["robustness"] == 0.0
    (line,) = [
        r
        for r in result["summary"]
        if (r["side"], r["tissue"], r["method"]) == ("all", "all", "rough")
    ]
    assert line["n_failed"] == 1 and line["n_pairs"] == 11
    mine = [r for r in result["pairs"] if r["method"] == "rough"]
    ok = [r for r in mine if not r["failed"]]
    # counted the official way in the headline columns, and shown apart
    assert line["median_rtre_median"] == pytest.approx(
        np.median([r["rtre_median"] for r in mine])
    )
    assert line["median_rtre_median_registered"] == pytest.approx(
        np.median([r["rtre_median"] for r in ok])
    )
    assert line["avg_robustness"] == pytest.approx(
        np.mean([r["robustness"] for r in mine])
    )
    assert line["avg_robustness"] < line["avg_robustness_registered"]
    assert any(f["method"] == "rough" for f in result["failures"])


def test_aggregates_are_by_side_tissue_and_set(scored, site):
    result, _, _ = scored
    split = site[1]
    keys = {
        (r["side"], r["tissue"]) for r in result["summary"] if r["method"] == "exact"
    }
    assert {("report", "all"), ("tune", "all"), ("all", "all")} <= keys
    assert ("report", "lung-lobes") in keys and ("tune", "COAD") in keys
    assert ("tune", "lung-lobes") not in keys  # a single set: wholly on the report side
    n = {
        r["side"]: r["n_pairs"]
        for r in result["summary"]
        if r["method"] == "exact" and r["tissue"] == "all"
    }
    assert n == {
        "report": split["pairs"]["report"],
        "tune": split["pairs"]["tune"],
        "all": 11,
    }
    sets = {r["set"]: r for r in result["by_set"] if r["method"] == "exact"}
    assert len(sets) == 10 and sets["lung-lobes_1"]["n_pairs"] == 2
    assert all(r["side"] == anhir.side_of(split, name) for name, r in sets.items())
    # BIRL's aggregate, on the rows of one group, by hand
    mine = [
        r for r in result["pairs"] if r["method"] == "rough" and r["side"] == "report"
    ]
    (line,) = [
        r
        for r in result["summary"]
        if (r["side"], r["tissue"], r["method"]) == ("report", "all", "rough")
    ]
    assert line["avg_rtre_median"] == pytest.approx(
        np.mean([r["rtre_median"] for r in mine])
    )
    assert line["avg_rtre_max"] == pytest.approx(np.mean([r["rtre_max"] for r in mine]))
    assert line["median_rtre_mean"] == pytest.approx(
        np.median([r["rtre_mean"] for r in mine])
    )
    robust = [r for r in mine if r["robustness"] > 0.5]
    assert line["n_robust"] == len(robust)
    assert line["avg_rtre_median_robust"] == pytest.approx(
        np.mean([r["rtre_median"] for r in robust])
    )


def test_micrometres_only_where_the_pixel_size_is_established(scored):
    result, _, _ = scored
    gland = [
        r
        for r in result["pairs"]
        if r["tissue"] == "mammary-gland" and r["method"] == "inverse"
    ]
    assert gland and all(math.isnan(r["tre_um_p50"]) for r in gland)
    assert all(r["rtre_median"] > 0 and r["tre_px_p50"] > 0 for r in gland)
    coad = [
        r for r in result["pairs"] if r["tissue"] == "COAD" and r["method"] == "inverse"
    ]
    assert all(r["tre_um_p50"] == pytest.approx(1.0 * 1.872) for r in coad)
    (line,) = [
        r
        for r in result["summary"]
        if (r["side"], r["tissue"], r["method"]) == ("all", "all", "inverse")
    ]
    assert line["n_pairs"] == 11 and line["n_pairs_um"] == 9


def test_the_tables_say_what_these_numbers_are_not(scored):
    result, out, _ = scored
    tables = Path(result["tables"])
    assert tables == out / "tables" / ANHIR
    for name in ("anhir_pairs", "anhir_summary", "anhir_by_set", "anhir_excluded"):
        assert (tables / f"{name}.csv").stat().st_size > 0
    assert not list((out / "tables").glob("*.csv"))  # none of the cohort's tables
    text = (tables / "summary.md").read_text()
    for said in (
        "TRAINING pairs",
        "hidden evaluation pairs",
        "not comparable row for row",
        "chosen on the tuning side of this same training set",
        "out-of-domain",
        "## Report side",
        "## Tuning side",
        "## All training pairs",
        "mammary-gland",
        "255 - (0.299 R + 0.587 G + 0.114 B)",
        "SOURCE image's frame",
        "`ashlar`",
    ):
        assert said in text, said
    assert text.index("## Report side") < text.index("## Tuning side")
    assert ("PROVISIONAL" in text) == (prep.load_frozen()["status"] == "provisional")


def test_cases_of_another_split_are_not_scored(prepared, scored, tmp_path):
    with pytest.raises(
        SystemExit, match="Results of two splits are never put together"
    ):
        challenge.score(prepared, scored[1], ANHIR)  # the TRACKED split, other cases


# ── choosing the preprocessing ────────────────────────────────────────────────
def tuned(prepared, out, skip=()):
    """Candidate results on the tuning side: candidate i is i + 1 px off."""
    for i, c in enumerate(prep.CANDIDATES):
        if c in skip:
            continue
        fake = Fake(f"stare_prep_{c}", MOV_TO_REF, (-6.0 + (4 - i) + 1, 4.0))
        for case in list_cases(prepared, ANHIR_TUNE):
            methods.run_case(fake, case, out)


def test_the_choice_is_made_on_the_tuning_side_and_frozen_with_its_evidence(
    prepared, frozen, tmp_path, capsys
):
    out = tmp_path / "out"
    tuned(prepared, out)
    target = tmp_path / "prep_frozen.json"
    argv = ["prep-select", "--cases", str(prepared), "--out", str(out)]
    assert cli.main(argv) == 0 and not target.exists()  # without --freeze: a table
    assert "selected: grey_smooth" in capsys.readouterr().out
    assert cli.main([*argv, "--freeze", "--frozen-file", str(target)]) == 0
    record = prep.load_frozen(target)
    assert record["status"] == "selected" and record["selected"] == "grey_smooth"
    assert record["split_sha256"] == frozen[1]["sha256"] and not record["note"]
    assert [r["candidate"] for r in record["evidence"]] == list(prep.CANDIDATES)[::-1]
    assert all(r["n_pairs"] == 3 and r["never_run"] == 0 for r in record["evidence"])
    medians = [r["median_of_median_rtre"] for r in record["evidence"]]
    assert medians == sorted(medians) and medians[0] == pytest.approx(
        1 / math.hypot(*HW)
    )
    # the tuning side's own tables show the same evidence, and say they are no result
    result = challenge.score(prepared, out, ANHIR_TUNE)
    assert [r["candidate"] for r in result["prep_evidence"]] == list(prep.CANDIDATES)[
        ::-1
    ]
    text = (Path(result["tables"]) / "summary.md").read_text()
    assert "TUNING SIDE. Nothing on this page is a result." in text
    assert "## Report side" not in text
    assert {r["side"] for r in result["pairs"]} == {"tune"}


def test_no_choice_on_part_of_the_evidence(prepared, frozen, tmp_path):
    out = tmp_path / "out"
    tuned(prepared, out, skip=("haem",))
    target = tmp_path / "prep_frozen.json"  # never the tracked file, even on a bug
    argv = ["prep-select", "--cases", str(prepared), "--out", str(out), "--freeze"]
    with pytest.raises(SystemExit, match="no result on some tuning pair for: haem"):
        cli.main([*argv, "--frozen-file", str(target)])
    assert not target.exists()


def test_a_report_side_pair_cannot_sit_in_the_tuning_dataset(
    prepared, frozen, tmp_path
):
    import shutil

    cases = tmp_path / "cases"
    shutil.copytree(prepared / ANHIR_TUNE, cases / ANHIR_TUNE)
    report = next(c for c in list_cases(prepared, ANHIR) if c.extra["side"] == "report")
    d = cases / ANHIR_TUNE / report.case_id
    shutil.copytree(report.dir, d)
    rec = json.loads((d / "case.json").read_text())
    rec["dataset"] = ANHIR_TUNE
    (d / "case.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="report-side pair in the tuning dataset"):
        challenge.score(cases, tmp_path / "out", ANHIR_TUNE)


# ── what STARE is given ───────────────────────────────────────────────────────
def test_stare_is_given_the_grey_image_and_a_placeholder_size_where_none_is_known(
    prepared, tmp_path
):
    pytest.importorskip("stare")
    from benchmarks.regbench.methods import stare as adapter

    by_set = {c.extra["set"]: c for c in list_cases(prepared, ANHIR)}
    argv = adapter.command(by_set["COAD_01"], tmp_path, {})
    assert argv[argv.index("--reference") + 1] == by_set["COAD_01"].ref_nuclear
    assert argv[argv.index("--moving") + 1].endswith("S1.grey.ome.tif")
    assert argv[argv.index("--nuclear-index") + 1] == "0"
    assert float(argv[argv.index("--pixel-size") + 1]) == pytest.approx(1.872)
    # no established pixel size: STITCH's header gets a placeholder, said aloud
    argv = adapter.command(by_set["mammary-gland_1"], tmp_path, {})
    assert float(argv[argv.index("--pixel-size") + 1]) == adapter.NOMINAL_PIXEL_SIZE
    # the preprocessing row registers its own files and nothing else changes
    plain = adapter.command(by_set["COAD_01"], tmp_path, {})
    prepped = adapter.command(by_set["COAD_01"], tmp_path, {}, ("/w/r.tif", "/w/m.tif"))
    changed = [i for i, (a, b) in enumerate(zip(plain, prepped)) if a != b]
    assert [plain[i - 1] for i in changed] == ["--reference", "--moving"]
    assert len(plain) == len(prepped)
