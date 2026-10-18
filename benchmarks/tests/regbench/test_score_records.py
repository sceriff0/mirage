"""Per-case score records, pair records and the scorer's parallelism: no table changes.

The scorer keeps what it computed under ``<out>/cache/scores``: the rows of each case
(``case.rec``) and, beneath them, every pair record (one ``pairs.record`` call). These
tests hold it to three things, on a cohort that has every kind of row
(``scored_cohort.py``):

* every way of getting the tables gives the SAME BYTES in every file under ``tables/``:
  serial, cases side by side, pair records side by side, one case at a time
  (``score_case``) and then merged, merged with half the records gone, and read back
  entirely from records;
* a record is reused only when everything it rests on is unchanged: after each kind of
  change the scorer's tables are byte-identical to a scoring that reads no record;
* each number in the cell tables is what a plain ``pairs.record`` call on that row's own
  polygons returns, with nothing shared between rows.

They were written against tables made by the code before the records existed (same
cohort, same folder): those were byte-identical too, which no committed test can show.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import pairs, variants
from benchmarks.regbench import score as score_mod
from benchmarks.regbench.cases import (
    list_cases,
    load_case,
    load_points,
    polys,
    result_dir,
    write_json,
)
from benchmarks.regbench.mapping import REF_TO_MOV

from . import scored_cohort

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
KW = scored_cohort.SCORE_KW
# what `score_case` takes of it (the primary rows are a matter of the tables)
CASE_KW = {k: v for k, v in KW.items() if k != "primary"}
NON = score_mod.NON


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root, out, _ = scored_cohort.build(tmp_path_factory.mktemp("scored"))
    return root, out


def copy_of(cohort, tmp, cases=None):
    """The cohort (or some of its cases) in a folder of its own, results and all, with
    no record and no table."""
    root, out = cohort
    new_root, new_out = tmp / "cases", tmp / "out"
    for case_json in sorted(root.glob("*/*/case.json")):
        d = case_json.parent
        if cases is None or d.name in cases:
            shutil.copytree(d, new_root / d.parent.name / d.name)
            for res in sorted(out.glob(f"*/{d.parent.name}/{d.name}")):
                if res.parts[-3] not in score_mod.RESERVED:
                    shutil.copytree(res, new_out / res.relative_to(out))
    return new_root, new_out


def tables(out):
    """Every file under ``tables/``, as bytes."""
    got = {p.name: p.read_bytes() for p in sorted((out / "tables").iterdir())}
    # (20 at the least: dropped.csv is gone with the dropped rows)
    assert "summary.md" in got and "cells.csv" in got and len(got) >= 20
    return got


def scored(root, out, **kw):
    score_mod.score(root, out, **{**KW, **kw})
    return tables(out)


def same(a, b):
    assert sorted(a) == sorted(b)
    differ = [name for name in a if a[name] != b[name]]
    assert not differ, f"tables that differ: {differ}"


class Spy:
    """Which cases were computed, and how many pair records, in THIS process."""

    def __init__(self, monkeypatch):
        self.cases, self.records = [], 0
        compute, record = score_mod._score_case, pairs.record

        def score_case(job, *a, **kw):
            self.cases.append(str(job[1].case_id))
            return compute(job, *a, **kw)

        def pair_record(*a, **kw):
            self.records += 1
            return record(*a, **kw)

        monkeypatch.setattr(score_mod, "_score_case", score_case)
        monkeypatch.setattr(pairs, "record", pair_record)

    def reset(self):
        self.cases, self.records = [], 0


# ── every path, the same bytes ────────────────────────────────────────────────
def test_every_way_of_scoring_writes_the_same_tables(cohort, tmp_path, monkeypatch):
    root, out = copy_of(cohort, tmp_path)
    serial = scored(root, out, workers=1, fresh=True)
    assert len(list_cases(root, "multiplex")) == 4
    # four workers, four cases to compute: whole cases side by side
    same(serial, scored(root, out, workers=4, fresh=True))
    # eight workers, four cases: one case after the other, its pair records side by side
    same(serial, scored(root, out, workers=8, fresh=True))

    # one case at a time (the cluster's array), then the merge
    shutil.rmtree(out / "cache" / "scores")
    shutil.rmtree(out / "tables")
    for i, case in enumerate(list_cases(root, "multiplex")):
        did = score_mod.score_case(
            root, out, "multiplex", index=i, workers=1 + i % 2 * 2, **CASE_KW
        )
        assert did == "scored" and not (out / "tables").exists()
        assert (score_mod.records_dir(out, case) / "case.rec").exists()
    assert score_mod.score_case(root, out, "multiplex", index=0, **CASE_KW) == "kept"
    spy = Spy(monkeypatch)
    same(serial, scored(root, out, workers=1))
    assert spy.cases == [] and spy.records == 0  # merged: nothing was computed

    # half the case records gone, and for one of them its pair records too
    for case_id in ("P1_r2", "P3_r1"):
        (out / "cache" / "scores" / "multiplex" / case_id / "case.rec").unlink()
    shutil.rmtree(out / "cache" / "scores" / "multiplex" / "P3_r1" / "pairs")
    same(serial, scored(root, out, workers=1))
    assert spy.cases == ["P1_r2", "P3_r1"]
    # P1_r2's pair records were all still there; P3_r1's were computed again
    assert (
        0
        < spy.records
        == len(list((out / "cache/scores/multiplex/P3_r1/pairs").iterdir()))
    )

    # a second scoring reads everything
    spy.reset()
    same(serial, scored(root, out, workers=4))
    assert spy.cases == [] and spy.records == 0


def test_an_interrupted_scoring_keeps_what_it_finished(cohort, tmp_path, monkeypatch):
    root, out = copy_of(cohort, tmp_path)
    whole = scored(root, out, workers=1, fresh=True)
    shutil.rmtree(out / "cache" / "scores")
    compute, seen = score_mod._score_case, []

    def dies_on_the_third(job, *a, **kw):
        seen.append(job[1].case_id)
        if len(seen) == 3:
            raise KeyboardInterrupt
        return compute(job, *a, **kw)

    monkeypatch.setattr(score_mod, "_score_case", dies_on_the_third)
    with pytest.raises(KeyboardInterrupt):
        score_mod.score(root, out, workers=1, **KW)
    kept = sorted(p.parent.name for p in (out / "cache/scores").glob("*/*/case.rec"))
    assert kept == sorted(seen[:2])
    monkeypatch.undo()
    spy = Spy(monkeypatch)
    same(whole, scored(root, out, workers=1))
    assert len(spy.cases) == 2 and not set(spy.cases) & set(seen[:2])


def test_a_pair_record_is_saved_as_it_finishes(cohort, tmp_path, monkeypatch):
    """A case that is killed half-way goes on from its pair records: a 24-hour task
    that ran out of time is not started over."""
    root, out = copy_of(cohort, tmp_path, cases=("P1_r1",))
    kw = {**CASE_KW, "expected": None}
    whole = scored(root, out, workers=1, fresh=True, expected=None)
    total = len(list((out / "cache/scores/multiplex/P1_r1/pairs").iterdir()))
    shutil.rmtree(out / "cache" / "scores")
    record, calls = pairs.record, []

    def dies_on_the_fifth(*a, **k):
        calls.append(1)
        if len(calls) == 5:
            raise KeyboardInterrupt
        return record(*a, **k)

    monkeypatch.setattr(pairs, "record", dies_on_the_fifth)
    with pytest.raises(KeyboardInterrupt):
        score_mod.score_case(root, out, "multiplex", index=0, **kw)
    assert len(list((out / "cache/scores/multiplex/P1_r1/pairs").iterdir())) == 4
    assert not (out / "cache/scores/multiplex/P1_r1/case.rec").exists()
    monkeypatch.undo()
    spy = Spy(monkeypatch)
    same(whole, scored(root, out, workers=1, expected=None))
    assert spy.records == total - 4 > 4


# ── each number is a plain pairs.record of that row's own polygons ────────────
def _plain_record(root, out, row):
    """``pairs.record`` for one table row, from the files, sharing nothing."""
    case = load_case(root / row["dataset"] / row["case_id"])
    points, which = load_points(case), row["set"]
    xy_key, band_key, band_ref_key = score_mod.WARPED_KEYS[which]
    ref, mov = polys(points, "ref", which), polys(points, "mov", which)
    variant, _, owner = row["method"].partition(NON)
    ref_in, mov_in = ref, mov
    if variant != "initial" and row["imputed_initial"] == "False":
        with np.load(result_dir(out, variant, case) / "warped.npz") as z:
            xy, direction = z[xy_key].astype(float), str(z["direction"])
        if direction == REF_TO_MOV:
            ref_in = pairs.with_xy(ref, xy)
        else:
            mov_in = pairs.with_xy(mov, xy)
    if owner:
        with np.load(result_dir(out, owner, case) / "warped.npz") as z:
            if band_ref_key in z.files:
                ref_in = score_mod._keep_features(ref_in, ~z[band_ref_key])
            if band_key in z.files:
                mov_in = score_mod._keep_features(mov_in, ~z[band_key])
    return pairs.record(ref, mov, ref_in, mov_in, case.pixel_size_um)


def test_every_cell_number_is_a_plain_record_of_its_own_row(cohort, tmp_path):
    """The scorer computes each DISTINCT pair record once and shares geometry between
    them; a row's numbers must still be exactly what computing that row alone gives."""
    root, out = copy_of(cohort, tmp_path)
    scored(root, out, workers=2, fresh=True)
    seen = {"imputed": 0, "owner": 0, "moving": 0}
    n = 0
    for name in ("cells", "nonband_cells", "cells_nuclei", "nonband_cells_nuclei"):
        with open(out / "tables" / f"{name}.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, name
        for row in rows:
            rec = _plain_record(root, out, row)
            for key in pairs.RECORD_KEYS:
                assert row[key] == str(rec[key]), (name, row["method"], key)
            n += 1
            seen["imputed"] += row["imputed_initial"] == "True"
            seen["owner"] += NON in row["method"]
            seen["moving"] += row["frame"] == "moving"
    assert n > 100 and all(seen.values()), seen


def test_geometry_is_kept_by_identity_and_changes_no_number():
    rng = np.random.default_rng(0)
    centres = scored_cohort.GRID + rng.uniform(-3, 3, scored_cohort.GRID.shape)
    ref = pairs.as_polyset(scored_cohort.squares(centres, 6.0))
    mov = pairs.as_polyset(scored_cohort.squares(centres + [2.0, -1.0], 6.0))
    moved = pairs.with_xy(mov, mov.xy + 0.7)
    plain = pairs.record(ref, mov, ref, moved, 0.5)
    kept = pairs.Geometry()
    area, cent = kept.area_cent(ref)
    assert kept.area_cent(ref)[1] is cent  # the same set: not computed again
    held = dict(kept._kept)
    for _ in range(2):
        again = pairs.record(ref, mov, ref, moved, 0.5, kept)
        assert list(again) == list(plain)
        assert all(str(again[k]) == str(plain[k]) for k in plain)
    assert kept._kept == held  # a record leaves nothing behind in the caller's
    # a copy with the same vertices is another set, and a moved one is never the cached
    twin = pairs.with_xy(ref, ref.xy.copy())
    assert kept.area_cent(twin)[1] is not cent
    np.testing.assert_array_equal(kept.area_cent(twin)[1], cent)
    shifted = pairs.with_xy(ref, ref.xy + 5.0)
    np.testing.assert_allclose(kept.area_cent(shifted)[1], cent + 5.0)
    a, c = pairs.cp.feature_area_centroid(ref)
    np.testing.assert_array_equal(area, a)
    np.testing.assert_array_equal(cent, c)


# ── a record is reused only when nothing it rests on has changed ──────────────
SMALL = ("P1_r1", "P2_r1")


@pytest.fixture
def small(cohort, tmp_path, monkeypatch):
    """Two cases, scored once: ``(root, out, spy, n pair records of P1_r1)``."""
    root, out = copy_of(cohort, tmp_path, cases=SMALL)
    spy = Spy(monkeypatch)
    scored(root, out, workers=1)
    assert sorted(spy.cases) == sorted(SMALL) and spy.records > 20
    n = len(list((out / "cache/scores/multiplex/P1_r1/pairs").iterdir()))
    spy.reset()
    return root, out, spy, n


def check(small, cases, records=None, **kw):
    """After a change: these cases are computed again (and that many pair records), and
    the tables are those of a scoring that reads no record at all."""
    root, out, spy, _ = small
    with_records = scored(root, out, workers=1, **kw)
    assert sorted(spy.cases) == sorted(cases), spy.cases
    if records is not None:
        assert spy.records == records
    same(with_records, scored(root, out, workers=1, fresh=True, **kw))
    spy.reset()
    # ... and what was just written is reused in its turn
    same(with_records, scored(root, out, workers=1, **kw))
    assert spy.cases == [] and spy.records == 0
    return with_records


def _rewrite_npz(path, **changes):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update({k: f(arrays[k]) for k, f in changes.items()})
    np.savez_compressed(path, **arrays)


def test_nothing_changed_nothing_is_computed(small):
    check(small, cases=[], records=0)


def test_a_changed_result_is_computed_again_and_only_what_read_it(small):
    root, out, spy, _ = small
    before = tables(out)
    case = load_case(root / "multiplex" / "P1_r1")
    _rewrite_npz(
        result_dir(out, "stare", case) / "warped.npz",
        cells_xy=lambda xy: xy + np.float32(1.25),
    )
    # what read that FILE: `stare` whole and off each of the two owners' seams, on
    # both sets (a pair record is keyed by the file's bytes, not by the array in it).
    # No other variant's pair record is computed again
    after = check(small, cases=["P1_r1"], records=6)
    assert after["cells.csv"] != before["cells.csv"]
    assert after["cells_nuclei.csv"] == before["cells_nuclei.csv"]


def test_a_changed_seam_mask_recomputes_every_row_off_that_owners_seams(small):
    root, out, spy, n = small
    case = load_case(root / "multiplex" / "P1_r1")
    _rewrite_npz(result_dir(out, "tiles", case) / "warped.npz", band=lambda b: ~b)
    # `tiles` itself (its file changed: whole and off two owners, on both sets), and
    # the OUTLINES of every other source off `tiles` (the unregistered pose and the
    # four other results: `dropme` is one since no row is dropped any more). Their
    # nuclei and their rows off the other owner are kept
    check(small, cases=["P1_r1"], records=6 + 5)
    # the same for an owner whose mask is on the REFERENCE nuclei
    _rewrite_npz(result_dir(out, "blocks", case) / "warped.npz", band_ref=lambda b: ~b)
    check(small, cases=["P1_r1"], records=6 + 5)


def test_what_a_result_says_changes_its_rows_and_no_pair_record(small):
    root, out, spy, _ = small
    before = tables(out)
    path = result_dir(out, "rtm", load_case(root / "multiplex" / "P1_r1")) / "run.json"
    run = json.loads(path.read_text())
    write_json(path, {**run, "selfcheck_status": "inconsistent", "problem": "odd"})
    after = check(small, cases=["P1_r1"], records=0)
    assert after["cells.csv"] != before["cells.csv"]
    # a result that now says it failed: its rows are the unregistered pose's, which
    # the case has a pair record of already
    write_json(path, {**run, "ok": False, "error": "it broke", "cause": "oom"})
    check(small, cases=["P1_r1"], records=0)
    # ... and one that vanished altogether
    shutil.rmtree(path.parent)
    check(small, cases=["P1_r1"], records=0)


def test_another_list_of_variants_is_another_record(small):
    root, out, spy, _ = small
    before = tables(out)
    # one more expected name: every case has one more (imputed) row, no new pair record
    after = check(small, cases=SMALL, records=0, expected=["ghost", "ghost2"])
    assert after["cells.csv"] != before["cells.csv"]
    # the same names in another order are another table (the rows follow that order)
    check(small, cases=SMALL, records=0, expected=["ghost2", "ghost"])
    # a new result folder joins the list
    case = load_case(root / "multiplex" / "P1_r1")
    # (on the dataset that has it; and its bytes are `stare`'s, so are its pair records)
    shutil.copytree(result_dir(out, "stare", case), result_dir(out, "newer", case))
    check(small, cases=SMALL, records=0, expected=["ghost2", "ghost"])


def test_other_seam_owners_other_checks_are_other_records(small):
    """One thing at a time, each from the state the one before left. (No row is ever
    dropped any more: what a selfcheck said is reported, never a gate.)"""
    root, out, spy, _ = small
    kw = {"seam_owners": ("tiles",)}
    check(small, cases=SMALL, records=0, **kw)
    # a selfcheck status is set at table time, not in the case key (fix round 1, I2):
    # the re-score reads every case record, and its tables are a fresh scoring's
    kw["row_checks"] = {**KW["row_checks"], "stare": "no_stage_image"}
    check(small, cases=[], records=0, **kw)


def test_another_submission_is_another_record(small, monkeypatch):
    root, out, spy, _ = small
    # no result of this folder was made by a submission: scored alone, none of them ran
    monkeypatch.setenv(score_mod.SUBMISSION_ENV, "20261012T000000-1")
    after = check(small, cases=SMALL, records=0)
    assert b"not run by this submission" in after["cells.csv"]


def test_changed_points_are_a_changed_case(small):
    root, out, spy, _ = small
    case = load_case(root / "multiplex" / "P1_r1")
    _rewrite_npz(case.points_path, cells_ref_xy=lambda xy: xy + np.float32(0.5))
    # the outlines changed: every result is stale for them (never a method's failure),
    # the unregistered pose is new, whole and off each owner that still has a mask
    after = check(small, cases=["P1_r1"])
    assert b"prepared again" in after["cells.csv"]
    # case.json alone (the pixel size): every pair record of the case is another
    path = Path(case.dir) / "case.json"
    write_json(path, {**json.loads(path.read_text()), "pixel_size_um": 0.25})
    check(small, cases=["P1_r1"])


def test_another_scorer_version_reads_no_record(small, monkeypatch):
    root, out, spy, n = small
    monkeypatch.setattr(score_mod, "RECORD_VERSION", score_mod.RECORD_VERSION + 1)
    scored(root, out, workers=1)
    assert sorted(spy.cases) == sorted(SMALL)
    assert spy.records >= n  # every pair record as well, not only the rows


def test_a_changed_constant_or_code_is_another_scorer(small, monkeypatch):
    root, out, spy, n = small
    before = tables(out)
    # a constant changed in the running process (not in a file): still another scorer
    monkeypatch.setattr(pairs, "NULL_DIRECTIONS", 4)
    after = check(small, cases=SMALL)
    assert after["cells.csv"] != before["cells.csv"]
    monkeypatch.undo()
    spy2 = Spy(monkeypatch)
    scored(root, out, workers=1)  # back to the constants of the files
    spy2.reset()
    # the pipeline's cell_pairs.py, were it another file
    other = root / "cell_pairs.py"
    shutil.copy(pairs.CELL_PAIRS_PATH, other)
    other.write_text(other.read_text() + "\n# changed\n")
    monkeypatch.setattr(pairs, "CELL_PAIRS_PATH", other)
    scored(root, out, workers=1)
    assert sorted(spy2.cases) == sorted(SMALL) and spy2.records >= n
    assert (
        score_mod.pair_code()["cell_pairs.py"] == score_mod.case_code()["cell_pairs.py"]
    )
    for name in ("pairs.py", "cases.py"):
        assert score_mod.pair_code()[name] == score_mod.file_sha256(
            REPO / "benchmarks" / "regbench" / name
        )
    monkeypatch.undo()
    # another NumPy, SciPy or scikit-image (the rasteriser) is another scorer too
    spy3 = Spy(monkeypatch)
    scored(root, out, workers=1)  # back to the pipeline's own file
    spy3.reset()
    scored(root, out, workers=1)
    assert spy3.cases == []
    libraries = {**score_mod._libraries(), "scikit-image": "0.0.1"}
    monkeypatch.setattr(score_mod, "_libraries", lambda: libraries)
    scored(root, out, workers=1)
    assert sorted(spy3.cases) == sorted(SMALL) and spy3.records >= n
    assert {"score.py", "pairs.py", "cases.py"} <= set(score_mod.case_code())


def test_the_keys_cover_every_module_that_decides_a_number(tmp_path, monkeypatch):
    """A record's key holds the bytes of every module its numbers are computed by,
    found by following the imports inside the package; and, of ``score.py``, the
    helpers and names a pair record is read and assembled with."""
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "__init__.py").write_text("from .. import far\n")
    (pkg / "top.py").write_text(
        "import numpy\nfrom . import helper\nfrom .sub import deep\n"
        "def f():\n    from .lazy import g\n    return g\n"
    )
    for name in ("helper", "lazy", "far", "unused"):
        (pkg / f"{name}.py").write_text(f"X = '{name}'\n")
    (pkg / "sub" / "deep.py").write_text("from ..helper import X\n")
    got = score_mod.code_files([pkg / "top.py"], pkg)
    assert list(got) == [
        "far.py",
        "helper.py",
        "lazy.py",
        "sub/__init__.py",
        "sub/deep.py",
        "top.py",
    ]

    def digest():
        files = score_mod.code_files([pkg / "top.py"], pkg).values()
        return [score_mod.file_sha256(f) for f in files]

    before = digest()
    (pkg / "unused.py").write_text("X = 'edited, and imported by nothing'\n")
    assert digest() == before
    (pkg / "sub" / "deep.py").write_text("from ..helper import X\nY = 2  # edited\n")
    assert digest() != before
    # the scorer's own keys
    pair, case = score_mod.pair_code(), score_mod.case_code()
    assert {"pairs.py", "cases.py", "mapping.py", "cell_pairs.py"} <= set(pair)
    assert {"mapping.py", "ladder.py", "stages.py", "methods/__init__.py"} <= set(case)
    for name in ("pairs.py", "mapping.py"):
        assert (
            pair[name]
            == case[name]
            == score_mod.file_sha256(REPO / "benchmarks" / "regbench" / name)
        )
    # a helper of score.py that reads a result: another helper is another pair key
    real = score_mod._load_run

    def other(out, variant, case):  # the same result, read by other code
        return real(out, variant, case)

    monkeypatch.setattr(score_mod, "_load_run", other)
    assert score_mod.pair_code() != pair
    monkeypatch.undo()
    assert score_mod.pair_code() == pair
    # and a name its arrays are read under
    monkeypatch.setitem(score_mod.WARPED_KEYS, "outlines", ("xy", "band", "band_ref"))
    assert score_mod.pair_code() != pair


def test_a_record_that_cannot_be_read_is_computed_again(small, capsys):
    root, out, spy, n = small
    folder = out / "cache" / "scores" / "multiplex" / "P1_r1"
    rec = folder / "case.rec"
    whole = rec.read_bytes()
    # cut short (a job killed while copying, a full disk)
    rec.write_bytes(whole[: len(whole) // 2])
    capsys.readouterr()
    check(small, cases=["P1_r1"], records=0)
    assert "unreadable record (cut short or damaged)" in capsys.readouterr().err
    assert rec.read_bytes() != whole[: len(whole) // 2]  # replaced by a whole one
    # not a record at all, and an empty file
    for junk in (b"not a record\n", b""):
        rec.write_bytes(junk)
        check(small, cases=["P1_r1"], records=0)
    # a whole, well-formed record of ANOTHER case in its place: its key is not this one
    other = folder.parent / "P2_r1" / "case.rec"
    rec.write_bytes(other.read_bytes())
    capsys.readouterr()
    check(small, cases=["P1_r1"], records=0)
    assert "made for other inputs" in capsys.readouterr().err
    # a pair record cut short: that one pair record is computed again, and no other
    rec.unlink()
    first = sorted((folder / "pairs").iterdir())[0]
    first.write_bytes(first.read_bytes()[:40])
    check(small, cases=["P1_r1"], records=1)


def test_the_key_is_written_where_a_person_can_read_it(small):
    root, out, _, _ = small
    said = json.loads((out / "cache/scores/multiplex/P1_r1/case.json").read_text())
    key = said["key"]
    assert said["key_sha256"] == score_mod._key_sha256(key)
    assert key["record_version"] == score_mod.RECORD_VERSION
    # the formerly dropped row is scored like any other: nothing is left out
    assert key["variants"][0] == "initial" and "dropme" in key["variants"]
    assert "dropped" not in key and key["submission"] == ""
    assert key["seam_owners"] == list(scored_cohort.OWNERS)
    assert set(key["results"]) == set(key["variants"]) - {"initial"}
    for name in ("stare", "rtm", "tiles", "blocks"):
        assert len(key["results"][name]["warped.npz"]) == 64
        assert len(key["results"][name]["run.json"]) == 64
    assert key["results"]["ghost"] == {"run.json": "", "warped.npz": ""}
    crash = key["results"]["crash"]
    assert crash["run.json"] and not crash["warped.npz"]
    assert "checks" not in key  # set at table time (fix round 1, I2)
    assert len(key["points.npz"]) == len(key["case.json"]) == 64
    assert {"numpy", "scipy", "scikit-image", "python"} <= set(key["libraries"])
    assert key["settings"]["null_directions"] == pairs.NULL_DIRECTIONS
    both = ["blocks", "tiles"]
    assert said["seam_masks"] == {"outlines": both, "nuclei": both}
    assert said["rows"]["cells"] > 0 and said["seconds"] >= 0
    # the records are under `cache`, which is never read as a variant
    assert score_mod.RECORDS[0] in score_mod.RESERVED
    assert "cache" not in score_mod.find_variants(out, "multiplex")


def test_one_worker_starts_no_process(small, monkeypatch):
    import multiprocessing

    root, out, _, _ = small

    def refused(*a, **kw):
        raise AssertionError("a pool was started with one worker")

    monkeypatch.setattr(multiprocessing, "get_context", refused)
    scored(root, out, workers=1, fresh=True)
    score_mod.score_case(
        root, out, "multiplex", index=0, workers=1, fresh=True, **CASE_KW
    )


def test_the_two_levels_are_never_nested(small, monkeypatch):
    """Whole cases side by side when there are at least as many as workers, else the
    pair records of one case at a time: never a pool inside a pool."""
    import multiprocessing

    root, out, _, _ = small
    made, given = [], []

    class Pool:
        def __init__(self, n):
            made.append(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, tasks):
            return [fn(task) for task in tasks]

        def imap_unordered(self, fn, tasks):
            given.append(fn.__name__)
            for task in reversed(list(tasks)):  # any order: the result must not care
                yield fn(task)

    context = types.SimpleNamespace(Pool=Pool)
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: context)
    quiet = dict(
        fresh=True, with_tissue=False
    )  # the tissue block has a pool of its own
    # two workers, two cases: the cases side by side, each serial inside
    two_at_once = scored(root, out, workers=2, **quiet)
    assert (made, given) == ([2], ["_case_task"])
    made.clear(), given.clear()
    # five workers, two cases: ONE pool, each case's pair records in it, in turn
    inside = scored(root, out, workers=5, **quiet)
    assert (made, given) == ([5], ["_pair_task", "_pair_task"])
    same(two_at_once, inside)
    monkeypatch.undo()
    same(inside, scored(root, out, workers=1, **quiet))


def test_score_case_names_its_case_and_leaves_dev_cases_alone(small):
    root, out, spy, _ = small
    with pytest.raises(SystemExit, match="--index 7 is not one of the 2 prepared"):
        score_mod.score_case(root, out, "multiplex", index=7, **CASE_KW)
    with pytest.raises(SystemExit, match="no prepared multiplex case named nope"):
        score_mod.score_case(root, out, "multiplex", case_id="nope", **CASE_KW)
    shutil.copytree(root / "multiplex" / "P1_r1", root / "multiplex" / "dev_x")
    path = root / "multiplex" / "dev_x" / "case.json"
    write_json(path, {**json.loads(path.read_text()), "case_id": "dev_x"})
    ids = [c.case_id for c in list_cases(root, "multiplex")]
    assert ids == ["P1_r1", "P2_r1", "dev_x"]  # the index counts every prepared case
    assert score_mod.score_case(root, out, "multiplex", index=2, **CASE_KW) == "skipped"
    assert score_mod.score_case(root, out, "multiplex", index=0, **CASE_KW) == "kept"
    assert spy.cases == []
    assert score_mod.score_case(
        root, out, "multiplex", index=2, dev=True, **CASE_KW
    ) == ("scored")


# ── the batch script: a task per case, then the tables, through the real CLI ──
def _batch(kind, *args, root, out, **env):
    base = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    base.pop("SLURM_ARRAY_TASK_ID", None)
    base.update(
        PATH=f"{Path(sys.executable).parent}:{base['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(root),
        REGBENCH_OUT=str(out),
        REGBENCH_SCORER_IMAGE=variants.SCORER_IMAGE,  # no image set: the python on PATH
        SLURM_CPUS_PER_TASK="2",
        SLURM_JOB_ID="77",  # the task's log and end line are named after these
        SLURM_JOB_NAME=f"rb-{kind}",
    )
    base.update(env)
    return subprocess.run(
        ["bash", str(SLURM / "score_cohort.sbatch"), kind, *args],
        env=base, capture_output=True, text=True, cwd=REPO,
    )  # fmt: skip


def test_the_array_tasks_records_are_the_ones_the_scorer_reads(cohort, tmp_path):
    """`score_cohort.sbatch case <dataset>` once per case, then `score`: both are told
    the submission the same way, so the scorer computes nothing."""
    root, out = copy_of(cohort, tmp_path)
    sub = "20261012T101010-5"
    for path in out.glob("*/*/*/run.json"):
        if path.parts[-4] != "rtm":  # one run this submission neither ran nor reused
            run = json.loads(path.read_text())
            write_json(path, {**run, "submissions": [sub]})
    env = {"REGBENCH_SUBMISSION": sub}
    for i in range(4):
        done = _batch(
            "case", "multiplex", root=root, out=out, SLURM_ARRAY_TASK_ID=str(i), **env
        )
        assert done.returncode == 0, done.stderr
        last = done.stdout.strip().splitlines()[-1]
        assert last.startswith("REGBENCH_END kind=score_case dataset=multiplex ")
        assert f"unit=index_{i} record=scored status=ok cause=- whose=-" in last
        assert "started (no record)" in done.stderr
        # the task made (or said it could not make) its case's tissue pieces first
        said = done.stdout.index("score-case multiplex index")
        tissue = re.search(r"tissue: \d+ row\(s\)|: not computed here", done.stdout)
        assert tissue and tissue.start() < said
    assert not (out / "tables").exists()  # a case task writes no table
    said = json.loads((out / "cache/scores/multiplex/P1_r1/case.json").read_text())
    key = said["key"]
    assert key["submission"] == sub and "dropped" not in key
    assert "stare" in key["variants"] and "dropme" in key["variants"]
    # the same task again finds its record
    again = _batch(
        "case", "multiplex", root=root, out=out, SLURM_ARRAY_TASK_ID="0", **env
    )
    assert "record=kept" in again.stdout and "started" not in again.stderr
    # a task without an index, and one past the end, fail and say whose that is
    none = _batch("case", "multiplex", root=root, out=out, **env)
    assert none.returncode != 0 and "submit with --array" in none.stderr
    past = _batch(
        "case", "multiplex", root=root, out=out, SLURM_ARRAY_TASK_ID="9", **env
    )
    assert past.returncode != 0
    assert (
        past.stdout.strip()
        .splitlines()[-1]
        .startswith(
            "REGBENCH_END kind=score_case dataset=multiplex unit=index_9 "
            "status=no_end_line cause=score_case_unfinished whose=benchmark"
        )
    )
    # the scorer: the four multiplex cases are read, none is computed, and the tables
    # are those of one scoring from nothing
    done = _batch("score", root=root, out=out, **env)
    assert done.returncode == 0, done.stderr
    read = [ln for ln in done.stderr.splitlines() if "rows read from its record" in ln]
    assert len(read) == 4 and all("multiplex/" in ln for ln in read)
    started = [ln for ln in done.stderr.splitlines() if ": started (" in ln]
    assert started == []
    assert done.stdout.strip().splitlines()[-1].startswith("REGBENCH_END kind=score ")
    # the verdicts first (no runs file: every run of the cohort), then the tables that name them
    verdicts = (out / "logs" / "verdicts.tsv").read_text().splitlines()
    assert verdicts[0] == "run\tcheck\tstatus\treference\trows\tlog"
    assert len(verdicts) == 1 + 2 * len(variants.runs_for(None))
    assert done.stdout.index("REGBENCH_END kind=verdict ") < done.stdout.index(
        "REGBENCH_END kind=score "
    )
    merged = tables(out)
    assert b"## What the selfchecks said" in merged["summary.md"]
    assert merged["launch.csv"].startswith(b"run,selfcheck,realcheck,reference,log")
    assert b"not run by this submission" in merged["cells.csv"]
    shutil.rmtree(out / "cache" / "scores")
    whole = _batch("score", root=root, out=out, **env)
    assert whole.returncode == 0 and whole.stderr.count(": started (no record)") == 4
    same(merged, tables(out))


# ── a result the scheduler ended: settled BEFORE the cases are scored ─────────
def _killed(out, case, variant, launch_id):
    """That result as the driver leaves it when its process is killed outright."""
    path = result_dir(out, variant, case) / "run.json"
    run = json.loads(path.read_text())
    killed = {**run, "ok": False, "cause": "killed", "launch_id": launch_id}
    write_json(path, {**killed, "error": "the process was killed"})
    return path


def _failure_causes(out, case_id, variant):
    with open(out / "tables" / "failure_log.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    return sorted(
        r["cause"] for r in rows if (r["case_id"], r["variant"]) == (case_id, variant)
    )


def test_a_record_made_before_a_result_was_settled_is_never_reused(cohort, tmp_path):
    """`regbench settle` rewrites the run.json of a result the scheduler ended. In the
    launcher's order (settle job, case tasks, scorer) no record is made before it. If a
    case IS scored before its result is settled (the order until 2026-10-12, and what
    happens when the settle job could not run), the record made then is not read: the
    case is scored again from the result as settled, and the tables are those of a
    scoring from nothing."""
    root, out = copy_of(cohort, tmp_path)
    sub = "20261012T111111-6"
    for path in out.glob("*/*/*/run.json"):
        run = json.loads(path.read_text())
        write_json(path, {**run, "submissions": [sub]})
    one = load_case(root / "multiplex" / "P1_r1")
    two = load_case(root / "multiplex" / "P2_r1")
    first = _killed(out, one, "stare", "500_0")
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "jobs.tsv").write_text("t\trb-stare-multiplex-512g\t500\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sacct").write_text(
        "#!/bin/bash\n"
        'echo "$*" >> "$SACCT_LOG"\n'
        'printf "500_0|NODE_FAIL|30|00:00:20\\n500_1|TIMEOUT|86400|4-00:00:00\\n"\n'
    )
    (fake / "sacct").chmod(0o755)
    env = {
        "REGBENCH_SUBMISSION": sub,
        "PATH": f"{fake}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "SACCT_LOG": str(tmp_path / "sacct.log"),
    }

    def case_tasks():
        for i in range(4):
            done = _batch(
                "case",
                "multiplex",
                root=root,
                out=out,
                SLURM_ARRAY_TASK_ID=str(i),
                **env,
            )
            assert done.returncode == 0, done.stderr
            assert "settle" not in done.stdout  # a case task never settles

    # 1. every case is scored BEFORE the settle: the wrong order
    case_tasks()
    key = json.loads((out / "cache/scores/multiplex/P1_r1/case.json").read_text())[
        "key"
    ]
    before = key["results"]["stare"]["run.json"]
    assert before == score_mod.file_sha256(first)
    # 2. the settle job: the scheduler says the node failed
    done = _batch("settle", root=root, out=out, **env)
    assert done.returncode == 0, done.stderr
    assert "settle: 1 infrastructure" in done.stdout
    assert (
        done.stdout.strip()
        .splitlines()[-1]
        .startswith("REGBENCH_END kind=settle status=ok cause=- whose=-")
    )
    assert json.loads(first.read_text())["cause"] == "infrastructure"
    assert (out / "logs" / "settled").read_text() == sub + "\n"
    assert score_mod.file_sha256(first) != before
    # 3. the scorer does not settle again, and does not read the record of P1_r1: it
    # says why, scores that case again, and reads the three whose results did not change
    done = _batch("score", root=root, out=out, **env)
    assert done.returncode == 0, done.stderr
    assert "settle: done by the settle job" in done.stdout
    assert len((tmp_path / "sacct.log").read_text().splitlines()) == 2  # + the last one
    started = [ln for ln in done.stderr.splitlines() if ": started (" in ln]
    again = [ln for ln in started if "multiplex/" in ln]
    assert (
        len(again) == 1
        and "multiplex/P1_r1: started (made for other inputs" in again[0]
    )
    assert "results.stare.run.json" in again[0]
    read = [ln for ln in done.stderr.splitlines() if "rows read from its record" in ln]
    assert len(read) == 3 and not any("P1_r1" in ln for ln in read)
    assert _failure_causes(out, "P1_r1", "stare") == ["infrastructure"]
    with open(out / "tables" / "failure_log.csv", newline="") as fh:
        assert "killed" not in {r["cause"] for r in csv.DictReader(fh)}
    # ... the pair records made before the settle are all still good (a failed result
    # is scored at the unregistered pose, which the case has a record of)
    (kept,) = [ln for ln in done.stderr.splitlines() if "P1_r1" in ln and "kept" in ln]
    assert kept.endswith(", 0 to compute")

    # 4. the fallback: the settle job left no word (it was lost), a second result was
    # killed, and its case was scored before anyone settled it
    (out / "logs" / "settled").unlink()
    second = _killed(out, two, "stare", "500_1")
    case_tasks()
    done = _batch("score", root=root, out=out, **env)
    assert done.returncode == 0, done.stderr
    assert "the settle job left no record for this submission; settling here" in (
        done.stdout
    )
    assert "settle: 1 timeout" in done.stdout
    assert json.loads(second.read_text())["cause"] == "timeout"
    started = [ln for ln in done.stderr.splitlines() if ": started (" in ln]
    again = [ln for ln in started if "multiplex/" in ln]
    assert (
        len(again) == 1
        and "multiplex/P2_r1: started (made for other inputs" in again[0]
    )
    assert _failure_causes(out, "P2_r1", "stare") == ["timeout"]
    assert _failure_causes(out, "P1_r1", "stare") == ["infrastructure"]
    assert (out / "logs" / "settled").read_text() == sub + "\n"
    # either way the tables are those of one scoring from nothing
    merged = tables(out)
    shutil.rmtree(out / "cache" / "scores")
    whole = _batch("score", root=root, out=out, **env)
    assert whole.returncode == 0 and "settle: done by the settle job" in whole.stdout
    same(merged, tables(out))


def test_a_settle_job_that_cannot_ask_the_scheduler_says_so_and_blocks_nothing(
    cohort, tmp_path
):
    root, out = copy_of(cohort, tmp_path, cases=SMALL)
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "settled").write_text("an-earlier-scoring\n")
    # no jobs.tsv: nothing was submitted into this folder by the launcher
    done = _batch("settle", root=root, out=out, REGBENCH_SUBMISSION="S-9")
    assert done.returncode == 0 and "settle: no " in done.stdout
    assert (
        done.stdout.strip()
        .splitlines()[-1]
        .startswith(
            "REGBENCH_END kind=settle status=failed cause=settle_unfinished whose=benchmark"
        )
    )
    assert not (out / "logs" / "settled").exists()  # an older word is not left standing
    # a scheduler that answers nothing
    # a run task (settle_ids asks the accounting of run tasks only: fleet wave 2, F6)
    (out / "logs" / "jobs.tsv").write_text("t\trb-x-multiplex-16g\t9\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sacct").write_text("#!/bin/bash\nexit 1\n")
    (fake / "sacct").chmod(0o755)
    path = f"{fake}:{Path(sys.executable).parent}:{os.environ['PATH']}"
    done = _batch("settle", root=root, out=out, REGBENCH_SUBMISSION="S-9", PATH=path)
    assert done.returncode == 0 and "sacct gave no accounting" in done.stdout
    assert not (out / "logs" / "settled").exists()
