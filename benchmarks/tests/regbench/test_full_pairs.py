"""Task 21b: the full pairings beside the headline (``regbench.full_pairs``).

``full_lsa``: the exact minimum-weight assignment in EVERY component, with no mutual-NN
fallback and no work cap (``pairs.exact_pairs(max_work=inf)``), solved only where it
differs from the pair record's own ``*_exact`` (``EXACT_TOO_LARGE``, or a record made
before the exact columns existed) and copied everywhere else. ``full_mnn``: the record's
own ``*_mnn`` columns (mutual nearest neighbour on the whole case, already uncapped),
reported beside the headline under that name. Both rank nothing.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import failure_log, full_pairs, losses, pairs, score
from benchmarks.regbench.paper import schema
from benchmarks.regbench.paper import tables as paper_tables
from benchmarks.tests.regbench import scored_cohort
from benchmarks.tests.regbench.test_pairs import (
    GRID,
    as_set,
    squares,
    tissue_centroids,
)

RB = Path(score.__file__).resolve().parent
NAN = float("nan")

# score.pair_code() at BASE 76ca9bd7, computed in-process before Task 21b touched a
# file: the full pairings move no pair record
BASE_PAIR_CODE = {
    "cases.py": "40afd42ef2a3f9453116251ef9cd254c9bf8a51dd7e5c7bf7b5ec74fe9ddf561",
    "cell_pairs.py": "83f7dd39ac8d8881d6fecb7e9e6045c03de379a5b2ee9e05ba5d2571b67f6fdc",
    "mapping.py": "b1a5a3a46b5675daf1343757bccc2d0541f9a0e585358cb8037bfde466a3824e",
    "pairs.py": "e572d9580e9d1c91a814c9a2291d1e7ff1cc27e8141f506341184c98a1927674",
    "score.py:_Case,_keep_features,_load_run,of_this_submission,_seam_masks": (
        "1145a3e665506c6f68565eefe06e3a4d832044d6056f2c72d21e139dd32ec096"
    ),
}
SCORED = tuple(k for k in pairs._SCORE_KEYS if k not in ("n_ref", "n_moving"))


def same(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def dense_optimum(a, b, radius):
    """An independent solve of the pipeline's objective (the most pairs, then the least
    total distance): one dense ``scipy.optimize.linear_sum_assignment`` over the whole
    case, a pair beyond the gate costing more than every real pair together."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    d = cdist(a, b)
    big = (len(a) + len(b)) * radius + 1.0
    r, c = linear_sum_assignment(np.where(d <= radius, d, big))
    keep = d[r, c] <= radius
    return r[keep], c[keep], d[r, c][keep]


@pytest.fixture
def small_fallback(monkeypatch):
    """The pipeline's fallback at 3 cells a component (so a tiny case uses it) and the
    exact solver's bound at 10 units of work (so the pair record leaves it blank)."""
    real = pairs.cp.match_cells
    monkeypatch.setattr(pairs.cp, "DEFAULT_MAX_COMPONENT_CELLS", 3)
    monkeypatch.setattr(
        pairs.cp,
        "match_cells",
        lambda *a, **k: real(*a, **{**k, "max_component_cells": 3}),
    )
    monkeypatch.setattr(pairs, "EXACT_MAX_WORK", 10.0)


# ── what a full-LSA number is ──────────────────────────────────────────────────
def test_above_the_cap_full_lsa_is_the_dense_optimum(small_fallback):
    a, b = tissue_centroids(14, seed=5)
    ref, mov = squares(a, half=6.0), squares(b, half=6.0)
    radius, px = 30.0, 0.5
    rec = pairs.score(ref, mov, radius, px, exact=True)
    rec["match_radius_px"] = radius
    assert rec["n_fallback_cells"] > 0
    assert rec["exact_status"].startswith("not solved: a component of ")
    assert math.isnan(rec["pair_fraction_exact"]) and full_pairs.needs_solve(rec)
    before = dict(rec)
    got = full_pairs.full_record(rec, lambda: (ref, mov, ref, mov, px, None))
    assert rec == before or all(same(rec[k], before[k]) for k in rec)  # headline kept
    assert got["full_lsa_status"] == full_pairs.SOLVED
    cent_a, cent_b = (pairs.area_centroid(s)[1] for s in (ref, mov))
    r, c, d = dense_optimum(cent_a, cent_b, radius)
    want = pairs._pair_scores(
        pairs.as_polyset(ref), pairs.as_polyset(mov), r, c, d, px, None, None
    )
    assert got["n_pairs_full_lsa"] == len(r) > rec["n_pairs"]
    for k in SCORED:
        assert got[f"{k}_full_lsa"] == pytest.approx(want[k], abs=1e-9), k
    ea, eb, *_ = full_pairs.full_lsa_pairs(cent_a, cent_b, radius)
    assert as_set(ea, eb) == as_set(r, c)
    # the cost of the solve is recorded, never capped
    assert got["full_lsa_seconds"] >= 0.0
    assert got["full_lsa_largest_component_cells"] == rec["largest_component_cells"]
    assert got["full_lsa_work"] > pairs.EXACT_MAX_WORK


@pytest.mark.parametrize("status", ["same", "solved"])
def test_a_record_the_exact_solver_finished_is_copied_and_nothing_is_solved(
    status, monkeypatch
):
    ref = squares(GRID)
    moved = pairs.with_xy(ref, ref.xy + [3.0, 1.0])
    rec = pairs.record(ref, ref, ref, moved, pixel_size_um=0.5)
    assert rec["exact_status"] == pairs.EXACT_SAME
    if status == "solved":
        rec = {
            **rec,
            "exact_status": pairs.EXACT_SOLVED.format(n=2),
            "pair_fraction_exact": 0.25,
        }

    def boom(*a, **k):
        raise AssertionError("the full LSA was solved where it is the exact one")

    monkeypatch.setattr(full_pairs, "solve", boom)
    monkeypatch.setattr(full_pairs, "full_lsa_pairs", boom)
    assert not full_pairs.needs_solve(rec)
    got = full_pairs.full_record(rec, boom)
    assert got["full_lsa_status"] == full_pairs.FROM_EXACT
    for k in SCORED:
        assert same(got[f"{k}_full_lsa"], rec[f"{k}_exact"]), k
    assert math.isnan(got["full_lsa_work"]) and math.isnan(got["full_lsa_seconds"])


def test_a_record_made_before_the_exact_columns_is_solved():
    ref = squares(GRID)
    moved = pairs.with_xy(ref, ref.xy + [3.0, 1.0])
    rec = pairs.record(ref, ref, ref, moved, pixel_size_um=0.5)
    old = {
        k: v for k, v in rec.items() if not k.endswith("_exact") and k != "exact_status"
    }
    assert full_pairs.needs_solve(old)
    got = full_pairs.full_record(old, lambda: (ref, ref, ref, moved, 0.5, None))
    assert got["full_lsa_status"] == full_pairs.SOLVED
    for k in SCORED:
        assert same(got[f"{k}_full_lsa"], rec[f"{k}_exact"]), k


def test_not_certified_and_a_failure_are_nan_with_the_reason(small_fallback, monkeypatch):
    a, b = tissue_centroids(10, seed=2)
    ref, mov = squares(a, half=6.0), squares(b, half=6.0)
    rec = {"exact_status": "not solved: a component of 9 cells", "match_radius_px": 30.0}
    monkeypatch.setattr(
        pairs, "solve_component", lambda *a, **k: (np.empty(0, int), np.empty(0, int), False)
    )
    got = full_pairs.full_record(rec, lambda: (ref, mov, ref, mov, 0.5, None))
    assert got["full_lsa_status"] == full_pairs.NOT_CERTIFIED
    assert got["full_lsa_detail"].startswith("not reported: the solver's result")
    assert all(math.isnan(got[f"{k}_full_lsa"]) for k in SCORED)

    def oom(*a, **k):
        raise MemoryError("no room")

    monkeypatch.setattr(full_pairs, "full_lsa_pairs", oom)
    got = full_pairs.full_record(rec, lambda: (ref, mov, ref, mov, 0.5, None))
    assert got["full_lsa_status"] == full_pairs.FAILED
    assert "MemoryError" in got["full_lsa_detail"] and "no room" in got["full_lsa_detail"]
    assert all(math.isnan(got[f"{k}_full_lsa"]) for k in SCORED)
    assert full_pairs.FAILED not in full_pairs.KEPT  # tried again by the next scoring


# ── its records: their own set, their own key ─────────────────────────────────
def test_the_pair_code_is_the_one_of_base():
    assert score.pair_code() == BASE_PAIR_CODE


def test_the_full_pairs_key_is_its_own(monkeypatch):
    code = score.full_pairs_code()
    assert code["full_pairs.py"] == score.file_sha256(RB / "full_pairs.py")
    assert not set(code) & set(score.pair_code())  # pairs.py is the pair code's
    real = score.file_sha256
    monkeypatch.setattr(
        score,
        "file_sha256",
        lambda p: "edited" if Path(p).name == "full_pairs.py" else real(p),
    )
    assert score.full_pairs_code() != code
    assert score.pair_code() == BASE_PAIR_CODE


class Spy:
    def __init__(self, monkeypatch):
        self.records = self.fulls = 0
        record, full = pairs.record, full_pairs.full_record

        def pair_record(*a, **k):
            self.records += 1
            return record(*a, **k)

        def full_record(*a, **k):
            self.fulls += 1
            return full(*a, **k)

        monkeypatch.setattr(pairs, "record", pair_record)
        monkeypatch.setattr(full_pairs, "full_record", full_record)


def _cells(out, name="cells.csv"):
    with open(out / "tables" / name, newline="") as fh:
        return list(csv.DictReader(fh))


def test_full_records_are_kept_beside_the_pair_records_and_keyed_apart(
    tmp_path, monkeypatch
):
    root, out, made = scored_cohort.build(tmp_path)
    spy = Spy(monkeypatch)
    score.score(root, out, **scored_cohort.SCORE_KW)
    folder = out / "cache" / "scores" / "multiplex" / "P1_r1"
    n_pairs = len(list((folder / "pairs").iterdir()))
    fulls = sorted((folder / full_pairs.RECORDS).iterdir())
    assert n_pairs and len(fulls) == n_pairs
    first = _cells(out)
    assert {r["full_lsa_status"] for r in first} == {full_pairs.FROM_EXACT}
    for r in first:
        for k in SCORED:
            assert r[f"{k}_full_lsa"] == r[f"{k}_exact"], k
    # nothing changed: every record is read
    spy.records = spy.fulls = 0
    score.score(root, out, **scored_cohort.SCORE_KW, fresh=False)
    assert spy.records == 0 and spy.fulls == 0
    # full_pairs.py changed: the full records are made again, and no pair record
    stamps = {p: p.stat().st_mtime_ns for p in (folder / "pairs").iterdir()}
    real = score.file_sha256
    monkeypatch.setattr(
        score,
        "file_sha256",
        lambda p: "edited" if Path(p).name == "full_pairs.py" else real(p),
    )
    score.score(root, out, **scored_cohort.SCORE_KW)
    assert spy.records == 0 and spy.fulls > 0
    assert {p: p.stat().st_mtime_ns for p in (folder / "pairs").iterdir()} == stamps
    assert _cells(out) == first


@pytest.mark.parametrize(
    "how, cause",
    [("not_certified", "full_lsa_not_certified"), ("raises", "full_lsa_failed")],
)
def test_a_full_lsa_that_cannot_be_reported_is_a_loss_and_drops_no_row(
    tmp_path, monkeypatch, how, cause
):
    root, out, made = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    before = _cells(out)
    monkeypatch.setattr(full_pairs, "needs_solve", lambda rec: True)
    if how == "not_certified":
        monkeypatch.setattr(
            full_pairs,
            "full_lsa_pairs",
            lambda *a, **k: (None, None, None, "not reported: test", 7, 12.0),
        )
    else:

        def raises(*a, **k):
            raise RuntimeError("the solver fell over")

        monkeypatch.setattr(full_pairs, "full_lsa_pairs", raises)
    import shutil

    for kept in (out / "cache" / "scores").glob(f"*/*/{full_pairs.RECORDS}"):
        shutil.rmtree(kept)  # the first scoring's full records: made again below
    got = score.score(root, out, **scored_cohort.SCORE_KW, fresh=True)
    after = _cells(out)
    assert len(after) == len(before)
    status = full_pairs.NOT_CERTIFIED if how == "not_certified" else full_pairs.FAILED
    for was, now in zip(before, after):
        assert {k: now[k] for k in was if "full_lsa" not in k} == {
            k: was[k] for k in was if "full_lsa" not in k
        }
        assert now["full_lsa_status"] == status
        assert now["pair_fraction_full_lsa"] in ("", "nan")
    lost = [r for r in got["losses"] if r["cause"] == cause]
    assert lost and {r["stage"] for r in lost} == {"full_pairs"}
    assert {r["effect"] for r in lost} == {"flagged"}
    with open(out / "tables" / "losses.csv", newline="") as fh:
        assert any(r["cause"] == cause for r in csv.DictReader(fh))
    # the solve is never saved when it raised: the next scoring tries it again
    folder = out / "cache" / "scores" / "multiplex" / "P1_r1" / full_pairs.RECORDS
    made = list(folder.iterdir()) if folder.is_dir() else []
    assert bool(made) == (how == "not_certified")


def test_the_loss_causes_have_an_owner():
    assert set(full_pairs.LOSS_CAUSE) == {full_pairs.NOT_CERTIFIED, full_pairs.FAILED}
    for cause in full_pairs.LOSS_CAUSE.values():
        assert losses.stage_of(cause) == "full_pairs"
        assert losses.blame_of(cause) == losses.INFRA  # the benchmark's own step
        assert failure_log.classify(cause) == failure_log.BENCHMARK


# ── beside every headline, ranking nothing ────────────────────────────────────
FULL_SUMMARY = (
    "pair_fraction_full_lsa_med",
    "pair_fraction_full_mnn_med",
    "dice_matched_full_lsa_med",
    "dice_matched_full_mnn_med",
    "fallback_frac_med",
    "fallback_frac_max",
    "n_full_lsa",
)


@pytest.fixture
def scored_out(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    got = score.score(root, out, **scored_cohort.SCORE_KW)
    return root, out, got


def test_the_summaries_carry_the_full_pairings_and_the_fallback(scored_out):
    root, out, got = scored_out
    for name in (
        "summary_cells.csv",
        "summary_cells_nuclei.csv",
        "summary_common_per_method.csv",
        "nonband_summary.csv",
    ):
        rows = _cells(out, name)
        assert rows and all(set(FULL_SUMMARY) <= set(r) for r in rows), name
    summary = {(r["method"], r["subset"]): r for r in got["summary_cells"]}
    stare = summary[("stare", "all")]
    cells = [r for r in got["cells"] if r["method"] == "stare"]
    assert stare["n_full_lsa"] == len(cells)
    assert stare["fallback_frac_med"] == 0.0 and stare["fallback_frac_max"] == 0.0
    per = score._per_patient(cells, "pair_fraction_mnn")
    assert stare["pair_fraction_full_mnn_med"] == pytest.approx(
        float(np.median(list(per.values())))
    )
    text = (out / "tables" / "summary.md").read_text()
    assert "fallback_frac" in text and "full_lsa" in text and "full_mnn" in text
    assert "no fallback and no cap" in text


def test_a_row_without_full_columns_is_nan_in_the_summary(scored_out):
    _, _, got = scored_out
    bare = [
        {k: v for k, v in r.items() if "full_lsa" not in k} for r in got["cells"]
    ]
    rows = score.summarise_cells(bare, scored_cohort.SCORE_KW["primary"])
    assert rows and all(math.isnan(r["dice_matched_full_lsa_med"]) for r in rows)
    assert all(r["n_full_lsa"] == 0 for r in rows)


def test_the_full_columns_change_no_selection(scored_out, monkeypatch):
    """Wild full-LSA / full-MNN numbers move no common block, no best row, no paired
    difference: they are read beside the headline only."""
    _, out, got = scored_out
    primary = scored_cohort.SCORE_KW["primary"]
    rng = np.random.default_rng(0)
    wild = [
        {
            **r,
            **{f"{k}_full_lsa": float(rng.random()) for k in SCORED},
            **{f"{k}_mnn": float(rng.random()) for k in pairs._SCORE_KEYS},
        }
        for r in got["cells"]
    ]
    assert score.common_block(wild, primary) == score.common_block(got["cells"], primary)
    a, b = score.paired_vs_primary(wild), score.paired_vs_primary(got["cells"])
    assert a and len(a) == len(b)
    assert all(set(x) == set(y) and all(same(x[k], y[k]) for k in x) for x, y in zip(a, b))
    fam = {"stare": "stare", "dropme": "stare", "rtm": "rtm", "crash": "rtm"}
    monkeypatch.setattr(paper_tables, "family_of", lambda v: fam.get(v))
    monkeypatch.setattr(paper_tables, "terminal_rows", lambda ds: {k: k for k in fam})
    pp = paper_tables.per_pair(out, "multiplex")
    rng = np.random.default_rng(1)
    pp_wild = [
        {
            **r,
            "dice_matched_full_lsa": float(rng.random()),
            "dice_matched_full_mnn": float(rng.random()),
        }
        for r in pp
    ]
    for f in ("stare", "rtm"):
        assert paper_tables.best_of(pp_wild, f, "multiplex", {}) == paper_tables.best_of(
            pp, f, "multiplex", {}
        )


PAPER_FULL = (
    "dice_full_lsa_med",
    "dice_full_mnn_med",
    "fallback_frac_med",
    "fallback_frac_max",
    "n_full_lsa",
)


def test_the_paper_tables_carry_them_from_the_schema(scored_out, monkeypatch):
    _, out, _ = scored_out
    for name in ("method_summary", "valis_stages", "stare_tiers"):
        assert set(PAPER_FULL) <= set(schema.columns(name)), name
    per_pair_cols = (
        "dice_matched_full_lsa",
        "dice_matched_full_mnn",
        "pair_fraction_full_lsa",
        "pair_fraction_full_mnn",
        "fallback_frac",
        "full_lsa_status",
    )
    assert set(per_pair_cols) <= set(schema.columns("per_pair"))
    fam = {"stare": "stare", "dropme": "stare", "rtm": "rtm", "crash": "rtm"}
    monkeypatch.setattr(paper_tables, "family_of", lambda v: fam.get(v))
    monkeypatch.setattr(paper_tables, "terminal_rows", lambda ds: {k: k for k in fam})
    built = paper_tables.build(out, "multiplex")
    rows = {(r["family"], r["which"]): r for r in built["method_summary"]}
    best = rows[("stare", "best")]
    assert best["n_full_lsa"] == best["n_pairs"]
    assert best["fallback_frac_max"] == 0.0 and best["fallback_frac_med"] == 0.0
    assert math.isfinite(best["dice_full_lsa_med"])
    assert math.isfinite(best["dice_full_mnn_med"])
    pp = [r for r in built["per_pair"] if r["row"] == "stare"]
    assert all(r["full_lsa_status"] == full_pairs.FROM_EXACT for r in pp)
    # a cells table made before Task 21b: every new column is the literal NA
    for name in ("cells.csv", "cells_nuclei.csv"):
        path = out / "tables" / name
        rows_ = _cells(out, name)
        cols = [c for c in rows_[0] if "full_lsa" not in c and c != "fallback_frac"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_)
    paper_tables.write_all(out, "multiplex")
    with open(out / "paper" / "method_summary.csv", newline="") as fh:
        got = list(csv.DictReader(fh))
    acc = [r for r in got if r["row"] in ("stare", "dropme", "rtm")]
    assert acc and all(r["dice_full_lsa_med"] == schema.NA for r in acc)
    assert all(r["fallback_frac_max"] == schema.NA for r in acc)
    assert all(r["dice_full_mnn_med"] != schema.NA for r in acc)  # the _mnn columns
    with open(out / "paper" / "per_pair.csv", newline="") as fh:
        pp = list(csv.DictReader(fh))
    assert all(r["full_lsa_status"] == schema.NA for r in pp)


@pytest.mark.usefixtures("public_datasets")
def test_fallback_frac_beside_every_headline_of_a_fluorescence_public_set(tmp_path):
    from benchmarks.tests.regbench.test_public_never_pooled import _public_case

    root, out, made = scored_cohort.build(tmp_path)
    _public_case(root, out)
    score.score(root, out, datasets=["multiplex", "nucset"], **scored_cohort.SCORE_KW)
    for where in (out / "tables", out / "tables" / "nucset"):
        with open(where / "summary_cells.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows and all(set(FULL_SUMMARY) <= set(r) for r in rows), where
        text = (where / "summary.md").read_text()
        assert "fallback_frac (med / max)" in text, where
        assert "pair_fraction_full_lsa" in text and "dice_matched_full_mnn" in text


def test_the_collector_copies_the_full_columns(scored_out):
    from benchmarks.regbench.paper import collect

    _, out, got = scored_out
    rows = collect._cell_rows(got["cells"], "regbench-interim:x", "outlines")
    stare = [r for r in rows if r["row"] == "stare"]
    assert stare and all(r["full_lsa_status"] == full_pairs.FROM_EXACT for r in stare)
    assert all(r["fallback_frac"] != schema.NA for r in stare)
    assert all(r["dice_matched_full_mnn"] != schema.NA for r in stare)

