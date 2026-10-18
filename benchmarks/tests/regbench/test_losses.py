"""Task 28: no loss is fatal to anything but itself, and every loss is a row.

Every test here scores the small cohort of ``scored_cohort.py`` (or a copy of it with one
thing broken) and reads what the scorer wrote: ``tables/losses.csv`` and the per-stage
tables beside it.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from benchmarks.regbench import cli, losses
from benchmarks.regbench import score as score_mod

from . import scored_cohort

KW = scored_cohort.SCORE_KW


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root, out, _ = scored_cohort.build(tmp_path_factory.mktemp("losses"))
    return root, out


def copy_of(cohort, tmp):
    root, out = cohort
    new_root, new_out = tmp / "cases", tmp / "out"
    shutil.copytree(root, new_root)
    for d in sorted(p for p in out.iterdir() if p.name not in score_mod.RESERVED):
        shutil.copytree(d, new_out / d.name)
    return new_root, new_out


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def run(root, out, **kw):
    return score_mod.score(root, out, ["multiplex"], with_tissue=False, **{**KW, **kw})


def run_json(out, method, case):
    return out / method / "multiplex" / case / "run.json"


def edit_run(out, method, case, **fields):
    path = run_json(out, method, case)
    rec = json.loads(path.read_text())
    rec.update(fields)
    path.write_text(json.dumps(rec))


def losses_of(out):
    return read(out / "tables" / "losses.csv")


def scored_values(out, skip=()):
    """``cells.csv`` rows keyed by (method, case), the value columns only."""
    keep = ("pair_fraction", "dice_matched", "within_1um", "within_3um")
    return {
        (r["method"], r["case_id"], r.get("cell_set", "")): tuple(r.get(k) for k in keep)
        for r in read(out / "tables" / "cells.csv")
        if (r["method"], r["case_id"]) not in skip
    }


# ── the table itself ─────────────────────────────────────────────────────────────────
def test_losses_csv_is_always_written_and_names_every_loss(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    run(root, out)
    path = out / "tables" / "losses.csv"
    assert header(path) == list(losses.COLUMNS)
    rows = losses_of(out)
    for r in rows:
        assert r["stage"] in losses.STAGES and r["blame"] in losses.BLAMES
        assert r["effect"] in losses.EFFECTS and r["cause"] in losses.CAUSES
        assert losses.CAUSES[r["cause"]][0] == r["blame"]
    crash = [r for r in rows if r["method"] == "crash"]
    assert {(r["cause"], r["blame"], r["effect"]) for r in crash} == {
        ("exception", "method", "imputed")
    }
    assert len(crash) == 4
    # the expected method nobody ran is a loss too, folded per method
    assert any(r["method"] == "ghost" and r["cause"] == "not_run" for r in rows)
    summary = json.loads((out / "tables" / "summary.json").read_text())
    assert summary["n_losses"] == len(rows)
    assert summary["losses"] and summary["n_scored_cases"] == 4
    assert "## Where every loss shows up" in (out / "tables" / "summary.md").read_text()
    readme = (out / "tables" / "README.txt").read_text()
    for cause in losses.CAUSES:
        assert cause in readme


def test_a_clean_run_writes_the_header_alone(tmp_path):
    path = tmp_path / "losses.csv"
    losses.write(path, [])
    assert path.read_text().strip() == ",".join(losses.COLUMNS)


# ── B1: a case the scorer could not score ────────────────────────────────────────────
def test_a_crashing_case_is_a_score_error_and_the_others_are_scored(
    cohort, tmp_path, monkeypatch
):
    clean_root, clean_out = copy_of(cohort, tmp_path / "clean")
    run(clean_root, clean_out)
    root, out = copy_of(cohort, tmp_path / "broken")
    real = score_mod._score_case

    def boom(job, *a, **kw):
        if job[1].case_id == "P2_r1":
            raise ValueError("the scorer fell over")
        return real(job, *a, **kw)

    monkeypatch.setattr(score_mod, "_score_case", boom)
    got = run(root, out)
    (err,) = read(out / "tables" / "score_errors.csv")
    assert err["case_id"] == "P2_r1" and "the scorer fell over" in err["error"]
    assert ".py:" in err["where"]  # the last frame
    rows = [r for r in losses_of(out) if r["cause"] == "score_error"]
    assert [(r["case_id"], r["blame"], r["stage"]) for r in rows] == [
        ("P2_r1", "infrastructure", "score")
    ]
    assert got["n_scored_cases"] == 3
    # the other cases' scores are those of the clean scoring
    clean = scored_values(clean_out)
    broken = scored_values(out)
    assert broken and all(clean[k] == v for k, v in broken.items())
    assert not any(k[1] == "P2_r1" for k in broken)


def test_nothing_scored_exits_non_zero_after_writing_losses(tmp_path, monkeypatch):
    root, out = tmp_path / "cases", tmp_path / "out"
    scored_cohort.multiplex_case(root, "P1_r1", "P1", 1)
    from benchmarks.regbench import methods

    methods.run_case(scored_cohort.forward("stare", 0.4), scored_cohort.multiplex_case(
        root, "P1_r1", "P1", 1), out)

    def boom(job, *a, **kw):
        raise ValueError("nothing scores")

    monkeypatch.setattr(score_mod, "_score_case", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "--cases", str(root), "--out", str(out), "--dataset",
                  "multiplex", "--no-tissue"])  # fmt: skip
    assert exc.value.code not in (0, None)
    assert [r["cause"] for r in losses_of(out)] == ["score_error"]


# ── S4 / S1: what a result's record cannot say ──────────────────────────────────────
def test_an_unreadable_result_is_its_own_cause(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    run_json(out, "stare", "P1_r1").write_text("{ not json")
    (out / "stare" / "multiplex" / "P1_r2" / "warped.npz").unlink()
    run(root, out)
    got = {
        (r["method"], r["case_id"]): r for r in losses_of(out) if r["method"] == "stare"
    }
    for case in ("P1_r1", "P1_r2"):
        r = got[("stare", case)]
        assert (r["cause"], r["blame"], r["stage"]) == (
            "result_unreadable", "infrastructure", "score"
        )
        assert r["source"].endswith(f"stare/multiplex/{case}/run.json")
    assert "result_unreadable" in score_mod.NOT_THE_METHOD


def test_an_unsettled_kill_is_not_the_methods(cohort, tmp_path, monkeypatch):
    root, out = copy_of(cohort, tmp_path)
    edit_run(out, "rtm", "P1_r1", ok=False, cause="killed", error="killed")
    (out / "rtm" / "multiplex" / "P1_r1" / "warped.npz").unlink()
    got = run(root, out)
    (r,) = [r for r in losses_of(out) if r["method"] == "rtm"
            and r["case_id"] == "P1_r1"]  # fmt: skip
    assert (r["cause"], r["blame"]) == ("killed_unsettled", "infrastructure")
    assert "killed_unsettled" in score_mod.NOT_THE_METHOD
    (att,) = [a for a in got["attrition"] if a["method"] == "rtm"]
    assert att["killed_unsettled"] == 1
    assert att["settled"] == ""  # no launcher submitted into the folder
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "jobs.tsv").write_text("")
    monkeypatch.setenv(score_mod.SUBMISSION_ENV, "s42")
    got = run(root, out)
    assert {a["settled"] for a in got["attrition"]} == {"no"}
    assert json.loads((out / "tables" / "summary.json").read_text())["settled"] == "no"
    (out / "logs" / "settled").write_text("s42\n")
    got = run(root, out)
    assert {a["settled"] for a in got["attrition"]} == {"yes"}


def test_a_capped_rung_is_the_methods_and_imputed(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    edit_run(out, "rtm", "P1_r2", ok=False, cause="capped",
             error="capped at 512 GB: the ladder's next rung is past the cap")  # fmt: skip
    (out / "rtm" / "multiplex" / "P1_r2" / "warped.npz").unlink()
    run(root, out)
    (r,) = [r for r in losses_of(out) if r["case_id"] == "P1_r2"
            and r["method"] == "rtm"]  # fmt: skip
    assert (r["cause"], r["blame"], r["effect"]) == ("capped", "method", "capped")
    assert "512 GB" in r["detail"]
    assert "capped" not in score_mod.NOT_THE_METHOD


def test_a_held_tile_submitter_is_held(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    edit_run(out, "rtm", "P3_r1", ok=False, cause="not_submitted",
             error=f"gave up: {losses.HELD_MARK} JobHeldAdmin) after 3 polls")  # fmt: skip
    (out / "rtm" / "multiplex" / "P3_r1" / "warped.npz").unlink()
    run(root, out)
    (r,) = [r for r in losses_of(out) if r["case_id"] == "P3_r1"
            and r["method"] == "rtm"]  # fmt: skip
    assert (r["cause"], r["blame"]) == ("held", "infrastructure")
    assert "JobHeldAdmin" in r["detail"]


# ── B2: the per-method common block, beside the strict one ───────────────────────────
def test_the_per_method_block_is_written_beside_the_strict_one(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    run(root, out)
    t = out / "tables"
    strict = read(t / "common_block.csv")
    per = read(t / "common_block_per_method.csv")
    assert strict and {r["common_block"] for r in strict} == {"strict"}
    assert per and {r["common_block"] for r in per} == {"per_method"}
    assert {r["common_block"] for r in read(t / "summary_common_per_method.csv")} == {
        "per_method"
    }
    # the strict block stays the headline: summary_cells is the strict one
    assert {r.get("common_block") for r in read(t / "summary_cells.csv")
            if r.get("common_block")} <= {"strict"}  # fmt: skip


# ── S10 / S11 / B9: flags carried on the rows ────────────────────────────────────────
def test_a_fallback_pixel_size_and_a_low_agreement_slide_are_flagged(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    path = root / "multiplex" / "P1_r1" / "case.json"
    rec = json.loads(path.read_text())
    rec.setdefault("extra", {})["pixel_size_source"] = "fallback"
    sets = rec["extra"].setdefault("cell_sets", {})
    sets.setdefault("mov", {}).setdefault("nuclei", {}).update(
        {"low_agreement": True, "agreement": {"one_nucleus_frac": 0.71}}
    )
    path.write_text(json.dumps(rec))
    got = run(root, out)
    cells = [r for r in read(out / "tables" / "cells.csv") if r["case_id"] == "P1_r1"]
    assert cells and {r["pixel_size_source"] for r in cells} == {"fallback_0.325"}
    assert {r["low_agreement"] for r in cells} == {"1"}
    others = [r for r in read(out / "tables" / "cells.csv") if r["case_id"] != "P1_r1"]
    assert {r["pixel_size_source"] for r in others} == {"metadata"}
    checks = read(out / "tables" / "nuclei_checks.csv")
    assert header(out / "tables" / "nuclei_checks.csv") == list(
        score_mod.NUCLEI_CHECK_COLUMNS
    )
    assert [r for r in checks if r["low_agreement"] == "1"]
    causes = {(r["cause"], r["effect"]) for r in losses_of(out)}
    assert ("low_agreement", "flagged") in causes
    assert ("pixel_size_fallback", "flagged") in causes
    summ = [r for r in got["summary_cells"] if r.get("n_low_agreement_slides") not in (None, "")]
    assert summ and max(int(r["n_low_agreement_slides"]) for r in summ) >= 1
    assert max(int(r["n_pixel_size_fallback"]) for r in got["summary_cells"]) >= 1


def test_every_finite_only_maximum_counts_its_nan(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    run(root, out)
    cols = header(out / "tables" / "summary_cells.csv")
    for name in ("unmapped_frac_n_nan", "fold_pct_n_nan", "coverage_n_nan"):
        assert name in cols, name


# ── the docs ─────────────────────────────────────────────────────────────────────────
def test_the_readme_section_lists_every_cause_with_its_blame():
    text = (Path(score_mod.__file__).parent / "README.md").read_text()
    start = text.index("## Where every loss shows up")
    section = text[start : text.index("\n## ", start + 1)]
    for cause, (blame, stage, _) in losses.CAUSES.items():
        assert f"| `{cause}` | {blame} | {stage} |" in section, cause
    for table in ("submit_dropped.csv", "manifest_dropped.csv", "prepare_dropped.csv",
                  "nuclei_checks.csv", "score_errors.csv", "unmarked_units.csv"):  # fmt: skip
        assert table in section, table
    assert "never gate" in section and "integrity refusals" in section


# ── end to end: six kinds of loss in one scoring ─────────────────────────────────────
def _keyed(rows):
    """The rows as a multiset, without what the crashing case's drop replaces: the
    rows of P2_r1 itself (it is one ``score_error`` row now) and the count of cases
    in the folded ``not_run`` row of the method nobody ran (one case fewer)."""
    from collections import Counter

    return Counter(
        (r["stage"], r["method"], r["case_id"], r["cause"], r["blame"], r["effect"])
        for r in rows
        if (r["case_id"] != "P2_r1" or r["cause"] == "score_error")
        and r["method"] != "ghost"
    )


def test_six_kinds_of_loss_are_six_rows_and_nothing_else_moves(
    cohort, tmp_path, monkeypatch
):
    """A crashing case, an unreadable run.json, an unsettled kill, a capped rung, a
    low-agreement slide and a run the launcher's preflight dropped, in ONE scoring:
    each is exactly its row of losses.csv, every table is written, and every other
    case keeps the scores of the clean scoring."""
    clean_root, clean_out = copy_of(cohort, tmp_path / "clean")
    run(clean_root, clean_out)
    root, out = copy_of(cohort, tmp_path / "broken")
    # 1. the scoring of one case raises
    real = score_mod._score_case

    def boom(job, *a, **kw):
        if job[1].case_id == "P2_r1":
            raise ValueError("the scorer fell over")
        return real(job, *a, **kw)

    monkeypatch.setattr(score_mod, "_score_case", boom)
    # 2. an unreadable record
    run_json(out, "stare", "P1_r1").write_text("{ not json")
    # 3. a kill nobody settled
    edit_run(out, "rtm", "P1_r1", ok=False, cause="killed", error="killed")
    (out / "rtm" / "multiplex" / "P1_r1" / "warped.npz").unlink()
    # 4. a capped rung
    edit_run(out, "rtm", "P1_r2", ok=False, cause="capped",
             error="capped at 512 GB: the next rung cannot be granted")  # fmt: skip
    (out / "rtm" / "multiplex" / "P1_r2" / "warped.npz").unlink()
    # 5. a slide under the agreement threshold, kept and flagged
    path = root / "multiplex" / "P3_r1" / "case.json"
    rec = json.loads(path.read_text())
    sets = rec.setdefault("extra", {}).setdefault("cell_sets", {})
    sets.setdefault("mov", {}).setdefault("nuclei", {}).update(
        {"low_agreement": True, "agreement": {"one_nucleus_frac": 0.71}}
    )
    path.write_text(json.dumps(rec))
    # 6. a run the launcher dropped at its preflight
    (out / "submit_dropped.csv").write_text(
        "run,reason,detail\nvalis_mirage,preflight_failed,the image would not start\n"
    )
    got = run(root, out)
    assert got["n_scored_cases"] == 3
    added = _keyed(losses_of(out)) - _keyed(losses_of(clean_out))
    gone = _keyed(losses_of(clean_out)) - _keyed(losses_of(out))
    assert not gone
    assert any(r["method"] == "ghost" and r["cause"] == "not_run" for r in losses_of(out))
    assert sorted(added.elements()) == sorted([
        ("score", "", "P2_r1", "score_error", "infrastructure", "dropped"),
        ("score", "stare", "P1_r1", "result_unreadable", "infrastructure", "dropped"),
        ("settle", "rtm", "P1_r1", "killed_unsettled", "infrastructure", "dropped"),
        ("run", "rtm", "P1_r2", "capped", "method", "capped"),
        ("nuclei", "", "P3_r1", "low_agreement", "input", "flagged"),
        ("submit", "", "", "preflight_failed", "infrastructure", "not_run"),
    ])  # fmt: skip
    # every table is written, the per-stage ones included
    t = out / "tables"
    for name in ("losses.csv", "score_errors.csv", "nuclei_checks.csv", "cells.csv",
                 "summary.json", "summary.md", "README.txt", "attrition.csv",
                 "common_block_per_method.csv"):  # fmt: skip
        assert (t / name).is_file(), name
    summary = json.loads((t / "summary.json").read_text())
    assert summary["n_losses"] == sum(1 for _ in losses_of(out))
    # nothing else moves: every other (method, case) has the clean scoring's numbers
    touched = {("stare", "P1_r1"), ("rtm", "P1_r1"), ("rtm", "P1_r2")}
    broken = scored_values(out, skip=touched)
    clean = scored_values(clean_out, skip=touched)
    assert broken and not any(k[1] == "P2_r1" for k in broken)
    assert {k: v for k, v in clean.items() if k[1] != "P2_r1"} == broken
