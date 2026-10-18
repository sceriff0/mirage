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


def test_nothing_scored_exits_non_zero_after_writing_losses(
    tmp_path, monkeypatch, capsys
):
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
    # the end line names the cause it was (score errors), not an unreadable record
    (end,) = [ln for ln in capsys.readouterr().out.splitlines()
              if ln.startswith("REGBENCH_END kind=score ")]  # fmt: skip
    assert "cause=score_error" in end and "unreadable_record" not in end


# ── S4 / S1: what a result's record cannot say ──────────────────────────────────────
def test_an_unreadable_result_is_its_own_cause(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    run_json(out, "stare", "P1_r1").write_text("{ not json")
    (out / "stare" / "multiplex" / "P1_r2" / "warped.npz").unlink()
    run(root, out)
    got = {
        (r["method"], r["case_id"]): r for r in losses_of(out) if r["method"] == "stare"
    }
    # the source is the file that failed (M7): the record, then the map
    for case, failed in (("P1_r1", "run.json"), ("P1_r2", "warped.npz")):
        r = got[("stare", case)]
        assert (r["cause"], r["blame"], r["stage"]) == (
            "result_unreadable", "infrastructure", "score"
        )
        assert r["source"].endswith(f"stare/multiplex/{case}/{failed}")
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


# ── I1: a result that is there but cannot be opened ──────────────────────────────────
def test_a_result_that_cannot_be_opened_is_one_loss_row_not_an_abort(cohort, tmp_path):
    """``case_key`` hashes every result's files before any per-case guard: a run.json
    that exists but cannot be opened (EACCES, EIO) raised out of the scorer, so there
    were no tables and no losses.csv at all."""
    import os

    root, out = copy_of(cohort, tmp_path)
    path = run_json(out, "stare", "P1_r1")
    os.chmod(path, 0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("this user can read a mode-0 file (root)")
        got = run(root, out)
    finally:
        os.chmod(path, 0o644)
    assert got["n_scored_cases"] == 4
    (r,) = [r for r in losses_of(out) if r["method"] == "stare" and r["case_id"] == "P1_r1"]
    assert (r["cause"], r["blame"], r["effect"]) == (
        "result_unreadable", "infrastructure", "dropped"
    )
    assert r["source"].endswith("stare/multiplex/P1_r1/run.json")


# ── I2: the re-score reads the case records ──────────────────────────────────────────
def test_a_rescore_with_new_selfcheck_statuses_reads_every_case_record(
    cohort, tmp_path, monkeypatch
):
    """B5: rb-score runs before the selfchecks end and rb-rescore after. The selfcheck
    row status was part of ``case_key``, so the re-score recomputed every case of a
    selfchecked run. It is set at table time now: every case is read, and the rows say
    the new status."""
    root, out = copy_of(cohort, tmp_path)
    run(root, out, row_checks={"stare": "not_checked", "rtm": "not_checked"})
    computed = []
    real = score_mod._case_record
    monkeypatch.setattr(
        score_mod, "_case_record",
        lambda job, *a, **kw: computed.append(job[1].case_id) or real(job, *a, **kw),
    )  # fmt: skip
    run(root, out, row_checks={"stare": "ok", "rtm": "no_stage_image"})
    assert computed == []
    cells = read(out / "tables" / "cells.csv")
    assert {r["selfcheck_row"] for r in cells if r["method"] == "stare"} == {"ok"}
    assert {r["selfcheck_row"] for r in cells if r["method"] == "rtm"} == {
        "no_stage_image"
    }


# ── I4 / M1: an empty dataset, and the tissue datasets ───────────────────────────────
def test_an_empty_cohort_beside_a_public_set_scores_the_public_set(tmp_path):
    """An empty dataset raised SystemExit out of the loop, so a public set named after
    it was never scored and the exit was non-zero. It is a dataset with nothing
    scored now: its losses.csv is written, its stale tables are removed, the next
    dataset is scored and the command exits 0."""
    from .test_public_never_pooled import _public_case

    root, out, _ = scored_cohort.build(tmp_path)
    score_mod.score(root, out, ["multiplex"], with_tissue=False, **KW)
    assert (out / "tables" / "cells.csv").is_file()
    _public_case(root, out)
    shutil.rmtree(root / "multiplex")
    got = score_mod.score(root, out, ["multiplex", "ibex"], with_tissue=False, **KW)
    assert got["n_scored_cases"] == 0
    assert got["public"]["ibex"]["n_scored_cases"] == 1
    assert not (out / "tables" / "cells.csv").exists()  # the stale one is gone
    assert header(out / "tables" / "losses.csv") == list(losses.COLUMNS)
    assert (out / "tables" / "ibex" / "cells.csv").is_file()
    code = cli.main(["score", "--cases", str(root), "--out", str(out), "--workers",
                     "1", "--dataset", "multiplex", "ibex"])  # fmt: skip
    assert code in (0, None)


def test_a_tissue_set_writes_its_losses_readme_and_summary(tmp_path):
    from .test_public_never_pooled import _public_case

    root, out, _ = scored_cohort.build(tmp_path)
    _public_case(root, out, dataset="acrobat", case_id="A1")
    score_mod.score(root, out, datasets=["acrobat"], expected=[])
    where = out / "tables" / "acrobat"
    assert header(where / "losses.csv") == list(losses.COLUMNS)
    assert "losses.csv" in (where / "README.txt").read_text()
    summary = json.loads((where / "summary.json").read_text())
    assert summary["n_losses"] == len(read(where / "losses.csv"))
    assert "losses" in summary


# ── I5: a per-stage table that is there and cannot be read ───────────────────────────
@pytest.mark.parametrize(
    "where", ["out/submit_dropped.csv", "out/logs/unmarked_units.csv",
              "cases/manifest_dropped.csv", "cases/prepare_dropped.csv"],
)  # fmt: skip
def test_a_per_stage_table_that_cannot_be_read_is_one_row(cohort, tmp_path, where):
    root, out = copy_of(cohort, tmp_path)
    path = tmp_path / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"run,reason\n\xff\xfe\x00 not text \xff\n")
    run(root, out)
    rows = [r for r in losses_of(out) if r["cause"] == "table_unreadable"]
    assert len(rows) == 1, rows
    (r,) = rows
    assert (r["blame"], r["effect"]) == ("infrastructure", "flagged")
    assert r["source"].endswith(path.name)


# ── M3: the realcheck verdict and the run-level rows ─────────────────────────────────
def test_an_unreadable_realcheck_verdict_is_a_score_row(cohort, tmp_path):
    root, out = copy_of(cohort, tmp_path)
    verdicts = out / "logs" / "verdicts.tsv"
    verdicts.parent.mkdir(parents=True, exist_ok=True)
    verdicts.write_text(
        "run\tcheck\tstatus\treference\trows\tlog\n"
        "stare\tselfcheck\tok\t-\tstare\t-\n"
        "stare\trealcheck\tunreadable\t-\tstare\trealcheck.json unreadable on P1_r1\n"
    )
    run(root, out, verdicts=verdicts)
    (r,) = [r for r in losses_of(out) if r["cause"] == "realcheck_unreadable"]
    assert (r["stage"], r["run"], r["blame"], r["effect"]) == (
        "score", "stare", "infrastructure", "flagged"
    )
    assert "P1_r1" in r["detail"] and r["source"] == "logs/verdicts.tsv"


def test_run_level_rows_are_the_cohorts_and_losses_run_csv_only(cohort, tmp_path):
    """A row of no dataset (a dropped run, an unmarked unit, a compute patient) was in
    every dataset's losses.csv, so a union counted it once per dataset."""
    from .test_public_never_pooled import _public_case

    root, out = copy_of(cohort, tmp_path)
    _public_case(root, out)
    (out / "submit_dropped.csv").write_text(
        "run,reason,detail\nvalis_mirage,preflight_failed,no start\n"
    )
    score_mod.score(root, out, ["multiplex", "ibex"], with_tissue=False, **KW)
    cohort_rows = [r for r in losses_of(out) if r["cause"] == "preflight_failed"]
    assert len(cohort_rows) == 1
    ibex = read(out / "tables" / "ibex" / "losses.csv")
    assert not [r for r in ibex if r["dataset"] == ""]
    run_level = read(out / "tables" / "losses_run.csv")
    assert [r["cause"] for r in run_level] == ["preflight_failed"]
    assert all(r["dataset"] == "" for r in run_level)


def test_an_image_the_tissue_block_could_not_score_is_a_row(tmp_path, monkeypatch):
    from benchmarks.regbench import tissue

    from .test_public_never_pooled import _public_case

    root, out, _ = scored_cohort.build(tmp_path)
    _public_case(root, out, dataset="acrobat", case_id="A1")
    real = tissue.score_all

    def one_missing(jobs, workers=1):
        got = real(jobs, workers)
        got["tissue_missing"] = [{"dataset": "acrobat", "case_id": "A1", "group": "A1",
                                  "variant": "stare", "reason": "no written image"}]
        return got  # fmt: skip

    monkeypatch.setattr(tissue, "score_all", one_missing)
    score_mod.score(root, out, datasets=["acrobat"], expected=[])
    (r,) = [r for r in read(out / "tables" / "acrobat" / "losses.csv")
            if r["cause"] == "tissue_missing"]  # fmt: skip
    assert (r["case_id"], r["method"], r["effect"]) == ("A1", "stare", "dropped")
    assert "no written image" in r["detail"]


# ── B2 (behaviour): strict for everyone, per_method for the method that lost it ──────
def test_a_not_the_methods_loss_shrinks_strict_for_all_but_per_method_for_its_own(
    cohort, tmp_path
):
    clean_root, clean_out = copy_of(cohort, tmp_path / "clean")
    clean = run(clean_root, clean_out)
    root, out = copy_of(cohort, tmp_path / "lost")
    edit_run(out, "rtm", "P1_r1", ok=False, cause="infrastructure", error="node lost")
    (out / "rtm" / "multiplex" / "P1_r1" / "warped.npz").unlink()
    lost = run(root, out)

    def n(got, table):
        return {r["method"]: r["n_cases_common"] for r in got[table]
                if r["in_common_block"]}  # fmt: skip

    before, after = n(clean, "common_block"), n(lost, "common_block")
    assert set(before) >= {"stare", "rtm"}
    for m in before:  # strict: one case fewer for every compared method
        assert after[m] == before[m] - 1, m
    before = n(clean, "common_block_per_method")
    after = n(lost, "common_block_per_method")
    assert after["rtm"] == before["rtm"] - 1
    assert all(after[m] == before[m] for m in before if m != "rtm")
    # the cost block: the same two rules, each named
    for table, rule in (("cost_summary", "strict"),
                        ("cost_summary_per_method", "per_method")):  # fmt: skip
        assert lost[table] and {r["common_block"] for r in lost[table]} == {rule}
    pm = {r["method"]: r["n_patients_common"] for r in lost["cost_summary_per_method"]}
    st = {r["method"]: r["n_patients_common"] for r in lost["cost_summary"]}
    assert st["stare"] == st["rtm"] == pm["rtm"] and pm["stare"] == st["stare"] + 1
    # each per-method table is published wherever its strict one is
    for name in ("cost_summary", "cost_end_to_end_summary"):
        strict = (out / "tables" / f"{name}.csv").is_file()
        assert (out / "tables" / f"{name}_per_method.csv").is_file() == strict, name
    assert (out / "tables" / "cost_summary_per_method.csv").is_file()


def test_a_failed_frame_check_is_an_input_row(cohort, tmp_path):
    """B10 (user ruling: no blocking): a STARE proxy whose frame check failed is
    written and run; the scorer makes its record a ``frame_check_failed`` row."""
    root, out = copy_of(cohort, tmp_path)
    variant = next(
        d.name for d in sorted(out.iterdir()) if (d / "multiplex" / "P1_r1").is_dir()
    )
    prep_dir = out / variant / "multiplex" / "P1_r1" / "stages" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    (prep_dir / "record.json").write_text(json.dumps({
        "proxy": "haem", "frame_check": "failed",
        "frame_check_px": {"dx": 3.0, "dy": 0.0, "max": 3.0, "n": 10},
    }))  # fmt: skip
    run(root, out)
    (r,) = [r for r in losses_of(out) if r["cause"] == "frame_check_failed"]
    assert (r["blame"], r["effect"], r["case_id"], r["dataset"]) == (
        "input", "flagged", "P1_r1", "multiplex"
    )
    assert r["patient_id"] == "P1" and "dx" in r["detail"]
    assert r["source"].endswith("stages/prep/record.json")
