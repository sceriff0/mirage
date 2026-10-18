"""What the selfchecks and the realchecks said, and what the scorer makes of it.

The checks RUN AND REPORT, and never gate (user ruling): `regbench verdict` writes one
line per run and check into ``<out>/logs/verdicts.tsv`` (run, check, status, reference,
rows, log) and always exits 0; the scorer names every verdict in ``launch.csv`` and in
the summary, and sets it on every row of the run. No run waits for a check and no row
is left out because of one.
"""

from __future__ import annotations

import csv
import json

import pytest

from benchmarks.regbench import cli, methods, score, selfcheck, variants
from benchmarks.regbench.cases import result_dir, write_json

from .test_ladder import a_case
from .test_pipeline import good, toy_case

HR = "deeperhistreg_hr_gpu"


@pytest.fixture(autouse=True)
def _no_submission(monkeypatch):
    # a record names a submission only under the launcher; these tests say when it does
    monkeypatch.delenv("REGBENCH_SUBMISSION", raising=False)


def _record(out, run, pairs=(), **statuses):
    path = out / "logs" / "selfcheck" / run / "selfcheck.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variants": statuses, "pairs": list(pairs)}))
    return path


def _verdict(out, *runs, submission=None):
    argv = ["verdict", "--out", str(out), "--runs", " ".join(runs)]
    if submission is not None:
        argv += ["--submission", submission]
    code = cli.main(argv)
    lines = (out / "logs" / "verdicts.tsv").read_text().splitlines()
    return code, lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def _line(lines, run, check="selfcheck"):
    (line,) = [x for x in lines if x[:2] == [run, check]]
    return line


# ── the verdicts file ─────────────────────────────────────────────────────────
def test_the_verdicts_file_has_a_line_per_run_and_check(tmp_path, capsys):
    out = tmp_path / "out"
    _record(out, "stare", stare_rigid="ok", stare="ok")
    code, header, lines = _verdict(out, "stare")
    assert code == 0
    assert header == list(cli.VERDICT_COLUMNS)
    assert header == ["run", "check", "status", "reference", "rows", "log"]
    assert lines == [
        ["stare", "selfcheck", "ok", "-", "stare_rigid,stare", "-"],
        ["stare", "realcheck", "-", "-", "-", "-"],
    ]
    assert all(all(line) for line in lines)  # no empty field
    # a verdict that is not ok is a line like any other, and the exit is still 0
    _record(out, "stare", stare_rigid="ok", stare="inconsistent")
    code, _, lines = _verdict(out, "stare")
    assert code == 0 and _line(lines, "stare")[2] == "inconsistent"
    # every run and both checks, one line each
    _, _, lines = _verdict(out, "stare", "palom", "valis_mirage")
    assert [x[:2] for x in lines] == [
        [run, check]
        for run in ("stare", "palom", "valis_mirage")
        for check in ("selfcheck", "realcheck")
    ]


def test_a_selfcheck_that_left_no_record_is_reported_as_no_selfcheck_record(tmp_path):
    out = tmp_path / "out"
    code, _, lines = _verdict(out, "palom")
    assert code == 0
    assert _line(lines, "palom") == [
        "palom", "selfcheck", "not_checked", "-",
        ",".join(variants.checked_rows("palom")), "no selfcheck record",
    ]  # fmt: skip
    # valid JSON that is not a record (a list) is the same, and nothing is raised
    _record(out, "palom").write_text("[1, 2]")
    code, _, lines = _verdict(out, "palom")
    assert code == 0 and _line(lines, "palom")[2::3] == [
        "not_checked",
        "no selfcheck record",
    ]
    # an unreadable one is the same, and nothing is raised
    path = _record(out, "palom")
    path.write_text("{not json")
    code, _, lines = _verdict(out, "palom")
    assert code == 0 and _line(lines, "palom")[2::3] == [
        "not_checked",
        "no selfcheck record",
    ]


def test_every_verdict_of_a_run_is_kept_and_its_rows_too(tmp_path):
    """DeeperHistReg's one row (the image its writer draws, under the run's label) is
    the row its status is read from, and the row stays in every table whatever its
    selfcheck says."""
    out = tmp_path / "out"
    _record(out, HR, **{HR: "inconsistent"})
    _, _, lines = _verdict(out, HR)
    assert _line(lines, HR)[2:5] == ["inconsistent", "-", HR]
    assert variants.checked_rows(HR) == (HR,)
    # and the row's own verdict travels beside its numbers (selfcheck_row)
    checks = variants.row_checks(out / "logs" / "selfcheck")
    assert checks[HR] == "inconsistent"


# ── a case of the selfcheck that died ─────────────────────────────────────────
def _row(status):
    return {"status": status}


@pytest.mark.parametrize("run", ["valis_mirage", "valis120_hires", "stare", "ashlar"])
@pytest.mark.parametrize("died", ["not_checked", "crashed"])
def test_the_verdict_is_never_better_than_a_case_that_died(tmp_path, run, died):
    """One case ok on every row, another whose process died without a row: the record's
    summary of the rows says ok everywhere, and the verdict must not."""
    names = variants.checked_rows(run)
    record = {
        "status": died,
        "variants": dict.fromkeys(names, "ok"),
        "pairs": [
            {"status": "ok", "variants": {v: _row("ok") for v in names}},
            {
                "status": died,
                "cause": "the selfcheck process exited -9",
                "variants": {},
            },
        ],
    }
    assert variants.selfcheck_status(run, record)[0] == died
    out = tmp_path / "out"
    path = _record(out, run)
    path.write_text(json.dumps(record))
    code, _, lines = _verdict(out, run)
    assert code == 0 and _line(lines, run)[2] == died
    # a case that carries a row of another status than its own: the row decides
    record["pairs"][1] = {
        "status": "ok",
        "variants": {**{v: _row("ok") for v in names}, names[-1]: _row("inconsistent")},
    }
    assert variants.selfcheck_status(run, record)[0] == "inconsistent"
    # and a case that lacks one checked row while it says ok: that row was not checked
    record["pairs"][1] = {"status": "ok", "variants": {names[0]: _row("ok")}}
    want = "ok" if len(names) == 1 else "not_checked"
    assert variants.selfcheck_status(run, record)[0] == want
    record["pairs"][1] = {"status": "ok", "variants": {v: _row("ok") for v in names}}
    assert variants.selfcheck_status(run, record)[0] == "ok"


@pytest.mark.parametrize(
    "method, label", [("valis_mirage", None), ("valis", "valis120_hires")]
)
@pytest.mark.parametrize(
    "code, died", [(-9, "not_checked"), (137, "not_checked"), (1, "crashed")]
)
def test_the_record_of_a_selfcheck_whose_case_died_is_reported_not_ok(
    tmp_path, monkeypatch, method, label, code, died
):
    """The record `selfcheck.run` itself assembles when the process of one case dies:
    every case carries every expected row, and the verdict reads it as not ok."""
    import subprocess
    import types

    expected = methods.variant_names(method, label)
    sizes = selfcheck.sizes_for(label or method)
    assert len(sizes) >= 2
    seen = []

    real = subprocess.run

    def fake(argv, **kw):
        if "--work" not in argv:  # another caller of the module (platform, uname)
            return real(argv, **kw)
        sub = argv[argv.index("--work") + 1]
        seen.append(sub)
        if len(seen) > 1:  # every case after the first: its process dies
            return types.SimpleNamespace(returncode=code)
        pairs = [{"status": "ok", "variants": {v: _row("ok") for v in expected}}]
        (tmp_path / "w").joinpath(sub).mkdir(parents=True, exist_ok=True)
        with open(f"{sub}/selfcheck.json", "w") as fh:
            json.dump({"pairs": pairs}, fh)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake)
    record = selfcheck.run(method, tmp_path / "w", label=label)
    assert len(seen) == len(sizes) and record["status"] == died
    for pair in record["pairs"]:
        assert set(pair["variants"]) >= set(expected)
    lost = [p for p in record["pairs"] if p["status"] != "ok"]
    assert lost and all(
        v["status"] == died and v["filled"]
        for p in lost
        for v in p["variants"].values()
    )
    assert set(record["variants"].values()) == {died}
    name = label or method
    assert variants.selfcheck_status(name, record)[0] == died
    # the summary rows alone (the shape a verdict read before): still not ok
    assert variants.selfcheck_status(name, {**record, "pairs": []})[0] == died
    assert record["not_ok"] is True


def test_a_run_behind_another_runs_selfcheck_is_recorded_under_its_own_name(
    tmp_path, capsys
):
    out = tmp_path / "out"
    _record(out, "ashlar", ashlar="inconsistent")
    assert variants.checked_by("ashlar_clean") == "ashlar"
    code, _, lines = _verdict(out, "ashlar_clean")
    assert code == 0
    assert _line(lines, "ashlar_clean")[:3] == [
        "ashlar_clean",
        "selfcheck",
        "inconsistent",
    ]
    assert _line(lines, "ashlar_clean")[5] == "via ashlar: -"
    # a name of no run is said, never raised, and has no line
    code, _, lines = _verdict(out, "nope", "ashlar")
    assert code == 0 and [x[0] for x in lines] == ["ashlar", "ashlar"]
    assert "no run named 'nope'" in capsys.readouterr().out


def test_the_reference_comparison_is_recorded_beside_the_status(tmp_path):
    """ASHLAR redraws the reference: its second comparison is already in the row's
    status, and is written out so the summary can say which of the two failed."""
    out = tmp_path / "out"
    pair = {"variants": {"ashlar": {"status": "inconsistent",
                                    "reference": {"status": "inconsistent"}}}}  # fmt: skip
    _record(out, "ashlar", [pair], ashlar="inconsistent")
    _, _, lines = _verdict(out, "ashlar")
    assert _line(lines, "ashlar")[2:4] == ["inconsistent", "inconsistent"]


# ── the realcheck ─────────────────────────────────────────────────────────────
def test_the_realcheck_verdict_reads_the_realcheck_rows_own_record(tmp_path):
    """Not the exit code of `run --realcheck`: that is 3 when ANY row is flagged."""
    root, out = tmp_path / "cases", tmp_path / "out"
    first = [a_case(root, "P1_a", "P1"), a_case(root, "P1_b", "P1")]
    a_case(root, "P2_a", "P2")

    def real():
        code, _, lines = _verdict(out, "valis_mirage")
        assert code == 0
        return _line(lines, "valis_mirage", "realcheck")

    assert real() == [
        "valis_mirage", "realcheck", "not_checked", "-", "valis_mirage",
        "no realcheck of this submission beside an ok result",
    ]  # fmt: skip
    for case in first:
        write_json(
            result_dir(out, "valis_mirage", case) / "realcheck.json", {"flag": False}
        )
    # a realcheck beside no run, or beside a failed one, answers for nothing: it is
    # what an earlier launch left behind (review item 13)
    assert real()[2] == "not_checked"
    for case in first:
        write_json(result_dir(out, "valis_mirage", case) / "run.json", {"ok": False})
    assert real()[2] == "not_checked"
    for case in first:
        write_json(
            result_dir(out, "valis_mirage", case) / "run.json",
            {"ok": True, "submissions": ["s1"]},
        )
    # a flag on a row that is not the realcheck's
    write_json(
        result_dir(out, "valis_mirage_rigid", first[0]) / "realcheck.json",
        {"flag": True},
    )
    assert real()[2::3] == ["ok", "-"]
    write_json(
        result_dir(out, "valis_mirage", first[1]) / "realcheck.json", {"flag": True}
    )
    assert real()[2::3] == ["flagged", "flagged on P1_b"]
    # a run without a realcheck has a line all the same, with nothing in it
    _, _, lines = _verdict(out, "palom")
    assert _line(lines, "palom", "realcheck")[2:] == ["-", "-", "-", "-"]


def test_the_realcheck_verdict_reports_not_checked_for_a_record_of_another_submission(
    tmp_path,
):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_a", "P1")
    d = result_dir(out, "valis_mirage", case)
    write_json(d / "realcheck.json", {"flag": True})
    write_json(d / "run.json", {"ok": True, "submissions": ["old"]})
    _, _, lines = _verdict(out, "valis_mirage", submission="new")
    assert _line(lines, "valis_mirage", "realcheck")[2] == "not_checked"
    _, _, lines = _verdict(out, "valis_mirage", submission="old")
    assert _line(lines, "valis_mirage", "realcheck")[2] == "flagged"


def test_the_selfcheck_verdict_reports_not_checked_for_a_record_of_another_submission(
    tmp_path, monkeypatch
):
    """Review item 13: REGBENCH_OUT is reused at the same commit; the new selfcheck
    died before it wrote anything and the old `ok` is still there."""
    out = tmp_path / "out"
    path = _record(out, "stare", stare_rigid="ok", stare="ok")
    path.write_text(json.dumps({**json.loads(path.read_text()), "submission": "old"}))
    _, _, lines = _verdict(out, "stare", submission="old")
    assert _line(lines, "stare")[2] == "ok"
    _, _, lines = _verdict(out, "stare", submission="new")
    assert _line(lines, "stare")[2::3] == [
        "not_checked",
        "selfcheck record of another submission (old)",
    ]
    # the default submission is the launcher's REGBENCH_SUBMISSION
    monkeypatch.setenv("REGBENCH_SUBMISSION", "old")
    _, _, lines = _verdict(out, "stare")
    assert _line(lines, "stare")[2] == "ok"


# ── what the scorer makes of the verdicts ─────────────────────────────────────
VERDICTS = (
    "run\tcheck\tstatus\treference\trows\tlog\n"
    "palom\tselfcheck\tinconsistent\t-\tpalom_affine,palom\t/logs/rb-sc-palom.log\n"
    "palom\trealcheck\t-\t-\t-\t-\n"
    "stare\tselfcheck\tok\t-\tstare_rigid,stare\t-\n"
    "stare\trealcheck\t-\t-\t-\t-\n"
    "valis120\tselfcheck\tcrashed\t-\tvalis120_rigid,valis120\tno selfcheck record\n"
    "valis120\trealcheck\t-\t-\t-\t-\n"
    "valis_mirage\tselfcheck\tok\t-\tvalis_mirage_rigid,valis_mirage\t-\n"
    "valis_mirage\trealcheck\tflagged\t-\tvalis_mirage\tflagged on P1_r1\n"
)


def _section(text):
    return text[text.index("## What the selfchecks said") :].split("\n## ")[0]


def test_the_summary_says_what_every_selfcheck_said(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    methods.run_case(good(), toy_case(root, "P1_r1", "P1"), out)
    path = out / "logs" / "verdicts.tsv"
    path.parent.mkdir(parents=True)
    path.write_text(VERDICTS)
    got = score.score(root, out, ["multiplex"], with_tissue=False, verdicts=path)
    by_run = {g["run"]: g for g in got["launch"]}
    assert set(by_run) == {"palom", "stare", "valis120", "valis_mirage"}
    assert by_run["palom"]["selfcheck"] == "inconsistent"
    assert by_run["valis_mirage"]["realcheck"] == "flagged"
    block = _section((out / "tables" / "summary.md").read_text())
    assert "- `palom`: selfcheck inconsistent; log /logs/rb-sc-palom.log" in block
    assert "- `stare`: selfcheck ok" in block
    assert "- `valis120`: selfcheck crashed; log no selfcheck record" in block
    assert (
        "- `valis_mirage`: selfcheck ok; realcheck flagged (flagged on P1_r1)" in block
    )
    with open(out / "tables" / "launch.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["run", "selfcheck", "realcheck", "reference", "log"]
    assert [r["run"] for r in rows] == ["palom", "stare", "valis120", "valis_mirage"]

    # through the command line
    cli.main(
        ["score", "--cases", str(root), "--out", str(out), "--only-run", "--no-tissue",
         "--verdicts", str(path)]
    )  # fmt: skip
    assert "## What the selfchecks said" in (out / "tables" / "summary.md").read_text()
    # without the verdicts there is no such block and no stale table
    score.score(root, out, ["multiplex"], with_tissue=False)
    assert "What the selfchecks said" not in (out / "tables" / "summary.md").read_text()
    assert not (out / "tables" / "launch.csv").exists()


def test_the_summary_never_says_a_run_was_blocked(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    methods.run_case(good(), toy_case(root, "P1_r1", "P1"), out)
    path = out / "logs" / "verdicts.tsv"
    path.parent.mkdir(parents=True)
    path.write_text(VERDICTS)
    score.score(root, out, ["multiplex"], with_tissue=False, verdicts=path)
    text = (out / "tables" / "summary.md").read_text()
    assert "## What the selfchecks said" in text
    assert "blocked" not in text.lower()
    assert "dropped" not in text.lower()
    assert score.launch_markdown([]) == []


def test_a_failed_selfcheck_is_reported_end_to_end(tmp_path):
    """A run whose selfcheck record says inconsistent, scored the way score_cohort.sbatch
    scores it (`verdict`, then `score --verdicts`): it is in verdicts.tsv, in the
    summary, in `selfcheck_status` on every row of the run, in selfcheck_masks.csv, and
    its rows are in the tables."""
    root, out = tmp_path / "cases", tmp_path / "out"
    for case_id, group in (("P1_r1", "P1"), ("P2_r1", "P2")):
        methods.run_case(good("palom"), toy_case(root, case_id, group), out)
    sets = {name: {"n_dots": 40, "found_frac": 0.5} for name in selfcheck.SETS}
    pair = {
        "case_id": "c1",
        "status": "inconsistent",
        "variants": {
            "palom_affine": {"status": "ok", "sets": sets},
            "palom": {"status": "inconsistent", "sets": sets},
        },
    }
    _record(out, "palom", [pair], palom_affine="ok", palom="inconsistent")
    assert cli.main(["verdict", "--out", str(out), "--runs", "palom"]) == 0
    verdicts = out / "logs" / "verdicts.tsv"
    assert "palom\tselfcheck\tinconsistent\t" in verdicts.read_text()
    assert cli.main(
        ["score", "--cases", str(root), "--out", str(out), "--only-run", "--no-tissue",
         "--verdicts", str(verdicts)]
    ) == 0  # fmt: skip
    tables = out / "tables"
    assert "- `palom`: selfcheck inconsistent" in _section(
        (tables / "summary.md").read_text()
    )
    with open(tables / "cells.csv", newline="") as fh:
        cells = [r for r in csv.DictReader(fh) if r["method"] == "palom"]
    assert len(cells) == 2  # its rows are scored, one per case
    assert {r["selfcheck_status"] for r in cells} == {"inconsistent"}
    assert {r["selfcheck_row"] for r in cells} == {"inconsistent"}
    with open(tables / "selfcheck_masks.csv", newline="") as fh:
        masks = list(csv.DictReader(fh))
    assert {(r["run"], r["variant"]) for r in masks} >= {("palom", "palom")}
    with open(tables / "summary_cells.csv", newline="") as fh:
        assert "palom" in {r["method"] for r in csv.DictReader(fh)}


# ── what the launcher sizes its arrays from ───────────────────────────────────
def test_list_sizes_counts_cases_patients_and_the_largest_patient(tmp_path, capsys):
    root = tmp_path / "cases"
    for case_id, group in (
        ("P1_a", "P1"),
        ("P1_b", "P1"),
        ("P1_c", "P1"),
        ("P2_a", "P2"),
    ):
        a_case(root, case_id, group)
    capsys.readouterr()
    assert (
        cli.main(["list", "--cases", str(root), "--dataset", "multiplex", "--sizes"])
        == 0
    )
    assert capsys.readouterr().out.split() == ["4", "2", "3"]
    # a cases folder with no prepared case: three zeros
    assert (
        cli.main(
            [
                "list",
                "--cases",
                str(tmp_path / "none"),
                "--dataset",
                "multiplex",
                "--sizes",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.split() == ["0", "0", "0"]


# ── the selfcheck records each attempt's allocation ───────────────────────────
def test_selfcheck_json_keeps_the_allocation_of_every_attempt(tmp_path, monkeypatch):
    """Ruling R3.5: the selfcheck climbs 64 -> 256 -> 512 GB, and its record says what
    each attempt was given."""
    monkeypatch.setattr(selfcheck, "sizes_for", lambda name: [((64, 64), None)])
    monkeypatch.setattr(
        selfcheck,
        "run_pair",
        lambda *a, **k: {
            "status": "not_checked",
            "variants": {"x": {"status": "not_checked", "cause": "oom"}},
        },
    )
    for attempt, gb in ((1, 64), (2, 256)):
        monkeypatch.setenv("REGBENCH_ATTEMPT", str(attempt))
        monkeypatch.setenv("REGBENCH_MEM_GB", str(gb))
        rec = selfcheck.run("initial", tmp_path)
    assert [
        (a["attempt"], a["mem_requested_gb"], a["status"]) for a in rec["attempts"]
    ] == [
        (1, 64.0, "not_checked"),
        (2, 256.0, "not_checked"),
    ]
    on_disk = json.loads((tmp_path / "selfcheck.json").read_text())
    assert on_disk["attempts"] == rec["attempts"]
    assert variants.selfcheck_status("stare", on_disk)[0] == "not_checked"


def test_an_unreadable_realcheck_is_reported_unreadable_not_skipped(tmp_path):
    """Task 28, S14: a realcheck.json that cannot be read was skipped, so the run read
    ``not_checked`` (or ``ok`` beside a readable one) and the loss was nowhere."""
    root, out = tmp_path / "cases", tmp_path / "out"
    first, second = a_case(root, "P1_a", "P1"), a_case(root, "P1_b", "P1")
    for case in (first, second):
        write_json(
            result_dir(out, "valis_mirage", case) / "run.json",
            {"ok": True, "submissions": ["s1"]},
        )
    write_json(result_dir(out, "valis_mirage", first) / "realcheck.json", {"flag": False})
    (result_dir(out, "valis_mirage", second) / "realcheck.json").write_text("{trunc")
    _, _, lines = _verdict(out, "valis_mirage")
    line = _line(lines, "valis_mirage", "realcheck")
    assert line[2] == "unreadable"
    assert line[5] == "realcheck.json unreadable on P1_b"
    # a flag still wins the status, and the unreadable one is still said
    write_json(result_dir(out, "valis_mirage", first) / "realcheck.json", {"flag": True})
    _, _, lines = _verdict(out, "valis_mirage")
    line = _line(lines, "valis_mirage", "realcheck")
    assert line[2] == "flagged"
    assert line[5] == "flagged on P1_a; realcheck.json unreadable on P1_b"
