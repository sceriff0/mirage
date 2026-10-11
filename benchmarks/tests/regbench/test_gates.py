"""What the launcher's gate jobs write, and what the scorer makes of it.

A gate record is one tab-separated line a shell can read: run, status, rows to drop,
blocked, reference. A blocked run never starts; a dropped row is left out of the tables;
both are named in the summary.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.regbench import cli, score, selfcheck, variants
from benchmarks.regbench.cases import result_dir, write_json

from .test_ladder import a_case
from .test_pipeline import good, toy_case

HR = "deeperhistreg_hr_gpu"


def _record(tmp_path, **statuses):
    path = tmp_path / "selfcheck.json"
    path.write_text(json.dumps({"variants": statuses, "pairs": []}))
    return path


def _gate(tmp_path, name, *extra, **statuses):
    out = tmp_path / "gates" / f"{name}.tsv"
    code = cli.main(
        [
            "gate", "--name", name, "--selfcheck", str(_record(tmp_path, **statuses)),
            "--record", str(out), *extra,
        ]
    )  # fmt: skip
    return code, out.read_text().rstrip("\n").split("\t")


# ── the selfcheck gate ────────────────────────────────────────────────────────
def test_the_record_is_one_line_a_shell_can_read(tmp_path, capsys):
    code, line = _gate(tmp_path, "stare", stare_rigid="ok", stare="ok")
    assert (code, line) == (0, ["stare", "ok", "-", "0", "-"])
    assert all(line)  # no empty field: a tab-separated `read` merges them
    code, line = _gate(tmp_path, "stare", stare_rigid="ok", stare="inconsistent")
    assert (code, line) == (1, ["stare", "inconsistent", "-", "1", "-"])


def test_a_dropped_row_does_not_block_and_is_recorded(tmp_path):
    code, line = _gate(tmp_path, HR, **{HR: "inconsistent", f"{HR}_asdrawn": "ok"})
    assert (code, line) == (0, [HR, "ok", HR, "0", "-"])


def test_a_selfcheck_that_left_no_record_blocks_the_run(tmp_path, capsys):
    out = tmp_path / "g.tsv"
    argv = ["gate", "--name", "palom", "--selfcheck", str(tmp_path / "none.json")]
    assert cli.main([*argv, "--record", str(out)]) == selfcheck.EXIT_CODE["not_checked"]
    assert out.read_text().split("\t")[:4] == ["palom", "not_checked", "-", "1"]


def test_an_allowed_run_starts_and_its_verdict_is_still_recorded(tmp_path):
    code, line = _gate(tmp_path, "palom", "--allow-unchecked", palom_affine="crashed")
    assert code == 0 and line == ["palom", "crashed", "-", "0", "-"]


def test_a_run_behind_another_runs_selfcheck_is_recorded_under_its_own_name(tmp_path):
    out = tmp_path / "g.tsv"
    rec = _record(tmp_path, ashlar="inconsistent")
    argv = [
        "gate",
        "--name",
        "ashlar",
        "--for",
        "ashlar_clean",
        "--selfcheck",
        str(rec),
    ]
    assert cli.main([*argv, "--record", str(out)]) == 1
    assert out.read_text().split("\t")[:4] == ["ashlar_clean", "inconsistent", "-", "1"]
    with pytest.raises(SystemExit):
        cli.main(["gate", "--name", "ashlar", "--for", "nope", "--selfcheck", str(rec)])


def test_rows_are_dropped_from_the_run_whose_selfcheck_it_is_and_no_other(tmp_path):
    """`dropped` is asked of the gating run; a run standing behind it drops nothing."""
    rec = tmp_path / "s.json"
    rec.write_text(json.dumps({"variants": {HR: "crashed", f"{HR}_asdrawn": "ok"}}))
    out = tmp_path / "g.tsv"
    argv = ["gate", "--name", HR, "--selfcheck", str(rec), "--record", str(out)]
    assert cli.main(argv) == 0 and out.read_text().split("\t")[2] == HR
    assert cli.main([*argv, "--for", "stare"]) == 0
    assert out.read_text().split("\t")[:3] == ["stare", "ok", "-"]


def test_the_reference_comparison_is_recorded_beside_the_status(tmp_path):
    """ASHLAR redraws the reference: its second comparison is already in the row's
    status, and is written out so the summary can say which of the two failed."""
    rec = tmp_path / "s.json"
    pair = {"variants": {"ashlar": {"status": "inconsistent",
                                    "reference": {"status": "inconsistent"}}}}  # fmt: skip
    rec.write_text(
        json.dumps({"variants": {"ashlar": "inconsistent"}, "pairs": [pair]})
    )
    out = tmp_path / "g.tsv"
    argv = ["gate", "--name", "ashlar", "--selfcheck", str(rec), "--record", str(out)]
    assert cli.main(argv) == 1
    assert out.read_text().rstrip("\n").split("\t") == [
        "ashlar", "inconsistent", "-", "1", "inconsistent",
    ]  # fmt: skip


# ── the realcheck gate ────────────────────────────────────────────────────────
def _realgate(root, out, tmp_path, *extra):
    record = tmp_path / "valis_mirage.realcheck.tsv"
    code = cli.main(
        [
            "realgate", "--cases", str(root), "--out", str(out), "--dataset",
            "multiplex", "--name", "valis_mirage", "--method", "valis_mirage",
            "--variant", "valis_mirage", "--record", str(record), *extra,
        ]
    )  # fmt: skip
    return code, record.read_text().rstrip("\n").split("\t")


def test_the_realcheck_gate_reads_the_gating_rows_own_record(tmp_path):
    """Not the exit code of `run --realcheck`: that is 3 when ANY row is flagged."""
    root, out = tmp_path / "cases", tmp_path / "out"
    first = [a_case(root, "P1_a", "P1"), a_case(root, "P1_b", "P1")]
    a_case(root, "P2_a", "P2")
    code, line = _realgate(root, out, tmp_path)
    assert (code, line[:4]) == (0, ["valis_mirage", "realcheck_not_checked", "-", "0"])

    for case in first:
        write_json(
            result_dir(out, "valis_mirage", case) / "realcheck.json", {"flag": False}
        )
    # a realcheck beside no run, or beside a failed one, answers for nothing: it is
    # what an earlier launch left behind (review item 13)
    code, line = _realgate(root, out, tmp_path)
    assert (code, line[:4]) == (0, ["valis_mirage", "realcheck_not_checked", "-", "0"])
    for case in first:
        write_json(result_dir(out, "valis_mirage", case) / "run.json", {"ok": False})
    assert _realgate(root, out, tmp_path)[1][1] == "realcheck_not_checked"
    for case in first:
        write_json(
            result_dir(out, "valis_mirage", case) / "run.json",
            {"ok": True, "submissions": ["s1"]},
        )
    # a flag on a row that does not gate, and on a patient that did not go ahead
    write_json(
        result_dir(out, "valis_mirage_rigid", first[0]) / "realcheck.json",
        {"flag": True},
    )
    code, line = _realgate(root, out, tmp_path)
    assert (code, line[:4]) == (0, ["valis_mirage", "realcheck_ok", "-", "0"])

    write_json(
        result_dir(out, "valis_mirage", first[1]) / "realcheck.json", {"flag": True}
    )
    code, line = _realgate(root, out, tmp_path)
    assert (code, line[:4]) == (1, ["valis_mirage", "realcheck_flagged", "-", "1"])
    code, line = _realgate(root, out, tmp_path, "--allow-unchecked")
    assert (code, line[:4]) == (0, ["valis_mirage", "realcheck_flagged", "-", "0"])


def test_the_realcheck_gate_refuses_a_record_of_another_submission(
    tmp_path, monkeypatch
):
    root, out = tmp_path / "cases", tmp_path / "out"
    case = a_case(root, "P1_a", "P1")
    d = result_dir(out, "valis_mirage", case)
    write_json(d / "realcheck.json", {"flag": True})
    write_json(d / "run.json", {"ok": True, "submissions": ["old"]})
    monkeypatch.setenv("REGBENCH_SUBMISSION", "new")
    code, line = _realgate(root, out, tmp_path)
    assert (code, line[1]) == (0, "realcheck_not_checked")
    monkeypatch.setenv("REGBENCH_SUBMISSION", "old")
    assert _realgate(root, out, tmp_path)[1][1] == "realcheck_flagged"


def test_the_selfcheck_gate_refuses_a_record_of_another_submission(
    tmp_path, monkeypatch, capsys
):
    """Review item 13: REGBENCH_OUT is reused at the same commit; the new selfcheck
    died before it wrote anything and the old `ok` is still there."""
    path = _record(tmp_path, stare_rigid="ok", stare="ok")
    rec = json.loads(path.read_text())
    path.write_text(json.dumps({**rec, "submission": "old"}))
    record = tmp_path / "gate.tsv"
    argv = [
        "gate",
        "--name",
        "stare",
        "--selfcheck",
        str(path),
        "--record",
        str(record),
    ]
    monkeypatch.setenv("REGBENCH_SUBMISSION", "old")
    assert cli.main(argv) == 0 and record.read_text().split("\t")[1] == "ok"
    monkeypatch.setenv("REGBENCH_SUBMISSION", "new")
    assert cli.main(argv) != 0
    assert record.read_text().split("\t")[1:4:2] == ["not_checked", "1"]
    assert "of another submission" in capsys.readouterr().out


# ── what the scorer makes of the gates ────────────────────────────────────────
def test_the_summary_names_every_blocked_run_and_every_dropped_row(tmp_path):
    root, out = tmp_path / "cases", tmp_path / "out"
    from benchmarks.regbench import methods

    methods.run_case(good(), toy_case(root, "P1_r1", "P1"), out)
    gates = out / "logs" / "gates"
    gates.mkdir(parents=True)
    (gates / "palom.tsv").write_text("palom\tinconsistent\t-\t1\t-\n")
    (gates / f"{HR}.tsv").write_text(f"{HR}\tok\t{HR}\t0\t-\n")
    (gates / "stare.tsv").write_text("stare\tok\t-\t0\t-\n")
    (gates / "valis120.tsv").write_text("valis120\tcrashed\t-\t0\t-\n")  # allowed
    (gates / "valis_mirage.realcheck.tsv").write_text(
        "valis_mirage\trealcheck_flagged\t-\t1\t-\n"
    )
    got = score.score(root, out, ["multiplex"], with_tissue=False, gates=gates)
    by_run = {(g["run"], g["gate"]): g for g in got["launch"]}
    assert by_run[("palom", "selfcheck")]["blocked"] is True
    assert by_run[(HR, "selfcheck")]["dropped"] == HR
    assert by_run[("valis_mirage", "realcheck")]["status"] == "realcheck_flagged"
    text = (out / "tables" / "summary.md").read_text()
    block = text[text.index("## What the gates decided") :].split("\n## ")[0]
    assert "`palom` (selfcheck: inconsistent)" in block
    assert "`valis_mirage` (realcheck: realcheck_flagged)" in block
    assert f"`{HR}` (of `{HR}`)" in block
    assert "`valis120` (selfcheck: crashed)" in block.split("started although")[1]
    assert "`stare`" not in block  # a run whose gate said ok is not a finding
    assert (out / "tables" / "launch.csv").read_text().count("\n") == 6

    # through the command line, with the rows the gates dropped
    cli.main(
        ["score", "--cases", str(root), "--out", str(out), "--only-run", "--no-tissue",
         "--gates", str(gates), "--drop", HR]
    )  # fmt: skip
    text = (out / "tables" / "summary.md").read_text()
    assert "## What the gates decided" in text and "## Rows left out" in text
    # without gate records there is no such block and no stale table
    score.score(root, out, ["multiplex"], with_tissue=False)
    assert "What the gates decided" not in (out / "tables" / "summary.md").read_text()
    assert not (out / "tables" / "launch.csv").exists()


def test_no_blocked_run_is_said_in_so_many_words():
    launch = [dict(run="stare", gate="selfcheck", status="ok", blocked=False,
                   dropped="", reference="")]  # fmt: skip
    assert "No run was blocked." in "\n".join(score.launch_markdown(launch))
    assert score.launch_markdown([]) == []


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
    assert (
        cli.main(["list", "--cases", str(root), "--dataset", "synthetic", "--sizes"])
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
    assert variants.gate_status("stare", on_disk) == "not_checked"
