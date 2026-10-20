"""``regbench status`` (benchmarks/regbench/status.py) on a stand-in result tree:
read-only counts per run and variant against the case list, the queue when squeue is
there, and what to resubmit. Fleet wave 2, B2."""

from __future__ import annotations

import csv
import io
import json
import os

from benchmarks.regbench import chains, cli, ladder, status
from benchmarks.tests.regbench.test_ladder import a_case

SUB = "s2"


def _tree(tmp_path):
    """Two palom units: P1_a's palom ok (reused by s2) and palom_affine failed oom;
    P2_a's palom ok (run by s2), palom_affine missing."""
    root = tmp_path / "cases"
    c1, c2 = a_case(root, "P1_a", "P1"), a_case(root, "P2_a", "P2")
    out = tmp_path / "out"

    def put(case, variant, **rec):
        (path,) = [
            p for p in ladder.unit_paths(out, "palom", "", [case])
            if p.parents[2].name == variant
        ]  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
        return path

    put(c1, "palom", ok=True, submissions=["s1", SUB], reused_by=[{"submission": SUB}])
    put(c1, "palom_affine", ok=False, cause="oom", submissions=[SUB])
    put(c2, "palom", ok=True, submissions=[SUB])
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "submission").write_text(SUB + "\n")
    (out / "logs" / "jobs.tsv").write_text(
        "t\trb-palom-multiplex-16g\t50\nt\trb-palom-multiplex-retry\t51\n"
    )
    return root, out


def _snapshot(path):
    return {p: p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_status_counts_each_variant_against_the_case_list(tmp_path, monkeypatch):
    root, out = _tree(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")  # no squeue here
    before = _snapshot(out)
    text = status.report(out, root, runs=["palom"])
    assert _snapshot(out) == before  # read-only
    rows = {
        line.split()[1]: line.split()
        for line in text.splitlines()
        if line.startswith("palom ")
    }
    # run variant dataset expected ok reused failed missing other_sub stale [causes]
    assert rows["palom"][3:10] == ["2", "2", "1", "0", "0", "0", "0"]
    assert rows["palom_affine"][3:11] == ["2", "0", "0", "1", "1", "0", "0", "oom:1"]
    assert "# 4 result(s) expected: 2 ok (1 reused), 1 failed, 1 missing" in text
    assert "# queue: not read (no squeue on this node)" in text
    assert (
        f"# next: 2 unit(s) of 1 run(s) (palom) are not all ok: submit the FULL run "
        f"list again into the same folder, which reuses every ok unit: "
        f"REGBENCH_OUT={out} submit_cohort.sh\n" in text
    )
    # a --runs subset into this folder would make every other run `not run` in the
    # tables (the scorer loads only what the last submission ran or reused)
    assert "a --runs subset needs its own REGBENCH_OUT" in text
    assert "--runs" not in text.split("# next:")[1].split("\n")[0].split("its own")[0]
    got = list(csv.DictReader(io.StringIO(status.report(out, root, as_csv=True,
                                                        runs=["palom"]))))  # fmt: skip
    got = {r["variant"]: r for r in got}
    assert set(got) == {"palom", "palom_affine"}
    assert got["palom_affine"]["causes"] == "oom:1"
    assert got["palom_affine"]["missing"] == "1"


def test_status_says_to_wait_while_jobs_of_the_folder_are_queued(tmp_path, monkeypatch):
    root, out = _tree(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text(
        "#!/bin/sh\necho '51|rb-palom-multiplex-retry|RUNNING|51|51'\n"
        "echo '99|someone-else|PENDING|99|99'\n"
    )
    (bin_ / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    text = status.report(out, root, runs=["palom"])
    assert "# queue: 1 job(s) of this folder still queued" in text
    assert "#   51 rb-palom-multiplex-retry RUNNING" in text
    assert "# next: wait; 1 job(s) are still queued" in text
    # a chain whose watcher link is still queued is not "unfinished" (S2) ...
    plan = chains.retry_root(out, SUB) / "rb-palom-multiplex"
    plan.mkdir(parents=True)
    assert "chain not finished" not in status.report(out, root, runs=["palom"])
    # ... once no link of it is queued, it is named
    (bin_ / "squeue").write_text("#!/bin/sh\necho '99|someone-else|PENDING|99|99'\n")
    assert "chain not finished by any link of its watcher: rb-palom-multiplex" in (
        status.report(out, root, runs=["palom"])
    )


def test_status_through_the_cli_and_into_the_summary(tmp_path, monkeypatch, capsys):
    root, out = _tree(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")
    assert cli.main(["status", "--out", str(out), "--cases", str(root), "--runs",
                     "palom", "--csv"]) == 0  # fmt: skip
    assert capsys.readouterr().out.splitlines()[0] == ",".join(status.COLUMNS)
    assert cli.main(["status", "--out", str(out), "--cases", str(tmp_path / "x")]) == 2
    summary = tmp_path / "summary.md"
    summary.write_text("# tables\n")
    assert cli.main(["status", "--out", str(out), "--summary-to", str(summary)]) == 0
    assert summary.read_text() == "# tables\n"  # every chain finished: nothing added
    (chains.retry_root(out, SUB) / "rb-palom-multiplex").mkdir(parents=True)
    assert cli.main(["status", "--out", str(out), "--summary-to", str(summary)]) == 0
    assert "## Chains no watcher finished" in summary.read_text()
    assert "- `rb-palom-multiplex`: before its first retry step" in summary.read_text()


def test_status_counts_ok_only_what_the_scorer_would_load(tmp_path, monkeypatch):
    """S1: an ok record of ANOTHER submission, or one made for points the case no longer
    has, is not scored for this submission: it is not `ok` here either."""
    root, out = _tree(tmp_path)
    c3 = a_case(root, "P3_a", "P3")
    c4 = a_case(root, "P4_a", "P4")
    for case, rec in (
        (c3, {"ok": True, "submissions": ["s1"]}),
        (c4, {"ok": True, "submissions": [SUB], "points_sha256": {"cells": "0" * 64}}),
    ):
        for path in ladder.unit_paths(out, "palom", "", [case]):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))
    monkeypatch.setenv("PATH", "/nonexistent")
    got = {r["variant"]: r for r in status.tally(out, root, submission=SUB,
                                                  runs=["palom"])[0]}  # fmt: skip
    assert (got["palom"]["ok"], got["palom"]["other_sub"], got["palom"]["stale"]) == (
        2, 1, 1
    )
    _, todo = status.tally(out, root, submission=SUB, runs=["palom"])
    assert len(todo["palom"]) == 4  # P1 (oom), P2 (missing), P3, P4


def test_status_finds_queued_jobs_by_their_ids_not_their_names(tmp_path, monkeypatch):
    """T3: a job of ANOTHER folder with the same name is not this folder's; an array
    task is matched by its array job id (%F)."""
    root, out = _tree(tmp_path)
    (out / "logs" / "jobs.tsv").write_text(
        "t\trb-palom-multiplex-16g\t50_0\nt\trb-palom-multiplex-retry\t51\n"
    )
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text(
        "#!/bin/sh\n"
        "echo '50_3|rb-palom-multiplex-16g|RUNNING|50|53'\n"
        "echo '77|rb-palom-multiplex-retry|PENDING|77|77'\n"
    )
    (bin_ / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    jobs, note = status.queued(out)
    assert note == "" and jobs == [("50_3", "rb-palom-multiplex-16g", "RUNNING")]


def test_status_of_another_dataset_reads_its_own_runs_and_cases(tmp_path, monkeypatch):
    """2026-10-20 on the cluster: `status --dataset synthetic` walked the COHORT's run list
    (stare_clahe listed, stare_medium_s128_none never counted) against REGBENCH_CASES (the
    cohort's folder), so every row said expected 0. A dataset is tallied against its own
    run list (variants.runs_for) and, without --cases, its own folder
    (REGBENCH_CASES_<NAME>, as the launcher's regbench_cases_of)."""
    from benchmarks.regbench import variants

    empty = tmp_path / "cases"
    empty.mkdir()
    rows, _ = status.tally(tmp_path / "out", empty, dataset="synthetic")
    want = {variants.run_name(r) for r in variants.runs_for("synthetic")}
    assert {r["run"] for r in rows} == want
    assert "stare_medium_s128_none" in want and "stare_clahe" not in want
    rows, _ = status.tally(tmp_path / "out", empty)  # the cohort: unchanged
    assert {r["run"] for r in rows} == {variants.run_name(r) for r in variants.runs_for(None)}

    cohort, synth = tmp_path / "c", tmp_path / "s"
    cohort.mkdir()
    synth.mkdir()
    monkeypatch.setenv("REGBENCH_CASES", str(cohort))
    monkeypatch.setenv("REGBENCH_CASES_SYNTHETIC", str(synth))
    monkeypatch.setenv("PATH", "/nonexistent")
    seen = []
    monkeypatch.setattr(status, "report", lambda out, root, *a, **k: seen.append(root) or "")
    assert cli.main(["status", "--out", str(tmp_path / "out"), "--dataset", "synthetic"]) == 0
    assert cli.main(["status", "--out", str(tmp_path / "out")]) == 0
    assert seen == [str(synth), str(cohort)]
