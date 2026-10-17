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
    # run variant dataset expected ok reused failed missing [causes]
    assert rows["palom"][3:8] == ["2", "2", "1", "0", "0"]
    assert rows["palom_affine"][3:9] == ["2", "0", "0", "1", "1", "oom:1"]
    assert "# 4 result(s) expected: 2 ok (1 reused), 1 failed, 1 missing" in text
    assert "# queue: not read (no squeue on this node)" in text
    assert (
        f"# next: 2 unit(s) of 1 run(s) are not all ok: submit again into the same "
        f"folder, which reuses every ok unit: REGBENCH_OUT={out} submit_cohort.sh "
        '--runs "palom"' in text
    )
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
        "#!/bin/sh\necho '51|rb-palom-multiplex-retry|RUNNING'\n"
        "echo '99|someone-else|PENDING'\n"
    )
    (bin_ / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    text = status.report(out, root, runs=["palom"])
    assert "# queue: 1 job(s) of this folder still queued" in text
    assert "#   51 rb-palom-multiplex-retry RUNNING" in text
    assert "# next: wait; 1 job(s) are still queued" in text
    # a chain no link finished is named
    plan = chains.retry_root(out, SUB) / "rb-palom-multiplex"
    plan.mkdir(parents=True)
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
