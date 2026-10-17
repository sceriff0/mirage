import json

from benchmarks.regbench import cli, methods, reuse
from benchmarks.tests.regbench import scored_cohort


def _rows(root, out):
    return {(r["variant"], r["case_id"]): r for r in reuse.report(root, out)}


def test_a_fresh_result_is_reusable_and_a_changed_key_is_named(tmp_path, monkeypatch):
    root, out, made = scored_cohort.build(tmp_path)
    rec_path = out / "stare" / "multiplex" / "P1_r1" / "run.json"
    rec = json.loads(rec_path.read_text())
    monkeypatch.setattr(methods, "reuse_key", lambda *a, **k: rec.get("reuse_key"))
    assert _rows(root, out)[("stare", "P1_r1")]["reusable"] == "yes"

    monkeypatch.setattr(
        methods, "reuse_key",
        lambda *a, **k: {**(rec.get("reuse_key") or {}), "code": "other"},
    )
    row = _rows(root, out)[("stare", "P1_r1")]
    assert row["reusable"] == "no" and "code" in row["differs"]


def test_a_stand_in_method_reports_key_not_computable(tmp_path):
    # stand-in records carry a reuse_key but methods.load() raises SystemExit
    root, out, made = scored_cohort.build(tmp_path)
    row = _rows(root, out)[("tiles", "P1_r1")]
    assert row["reusable"] == "no"
    assert row["detail"].startswith(reuse.KEY_NOT_COMPUTABLE)
    assert "code" in row["differs"]


def test_a_failed_result_is_never_reusable(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    rows = [r for r in reuse.report(root, out) if r["variant"] == "crash"]
    assert rows and all(r["reusable"] == "no" and "record" in r["differs"] for r in rows)


def test_re_prepared_points_are_named(tmp_path, monkeypatch):
    root, out, made = scored_cohort.build(tmp_path)
    path = out / "stare" / "multiplex" / "P1_r1" / "run.json"
    rec = json.loads(path.read_text())
    monkeypatch.setattr(methods, "reuse_key", lambda *a, **k: rec["reuse_key"])
    rec["points_sha256"] = {g: "0" * 64 for g in rec["points_sha256"]}
    path.write_text(json.dumps(rec))
    row = _rows(root, out)[("stare", "P1_r1")]
    assert row["reusable"] == "no" and "points" in row["differs"]


def test_the_report_writes_nothing_under_out(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    before = sorted(p for p in out.rglob("*"))
    reuse.report(root, out)
    assert sorted(p for p in out.rglob("*")) == before


def test_the_cli_prints_rows_and_the_summary_line(tmp_path, capsys):
    root, out, made = scored_cohort.build(tmp_path)
    csv_path = tmp_path / "reuse.csv"
    assert cli.main(["reuse", "--cases", str(root), "--out", str(out),
                     "--csv", str(csv_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t")[:2] == ["variant", "case_id"]
    assert lines[-1].startswith("# reusable: 0 of ")
    assert csv_path.read_text().startswith("variant,case_id")
