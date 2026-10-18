# benchmarks/tests/regbench/test_paper_fits.py
import csv
import json
from dataclasses import asdict

import pytest

from benchmarks.regbench.cases import Case
from benchmarks.regbench.paper import fits, schema, tables

# Three tasks of TILED_STITCH at 1, 2 and 3 GiB of input, in the pipeline's size-log layout
# (lib/ProcessEnvelope.groovy SIZE_LOG_COLUMNS = process, sample_id, filename, bytes;
# bin/generate_resource_report.py parse_size_log refuses any other header).
SIZE_LOG_FIXTURE = (
    "process,sample_id,filename,bytes\n"
    f"TILED_STITCH,P1_a,a.ome.tif,{1 * 2**30}\n"
    f"TILED_STITCH,P1_b,b.ome.tif,{2 * 2**30}\n"
    f"TILED_STITCH,P1_c,c.ome.tif,{3 * 2**30}\n"
)


def test_a_line_is_recovered():
    r = fits.fit([1, 2, 3, 4], [3, 5, 7, 9])
    assert r["fit_ok"] and abs(r["slope"] - 2) < 1e-12 and abs(r["intercept"] - 1) < 1e-12
    assert abs(r["r2"] - 1) < 1e-12 and r["n"] == 4


def test_too_few_or_one_size_is_not_a_fit():
    assert fits.fit([1, 2], [3, 4])["fit_ok"] is False
    r = fits.fit([5, 5, 5], [1, 2, 3])
    assert r["fit_ok"] is False and r["slope"] is None and r["n_distinct_x"] == 1


def test_pipeline_fits_join_size_logs(tmp_path):
    out = tmp_path / "out"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "pipeline_cost.csv").write_text(
        "backend,patient,process,tag,realtime_s,peak_rss_gb\n"
        "tiled,P1,TILED_STITCH,P1_a,10,2.0\ntiled,P1,TILED_STITCH,P1_b,20,2.0\n"
        "tiled,P1,TILED_STITCH,P1_c,30,2.0\n")
    logs = out / "compute" / "tiled" / "P1" / "results" / "size_logs"
    logs.mkdir(parents=True)
    (logs / "input_sizes.csv").write_text(SIZE_LOG_FIXTURE)
    rows = {(r["process"], r["target"]): r for r in fits.pipeline_fits(out)}
    rt = rows[("TILED_STITCH", "realtime_s")]
    assert rt["n"] == 3 and rt["fit_ok"] and abs(rt["slope"] - 10.0) < 1e-9
    assert rows[("TILED_STITCH", "peak_rss_gb")]["slope"] == 0.0


def test_pipeline_fits_count_the_tasks_they_drop(tmp_path):
    """Task 28, S12: a completed task with no input size or no finite target was left
    out of its fit and counted nowhere; ``n_dropped`` says how many."""
    out = tmp_path / "out"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "pipeline_cost.csv").write_text(
        "backend,patient,process,tag,realtime_s,peak_rss_gb\n"
        "tiled,P1,TILED_STITCH,P1_a,10,2.0\ntiled,P1,TILED_STITCH,P1_b,20,2.0\n"
        "tiled,P1,TILED_STITCH,P1_c,,2.0\ntiled,P1,TILED_STITCH,P1_nosize,40,2.0\n")
    logs = out / "compute" / "tiled" / "P1" / "results" / "size_logs"
    logs.mkdir(parents=True)
    (logs / "input_sizes.csv").write_text(SIZE_LOG_FIXTURE)
    rows = {(r["process"], r["target"]): r for r in fits.pipeline_fits(out)}
    assert rows[("TILED_STITCH", "realtime_s")]["n"] == 2
    assert rows[("TILED_STITCH", "realtime_s")]["n_dropped"] == 2  # no size + blank
    assert rows[("TILED_STITCH", "peak_rss_gb")]["n_dropped"] == 1  # no size
    assert all(list(r) == schema.columns("pipeline_cost_fits") for r in rows.values())


# ── points ───────────────────────────────────────────────────────────────────
def _write(path, rows):
    cols = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """A per-patient row ``pp`` (P1 complete, P2 with a failed round, P3 complete with an
    excluded round) and a pair row ``pair`` (a round per case), over cases of 1 GiB
    reference + 1 GiB moving (sparse files)."""
    monkeypatch.setattr(tables, "family_of", lambda v: "fam")
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {"pp": "pp", "pair": "pair"})
    out, cases = tmp_path / "out", tmp_path / "cases"
    (out / "tables").mkdir(parents=True)
    ids = {"P1": ["P1_r1", "P1_r2"], "P2": ["P2_r1", "P2_r2"], "P3": ["P3_r1", "P3_r2"]}
    for g, cs in ids.items():
        for c in cs:
            d = cases / "multiplex" / c
            d.mkdir(parents=True)
            ref, mov = d / "ref.tif", d / "mov.tif"
            for f in (ref, mov):
                with open(f, "wb") as fh:
                    fh.truncate(2**20)
            case = Case("multiplex", c, g, "", "", str(ref), str(mov), "fluorescence", 1.0)
            (d / "case.json").write_text(json.dumps({k: v for k, v in asdict(case).items() if k != "dir"}))
    res, summ = [], []
    for g, cs in ids.items():
        for i, c in enumerate(cs):
            bad = g == "P2" and i == 1
            res.append(dict(dataset="multiplex", group=g, method="pp", case_id=c, unit="patient",
                            ok=str(not bad), register_s=100 * (1 + i), peak_rss_gb=5, cpu_h_used=1,
                            moving_gpx=0.5, device="cpu", n_jobs=""))
            res.append(dict(dataset="multiplex", group=g, method="pair", case_id=c, unit="pair",
                            ok=str(not bad), register_s=10 * (1 + i), peak_rss_gb=2, cpu_h_used=0.1,
                            moving_gpx=0.5, device="cpu", n_jobs=""))
        summ.append(dict(dataset="multiplex", group=g, method="pp", unit="patient",
                         complete=str(g != "P2"), register_s=200, peak_rss_gb=5, cpu_h_used=2,
                         moving_gpx=1.0, device="cpu", n_jobs=""))
    _write(out / "tables" / "resources.csv", res)
    _write(out / "tables" / "summary_resources.csv", summ)
    return out, cases


def test_an_incomplete_patient_is_not_a_point(cohort):
    out, cases = cohort
    pts = fits.points(out, "multiplex", cases, excluded=frozenset())
    patients = {p["unit_id"] for p in pts if p["row"] == "pp"}
    assert patients == {"P1", "P3"}  # P2 registered one round of two: not its cost
    assert all(p["unit_kind"] == "patient" for p in pts if p["row"] == "pp")


def test_an_excluded_round_is_not_a_point_and_makes_its_patient_one_less(cohort):
    out, cases = cohort
    pts = fits.points(out, "multiplex", cases, excluded={("P3_r1", "pp"), ("P3_r1", "pair")})
    assert {p["unit_id"] for p in pts if p["row"] == "pp"} == {"P1"}
    assert {p["unit_id"] for p in pts if p["row"] == "pair"} == {
        "P1_r1", "P1_r2", "P2_r1", "P3_r2"}  # P2_r2 failed: not ok


def test_a_point_carries_the_bytes_of_the_files_it_read(cohort):
    out, cases = cohort
    pts = {(p["row"], p["unit_id"]): p for p in fits.points(out, "multiplex", cases, excluded=frozenset())}
    assert pts[("pair", "P1_r1")]["input_gb"] == pytest.approx(2 * 2**20 / 2**30)
    assert pts[("pp", "P1")]["input_gb"] == pytest.approx(4 * 2**20 / 2**30)  # both rounds
    assert pts[("pp", "P1")]["input_gpx"] == 1.0 and pts[("pp", "P1")]["wall_s"] == 200
    no_cases = fits.points(out, "multiplex", None, excluded=frozenset())
    assert all(p["input_gb"] is None for p in no_cases)


def test_method_fits_follow_the_schema(cohort):
    out, cases = cohort
    rows = fits.method_fits(fits.points(out, "multiplex", cases, excluded=frozenset()))
    assert len(rows) == 2 * 2 * 2
    assert all(list(r) == schema.columns("cost_fits") for r in rows)
    pair = {(r["target"], r["predictor"]): r for r in rows if r["row"] == "pair"}
    assert pair[("wall_s", "input_gb")]["n"] == 5 and pair[("wall_s", "input_gb")]["fit_ok"] is False
    assert pair[("peak_rss_gb", "input_gpx")]["fit_ok"] is False  # one size
    # every point of the row is counted: used or dropped (S12)
    points = fits.points(out, "multiplex", cases, excluded=frozenset())
    n_pair = sum(p["row"] == "pair" for p in points)
    assert all(r["n"] + r["n_dropped"] == n_pair for r in rows if r["row"] == "pair")
