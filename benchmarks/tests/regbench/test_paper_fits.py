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
    # pp is the family's primary: the common set (P1, P3) is over it (T31 fix I4)
    monkeypatch.setattr(tables, "primary_of", lambda fam, dataset=None: "pp")
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


def test_every_patient_is_a_point_and_says_what_became_of_it(cohort):
    # T31 rule 6: an incomplete patient is a point with its status, never dropped
    # silently; only a complete one enters a fit
    out, cases = cohort
    pts = fits.points(out, "multiplex", cases)
    status = {p["unit_id"]: p["status"] for p in pts
              if p["row"] == "pp" and p["unit_kind"] == "patient"}
    assert status == {"P1": "complete", "P2": "failed", "P3": "complete"}
    # M2: a per-patient launch that did not time its rounds has them as points of
    # status not_split (no figure), so fig. 2c counts them; the failed round is none
    rounds = {p["unit_id"]: p["status"] for p in pts if p["row"] == "pp" and p["unit_kind"] == "round"}
    assert rounds == {c: "not_split" for c in ("P1_r1", "P1_r2", "P2_r1", "P3_r1", "P3_r2")}
    fitted = {(r["target"], r["predictor"]): r for r in fits.method_fits(pts) if r["row"] == "pp"}
    assert all(r["n"] + r["n_dropped"] == 2 for r in fitted.values())  # P1, P3


def test_a_failed_round_is_not_a_round_point(cohort):
    out, cases = cohort
    pts = fits.points(out, "multiplex", cases)
    assert {p["unit_id"] for p in pts if p["row"] == "pair"} == {
        "P1_r1", "P1_r2", "P2_r1", "P3_r1", "P3_r2"}  # P2_r2 failed: not ok
    assert all(p["unit_kind"] == "round" for p in pts if p["row"] == "pair")


def test_a_point_carries_the_bytes_of_the_files_it_read(cohort):
    out, cases = cohort
    pts = {(p["row"], p["unit_id"]): p for p in fits.points(out, "multiplex", cases)}
    assert pts[("pair", "P1_r1")]["input_gb"] == pytest.approx(2 * 2**20 / 2**30)
    assert pts[("pp", "P1")]["input_gb"] == pytest.approx(4 * 2**20 / 2**30)  # both rounds
    assert pts[("pp", "P1")]["input_gpx"] == 1.0 and pts[("pp", "P1")]["wall_s"] == 200
    no_cases = fits.points(out, "multiplex", None)
    assert all(p["input_gb"] is None for p in no_cases)


def test_method_fits_follow_the_schema(cohort):
    # T31 P10/P13: per patient for every method (a pair row without patient sums has
    # no fit), log-log, and no fit below MIN_FIT_POINTS patients
    out, cases = cohort
    points = fits.points(out, "multiplex", cases)
    rows = fits.method_fits(points)
    assert len(rows) == len(fits.METHOD_TARGETS) * len(fits.METHOD_PREDICTORS)
    assert {r["row"] for r in rows} == {"pp"}
    assert all(list(r) == schema.columns("cost_fits") for r in rows)
    # T31 fix I4: fitted over the COMMON patients (rule 2), as every cost figure draws
    assert all(r["unit"] == "patient" and r["patient_set"] == "common" for r in rows)
    assert all(r["fit_ok"] is False and r["n"] == 2 for r in rows)  # 2 < 5 patients


def test_a_fit_on_five_patients_has_a_slope_interval():
    r = fits.fit([1, 2, 4, 8, 16], [2.1, 3.9, 8.2, 15.8, 32.5], log=True, min_n=5)
    assert r["fit_ok"] and r["scale"] == "log-log" and r["slope"] == pytest.approx(1.0, abs=0.05)
    assert r["slope_ci_low"] < r["slope"] < r["slope_ci_high"]
    assert fits.fit([1, 2, 4, 8], [2, 4, 8, 16], log=True, min_n=5)["fit_ok"] is False
