"""CPU-hours and memory-hours by one rule for every method (``regbench.billing``).

Every expected number is worked out by hand from the attempts written in the test: CPUs
x seconds / 3600, GB x seconds / 3600, CPU seconds / 3600.
"""

import json
import math
import types

import pytest

from benchmarks.regbench import billing, score, stages

H = 3600.0


def single(**kw):
    """A run that registered in one process at the second rung, after a first rung
    that ran out of memory: 4 CPUs, 64 GB, 1800 s, 3600 CPU seconds."""
    run = {
        "ok": True,
        "cause": "",
        "method": "palom",
        "unit": "pair",
        "launch_id": "20_3",
        "attempt": 2,
        "cpus_requested": 4,
        "mem_requested_gb": 64.0,
        "register_s": 1800.0,
        "write_s": 600.0,
        "resources": {"wall_s": 1800.0, "cpu_s": 3600.0, "peak_rss_mb": 2048.0},
        "previous_attempts": [
            {
                "attempt": 1,
                "cause": "oom",
                "mem_requested_gb": 16.0,
                "launch_id": "10_3",
                "wall_s": 900.0,
                "cpu_s": 1200.0,
            }
        ],
        "info": {},
    }
    run.update(kw)
    return run


def test_a_single_process_row_with_a_failed_rung():
    row = score.resource_row(single(), moving_px=2_000_000_000)
    assert row["cpu_h_used"] == pytest.approx(3600 / H)  # 1.0
    assert row["cpu_h_reserved"] == pytest.approx(4 * 1800 / H)  # 2.0
    assert row["cpu_efficiency"] == pytest.approx(0.5)
    assert row["gb_h_reserved"] == pytest.approx(64 * 1800 / H)  # 32
    # the first rung: the same 4 CPUs, its own 16 GB, 900 s, 1200 CPU seconds
    assert row["wasted_cpu_h_used"] == pytest.approx(1200 / H)
    assert row["wasted_cpu_h_reserved"] == pytest.approx(4 * 900 / H)  # 1.0
    assert row["wasted_gb_h_reserved"] == pytest.approx(16 * 900 / H)  # 4
    assert row["total_cpu_h_used"] == pytest.approx(4800 / H)
    assert row["total_cpu_h_reserved"] == pytest.approx(3.0)
    assert row["total_gb_h_reserved"] == pytest.approx(36.0)
    # the writer is apart: the job's CPUs for its 600 s; its CPU time is not measured
    assert row["write_cpu_h_reserved"] == pytest.approx(4 * 600 / H)
    assert row["write_cpu_h_used"] == "" and row["write_wasted_cpu_h_reserved"] == ""
    assert row["moving_gpx"] == 2.0 and row["cpu_h_used_per_gpx"] == pytest.approx(0.5)
    assert (row["device"], row["device_name"], row["gpu_h"]) == ("cpu", "cpu", 0.0)
    # the names the tables had before stand for the same numbers
    assert row["cpu_h"] == row["cpu_h_used"]
    assert row["wasted_cpu_h"] == row["wasted_cpu_h_used"]
    # no accounting: the cross-check is blank, never zero
    assert row["cpu_h_scheduler"] == "" and row["cpu_h_used_vs_scheduler"] == ""
    assert set(billing.COLUMNS) <= set(row)


def test_the_scheduler_figure_is_a_cross_check_and_blank_without_one():
    booked = {"20_3": {"cpu_s": 5400.0}, "10_3": {"cpu_s": 999.0}}
    row = score.resource_row(single(), accounting=booked)
    # the job that registered, not the rung that failed: 5400 CPU s for the whole job
    assert row["cpu_h_scheduler"] == pytest.approx(1.5) and row["scheduler_jobs"] == 1
    assert row["cpu_h_used_vs_scheduler"] == pytest.approx(1.0 / 1.5)
    for nothing in ({}, {"20_3": {"cpu_s": None}}, {"20_3": {"cpu_s": 0.0}}):
        row = score.resource_row(single(), accounting=nothing)
        assert row["cpu_h_scheduler"] == "" and row["cpu_h_used_vs_scheduler"] == ""
    # a row that registered nothing has no good job to ask about
    lost = score.resource_row(single(ok=False, cause="oom"), accounting=booked)
    assert lost["cpu_h_scheduler"] == ""


def test_a_gpu_row_says_so_beside_its_cpu_hours():
    run = single(
        info={"device": "cuda:0"}, gpus=1, gpu_model="NVIDIA H200", method="dhr"
    )
    row = score.resource_row(run)
    assert (row["device"], row["device_name"]) == ("gpu", "cuda:0")
    assert row["gpu_h"] == pytest.approx(1800 / H) and row["gpu_model"] == "NVIDIA H200"
    assert row["cpu_h_used"] == pytest.approx(1.0)  # the same rule as a CPU row
    assert row["cpu_h_reserved"] == pytest.approx(2.0)
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}
    gpu = {**row, **base, "method": "dhr_gpu"}
    cpu = {**score.resource_row(single()), **base, "method": "palom"}
    per = {r["method"]: r for r in score.summarise_resources([gpu, cpu])}
    assert (per["dhr_gpu"]["device"], per["palom"]["device"]) == ("gpu", "cpu")
    summary = {r["method"]: r for r in score.summarise_cost(list(per.values()))}
    assert summary["dhr_gpu"]["device"] == "gpu" and summary["palom"]["device"] == "cpu"
    assert summary["dhr_gpu"]["gpu_model"] == "NVIDIA H200"
    text = score.summary_markdown(
        [], [], list(per.values()), cost_summary=list(summary.values())
    )
    head = text[text.index("### Cost per method") :]
    names = [c.strip() for c in head.splitlines()[4].strip("|").split("|")]
    # the device stands right after the GPU-hours, before any time
    assert names[names.index("gpu_h: complete / common") + 1] == "device"
    (line,) = [ln for ln in head.splitlines() if "| dhr_gpu |" in ln]
    assert (
        dict(zip(names, (c.strip() for c in line.strip("|").split("|"))))["device"]
        == "gpu"
    )
    assert "a `gpu` row's CPU-hours or time against a `cpu` row's" in head


def test_a_row_that_failed_is_all_wasted():
    run = single(
        ok=False,
        cause="exception",
        attempt=1,
        mem_requested_gb=16.0,
        register_s=600.0,
        resources={"wall_s": 600.0, "cpu_s": 900.0},
        previous_attempts=[],
    )
    row = score.resource_row(run)
    assert math.isnan(row["cpu_h_used"]) and math.isnan(row["cpu_h_reserved"])
    assert row["wasted_cpu_h_used"] == pytest.approx(900 / H)
    assert row["wasted_cpu_h_reserved"] == pytest.approx(4 * 600 / H)
    assert row["wasted_gb_h_reserved"] == pytest.approx(16 * 600 / H)
    assert row["total_cpu_h_reserved"] == row["wasted_cpu_h_reserved"]
    assert row["write_cpu_h_reserved"] == ""
    # a result nobody ran has no attempt: nothing good, nothing wasted, no total
    none = score.resource_row({"cause": "not_run"})
    assert math.isnan(none["cpu_h_used"]) and none["wasted_cpu_h_reserved"] == 0.0
    assert math.isnan(none["total_cpu_h_reserved"])
    # a writer that failed after a good registration: the registration is good
    kept = score.resource_row({**run, "registered": True, "cause": "writer"})
    assert kept["cpu_h_used"] == pytest.approx(900 / H)
    assert kept["wasted_cpu_h_reserved"] == 0.0
    # an attempt nobody could bill is in no wasted sum (it is counted as unknown)
    unknown = score.resource_row(
        single(previous_attempts=[{"attempt": 1, "mem_requested_gb": 16.0}])
    )
    assert unknown["wasted_unknown"] == 1 and unknown["wasted_cpu_h_reserved"] == 0.0


def test_per_patient_sums_for_both_kinds_of_unit():
    base = {"dataset": "multiplex", "group": "P1"}

    def row(case_id, method, run, px):
        return {
            **score.resource_row(run, moving_px=px),
            **base,
            "case_id": case_id,
            "method": method,
        }

    # a pair method: each round its own launch; the patient is the sum
    r1 = row("P1_r1", "pair", single(), 2_000_000_000)
    r2 = row(
        "P1_r2",
        "pair",
        single(
            launch_id="20_4",
            register_s=3600.0,
            resources={"wall_s": 3600.0, "cpu_s": 3600.0},
            previous_attempts=[],
            mem_requested_gb=16.0,
        ),
        1_000_000_000,
    )
    booked = {"20_3": {"cpu_s": 5400.0}, "20_4": {"cpu_s": 7200.0}}
    cases = [
        types.SimpleNamespace(dataset="multiplex", case_id=c, mov_hw=None)
        for c in ("P1_r1", "P1_r2")
    ]
    score.bill_from_disk([r1, r2], "/nowhere", cases, booked)
    (pair,) = score.summarise_resources([r1, r2])
    assert pair["cpu_h_used"] == pytest.approx(2.0)  # 1 + 1
    assert pair["cpu_h_reserved"] == pytest.approx(2.0 + 4.0)
    assert pair["cpu_efficiency"] == pytest.approx(2.0 / 6.0)
    assert pair["gb_h_reserved"] == pytest.approx(32.0 + 16.0)
    assert pair["wasted_cpu_h_reserved"] == pytest.approx(1.0)
    assert pair["total_cpu_h_reserved"] == pytest.approx(7.0)
    assert pair["total_gb_h_reserved"] == pytest.approx(48.0 + 4.0)
    assert pair["moving_gpx"] == pytest.approx(3.0)
    assert pair["cpu_h_used_per_gpx"] == pytest.approx(2.0 / 3.0)
    assert pair["cpu_h_scheduler"] == pytest.approx(1.5 + 2.0)
    assert pair["cpu_h_used_vs_scheduler"] == pytest.approx(2.0 / 3.5)
    assert pair["write_cpu_h_reserved"] == pytest.approx(2 * 4 * 600 / H)

    # a per-patient method: ONE launch, one clock, every row holds it up to its yield
    def launch(register_s, cpu_s, **kw):
        return single(
            unit="patient",
            method="group",
            launch_id="30_0",
            register_s=register_s,
            resources={"wall_s": register_s, "cpu_s": cpu_s},
            **kw,
        )

    g1 = row("P1_r1", "group", launch(1000.0, 2000.0), 2_000_000_000)
    g2 = row("P1_r2", "group", launch(3600.0, 7200.0), 1_000_000_000)
    g3 = row(
        "P1_r3",
        "group",
        launch(2000.0, 100.0, ok=False, cause="exception"),
        1_000_000_000,
    )
    assert g1["cpu_h_used_per_gpx"] != g1["cpu_h_used_per_gpx"]  # not per round
    (group,) = score.summarise_resources([g1, g2, g3])
    assert group["cpu_h_used"] == pytest.approx(7200 / H)  # the launch, not a sum
    assert group["cpu_h_reserved"] == pytest.approx(4 * 3600 / H)
    assert group["gb_h_reserved"] == pytest.approx(64 * 3600 / H)
    # the failed rung before it, once; the failed round inside the launch is no attempt
    assert group["wasted_cpu_h_reserved"] == pytest.approx(1.0)
    assert group["wasted_cpu_h_used"] == pytest.approx(1200 / H)
    assert group["total_cpu_h_reserved"] == pytest.approx(5.0)
    assert group["moving_gpx"] == pytest.approx(3.0)  # the rounds that registered
    assert group["cpu_h_used_per_gpx"] == pytest.approx(2.0 / 3.0)
    # a patient launch that registered nothing: its one failed attempt, once
    f = [
        row(f"P1_r{k}", "group", launch(500.0, 250.0, ok=False, cause="oom"), None)
        for k in (1, 2)
    ]
    (dead,) = score.summarise_resources(f)
    assert math.isnan(dead["cpu_h_used"])
    assert dead["wasted_cpu_h_reserved"] == pytest.approx(1.0 + 4 * 500 / H)
    assert dead["total_cpu_h_used"] == pytest.approx((1200 + 250) / H)


# ── a registration in several scheduler jobs ──────────────────────────────────────
def attempt_record(stage, attempt, ok, cpus, mem_gb, wall_s, cpu_s, job, row=None):
    return {
        "stage": stage,
        "row": row,
        "attempt": attempt,
        "ok": ok,
        "exit": 0 if ok else 137,
        "cause": "" if ok else "oom",
        "started": 1000.0,
        "ended": 1000.0 + wall_s,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "peak_rss_mb": 500.0,
        "cpus": cpus,
        "mem_gb": mem_gb,
        "launch_id": job,
        "history": [],
    }


def staged_round(folder):
    """COARSE (2 CPUs, 2 GB, 60 s), tile 0 retried (2 CPUs: 4 GB for 20 s, then 8 GB
    for 50 s), tile 1 (2 CPUs, 4 GB, 40 s), SOLVE (1 CPU, 1 GB, 10 s); and STITCH."""
    coarse = attempt_record("coarse", 1, True, 2, 2, 60.0, 90.0, "c1")
    first = attempt_record("tile", 1, False, 2, 4, 20.0, 30.0, "t0a", 0)
    tile0 = attempt_record("tile", 2, True, 2, 8, 50.0, 70.0, "t0b", 0)
    tile0["history"] = [stages._summary(first)]
    tile1 = attempt_record("tile", 1, True, 2, 4, 40.0, 50.0, "t1", 1)
    solve = attempt_record("solve", 1, True, 1, 1, 10.0, 8.0, "s1")
    stitch = attempt_record("stitch", 1, True, 4, 4, 30.0, 45.0, "20_3")
    jobs = folder / "stages" / "jobs"
    jobs.mkdir(parents=True)
    for name, rec in (
        ("coarse", coarse),
        ("tile_000000", tile0),
        ("tile_000001", tile1),
        ("solve", solve),
        ("stitch", stitch),
    ):
        (jobs / f"{name}.json").write_text(json.dumps(rec))
    spent = {**stages.summarise(coarse, [tile0, tile1], solve), "execution": "jobs"}
    return single(
        method="stare",
        register_s=spent["wall_unlimited_parallel_s"],
        resources={"wall_s": 120.0, "cpu_s": spent["cpu_h_measured"] * H},
        previous_attempts=[],
        write_s=30.0,
        info={"jobs": spent},
        native={
            "stitch_job": {
                "attempt": 2,
                "cpus": 4,
                "mem_gb": 8,
                "wall_s": 30.0,
                "cpu_s": 45.0,
                "peak_rss_mb": 900.0,
                "failed_attempts": [{"attempt": 1, "cpus": 4, "wall_s": 100.0}],
            }
        },
    )


GOOD_RESERVED = (2 * 60 + 2 * 50 + 2 * 40 + 1 * 10) / H
GOOD_USED = (90 + 70 + 50 + 8) / H
GOOD_GB = (2 * 60 + 8 * 50 + 4 * 40 + 1 * 10) / H


def test_a_staged_row_with_a_retry(tmp_path):
    out = tmp_path / "out"
    folder = out / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    # from what the run recorded alone: CPU-hours, and no memory-hours
    plain = score.resource_row(run, moving_px=1_000_000_000)
    assert plain["cpu_h_used"] == pytest.approx(GOOD_USED)
    assert plain["cpu_h_reserved"] == pytest.approx(GOOD_RESERVED)
    assert plain["cpu_efficiency"] == pytest.approx(GOOD_USED / GOOD_RESERVED)
    assert plain["wasted_cpu_h_used"] == pytest.approx(30 / H)
    assert plain["wasted_cpu_h_reserved"] == pytest.approx(2 * 20 / H)
    assert math.isnan(plain["gb_h_reserved"])
    assert plain["billing_note"] == billing.NOT_IN_RECORDS
    # the names the tables had before
    assert plain["cpu_h_allocated"] == plain["cpu_h_reserved"]
    assert plain["write_cpu_h_allocated"] == plain["write_cpu_h_reserved"]
    # STITCH is the writer: its own pair, and its failed attempt is the writer's
    assert plain["write_cpu_h_used"] == pytest.approx(45 / H)
    assert plain["write_cpu_h_reserved"] == pytest.approx(4 * 30 / H)
    assert plain["write_wasted_cpu_h_reserved"] == pytest.approx(4 * 100 / H)
    # with the stage jobs' records: each attempt's own memory, and its job
    stage = billing.stage_attempts(folder / "stages")
    assert len(stage[0]) == 5 and stage[1] == 0  # STITCH is not among them
    booked = {j: {"cpu_s": 100.0} for j in ("c1", "t0b", "t1", "s1")}
    booked["t0a"] = {"cpu_s": 5000.0}  # the retried attempt's job: not a good one
    row = score.resource_row(run, 1_000_000_000, stage, booked)
    assert row["gb_h_reserved"] == pytest.approx(GOOD_GB)
    assert row["wasted_gb_h_reserved"] == pytest.approx(4 * 20 / H)
    assert row["total_gb_h_reserved"] == pytest.approx(GOOD_GB + 80 / H)
    assert row["cpu_h_used"] == pytest.approx(GOOD_USED)
    assert row["cpu_h_reserved"] == pytest.approx(GOOD_RESERVED)
    assert row["total_cpu_h_reserved"] == pytest.approx(GOOD_RESERVED + 40 / H)
    assert row["cpu_h_scheduler"] == pytest.approx(400 / H)
    assert row["scheduler_jobs"] == 4 and row["billing_note"] == ""
    assert row["cpu_h_used_vs_scheduler"] == pytest.approx(GOOD_USED / (400 / H))
    assert row["cpu_h_used_per_gpx"] == pytest.approx(GOOD_USED)
    del booked["t1"]  # one job the scheduler says nothing of: no figure
    assert score.resource_row(run, None, stage, booked)["cpu_h_scheduler"] == ""
    # a round that failed: everything its stage jobs did is wasted
    lost = score.resource_row({**run, "ok": False, "cause": "exception"}, None, stage)
    assert math.isnan(lost["cpu_h_used"])
    assert lost["wasted_cpu_h_reserved"] == pytest.approx(GOOD_RESERVED + 40 / H)
    assert lost["wasted_gb_h_reserved"] == pytest.approx(GOOD_GB + 80 / H)
    # records that are not this result's are not used
    solve = folder / "stages" / "jobs" / "solve.json"
    solve.write_text(json.dumps({**json.loads(solve.read_text()), "wall_s": 999.0}))
    other = score.resource_row(run, None, billing.stage_attempts(folder / "stages"))
    assert other["billing_note"] == billing.RECORDS_CHANGED
    assert math.isnan(other["gb_h_reserved"])
    assert other["cpu_h_reserved"] == pytest.approx(GOOD_RESERVED)
    assert billing.stage_attempts(tmp_path / "nothing") is None


def test_the_tables_read_the_stage_records_and_name_the_parallel_time(tmp_path):
    out = tmp_path / "out"
    folder = out / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    (folder / "run.json").write_text(json.dumps(run))
    rigid = out / "stare_rigid" / "multiplex" / "P1_r1"
    rigid.mkdir(parents=True)
    (rigid / "run.json").write_text(json.dumps({**run, "native": None, "write_s": 0}))
    case = types.SimpleNamespace(
        dataset="multiplex", case_id="P1_r1", mov_hw=[20000, 50000]
    )
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}
    rows = [
        {
            **score.resource_row(run, score.moving_pixels(case)),
            **base,
            "method": "stare",
        },
        {
            **score.resource_row({**run, "native": None, "write_s": 0}, 10**9),
            **base,
            "method": "stare_rigid",
        },
        {**score.resource_row(single(), 10**9), **base, "method": "palom"},
    ]
    assert all(math.isnan(r["gb_h_reserved"]) for r in rows[:2])
    booked = {j: {"cpu_s": 100.0} for j in ("c1", "t0b", "t1", "s1", "20_3")}
    score.bill_from_disk(rows, out, [case], booked)
    # both results of the round share its stage records, found through the launch
    for r in rows[:2]:
        assert r["gb_h_reserved"] == pytest.approx(GOOD_GB)
        assert (
            r["cpu_h_scheduler"] == pytest.approx(400 / H) and r["billing_note"] == ""
        )
    assert rows[0]["moving_gpx"] == 1.0
    assert rows[2]["cpu_h_scheduler"] == pytest.approx(100 / H)
    per = score.summarise_resources(rows)
    summary = {r["method"]: r for r in score.summarise_cost(per)}
    stare, palom = summary["stare"], summary["palom"]
    # its time is not called like a single process's, and says how many CPUs it took
    assert "register_s_med" not in stare and "total_to_image_s_med" not in stare
    assert stare["time_unlimited_parallel_tiles_s_med"] == pytest.approx(120.0)
    assert stare["to_image_unlimited_parallel_tiles_s_med"] == pytest.approx(150.0)
    # every attempt of the fixture starts at the same moment: 2 + 2 + 2 + 2 + 1 CPUs
    assert stare["peak_cpus_at_once"] == 9
    assert palom["register_s_med"] == 1800.0 and palom["peak_cpus_at_once"] == ""
    assert "time_unlimited_parallel_tiles_s_med" not in palom
    assert stare["cpu_h_used_med"] == pytest.approx(GOOD_USED)
    assert stare["gb_h_reserved_med"] == pytest.approx(GOOD_GB)
    text = score.summary_markdown([], [], per, cost_summary=list(summary.values()))
    head = text[text.index("### Cost per method") :]
    names = [c.strip() for c in head.splitlines()[4].strip("|").split("|")]
    (line,) = [ln for ln in head.splitlines() if "| stare |" in ln]
    cells = dict(zip(names, (c.strip() for c in line.strip("|").split("|"))))
    assert cells["register_s: complete / common"] == "-"
    assert cells["time_unlimited_parallel_tiles_s: complete / common"] == "120 / 120"
    assert cells["peak_cpus_at_once"] == "9"
    assert names.index("cpu_h_used: complete / common") == 4
    block = text[text.index("## Resources (per patient") : text.index("### Cost per")]
    assert "| time_unlimited_parallel_tiles_s | peak_cpus_at_once |" in block
    assert score.VALID_RATIOS.strip() in block
