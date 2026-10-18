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


def test_the_writers_cpu_time_of_a_single_process_row():
    """``write_cpu_s`` is what the run's counter advanced by while the writer ran; the
    note says when that stretch holds more than the writer."""
    own = score.resource_row(
        single(write_cpu_s=900.0, write_cpu_from=billing.WRITE_CPU_CALL)
    )
    assert own["write_cpu_h_used"] == pytest.approx(0.25)
    assert own["billing_note"] == ""
    # an adapter that times its writer apart from the benchmark's own copying: the
    # CPU is the whole step's, and the row says so
    parts = {"convert": 0.0, "writer": 500.0, "benchmark": 100.0}
    whole = score.resource_row(
        single(
            write_cpu_s=900.0,
            write_cpu_from=billing.WRITE_CPU_CALL,
            native={"write_parts_s": parts},
        )
    )
    assert whole["write_cpu_h_used"] == pytest.approx(0.25)
    assert whole["billing_note"] == billing.WRITE_CPU_HOLDS.format("benchmark")
    assert whole["write_cpu_h_reserved"] == pytest.approx(4 * 500 / H)
    # a writer that ran at a yield, inside the adapter's own paused stretch
    shared = score.resource_row(
        single(write_cpu_s=900.0, write_cpu_from=billing.WRITE_CPU_STRETCH)
    )
    assert shared["billing_note"] == billing.WRITE_CPU_IS_THE_STRETCH
    # a row that failed has no writer's cost
    lost = score.resource_row(single(ok=False, cause="writer", write_cpu_s=900.0))
    assert lost["write_cpu_h_used"] == "" and lost["billing_note"] == ""
    # the counter's name is a column
    named = single()
    named["resources"]["cpu_instrument"] = "cgroup_v2"
    assert score.resource_row(named)["cpu_instrument"] == "cgroup_v2"
    assert score.resource_row(single())["cpu_instrument"] == ""


def test_the_summary_names_the_counters_and_warns_only_of_the_sampler():
    exact = score.cpu_instruments_line(
        [{"cpu_instrument": "cgroup_v2"}, {"cpu_instrument": "cgroup_v2; rusage+live"}]
    )
    assert exact == "here: `cgroup_v2`, `rusage+live`"
    polled = score.cpu_instruments_line([{"cpu_instrument": "polled (may undercount)"}])
    assert "`polled (may undercount)`" in polled and "LOWER BOUND" in polled
    assert "LOWER BOUND" not in exact
    assert "no row" in score.cpu_instruments_line([{"cpu_instrument": ""}, {}])


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
    # the writer is apart: the job's CPUs for its 600 s; this record carries no CPU
    # time of it, and a blank is not a zero
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
    # an attempt nobody could bill is counted as unknown, and a wasted sum made of
    # such attempts alone is blank, never zero; the total is then the good part
    unknown = score.resource_row(
        single(previous_attempts=[{"attempt": 1, "mem_requested_gb": 16.0}])
    )
    assert unknown["wasted_unknown"] == 1
    assert math.isnan(unknown["wasted_cpu_h_reserved"])
    assert unknown["total_cpu_h_reserved"] == pytest.approx(2.0)


def test_a_sum_is_blank_where_its_figure_is_missing_and_never_zero():
    """4 CPUs and 16 GB for 900 s good, with no CPU time and no memory recorded for
    it; one wasted attempt of 2 CPUs and 8 GB for 1800 s that used 600 CPU seconds."""
    good = billing.attempt(billing.GOOD, 4, None, 900.0, None)
    bad = billing.attempt(billing.WASTED, 2, 8, 1800.0, 600.0)
    got = billing.sums([good, bad])
    assert got["cpu_h_reserved"] == pytest.approx(1.0)
    assert got["wasted_cpu_h_reserved"] == pytest.approx(1.0)
    assert got["total_cpu_h_reserved"] == pytest.approx(2.0)
    # the good figure is missing: the total is not the wasted part alone
    assert math.isnan(got["gb_h_reserved"]) and math.isnan(got["total_gb_h_reserved"])
    assert got["wasted_gb_h_reserved"] == pytest.approx(8 * 1800 / H)  # 4.0
    assert math.isnan(got["cpu_h_used"]) and math.isnan(got["total_cpu_h_used"])
    # wasted attempts that all lack a figure: blank; one that has it: its sum
    blind = billing.attempt(billing.WASTED, 2, None, 1800.0, None)
    got = billing.sums([blind, blind])
    assert math.isnan(got["wasted_gb_h_reserved"])
    assert math.isnan(got["wasted_cpu_h_used"])
    assert got["wasted_cpu_h_reserved"] == pytest.approx(2.0)
    assert math.isnan(got["total_gb_h_reserved"])
    assert billing.sums([blind, bad])["wasted_gb_h_reserved"] == pytest.approx(4.0)
    # no attempt of a kind is a zero of that kind, and no attempt at all no total
    none = billing.sums([])
    assert none["wasted_gb_h_reserved"] == 0.0 and none["lost_gb_h_reserved"] == 0.0
    assert math.isnan(none["total_gb_h_reserved"])


def test_an_attempt_lost_to_the_cluster_is_no_cost_of_the_method():
    """The second rung registered (2.0 reserved CPU-hours, 32 GB-hours, 1.0 used).
    Before it: a rung that ran out of memory (4 CPUs, 16 GB, 900 s, 1200 CPU s) and a
    node that failed under the retry (4 CPUs, 16 GB, 450 s by the scheduler, 300 CPU
    s by the scheduler)."""
    oom = single()["previous_attempts"][0]
    node = {
        "attempt": 1,
        "retry": 1,
        "cause": "infrastructure",
        "mem_requested_gb": 16.0,
        "launch_id": "11_3",
        "scheduler_state": "NODE_FAIL",
        "scheduler_elapsed_s": 450.0,
        "scheduler_cpu_s": 300.0,
    }
    row = score.resource_row(single(previous_attempts=[oom, node]))
    assert row["wasted_cpu_h_reserved"] == pytest.approx(4 * 900 / H)  # 1.0
    assert row["wasted_cpu_h_used"] == pytest.approx(1200 / H)
    assert row["wasted_gb_h_reserved"] == pytest.approx(4.0)
    assert row["lost_cpu_h_reserved"] == pytest.approx(4 * 450 / H)  # 0.5
    assert row["lost_cpu_h_used"] == pytest.approx(300 / H)
    assert row["lost_gb_h_reserved"] == pytest.approx(16 * 450 / H)  # 2.0
    assert row["total_cpu_h_reserved"] == pytest.approx(3.0)  # 2 good + 1 wasted
    assert row["total_gb_h_reserved"] == pytest.approx(36.0)
    assert row["total_cpu_h_used"] == pytest.approx(4800 / H)
    assert (row["wasted_attempts"], row["lost_attempts"]) == (1, 1)
    assert row["wasted_cpu_h"] == pytest.approx(1200 / H)
    assert row["wasted_wall_s"] == 900.0 and row["wasted_from_scheduler"] == 0
    assert set(billing.COLUMNS) <= set(row)
    # a unit whose last attempt was lost too: nothing of it is the method's
    gone = score.resource_row(
        single(
            ok=False,
            cause="infrastructure",
            resources=None,
            register_s=None,
            scheduler={"state": "NODE_FAIL", "elapsed_s": 900.0, "cpu_s": 600.0},
            previous_attempts=[node],
        )
    )
    assert gone["lost_cpu_h_reserved"] == pytest.approx(4 * (450 + 900) / H)  # 1.5
    assert gone["lost_cpu_h_used"] == pytest.approx(900 / H)
    assert gone["wasted_cpu_h_reserved"] == 0.0 and gone["wasted_attempts"] == 0
    assert math.isnan(gone["total_cpu_h_reserved"]) and gone["lost_attempts"] == 2
    assert gone["wasted_from_scheduler"] == 0 and gone["wasted_unknown"] == 0
    # per patient: the lost attempts are summed apart, and are in no total
    base = {"dataset": "multiplex", "group": "P1", "method": "pair"}
    rows = [
        {**row, **base, "case_id": "P1_r1"},
        {**gone, **base, "case_id": "P1_r2"},
    ]
    (patient,) = score.summarise_resources(rows)
    assert patient["lost_cpu_h_reserved"] == pytest.approx(2.0)
    assert patient["lost_attempts"] == 3 and patient["wasted_attempts"] == 1
    assert patient["wasted_cpu_h_reserved"] == pytest.approx(1.0)
    assert patient["total_cpu_h_reserved"] == pytest.approx(3.0)
    text = score.summary_markdown([], [], [patient])
    assert "| lost_cpu_h_reserved |" in text and "`lost_attempts`" in text


def test_a_stage_attempt_lost_to_the_cluster_is_lost(tmp_path):
    jobs = tmp_path / "stages" / "jobs"
    jobs.mkdir(parents=True)
    first = attempt_record("tile", 1, False, 2, 4, 20.0, 30.0, "t0a", 0)
    first["cause"] = "infrastructure"
    tile = attempt_record("tile", 2, True, 2, 8, 50.0, 70.0, "t0b", 0)
    tile["history"] = [stages._summary(first)]
    (jobs / "tile_000000.json").write_text(json.dumps(tile))
    attempts, _ = billing.stage_attempts(tmp_path / "stages")
    assert [a["kind"] for a in attempts] == [billing.LOST, billing.GOOD]
    got = billing.sums(attempts)
    assert got["lost_cpu_h_reserved"] == pytest.approx(2 * 20 / H)
    assert got["wasted_cpu_h_reserved"] == 0.0
    assert got["total_cpu_h_reserved"] == pytest.approx(2 * 50 / H)


def test_a_patients_columns_are_over_the_same_rounds():
    """Two rounds of a pair method, each its own job. Round 1: 1.0 CPU-hour used, 2.0
    reserved, 32 GB-hours, and 1.5 CPU-hours by the scheduler. Round 2: 1.0 used, 4.0
    reserved; no memory request recorded, and nothing in the accounting."""
    base = {"dataset": "multiplex", "group": "P1", "method": "pair"}
    r1 = {**score.resource_row(single(previous_attempts=[])), **base}
    r2 = {
        **score.resource_row(
            single(
                launch_id="20_4",
                register_s=3600.0,
                resources={"wall_s": 3600.0, "cpu_s": 3600.0},
                previous_attempts=[],
                mem_requested_gb=None,
            )
        ),
        **base,
    }
    r1["case_id"], r2["case_id"] = "P1_r1", "P1_r2"
    cases = [
        types.SimpleNamespace(dataset="multiplex", case_id=c, mov_hw=None)
        for c in ("P1_r1", "P1_r2")
    ]
    score.bill_from_disk([r1, r2], "/nowhere", cases, {"20_3": {"cpu_s": 5400.0}})
    assert r1["cpu_h_scheduler"] == pytest.approx(1.5) and r2["cpu_h_scheduler"] == ""
    (patient,) = score.summarise_resources([r1, r2])
    assert patient["cpu_h_used"] == pytest.approx(2.0)
    assert patient["cpu_h_reserved"] == pytest.approx(6.0)
    assert patient["total_cpu_h_reserved"] == pytest.approx(6.0)
    # one round lacks the figure: no sum over the other one alone, and no ratio of a
    # two-round numerator over a one-round denominator
    assert math.isnan(patient["gb_h_reserved"])
    assert math.isnan(patient["total_gb_h_reserved"])
    assert math.isnan(patient["cpu_h_scheduler"])
    assert math.isnan(patient["cpu_h_used_vs_scheduler"])
    # with both: 1.5 + 2.0 by the scheduler
    score.bill_from_disk(
        [r1, r2], "/nowhere", cases, {k: {"cpu_s": 7200.0} for k in ("20_4",)}
    )
    (patient,) = score.summarise_resources([r1, r2])
    assert patient["cpu_h_scheduler"] == pytest.approx(3.5)
    assert patient["cpu_h_used_vs_scheduler"] == pytest.approx(2.0 / 3.5)


def test_a_round_whose_writer_failed_is_in_its_patients_cost():
    """Round 1 registered and wrote (1.0 used, 2.0 reserved, 32 GB-hours). Round 2
    registered (4 CPUs, 16 GB, 600 s, 900 CPU s) and its writer failed."""
    base = {"dataset": "multiplex", "group": "P1", "method": "pair"}
    r1 = {**score.resource_row(single(previous_attempts=[])), **base}
    r2 = {
        **score.resource_row(
            single(
                ok=False,
                cause="writer",
                registered=True,
                launch_id="20_4",
                mem_requested_gb=16.0,
                register_s=600.0,
                resources={"wall_s": 600.0, "cpu_s": 900.0},
                previous_attempts=[],
            )
        ),
        **base,
    }
    r1["case_id"], r2["case_id"] = "P1_r1", "P1_r2"
    (patient,) = score.summarise_resources([r1, r2])
    assert patient["cpu_h_used"] == pytest.approx((3600 + 900) / H)  # 1.25
    assert patient["cpu_h_reserved"] == pytest.approx(2.0 + 4 * 600 / H)
    assert patient["gb_h_reserved"] == pytest.approx(32.0 + 16 * 600 / H)
    assert patient["total_cpu_h_reserved"] == pytest.approx(2.0 + 4 * 600 / H)
    assert patient["wasted_cpu_h_reserved"] == 0.0 and patient["wasted_attempts"] == 0
    assert not patient["complete"] and patient["n_ok"] == 1


def test_the_cost_table_counts_what_a_median_is_over_and_names_the_dataset():
    def patient(ds, g, m, gb, sched):
        return {
            "dataset": ds, "group": g, "method": m, "unit": "pair", "n_rounds": 1,
            "n_ok": 1, "complete": True, "launched": True, "cpu_h_used": 1.0,
            "gb_h_reserved": gb, "cpu_h_scheduler": sched,
        }  # fmt: skip

    nan = math.nan
    per = [
        patient("multiplex", "P1", "a", 8.0, 2.0),
        patient("multiplex", "P2", "a", 4.0, nan),
        patient("multiplex", "P3", "a", nan, nan),
        patient("multiplex", "P1", "b", 1.0, 1.0),
        patient("multiplex", "P2", "b", 1.0, 1.0),
    ]
    rows = score.summarise_cost(per)
    a = next(r for r in rows if (r["dataset"], r["method"]) == ("multiplex", "a"))
    assert (a["n_patients_complete"], a["n_patients_common"]) == (3, 2)
    assert a["gb_h_reserved_med"] == 6.0 and a["gb_h_reserved_n"] == 2
    assert a["gb_h_reserved_common_med"] == 6.0 and a["gb_h_reserved_common_n"] == 2
    assert a["cpu_h_scheduler_med"] == 2.0
    assert (a["cpu_h_scheduler_n"], a["cpu_h_scheduler_common_n"]) == (1, 1)
    assert "cpu_h_used_n" not in a
    text = score.summary_markdown([], [], [], cost_summary=rows)
    assert "(compared rows: multiplex: a, b)" in text
    (line,) = [ln for ln in text.splitlines() if ln.startswith("| multiplex | a |")]
    assert "6 / 6 (n 2 / 2)" in line


def test_a_staged_round_that_failed_has_its_own_time_and_allocation(tmp_path):
    """The fixture's round, failed before its writer: 218 good and 30 retried CPU
    seconds, on 310 good and 40 retried allocated CPU seconds; 120 s with every tile at
    once plus the retried attempt's 20 s. STITCH's failed 100 s are the writer's."""
    run = staged_round(tmp_path / "out" / "stare" / "multiplex" / "P1_r1")
    row = score.resource_row({**run, "ok": False, "cause": "exception"})
    assert row["cpu_h"] == pytest.approx(248 / H)
    assert row["cpu_efficiency"] == pytest.approx(248 / 350)
    assert row["register_s"] == pytest.approx(140.0)


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


@pytest.mark.parametrize(
    "cause, column, other",
    [("killed", "wasted", "lost"), ("not_submitted", "lost", "wasted")],
)
def test_a_staged_round_without_its_summary_is_billed_from_its_stage_records(
    tmp_path, cause, column, other
):
    """The round's last job was killed (or the run was refused) before it wrote what
    its stage jobs cost: `info.jobs` is missing. The stage records are on disk: 310
    good and 40 retried allocated CPU seconds, none of which registered anything."""
    out = tmp_path / "out"
    folder = out / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    run.update(ok=False, cause=cause, resources=None, register_s=None, info={})
    run.pop("native")
    (folder / "run.json").write_text(json.dumps(run))
    case = types.SimpleNamespace(dataset="multiplex", case_id="P1_r1", mov_hw=None)
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}
    row = {**score.resource_row(run), **base, "method": "stare"}
    assert row["n_jobs"] == "" and row["adapter"] == "stare"
    assert row[f"{column}_cpu_h_reserved"] == 0.0  # as one process: nothing of them
    assert other in ("wasted", "lost")
    score.bill_from_disk([row], out, [case], {})
    assert math.isnan(row["cpu_h_used"])
    if cause == "killed":  # the method's: every attempt of its stage jobs is wasted
        assert row["wasted_cpu_h_reserved"] == pytest.approx(GOOD_RESERVED + 40 / H)
        assert row["wasted_gb_h_reserved"] == pytest.approx(GOOD_GB + 80 / H)
        assert row["wasted_cpu_h_used"] == pytest.approx(GOOD_USED + 30 / H)
        assert row["total_cpu_h_reserved"] == pytest.approx(GOOD_RESERVED + 40 / H)
        assert row["lost_cpu_h_reserved"] == 0.0
    else:
        # not the method's: what ended ok is lost; the tile attempt that was retried
        # (2 CPUs, 4 GB, 20 s) stays the method's own
        assert row["lost_cpu_h_reserved"] == pytest.approx(GOOD_RESERVED)
        assert row["lost_gb_h_reserved"] == pytest.approx(GOOD_GB)
        assert row["lost_cpu_h_used"] == pytest.approx(GOOD_USED)
        assert row["wasted_cpu_h_reserved"] == pytest.approx(40 / H)
        assert row["total_cpu_h_reserved"] == pytest.approx(40 / H)
    # a method that is one process is never looked up
    plain = {
        **score.resource_row(single(ok=False, cause="killed")),
        **base,
        "method": "palom",
    }
    before = dict(plain)
    score.bill_from_disk([plain], out, [case], {})
    assert plain == before


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


def test_the_jvm_heap_is_in_the_tables_as_the_run_recorded_it():
    """What the JVM itself reported, beside what the adapter asked for: recorded, and
    set by nothing here."""
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}
    asked = single(method="valis", info={"jvm_heap_gb": 60, "jvm_max_heap_gb": 60.0})
    kept = single(
        method="valis_mirage", info={"jvm_heap_gb": 60, "jvm_max_heap_gb": 10.0}
    )
    rows = [
        {**score.resource_row(asked), **base, "method": "valis120_hires"},
        {**score.resource_row(kept), **base, "method": "valis_mirage"},
        {**score.resource_row(single()), **base, "method": "palom"},
    ]
    assert (rows[0]["jvm_heap_asked_gb"], rows[0]["jvm_max_heap_gb"]) == (60.0, 60.0)
    assert (rows[1]["jvm_heap_asked_gb"], rows[1]["jvm_max_heap_gb"]) == (60.0, 10.0)
    assert rows[2]["jvm_heap_asked_gb"] == "" and rows[2]["jvm_max_heap_gb"] == ""
    per = {r["method"]: r for r in score.summarise_resources(rows)}
    assert per["valis_mirage"]["jvm_max_heap_gb"] == 10.0
    assert per["palom"]["jvm_max_heap_gb"] == ""
    (line,) = score.jvm_heap_lines(list(per.values()))
    assert "`valis120_hires` 60 GiB (asked 60)" in line
    assert "`valis_mirage` 10 GiB (asked 60)" in line and "palom" not in line
    assert score.jvm_heap_lines([per["palom"]]) == []
    assert line in score.summary_markdown([], [], list(per.values()))


def test_an_unreadable_stage_record_is_counted_not_skipped(tmp_path):
    """S5 (Task 28): a stage record nobody can read has no attempt in any figure; the
    row says how many there were, so its times and CPU-hours read as lower bounds."""
    out = tmp_path / "out"
    folder = out / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    (folder / "run.json").write_text(json.dumps(run))
    jobs = folder / "stages" / "jobs"
    assert billing.unreadable_stage_records(folder / "stages") == 0
    (jobs / "solve.json").write_text("{ cut short")
    (jobs / "stitch.json").write_text("{ cut short")  # the writer's: not counted
    assert billing.unreadable_stage_records(folder / "stages") == 1
    case = types.SimpleNamespace(dataset="multiplex", case_id="P1_r1", mov_hw=None)
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}
    row = {**score.resource_row(run), **base, "method": "stare"}
    assert row["unmeasured_records"] == ""
    score.bill_from_disk([row], out, [case], {})
    assert row["unmeasured_records"] == 1


def test_an_unsettled_kill_is_billed_as_wasted_by_the_schedulers_upper_bound():
    """R26 (resources review): `killed_unsettled` is not the method's for the scores
    (score.NOT_THE_METHOD) but its attempt is billed as WASTED, never lost: an OOM
    kill nobody settled is otherwise billed to no one. Its cost is the scheduler's
    upper bound (elapsed x allocated CPUs, requested memory), and the row says
    `cost_bound = upper`."""
    assert "killed_unsettled" in score.NOT_THE_METHOD
    assert billing.kind_of("killed_unsettled") == billing.WASTED
    assert billing.kind_of("watcher_gave_up") == billing.LOST  # still lost
    run = single(ok=False, cause="killed_unsettled", resources=None, register_s=None,
                 previous_attempts=[])  # fmt: skip
    base = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1", "method": "palom"}
    row = {**score.resource_row(run), **base}
    assert row["cost_bound"] == "upper" and row["lost_attempts"] == 0
    assert row["wasted_unknown"] == 1
    case = types.SimpleNamespace(dataset="multiplex", case_id="P1_r1", mov_hw=None)
    booked = {"20_3": {"state": "RUNNING", "elapsed_s": 7200.0, "cpu_s": 3600.0}}
    score.bill_from_disk([row], "/nowhere", [case], booked)
    assert row["wasted_cpu_h_reserved"] == pytest.approx(4 * 2.0)
    assert row["wasted_gb_h_reserved"] == pytest.approx(64 * 2.0)
    assert row["total_cpu_h_reserved"] == pytest.approx(4 * 2.0)
    assert row["wasted_unknown"] == 0 and row["wasted_from_scheduler"] == 1
    assert row["lost_cpu_h_reserved"] == 0.0
    # a row whose cause is the method's is no bound
    assert score.resource_row(single())["cost_bound"] == ""
