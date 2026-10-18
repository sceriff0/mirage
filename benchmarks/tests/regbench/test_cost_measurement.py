"""Task 31, the measurement side of the cost (review-resources-measurement.md, R1-R26).

Every expected number is worked out by hand from the records written in the test.
"""

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from benchmarks.regbench import billing, cost, score, stages, telemetry
from benchmarks.tests.regbench.test_billing import (
    GOOD_RESERVED,
    attempt_record,
    single,
    staged_round,
)

H = 3600.0
SLURM = Path(__file__).resolve().parents[2] / "regbench" / "slurm"
BASE = {"dataset": "multiplex", "group": "P1", "case_id": "P1_r1"}


# ── R8: memory-hours used, beside reserved ───────────────────────────────────────
def test_gb_h_used_is_each_attempts_own_peak_times_its_duration():
    """One process: 2 GB peak (2048 MB) for 1800 s = 1.0 GB-hour used, beside the
    64 GB x 1800 s reserved. The earlier rung recorded no peak: its used memory-hours
    are unknown (NaN), never zero."""
    row = score.resource_row(single())
    assert row["gb_h_used"] == pytest.approx(2.0 * 1800 / H)
    assert row["gb_h_reserved"] == pytest.approx(64.0 * 1800 / H)
    assert math.isnan(row["wasted_gb_h_used"])
    assert "gb_h_used" in billing.COLUMNS and "total_gb_h_used" in billing.COLUMNS
    known = score.resource_row(
        single(
            previous_attempts=[
                {**single()["previous_attempts"][0], "peak_rss_mb": 1024.0}
            ]
        )
    )
    assert known["wasted_gb_h_used"] == pytest.approx(1.0 * 900 / H)


def test_a_staged_rounds_gb_h_used_is_from_its_stage_records(tmp_path):
    """Every stage attempt of the fixture peaked at 500 MB: good 60 + 50 + 40 + 10 s,
    the retried tile 20 s."""
    folder = tmp_path / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    stage = billing.stage_attempts(folder / "stages")
    row = score.resource_row(run, None, stage)
    assert row["gb_h_used"] == pytest.approx(500 / 1024 * 160 / H)
    assert row["wasted_gb_h_used"] == pytest.approx(500 / 1024 * 20 / H)


# ── R6: a stage attempt the scheduler killed is settled from its accounting ──────
def _killed_tile(folder):
    jobs = folder / "stages" / "jobs"
    first = attempt_record("tile", 1, False, 2, 4, 20.0, 30.0, "t0a", 0)
    first.update(wall_s=None, cpu_s=None, exit=None, cause="killed", ended=None)
    tile = json.loads((jobs / "tile_000000.json").read_text())
    tile["history"] = [stages._summary(first)]
    (jobs / "tile_000000.json").write_text(json.dumps(tile))


def test_a_killed_stage_attempt_is_settled_as_wasted_from_the_stage_accounting(
    tmp_path,
):
    folder = tmp_path / "stare" / "multiplex" / "P1_r1"
    staged_round(folder)
    _killed_tile(folder)
    attempts, unmeasured = billing.stage_attempts(folder / "stages")
    assert unmeasured == 1 and len(attempts) == 4  # nobody knows what it cost
    booked = {"t0a": {"state": "OUT_OF_MEMORY", "elapsed_s": 100.0, "cpu_s": None}}
    attempts, unmeasured = billing.stage_attempts(folder / "stages", booked)
    assert unmeasured == 0
    (settled,) = [a for a in attempts if a.get("from_scheduler")]
    assert settled["kind"] == billing.WASTED
    assert settled["reserved_h"] == pytest.approx(2 * 100 / H)
    assert settled["gb_h"] == pytest.approx(4 * 100 / H)
    assert math.isnan(settled["used_h"])
    # through the scorer: the row counts it, and its unmeasured attempts go down
    run = json.loads(json.dumps(staged_round(tmp_path / "again" / "x")))
    run["info"]["jobs"]["unmeasured_attempts"] = 1
    row = score.resource_row(run, None, (attempts, unmeasured), booked)
    assert row["wasted_from_scheduler"] == 1 and row["unmeasured_attempts"] == 0
    assert row["wasted_cpu_h_reserved"] == pytest.approx(2 * 100 / H)


# ── R7: what the cluster held, from sacct ─────────────────────────────────────────
ACCT = """\
c1|COMPLETED|60|00:01:30|2|2G|
c1.batch|COMPLETED|60|00:01:30|2||1500K
20_3|COMPLETED|120|00:02:00|4|16Gn|
20_3.batch|COMPLETED|120|00:02:00|4||3G
t0b|COMPLETED|50|00:01:10|2|4096M|
t1|COMPLETED|40|00:00:50|2|1Gc|
s1|COMPLETED|10|00:00:08|1|1G|
old|COMPLETED|10|00:00:08
"""


def test_the_schedulers_reservation_is_parsed_and_summed():
    res = billing.parse_reservations(ACCT)
    assert set(res) == {"c1", "20_3", "t0b", "t1", "s1"}  # four fields: nothing
    assert res["c1"] == {
        "alloc_cpus": 2.0,
        "req_mem_gb": 2.0,
        "elapsed_s": 60.0,
        "max_rss_gb": pytest.approx(1500 / 1024**2),
    }
    assert res["20_3"]["req_mem_gb"] == 16.0 and res["20_3"]["max_rss_gb"] == 3.0
    assert res["t0b"]["req_mem_gb"] == 4.0
    assert res["t1"]["req_mem_gb"] == 2.0  # 1 GB per CPU x 2 CPUs
    cpu_h, gb_h, n = billing.reserved_hours(["c1", "20_3", "c1"], res)
    assert n == 2
    assert cpu_h == pytest.approx((2 * 60 + 4 * 120) / H)
    assert gb_h == pytest.approx((2 * 60 + 16 * 120) / H)
    # a job the accounting lacks: no partial sum
    assert all(math.isnan(v) for v in billing.reserved_hours(["c1", "zz"], res)[:2])
    assert all(math.isnan(v) for v in billing.reserved_hours(["c1"], {})[:2])


def _rows(tmp_path):
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
    rows = [
        {**score.resource_row(run, score.moving_pixels(case)), **BASE, "method": "stare"},
        {
            **score.resource_row({**run, "native": None, "write_s": 0}, 10**9),
            **BASE,
            "method": "stare_rigid",
        },
        {
            **score.resource_row(single(launch_id="30_1"), 10**9),
            **BASE,
            "method": "palom",
        },
    ]
    return out, case, rows


def test_the_reservation_of_a_staged_row_is_its_good_stage_jobs_and_its_driver(
    tmp_path,
):
    out, case, rows = _rows(tmp_path)
    res = billing.parse_reservations(ACCT)  # the staged row's driver job is 20_3
    res["30_1"] = {"alloc_cpus": 4.0, "req_mem_gb": 64.0, "elapsed_s": 1800.0}
    score.bill_from_disk(rows, out, [case], {}, res, {})
    stare, rigid, palom = rows
    good = (2 * 60 + 2 * 50 + 2 * 40 + 1 * 10 + 4 * 120) / H
    assert stare["cpu_h_reserved_scheduler"] == pytest.approx(good)
    assert stare["reservation_jobs"] == 5
    assert stare["gb_h_reserved_scheduler"] == pytest.approx(
        (2 * 60 + 4 * 50 + 2 * 40 + 1 * 10 + 16 * 120) / H
    )
    assert palom["cpu_h_reserved_scheduler"] == pytest.approx(4 * 1800 / H)
    assert palom["gb_h_reserved_scheduler"] == pytest.approx(64 * 1800 / H)
    # R21: the rigid result of the same launch is billed with stare, never twice
    assert rigid["cost_blank_reason"] == "billed with stare"
    for col in ("cpu_h_used", "cpu_h_reserved", "gb_h_reserved", "gb_h_used", "cpu_h"):
        assert rigid[col] == "", col
    assert stare["cost_blank_reason"] == "" and stare["cpu_h_reserved"] == pytest.approx(
        GOOD_RESERVED
    )
    # without the accounting: blank, never zero
    _, _, again = _rows(tmp_path / "b")
    score.bill_from_disk(again, out, [case], {}, {}, {})
    assert again[0]["cpu_h_reserved_scheduler"] == ""
    per = {r["method"]: r for r in score.summarise_resources(rows)}
    assert per["stare_rigid"]["cost_blank_reason"] == "billed with stare"
    assert per["palom"]["cpu_h_reserved_scheduler"] == pytest.approx(2.0)


# ── R4, R5, R9: the stage jobs' footprint ────────────────────────────────────────
def test_the_stage_footprint_is_one_jobs_need_and_the_concurrent_demand(tmp_path):
    """Three attempts: COARSE 0-10 s at 1 GB, two tiles 10-30 s at 3 and 4 GB (at once),
    SOLVE 30-35 s at 2 GB. The largest job needs 4 GB; at once they held 7 GB."""
    jobs = tmp_path / "stages" / "jobs"
    jobs.mkdir(parents=True)

    def rec(stage, start, end, mb, job, **kw):
        r = attempt_record(stage, 1, True, 2, 4, end - start, (end - start) * 2.5, job)
        r.update(started=start, ended=end, peak_rss_mb=mb, **kw)
        return r

    (jobs / "coarse.json").write_text(
        json.dumps(rec("coarse", 0, 10, 1024, "c", cgroup_peak_mb=1500.0))
    )
    (jobs / "tile_000000.json").write_text(
        json.dumps(
            rec("tile", 10, 30, 3072, "t0", cpus_requested=2, cpus_affinity=2, cpu_model="X")
        )
    )
    (jobs / "tile_000001.json").write_text(
        json.dumps(
            rec("tile", 10, 30, 4096, "t1", cpus_requested=2, cpus_affinity=8, cpu_model="Y")
        )
    )
    (jobs / "solve.json").write_text(json.dumps(rec("solve", 30, 35, 2048, "s")))
    tel = {"s": {"cgroup_peak_mb": "6144", "loadavg_start": "3.5 1 1"}}
    got = billing.stage_footprint(tmp_path / "stages", tel)
    assert got["peak_largest_job_gb"] == 4.0
    assert got["peak_concurrent_gb"] == 7.0
    # the record's own cgroup peak, else the job's telemetry
    assert got["stage_cgroup_peak_gb"] == 6.0
    assert got["stage_jobs_known"] == 2 and got["stage_jobs_confined"] == 1
    assert got["stage_cpu_efficiency_max"] == pytest.approx(1.25)
    assert got["stage_cpu_model"] == "X|Y"
    assert billing.stage_footprint(tmp_path / "none") is None


def test_a_staged_rows_cgroup_peak_is_blank_and_its_driver_job_is_named(tmp_path):
    folder = tmp_path / "stare" / "multiplex" / "P1_r1"
    run = staged_round(folder)
    run["resources"]["cgroup_peak_mb"] = 3072.0
    row = score.resource_row(run)
    assert row["cgroup_peak_gb"] == "" and row["driver_job_peak_gb"] == 3.0
    plain = score.resource_row(single(resources={**single()["resources"], "cgroup_peak_mb": 1024.0}))
    assert plain["cgroup_peak_gb"] == 1.0 and plain["driver_job_peak_gb"] == ""



def test_a_staged_rows_hwm_is_its_stage_jobs_never_its_driver_job(tmp_path):
    """T31 fix I2: the sidecar's VmHWM of a STARE row in several jobs is its driver
    job's (STITCH and the benchmark's work), so `peak_hwm_gb` is blank there, as
    `cgroup_peak_gb` is; `stage_hwm_peak_gb` is the largest of its GOOD stage attempts'
    own telemetry. A one-process row keeps its own job's."""
    out, case, rows = _rows(tmp_path)
    stare, _, palom = rows
    tel = {
        stare["launch_id"]: {"hwm_peak_mb": "9216"},  # the driver job: not the method's
        "c1": {"hwm_peak_mb": "1024"},
        "t0b": {"hwm_peak_mb": "5120"},
        "t0a": {"hwm_peak_mb": "20480"},  # a failed attempt: not the good one's need
        "30_1": {"hwm_peak_mb": "2048"},
    }
    score.bill_from_disk(rows, out, [case], {}, {}, tel)
    assert stare["peak_hwm_gb"] == ""
    assert stare["stage_hwm_peak_gb"] == 5.0
    assert palom["peak_hwm_gb"] == 2.0 and palom["stage_hwm_peak_gb"] == ""
    per = {r["method"]: r for r in score.summarise_resources(rows)}
    assert per["stare"]["stage_hwm_peak_gb"] == 5.0
    assert per["stare"]["peak_hwm_gb"] in ("", None) or per["stare"]["peak_hwm_gb"] != per["stare"]["peak_hwm_gb"]

# ── R11, R16, R19, R1: the new columns of a row ──────────────────────────────────
def test_a_row_names_its_gpu_peak_its_instruments_and_where_its_round_time_is_from():
    gpu = score.resource_row(
        single(
            info={"device": "cuda:0"},
            gpu_model="A100",
            resources={**single()["resources"], "gpu_peak_mb": 5120.0},
        )
    )
    assert gpu["gpu_peak_gb"] == 5.0
    assert score.resource_row(single())["gpu_peak_gb"] == ""
    lower = score.resource_row(
        single(
            resources={**single()["resources"], "rss_method": "rusage", "cpu_read_failures": 2}
        )
    )
    assert lower["peak_rss_comparable"] is False and lower["cpu_read_failures"] == 2.0
    assert lower["peak_rss_sampled_gb"] == lower["peak_rss_gb"] == 2.0
    assert score.resource_row(single())["peak_rss_comparable"] is True
    assert score.resource_row(single())["register_s_per_round_from"] == "own"
    patient = single(unit="patient", group_cases=["a", "b"])
    assert score.resource_row(patient)["register_s_per_round_from"] == "split"
    timed = single(unit="patient", group_cases=["a", "b"], info={"register_s_round": 9})
    assert score.resource_row(timed)["register_s_per_round_from"] == "info"
    jvm = score.resource_row(single(info={"jvm_max_heap_gb": 10.0}))
    assert jvm["jvm_peak_heap_gb"].startswith("NA")
    assert score.resource_row(single())["jvm_peak_heap_gb"] == ""


# ── R20: a patient's time is over every round or blank ───────────────────────────
def test_a_patients_time_and_cpu_are_blank_when_one_round_lacks_them():
    rows = [
        {**score.resource_row(single()), **BASE, "method": "palom"},
        {
            **score.resource_row(single(register_s=None, resources={"cpu_s": None})),
            **BASE,
            "case_id": "P1_r2",
            "method": "palom",
        },
    ]
    rows[1]["register_s"] = math.nan
    rows[1]["cpu_h"] = math.nan
    (patient,) = score.summarise_resources(rows)
    assert math.isnan(patient["register_s"]) and math.isnan(patient["cpu_h"])


# ── R17: a stage attempt without CPU time is unknown, never zero ─────────────────
def test_a_stage_attempt_without_cpu_time_makes_the_rounds_cpu_unknown():
    coarse = attempt_record("coarse", 1, True, 2, 2, 60.0, 90.0, "c1")
    tile = attempt_record("tile", 1, True, 2, 4, 40.0, None, "t1", 0)
    solve = attempt_record("solve", 1, True, 1, 1, 10.0, 8.0, "s1")
    got = stages.summarise(coarse, [tile], solve)
    assert math.isnan(got["cpu_h_measured"])
    tile["cpu_s"] = 50.0
    assert stages.summarise(coarse, [tile], solve)["cpu_h_measured"] == pytest.approx(
        148 / H
    )


# ── R5, R9, R14: what a stage job records of its host ────────────────────────────
def test_a_stage_record_names_its_cpu_its_cores_its_load_and_its_cgroup(tmp_path):
    rec = stages.run_command(
        tmp_path / "stages", stages.SOLVE, [sys.executable, "-c", "pass"]
    )
    for key in (
        "cpu_model",
        "cpus_requested",
        "cpus_affinity",
        "cpus_node",
        "loadavg_start",
        "loadavg_end",
        "cgroup_peak_mb",
    ):
        assert key in rec, key
        assert key in stages._summary(rec), key
    assert rec["loadavg_start"] is None or len(rec["loadavg_start"].split()) == 3


# ── R22, R23, R8, R16, R19, R14: the header lines ────────────────────────────────
def test_the_header_lines_say_what_differs_between_the_paths():
    rows = [
        {
            "cause": "",
            "ok": True,
            "adapter": "stare",
            "peak_cpus_at_once": 9.0,
            "n_tiles": 2.0,
            "rss_method": "rusage",
            "cpu_read_failures": 3.0,
            "telemetry": "no",
        },
        {
            "cause": "",
            "ok": True,
            "adapter": "valis",
            "jvm_max_heap_gb": 10.0,
            "rss_method": "psutil",
            "telemetry": "yes",
            "load_start": 2.0,
            "load_end": 4.0,
            "other_jobs_start": 1.0,
            "other_jobs_end": 3.0,
        },
    ]
    text = "\n".join(cost.header_lines(rows))
    assert cost.STARE_LADDER in text and "16 GB" in cost.STARE_LADDER
    assert cost.INSTRUMENTS_DIFFER in text and "[MEMORY" in cost.INSTRUMENTS_DIFFER
    assert cost.GB_H_POLICY in text and cost.JVM_RSS in text and "-Xmx" in cost.JVM_RSS
    assert "1 result(s) read their peak by `rusage` alone" in text
    assert "1 result(s) had CPU counter reads that failed" in text
    assert "shared nodes" in text and "headline is CPU-hours" in text
    assert "1 of 2 results' jobs left their telemetry" in text
    assert "REGBENCH_WAVES=1" in text
    # without a staged row and without a JVM, neither line
    plain = "\n".join(cost.header_lines(rows[1:2] and [{**rows[1], "jvm_max_heap_gb": ""}]))
    assert cost.STARE_LADDER not in plain and cost.JVM_RSS not in plain


def test_cores_confined_counts_the_stage_jobs_and_the_largest_efficiency():
    rows = [
        {"cause": "", "adapter": "palom", "cpus_asked": 4, "cpus_affinity": 4, "cpu_efficiency": 1.3},
        {
            "cause": "",
            "adapter": "stare",
            "cpus_asked": 1,
            "cpus_affinity": 1,
            "cpu_efficiency": 0.9,
            "stage_jobs_known": 4,
            "stage_jobs_confined": 3,
            "stage_cpu_efficiency_max": 1.5,
        },
    ]
    line = cost.cores_confined(rows)
    assert "Stage jobs: 3 of 4 attempts held to their CPUs." in line
    assert "palom 1.30" in line and "stare 1.50" in line
    assert "1 row(s) above 1.05" in line


# ── R14: the node telemetry ──────────────────────────────────────────────────────
def test_the_telemetry_files_are_read_per_launch(tmp_path):
    base = telemetry.folder(tmp_path)
    base.mkdir(parents=True)
    (base / "12_3.tsv").write_text(
        "kind\trun\nloadavg_start\t2.50 1.0 1.0 2/300 9\nother_jobs_start\t4\n"
        "hwm_peak_mb\t2048\nnot_a_key\tx\nbroken line\n"
    )
    tel = telemetry.read(tmp_path)
    assert set(tel["12_3"]) == {"kind", "loadavg_start", "other_jobs_start", "hwm_peak_mb"}
    got = telemetry.columns(tel, "12_3")
    assert got["load_start"] == 2.5 and got["other_jobs_start"] == 4.0
    assert got["peak_hwm_gb"] == 2.0 and got["telemetry"] == "yes"
    assert got["load_end"] == ""  # blank, never a zero
    none = telemetry.columns(tel, "99")
    assert none["telemetry"] == "no" and none["load_start"] == ""
    assert "the load is unknown" in telemetry.contention_line([{"cause": ""}])


def test_the_job_shell_leaves_its_telemetry(tmp_path):
    """env.sh's telemetry, run by bash against a /proc of the test's own, with a
    scheduler that lists this job and two others on the node."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "loadavg").write_text("1.50 1.00 0.50 2/100 7\n")
    (proc / "cpuinfo").write_text("processor\t: 0\nmodel name\t: Test CPU 9000\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "squeue").write_text("#!/bin/bash\nprintf '77\\n78\\n79\\n'\n")
    (fake / "squeue").chmod(0o755)
    out = tmp_path / "out"
    env = {
        **os.environ,
        "PATH": f"{fake}:{os.environ['PATH']}",
        "REGBENCH_PROC": str(proc),
        "REGBENCH_OUT": str(out),
        "SLURM_JOB_ID": "77",
        "SLURM_CPUS_PER_TASK": "4",
    }
    env.pop("SLURM_ARRAY_JOB_ID", None)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    script = (
        f'source "{SLURM}/env.sh"\nregbench_telemetry_start run\nregbench_telemetry_end\n'
        "regbench_telemetry_start score\n"
    )
    done = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    got = telemetry.read(out)
    assert set(got) == {"77"}  # the score job left none
    t = got["77"]
    assert t["kind"] == "run" and t["cpu_model"] == "Test CPU 9000"
    assert t["loadavg_start"].startswith("1.50") and t["loadavg_end"].startswith("1.50")
    assert t["other_jobs_start"] == "2" and t["cpus_per_task"] == "4"
    off = subprocess.run(
        ["bash", "-c", script],
        env={**env, "REGBENCH_TELEMETRY": "0", "SLURM_JOB_ID": "80"},
        capture_output=True,
        text=True,
    )
    assert off.returncode == 0 and "80" not in telemetry.read(out)
