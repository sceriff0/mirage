"""STARE's peak memory per stage per round (``regbench.stage_peaks``), on the real shape of a
stage job's record: ``stages.record_path`` / ``stages.read_record``, a ``history`` list of the
earlier attempts and the last attempt's own fields (``stages.attempts_of``)."""

import json

from benchmarks.regbench import score, stage_peaks, stages, table_schema


def _rec(*peaks, last_ok=True):
    """A record: every peak but the last is a failed attempt in ``history``; the last one
    is the record's own fields."""
    *before, final = peaks
    return {
        "history": [{"ok": False, "peak_rss_mb": p, "wall_s": 1.0} for p in before],
        "ok": last_ok,
        "peak_rss_mb": final,
        "wall_s": 1.0,
    }


def _put(stage_dir, stage, rec, row=None):
    path = stages.record_path(stage_dir, stage, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))


def test_one_round(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    sd = tmp_path / "stages"
    _put(sd, "coarse", _rec(900))
    for i, p in enumerate([1000, 3000, 2000]):
        _put(sd, "tile", _rec(p), row=i)
    _put(sd, "solve", _rec(300))
    run = {
        "info": {"implementation": {"tier": "high"}, "stride": 64},
        "native": {"stitch_job": {"peak_rss_mb": 2048}},
    }
    row = stage_peaks.round_row(run, sd, case)
    assert round(row["coarse_peak_gb"], 3) == round(900 / 1024, 3)
    assert row["tile_peak_gb_max"] == 3000 / 1024
    assert row["tile_peak_gb_p50"] == 2000 / 1024
    assert row["n_tiles"] == 3 and row["stitch_peak_gb"] == 2.0
    assert row["solve_peak_gb"] == 300 / 1024
    assert row["tier"] == "high" and row["stride"] == 64
    assert row["read_box_px"] == 2048 + 384 and row["pipeline_window_px"] == 2048 + 512
    assert row["read_box_mpx"] == (2048 + 384) ** 2 / 1e6
    assert row["dataset"] == case.dataset and row["case_id"] == case.case_id
    assert row["patient_id"] == case.group and row["panel"] == score.panel_of(case)
    assert set(row) == set(table_schema.columns("stare_stage_peaks")[1:])


def test_round_without_stage_records(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    row = stage_peaks.round_row({"info": {}}, tmp_path / "none", case)
    assert row["coarse_peak_gb"] is None and row["n_tiles"] == 0
    assert row["read_box_px"] is None and row["stitch_peak_gb"] is None


def test_a_retried_tile_counts_the_attempt_that_ended_ok(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    sd = tmp_path / "stages"
    _put(sd, "tile", _rec(4096, 1024), row=0)
    row = stage_peaks.round_row({"info": {}}, sd, case)
    assert row["tile_peak_gb_max"] == 1.0 and row["n_tiles"] == 1


def test_a_job_that_did_not_end_ok_has_no_peak(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    sd = tmp_path / "stages"
    _put(sd, "coarse", _rec(900, last_ok=False))
    _put(sd, "tile", _rec(5000, last_ok=False), row=0)
    _put(sd, "tile", _rec(1024), row=1)
    row = stage_peaks.round_row({"info": {}}, sd, case)
    assert row["coarse_peak_gb"] is None
    assert row["tile_peak_gb_max"] == 1.0 and row["n_tiles"] == 1


def test_a_tile_changed_from_its_tier_sizes_the_windows_with_the_change(
    tmp_path, toy_cases
):
    case = next(iter(toy_cases.values()))
    run = {
        "info": {
            "stride": 128,
            "implementation": {
                "tier": "high",
                "changed_from_tier": {"tile": "1024", "halo": "512"},
            },
        }
    }
    row = stage_peaks.round_row(run, tmp_path / "none", case)
    assert row["read_box_px"] == 1024 + 2 * 3 * 128
    assert row["pipeline_window_px"] == 1024 + 2 * max(512, 3 * 128)


def test_rows_cover_every_stare_result_present(tmp_path, toy_cases):
    case = next(iter(toy_cases.values()))
    out = tmp_path / "out"
    from benchmarks.regbench.cases import result_dir

    d = result_dir(out, "stare", case)
    (d).mkdir(parents=True)
    (d / "run.json").write_text(
        json.dumps({"info": {"implementation": {"tier": "low"}, "stride": 64}})
    )
    _put(d / "stages", "coarse", _rec(512))
    other = result_dir(out, "ashlar", case)
    other.mkdir(parents=True)
    (other / "run.json").write_text("{}")
    got = stage_peaks.rows(out, [case], ["initial", "stare", "ashlar", "stare_rigid"])
    assert [r["row"] for r in got] == ["stare"]
    assert got[0]["coarse_peak_gb"] == 0.5 and got[0]["tier"] == "low"
