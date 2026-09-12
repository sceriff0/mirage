"""warp.py: the STARE leg through the REAL bin/utils warper, the baseline reader,
slide-name resolution, the trace-time reader, and the batch driver."""

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.anhir import prepare, warp
from benchmarks.anhir.landmarks import read_landmarks
from benchmarks.tests.anhir.conftest import OFFSET


def _manifest(moving_name: str, dx: float, dy: float, with_mesh: bool = False) -> dict:
    mesh = None
    if with_mesh:
        # a constant +1,+1 residual over the whole reference frame
        mesh = {
            "grid_x": [0.0, 5000.0],
            "grid_y": [0.0, 5000.0],
            "displacements": [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        }
    return {
        "ref_slide": "ref",
        "slides": {
            "ref": {"M0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "mesh": None},
            moving_name: {"M0": [[1, 0, dx], [0, 1, dy], [0, 0, 1]], "mesh": mesh},
        },
    }


def test_warp_tiled_applies_m0_then_the_mesh_through_the_pipelines_own_warper(tmp_path):
    m = tmp_path / "x_manifest.json"
    m.write_text(json.dumps(_manifest("DAPI_S1", 5.0, -3.0, with_mesh=True)))
    xy = np.array([[10.0, 10.0], [100.0, 200.0]])
    rigid = warp.warp_tiled(m, "DAPI_S1", xy, stage="rigid")
    np.testing.assert_allclose(rigid, xy + [5.0, -3.0])
    refined = warp.warp_tiled(m, "DAPI_S1", xy)
    np.testing.assert_allclose(refined, xy + [6.0, -2.0])


def test_pick_moving_prefers_the_named_slide_then_the_single_non_reference():
    assert warp.pick_moving(["ref", "DAPI_S1"], "ref", "DAPI_S1") == "DAPI_S1"
    assert warp.pick_moving(["ref", "anything"], "ref", "DAPI_S1") == "anything"
    with pytest.raises(KeyError, match="cannot pick"):
        warp.pick_moving(["ref", "a", "b"], "ref", "DAPI_S1")


def test_find_transform_reads_the_published_layout_and_names_what_is_missing(tmp_path):
    out = tmp_path / "results"
    p = out / "anhir0" / "registered" / "manifest" / "anhir0_DAPI_S1_manifest.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    assert warp.find_transform(out, "anhir0", "tiled") == p
    with pytest.raises(FileNotFoundError, match="registered/transform"):
        warp.find_transform(out, "anhir0", "valis")
    with pytest.raises(FileNotFoundError, match="anhir1"):
        warp.find_transform(out, "anhir1", "tiled")


def test_bunwarpj_case_reads_points_and_milliseconds(anhir_root):
    xy, t = warp.bunwarpj_case(anhir_root["baseline"], 0)
    assert xy.shape == (8, 2)
    assert t == pytest.approx(2.0)
    with pytest.raises(FileNotFoundError):
        warp.bunwarpj_case(anhir_root["baseline"], 99)


def test_times_from_trace_sums_registration_processes_per_patient(tmp_path):
    trace = tmp_path / "trace.txt"
    cols = [
        "task_id",
        "process",
        "tag",
        "status",
        "exit",
        "peak_rss",
        "peak_vmem",
        "realtime",
        "duration",
        "cpus",
    ]
    rows = [
        [
            1,
            "MIRAGE:REGISTRATION:TILED_ADAPTER:TILED_COARSE",
            "anhir0",
            "COMPLETED",
            0,
            "1 GB",
            "2 GB",
            "1m",
            "1m 10s",
            2,
        ],
        [
            2,
            "MIRAGE:REGISTRATION:TILED_ADAPTER:TILED_SOLVE",
            "anhir0",
            "COMPLETED",
            0,
            "1 GB",
            "2 GB",
            "30s",
            "40s",
            1,
        ],
        [
            3,
            "MIRAGE:REGISTRATION:TILED_ADAPTER:TILED_STITCH",
            "anhir0",
            "COMPLETED",
            0,
            "1 GB",
            "2 GB",
            "10m",
            "10m",
            1,
        ],  # stitching is not registration time
        [
            4,
            "MIRAGE:REGISTRATION:VALIS_ADAPTER:REGISTER",
            "anhir1",
            "COMPLETED",
            0,
            "1 GB",
            "2 GB",
            "3m",
            "3m",
            8,
        ],
    ]
    pd.DataFrame(rows, columns=cols).to_csv(trace, sep="\t", index=False)
    t = warp.times_from_trace(trace, ["anhir0", "anhir1", "anhir2"])
    assert t["anhir0"] == pytest.approx(1.5)
    assert t["anhir1"] == pytest.approx(3.0)
    assert "anhir2" not in t


def _pairs(anhir_root, tmp_path):
    _, pairs = prepare.build_inputs(
        anhir_root["cases"], anhir_root["root"], tmp_path / "w", convert=False
    )
    return pairs


def test_warp_all_initial_copies_the_source_landmarks(anhir_root, tmp_path):
    pairs = _pairs(anhir_root, tmp_path)
    index = warp.warp_all("initial", pairs, tmp_path / "initial")
    assert list(index["case_id"]) == [0, 1, 2] and (index["error"] == "").all()
    src = read_landmarks(pairs.loc[0, "source_landmarks"])
    np.testing.assert_allclose(read_landmarks(tmp_path / "initial" / "0.csv"), src)
    assert (tmp_path / "initial" / "warp_index.csv").exists()


def test_warp_all_tiled_finds_each_patients_manifest_and_records_failures(
    anhir_root, tmp_path
):
    pairs = _pairs(anhir_root, tmp_path)
    outdir = tmp_path / "results" / "tiled"
    for rec in pairs.itertuples():
        if rec.case_id == 2:
            continue  # simulate one case whose run produced nothing
        p = (
            outdir
            / rec.patient_id
            / "registered"
            / "manifest"
            / f"{rec.patient_id}_manifest.json"
        )
        p.parent.mkdir(parents=True)
        # a perfect rigid correction: undo the fixture's constant OFFSET
        p.write_text(json.dumps(_manifest(rec.moving_channels, -OFFSET[0], -OFFSET[1])))
    index = warp.warp_all("tiled", pairs, tmp_path / "tiled", outdir=outdir)
    by_case = index.set_index("case_id")
    assert by_case.loc[0, "error"] == "" and by_case.loc[1, "error"] == ""
    assert "FileNotFoundError" in by_case.loc[2, "error"]
    assert not (tmp_path / "tiled" / "2.csv").exists()
    target = read_landmarks(pairs.loc[0, "target_landmarks"])
    np.testing.assert_allclose(read_landmarks(tmp_path / "tiled" / "0.csv"), target)


def test_warp_all_refuses_a_pipeline_method_without_an_outdir(anhir_root, tmp_path):
    pairs = _pairs(anhir_root, tmp_path)
    with pytest.raises(ValueError, match="--outdir"):
        warp.warp_all("tiled", pairs, tmp_path / "x")
    with pytest.raises(ValueError, match="--baseline-root"):
        warp.warp_all("bunwarpj", pairs, tmp_path / "x")
    with pytest.raises(ValueError, match="unknown method"):
        warp.warp_all("elastix", pairs, tmp_path / "x")


def test_cli_bunwarpj_leg_runs_end_to_end(anhir_root, tmp_path):
    pairs_csv = tmp_path / "pairs.csv"
    _pairs(anhir_root, tmp_path).to_csv(pairs_csv, index=False)
    rc = warp.main(
        [
            "--method",
            "bunwarpj",
            "--pairs",
            str(pairs_csv),
            "--baseline-root",
            str(anhir_root["baseline"]),
            "--out",
            str(tmp_path / "bw"),
        ]
    )
    assert rc == 0
    idx = pd.read_csv(tmp_path / "bw" / "warp_index.csv")
    assert idx["time_min"].tolist() == pytest.approx([2.0, 2.0, 2.0])
