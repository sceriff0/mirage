import csv
import json
import subprocess
import sys

from merge_seg_eval import flatten


def test_merge_two(tmp_path):
    for pid, qs in [("p1", 0.5), ("p2", 0.7)]:
        (tmp_path / f"{pid}.json").write_text(
            json.dumps(
                {
                    "id": pid,
                    "QualityScore": qs,
                    "downsample_factor": 2,
                    "effective_pixel_size_um": 0.65,
                    "metrics": {
                        "Matched Cell": {"NumberOfCellsPer100SquareMicrons": 1.0}
                    },
                }
            )
        )
    out = tmp_path / "segmentation_metrics.csv"
    subprocess.run(
        [
            sys.executable,
            "bin/merge_seg_eval.py",
            "--inputs",
            str(tmp_path / "p1.json"),
            str(tmp_path / "p2.json"),
            "--out",
            str(out),
        ],
        check=True,
    )
    rows = list(csv.DictReader(out.open()))
    assert {r["id"] for r in rows} == {"p1", "p2"}
    assert any(r["QualityScore"] == "0.7" for r in rows)
    # Drift-visibility columns must round-trip through the merged CSV.
    assert "downsample_factor" in rows[0]
    assert "effective_pixel_size_um" in rows[0]
    assert any(r["downsample_factor"] == "2" for r in rows)
    assert any(r["effective_pixel_size_um"] == "0.65" for r in rows)


def test_flatten_carries_downsample_fields():
    doc = {
        "id": "p1",
        "QualityScore": 0.9,
        "downsample_factor": 4,
        "effective_pixel_size_um": 1.3,
        "metrics": {},
    }
    row = flatten(doc)
    assert row["downsample_factor"] == 4
    assert row["effective_pixel_size_um"] == 1.3


def test_flatten_defaults_gracefully_on_older_format():
    # Per-patient JSONs without the drift fields must not crash flatten().
    doc = {"id": "p1", "QualityScore": 0.5, "metrics": {}}
    row = flatten(doc)
    assert row["downsample_factor"] is None
    assert row["effective_pixel_size_um"] is None
