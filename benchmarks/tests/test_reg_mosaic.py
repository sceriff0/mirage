"""reg_mosaic: arm directories in, exactly --rows rows out, from the checkpoints alone.

A miniature arm root is built from synthetic OME-TIFFs: a reference with a
tissue disc and nuclei, a "preprocessed" moving slide that is the same tissue
shifted (the Before column), one arm whose registered slide is perfectly back
on the reference (armA) and one that is 3 px off (armB). No samplesheet, no raw
acquisition: the tool must find everything through csv/registered.csv and
preprocess_shared/csv/preprocessed.csv, exactly as run_arms.sh writes them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks import reg_mosaic as rm

SIZE = 320
PX = 0.5  # µm


def _tissue(rng: np.random.Generator) -> np.ndarray:
    """uint16 nuclear plane: a bright tissue disc with nuclei, dark glass around it."""
    yy, xx = np.indices((SIZE, SIZE))
    disc = np.hypot(yy - SIZE / 2, xx - SIZE / 2) < SIZE * 0.42
    img = np.where(disc, 300.0, 20.0)
    n = 0
    while n < 700:
        y, x = rng.integers(8, SIZE - 8, 2)
        if disc[y, x]:
            r = rng.integers(3, 6)
            img[np.hypot(yy - y, xx - x) < r] = 2000.0 + rng.normal(0, 100)
            n += 1
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _write(path: Path, dapi: np.ndarray, stain: np.ndarray, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.stack([dapi, stain]),
        metadata={"axes": "CYX", "Channel": {"Name": names}},
        tile=(64, 64),
    )


def _checkpoint(path: Path, image_col: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "patient_id",
                "id",
                image_col,
                "is_reference",
                "channels",
                "pixel_size",
            ],
        )
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def arm_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("arms")
    rng = np.random.default_rng(3)
    ref = _tissue(rng)
    stain = (ref // 3).astype(np.uint16)
    # the moving slide as acquired: the same tissue, 12 px down and 9 px left
    moved = np.roll(np.roll(ref, 12, axis=0), -9, axis=1)
    pre = root / "preprocess_shared"
    _write(
        pre / "P1" / "preprocessed" / "P1_ref.ome.tif", ref, stain, ["DAPI", "PANCK"]
    )
    _write(
        pre / "P1" / "preprocessed" / "P1_cd3.ome.tif", moved, stain, ["DAPI", "CD3"]
    )
    _write(
        pre / "P1" / "preprocessed" / "P1_cd8.ome.tif", moved, stain, ["DAPI", "CD8"]
    )
    ref_row = {
        "patient_id": "P1",
        "id": "P1_ref",
        "is_reference": "true",
        "channels": "DAPI|PANCK",
        "pixel_size": PX,
    }
    _checkpoint(
        pre / "csv" / "preprocessed.csv",
        "preprocessed_image",
        [
            {
                **ref_row,
                "preprocessed_image": str(
                    pre / "P1" / "preprocessed" / "P1_ref.ome.tif"
                ),
            },
            {
                "patient_id": "P1",
                "id": "P1_cd3",
                "preprocessed_image": str(
                    pre / "P1" / "preprocessed" / "P1_cd3.ome.tif"
                ),
                "is_reference": "false",
                "channels": "DAPI|CD3",
                "pixel_size": PX,
            },
            {
                "patient_id": "P1",
                "id": "P1_cd8",
                "preprocessed_image": str(
                    pre / "P1" / "preprocessed" / "P1_cd8.ome.tif"
                ),
                "is_reference": "false",
                "channels": "DAPI|CD8",
                "pixel_size": PX,
            },
        ],
    )
    for arm, shift in (("armA", 0), ("armB", 3)):
        d = root / arm
        reg = np.roll(ref, shift, axis=1)
        rows = [
            {
                **ref_row,
                "registered_image": str(pre / "P1" / "preprocessed" / "P1_ref.ome.tif"),
            }
        ]
        for stem, ch in (("cd3", "DAPI|CD3"), ("cd8", "DAPI|CD8")):
            p = (
                d
                / "P1"
                / "registered"
                / "registered_slides"
                / f"P1_{stem}_registered.ome.tif"
            )
            _write(p, reg, stain, ch.split("|"))
            rows.append(
                {
                    "patient_id": "P1",
                    "id": f"P1_{stem}_registered",
                    "registered_image": str(p),
                    "is_reference": "false",
                    "channels": ch,
                    "pixel_size": PX,
                }
            )
        _checkpoint(d / "csv" / "registered.csv", "registered_image", rows)
    return root


# --- rows -----------------------------------------------------------------------
def test_plan_rows_is_roi_major_and_exact():
    assert rm.plan_rows(["a", "b", "c"], 2, 4) == [
        ("a", 0),
        ("b", 0),
        ("c", 0),
        ("a", 1),
    ]
    assert rm.plan_rows(["a"], 3, 3) == [("a", 0), ("a", 1), ("a", 2)]
    with pytest.raises(ValueError, match="fewer than --rows"):
        rm.plan_rows(["a", "b"], 1, 3)
    with pytest.raises(ValueError, match=">= 1"):
        rm.plan_rows(["a"], 1, 0)


# --- checkpoints ----------------------------------------------------------------
def test_read_checkpoint_keys_rounds_by_their_stains(arm_root):
    per = rm.read_checkpoint(
        arm_root / "armA" / "csv" / "registered.csv", "registered_image"
    )
    assert list(per) == ["P1"]
    keys = {s.key: s for s in per["P1"]}
    assert set(keys) == {"PANCK", "CD3", "CD8"}
    assert keys["PANCK"].is_reference and keys["PANCK"].pixel_size == PX
    assert keys["CD3"].image.name == "P1_cd3_registered.ome.tif"


def test_read_checkpoint_names_what_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        rm.read_checkpoint(tmp_path / "csv" / "registered.csv", "registered_image")
    p = tmp_path / "x.csv"
    p.write_text("patient_id,id\nP1,a\n")
    with pytest.raises(SystemExit, match="missing column"):
        rm.read_checkpoint(p, "registered_image")


def test_find_before_looks_beside_the_arms(arm_root, tmp_path):
    assert rm.find_before([arm_root / "armA"], None) == arm_root / "preprocess_shared"
    assert rm.find_before([arm_root / "armA"], tmp_path) == tmp_path
    with pytest.raises(SystemExit, match="pass --before"):
        rm.find_before([tmp_path / "lonely_arm"], None)


# --- end to end -----------------------------------------------------------------
def _run(arm_root, out, *extra):
    argv = [
        str(arm_root / "armA"),
        str(arm_root / "armB"),
        "--rows",
        "3",
        "-o",
        str(out),
        "--patch-px",
        "64",
        "--lowres-um",
        "2",
        "--formats",
        "png",
        "--dpi",
        "50",
        "--annotate",
        "dice,shift",
        *extra,
    ]
    assert rm.main(argv) == 0
    return json.loads((out / "P1_rois.json").read_text())


def test_mosaic_has_exactly_the_requested_rows_and_one_column_per_arm(
    arm_root, tmp_path
):
    out = tmp_path / "figs"
    m = _run(arm_root, out)
    assert (out / "P1_mosaic.png").exists() and (out / "P1_locator.png").exists()
    assert m["rows"] == 3 and len(m["row_plan"]) == 3
    # ROI-major over the two rounds: CD3, CD8 at ROI 1, then CD3 at ROI 2
    assert [(r["round"], r["roi"]) for r in m["row_plan"]] == [
        ("CD3", 1),
        ("CD8", 1),
        ("CD3", 2),
    ]
    assert list(m["columns"]) == ["Before", "armA", "armB"]
    pngs = list((out / "P1_patches").glob("*.png"))
    assert len(pngs) == 3 * 3  # rows x columns, overlay only
    assert m["pixel_size_um"] == PX and m["patch_px"] == 64


def test_the_perfect_arm_beats_before_and_the_offset_arm_on_dice(arm_root, tmp_path):
    m = _run(arm_root, tmp_path / "figs")
    for row in m["row_plan"]:
        cells = row["cells"]
        assert cells["armA"]["dice"] > 0.95, cells
        assert (
            cells["armA"]["dice"] > cells["armB"]["dice"] > cells["Before"]["dice"]
        ), cells
        # the residual shift the metric recovers matches what the fixture injected
        dx, dy = cells["armB"]["shift_px"]
        assert abs(abs(dx) - 3) < 0.6 and abs(dy) < 0.6, cells["armB"]


def test_labels_rename_columns_and_duplicates_are_refused(arm_root, tmp_path):
    m = _run(
        arm_root,
        tmp_path / "figs",
        "--label",
        "armA=VALIS best",
        "--label",
        "armB=STARE best",
    )
    assert list(m["columns"]) == ["Before", "VALIS best", "STARE best"]
    with pytest.raises(SystemExit, match="share a title"):
        rm.main(
            [
                str(arm_root / "armA"),
                str(arm_root / "armB"),
                "--rows",
                "1",
                "-o",
                str(tmp_path / "x"),
                "--label",
                "armB=armA",
                "--patch-px",
                "64",
            ]
        )


def test_rows_beyond_the_fixed_rois_is_an_error_not_a_shorter_figure(
    arm_root, tmp_path
):
    with pytest.raises(ValueError, match="fewer than --rows"):
        rm.main(
            [
                str(arm_root / "armA"),
                "--rows",
                "5",
                "-o",
                str(tmp_path / "x"),
                "--patch-px",
                "64",
                "--roi",
                "100,100",
                "--formats",
                "png",
                "--dpi",
                "50",
            ]
        )


def test_rounds_filter_and_reused_rois(arm_root, tmp_path):
    first = _run(arm_root, tmp_path / "a", "--rounds", "CD8")
    assert {r["round"] for r in first["row_plan"]} == {"CD8"} and len(
        first["row_plan"]
    ) == 3
    second = _run(
        arm_root, tmp_path / "b", "--rois-json", str(tmp_path / "a" / "P1_rois.json")
    )
    assert second["rois"] == first["rois"]


def test_a_missing_round_in_an_arm_is_named(arm_root, tmp_path):
    d = tmp_path / "armC"
    rows = list(csv.DictReader(open(arm_root / "armA" / "csv" / "registered.csv")))
    _checkpoint(
        d / "csv" / "registered.csv",
        "registered_image",
        [r for r in rows if "CD8" not in r["channels"]],
    )
    with pytest.raises(
        SystemExit, match=r"no registered slide for round\(s\) \['CD8'\]"
    ):
        rm.main(
            [
                str(d),
                "--rows",
                "2",
                "-o",
                str(tmp_path / "x"),
                "--patch-px",
                "64",
                "--before",
                str(arm_root / "preprocess_shared"),
                "--formats",
                "png",
                "--dpi",
                "50",
            ]
        )
