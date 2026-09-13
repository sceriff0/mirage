"""reg_mosaic: arm directories in, exactly --rows rows out, from the arms' own QC alone.

A miniature arm root: each arm carries csv/registered.csv and, per moving
slide, the pipeline's two-panel QC composite (<stem>_QC_RGB_fullres.tif: Before
| blue separator | After, red = moving, green = reference), its preview, a
*_seg_qc.json and a *_reg_residuals.csv -- exactly what GENERATE_REGISTRATION_QC
and WARP_SEG_QC publish (and what run_ashlar_arm.sh now writes too). armA is
perfectly registered, armB is 3 px off; the numbers in the cells must be the
scorer's, not something the tool recomputed.
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
GAP = 8


def _tissue(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """uint16 nuclear plane (tissue disc + nuclei) and the nucleus centres."""
    yy, xx = np.indices((SIZE, SIZE))
    disc = np.hypot(yy - SIZE / 2, xx - SIZE / 2) < SIZE * 0.42
    img = np.where(disc, 300.0, 20.0)
    centres = []
    while len(centres) < 700:
        y, x = rng.integers(8, SIZE - 8, 2)
        if disc[y, x]:
            r = rng.integers(3, 6)
            img[np.hypot(yy - y, xx - x) < r] = 2000.0 + rng.normal(0, 100)
            centres.append((int(y), int(x)))
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16), centres


def _u8(plane: np.ndarray) -> np.ndarray:
    p = plane.astype(np.float32)
    return np.round((p - p.min()) / (p.max() - p.min()) * 255).astype(np.uint8)


def _composite(ref, native, registered) -> np.ndarray:
    """(3, H, 2W+GAP) uint8 exactly as bin/utils/qc.py renders it."""

    def panel(mov):
        out = np.zeros((3, SIZE, SIZE), np.uint8)
        out[0], out[1] = _u8(mov), _u8(ref)
        return out

    sep = np.zeros((3, SIZE, GAP), np.uint8)
    sep[2] = 255
    return np.concatenate([panel(native), sep, panel(registered)], axis=2)


def _write_qc(qc_dir: Path, stem: str, comp: np.ndarray) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    kw = {"resolution": (1 / PX, 1 / PX), "resolutionunit": "MICROMETER"}
    tifffile.imwrite(
        str(qc_dir / f"{stem}_QC_RGB_fullres.tif"),
        comp,
        imagej=True,
        metadata={"axes": "CYX", "unit": "um"},
        compression="zlib",
        **kw,
    )
    preview = comp[:, ::4, :]
    preview = np.concatenate(
        [
            preview[:, :, : SIZE // 4],
            preview[:, :, SIZE : SIZE + GAP],
            preview[:, :, SIZE + GAP :: 4],
        ],
        axis=2,
    )
    tifffile.imwrite(
        str(qc_dir / f"{stem}_QC_RGB.tif"),
        preview,
        imagej=True,
        metadata={"axes": "CYX"},
    )


def _write_seg_qc(
    qc_dir: Path,
    prefix: str,
    moving_name: str,
    dice: float,
    disp_px: float,
    centres,
    residual_px: float,
) -> None:
    stages = {
        "rigid": {
            "dice_matched": dice * 0.9,
            "displacement_px_p50": disp_px + 2,
            "displacement_um_p50": (disp_px + 2) * PX,
            "n_pairs": len(centres),
        },
        "refined": {
            "dice_matched": dice,
            "displacement_px_p50": disp_px,
            "displacement_um_p50": disp_px * PX,
            "n_pairs": len(centres),
        },
    }
    (qc_dir / f"{prefix}_seg_qc.json").write_text(
        json.dumps(
            {
                "patient_id": "P1",
                "moving": moving_name,
                "stage_order": ["native", "rigid", "refined"],
                "stages": stages,
            }
        )
    )
    with open(qc_dir / f"{prefix}_reg_residuals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["moving", "ref_x", "ref_y", "residual_px", "stage"])
        for y, x in centres:
            w.writerow([moving_name, x, y, residual_px + 3, "rigid"])
            w.writerow([moving_name, x, y, residual_px, "refined"])


def _checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "patient_id",
                "id",
                "registered_image",
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
    ref, centres = _tissue(rng)
    native = np.roll(
        np.roll(ref, 12, axis=0), -9, axis=1
    )  # as acquired: 12 px down, 9 px left
    ref_path = root / "preprocess_shared" / "P1" / "preprocessed" / "P1_ref.ome.tif"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_bytes(
        b""
    )  # never read by the tool: the checkpoint names it, the QC composite carries the pixels
    # armA: STARE-style names (channels joined), perfect; armB: VALIS-style names (file stem), 3 px off
    for arm, shift, dice, naming in (
        ("armA", 0, 0.92, "stare"),
        ("armB", 3, 0.74, "valis"),
    ):
        d = root / arm
        registered = np.roll(ref, shift, axis=1)
        rows = [
            {
                "patient_id": "P1",
                "id": "P1_ref",
                "registered_image": str(ref_path),
                "is_reference": "true",
                "channels": "DAPI|PANCK",
                "pixel_size": PX,
            }
        ]
        for stem_base, ch in (("cd3", "DAPI|CD3"), ("cd8", "DAPI|CD8")):
            reg_name = f"P1_{stem_base}_registered.ome.tiff"
            reg_path = d / "P1" / "registered" / "registered_slides" / reg_name
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            reg_path.write_bytes(b"")
            rows.append(
                {
                    "patient_id": "P1",
                    "id": f"P1_{stem_base}_registered",
                    "registered_image": str(reg_path),
                    "is_reference": "false",
                    "channels": ch,
                    "pixel_size": PX,
                }
            )
            qc_dir = d / "P1" / "qc" / "registration"
            _write_qc(
                qc_dir,
                f"P1_{stem_base}_registered",
                _composite(ref, native, registered),
            )
            moving_name = (
                ch.replace("|", "_")
                if naming == "stare"
                else f"P1_{stem_base}_registered"
            )
            prefix = f"P1_{moving_name}" if naming == "stare" else moving_name
            _write_seg_qc(
                qc_dir, prefix, moving_name, dice, float(shift), centres, float(shift)
            )
        _checkpoint(d / "csv" / "registered.csv", rows)
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


# --- the pieces -------------------------------------------------------------------
def test_read_checkpoint_keys_rounds_by_their_stains(arm_root):
    per = rm.read_checkpoint(arm_root / "armA" / "csv" / "registered.csv")
    keys = {s.key: s for s in per["P1"]}
    assert set(keys) == {"PANCK", "CD3", "CD8"}
    assert keys["PANCK"].is_reference and keys["CD3"].stem == "P1_cd3_registered"
    assert {"P1_cd3_registered", "P1_cd3", "DAPI_CD3", "P1_DAPI_CD3"} <= keys[
        "CD3"
    ].names


def test_composite_splits_on_the_blue_separator(arm_root):
    c = rm.Composite(
        arm_root
        / "armA"
        / "P1"
        / "qc"
        / "registration"
        / "P1_cd3_registered_QC_RGB_fullres.tif"
    )
    assert (
        c.has_before
        and c.before_width == SIZE
        and c.after_offset == SIZE + GAP
        and c.after_width == SIZE
    )
    assert c.canvas == (SIZE, SIZE) and c.px == pytest.approx(PX)
    ref_b, mov_b = c.crop("before", 100, 100, 32, 32)
    ref_a, mov_a = c.crop("after", 100, 100, 32, 32)
    np.testing.assert_array_equal(
        ref_b, ref_a
    )  # the reference is the same in both panels
    assert not np.array_equal(mov_b, mov_a)  # the moving slide is not
    c.close()


def test_seg_qc_is_matched_under_both_naming_conventions(arm_root):
    for arm in ("armA", "armB"):
        a = rm.Arm(arm_root / arm)
        a.open("P1")
        qc = a.seg_qc("CD3")
        assert qc is not None and qc.stage == "refined"
        assert qc.dice == pytest.approx(0.92 if arm == "armA" else 0.74)
        assert qc.residuals.shape[0] == 700  # only the final stage's rows
        local, n = qc.local_displacement_px(100, 100, 80, 80, 5)
        assert n > 5 and local == pytest.approx(0.0 if arm == "armA" else 3.0)
        assert qc.local_displacement_px(0, 0, 4, 4, 5) == (None, 0)


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
        "--formats",
        "png",
        "--dpi",
        "50",
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
    assert [(r["round"], r["roi"]) for r in m["row_plan"]] == [
        ("CD3", 1),
        ("CD8", 1),
        ("CD3", 2),
    ]
    assert list(m["columns"]) == ["Before", "armA", "armB"]
    assert (
        m["columns"]["Before"]["panel"] == "before"
        and m["columns"]["armA"]["panel"] == "after"
    )
    assert len(list((out / "P1_patches").glob("*.png"))) == 9
    assert m["pixel_size_um"] == pytest.approx(PX) and m["patch_px"] == 64


def test_cells_carry_the_scorers_numbers_not_recomputed_ones(arm_root, tmp_path):
    m = _run(arm_root, tmp_path / "figs")
    for row in m["row_plan"]:
        a, b = row["cells"]["armA"], row["cells"]["armB"]
        assert a["dice_matched"] == pytest.approx(0.92) and b[
            "dice_matched"
        ] == pytest.approx(0.74)
        assert a["stage"] == "refined"
        assert a["n_nuclei_in_roi"] >= 5
        assert a["roi_displacement_um"] == pytest.approx(0.0)
        assert b["roi_displacement_um"] == pytest.approx(3.0 * PX)
        assert "dice_matched" not in row["cells"]["Before"]


def test_a_sparse_roi_falls_back_to_the_slide_level_displacement(arm_root, tmp_path):
    m = _run(
        arm_root,
        tmp_path / "figs",
        "--roi",
        "2,2",
        "--rows",
        "2",
        "--min-nuclei",
        "5000",
    )
    cell = m["row_plan"][0]["cells"]["armB"]
    assert "roi_displacement_um" not in cell and cell[
        "slide_displacement_um"
    ] == pytest.approx(1.5)


def test_labels_rename_columns_and_duplicates_are_refused(arm_root, tmp_path):
    m = _run(
        arm_root,
        tmp_path / "figs",
        "--label",
        "armA=STARE robust",
        "--label",
        "armB=VALIS",
    )
    assert list(m["columns"]) == ["Before", "STARE robust", "VALIS"]
    with pytest.raises(SystemExit, match="distinct"):
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


def test_a_missing_composite_or_round_is_named(arm_root, tmp_path):
    d = tmp_path / "armC"
    rows = list(csv.DictReader(open(arm_root / "armA" / "csv" / "registered.csv")))
    _checkpoint(d / "csv" / "registered.csv", rows)  # the checkpoint, but no QC files
    with pytest.raises(SystemExit, match="no registration QC composite"):
        rm.main(
            [
                str(d),
                "--rows",
                "2",
                "-o",
                str(tmp_path / "x"),
                "--patch-px",
                "64",
                "--formats",
                "png",
                "--dpi",
                "50",
            ]
        )
    _checkpoint(
        d / "csv" / "registered.csv", [r for r in rows if "CD8" not in r["channels"]]
    )
    with pytest.raises(SystemExit, match="no registered slide for round 'CD8'"):
        rm.main(
            [
                str(arm_root / "armA"),
                str(d),
                "--rows",
                "2",
                "-o",
                str(tmp_path / "x"),
                "--patch-px",
                "64",
                "--formats",
                "png",
                "--dpi",
                "50",
            ]
        )
