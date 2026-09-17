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
        "native": {
            "dice_matched": dice * 0.5,
            "displacement_px_p50": disp_px + 9,
            "displacement_um_p50": (disp_px + 9) * PX,
            "n_pairs": len(centres),
        },
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
        # Before is the first arm's composite, so it carries the first arm's NATIVE stage
        assert row["cells"]["Before"]["dice_matched"] == pytest.approx(0.46)
        assert row["cells"]["Before"]["stage"] == "native"


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


def test_a_pipeline_arm_is_read_from_the_layout_the_pipeline_publishes(
    arm_root, tmp_path
):
    """A NEXTFLOW arm does not publish what run_ashlar_arm.sh writes. conf/modules.config's
    GENERATE_REGISTRATION_QC publishDir pattern is "qc/*_QC_RGB_fullres.tif", and a
    pattern keeps its relative path, so the composite lands in
    <patient>/qc/registration/qc/ -- one level below the seg QC json. WARP_SEG_QC's
    publishDir names *_seg_qc.json only, so *_reg_residuals.csv is not published at all.
    Measured 2026-09-16 on a stub run of `--start registration --stop registration`.
    Before this, every VALIS/STARE column died on "no registration QC composite"; only
    the ASHLAR arm, which writes flat, could be drawn."""
    import shutil

    pipe = tmp_path / "pipe"
    shutil.copytree(arm_root / "armB", pipe)
    qc = pipe / "P1" / "qc" / "registration"
    (qc / "qc").mkdir()
    for f in qc.glob("*_QC_RGB*"):
        f.rename(qc / "qc" / f.name)
    for f in qc.glob("*_reg_residuals.csv"):
        f.unlink()

    out = tmp_path / "figs"
    argv = [str(arm_root / "armA"), str(pipe), "--rows", "3", "-o", str(out)]
    argv += ["--patch-px", "64", "--formats", "png", "--dpi", "50"]
    assert rm.main(argv) == 0
    m = json.loads((out / "P1_rois.json").read_text())
    cell = m["row_plan"][0]["cells"]["pipe"]
    assert cell["dice_matched"] == pytest.approx(0.74)
    # no residuals published -> the slide-level number, flagged, never a fabricated local one
    assert "roi_displacement_um" not in cell
    assert cell["slide_displacement_um"] == pytest.approx(1.5)
    assert "/qc/registration/qc/" in m["columns"]["pipe"]["files"]["CD3"]


def test_every_cell_reads_dice_equals_and_the_scale_bar_sits_in_the_top_left_cell(
    arm_root, tmp_path, monkeypatch
):
    """Each cell is labelled `Dice = X` -- the Before cell with the native stage -- and the
    scale bar is drawn once, in row 0 / column 0, the convention of an IF figure panel."""
    drawn = {"notes": None, "bars": []}
    real_assemble = rm.assemble_figure

    def spy_assemble(grid, notes, *a, **k):
        drawn["notes"] = notes
        return real_assemble(grid, notes, *a, **k)

    def spy_bar(ax, h, w, bar_px, label, font, **k):
        drawn["bars"].append(
            (
                ax.get_subplotspec().rowspan.start,
                ax.get_subplotspec().colspan.start,
                label,
                bar_px,
            )
        )

    monkeypatch.setattr(rm, "assemble_figure", spy_assemble)
    monkeypatch.setattr(rm, "draw_scalebar", spy_bar)
    # the locator draws its own bar with the same helper; this test is about the mosaic
    monkeypatch.setattr(rm, "save_locator", lambda *a, **k: None)
    _run(arm_root, tmp_path / "figs", "--orient", "rounds-as-rows")
    first_row = drawn["notes"][0]
    assert first_row[0].startswith("Dice = 0.46")  # Before: native stage
    assert first_row[1].startswith("Dice = 0.92") and first_row[2].startswith(
        "Dice = 0.74"
    )
    assert "Δ = " in first_row[1]
    assert len(drawn["bars"]) == 1
    row, col, label, bar_px = drawn["bars"][0]
    assert (row, col) == (0, 0) and label.endswith("µm")
    # the bar is as long as it says: label µm / pixel size, in the crop's own pixels
    assert bar_px == pytest.approx(float(label.split()[0]) / PX)


# --- numbers without WARP_SEG_QC (reg_qc=1) ----------------------------------------
def test_image_metrics_read_alignment_off_the_pixels():
    rng = np.random.default_rng(7)
    ref, _ = _tissue(rng)
    ref = _u8(ref)[60:188, 60:188]
    dice, shift = rm.image_metrics(ref, ref)
    assert dice == pytest.approx(1.0) and shift == pytest.approx(0.0, abs=0.05)
    moved = np.roll(np.roll(ref, 3, axis=0), 4, axis=1)
    dice_off, shift_off = rm.image_metrics(ref, moved)
    assert shift_off == pytest.approx(5.0, abs=0.3)  # sqrt(3^2 + 4^2)
    assert dice_off < dice


def test_numbers_image_computes_every_cell_from_the_crop(arm_root, tmp_path):
    m = _run(arm_root, tmp_path / "figs", "--numbers", "image")
    assert m["number_sources"] == ["image"] and m["palette"] == "magenta-cyan"
    for row in m["row_plan"]:
        before, a, b = (row["cells"][c] for c in ("Before", "armA", "armB"))
        assert {before["source"], a["source"], b["source"]} == {"image"}
        assert a["shift_px"] == pytest.approx(0.0, abs=0.2)  # armA is perfect
        assert b["shift_px"] == pytest.approx(3.0, abs=0.3)  # armB is 3 px off
        # native: 12 down, 9 left = 15 px; np.roll wraps nuclei across the crop border,
        # which costs the estimate a little
        assert before["shift_px"] == pytest.approx(15.0, abs=1.0)
        assert a["dice_pixel"] > b["dice_pixel"] > before["dice_pixel"]


def test_auto_falls_back_to_the_image_when_warp_seg_qc_did_not_run(arm_root, tmp_path):
    import shutil

    for arm in ("armA", "armB"):
        shutil.copytree(arm_root / arm, tmp_path / arm)
        for f in (tmp_path / arm).rglob("*_seg_qc.json"):
            f.unlink()
    m = _run(tmp_path, tmp_path / "figs")  # --numbers auto is the default
    assert m["number_sources"] == ["image"]
    assert m["row_plan"][0]["cells"]["armB"]["shift_um"] == pytest.approx(1.5, abs=0.15)


def test_select_rois_keeps_clear_of_excluded_boxes():
    rng = np.random.default_rng(5)
    low, _ = _tissue(rng)
    first = rm.select_rois(low, 1.0, 48, 1, 0.15, 1.0, low.shape)
    (y0, x0), size = first[0], 48
    again = rm.select_rois(
        low, 1.0, 48, 3, 0.15, 1.0, low.shape, exclude=[(y0, x0, size)]
    )
    for y, x in again:
        assert abs(y - y0) >= size or abs(x - x0) >= size


def test_rounds_are_columns_by_default_and_methods_read_down_each_column(
    arm_root, tmp_path, monkeypatch
):
    seen = {}
    real = rm.assemble_figure

    def spy(grid, notes, row_labels, col_labels, *a, **k):
        seen.update(notes=notes, rows=row_labels, cols=col_labels, bar=a[4], where=a[5])
        return real(grid, notes, row_labels, col_labels, *a, **k)

    monkeypatch.setattr(rm, "assemble_figure", spy)
    m = _run(arm_root, tmp_path / "figs")
    assert m["orient"] == "rounds-as-columns"
    assert seen["rows"] == ["Before", "armA", "armB"]
    assert len(seen["cols"]) == 3 and "CD3" in seen["cols"][0]
    column0 = [seen["notes"][r][0] for r in range(3)]
    assert column0[0].startswith("Dice = 0.46")  # Before
    assert column0[1].startswith("Dice = 0.92") and column0[2].startswith("Dice = 0.74")
    assert seen["where"] == "first"  # the bar still sits in the top-left cell


def test_numbers_none_leaves_every_cell_blank(arm_root, tmp_path, monkeypatch):
    """The fast prototype prints no Dice/Δ at all: the numbers come later, from reg_qc=2."""
    seen = {}
    real = rm.assemble_figure

    def spy(grid, notes, *a, **k):
        seen.update(notes=notes, footer=k.get("footer", ""))
        return real(grid, notes, *a, **k)

    monkeypatch.setattr(rm, "assemble_figure", spy)
    m = _run(arm_root, tmp_path / "figs", "--numbers", "none")
    assert all(note == "" for row in seen["notes"] for note in row)
    assert m["number_sources"] == []
    assert "Dice" not in seen["footer"] and "magenta" in seen["footer"]
    cells = m["row_plan"][0]["cells"]
    assert all(
        "dice_matched" not in c and "dice_pixel" not in c for c in cells.values()
    )


# --- pixels from the original 16-bit slides, not the 8-bit composite ---------------
def _ome(path: Path, planes, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.stack(planes).astype(np.uint16),
        ome=True,
        tile=(128, 128),
        compression="zlib",
        metadata={
            "axes": "CYX",
            "Channel": {"Name": names},
            "PhysicalSizeX": PX,
            "PhysicalSizeY": PX,
        },
    )


@pytest.fixture(scope="module")
def originals_root(tmp_path_factory):
    """One run as a Nextflow arm publishes it, WITH the slides behind its QC composite.

    The registered slide carries a few saturated pixels (65535) the way a resampled slide
    can. The composite is built from it by the pipeline's own rule -- min-max over the WHOLE
    plane to uint8 (bin/utils/qc.py autoscale_for_display) -- which puts all real tissue in
    2-3 grey levels: the flat, speckled mosaic cells seen on a real run, 2026-09-17.
    """
    root = tmp_path_factory.mktemp("orig")
    rng = np.random.default_rng(11)
    ref16, _ = _tissue(rng)
    ref16 = (ref16 * 0.25).astype(np.uint16) + 100  # DAPI far below 16-bit full scale
    native16 = np.roll(np.roll(ref16, 12, axis=0), -9, axis=1)
    reg16 = np.roll(ref16, 3, axis=1)
    reg16[5:8, 5:8] = 65535  # a few extreme pixels
    other = (rng.random(ref16.shape) * 500).astype(np.uint16)

    pre = root / "preprocess_shared" / "P1" / "preprocessed"
    _ome(pre / "P1_ref.ome.tif", [ref16, other], ["DAPI", "PANCK"])
    _ome(pre / "P1_cd3.ome.tif", [other, native16], ["CD3", "DAPI"])  # DAPI not first
    _csv(
        root / "preprocess_shared" / "csv" / "preprocessed.csv",
        [
            {
                "patient_id": "P1",
                "id": "P1_ref",
                "preprocessed_image": str(pre / "P1_ref.ome.tif"),
                "is_reference": "true",
                "channels": "DAPI|PANCK",
                "pixel_size": PX,
            },
            {
                "patient_id": "P1",
                "id": "P1_cd3",
                "preprocessed_image": str(pre / "P1_cd3.ome.tif"),
                "is_reference": "false",
                "channels": "DAPI|CD3",
                "pixel_size": PX,
            },
        ],
    )
    arm = root / "armR"
    reg = arm / "P1" / "registered" / "registered_slides"
    _ome(reg / "P1_ref_registered.ome.tiff", [ref16, other], ["DAPI", "PANCK"])
    _ome(reg / "P1_cd3_registered.ome.tiff", [reg16, other], ["DAPI", "CD3"])

    def minmax(a):
        a = a.astype(np.float64)
        return np.round((a - a.min()) * 255 / (a.max() - a.min())).astype(np.uint8)

    qc = arm / "P1" / "qc" / "registration" / "qc"  # where a Nextflow arm publishes it
    sep = np.zeros((3, SIZE, GAP), np.uint8)
    sep[2] = 255
    before = np.stack([minmax(native16), minmax(ref16), np.zeros_like(ref16, np.uint8)])
    after = np.stack([minmax(reg16), minmax(ref16), np.zeros_like(ref16, np.uint8)])
    _write_qc(qc, "P1_cd3_registered", np.concatenate([before, sep, after], axis=2))
    _checkpoint(
        arm / "csv" / "registered.csv",
        [
            {
                "patient_id": "P1",
                "id": "P1_ref",
                "registered_image": str(reg / "P1_ref_registered.ome.tiff"),
                "is_reference": "true",
                "channels": "DAPI|PANCK",
                "pixel_size": PX,
            },
            {
                "patient_id": "P1",
                "id": "P1_cd3_registered",
                "registered_image": str(reg / "P1_cd3_registered.ome.tiff"),
                "is_reference": "false",
                "channels": "DAPI|CD3",
                "pixel_size": PX,
            },
        ],
    )
    return root


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _mosaic(arm_dir, out, *extra):
    argv = [
        str(arm_dir),
        "--rows",
        "1",
        "-o",
        str(out),
        "--patch-px",
        "96",
        "--formats",
        "png",
        "--dpi",
        "50",
        "--numbers",
        "image",
        *extra,
    ]
    assert rm.main(argv) == 0
    return json.loads((out / "P1_rois.json").read_text())


def test_the_8bit_composite_crushes_a_slide_with_outliers_the_originals_do_not(
    originals_root, tmp_path
):
    crushed = _mosaic(originals_root / "armR", tmp_path / "c", "--source", "composite")
    cell = crushed["row_plan"][0]["cells"]["armR"]
    assert (
        cell["pixels"] == "composite" and cell["mov_limits"][1] <= 3
    )  # the bug, reproduced

    good = _mosaic(originals_root / "armR", tmp_path / "o")  # --source auto
    after, before = (
        good["row_plan"][0]["cells"]["armR"],
        good["row_plan"][0]["cells"]["Before"],
    )
    assert after["pixels"] == before["pixels"] == "originals"
    assert (
        after["mov_limits"][1]
        > 100 * crushed["row_plan"][0]["cells"]["armR"]["mov_limits"][1]
    )
    assert after["shift_px"] == pytest.approx(3.0, abs=0.3)  # the registered slide
    assert before["shift_px"] == pytest.approx(
        15.0, abs=1.0
    )  # the NATIVE slide, DAPI found by name


def test_without_the_native_slides_it_falls_back_to_the_composite(
    originals_root, tmp_path, caplog
):
    import shutil

    shutil.copytree(
        originals_root / "armR", tmp_path / "armR"
    )  # no sibling preprocess_shared/
    with caplog.at_level("WARNING"):
        m = _mosaic(tmp_path / "armR", tmp_path / "out")
    assert m["row_plan"][0]["cells"]["armR"]["pixels"] == "composite"
    assert "original slides unusable" in caplog.text


def test_each_cell_is_drawn_at_one_image_pixel_per_output_pixel(
    originals_root, tmp_path, monkeypatch
):
    seen = {}
    real = rm.assemble_figure

    def spy(
        grid, notes, row_labels, col_labels, out_stem, formats, cell_in, dpi, *a, **k
    ):
        seen.update(cell_in=cell_in, dpi=dpi)
        return real(
            grid,
            notes,
            row_labels,
            col_labels,
            out_stem,
            formats,
            cell_in,
            dpi,
            *a,
            **k,
        )

    monkeypatch.setattr(rm, "assemble_figure", spy)
    _mosaic(originals_root / "armR", tmp_path / "o")
    assert seen["cell_in"] * seen["dpi"] == pytest.approx(
        96
    )  # the patch, not resampled


def test_an_arm_without_a_qc_composite_is_drawn_from_its_slides(
    originals_root, tmp_path
):
    """An external arm whose QC-composite step failed still has its stitched slides."""
    import shutil

    shutil.copytree(originals_root, tmp_path / "root")
    ext = tmp_path / "root" / "armX"
    shutil.copytree(tmp_path / "root" / "armR", ext)
    shutil.rmtree(ext / "P1" / "qc")
    argv = [
        str(tmp_path / "root" / "armR"),
        str(ext),
        "--rows",
        "1",
        "-o",
        str(tmp_path / "out"),
        "--patch-px",
        "96",
        "--formats",
        "png",
        "--dpi",
        "50",
        "--numbers",
        "none",
    ]
    assert rm.main(argv) == 0
    cells = json.loads((tmp_path / "out" / "P1_rois.json").read_text())["row_plan"][0][
        "cells"
    ]
    assert cells["armX"]["pixels"] == "originals"


def test_lzw_compressed_slides_are_read(originals_root, tmp_path):
    """The real registered slides are LZW (job 6844142); decoding them needs imagecodecs."""
    pytest.importorskip("imagecodecs")
    import shutil

    shutil.copytree(originals_root, tmp_path / "root")
    reg = (
        tmp_path
        / "root"
        / "armR"
        / "P1"
        / "registered"
        / "registered_slides"
        / "P1_cd3_registered.ome.tiff"
    )
    data = tifffile.imread(str(reg))
    tifffile.imwrite(
        str(reg),
        data,
        ome=True,
        tile=(128, 128),
        compression="lzw",
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}},
    )
    m = _mosaic(tmp_path / "root" / "armR", tmp_path / "out")
    assert m["row_plan"][0]["cells"]["armR"]["pixels"] == "originals"


def test_an_undecodable_slide_falls_back_to_the_composite_instead_of_crashing(
    originals_root, tmp_path, monkeypatch, caplog
):
    def no_codec(self, *a, **k):
        raise ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    monkeypatch.setattr(rm.TiffSource, "read_patch", no_codec)
    monkeypatch.setattr(
        rm.Composite,
        "crop",
        lambda self, panel, y, x, h, w: (np.zeros((h, w), np.uint8),) * 2,
    )
    with caplog.at_level("WARNING"):
        m = _mosaic(originals_root / "armR", tmp_path / "out")
    assert m["row_plan"][0]["cells"]["armR"]["pixels"] == "composite"
    assert "imagecodecs" in caplog.text
