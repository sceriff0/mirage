"""reg_zoom: whole-slide DAPI overview + a zoom with the pipeline's segmented cells outlined."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks import reg_mosaic as rm
from benchmarks import reg_zoom as rz
from benchmarks.tests import test_reg_mosaic as tm


@pytest.fixture(scope="module")
def seg_run(tmp_path_factory):
    root = tm.originals_root.__wrapped__(tmp_path_factory)
    arm = root / "armR"
    reg = arm / "P1" / "registered" / "registered_slides" / "P1_ref_registered.ome.tiff"
    labels = np.zeros((tm.SIZE, tm.SIZE), np.uint32)
    for i, (cy, cx) in enumerate([(40, 40), (40, 56), (100, 150), (200, 200)], start=1):
        labels[cy - 8 : cy + 8, cx - 8 : cx + 8] = i  # two touching cells, two apart
    mask = arm / "P1" / "segmentation" / "P1_cell_mask.tif"
    mask.parent.mkdir(parents=True)
    tifffile.imwrite(str(mask), labels, compression="zlib")
    rows = [
        {
            "patient_id": "P1",
            "id": "P1_ref",
            "registered_image": str(reg),
            "is_reference": "true",
            "channels": "DAPI|PANCK",
            "cell_mask": str(mask),
            "nuclei_mask": str(mask),
            "contours": "",
            "nucleus_contours": "",
            "pixel_size": tm.PX,
        },
    ]
    with open(arm / "csv" / "segmented.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return arm, labels


def _zoom(arm, out, *extra, monkeypatch=None):
    argv = [
        str(arm),
        "-o",
        str(out),
        "--field-um",
        "60",
        "--roi",
        "20,20",
        "--overview-px",
        "160",
        "--formats",
        "png",
        "--dpi",
        "50",
        *extra,
    ]
    assert rz.main(argv) == 0
    return json.loads((out / "P1_zoom.json").read_text())


def test_outlines_are_inside_each_cell_and_neighbours_stay_apart(seg_run):
    _, labels = seg_run
    one = rz.outlines(labels[20:80, 20:80], 1)
    # cell 1 spans rows/cols 12..27 of this crop, cell 2 cols 28..43 (touching at 27|28)
    assert one[12, 17] and not one[20, 20]  # top edge of cell 1 on, its centre off
    assert one[20, 27] and one[20, 28]  # both sides of the shared border are drawn
    thick = rz.outlines(labels[20:80, 20:80], 3)
    assert thick.sum() > one.sum() and not thick[labels[20:80, 20:80] == 0].any()


def test_zoom_figure_draws_cells_in_the_chosen_colour_with_bars_and_legend(
    seg_run, tmp_path, monkeypatch
):
    arm, _ = seg_run
    seen = {"legend": None, "bars": []}
    monkeypatch.setattr(
        rm, "draw_legend", lambda ax, entries, font, **k: seen.update(legend=entries)
    )
    real_bar = rm.draw_scalebar
    monkeypatch.setattr(
        rm,
        "draw_scalebar",
        lambda ax, h, w, bar_px, label, font, **k: (
            seen["bars"].append((bar_px, label)),
            real_bar(ax, h, w, bar_px, label, font, **k),
        ),
    )
    m = _zoom(
        arm,
        tmp_path / "z",
        "--outline-color",
        "#00ff00",
        "--outline-width",
        "2",
        "--cells-label",
        "StarDist cells",
    )
    assert m["zoom"] == {
        "y": 20,
        "x": 20,
        "size_px": 120,
        "size_um": pytest.approx(60.0),
    }
    assert m["cells_in_zoom"] == 2 and m["outline"] == {
        "color": "#00ff00",
        "width_px": 2,
    }
    assert seen["legend"] == [
        ("DAPI", (1.0, 1.0, 1.0)),
        ("StarDist cells", (0.0, 1.0, 0.0)),
    ]
    assert len(seen["bars"]) == 2
    for bar_px, label in seen["bars"]:
        assert bar_px > 0 and label.endswith(("µm", "mm"))
    zoom_bar = seen["bars"][1]
    assert zoom_bar[0] == pytest.approx(m["scalebars"]["zoom_um"] / tm.PX)
    assert (tmp_path / "z" / "P1_zoom.png").is_file()


def test_the_outline_pixels_carry_the_colour(seg_run):
    _, labels = seg_run
    zoom = rz.white(np.full((60, 60), 100, np.uint16), 1, 99, 1.0)
    edge = rz.outlines(labels[20:80, 20:80], 1)
    zoom[edge] = rz._rgb("yellow")
    assert tuple(zoom[edge][0]) == (1.0, 1.0, 0.0)


def test_a_mask_off_the_reference_canvas_is_refused(seg_run, tmp_path):
    arm, _ = seg_run
    import shutil

    shutil.copytree(arm, tmp_path / "arm")
    bad = tmp_path / "small.tif"
    tifffile.imwrite(str(bad), np.zeros((50, 50), np.uint32))
    csv_path = tmp_path / "arm" / "csv" / "segmented.csv"
    rows = list(csv.DictReader(open(csv_path)))
    rows[0]["cell_mask"] = str(bad)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="not on the reference canvas"):
        _zoom(tmp_path / "arm", tmp_path / "z")


def test_a_zoom_too_large_to_show_cells_is_refused(seg_run, tmp_path):
    arm, _ = seg_run
    with pytest.raises(SystemExit, match="max-zoom-px"):
        _zoom(arm, tmp_path / "z", "--max-zoom-px", "50")


def test_a_checkpoint_naming_preprocessed_for_a_converted_slide_still_draws(
    seg_run, tmp_path, caplog
):
    """A run written before the passthrough fix records <pid>/preprocessed/<name> for a
    slide published under <pid>/converted/<name> (job 6847276)."""
    arm, _ = seg_run
    import shutil

    shutil.copytree(arm, tmp_path / "arm")
    csv_path = tmp_path / "arm" / "csv" / "segmented.csv"
    rows = list(csv.DictReader(open(csv_path)))
    real = Path(rows[0]["registered_image"])
    converted = tmp_path / "arm" / "P1" / "converted" / "P1_ref.ome.tif"
    converted.parent.mkdir(parents=True)
    shutil.copy(real, converted)
    rows[0]["registered_image"] = str(
        tmp_path / "arm" / "P1" / "preprocessed" / "P1_ref.ome.tif"
    )
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with caplog.at_level("WARNING"):
        m = _zoom(tmp_path / "arm", tmp_path / "z")
    assert m["image"] == str(converted)
    assert "does not exist; using" in caplog.text


def test_an_anonymous_reference_falls_back_to_the_checkpoint_channels(
    seg_run, tmp_path
):
    """A slide stitched without --channel-names has no names of its own; segmented.csv's
    channels column is the same order (see test_reg_mosaic's anonymous-slide case)."""
    import shutil

    arm, _ = seg_run
    dst = tmp_path / "armR"
    shutil.copytree(arm, dst)
    reg = dst / "P1" / "registered" / "registered_slides" / "P1_ref_registered.ome.tiff"
    tm._anonymous_ome(reg, list(tifffile.imread(str(reg))))
    for path in (dst / "csv" / "segmented.csv",):
        path.write_text(path.read_text().replace(str(arm), str(dst)))
    m = _zoom(dst, tmp_path / "out")
    assert m["channel_index"] == 0 and m["channel_names_from"] == "checkpoint"
