"""prepare.py: the samplesheet/pairs layout (dry), and a real JPEG -> OME-TIFF conversion."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.anhir import prepare


def test_channel_names_are_samplesheet_safe_and_never_nuclear():
    assert prepare.channel_for_stem("HE") == "HE"
    assert prepare.channel_for_stem("S5-v1") == "S5_V1"
    assert (
        prepare.channel_for_stem("29-041-Izd2-w35-Cc10-5-les1")
        == "29_041_IZD2_W35_CC10_5_LES1"
    )
    assert prepare.channel_for_stem("dapi") == "DAPI_STAIN"
    assert prepare.channels_for("HE") == "DAPI|HE"


def test_build_inputs_dry_run_lays_out_one_patient_per_case(anhir_root, tmp_path):
    sheet, pairs = prepare.build_inputs(
        anhir_root["cases"], anhir_root["root"], tmp_path / "w", convert=False
    )
    assert list(sheet.columns) == prepare.SAMPLESHEET_COLUMNS
    assert len(sheet) == 6 and len(pairs) == 3
    # the target is the reference; every row's basename is unique even though two
    # cases share HE.jpg as their target
    assert sheet["is_reference"].tolist() == ["true", "false"] * 3
    basenames = [p.rsplit("/", 1)[-1] for p in sheet["preprocessed_image"]]
    assert len(set(basenames)) == 6
    assert basenames[0] == "anhir0_HE.ome.tif" and basenames[1] == "anhir0_S1.ome.tif"
    # channels differ between the two slides of a case, and the nuclear one leads
    assert (
        sheet.loc[0, "channels"] == "DAPI|HE" and sheet.loc[1, "channels"] == "DAPI|S1"
    )
    p = pairs.set_index("case_id")
    assert p.loc[0, "patient_id"] == "anhir0"
    assert (
        p.loc[0, "moving_stem"] == "anhir0_S1"
        and p.loc[0, "reference_stem"] == "anhir0_HE"
    )
    assert p.loc[0, "moving_channels"] == "DAPI_S1"
    assert p.loc[0, "source_landmarks"].endswith("landmarks/COAD_01/scale-25pc/S1.csv")
    assert p.loc[0, "diagonal"] == pytest.approx(2000.0)
    assert p.loc[2, "status"] == "evaluation"
    # a dry run writes nothing
    assert not (tmp_path / "w").exists()


def test_convert_image_writes_a_two_channel_ome_tiff_with_inverted_nuclei(tmp_path):
    import tifffile
    from PIL import Image

    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[10:20, 10:30] = (120, 60, 200)  # a "nucleus": darker than the white background
    rgb[rgb.sum(axis=2) == 0] = 255
    jpg = tmp_path / "S1.jpg"
    Image.fromarray(rgb).save(jpg, quality=95)
    out = prepare.convert_image(jpg, tmp_path / "S1.ome.tif", "S1", pixel_size_um=0.5)
    arr = tifffile.imread(out)
    assert arr.shape == (2, 40, 60) and arr.dtype == np.uint8
    dapi, stain = arr
    # the dark nucleus becomes BRIGHT on the nuclear channel and stays dark on the stain
    assert dapi[15, 20] > dapi[5, 5]
    assert stain[15, 20] < stain[5, 5]
    np.testing.assert_array_equal(dapi, 255 - stain)
    with tifffile.TiffFile(out) as tf:
        assert tf.ome_metadata and "DAPI" in tf.ome_metadata and "S1" in tf.ome_metadata
    # idempotent: a second call does not rewrite
    mtime = out.stat().st_mtime
    assert prepare.convert_image(jpg, out, "S1", pixel_size_um=0.5) == out
    assert out.stat().st_mtime == mtime


def test_build_inputs_converts_each_distinct_image_once_and_links_per_case(
    anhir_root, tmp_path, monkeypatch
):
    calls = []

    def fake_convert(jpeg_path, out_path, stem, pixel_size_um):
        calls.append(str(jpeg_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"tif")
        return out_path

    monkeypatch.setattr(prepare, "convert_image", fake_convert)
    images = anhir_root["images"]
    for c in anhir_root["cases"]:
        for rel in (c.source_image, c.target_image):
            (images / rel).parent.mkdir(parents=True, exist_ok=True)
            (images / rel).write_bytes(b"jpg")
    work = tmp_path / "w"
    sheet, pairs = prepare.build_inputs(anhir_root["cases"], anhir_root["root"], work)
    # 5 distinct images across 3 cases (HE.jpg is shared by cases 0 and 1)
    assert len(calls) == 5 and len(set(calls)) == 5
    assert (work / "samplesheet.csv").exists() and (
        work / "pairs_manifest.csv"
    ).exists()
    for p in sheet["preprocessed_image"]:
        link = Path(p)
        assert link.is_symlink(), f"{p} should be a per-case symlink"
        assert link.resolve().read_bytes() == b"tif"


def test_build_inputs_names_the_missing_image(anhir_root, tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare.py join"):
        prepare.build_inputs(anhir_root["cases"], anhir_root["root"], tmp_path / "w")
