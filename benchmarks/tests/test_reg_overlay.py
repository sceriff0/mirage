"""reg_overlay: one registration dir in, a Before and an After image of the same crop out.

Reuses test_reg_mosaic's miniature arm root (armA perfect, armB 3 px off, native 12 px
down / 9 px left), so the two tools are exercised on identical composites.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import reg_mosaic as rm
from benchmarks import reg_overlay as ro
from benchmarks.tests import test_reg_mosaic as tm

PX, _run = tm.PX, tm._run


@pytest.fixture(scope="module")
def arm_root(tmp_path_factory):
    return tm.arm_root.__wrapped__(tmp_path_factory)


def _overlay(arm_dir, out, *extra):
    argv = [str(arm_dir), "-o", str(out), "--field-px", "96", "--rounds", "CD3"]
    argv += ["--formats", "png", "--dpi", "50", *extra]
    assert ro.main(argv) == 0
    return json.loads((out / "P1_CD3_overlay.json").read_text())


def test_before_and_after_images_of_one_crop_with_numbers_and_a_scale_bar(
    arm_root, tmp_path
):
    out = tmp_path / "ov"
    m = _overlay(arm_root / "armB", out, "--numbers", "image")
    for name in ("before", "after", "locator"):
        assert (out / f"P1_CD3_{name}.png").is_file()
    assert m["palette"] == "magenta-cyan" and m["pixel_size_um"] == pytest.approx(PX)
    assert m["crop"]["size_px"] == 96 and m["scalebar_um"] > 0
    before, after = m["numbers"]["before"], m["numbers"]["after"]
    assert before["source"] == after["source"] == "image"
    assert after["shift_px"] == pytest.approx(3.0, abs=0.3)  # armB's residual
    assert before["shift_px"] > after["shift_px"] + 8  # native is ~15 px off


def test_the_scorer_numbers_are_used_when_warp_seg_qc_ran(arm_root, tmp_path):
    m = _overlay(arm_root / "armB", tmp_path / "ov")  # --numbers auto
    assert m["numbers"]["after"]["dice_matched"] == pytest.approx(0.74)
    assert m["numbers"]["before"]["dice_matched"] == pytest.approx(0.37)  # native stage


def test_the_crop_keeps_clear_of_the_mosaics_rois(arm_root, tmp_path):
    """Exclude exactly the box the tool would otherwise pick: it has to move. (Excluding a
    real mosaic's ROIs cannot fail on this fixture -- they happen not to overlap.)"""
    first = _overlay(arm_root / "armB", tmp_path / "free")
    free, m_ref = first["crop"], first["reference"]
    rois_json = tmp_path / "P1_rois.json"
    rois_json.write_text(
        json.dumps(
            {
                "reference": m_ref,
                "patch_px": free["size_px"],
                "rois": [{"id": 1, "y": free["y"], "x": free["x"]}],
            }
        )
    )
    c = _overlay(
        arm_root / "armB", tmp_path / "ov", "--avoid-rois-json", str(rois_json)
    )["crop"]
    s = free["size_px"]
    assert not (abs(c["y"] - free["y"]) < s and abs(c["x"] - free["x"]) < s), (c, free)
    mosaic = _run(
        arm_root, tmp_path / "mosaic"
    )  # and the real mosaic ROIs load and apply
    m = _overlay(
        arm_root / "armB",
        tmp_path / "ov2",
        "--avoid-rois-json",
        str(tmp_path / "mosaic" / "P1_rois.json"),
    )
    assert m["avoided_rois_from"].endswith("P1_rois.json") and mosaic["rois"]


def test_a_manual_roi_is_honoured(arm_root, tmp_path):
    m = _overlay(arm_root / "armA", tmp_path / "ov", "--roi", "20,30")
    assert (m["crop"]["y"], m["crop"]["x"]) == (20, 30)


def test_scale_bar_and_palette_reach_the_drawn_panels(arm_root, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        ro,
        "draw_panel",
        lambda img, title, note, bar, *a, **k: seen.append((title, note, bar)),
    )
    _overlay(arm_root / "armB", tmp_path / "ov", "--numbers", "image")
    assert [t for t, _, _ in seen] == ["Before", "After (armB)"]
    assert all(bar and bar[1].endswith("µm") for _, _, bar in seen)
    assert all(note.startswith("Dice = ") and "Δ = " in note for _, note, _ in seen)
    assert rm.PALETTES["magenta-cyan"] == ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0))


def test_a_mosaic_on_another_reference_is_not_avoided(arm_root, tmp_path):
    """Coordinates of another reference's frame name other tissue: ignored, not applied."""
    free = _overlay(arm_root / "armB", tmp_path / "free")["crop"]
    rois_json = tmp_path / "P1_rois.json"
    rois_json.write_text(
        json.dumps(
            {
                "reference": "/elsewhere/P1_other_reference.ome.tif",
                "patch_px": free["size_px"],
                "rois": [{"id": 1, "y": free["y"], "x": free["x"]}],
            }
        )
    )
    c = _overlay(
        arm_root / "armB", tmp_path / "ov", "--avoid-rois-json", str(rois_json)
    )
    assert (c["crop"]["y"], c["crop"]["x"]) == (free["y"], free["x"])


@pytest.fixture(scope="module")
def originals_root(tmp_path_factory):
    return tm.originals_root.__wrapped__(tmp_path_factory)


def test_overlay_reads_the_original_slides_at_one_pixel_per_pixel(
    originals_root, tmp_path, monkeypatch
):
    sizes = []
    real = ro.draw_panel

    def spy(img, title, note, bar, out_stem, formats, dpi, legend, size_in=None):
        sizes.append((img.shape[1], dpi))
        return real(img, title, note, bar, out_stem, formats, dpi, legend, size_in)

    monkeypatch.setattr(ro, "draw_panel", spy)
    m = _overlay(originals_root / "armR", tmp_path / "ov", "--numbers", "image")
    assert m["pixels"] == {"before": "originals", "after": "originals"}
    assert m["numbers"]["after"]["shift_px"] == pytest.approx(3.0, abs=0.3)
    assert m["numbers"]["before"]["shift_px"] == pytest.approx(15.0, abs=1.0)
    png = tmp_path / "ov" / "P1_CD3_after.png"
    import matplotlib.image

    assert (
        matplotlib.image.imread(str(png)).shape[1] >= 96
    )  # never smaller than the crop


def test_the_channel_names_are_drawn_in_their_colours(arm_root, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        ro.rm, "draw_legend", lambda ax, entries, font, **k: seen.append(entries)
    )
    _overlay(arm_root / "armB", tmp_path / "ov", "--numbers", "none")
    assert seen and all(
        e == [("reference DAPI", (0.0, 1.0, 1.0)), ("moving DAPI", (1.0, 0.0, 1.0))]
        for e in seen
    )


def test_a_run_without_qc_composites_is_drawn_from_its_slides(originals_root, tmp_path):
    import shutil

    shutil.copytree(originals_root, tmp_path / "root")
    shutil.rmtree(tmp_path / "root" / "armR" / "P1" / "qc")
    m = _overlay(tmp_path / "root" / "armR", tmp_path / "ov", "--numbers", "none")
    assert m["composite"] is None and m["pixels"] == {
        "before": "originals",
        "after": "originals",
    }


def test_variants_draw_alternative_crops_to_choose_between(arm_root, tmp_path):
    """One crop is one roll of the dice: --variants N gives N pairs, each on tissue no
    earlier variant used, so a bad-looking crop is not the only output."""
    out = tmp_path / "ov3"
    assert (
        ro.main(
            [
                str(arm_root / "armB"),
                "-o",
                str(out),
                "--field-px",
                "96",
                "--rounds",
                "CD3",
                "--formats",
                "png",
                "--dpi",
                "50",
                "--variants",
                "3",
                "--numbers",
                "none",
            ]
        )
        == 0
    )
    crops = []
    for v in (1, 2, 3):
        for name in ("before", "after", "locator"):
            assert (out / f"P1_CD3_v{v}_{name}.png").is_file(), (v, name)
        m = json.loads((out / f"P1_CD3_v{v}_overlay.json").read_text())
        assert m["variant"] == v
        crops.append((m["crop"]["y"], m["crop"]["x"]))
    assert not (out / "P1_CD3_before.png").exists()  # tagged, so nothing is overwritten
    for i, (y, x) in enumerate(crops):
        for y2, x2 in crops[i + 1 :]:
            assert abs(y - y2) >= 96 or abs(x - x2) >= 96, crops  # no overlap at all


def test_one_variant_keeps_the_untagged_names(arm_root, tmp_path):
    out = tmp_path / "ov1"
    _overlay(arm_root / "armB", out, "--numbers", "none", "--variants", "1")
    assert (out / "P1_CD3_before.png").is_file()


def test_variants_are_ignored_when_the_crop_is_pinned(arm_root, tmp_path, caplog):
    out = tmp_path / "ovpin"
    with caplog.at_level("WARNING"):
        m = _overlay(
            arm_root / "armB",
            out,
            "--numbers",
            "none",
            "--variants",
            "4",
            "--roi",
            "40,40",
        )
    assert m["crop"]["y"] == 40 and m["crop"]["x"] == 40
    assert "--variants is ignored" in caplog.text
    assert not (out / "P1_CD3_v2_overlay.json").exists()


def test_running_out_of_tissue_keeps_the_variants_already_drawn(
    arm_root, tmp_path, caplog
):
    """A slide has only so much distinct tissue. Asking for more variants than fit must not
    throw away the ones that did."""
    out = tmp_path / "ovmany"
    with caplog.at_level("WARNING"):
        assert (
            ro.main(
                [
                    str(arm_root / "armB"),
                    "-o",
                    str(out),
                    "--field-px",
                    "150",  # nearly half the 320 px fixture: two fit, ten do not
                    "--rounds",
                    "CD3",
                    "--formats",
                    "png",
                    "--dpi",
                    "50",
                    "--variants",
                    "10",
                    "--numbers",
                    "none",
                ]
            )
            == 0
        )
    drawn = sorted(out.glob("P1_CD3_v*_after.png"))
    assert 1 <= len(drawn) < 10
    assert "stopping at" in caplog.text
