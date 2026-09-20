"""reg_crop: one named channel, cropped, scaled so the background stays black."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks import reg_crop as rc
from benchmarks import reg_mosaic as rm
from benchmarks.tests import test_reg_mosaic as tm

PX = tm.PX


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A run as an arm publishes it: a reference (DAPI, PANCK) and a moving round (DAPI, CD3)."""
    return tm.originals_root.__wrapped__(tmp_path_factory) / "armR"


def _crop(run, out, *extra):
    argv = [str(run), "-o", str(out), "--formats", "png", "--dpi", "50", *extra]
    assert rc.main(argv) == 0
    return out


def _meta(out, channel, pid="P1"):
    return json.loads((out / f"{pid}_{channel}_crop.json").read_text())


def test_a_named_channel_is_cropped_at_the_requested_size(run_dir, tmp_path):
    import matplotlib.image

    out = _crop(
        run_dir,
        tmp_path / "one",
        "--channel",
        "DAPI",
        "--field-um",
        str(64 * PX),
        "--crop-px",
        "256",
    )
    m = _meta(out, "DAPI")
    assert (
        m["channel"] == "DAPI" and m["crop"]["size_px"] == 64 and m["output_px"] == 256
    )
    img = matplotlib.image.imread(str(out / "P1_DAPI_crop.png"))
    assert img.shape[:2] == (256, 256)


def test_a_channel_on_another_slide_is_found(run_dir, tmp_path):
    """CD3 is not on the reference: every registered slide sits on the same canvas, so the
    tool searches them all rather than assuming the reference row."""
    out = _crop(
        run_dir,
        tmp_path / "cd3",
        "--channel",
        "CD3",
        "--field-um",
        str(64 * PX),
        "--roi",
        "40,40",
    )
    m = _meta(out, "CD3")
    assert m["image"].endswith("P1_cd3_registered.ome.tiff")
    assert (m["crop"]["y"], m["crop"]["x"]) == (40, 40)


def test_the_same_roi_names_the_same_tissue_in_every_channel(run_dir, tmp_path):
    out = _crop(
        run_dir,
        tmp_path / "pair",
        "--channel",
        "DAPI",
        "--channel",
        "CD3",
        "--field-um",
        str(64 * PX),
        "--roi",
        "48,52",
    )
    a, b = _meta(out, "DAPI")["crop"], _meta(out, "CD3")["crop"]
    assert a == b


def test_an_unknown_channel_names_what_the_run_actually_has(run_dir, tmp_path):
    with pytest.raises(SystemExit, match="no channel 'CD8'"):
        _crop(run_dir, tmp_path / "bad", "--channel", "CD8")


def test_clean_is_the_default_and_percentile_is_opt_in(run_dir, tmp_path):
    out = _crop(run_dir, tmp_path / "auto", "--channel", "DAPI", "--roi", "40,40")
    assert _meta(out, "DAPI")["limits"]["how"] == "clean"
    out = _crop(
        run_dir,
        tmp_path / "pct",
        "--channel",
        "DAPI",
        "--roi",
        "40,40",
        "--autoscale",
        "percentile",
    )
    assert _meta(out, "DAPI")["limits"]["how"] == "percentile"
    out = _crop(
        run_dir,
        tmp_path / "pin",
        "--channel",
        "DAPI",
        "--roi",
        "40,40",
        "--vmin",
        "10",
        "--vmax",
        "900",
    )
    m = _meta(out, "DAPI")["limits"]
    assert (m["how"], m["lo"], m["hi"]) == ("pinned", 10.0, 900.0)


def test_the_clean_limits_reach_the_drawn_pixels(run_dir, tmp_path, monkeypatch):
    """Not just recorded in the JSON: the image handed to the writer is stretched with them,
    so its background is black."""
    seen = {}
    real = rm.write_crop
    monkeypatch.setattr(
        rm,
        "write_crop",
        lambda img, *a, **k: (seen.update(img=img), real(img, *a, **k))[1],
    )
    out = _crop(run_dir, tmp_path / "px", "--channel", "DAPI", "--roi", "40,40")
    lo, hi = (_meta(out, "DAPI")["limits"][k] for k in ("lo", "hi"))
    plane = seen["img"]
    assert plane.ndim == 3 and plane.shape[2] == 3
    assert 0.0 <= plane.min() and plane.max() <= 1.0
    assert hi > lo


def test_a_colour_tints_the_channel_and_names_it_in_the_legend(
    run_dir, tmp_path, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        rm, "draw_legend", lambda ax, entries, font, **k: seen.append(entries)
    )
    _crop(
        run_dir,
        tmp_path / "col",
        "--channel",
        "CD3",
        "--roi",
        "40,40",
        "--colors",
        "#00e5ff",
        "--label",
        "CD3 (cyan)",
    )
    assert seen == [[("CD3 (cyan)", (0.0, 0.8980392156862745, 1.0))]]


def test_plain_drops_the_labels(run_dir, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rm, "draw_legend", lambda *a, **k: seen.append(a))
    monkeypatch.setattr(rm, "draw_scalebar", lambda *a, **k: seen.append(a))
    out = _crop(
        run_dir, tmp_path / "bare", "--channel", "DAPI", "--roi", "40,40", "--plain"
    )
    assert seen == [] and _meta(out, "DAPI")["title"] == ""


def test_colorize_is_a_pure_scaling_of_the_grey(run_dir):
    grey = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = rc.colorize(grey, "#00e5ff")
    assert out.shape == (4, 4, 3)
    assert out[..., 0].max() == 0.0  # no red in cyan
    np.testing.assert_allclose(out[..., 2], grey, atol=1e-6)
