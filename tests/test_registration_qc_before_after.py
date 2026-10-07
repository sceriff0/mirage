"""The registration QC figure shows BEFORE and AFTER, on the reference canvas.

``GENERATE_REGISTRATION_QC`` used to render one composite — reference + registered
— which shows what registration *produced* and not what it *corrected*. This file
pins the two properties that make the added "before" panel honest:

1. The native (pre-registration) moving slide is placed on the REFERENCE canvas by
   pad-or-crop at the origin, never by rescaling. Rescaling a differently-sized
   native image to the reference's shape silently removes the scale component of
   the misalignment, which makes the "before" panel understate the error and
   registration look better than it was.
2. The two panels are one image with a deterministic layout, so the published
   ``*_QC_RGB*`` names keep meaning one artifact per moving slide.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

UTILS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

pytest.importorskip("skimage")
pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
cv2 = pytest.importorskip("cv2")

import qc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "bin" / "generate_registration_qc.py"

# The ResolutionUnit enum moved: it is a MODULE-level `tifffile.RESUNIT` in current
# tifffile (2025.5.10, the version requirements/constraints.txt pins and CI installs),
# and only a `tifffile.TIFF.RESUNIT` attribute in older ones (2023.4.12, what a local
# venv may still hold). Reading it off `TIFF` alone made these assertions raise
# `AttributeError: '_TIFF' object has no attribute 'RESUNIT'` in CI while passing
# locally. The value itself is stable across both: RESUNIT.MICROMETER == 5 (standard
# TIFF defines only 1-3; tifffile extends it with 4 = mm and 5 = um, following DNG).
RESUNIT = getattr(tifffile, "RESUNIT", None) or tifffile.TIFF.RESUNIT
assert int(RESUNIT.MICROMETER) == 5


def test_smaller_moving_image_is_zero_padded_at_the_origin():
    ref = np.zeros((10, 12), dtype=np.uint16)
    mov = np.full((6, 8), 7, dtype=np.uint16)

    out = qc.compose_on_reference_canvas(ref, mov)

    assert out.shape == (10, 12)
    assert out.dtype == np.uint16
    assert (out[:6, :8] == 7).all()
    assert (out[6:, :] == 0).all()
    assert (out[:, 8:] == 0).all()


def test_larger_moving_image_is_cropped_not_rescaled():
    ref = np.zeros((4, 5), dtype=np.uint16)
    mov = np.arange(9 * 11, dtype=np.uint16).reshape(9, 11)

    out = qc.compose_on_reference_canvas(ref, mov)

    assert out.shape == (4, 5)
    # Cropped at the origin: every value is the SAME value the input had at that
    # coordinate. A rescale would have produced interpolated values instead.
    assert np.array_equal(out, mov[:4, :5])


def test_equal_shapes_pass_through_unchanged():
    ref = np.zeros((5, 5), dtype=np.uint16)
    mov = np.arange(25, dtype=np.uint16).reshape(5, 5)

    out = qc.compose_on_reference_canvas(ref, mov)

    assert np.array_equal(out, mov)


def test_one_dimension_bigger_one_smaller_is_both_cropped_and_padded():
    ref = np.zeros((10, 4), dtype=np.uint16)
    mov = np.full((6, 9), 3, dtype=np.uint16)

    out = qc.compose_on_reference_canvas(ref, mov)

    assert out.shape == (10, 4)
    assert (out[:6, :] == 3).all()  # padded down the Y axis
    assert (out[6:, :] == 0).all()
    # cropped on X: nothing beyond column 4 survives, and nothing was resampled
    assert np.array_equal(out[:6, :], mov[:6, :4])


def _plane(seed, h, w):
    return np.random.default_rng(seed).integers(0, 65536, size=(h, w), dtype=np.uint16)


def test_render_before_after_is_two_panels_wide_plus_the_gap():
    ref = _plane(1, 20, 30)
    native = _plane(2, 20, 30)
    registered = _plane(3, 20, 30)

    out = qc.render_before_after(ref, native, registered, scale_factor=1.0, gap=8)

    assert out.dtype == np.uint8
    assert out.shape == (3, 20, 30 * 2 + 8)


def test_render_before_after_puts_registered_on_the_right_and_native_on_the_left():
    """Red is the MOVING slide in both panels; green is the reference in both.
    A native and a registered plane that differ must therefore produce panels
    whose red channels differ, while their green channels are identical.

    ``native`` is a constant-zero plane and ``registered`` is a plane with real
    variance, NOT a second constant. ``create_nuclear_overlay`` autoscales each
    plane independently with a min-max stretch, and a min-max stretch of ANY
    constant-valued plane collapses to all-zero regardless of the constant's
    value (``(x - x) / eps == 0`` everywhere) -- so two different constants
    would produce identical (all-zero) red channels and the panels would be
    indistinguishable for a reason that has nothing to do with
    ``render_before_after`` itself.
    """
    ref = _plane(1, 16, 16)
    native = np.zeros((16, 16), dtype=np.uint16)
    registered = _plane(3, 16, 16)

    out = qc.render_before_after(ref, native, registered, scale_factor=1.0, gap=4)

    left = out[:, :, :16]
    right = out[:, :, 16 + 4 :]

    # Green (index 1) is the reference in both panels -> identical.
    assert np.array_equal(left[1], right[1])
    # Red (index 0) is the moving slide -> a constant-zero native and a
    # variance-carrying registered plane cannot produce the same red channel.
    assert not np.array_equal(left[0], right[0])
    assert left[0].max() == 0
    assert right[0].max() > 0


def test_render_before_after_marks_the_panel_boundary_in_blue():
    ref = _plane(1, 8, 9)
    out = qc.render_before_after(
        ref, _plane(2, 8, 9), _plane(3, 8, 9), scale_factor=1.0, gap=5
    )

    band = out[:, :, 9 : 9 + 5]
    assert (band[2] == 255).all(), "the separator band must be blue 255"
    assert (band[0] == 0).all() and (band[1] == 0).all(), (
        "the separator must not carry red or green, or it would read as signal"
    )
    # Blue is the annotation channel and is unused everywhere else.
    assert out[2, :, :9].max() == 0
    assert out[2, :, 9 + 5 :].max() == 0


def test_render_before_after_without_a_native_is_a_single_panel():
    """The script stays runnable by hand against a pair of images. With no
    native there is no 'before' to draw, and the output degenerates to exactly
    the one-panel composite this process produced before the change."""
    ref = _plane(1, 12, 14)
    registered = _plane(3, 12, 14)

    out = qc.render_before_after(ref, None, registered, scale_factor=1.0, gap=8)

    assert out.shape == (3, 12, 14)
    assert out[2].max() == 0


def test_render_before_after_pads_a_smaller_native_onto_the_reference_canvas():
    """The panels must be the same height and width even when the native slide
    is a different size — that is what compose_on_reference_canvas is for."""
    ref = _plane(1, 20, 24)
    native = _plane(2, 11, 13)  # smaller in both axes
    registered = _plane(3, 20, 24)

    out = qc.render_before_after(ref, native, registered, scale_factor=1.0, gap=6)

    assert out.shape == (3, 20, 24 * 2 + 6)


def test_render_before_after_warns_when_the_registered_plane_is_off_canvas(caplog):
    """A registered slide is supposed to already be on the reference canvas.

    Routing the REGISTERED plane through ``compose_on_reference_canvas`` (which the
    native panel needs) means one that is not gets silently pad-or-cropped, where
    ``create_nuclear_overlay`` used to raise ``ValueError("Shape mismatch...")``.
    Drawing it is right -- a QC figure that refuses to render tells its reader
    nothing -- but the discrepancy is a registration-output defect and must be said
    out loud, naming BOTH shapes so the reader can tell which way it went.
    """
    ref = _plane(1, 20, 24)
    native = _plane(2, 11, 13)
    registered = _plane(3, 18, 24)  # NOT the reference canvas

    with caplog.at_level(logging.WARNING, logger=qc.logger.name):
        out = qc.render_before_after(ref, native, registered, scale_factor=1.0, gap=6)

    # Still rendered, still on the reference canvas.
    assert out.shape == (3, 20, 24 * 2 + 6)
    assert "(18, 24)" in caplog.text and "(20, 24)" in caplog.text


def test_render_before_after_is_silent_when_only_the_native_differs(caplog):
    """The native plane differing from the reference IS the normal case -- warning on
    it would train a reader to ignore the warning above."""
    ref = _plane(1, 20, 24)
    native = _plane(2, 11, 13)
    registered = _plane(3, 20, 24)

    with caplog.at_level(logging.WARNING, logger=qc.logger.name):
        qc.render_before_after(ref, native, registered, scale_factor=1.0, gap=6)

    assert caplog.text.strip() == ""


def test_render_before_after_honours_the_scale_factor():
    ref = _plane(1, 40, 40)
    out = qc.render_before_after(
        ref, _plane(2, 40, 40), _plane(3, 40, 40), scale_factor=0.5, gap=4
    )

    # Each panel is 20 wide after a 0.5 rescale; the gap is not rescaled.
    assert out.shape == (3, 20, 20 * 2 + 4)


def test_cyx_to_bgr_reorders_for_opencv():
    cyx = np.zeros((3, 2, 2), dtype=np.uint8)
    cyx[0] = 10  # red
    cyx[1] = 20  # green
    cyx[2] = 30  # blue

    bgr = qc.cyx_to_bgr(cyx)

    assert bgr.shape == (2, 2, 3)
    assert (bgr[:, :, 0] == 30).all()  # OpenCV channel 0 is blue
    assert (bgr[:, :, 1] == 20).all()
    assert (bgr[:, :, 2] == 10).all()


def _write_triplet(tmp_path, h=48, w=40):
    """A reference, a native moving slide and its registered counterpart, all
    OME-TIFF with the nuclear marker deliberately NOT at channel 0 so the test
    exercises pick_nuclear_index rather than the channel-0 fallback."""
    sys.path.insert(0, UTILS_DIR)
    from ome_io import write_ome_tiff

    def _stack(seed):
        return np.stack([_plane(seed, h, w), _plane(seed + 100, h, w)])

    paths = {}
    for name, seed in (("reference", 11), ("native", 22), ("registered", 33)):
        p = tmp_path / f"{name}.ome.tiff"
        write_ome_tiff(p, _stack(seed), channels=["SMA", "DAPI"], pixel_size_um=0.5)
        paths[name] = p
    return paths


def test_create_registration_qc_with_a_native_writes_a_two_panel_preview(tmp_path):
    p = _write_triplet(tmp_path)
    out = tmp_path / "P001_mov1_QC_RGB.tif"

    qc.create_registration_qc(
        reference_path=p["reference"],
        registered_path=p["registered"],
        output_path=out,
        scale_factor=1.0,
        save_fullres=True,
        save_png=True,
        save_tiff=True,
        native_path=p["native"],
    )

    preview = tifffile.imread(str(out))
    fullres = tifffile.imread(str(tmp_path / "P001_mov1_QC_RGB_fullres.tif"))
    png = cv2.imread(str(tmp_path / "P001_mov1_QC_RGB.png"))

    # 40-wide reference -> two 40-wide panels plus the 8px separator.
    assert preview.shape == (3, 48, 88)
    assert fullres.shape == (3, 48, 88)
    assert png.shape == (48, 88, 3)


def test_create_registration_qc_without_a_native_stays_single_panel(tmp_path):
    p = _write_triplet(tmp_path)
    out = tmp_path / "P001_mov1_QC_RGB.tif"

    qc.create_registration_qc(
        reference_path=p["reference"],
        registered_path=p["registered"],
        output_path=out,
        scale_factor=1.0,
        save_fullres=False,
        save_png=False,
        save_tiff=True,
    )

    assert tifffile.imread(str(out)).shape == (3, 48, 40)


def test_create_registration_qc_rejects_a_missing_native(tmp_path):
    p = _write_triplet(tmp_path)

    with pytest.raises(FileNotFoundError, match="Native image not found"):
        qc.create_registration_qc(
            reference_path=p["reference"],
            registered_path=p["registered"],
            output_path=tmp_path / "x_QC_RGB.tif",
            native_path=tmp_path / "does_not_exist.ome.tiff",
        )


def test_cli_with_native_writes_a_two_panel_figure(tmp_path):
    p = _write_triplet(tmp_path)
    outdir = tmp_path / "qc"

    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reference",
            str(p["reference"]),
            "--registered",
            str(p["registered"]),
            "--native",
            str(p["native"]),
            "--output",
            str(outdir),
            "--scale-factor",
            "1.0",
            "--nuclear-markers",
            "DAPI",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": UTILS_DIR},
    )

    assert r.returncode == 0, r.stderr
    preview = tifffile.imread(str(outdir / "registered_QC_RGB.tif"))
    assert preview.shape == (3, 48, 88)


def test_cli_without_native_stays_single_panel(tmp_path):
    p = _write_triplet(tmp_path)
    outdir = tmp_path / "qc"

    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reference",
            str(p["reference"]),
            "--registered",
            str(p["registered"]),
            "--output",
            str(outdir),
            "--scale-factor",
            "1.0",
            "--nuclear-markers",
            "DAPI",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": UTILS_DIR},
    )

    assert r.returncode == 0, r.stderr
    assert tifffile.imread(str(outdir / "registered_QC_RGB.tif")).shape == (3, 48, 40)


def test_cli_rejects_a_native_list_of_the_wrong_length(tmp_path):
    p = _write_triplet(tmp_path)

    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reference",
            str(p["reference"]),
            "--registered",
            str(p["registered"]),
            str(p["native"]),
            "--native",
            str(p["native"]),
            "--output",
            str(tmp_path / "qc"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": UTILS_DIR},
    )

    assert r.returncode == 1
    assert "--native" in (r.stdout + r.stderr)


def test_cli_pixel_size_um_stamps_both_qc_tiffs(tmp_path):
    p = _write_triplet(tmp_path)
    outdir = tmp_path / "qc"

    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reference",
            str(p["reference"]),
            "--registered",
            str(p["registered"]),
            "--output",
            str(outdir),
            "--scale-factor",
            "0.5",
            "--nuclear-markers",
            "DAPI",
            "--pixel-size-um",
            "0.325",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": UTILS_DIR},
    )

    assert r.returncode == 0, r.stderr

    scale_factor = 0.5
    pixel_size_um = 0.325

    with tifffile.TiffFile(str(outdir / "registered_QC_RGB.tif")) as tf:
        tags = tf.pages[0].tags
        assert tags["ResolutionUnit"].value == RESUNIT.MICROMETER
        x_num, x_den = tags["XResolution"].value
        preview_resolution = x_num / x_den

    with tifffile.TiffFile(str(outdir / "registered_QC_RGB_fullres.tif")) as tf:
        tags = tf.pages[0].tags
        assert tags["ResolutionUnit"].value == RESUNIT.MICROMETER
        x_num, x_den = tags["XResolution"].value
        fullres_resolution = x_num / x_den

    assert fullres_resolution == pytest.approx(1.0 / pixel_size_um, rel=1e-3)
    assert preview_resolution == pytest.approx(scale_factor / pixel_size_um, rel=1e-3)

    # A resolution TAG alone is not a usable scale bar in the tools this figure is opened
    # in: ImageJ reads its scale off the ImageDescription's `unit=` line, and an OME-XML
    # reader reads PhysicalSizeX/Y, neither of which is the XResolution/ResolutionUnit tag
    # checked above.
    with tifffile.TiffFile(str(outdir / "registered_QC_RGB_fullres.tif")) as tf:
        description = tf.pages[0].tags["ImageDescription"].value
    assert "unit=um" in description

    with tifffile.TiffFile(str(outdir / "registered_QC_RGB.tif")) as tf:
        ome_xml = tf.ome_metadata
    match = re.search(r'PhysicalSizeX="([\d.]+)"', ome_xml)
    assert match, f"no PhysicalSizeX in OME-XML: {ome_xml}"
    assert float(match.group(1)) == pytest.approx(
        pixel_size_um / scale_factor, rel=1e-3
    )
    # A bare PhysicalSizeX is not a scale: OME's default unit is um, but a reader that
    # honours the attribute reads whatever PhysicalSizeXUnit says, so a number written
    # without its unit is one ome_io._MICRON change away from silently meaning nm.
    # ome_io.ome_metadata writes the pair together (bin/utils/ome_io.py:262-266); this
    # pins that BOTH halves reach the file.
    assert 'PhysicalSizeXUnit="\u00b5m"' in ome_xml, (
        f"no PhysicalSizeXUnit in OME-XML: {ome_xml}"
    )
    assert 'PhysicalSizeYUnit="\u00b5m"' in ome_xml, (
        f"no PhysicalSizeYUnit in OME-XML: {ome_xml}"
    )


def test_cli_without_pixel_size_um_leaves_output_unstamped(tmp_path):
    p = _write_triplet(tmp_path)
    outdir = tmp_path / "qc"

    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reference",
            str(p["reference"]),
            "--registered",
            str(p["registered"]),
            "--output",
            str(outdir),
            "--nuclear-markers",
            "DAPI",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": UTILS_DIR},
    )

    assert r.returncode == 0, r.stderr

    # tifffile always writes an XResolution tag (it defaults to (1, 1) /
    # RESUNIT.NONE when no resolution is given); the thing pixel_size_um
    # controls is whether that tag is STAMPED WITH A MICROMETER SCALE, so the
    # "unstamped" assertion is on the unit, not on the tag's mere presence.
    with tifffile.TiffFile(str(outdir / "registered_QC_RGB.tif")) as tf:
        assert tf.pages[0].tags["ResolutionUnit"].value != RESUNIT.MICROMETER
        assert "PhysicalSizeX" not in tf.ome_metadata

    with tifffile.TiffFile(str(outdir / "registered_QC_RGB_fullres.tif")) as tf:
        assert "unit=um" not in tf.pages[0].tags["ImageDescription"].value


def test_create_registration_qc_stamps_resolution_on_both_tiffs_when_given(tmp_path):
    """Folded requirement: pixel_size_um stamps XResolution/ResolutionUnit on BOTH
    the fullres and the downsampled composite, and the two differ by exactly the
    scale_factor -- the downsampled preview covers the same tissue in fewer, larger
    pixels, so its pixel size in microns is larger and its resolution (pixels/um)
    is smaller."""
    p = _write_triplet(tmp_path)
    out = tmp_path / "P001_mov1_QC_RGB.tif"
    scale_factor = 0.5
    pixel_size_um = 0.325

    qc.create_registration_qc(
        reference_path=p["reference"],
        registered_path=p["registered"],
        output_path=out,
        scale_factor=scale_factor,
        save_fullres=True,
        save_png=False,
        save_tiff=True,
        pixel_size_um=pixel_size_um,
    )

    with tifffile.TiffFile(str(out)) as tf:
        tags = tf.pages[0].tags
        x_num, x_den = tags["XResolution"].value
        preview_resolution = x_num / x_den
        assert tags["ResolutionUnit"].value == RESUNIT.MICROMETER

    with tifffile.TiffFile(str(tmp_path / "P001_mov1_QC_RGB_fullres.tif")) as tf:
        tags = tf.pages[0].tags
        x_num, x_den = tags["XResolution"].value
        fullres_resolution = x_num / x_den
        assert tags["ResolutionUnit"].value == RESUNIT.MICROMETER

    assert fullres_resolution == pytest.approx(1.0 / pixel_size_um, rel=1e-3)
    # The downsampled preview's pixel is 1/scale_factor times bigger in microns,
    # so its resolution (pixels/um) is scale_factor times smaller.
    assert preview_resolution == pytest.approx(
        fullres_resolution * scale_factor, rel=1e-3
    )

    with tifffile.TiffFile(str(tmp_path / "P001_mov1_QC_RGB_fullres.tif")) as tf:
        assert "unit=um" in tf.pages[0].tags["ImageDescription"].value

    with tifffile.TiffFile(str(out)) as tf:
        ome_xml = tf.ome_metadata
    match = re.search(r'PhysicalSizeX="([\d.]+)"', ome_xml)
    assert match, f"no PhysicalSizeX in OME-XML: {ome_xml}"
    assert float(match.group(1)) == pytest.approx(
        pixel_size_um / scale_factor, rel=1e-3
    )
    # A bare PhysicalSizeX is not a scale: OME's default unit is um, but a reader that
    # honours the attribute reads whatever PhysicalSizeXUnit says, so a number written
    # without its unit is one ome_io._MICRON change away from silently meaning nm.
    # ome_io.ome_metadata writes the pair together (bin/utils/ome_io.py:262-266); this
    # pins that BOTH halves reach the file.
    assert 'PhysicalSizeXUnit="\u00b5m"' in ome_xml, (
        f"no PhysicalSizeXUnit in OME-XML: {ome_xml}"
    )
    assert 'PhysicalSizeYUnit="\u00b5m"' in ome_xml, (
        f"no PhysicalSizeYUnit in OME-XML: {ome_xml}"
    )


def test_create_registration_qc_without_pixel_size_um_leaves_output_unstamped(tmp_path):
    """``pixel_size_um=None`` (the default) must write exactly what this function
    wrote before the parameter existed. tifffile always writes SOME XResolution
    tag (defaulting to (1, 1) / RESUNIT.NONE), so "unstamped" means no explicit
    resolution kwargs reached write_tiff, not that the tag is absent."""
    p = _write_triplet(tmp_path)
    out = tmp_path / "P001_mov1_QC_RGB.tif"

    qc.create_registration_qc(
        reference_path=p["reference"],
        registered_path=p["registered"],
        output_path=out,
        scale_factor=1.0,
        save_fullres=True,
        save_png=False,
        save_tiff=True,
        pixel_size_um=None,
    )

    with tifffile.TiffFile(str(out)) as tf:
        tags = tf.pages[0].tags
        assert tags["ResolutionUnit"].value != RESUNIT.MICROMETER
        assert tags["XResolution"].value == (1, 1)
        assert "PhysicalSizeX" not in tf.ome_metadata

    with tifffile.TiffFile(str(tmp_path / "P001_mov1_QC_RGB_fullres.tif")) as tf:
        tags = tf.pages[0].tags
        assert tags["ResolutionUnit"].value != RESUNIT.MICROMETER
        assert tags["XResolution"].value == (1, 1)
        assert "unit=um" not in tags["ImageDescription"].value


def test_a_fullres_composite_too_large_for_a_classic_tiff_is_written_as_bigtiff(
    tmp_path, monkeypatch
):
    """An ImageJ TIFF cannot be BigTIFF, so a whole-slide composite overflowed its 32-bit
    offsets (``'I' format requires 0 <= number <= 4294967295``) and failed the run. Above
    the payload threshold the same pixels and scale are written as an OME BigTIFF."""
    p = _write_triplet(tmp_path)

    def _fullres(name, threshold):
        out = tmp_path / f"{name}_QC_RGB.tif"
        if threshold is not None:
            monkeypatch.setattr(qc, "BIGTIFF_THRESHOLD_BYTES", threshold)
        qc.create_registration_qc(
            reference_path=p["reference"],
            registered_path=p["registered"],
            output_path=out,
            scale_factor=1.0,
            save_fullres=True,
            save_png=False,
            save_tiff=False,
            native_path=p["native"],
            pixel_size_um=0.5,
        )
        return tmp_path / f"{name}_QC_RGB_fullres.tif"

    small = _fullres("small", None)
    large = _fullres("large", 1)

    with tifffile.TiffFile(str(small)) as tf:
        assert tf.is_imagej and not tf.is_bigtiff
        small_px = tf.asarray()
        small_res = tf.pages[0].tags["XResolution"].value
    with tifffile.TiffFile(str(large)) as tf:
        assert tf.is_bigtiff and tf.is_ome
        assert 'PhysicalSizeX="0.5"' in tf.ome_metadata
        large_px = tf.asarray()
        large_res = tf.pages[0].tags["XResolution"].value

    assert np.array_equal(small_px, large_px)
    assert small_res == large_res
