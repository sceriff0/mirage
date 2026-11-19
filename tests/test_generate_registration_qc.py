"""bin/generate_registration_qc.py — the (success, message) contract.

_process_single_image catches every exception and reports (False, message), so
one bad slide does not abort the batch, but a broken generator can also fail
every slide while main() carries on. Both halves are asserted.

Asserted at the (bool, str) contract and 'the output directory is not empty',
not on the panel layout or filenames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")  # bin/utils/qc.py imports it at module scope

import generate_registration_qc as grq  # noqa: E402
from utils import ome_io  # noqa: E402


def _slide(path: Path, channels: list[str], shift: int = 0) -> Path:
    data = np.zeros((len(channels), 64, 64), dtype=np.uint16)
    for c in range(len(channels)):
        data[c, 10 + shift : 40 + shift, 10:40] = 3000 - 500 * c
    ome_io.write_ome_tiff(path, data, channels=channels, pixel_size_um=0.325)
    return path


def test_a_valid_pair_reports_success_and_writes_something(tmp_path):
    reference = _slide(tmp_path / "P001_ref.ome.tiff", ["DAPI", "PANCK"])
    registered = _slide(
        tmp_path / "P001_mov1_registered.ome.tiff", ["DAPI", "CD3"], shift=2
    )
    outdir = tmp_path / "qc"
    outdir.mkdir()

    ok, message = grq._process_single_image(
        reference_path=reference,
        registered_path=registered,
        output_dir=outdir,
        scale_factor=0.5,
        save_fullres=False,
        save_png=True,
        save_tiff=True,
        logger=logging.getLogger("test"),
    )

    assert ok is True, f"_process_single_image reported failure: {message}"
    written = sorted(p.name for p in outdir.iterdir())
    assert written, "reported success and wrote nothing"
    assert any(name.startswith("P001_mov1_registered") for name in written), (
        f"no output is named after the registered slide: {written}"
    )


def test_a_broken_slide_is_REPORTED_not_raised(tmp_path):
    """The batch loop depends on this: an exception here would abort every
    remaining slide, and the operator would see one traceback instead of a
    per-slide report."""
    reference = _slide(tmp_path / "P001_ref.ome.tiff", ["DAPI", "PANCK"])
    broken = tmp_path / "P001_broken_registered.ome.tiff"
    broken.write_bytes(b"not a tiff at all")
    outdir = tmp_path / "qc"
    outdir.mkdir()

    ok, message = grq._process_single_image(
        reference_path=reference,
        registered_path=broken,
        output_dir=outdir,
        scale_factor=0.5,
        save_fullres=False,
        save_png=True,
        save_tiff=True,
        logger=logging.getLogger("test"),
    )

    assert ok is False
    assert message and message != "Success", (
        "a failure must carry a message the operator can act on, not an empty string"
    )


def test_the_nuclear_channel_is_selected_by_NAME_not_by_position(tmp_path):
    """The overlay is built from the nuclear channel, resolved from the OME
    channel names against params.nuclear_markers. A positional fallback would
    build the overlay from a marker channel and make a correct registration look
    misaligned -- the QC image would be wrong while the registration was right."""
    reference = _slide(tmp_path / "P001_ref.ome.tiff", ["PANCK", "CELLTOX"])
    registered = _slide(
        tmp_path / "P001_mov1_registered.ome.tiff", ["CD3", "CELLTOX"], shift=2
    )
    outdir = tmp_path / "qc"
    outdir.mkdir()

    ok, message = grq._process_single_image(
        reference_path=reference,
        registered_path=registered,
        output_dir=outdir,
        scale_factor=0.5,
        save_fullres=False,
        save_png=True,
        save_tiff=True,
        logger=logging.getLogger("test"),
        nuclear_markers=["CELLTOX"],
    )

    assert ok is True, f"CELLTOX was not accepted as the nuclear marker: {message}"
    assert list(outdir.iterdir())


def test_a_marker_list_that_matches_nothing_fails_loudly(tmp_path):
    """Silently falling back to channel 0 is the failure mode this pins out."""
    reference = _slide(tmp_path / "P001_ref.ome.tiff", ["PANCK", "SMA"])
    registered = _slide(
        tmp_path / "P001_mov1_registered.ome.tiff", ["CD3", "CD8"], shift=2
    )
    outdir = tmp_path / "qc"
    outdir.mkdir()

    ok, message = grq._process_single_image(
        reference_path=reference,
        registered_path=registered,
        output_dir=outdir,
        scale_factor=0.5,
        save_fullres=False,
        save_png=True,
        save_tiff=True,
        logger=logging.getLogger("test"),
        nuclear_markers=["DAPI"],
    )

    assert ok is False
    assert message and message != "Success"
