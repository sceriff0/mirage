"""Tests for bin/preflight_scale.py -- the pre-flight metadata-only scale scan.

Four outcomes:

    pixel_size    metadata       result
    ----------    -----------    ------------------------------------------------
    auto          present        exit 0, report.json source="metadata"
    auto          absent         exit 1, message names EVERY unresolvable slide
    number        disagrees      exit 0, WARNING naming both numbers and the file
    number        absent         exit 0, WARNING naming the file(s) ("unconfirmed")

Only OME metadata is read -- these tests never call anything that decodes pixel data,
matching the script's own contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "bin" / "preflight_scale.py"


def _write_with_scale(tmp_path: Path, name: str, pixel_size_um: float = 0.5) -> Path:
    p = tmp_path / name
    tifffile.imwrite(
        p,
        np.zeros((4, 4), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": pixel_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size_um,
            "PhysicalSizeYUnit": "µm",
        },
    )
    return p


def _write_no_scale(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    tifffile.imwrite(p, np.zeros((4, 4), dtype=np.uint16))
    return p


def _run(args, tmp_path: Path):
    output = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    return proc.returncode, out, output


# ── auto + metadata present ─────────────────────────────────────────────────────
def test_auto_with_metadata_resolves_and_reports_source_metadata(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.5)
    rc, out, output = _run(["--images", str(a), "--pixel-size", "auto"], tmp_path)
    assert rc == 0, out
    report = json.loads(output.read_text())
    entry = report[str(a)]
    assert entry["source"] == "metadata"
    assert entry["pixel_size"] == pytest.approx(0.5, rel=1e-6)


# ── auto + no metadata: fail naming EVERY offender ──────────────────────────────
def test_auto_with_no_metadata_fails_and_names_every_offender(tmp_path):
    a, b = (
        _write_no_scale(tmp_path, "a.ome.tiff"),
        _write_no_scale(tmp_path, "b.ome.tiff"),
    )
    rc, out, output = _run(
        ["--images", str(a), str(b), "--pixel-size", "auto"], tmp_path
    )
    assert rc == 1
    assert "a.ome.tiff" in out and "b.ome.tiff" in out, (
        "the failure must name EVERY unresolvable slide, not just the first"
    )
    assert not output.exists()


def test_auto_mixed_metadata_still_names_only_the_offender(tmp_path):
    """One good slide plus one bad slide: fail, and the FAILURE message names only
    the bad one -- resolving the good slide may still be logged separately, but the
    offender list itself must not blame a slide that resolved fine."""
    good = _write_with_scale(tmp_path, "good.ome.tiff", 0.325)
    bad = _write_no_scale(tmp_path, "bad.ome.tiff")
    rc, out, _output = _run(
        ["--images", str(good), str(bad), "--pixel-size", "auto"], tmp_path
    )
    assert rc == 1
    error_line = next(
        line for line in out.splitlines() if "carry no usable OME" in line
    )
    assert "bad.ome.tiff" in error_line
    assert "good.ome.tiff" not in error_line


# ── number + disagreement: warn, exit 0 ─────────────────────────────────────────
def test_number_disagreeing_with_metadata_warns_but_succeeds(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.2125)
    rc, out, output = _run(["--images", str(a), "--pixel-size", "0.325"], tmp_path)
    assert rc == 0, out
    assert "SCALE MISMATCH" in out
    assert "0.2125" in out and "0.325" in out
    assert "a.ome.tiff" in out
    report = json.loads(output.read_text())
    assert report[str(a)] == {"pixel_size": 0.325, "source": "operator"}


# ── number + no metadata: warn "unconfirmed", exit 0 ────────────────────────────
def test_number_with_no_metadata_warns_unconfirmed_but_succeeds(tmp_path):
    a, b = (
        _write_no_scale(tmp_path, "a.ome.tiff"),
        _write_no_scale(tmp_path, "b.ome.tiff"),
    )
    rc, out, output = _run(
        ["--images", str(a), str(b), "--pixel-size", "0.325"], tmp_path
    )
    assert rc == 0, out
    assert "could not be verified" in out
    assert "a.ome.tiff" in out and "b.ome.tiff" in out
    report = json.loads(output.read_text())
    assert report[str(a)]["source"] == "operator"
    assert report[str(b)]["source"] == "operator"


# ── number + agreement: no warning at all ───────────────────────────────────────
def test_number_agreeing_with_metadata_is_silent(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.325)
    rc, out, output = _run(["--images", str(a), "--pixel-size", "0.325"], tmp_path)
    assert rc == 0, out
    assert "SCALE MISMATCH" not in out
    assert "could not be verified" not in out
    report = json.loads(output.read_text())
    assert report[str(a)] == {"pixel_size": 0.325, "source": "operator"}


# ── auto + two slides resolving to genuinely different scales: warn, exit 0 ────
def test_auto_heterogeneous_scales_warns_but_succeeds(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.325)
    b = _write_with_scale(tmp_path, "b.ome.tiff", 0.65)
    rc, out, output = _run(
        ["--images", str(a), str(b), "--pixel-size", "auto"], tmp_path
    )
    assert rc == 0, out
    assert "SCALE HETEROGENEITY" in out
    assert "0.325" in out and "0.65" in out
    assert "a.ome.tiff" in out and "b.ome.tiff" in out
    report = json.loads(output.read_text())
    assert report[str(a)]["pixel_size"] == pytest.approx(0.325, rel=1e-6)
    assert report[str(b)]["pixel_size"] == pytest.approx(0.65, rel=1e-6)


def test_auto_matching_scales_does_not_warn(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.325)
    b = _write_with_scale(tmp_path, "b.ome.tiff", 0.325)
    rc, out, _output = _run(
        ["--images", str(a), str(b), "--pixel-size", "auto"], tmp_path
    )
    assert rc == 0, out
    assert "SCALE HETEROGENEITY" not in out


def test_invalid_pixel_size_is_rejected(tmp_path):
    a = _write_no_scale(tmp_path, "a.ome.tiff")
    rc, out, output = _run(
        ["--images", str(a), "--pixel-size", "not-a-number"], tmp_path
    )
    assert rc == 1
    assert not output.exists()
