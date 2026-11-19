"""The shipped-defaults smoke fixture: the one image pair with a real OME scale.

Every other fixture `tests/testdata/generate_complete_testdata.py` writes is
scale-less, so `--pixel_size auto` (the shipped default) hard-fails at
PREFLIGHT_SCALE against them (see `conf/test.config`).
`tests/testdata/P900_ref_scaled.ome.tiff` and `tests/testdata/P900_mov_scaled.ome.tiff`
(also used by `tests/testdata/shipped_defaults_input.csv`) carry a real
`PhysicalSizeX`/`PhysicalSizeY`, so `auto` resolves for them.

`tests/test_preflight_scale.py` covers the same code on synthetic tmp_path images;
this covers the checked-in fixture the shipped-defaults CI job uses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TESTDATA = REPO / "tests" / "testdata"
PREFLIGHT_SCRIPT = REPO / "bin" / "preflight_scale.py"

sys.path.insert(0, str(REPO / "bin" / "utils"))
from pixel_size import read_ome_pixel_size  # noqa: E402

FIXTURES = [
    TESTDATA / "P900_ref_scaled.ome.tiff",
    TESTDATA / "P900_mov_scaled.ome.tiff",
]
SAMPLESHEET = TESTDATA / "shipped_defaults_input.csv"


def _skip_if_not_generated():
    missing = [str(p) for p in (*FIXTURES, SAMPLESHEET) if not p.exists()]
    if missing:
        pytest.skip(
            "shipped-defaults fixture(s) not generated -- run "
            "tests/testdata/generate_complete_testdata.py first: " + ", ".join(missing)
        )


def test_fixtures_carry_a_real_ome_physical_size():
    _skip_if_not_generated()
    for path in FIXTURES:
        det_x, det_y = read_ome_pixel_size(path)
        assert det_x == pytest.approx(0.325, rel=1e-6), path
        assert det_y == pytest.approx(0.325, rel=1e-6), path


def test_samplesheet_names_both_fixtures():
    _skip_if_not_generated()
    text = SAMPLESHEET.read_text()
    for path in FIXTURES:
        assert path.name in text, f"{SAMPLESHEET} does not reference {path.name}"


def test_preflight_scale_resolves_auto_against_the_real_fixture(tmp_path):
    """The actual entry point PREFLIGHT_SCALE uses, against the checked-in fixture
    (not a synthetic tmp_path image), resolving under the literal shipped default."""
    _skip_if_not_generated()
    output = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_SCRIPT),
            "--images",
            *[str(p) for p in FIXTURES],
            "--pixel-size",
            "auto",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.exists()
