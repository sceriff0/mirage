"""Foreground fraction: ``ref_fg`` and ``mov_fg`` on every control point, gated by nothing.

Both crops are measured because ``mov_fg / ref_fg`` separates wrong shifts from correct
ones better than ``mov_fg`` alone. On a dense synthetic sweep a foreground threshold
rejected about 60-65% of wrong shifts at no cost to correct tiles, but the distributions
overlap, so foreground alone is not a sufficient gate. The values are emitted so a future
gate can be measured on real slides and ``--out-tre`` can say why a tile was dropped; SOLVE
does not read them.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

np = pytest.importorskip("numpy")
pytest.importorskip("skimage")
tifffile = pytest.importorskip("tifffile")

import tiled_reg_tile  # noqa: E402
from tile_residual import foreground_fraction  # noqa: E402


def _tissue(seed=1234, n=192):
    """Sparse bright objects on a dim floor -- the shape real IF actually has.

    NOT a smooth filtered field. A smooth field is unimodal, and Otsu on a unimodal tile just
    splits it near the middle and reports ~0.50 -- which is what the first version of these
    tests used, and why they failed against a function that is behaving correctly. Modelled on
    the `_tissue_field` of the STARE v1 confidence-gate tests (retired with the gate).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.zeros((n, n), dtype=float)
    for _ in range(40):
        cy, cx = rng.integers(0, n, 2)
        field += 800.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0**2)))
    return field + rng.normal(30.0, 3.0, field.shape)


def test_a_blank_tile_reports_no_foreground_at_all():
    """Otsu ALWAYS returns a threshold, so a raw split reports ~0.51 here -- higher than real
    tissue's 0.087, and a "reject low foreground" rule would then keep background and drop
    tissue. The bimodality check is what makes this 0.0."""
    rng = np.random.default_rng(7)
    blank = rng.normal(30.0, 3.0, (192, 192))

    assert foreground_fraction(blank) == 0.0
    assert foreground_fraction(_tissue()) > 0.0


def test_a_smooth_unimodal_tile_reports_no_foreground():
    """The other unimodal case: a raw Otsu split reports ~0.50 and means nothing."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    smooth = gaussian_filter(rng.uniform(0, 1, size=(192, 192)), 3.0)

    assert foreground_fraction(smooth * 60000.0) == 0.0


def test_the_fraction_falls_monotonically_as_tissue_is_removed():
    """The property a gate would depend on, and that a raw Otsu split does NOT have."""
    rng = np.random.default_rng(5)
    tile = _tissue()
    values = []
    for blanked in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = tile.copy()
        cut = int(tile.shape[1] * blanked)
        if cut:
            t[:, :cut] = rng.normal(30.0, 3.0, (tile.shape[0], cut))
        values.append(foreground_fraction(t))

    assert values == sorted(values, reverse=True), values
    assert values[-1] == 0.0


def test_the_fraction_is_bounded_to_zero_one():
    for img in (_tissue(), np.random.default_rng(3).normal(30.0, 3.0, (64, 64))):
        value = foreground_fraction(img)
        assert 0.0 <= value <= 1.0


def test_a_constant_tile_reports_no_foreground():
    """Otsu has no threshold to find; 0.0 is the honest answer, not a crash."""
    assert foreground_fraction(np.full((32, 32), 42.0)) == 0.0


def test_an_all_zero_tile_reports_no_foreground():
    """The manufactured out-of-slide crop, which tiled_reg_tile.py:108 creates."""
    assert foreground_fraction(np.zeros((32, 32))) == 0.0


def test_blanking_half_a_tile_roughly_halves_its_foreground():
    """The signal has to actually track how much of the tile is off-section."""
    rng = np.random.default_rng(11)
    tile = _tissue()
    half = tile.copy()
    half[:, : tile.shape[1] // 2] = rng.normal(
        30.0, 3.0, (tile.shape[0], tile.shape[1] // 2)
    )

    full_fg = foreground_fraction(tile)
    half_fg = foreground_fraction(half)

    assert half_fg < full_fg
    assert half_fg == pytest.approx(full_fg / 2, rel=0.6)


# ---------------------------------------------------------------------------
# the control point carries the signal
# ---------------------------------------------------------------------------


def _run_tile(tmp_path, nuclear_index=0):
    img = _tissue(n=256).astype(np.uint16)
    stack = np.stack([img, img])
    ref_f, mov_f = tmp_path / "ref.ome.tiff", tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref_f), stack, photometric="minisblack")
    tifffile.imwrite(str(mov_f), stack, photometric="minisblack")

    m0 = tmp_path / "m0.json"
    m0.write_text(json.dumps({"M0": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]}))
    out = tmp_path / "ctrl.json"

    tiled_reg_tile.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--m0",
            str(m0),
            "--nuclear-index",
            str(nuclear_index),
            "--ix",
            "0",
            "--iy",
            "0",
            "--cx",
            "64",
            "--cy",
            "64",
            "--rx0",
            "0",
            "--ry0",
            "0",
            "--rx1",
            "128",
            "--ry1",
            "128",
            "--out",
            str(out),
        ]
    )
    return json.loads(out.read_text())


def test_the_control_point_carries_both_foreground_fractions(tmp_path):
    control = _run_tile(tmp_path)

    assert "ref_fg" in control
    assert "mov_fg" in control
    assert 0.0 <= control["ref_fg"] <= 1.0
    assert 0.0 <= control["mov_fg"] <= 1.0


def test_the_existing_control_point_keys_are_unchanged(tmp_path):
    """The fraction adds keys; it must not move or drop any."""
    control = _run_tile(tmp_path)

    for key in ("ix", "iy", "cx", "cy", "dx", "dy", "tre", "error"):
        assert key in control


def test_nothing_gates_on_the_foreground_fraction_yet():
    """The fraction is emitted, not gated on.

    The per-tile ``ref_fg``/``mov_fg`` must not change what SOLVE takes from a tile: a tile's
    acceptance (``stare.solve.tile_accepted``) reads only its vectors and the range gate.
    """
    from stare.solve import tile_accepted

    base = {
        "ix": 0,
        "iy": 0,
        "cx": 0.0,
        "cy": 0.0,
        "dx": 1.0,
        "dy": 0.0,
        "tre": 1.0,
        "error": 0.04,
        "lattice": {"stride": 128, "window": 256, "origin": 128},
        "vectors": [[0, 0, 128.0, 128.0, 1.0, 0.0, 5.0, 2.0, 1.0]],
        "rejected": [],
    }

    accepted_without = tile_accepted(dict(base), max_disp=256)
    accepted_with_zero = tile_accepted(dict(base, ref_fg=0.9, mov_fg=0.0), max_disp=256)

    assert accepted_without is True
    assert accepted_with_zero is True, (
        "a gate started reading mov_fg. That is Phase 2 work and it changes registration "
        "output -- it needs the dense-sweep acceptance and a real-slide before/after first."
    )
