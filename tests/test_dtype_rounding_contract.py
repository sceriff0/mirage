"""`.astype()` floors. Every float->integer cast in this repo must round first.

`bin/apply_basic_profiles.py` states the convention and explains it (it inherited the
rule from the deleted `bin/preprocess.py`, which stated it first):

    * **The storage dtype rule.** Clip to the dtype's range, ROUND (half-to-even),
      then cast. ... astype() alone floors, a systematic -0.5 LSB bias.

Three sites did not follow it, and two of them write published artefacts:

  * `merge_channels_pyramid.write_pyramidal_ome_tiff` -- the QuPath-facing pyramid;
  * `merge_channels_pyramid._load_channel_stack` -- the same cast, same file, 250 lines apart;
  * `generate_preprocess_qc.normalize_image` -- the 8-bit QC render.

Truncation is not a rounding-mode quibble. It is a systematic *downward* bias of up to one
count on every pixel, mean -0.5 LSB, applied to intensity data that later becomes a measured
per-cell statistic. It never averages out, because it is one-sided.

On the 8-bit QC path it is worse than a bias: with `(img * 255).astype(np.uint8)` only an
input of exactly 1.0 reaches 255, so the top bin is 1/255 as wide as every other bin and the
brightest tissue is systematically reported one level dark.

The float->uint16 rule now has ONE owner, `merge_channels_pyramid.to_uint16`, because the two
sites that drifted from the convention drifted together while sitting in the same file.
"""

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
tifffile = pytest.importorskip("tifffile")

import generate_preprocess_qc as gpq  # noqa: E402
import merge_channels_pyramid as mcp  # noqa: E402

# ---------------------------------------------------------------------------
# The float -> uint16 rule, as a function
# ---------------------------------------------------------------------------


def test_to_uint16_rounds_to_nearest_rather_than_truncating():
    """100.7 must become 101. astype alone gives 100 on every pixel that is not exact."""
    data = np.array([[0.4, 0.6, 100.7, 100.2]], dtype=np.float32)

    out = mcp.to_uint16(data)

    assert out.tolist() == [[0, 1, 101, 100]]


def test_to_uint16_uses_half_to_even_like_the_documented_convention():
    """apply_basic_profiles.py says half-to-even; a different tie rule here would be a second convention."""
    data = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float64)

    out = mcp.to_uint16(data)

    assert out.tolist() == [[0, 2, 2, 4]]


def test_to_uint16_still_clips_both_ends():
    """Rounding must not cost the clip -- 65535.6 would wrap to 0 without it."""
    data = np.array([[-5.0, 65535.6, 70000.0]], dtype=np.float64)

    out = mcp.to_uint16(data)

    assert out.tolist() == [[0, 65535, 65535]]
    assert out.dtype == np.uint16


def test_to_uint16_returns_integer_data_unchanged():
    """The helper is the one owner of the rule, so it must be safe to call on already-integer data."""
    data = np.array([[7, 8]], dtype=np.uint16)

    out = mcp.to_uint16(data)

    assert out.tolist() == [[7, 8]]
    assert out.dtype == np.uint16


def test_to_uint16_does_not_bias_a_uniform_field_downward():
    """The point of the fix: truncation is one-sided, so it never averages out."""
    rng = np.random.default_rng(20260816)
    data = rng.uniform(1000.0, 2000.0, size=(4, 64, 64))

    out = mcp.to_uint16(data)

    # Mean absolute error under rounding is <= 0.5; under truncation the *signed* mean error
    # is about -0.5, which is what this asserts against.
    assert abs(float(out.mean()) - float(data.mean())) < 0.05


# ---------------------------------------------------------------------------
# The published pyramid actually uses it
# ---------------------------------------------------------------------------


def test_the_published_pyramid_rounds_rather_than_truncates(tmp_path):
    """The QuPath-facing deliverable is the highest-stakes of the three sites."""
    data = np.full((2, 64, 64), 100.7, dtype=np.float32)
    out = tmp_path / "merged.ome.tiff"

    mcp.write_pyramidal_ome_tiff(
        data,
        str(out),
        channel_names=["DAPI", "CD3"],
        channel_colors=[(255, 0, 0), (0, 255, 0)],
        pyramid_resolutions=1,
    )

    written = tifffile.imread(str(out))

    assert int(np.asarray(written).ravel()[0]) == 101


# ---------------------------------------------------------------------------
# The 8-bit QC render
# ---------------------------------------------------------------------------


def test_normalize_image_rounds_to_nearest():
    """Midway up the range is 127.5: truncation gives 127, rounding gives 128."""
    # percentile_low=0 / high=100 makes the mapping exactly (v - min) / (max - min), so the
    # midpoint lands on a genuine .5 and the two behaviours differ. A value like 0.6 would
    # NOT discriminate -- 0.6 * 255 is exactly 153.0 and both rules agree.
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    out = gpq.normalize_image(image, percentile_low=0.0, percentile_high=100.0)

    assert out.tolist() == [[0, 128, 255]]


def test_normalize_image_top_bin_is_not_starved():
    """With truncation only an exact 1.0 reaches 255; the brightest tissue reads one level dark.

    The input must NOT be aligned to the 8-bit quantisation grid. `linspace(0, 1, 256)` gives
    exactly i/255, so every value scales to a whole integer and rounding and truncation agree --
    that input cannot see this bug at all. A dense off-grid ramp can.

    Under truncation, `out == 255` holds only for v == 1.0 exactly: one pixel, a bin of measure
    zero. Under rounding it holds for v >= 254.5/255, a bin of the usual half-width at the
    endpoint.
    """
    image = np.linspace(0.0, 1.0, 10001, dtype=np.float64).reshape(1, 10001)

    out = gpq.normalize_image(image, percentile_low=0.0, percentile_high=100.0)

    assert int((out == 255).sum()) > 1


def test_normalize_image_still_returns_uint8():
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    out = gpq.normalize_image(image, percentile_low=0.0, percentile_high=100.0)

    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# The two halves of one phase correlation, twelve lines apart
# ---------------------------------------------------------------------------
#
# bin/tiled_reg_tile.py read the reference tile as float32 and the moving crop as `dtype=float`
# (= float64). scikit-image promotes the pair internally, so the split cost float64 memory on
# the two largest arrays in the task -- the warped moving tile and its source crop -- and bought
# nothing: the source data is uint16, and float32 carries 24 mantissa bits.
#
# Measured over 6 seeded 384x384 tiles with a known (-3.4, +2.7) shift, comparing float32/float64
# against float32/float32:
#   recovered shift  identical to every printed digit (delta = 0.000e+00; upsample=10 quantises
#                    to 0.1 px, orders of magnitude coarser than the difference)
#   correlation error  differs by 7e-11 to 1.3e-09, against a gate threshold of 0.99
# So unifying is inert where it is read, which is what keeps this change out of registration
# output.


def test_both_halves_of_the_phase_correlation_arrive_at_the_same_precision(
    tmp_path, monkeypatch
):
    """A split-precision correlation is a smell that costs 2x memory on the biggest arrays."""
    import json

    import tiled_reg_tile
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    img = gaussian_filter(rng.uniform(0, 1, size=(256, 256)), 3.0)
    img = ((img - img.min()) / np.ptp(img) * 60000).astype(np.uint16)
    stack = np.stack([img, img])

    ref_f = tmp_path / "ref.ome.tiff"
    mov_f = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref_f), stack, photometric="minisblack")
    tifffile.imwrite(str(mov_f), stack, photometric="minisblack")

    m0_f = tmp_path / "m0.json"
    m0_f.write_text(
        json.dumps({"M0": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]})
    )

    seen = {}
    # the correlation's two halves enter at the window-vector estimator (stare.vector_grid)
    real = tiled_reg_tile.estimate_tile_vectors

    def capture(ref_tile, mov_tile, *args, **kw):
        seen["ref"] = ref_tile.dtype
        seen["mov"] = mov_tile.dtype
        return real(ref_tile, mov_tile, *args, **kw)

    monkeypatch.setattr(tiled_reg_tile, "estimate_tile_vectors", capture)

    tiled_reg_tile.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--m0",
            str(m0_f),
            "--nuclear-index",
            "0",
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
            str(tmp_path / "ctrl.json"),
        ]
    )

    assert seen["ref"] == seen["mov"], (
        f"phase correlation received {seen['ref']} against {seen['mov']} -- "
        "the two halves of one measurement are at different precision"
    )


def test_the_correlation_runs_at_float32_not_float64():
    """Direction matters: unifying UP to float64 would double memory on the two largest arrays."""

    import tiled_reg_tile

    assert tiled_reg_tile.TILE_DTYPE == np.float32
