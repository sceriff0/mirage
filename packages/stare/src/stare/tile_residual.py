"""Per-tile refinement primitive for the STARE registration method.

After the global M0 anchor, a tile's moving crop and the reference crop differ only by a small,
near-translational residual. :func:`residual_displacement` recovers it sub-pixel with phase
correlation and returns it as ``(dx, dy)`` — the mesh control-point displacement for the tile,
i.e. the amount to add to a moving-tile coordinate to land it on the reference — together with the
tile's local Target Registration Error (the magnitude of that residual) and the correlation's
normalised error, which is how a caller tells a real match from a peak in noise.

Pure NumPy + scikit-image.
"""

from __future__ import annotations

import numpy as np

__all__ = ["residual_displacement", "foreground_fraction"]

# Below this between-class separation (in units of the pooled within-class spread) an Otsu split
# is not evidence of foreground -- see foreground_fraction. Measured gap on synthetic tiles:
# background 2.64-2.66, any-tissue 6.61-35.72.
BIMODAL_MIN_SEPARATION = 4.0


def foreground_fraction(tile):
    """Fraction of ``tile`` above its own Otsu threshold — how much of it is on-section.

    Emitted on every control point (``ref_fg`` / ``mov_fg``) and, as of this change, **gated on
    by nothing**. It exists so the next phase can be measured on real slides rather than
    synthetic tiles, and so ``--out-tre`` can eventually say WHY a tile was dropped.

    What it is worth, measured over the same dense sweep that pins the accepted-but-wrong band
    (21 blanking fractions x 4 geometries x 3 seeds; 200 accepted configs, 68 of them wrong):

        signal              threshold losing 0 correct    wrong tiles rejected
        mov_fg                      >= 0.0371                 41/68  (60.3%)
        mov_fg / ref_fg             >= 0.4165                 44/68  (64.7%)

    So it is a REAL separator and a PARTIAL one: ~60-65% of the exposure removed at no cost to
    real tissue, but the distributions still overlap (correct tiles reach down to 0.0371, wrong
    tiles up to 0.0624), so about a third survives any per-tile threshold. Neighbourhood
    consistency is still needed. The ratio beating ``mov_fg`` alone is why both crops are
    measured rather than only the moving one.

    Returns 0.0 for a constant tile — including the all-zeros crop ``tiled_reg_tile`` fabricates
    when the moving region falls outside the slide. Otsu has no threshold to find there, and 0.0
    is the honest answer rather than an exception.
    """
    from skimage.filters import threshold_otsu

    a = np.asarray(tile, dtype=float)
    if a.size == 0:
        return 0.0
    lo, hi = float(a.min()), float(a.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return 0.0

    threshold = threshold_otsu(a)
    below, above = a[a <= threshold], a[a > threshold]
    if below.size == 0 or above.size == 0:
        return 0.0

    # Otsu ALWAYS returns a threshold, including on a tile that has no foreground at all -- it
    # simply splits the distribution near its middle. Measured: a pure-background tile scores
    # 0.51 and a smooth unimodal field 0.50, i.e. HIGHER than real tissue's 0.087. Reporting
    # that raw would make the signal non-monotonic in how much of the tile is off-section (it
    # falls as tissue is removed, then jumps back to ~0.5 once the tile is pure background) and
    # a "reject low foreground" rule would keep background and drop tissue.
    #
    # So the split is only believed when the two classes are actually separated. Measured over
    # 255 tiles (the dense band sweep plus explicit background tiles):
    #     tiles with any tissue    separation 6.61 .. 35.72
    #     pure-background tiles    separation  2.64 ..  2.66
    # a gap of [2.66, 6.61]. The cut sits in it with margin both ways.
    #
    # CAVEAT: measured on synthetic tiles only. Real IF backgrounds are not Gaussian noise, so
    # re-measure before relying on this on real slides -- which is what Phase 1 emitting the
    # value, rather than gating on it, is for.
    within = np.sqrt((below.var() * below.size + above.var() * above.size) / a.size)
    separation = (above.mean() - below.mean()) / (within + 1e-12)
    if separation < BIMODAL_MIN_SEPARATION:
        return 0.0

    return float(above.size / a.size)


def residual_displacement(ref_tile, mov_tile, upsample=10, whiten_sigma=3.0):
    """Sub-pixel residual displacement aligning ``mov_tile`` to ``ref_tile``.

    Returns ``(dx, dy, tre, error)``:

    - ``(dx, dy)`` is the control-point displacement (add it to a moving coordinate to reach the
      reference);
    - ``tre = hypot(dx, dy)`` is the tile's local rigid-stage TRE — the pre-refinement
      misalignment magnitude used to gate whether the tile needs refining;
    - ``error`` is scikit-image's normalised correlation error, in ``[0, 1]``, where 0 is a
      perfect match and ~1 means the two crops share no structure at all.

    ``error`` is the *confidence* of the match and is not optional bookkeeping. Phase correlation
    always returns a peak, so a background tile or a tile straddling the section edge yields a
    displacement that is indistinguishable from a small real residual by magnitude — measured on
    the synthetic tiles in ``tests/test_tile_residual_confidence.py``, a tissue-vs-background tile
    scores ``|d| = 5.66 px`` (well inside ``reg_tiled_halo``) and only its ``error`` of 0.999961,
    against real tissue's 0.040956, gives it away. ``tiled_solve._grid_from_controls`` rejects
    control points above ``--max-error`` for exactly that reason.

    ``error`` is **NaN** where scikit-image cannot compute it at all — notably when either crop is
    empty, which is what ``tiled_reg_tile.py`` manufactures when the moving crop falls outside the
    slide. Callers must treat NaN as "reject", not as "small".

    The correlation kernel mirrors ASHLAR's: **whiten** each crop with a high-pass (subtract a
    Gaussian blur) to strip the low-frequency intensity mismatch between stains, then **Hann
    window** to taper the non-periodic tile edges the FFT would otherwise wrap. Plain (un-phase-
    normalised) cross-correlation is used deliberately — phase normalisation amplifies the
    high-frequency noise the whitening leaves and destabilises the sub-pixel peak on small tiles.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import window
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(ref_tile, dtype=float)
    mov = np.asarray(mov_tile, dtype=float)
    win = window("hann", ref.shape)
    ref_p = (ref - gaussian_filter(ref, whiten_sigma)) * win
    mov_p = (mov - gaussian_filter(mov, whiten_sigma)) * win
    # shift registers `mov` onto `ref`; axis order is (row, col) = (y, x)
    shift, error, _phasediff = phase_cross_correlation(
        ref_p, mov_p, upsample_factor=upsample, normalization=None
    )
    dy = float(shift[0])
    dx = float(shift[1])
    tre = float(np.hypot(dx, dy))
    return dx, dy, tre, float(error)
