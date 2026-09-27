"""stare.vector_grid: REG_TILE's window-vector estimator.

What each test pins (research/stare-optimal-design-2026-09-27.md §2 and
research/stare-sota-review-2026-09-27.md Part C §3, §6):

* sub-pixel accuracy per vector, with the 3-point Gaussian peak fit;
* the sign convention of the one-point-per-tile control (``tile_residual``);
* a smooth field recovered inside one full-size tile;
* the foreground rule (blank tile -> nothing; half-blank -> only the tissue half);
* the peak ratio separates a real match from noise;
* the global lattice is partitioned across tiles exactly once;
* pass 1 is REQUIRED at a +100 px residual offset (a single pass fails).
"""

from __future__ import annotations

import numpy as np
import pytest
from stare import vector_grid as vg
from stare.tile_grid import tile_grid
from stare.tile_residual import residual_displacement
from stare_synthetic import make_pair, nuclei, true_control


def _uniform(ux, uy):
    return lambda X, Y: (np.full_like(X, ux), np.full_like(Y, uy))


def _vectors(res):
    v = np.asarray(res["vectors"], dtype=float)
    assert v.ndim == 2 and v.shape[1] == 9, v.shape
    return v


def test_a_uniform_subpixel_shift_is_recovered_by_every_vector():
    ref, mov = make_pair(1024, _uniform(2.3, -1.7), noise=0.005)
    res = vg.estimate_tile_vectors(ref, mov, (0, 0), (0, 0, 1024, 1024), 128)
    v = _vectors(res)
    assert len(v) == 49, "7 x 7 owned nodes, all on tissue"
    # control convention: d = -u
    err = np.hypot(v[:, 4] + 2.3, v[:, 5] - 1.7)
    assert err.max() < 0.1, f"worst vector {err.max():.3f} px"


def test_the_sign_matches_the_one_point_per_tile_control():
    """The same pair through the old estimator and the new one: the same (dx, dy)."""
    ref, mov = make_pair(1024, _uniform(-3.4, 1.2), seed=3)
    dx_old, dy_old, _tre, _err = residual_displacement(ref, mov)
    v = _vectors(vg.estimate_tile_vectors(ref, mov, (0, 0), (0, 0, 1024, 1024), 128))
    assert np.median(v[:, 4]) == pytest.approx(dx_old, abs=0.1)
    assert np.median(v[:, 5]) == pytest.approx(dy_old, abs=0.1)
    # and both are the control convention d = -u (the old one quantised to 0.1 px)
    assert dx_old == pytest.approx(3.4, abs=0.15) and dy_old == pytest.approx(
        -1.2, abs=0.15
    )


def _smooth(X, Y):
    ux = 2.5 + 0.001 * (Y - 1280) + 3.0 * np.sin(2 * np.pi * Y / 1800)
    uy = -1.5 - 0.001 * (X - 1280) + 3.0 * np.cos(2 * np.pi * X / 1400)
    return ux, uy


def test_a_smooth_field_inside_one_full_size_tile_is_recovered():
    """2560^2 read box, 2048 core: per-vector median error < 0.3 px against the truth."""
    ref, mov = make_pair(2560, _smooth, seed=1)
    core = (256, 256, 2304, 2304)
    v = _vectors(vg.estimate_tile_vectors(ref, mov, (0, 0), core, 128))
    assert len(v) == 256
    truth = true_control(_smooth, v[:, 2:4])
    err = np.hypot(*(v[:, 4:6] - truth).T)
    assert np.median(err) < 0.3, f"median {np.median(err):.3f} px"


def test_a_blank_tile_emits_no_vector():
    rng = np.random.default_rng(0)
    blank = rng.normal(30.0, 3.0, (1024, 1024)).astype(np.float32)
    other = rng.normal(30.0, 3.0, (1024, 1024)).astype(np.float32)
    res = vg.estimate_tile_vectors(blank, other, (0, 0), (0, 0, 1024, 1024), 128)
    assert res["vectors"] == []
    assert len(res["rejected"]) == 49
    # rejected for foreground, never correlated: the peak ratio is null
    assert all(r[5] is None for r in res["rejected"])


def test_a_half_blank_tile_emits_only_its_tissue_windows():
    n = 1024
    mask = np.zeros((n, n), dtype=bool)
    mask[:, n // 2 :] = True
    ref, mov = make_pair(n, _uniform(1.0, 0.5), seed=2, mask=mask)
    v = _vectors(vg.estimate_tile_vectors(ref, mov, (0, 0), (0, 0, n, n), 128))
    # a window centred at cx spans [cx - 128, cx + 128): >= 25 % tissue needs cx >= 448
    assert len(v) > 0
    assert v[:, 2].min() >= 448, sorted(set(v[:, 2]))
    # and every node well inside the tissue half is kept
    assert set(v[:, 2]) >= {640.0, 768.0, 896.0}


def test_the_peak_ratio_is_low_on_noise_and_high_on_a_match():
    rng = np.random.default_rng(5)
    W = 256
    n = 16
    from skimage.filters import window

    han = window("hann", (W, W)).astype(np.float32)
    noise_r = np.stack([vg._whiten(rng.normal(0, 1, (W, W))) for _ in range(n)]) * han
    noise_m = np.stack([vg._whiten(rng.normal(0, 1, (W, W))) for _ in range(n)]) * han
    pr_noise = vg.correlate_windows(noise_r, noise_m)[:, 2]
    tiles = [vg._whiten(nuclei(W, rng)) for _ in range(n)]
    match_r = np.stack(tiles) * han
    match_m = np.stack([np.roll(t, (2, -3), axis=(0, 1)) for t in tiles]) * han
    out = vg.correlate_windows(match_r, match_m)
    assert np.median(pr_noise) < vg.PEAK_RATIO_MIN, pr_noise
    assert np.all(out[:, 2] > 2.0), out[:, 2]
    # np.roll by (+2 rows, -3 cols): M(x) = R(x - (-3, 2)) -> d = (-3, +2)... and the
    # control convention is R(x) = M(x - d), i.e. d = (+3, -2)
    assert np.allclose(np.median(out[:, :2], axis=0), [3.0, -2.0], atol=0.1)


@pytest.mark.parametrize(
    "width,height,tile,stride",
    [
        (5000, 3000, 2048, 128),  # tile not a multiple of the image, short last tile
        (4100, 777, 1000, 128),  # tile not divisible by stride
        (2048, 2048, 2048, 100),  # one tile
        (3333, 2500, 777, 64),  # odd everything
    ],
)
def test_the_lattice_is_partitioned_across_tiles_exactly_once(
    width, height, tile, stride
):
    owned = []
    for t in tile_grid(width, height, tile, 256):
        kxs, kys = vg.owned_nodes(t.core, stride)
        for ky in kys:
            for kx in kxs:
                cx, cy = stride * (kx + 1), stride * (ky + 1)
                assert t.core[0] <= cx < t.core[2] and t.core[1] <= cy < t.core[3]
                owned.append((kx, ky))
    assert len(owned) == len(set(owned)), "a node is owned by two tiles"
    # every node whose centre lies on the slide is owned by someone
    all_nodes = {
        (kx, ky)
        for kx in range((width - 1) // stride)
        for ky in range((height - 1) // stride)
    }
    assert set(owned) == all_nodes


def test_the_read_box_covers_every_owned_window_plus_the_capture_margin():
    core, stride = (2048, 2048, 4096, 4096), 128
    assert vg.read_box(core, stride, (10_000, 10_000)) == (1664, 1664, 4480, 4480)
    # clamped at the slide edge; image_shape is (height, width)
    assert vg.read_box((0, 0, 2048, 2048), stride, (2500, 2300)) == (0, 0, 2300, 2432)


def test_pass_one_is_required_for_a_100_px_residual():
    """+100 px left over from COARSE: two passes recover it, a single pass does not.

    Measured on this pair: two passes put a correct vector (< 1 px) on ~all 64 owned nodes;
    a single pass keeps ~29, of which a third are confident-but-wrong by 80-170 px -- the
    peak ratio gate cannot see them, and SOLVE then has to fight them
    (research/stare-sota-review-2026-09-27.md Part C §6).
    """
    shift = _uniform(100.3, -50.2)
    ref, mov = make_pair(1792, shift, seed=4)
    core = (384, 384, 1408, 1408)
    n_owned = 64

    def correct(res):
        v = np.asarray(res["vectors"], dtype=float).reshape(-1, 9)
        err = np.hypot(v[:, 4] + 100.3, v[:, 5] - 50.2)
        return int((err < 1.0).sum()), int((err > 20.0).sum())

    ok2, bad2 = correct(vg.estimate_tile_vectors(ref, mov, (0, 0), core, 128))
    assert ok2 >= 0.9 * n_owned and bad2 == 0, (ok2, bad2)

    ok1, bad1 = correct(
        vg.estimate_tile_vectors(ref, mov, (0, 0), core, 128, two_pass=False)
    )
    assert ok1 < 0.5 * n_owned, (ok1, bad1)
    assert bad1 >= 1, "premise: a single pass emits confident-but-wrong vectors"
