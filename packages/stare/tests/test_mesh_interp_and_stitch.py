"""The mesh interpolant, the stitch's sub-grid inverse map, and the QC-seam <-> stitch round trip.

What each test pins:

* ``"interp": "cubic"`` in a mesh spec is an interpolating cubic B-spline; an absent key is the
  bilinear field every earlier manifest was written with;
* the QC seam (``stage_warp.make_warper``) and the stitch build their field through ONE
  constructor (``MeshField.from_spec``), so they cannot sample different fields;
* the stitch's sub-grid inverse map (every ``FIELD_STEP`` px, bilinearly upsampled) is within
  0.01 px of the exact per-pixel inverse, and within 0.02 px of the QC seam's exact forward
  map, at 10k random pixels; the ``(h^2/8)(max|u_xx| + max|u_yy|)`` bound is < 0.01 px on
  the field;
* ``resample_bilinear`` (now ``map_coordinates(order=1)``) equals the explicit four-tap form;
* a moving point pushed through the QC seam's ``refined`` stage and pulled back out of the
  STITCHED image lands on itself within 0.05 px, for both interpolants.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from stare import solve, warp
from stare.manifest import build_manifest, slide_entry
from stare.mesh_field import MeshField, resample_bilinear
from stare.stage_warp import STAGE_REFINED, make_warper
from stare.stages import stitch
from stare.warp import _invert, new_inverse_stats, source_coords

S = 128
TH = np.radians(0.2)
M0 = np.array(
    [[np.cos(TH), -np.sin(TH), 6.0], [np.sin(TH), np.cos(TH), -4.0], [0.0, 0.0, 1.0]]
)


def _truth(X, Y, c=1024.0):
    """A SOLVE-like residual: offset, 0.15 deg of rotation, two waves (the P2 e2e's)."""
    th = np.radians(0.15)
    dx = (np.cos(th) - 1) * (X - c) - np.sin(th) * (Y - c) + 20.0
    dx = dx + 3.0 * np.sin(2 * np.pi * Y / 3000) + 1.5 * np.sin(2 * np.pi * X / 1500)
    dy = np.sin(th) * (X - c) + (np.cos(th) - 1) * (Y - c) - 10.0
    dy = dy + 3.0 * np.cos(2 * np.pi * X / 3000) + 1.5 * np.cos(2 * np.pi * Y / 1500)
    return dx, dy


def _solved(n, interp, seed=0, noise=0.1):
    """``solve_dctpls`` on an ``n x n`` vector lattice of the truth plus noise."""
    rng = np.random.default_rng(seed)
    X = S * (np.arange(n) + 1.0)
    vecs = []
    for ky in range(n):
        for kx in range(n):
            dx, dy = _truth(X[kx], X[ky])
            vecs.append(
                [
                    kx,
                    ky,
                    X[kx],
                    X[ky],
                    dx + rng.normal(0, noise),
                    dy + rng.normal(0, noise),
                    5.0,
                    2.0,
                    1.0,
                ]
            )
    control = {
        "ix": 0,
        "iy": 0,
        "cx": 0.0,
        "cy": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "tre": 0.0,
        "error": 0.05,
        "lattice": {"stride": S, "window": 2 * S, "origin": S},
        "vectors": vecs,
        "rejected": [],
    }
    gx, gy, disp, report = solve.solve_dctpls([control], max_disp=256, interp=interp)
    return gx, gy, disp, report


def _manifest(gx, gy, disp, interp, out_shape):
    entry = slide_entry(M0, gx, gy, disp, interp=interp)
    entry["out_shape"] = list(out_shape)
    return build_manifest("ref", {"ref": slide_entry(np.eye(3)), "mov": entry})


# ── the interpolant ─────────────────────────────────────────────────────────────
def test_cubic_passes_through_every_node_even_on_a_short_last_column():
    gx = [64.0, 192.0, 320.0, 400.0]  # the per-tile grid's short last tile
    gy = [0.0, 100.0, 200.0]
    d = np.random.default_rng(0).normal(size=(3, 4, 2))
    m = MeshField(gx, gy, d, interp="cubic")
    nodes = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    assert np.abs(m.displacement(nodes) - d.reshape(-1, 2)).max() < 1e-9
    # and clamps past the outermost nodes like the bilinear field
    far = np.array([[-500.0, -500.0], [5000.0, 5000.0]])
    assert np.allclose(m.displacement(far), [d[0, 0], d[-1, -1]])


def test_cubic_reproduces_a_smooth_field_between_nodes_better_than_bilinear():
    g = S * (np.arange(24) + 1.0)
    GX, GY = np.meshgrid(g, g)
    D = np.stack(_truth(GX, GY), axis=-1)
    pts = np.random.default_rng(1).uniform(g[2], g[-3], (5000, 2))
    true = np.stack(_truth(pts[:, 0], pts[:, 1]), axis=1)
    err = {
        k: np.abs(MeshField(g, g, D, interp=k).displacement(pts) - true).max()
        for k in ("bilinear", "cubic")
    }
    # measured 0.015 vs 0.079 px worst of 5000
    assert err["cubic"] < err["bilinear"] / 4, err


def test_an_absent_interp_key_is_bilinear_and_an_unknown_one_is_refused():
    spec = {
        "grid_x": [0.0, 10.0],
        "grid_y": [0.0, 10.0],
        "displacements": [[[0, 0]] * 2] * 2,
    }
    assert MeshField.from_spec(spec).interp == "bilinear"
    assert MeshField.from_spec(None) is None
    assert MeshField.from_spec({**spec, "interp": "cubic"}).interp == "cubic"
    with pytest.raises(ValueError, match="unknown mesh interp"):
        MeshField.from_spec({**spec, "interp": "lanczos"})


def test_a_bilinear_manifest_carries_no_interp_key_and_a_cubic_one_does():
    gx, gy, d = [0.0, 10.0], [0.0, 10.0], [[[1.0, 0.0]] * 2] * 2
    assert "interp" not in slide_entry(np.eye(3), gx, gy, d)["mesh"]
    assert "interp" not in slide_entry(np.eye(3), gx, gy, d, interp="bilinear")["mesh"]
    assert (
        slide_entry(np.eye(3), gx, gy, d, interp="cubic")["mesh"]["interp"] == "cubic"
    )


def test_solve_writes_the_vector_mesh_cubic():
    _gx, _gy, _d, report = _solved(12, solve.VECTOR_MESH_INTERP)
    assert solve.VECTOR_MESH_INTERP == "cubic"
    assert report["mesh_interp"] == "cubic"


@pytest.mark.parametrize("interp", ["bilinear", "cubic"])
def test_the_qc_seam_and_the_stitch_read_one_field(interp):
    gx, gy, disp, _r = _solved(12, interp)
    man = json.loads(json.dumps(_manifest(gx, gy, disp, interp, (1600, 1600))))
    mesh, _margin = stitch._mesh_and_margin(man["slides"]["mov"])
    assert mesh.interp == interp
    p = np.random.default_rng(2).uniform(100, 1500, (500, 2))
    v = p @ M0[:2, :2].T + M0[:2, 2]
    qc = make_warper(man)("mov", p, STAGE_REFINED)
    assert np.array_equal(qc, v + mesh.displacement(v))


# ── the stitch's sub-grid inverse map ──────────────────────────────────────────
@pytest.mark.parametrize("interp", ["bilinear", "cubic"])
def test_the_sub_grid_inverse_is_within_bound_of_the_exact_field(interp):
    """10k random output pixels: sub-grid vs exact < 0.01 px; vs the QC seam < 0.02 px."""
    n = 30
    gx, gy, disp, _r = _solved(n, interp)
    mesh = MeshField(gx, gy, disp, interp=interp)
    h = stitch.FIELD_STEP
    if interp == "cubic":
        # the interpolation bound h^2/8 (max|u_xx| + max|u_yy|), from the field itself
        f = np.arange(gx[0], gx[-1], 2.0)
        FX, FY = np.meshgrid(f, f)
        U = mesh.displacement(np.column_stack([FX.ravel(), FY.ravel()]))
        U = U.reshape(f.size, f.size, 2)
        uxx = np.abs(np.diff(U, 2, axis=1)).max() / 4.0
        uyy = np.abs(np.diff(U, 2, axis=0)).max() / 4.0
        assert h**2 / 8 * (uxx + uyy) < 0.01, (uxx, uyy)

    extent = int(S * (n + 1))
    q = np.random.default_rng(3).integers(0, extent, (10000, 2))
    stitched = np.array(
        [
            source_coords(M0, mesh, (1, 1), (int(x), int(y)), field_step=h)[0, 0]
            for x, y in q
        ]
    )
    exact = _invert(M0, mesh, q.astype(float))
    sub = np.hypot(*(stitched - exact).T)
    # a bilinear mesh adds h |jump u'| / 4 at its cell edges (0.013 px measured): only the
    # cubic field (the one SOLVE writes) is held to the h^2/8 figure
    assert sub.max() < (0.01 if interp == "cubic" else 0.02), (interp, sub.max())
    # the QC seam's field is the exact forward map; the stitch's inverse must invert it
    man = _manifest(gx, gy, disp, interp, (extent, extent))
    back = make_warper(man)("mov", stitched, STAGE_REFINED)
    seam = np.hypot(*(back - q).T)
    assert seam.max() < 0.02, (interp, seam.max())


def test_a_window_of_the_sub_grid_map_equals_the_same_pixels_of_a_larger_window():
    """Global sub-grid nodes: output tiles share nodes, so there is no seam at a tile edge."""
    gx, gy, disp, _r = _solved(12, "cubic")
    mesh = MeshField(gx, gy, disp, interp="cubic")
    big = source_coords(M0, mesh, (300, 260), (40, 70), field_step=8)
    small = source_coords(M0, mesh, (37, 53), (101, 133), field_step=8)
    assert np.abs(big[133 - 70 : 170 - 70, 101 - 40 : 154 - 40] - small).max() < 1e-9


# ── the inverse map runs to a tolerance ───────────────────────────────────────
def _contraction_mesh(lip, offset=(100.0, 0.0), ctr=512.0):
    """A bilinear lattice holding F(v) = offset + lip * rot90 (v - ctr): Lipschitz ``lip``.

    Linear, so the bilinear mesh is exact inside the lattice and the fixed point has a closed
    form: (I + B) v = u - offset + B ctr. The skew B contracts the iteration at exactly
    ``lip`` per step -- the worst case the certificate allows at ``lip`` < 0.5.
    """
    g = np.arange(-512.0, 1536.0 + 1, 64.0)
    GX, GY = np.meshgrid(g, g)
    B = lip * np.array([[0.0, -1.0], [1.0, 0.0]])
    rel = np.stack([GX - ctr, GY - ctr], axis=-1)
    disp = np.asarray(offset) + rel @ B.T
    return MeshField(g, g, disp, interp="bilinear"), B, np.asarray(offset), ctr


def _fixed_point(u, B, offset, ctr):
    rhs = u - offset + (B @ np.array([ctr, ctr]))
    return np.linalg.solve(np.eye(2) + B, rhs.T).T


def test_the_inverse_of_a_certificate_edge_field_converges_to_tolerance():
    """L = 0.45, |F| ~ 100 px: inside SOLVE's fold certificate (L < 0.5). Three fixed-point
    steps (the pre-2026-09-27 hard-coded count) leave ~0.45^3 x 100 = 9 px; the loop runs to
    the tolerance and records how far it got (field NOTES F2c)."""
    mesh, B, off, ctr = _contraction_mesh(0.45)
    m0 = np.eye(3)
    out = source_coords(m0, mesh, (32, 32), (480, 490))
    ys, xs = np.mgrid[490:522, 480:512]
    u = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    err = np.hypot(*(out.reshape(-1, 2) - _fixed_point(u, B, off, ctr)).T)
    assert err.max() < 1e-2, err.max()
    stats = new_inverse_stats()
    _invert(m0, mesh, u, stats=stats)
    assert stats["inverse_cap_hits"] == 0
    assert stats["inverse_residual_px"] < warp.INVERSE_TOL_PX
    assert stats["inverse_iterations_max"] > 3


def test_a_small_field_still_inverts_in_a_few_iterations():
    """SOLVE's own meshes (L ~ 0.02) stop long before the cap: a handful of steps."""
    gx, gy, disp, _r = _solved(12, "cubic")
    mesh = MeshField(gx, gy, disp, interp="cubic")
    stats = new_inverse_stats()
    source_coords(M0, mesh, (256, 256), (300, 300), field_step=8, stats=stats)
    assert stats["inverse_cap_hits"] == 0
    assert stats["inverse_iterations_max"] <= 5, stats


def test_a_field_outside_the_certificate_hits_the_cap_and_says_so():
    """L > 1 is not a contraction: the loop stops at the cap and records it, never hangs."""
    mesh, *_ = _contraction_mesh(1.2, offset=(10.0, 0.0))
    stats = new_inverse_stats()
    _invert(np.eye(3), mesh, np.array([[500.0, 520.0]]), stats=stats)
    assert stats["inverse_cap_hits"] == 1
    assert stats["inverse_iterations_max"] == warp.INVERSE_MAX_ITERATIONS
    assert stats["inverse_residual_px"] >= warp.INVERSE_TOL_PX


def test_the_stitch_warns_when_the_inverse_hit_the_cap(caplog):
    """stream_tiles accumulates the record over tiles; a capped inverse is a WARNING.

    L = 1 exactly (a skew): the fixed-point error neither shrinks nor grows, so the source
    boxes stay on the image and every tile is warped -- and none converges.
    """
    import logging

    mesh, *_ = _contraction_mesh(1.0, offset=(10.0, 0.0), ctr=32.0)
    src = np.ones((1, 64, 64), np.float32)
    stats = new_inverse_stats()
    list(
        stitch.stream_tiles(
            src, np.eye(3), mesh, 8, 64, 64, 32, np.float32, stats=stats
        )
    )
    assert stats["inverse_calls"] == 4 and stats["inverse_cap_hits"] == 4
    with caplog.at_level(logging.INFO):
        stitch.log_inverse(stats)
    assert any(
        r.levelno == logging.WARNING and "cap" in r.message for r in caplog.records
    )
    good = new_inverse_stats()
    mesh, *_ = _contraction_mesh(0.1, offset=(10.0, 0.0), ctr=32.0)
    list(
        stitch.stream_tiles(src, np.eye(3), mesh, 8, 64, 64, 32, np.float32, stats=good)
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        stitch.log_inverse(good)
    assert good["inverse_calls"] == 4 and good["inverse_cap_hits"] == 0
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


# ── the resampler ──────────────────────────────────────────────────────────────
def _explicit_bilinear(image, xy):
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0], 0.0, w - 1.0)
    y = np.clip(xy[:, 1], 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = x - x0, y - y0
    if image.ndim == 3:
        tx, ty = tx[:, None], ty[:, None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bot = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bot * ty


@pytest.mark.parametrize("shape", [(90, 70), (90, 70, 3)])
def test_resample_equals_the_explicit_four_tap_form(shape):
    rng = np.random.default_rng(4)
    image = rng.uniform(0, 60000, shape)
    xy = rng.uniform(-3, 95, (20000, 2))
    got = resample_bilinear(image, xy)
    assert got.shape == ((20000,) if len(shape) == 2 else (20000, 3))
    assert np.abs(got - _explicit_bilinear(image, xy)).max() < 1e-4


# ── the round trip: QC seam forward, stitched image back ───────────────────────
@pytest.mark.parametrize("interp", ["bilinear", "cubic"])
def test_a_point_through_the_qc_seam_comes_back_out_of_the_stitched_image(
    interp, tmp_path
):
    """Stitch a moving image whose pixels ARE their coordinates, then read it at the QC point.

    Channel 0 holds x and channel 1 holds y. Bilinear resampling of a linear ramp is exact,
    so the stitched image at reference pixel ``u`` holds the moving point the stitch drew
    ``u`` from. Push a moving point ``p`` through ``make_warper``'s refined stage to ``u``,
    read the stitched ramps at ``u``, and ``p`` must come back.
    """
    tifffile = pytest.importorskip("tifffile")
    n, size = 15, 2048
    gx, gy, disp, report = _solved(n, interp)
    assert report["mesh_interp"] == interp
    man = _manifest(gx, gy, disp, interp, (size, size))
    man_f = tmp_path / "manifest.json"
    man_f.write_text(json.dumps(man))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mov_f = tmp_path / "ramps.ome.tiff"
    tifffile.imwrite(str(mov_f), np.stack([xx, yy]), photometric="minisblack")
    out_f = tmp_path / "stitched.ome.tiff"
    stitch.main(
        [
            "--moving",
            str(mov_f),
            "--manifest",
            str(man_f),
            "--moving-name",
            "mov",
            "--out",
            str(out_f),
            "--out-tile",
            "512",
            "--pixel-size",
            "0.5",
        ]
    )
    stitched = tifffile.imread(str(out_f)).astype(float)

    p = np.random.default_rng(5).uniform(300, size - 300, (2000, 2))
    u = make_warper(man)("mov", p, STAGE_REFINED)
    back = np.stack(
        [map_coordinates(stitched[k], [u[:, 1], u[:, 0]], order=1) for k in range(2)],
        axis=1,
    )
    err = np.hypot(*(back - p).T)
    assert err.max() < 0.05, (interp, err.max())
