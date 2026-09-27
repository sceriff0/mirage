"""End-to-end STARE registration of one moving slide against the reference, in process.

``register_slide`` composes the method's pieces in the order the Nextflow DAG runs them, but
in a single call so it is unit-testable and reusable by the ``bin/`` CLIs:

    coarse_align.estimate_rigid        -> global rigid M0 (moving -> reference)
    warp.warp_image (rigid)            -> the moving slide pre-warped into the reference frame
    tile_grid + vector_grid            -> per tile, a grid of window vectors on the global lattice
    solve.solve_dctpls                 -> the mesh (robust affine + robust DCT-PLS)
    manifest.slide_entry               -> the manifest entry
    warp.warp_image (mesh)             -> the registered slide (non-negative)

Registration is estimated from the DAPI channel; the same transform can warp any channel image.
Pure NumPy + SciPy + scikit-image -- no VALIS, no JVM.

Memory note: this reference implementation rigid-pre-warps the whole slide for clarity. The
Nextflow REG_TILE/STITCH path does the equivalent tile-by-tile to hold peak memory to one tile.
"""

from __future__ import annotations

import numpy as np

from stare.coarse_align import estimate_rigid
from stare.manifest import slide_entry
from stare.mesh_field import MeshField
from stare.solve import solve_dctpls
from stare.tile_grid import tile_grid
from stare.tile_residual import residual_displacement
from stare.vector_grid import estimate_tile_vectors, read_box
from stare.warp import warp_image

__all__ = ["register_slide"]


def register_slide(
    ref_dapi,
    mov_dapi,
    *,
    tile=512,
    halo=64,
    stride=128,
    model="euclidean",
    warp_data=None,
    out_shape=None,
):
    """Register ``mov_dapi`` to ``ref_dapi`` and warp an image into the reference frame.

    Parameters
    ----------
    ref_dapi, mov_dapi : 2-D arrays
        Reference and moving DAPI images used to *estimate* the transform.
    tile, halo : int
        Tile core size and read halo (px). ``halo`` is also the range gate (``max_disp``),
        as in the pipeline.
    stride : int
        Vector-lattice stride (px): one displacement vector per ``stride`` px, window
        ``2 * stride`` -- the mesh resolution.
    warp_data : ndarray, optional
        Image actually warped to produce the registered output (e.g. the full multi-channel
        moving slide). Defaults to ``mov_dapi``.
    out_shape : (int, int), optional
        Output raster ``(H, W)``. Defaults to the reference DAPI shape.

    Returns a dict with ``registered``, ``entry`` (manifest slide entry), ``mesh``, ``M0``,
    ``tiles``, ``controls`` (the per-tile control dicts SOLVE consumed), ``solve`` (its report),
    ``tre_px`` (per-tile rigid-stage residual: the median |d| of the tile's vectors),
    ``tre_after_px``, ``coarse_tre`` and ``n_inliers``.
    """
    ref_dapi = np.asarray(ref_dapi, dtype=float)
    mov_dapi = np.asarray(mov_dapi, dtype=float)
    out_shape = tuple(out_shape) if out_shape is not None else ref_dapi.shape

    # 1. global rigid anchor
    m0, coarse_tre, n_inliers = estimate_rigid(ref_dapi, mov_dapi, model=model)

    # 2. rigid pre-warp into the reference frame, then a grid of window vectors per tile
    mov_rigid = warp_image(mov_dapi, m0, None, ref_dapi.shape)
    tiles = tile_grid(ref_dapi.shape[1], ref_dapi.shape[0], tile, halo)
    controls = []
    for t in tiles:
        bx0, by0, bx1, by1 = read_box(t.core, stride, ref_dapi.shape)
        vec = estimate_tile_vectors(
            ref_dapi[by0:by1, bx0:bx1],
            mov_rigid[by0:by1, bx0:bx1],
            (bx0, by0),
            t.core,
            stride,
        )
        vs = vec["vectors"]
        dx = float(np.median([v[4] for v in vs])) if vs else 0.0
        dy = float(np.median([v[5] for v in vs])) if vs else 0.0
        controls.append(
            {
                "ix": t.ix,
                "iy": t.iy,
                "cx": float(t.cx),
                "cy": float(t.cy),
                "dx": dx,
                "dy": dy,
                "tre": float(np.hypot(dx, dy)),
                "lattice": vec["lattice"],
                "vectors": vs,
                "rejected": vec["rejected"],
            }
        )

    # 3. the mesh: dctpls on the vector lattice, range-gated at the halo like the pipeline
    grid_x, grid_y, disp, solve_report = solve_dctpls(controls, max_disp=halo)
    entry = slide_entry(
        m0, grid_x, grid_y, disp, interp=solve_report.get("mesh_interp")
    )
    mesh = MeshField.from_spec(entry["mesh"])

    # 4. Post-refinement per-tile residual = STARE's final-accuracy TRE (the analogue of VALIS's
    #    non-rigid error). Re-warp the DAPI with M0 + mesh and re-measure each tile against the
    #    reference with one whole-read-box correlation. With no mesh the rigid warp is final.
    tre_rigid = [c["tre"] for c in controls]
    if mesh is not None:
        mov_refined = warp_image(mov_dapi, m0, mesh, ref_dapi.shape)
        tre_after = []
        for t in tiles:
            rx0, ry0, rx1, ry1 = t.read
            _dx, _dy, tre, _error = residual_displacement(
                ref_dapi[ry0:ry1, rx0:rx1], mov_refined[ry0:ry1, rx0:rx1]
            )
            tre_after.append(tre)
    else:
        tre_after = list(tre_rigid)

    # 5. warp the (possibly multi-channel) moving image with M0 + mesh
    data = warp_data if warp_data is not None else mov_dapi
    registered = warp_image(data, m0, mesh, out_shape)

    return {
        "registered": registered,
        "entry": entry,
        "mesh": mesh,
        "M0": m0,
        "tiles": tiles,
        "controls": controls,
        "solve": solve_report,
        "tre_px": tre_rigid,  # per-tile rigid-stage misalignment
        "tre_after_px": tre_after,  # per-tile residual after refinement (final accuracy)
        "coarse_tre": coarse_tre,
        "n_inliers": n_inliers,
    }
