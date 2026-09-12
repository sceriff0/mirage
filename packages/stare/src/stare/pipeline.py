"""End-to-end STARE registration of one moving slide against the reference, in process.

``register_slide`` composes the method's pieces in the order the Nextflow DAG runs them, but
in a single call so it is unit-testable and reusable by the ``bin/`` CLIs:

    coarse_align.estimate_rigid   -> global rigid M0 (moving -> reference)
    tiled_warp.warp_image (rigid) -> the moving slide pre-warped into the reference frame
    tile_grid + tile_residual     -> a sub-pixel residual per reference-frame tile
    tiled_manifest.assemble       -> a TRE-gated mesh control grid + manifest entry
    tiled_warp.warp_image (mesh)  -> the registered slide (bilinear, non-negative)

Registration is estimated from the DAPI channel; the same transform can warp any channel image.
Pure NumPy + scikit-image — no VALIS, no JVM.

Memory note: this reference implementation rigid-pre-warps the whole slide for clarity. The
Nextflow WARP/STITCH path does the equivalent tile-by-tile to hold peak memory to one tile, but
produces the identical manifest and output.
"""

from __future__ import annotations

import numpy as np

from stare.coarse_align import estimate_rigid
from stare.manifest import assemble_control_grid, slide_entry
from stare.mesh_field import MeshField
from stare.tile_grid import tile_grid
from stare.tile_residual import residual_displacement
from stare.warp import warp_image

__all__ = ["register_slide"]


def register_slide(
    ref_dapi,
    mov_dapi,
    *,
    tile=512,
    halo=64,
    gate_tre=1.0,
    upsample=10,
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
        Tile core size and read halo (px). Tile size is the mesh grid resolution.
    gate_tre : float
        Only tiles whose rigid-stage residual TRE (px) meets this threshold contribute a mesh
        displacement; the rest stay rigid.
    warp_data : ndarray, optional
        Image actually warped to produce the registered output (e.g. the full multi-channel
        moving slide). Defaults to ``mov_dapi``.
    out_shape : (int, int), optional
        Output raster ``(H, W)``. Defaults to the reference DAPI shape.

    Returns a dict with ``registered``, ``entry`` (manifest slide entry), ``mesh``, ``M0``,
    ``tiles``, ``residuals``, ``tre_px`` (per-tile TRE), ``coarse_tre`` and ``n_inliers``.
    """
    ref_dapi = np.asarray(ref_dapi, dtype=float)
    mov_dapi = np.asarray(mov_dapi, dtype=float)
    out_shape = tuple(out_shape) if out_shape is not None else ref_dapi.shape

    # 1. global rigid anchor
    m0, coarse_tre, n_inliers = estimate_rigid(ref_dapi, mov_dapi, model=model)

    # 2. rigid pre-warp into the reference frame, then measure the residual per reference tile
    mov_rigid = warp_image(mov_dapi, m0, None, ref_dapi.shape)
    tiles = tile_grid(ref_dapi.shape[1], ref_dapi.shape[0], tile, halo)
    residuals = []
    for t in tiles:
        rx0, ry0, rx1, ry1 = t.read
        # NOTE: `_error` (the correlation confidence) is deliberately unused here. The confidence
        # and range gates live in the FAN-OUT reduction, tiled_solve._grid_from_controls, which is
        # what the Nextflow DAG actually runs; this in-process composition is a reference/test
        # harness with no CLI or module caller, and `assemble_control_grid` still gates on TRE
        # alone. Consequence: on background-heavy input this path is no longer bit-identical to
        # the fan-out — it keeps control points the fan-out now rejects.
        dx, dy, tre, _error = residual_displacement(
            ref_dapi[ry0:ry1, rx0:rx1],
            mov_rigid[ry0:ry1, rx0:rx1],
            upsample=upsample,
        )
        residuals.append((dx, dy, tre))

    # 3. TRE-gated control grid (reference-frame tile centres) -> manifest entry + mesh
    grid_x, grid_y, disp = assemble_control_grid(tiles, residuals, gate_tre)
    entry = slide_entry(m0, grid_x, grid_y, disp)
    mesh = (
        MeshField(grid_x, grid_y, np.asarray(disp, dtype=float))
        if entry["mesh"] is not None
        else None
    )

    # 4. Post-refinement per-tile residual = STARE's final-accuracy TRE (the analogue of VALIS's
    #    non-rigid error). Re-warp the DAPI with M0 + mesh and re-measure each tile against the
    #    reference. With no mesh the rigid warp is final, so the rigid residual is already the
    #    final residual — no second measurement needed.
    if mesh is not None:
        mov_refined = warp_image(mov_dapi, m0, mesh, ref_dapi.shape)
        tre_after = []
        for t in tiles:
            rx0, ry0, rx1, ry1 = t.read
            _dx, _dy, tre, _error = residual_displacement(
                ref_dapi[ry0:ry1, rx0:rx1],
                mov_refined[ry0:ry1, rx0:rx1],
                upsample=upsample,
            )
            tre_after.append(tre)
    else:
        tre_after = [r[2] for r in residuals]

    # 5. warp the (possibly multi-channel) moving image with M0 + mesh
    data = warp_data if warp_data is not None else mov_dapi
    registered = warp_image(data, m0, mesh, out_shape)

    return {
        "registered": registered,
        "entry": entry,
        "mesh": mesh,
        "M0": m0,
        "tiles": tiles,
        "residuals": residuals,
        "tre_px": [r[2] for r in residuals],  # per-tile rigid-stage misalignment
        "tre_after_px": tre_after,  # per-tile residual after refinement (final accuracy)
        "coarse_tre": coarse_tre,
        "n_inliers": n_inliers,
    }
