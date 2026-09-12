"""Assemble and serialize the STARE transform manifest.

A manifest is the whole transform for a patient: a global affine ``M0`` per slide plus, for slides
that needed refining, a control-grid mesh field. It is JSON-native and is consumed unchanged by
:func:`tiled_stage_warp.make_warper` (reg_qc=2) and by the image warp — one artifact, three
readers.

:func:`assemble_control_grid` lays per-tile residuals onto the grid with **TRE gating**: a tile
whose rigid-stage TRE is below the threshold is already well aligned, so it contributes a zero
displacement and stays at its rigid position; only tiles above the threshold carry their
correction. A slide whose every tile is gated out gets no mesh at all (rigid-only entry).
"""

from __future__ import annotations

import numpy as np

__all__ = ["assemble_control_grid", "slide_entry", "build_manifest"]


def assemble_control_grid(tiles, residuals, gate_tre):
    """Lay ``residuals`` (one ``(dx, dy, tre)`` per tile) onto the control grid, TRE-gated.

    Returns ``(grid_x, grid_y, displacements)`` where ``displacements`` is a nested
    ``[ny][nx][2]`` list, zero wherever ``tre < gate_tre``.
    """
    nx = max(t.ix for t in tiles) + 1
    ny = max(t.iy for t in tiles) + 1
    grid_x = [0.0] * nx
    grid_y = [0.0] * ny
    disp = [[[0.0, 0.0] for _ in range(nx)] for _ in range(ny)]
    for t, (dx, dy, tre) in zip(tiles, residuals):
        grid_x[t.ix] = float(t.cx)
        grid_y[t.iy] = float(t.cy)
        if tre >= gate_tre:
            disp[t.iy][t.ix] = [float(dx), float(dy)]
    return grid_x, grid_y, disp


def slide_entry(m0, grid_x=None, grid_y=None, displacements=None):
    """Build one slide's manifest entry: ``{"M0": ..., "mesh": ... | None}``.

    A mesh is emitted only when some displacement is non-zero; an all-zero (fully TRE-gated) grid
    collapses to ``mesh: None`` so the warper cleanly falls back to rigid.
    """
    entry = {"M0": np.asarray(m0, dtype=float).tolist(), "mesh": None}
    if displacements is not None:
        d = np.asarray(displacements, dtype=float)
        if np.any(d != 0.0):
            entry["mesh"] = {
                "grid_x": [float(v) for v in grid_x],
                "grid_y": [float(v) for v in grid_y],
                "displacements": d.tolist(),
            }
    return entry


def build_manifest(ref_slide, slides):
    """Assemble the patient manifest from a ``{slide_name: entry}`` mapping."""
    return {"ref_slide": ref_slide, "slides": dict(slides)}
