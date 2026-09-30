"""Assemble and serialize the STARE transform manifest.

A manifest is the whole transform for a patient: a global affine ``M0`` per slide plus, for slides
that needed refining, a control-grid mesh field. It is JSON-native and is consumed unchanged by
:func:`tiled_stage_warp.make_warper` (reg_qc=2) and by the image warp — one artifact, three
readers.

The mesh itself comes from SOLVE (``stare.solve.solve_dctpls``); :func:`slide_entry` wraps it
with the slide's ``M0``. A slide whose solved field is identically zero gets no mesh at all
(rigid-only entry).
"""

from __future__ import annotations

import numpy as np

__all__ = ["slide_entry", "build_manifest"]


def slide_entry(m0, grid_x=None, grid_y=None, displacements=None, interp=None):
    """Build one slide's manifest entry: ``{"M0": ..., "mesh": ... | None}``.

    A mesh is emitted only when some displacement is non-zero; an all-zero (or empty) grid
    collapses to ``mesh: None`` so the warper cleanly falls back to rigid. ``interp`` other than
    ``None``/``"bilinear"`` is recorded as the mesh's ``"interp"`` (``MeshField.from_spec``);
    a bilinear mesh carries no key, so its manifest is byte-identical to one written before
    the key existed.
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
            if interp not in (None, "bilinear"):
                entry["mesh"]["interp"] = interp
    return entry


def build_manifest(ref_slide, slides):
    """Assemble the patient manifest from a ``{slide_name: entry}`` mapping."""
    return {"ref_slide": ref_slide, "slides": dict(slides)}
