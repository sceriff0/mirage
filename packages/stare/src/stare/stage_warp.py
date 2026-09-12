"""reg_qc=2 stage warper for the tiled ('STARE') registration method.

The reg_qc=2 scorer (``bin/warp_seg_qc.py``) is method-agnostic: it scores every stage through an
injected ``warp(slide_name, xy, stage) -> xy`` callable and never imports the registrar itself.
For VALIS that callable comes from ``valis_stage_warp.make_warper`` (a loaded registrar pickle +
BioFormats JVM). This module is its tiled-method counterpart: it builds the *same* callable from a
lightweight STARE transform manifest — a global rigid ``M0`` per slide plus a control-grid mesh
field — with **no VALIS, no JVM, pure NumPy**.

Manifest shape::

    {
      "ref_slide": "<name>",
      "slides": {
        "<name>": {
          "M0":   3x3 affine (forward: native moving coords -> reference frame),
          "mesh": {"grid_x": [...], "grid_y": [...], "displacements": [[[dx,dy],...],...]} | null,
        },
        ...
      }
    }

Stages
------
``native``   no transform.
``rigid``    the global affine ``M0`` only  — **the reg_qc anchor**.
``refined``  ``M0`` plus the mesh residual, sampled at the native point.

There is no ``micro`` stage and nothing is composed destructively, so — unlike VALIS — every
stage is a first-class transform the manifest can reproduce directly; no pre-micro checkpoint is
needed. The mesh lives in the *reference frame* (the grid is laid on the rigid-warped slide's tile
centres) and is sampled at the rigid position ``M0·xy``; both this warper and the per-tile image
warp sample that same field, so the QC measures exactly the transform that shipped.
"""

from __future__ import annotations

import numpy as np

from stare.mesh_field import MeshField

__all__ = [
    "STAGE_NATIVE",
    "STAGE_RIGID",
    "STAGE_REFINED",
    "STAGES",
    "make_warper",
]

STAGE_NATIVE = "native"
STAGE_RIGID = "rigid"
STAGE_REFINED = "refined"
STAGES = (STAGE_NATIVE, STAGE_RIGID, STAGE_REFINED)


def _apply_affine(m, xy):
    """Apply a 3x3 forward affine to ``(N, 2)`` points."""
    m = np.asarray(m, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return xy.reshape(-1, 2).copy()
    homog = np.column_stack([xy, np.ones(len(xy))])
    return (homog @ m.T)[:, :2]


def _mesh_from_spec(spec):
    if spec is None:
        return None
    return MeshField(spec["grid_x"], spec["grid_y"], spec["displacements"])


def make_warper(manifest):
    """Build ``warp(slide_name, xy, stage) -> xy`` over a STARE transform manifest."""
    slides = manifest["slides"]
    affines = {name: np.asarray(s["M0"], dtype=float) for name, s in slides.items()}
    meshes = {name: _mesh_from_spec(s.get("mesh")) for name, s in slides.items()}

    def warp(slide_name, xy, stage):
        xy = np.asarray(xy, dtype=float)
        if stage == STAGE_NATIVE:
            return xy.copy()
        if stage not in (STAGE_RIGID, STAGE_REFINED):
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        warped = _apply_affine(
            affines[slide_name], xy
        )  # rigid position in the reference frame
        if stage == STAGE_REFINED:
            mesh = meshes.get(slide_name)
            if mesh is not None:
                # the mesh lives in the reference frame, so it is sampled at the rigid position
                warped = warped + mesh.displacement(warped)
        return warped

    return warp
