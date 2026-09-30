"""Tests for bin/utils/tiled_stage_warp.py — the reg_qc=2 stage warper for the tiled ('STARE')
registration method.

This is the seam that lets the tiled method reuse the reg_qc=2 scorer (bin/warp_seg_qc.py)
unchanged: it turns a STARE transform manifest (a global rigid ``M0`` per slide + a control-grid
mesh field) into the same ``warp(slide_name, xy, stage)`` callable the scorer injects for VALIS.
Stages are ``native`` / ``rigid`` / ``refined`` — no destructive micro composition, so every
stage is a first-class, independently reachable transform.

The final test drives the *real* warper through the *real* scorer, proving reg_qc=2 works for the
tiled method before any registration code exists.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
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

from utils.tiled_stage_warp import (
    STAGE_NATIVE,
    STAGE_REFINED,
    STAGE_RIGID,
    make_warper,
)

REF = "ref_slide"
MOV = "mov_slide"


def _identity():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _translation(tx, ty):
    return [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]


def _uniform_mesh(dx, dy, extent=1000.0):
    """A 2x2 control grid whose four nodes all carry the same displacement -> a constant field."""
    return {
        "grid_x": [0.0, extent],
        "grid_y": [0.0, extent],
        "displacements": [[[dx, dy], [dx, dy]], [[dx, dy], [dx, dy]]],
    }


def _manifest(mov_M0, mov_mesh=None):
    return {
        "ref_slide": REF,
        "slides": {
            REF: {"M0": _identity(), "mesh": None},
            MOV: {"M0": mov_M0, "mesh": mov_mesh},
        },
    }


# ── stage mapping ────────────────────────────────────────────────────────────────
def test_native_stage_returns_points_unchanged():
    warp = make_warper(_manifest(_translation(-96.0, 0.0)))
    xy = np.array([[100.0, 0.0], [140.0, 10.0]])
    np.testing.assert_allclose(warp(MOV, xy, STAGE_NATIVE), xy)


def test_rigid_stage_applies_the_global_affine_M0():
    warp = make_warper(_manifest(_translation(-96.0, 5.0)))
    xy = np.array([[100.0, 0.0], [140.0, 10.0]])
    np.testing.assert_allclose(warp(MOV, xy, STAGE_RIGID), xy + np.array([-96.0, 5.0]))


def test_refined_stage_adds_the_mesh_field_on_top_of_rigid():
    warp = make_warper(
        _manifest(_translation(-96.0, 0.0), mov_mesh=_uniform_mesh(-4.0, 0.0))
    )
    xy = np.array([[100.0, 0.0], [140.0, 10.0]])
    # rigid puts it at xy+(-96,0); the constant mesh finishes the last -4 in x
    np.testing.assert_allclose(
        warp(MOV, xy, STAGE_REFINED), xy + np.array([-100.0, 0.0])
    )


def test_refined_equals_rigid_when_the_slide_has_no_mesh():
    warp = make_warper(_manifest(_translation(-96.0, 0.0), mov_mesh=None))
    xy = np.array([[100.0, 0.0]])
    np.testing.assert_allclose(warp(MOV, xy, STAGE_REFINED), warp(MOV, xy, STAGE_RIGID))


def test_reference_slide_is_the_identity_at_every_stage():
    warp = make_warper(_manifest(_translation(-96.0, 0.0)))
    xy = np.array([[10.0, 20.0], [0.0, 0.0]])
    for stage in (STAGE_NATIVE, STAGE_RIGID, STAGE_REFINED):
        np.testing.assert_allclose(warp(REF, xy, stage), xy)


def test_unknown_stage_is_rejected():
    warp = make_warper(_manifest(_translation(0.0, 0.0)))
    with pytest.raises(ValueError, match="stage"):
        warp(MOV, np.array([[0.0, 0.0]]), "micro")


# ── the integration proof: real warper → real reg_qc=2 scorer ───────────────────
def _square(x0, y0, s=10):
    ring = [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s], [x0, y0]]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _grid_fc(n=6, pitch=40, dx=0.0, size=10):
    return {
        "type": "FeatureCollection",
        "features": [_square(i * pitch + dx, 0.0, size) for i in range(n)],
    }


def _write(tmp_path, name, fc):
    p = tmp_path / name
    p.write_text(json.dumps(fc))
    return str(p)


def test_reg_qc2_scorer_runs_a_native_rigid_refined_plan_through_the_real_warper(
    tmp_path,
):
    """End-to-end proof that the tiled method is reg_qc=2 compatible with no scorer changes.

    A manifest whose global M0 gets the moving cells within 4 px and whose constant mesh field
    finishes the last 4 px must make the scorer report: pairing fixed at the rigid anchor, rigid
    residual 4 px, refined residual 0 px, and IoU rising rigid -> refined to 1.0.
    """
    pytest.importorskip("skimage")
    pytest.importorskip("scipy")
    import warp_seg_qc as wsq

    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))  # 100 px off natively

    warp = make_warper(
        _manifest(_translation(-96.0, 0.0), mov_mesh=_uniform_mesh(-4.0, 0.0))
    )

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_REFINED],
        ref_slide=REF,
        moving_slide=MOV,
        supersample=4,
    )

    s = out["stages"]
    # pairing fixed at the anchor: every stage reports the same six pairs
    assert out["matching"]["anchor_stage"] == STAGE_RIGID
    assert out["matching"]["n_pairs"] == 6
    assert {st: s[st]["n_pairs"] for st in out["stage_order"]} == {
        st: 6 for st in out["stage_order"]
    }
    # geometry improves rigid -> refined
    assert s[STAGE_RIGID]["displacement_px_p50"] == pytest.approx(4.0)
    assert s[STAGE_REFINED]["displacement_px_p50"] == pytest.approx(0.0, abs=1e-9)
    assert s[STAGE_NATIVE]["displacement_px_p50"] == pytest.approx(100.0)
    assert s[STAGE_RIGID]["iou_mean"] < s[STAGE_REFINED]["iou_mean"]
    assert s[STAGE_REFINED]["iou_mean"] == pytest.approx(1.0)
    # deltas read against the anchor, with the sign that says "refined helped"
    assert out["delta_vs_anchor"][STAGE_REFINED][
        "displacement_px_p50"
    ] == pytest.approx(-4.0)
