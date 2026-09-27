"""Tests for bin/utils/tiled_manifest.py — assembling the DRAPE transform manifest.

SOLVE (``drape.solve.solve_dctpls``) produces the mesh; ``slide_entry`` wraps it with the slide's
M0, collapsing an all-zero field to a rigid-only entry. The manifest round-trips through JSON and
is consumed unchanged by
``tiled_stage_warp.make_warper`` — the same object the reg_qc=2 scorer and the image warp read.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "utils"
    ),
)

from tile_grid import tile_grid
from tiled_manifest import build_manifest, slide_entry
from tiled_stage_warp import STAGE_REFINED, STAGE_RIGID, make_warper


def test_an_all_zero_field_yields_a_rigid_only_entry():
    tiles = tile_grid(80, 40, 40, halo=4)
    gx = sorted({t.cx for t in tiles})
    gy = sorted({t.cy for t in tiles})
    disp = np.zeros((len(gy), len(gx), 2))
    entry = slide_entry(np.eye(3), gx, gy, disp)
    assert (
        entry["mesh"] is None
    )  # nothing to refine -> no mesh, warper falls back to rigid


def test_an_empty_field_yields_a_rigid_only_entry():
    """SOLVE returns empty arrays when no tile reported a lattice node."""
    assert slide_entry(np.eye(3), [], [], [])["mesh"] is None


def test_manifest_round_trips_through_json_into_a_working_warper():
    tiles = tile_grid(100, 100, 50, halo=8)  # 2x2
    # one node carries a real correction; build the manifest around it
    gx = sorted({t.cx for t in tiles})
    gy = sorted({t.cy for t in tiles})
    disp = np.zeros((2, 2, 2))
    disp[1, 1] = [4.0, 0.0]

    manifest = build_manifest(
        "ref",
        {
            "ref": slide_entry(np.eye(3)),
            # identity M0 so the rigid position equals the native point, landing exactly on the
            # (1,1) control node — makes the mesh contribution assertable to the node value.
            "mov": slide_entry(np.eye(3), gx, gy, disp),
        },
    )
    # must survive serialization intact (all json-native types)
    manifest = json.loads(json.dumps(manifest))

    warp = make_warper(manifest)
    # a point at the (1,1) tile centre (a control node): rigid is identity, mesh adds +4 in x
    centre = np.array([[gx[1], gy[1]]])
    rigid = warp("mov", centre, STAGE_RIGID)
    refined = warp("mov", centre, STAGE_REFINED)
    np.testing.assert_allclose(rigid, centre)
    np.testing.assert_allclose(refined, centre + np.array([4.0, 0.0]))
