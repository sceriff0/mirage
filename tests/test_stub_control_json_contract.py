"""The stub control JSON must exercise the solve every real run takes, not route around it.

`-stub` already cannot see a `script:` block. If the stub's own output also dodges the branch
under test, stub coverage of this module is worth nothing. Under STARE v1 the trap was the
`error` key (a control point without it took a legacy accept-with-warning path). Since STARE v2
SOLVE solves only the window-vector lattice and REFUSES a control JSON without `lattice` and
`vectors` -- so a stub missing them would fail every stub run's TILED_SOLVE, and a stub whose
vectors are all out of range would solve to no mesh while real runs refine one.

The test drives the real consumer rather than string-matching the stub, so the two cannot
drift: it parses the JSON the stub actually writes and runs `drape.solve.solve_dctpls` on it.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, os.path.join(str(REPO), "bin"))
sys.path.insert(0, os.path.join(str(REPO), "bin", "utils"))

pytest.importorskip("numpy")

import tiled_solve  # noqa: E402

MODULE = REPO / "modules" / "local" / "tiled_reg_tile.nf"


def _stub_control():
    """Parse the control JSON the stub block writes, with Nextflow interpolation filled in."""
    text = MODULE.read_text()
    stub = text.split("stub:", 1)[1]
    m = re.search(r"echo\s+'(\{.*?\})'", stub, re.S)
    assert m, f"no control-JSON echo found in {MODULE}'s stub block"
    literal = m.group(1)
    # `${row.ix}` etc. -- the values are irrelevant to the contract under test; the KEYS are not.
    literal = re.sub(r"\$\{[^}]*\}", "0", literal)
    return json.loads(literal)


def test_the_stub_control_json_is_accepted_by_the_shipped_range_gate():
    """The stub must model a GOOD tile -- a stub that models a rejected tile tests the wrong path."""
    control = _stub_control()

    assert tiled_solve.tile_accepted(control, max_disp=256), (
        "stub control JSON contributes no in-range vector"
    )


def test_the_stub_carries_every_key_the_consumer_reads():
    """Guard the whole contract, not just `error`, so a future key cannot be forgotten here."""
    control = _stub_control()

    for key in ("ix", "iy", "cx", "cy", "dx", "dy", "tre", "lattice", "vectors"):
        assert key in control, f"stub control JSON is missing {key!r}"


def test_the_parser_would_notice_if_the_stub_stopped_emitting_a_control_json():
    """A guard that silently finds nothing checks nothing."""
    assert _stub_control(), (
        "parsed an empty control JSON -- the extraction regex has rotted"
    )


# ---------------------------------------------------------------------------
# The window-vector grid (drape.vector_grid): the stub must drive the VECTOR solve
# ---------------------------------------------------------------------------
# `drape.solve.solve_dctpls` refuses a control without `vectors`, so a stub without them would
# fail TILED_SOLVE on every stub run.


def test_the_stub_carries_a_lattice_and_a_vector_list():
    control = _stub_control()

    assert "lattice" in control, "stub control JSON has no 'lattice'"
    lattice = control["lattice"]
    for key in ("stride", "window", "origin"):
        assert key in lattice, f"stub lattice is missing {key!r}"
    assert lattice["window"] == 2 * lattice["stride"]
    assert lattice["origin"] == lattice["window"] / 2

    vectors = control.get("vectors")
    assert isinstance(vectors, list) and vectors, "stub must emit at least one vector"
    for v in vectors:
        # [kx, ky, cx, cy, dx, dy, peak_ratio, sharpness, fg]
        assert len(v) == 9, f"a vector is 9 numbers, got {v!r}"


def test_the_stub_vectors_reach_the_vector_solve():
    """Drive the real dctpls consumer: the stub's vectors must be laid on the lattice."""
    from drape.solve import solve_dctpls

    control = _stub_control()
    _gx, _gy, _disp, report = solve_dctpls([control], max_disp=256)

    assert report.get("input") == "vectors"
    assert report["n_valid"] >= 1
