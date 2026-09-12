"""The stub control JSON must exercise the gate, not route around it.

`bin/tiled_solve._accept` treats a control point with no `"error"` key as legacy -- written
before confidence gating existed -- and accepts it unconditionally with a warning, so a run
resumed across that change does not lose every tile. That contract is right for real data and
wrong for a stub: `modules/local/tiled_reg_tile.nf`'s stub block emitted no `"error"`, so
**every stub run took the legacy accept-with-warning path** and the gate CI exercises was not
the gate production runs.

`-stub` already cannot see a `script:` block. If the stub's own output also dodges the one
branch under test, stub coverage of this module is worth nothing.

The test drives the real consumer rather than string-matching the stub, so the two cannot
drift: it parses the JSON the stub actually writes and asserts `_grid_from_controls` does not
report it as legacy.
"""

import json
import logging
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


def test_the_stub_control_json_is_gated_not_legacy_accepted(caplog):
    """A stub control point must go through the confidence gate like a real one."""
    control = _stub_control()

    with caplog.at_level(logging.WARNING, logger="stare.solve"):
        tiled_solve._grid_from_controls(
            [control], gate_tre=1.0, max_error=0.99, max_disp=256
        )

    assert "carry no 'error' key" not in caplog.text


def test_the_stub_control_json_is_accepted_by_the_shipped_gate():
    """The stub must model a GOOD tile -- a stub that models a rejected tile tests the wrong path."""
    control = _stub_control()

    accepted, reason = tiled_solve._accept(control, max_error=0.99, max_disp=256)

    assert accepted, f"stub control point is rejected by the shipped gate ({reason})"


def test_the_stub_carries_every_key_the_consumer_reads():
    """Guard the whole contract, not just `error`, so a future key cannot be forgotten here."""
    control = _stub_control()

    for key in ("ix", "iy", "cx", "cy", "dx", "dy", "tre", "error"):
        assert key in control, f"stub control JSON is missing {key!r}"


def test_the_parser_would_notice_if_the_stub_stopped_emitting_a_control_json():
    """A guard that silently finds nothing checks nothing."""
    assert _stub_control(), (
        "parsed an empty control JSON -- the extraction regex has rotted"
    )
