"""ASHLAR solve: the reference cycle's tiles are placed EXACTLY, not re-registered.

retile.py cuts every cycle from one already-stitched slide, so reference tile positions are
exact by construction and adjacent tiles' overlaps are IDENTICAL pixels. ASHLAR's
EdgeAligner registers those overlaps anyway, and utils.nccw raises when correlation exceeds
total amplitude by more than an ABSOLUTE 1e-5 -- which identical, bright overlaps do through
float rounding: job 6844139, "RuntimeError: correlation > total_amplitude
(diff=1.1444091796875e-05)" on tile pair (627, 628). So by default the reference aligner is
built from the nominal positions with an identity model, and only LayerAligner (the
cross-cycle step) registers. ASHLAR is not installed in CI; a stand-in ashlar.reg checks
the wiring.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from benchmarks.ashlar import solve


class _Meta:
    def __init__(self):
        self.positions = np.array(
            [[0.0, 0.0], [0.0, 900.0], [900.0, 0.0], [900.0, 900.0]]
        )
        self.size = np.array([1000.0, 1000.0])
        self.pixel_size = 0.325


class _FakeEdge:
    runs = 0

    def __init__(self, reader, channel=0, max_shift=15, verbose=False):
        self.reader, self.channel, self.max_shift = reader, channel, max_shift
        self.metadata = _Meta()
        self.thumbnail_made = False

    def make_thumbnail(self):
        self.thumbnail_made = True

    def run(self):
        _FakeEdge.runs += 1
        raise RuntimeError("correlation > total_amplitude (diff=1.1444091796875e-05)")


class _FakeLayer:
    def __init__(
        self, reader, reference_aligner, channel=None, max_shift=15, verbose=False
    ):
        self.reference_aligner = reference_aligner

    def run(self):
        ref = self.reference_aligner
        assert ref.thumbnail_made, "LayerAligner needs the reference thumbnail"
        np.testing.assert_allclose(ref.positions, ref.metadata.positions)
        np.testing.assert_allclose(
            ref.lr.predict(ref.metadata.positions), ref.metadata.positions
        )
        np.testing.assert_allclose(ref.centers, ref.positions + ref.metadata.size / 2)


@pytest.fixture
def fake_ashlar(monkeypatch):
    reg = types.ModuleType("ashlar.reg")
    reg.EdgeAligner, reg.LayerAligner = _FakeEdge, _FakeLayer

    class DataWarning(UserWarning):
        pass

    reg.DataWarning = DataWarning
    pkg = types.ModuleType("ashlar")
    pkg.reg = reg
    monkeypatch.setitem(sys.modules, "ashlar", pkg)
    monkeypatch.setitem(sys.modules, "ashlar.reg", reg)
    monkeypatch.setattr(solve, "_reader", lambda d, g, px: object())
    _FakeEdge.runs = 0


def test_the_reference_grid_is_placed_exactly_without_registering_its_edges(
    fake_ashlar,
):
    edge, layer, warned = solve._run_aligners("r", {}, "m", {}, 0, 30.0, 0.325)
    assert _FakeEdge.runs == 0 and warned == []
    assert layer.reference_aligner is edge


def test_register_mode_keeps_ashlars_own_edge_registration(fake_ashlar):
    with pytest.raises(RuntimeError, match="correlation > total_amplitude"):
        solve._run_aligners(
            "r", {}, "m", {}, 0, 30.0, 0.325, reference_edges="register"
        )
    assert _FakeEdge.runs == 1
