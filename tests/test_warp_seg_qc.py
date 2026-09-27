"""Tests for bin/warp_seg_qc.py — the staged, fixed-correspondence registration QC (reg_qc=2).

The VALIS warp is injected as a plain ``warp(slide_name, xy, stage)`` callable, so the parts
that carry the scientific claim — which stages are reportable, that the pairing is fixed at the
anchor, and that the deltas read against that anchor — are testable without VALIS or a GPU.

The fake warp models the real thing: each stage moves the moving slide's cells a bit closer to
the reference, exactly as rigid -> non-rigid -> micro is supposed to.
"""

from __future__ import annotations

import json
import math
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

pytest.importorskip("skimage")
pytest.importorskip("scipy")

import warp_seg_qc as wsq
from utils.valis_stage_warp import (
    STAGE_MICRO,
    STAGE_NATIVE,
    STAGE_NON_RIGID,
    STAGE_RIGID,
)


# ── fixtures ───────────────────────────────────────────────────────────────────
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


def _pair_fc(centroid_xs, s=2.0, y=0.0):
    """Two (or more) small squares whose centroids sit exactly at ``centroid_xs`` on the
    x-axis -- lets a test place cells at precise distances instead of relying on a grid
    pitch, which is what a geometry that must distinguish two matchers needs."""
    return {
        "type": "FeatureCollection",
        "features": [_square(cx - s / 2, y - s / 2, s) for cx in centroid_xs],
    }


def _write(tmp_path, name, fc):
    p = tmp_path / name
    p.write_text(json.dumps(fc))
    return str(p)


REF = "ref_slide"
MOV = "mov_slide"


def _shift_warp(offsets):
    """warp(slide, xy, stage) that shifts only the moving slide, by ``offsets[stage]`` in x.

    The reference is left alone at every stage, which is the geometry that matters: what is
    being measured is the *relative* position of the two slides in a shared frame.
    """
    calls = []

    def warp(slide_name, xy, stage):
        calls.append((slide_name, stage))
        xy = np.asarray(xy, dtype=float)
        if slide_name == REF:
            return xy.copy()
        return xy + np.array([offsets[stage], 0.0])

    warp.calls = calls
    return warp


class _FakeCheckpoint:
    def __init__(self, micro=True, slides=(REF, MOV)):
        self.micro_registration = micro
        self.slide_names = list(slides)

    def has_slide(self, name):
        return name in self.slide_names


# ── stage planning ─────────────────────────────────────────────────────────────
def test_plan_stages_without_a_checkpoint_cannot_separate_non_rigid():
    stages, separable, note = wsq.plan_stages(None)
    assert stages == [STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO]
    assert separable is False
    assert "micro residual" in note


def test_plan_stages_with_a_checkpoint_reports_all_four():
    stages, separable, note = wsq.plan_stages(_FakeCheckpoint(micro=True))
    assert stages == [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO]
    assert separable is True
    assert note == ""


def test_plan_stages_omits_micro_when_micro_registration_was_skipped():
    """A 'micro' stage identical to 'non_rigid' would be a duplicate reported as a result."""
    stages, separable, note = wsq.plan_stages(_FakeCheckpoint(micro=False))
    assert stages == [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID]
    assert separable is True
    assert "did not run" in note


# ── the anchor contract ────────────────────────────────────────────────────────
def test_run_fixes_the_pairing_at_the_anchor_not_per_stage(tmp_path):
    """Every stage must report the same ``n_pairs`` — that is what "fixed" means."""
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))  # wildly offset natively
    warp = _shift_warp(
        {
            STAGE_NATIVE: 0.0,
            STAGE_RIGID: -96.0,
            STAGE_NON_RIGID: -99.0,
            STAGE_MICRO: -100.0,
        }
    )

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
    )

    n = out["matching"]["n_pairs"]
    assert n == 6
    assert {s: out["stages"][s]["n_pairs"] for s in out["stage_order"]} == {
        s: n for s in out["stage_order"]
    }


def test_run_pairs_at_the_anchor_even_when_native_is_unmatchable(tmp_path):
    """The point of anchoring after rigid: native distances are too wide to pair on.

    Natively the moving cells sit 100 px away — far past any sane match radius — yet the run
    still reports six pairs, because the pairing was never attempted there.
    """
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp({STAGE_NATIVE: 0.0, STAGE_RIGID: -100.0, STAGE_MICRO: -100.0})

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
    )

    assert out["matching"]["n_pairs"] == 6
    # ...and the native stage still gets a number, measured on those anchored pairs.
    assert out["stages"][STAGE_NATIVE]["displacement_px_p50"] == pytest.approx(100.0)
    assert out["stages"][STAGE_NATIVE]["iou_mean"] == pytest.approx(0.0)


def test_match_radius_is_derived_from_the_cells_themselves(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc(size=10))
    mov = _write(tmp_path, "mov.geojson", _grid_fc(size=10, dx=1.0))
    warp = _shift_warp({STAGE_RIGID: 0.0, STAGE_MICRO: 0.0})
    out = wsq.run(
        ref, mov, warp, [STAGE_RIGID, STAGE_MICRO], ref_slide=REF, moving_slide=MOV
    )
    # A 10x10 square has equivalent radius sqrt(100/pi) ~= 5.64; factor 1.5 -> 1.5x that.
    assert out["matching"]["median_cell_radius_px"] == pytest.approx(
        math.sqrt(100 / math.pi)
    )
    assert out["matching"]["radius_px"] == pytest.approx(
        1.5 * out["matching"]["median_cell_radius_px"]
    )
    assert out["matching"]["method"] == "lsa_centroid"
    assert out["matching"]["anchor_stage"] == STAGE_RIGID


def test_absolute_match_radius_overrides_the_factor(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    warp = _shift_warp({STAGE_RIGID: 0.0})
    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID],
        ref_slide=REF,
        moving_slide=MOV,
        match_radius_px=3.0,
    )
    assert out["matching"]["radius_px"] == pytest.approx(3.0)
    assert out["matching"]["radius_factor"] is None


def test_a_match_radius_too_tight_to_pair_anything_is_reported_not_crashed(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=20.0))
    warp = _shift_warp({STAGE_RIGID: 0.0})
    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID],
        ref_slide=REF,
        moving_slide=MOV,
        match_radius_px=0.5,
    )
    assert out["matching"]["n_pairs"] == 0
    assert out["stages"][STAGE_RIGID]["n_pairs"] == 0
    assert out["stages"][STAGE_RIGID]["iou_n"] == 0


def test_run_records_lsa_as_the_default_pairing(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=2.0))
    warp = _shift_warp({STAGE_NATIVE: 0.0, STAGE_RIGID: -2.0})
    out = wsq.run(
        ref, mov, warp, [STAGE_NATIVE, STAGE_RIGID], ref_slide=REF, moving_slide=MOV
    )
    m = out["matching"]
    assert m["method"] == "lsa_centroid"
    assert m["n_pairs"] == 6
    assert m["n_components"] >= 1
    assert m["n_fallback_components"] == 0
    assert m["n_fallback_cells"] == 0


def test_run_honours_pairing_mutual_nn(tmp_path):
    """A geometry where LSA and mutual-NN genuinely disagree on the pair set -- same premise
    as test_cell_pairs.py::test_lsa_recovers_the_pair_mutual_nn_drops (ref centroids at
    x=0,3; moving at x=1,7; radius 5). a1's only in-radius candidate is b0 (dist 2), but
    b0's nearest is a0 (dist 1), so mutual-NN drops a1 entirely and pairs only (a0,b0).
    LSA maximises cardinality first and pairs both (a0,b0) and (a1,b1) (dist 1 and 4). On a
    uniform-shift grid (the old fixture) both backends pair everything, so only the
    ``method`` string distinguished them; this geometry would fail if ``--pairing`` were
    silently ignored."""
    ref = _write(tmp_path, "ref.geojson", _pair_fc([0.0, 3.0]))
    mov = _write(tmp_path, "mov.geojson", _pair_fc([1.0, 7.0]))
    warp = _shift_warp({STAGE_RIGID: 0.0})

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID],
        ref_slide=REF,
        moving_slide=MOV,
        pairing="mutual_nn",
        match_radius_px=5.0,
    )
    m = out["matching"]
    assert m["method"] == "mutual_nn_centroid"
    assert m["n_pairs"] == 1
    # The mutual-NN backend reports no component diagnostics -- it has none.
    assert "n_components" not in m

    # Same geometry, default (lsa) pairing: both cells pair. If --pairing were silently
    # ignored, the mutual_nn run above would have produced this result instead.
    out_lsa = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID],
        ref_slide=REF,
        moving_slide=MOV,
        match_radius_px=5.0,
    )
    assert out_lsa["matching"]["method"] == "lsa_centroid"
    assert out_lsa["matching"]["n_pairs"] == 2


def test_default_match_radius_factor_is_widened_for_lsa():
    assert wsq.DEFAULT_MATCH_RADIUS_FACTOR == 1.5


# ── per-stage metrics + deltas ─────────────────────────────────────────────────
def test_successive_stages_improve_iou_and_shrink_displacement(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp(
        {
            STAGE_NATIVE: 0.0,
            STAGE_RIGID: -96.0,
            STAGE_NON_RIGID: -99.0,
            STAGE_MICRO: -100.0,
        }
    )

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
        supersample=4,
    )
    s = out["stages"]

    assert s[STAGE_RIGID]["displacement_px_p50"] == pytest.approx(4.0)
    assert s[STAGE_NON_RIGID]["displacement_px_p50"] == pytest.approx(1.0)
    assert s[STAGE_MICRO]["displacement_px_p50"] == pytest.approx(0.0, abs=1e-9)
    assert (
        s[STAGE_RIGID]["iou_mean"]
        < s[STAGE_NON_RIGID]["iou_mean"]
        < s[STAGE_MICRO]["iou_mean"]
    )
    assert s[STAGE_MICRO]["iou_mean"] == pytest.approx(1.0)
    assert s[STAGE_MICRO]["dice_matched"] == pytest.approx(1.0)
    assert s[STAGE_MICRO]["frac_iou_ge_0.5"] == pytest.approx(1.0)


def test_deltas_are_measured_against_the_anchor_and_carry_the_sign(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp(
        {
            STAGE_NATIVE: 0.0,
            STAGE_RIGID: -96.0,
            STAGE_NON_RIGID: -99.0,
            STAGE_MICRO: -100.0,
        }
    )

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
        supersample=4,
    )

    assert (
        STAGE_RIGID not in out["delta_vs_anchor"]
    )  # the anchor has no delta to itself
    d = out["delta_vs_anchor"]
    assert d[STAGE_MICRO]["displacement_px_p50"] == pytest.approx(-4.0)  # improvement
    assert d[STAGE_MICRO]["iou_mean"] > 0
    assert d[STAGE_NATIVE]["displacement_px_p50"] == pytest.approx(
        +96.0
    )  # native is worse


def test_a_stage_that_makes_things_worse_shows_a_positive_displacement_delta(tmp_path):
    """The failure this whole design exists to surface: micro-registration regressing."""
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp(
        {STAGE_RIGID: -100.0, STAGE_NON_RIGID: -100.0, STAGE_MICRO: -97.0}
    )

    out = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
        supersample=4,
    )

    assert out["delta_vs_anchor"][STAGE_NON_RIGID][
        "displacement_px_p50"
    ] == pytest.approx(0.0)
    assert out["delta_vs_anchor"][STAGE_MICRO]["displacement_px_p50"] == pytest.approx(
        +3.0
    )
    assert out["delta_vs_anchor"][STAGE_MICRO]["iou_mean"] < 0


def test_micron_stats_appear_only_with_a_pixel_size(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp({STAGE_RIGID: -98.0})

    px = wsq.run(ref, mov, warp, [STAGE_RIGID], ref_slide=REF, moving_slide=MOV)
    assert "displacement_um_p50" not in px["stages"][STAGE_RIGID]

    um = wsq.run(
        ref,
        mov,
        warp,
        [STAGE_RIGID],
        ref_slide=REF,
        moving_slide=MOV,
        pixel_size_um=0.325,
    )
    assert um["stages"][STAGE_RIGID]["displacement_um_p50"] == pytest.approx(
        2.0 * 0.325
    )
    assert um["params"]["pixel_size_um"] == pytest.approx(0.325)


def test_run_rejects_a_stage_plan_without_the_anchor(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    with pytest.raises(ValueError, match="anchor stage"):
        wsq.run(
            ref,
            mov,
            _shift_warp({STAGE_NATIVE: 0.0}),
            [STAGE_NATIVE],
            ref_slide=REF,
            moving_slide=MOV,
        )


def test_counts_report_the_native_feature_totals(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc(n=6))
    mov = _write(tmp_path, "mov.geojson", _grid_fc(n=4, dx=1.0))
    warp = _shift_warp({STAGE_RIGID: 0.0})
    out = wsq.run(ref, mov, warp, [STAGE_RIGID], ref_slide=REF, moving_slide=MOV)
    assert out["counts"] == {"features_ref": 6, "features_moving": 4}
    assert out["matching"]["n_pairs"] == 4
    assert out["matching"]["pair_fraction"] == pytest.approx(1.0)  # vs min(6, 4)
    assert out["matching"]["pair_fraction_ref"] == pytest.approx(4 / 6)


# ── report writing ─────────────────────────────────────────────────────────────
def test_write_report_emits_the_documented_record(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    out_f = tmp_path / "p1_mov_seg_qc.json"
    warp = _shift_warp({STAGE_NATIVE: 0.0, STAGE_RIGID: -98.0, STAGE_MICRO: -100.0})

    rec = wsq.write_report(
        None,
        REF,
        MOV,
        ref,
        mov,
        str(out_f),
        patient_id="p1",
        warp=warp,
        stages=[STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO],
        separable=False,
        note="test note",
    )

    on_disk = json.loads(out_f.read_text())
    assert on_disk == rec
    assert rec["patient_id"] == "p1"
    assert rec["moving"] == MOV and rec["reference"] == REF
    assert rec["stages_separable"] is False
    assert rec["note"] == "test note"
    assert rec["stage_order"] == [STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO]
    assert set(rec["stages"]) == {STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO}
    assert "iou_mean" in rec["stages"][STAGE_RIGID]
    assert "displacement_px_p90" in rec["stages"][STAGE_RIGID]


def test_record_is_honest_that_rigid_includes_micro_rigid_at_level_1(tmp_path):
    """At micro_reg>=1 the 'rigid' stage is really affine ∘ micro-rigid (VALIS composes them
    into slide.M). The record must say so rather than let a reader assume pure affine."""
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    rec = wsq.write_report(
        None,
        REF,
        MOV,
        ref,
        mov,
        str(tmp_path / "o.json"),
        warp=_shift_warp({STAGE_RIGID: 0.0}),
        stages=[STAGE_RIGID],
        micro_reg=1,
    )
    assert rec["micro_reg"] == 1
    assert rec["rigid_includes_micro_rigid"] is True


def test_record_says_rigid_is_pure_affine_at_level_0(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    rec = wsq.write_report(
        None,
        REF,
        MOV,
        ref,
        mov,
        str(tmp_path / "o.json"),
        warp=_shift_warp({STAGE_RIGID: 0.0}),
        stages=[STAGE_RIGID],
        micro_reg=0,
    )
    assert rec["micro_reg"] == 0
    assert rec["rigid_includes_micro_rigid"] is False


def test_record_omits_micro_reg_fields_when_it_is_unknown(tmp_path):
    """When the level isn't supplied (e.g. an injected-warp unit run), don't fabricate it."""
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    rec = wsq.write_report(
        None,
        REF,
        MOV,
        ref,
        mov,
        str(tmp_path / "o.json"),
        warp=_shift_warp({STAGE_RIGID: 0.0}),
        stages=[STAGE_RIGID],
    )
    assert "micro_reg" not in rec
    assert "rigid_includes_micro_rigid" not in rec


def test_write_report_defaults_display_names_to_the_slide_names(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    rec = wsq.write_report(
        None,
        REF,
        MOV,
        ref,
        mov,
        str(tmp_path / "o.json"),
        warp=_shift_warp({STAGE_RIGID: 0.0}),
        stages=[STAGE_RIGID],
    )
    assert rec["moving"] == MOV and rec["reference"] == REF


# ── the warp is asked for exactly the stages that were planned ─────────────────
def test_each_planned_stage_is_warped_for_both_slides_and_no_others(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=1.0))
    warp = _shift_warp({STAGE_NATIVE: 0.0, STAGE_RIGID: 0.0, STAGE_MICRO: 0.0})

    wsq.run(
        ref,
        mov,
        warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO],
        ref_slide=REF,
        moving_slide=MOV,
    )

    assert sorted(warp.calls) == sorted(
        [(s, st) for st in (STAGE_NATIVE, STAGE_RIGID, STAGE_MICRO) for s in (REF, MOV)]
    )


# ── CLI surface ────────────────────────────────────────────────────────────────
def test_parse_args_defaults_match_the_documented_behaviour():
    a = wsq.parse_args(
        [
            "--pickle",
            "p.pickle",
            "--ref-slide",
            "r",
            "--moving-slide",
            "m",
            "--ref-geojson",
            "r.geojson",
            "--moving-geojson",
            "m.geojson",
            "--output",
            "o.json",
        ]
    )
    assert a.checkpoint_dir is None
    assert a.clip_to_frame is False  # QC does not flatten boundary cells onto the crop
    assert a.match_radius_factor == wsq.DEFAULT_MATCH_RADIUS_FACTOR
    assert a.match_radius_px is None
    assert a.supersample == wsq.DEFAULT_SUPERSAMPLE
    assert a.iou_thresh == wsq.DEFAULT_IOU_THRESH
    assert a.crop == "overlap"
    assert a.micro_reg is None  # unknown unless the caller passes it
    assert a.pairing == "lsa"


def test_parse_args_accepts_micro_reg_level():
    a = wsq.parse_args(
        [
            "--pickle",
            "p.pickle",
            "--ref-slide",
            "r",
            "--moving-slide",
            "m",
            "--ref-geojson",
            "r.geojson",
            "--moving-geojson",
            "m.geojson",
            "--output",
            "o.json",
            "--micro-reg",
            "1",
        ]
    )
    assert a.micro_reg == 1


# ── the tiled (STARE) dispatch: JVM-free, driven through the real CLI main() ────
#
# Folded in from the former tests/test_warp_seg_qc_tiled.py when
# modules/local/warp_seg_qc_tiled.nf was merged into modules/local/warp_seg_qc.nf
# (lib/WarpBackends.groovy) -- both exercise the same, method-agnostic
# bin/warp_seg_qc.py, which this merge left unchanged.
def test_reg_qc2_tiled_cli_scores_through_a_manifest_without_a_jvm(tmp_path):
    # manifest: reference is identity; moving rigid-aligns to within 4px, a uniform mesh finishes it
    manifest = {
        "ref_slide": "ref",
        "slides": {
            "ref": {"M0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "mesh": None},
            "mov": {
                "M0": [[1, 0, -96], [0, 1, 0], [0, 0, 1]],
                "mesh": {
                    "grid_x": [0.0, 1000.0],
                    "grid_y": [0.0, 1000.0],
                    "displacements": [
                        [[-4.0, 0.0], [-4.0, 0.0]],
                        [[-4.0, 0.0], [-4.0, 0.0]],
                    ],
                },
            },
        },
    }
    man_f = _write(tmp_path, "manifest.json", manifest)
    ref_gj = _write(tmp_path, "ref.geojson", _grid_fc())
    mov_gj = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    out_f = tmp_path / "seg_qc.json"

    # returns None (no JVM path taken); would raise if it tried to import/boot VALIS
    wsq.main(
        [
            "--method",
            "tiled",
            "--pickle",
            man_f,
            "--ref-slide",
            "ref",
            "--moving-slide",
            "mov",
            "--ref-geojson",
            ref_gj,
            "--moving-geojson",
            mov_gj,
            "--output",
            str(out_f),
            "--patient-id",
            "P001",
            "--supersample",
            "4",
        ]
    )

    rec = json.loads(out_f.read_text())
    assert rec["stage_order"] == ["native", "rigid", "refined"]
    assert rec["stages_separable"] is True
    assert "micro_reg" not in rec  # a tiled run has no micro stage to report
    s = rec["stages"]
    assert s["rigid"]["displacement_px_p50"] == pytest.approx(4.0)
    assert s["refined"]["displacement_px_p50"] == pytest.approx(0.0, abs=1e-9)
    assert s["rigid"]["iou_mean"] < s["refined"]["iou_mean"]
    assert rec["matching"]["n_pairs"] == 6


# ── per-cell IoU ───────────────────────────────────────────────────────────────
def test_per_cell_carries_the_final_stage_iou(tmp_path):
    ref = _write(tmp_path, "ref.geojson", _grid_fc())
    mov = _write(tmp_path, "mov.geojson", _grid_fc(dx=100.0))
    warp = _shift_warp(
        {STAGE_NATIVE: 0.0, STAGE_RIGID: -96.0, STAGE_NON_RIGID: -99.0, STAGE_MICRO: -100.0}
    )
    out = wsq.run(
        ref, mov, warp,
        [STAGE_NATIVE, STAGE_RIGID, STAGE_NON_RIGID, STAGE_MICRO],
        ref_slide=REF, moving_slide=MOV, supersample=4,
    )
    per_cell = out["_per_cell"]
    assert per_cell["stage"] == STAGE_MICRO
    assert per_cell["iou"].shape == per_cell["residual_px"].shape
    assert np.allclose(per_cell["iou"], 1.0)


def test_per_cell_csv_has_an_iou_column(tmp_path):
    per_cell = {
        "stage": "micro",
        "ref_xy": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "residual_px": np.array([0.5, 1.5]),
        "iou": np.array([0.9, np.nan]),
    }
    path = tmp_path / "r.csv"
    assert wsq.write_per_cell_csv(str(path), per_cell, "mov") == 2
    lines = path.read_text().splitlines()
    assert lines[0] == "moving,ref_x,ref_y,residual_px,iou,stage"
    assert lines[1].split(",")[4] == "0.900000"
    assert lines[2].split(",")[4] == ""
