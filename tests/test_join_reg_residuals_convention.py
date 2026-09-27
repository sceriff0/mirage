"""Tie ``export_spatialdata.join_reg_residuals`` to the premise that makes it correct.

``join_reg_residuals`` is the last convention-sensitive pairing in this repo with
nothing testing its premise. It matches ``warp_seg_qc``'s per-cell residuals --
whose ``ref_x``/``ref_y`` trace back to ``bin/mask_to_geojson.py``'s rings, and are
therefore **centre**-of-pixel -- against ``quant[["x", "y"]]``, raw regionprops
output and therefore **centre** too. It reads the quantification CSV directly
rather than ``obsm["spatial"]``, so the centre->corner conversion
``build_table`` applies never touches it. The join needs no offset.

That correctness rests entirely on ``bin/mask_to_geojson.py`` keeping its
deliberate opt-out from ``bin/utils/pixel_convention.py``. Nothing in the suite
failed if somebody "fixed" that opt-out for consistency: both producers would
still run, the join would still pair every residual with a cell, and every cell
would silently acquire a systematic ``hypot(0.5, 0.5) == 0.707 px`` bias in the
distance it was matched at. This file is what fails instead.

Three assertions, in increasing strength:

1. **The opt-out holds, measured over the real modules.** Not a grep for the
   literal ``0.5`` -- that passes the moment somebody renames the constant, and
   it would not see an offset applied through ``centre_to_corner()``. The two
   producers are RUN, on the same mask, and their centroids must coincide.
2. **The join reads the CSV, not the store.** Structural, over ``ast`` rather
   than text: the array handed to ``join_reg_residuals`` must come from
   ``quant[["x", "y"]]`` and must not pass through ``centre_to_corner``.
3. **A wrong offset is DETECTABLE at the public interface.** The premise is not
   "the source text currently looks right", it is "the join changes if the
   convention changes". Perturbing one side by ``CORNER_OFFSET`` must change
   what ``join_reg_residuals`` returns.

What this does NOT claim: that the shipped join radius
(``spatialdata_residual_join_max_px = 15.0``) would notice. It would not -- and
``test_the_offset_is_invisible_at_the_shipped_join_radius`` pins exactly that,
because it is the reason a silent bias is the realistic failure and a broken
join is not. The tight radius used in the detection test is the instrument that
makes the sub-pixel bias observable through a function that only ever reports
whether a pair fell inside a radius.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")
pytest.importorskip("scipy")

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BIN / "utils"))

import mask_to_geojson  # noqa: E402
from pixel_convention import CORNER_OFFSET, centre_to_corner  # noqa: E402
from utils import cell_pairs as cp  # noqa: E402

_ESD_PATH = BIN / "export_spatialdata.py"
_CQ_PATH = BIN / "cell_qc.py"
_REG_PATH = BIN / "utils" / "reg_residuals.py"
_esd_spec = importlib.util.spec_from_file_location("export_spatialdata", _ESD_PATH)
esd = importlib.util.module_from_spec(_esd_spec)
_esd_spec.loader.exec_module(esd)

_ecp_spec = importlib.util.spec_from_file_location(
    "extract_cell_properties", BIN / "extract_cell_properties.py"
)
ecp = importlib.util.module_from_spec(_ecp_spec)
_ecp_spec.loader.exec_module(ecp)


# The bias a "consistency fix" to mask_to_geojson would introduce: half a pixel
# in each axis, so 0.707 px in the plane. Derived from CORNER_OFFSET rather than
# written out, so renaming or re-valuing the constant cannot leave a stale
# number here.
_DIAGONAL_BIAS = float(np.hypot(CORNER_OFFSET, CORNER_OFFSET))

# Solid squares, deliberately: a square's marching-squares outline is traced
# exactly (its corners survive `approximate_polygon`), so its polygon centroid
# equals its pixel centroid to floating-point exactness. A thin or irregular
# blob picks up a simplification error of ~0.1 px, which is the same order as
# the 0.707 px effect under test -- the fixture would then be measuring the
# simplifier, not the convention. `(row, col, side)`.
_SQUARES = ((3, 3, 6), (20, 10, 8), (40, 35, 5), (10, 45, 7), (48, 3, 4))


def _mask():
    m = np.zeros((60, 60), dtype=np.int32)
    for i, (r, c, side) in enumerate(_SQUARES, start=1):
        m[r : r + side, c : c + side] = i
    return m


def _qc_side(mask):
    """What warp_seg_qc's ``ref_x``/``ref_y`` are: centroids of mask_to_geojson rings.

    ``warp_seg_qc`` runs its own segmentation and takes the reference-cell
    centroids off the traced rings (``cell_pairs.feature_area_centroid``), which
    is what is reproduced here. Returns ``(labels, centroids_xy)``.
    """
    fc = mask_to_geojson.mask_to_feature_collection(mask)
    labels = np.array(
        [f["properties"]["label"] for f in fc["features"]], dtype=np.int64
    )
    _area, cent = cp.feature_area_centroid(cp.from_feature_collection(fc))
    return labels, np.asarray(cent, dtype=float)


def _quant_side(mask, labels):
    """What ``quant[["x", "y"]]`` is: raw regionprops centroids, in label order."""
    props, _filtered, _ = ecp.extract_morphology(mask)
    return props.loc[labels][["x", "y"]].to_numpy(dtype=float)


def _residual_csv(path, centroids_xy, residuals, moving="cycle2"):
    pd.DataFrame(
        {
            "moving": moving,
            "ref_x": centroids_xy[:, 0],
            "ref_y": centroids_xy[:, 1],
            "residual_px": residuals,
            "stage": "micro",
        }
    ).to_csv(path, index=False)
    return str(path)


# ── 1. the opt-out holds ───────────────────────────────────────────────────────
def test_mask_to_geojson_rings_are_centre_of_pixel_like_regionprops():
    """The two sides of the join are produced in the SAME convention, measured.

    Both producers are run on one mask. ``mask_to_geojson``'s ring centroids and
    ``extract_morphology``'s regionprops centroids must coincide. Adding
    ``CORNER_OFFSET`` inside ``mask_to_geojson`` -- by any spelling, including a
    call to ``centre_to_corner`` or a renamed constant -- moves every ring and
    fails this, which a grep for the literal ``0.5`` would not.
    """
    mask = _mask()
    labels, qc = _qc_side(mask)
    quant = _quant_side(mask, labels)

    assert len(labels) == len(_SQUARES), (
        f"only {len(labels)} of {len(_SQUARES)} cells produced a ring -- the "
        "comparison below would be made on a subset chosen by accident"
    )
    assert np.abs(qc - quant).max() < 1e-9, (
        "bin/mask_to_geojson.py's rings no longer sit in the same pixel convention "
        "as regionprops. That opt-out is what makes "
        "export_spatialdata.join_reg_residuals correct: it matches these rings' "
        "centroids against the quantification CSV's raw regionprops x/y, so a "
        "corner offset here gives every joined cell a systematic "
        f"{_DIAGONAL_BIAS:.3f} px bias and nothing else in the suite notices. "
        "See bin/utils/pixel_convention.py."
    )


def test_the_premise_is_not_vacuous():
    """The offset under discussion is real and non-zero.

    If ``CORNER_OFFSET`` were ever 0, every assertion in this file would still
    pass while checking nothing -- the perturbed and unperturbed fixtures would
    be identical. Pin it.
    """
    assert CORNER_OFFSET != 0
    assert _DIAGONAL_BIAS == pytest.approx(0.7071, abs=1e-4)
    mask = _mask()
    _labels, qc = _qc_side(mask)
    shifted = centre_to_corner(qc)
    assert np.linalg.norm(shifted - qc, axis=1).min() == pytest.approx(_DIAGONAL_BIAS)


# ── 2. the join reads the CSV, not the store ───────────────────────────────────
def _call_and_enclosing_function(tree, name):
    """The single call to ``name(...)`` and the FunctionDef containing it."""
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
                and node is not func
            ):
                return node, func
    return None, None


def test_join_reg_residuals_is_fed_the_quantification_csv_not_obsm_spatial():
    """The centroids handed to the join come from ``quant[["x", "y"]]`` unconverted.

    Structural rather than behavioural: the array is assembled in the
    ``build_sdata`` composition root, whose remaining body needs the real
    ``spatialdata``/``geopandas`` stack. So this reads the module with ``ast`` --
    a parse, not a grep -- and asserts the argument's provenance.

    The failure it exists to catch is somebody "simplifying" the builder to reuse
    ``adata.obsm["spatial"]``, which ``build_table`` has already converted to
    corner-of-pixel. That reads as a tidy-up and is a 0.707 px bias.
    """
    src = _ESD_PATH.read_text()
    tree = ast.parse(src)

    call, func = _call_and_enclosing_function(tree, "join_reg_residuals")
    assert call is not None, (
        "no call to join_reg_residuals found in bin/export_spatialdata.py -- this "
        "guard has lost its subject; repoint it or delete it"
    )
    assert len(call.args) >= 2, "join_reg_residuals is no longer called positionally"

    arg = call.args[1]
    assert isinstance(arg, ast.Name), (
        "the centroids argument is no longer a plain name; this guard can no "
        "longer trace its provenance and must be rewritten"
    )

    assigns = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == arg.id for t in node.targets)
    ]
    assert len(assigns) == 1, (
        f"expected exactly one assignment to {arg.id!r} in {func.name}(), found "
        f"{len(assigns)}"
    )

    source = ast.get_source_segment(src, assigns[0]) or ""
    assert 'quant[["x", "y"]]' in source, (
        f"{arg.id!r} is no longer read straight off the quantification CSV's x/y "
        "columns. Those are raw regionprops centroids -- centre-of-pixel, the same "
        "convention as the residual CSV's ref_x/ref_y. Any other source (obsm, a "
        "converted copy) silently biases the join by "
        f"{_DIAGONAL_BIAS:.3f} px. See bin/utils/pixel_convention.py.\n  {source}"
    )
    for forbidden in ("centre_to_corner", "obsm"):
        assert forbidden not in source, (
            f"{arg.id!r} now passes through {forbidden!r}: that is the "
            "corner-of-pixel side, and the residual CSV it is matched against is "
            f"centre-of-pixel.\n  {source}"
        )

    reg_src = _REG_PATH.read_text()
    reg_tree = ast.parse(reg_src)
    for fn_name in ("join_one", "join_reg_residuals"):
        body = (
            ast.get_source_segment(
                reg_src,
                next(
                    n
                    for n in ast.walk(reg_tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn_name
                ),
            )
            or ""
        )
        for forbidden in ("centre_to_corner", "obsm"):
            assert forbidden not in body, (
                f"{fn_name} itself now mentions {forbidden!r}; both of its "
                "sides are centre-of-pixel and it must not convert either"
            )

    # Non-vacuous: the conversion IS available in this module and IS used
    # elsewhere in it, so its absence on this path is a choice, not an omission.
    assert "centre_to_corner" in src


# ── 3. a wrong offset is detectable ────────────────────────────────────────────
def test_a_corner_offset_on_either_side_changes_the_join(tmp_path):
    """Perturbing one side by ``CORNER_OFFSET`` must change what the join returns.

    This is the assertion that actually ties the join to its premise. The first
    half is the world as shipped: residual ref_x/ref_y taken from
    ``mask_to_geojson`` rings, cell centroids taken from regionprops, joined at a
    radius tighter than the bias -- every residual lands on its own cell at
    distance zero. The second half is the world after somebody "fixes"
    ``mask_to_geojson`` for consistency: the same join, at the same radius, now
    matches nothing.
    """
    mask = _mask()
    labels, qc = _qc_side(mask)
    quant = _quant_side(mask, labels)
    residuals = np.arange(1, len(labels) + 1, dtype=float)

    # A radius below the 0.707 px bias, so the effect is visible through a
    # function whose only report is "inside the radius or not". See the module
    # docstring: at the shipped 15.0 px this is invisible, which is the point.
    tight = _DIAGONAL_BIAS / 2.0

    ok_csv = _residual_csv(tmp_path / "centre.csv", qc, residuals)
    out, stats = esd.join_reg_residuals([ok_csv], quant, labels, max_dist_px=tight)
    assert out["cycle2"].tolist() == residuals.tolist(), (
        "with both sides in centre-of-pixel each residual must land on its own cell"
    )
    assert stats["slides"]["cycle2"]["join_fraction"] == 1.0

    bad_csv = _residual_csv(tmp_path / "corner.csv", centre_to_corner(qc), residuals)
    out_bad, stats_bad = esd.join_reg_residuals(
        [bad_csv], quant, labels, max_dist_px=tight
    )
    assert stats_bad["slides"]["cycle2"]["joined"] == 0, (
        "a CORNER_OFFSET applied to one side of this join left the result "
        "unchanged, so this guard cannot detect the convention it exists to "
        "protect -- the fixture is wrong, not the pipeline"
    )
    assert out_bad["cycle2"].isna().all()


def test_the_offset_is_invisible_at_the_shipped_join_radius(tmp_path):
    """Why the failure would be a silent bias and not a broken join.

    ``spatialdata_residual_join_max_px`` ships at 15.0 px. A 0.707 px shift is
    two orders of magnitude inside that, so every pair still joins, to the same
    cell, and every summary statistic in ``stats`` is byte-identical. Only the
    match DISTANCE moves, and that is never reported. Pinning this is what makes
    the tight-radius test above honest about what it is doing, and it is the
    reason the opt-out needs a guard at all rather than being caught in practice.
    """
    mask = _mask()
    labels, qc = _qc_side(mask)
    quant = _quant_side(mask, labels)
    residuals = np.arange(1, len(labels) + 1, dtype=float)
    shipped = 15.0
    assert shipped > _DIAGONAL_BIAS * 10

    ok_csv = _residual_csv(tmp_path / "centre.csv", qc, residuals)
    bad_csv = _residual_csv(tmp_path / "corner.csv", centre_to_corner(qc), residuals)

    out, stats = esd.join_reg_residuals([ok_csv], quant, labels, max_dist_px=shipped)
    out_bad, stats_bad = esd.join_reg_residuals(
        [bad_csv], quant, labels, max_dist_px=shipped
    )

    assert out["cycle2"].tolist() == out_bad["cycle2"].tolist()
    assert stats["slides"] == stats_bad["slides"]


# ── 2b. the same call-site check, for CELL_QC's join_one call ──────────────────
def test_cell_qc_feeds_join_one_the_quantification_csv_not_a_converted_copy():
    """CELL_QC's residual join has the same premise as EXPORT_SPATIALDATA's.

    ``bin/cell_qc.py`` calls ``join_one`` directly (rather than
    ``join_reg_residuals``) with the quantification table's raw ``x``/``y``, named
    ``centroids`` and assigned from ``quant_xy[["x", "y"]]``. Same failure mode as
    the ``export_spatialdata`` guard above: a "simplification" that routes this
    through a corner-converted copy would give every joined cell the same
    ``0.707 px`` bias, silently, because the shipped join radius does not notice it.
    """
    src = _CQ_PATH.read_text()
    tree = ast.parse(src)

    call, func = _call_and_enclosing_function(tree, "join_one")
    assert call is not None, (
        "no call to join_one found in bin/cell_qc.py -- this guard has lost its "
        "subject; repoint it or delete it"
    )
    assert len(call.args) >= 2, "join_one is no longer called positionally"

    arg = call.args[1]
    assert isinstance(arg, ast.Name), (
        "the centroids argument is no longer a plain name; this guard can no "
        "longer trace its provenance and must be rewritten"
    )

    assigns = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == arg.id for t in node.targets)
    ]
    assert len(assigns) == 1, (
        f"expected exactly one assignment to {arg.id!r} in {func.name}(), found "
        f"{len(assigns)}"
    )

    source = ast.get_source_segment(src, assigns[0]) or ""
    assert 'quant_xy[["x", "y"]]' in source, (
        f"{arg.id!r} is no longer read straight off the quantification table's "
        "x/y columns. Those are raw regionprops centroids -- centre-of-pixel, the "
        "same convention as the residual CSV's ref_x/ref_y. Any other source "
        "(obsm, a converted copy) silently biases the join by "
        f"{_DIAGONAL_BIAS:.3f} px. See bin/utils/pixel_convention.py.\n  {source}"
    )
    for forbidden in ("centre_to_corner", "obsm"):
        assert forbidden not in source, (
            f"{arg.id!r} now passes through {forbidden!r}: that is the "
            "corner-of-pixel side, and the residual CSV it is matched against is "
            f"centre-of-pixel.\n  {source}"
        )
