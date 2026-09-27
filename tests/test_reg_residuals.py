"""The one spatial join of warp_seg_qc per-cell residuals onto cell labels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin" / "utils"))

from reg_residuals import join_one, join_reg_residuals  # noqa: E402

CENTROIDS = np.array([[10.0, 10.0], [50.0, 10.0], [90.0, 10.0]])


def _csv(tmp_path, rows, header="moving,ref_x,ref_y,residual_px,iou,stage"):
    p = tmp_path / "r.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(p)


def test_join_one_matches_nearest_cell_and_carries_iou(tmp_path):
    p = _csv(tmp_path, ["m,11,10,2.0,0.8,micro", "m,51,10,0.5,0.95,micro"])
    resid, iou, stats = join_one(p, CENTROIDS, 5.0)
    assert resid[0] == 2.0 and iou[0] == pytest.approx(0.8)
    assert resid[1] == 0.5 and iou[1] == pytest.approx(0.95)
    assert np.isnan(resid[2]) and np.isnan(iou[2])
    assert stats["joined"] == 2


def test_two_pairs_on_one_cell_keep_the_worst_residual_and_its_iou(tmp_path):
    p = _csv(tmp_path, ["m,11,10,1.0,0.9,micro", "m,9,10,3.0,0.4,micro"])
    resid, iou, _ = join_one(p, CENTROIDS, 5.0)
    assert resid[0] == 3.0 and iou[0] == pytest.approx(0.4)


def test_residual_csv_without_iou_gives_nan_iou(tmp_path):
    p = _csv(tmp_path, ["m,11,10,2.0,micro"], header="moving,ref_x,ref_y,residual_px,stage")
    resid, iou, _ = join_one(p, CENTROIDS, 5.0)
    assert resid[0] == 2.0 and np.isnan(iou).all()


def test_empty_csv_joins_nothing(tmp_path):
    p = _csv(tmp_path, [])
    resid, iou, _ = join_one(p, CENTROIDS, 5.0)
    assert resid is None and iou is None


def test_join_reg_residuals_keeps_its_frame_shape(tmp_path):
    p = _csv(tmp_path, ["m,11,10,2.0,0.8,micro"])
    frame, stats = join_reg_residuals([p], CENTROIDS, np.array([1, 2, 3]), 5.0)
    assert list(frame.columns) == ["m"]
    assert frame.loc[1, "m"] == 2.0 and np.isnan(frame.loc[2, "m"])
    assert stats["slides"]["m"]["cells_with_residual"] == 1
