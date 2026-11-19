"""A cell with a NaN centroid must be skipped, not crash the combined export.

cells.geojson is the file qupath-extension-flowpath consumes; one bad cell must
not take the whole export down with it.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

# Imported at collection time: the first import runs logger.configure_logging(),
# which clears the root logger's handlers once; doing it before test setup keeps
# that from removing pytest's caplog handler.
from export_geojson import export_combined_geojson


def test_nan_centroid_is_skipped_and_valid_cells_still_export(tmp_path, caplog):
    """One NaN-x row, one NaN-y row, one valid row: both bad rows drop, the good one survives."""
    df = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "x": [10.0, np.nan, 30.0],
            "y": [10.0, 20.0, np.nan],
            "area": [100.0, 100.0, 100.0],
        }
    )
    cell_contours = {
        "1": [[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]],
    }
    nucleus_contours = {}

    with caplog.at_level(logging.INFO, logger="export_geojson"):
        counts = export_combined_geojson(
            df,
            str(tmp_path),
            0.325,
            marker_cols=["area"],
            cell_contours=cell_contours,
            nucleus_contours=nucleus_contours,
            prefix="cells",
        )

    # The point is that this returns at all with a NaN centroid in row 2.
    assert counts == {"cells": 1, "cells_wholecell": 1}

    combined = (tmp_path / "cells.geojson").read_text()
    gj = json.loads(combined)
    assert len(gj["features"]) == 1

    # The skipped count reported to the operator must reflect both dropped
    # rows (NaN x and NaN y).
    skip_lines = [r.message for r in caplog.records if "skipped" in r.message]
    assert skip_lines, "expected a log line reporting the skipped-cell count"
    assert "skipped 2" in skip_lines[-1]
