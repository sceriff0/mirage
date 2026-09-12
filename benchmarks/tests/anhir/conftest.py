"""A miniature ANHIR download: cover CSV, landmark tree, bUnwarpJ baseline folder.

Three cases mirror the real layout (``<tissue>_<n>/scale-<k>pc/<name>``): two
training cases with public target landmarks and one evaluation case with a
target landmark path that resolves to nothing, exactly as the real archive
behaves for held-out cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.anhir import dataset as ds
from benchmarks.anhir.landmarks import write_landmarks

RNG = np.random.default_rng(7)

CASES = [
    # case_id, source, target, status, size
    (
        0,
        "COAD_01/scale-25pc/S1.jpg",
        "COAD_01/scale-25pc/HE.jpg",
        "training",
        (1600, 1200),
    ),
    (
        1,
        "COAD_01/scale-25pc/S3.jpg",
        "COAD_01/scale-25pc/HE.jpg",
        "training",
        (1600, 1200),
    ),
    (
        2,
        "lung-lesion_2/scale-100pc/29-041-Izd2-w35-Cc10-5-les1.jpg",
        "lung-lesion_2/scale-100pc/29-041-Izd2-w35-CD31-3-les1.jpg",
        "evaluation",
        (800, 600),
    ),
]

# Source landmarks are the target landmarks shifted by a constant offset, so the
# "initial" pose has a known TRE and any warp that undoes the offset is perfect.
OFFSET = np.array([30.0, -20.0])


def _landmarks(n: int, size) -> np.ndarray:
    w, h = size
    return np.column_stack([RNG.uniform(50, w - 50, n), RNG.uniform(50, h - 50, n)])


@pytest.fixture
def anhir_root(tmp_path):
    """``{root, cover, images, landmarks, baseline, cases}``."""
    root = tmp_path / "anhir"
    images = root / "images"
    landmarks = root / "landmarks"
    baseline = root / "BmUnwarpJ"
    rows = []
    # One landmark set per TARGET image, as in the real archive: COAD_01's HE.csv
    # is the target of every source in that folder, so the source files are all
    # index-aligned to the same target file.
    targets: dict = {}
    for case_id, src, tgt, status, size in CASES:
        target_xy = targets.setdefault(tgt, _landmarks(8, size))
        source_xy = target_xy + OFFSET
        src_l = src.replace(".jpg", ".csv")
        tgt_l = tgt.replace(".jpg", ".csv")
        write_landmarks(landmarks / src_l, source_xy)
        if status == "training":
            write_landmarks(landmarks / tgt_l, target_xy)
        # the baseline: a warp that removes 90 % of the offset, in ImageJ point format
        bdir = baseline / str(case_id)
        bdir.mkdir(parents=True)
        warped = source_xy - 0.9 * OFFSET
        lines = ["point", str(len(warped))] + [f"{x} {y}" for x, y in warped]
        (bdir / "warped_source_landmarks.txt").write_text("\n".join(lines) + "\n")
        (bdir / "TIME.txt").write_text("120000")  # 2 min in ms
        rows.append(
            {
                ds.COL_DIAGONAL: float(np.hypot(*size)),
                ds.COL_SIZE: str(tuple(size)),
                ds.COL_SOURCE: src,
                ds.COL_SOURCE_LND: src_l,
                ds.COL_TARGET: tgt,
                ds.COL_TARGET_LND: tgt_l,
                ds.COL_STATUS: status,
                ds.COL_WARPED_TARGET: np.nan,
                ds.COL_WARPED_SOURCE: np.nan,
                ds.COL_TIME: np.nan,
            }
        )
    cover = root / "dataset_medium.csv"
    pd.DataFrame(rows).to_csv(cover, index=True)
    return {
        "root": root,
        "cover": cover,
        "images": images,
        "landmarks": landmarks,
        "baseline": baseline,
        "cases": ds.load_cases(cover),
    }
