"""End-to-end test of the per-tile fan-out CLIs: coarse -> reg_tile(xN) -> solve -> stitch.

Runs the four little-process steps in sequence (as the Nextflow fan-out does) on a synthetic
reference/moving pair and checks the stitched slide is actually registered (residual TRE drops)
and non-negative. This proves the fan-out scripts compose into the same registration the monolithic
path produces — the distribution across tasks changes orchestration, not the math.
"""

from __future__ import annotations

import csv
import glob
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
pytest.importorskip("skimage")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")

import tiled_coarse  # noqa: E402
import tiled_reg_tile  # noqa: E402
import tiled_solve  # noqa: E402
import tiled_stitch  # noqa: E402
from reg_benchmark import benchmark  # noqa: E402


def _textured(seed, n=384):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(n, n)), 2.0)
    return (img - img.min()) / (img.ptp() + 1e-9)


def test_fanout_scripts_chain_into_a_registered_slide(tmp_path):
    dapi = _textured(0)
    marker = _textured(1)
    from skimage.transform import EuclideanTransform
    from skimage.transform import warp as sk_warp

    tform = EuclideanTransform(rotation=np.deg2rad(2.0), translation=(10.0, -6.0))
    ref = (np.stack([dapi, marker]) * 60000).astype(np.uint16)
    mov = (
        np.stack(
            [
                sk_warp(dapi, tform, mode="reflect"),
                sk_warp(marker, tform, mode="reflect"),
            ]
        )
        * 60000
    ).astype(np.uint16)

    ref_f = tmp_path / "ref.ome.tiff"
    mov_f = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref_f), ref, photometric="minisblack")
    tifffile.imwrite(str(mov_f), mov, photometric="minisblack")

    m0_f = tmp_path / "m0.json"
    tiles_f = tmp_path / "tiles.csv"
    tiled_coarse.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--nuclear-index",
            "0",
            "--tile",
            "128",
            "--halo",
            "32",
            # --max-dim is REQUIRED (no default): a default in bin/tiled_coarse.py would be a
            # fourth, unpinned copy of RegPresets.STARE.high.coarse_max_dim. 256 keeps this
            # fixture's 512 px planes at decimation factor 2, which is what the chain asserts.
            "--max-dim",
            "256",
            "--out-m0",
            str(m0_f),
            "--out-tiles",
            str(tiles_f),
        ]
    )

    # 2. one reg_tile task per row of the tile plan
    with open(tiles_f) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4
    for r in rows:
        out = tmp_path / f"ctrl_{r['ix']}_{r['iy']}.json"
        tiled_reg_tile.main(
            [
                "--reference",
                str(ref_f),
                "--moving",
                str(mov_f),
                "--m0",
                str(m0_f),
                "--nuclear-index",
                "0",
                "--ix",
                r["ix"],
                "--iy",
                r["iy"],
                "--cx",
                r["cx"],
                "--cy",
                r["cy"],
                "--rx0",
                r["rx0"],
                "--ry0",
                r["ry0"],
                "--rx1",
                r["rx1"],
                "--ry1",
                r["ry1"],
                "--out",
                str(out),
            ]
        )
    assert len(glob.glob(str(tmp_path / "ctrl_*.json"))) == len(rows)

    man_f = tmp_path / "manifest.json"
    tre_f = tmp_path / "tre.json"
    tiled_solve.main(
        [
            "--m0",
            str(m0_f),
            "--controls",
            str(tmp_path / "ctrl_*.json"),
            "--reference-name",
            "ref",
            "--moving-name",
            "mov",
            "--out-manifest",
            str(man_f),
            "--out-tre",
            str(tre_f),
        ]
    )
    # fix: the fan-out now emits the intrinsic TRE (spatial rigid heatmap), not just the manifest
    tre = json.loads(tre_f.read_text())
    assert tre["n_tiles"] == len(rows)
    assert len(tre["tiles"]) == len(rows)
    assert "coarse_tre_px" in tre and tre["rigid_tre_px"]["p50"] is not None

    reg_f = tmp_path / "mov_registered.ome.tiff"
    tiled_stitch.main(
        [
            "--moving",
            str(mov_f),
            "--manifest",
            str(man_f),
            "--out",
            str(reg_f),
            "--pixel-size",
            "0.325",
        ]
    )

    registered = tifffile.imread(str(reg_f))
    assert registered.shape == ref.shape
    assert registered.dtype == np.uint16
    assert registered.min() >= 0

    report = benchmark(
        ref[0].astype(float), mov[0].astype(float), registered[0].astype(float)
    )
    assert report["after"]["tre_px_median"] < report["before"]["tre_px_median"]
    assert report["after"]["tre_px_median"] < 3.0
