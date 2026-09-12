"""`stare register` and the four-stage fan-out produce the same manifest and the same pixels.

Two executors of one method:

  (a) the pipeline's shape -- ``bin/tiled_coarse.py``, one ``bin/tiled_reg_tile.py`` per
      row of the tile plan, ``bin/tiled_solve.py``, ``bin/tiled_stitch.py`` -- driven here
      exactly as ``tests/test_tiled_fanout.py`` drives it, on the synthetic pair that file
      builds;
  (b) ``stare register --workers 2``, the package's single-process command, which maps the
      same reg-tile function over the same rows with a local pool.

Both are run with ``--solver legacy`` so the solve is the byte-for-byte pre-package one
and any disagreement is the executor's, not the solver's. The manifests must be equal
(JSON structure exact, floats to within 1e-9) and the stitched arrays identical.

WHY THERE IS NO "PRE-MOVE SCRIPTS vs SHIMS" LEG. The ``bin/tiled_*.py`` shims ARE the
package: each one imports its stage's ``main`` from ``stare.stages`` and replaces itself in
``sys.modules`` with the stage module, so there is no second implementation for (a) to be
compared against -- running (a) against "the old scripts" would be running the package
against itself. The parity that can drift is between the two EXECUTORS, and that is what
this file pins. The legacy solve's own byte-for-byte claim against the pre-package code is
pinned separately, from a verbatim copy, in ``packages/stare/tests/test_solve.py``.

Tagged like ``tests/test_tiled_fanout.py``: the COARSE anchor is DISK + LightGlue, so this
needs torch and kornia (CI installs both; ``tests/test_disk_test_actually_runs.py`` pins it).
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
pytest.importorskip("torch")
pytest.importorskip("kornia")
tifffile = pytest.importorskip("tifffile")

import tiled_coarse  # noqa: E402
import tiled_reg_tile  # noqa: E402
import tiled_solve  # noqa: E402
import tiled_stitch  # noqa: E402
from stare import cli  # noqa: E402

TILE, HALO, MAX_DIM = 128, 32, 256


def _textured(seed, n=384):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.uniform(0, 1, size=(n, n)), 2.0)
    return (img - img.min()) / (img.ptp() + 1e-9)


def _write_pair(tmp_path):
    """The synthetic pair of tests/test_tiled_fanout.py: a 2-channel slide and its
    rotated + shifted copy."""
    from skimage.transform import EuclideanTransform
    from skimage.transform import warp as sk_warp

    dapi, marker = _textured(0), _textured(1)
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
    ref_f, mov_f = tmp_path / "ref.ome.tiff", tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(ref_f), ref, photometric="minisblack")
    tifffile.imwrite(str(mov_f), mov, photometric="minisblack")
    return ref_f, mov_f


def _fanout(work, ref_f, mov_f):
    """Path (a): the shims, one call per stage invocation, as the Nextflow DAG runs them."""
    work.mkdir()
    m0_f, tiles_f = work / "m0.json", work / "tiles.csv"
    assert (
        tiled_coarse.main(
            [
                "--reference",
                str(ref_f),
                "--moving",
                str(mov_f),
                "--nuclear-index",
                "0",
                "--tile",
                str(TILE),
                "--halo",
                str(HALO),
                "--max-dim",
                str(MAX_DIM),
                "--out-m0",
                str(m0_f),
                "--out-tiles",
                str(tiles_f),
            ]
        )
        == 0
    )
    with open(tiles_f) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4
    for r in rows:
        out = work / f"ctrl_{r['ix']}_{r['iy']}.json"
        argv = ["--reference", str(ref_f), "--moving", str(mov_f), "--m0", str(m0_f)]
        argv += ["--nuclear-index", "0"]
        for k in ("ix", "iy", "cx", "cy", "rx0", "ry0", "rx1", "ry1"):
            argv += [f"--{k}", r[k]]
        argv += ["--out", str(out)]
        assert tiled_reg_tile.main(argv) == 0
    assert len(glob.glob(str(work / "ctrl_*.json"))) == len(rows)

    man_f, tre_f = work / "manifest.json", work / "tre.json"
    assert (
        tiled_solve.main(
            [
                "--m0",
                str(m0_f),
                "--controls",
                str(work / "ctrl_*.json"),
                "--gate-tre",
                "0.0",
                "--max-disp",
                str(HALO),
                "--solver",
                "legacy",
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
        == 0
    )
    reg_f = work / "mov_registered.ome.tiff"
    assert (
        tiled_stitch.main(
            [
                "--moving",
                str(mov_f),
                "--manifest",
                str(man_f),
                "--moving-name",
                "mov",
                "--out",
                str(reg_f),
                "--pixel-size",
                "0.325",
            ]
        )
        == 0
    )
    return man_f, tre_f, reg_f


def _register(work, ref_f, mov_f):
    """Path (b): the package's one-process command, tile stage over a 2-worker pool."""
    work.mkdir()
    man_f, tre_f = work / "manifest.json", work / "tre.json"
    reg_f = work / "mov_registered.ome.tiff"
    rc = cli.main(
        [
            "register",
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--out",
            str(reg_f),
            "--manifest",
            str(man_f),
            "--tre",
            str(tre_f),
            "--workdir",
            str(work / "stages"),
            "--workers",
            "2",
            "--reference-name",
            "ref",
            "--moving-name",
            "mov",
            "--tile",
            str(TILE),
            "--halo",
            str(HALO),
            "--max-dim",
            str(MAX_DIM),
            "--gate-tre",
            "0.0",
            "--solver",
            "legacy",
            "--pixel-size",
            "0.325",
        ]
    )
    assert rc == 0
    return man_f, tre_f, reg_f


def _assert_json_equal(a, b, path="$"):
    """Structural equality; floats to 1e-9, everything else exact."""
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), path
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)), f"{path}: {a} != {b}"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys differ"
        for k in a:
            _assert_json_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_json_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_stare_register_equals_the_pipeline_fanout(tmp_path):
    ref_f, mov_f = _write_pair(tmp_path)
    man_a, tre_a, reg_a = _fanout(tmp_path / "fanout", ref_f, mov_f)
    man_b, tre_b, reg_b = _register(tmp_path / "register", ref_f, mov_f)

    manifest_a = json.loads(man_a.read_text())
    manifest_b = json.loads(man_b.read_text())
    # premise: both ran the legacy solve and actually refined something
    assert manifest_a["slides"]["mov"]["solver"] == "legacy"
    assert manifest_a["slides"]["mov"]["mesh"] is not None
    _assert_json_equal(manifest_a, manifest_b)

    tre_a_doc, tre_b_doc = json.loads(tre_a.read_text()), json.loads(tre_b.read_text())
    assert tre_a_doc["n_tiles"] == tre_b_doc["n_tiles"] >= 4
    _assert_json_equal(tre_a_doc["solve"], tre_b_doc["solve"])

    pixels_a, pixels_b = tifffile.imread(str(reg_a)), tifffile.imread(str(reg_b))
    assert pixels_a.shape == pixels_b.shape and pixels_a.dtype == pixels_b.dtype
    assert np.array_equal(pixels_a, pixels_b), (
        "stitched pixels differ between executors"
    )
    assert pixels_a.any(), "premise: the stitched slide is not blank"


def test_the_plan_row_form_names_the_same_tile_as_the_explicit_form(tmp_path):
    """``--plan tiles.csv --row N`` and the explicit geometry write the identical control
    JSON -- the property a SLURM array job relies on."""
    ref_f, mov_f = _write_pair(tmp_path)
    m0_f, tiles_f = tmp_path / "m0.json", tmp_path / "tiles.csv"
    tiled_coarse.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--tile",
            str(TILE),
            "--halo",
            str(HALO),
            "--max-dim",
            str(MAX_DIM),
            "--out-m0",
            str(m0_f),
            "--out-tiles",
            str(tiles_f),
        ]
    )
    with open(tiles_f) as f:
        rows = list(csv.DictReader(f))
    row = len(rows) // 2  # an interior row, not the corner case at 0
    r = rows[row]
    common = ["--reference", str(ref_f), "--moving", str(mov_f), "--m0", str(m0_f)]
    explicit = tmp_path / "explicit.json"
    argv = list(common)
    for k in ("ix", "iy", "cx", "cy", "rx0", "ry0", "rx1", "ry1"):
        argv += [f"--{k}", r[k]]
    assert tiled_reg_tile.main(argv + ["--out", str(explicit)]) == 0
    by_row = tmp_path / "by_row.json"
    assert (
        tiled_reg_tile.main(
            common + ["--plan", str(tiles_f), "--row", str(row), "--out", str(by_row)]
        )
        == 0
    )
    assert json.loads(explicit.read_text()) == json.loads(by_row.read_text())
    assert json.loads(by_row.read_text())["ix"] == int(r["ix"])
