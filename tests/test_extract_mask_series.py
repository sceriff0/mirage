import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

BIN = Path(__file__).resolve().parent.parent / "bin"
_SPEC = importlib.util.spec_from_file_location(
    "merge_channels_pyramid", BIN / "merge_channels_pyramid.py"
)
mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcp)


def _pyramid_with_masks(path):
    intens = np.random.randint(0, 4000, size=(2, 96, 96), dtype=np.uint16)
    masks = np.stack(
        [
            np.arange(96 * 96, dtype=np.uint32).reshape(96, 96),
            np.arange(96 * 96, dtype=np.uint32).reshape(96, 96),
        ]
    )
    mcp.write_pyramidal_ome_tiff(
        intens,
        str(path),
        channel_names=["DAPI", "CD8"],
        channel_colors=[(0, 0, 255), (255, 0, 0)],
        mask_stack=masks,
        mask_names=["cell_mask", "nuclei_mask"],
        pyramid_resolutions=2,
    )
    return masks


def test_extract_recovers_masks(tmp_path):
    masks = _pyramid_with_masks(tmp_path / "p.ome.tiff")
    subprocess.run(
        [
            sys.executable,
            str(BIN / "extract_mask_series.py"),
            "--pyramid",
            str(tmp_path / "p.ome.tiff"),
            "--outdir",
            str(tmp_path),
        ],
        check=True,
    )
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "cell_mask.tif"), masks[0])
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "nuclei_mask.tif"), masks[1]
    )


def test_missing_series_fails_fast(tmp_path):
    intens = np.random.randint(0, 4000, size=(2, 64, 64), dtype=np.uint16)
    mcp.write_pyramidal_ome_tiff(
        intens,
        str(tmp_path / "n.ome.tiff"),
        channel_names=["DAPI", "CD8"],
        channel_colors=[(0, 0, 255), (255, 0, 0)],
        mask_stack=None,
        mask_names=None,
        pyramid_resolutions=2,
    )
    r = subprocess.run(
        [
            sys.executable,
            str(BIN / "extract_mask_series.py"),
            "--pyramid",
            str(tmp_path / "n.ome.tiff"),
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "no mask series" in (r.stderr + r.stdout).lower()


def test_float_mask_series_fails_fast(tmp_path):
    intens = np.random.randint(0, 4000, size=(2, 64, 64), dtype=np.uint16)
    fmask = np.random.rand(2, 64, 64).astype(np.float32)
    out = tmp_path / "float.ome.tiff"
    with tifffile.TiffWriter(out, bigtiff=True, ome=True) as tif:
        tif.write(
            intens,
            metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD8"]}},
            photometric="minisblack",
        )
        tif.write(
            fmask,
            metadata={"axes": "CYX", "Channel": {"Name": ["cell_mask", "nuclei_mask"]}},
            photometric="minisblack",
        )
    r = subprocess.run(
        [
            sys.executable,
            str(BIN / "extract_mask_series.py"),
            "--pyramid",
            str(out),
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "dtype" in (r.stderr + r.stdout).lower()
