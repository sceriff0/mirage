import importlib.util
from pathlib import Path

import numpy as np
import tifffile

_SPEC = importlib.util.spec_from_file_location(
    "merge_channels_pyramid",
    Path(__file__).resolve().parent.parent / "bin" / "merge_channels_pyramid.py",
)
mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcp)


def test_second_series_holds_uint32_masks(tmp_path):
    intens = np.random.randint(0, 4000, size=(3, 128, 128), dtype=np.uint16)
    masks = np.stack(
        [
            np.random.randint(
                0, 100000, size=(128, 128), dtype=np.uint32
            ),  # cell (>65535)
            np.random.randint(0, 100000, size=(128, 128), dtype=np.uint32),  # nuclei
        ]
    )
    out = tmp_path / "pyr.ome.tiff"
    mcp.write_pyramidal_ome_tiff(
        intens,
        str(out),
        channel_names=["DAPI", "CD8", "PANCK"],
        channel_colors=[(0, 0, 255), (255, 0, 0), (0, 255, 0)],
        mask_stack=masks,
        mask_names=["cell_mask", "nuclei_mask"],
        pyramid_resolutions=3,
    )
    with tifffile.TiffFile(out) as tif:
        assert len(tif.series) == 2
        assert tif.series[0].dtype == np.uint16  # intensities untouched
        s1 = tif.series[1].asarray()
        assert s1.dtype == np.uint32 and s1.shape == (2, 128, 128)
        np.testing.assert_array_equal(s1, masks)  # labels preserved exactly


def test_no_mask_stack_writes_single_series(tmp_path):
    intens = np.random.randint(0, 4000, size=(2, 64, 64), dtype=np.uint16)
    out = tmp_path / "pyr_nomask.ome.tiff"
    mcp.write_pyramidal_ome_tiff(
        intens,
        str(out),
        channel_names=["DAPI", "CD8"],
        channel_colors=[(0, 0, 255), (255, 0, 0)],
        mask_stack=None,
        mask_names=None,
        pyramid_resolutions=3,
    )
    with tifffile.TiffFile(out) as tif:
        assert len(tif.series) == 1  # unchanged behavior


def test_merge_channels_embeds_masks_with_real_segment_filenames(tmp_path):
    """merge_channels() finds patient-prefixed mask names.

    SEGMENT emits `${patient_id}_cell_mask.tif` / `${patient_id}_nuclei_mask.tif`
    (see modules/local/segment.nf), and Nextflow's `stageAs: 'masks/*'` preserves
    that real basename inside masks/ -- it does NOT rename files down to the bare
    `cell_mask.tif` / `nuclei_mask.tif`. merge_channels() must find masks named
    with a patient prefix, not just the bare names.
    """
    h, w = 96, 96

    channels_dir = tmp_path / "channels"
    channels_dir.mkdir()
    tifffile.imwrite(
        str(channels_dir / "DAPI.tif"),
        np.random.randint(0, 4000, size=(h, w), dtype=np.uint16),
    )
    tifffile.imwrite(
        str(channels_dir / "CD8.tif"),
        np.random.randint(0, 4000, size=(h, w), dtype=np.uint16),
    )

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    cell_mask = np.random.randint(0, 100000, size=(h, w), dtype=np.uint32)
    nuclei_mask = np.random.randint(0, 100000, size=(h, w), dtype=np.uint32)
    # Real SEGMENT/Nextflow-staged naming convention: patient-id prefixed.
    tifffile.imwrite(str(masks_dir / "P001_cell_mask.tif"), cell_mask)
    tifffile.imwrite(str(masks_dir / "P001_nuclei_mask.tif"), nuclei_mask)

    out = tmp_path / "pyramid.ome.tiff"
    mcp.merge_channels(
        input_dir=str(channels_dir),
        output_path=str(out),
        masks_dir=str(masks_dir),
        pyramid_resolutions=3,
    )

    with tifffile.TiffFile(str(out)) as tif:
        assert len(tif.series) == 2
        mask_series = tif.series[1]
        assert mask_series.dtype == np.uint32
        assert tuple(mask_series.shape) == (2, h, w)
        s1 = mask_series.asarray()
        np.testing.assert_array_equal(s1[0], cell_mask)
        np.testing.assert_array_equal(s1[1], nuclei_mask)


def test_merge_channels_masks_dir_with_no_masks_raises(tmp_path):
    """--masks-dir is only ever passed when the mask series is REQUIRED.

    A masks/ directory that exists but contains no matching *cell_mask.tif /
    *nuclei_mask.tif files must be a hard error, never a silent skip.
    """
    h, w = 32, 32

    channels_dir = tmp_path / "channels"
    channels_dir.mkdir()
    tifffile.imwrite(
        str(channels_dir / "DAPI.tif"),
        np.random.randint(0, 4000, size=(h, w), dtype=np.uint16),
    )

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    # Empty: no cell/nuclei mask files present.

    out = tmp_path / "pyramid.ome.tiff"
    import pytest

    with pytest.raises(ValueError, match="no \\*cell_mask.tif"):
        mcp.merge_channels(
            input_dir=str(channels_dir),
            output_path=str(out),
            masks_dir=str(masks_dir),
            pyramid_resolutions=3,
        )
