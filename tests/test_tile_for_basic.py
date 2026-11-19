"""TILE_FOR_BASIC: a stitched slide -> a multi-SITE OME-TIFF the nf-core module accepts.

nf-core's ``basicpy`` module runs ``labsyspharm/basicpy-docker-mcmicro``'s ``/opt/main.py``,
which builds its field-of-view axis as::

    istack = istack.stack(I=('M', 'T', 'Z')).transpose('C', 'I', 'Y', 'X')
    if len(istack.coords['I']) < 2 and not args.ignore_single_image_error:
        raise RuntimeError("The image is single sited. Was it saved in the correct way?")

so a mirage slide -- one stitched plane per channel, ``SizeM = SizeT = SizeZ = 1`` -- is
refused outright. ``bin/tile_for_basic.py`` is what makes it acceptable: mirage's existing
non-overlapping FOV grid is written onto the ``Z`` axis, so ``len(I) == n_tiles``.

The same script iterates channels the way ``/opt/main.py`` does
(``for c, channel_stack in enumerate(istack, 1)``, one profile fitted per channel), which is
why the axis order has to be ``CZYX`` and not ``ZCYX``.

Nothing here fits a profile -- BaSiC is the nf-core module's job. What is pinned is the
*container* the module is handed and the sidecar that lets APPLY_PROFILES put the slide
back together without re-deriving the grid.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

# No basicpy stub: the FOV-tiling helpers in `bin/utils/fov_tiling.py` import only numpy,
# so a stub here could only hide a real import.
import tile_for_basic  # noqa: E402
from fov_tiling import reconstruct_image_from_fovs  # noqa: E402


def _write_slide(path, stack, channel_names):
    tifffile.imwrite(
        str(path),
        stack,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(channel_names)}},
        ome=True,
    )


def _rng_stack(n_channels, height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(n_channels, height, width), dtype=np.uint16)


# ---------------------------------------------------------------------------
# The single-site check, which is the whole reason this process exists
# ---------------------------------------------------------------------------


def test_tiles_land_on_the_z_axis_so_the_module_sees_more_than_one_site(tmp_path):
    """SizeZ must equal the tile count, and SizeC the channel count.

    This is the precondition ``/opt/main.py`` tests. ``len(I) = SizeM * SizeT * SizeZ``;
    mirage writes no M and no T, so every site has to come from Z.
    """
    slide = tmp_path / "P001_ref.ome.tif"
    _write_slide(slide, _rng_stack(3, 200, 300), ["DAPI", "PANCK", "SMA"])

    out = tmp_path / "P001_ref_tiles.ome.tif"
    sidecar = tmp_path / "P001_ref_tiles.json"
    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(out),
        str(sidecar),
        channel_names=["DAPI", "PANCK", "SMA"],
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["DAPI", "CELLTOX"],
    )

    with tifffile.TiffFile(str(out)) as tif:
        assert tif.is_ome, (
            "the module reads the file through Bio-Formats; it must be OME"
        )
        ome = tif.ome_metadata
        series = tif.series[0]

    n_tiles = manifest["n_fovs_y"] * manifest["n_fovs_x"]
    assert n_tiles >= 2, "a single-sited stack is exactly what the module refuses"
    assert f'SizeZ="{n_tiles}"' in ome
    # DAPI is the fiducial and is not corrected, so only PANCK and SMA are fitted.
    assert f'SizeC="{len(manifest["profile_channels"])}"' in ome
    assert series.axes == "CZYX"
    assert series.shape == (
        len(manifest["profile_channels"]),
        n_tiles,
        *manifest["tile_shape"],
    )


def test_a_grid_of_one_tile_is_refused_rather_than_silently_written(tmp_path):
    """An image smaller than one FOV yields one site, which the module rejects.

    Failing here, with the knob named, beats failing inside a vendored module's
    container with "Was it saved in the correct way?".
    """
    slide = tmp_path / "small.ome.tif"
    _write_slide(slide, _rng_stack(2, 64, 64), ["DAPI", "PANCK"])

    with pytest.raises(ValueError, match="preproc_tile_size"):
        tile_for_basic.tile_for_basic(
            str(slide),
            str(tmp_path / "small_tiles.ome.tif"),
            str(tmp_path / "small_tiles.json"),
            channel_names=["DAPI", "PANCK"],
            fov_size=(1950, 1950),
        )


# ---------------------------------------------------------------------------
# The nuclear/fiducial skip
# ---------------------------------------------------------------------------


def test_the_configured_fiducial_is_excluded_from_the_fit(tmp_path):
    """CELLTOX, not DAPI, is the fiducial here.

    The decision goes through ``utils.metadata.is_nuclear`` (not a ``"DAPI" in name``
    check), is made once, here, and is recorded in the sidecar, so APPLY_PROFILES reads
    the answer rather than re-deciding.
    """
    slide = tmp_path / "s.ome.tif"
    names = ["CELLTOX", "PANCK", "DAPI_like"]
    _write_slide(slide, _rng_stack(3, 200, 200), names)

    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "s_tiles.ome.tif"),
        str(tmp_path / "s_tiles.json"),
        channel_names=names,
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["CELLTOX"],
    )

    assert manifest["corrected_channels"] == [1, 2]
    assert manifest["skipped_channels"] == [0]


def test_skip_nuclear_off_corrects_every_channel(tmp_path):
    slide = tmp_path / "s.ome.tif"
    names = ["DAPI", "PANCK"]
    _write_slide(slide, _rng_stack(2, 200, 200), names)

    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "s_tiles.ome.tif"),
        str(tmp_path / "s_tiles.json"),
        channel_names=names,
        fov_size=(100, 100),
        skip_nuclear=False,
        nuclear_markers=["DAPI", "CELLTOX"],
    )

    assert manifest["corrected_channels"] == [0, 1]
    assert manifest["skipped_channels"] == []


def test_a_celltox_only_panel_still_produces_a_readable_stack(tmp_path):
    """A panel whose every channel is the fiducial is a supported input.

    Nothing is corrected, so nothing NEEDS fitting -- but the module still runs, and an
    OME-TIFF with zero channels is not a thing. The stack therefore carries the channels
    anyway (``profile_channels``) while ``corrected_channels`` stays empty, and
    APPLY_PROFILES applies nothing. The two lists are separate for this case.
    """
    slide = tmp_path / "celltox.ome.tif"
    _write_slide(slide, _rng_stack(1, 200, 200), ["CELLTOX"])

    out = tmp_path / "celltox_tiles.ome.tif"
    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(out),
        str(tmp_path / "celltox_tiles.json"),
        channel_names=["CELLTOX"],
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["CELLTOX"],
    )

    assert manifest["corrected_channels"] == []
    assert manifest["profile_channels"] == [0]
    # tifffile drops a singleton C from the SERIES axes ("ZYX", shape (4, y, x)), but the
    # OME header -- which is what Bio-Formats, and therefore /opt/main.py, reads -- still
    # says SizeC=1 and SizeZ=4. Assert the header, not the numpy shape.
    with tifffile.TiffFile(str(out)) as tif:
        ome = tif.ome_metadata
    assert 'SizeC="1"' in ome
    assert f'SizeZ="{manifest["n_fovs_y"] * manifest["n_fovs_x"]}"' in ome


# ---------------------------------------------------------------------------
# The sidecar: APPLY_PROFILES must not have to re-derive the grid
# ---------------------------------------------------------------------------


def test_sidecar_positions_reassemble_the_original_exactly(tmp_path):
    """tile -> (identity) -> reassemble is the identity, byte for byte.

    Output-equivalence on this script's half of the round trip. If the positions
    are wrong by a pixel, or the padding of an edge tile leaks back in, this fails.
    """
    slide = tmp_path / "s.ome.tif"
    names = ["DAPI", "PANCK", "SMA"]
    # 250x330 divides by neither 100 nor 100, so every remainder branch of
    # split_image_into_fovs is exercised (some tiles get +1 pixel, edges are padded).
    original = _rng_stack(3, 250, 330, seed=7)
    _write_slide(slide, original, names)

    out = tmp_path / "s_tiles.ome.tif"
    sidecar = tmp_path / "s_tiles.json"
    tile_for_basic.tile_for_basic(
        str(slide),
        str(out),
        str(sidecar),
        channel_names=names,
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["DAPI"],
    )

    manifest = json.loads(sidecar.read_text())
    positions = [tuple(p) for p in manifest["positions"]]
    tiles = tifffile.imread(str(out))

    for stack_index, source_index in enumerate(manifest["profile_channels"]):
        rebuilt = reconstruct_image_from_fovs(
            tiles[stack_index], positions, tuple(manifest["image_shape"])
        )
        assert np.array_equal(rebuilt, original[source_index])


def test_sidecar_records_what_task_4_needs_and_says_which_version_it_is(tmp_path):
    slide = tmp_path / "s.ome.tif"
    names = ["DAPI", "PANCK"]
    _write_slide(slide, _rng_stack(2, 250, 250), names)

    sidecar = tmp_path / "s_tiles.json"
    tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "s_tiles.ome.tif"),
        str(sidecar),
        channel_names=names,
        fov_size=(100, 100),
        skip_nuclear=True,
        nuclear_markers=["DAPI"],
    )

    manifest = json.loads(sidecar.read_text())
    assert manifest["format_version"] == 1
    assert manifest["source_image"] == "s.ome.tif"
    assert manifest["source_dtype"] == "uint16"
    assert manifest["image_shape"] == [250, 250]
    assert manifest["channel_names"] == names
    assert manifest["fov_size"] == [100, 100]
    assert len(manifest["positions"]) == manifest["n_fovs_y"] * manifest["n_fovs_x"]
    assert all(len(p) == 4 for p in manifest["positions"])


def test_a_two_dimensional_slide_is_treated_as_one_channel(tmp_path):
    slide = tmp_path / "flat.ome.tif"
    tifffile.imwrite(
        str(slide),
        _rng_stack(1, 250, 250)[0],
        photometric="minisblack",
        metadata={"axes": "YX"},
        ome=True,
    )

    manifest = tile_for_basic.tile_for_basic(
        str(slide),
        str(tmp_path / "flat_tiles.ome.tif"),
        str(tmp_path / "flat_tiles.json"),
        channel_names=["PANCK"],
        fov_size=(100, 100),
    )

    assert manifest["channel_names"] == ["PANCK"]
    assert manifest["corrected_channels"] == [0]
