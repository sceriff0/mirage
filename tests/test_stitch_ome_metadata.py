"""The stitched slide's OME header must carry what a `.ome.tiff` promises.

The source review recorded this as "`tiled_stitch.py:123` writes `*_registered.ome.tiff`
containing **no OME-XML**". Measured against the working tree, that is **not what happens** --
tifffile infers OME mode from the `.ome.tiff` suffix, so a header is written whether or not the
code asks for one:

    is_ome        : True
    ome_metadata  : <?xml version="1.0" ...><OME ...>

What is true is that the header was **content-free**:

    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX=... SizeY=... SizeC="2">
    <Channel ID="Channel:0:0" SamplesPerPixel="1">      <- no Name
    <Channel ID="Channel:0:1" SamplesPerPixel="1">      <- no Name

No channel names, and no `PhysicalSizeX`/`PhysicalSizeY` -- the pixel size reached the file only
as TIFF-level `resolution` tags. A consumer that reads OME metadata, which is the natural thing
to do with a file called `.ome.tiff`, got neither.

That also retires the reason the code gave for not writing one: "an OME header here would become
a second, channel-less source of channel names for SPLIT_CHANNELS to find". tifffile was writing
that header anyway, so the situation the comment argued against was the situation in effect. And
SPLIT_CHANNELS could not have been misled by it in any case -- `split_channels.nf:31` passes
`--channels ${meta.channels.join(' ')}` unconditionally, and `split_multichannel.py:92` only
falls back to `extract_channel_names_from_ome` when no names were passed.

So the fix is to fill the header in, not to suppress it.
"""

import json
import os
import re
import sys

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

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

import tiled_stitch  # noqa: E402

PIXEL_SIZE = 0.325


def _inputs(tmp_path, channels=("DAPI", "CD3", "CD8"), size=256):
    data = np.stack(
        [
            np.full((size, size), 100 * (i + 1), dtype=np.uint16)
            for i in range(len(channels))
        ]
    )
    mov = tmp_path / "mov.ome.tiff"
    tifffile.imwrite(str(mov), data, photometric="minisblack")

    manifest = {
        "reference": "ref",
        "slides": {
            "ref": {"M0": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "mesh": None},
            "mov": {
                "M0": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                "mesh": None,
                "out_shape": [size, size],
            },
        },
    }
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(manifest))
    return mov, man


def _stitch(tmp_path, channels=("DAPI", "CD3", "CD8")):
    mov, man = _inputs(tmp_path, channels)
    out = tmp_path / "P001_registered.ome.tiff"
    argv = [
        "--moving",
        str(mov),
        "--manifest",
        str(man),
        "--moving-name",
        "mov",
        "--out",
        str(out),
        "--pixel-size",
        str(PIXEL_SIZE),
        "--channel-names",
        *channels,
    ]
    assert tiled_stitch.main(argv) == 0
    return out


def test_the_ome_header_names_the_channels(tmp_path):
    """A `.ome.tiff` whose channels are anonymous makes the reader guess what CD3 was."""
    out = _stitch(tmp_path)

    with tifffile.TiffFile(str(out)) as tif:
        md = tif.ome_metadata

    names = re.findall(r'<Channel[^>]*Name="([^"]+)"', md)

    assert names == ["DAPI", "CD3", "CD8"]


def test_the_ome_header_records_the_physical_pixel_size(tmp_path):
    """The scale reached the file only as TIFF resolution tags; an OME reader saw nothing."""
    out = _stitch(tmp_path)

    with tifffile.TiffFile(str(out)) as tif:
        md = tif.ome_metadata

    sizes = re.findall(r'PhysicalSize([XY])="([^"]+)"', md)

    assert {axis for axis, _ in sizes} == {"X", "Y"}
    for _axis, value in sizes:
        assert float(value) == pytest.approx(PIXEL_SIZE)


def test_the_tiff_resolution_tags_are_not_lost(tmp_path):
    """The existing scale channel must survive -- consumers downstream of STARE read these."""
    out = _stitch(tmp_path)

    with tifffile.TiffFile(str(out)) as tif:
        tags = tif.pages[0].tags
        xres = tags["XResolution"].value

    # CENTIMETER means pixels-per-cm: 1e4 um/cm over um/px. The tolerance is loose because TIFF
    # stores resolution as a RATIONAL (numerator/denominator, both uint32), so 30769.23... is
    # only approximable -- measured 30769.327, a relative error of 3.1e-06. That is inherent to
    # the tag, not to this change; a genuinely wrong pixel size would be out by percent or more.
    assert xres[0] / xres[1] == pytest.approx(1e4 / PIXEL_SIZE, rel=1e-4)


def test_the_pixels_survive_unchanged(tmp_path):
    """Metadata only -- this must not touch a single pixel."""
    out = _stitch(tmp_path)

    written = tifffile.imread(str(out))

    assert written.shape == (3, 256, 256)
    assert [int(written[c].flat[0]) for c in range(3)] == [100, 200, 300]
