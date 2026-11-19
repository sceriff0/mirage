"""bin/create_channels_manifest.py — the map REGISTER and RegisteredMatch pair on.

Its JSON is filename -> [channel names], which RegisteredMatch.pair uses to reunite
VALIS's registered outputs with their slide metas by channel signature. A manifest that
skips a file or misorders channels re-pairs one slide's meta with another slide's
pixels, with no visible error.
"""

from __future__ import annotations

import json
from pathlib import Path

import create_channels_manifest
import numpy as np
import pytest
from utils import ome_io


def _registered(directory: Path, stem: str, channels: list[str]) -> Path:
    """Write a file named exactly the way REGISTER's output is named."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_registered.ome.tiff"
    ome_io.write_ome_tiff(
        path,
        np.zeros((len(channels), 8, 8), dtype=np.uint16),
        channels=channels,
        pixel_size_um=0.325,
    )
    return path


def _run(monkeypatch, indir: Path, out: Path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_channels_manifest.py",
            "--input-dir",
            str(indir),
            "--output",
            str(out),
        ],
    )
    create_channels_manifest.main()


def test_the_manifest_maps_each_file_to_its_channel_names_in_order(
    tmp_path, monkeypatch
):
    indir = tmp_path / "in"
    _registered(indir, "P001_ref", ["DAPI", "PANCK", "SMA"])
    _registered(indir, "P001_mov1", ["DAPI", "CD3", "CD8"])
    out = tmp_path / "channels.json"

    _run(monkeypatch, indir, out)

    manifest = json.loads(out.read_text())
    assert manifest == {
        "P001_ref_registered.ome.tiff": ["DAPI", "PANCK", "SMA"],
        "P001_mov1_registered.ome.tiff": ["DAPI", "CD3", "CD8"],
    }


def test_files_that_are_not_registered_outputs_are_skipped(tmp_path, monkeypatch):
    """The directory REGISTER hands over also holds its inputs and its logs. A
    manifest that indexed those would hand RegisteredMatch more keys than there
    are metas, which is its 'count mismatch' failure -- at a later step, with a
    less useful message."""
    indir = tmp_path / "in"
    _registered(indir, "P001_ref", ["DAPI", "PANCK"])
    ome_io.write_ome_tiff(
        indir / "P001_ref.ome.tiff",
        np.zeros((2, 8, 8), dtype=np.uint16),
        channels=["DAPI", "PANCK"],
        pixel_size_um=0.325,
    )
    (indir / "notes.txt").write_text("not an image")
    out = tmp_path / "channels.json"

    _run(monkeypatch, indir, out)

    assert list(json.loads(out.read_text())) == ["P001_ref_registered.ome.tiff"]


def test_the_manifest_is_written_in_sorted_filename_order(tmp_path, monkeypatch):
    """`sorted(os.listdir(...))`, not arrival order: the manifest is read back by
    a process whose caching hashes its bytes, so a directory-order manifest would
    make an identical rerun miss."""
    indir = tmp_path / "in"
    for stem in ("P001_zed", "P001_alpha", "P001_mid"):
        _registered(indir, stem, ["DAPI", "CD3"])
    out = tmp_path / "channels.json"

    _run(monkeypatch, indir, out)

    assert list(json.loads(out.read_text())) == [
        "P001_alpha_registered.ome.tiff",
        "P001_mid_registered.ome.tiff",
        "P001_zed_registered.ome.tiff",
    ]


def test_a_registered_file_with_no_ome_metadata_stops_the_run(tmp_path, monkeypatch):
    """The dangerous alternative is an empty channel list in the manifest, which
    reads downstream as 'this slide has no channels' rather than as an error.

    ``ome=False`` matters here: tifffile infers OME
    output from the ``.ome.tiff`` suffix alone and synthesizes a
    ``Channel:0:0`` OME-XML block even without an explicit ``ome=True`` --
    observed directly (``extract_channel_names_from_ome`` returned
    ``['Channel:0:0']`` rather than ``[]`` on a first pass of this test that
    omitted the flag). ``ome=False`` is what actually produces a file with no
    OME metadata for the filename this script requires."""
    import tifffile

    indir = tmp_path / "in"
    indir.mkdir()
    tifffile.imwrite(
        indir / "P001_ref_registered.ome.tiff",
        np.zeros((8, 8), dtype=np.uint16),
        photometric="minisblack",
        ome=False,
    )
    out = tmp_path / "channels.json"

    with pytest.raises(RuntimeError, match="No OME metadata"):
        _run(monkeypatch, indir, out)
