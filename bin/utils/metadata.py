"""OME-TIFF channel-name and nuclear-marker helpers."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import tifffile

__all__ = [
    "extract_channel_names_from_ome",
    "is_nuclear",
    "pick_nuclear_index",
    "DEFAULT_NUCLEAR_MARKERS",
]

# Ordered preference for the nuclear/fiducial channel (drives segmentation and the
# registration fiducial); the first marker matching any channel wins. Mirrors
# params.nuclear_markers. Pipeline callers always pass the list; this fallback is for
# running a script by hand.
DEFAULT_NUCLEAR_MARKERS = ("DAPI", "CELLTOX")


def _resolve_markers(nuclear_markers):
    """The configured markers, trimmed and upper-cased, or the default mirror.

    Each element is also split on commas and whitespace, like ``MarkerUtils.markerList``:
    an unsplit ``"DAPI,CELLTOX"`` would silently match no channel.
    """
    markers = [
        part.upper()
        for m in (nuclear_markers or ())
        for part in re.split(r"[,\s]+", str(m).strip())
        if part
    ]
    return markers or [str(m).upper() for m in DEFAULT_NUCLEAR_MARKERS]


def is_nuclear(channel_name, nuclear_markers=None):
    """True when ``channel_name`` names one of the configured nuclear markers.

    Case-insensitive substring match (``DAPI_nuclear`` counts). Must agree with
    ``MarkerUtils.isNuclear``, which pre-computes how many channels SPLIT_CHANNELS emits;
    a mismatch silently miscounts a ``groupTuple`` size.

    Parameters
    ----------
    channel_name : str
        A single channel name, from metadata (never a filename).
    nuclear_markers : sequence of str, optional
        Marker names. Defaults to ``DEFAULT_NUCLEAR_MARKERS``.

    Returns
    -------
    bool
    """
    name = str(channel_name or "").strip().upper()
    if not name:
        return False
    return any(marker in name for marker in _resolve_markers(nuclear_markers))


def pick_nuclear_index(channel_names, nuclear_markers=None):
    """Index of the nuclear/fiducial channel, resolved by marker name.

    Channel identity comes from metadata (OME ``Channel:Name`` or the declared
    channel names), never from the filename. Markers are tried in order; the first
    marker that matches any channel (case-insensitive substring) wins, and that
    channel's index is returned.

    Parameters
    ----------
    channel_names : list of str
        Channel names in image/channel order.
    nuclear_markers : sequence of str, optional
        Ordered preference list of nuclear marker names. Defaults to
        ``DEFAULT_NUCLEAR_MARKERS`` (``('DAPI', 'CELLTOX')``).

    Returns
    -------
    int or None
        Index of the first matching channel, or ``None`` if no marker matches.
    """
    names = [str(n).upper() for n in (channel_names or [])]
    for marker in _resolve_markers(nuclear_markers):
        for i, name in enumerate(names):
            if marker in name:
                return i
    return None


def extract_channel_names_from_ome(filepath: str | Path) -> List[str]:
    """Extract channel names from OME-TIFF metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to OME-TIFF file.

    Returns
    -------
    List[str]
        List of channel names from OME-XML metadata.
        Returns empty list if:
        - File is not OME-TIFF
        - OME metadata is missing/malformed
        - No channel information found

    Notes
    -----
    Falls back to a namespace-agnostic search for files without a proper OME namespace.
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            if not hasattr(tif, "ome_metadata") or not tif.ome_metadata:
                return []

            root = ET.fromstring(tif.ome_metadata)

            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

            channels = root.findall(".//ome:Channel", ns)

            if not channels:
                # Some files lack a proper OME namespace.
                channels = root.findall(".//{*}Channel")

            names = []
            for ch in channels:
                name = ch.get("Name") or ch.get("ID", f"Channel_{len(names)}")
                names.append(name)

            return names

    except Exception as e:
        logging.getLogger(__name__).debug(f"OME metadata parsing failed: {e}")
        return []
