"""OME-TIFF and channel metadata utilities.

This module provides standardized functions for extracting and creating
metadata for microscopy images, particularly OME-TIFF format.

Examples
--------
>>> from lib.metadata import get_channel_names, create_ome_xml
>>> channels = get_channel_names("sample_DAPI_SMA_panCK.tif")
>>> ['DAPI', 'SMA', 'panCK']
>>> xml = create_ome_xml(channels, np.uint16, 2048, 2048)

Notes
-----
This module consolidates metadata extraction logic that was previously
duplicated across multiple scripts (register.py, register_cpu.py,
register_gpu.py, merge_registered.py, etc.).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import tifffile

__all__ = [
    "get_channel_names",
    "extract_channel_names_from_ome",
    "extract_channel_names_from_filename",
    "extract_markers_from_filename",
    "create_ome_xml",
    "get_ome_metadata",
]


def get_channel_names(filename: str | Path) -> List[str]:
    """Extract channel names from filename using standard naming convention.

    Parameters
    ----------
    filename : str or Path
        Image filename following the pattern: `SampleID_Channel1_Channel2_...`
        Examples:
        - 'B19-10215_DAPI_SMA_panCK.tif'
        - 'Sample01_DAPI_CD3_CD8_corrected.ome.tiff'

    Returns
    -------
    List[str]
        List of channel names extracted from filename.

    Notes
    -----
    This function implements the standard naming convention used throughout
    the ATEIA pipeline:
    - First part (before first underscore) is the sample ID
    - Remaining parts are channel marker names
    - Suffixes like '_corrected', '_padded', '_registered', '_preprocessed'
      are automatically removed

    The function does NOT extract from OME metadata. Use
    `extract_channel_names_from_ome()` for that purpose.

    Examples
    --------
    >>> get_channel_names("B19-10215_DAPI_SMA_panCK.tif")
    ['DAPI', 'SMA', 'panCK']

    >>> get_channel_names("Sample01_DAPI_CD3_CD8_corrected.ome.tiff")
    ['DAPI', 'CD3', 'CD8']

    >>> get_channel_names("/path/to/data/S001_DAPI_registered.tif")
    ['DAPI']

    See Also
    --------
    extract_channel_names_from_ome : Extract from OME-XML metadata
    extract_channel_names_from_filename : Lower-level filename parsing
    """
    # Get basename without directory path
    base = os.path.basename(str(filename))

    # Remove known suffixes that don't contain channel information
    name_part = (base.replace('_corrected', '')
                    .replace('_padded', '')
                    .replace('_preprocessed', '')
                    .replace('_registered', '')
                    .split('.')[0])  # Remove file extension

    # Split by underscore
    parts = name_part.split('_')

    # First part is sample ID (typically has format like "B19-10215" or "Sample01")
    # Remaining parts are channel names
    if len(parts) > 1:
        channels = parts[1:]
    else:
        # Fallback: if no underscore, return the whole name
        channels = [name_part]

    return channels


def extract_channel_names_from_filename(
    filename: str | Path,
    expected_channels: Optional[int] = None
) -> List[str]:
    """Extract channel names from filename with validation.

    Parameters
    ----------
    filename : str or Path
        Path to image file.
    expected_channels : int or None, optional
        Expected number of channels for validation.
        If provided and mismatch occurs, generates numbered fallback names.

    Returns
    -------
    List[str]
        List of channel names, with fallback to generic names if needed.

    Notes
    -----
    This is a more robust version of `get_channel_names()` that handles
    edge cases and provides fallback behavior.

    Examples
    --------
    >>> extract_channel_names_from_filename("sample_DAPI_SMA.tif", expected_channels=2)
    ['DAPI', 'SMA']

    >>> extract_channel_names_from_filename("sample_DAPI.tif", expected_channels=3)
    ['DAPI', 'Channel_1', 'Channel_2']

    See Also
    --------
    get_channel_names : Standard channel name extraction
    """
    markers = get_channel_names(filename)

    if expected_channels is None:
        return markers

    # Validate and handle mismatches
    if len(markers) == expected_channels:
        return markers
    elif len(markers) < expected_channels:
        # Pad with generic names
        return markers + [f"Channel_{i}" for i in range(len(markers), expected_channels)]
    else:
        # Too many names - truncate
        return markers[:expected_channels]


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
    This function parses the OME-XML metadata block embedded in TIFF files.
    It handles both the standard OME namespace and files without proper namespace
    declarations.

    The OME-XML specification is defined at:
    https://www.openmicroscopy.org/Schemas/OME/2016-06

    Examples
    --------
    >>> extract_channel_names_from_ome("registered.ome.tif")
    ['DAPI', 'SMA', 'panCK', 'CD3']

    >>> extract_channel_names_from_ome("not_ome.tif")
    []

    See Also
    --------
    get_ome_metadata : Extract full OME metadata dictionary
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            if not hasattr(tif, 'ome_metadata') or not tif.ome_metadata:
                return []

            # Parse XML
            root = ET.fromstring(tif.ome_metadata)

            # Define OME namespace
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

            # Try with namespace first
            channels = root.findall('.//ome:Channel', ns)

            if not channels:
                # Fallback: try without namespace (some files may not use it properly)
                channels = root.findall('.//{*}Channel')

            # Extract channel names
            names = []
            for ch in channels:
                # Try 'Name' attribute first, fallback to 'ID'
                name = ch.get('Name') or ch.get('ID', f'Channel_{len(names)}')
                names.append(name)

            return names

    except Exception as e:
        # Silently fail and return empty list
        # Caller can fall back to filename parsing
        return []


def create_ome_xml(
    channel_names: List[str],
    dtype: np.dtype,
    width: int,
    height: int,
    pixel_size_um: float = 0.325,
    size_z: int = 1,
    size_t: int = 1
) -> str:
    """Create OME-XML metadata string for TIFF files.

    Parameters
    ----------
    channel_names : List[str]
        List of channel names in order.
    dtype : np.dtype
        NumPy data type of the image (e.g., np.uint16, np.uint8).
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    pixel_size_um : float, default=0.325
        Physical pixel size in micrometers.
        Default matches typical microscopy resolution.
    size_z : int, default=1
        Number of Z-slices (depth).
    size_t : int, default=1
        Number of time points.

    Returns
    -------
    str
        OME-XML metadata string formatted for embedding in TIFF.

    Notes
    -----
    The generated XML follows the OME-XML 2016-06 schema specification.
    This is the standard format for storing microscopy metadata.

    The dimension order is always 'XYCZT' (width, height, channels, z, time),
    which is the recommended order for most applications.

    Physical size units are in micrometers (µm).

    Examples
    --------
    >>> channels = ['DAPI', 'SMA', 'panCK']
    >>> xml = create_ome_xml(channels, np.uint16, 2048, 2048)
    >>> print(xml[:100])
    <?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schema...

    >>> # For 8-bit image
    >>> xml = create_ome_xml(['Brightfield'], np.uint8, 1024, 1024, pixel_size_um=0.5)

    See Also
    --------
    extract_channel_names_from_ome : Parse OME-XML from existing files
    """
    num_channels = len(channel_names)

    # Map NumPy dtype to OME dtype name
    dtype_name = dtype.name

    # Create channel XML elements
    channel_xml = '\n'.join(
        f'            <Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1" />'
        for i, name in enumerate(channel_names)
    )

    # Construct complete OME-XML
    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Image">
        <Pixels ID="Pixels:0" Type="{dtype_name}"
                SizeX="{width}" SizeY="{height}" SizeZ="{size_z}" SizeC="{num_channels}" SizeT="{size_t}"
                DimensionOrder="XYCZT"
                PhysicalSizeX="{pixel_size_um}" PhysicalSizeY="{pixel_size_um}" PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm">
{channel_xml}
            <TiffData />
        </Pixels>
    </Image>
</OME>'''

    return ome_xml


def extract_markers_from_filename(filename: str | Path) -> List[str]:
    """Extract marker names from filename with more flexible parsing.

    This is an alternative to get_channel_names() that tries to be more
    flexible about detecting sample IDs vs marker names.

    Parameters
    ----------
    filename : str or Path
        Filename or path to parse.

    Returns
    -------
    List[str]
        List of extracted marker names.

    Notes
    -----
    This function removes common suffixes (_registered, _corrected, _padded)
    and attempts to filter out sample IDs based on patterns like:
    - Contains both letters and numbers with hyphens (e.g., "B19-10215")
    - Is recognized as a common sample ID pattern

    The function is more lenient than get_channel_names() which assumes
    the first underscore-delimited part is always the sample ID.

    Examples
    --------
    >>> extract_markers_from_filename("B19-10215_DAPI_SMA_panck_corrected.ome.tif")
    ['DAPI', 'SMA', 'panck']

    >>> extract_markers_from_filename("Sample_DAPI_CD3.tif")
    ['DAPI', 'CD3']

    See Also
    --------
    get_channel_names : Standard channel name extraction
    extract_channel_names_from_filename : With validation
    """
    # Get basename
    base = os.path.basename(str(filename))

    # Remove suffixes
    name = base.replace('_registered.ome.tif', '').replace('_registered.ome.tiff', '')
    name = name.replace('_corrected', '').replace('_padded', '').replace('_preprocessed', '')

    # Remove file extension
    if '.' in name:
        name = name.split('.')[0]

    # Split by underscore
    parts = name.split('_')

    # Try to detect and filter out sample ID
    # Pattern: contains letters, numbers, and hyphen (e.g., "B19-10215")
    markers = []
    for p in parts:
        if not p:
            continue
        # Skip if it looks like a sample ID (has hyphen and alphanumeric)
        if '-' in p and any(c.isalpha() for c in p) and any(c.isdigit() for c in p):
            continue
        # Skip if it's just "Sample" or similar generic prefix
        if p.lower() in ['sample', 'slide', 'image']:
            continue
        markers.append(p)

    # If no markers found, return the whole name
    if not markers:
        markers = [name]

    return markers


def get_ome_metadata(filepath: str | Path) -> Dict[str, Any]:
    """Extract all relevant metadata from OME-TIFF file.

    Parameters
    ----------
    filepath : str or Path
        Path to OME-TIFF file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'channel_names': List[str] - Channel names
        - 'num_channels': int - Number of channels
        - 'width': int - Image width
        - 'height': int - Image height
        - 'dtype': str - Data type name
        - 'pixel_size_um': float or None - Physical pixel size
        - 'ome_xml': str or None - Full OME-XML string

    Returns empty dict if file cannot be read or is not OME-TIFF.

    Examples
    --------
    >>> meta = get_ome_metadata("registered.ome.tif")
    >>> meta['channel_names']
    ['DAPI', 'SMA', 'panCK']
    >>> meta['width']
    2048
    >>> meta['pixel_size_um']
    0.325

    See Also
    --------
    extract_channel_names_from_ome : Extract only channel names
    create_ome_xml : Create OME-XML metadata
    """
    metadata = {}

    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            # Get basic shape information
            if tif.pages:
                page = tif.pages[0]
                metadata['width'] = page.imagewidth
                metadata['height'] = page.imagelength
                metadata['dtype'] = str(page.dtype)

            # Get channel information
            channel_names = extract_channel_names_from_ome(filepath)
            metadata['channel_names'] = channel_names
            metadata['num_channels'] = len(channel_names)

            # Get OME-XML if available
            if hasattr(tif, 'ome_metadata'):
                metadata['ome_xml'] = tif.ome_metadata

                # Try to extract physical pixel size
                try:
                    root = ET.fromstring(tif.ome_metadata)
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//ome:Pixels', ns)
                    if pixels is not None:
                        px_size = pixels.get('PhysicalSizeX')
                        if px_size:
                            metadata['pixel_size_um'] = float(px_size)
                except:
                    metadata['pixel_size_um'] = None
            else:
                metadata['ome_xml'] = None
                metadata['pixel_size_um'] = None

    except Exception:
        return {}

    return metadata
