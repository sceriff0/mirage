"""Quality control image generation for registration.

This module provides functions for creating QC visualizations to assess
registration quality, particularly RGB composite overlays of registered
and reference images.

Examples
--------
>>> from qc import create_registration_qc
>>> create_registration_qc(
...     reference_path="ref.tif",
...     registered_path="registered.tif",
...     output_path="qc.tif"
... )

Notes
-----
``create_registration_qc`` decodes only the resolved nuclear/fiducial channel of
each slide, at full resolution: the published ``*_fullres.tif`` needs every pixel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from logger import get_logger
from metadata import extract_channel_names_from_ome, pick_nuclear_index
from numpy.typing import NDArray
from ome_io import (
    BIGTIFF_THRESHOLD_BYTES,
    ome_metadata,
    read_info,
    read_plane,
    write_tiff,
)
from registration_utils import autoscale
from skimage.transform import rescale
from tiled_io import decimation_factor, open_lazy, read_decimated

__all__ = [
    "create_registration_qc",
    "create_nuclear_overlay",
    "compose_on_reference_canvas",
    "render_before_after",
    "cyx_to_bgr",
    "autoscale_for_display",
]

logger = get_logger(__name__)


def autoscale_for_display(img: NDArray, method: str = "minmax") -> NDArray[np.uint8]:
    """Autoscale image for display purposes.

    Parameters
    ----------
    img : NDArray
        Input image of any numeric dtype.
    method : str, default="minmax"
        Scaling method:
        - 'minmax': Min-max normalization (linear)
        - 'percentile': Percentile-based (robust to outliers)

    Returns
    -------
    NDArray[np.uint8]
        Scaled image in 0-255 range.

    Examples
    --------
    >>> img = np.random.rand(100, 100) * 1000
    >>> scaled = autoscale_for_display(img, method='minmax')
    >>> scaled.min(), scaled.max()
    (0, 255)

    See Also
    --------
    autoscale : Percentile-based scaling from registration_utils
    """
    if method == "minmax":
        img_min = img.min()
        img_max = img.max()
        range_val = max(img_max - img_min, 1e-6)
        normalized = (img - img_min) / range_val
        # Round before the uint8 cast to avoid truncation artifacts
        return np.round(normalized * 255).astype(np.uint8)
    elif method == "percentile":
        return autoscale(img, low_p=1.0, high_p=99.0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'percentile'.")


def compose_on_reference_canvas(ref: NDArray, mov: NDArray) -> NDArray:
    """Place ``mov`` on ``ref``'s canvas at the origin by pad-or-crop.

    Parameters
    ----------
    ref : NDArray
        Reference plane (2-D). Only its ``shape`` is used.
    mov : NDArray
        Moving plane (2-D), of any shape.

    Returns
    -------
    NDArray
        An array of exactly ``ref.shape`` and ``mov.dtype`` holding ``mov``
        anchored at ``(0, 0)``: zero-filled where ``mov`` is smaller, truncated
        where it is larger.

    Notes
    -----
    Pad-or-crop, never rescale: rescaling a differently-sized native image would
    absorb the scale part of the misalignment and make the "before" panel
    understate the pre-registration error.

    Origin-aligned because both registration backends put slides in the
    reference's frame from the same corner.
    """
    if ref.ndim != 2 or mov.ndim != 2:
        raise ValueError(
            f"compose_on_reference_canvas expects 2-D planes, got "
            f"ref.ndim={ref.ndim}, mov.ndim={mov.ndim}"
        )
    h, w = ref.shape
    out = np.zeros((h, w), dtype=mov.dtype)
    hh = min(h, mov.shape[0])
    ww = min(w, mov.shape[1])
    out[:hh, :ww] = mov[:hh, :ww]
    return out


def create_nuclear_overlay(
    reference_nuc: NDArray, registered_nuc: NDArray, scale_factor: float = 0.25
) -> Tuple[NDArray, NDArray]:
    """Create RGB overlay of the reference and registered nuclear/fiducial channels.

    Parameters
    ----------
    reference_nuc : NDArray
        Reference nuclear/fiducial channel (2D array).
    registered_nuc : NDArray
        Registered nuclear/fiducial channel (2D array).
    scale_factor : float, default=0.25
        Downsampling factor for output images (0.25 = 4x smaller).

    Returns
    -------
    rgb_bgr : NDArray
        RGB composite in BGR order (for OpenCV/PNG), shape (H, W, 3).
        Red = registered, Green = reference, Blue = 0.
    rgb_cyx : NDArray
        RGB composite in CYX order (for TIFF/ImageJ), shape (3, H, W).
        Same channel assignment as rgb_bgr.

    Notes
    -----
    Each channel is independently autoscaled using min-max normalization
    before creating the composite. This ensures both channels are visible
    even if they have different intensity ranges.

    The downsampling uses anti-aliasing to prevent artifacts.

    Examples
    --------
    >>> ref = np.random.randint(0, 4096, (2048, 2048), dtype=np.uint16)
    >>> reg = np.random.randint(0, 4096, (2048, 2048), dtype=np.uint16)
    >>> rgb_bgr, rgb_cyx = create_nuclear_overlay(ref, reg, scale_factor=0.25)
    >>> rgb_bgr.shape
    (512, 512, 3)
    >>> rgb_cyx.shape
    (3, 512, 512)

    See Also
    --------
    create_registration_qc : High-level QC generation function
    autoscale_for_display : Scaling function used internally
    """
    # Validate matching shapes
    if reference_nuc.shape != registered_nuc.shape:
        raise ValueError(
            f"Shape mismatch: reference nuclear channel {reference_nuc.shape} != "
            f"registered nuclear channel {registered_nuc.shape}. "
            f"Images must have the same dimensions after registration."
        )

    # Autoscale each channel independently
    ref_scaled = autoscale_for_display(reference_nuc, method="minmax")
    reg_scaled = autoscale_for_display(registered_nuc, method="minmax")

    # Downsample if requested
    if scale_factor != 1.0:
        # Round before the uint8 cast to avoid truncation artifacts
        ref_down = np.round(
            rescale(
                ref_scaled, scale=scale_factor, anti_aliasing=True, preserve_range=True
            )
        ).astype(np.uint8)
        reg_down = np.round(
            rescale(
                reg_scaled, scale=scale_factor, anti_aliasing=True, preserve_range=True
            )
        ).astype(np.uint8)
    else:
        ref_down = ref_scaled
        reg_down = reg_scaled

    h, w = reg_down.shape

    # Create RGB composite in BGR order (for OpenCV)
    rgb_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_bgr[:, :, 2] = reg_down  # Red = registered (BGR so index 2 is Red)
    rgb_bgr[:, :, 1] = ref_down  # Green = reference
    rgb_bgr[:, :, 0] = 0  # Blue = 0

    # Create CYX version for TIFF
    rgb_cyx = np.stack(
        [
            reg_down,  # Red channel (registered)
            ref_down,  # Green channel (reference)
            np.zeros_like(ref_down, dtype=np.uint8),  # Blue channel
        ],
        axis=0,
    )

    return rgb_bgr, rgb_cyx


def cyx_to_bgr(cyx: NDArray) -> NDArray:
    """Reorder a ``(3, H, W)`` RGB composite into OpenCV's ``(H, W, 3)`` BGR.

    Parameters
    ----------
    cyx : NDArray
        ``(3, H, W)`` uint8, channel order red, green, blue.

    Returns
    -------
    NDArray
        ``(H, W, 3)`` uint8, channel order blue, green, red — what
        ``cv2.imwrite`` expects.
    """
    return np.ascontiguousarray(np.stack([cyx[2], cyx[1], cyx[0]], axis=-1))


def _hstack_panels(panels, gap: int) -> NDArray:
    """Concatenate ``(3, H, W)`` panels left-to-right with a blue separator.

    Blue rather than black: black looks like a dark region of a panel, and blue is
    the one channel the overlay never uses.
    """
    if len(panels) == 1:
        return panels[0]
    h = panels[0].shape[1]
    sep = np.zeros((3, h, gap), dtype=np.uint8)
    sep[2] = 255
    out = []
    for i, panel in enumerate(panels):
        if i:
            out.append(sep)
        out.append(panel)
    return np.concatenate(out, axis=2)


def render_before_after(
    reference: NDArray,
    native: NDArray | None,
    registered: NDArray,
    *,
    scale_factor: float = 1.0,
    gap: int = 8,
) -> NDArray:
    """Render the two-panel registration QC composite.

    Parameters
    ----------
    reference : NDArray
        Reference nuclear/fiducial plane (2-D). Defines the canvas.
    native : NDArray or None
        The moving slide's nuclear/fiducial plane BEFORE registration, i.e.
        exactly what entered the registration step. ``None`` renders the
        "after" panel alone.
    registered : NDArray
        The moving slide's nuclear/fiducial plane AFTER registration (2-D).
    scale_factor : float, default=1.0
        Downsampling factor applied to each panel. The separator width is NOT
        rescaled — it is a fixed number of output pixels.
    gap : int, default=8
        Width in output pixels of the blue separator between panels.

    Returns
    -------
    NDArray
        ``(3, H', 2*W' + gap)`` uint8 CYX, or ``(3, H', W')`` when ``native`` is
        None. Left panel = *Before* (reference + native), right = *After*
        (reference + registered). Red = moving slide, green = reference, blue =
        0 except the separator.

    Notes
    -----
    Both panels are built on the REFERENCE canvas via
    ``compose_on_reference_canvas``, so they are the same size and the same
    pixel coordinate means the same tissue location in both. That is what makes
    the pair comparable at a glance: fringing that shrinks from left to right is
    exactly the correction registration applied.
    """
    # The registered plane should already be on the reference canvas; if not, it is
    # pad-or-cropped to draw, with a warning, because that is a registration defect.
    # The native plane differing in shape is normal.
    if registered.shape != reference.shape:
        logger.warning(
            "Registered plane %s does not match the reference canvas %s; the 'after' "
            "panel was pad-or-cropped at the origin to fit. A registered slide is "
            "expected to already be on the reference canvas, so this is a registration "
            "output to check, not a QC setting.",
            registered.shape,
            reference.shape,
        )

    panels = []
    if native is not None:
        panels.append(
            create_nuclear_overlay(
                reference,
                compose_on_reference_canvas(reference, native),
                scale_factor=scale_factor,
            )[1]
        )
    panels.append(
        create_nuclear_overlay(
            reference,
            compose_on_reference_canvas(reference, registered),
            scale_factor=scale_factor,
        )[1]
    )
    return _hstack_panels(panels, gap)


def create_registration_qc(
    reference_path: str | Path,
    registered_path: str | Path,
    output_path: str | Path,
    scale_factor: float = 0.25,
    save_fullres: bool = True,
    save_png: bool = True,
    save_tiff: bool = True,
    nuclear_markers=None,
    native_path: str | Path | None = None,
    pixel_size_um: float | None = None,
) -> None:
    """Create QC visualizations for registration assessment.

    This function generates RGB composite images showing the overlay of the
    reference and registered nuclear/fiducial channels. Perfect registration
    appears as yellow (red + green), while misalignment shows red/green fringing.

    Parameters
    ----------
    reference_path : str or Path
        Path to reference image (OME-TIFF).
    registered_path : str or Path
        Path to registered image (OME-TIFF).
    output_path : str or Path
        Base path for output files (extension will be modified).
    scale_factor : float, default=0.25
        Downsampling factor for preview images (0.25 = 4x smaller).
        Full resolution is always saved if save_fullres=True.
    save_fullres : bool, default=True
        Save full-resolution TIFF (with compression).
    save_png : bool, default=True
        Save downsampled PNG preview.
    save_tiff : bool, default=True
        Save downsampled TIFF for ImageJ.
    nuclear_markers : sequence of str, optional
        Ordered preference list of nuclear/fiducial marker names, i.e.
        ``params.nuclear_markers``. Resolved against each image's OME channel
        names by ``metadata.pick_nuclear_index``, the same rule
        ``lib/MarkerUtils.groovy`` and ``split_multichannel.py`` use. Defaults to
        ``metadata.DEFAULT_NUCLEAR_MARKERS`` so the script stays usable by hand.
    native_path : str or Path, optional
        The moving slide's image BEFORE registration -- exactly what entered the
        registration step. When given, the output is a TWO-panel figure: left
        *Before* (reference + native), right *After* (reference + registered),
        both on the reference canvas. When omitted, only the "after" panel is
        rendered.
    pixel_size_um : float, optional
        The reference/registered images' pixel size in micrometers, i.e.
        ``pixel_size.read_ome_pixel_size``'s answer for this patient. When given,
        it is stamped as the ``XResolution``/``ResolutionUnit`` TIFF tags on BOTH
        ``*_QC_RGB.tif`` and ``*_QC_RGB_fullres.tif`` -- ``resolution`` is
        pixels-per-unit, so the tag is ``1 / pixel_size_um`` at full resolution
        and scaled by ``scale_factor`` for the downsampled preview, which covers
        the same tissue in fewer, larger pixels. When omitted, no resolution tag
        is written.

    Returns
    -------
    None
        Saves files to disk.

    Notes
    -----
    Output files created:
    - {output_path}_fullres.tif: Full resolution, compressed (if save_fullres=True)
    - {output_path}.png: Downsampled PNG (if save_png=True)
    - {output_path}.tif: Downsampled ImageJ TIFF (if save_tiff=True)

    Channel assignment in composites:
    - Red: Registered image
    - Green: Reference image
    - Blue: 0 (black)

    Perfect alignment appears yellow (red + green).
    Misalignment shows red/green fringing.

    Examples
    --------
    >>> create_registration_qc(
    ...     reference_path="reference_DAPI_SMA.tif",
    ...     registered_path="registered_DAPI_CD3.tif",
    ...     output_path="qc_output.tif"
    ... )
    # Creates: qc_output_fullres.tif, qc_output.png, qc_output.tif

    Custom scaling and selective output:
    >>> create_registration_qc(
    ...     reference_path="ref.tif",
    ...     registered_path="reg.tif",
    ...     output_path="qc.tif",
    ...     scale_factor=0.5,
    ...     save_fullres=False,
    ...     save_png=True,
    ...     save_tiff=False
    ... )

    See Also
    --------
    create_nuclear_overlay : Lower-level overlay creation
    """
    reference_path = Path(reference_path)
    registered_path = Path(registered_path)
    output_path = Path(output_path)

    logger.info(f"Creating QC composite: {output_path.name}")

    # Validate input files exist
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    if not registered_path.exists():
        raise FileNotFoundError(f"Registered image not found: {registered_path}")

    native_path = Path(native_path) if native_path is not None else None
    if native_path is not None and not native_path.exists():
        raise FileNotFoundError(f"Native image not found: {native_path}")

    # Resolve the nuclear channel from OME metadata before reading pixels, so bad
    # metadata fails fast. Use pick_nuclear_index, never a literal "DAPI": the marker
    # list is configurable (e.g. CELLTOX-only panels).
    ref_channels = extract_channel_names_from_ome(reference_path)
    reg_channels = extract_channel_names_from_ome(registered_path)

    # No match raises rather than falling back to channel 0: a fallback could build the
    # overlay from a marker channel and make a misregistration look aligned.
    ref_nuc_idx = pick_nuclear_index(ref_channels, nuclear_markers)
    if ref_nuc_idx is None:
        raise ValueError(
            f"No nuclear/fiducial channel found in reference image "
            f"{reference_path.name} (channels: {ref_channels}, markers: "
            f"{list(nuclear_markers) if nuclear_markers else 'default'})."
        )

    reg_nuc_idx = pick_nuclear_index(reg_channels, nuclear_markers)
    if reg_nuc_idx is None:
        raise ValueError(
            f"No nuclear/fiducial channel found in registered image "
            f"{registered_path.name} (channels: {reg_channels}, markers: "
            f"{list(nuclear_markers) if nuclear_markers else 'default'})."
        )

    if ref_channels:
        logger.debug(
            f"Reference nuclear channel: {ref_nuc_idx} ({ref_channels[ref_nuc_idx]})"
        )
    if reg_channels:
        logger.debug(
            f"Registered nuclear channel: {reg_nuc_idx} ({reg_channels[reg_nuc_idx]})"
        )

    # Read only the resolved channel through a lazy zarr view. factor stays 1: the
    # *_fullres.tif output needs full-resolution pixels.
    #
    # Keep the read float32. A uint16 read would go through autoscale_for_display in
    # float64 and round some pixels differently (tests/test_registration_qc_lazy_read.py
    # has a counter-example); widening back to fix that erases the memory saving.
    ref_close = reg_close = None
    try:
        ref_arr, _ref_dtype, ref_close = open_lazy(reference_path)
        reg_arr, _reg_dtype, reg_close = open_lazy(registered_path)
        factor = decimation_factor([ref_arr.shape[1:], reg_arr.shape[1:]], max_dim=None)
        ref_nuc = read_decimated(ref_arr, ref_nuc_idx, factor)
        reg_nuc = read_decimated(reg_arr, reg_nuc_idx, factor)
    finally:
        # open_lazy can raise before returning, so close each handle only if bound.
        if ref_close is not None:
            ref_close()
        if reg_close is not None:
            reg_close()

    # The native (pre-registration) plane, via ome_io's read_info/read_plane, with the
    # channel resolved by pick_nuclear_index like the other two.
    native_nuc = None
    if native_path is not None:
        native_info = read_info(native_path)
        native_channels = list(native_info.channels or [])
        # Raise rather than fall back to channel 0, as above.
        native_nuc_idx = pick_nuclear_index(native_channels, nuclear_markers)
        if native_nuc_idx is None:
            raise ValueError(
                f"No nuclear/fiducial channel found in native image "
                f"{native_path.name} (channels: {native_channels}, markers: "
                f"{list(nuclear_markers) if nuclear_markers else 'default'})."
            )
        # Widen to float32 so this panel is scaled with the same arithmetic as the others.
        native_nuc = np.asarray(
            read_plane(native_path, native_nuc_idx, lazy=True), dtype=np.float32
        )

    # Resolution tags, only when a pixel size is given. tifffile's `resolution` is pixels
    # per unit. The preview's pixels span pixel_size_um / scale_factor, so its tag differs
    # from the full-res one.
    def _resolution_kwargs(px_um):
        if px_um is None:
            return {}
        return {
            "resolution": (1.0 / px_um, 1.0 / px_um),
            "resolutionunit": "MICROMETER",
        }

    fullres_resolution_kwargs = _resolution_kwargs(pixel_size_um)
    preview_pixel_size_um = (
        pixel_size_um / scale_factor if pixel_size_um is not None else None
    )
    preview_resolution_kwargs = _resolution_kwargs(preview_pixel_size_um)

    # ImageJ reads its scale from the description's `unit=` and OME readers from
    # PhysicalSizeX/Y, not from the TIFF tags, so set those too, from the same values.
    fullres_metadata = {"axes": "CYX", "mode": "composite"}
    if pixel_size_um is not None:
        fullres_metadata["unit"] = "um"

    # channels=None: the composite is an RGB display overlay, with no channel names.
    preview_metadata = ome_metadata(None, preview_pixel_size_um, axes="CYX")
    preview_metadata["mode"] = "composite"

    # Save full-resolution QC (compressed). scale_factor=1.0 means no rescale.
    if save_fullres:
        rgb_stack_full = render_before_after(
            ref_nuc, native_nuc, reg_nuc, scale_factor=1.0
        )
        fullres_output_path = output_path.with_name(output_path.stem + "_fullres.tif")
        # ImageJ TIFFs are classic TIFFs, whose 32-bit offsets overflow on a whole-slide
        # composite. Above ome_io's BigTIFF threshold, write an OME BigTIFF instead.
        if rgb_stack_full.nbytes > BIGTIFF_THRESHOLD_BYTES:
            fullres_ome_metadata = ome_metadata(None, pixel_size_um, axes="CYX")
            fullres_ome_metadata["mode"] = "composite"
            write_tiff(
                str(fullres_output_path),
                rgb_stack_full,
                ome=True,
                bigtiff=True,
                metadata=fullres_ome_metadata,
                compression="zlib",
                **fullres_resolution_kwargs,
            )
        else:
            write_tiff(
                str(fullres_output_path),
                rgb_stack_full,
                imagej=True,
                metadata=fullres_metadata,
                compression="zlib",
                **fullres_resolution_kwargs,
            )
        logger.info(f"  Saved full-res QC TIFF: {fullres_output_path}")
        del rgb_stack_full

    # Downsampled preview, same panel(s) as the full-res write above.
    preview_cyx = render_before_after(
        ref_nuc, native_nuc, reg_nuc, scale_factor=scale_factor
    )

    # Save PNG (OpenCV uses BGR order)
    if save_png:
        png_output_path = output_path.with_suffix(".png")
        cv2.imwrite(str(png_output_path), cyx_to_bgr(preview_cyx))
        logger.info(f"  Saved QC PNG: {png_output_path}")

    # Save TIFF (ImageJ-compatible, CYX order)
    if save_tiff:
        tiff_output_path = output_path.with_suffix(".tif")
        write_tiff(
            str(tiff_output_path),
            preview_cyx,
            ome=True,
            bigtiff=True,
            metadata=preview_metadata,
            **preview_resolution_kwargs,
        )
        logger.info(f"  Saved QC TIFF: {tiff_output_path}")

    logger.info("QC generation complete")
