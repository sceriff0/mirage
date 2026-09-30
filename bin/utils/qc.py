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
``create_registration_qc`` reads its nuclear/fiducial channel through
``tiled_io.open_lazy`` + ``read_decimated`` rather than ``tifffile.imread``ing
every marker channel of the whole slide. What that removes is the
**N-channel overread**: only the ``pick_nuclear_index``-resolved channel is
ever decoded, not every marker channel in the panel. It does **not** remove
any spatial overread -- the read stays at full resolution
(``decimation_factor(..., max_dim=None)`` always returns ``1`` here), because
``GENERATE_REGISTRATION_QC`` runs with ``save_fullres=True`` unconditionally
in production (see ``conf/modules.config``'s ``GENERATE_REGISTRATION_QC``
block: no ``--no-fullres`` is ever passed), so the ``*_fullres.tif`` artifact
this function publishes genuinely needs full-resolution pixels. Decimating
the read would change that published artifact, not just the (already
downsampled) preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from logger import get_logger
from metadata import extract_channel_names_from_ome, pick_nuclear_index
from numpy.typing import NDArray
from ome_io import ome_metadata, read_info, read_plane, write_tiff
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
        # Fix: Round before converting to uint8 to avoid truncation artifacts
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
    **Pad-or-crop, never rescale — this is the whole point of the function.**
    The "before" panel of the registration QC figure overlays the NATIVE moving
    slide on the reference. Rescaling a differently-sized native image onto the
    reference's shape would absorb the scale component of the misalignment into
    the resampling, so the panel would understate the pre-registration error and
    make registration look better than it was. Padding and cropping preserve
    every pixel's original position in the reference frame, which is what a
    reader of the figure is entitled to assume they are seeing.

    Origin-aligned rather than centre-aligned because that is the convention
    both registration backends use: VALIS resolves slides into a shared space
    whose origin is the reference's, and the tiled/STARE backend warps each
    moving slide into the reference's shape from the same corner.
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
        # Fix: Round before converting to uint8 to avoid truncation artifacts
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

    The separator is blue 255 rather than black. Black is indistinguishable from
    a dark region of either panel, so a reader cannot tell where one panel ends;
    blue is the one channel a nuclear overlay never uses (red = moving, green =
    reference, blue = 0), so the band is unmistakable and cannot be mistaken for
    signal.
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
        "after" panel alone, which is what this process produced before the
        before/after change and keeps the script usable by hand on a pair.
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
    # The REGISTERED plane is expected to already be on the reference canvas -- that is
    # what registration produced it for. Routing it through compose_on_reference_canvas
    # (needed because the NATIVE plane genuinely differs in shape) means a registered
    # slide that is NOT on the reference canvas is now silently padded or cropped, where
    # create_nuclear_overlay used to raise "Shape mismatch". Pad-or-crop is the right
    # rendering behaviour -- a QC figure that refuses to draw tells its reader nothing --
    # but it must not be silent, because the discrepancy is a registration-output defect,
    # not a QC one. The native panel is deliberately NOT warned about: differing there is
    # the normal case.
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
        rendered, which is what this function produced before the before/after
        change and keeps it usable by hand on a plain pair.
    pixel_size_um : float, optional
        The reference/registered images' pixel size in micrometers, i.e.
        ``pixel_size.read_ome_pixel_size``'s answer for this patient. When given,
        it is stamped as the ``XResolution``/``ResolutionUnit`` TIFF tags on BOTH
        ``*_QC_RGB.tif`` and ``*_QC_RGB_fullres.tif`` -- ``resolution`` is
        pixels-per-unit, so the tag is ``1 / pixel_size_um`` at full resolution
        and scaled by ``scale_factor`` for the downsampled preview, which covers
        the same tissue in fewer, larger pixels. When omitted (the default), no
        resolution tag is written and the output is unchanged from before this
        parameter existed.

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

    # Resolve the nuclear/fiducial channel from OME metadata (convert_image guarantees
    # OME-XML is always present) BEFORE touching pixel data, so a malformed/missing
    # OME-XML fails fast without paying for opening either slide's pixel store. This MUST
    # go through pick_nuclear_index rather than testing for the literal "DAPI":
    # nuclear_markers is configurable and a CELLTOX-only panel is a supported input. The
    # old literal test worked only by accident -- it fell back to channel 0, which
    # CONVERT_IMAGE happens to reserve for the nuclear channel -- so it produced a correct
    # overlay and a spurious warning on every non-DAPI slide, and would have picked the
    # wrong channel the moment that promotion invariant changed.
    ref_channels = extract_channel_names_from_ome(reference_path)
    reg_channels = extract_channel_names_from_ome(registered_path)

    # A silent fallback to channel 0 here used to work ONLY by accident: channel 0 is
    # CONVERT_IMAGE's reserved nuclear-channel slot, so the fallback happened to
    # reproduce the correct overlay on every slide that followed that convention, and
    # emitted a spurious warning on every slide that did not. The moment a panel is
    # nuclear-marker-only under a NON-default --nuclear-markers list that fails to
    # match (e.g. a CELLTOX-only panel queried with the DAPI-only default), the
    # fallback silently builds the overlay from a marker channel instead -- a
    # misregistration would then be masked by a QC image that looks aligned only
    # because it was never looking at the nuclear signal. Raise instead: a QC failure
    # under retry-then-fail is the correct outcome for "this panel does not have any
    # of the configured nuclear/fiducial markers", not a mis-rendered PNG that still
    # exits 0. See tests/test_generate_registration_qc.py::
    # test_a_marker_list_that_matches_nothing_fails_loudly.
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

    # Read only the resolved nuclear/fiducial channel of each slide, through a lazy,
    # region-readable zarr view (open_lazy) rather than tifffile.imread's whole-stack
    # read of every marker channel at full resolution -- the N-CHANNEL overread this step
    # used to pay, twice, on the two largest inputs the QC step sees. This is NOT a
    # spatial decimation: `factor` is 1 here, and stays 1, because this function always
    # needs true full-resolution pixels (the *_fullres.tif output is unconditional in
    # production -- see the module docstring). Only the unused channels are no longer
    # read; every pixel of the used one still is. decimation_factor is still the source of
    # true decimation.
    #
    # read_decimated always returns float32 here -- do NOT pass a narrower dtype
    # (uint8/uint16) even though the only consumer of ref_nuc/reg_nuc is
    # autoscale_for_display on the way to uint8 (below, and inside create_nuclear_overlay).
    # This was tried (task 3, first pass) and reverted. Two things are both true and worth
    # keeping straight:
    #   1. Narrowing the read is NOT safe on its own: a uint16 array divided by a uint16
    #      range_val promotes to float64 under numpy's true-divide, where the float32
    #      arithmetic autoscale_for_display has always run stays float32 -- different
    #      intermediate precision that demonstrably rounds np.round(normalized * 255)
    #      differently for real (min, max, value) triples (e.g. min=13286, max=62449,
    #      value=52520: uint16-input -> 203, float32-input -> 204 -- float32's coarser
    #      spacing manufactures a spurious .5 tie that round-half-to-even then resolves the
    #      other way). See
    #      tests/test_tiled_io.py::test_autoscale_uint16_vs_float32_disagree_at_a_real_triple
    #      and tests/test_registration_qc_lazy_read.py's adversarial-pixel test for the
    #      reproduced counter-example. Fixing that requires widening back to float32 right
    #      before autoscale_for_display sees the array.
    #   2. That widen-back is exactly what erases the memory saving the narrow read
    #      appeared to offer. Measured (docs/perf/2026-08-26-rss.md, isolated processes,
    #      7000x6000 fixture): a narrow read that is NEVER widened back does measure ~48%
    #      lower peak RSS than this always-float32 read -- but once the mandatory widen-back
    #      cast runs, the measured peak RSS is statistically indistinguishable from never
    #      narrowing at all (670.8 MB either way). So narrowing here buys nothing once it is
    #      made correct, and only adds a cast plus branching. Left at float32, unconditionally.
    #
    # The real, measured win at this call site is read_decimated's pre-allocated
    # destination (no band-list-plus-concatenate doubling, at any factor): measured ~24%
    # lower peak RSS than the prior list+concatenate implementation, at this exact
    # (default-dtype, no narrowing) call shape -- but that number is a READ-PHASE-ONLY
    # figure (isolated open_lazy + read_decimated, no render). The measured full
    # create_registration_qc peak (read + autoscale/stack/rescale/PNG-TIFF write) showed no
    # equivalent improvement -- statistically within noise of the pre-branch baseline. See
    # docs/perf/2026-08-26-rss.md for both sets of numbers; do not extend the read-phase
    # figure into an end-to-end claim it doesn't support.
    ref_close = reg_close = None
    try:
        ref_arr, _ref_dtype, ref_close = open_lazy(reference_path)
        reg_arr, _reg_dtype, reg_close = open_lazy(registered_path)
        factor = decimation_factor([ref_arr.shape[1:], reg_arr.shape[1:]], max_dim=None)
        ref_nuc = read_decimated(ref_arr, ref_nuc_idx, factor)
        reg_nuc = read_decimated(reg_arr, reg_nuc_idx, factor)
    finally:
        # Both opens now live inside this try, so a failure on the SECOND open_lazy call no
        # longer leaks the first store's handle -- but it also means ref_close may never get
        # bound (open_lazy(reference_path) itself can raise before returning). Guard each
        # close individually rather than assuming both succeeded.
        if ref_close is not None:
            ref_close()
        if reg_close is not None:
            reg_close()

    # The native (pre-registration) plane, read through the ome_io seam that owns every
    # image read added after plan 05 (read_info/read_plane). Its channel index is resolved
    # the same way the reference's and the registered slide's are -- pick_nuclear_index
    # over the OME channel names, never a literal "DAPI" test -- because the native slide
    # of a CELLTOX-only panel is as supported an input as any other.
    native_nuc = None
    if native_path is not None:
        native_info = read_info(native_path)
        native_channels = list(native_info.channels or [])
        # Same rule as the reference/registered channel resolution above: raise rather
        # than silently fall back to channel 0, for the same reason (a fallback that
        # happens to be right only when a panel follows the channel-0-is-nuclear
        # convention, and silently wrong otherwise).
        native_nuc_idx = pick_nuclear_index(native_channels, nuclear_markers)
        if native_nuc_idx is None:
            raise ValueError(
                f"No nuclear/fiducial channel found in native image "
                f"{native_path.name} (channels: {native_channels}, markers: "
                f"{list(nuclear_markers) if nuclear_markers else 'default'})."
            )
        # read_plane returns the source's native dtype (uint16 for this pipeline's
        # slides), while ref_nuc/reg_nuc above are always float32 (read_decimated has no
        # dtype= parameter -- see the comment above that read for why that is load-bearing:
        # autoscale_for_display's min-max stretch rounds a uint16 input and a float32 input
        # to different uint8 values at some real (min, max, value) triples, a ±1 LSB
        # difference in the SAME pixel depending only on which dtype it arrived as). Widen
        # here so the native panel goes through create_nuclear_overlay with the same
        # arithmetic precision the reference/registered panels already use.
        native_nuc = np.asarray(
            read_plane(native_path, native_nuc_idx, lazy=True), dtype=np.float32
        )

    # Resolution tags for both QC TIFFs, only when a pixel size was given (the default
    # `None` writes no tag, leaving output byte-for-byte unchanged from before this
    # parameter existed). tifffile's `resolution` is PIXELS PER UNIT, i.e. the inverse of
    # a µm-per-pixel size. The full-res write is always at scale_factor=1.0 (see above), so
    # its pixel size is exactly `pixel_size_um`; the downsampled preview covers the same
    # tissue in `scale_factor` as many pixels per side, so each of its pixels spans
    # `pixel_size_um / scale_factor` -- a DIFFERENT physical size from the full-res write,
    # not the same tag copied twice.
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

    # The XResolution/ResolutionUnit TIFF tags above are not, on their own, a usable scale
    # bar in the tools this figure is actually opened in: ImageJ reads its scale off the
    # ImageDescription's `unit=` line (present only when `metadata={"unit": ...}` is passed
    # alongside `imagej=True`), and an OME-XML reader reads PhysicalSizeX/Y, not the TIFF
    # tags. Both dicts below are built from the SAME `pixel_size_um`/`preview_pixel_size_um`
    # `_resolution_kwargs` already used, so the tag and the description/XML can never drift
    # apart, and both stay absent (byte-for-byte the old dict) when pixel_size_um is None.
    fullres_metadata = {"axes": "CYX", "mode": "composite"}
    if pixel_size_um is not None:
        fullres_metadata["unit"] = "um"

    # ome_metadata is ome_io.py's one builder of the PhysicalSizeX/Y(+Unit) dict (see its
    # own docstring for why a second hand-built copy is exactly the drift this module
    # exists to prevent). channels=None: this composite is an RGB display overlay, not a
    # labelled multi-marker stack, so there is no Channel/Name list to carry.
    preview_metadata = ome_metadata(None, preview_pixel_size_um, axes="CYX")
    preview_metadata["mode"] = "composite"

    # Save full-resolution QC (compressed). scale_factor=1.0 means no rescale.
    if save_fullres:
        rgb_stack_full = render_before_after(
            ref_nuc, native_nuc, reg_nuc, scale_factor=1.0
        )
        fullres_output_path = output_path.with_name(output_path.stem + "_fullres.tif")
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
