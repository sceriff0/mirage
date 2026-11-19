#!/usr/bin/env python3
"""Cell segmentation using CellSAM on the nuclear (DAPI) channel.

CellSAM (https://github.com/vanvalenlab/cellSAM) is a SAM-based foundation
model for cell segmentation. ``cellsam_pipeline`` returns a *single* instance
label map; it does not separately predict nuclei vs whole-cell.

Behaves like the StarDist backend (``segment.py``): DAPI goes in the nuclear slot of
CellSAM's ``(blank, nuclear, whole-cell)`` input, the instance map is the nuclei mask,
and the cell mask is the nuclei expanded by ``--expand-distance``.

Writes ``{prefix}_nuclei_mask.tif`` and ``{prefix}_cell_mask.tif`` (uint32, zlib), the
same contract as ``segment.py`` and ``segment_instantseg.py``. ``segment_io`` has no ML
dependencies, so importing it does not pull in stardist/tensorflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent / "utils"))

import numpy as np
from image_utils import ensure_dir
from logger import configure_logging, get_logger
from numpy.typing import NDArray
from ome_io import write_tiff
from segment_io import expand_labels_tiled, extract_dapi_channel

logger = get_logger(__name__)

#: TIFF tile size (px) of the mask files, a layout choice, not a processing tile. Smaller
#: than the image writers' 2048 because masks are read in many small windows.
MASK_TIFF_TILE = 1024


def _build_cellsam_input(dapi_image: NDArray) -> NDArray:
    """Lay out the nuclear channel as CellSAM's ``(H, W, 3)`` input.

    DAPI goes in the nuclear slot, the other two are zeros. A bare 2D array may be
    treated as brightfield.
    """
    blank = np.zeros_like(dapi_image)
    whole_cell = np.zeros_like(dapi_image)
    cellsam_input = np.stack([blank, dapi_image, whole_cell], axis=-1)
    logger.info(
        f"  Built CellSAM input (blank, nuclear, whole-cell): "
        f"shape={cellsam_input.shape}, dtype={cellsam_input.dtype}"
    )
    return cellsam_input


def run_cellsam(
    image_path: str,
    output_dir: str,
    dapi_channel_index: int = 0,
    expand_distance: int = 10,
    use_gpu: bool = False,
    use_wsi: bool = True,
    bbox_threshold: float = 0.4,
    block_size: int = 400,
    overlap: int = 56,
    model_path: str | None = None,
    prefix: str | None = None,
) -> Tuple[str, str]:
    """Run the CellSAM segmentation pipeline on the nuclear channel.

    Parameters
    ----------
    image_path
        Path to multichannel OME-TIFF (e.g. registration output).
    output_dir
        Directory where ``*_nuclei_mask.tif`` / ``*_cell_mask.tif`` are written.
    dapi_channel_index
        Index of the nuclear (DAPI) channel. Default 0.
    expand_distance
        Distance (px) to expand CellSAM nuclei labels into the whole-cell mask
        (StarDist-style). Default 10.
    use_gpu
        Whether GPU is requested. ``cellsam_pipeline`` selects the device itself; this
        raises if CUDA is unavailable.
    use_wsi
        Enable CellSAM's native whole-slide tiling (recommended for large
        registered images).
    bbox_threshold
        CellSAM bounding-box confidence threshold (precision/recall knob).
    block_size
        Tile side length (px) for WSI mode. Upstream-recommended range [256, 2048].
    overlap
        Tile overlap (px) for WSI mode.
    model_path
        Path to pre-downloaded CellSAM weights. If None, ``cellsam_pipeline``
        auto-downloads via ``get_model`` (needs ``DEEPCELL_ACCESS_TOKEN``).
    prefix
        Output filename prefix. Defaults to the input image's stem.
    """
    logger.info("=" * 80)
    logger.info("CELL SEGMENTATION PIPELINE (CellSAM)")
    logger.info("=" * 80)
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Nuclear (DAPI) channel index: {dapi_channel_index}")
    logger.info(f"Cell expansion distance: {expand_distance}px")
    logger.info(f"Use GPU (requested): {use_gpu}")
    logger.info(f"use_wsi: {use_wsi}  block_size: {block_size}  overlap: {overlap}")
    logger.info(f"bbox_threshold: {bbox_threshold}")
    logger.info(f"model_path: {model_path or '(auto-download)'}")
    logger.info("")

    ensure_dir(output_dir)

    # Defer heavy imports until after argparse/help works without GPU stacks.
    from cellSAM import cellsam_pipeline  # type: ignore

    # cellsam_pipeline has no device argument and falls back to CPU silently, which
    # takes days on a WSI; fail fast instead.
    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--use-gpu requested but torch.cuda.is_available() is False. "
                "CellSAM would silently fall back to CPU (multi-day WSI runs). "
                "Check that the GPU is bound into the container: Docker needs "
                "'--gpus all', Singularity needs '--nv', and the host must have "
                "the NVIDIA Container Toolkit / drivers installed. To deliberately "
                "run on CPU, launch the pipeline with seg_gpu=false."
            )
        logger.info(f"  CUDA available: {torch.cuda.get_device_name(0)}")

    dapi_image, _metadata = extract_dapi_channel(
        image_path, dapi_channel_index, logger=logger
    )
    cellsam_input = _build_cellsam_input(dapi_image)
    del dapi_image

    # use_wsi tiles internally so peak memory scales with block_size; needed for WSIs.
    logger.info("Running cellsam_pipeline...")
    seg_start = time.time()
    nuclei_labels = cellsam_pipeline(
        cellsam_input,
        model_path=model_path,
        bbox_threshold=bbox_threshold,
        use_wsi=use_wsi,
        block_size=block_size,
        overlap=overlap,
    )
    logger.info(f"  Segmentation time: {time.time() - seg_start:.2f}s")
    del cellsam_input

    nuclei_mask = np.asarray(nuclei_labels).astype(np.uint32, copy=False)

    logger.info(
        f"Expanding nuclei labels to create cell mask (distance={expand_distance}px)..."
    )
    cell_mask = expand_labels_tiled(nuclei_mask, distance=expand_distance).astype(
        np.uint32
    )

    n_nuclei = int(nuclei_mask.max())
    n_cells = int(cell_mask.max())
    logger.info(f"  Nuclei detected: {n_nuclei}")
    logger.info(f"  Cells (after expansion): {n_cells}")

    basename = prefix if prefix else Path(image_path).stem
    nuclei_mask_path = Path(output_dir) / f"{basename}_nuclei_mask.tif"
    cell_mask_path = Path(output_dir) / f"{basename}_cell_mask.tif"

    logger.info("")
    logger.info("Saving segmentation masks...")
    logger.info(f"  Nuclei mask: {nuclei_mask_path.name}")
    # bigtiff: a 40k x 40k uint32 mask is 6.4 GB raw, past classic TIFF's 4 GB offsets.
    write_tiff(
        nuclei_mask_path,
        nuclei_mask,
        compression="zlib",
        bigtiff=True,
        tile=(MASK_TIFF_TILE, MASK_TIFF_TILE),
    )
    del nuclei_mask

    logger.info(f"  Cell mask: {cell_mask_path.name}")
    write_tiff(
        cell_mask_path,
        cell_mask,
        compression="zlib",
        bigtiff=True,
        tile=(MASK_TIFF_TILE, MASK_TIFF_TILE),
    )
    del cell_mask

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ CellSAM SEGMENTATION COMPLETE")
    logger.info("=" * 80)

    return str(nuclei_mask_path), str(cell_mask_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cell segmentation using CellSAM on the nuclear (DAPI) channel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to multichannel OME-TIFF image (e.g., the registered image)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for segmentation masks",
    )

    parser.add_argument(
        "--dapi-channel",
        type=int,
        default=0,
        help="Index of the nuclear (DAPI) channel in the multichannel image",
    )

    parser.add_argument(
        "--expand-distance",
        type=int,
        default=10,
        help="Distance (pixels) to expand nuclei labels for the whole-cell mask",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Require a GPU: exit with an error if CUDA is unavailable (cellsam_pipeline picks the device itself)",
    )

    parser.add_argument(
        "--use-wsi",
        action="store_true",
        default=False,
        help="Enable CellSAM native WSI tiling (recommended for large images)",
    )

    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.4,
        help="CellSAM bounding-box confidence threshold (precision/recall knob)",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Tile block size (px) for WSI mode",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=56,
        help="Tile overlap (px) for WSI mode",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-downloaded CellSAM weights. If omitted, weights are "
        "auto-downloaded (requires DEEPCELL_ACCESS_TOKEN).",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (defaults to input image stem)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)
    args = parse_args()

    run_cellsam(
        image_path=args.image,
        output_dir=args.output_dir,
        dapi_channel_index=args.dapi_channel,
        expand_distance=args.expand_distance,
        use_gpu=args.use_gpu,
        use_wsi=args.use_wsi,
        bbox_threshold=args.bbox_threshold,
        block_size=args.block_size,
        overlap=args.overlap,
        model_path=args.model_path,
        prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
