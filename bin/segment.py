#!/usr/bin/env python3
"""Cell segmentation using StarDist on DAPI channel from multichannel images.

Simplified segmentation pipeline:
- Loads multichannel OME-TIFF (from registration output)
- Extracts DAPI channel (first channel)
- Normalizes using CSBDeep normalize
- Segments using StarDist whole-image processing
"""

from __future__ import annotations

import argparse
import logging
import os
import re

# Add parent directory to path to import lib modules
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from typing import Optional, Tuple

import numpy as np
from csbdeep.utils import normalize
from image_utils import ensure_dir
from logger import configure_logging, get_logger
from numpy.typing import NDArray
from ome_io import write_tiff
from segment_io import expand_labels_tiled, extract_dapi_channel
from stardist.models import StarDist2D

logger = get_logger(__name__)

#: TIFF tile size (px) for the nuclei/cell mask writes below -- a TIFF LAYOUT choice, not a
#: processing tile size. See ``bin/convert_image.py:35-49`` for why an untiled write forces
#: every downstream region read to decode the whole plane. Deliberately half of
#: CONVERT_TIFF_TILE/WRITE_TILE's 2048, not copied from it: measured directly on the mask's
#: own read pattern (many small windowed per-cell/per-region reads, not whole-plane decodes)
#: -- see docs/perf/2026-08-26-rss.md's tiled-vs-striped table.
MASK_TIFF_TILE = 1024

#: A Hugging Face model reference: ``hf://<owner>/<repo>@<40-hex commit>``. The commit is
#: REQUIRED: a branch name like ``main`` moves, and a segmentation run whose weights can
#: change under it is not reproducible. The commit pins every file of the snapshot.
HF_PREFIX = "hf://"
HF_SPEC = re.compile(r"^hf://(?P<repo>[\w.-]+/[\w.-]+)@(?P<rev>[0-9a-f]{40})$")
#: The three files a StarDist2D model folder needs (weights_last.h5 and the logs are not
#: read by StarDist2D(None, name, basedir) and are not downloaded).
HF_FILES = ("config.json", "thresholds.json", "weights_best.h5")

__all__ = [
    "normalize_dapi",
    "resolve_hf_model",
    "load_stardist_model",
    "segment_nuclei",
    "run_segmentation",
]


def normalize_dapi(
    dapi_image: NDArray, pmin: float = 1.0, pmax: float = 99.8
) -> NDArray:
    """
    Normalize DAPI image using CSBDeep percentile normalization.

    Parameters
    ----------
    dapi_image : ndarray, shape (Y, X)
        DAPI channel image.
    pmin : float, optional
        Lower percentile for normalization. Default is 1.0.
    pmax : float, optional
        Upper percentile for normalization. Default is 99.8.

    Returns
    -------
    normalized : ndarray, shape (Y, X)
        Normalized DAPI image with values in [0, 1].

    Notes
    -----
    Uses CSBDeep's normalize function which clips to percentiles
    and scales to [0, 1] range. Converts to float32 to save memory.
    """
    logger.info("Normalizing DAPI channel...")
    logger.info(f"  Percentiles: [{pmin}, {pmax}]")

    # CSBDeep normalize clips to percentiles and scales to [0, 1].
    # It internally converts to float32 via x.astype(float32, copy=True),
    # so pre-converting here would just create a redundant full-image copy.
    normalized = normalize(dapi_image, pmin, pmax, axis=(0, 1))

    logger.info(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    return normalized


def resolve_hf_model(spec: str, cache_dir: Optional[str] = None) -> Tuple[str, str]:
    """Download (once) a StarDist model folder from the Hugging Face Hub.

    Parameters
    ----------
    spec : str
        ``hf://<owner>/<repo>@<commit>``, e.g. the shipped default
        ``hf://DSUnit/stardist@68b94406104bb069a58b64f4c890375c7f2f15df``.
    cache_dir : str, optional
        Where the folder is kept. A folder already there is used WITHOUT any network
        access, which is how an air-gapped site runs: pre-download it into
        ``segmentation_model_dir``. Default: ``./stardist_models`` in the task directory.

    Returns
    -------
    (basedir, name) : tuple of str
        For ``StarDist2D(None, name=name, basedir=basedir)``.
    """
    m = HF_SPEC.match(spec)
    if not m:
        raise ValueError(
            f"Hugging Face model reference {spec!r} is not hf://<owner>/<repo>@<40-hex commit>; "
            "a branch name is refused because the weights behind it can change."
        )
    repo, rev = m.group("repo"), m.group("rev")
    basedir = Path(
        cache_dir if cache_dir not in (None, "", "null", "None") else "stardist_models"
    )
    name = f"{repo.replace('/', '__')}__{rev[:12]}"
    folder = basedir / name
    if all((folder / f).exists() for f in HF_FILES):
        logger.info(f"  Hugging Face model {repo}@{rev[:12]} found in cache: {folder}")
        return str(basedir), name
    # HF_ENDPOINT is the Hub's own variable for a mirror; honour it like huggingface_hub does.
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    folder.mkdir(parents=True, exist_ok=True)
    for fname in HF_FILES:
        dest = folder / fname
        if dest.exists():
            continue
        url = f"{endpoint}/{repo}/resolve/{rev}/{fname}"
        logger.info(f"  Downloading {url}")
        last = None
        for attempt in range(1, 4):
            try:
                tmp = dest.with_name(dest.name + ".part")
                with (
                    urllib.request.urlopen(url, timeout=120) as r,
                    open(tmp, "wb") as fh,
                ):
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                tmp.replace(dest)  # atomic: a killed download never looks complete
                last = None
                break
            except OSError as exc:  # URLError and HTTPError are OSError subclasses
                last = exc
                logger.warning(f"  download attempt {attempt}/3 failed: {exc}")
                time.sleep(2 * attempt)
        if last is not None:
            raise RuntimeError(
                f"Could not download {url} ({last}). Without network access, pre-download "
                f"{', '.join(HF_FILES)} of {repo} at {rev} into "
                f"<segmentation_model_dir>/{name}/ and pass segmentation_model_dir."
            ) from last
    return str(basedir), name


def load_stardist_model(
    model_dir: str, model_name: str, use_gpu: bool = True
) -> StarDist2D:
    """
    Load pre-trained StarDist model.

    Parameters
    ----------
    model_dir : str
        Directory containing the model.
    model_name : str
        Name of the model (e.g., '2D_versatile_fluo').
    use_gpu : bool, optional
        Use GPU acceleration if available. Default is True.

    Returns
    -------
    model : StarDist2D
        Loaded StarDist model.
    """
    logger.info("Loading StarDist model...")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Model directory: {model_dir}")
    logger.info(f"  GPU enabled: {use_gpu}")

    # StarDist ships built-in pretrained models that are downloaded/cached on
    # first use. We fall back to these when no usable custom model directory is
    # given, so the pipeline runs out of the box (and CI/real tests don't need a
    # multi-MB trained model shipped in the repo).
    BUILTIN_MODELS = {
        "2D_versatile_fluo",
        "2D_versatile_he",
        "2D_paper_dsb2018",
        "2D_demo",
    }

    if model_name.startswith(HF_PREFIX):
        # Shipped default: a pinned Hugging Face snapshot, downloaded once into
        # model_dir (or the task directory) and then loaded as a custom model.
        model_dir, model_name = resolve_hf_model(model_name, model_dir)
        logger.info(f"  Resolved to local model folder: {Path(model_dir) / model_name}")

    have_model_dir = model_dir not in (None, "", "null", "None")
    model_path = (Path(model_dir) / model_name) if have_model_dir else None

    if have_model_dir and model_path.exists():
        # Custom model on disk (production path).
        config_file = model_path / "config.json"
        logger.info(f"  Loading custom model from: {model_path}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        model = StarDist2D(None, name=model_name, basedir=model_dir)
    elif model_name in BUILTIN_MODELS:
        # No usable custom dir — load a StarDist built-in pretrained model.
        logger.info(
            f"  No custom model directory; loading built-in pretrained model: {model_name}"
        )
        model = StarDist2D.from_pretrained(model_name)
    else:
        raise FileNotFoundError(
            f"StarDist model '{model_name}' could not be loaded: no custom model at "
            f"{model_path} and '{model_name}' is not a built-in pretrained model "
            f"(known built-ins: {sorted(BUILTIN_MODELS)}). Provide --model-dir or use "
            f"a built-in model name."
        )

    if hasattr(model, "config"):
        model.config.use_gpu = use_gpu

    logger.info("  ✓ Model loaded successfully")

    return model


def segment_nuclei(
    normalized_dapi: NDArray,
    model: StarDist2D,
    n_tiles: Tuple[int, int] = (24, 24),
    expand_distance: int = 10,
    prob_thresh: Optional[float] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Segment nuclei and create whole-cell masks.

    Parameters
    ----------
    normalized_dapi : ndarray, shape (Y, X)
        Normalized DAPI image.
    model : StarDist2D
        Loaded StarDist model.
    n_tiles : tuple of int, optional
        Number of tiles for tiled processing. Default is (24, 24).
        Larger images benefit from tiling.
    expand_distance : int, optional
        Distance (pixels) to expand nuclei labels to create whole-cell masks.
        Default is 10.

    Returns
    -------
    nuclei_mask : ndarray, shape (Y, X), dtype uint32
        Nuclei segmentation mask with unique cell labels.
    cell_mask : ndarray, shape (Y, X), dtype uint32
        Whole-cell segmentation mask (expanded from nuclei).

    Notes
    -----
    - Uses StarDist predict_instances with tiling for memory efficiency
    - Expands nuclei labels using skimage.segmentation.expand_labels
    - Label 0 is background, labels ≥1 are cells
    """
    logger.info("Segmenting nuclei on whole image...")
    logger.info(f"  Image shape: {normalized_dapi.shape}")
    logger.info(f"  Tiling: {n_tiles}")
    logger.info(f"  Cell expansion distance: {expand_distance}px")

    start_time = time.time()

    # Predict nuclei instances using StarDist. prob_thresh=None uses the model's
    # built-in default; a lower value detects fainter/smaller nuclei (used by the
    # test profile so the built-in model picks up small synthetic nuclei).
    if prob_thresh is not None:
        logger.info(f"  Probability threshold: {prob_thresh}")
    nuclei_labels, _ = model.predict_instances(
        normalized_dapi,
        n_tiles=n_tiles,
        prob_thresh=prob_thresh,
        show_tile_progress=False,
        verbose=False,
    )

    logger.info("  ✓ Nuclei detection complete")

    # Free normalized DAPI to reduce memory pressure before expansion
    logger.info("  Freeing normalized DAPI from memory...")
    del normalized_dapi
    import gc

    gc.collect()

    # Expand nuclei labels to create whole-cell masks (tiled for memory efficiency)
    logger.info("  Expanding nuclei labels to create cell masks...")
    cell_labels = expand_labels_tiled(nuclei_labels, distance=expand_distance)

    elapsed = time.time() - start_time

    # StarDist labels are dense 0..N; .max() avoids np.unique's full-array sort copy
    n_nuclei = int(nuclei_labels.max())
    n_cells = int(cell_labels.max())

    logger.info(f"  Nuclei detected: {n_nuclei}")
    logger.info(f"  Cells (after expansion): {n_cells}")
    logger.info(f"  Segmentation time: {elapsed:.2f}s")

    # Ensure uint32 for large label counts
    return nuclei_labels.astype(np.uint32), cell_labels.astype(np.uint32)


def run_segmentation(
    image_path: str,
    output_dir: str,
    model_dir: str,
    model_name: str,
    dapi_channel_index: int = 0,
    use_gpu: bool = True,
    n_tiles: Tuple[int, int] = (24, 24),
    expand_distance: int = 10,
    pmin: float = 1.0,
    pmax: float = 99.8,
    prob_thresh: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Run complete segmentation pipeline on multichannel image.

    Parameters
    ----------
    image_path : str
        Path to multichannel OME-TIFF image (e.g., from VALIS registration).
    output_dir : str
        Output directory for segmentation masks.
    model_dir : str
        StarDist model directory.
    model_name : str
        StarDist model name.
    dapi_channel_index : int, optional
        Index of DAPI channel in multichannel image. Default is 0.
    use_gpu : bool, optional
        Use GPU acceleration. Default is True.
    n_tiles : tuple of int, optional
        Number of tiles for processing. Default is (24, 24).
    expand_distance : int, optional
        Distance to expand nuclei for whole-cell masks. Default is 10.
    pmin : float, optional
        Lower percentile for normalization. Default is 1.0.
    pmax : float, optional
        Upper percentile for normalization. Default is 99.8.

    Returns
    -------
    nuclei_mask_path : str
        Path to saved nuclei segmentation mask.
    cell_mask_path : str
        Path to saved whole-cell segmentation mask.
    """
    logger.info("=" * 80)
    logger.info("CELL SEGMENTATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Create output directory
    ensure_dir(output_dir)

    # 1. Extract DAPI channel from multichannel image
    dapi_image, metadata = extract_dapi_channel(
        image_path, dapi_channel_index, logger=logger
    )

    # 2. Normalize using CSBDeep
    normalized_dapi = normalize_dapi(dapi_image, pmin=pmin, pmax=pmax)
    del dapi_image  # Free uint16 original — only normalized float32 needed from here

    # 3. Load StarDist model
    model = load_stardist_model(model_dir, model_name, use_gpu=use_gpu)

    # 4. Segment nuclei and create cell masks
    nuclei_mask, cell_mask = segment_nuclei(
        normalized_dapi,
        model,
        n_tiles=n_tiles,
        expand_distance=expand_distance,
        prob_thresh=prob_thresh,
    )
    del normalized_dapi, model  # Free float32 image + TF weights before save phase

    # 5. Save outputs
    basename = Path(image_path).stem
    nuclei_mask_path = Path(output_dir) / f"{basename}_nuclei_mask.tif"
    cell_mask_path = Path(output_dir) / f"{basename}_cell_mask.tif"

    logger.info("")
    logger.info("Saving segmentation masks...")
    logger.info(f"  Nuclei mask: {nuclei_mask_path.name}")
    # bigtiff for the same reason tiled_stitch.py:118 and extract_mask_series.py give:
    # a 40000x40000 uint32 label mask is 6.4 GB before compression, and classic TIFF's
    # 32-bit offsets overflow past 4 GB. Compression usually keeps it under -- usually is
    # not a contract. Guarded by tests/test_slide_io_seam.py.
    write_tiff(
        nuclei_mask_path,
        nuclei_mask,
        compression="zlib",
        bigtiff=True,
        tile=(MASK_TIFF_TILE, MASK_TIFF_TILE),
    )
    del nuclei_mask  # Free before writing cell mask

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
    logger.info("✓ SEGMENTATION COMPLETE")
    logger.info("=" * 80)

    return str(nuclei_mask_path), str(cell_mask_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cell segmentation using StarDist on multichannel images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to multichannel OME-TIFF image (e.g., registered image from VALIS)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing a custom StarDist model. If omitted, "
        "--model-name is loaded as a StarDist built-in pretrained model "
        '(e.g. "2D_versatile_fluo"). With an hf:// --model-name it is the '
        "download cache (a model already there is used offline).",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help='StarDist model: a folder name under --model-dir, a built-in (e.g. "2D_versatile_fluo"), '
        'or a pinned Hugging Face snapshot "hf://<owner>/<repo>@<commit>"',
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./segmentation_output",
        help="Output directory for segmentation masks",
    )

    parser.add_argument(
        "--dapi-channel",
        type=int,
        default=0,
        help="Index of DAPI channel in multichannel image",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration (if available)",
    )

    parser.add_argument(
        "--n-tiles",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("Y", "X"),
        help="Number of tiles for processing (Y X)",
    )

    parser.add_argument(
        "--expand-distance",
        type=int,
        default=10,
        help="Distance (pixels) to expand nuclei labels for whole-cell masks",
    )

    parser.add_argument(
        "--pmin", type=float, default=1.0, help="Lower percentile for normalization"
    )

    parser.add_argument(
        "--pmax", type=float, default=99.8, help="Upper percentile for normalization"
    )

    parser.add_argument(
        "--prob-thresh",
        type=float,
        default=None,
        help="StarDist detection probability threshold. Omit to use the model "
        "default; lower it to detect fainter/smaller nuclei.",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)

    args = parse_args()

    run_segmentation(
        image_path=args.image,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        dapi_channel_index=args.dapi_channel,
        use_gpu=args.use_gpu,
        n_tiles=tuple(args.n_tiles),
        expand_distance=args.expand_distance,
        pmin=args.pmin,
        pmax=args.pmax,
        prob_thresh=args.prob_thresh,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
