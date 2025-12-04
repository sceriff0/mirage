#!/usr/bin/env python3
"""Cell phenotyping based on marker expression.

This module assigns phenotype labels to cells based on marker expression
thresholds and creates phenotype masks from segmentation labels.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import zscore
from numpy.typing import NDArray

from _common import (
    ensure_dir,
    save_tiff,
)

logger = logging.getLogger(__name__)


__all__ = [
    "labels_to_phenotype",
    "run_phenotyping_pipeline",
]


# Phenotyping configuration - can be overridden via parameters
DEFAULT_MARKERS = [
    'CD163', 'CD14', 'CD45', 'CD3', 'CD8', 'CD4', 'FOXP3',
    'PANCK', 'VIMENTIN', 'SMA', 'L1CAM', 'PAX2', 'CD74',
    'GZMB', 'PD1', 'PDL1'
]

DEFAULT_CUTOFFS = [
    0.7, 0.9, 0.4, 0.2, 0.4, 0.9, 1.3,
    0.2, 0.2, 0.2, 0.3, 1.0, 1.3,
    1.2, 1.5, 0.4
]


def labels_to_phenotype(mask: NDArray, phenotype_df: pd.DataFrame) -> NDArray:
    """Map label array to phenotype numbers.

    Parameters
    ----------
    mask : ndarray
        Label image where each integer corresponds to a cell label.
    phenotype_df : DataFrame
        DataFrame with columns 'label' and 'phenotype_num'.

    Returns
    -------
    ndarray
        Array with same shape as mask where labels are replaced by phenotype codes.
    """
    logger.info("Mapping labels to phenotypes")

    map_arr = phenotype_df[['label', 'phenotype_num']].to_numpy()
    max_val = max(map_arr[:, 0].max(), mask.max()) + 1
    lookup = np.zeros(max_val + 1, dtype=map_arr[:, 1].dtype)
    lookup[map_arr[:, 0]] = map_arr[:, 1]
    remapped = lookup[mask]

    logger.info("Label to phenotype mapping completed")
    return remapped


def run_phenotyping_pipeline(
    cell_df: pd.DataFrame,
    mask: NDArray,
    markers: list = None,
    cutoffs: list = None,
    quality_percentile: float = 1.0,
    noise_percentile: float = 0.01
) -> Tuple[pd.DataFrame, NDArray]:
    """Run cell phenotyping pipeline.

    Parameters
    ----------
    cell_df : DataFrame
        Cell data with marker intensities and morphological features.
    mask : ndarray
        Segmentation mask.
    markers : list, optional
        List of marker names for phenotyping.
    cutoffs : list, optional
        Expression cutoffs for each marker.
    quality_percentile : float, optional
        Percentile for quality filtering.
    noise_percentile : float, optional
        Percentile for noise removal.

    Returns
    -------
    phenotypes_df : DataFrame
        Cell data with phenotype assignments.
    phenotypes_mask : ndarray
        Phenotype mask image.
    """
    if markers is None:
        markers = DEFAULT_MARKERS
    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS

    logger.info("Starting phenotyping pipeline")
    logger.info(f"Markers: {markers}")
    logger.info(f"Number of cells: {len(cell_df)}")

    # Convert percentile parameters if they're passed as percentages (>1) instead of decimals
    if quality_percentile > 1:
        logger.info(f"Converting quality_percentile from {quality_percentile}% to {quality_percentile/100}")
        quality_percentile = quality_percentile / 100.0
    if noise_percentile > 1:
        logger.info(f"Converting noise_percentile from {noise_percentile}% to {noise_percentile/100}")
        noise_percentile = noise_percentile / 100.0

    # Reorder columns (keep available ones)
    base_cols = ['y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
                 'axis_major_length', 'axis_minor_length', 'label']

    # Get marker columns that exist in the dataframe
    available_markers = [m for m in markers if m in cell_df.columns]
    available_markers.append('DAPI')  # Always include DAPI

    available_cols = [c for c in base_cols if c in cell_df.columns]
    all_cols = available_cols + available_markers

    cell_df = cell_df[[c for c in all_cols if c in cell_df.columns]]

    # Quality filtering
    logger.info("Applying quality filters")
    nuc_thres = np.percentile(cell_df['DAPI'], quality_percentile)
    size_thres = np.percentile(cell_df['area'], quality_percentile)
    cell_df_filtered = cell_df[
        (cell_df['DAPI'] > nuc_thres) & (cell_df['area'] > size_thres)
    ].copy()

    logger.info(f"Cells after quality filter: {len(cell_df_filtered)}")

    # Normalization
    logger.info("Normalizing marker intensities")
    list_out = ['eccentricity', 'perimeter', 'convex_area',
                'axis_major_length', 'axis_minor_length']
    list_keep = ['DAPI', 'x', 'y', 'area', 'label']

    # Remove morphological features (keep only markers + metadata)
    dfin = cell_df_filtered.drop(
        [c for c in list_out if c in cell_df_filtered.columns],
        axis=1
    )
    df_loc = dfin[[c for c in list_keep if c in dfin.columns]]
    dfz = dfin.drop([c for c in list_keep if c in dfin.columns], axis=1)

    # Z-score normalization
    dfz1 = pd.DataFrame(
        zscore(dfz, axis=0),
        index=dfz.index,
        columns=dfz.columns
    )
    dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")

    # Noise removal
    logger.info("Removing noise")
    last_marker = 'VIMENTIN' if 'VIMENTIN' in dfz_all.columns else available_markers[-1]
    col_num_last_marker = dfz_all.columns.get_loc(last_marker)

    dfz_copy = dfz_all.copy()
    dfz_copy["Count"] = dfz_all.iloc[:, :col_num_last_marker + 1].ge(0).sum(axis=1)
    dfz_copy["z_sum"] = dfz_all.iloc[:, :col_num_last_marker + 1].sum(axis=1)

    count_threshold = dfz_copy["Count"].quantile(1 - noise_percentile)
    z_sum_threshold = dfz_copy["z_sum"].quantile(1 - noise_percentile)

    # Keep cells below noise thresholds
    df_nn = dfz_copy.copy()

    # Phenotyping based on marker expression
    logger.info("Assigning phenotypes")

    # Ensure all markers have cutoffs
    marker_cutoffs = dict(zip(markers[:len(cutoffs)], cutoffs))

    df_nn['pheno_markers'] = [[] for _ in range(len(df_nn))]

    # Mark positive cells for each marker
    for marker, cutoff in marker_cutoffs.items():
        if marker in df_nn.columns:
            sel = df_nn[df_nn[marker] >= cutoff]
            sel_idx = sel.index
            for idx in sel_idx:
                df_nn.at[idx, 'pheno_markers'].append(marker)

    # Assign phenotypes based on marker combinations
    df_nn['phenotype'] = 'Unknown'

    # Immune compartment (CD45+)
    df_nn.loc[
        df_nn['pheno_markers'].apply(lambda x: "CD45" in x),
        'phenotype'
    ] = "Immune"

    # T cell subsets (CD45+ CD3+)
    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" in x and "CD4" in x and
            "FOXP3" in x and "CD8" not in x
        ),
        'phenotype'
    ] = "CD4 T regulatory"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" in x and "CD4" in x and
            "FOXP3" not in x and "CD8" not in x
        ),
        'phenotype'
    ] = "T helper"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" in x and "GZMB" not in x and
            "FOXP3" not in x and "CD8" in x and "CD4" not in x
        ),
        'phenotype'
    ] = "T cytotoxic"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" in x and "GZMB" not in x and
            "FOXP3" in x and "CD8" in x and "CD4" not in x
        ),
        'phenotype'
    ] = "CD8 T regulatory"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" in x and "GZMB" in x and
            "CD8" in x and "CD4" not in x
        ),
        'phenotype'
    ] = "activated T cytotoxic"

    # Macrophages (CD45+ CD3-)
    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" not in x and "CD14" in x and "CD163" in x
        ),
        'phenotype'
    ] = "Macrophages"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" not in x and "CD14" in x and "CD163" not in x
        ),
        'phenotype'
    ] = "M1"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" in x and "CD3" not in x and "CD14" not in x and "CD163" in x
        ),
        'phenotype'
    ] = "M2"

    # Stroma (CD45-)
    df_nn.loc[
        df_nn['pheno_markers'].apply(lambda x: "CD45" not in x),
        'phenotype'
    ] = "Stroma"

    df_nn.loc[
        df_nn['pheno_markers'].apply(lambda x: "CD45" not in x and "SMA" in x),
        'phenotype'
    ] = "Stroma"

    # Tumor (CD45- SMA-)
    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" not in x and "SMA" not in x and "PANCK" in x
        ),
        'phenotype'
    ] = "PANCK+ Tumor"

    df_nn.loc[
        df_nn['pheno_markers'].apply(
            lambda x: "CD45" not in x and "SMA" not in x and "VIMENTIN" in x
        ),
        'phenotype'
    ] = "VIM+ Tumor"

    # Add numeric phenotype labels
    pheno_complete = df_nn['phenotype'].value_counts().index.values
    for pp, p in enumerate(pheno_complete, start=1):
        sel = df_nn[df_nn['phenotype'] == p].index
        df_nn.loc[sel, 'phenotype_num'] = pp

    df_nn['phenotype_num'] = df_nn['phenotype_num'].fillna(0).astype(int)

    logger.info(f"Phenotype distribution:\n{df_nn['phenotype'].value_counts()}")

    # Create phenotype mask
    logger.info("Creating phenotype mask")
    phenotype_mask = labels_to_phenotype(mask, df_nn)

    return df_nn, phenotype_mask


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell phenotyping based on marker expression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--cell_data',
        required=True,
        help='Path to input CSV file with cell data'
    )
    parser.add_argument(
        '--segmentation_mask',
        required=True,
        help='Path to segmentation mask file (npy)'
    )
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Output directory'
    )

    # Optional arguments
    parser.add_argument(
        '--markers',
        nargs='+',
        default=DEFAULT_MARKERS,
        help='List of marker names for phenotyping'
    )
    parser.add_argument(
        '--cutoffs',
        nargs='+',
        type=float,
        default=DEFAULT_CUTOFFS,
        help='Expression cutoffs for each marker'
    )
    parser.add_argument(
        '--quality_percentile',
        type=float,
        default=1.0,
        help='Percentile for quality filtering (DAPI/area)'
    )
    parser.add_argument(
        '--noise_percentile',
        type=float,
        default=0.01,
        help='Percentile for noise removal'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    ensure_dir(args.output_dir)

    logger.info("=" * 80)
    logger.info("Starting Cell Phenotyping Pipeline")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading cell data: {args.cell_data}")
    cell_df = pd.read_csv(args.cell_data).drop_duplicates(subset='label', keep='first')

    logger.info(f"Loading segmentation mask: {args.segmentation_mask}")
    # Support both .npy and TIFF formats
    if args.segmentation_mask.endswith('.npy'):
        mask = np.load(args.segmentation_mask)
    else:
        # Load TIFF format
        mask = tifffile.imread(args.segmentation_mask)
        # Ensure 2D
        if mask.ndim > 2:
            logger.warning(f"Mask has {mask.ndim} dimensions, taking first channel")
            mask = mask[0] if mask.shape[0] < mask.shape[-1] else mask[..., 0]

    # Run phenotyping
    phenotypes_data, phenotypes_mask = run_phenotyping_pipeline(
        cell_df,
        mask,
        markers=args.markers,
        cutoffs=args.cutoffs,
        quality_percentile=args.quality_percentile,
        noise_percentile=args.noise_percentile
    )

    # Save outputs
    output_csv = os.path.join(args.output_dir, 'phenotypes_data.csv')
    output_mask = os.path.join(args.output_dir, 'phenotypes_mask.tiff')

    logger.info(f"Saving phenotype data: {output_csv}")
    phenotypes_data.to_csv(output_csv, index=False)

    logger.info(f"Saving phenotype mask: {output_mask}")
    save_tiff(phenotypes_mask, output_mask)

    logger.info("=" * 80)
    logger.info("Cell Phenotyping Pipeline Completed Successfully")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
