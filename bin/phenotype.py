#!/usr/bin/env python3
"""Cell phenotyping based on marker expression.

This module assigns phenotype labels to cells based on marker expression
thresholds and exports results in QuPath-compatible GeoJSON format.

Outputs:
    - phenotypes_data.csv: Full cell data with phenotype assignments
    - phenotypes.geojson: QuPath-compatible cell detections with classifications
    - phenotypes.classifications.json: Color definitions for QuPath
"""

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore
from numpy.typing import NDArray

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

try:
    from logger import get_logger, configure_logging
    from image_utils import ensure_dir
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    def configure_logging(level=logging.INFO):
        logging.basicConfig(level=level)
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)


__all__ = [
    "run_phenotyping_pipeline",
    "export_to_geojson",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default markers and cutoffs
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

# Phenotype colors for QuPath visualization (RGB)
PHENOTYPE_COLORS = {
    "Background": (0, 0, 0),
    "Unknown": (64, 64, 64),
    "Immune": (0, 255, 0),
    "T helper": (255, 255, 0),
    "T cytotoxic": (0, 255, 255),
    "activated T cytotoxic": (0, 200, 255),
    "CD4 T regulatory": (255, 200, 0),
    "CD8 T regulatory": (200, 255, 0),
    "Macrophages": (255, 128, 0),
    "M1": (255, 80, 80),
    "M2": (255, 160, 80),
    "PANCK+ Tumor": (255, 0, 0),
    "VIM+ Tumor": (255, 0, 255),
    "Stroma": (128, 128, 128),
}


# =============================================================================
# GEOJSON EXPORT FUNCTIONS
# =============================================================================

def rgb_to_qupath_color(r: int, g: int, b: int, a: int = 255) -> int:
    """Convert RGB to QuPath's signed 32-bit ARGB color format."""
    value = (a << 24) | (r << 16) | (g << 8) | b
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def get_phenotype_color(phenotype: str, index: int = 0) -> Tuple[int, int, int]:
    """Get color for a phenotype, generating one if not predefined."""
    if phenotype in PHENOTYPE_COLORS:
        return PHENOTYPE_COLORS[phenotype]
    
    # Generate deterministic color for unknown phenotypes
    h = (index * 0.618033988749895) % 1.0
    s = 0.7 + (index % 3) * 0.1
    v = 0.85 + (index % 2) * 0.1
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def export_to_geojson(
    df: pd.DataFrame,
    output_path: str,
    pixel_size: float = 0.325,
    x_col: str = 'x',
    y_col: str = 'y',
    phenotype_col: str = 'phenotype',
    cell_id_col: str = 'label',
    exclude_background: bool = True,
    measurement_cols: Optional[List[str]] = None,
) -> Tuple[int, Dict]:
    """
    Export phenotyped cells to QuPath-compatible GeoJSON.
    
    Parameters
    ----------
    df : DataFrame
        Cell data with coordinates and phenotype assignments.
    output_path : str
        Path for output GeoJSON file.
    pixel_size : float
        Micrometers per pixel for coordinate conversion.
    x_col : str
        Column name for X centroid (in pixels).
    y_col : str
        Column name for Y centroid (in pixels).
    phenotype_col : str
        Column name for phenotype classification.
    cell_id_col : str
        Column name for cell ID/label.
    exclude_background : bool
        Skip cells with phenotype "Background" or "Unknown".
    measurement_cols : list, optional
        Columns to include as measurements. If None, includes all numeric columns.
    
    Returns
    -------
    num_exported : int
        Number of cells exported.
    phenotype_colors : dict
        Mapping of phenotype names to RGB colors.
    """
    logger.info(f"Exporting {len(df)} cells to GeoJSON: {output_path}")
    
    # Determine measurement columns
    if measurement_cols is None:
        exclude_cols = {x_col, y_col, phenotype_col, cell_id_col, 'phenotype_num', 'pheno_markers'}
        measurement_cols = [
            col for col in df.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    # Get unique phenotypes and assign colors
    phenotypes = df[phenotype_col].unique()
    phenotype_color_map = {}
    for i, pheno in enumerate(phenotypes):
        phenotype_color_map[pheno] = get_phenotype_color(pheno, i)
    
    # Build GeoJSON features
    features = []
    skipped = 0
    
    for idx, row in df.iterrows():
        phenotype = row[phenotype_col]
        
        # Skip background/unknown if requested
        if exclude_background and phenotype in ('Background', 'Unknown'):
            skipped += 1
            continue
        
        # Get coordinates (keep in pixels for QuPath)
        x_px = row[x_col]
        y_px = row[y_col]

        if pd.isna(x_px) or pd.isna(y_px):
            skipped += 1
            continue

        x = float(x_px)
        y = float(y_px)
        
        # Get cell ID
        cell_id = row.get(cell_id_col, idx)
        
        # Build measurements dict (convert to µm for display)
        measurements = {
            "Centroid X µm": round(x * pixel_size, 3),
            "Centroid Y µm": round(y * pixel_size, 3),
        }

        for col in measurement_cols:
            if col in row and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    # Convert area to µm² if it's an area column
                    if 'area' in col.lower():
                        measurements[f"{col} µm²"] = round(val * pixel_size * pixel_size, 3)
                    else:
                        measurements[col] = round(val, 4)
                except (ValueError, TypeError):
                    pass

        # Create GeoJSON feature (coordinates in pixels for QuPath)
        feature = {
            "type": "Feature",
            "id": str(cell_id),
            "geometry": {
                "type": "Point",
                "coordinates": [x, y]
            },
            "properties": {
                "objectType": "detection",
                "classification": {
                    "name": phenotype
                },
                "measurements": measurements
            }
        }
        
        features.append(feature)
    
    logger.info(f"  Exported {len(features)} cells, skipped {skipped}")
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save GeoJSON
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"  GeoJSON size: {file_size_mb:.1f} MB")
    
    return len(features), phenotype_color_map


def export_classifications(
    phenotype_colors: Dict[str, Tuple[int, int, int]],
    output_path: str,
    exclude_background: bool = True
):
    """
    Export phenotype classifications JSON for QuPath color setup.
    
    Parameters
    ----------
    phenotype_colors : dict
        Mapping of phenotype name to (R, G, B) tuple.
    output_path : str
        Output path for classifications JSON.
    exclude_background : bool
        Exclude "Background" from output.
    """
    classifications = []
    
    for name, rgb in phenotype_colors.items():
        if exclude_background and name in ('Background', 'Unknown'):
            continue
        
        r, g, b = rgb
        color_int = rgb_to_qupath_color(r, g, b)
        
        classifications.append({
            "name": name,
            "color": color_int,
            "rgb": [r, g, b],
            "hex": f"#{r:02x}{g:02x}{b:02x}"
        })
    
    with open(output_path, 'w') as f:
        json.dump(classifications, f, indent=2)
    
    logger.info(f"Saved {len(classifications)} classifications to {output_path}")


# =============================================================================
# PHENOTYPING PIPELINE
# =============================================================================

def run_phenotyping_pipeline(
    cell_df: pd.DataFrame,
    markers: list = None,
    cutoffs: list = None,
    quality_percentile: float = 1.0,
    noise_percentile: float = 0.01
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Run cell phenotyping pipeline.

    Parameters
    ----------
    cell_df : DataFrame
        Cell data with marker intensities and morphological features.
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
    phenotype_mapping : dict
        Mapping of phenotype number to name.
    """
    if markers is None:
        markers = DEFAULT_MARKERS
    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS

    logger.info("Starting phenotyping pipeline")
    logger.info(f"Markers: {markers}")
    logger.info(f"Number of cells: {len(cell_df)}")

    # Convert percentile parameters if passed as percentages
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
    if 'DAPI' in cell_df.columns:
        available_markers.append('DAPI')

    available_cols = [c for c in base_cols if c in cell_df.columns]
    all_cols = available_cols + available_markers

    cell_df = cell_df[[c for c in all_cols if c in cell_df.columns]].copy()

    # Quality filtering
    logger.info("Applying quality filters")
    if 'DAPI' in cell_df.columns:
        nuc_thres = np.percentile(cell_df['DAPI'], quality_percentile)
        size_thres = np.percentile(cell_df['area'], quality_percentile)
        cell_df_filtered = cell_df[
            (cell_df['DAPI'] > nuc_thres) & (cell_df['area'] > size_thres)
        ].copy()
    else:
        size_thres = np.percentile(cell_df['area'], quality_percentile)
        cell_df_filtered = cell_df[cell_df['area'] > size_thres].copy()

    logger.info(f"Cells after quality filter: {len(cell_df_filtered)}")

    # Normalization
    logger.info("Normalizing marker intensities")
    list_out = ['eccentricity', 'perimeter', 'convex_area',
                'axis_major_length', 'axis_minor_length']
    list_keep = ['DAPI', 'x', 'y', 'area', 'label']

    dfin = cell_df_filtered.drop(
        [c for c in list_out if c in cell_df_filtered.columns],
        axis=1
    )
    df_loc = dfin[[c for c in list_keep if c in dfin.columns]]
    dfz = dfin.drop([c for c in list_keep if c in dfin.columns], axis=1)

    # Z-score normalization
    if len(dfz.columns) > 0:
        dfz1 = pd.DataFrame(
            zscore(dfz, axis=0),
            index=dfz.index,
            columns=dfz.columns
        )
        dfz_all = pd.concat([dfz1, df_loc], axis=1, join="inner")
    else:
        dfz_all = df_loc.copy()

    # Noise removal
    logger.info("Removing noise")
    last_marker = 'VIMENTIN' if 'VIMENTIN' in dfz_all.columns else (
        available_markers[-1] if available_markers else None
    )
    
    df_nn = dfz_all.copy()
    
    if last_marker and last_marker in dfz_all.columns:
        col_num_last_marker = dfz_all.columns.get_loc(last_marker)
        df_nn["Count"] = dfz_all.iloc[:, :col_num_last_marker + 1].ge(0).sum(axis=1)
        df_nn["z_sum"] = dfz_all.iloc[:, :col_num_last_marker + 1].sum(axis=1)

    # Phenotyping based on marker expression
    logger.info("Assigning phenotypes")

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
    phenotype_mapping = {0: "Background"}
    for pp, p in enumerate(pheno_complete, start=1):
        sel = df_nn[df_nn['phenotype'] == p].index
        df_nn.loc[sel, 'phenotype_num'] = pp
        phenotype_mapping[pp] = p

    df_nn['phenotype_num'] = df_nn['phenotype_num'].fillna(0).astype(int)

    logger.info(f"Phenotype distribution:\n{df_nn['phenotype'].value_counts()}")
    logger.info(f"Phenotype mapping: {phenotype_mapping}")

    # Convert pheno_markers list to string for CSV export
    df_nn['pheno_markers_str'] = df_nn['pheno_markers'].apply(lambda x: ','.join(x) if x else '')

    return df_nn, phenotype_mapping


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell phenotyping with QuPath GeoJSON export',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--cell_data',
        required=True,
        help='Path to input CSV file with cell data (from regionprops)'
    )
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help='Output directory'
    )

    # Pixel size for coordinate conversion
    parser.add_argument(
        '--pixel_size',
        type=float,
        default=0.325,
        help='Pixel size in micrometers (for GeoJSON coordinate conversion)'
    )

    # Column name overrides
    parser.add_argument(
        '--x_col',
        default='x',
        help='Column name for X centroid (pixels)'
    )
    parser.add_argument(
        '--y_col',
        default='y',
        help='Column name for Y centroid (pixels)'
    )
    parser.add_argument(
        '--cell_id_col',
        default='label',
        help='Column name for cell ID'
    )

    # Phenotyping parameters
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

    # Output options
    parser.add_argument(
        '--include_unknown',
        action='store_true',
        help='Include Unknown phenotype cells in GeoJSON output'
    )
    parser.add_argument(
        '--output_prefix',
        default='phenotypes',
        help='Prefix for output files'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging(level=logging.INFO)
    args = parse_args()

    ensure_dir(args.output_dir)

    logger.info("=" * 80)
    logger.info("Cell Phenotyping Pipeline with QuPath Export")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading cell data: {args.cell_data}")
    cell_df = pd.read_csv(args.cell_data)
    
    # Remove duplicates by label if present
    if args.cell_id_col in cell_df.columns:
        cell_df = cell_df.drop_duplicates(subset=args.cell_id_col, keep='first')
    
    logger.info(f"Loaded {len(cell_df)} cells")

    # Run phenotyping
    phenotypes_df, phenotype_mapping = run_phenotyping_pipeline(
        cell_df,
        markers=args.markers,
        cutoffs=args.cutoffs,
        quality_percentile=args.quality_percentile,
        noise_percentile=args.noise_percentile
    )

    # Define output paths
    output_csv = os.path.join(args.output_dir, f'{args.output_prefix}_data.csv')
    output_geojson = os.path.join(args.output_dir, f'{args.output_prefix}.geojson')
    output_classifications = os.path.join(args.output_dir, f'{args.output_prefix}.classifications.json')
    output_mapping = os.path.join(args.output_dir, f'{args.output_prefix}_mapping.json')

    # Save CSV (convert pheno_markers list to string for CSV compatibility)
    logger.info(f"Saving phenotype data: {output_csv}")
    csv_df = phenotypes_df.copy()
    if 'pheno_markers' in csv_df.columns:
        csv_df = csv_df.drop(columns=['pheno_markers'])
    csv_df.to_csv(output_csv, index=False)

    # Export GeoJSON for QuPath
    logger.info(f"Exporting GeoJSON: {output_geojson}")
    num_exported, phenotype_colors = export_to_geojson(
        df=phenotypes_df,
        output_path=output_geojson,
        pixel_size=args.pixel_size,
        x_col=args.x_col,
        y_col=args.y_col,
        phenotype_col='phenotype',
        cell_id_col=args.cell_id_col,
        exclude_background=not args.include_unknown,
    )

    # Export classifications JSON (for QuPath color setup)
    logger.info(f"Saving classifications: {output_classifications}")
    export_classifications(
        phenotype_colors=phenotype_colors,
        output_path=output_classifications,
        exclude_background=not args.include_unknown
    )

    # Save phenotype mapping
    logger.info(f"Saving phenotype mapping: {output_mapping}")
    with open(output_mapping, 'w') as f:
        json.dump(phenotype_mapping, f, indent=2)

    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Output files:")
    logger.info(f"  CSV:             {output_csv}")
    logger.info(f"  GeoJSON:         {output_geojson} ({num_exported} cells)")
    logger.info(f"  Classifications: {output_classifications}")
    logger.info(f"  Mapping:         {output_mapping}")
    logger.info("")
    logger.info("To view in QuPath:")
    logger.info("  1. Open your pyramidal OME-TIFF image")
    logger.info("  2. Run the import_phenotypes_simple.groovy script")
    logger.info(f"  3. Select: {output_geojson}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())