#!/usr/bin/env python3
"""
Pixie Cell Clustering - Automated version of notebook workflow.

Based on: https://github.com/angelolab/pixie/blob/main/templates/2_Pixie_Cluster_Cells.ipynb

This script performs cell-level clustering by aggregating pixel clusters
and using Self-Organizing Maps (SOM) followed by consensus clustering.

Steps:
1. Create cell-to-pixel-cluster (c2pc) data
2. Compute weighted channel expression per cell
3. Train cell SOM
4. Assign cells to SOM clusters
5. Perform consensus clustering for meta-clusters
6. Generate cluster mappings and profiles
7. Update cell table with cluster assignments
8. Adjust coordinates for tiled inputs (if applicable)
9. Export QuPath-compatible GeoJSON and classifications

Supports tiled inputs from pixie_pixel_cluster.py with automatic coordinate
reconstruction to original image space.

Author: ATEIA Pipeline (adapted from ark-analysis)
"""

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.tiling import load_tile_positions, TileInfo


# =============================================================================
# Tiled c2pc Data Creation (for images split into tiles)
# =============================================================================

def create_c2pc_data_tiled(
    fovs: List[str],
    pixel_data_path: str,
    cell_table_path: str,
    pixel_cluster_col: str = 'pixel_meta_cluster_rename'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create cell-to-pixel-cluster data by aggregating across all tiles.

    Unlike the original `cell_cluster_utils.create_c2pc_data` which filters
    cells by FOV, this function aggregates pixel cluster counts by global
    cell label across ALL tiles. This preserves boundary cell pixels that
    span multiple tiles.

    Parameters
    ----------
    fovs : List[str]
        List of FOV/tile names (e.g., ['sample_tile_0_0', 'sample_tile_0_1', ...])
    pixel_data_path : str
        Path to directory containing per-FOV feather files with pixel cluster data.
    cell_table_path : str
        Path to cell table CSV containing 'fov', 'label', and 'cell_size' columns.
    pixel_cluster_col : str
        Name of pixel cluster column (e.g., 'pixel_meta_cluster_rename')

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (cluster_counts, cluster_counts_size_norm)
        - cluster_counts: DataFrame with fov, label, cell_size, + cluster count columns
        - cluster_counts_size_norm: Same but counts normalized by cell_size

    Notes
    -----
    This function is Pixie-compatible and returns the same DataFrame format as
    `cell_cluster_utils.create_c2pc_data`. Downstream functions like `train_cell_som`,
    `cluster_cells`, and `cell_consensus_cluster` work unchanged.

    The key difference is that pixel counts are summed across ALL tiles for each
    cell label, rather than filtering by FOV. This ensures boundary cells get
    100% of their pixels counted.
    """
    import feather
    from collections import defaultdict

    # Read cell table
    cell_table = pd.read_csv(cell_table_path)

    # Verify required columns exist
    required_cols = ['fov', 'label', 'cell_size']
    missing_cols = [col for col in required_cols if col not in cell_table.columns]
    if missing_cols:
        raise ValueError(f"Cell table missing required columns: {missing_cols}")

    cell_table['label'] = cell_table['label'].astype(int)

    # Accumulate pixel cluster counts per cell (by global label)
    # Using defaultdict of defaultdict for easy accumulation
    cell_cluster_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_cluster_cols = set()

    print(f"    Processing {len(fovs)} tiles...")
    tiles_processed = 0

    for fov in fovs:
        fov_path = os.path.join(pixel_data_path, f"{fov}.feather")
        if not os.path.exists(fov_path):
            print(f"    Warning: Tile data not found: {fov_path}")
            continue

        pixel_data = feather.read_dataframe(fov_path)

        # Handle column name variations
        if 'segmentation_label' in pixel_data.columns:
            pixel_data.rename(columns={'segmentation_label': 'label'}, inplace=True)

        if pixel_cluster_col not in pixel_data.columns:
            print(f"    Warning: {pixel_cluster_col} not in {fov}.feather, skipping")
            continue

        # Group by label and cluster, count pixels
        counts = pixel_data.groupby(['label', pixel_cluster_col]).size()

        # Accumulate counts across tiles (sums boundary cell pixels)
        for (label, cluster), count in counts.items():
            col_name = f"{pixel_cluster_col}_{int(cluster)}"
            cell_cluster_counts[int(label)][col_name] += int(count)
            all_cluster_cols.add(col_name)

        tiles_processed += 1

    print(f"    Processed {tiles_processed} tiles with pixel data")

    if not cell_cluster_counts:
        raise ValueError("No pixel cluster data found in any tile")

    # Convert nested dict to DataFrame
    counts_df = pd.DataFrame.from_dict(cell_cluster_counts, orient='index')
    counts_df = counts_df.fillna(0).astype(int)
    counts_df.index.name = 'label'
    counts_df = counts_df.reset_index()

    # Get valid labels (exist in both cell_table and pixel data)
    cell_table_labels = set(cell_table['label'])
    pixel_data_labels = set(counts_df['label'])
    valid_labels = cell_table_labels.intersection(pixel_data_labels)

    if len(valid_labels) == 0:
        raise ValueError("No matching cell labels between cell_table and pixel data")

    missing_from_pixels = len(cell_table_labels - pixel_data_labels)
    missing_from_cells = len(pixel_data_labels - cell_table_labels)

    if missing_from_pixels > 0:
        print(f"    Note: {missing_from_pixels} cells in table have no pixel data (small cells)")
    if missing_from_cells > 0:
        print(f"    Note: {missing_from_cells} pixel labels not in cell table (filtered out)")

    # Subset cell table to valid labels and select required columns
    cell_subset = cell_table[cell_table['label'].isin(valid_labels)][
        ['fov', 'label', 'cell_size']
    ].copy()

    # Subset counts to valid labels
    counts_df = counts_df[counts_df['label'].isin(valid_labels)]

    # Merge cell info with cluster counts
    cluster_counts = cell_subset.merge(counts_df, on='label', how='left').fillna(0)

    # Ensure cluster columns are int
    cluster_cols = sorted([c for c in cluster_counts.columns if pixel_cluster_col in c])
    for col in cluster_cols:
        cluster_counts[col] = cluster_counts[col].astype(int)

    # Create size-normalized version (required by Pixie)
    cluster_counts_norm = cluster_counts.copy()
    cluster_counts_norm[cluster_cols] = cluster_counts_norm[cluster_cols].div(
        cluster_counts_norm['cell_size'], axis=0
    )
    # Handle division by zero: cells with cell_size=0 produce inf/NaN
    nan_inf_mask = (
        cluster_counts_norm[cluster_cols].isin([np.inf, -np.inf]).any(axis=1)
        | cluster_counts_norm[cluster_cols].isna().any(axis=1)
    )
    if nan_inf_mask.any():
        print(f"    Warning: {nan_inf_mask.sum()} cells have inf/NaN after size normalization (cell_size=0). Replacing with 0.")
    cluster_counts_norm[cluster_cols] = cluster_counts_norm[cluster_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # Sort columns to match Pixie's expected order: fov, label, cell_size, then cluster cols
    meta_cols = ['fov', 'label', 'cell_size']
    final_cols = meta_cols + cluster_cols
    cluster_counts = cluster_counts[final_cols]
    cluster_counts_norm = cluster_counts_norm[final_cols]

    # Reset index for consistency with original function
    cluster_counts = cluster_counts.reset_index(drop=True)
    cluster_counts_norm = cluster_counts_norm.reset_index(drop=True)

    print(f"    Created c2pc data: {len(cluster_counts)} cells, {len(cluster_cols)} cluster columns")

    return cluster_counts, cluster_counts_norm


# =============================================================================
# QuPath Export Functions
# =============================================================================

def generate_cluster_colors(n_clusters: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate visually distinct colors for clusters using HSV color space.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to generate colors for.

    Returns
    -------
    Dict[int, Tuple[int, int, int]]
        Mapping of cluster ID to RGB color tuple.
    """
    colors = {}
    for i in range(n_clusters):
        # Use golden ratio to distribute hues evenly
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.85 + (i % 2) * 0.1  # Vary brightness slightly

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i + 1] = (int(r * 255), int(g * 255), int(b * 255))

    return colors


def export_to_geojson(
    df: pd.DataFrame,
    output_path: str,
    cluster_colors: Dict[int, Tuple[int, int, int]],
    cluster_mapping: Dict[int, str],
    pixel_size: float = 0.325,
    x_col: str = 'x',
    y_col: str = 'y',
    cluster_col: str = 'cell_meta_cluster',
    cell_id_col: str = 'label',
) -> int:
    """Export clustered cells to QuPath-compatible GeoJSON.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with coordinates and cluster assignments.
    output_path : str
        Path for output GeoJSON file.
    cluster_colors : dict
        Mapping of cluster ID to RGB color tuple.
    cluster_mapping : dict
        Mapping of cluster ID to cluster name.
    pixel_size : float
        Micrometers per pixel for coordinate conversion.
    x_col : str
        Column name for X centroid (in pixels).
    y_col : str
        Column name for Y centroid (in pixels).
    cluster_col : str
        Column name for cluster assignment.
    cell_id_col : str
        Column name for cell ID/label.

    Returns
    -------
    int
        Number of cells exported.
    """
    features = []

    for _, row in df.iterrows():
        # Skip cells without cluster assignment
        if pd.isna(row.get(cluster_col)):
            continue

        cluster_id = int(row[cluster_col])
        cluster_name = cluster_mapping.get(cluster_id, f'Cell_Type_{cluster_id}')
        color = cluster_colors.get(cluster_id, (128, 128, 128))

        # Get coordinates (in pixels - QuPath expects pixel coordinates)
        x_px = float(row[x_col])
        y_px = float(row[y_col])

        # Create point geometry (cell centroid)
        # For full cell outlines, would need segmentation contours
        feature = {
            "type": "Feature",
            "id": f"cell_{int(row[cell_id_col])}",
            "geometry": {
                "type": "Point",
                "coordinates": [x_px, y_px]
            },
            "properties": {
                "objectType": "detection",
                "classification": {
                    "name": cluster_name,
                    "color": color
                },
                "measurements": {
                    "Cell ID": int(row[cell_id_col]),
                    "Cluster ID": cluster_id,
                    "Centroid X (px)": float(row[x_col]),
                    "Centroid Y (px)": float(row[y_col]),
                }
            }
        }

        # Add SOM cluster if available
        if 'cell_som_cluster' in row.index and not pd.isna(row['cell_som_cluster']):
            feature["properties"]["measurements"]["SOM Cluster"] = int(row['cell_som_cluster'])

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f)

    return len(features)


def export_classifications(
    cluster_colors: Dict[int, Tuple[int, int, int]],
    cluster_mapping: Dict[int, str],
    output_path: str,
) -> None:
    """Export cluster classifications JSON for QuPath color setup.

    Parameters
    ----------
    cluster_colors : dict
        Mapping of cluster ID to RGB color tuple.
    cluster_mapping : dict
        Mapping of cluster ID to cluster name.
    output_path : str
        Output path for classifications JSON.
    """
    classifications = []

    for cluster_id, cluster_name in sorted(cluster_mapping.items()):
        color = cluster_colors.get(cluster_id, (128, 128, 128))

        # QuPath classification format (matches phenotyping.py format)
        classification = {
            "name": cluster_name,
            "rgb": list(color),
            "colorRGB": (color[0] << 16) | (color[1] << 8) | color[2]
        }
        classifications.append(classification)

    with open(output_path, 'w') as f:
        json.dump(classifications, f, indent=2)


def export_cluster_mapping_json(
    cluster_mapping: Dict[int, str],
    cluster_colors: Dict[int, Tuple[int, int, int]],
    output_path: str,
) -> None:
    """Export cluster mapping with colors as JSON.

    Parameters
    ----------
    cluster_mapping : dict
        Mapping of cluster ID to cluster name.
    cluster_colors : dict
        Mapping of cluster ID to RGB color tuple.
    output_path : str
        Output path for mapping JSON.
    """
    mapping = {}
    for cluster_id, cluster_name in sorted(cluster_mapping.items()):
        color = cluster_colors.get(cluster_id, (128, 128, 128))
        mapping[cluster_name] = {
            "cluster_id": cluster_id,
            "color_rgb": list(color),
            "color_hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        }

    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Pixie Cell Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--base_dir', required=True,
                        help='Base directory for processing')
    parser.add_argument('--pixel_output_dir', required=True,
                        help='Directory containing pixel clustering outputs')
    parser.add_argument('--cell_table_path', required=True,
                        help='Path to cell table CSV (merged quantification)')
    parser.add_argument('--cell_params_path', required=True,
                        help='Path to cell_clustering_params.json from pixel clustering')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for cell clustering results')
    parser.add_argument('--pixel_cluster_col', default='pixel_meta_cluster_rename',
                        help='Pixel cluster column to use for cell clustering')
    parser.add_argument('--max_k', type=int, default=20,
                        help='Maximum number of cell meta-clusters')
    parser.add_argument('--cap', type=int, default=3,
                        help='Z-score capping value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--pixel_size', type=float, default=0.325,
                        help='Pixel size in micrometers for QuPath export')
    args = parser.parse_args()

    # Import ark modules (delayed import to check availability)
    try:
        import feather
        from ark.phenotyping import (
            cell_cluster_utils,
            cell_meta_clustering,
            cell_som_clustering,
            weighted_channel_comp
        )
    except ImportError as e:
        print(f"ERROR: Failed to import ark-analysis modules: {e}", file=sys.stderr)
        print("Please ensure ark-analysis==0.6.4 is installed.", file=sys.stderr)
        sys.exit(1)

    # Load pixel clustering parameters
    print("Loading pixel clustering parameters...")
    with open(args.cell_params_path) as f:
        cell_params = json.load(f)

    fovs = cell_params['fovs']
    channels = cell_params['channels']
    pixel_data_dir = cell_params['pixel_data_dir']
    pc_chan_avg_meta_cluster_name = cell_params['pc_chan_avg_meta_cluster_name']

    # Check for tiling information
    is_tiled = cell_params.get('is_tiled', False)
    original_fov = cell_params.get('original_fov', fovs[0] if fovs else None)
    tile_positions_path = cell_params.get('tile_positions_path')
    tile_positions: Optional[Dict[str, TileInfo]] = None

    if is_tiled and tile_positions_path:
        # Check multiple locations for tile positions file (priority: staged in cwd first)
        tile_pos_basename = os.path.basename(tile_positions_path)
        tile_pos_candidates = [
            tile_pos_basename,  # Current directory (staged by Nextflow)
            tile_positions_path,  # Original path from JSON
            os.path.join(args.base_dir, tile_pos_basename),  # base_dir
        ]

        actual_tile_pos_path = None
        for path in tile_pos_candidates:
            if os.path.exists(path):
                actual_tile_pos_path = path
                break

        if actual_tile_pos_path:
            print(f"Loading tile positions from: {actual_tile_pos_path}")
            tile_positions, tile_metadata = load_tile_positions(Path(actual_tile_pos_path))
            print(f"  Loaded {len(tile_positions)} tile positions")
            print(f"  Original image: {tile_metadata['original_width']}x{tile_metadata['original_height']}")
        else:
            print(f"  Warning: Tile positions file not found, coordinate adjustment will be skipped")
            print(f"  Searched: {tile_pos_candidates}")

    base_dir = args.base_dir
    cell_output_dir = args.output_dir
    os.makedirs(os.path.join(base_dir, cell_output_dir), exist_ok=True)

    print(f"Pixie Cell Clustering")
    print(f"=" * 50)
    print(f"FOVs: {', '.join(fovs[:5])}{'...' if len(fovs) > 5 else ''}")
    print(f"Channels: {', '.join(channels)}")
    print(f"Pixel cluster column: {args.pixel_cluster_col}")
    print(f"Max K: {args.max_k}")
    print(f"Cap: {args.cap}")
    print(f"Seed: {args.seed}")
    if is_tiled:
        print(f"Tiled input: {len(fovs)} tiles from {original_fov}")
    print()

    # Output file names (matching notebook structure)
    cluster_counts_name = os.path.join(cell_output_dir, 'cluster_counts.feather')
    cluster_counts_size_norm_name = os.path.join(cell_output_dir, 'cluster_counts_size_norm.feather')
    weighted_cell_channel_name = os.path.join(cell_output_dir, 'weighted_cell_channel.feather')
    cell_som_weights_name = os.path.join(cell_output_dir, 'cell_som_weights.feather')
    cell_som_cluster_count_avg_name = os.path.join(cell_output_dir, 'cell_som_cluster_count_avg.csv')
    cell_meta_cluster_count_avg_name = os.path.join(cell_output_dir, 'cell_meta_cluster_count_avg.csv')
    cell_som_cluster_channel_avg_name = os.path.join(cell_output_dir, 'cell_som_cluster_channel_avg.csv')
    cell_meta_cluster_channel_avg_name = os.path.join(cell_output_dir, 'cell_meta_cluster_channel_avg.csv')
    cell_meta_cluster_remap_name = os.path.join(cell_output_dir, 'cell_meta_cluster_mapping.csv')

    pixel_cluster_col = args.pixel_cluster_col

    # =========================================================================
    # Step 1: Create cell-to-pixel-cluster (c2pc) data
    # =========================================================================
    print("Step 1: Creating cell-to-pixel-cluster data...")

    # The pixel_data_dir from JSON is the original relative path from pixie_pixel_cluster.py
    # But Nextflow stages it differently, so use args.pixel_output_dir directly
    pixel_data_path = args.pixel_output_dir
    if not os.path.isabs(pixel_data_path):
        pixel_data_path = os.path.join(base_dir, args.pixel_output_dir)

    # Fallback to JSON path only if the direct path doesn't exist
    if not os.path.exists(pixel_data_path):
        fallback_path = os.path.join(base_dir, pixel_data_dir)
        if os.path.exists(fallback_path):
            pixel_data_path = fallback_path

    print(f"  Pixel data path: {pixel_data_path}")

    # Use tiled aggregation for tiled inputs, standard Pixie for single FOV
    if is_tiled and len(fovs) > 1:
        print(f"  Using tiled aggregation (preserves boundary cell pixels)...")
        cluster_counts, cluster_counts_size_norm = create_c2pc_data_tiled(
            fovs,
            pixel_data_path,
            args.cell_table_path,
            pixel_cluster_col
        )
    else:
        # Standard Pixie for non-tiled (single FOV) inputs
        cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
            fovs,
            pixel_data_path,
            args.cell_table_path,
            pixel_cluster_col
        )

    feather.write_dataframe(
        cluster_counts,
        os.path.join(base_dir, cluster_counts_name),
        compression='uncompressed'
    )
    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    cell_som_cluster_cols = cluster_counts_size_norm.filter(
        regex=f'{pixel_cluster_col}.*'
    ).columns.values

    print(f"  Created c2pc data with {len(cell_som_cluster_cols)} cluster columns.")
    print(f"  Cells: {len(cluster_counts_size_norm)}")

    # Filter degenerate cells (cell_size <= 0) before clustering
    _degenerate = cluster_counts_size_norm['cell_size'] <= 0
    if _degenerate.any():
        print(f"  Filtering {_degenerate.sum()} cells with cell_size <= 0 (degenerate, no pixel data)")
        cluster_counts_size_norm = cluster_counts_size_norm[~_degenerate].reset_index(drop=True)
        cluster_counts = cluster_counts[~_degenerate].reset_index(drop=True)
        # Re-save filtered data
        feather.write_dataframe(cluster_counts, os.path.join(base_dir, cluster_counts_name), compression='uncompressed')
        feather.write_dataframe(cluster_counts_size_norm, os.path.join(base_dir, cluster_counts_size_norm_name), compression='uncompressed')

    # Safety net: clean any remaining NaN/inf in feature columns
    _bad = (
        cluster_counts_size_norm[cell_som_cluster_cols].isin([np.inf, -np.inf]).any(axis=1)
        | cluster_counts_size_norm[cell_som_cluster_cols].isna().any(axis=1)
    )
    if _bad.any():
        print(f"  Warning: {_bad.sum()} cells still have NaN/inf in cluster columns. Replacing with 0.")
        cluster_counts_size_norm[cell_som_cluster_cols] = cluster_counts_size_norm[cell_som_cluster_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    # =========================================================================
    # Step 2: Compute weighted channel expression per cell
    # =========================================================================
    print("Step 2: Computing weighted channel expression...")

    # Determine which channel average file to use
    # Priority: 1) current directory (staged by Nextflow), 2) base_dir + path from JSON,
    # 3) pixel_output_dir + basename
    pc_chan_avg_basename = os.path.basename(pc_chan_avg_meta_cluster_name)
    search_paths = [
        pc_chan_avg_basename,  # Current directory (staged by Nextflow)
        os.path.join(base_dir, pc_chan_avg_basename),  # base_dir
        os.path.join(base_dir, pc_chan_avg_meta_cluster_name),  # Full path from JSON
        os.path.join(args.pixel_output_dir, pc_chan_avg_basename),  # pixel_output_dir
    ]

    pc_chan_avg_path = None
    for path in search_paths:
        if os.path.exists(path):
            pc_chan_avg_path = path
            break

    if pc_chan_avg_path is None:
        raise FileNotFoundError(
            f"Cannot find pixel channel average file '{pc_chan_avg_basename}'. "
            f"Searched paths:\n  - " + "\n  - ".join(search_paths)
        )

    print(f"  Using pixel channel averages: {pc_chan_avg_path}")
    pixel_channel_avg = pd.read_csv(pc_chan_avg_path)
    # For tiled workflows, use FOVs from cluster_counts (original FOV names from cell table)
    # not the tile FOV names. The cluster_counts already aggregates across all tiles.
    cell_fovs = cluster_counts['fov'].unique().tolist() if is_tiled else fovs
    weighted_cell_channel = weighted_channel_comp.compute_p2c_weighted_channel_avg(
        pixel_channel_avg=pixel_channel_avg,
        channels=channels,
        cell_counts=cluster_counts,
        fovs=cell_fovs,
        pixel_cluster_col=pixel_cluster_col
    )
    # Clean NaN/inf from weighted channel expression (division by cell_size)
    _wc_cols = [c for c in weighted_cell_channel.columns if c in channels]
    if weighted_cell_channel[_wc_cols].isna().any().any() or weighted_cell_channel[_wc_cols].isin([np.inf, -np.inf]).any().any():
        print("  Warning: Weighted channel expression contains NaN/inf. Replacing with 0.")
        weighted_cell_channel[_wc_cols] = weighted_cell_channel[_wc_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
    feather.write_dataframe(
        weighted_cell_channel,
        os.path.join(base_dir, weighted_cell_channel_name),
        compression='uncompressed'
    )
    print("  Weighted channel expression computed.")

    # =========================================================================
    # Step 3: Train cell SOM
    # =========================================================================
    print("Step 3: Training cell SOM...")
    cell_pysom = cell_som_clustering.train_cell_som(
        fovs=cell_fovs,
        base_dir=base_dir,
        cell_table_path=args.cell_table_path,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        som_weights_name=cell_som_weights_name,
        num_passes=1,
        seed=args.seed
    )
    print("  Cell SOM training complete.")

    # =========================================================================
    # Step 4: Assign cells to SOM clusters
    # =========================================================================
    print("Step 4: Assigning cells to SOM clusters...")
    cluster_counts_size_norm = cell_som_clustering.cluster_cells(
        base_dir=base_dir,
        cell_pysom=cell_pysom,
        cell_som_cluster_cols=cell_som_cluster_cols
    )

    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    cell_som_clustering.generate_som_avg_files(
        base_dir=base_dir,
        cell_som_input_data=cluster_counts_size_norm,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name
    )
    print("  Cell SOM cluster assignment complete.")

    # Safety net: clean NaN/inf in SOM cluster average file before consensus clustering
    _som_avg_path = os.path.join(base_dir, cell_som_cluster_count_avg_name)
    _som_avg = pd.read_csv(_som_avg_path)
    _som_cols = [c for c in _som_avg.columns if c in cell_som_cluster_cols]
    if _som_avg[_som_cols].isna().any().any() or _som_avg[_som_cols].isin([np.inf, -np.inf]).any().any():
        print("  Warning: SOM cluster average file contains NaN/inf. Cleaning before consensus clustering.")
        _som_avg[_som_cols] = _som_avg[_som_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        _som_avg.to_csv(_som_avg_path, index=False)

    # Safety net: clean NaN/inf in per-cell data before consensus clustering
    _cell_cols = [c for c in cluster_counts_size_norm.columns if c in cell_som_cluster_cols]
    if cluster_counts_size_norm[_cell_cols].isna().any().any() or cluster_counts_size_norm[_cell_cols].isin([np.inf, -np.inf]).any().any():
        print("  Warning: Per-cell cluster data contains NaN/inf. Cleaning before consensus clustering.")
        cluster_counts_size_norm[_cell_cols] = cluster_counts_size_norm[_cell_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    # =========================================================================
    # Step 5: Consensus clustering for meta-clusters
    # =========================================================================
    print("Step 5: Running consensus clustering...")
    cell_cc, cluster_counts_size_norm = cell_meta_clustering.cell_consensus_cluster(
        base_dir=base_dir,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name,
        max_k=args.max_k,
        cap=args.cap,
        seed=args.seed
    )

    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    cell_meta_clustering.generate_meta_avg_files(
        base_dir=base_dir,
        cell_cc=cell_cc,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name,
        cell_meta_expr_col_avg_name=cell_meta_cluster_count_avg_name
    )

    weighted_channel_comp.generate_wc_avg_files(
        fovs=cell_fovs,
        channels=channels,
        base_dir=base_dir,
        cell_cc=cell_cc,
        cell_som_input_data=cluster_counts_size_norm,
        weighted_cell_channel_name=weighted_cell_channel_name,
        cell_som_cluster_channel_avg_name=cell_som_cluster_channel_avg_name,
        cell_meta_cluster_channel_avg_name=cell_meta_cluster_channel_avg_name
    )
    print("  Consensus clustering complete.")

    # =========================================================================
    # Step 6: Create default cluster mapping (replacing interactive GUI)
    # =========================================================================
    print("Step 6: Creating cluster mapping...")

    # Get unique meta-clusters
    if 'cell_meta_cluster' in cluster_counts_size_norm.columns:
        unique_meta = sorted(cluster_counts_size_norm['cell_meta_cluster'].dropna().unique())
    else:
        # Fallback if column doesn't exist
        unique_meta = list(range(1, args.max_k + 1))

    # Create default mapping: cluster_id -> "Cell_Type_N"
    mapping_df = pd.DataFrame({
        'cluster_id': unique_meta,
        'cluster_name': [f'Cell_Type_{int(i)}' for i in unique_meta]
    })
    mapping_df.to_csv(os.path.join(base_dir, cell_meta_cluster_remap_name), index=False)
    print(f"  Created mapping for {len(unique_meta)} cell meta-clusters.")
    print(f"  To rename clusters, edit: {cell_meta_cluster_remap_name}")

    # =========================================================================
    # Step 7: Append cluster labels to cell table
    # =========================================================================
    print("Step 7: Updating cell table with cluster assignments...")
    cell_table = pd.read_csv(args.cell_table_path)

    # Determine which columns to merge
    cluster_cols_to_merge = ['label']
    if 'cell_som_cluster' in cluster_counts_size_norm.columns:
        cluster_cols_to_merge.append('cell_som_cluster')
    if 'cell_meta_cluster' in cluster_counts_size_norm.columns:
        cluster_cols_to_merge.append('cell_meta_cluster')
    if 'fov' in cluster_counts_size_norm.columns:
        cluster_cols_to_merge.insert(0, 'fov')

    # Merge cluster assignments to cell table
    merge_data = cluster_counts_size_norm[cluster_cols_to_merge].copy()

    # Handle label column type matching
    if 'label' in cell_table.columns and 'label' in merge_data.columns:
        # Ensure types match
        cell_table['label'] = cell_table['label'].astype(int)
        merge_data['label'] = merge_data['label'].astype(int)

    cell_table_clustered = cell_table.merge(
        merge_data,
        on='label',
        how='left'
    )

    # =========================================================================
    # Step 7b: Adjust coordinates for tiled inputs
    # =========================================================================
    if tile_positions is not None and 'fov' in cell_table_clustered.columns:
        print("Step 7b: Adjusting coordinates for tiled input...")

        # Adjust x and y coordinates based on tile position (vectorized)
        if 'x' in cell_table_clustered.columns and 'y' in cell_table_clustered.columns:
            adjusted_count = 0
            for fov_name, tile_info in tile_positions.items():
                mask = cell_table_clustered['fov'] == fov_name
                n_cells = mask.sum()
                if n_cells > 0:
                    cell_table_clustered.loc[mask, 'x'] += tile_info.x_start
                    cell_table_clustered.loc[mask, 'y'] += tile_info.y_start
                    adjusted_count += n_cells

            print(f"  Adjusted coordinates for {adjusted_count} cells")
        else:
            print("  Warning: x/y columns not found, skipping coordinate adjustment")

    # Save updated cell table
    output_path = os.path.join(base_dir, cell_output_dir, 'cell_table_with_clusters.csv')
    cell_table_clustered.to_csv(output_path, index=False)

    # =========================================================================
    # Step 8: Export QuPath-compatible outputs
    # =========================================================================
    print("Step 8: Exporting QuPath-compatible outputs...")

    # Generate cluster colors
    cluster_colors = generate_cluster_colors(len(unique_meta))

    # Create cluster ID to name mapping
    cluster_mapping = {int(row['cluster_id']): row['cluster_name']
                       for _, row in mapping_df.iterrows()}

    # Export GeoJSON
    geojson_path = os.path.join(base_dir, cell_output_dir, 'pixie_clusters.geojson')
    num_exported = export_to_geojson(
        df=cell_table_clustered,
        output_path=geojson_path,
        cluster_colors=cluster_colors,
        cluster_mapping=cluster_mapping,
        pixel_size=args.pixel_size,
        x_col='x',
        y_col='y',
        cluster_col='cell_meta_cluster',
        cell_id_col='label'
    )
    print(f"  GeoJSON exported: {num_exported} cells")

    # Export classifications JSON for QuPath
    classifications_path = os.path.join(base_dir, cell_output_dir, 'pixie_clusters.classifications.json')
    export_classifications(
        cluster_colors=cluster_colors,
        cluster_mapping=cluster_mapping,
        output_path=classifications_path
    )
    print(f"  Classifications exported: {len(cluster_mapping)} clusters")

    # Export mapping JSON with colors
    mapping_json_path = os.path.join(base_dir, cell_output_dir, 'pixie_clusters_mapping.json')
    export_cluster_mapping_json(
        cluster_mapping=cluster_mapping,
        cluster_colors=cluster_colors,
        output_path=mapping_json_path
    )
    print(f"  Mapping JSON exported")

    # =========================================================================
    # Summary
    # =========================================================================
    n_cells = len(cell_table_clustered)
    n_clusters = len(unique_meta)
    n_assigned = cell_table_clustered['cell_meta_cluster'].notna().sum() if 'cell_meta_cluster' in cell_table_clustered.columns else 0

    print()
    print(f"Summary")
    print(f"-" * 50)
    print(f"  Total cells: {n_cells}")
    print(f"  Cells with cluster assignment: {n_assigned}")
    print(f"  Cell meta-clusters identified: {n_clusters}")
    print(f"  Channels used: {len(channels)}")
    if is_tiled:
        print(f"  FOV tiles processed: {len(fovs)}")
        print(f"  Coordinates adjusted to original image space")
    print()
    print("Output files:")
    print(f"  Cell table:       {output_path}")
    print(f"  GeoJSON:          {geojson_path}")
    print(f"  Classifications:  {classifications_path}")
    print(f"  Mapping:          {mapping_json_path}")
    print()
    print("To view in QuPath:")
    print("  1. Open your pyramidal OME-TIFF image")
    print("  2. File > Import objects from file")
    print(f"  3. Select: {geojson_path}")
    print()
    print("Cell clustering complete!")


if __name__ == '__main__':
    main()
