import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add bin directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'bin'))

from phenotype import run_phenotyping_pipeline, export_to_geojson


def test_run_phenotyping_pipeline_minimal():
    """Test that the phenotyping pipeline assigns phenotypes to cells."""
    # Build a minimal cell dataframe with required columns
    cols = [
        'y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
        'axis_major_length', 'axis_minor_length', 'label',
        'CD14', 'CD163', 'CD3', 'CD4', 'CD45', 'CD8', 'FOXP3',
        'PANCK', 'PAX2', 'PD1', 'PDL1', 'SMA', 'GZMB', 'CD74', 'VIMENTIN', 'DAPI'
    ]

    # Two cells with different marker profiles
    data = [
        [10, 10, 0.5, 100, 200, 150, 20, 15, 1] + [0.5] * (len(cols) - 9),
        [20, 20, 0.5, 100, 200, 160, 20, 15, 2] + [0.5] * (len(cols) - 9),
    ]

    df = pd.DataFrame(data, columns=cols)

    pheno_df, phenotype_mapping = run_phenotyping_pipeline(df)

    assert 'phenotype' in pheno_df.columns
    assert 'phenotype_num' in pheno_df.columns
    assert len(pheno_df) > 0
    assert isinstance(phenotype_mapping, dict)


def test_export_to_geojson(tmp_path):
    """Test GeoJSON export produces valid QuPath-compatible output."""
    df = pd.DataFrame({
        'x': [100, 200, 300],
        'y': [150, 250, 350],
        'label': [1, 2, 3],
        'phenotype': ['Immune', 'Stroma', 'PANCK+ Tumor'],
        'area': [500, 600, 700],
    })

    output_path = tmp_path / 'test.geojson'
    num_exported, colors = export_to_geojson(
        df,
        str(output_path),
        pixel_size=0.325,
        exclude_background=False
    )

    assert output_path.exists()
    assert num_exported == 3

    # Verify GeoJSON structure
    import json
    with open(output_path) as f:
        geojson = json.load(f)

    assert geojson['type'] == 'FeatureCollection'
    assert len(geojson['features']) == 3

    # Check first feature structure
    feature = geojson['features'][0]
    assert feature['type'] == 'Feature'
    assert feature['geometry']['type'] == 'Point'
    assert feature['properties']['objectType'] == 'detection'
    assert 'classification' in feature['properties']
    assert 'name' in feature['properties']['classification']
