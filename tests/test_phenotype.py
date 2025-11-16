import numpy as np
import pandas as pd

from scripts.phenotype import labels_to_phenotype, run_phenotyping_pipeline


def test_labels_to_phenotype_simple():
    arr = np.array([[0, 1], [2, 1]], dtype=np.int32)
    phenotype_df = pd.DataFrame({"label": [1, 2], "phenotype_num": [10, 20]})

    mapped = labels_to_phenotype(arr, phenotype_df)
    assert mapped.shape == arr.shape
    assert mapped[0, 1] == 10
    assert mapped[1, 0] == 20


def test_run_phenotyping_pipeline_minimal():
    # build a minimal cell dataframe with required columns
    cols = [
        'y', 'x', 'eccentricity', 'perimeter', 'convex_area', 'area',
        'axis_major_length', 'axis_minor_length', 'label',
        'ARID1A', 'CD14', 'CD163', 'CD3', 'CD4', 'CD45', 'CD8', 'FOXP3',
        'L1CAM', 'P53', 'PANCK', 'PAX2', 'PD1', 'PDL1', 'SMA', 'GZMB', 'CD74', 'VIMENTIN', 'DAPI'
    ]

    # two cells
    data = [
        [0, 0, 0, 0, 0, 50, 0, 0, 1] + [0]* (len(cols)-9),
        [1, 1, 0, 0, 0, 60, 0, 0, 2] + [0]* (len(cols)-9),
    ]

    df = pd.DataFrame(data, columns=cols)
    mask = np.array([[0, 1], [0, 2]], dtype=np.int32)

    pheno_df, pheno_mask = run_phenotyping_pipeline(df, mask, output_dir='.')
    assert 'phenotype' in pheno_df.columns
    assert pheno_mask.shape == mask.shape
