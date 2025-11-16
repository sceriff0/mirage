import numpy as np

from scripts.segment import remap_labels, normalize_image


def test_remap_labels_consecutive():
    mask = np.array([[0, 5, 5], [0, 10, 10]])
    remapped = remap_labels(mask)
    uniq = np.unique(remapped)
    assert list(uniq) == [0, 1, 2]


def test_normalize_image_basic():
    img = np.linspace(0, 255, 100).reshape(10, 10).astype(float)
    norm = normalize_image(img, pmin=1.0, pmax=99.0)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
