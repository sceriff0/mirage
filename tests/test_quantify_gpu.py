import numpy as np
import tifffile

from scripts.quantify_gpu import import_images, extract_features_gpu


def test_import_images_tifffile(tmp_path):
    arr = np.zeros((4, 4), dtype=np.uint8)
    p = tmp_path / "img.tiff"
    tifffile.imwrite(str(p), arr)
    img, meta = import_images(str(p))
    assert img.shape == arr.shape


def test_extract_features_gpu_no_gpu_returns_empty(monkeypatch):
    # If GPU libs unavailable, gpu_extract_features will raise; ensure extract
    # wrapper handles empty results gracefully when no channels are present.
    res = extract_features_gpu([], np.zeros((4, 4), dtype=np.int32))
    assert res.empty
