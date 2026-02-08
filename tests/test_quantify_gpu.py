import numpy as np
import pytest

import quantify_gpu


def test_gpu_extract_features_requires_gpu_when_gpu_libs_missing():
    if quantify_gpu.cp is not None and quantify_gpu.gpu_regionprops_table is not None:
        pytest.skip("GPU libraries are available in this environment")

    segmentation_mask = np.array([[0, 1], [0, 2]], dtype=np.int32)
    channel_image = np.array([[0, 10], [0, 20]], dtype=np.uint16)

    with pytest.raises(RuntimeError, match="GPU libraries"):
        quantify_gpu.gpu_extract_features(
            segmentation_mask=segmentation_mask,
            channel_image=channel_image,
            chan_name="DAPI",
        )


def test_extract_features_gpu_propagates_missing_gpu_error():
    if quantify_gpu.cp is not None and quantify_gpu.gpu_regionprops_table is not None:
        pytest.skip("GPU libraries are available in this environment")

    multichannel = np.zeros((1, 4, 4), dtype=np.uint16)
    segmentation_mask = np.zeros((4, 4), dtype=np.int32)

    with pytest.raises(RuntimeError, match="GPU libraries"):
        quantify_gpu.extract_features_gpu(
            multichannel_image=multichannel,
            channel_names=["DAPI"],
            segmentation_mask=segmentation_mask,
            verbose=False,
        )
