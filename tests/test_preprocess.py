import numpy as np

from preprocess import split_image_into_fovs, reconstruct_image_from_fovs, preprocess_multichannel_image


def test_split_and_reconstruct():
    img = np.arange(16).reshape(4, 4)
    fov_stack, positions, max_fov = split_image_into_fovs(img, (2, 2), overlap=0)
    recon = reconstruct_image_from_fovs(fov_stack, positions, img.shape)
    assert (recon == img).all()


def test_preprocess_multichannel_monkeypatch(tmp_path, monkeypatch):
    # Create two fake channel files
    ch1 = (tmp_path / "ch1.tiff").as_posix()
    ch2 = (tmp_path / "ch2.tiff").as_posix()

    import tifffile
    tifffile.imwrite(ch1, np.zeros((4, 4), dtype=np.uint8))
    tifffile.imwrite(ch2, np.ones((4, 4), dtype=np.uint8))

    # Monkeypatch apply_basic_correction to return input unchanged
    def fake_apply(img, **kwargs):
        return img, None

    monkeypatch.setattr('preprocess.apply_basic_correction', fake_apply)

    # Monkeypatch save_h5 to a no-op
    monkeypatch.setattr('utils.io.save_h5', lambda arr, p: None)

    pre = preprocess_multichannel_image([ch1, ch2], str(tmp_path / 'out.h5'))
    assert pre.shape[0] == 2
