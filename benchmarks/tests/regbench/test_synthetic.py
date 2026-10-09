"""The tile-streamed generator: seamless, reproducible, and the same slide at any tile size."""

import numpy as np
import tifffile
from regbench.datasets import synthetic

SPEC = {**synthetic._BASE, "family": "t", "n": 1536, "theta": 20.0, "shift": (25.0, -40.0),
        "amp": 12.0, "wavelength": 500.0, "dropout": 0.2, "noise": 0.0, "seed": 9,
        "mov_hw": (1300, 1400)}


def test_tiles_join_into_the_slide_a_single_tile_would_render():
    """No seams: with the noise off, 256 px tiles and one whole-slide tile give the same
    pixels, for the reference and for the resampled moving slide (20 degrees, so every
    moving tile reaches into a differently placed patch of the base)."""
    ref_a, mov_a, _ = synthetic.make_pair(SPEC, tile=256)
    ref_b, mov_b, _ = synthetic.make_pair(SPEC, tile=4096)
    assert ref_a.shape == (1536, 1536) and mov_a.shape == (1300, 1400)
    assert ref_a.max() > 10000  # there are nuclei
    assert np.abs(ref_a.astype(int) - ref_b.astype(int)).max() <= 1
    assert np.abs(mov_a.astype(int) - mov_b.astype(int)).max() <= 1


def test_streamed_files_equal_the_in_memory_pair_and_do_not_depend_on_workers(tmp_path):
    spec = {**SPEC, "noise": 0.05}
    ref, mov, _ = synthetic.make_pair(spec)
    for workers in (1, 2):
        r, m = tmp_path / f"r{workers}.ome.tif", tmp_path / f"m{workers}.ome.tif"
        synthetic.write_pair(spec, r, m, workers=workers)
        assert np.array_equal(tifffile.imread(r).squeeze(), ref)
        assert np.array_equal(tifffile.imread(m).squeeze(), mov)  # incl. the padded edge tiles


def test_scale_suite_runs_one_deformation_from_1024_to_65536():
    suite = synthetic.SUITES["scale"]
    names = sorted(suite)
    assert [suite[k]["n"] for k in names] == [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    deform = {tuple((k, str(v)) for k, v in s.items() if k not in ("n",)) for s in suite.values()}
    assert len(deform) == 1  # only the size differs
    # a huge slide is described without allocating it
    scene = synthetic.Scene({**suite["scale_65536"], "n": 65536})
    assert scene.mask.shape == (2048, 2048) and len(scene.ref[0]) > 2_000_000
    assert scene.tile("mov", 40000, 50000, 40256, 50256).shape == (256, 256)
