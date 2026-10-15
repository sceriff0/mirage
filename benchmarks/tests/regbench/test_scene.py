"""The selfcheck's scene generator (``regbench.scene``): seamless, reproducible, the same
slide at any tile size; and owned off the synthetic dataset, as the toy cases are."""

import numpy as np
import tifffile

from benchmarks.regbench import scene

SPEC = {
    **scene._BASE,
    "family": "t",
    "n": 1536,
    "theta": 20.0,
    "shift": (25.0, -40.0),
    "amp": 12.0,
    "wavelength": 500.0,
    "dropout": 0.2,
    "noise": 0.0,
    "seed": 9,
    "mov_hw": (1300, 1400),
}


def test_tiles_join_into_the_slide_a_single_tile_would_render():
    """No seams: with the noise off, 256 px tiles and one whole-slide tile give the same
    pixels, for the reference and for the resampled moving slide (20 degrees, so every
    moving tile reaches into a differently placed patch of the base)."""
    ref_a, mov_a, _ = scene.make_pair(SPEC, tile=256)
    ref_b, mov_b, _ = scene.make_pair(SPEC, tile=4096)
    assert ref_a.shape == (1536, 1536) and mov_a.shape == (1300, 1400)
    assert ref_a.max() > 10000  # there are nuclei
    assert np.abs(ref_a.astype(int) - ref_b.astype(int)).max() <= 1
    assert np.abs(mov_a.astype(int) - mov_b.astype(int)).max() <= 1


def test_streamed_files_equal_the_in_memory_pair_and_do_not_depend_on_workers(tmp_path):
    spec = {**SPEC, "noise": 0.05}
    ref, mov, _ = scene.make_pair(spec)
    for workers in (1, 2):
        r, m = tmp_path / f"r{workers}.ome.tif", tmp_path / f"m{workers}.ome.tif"
        scene.write_pair(spec, r, m, workers=workers)
        assert np.array_equal(tifffile.imread(r).squeeze(), ref)
        assert np.array_equal(
            tifffile.imread(m).squeeze(), mov
        )  # incl. the padded edge tiles


def test_selfcheck_and_the_test_fixture_do_not_import_the_synthetic_dataset():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    for rel in (
        "benchmarks/regbench/selfcheck.py",
        "benchmarks/regbench/scene.py",
        "benchmarks/tests/regbench/conftest.py",
        "benchmarks/tests/regbench/toy_cases.py",
    ):
        tree = ast.parse((root / rel).read_text())
        names = {
            a.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        } | {
            getattr(n, "module", "") or ""
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
        }
        assert not any("synthetic" in x or x == "truth" for x in names), rel


def test_a_toy_case_is_a_multiplex_case_of_its_own_group(tmp_path):
    from benchmarks.regbench.cases import load_case

    from . import toy_cases

    d = toy_cases.prepare_toy(tmp_path, "rigid", toy_cases.SPECS["rigid"])
    c = load_case(d)
    assert (c.dataset, c.group) == ("multiplex", "rigid")
    assert "truth_mov" not in np.load(d / "points.npz").files
