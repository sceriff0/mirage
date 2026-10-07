"""ASHLAR as published (benchmarks/ashlar/original.py) on synthetic raw tiles (retile.py).

The slides exist only stitched, so the raw tiles are synthesised: a regular grid whose
tiles are cut a little off their nominal corner (the stage error ASHLAR's stitching exists
to find) and carry their own sensor noise. The original `ashlar` command then runs
unmodified; these tests pin the two things added around it -- the synthetic tiles and the
scoring manifest -- and, where ashlar itself is installed, the whole chain against a
known shift. ASHLAR is not installed in CI, so that last test skips there.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from benchmarks.ashlar import original, retile


def _slide(path, plane, n_channels=2):
    stack = np.stack([plane * (0.5**c) for c in range(n_channels)]).astype(np.uint16)
    tifffile.imwrite(path, stack, tile=(64, 64), metadata={"axes": "CYX"}, ome=True)
    return path


def _texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(200, 4000, size=shape).astype(np.float32)


# ------------------------------------------------------------------ synthetic tiles --
def test_a_jittered_tile_holds_the_slide_at_its_true_position(tmp_path):
    """A tile's file name says (row, col); its pixels come from nominal + stage error, and
    grid.json records that true corner. Whole pixels: nothing is resampled."""
    plane = _texture((300, 420))
    slide = _slide(tmp_path / "s.ome.tif", plane)
    grid_path = retile.write_tiles(
        slide,
        tmp_path / "t",
        128,
        0.125,
        pixel_size_um=0.5,
        stage_jitter_um=3.0,  # 6 px
        seed=4,
        exact_overlap=True,
    )
    grid = json.loads(grid_path.read_text())
    stride = grid["stride"]
    assert grid["overlap"] == pytest.approx(1 - stride / 128)  # ashlar's pitch is exact
    true = np.array(grid["true_positions_yx"])
    nominal = np.array(
        [
            [r * stride, c * stride]
            for r in range(grid["n_rows"])
            for c in range(grid["n_cols"])
        ]
    )
    off = true - nominal
    assert np.abs(off).max() <= 6 and np.abs(off).max() > 0  # jittered, within budget
    k = 1 * grid["n_cols"] + 1  # an interior tile
    y, x = true[k]
    tile = tifffile.imread(tmp_path / "t" / "r001_c001.tif")
    assert tile.shape == (2, 128, 128)
    np.testing.assert_array_equal(
        tile[0], plane[y : y + 128, x : x + 128].astype(np.uint16)
    )
    # the same call again is the same tiles: seeded
    again = retile.write_tiles(
        slide,
        tmp_path / "t2",
        128,
        0.125,
        pixel_size_um=0.5,
        stage_jitter_um=3.0,
        seed=4,
        exact_overlap=True,
    )
    assert (
        json.loads(again.read_text())["true_positions_yx"] == grid["true_positions_yx"]
    )
    # ... and no jitter is the nominal grid, as before
    plain = json.loads(
        retile.write_tiles(
            slide, tmp_path / "t3", 128, 0.125, pixel_size_um=0.5
        ).read_text()
    )
    np.testing.assert_array_equal(np.array(plain["true_positions_yx"]), nominal)


def test_sensor_noise_is_per_tile_so_overlaps_are_no_longer_identical(tmp_path):
    """Two neighbouring tiles see the same tissue in their overlap; each being its own
    exposure, their pixels differ by the noise. On identical overlaps ASHLAR's edge
    registration fails on a rounding difference, so this is what lets it run at all."""
    plane = _texture((300, 420))
    slide = _slide(tmp_path / "s.ome.tif", plane)
    grid = json.loads(
        retile.write_tiles(
            slide,
            tmp_path / "n",
            128,
            0.125,
            pixel_size_um=0.5,
            noise_frac=0.02,
            exact_overlap=True,
        ).read_text()
    )
    stride, ov = grid["stride"], 128 - grid["stride"]
    left = tifffile.imread(tmp_path / "n" / "r000_c000.tif")[0].astype(float)
    right = tifffile.imread(tmp_path / "n" / "r000_c001.tif")[0].astype(float)
    a, b = left[:, stride:], right[:, :ov]  # the same tissue, twice
    truth = plane[:128, stride : stride + ov]
    assert not np.array_equal(a, b)
    sd = grid["noise_sd"][0]
    assert sd == pytest.approx(0.02 * np.ptp(np.percentile(plane, (1, 99))), rel=0.05)
    assert np.std(a - truth) == pytest.approx(sd, rel=0.2)
    assert np.std(a - b) == pytest.approx(sd * np.sqrt(2), rel=0.2)  # independent draws
    clean = retile.write_tiles(slide, tmp_path / "c", 128, 0.125, pixel_size_um=0.5)
    assert json.loads(clean.read_text())["noise_sd"] == [0.0, 0.0]
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "c" / "r000_c000.tif")[0],
        plane[:128, :128].astype(np.uint16),
    )


# -------------------------------------------------------------------- the command --
def test_the_command_is_what_a_user_would_type():
    grid = {
        "pattern": "r{row:03}_c{col:03}.tif",
        "overlap": 0.125,
        "pixel_size_um": 0.325,
    }
    argv = original.ashlar_argv(
        ["/t/ref", "/t/c1"], [grid, grid], "/o/ashlar_output.ome.tif", 0, 15, ["-q"]
    )
    assert argv == [
        "ashlar",
        "filepattern|/t/ref|pattern=r{row:03}_c{col:03}.tif|overlap=0.125|pixel_size=0.325",
        "filepattern|/t/c1|pattern=r{row:03}_c{col:03}.tif|overlap=0.125|pixel_size=0.325",
        "-o",
        "/o/ashlar_output.ome.tif",
        "-c",
        "0",
        "-m",
        "15.0",
        "-q",
    ]
    with pytest.raises(ValueError, match="separator"):
        original.reader_spec("/t/a=b", grid)


def test_the_nuclear_channel_is_one_index_for_every_cycle():
    """ashlar's -c is a single index: the nuclear stain must sit there in every cycle."""
    cycles = ["ref", "c1"]
    assert original.nuclear_index(None, ["SMA|DAPI", "CD3|DAPI"], "DAPI", cycles) == 1
    assert original.nuclear_index(2, ["SMA|DAPI", "CD3|DAPI"], "DAPI", cycles) == 2
    assert original.nuclear_index(None, None, "DAPI", cycles) == 0
    with pytest.raises(SystemExit, match="not at one index"):
        original.nuclear_index(None, ["DAPI|SMA", "CD3|DAPI"], "DAPI", cycles)
    with pytest.raises(SystemExit, match="no 'DAPI'"):
        original.nuclear_index(None, ["DAPI|SMA", "CD3|CD8"], "DAPI", cycles)


# ------------------------------------------------------------------- the manifest --
def test_displacement_uses_where_the_pixels_came_from_not_the_stage_positions():
    """D = (L - c_mov) - (E - c_ref): with the tiles' TRUE corners, a moving cycle whose
    content sits 5 px right of the reference comes out as -5 in x whatever the jitter."""
    c_ref = np.array([[0.0, 0.0], [0.0, 100.0]]) + [[1, -2], [0, 3]]  # jittered truths
    c_mov = np.array([[0.0, 0.0], [0.0, 100.0]]) + [[-1, 1], [2, 0]]
    origin = np.array([7.0, 9.0])
    e = (
        c_ref + origin
    )  # a perfect stitch: every tile at its true place, plus the origin
    layer = c_mov + origin + [0.0, -5.0]
    d = original.tile_displacements(layer, c_mov, e, c_ref, [0, 1])
    np.testing.assert_allclose(d, [[0.0, -5.0], [0.0, -5.0]])
    # a moving tile matched to the NEIGHBOURING reference tile uses that tile's frame
    d2 = original.tile_displacements(layer, c_mov, e + [[0, 0], [4, 0]], c_ref, [1, 1])
    np.testing.assert_allclose(d2[:, 0], [-4.0, -4.0])


def test_the_field_is_rigid_inside_a_tile_and_ramps_only_across_the_overlap():
    """ASHLAR moves each tile as a block. The scoring field must do the same: constant
    over the part of a tile no neighbour covers, a ramp across the band two tiles share."""
    from stare.mesh_field import MeshField

    tile, stride = 100, 80  # 20 px overlap
    residual = np.array([[[1.0, 0.0], [5.0, 0.0], [-3.0, 0.0]]])  # one row, three tiles
    gx, gy, disp = original.piecewise_mesh(1, 3, tile, stride, residual, (0.0, 0.0))
    assert gx == [20, 80, 100, 160, 180, 240] and gy == [20, 80]
    field = MeshField(gx, gy, disp)
    x = np.array([5.0, 20, 50, 80, 90, 100, 130, 160, 170, 180, 230, 260])
    got = field.displacement(np.stack([x, np.full_like(x, 50)], 1))[:, 0]
    np.testing.assert_allclose(got, [1, 1, 1, 1, 3, 5, 5, 5, 1, -3, -3, -3])
    with pytest.raises(ValueError, match="half a tile"):
        original.piecewise_mesh(1, 2, 100, 40, np.zeros((1, 2, 2)), (0, 0))


def test_a_cycle_entry_warps_points_by_each_tiles_own_displacement():
    """End to end without ashlar: recorded positions -> manifest -> the pipeline's own
    warper lands a point of each tile where that tile was put."""
    from stare import stage_warp

    n_rows, n_cols, tile, stride = 2, 2, 100, 80
    nominal = np.array(
        [[r * stride, c * stride] for r in range(n_rows) for c in range(n_cols)], float
    )
    grid = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "tile_size": tile,
        "stride": stride,
        "orig_shape": [180, 180],
        "true_positions_yx": nominal.tolist(),
    }
    per_tile = np.array([[3.0, -4.0], [3.0, -4.0], [3.0, -4.0], [6.0, -1.0]])  # (y, x)
    edge = {"positions_yx": nominal.tolist()}
    layer = {
        "positions_yx": (nominal + per_tile).tolist(),
        "reference_idx": [0, 1, 2, 3],
        "discard": [False] * 4,
        "errors": [0.1] * 4,
        "cycle_offset_yx": [0.0, 0.0],
    }
    entry, translation, d_yx = original.cycle_entry(edge, layer, grid, grid)
    np.testing.assert_allclose(translation, [-4.0, 3.0])  # (x, y): the median tile
    assert entry["out_shape"] == [180, 180]
    manifest = {
        "reference": "ref",
        "slides": {"ref": {"M0": np.eye(3).tolist(), "mesh": None}, "mov": entry},
    }
    warp = stage_warp.make_warper(manifest)
    pts = np.array([[40.0, 40.0], [150.0, 150.0]])  # inside tile (0,0) and tile (1,1)
    out = warp("mov", pts, stage_warp.STAGE_REFINED)
    np.testing.assert_allclose(out, pts + [[-4.0, 3.0], [-1.0, 6.0]], atol=1e-9)
    np.testing.assert_allclose(
        warp("mov", pts, stage_warp.STAGE_RIGID), pts + [-4.0, 3.0]
    )
    diag = original.cycle_diagnostics(layer, d_yx, translation)
    assert diag["n_discarded"] == 0 and diag["mesh_residual_px"][
        "max"
    ] == pytest.approx(np.hypot(3, 3))
    err = original.reference_stitch_error(
        {"positions_yx": (nominal + [5, 5]).tolist()}, grid
    )
    assert err["max"] == pytest.approx(
        0.0
    )  # one constant offset is the origin, not an error


def test_the_mosaic_is_split_into_named_slides_pixel_for_pixel(tmp_path):
    rng = np.random.default_rng(0)
    mosaic = rng.integers(0, 60000, size=(5, 700, 900)).astype(np.uint16)
    tifffile.imwrite(tmp_path / "m.ome.tif", mosaic, metadata={"axes": "CYX"}, ome=True)
    outs = [
        (tmp_path / "ref.ome.tiff", ["DAPI", "CD3"]),
        (tmp_path / "c1.ome.tiff", ["DAPI", "CD8", "CD4"]),
    ]
    original.split_mosaic(tmp_path / "m.ome.tif", outs, 0.325)
    np.testing.assert_array_equal(tifffile.imread(outs[0][0]), mosaic[:2])
    np.testing.assert_array_equal(tifffile.imread(outs[1][0]), mosaic[2:])
    with tifffile.TiffFile(outs[1][0]) as tf:
        assert all(f'Name="{n}"' in tf.ome_metadata for n in outs[1][1])
        assert 'PhysicalSizeX="0.325"' in tf.ome_metadata and tf.pages[0].is_tiled
    with pytest.raises(ValueError, match="planes"):
        original.split_mosaic(tmp_path / "m.ome.tif", outs[:1], 0.325)


# ------------------------------------------------------------- the original, for real --
def test_the_original_ashlar_recovers_a_known_shift_and_its_images_are_aligned(
    tmp_path,
):
    """The whole chain with ASHLAR itself (skipped where it is not installed): synthetic
    raw tiles of three cycles with known offsets, the unmodified command, the manifest
    scored against the truth, and ASHLAR's own registered images compared."""
    pytest.importorskip("ashlar")
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation
    from stare import stage_warp

    h, w, tile = 1500, 1800, 512
    rng = np.random.default_rng(1)
    canvas = np.zeros((h + 200, w + 200), np.float32)
    ys, xs = rng.integers(0, h + 200, 9000), rng.integers(0, w + 200, 9000)
    canvas[ys, xs] = rng.uniform(2000, 9000, 9000)
    base = ndimage.gaussian_filter(canvas, 2.5) * 25 + 300
    shifts = {
        "ref": (0, 0),
        "cyc1": (7, -11),
        "cyc2": (-20, 30),
    }  # content moved (dy, dx)
    slides = [
        _slide(
            tmp_path / f"{n}.ome.tif",
            base[100 - dy : 100 - dy + h, 100 - dx : 100 - dx + w],
        )
        for n, (dy, dx) in shifts.items()
    ]
    names = list(shifts)
    for k, s in enumerate(slides):
        retile.write_tiles(
            s,
            tmp_path / "tiles" / names[k],
            tile,
            0.125,
            cycle=k,
            canvas_like=slides,
            pixel_size_um=0.325,
            stage_jitter_um=2.0,
            noise_frac=0.01,
            seed=7,
            exact_overlap=True,
        )
    reg = [str(tmp_path / "reg" / f"{n}.ome.tiff") for n in names]
    rc = original.main(
        [
            "--tiles",
            *[str(tmp_path / "tiles" / n) for n in names],
            "--names",
            *names,
            "--outdir",
            str(tmp_path / "out"),
            "--maximum-shift",
            "15",
            "--split",
            *reg,
            "--channels",
            "DAPI|CD3",
            "DAPI|CD8",
            "DAPI|CD4",
            "--ashlar-args",
            "-q",
        ]
    )
    assert rc == 0
    record = json.loads((tmp_path / "out" / "positions.json").read_text())
    assert record["command"][0] == "ashlar" and record["command"][-1] == "-q"
    assert [a["kind"] for a in record["aligners"]] == [
        "EdgeAligner",
        "LayerAligner",
        "LayerAligner",
    ]
    assert record["reference_stitch_error_px"]["p50"] < 1.0  # it found the stage errors
    pts = np.stack([rng.uniform(150, 1650, 3000), rng.uniform(150, 1350, 3000)], 1)
    ref = tifffile.imread(reg[0])[0].astype(float)
    for n in names[1:]:
        dy, dx = shifts[n]
        manifest = json.loads((tmp_path / "out" / n / "manifest.json").read_text())
        out = stage_warp.make_warper(manifest)(n, pts, stage_warp.STAGE_REFINED)
        err = np.linalg.norm(out - (pts - [dx, dy]), axis=1)
        assert np.median(err) < 0.5 and np.percentile(err, 95) < 1.0, n
        mov = tifffile.imread(reg[names.index(n)])[0].astype(float)
        residual, _, _ = phase_cross_correlation(
            ref[200:1300, 200:1600], mov[200:1300, 200:1600], upsample_factor=10
        )
        assert np.abs(residual).max() < 0.5, n  # ASHLAR's own image is on the reference


# ------------------------------------------------------ the same defect, whole slides --
def test_whole_slides_get_noise_of_the_strength_the_tiles_get(tmp_path):
    """VALIS and STARE register whole slides. They get no stage error (a stitched slide
    has none) but the SAME sensor noise ASHLAR's tiles carry: same fraction, measured the
    same way on the same clean slide. The copies keep their names, channel names and pixel
    size, and the checkpoint is rewritten to name them."""
    from benchmarks.ashlar import degrade

    plane = _texture((500, 640), seed=3)
    clean = tmp_path / "clean" / "P1" / "preprocessed"
    clean.mkdir(parents=True)
    src = clean / "P1_DAPI_CD3.ome.tif"
    tifffile.imwrite(
        src,
        np.stack([plane, plane * 0.5]).astype(np.uint16),
        tile=(64, 64),
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}},
        ome=True,
    )
    chk = tmp_path / "clean" / "csv" / "preprocessed.csv"
    chk.parent.mkdir()
    chk.write_text(
        "patient_id,id,preprocessed_image,is_reference,channels,pixel_size\n"
        f"P1,P1_DAPI_CD3,{src},true,DAPI|CD3,0.325\n"
    )
    out = degrade.degrade_checkpoint(chk, tmp_path / "noisy", noise_frac=0.02, seed=5)
    row = out.read_text().splitlines()[1].split(",")
    dst = tmp_path / "noisy" / "P1" / "preprocessed" / "P1_DAPI_CD3.ome.tif"
    assert row[2] == str(dst) and row[3:] == ["true", "DAPI|CD3", "0.325"]
    noisy = tifffile.imread(dst).astype(float)
    truth = tifffile.imread(src).astype(float)
    assert noisy.shape == truth.shape
    # the strength retile.py gives the tiles of this very slide
    grid = json.loads(
        retile.write_tiles(
            src, tmp_path / "tiles", 128, 0.125, pixel_size_um=0.325, noise_frac=0.02
        ).read_text()
    )
    for c in range(2):
        assert np.std(noisy[c] - truth[c]) == pytest.approx(
            grid["noise_sd"][c], rel=0.1
        )
    with tifffile.TiffFile(dst) as tf:
        assert 'Name="DAPI"' in tf.ome_metadata and 'Name="CD3"' in tf.ome_metadata
        assert 'PhysicalSizeX="0.325"' in tf.ome_metadata
        assert len(tf.pages) == 2 and tf.pages[0].is_tiled  # one page per channel
    # seeded, and an existing copy is kept (an interrupted run continues)
    before = dst.stat().st_mtime_ns
    degrade.degrade_checkpoint(chk, tmp_path / "noisy", noise_frac=0.02, seed=5)
    assert dst.stat().st_mtime_ns == before
    again = degrade.degrade_checkpoint(
        chk, tmp_path / "noisy2", noise_frac=0.02, seed=5
    )
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "noisy2" / "P1" / "preprocessed" / src.name),
        tifffile.imread(dst),
    )
    assert again.is_file()
