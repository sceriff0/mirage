"""The benchmark-side one-channel proxies of an RGB slide (``regbench.prep``).

Restored from commit 5a46f778 (``benchmarks/tests/regbench/test_prep.py``) without its
tests of the candidate list, the frozen selection and the ``stare_prep`` row, which
``prep`` no longer has. What must hold: the proxy list is closed; no proxy moves image
content or changes a shape; ``grey`` is the stated formula; a whole-slide RGB TIFF is read
and written by tiles, in memory bounded by a tile; and a refusal is the benchmark's cause.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.regbench import prep

RNG = np.random.default_rng(3)


def smooth_noise(hw, dtype, top):
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(RNG.random(hw), 4)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * top).astype(dtype)


def rgb_noise(hw):
    return np.stack([smooth_noise(hw, np.uint8, 255) for _ in range(3)], -1)


# ── the list ──────────────────────────────────────────────────────────────────
def test_the_proxy_list_is_closed_and_documented():
    from benchmarks.regbench import regimes

    assert list(prep.PROXIES) == ["grey", "haem", "grey_smooth"]
    # every proxy of a regime but the channel itself, and nothing else
    assert set(prep.PROXIES) == set(regimes.PROXIES) - {"channel"}
    for name in prep.PROXIES:
        assert f"``{name}``" in prep.__doc__
    with pytest.raises(ValueError, match="expected one of"):
        prep.proxy("otsu")
    assert "5a46f778" in prep.__doc__


# ── what a proxy does to an image ─────────────────────────────────────────────
@pytest.mark.parametrize("name", list(prep.PROXIES))
def test_no_proxy_moves_image_content(name):
    check = prep.frame_check(name)
    assert check["ok"] and check["n"] > 40
    assert (
        abs(check["dx"]) < prep.FRAME_GATE_PX and abs(check["dy"]) < prep.FRAME_GATE_PX
    )


def test_a_proxy_that_moves_content_is_caught(monkeypatch):
    def shifted(reader, name):
        return np.roll(prep.to_array(reader, name), 1, axis=1)

    check = prep.frame_check("grey", process=shifted)
    assert not check["ok"] and check["dx"] == pytest.approx(1.0, abs=0.05)

    # a move with a ZERO mean: the left half one pixel right, the right half one left
    # (a rotation or a scaling about the centre has the same signature)
    def torn(reader, name):
        a = prep.to_array(reader, name)
        half = a.shape[1] // 2
        return np.hstack(
            [np.roll(a[:, :half], 1, axis=1), np.roll(a[:, half:], -1, axis=1)]
        )

    whole = prep.frame_check("grey")["n"]
    monkeypatch.setattr(prep, "FRAME_GATE_PX", 5.0)  # the mean cannot object here
    check = prep.frame_check("grey", process=torn)
    assert check["n"] == whole  # every dot was found: only the per-dot gate is left
    assert not check["ok"] and check["max"] == pytest.approx(1.0, abs=0.05)
    assert abs(check["dx"]) < 0.3  # +1 px and -1 px nearly cancel in the mean
    monkeypatch.undo()

    def turned(reader, name):
        from scipy.ndimage import rotate

        return rotate(prep.to_array(reader, name), 0.2, reshape=False, order=1)

    check = prep.frame_check("grey", process=turned)
    assert not check["ok"] and check["max"] > 1.0

    def cropped(reader, name):
        return prep.to_array(reader, name)[:-2]

    assert not prep.frame_check("grey_smooth", process=cropped)["ok"]


def test_the_result_does_not_depend_on_the_block_size():
    rgb = rgb_noise((600, 800))
    for name in prep.PROXIES:
        whole = prep.to_array(prep.array_reader(rgb), name)
        blocks = prep.to_array(prep.array_reader(rgb), name, block=256)
        assert whole.shape == rgb.shape[:2] and whole.dtype == np.uint8
        assert np.array_equal(whole, blocks), name


def test_grey_is_the_stated_formula_and_the_slide_is_dark():
    from . import rgb_cases

    rgb = RNG.integers(0, 256, (50, 60, 3), dtype=np.uint8)
    want = 255.0 - (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    got = prep.grey_plane(rgb)
    assert got.dtype == np.uint8 and np.abs(got - want).max() <= 0.5 + 1e-3
    assert np.array_equal(prep.to_array(prep.array_reader(rgb), "grey"), got)
    # the tests' stand-in proxy is the same formula (float64 there, float32 here)
    assert np.abs(got.astype(int) - rgb_cases.grey(np.moveaxis(rgb, -1, 0))).max() <= 1
    white = np.full((40, 40, 3), 255, np.uint8)
    assert prep.to_array(prep.array_reader(white), "grey").max() == 0  # paper is dark
    assert prep.to_array(prep.array_reader(white), "haem").max() == 0
    smooth = prep.to_array(prep.array_reader(rgb_noise((300, 400))), "grey_smooth")
    rough = prep.to_array(prep.array_reader(rgb_noise((300, 400))), "grey")
    assert (
        smooth.dtype == np.uint8
        and np.diff(smooth.astype(float)).std() < np.diff(rough.astype(float)).std()
    )
    with pytest.raises(prep.Refused, match="RGB"):
        prep.array_reader(np.zeros((8, 8), np.uint8))


# ── a whole-slide brightfield TIFF: read and written by tiles ─────────────────
def write_rgb_tiff(path, hw, tile=512, seed=0):
    """A tiled RGB TIFF written from a generator of tiles (never held whole): a smooth
    texture, shifted per tile so no two tiles are equal."""
    import tifffile
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    texture = np.stack(
        [gaussian_filter(rng.random((tile * 2, tile * 2)), 3) for _ in range(3)], -1
    )
    texture = ((texture - texture.min()) / np.ptp(texture) * 255).astype(np.uint8)
    h, w = hw

    def tiles():
        for ty in range(0, h, tile):
            for tx in range(0, w, tile):
                oy, ox = (ty * 7 + tx * 3) % tile, (tx * 5 + ty) % tile
                yield np.ascontiguousarray(texture[oy : oy + tile, ox : ox + tile])

    tifffile.imwrite(
        path,
        tiles(),
        shape=(h, w, 3),
        dtype=np.uint8,
        tile=(tile, tile),
        photometric="rgb",
        compression="zlib",
        bigtiff=True,
    )
    return path


def bf_case(tif, hw, modality="brightfield"):
    return SimpleNamespace(
        case_id="bf_tif",
        modality=modality,
        ref_image=str(tif),
        mov_image=str(tif),
        ref_hw=list(hw),
        mov_hw=list(hw),
        pixel_size_um=0.24,
    )


@pytest.mark.parametrize("name", ["haem", "grey_smooth"])
def test_a_brightfield_tiff_is_written_like_the_whole_image(tmp_path, name):
    """HyReCo's slides are RGB TIFFs: the proxy reads them by tiles and gives the pixels
    the whole-image path gives (a size that is not a multiple of a tile)."""
    import tifffile

    hw = (1500, 2300)
    tif = write_rgb_tiff(tmp_path / "he.tif", hw)
    ref, mov = prep.write_inputs(bf_case(tif, hw), name, tmp_path / "out")
    assert (ref, mov) == tuple(map(str, prep.input_paths(tmp_path / "out", name)))
    got = tifffile.imread(mov)
    want = prep.to_array(prep.array_reader(tifffile.imread(tif)), name)
    assert got.shape == hw and np.array_equal(got, want)
    record = json.loads((tmp_path / "out" / prep.RECORD).read_text())
    assert record["hw"] == {"ref": list(hw), "mov": list(hw)}
    assert record["proxy"] == name and record["prep_s"] > 0
    assert prep.current(bf_case(tif, hw), name, tmp_path / "out") == record


@pytest.mark.parametrize("interleaved", [True, False])
def test_an_rgb_ome_tiff_is_read_in_either_layout(tmp_path, interleaved):
    """A dataset's RGB OME-TIFF has three samples per pixel, or three planes R, G, B: the
    same proxy from both."""
    import tifffile

    from . import rgb_cases

    rgb = rgb_noise((700, 900))
    path = tmp_path / f"s{int(interleaved)}.ome.tif"
    rgb_cases.write_rgb(path, np.moveaxis(rgb, -1, 0), interleaved)
    ref, _ = prep.write_inputs(bf_case(path, rgb.shape[:2]), "haem", tmp_path / "o")
    want = prep.to_array(prep.array_reader(rgb), "haem")
    assert np.array_equal(tifffile.imread(ref), want)


def test_a_proxy_is_current_only_for_the_slides_and_code_it_was_written_from(tmp_path):
    import os

    hw = (300, 400)
    tif = write_rgb_tiff(tmp_path / "he.tif", hw, tile=256)
    case, out = bf_case(tif, hw), tmp_path / "out"
    assert prep.current(case, "haem", out) is None  # nothing written yet
    prep.write_inputs(case, "haem", out)
    assert prep.current(case, "haem", out) is not None
    assert prep.current(case, "grey_smooth", out) is None  # another proxy
    st = os.stat(tif)
    os.utime(tif, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert prep.current(case, "haem", out) is None  # the slide was replaced
    prep.write_inputs(case, "haem", out)
    prep.input_paths(out, "haem")[1].unlink()
    assert prep.current(case, "haem", out) is None  # a file is gone


MEMORY_BOUND = 64 * 2**20


def prep_peak(tif, hw, out, name):
    """tracemalloc's peak (numpy, zarr and tifffile buffers) over ``write_inputs``."""
    import tracemalloc

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        prep.write_inputs(bf_case(tif, hw), name, out)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.fixture(scope="module")
def two_slides(tmp_path_factory):
    d = tmp_path_factory.mktemp("bf_memory")
    small, big = (2048, 3072), (6144, 8192)  # the big one has 8x the pixels
    return d, {
        hw: write_rgb_tiff(d / f"{hw[0]}x{hw[1]}.tif", hw) for hw in (small, big)
    }


@pytest.mark.parametrize("name", ["haem", "grey_smooth"])
def test_a_brightfield_tiff_is_written_in_memory_bounded_by_a_tile(two_slides, name):
    """The memory contract (every reader streams): the peak is a tile's working set,
    not the slide's. Measured at two slide sizes, 8x apart in pixels: the peak does not
    grow with the slide, and it stays under a bound that a whole-slide read of the big
    fixture alone (H x W x 3 bytes, before any float conversion) exceeds more than twice
    over."""
    d, slides = two_slides
    (small, big) = sorted(slides, key=lambda hw: hw[0] * hw[1])
    whole_read = big[0] * big[1] * 3
    assert whole_read > 2 * MEMORY_BOUND  # a whole-slide read could not pass
    peaks = {
        hw: prep_peak(slides[hw], hw, d / f"out_{name}_{hw[0]}", name)
        for hw in (small, big)
    }
    mib = {hw: round(p / 2**20, 1) for hw, p in peaks.items()}
    assert peaks[big] < MEMORY_BOUND, mib
    assert peaks[big] <= peaks[small] + 8 * 2**20, (
        f"the peak grows with the slide: {mib}"
    )


def test_the_tile_writer_uses_one_encoder_thread(tmp_path, monkeypatch):
    """tifffile's default encoder pool runs ahead of the writer and queues tiles in
    proportion to the slide (58 / 199 / 269 MiB for 48 / 192 / 384 tiles, measured on
    2023.4.12 at 5a46f778): ``maxworkers=1`` is a memory bound. Asserted on the argument,
    which holds at every tifffile version."""
    import tifffile

    from benchmarks.regbench import imageio

    seen = []
    real = tifffile.TiffWriter.write

    def spy(self, data=None, **kwargs):
        seen.append(kwargs.get("maxworkers", "<default>"))
        return real(self, data, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", spy)
    tiles = (np.zeros((64, 64), np.uint8) for _ in range(4))
    imageio.write_nuclear_tiles(tmp_path / "t.ome.tif", (128, 128), np.uint8, tiles, 64)
    assert seen == [1]


def test_a_refusal_is_the_benchmarks_cause(tmp_path):
    """What the proxy writer refuses (here: a 16-bit 'RGB' TIFF, and a fluorescence
    case, whose proxy is the channel itself) is the benchmark's own step failing, never
    the method's exception."""
    import tifffile

    from benchmarks.regbench import failure_log, resources

    tif = tmp_path / "rgb16.tif"
    tifffile.imwrite(tif, np.zeros((64, 64, 3), np.uint16), photometric="rgb")
    with pytest.raises(prep.Refused) as err:
        prep.write_inputs(bf_case(tif, (64, 64)), "haem", tmp_path / "out")
    cause = resources.classify_error(err.value)
    assert cause == "benchmark" and cause in resources.NOT_THE_METHOD
    assert failure_log.WHOSE[cause] == failure_log.BENCHMARK
    with pytest.raises(prep.Refused, match="fluorescence"):
        prep.write_inputs(bf_case(tif, (64, 64), "fluorescence"), "haem", tmp_path)


def test_read_rgb_reduced_is_the_block_mean_of_the_slide(tmp_path):
    """``imageio.read_rgb_reduced``: long side <= ``max_dim``, each pixel the rounded mean
    of an f x f block (the edge that is not a whole block is dropped), and the factor."""
    from benchmarks.regbench import imageio

    tif = write_rgb_tiff(str(tmp_path / "s.tif"), (200, 300), tile=64)
    import tifffile

    full = tifffile.imread(tif)
    got, f = imageio.read_rgb_reduced(tif, 100, rows=7)
    assert f == 3 and got.dtype == np.uint8 and max(got.shape[:2]) <= 100
    hh, ww = (200 // f) * f, (300 // f) * f
    want = (
        full[:hh, :ww]
        .astype(np.float32)
        .reshape(hh // f, f, ww // f, f, 3)
        .mean(axis=(1, 3))
        .round()
        .astype(np.uint8)
    )
    assert got.shape == want.shape == (66, 100, 3)
    assert np.array_equal(got, want)
