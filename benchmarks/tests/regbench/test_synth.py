"""The synthetic scaling set's generator (benchmarks/regbench/synth.py, T32).

Every test runs on tiny stand-in slides: a test-only grid of a few hundred px and a small
tile, never the paper's 4096..65536 px grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import imageio, synth
from benchmarks.regbench.cases import list_cases


def _source(tmp_path, side=300, dtype=np.uint8, names=("DAPI", "CD3"), px=0.325):
    """A two-channel stand-in slide: channel 0 a blob texture, channel 1 noise."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:side, 0:side]
    tex = np.zeros((side, side))
    for cy, cx in rng.uniform(0, side, (60, 2)):
        tex += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    tex = (tex / tex.max() * (np.iinfo(dtype).max * 0.8)).astype(dtype)
    other = rng.integers(0, 50, (side, side)).astype(dtype)
    path = tmp_path / "src.ome.tif"
    imageio.write_channels(path, np.stack([tex, other]), list(names), px)
    return path


SIZES = (64, 128, 256)


def _plan(tmp_path, src, sizes=SIZES, seed=0, to=None):
    to = Path(to or tmp_path / "cases")
    synth.plan(src, to, sizes=sizes, seed=seed, tile=32, texture_side=128)
    return to


def _generate(tmp_path, src, sizes=SIZES, seed=0, to=None):
    to = _plan(tmp_path, src, sizes, seed, to)
    for i in range(synth.n_units(to)):
        synth.render_unit(to, i)
    synth.finish(to)
    return to


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the design --------------------------------------------------------------------------


def test_the_grids_are_the_stated_sizes_and_small_is_a_subset_of_default():
    assert synth.GRIDS["default"] == (4096, 8192, 16384, 32768, 65536)
    assert synth.GRIDS["small"] == (4096,)
    assert set(synth.GRIDS["small"]) < set(synth.GRIDS["default"])


def test_the_constants_are_the_brief_s():
    assert synth.CHANNELS == ("DAPI", "S1", "S2")
    assert synth.PIXEL_SIZE_UM == 0.325
    assert synth.DTYPE == np.uint16
    assert synth.N_PANELS == 11 and synth.N_MOVING == 10
    assert synth.ROTATION_DEG == 1.0 and synth.SHIFT_FRACTION == 0.02
    assert synth.WAVELENGTH_UM == 1000.0 and synth.AMPLITUDE_UM == 5.0
    assert synth.N_MODES == 4 and synth.NOISE_SD == 3.0
    # 1 mm at 0.325 um/px is 3077 px; 5 um is ~15 px
    assert round(synth.WAVELENGTH_UM / synth.PIXEL_SIZE_UM) == 3077
    assert round(synth.AMPLITUDE_UM / synth.PIXEL_SIZE_UM) == 15


def test_the_byte_estimate_is_the_brief_s():
    est = synth.estimate(synth.GRIDS["default"])
    per_image = [r["bytes_per_image"] for r in est["sizes"]]
    assert [round(b / 1e9, 1) for b in per_image] == [0.1, 0.4, 1.6, 6.4, 25.8]
    assert round(est["total_bytes"] / 1e9) == 378


def test_compute_target_shape_is_the_archive_s():
    assert synth.compute_target_shape((100, 200), 400) == (200, 400)
    assert synth.compute_target_shape((300, 300), 150) == (150, 150)
    with pytest.raises(ValueError):
        synth.compute_target_shape((0, 0), 10)


def test_synthesize_channels_is_the_archive_s_perturbation():
    src = (np.arange(64 * 64) % 200).reshape(64, 64).astype(np.uint16)
    out = synth.synthesize_channels(src, 3, seed=1, block_rows=16)
    assert out.shape == (3, 64, 64) and out.dtype == np.uint16
    assert np.array_equal(out[0], src)
    assert not np.array_equal(out[1], src) and not np.array_equal(out[1], out[2])
    assert np.array_equal(out, synth.synthesize_channels(src, 3, seed=1, block_rows=64))


# ---- the transform -----------------------------------------------------------------------


def test_the_transform_is_within_the_stated_bounds():
    for size in (4096, 65536):
        for k in range(1, 11):
            t = synth.transform(0, size, k)
            assert abs(t["rotation_deg"]) <= 1.0
            assert all(abs(v) <= 0.02 * size for v in t["translation_px"])
            assert len(t["modes"]) == 4
            assert all(
                abs(m["wavelength_px"] - 1000 / 0.325) < 1e-9 for m in t["modes"]
            )
            amp_um = sum(m["amplitude_px"] for m in t["modes"]) * 0.325
            assert amp_um == pytest.approx(5.0)


def test_the_deformation_amplitude_is_the_same_in_microns_at_two_sizes():
    a, b = synth.transform(0, 4096, 3), synth.transform(0, 32768, 3)
    um = [
        sum(m["amplitude_px"] for m in t["modes"]) * t["pixel_size_um"] for t in (a, b)
    ]
    assert um[0] == pytest.approx(um[1]) == pytest.approx(5.0)
    # and the displacement field never exceeds it
    yy, xx = np.mgrid[0:4096:97, 0:4096:97].astype(float)
    dy, dx = synth.displacement(a, yy, xx)
    assert float(np.hypot(dy, dx).max()) <= 5.0 / 0.325 + 1e-9


def test_the_transform_is_seeded_by_seed_size_and_panel():
    assert synth.transform(0, 4096, 1) == synth.transform(0, 4096, 1)
    assert synth.transform(0, 4096, 1) != synth.transform(0, 4096, 2)
    assert synth.transform(0, 4096, 1) != synth.transform(0, 8192, 1)
    assert synth.transform(0, 4096, 1) != synth.transform(1, 4096, 1)
    with pytest.raises(ValueError):
        synth.transform(0, 4096, 0)  # the reference has no transform


# ---- the panels --------------------------------------------------------------------------


def test_panels_have_the_stated_shape_channels_dtype_and_pixel_size(tmp_path):
    to = _generate(tmp_path, _source(tmp_path))
    for size in SIZES:
        for k in (0, 5, 10):
            p = synth.panel_path(to, size, k)
            assert imageio.channel_names(p) == ["DAPI", "S1", "S2"]
            assert imageio.pixel_size_um(p) == pytest.approx(0.325)
            arr = tifffile.imread(p)
            assert arr.shape == (3, size, size) and arr.dtype == np.uint16


def test_the_reference_panel_is_the_source_crop_and_a_moving_panel_differs(tmp_path):
    src = _source(tmp_path)
    to = _generate(tmp_path, src)
    crop = synth.texture(src, texture_side=128)
    assert crop.dtype == np.uint16 and crop.shape == (128, 128)
    # uint8 sources are widened by 257, so 255 is 65535
    raw = tifffile.imread(src)[0]
    c0 = (raw.shape[0] - 128) // 2
    assert np.array_equal(
        crop, raw[c0 : c0 + 128, c0 : c0 + 128].astype(np.uint16) * 257
    )
    ref = tifffile.imread(synth.panel_path(to, 64, 0))
    assert np.array_equal(ref[0], crop[:64, :64])
    big = tifffile.imread(synth.panel_path(to, 256, 0))[0]
    assert np.array_equal(big[:128, :128], crop)  # tiled: the crop, then its mirror
    assert np.array_equal(big[:128, 128:256], crop[:, ::-1])
    mov = tifffile.imread(synth.panel_path(to, 64, 1))
    assert not np.array_equal(mov[0], ref[0])


def test_the_moving_panel_is_the_reference_moved_by_its_transform(tmp_path):
    """Noise aside, moving DAPI at q is the reference texture at the transform of q."""
    src = _source(tmp_path)
    to = _generate(tmp_path, src)
    tex = synth.texture(src, texture_side=128).astype(float)
    t = synth.transform(0, 256, 4)
    mov = tifffile.imread(synth.panel_path(to, 256, 4))[0].astype(float)
    yy, xx = np.mgrid[0:256, 0:256].astype(float)
    py, px = synth.source_coords(t, yy, xx)
    want = synth.bilinear(tex, py, px)
    resid = mov - want
    assert abs(resid.mean()) < 1.0 and 2.0 < resid.std() < 4.0  # the s.d. 3 noise


def test_a_source_at_another_pixel_size_is_resampled_to_0_325_um(tmp_path):
    src = _source(tmp_path, px=0.65)  # nuclei twice as large in px at 0.325 um
    tex = synth.texture(src, texture_side=128)
    assert tex.shape == (128, 128) and tex.dtype == np.uint16
    raw = tifffile.imread(src)[0].astype(np.uint16) * 257
    c0 = (300 - 64) // 2
    # the 64 px source crop, upsampled 2x: every other pixel is a source pixel
    assert np.allclose(
        tex[::2, ::2][:60, :60].astype(float),
        raw[c0 : c0 + 64, c0 : c0 + 64][:60, :60].astype(float),
        atol=257 * 40,
    )


def test_the_nuclear_channel_is_found_by_name(tmp_path):
    src = _source(tmp_path, names=("CD3", "DAPI"))
    # the blob texture is channel 0 whatever its name; by name the picker takes DAPI = 1
    assert synth._nuclear_index(src, None) == 1
    assert synth._nuclear_index(src, 0) == 0


def test_the_same_seed_gives_byte_identical_panels(tmp_path):
    src = _source(tmp_path)
    a = _generate(tmp_path, src, to=tmp_path / "a")
    b = _generate(tmp_path, src, to=tmp_path / "b")
    for size in SIZES:
        for k in range(11):
            assert _sha(synth.panel_path(a, size, k)) == _sha(
                synth.panel_path(b, size, k)
            )
    c = _generate(tmp_path, src, seed=1, to=tmp_path / "c")
    assert _sha(synth.panel_path(a, 64, 3)) != _sha(synth.panel_path(c, 64, 3))


def test_tile_generation_memory_does_not_grow_with_the_slide(tmp_path):
    """The archive's row-block bound, as tiles: peak memory is a tile's, not a slide's."""
    tex = synth.texture(_source(tmp_path), texture_side=128)

    def peak(size):
        tracemalloc.start()
        for _ in synth.panel_tiles(tex, 0, size, 3, tile=32):
            pass
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    small, large = peak(128), peak(1024)
    # 64x the pixels; the peak stays a few tiles (2 MB is ~ 40 float64 32x32 tiles)
    assert large < 2 * small + 2**20
    assert large < 1024 * 1024 * 3 * 2 / 8


# ---- the cases, the refusals, the estimate ------------------------------------------------


def test_each_size_is_one_patient_of_ten_cases_on_one_reference(tmp_path):
    to = _generate(tmp_path, _source(tmp_path))
    cases = list_cases(to, "synthetic")
    assert len(cases) == 3 * 10
    for size in SIZES:
        mine = [c for c in cases if c.group == synth.patient_of(size)]
        assert len(mine) == 10
        assert {c.ref_image for c in mine} == {str(synth.panel_path(to, size, 0))}
        assert len({Path(c.mov_image).name for c in mine}) == 10
        for c in mine:
            assert c.modality == "fluorescence" and c.pixel_size_um == 0.325
            assert c.ref_hw == [size, size] and c.mov_hw == [size, size]
            assert c.extra["mov_channel"] == 0 and c.extra["ref_channel"] == 0
            assert c.extra["channel_names"] == ["DAPI", "S1", "S2"]
            assert c.ref_nuclear == "" and c.mov_nuclear == ""
            tj = json.loads((Path(c.dir) / "synth_transform.json").read_text())
            k = int(c.extra["synthetic"]["panel"])
            assert tj == json.loads(json.dumps(synth.transform(0, size, k)))
            assert c.diagonal == pytest.approx(math.hypot(size, size))


def test_the_bytes_are_printed_before_anything_is_written(
    tmp_path, capsys, monkeypatch
):
    src = _source(tmp_path)
    seen = []

    def spy(*a, **k):
        seen.append(capsys.readouterr().out)
        raise SystemExit("stop")

    monkeypatch.setattr(synth, "_write_plan", spy)
    with pytest.raises(SystemExit):
        synth.plan(src, tmp_path / "cases", sizes=SIZES, tile=32, texture_side=128)
    assert "bytes" in seen[0] and str(synth.estimate(SIZES)["total_bytes"]) in seen[0]
    assert not (tmp_path / "cases" / "synthetic").exists()


def test_it_refuses_to_generate_over_an_existing_cases_folder(tmp_path):
    src = _source(tmp_path)
    to = _generate(tmp_path, src)
    before = _sha(synth.panel_path(to, 64, 2))
    with pytest.raises(SystemExit, match="exists"):
        synth.plan(src, to, sizes=SIZES, tile=32, texture_side=128)
    (tmp_path / "other" / "synthetic").mkdir(parents=True)
    with pytest.raises(SystemExit, match="exists"):
        synth.plan(src, tmp_path / "other", sizes=SIZES, tile=32, texture_side=128)
    assert _sha(synth.panel_path(to, 64, 2)) == before


def test_finish_refuses_a_set_with_a_missing_panel(tmp_path):
    to = _plan(tmp_path, _source(tmp_path))
    for i in range(synth.n_units(to) - 1):
        synth.render_unit(to, i)
    with pytest.raises(SystemExit, match="missing"):
        synth.finish(to)
    assert list_cases(to, "synthetic") == []


def test_a_rendered_unit_is_kept_when_its_array_task_runs_again(tmp_path):
    to = _plan(tmp_path, _source(tmp_path))
    synth.render_unit(to, 0)
    p = synth.panel_path(to, 64, 0)
    mtime = p.stat().st_mtime_ns
    synth.render_unit(to, 0)
    assert p.stat().st_mtime_ns == mtime


def test_the_status_says_complete_planned_or_absent(tmp_path):
    src = _source(tmp_path)
    assert synth.status(tmp_path / "x") == "absent"
    to = _plan(tmp_path, src)
    assert synth.status(to) == "planned"
    for i in range(synth.n_units(to)):
        synth.render_unit(to, i)
    synth.finish(to)
    assert synth.status(to) == "complete"


def test_the_cli_runs_the_whole_set_and_answers_the_unit_count(tmp_path, capsys):
    from benchmarks.regbench import cli

    src = _source(tmp_path)
    to = tmp_path / "cli"
    # the test-only grid is not a CLI choice: the CLI prints the estimate of the grid named
    assert (
        cli.main(
            [
                "synth",
                "--source",
                str(src),
                "--to",
                str(to),
                "--estimate",
                "--grid",
                "small",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert str(synth.estimate((4096,))["total_bytes"]) in out
    assert not to.exists()


def test_units_are_the_grids_before_the_plan_and_the_plans_after(tmp_path, capsys):
    """The launcher sizes the array before the plan job has run: ``--units`` is the
    grid's count (one per patient and panel) until a plan is written, then the plan's."""
    from types import SimpleNamespace

    args = dict(estimate=False, status=False, units=True, plan=False, unit=None,
                finish=False, source=None, seed=0, channel=None)  # fmt: skip
    to = tmp_path / "cases"
    for grid, want in (("default", 55), ("small", 11)):
        synth.main(SimpleNamespace(to=str(to), grid=grid, **args))
        assert capsys.readouterr().out.strip() == str(want)
    _plan(tmp_path, _source(tmp_path), sizes=(64, 128), to=to)
    capsys.readouterr()
    synth.main(SimpleNamespace(to=str(to), grid="default", **args))
    assert capsys.readouterr().out.strip() == "22"
