import csv
from pathlib import Path

import numpy as np
import pytest

from benchmarks.generate_matrix import compute_target_shape, synthesize_channels


def test_compute_target_shape_scales_long_edge_preserving_aspect():
    # source 4000x2000 (HxW), target long edge 1000 -> 1000x500
    assert compute_target_shape((4000, 2000), 1000) == (1000, 500)


def test_compute_target_shape_long_edge_is_height_when_taller():
    assert compute_target_shape((2000, 4000), 1000) == (500, 1000)


def test_synthesize_channels_returns_requested_count_and_shape():
    src = np.full((8, 8), 100, dtype=np.uint8)
    out = synthesize_channels(src, n_channels=4, seed=0)
    assert out.shape == (4, 8, 8)
    assert out.dtype == np.uint8


def test_synthesize_channels_first_channel_is_source_unchanged():
    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = synthesize_channels(src, n_channels=3, seed=0)
    np.testing.assert_array_equal(out[0], src)


def test_synthesize_channels_extra_channels_differ_from_source():
    src = np.full((16, 16), 120, dtype=np.uint8)
    out = synthesize_channels(src, n_channels=2, seed=0)
    assert not np.array_equal(out[1], src)  # jitter+noise+offset perturbs it


def test_synthesize_channels_is_deterministic_for_seed():
    src = np.full((16, 16), 120, dtype=np.uint8)
    a = synthesize_channels(src, n_channels=3, seed=7)
    b = synthesize_channels(src, n_channels=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_synthesize_channels_single_channel_returns_unchanged_source():
    src = np.arange(64, dtype=np.uint8).reshape(8, 8)
    out = synthesize_channels(src, n_channels=1, seed=0)
    assert out.shape == (1, 8, 8)
    np.testing.assert_array_equal(out[0], src)


def test_synthesize_channels_output_stays_in_dtype_range():
    src = np.full((8, 8), 250, dtype=np.uint8)
    out = synthesize_channels(src, n_channels=4, seed=1)
    assert out.max() <= 255 and out.min() >= 0


def test_synthesize_single_block_matches_reference_formula():
    """When the image fits in one block (h <= block_rows), the block-wise path must compute
    EXACTLY clip(roll(src, c)*gain + noise) with the documented rng draw order (gain, then one
    whole-image noise draw per channel) — i.e. it is unchanged from the original whole-image code."""
    src = (np.arange(6 * 5, dtype=np.uint16).reshape(6, 5) % 60).astype(np.uint16)
    out = synthesize_channels(
        src, n_channels=2, seed=4, block_rows=4096
    )  # 6 rows -> 1 block
    rng = np.random.default_rng(4)
    info = np.iinfo(np.uint16)
    gain = 1.0 + rng.uniform(-0.1, 0.1)  # same first draw
    shifted = np.roll(src, shift=1, axis=1)
    noise = rng.normal(0.0, 3.0, size=src.shape)  # same second draw (whole image)
    ref = np.clip(shifted.astype(np.float64) * gain + noise, info.min, info.max).astype(
        np.uint16
    )
    np.testing.assert_array_equal(out[1], ref)


def test_synthesize_blockwise_covers_all_rows_incl_partial_last_block():
    """Non-divisible boundary (h=10, block_rows=4 -> rows 0:4, 4:8, 8:10). Every row of every
    channel must be written into the np.empty output (no uninitialized gap at block seams), the
    result is deterministic, and channel 0 is the exact source across all blocks."""
    src = (np.arange(10 * 6, dtype=np.uint16).reshape(10, 6) * 7 % 500).astype(
        np.uint16
    )
    a = synthesize_channels(src, n_channels=3, seed=3, block_rows=4)
    b = synthesize_channels(src, n_channels=3, seed=3, block_rows=4)
    np.testing.assert_array_equal(a, b)  # deterministic for a fixed block_rows
    np.testing.assert_array_equal(
        a[0], src
    )  # ch0 unchanged incl. the partial last block
    assert a.shape == (3, 10, 6) and a.dtype == np.uint16
    assert a.max() <= np.iinfo(np.uint16).max
    for c in (1, 2):
        assert not np.array_equal(a[c], src)  # every extra channel actually perturbed
        # the partial last block (rows 8:9) was written, not left as np.empty garbage matching src
        assert not np.array_equal(a[c, 8:], src[8:])


def test_synthesize_channels_rejects_3d_input():
    with pytest.raises(ValueError):
        synthesize_channels(np.zeros((2, 4, 4), dtype=np.uint8), n_channels=2)


def test_synthesize_channels_rejects_zero_channels():
    with pytest.raises(ValueError):
        synthesize_channels(np.zeros((4, 4), dtype=np.uint8), n_channels=0)


def test_synthesize_channels_rejects_float_dtype():
    with pytest.raises(ValueError):
        synthesize_channels(np.zeros((4, 4), dtype=np.float32), n_channels=2)


def test_compute_target_shape_rejects_zero_dimension():
    with pytest.raises(ValueError):
        compute_target_shape((0, 0), 512)


def test_run_matrix_preserves_uint16_dtype(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "src16.tif"
    tifffile.imwrite(src, np.full((200, 100), 4000, dtype=np.uint16))

    manifest = run_matrix(
        source=src,
        outdir=tmp_path / "m16",
        target_px=[50],
        n_channels=[1, 2],
        seed=0,
    )
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        arr = tifffile.imread(r["path"])
        assert arr.dtype == np.uint16
        n, h, w = int(r["n_channels"]), int(r["height"]), int(r["width"])
        if n == 1:
            assert arr.shape == (h, w)
        else:
            assert arr.shape == (n, h, w)


def test_run_matrix_writes_cells_and_manifest(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "src.tif"
    tifffile.imwrite(src, np.full((400, 200), 100, dtype=np.uint8))

    manifest = run_matrix(
        source=src,
        outdir=tmp_path / "matrix",
        target_px=[100, 50],
        n_channels=[1, 2],
        seed=0,
    )

    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    # 2 sizes x 2 channel-counts = 4 cells
    assert len(rows) == 4
    assert set(rows[0].keys()) == {
        "cell_id",
        "target_px",
        "width",
        "height",
        "n_channels",
        "bytes",
        "path",
    }
    for r in rows:
        p = Path(r["path"])
        assert p.exists() and int(r["bytes"]) == p.stat().st_size
        arr = tifffile.imread(p)
        n = int(r["n_channels"])
        # single-channel cells are 2-D; multi-channel are (C, H, W)
        assert (arr.ndim == 2 and n == 1) or (arr.shape[0] == n)


def test_run_matrix_paired_writes_moving_with_distinct_channels(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "s.tif"
    tifffile.imwrite(src, np.full((200, 100), 100, dtype=np.uint8))
    manifest = run_matrix(
        source=src,
        outdir=tmp_path / "m",
        target_px=[50],
        n_channels=[1, 2],
        seed=0,
        paired=True,
    )
    rows = {r["cell_id"]: r for r in csv.DictReader(open(manifest))}
    assert "moving_paths" in next(iter(rows.values()))
    # n>=2 gets one moving file (default n_moving=1); n==1 does not (would collide on {DAPI})
    movs = rows["px50_ch2"]["moving_paths"].split(";")
    assert len(movs) == 1 and Path(movs[0]).exists()
    assert rows["px50_ch1"]["moving_paths"] == ""
    mov = tifffile.imread(movs[0])
    assert mov.shape[0] == 2  # (C,H,W)


def test_run_matrix_n_moving_writes_distinct_panels(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "s.tif"
    tifffile.imwrite(src, np.full((200, 100), 100, dtype=np.uint8))
    manifest = run_matrix(
        source=src,
        outdir=tmp_path / "m",
        target_px=[50],
        n_channels=[2],
        seed=0,
        paired=True,
        n_moving=3,
    )
    row = next(iter(csv.DictReader(open(manifest))))
    movs = row["moving_paths"].split(";")
    assert len(movs) == 3  # one distinct moving image per extra registration panel
    assert all(Path(p).exists() for p in movs)
    assert len(set(movs)) == 3  # distinct filenames


def test_derive_from_sweep_computes_matrix_shape(tmp_path):
    from benchmarks.generate_matrix import derive_from_sweep

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "baseline:\n"
        "  target_px: 4096\n"
        "  n_channels: 2\n"
        "  n_register_images: 2\n"
        "axes:\n"
        "  target_px: [2048, 4096, 8192]\n"
        "  n_channels: [1, 2, 4]\n"
        "  n_register_images: [2, 4, 8]\n"
    )
    d = derive_from_sweep(sweep)
    assert d["target_px"] == [2048, 4096, 8192]  # axis ∪ baseline, sorted
    assert d["n_channels"] == [1, 2, 4]
    assert d["n_moving"] == 7  # max(n_register_images) - 1 = 8 - 1
    assert d["paired"] is True  # >1 panel requested


def test_derive_from_sweep_matches_repo_sweep():
    """The shipped sweep.yaml derives a self-consistent matrix (no manual --n-moving sync)."""
    from pathlib import Path

    from benchmarks.generate_matrix import derive_from_sweep

    d = derive_from_sweep(Path(__file__).parents[1] / "configs" / "sweep.yaml")
    assert d["n_moving"] == 7 and d["paired"] is True
    # Input-scale cells come from the scaling_grid: 65536 is the largest benchmarked
    # size (131072 is not, at ~69 GB/cell) and 4096 the smallest, because VALIS refuses
    # a slide no larger than its 2048 px non-rigid size. Channels are 2-4; 1 channel
    # does not exercise the multi-channel path and is not benchmarked.
    assert max(d["target_px"]) == 65536
    assert 131072 not in d["target_px"]
    assert min(d["target_px"]) == 4096
    assert d["n_channels"] == [2, 3, 4]


def test_derive_moving_map_matches_registration_grid():
    from pathlib import Path

    from benchmarks.generate_matrix import derive_from_sweep

    d = derive_from_sweep(Path(__file__).parents[1] / "configs" / "sweep.yaml")
    mm = d["n_moving_map"]
    # The registration grid runs N=8 at EVERY size and at EVERY channel count it lists, so
    # each of its cells must carry 7 moving panels (N=8 => 7 moving slides) — including the
    # big ones, now that they're registered with more than 2 rounds. Read from the shipped
    # grid rather than a second copy of its values, which is how this drifted before.
    import yaml

    rg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "sweep.yaml").read_text()
    )["registration_grid"]
    assert rg["target_px"] and rg["n_channels"], "registration_grid lost its cells"
    for t in rg["target_px"]:
        for c in rg["n_channels"]:
            assert mm[(t, c)] == 7, f"{c}-ch cell {t} should carry 7 panels"


def test_run_matrix_moving_map_generates_per_cell_counts(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "s.tif"
    tifffile.imwrite(src, np.full((64, 64), 100, dtype=np.uint8))
    mm = {(32, 2): 3, (48, 2): 1}
    manifest = run_matrix(
        source=src,
        outdir=tmp_path / "m",
        target_px=[32, 48],
        n_channels=[2],
        seed=0,
        paired=True,
        n_moving_map=mm,
    )
    rows = {r["cell_id"]: r for r in csv.DictReader(open(manifest))}
    assert (
        len(rows["px32_ch2"]["moving_paths"].split(";")) == 3
    )  # 3 panels for this cell
    assert len(rows["px48_ch2"]["moving_paths"].split(";")) == 1  # only 1 for this cell
    assert (tmp_path / "m" / "px32_ch2_moving3.ome.tif").exists()
    assert not (
        tmp_path / "m" / "px48_ch2_moving2.ome.tif"
    ).exists()  # not over-generated


def test_run_matrix_default_unpaired_manifest_columns_unchanged(tmp_path):
    import tifffile

    from benchmarks.generate_matrix import run_matrix

    src = tmp_path / "s.tif"
    tifffile.imwrite(src, np.full((80, 80), 100, dtype=np.uint8))
    manifest = run_matrix(
        source=src, outdir=tmp_path / "m", target_px=[40], n_channels=[2], seed=0
    )
    rows = list(csv.DictReader(open(manifest)))
    assert set(rows[0].keys()) == {
        "cell_id",
        "target_px",
        "width",
        "height",
        "n_channels",
        "bytes",
        "path",
    }


# ---------------------------------------------------------------------------
# Source formats — the matrix generator must accept what the PIPELINE accepts
# ---------------------------------------------------------------------------


def test_bioio_only_suffixes_are_a_subset_of_the_pipeline_format_table():
    """bin/utils/ome_io.py is the authority on what the pipeline can read.

    bin/convert_image.py's old BIOIO_NATIVE_FORMATS set is gone: v1.0.0's plan 05
    moved format dispatch to ome_io.py's `_SUFFIX_TO_READER` (suffix -> reader name;
    readers are "bioio", "bioio-bioformats", "tifffile", "hdf5" -- see `READERS`).
    If the pipeline grows a vendor format there, the benchmark's matrix generator
    must not be the thing that rejects it — a source you can run the pipeline on
    should not be one you cannot benchmark against.

    Imports the table rather than regexing the file: a regex on file text is
    exactly what went stale here when the dispatch table moved modules.
    """
    import sys
    from pathlib import Path

    from benchmarks.generate_matrix import _BIOIO_ONLY_SUFFIXES

    repo = Path(__file__).resolve().parents[2]
    # benchmarks/tests/ has no conftest putting bin/ on sys.path the way the
    # pipeline's own tests/conftest.py:68-70 does -- ome_io.py's flat sibling
    # imports (`from pixel_size import ...`) need bin/utils itself on sys.path.
    bin_utils = str(repo / "bin" / "utils")
    if bin_utils not in sys.path:
        sys.path.insert(0, bin_utils)
    import ome_io

    pipeline_bioio_suffixes = {
        suffix
        for suffix, reader in ome_io._SUFFIX_TO_READER.items()
        if reader in ("bioio", "bioio-bioformats")
    }
    missing = sorted(_BIOIO_ONLY_SUFFIXES - pipeline_bioio_suffixes)
    assert not missing, (
        f"generate_matrix routes {missing} to bioio but ome_io._SUFFIX_TO_READER "
        f"does not route them through bioio/bioio-bioformats: "
        f"{sorted(pipeline_bioio_suffixes)}"
    )


def test_nd2_without_bioio_fails_with_an_actionable_message(tmp_path, monkeypatch):
    """tifffile.imread on an ND2 raises a bare 'cannot determine format', which
    reads like a corrupt file rather than a missing dependency."""
    import builtins

    import pytest

    from benchmarks.generate_matrix import _read_source_2d

    real_import = builtins.__import__

    def no_bioio(name, *a, **kw):
        if name == "bioio":
            raise ImportError("no bioio")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_bioio)
    src = tmp_path / "slide.nd2"
    src.write_bytes(b"not really an nd2")
    with pytest.raises(SystemExit) as ei:
        _read_source_2d(src)
    msg = str(ei.value)
    assert "bioio" in msg and "convert one slide" in msg


def test_only_one_synthesized_stack_is_alive_at_a_time(tmp_path, monkeypatch):
    """The peak this job is sized from is resized + ONE stack, not two.

    run_matrix built the reference stack, wrote it, and then allocated each moving
    panel's stack while the reference was STILL BOUND. For a 90000px x 4ch cell that
    is 64.8 GB each: 16.2 (resized) + 64.8 + 64.8 = 145.8 GB, which SIGKILLed a
    128 GB matrix job 77 minutes in. Nothing caught it because every fixture here is
    small enough that two live stacks fit anyway.

    Measured by weakref rather than by memory: on entry to each synthesize_channels
    call, count how many previously-returned stacks are still alive. Correct is zero
    -- the previous one must have been released before the next is allocated.
    """
    import weakref

    import numpy as np

    from benchmarks import generate_matrix as gm

    alive, seen_max = [], []
    real = gm.synthesize_channels

    def counting(src_2d, n_channels, seed=0, **kw):
        live = sum(1 for r in alive if r() is not None)
        seen_max.append(live)
        out = real(src_2d, n_channels, seed=seed, **kw)
        alive.append(weakref.ref(out))
        return out

    monkeypatch.setattr(gm, "synthesize_channels", counting)

    src = tmp_path / "src.tif"
    import tifffile

    tifffile.imwrite(src, np.full((64, 64), 700, dtype=np.uint16))

    gm.run_matrix(
        src, tmp_path / "m", target_px=[32, 48], n_channels=[2], paired=True, n_moving=3
    )

    assert seen_max, "synthesize_channels was never called"
    assert max(seen_max) == 0, (
        f"up to {max(seen_max)} previously-built stack(s) were still alive when the "
        "next was allocated; the peak is then N+1 stacks, not one. Release each "
        "stack (del) before building the next."
    )
