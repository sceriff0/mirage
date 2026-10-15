"""Raw-tile datasets prepared per stitching front-end (``datasets/rawtiles.py``), the
``@raw`` case set ASHLAR reads natively, their run lists and their scoring.

The stand-in is a PhenoCycler tree (``MICROSCOPE_OUTPUT/cycNNN_reg001``, the dataset's
real layout) of two cycles over one snake grid: cycle 2 is cycle 1's scene shifted by
(7, -4) px, each tile with its own noise. ``ashlar_edge`` runs against ``fake_ashlar``
(which returns the stage positions) and ``stare_mosaic``'s subprocess is replaced by one
returning the stage positions, as in ``test_frontends.py``.
"""

from __future__ import annotations

import csv
import gc
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from benchmarks.regbench import (
    RAW_FRONT_ENDS,
    RAW_TILE_CASE_SETS,
    RAW_TILE_DATASETS,
    cells,
    cli,
    frontends,
    methods,
    regimes,
    score,
    stitching,
    tilemap,
    variants,
)
from benchmarks.regbench.cases import list_cases, load_points, polys
from benchmarks.regbench.datasets import rawtiles
from benchmarks.regbench.methods import ashlar as am
from benchmarks.regbench.readers._common import snake_rowcol

from . import fake_ashlar

SHIFT = (7, -4)  # cycle 2's content = cycle 1's moved by +7 px in x, -4 px in y
MOSAIC_FES = ("stage", "ashlar_edge", "stare_mosaic")


# ── the stand-in ──────────────────────────────────────────────────────────────
def scene(n, seed=1, density=1 / 500, sigma=2.5):
    rng = np.random.default_rng(seed)
    k = int(n * n * density)
    im = np.zeros((n, n), np.float32)
    np.add.at(im, (rng.integers(0, n, k), rng.integers(0, n, k)), rng.uniform(0.6, 1.4, k))
    im = gaussian_filter(im, sigma)
    return im / float(im.max())


def write_pheno(root, grid=2, size=96, overlap=0.25, shift=SHIFT, noise=0.01, seed=0,
                n_cycles=2):
    """A PhenoCycler tree with ``n_cycles`` cycles; cycle k > 1 is shifted by ``shift``."""
    out = root / "MICROSCOPE_OUTPUT"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text(json.dumps({
        "regionWidth": grid, "regionHeight": grid, "tileWidth": size, "tileHeight": size,
        "tileOverlapX": overlap, "tileOverlapY": overlap, "tilingMode": "snakerows",
        "numZPlanes": 1, "xyResolution": 500,
        "channelNames": {"channelNamesArray": [f"DAPI{k}" for k in range(1, n_cycles + 1)]},
    }))
    pitch = int(round(size * (1 - overlap)))
    margin = 16
    n = pitch * (grid - 1) + size + 2 * margin
    sc = scene(n)
    rng = np.random.default_rng(seed)
    for cyc in range(1, n_cycles + 1):
        d = out / f"cyc{cyc:03d}_reg001"
        d.mkdir()
        dx, dy = (0, 0) if cyc == 1 else shift
        for tile in range(1, grid * grid + 1):
            row, col = snake_rowcol(tile - 1, grid)
            x, y = col * pitch, row * pitch
            img = sc[margin + y - dy: margin + y - dy + size, margin + x - dx: margin + x - dx + size]
            img = img + rng.normal(0, noise, img.shape)
            plane = np.clip(np.rint(img * 20000), 0, 65535).astype(np.uint16)
            tifffile.imwrite(d / f"1_{tile:05d}_Z001_CH1.tif", plane)
    return root


class Counting(stitching.Segmenter):
    """The built-in segmenter, counting its calls."""

    def __init__(self, pixel_size_um):
        inner = stitching.builtin_segmenter(pixel_size_um, diameter_um=3.0)
        self.calls = 0

        def fn(path):
            self.calls += 1
            return inner(path)

        super().__init__(fn, inner.tag)


@pytest.fixture
def fakes(monkeypatch):
    """``ashlar_edge`` and ``stare_mosaic`` without their images: both return the stage."""
    fake_ashlar.install(monkeypatch.setitem, edge_positions=lambda md: md.positions)

    def fake_stare(argv):
        cyc = frontends.tiles.read_cycle(Path(argv[argv.index("--tiles") + 1]).parent)
        doc = {"tiles": [{"tile": str(r.tile), "x": r.x_px, "y": r.y_px} for r in cyc.tiles],
               "report": {"mosaic": None}}
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(doc))

    monkeypatch.setattr(frontends, "_run_stare_mosaic", fake_stare)
    monkeypatch.setattr(frontends, "_require_stare", lambda: "1.3.0")
    yield
    fake_ashlar.configure()


def prepare(cases, source, fe, seg=None, **kw):
    units = rawtiles.units("phenocycler", source)
    (unit,) = units
    factory = (lambda cycle, work: seg) if seg is not None else (
        lambda cycle, work: Counting(cycle.pixel_size_um)
    )
    out = rawtiles.prepare_unit(cases, "phenocycler", unit, units[unit], fe,
                                segmenter_factory=factory, **kw)
    if fe != "raw":
        rawtiles.write_stitching_csv(cases, f"phenocycler@{fe}")
    return out


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_pheno(tmp_path_factory.mktemp("pheno") / "tonsil")


# ── the vocabulary ────────────────────────────────────────────────────────────
def test_the_front_ends_are_frontends_plus_raw():
    assert RAW_FRONT_ENDS == (*frontends.FRONT_ENDS, "raw")
    assert RAW_TILE_DATASETS == ("mcmicro", "rashid", "phenocycler", "codex")
    assert "phenocycler@raw" in RAW_TILE_CASE_SETS
    for d in RAW_TILE_DATASETS:
        assert regimes.regime_of(d) == "cyclic_if"
        for fe in RAW_FRONT_ENDS:
            assert regimes.regime_of(f"{d}@{fe}") == "cyclic_if"


def test_prepare_images():
    assert rawtiles.image_for("phenocycler", "ashlar_edge") == "ASHLAR"
    assert rawtiles.image_for("phenocycler", "stare_mosaic") == "STARE_REGBENCH"
    assert rawtiles.image_for("mcmicro", None, export_only=True) == "ASHLAR"
    assert rawtiles.image_for("codex", None, export_only=True) == "STARE_REGBENCH"


# ── the run lists (user ruling 2026-10-15) ────────────────────────────────────
def runs_cli(capsys, *args):
    assert cli.main(["runs", *args]) == 0
    return [line.split("\t") for line in capsys.readouterr().out.splitlines() if line]


def test_runs_on_raw_are_ashlar_as_published_and_the_shift_cross(capsys):
    rows = runs_cli(capsys, "--dataset", "phenocycler@raw")
    names = [r[0] for r in rows]
    assert names == ["ashlar_clean", "ashlar_anone_n0_s60", "ashlar_anone_n0_s240"]
    assert {r[0]: r[10] for r in rows}["ashlar_clean"] == "ashlar_clean"  # primary column
    assert {r[0]: r[8] for r in rows} == dict.fromkeys(names, "ashlar_clean")  # gated_by
    assert variants.primary_for("phenocycler@raw") == ["ashlar_clean"]
    why = dict(r for r in runs_cli(capsys, "--dataset", "phenocycler@raw", "--excluded"))
    assert why["ashlar_anone_n05_s15"] == variants.RAW_NOISE
    assert why["ashlar"] == variants.RAW_NOISE
    assert why["affine_ashlar_clean"] == variants.RAW_AFFINE
    assert why["stare"] == variants.RAW_NOT_ASHLAR
    # every active run is either kept or left out with a reason
    kept = set(names) | set(why)
    assert kept == {variants.run_name(r) for r in variants.active_runs()}


def test_runs_on_a_mosaic_case_set_keep_ashlars_factorial(capsys):
    names = [r[0] for r in runs_cli(capsys, "--dataset", "phenocycler@stage")]
    assert "ashlar_anone_n05_s15" in names and "ashlar_clean" in names
    assert set(am.ASHLAR_CELLS) <= set(names)
    assert variants.primary_for("phenocycler@stage")[:1] == ["stare"]
    assert "ashlar" in variants.primary_for("phenocycler@stage")
    dhr = [n for n in names if n.startswith("deeperhistreg")]
    assert dhr and all("fluo" in n for n in dhr)
    why = dict(r for r in runs_cli(capsys, "--dataset", "phenocycler@stage", "--excluded"))
    assert set(why) == {
        variants.run_name(r) for r in variants.active_runs()
        if r.method == "deeperhistreg" and r.opts.get("fluo") != "1"
    }
    assert set(why.values()) == {variants.RAW_DHR}


# ── prepare per front-end ─────────────────────────────────────────────────────
def test_one_case_per_front_end_with_its_report_and_the_same_nuclei(tmp_path, source, fakes):
    cases = tmp_path / "cases"
    n_nuclei = {}
    for fe in MOSAIC_FES:
        (d,) = prepare(cases, source, fe)
        got = list_cases(cases, f"phenocycler@{fe}")
        assert [c.case_id for c in got] == ["tonsil__cyc002_reg001"]
        case = got[0]
        assert case.extra["front_end"] == fe and case.group == "tonsil"
        assert "tiles" not in case.extra  # only @raw is read as tiles
        st = case.extra["stitching"]
        assert st["ref"]["front_end"] == fe and st["mov"]["front_end"] == fe
        if fe != "stage":
            assert st["ref"]["benchmark_side"]  # every departure is recorded
        assert Path(case.ref_nuclear).name == "mosaic.ome.tif"
        with tifffile.TiffFile(case.ref_nuclear) as tf:
            assert list(tf.series[0].shape[-2:]) == case.ref_hw
        rows = list(csv.DictReader(open(cases / f"phenocycler@{fe}" / "stitching.csv")))
        assert [r["cycle"] for r in rows] == ["cyc001_reg001", "cyc002_reg001"]
        assert all(r["front_end"] == fe for r in rows)
        n_nuclei[fe] = [int(r["n_nuclei"]) for r in rows]
        pts = load_points(case)
        assert polys(pts, "ref", "nuclei").n_features == n_nuclei[fe][0]
    assert len({tuple(v) for v in n_nuclei.values()}) == 1  # one population
    assert min(n_nuclei["stage"]) > 5


def test_a_second_front_end_reads_the_nuclei_cache(tmp_path, source, fakes):
    cases = tmp_path / "cases"
    seg = Counting(0.5)
    prepare(cases, source, "stage", seg=seg)
    first = seg.calls
    assert first == 8  # 4 tiles x 2 cycles, once each
    cache = rawtiles.nuclei_cache(cases, "phenocycler", "tonsil", "cyc001_reg001")
    assert sorted(p.name for p in cache.iterdir()) == [f"t{t:05d}.npz" for t in range(1, 5)]
    prepare(cases, source, "ashlar_edge", seg=seg)
    prepare(cases, source, "stage", seg=seg, force=True)
    assert seg.calls == first  # neither segmented a tile again


def test_prepare_without_force_keeps_the_cases(tmp_path, source, fakes):
    cases = tmp_path / "cases"
    (d,) = prepare(cases, source, "stage")
    t0 = (d / "case.json").stat().st_mtime_ns
    prepare(cases, source, "stage")
    assert (d / "case.json").stat().st_mtime_ns == t0


# ── scoring ───────────────────────────────────────────────────────────────────
def test_initial_scored_per_front_end_with_the_stitching_block(tmp_path, source, fakes, capsys):
    cases, out = tmp_path / "cases", tmp_path / "out"
    for fe in MOSAIC_FES:
        prepare(cases, source, fe)
        for case in list_cases(cases, f"phenocycler@{fe}"):
            methods.run_case("initial", case, out)
    assert cli.main([
        "score", "--cases", str(cases), "--out", str(out), "--dataset", "phenocycler",
        "--only-run", "--workers", "1",
    ]) == 0
    tables = out / "tables" / "phenocycler"
    summary = list(csv.DictReader(open(tables / "summary_cells.csv")))
    initial = [r for r in summary if r["method"] == "initial" and r["subset"] == "all"]
    assert sorted(r["front_end"] for r in initial) == sorted(MOSAIC_FES)
    assert {r["dataset"] for r in initial} == {f"phenocycler@{fe}" for fe in MOSAIC_FES}
    stitch = list(csv.DictReader(open(tables / "stitching.csv")))
    assert sorted({r["front_end"] for r in stitch}) == sorted(MOSAIC_FES)
    assert len(stitch) == 3 * 2  # front-end x cycle
    text = (tables / "summary.md").read_text()
    assert "## Stitching error by front-end" in text
    assert "n_nuclei equal for every front-end: yes" in text
    assert not (out / "tables" / "summary.md").exists()  # never pooled with the cohort


# ── @raw: ASHLAR on the tiles ─────────────────────────────────────────────────
def test_raw_cases_name_the_tiles_and_sit_in_the_stage_layout(tmp_path, source, fakes):
    cases = tmp_path / "cases"
    prepare(cases, source, "stage")
    prepare(cases, source, "raw")
    (raw,) = list_cases(cases, "phenocycler@raw")
    (stage,) = list_cases(cases, "phenocycler@stage")
    assert raw.extra["front_end"] == "raw"
    assert set(raw.extra["tiles"]) == {"ref", "mov"}
    assert raw.extra["nuclei"] == rawtiles.NUCLEI_RAW
    assert raw.ref_hw == stage.ref_hw  # the stage layout is the stage mosaic's frame
    for side in ("ref", "mov"):
        a = polys(load_points(raw), side, "nuclei")
        b = polys(load_points(stage), side, "nuclei")
        np.testing.assert_array_equal(a.xy, b.xy)
    assert not (cases / "phenocycler@raw" / "stitching.csv").exists()


def nearest(a, b):
    d, _ = cKDTree(b).query(a)
    return d


def test_ashlar_reads_the_tiles_as_published_and_maps_mov_onto_ref(tmp_path, source, monkeypatch):
    cases, out = tmp_path / "cases", tmp_path / "out"
    prepare(cases, source, "raw")
    shift_yx = np.array([-SHIFT[1], -SHIFT[0]], dtype=float)  # what ASHLAR would find
    fake_ashlar.install(
        monkeypatch.setitem,
        edge_positions=lambda md: md.positions,
        layer_shifts=lambda cycle, r, c: np.tile(shift_yx, (r * c, 1)),
    )
    try:
        got = list_cases(cases, "phenocycler@raw")
        done = methods.run_group(
            "ashlar", got, out, {"noise_frac": "0", "affine": "none"}, label="ashlar_clean"
        )
        assert done == {"tonsil__cyc002_reg001": ["ashlar_clean"]}
        (main,) = fake_ashlar.calls("main")
        argv = main["argv"]
        assert [a.split("|")[0] for a in argv[1:3]] == ["regbench_tiles"] * 2
        assert argv[3:] == ["-o", argv[4], "-c", "0", "-m", "15.0"]
        # ASHLAR's own CachingReader is kept: the edge aligner read through one
        assert {c["reader"] for c in fake_ashlar.calls("EdgeAligner.run")} == {"CachingReader"}
        assert "regbench_tiles" not in fake_ashlar.readers  # unregistered afterwards
        run = json.loads((out / "ashlar_clean" / "phenocycler@raw" / got[0].case_id / "run.json").read_text())
        assert run["ok"], run.get("error")
        side = run["info"]["implementation"]["benchmark_side"]
        assert any("regbench_tiles" in s for s in side)
        with np.load(out / "ashlar_clean" / "phenocycler@raw" / got[0].case_id / "warped.npz") as z:
            warped = z["nuclei_xy"] if "nuclei_xy" in z.files else z["cells_xy"]
        pts = load_points(got[0])
        ref = cells.area_centroid(cells.Polys(polys(pts, "ref", "nuclei").xy,
                                              polys(pts, "ref", "nuclei").ring_off))[1]
        mov = polys(pts, "mov", "nuclei")
        moved = cells.area_centroid(cells.Polys(np.asarray(warped, float).reshape(-1, 2),
                                                mov.ring_off))[1]
        before = cells.area_centroid(cells.Polys(mov.xy, mov.ring_off))[1]
        assert np.median(nearest(before, ref)) > 5  # the shift is there before
        assert np.median(nearest(moved, ref)) < 1.0  # and ASHLAR's map takes it out
        # scored: the off-seam rows of @raw are owned by ashlar_clean, its primary there
        for case in got:
            methods.run_case("initial", case, out)
        res = score.score_rawtiles(
            cases, out, "phenocycler", 1,
            expected={"phenocycler@raw": variants.expected_variants("phenocycler@raw")},
            primary={"phenocycler@raw": variants.primary_for("phenocycler@raw")},
        )
        non = {r["method"] for r in res["nonband_cells"]}
        assert "ashlar_clean:non_ashlar_clean" in non
        assert "initial:non_ashlar_clean" in non
        assert not any(m.endswith(":non_ashlar") for m in non)
        block = [r for r in res["common_block"] if r["dataset"] == "phenocycler@raw"]
        assert {r["method"] for r in block} == {"ashlar_clean"}  # no "no row" clutter
        assert block[0]["in_common_block"] is True
        text = (out / "tables" / "phenocycler" / "summary.md").read_text()
        assert "## `@raw`: what it does not have" in text
    finally:
        fake_ashlar.configure()


def test_raw_mode_refuses_noise_or_an_anchor(tmp_path, source, monkeypatch):
    cases = tmp_path / "cases"
    prepare(cases, source, "raw")
    fake_ashlar.install(monkeypatch.setitem, edge_positions=lambda md: md.positions)
    try:
        got = list_cases(cases, "phenocycler@raw")
        for opts in ({"noise_frac": "0.002"}, {"noise_frac": "0", "affine": "stare", "label": "affine_ashlar_clean"}):
            with pytest.raises(ValueError, match="as published"):
                list(am.register_group(got, tmp_path / "w", opts))
    finally:
        fake_ashlar.configure()


def test_rectangular_tiles_have_depths_to_their_nearest_edge():
    origins = np.array([[0.0, 0.0], [0.0, 90.0]])  # (y, x): two 60 x 100 tiles side by side
    depth, idx = tilemap._signed_depths(origins, (60, 100), np.array([[95.0, 30.0]]), 2)
    assert idx[0].tolist() == [1, 0]  # deepest first: 5 px into tile 1, 5 px into tile 0
    np.testing.assert_allclose(depth[0], [5.0, 5.0])
    depth, idx = tilemap._signed_depths(origins, (60, 100), np.array([[50.0, 30.0]]), 2)
    assert idx[0, 0] == 0 and depth[0, 0] == pytest.approx(30.0)  # half the height
    assert tilemap.in_band(origins, (60, 100), [[95.0, 30.0], [50.0, 30.0]]).tolist() == [True, False]
    # a square tile given as (h, w) reads like one number
    sq = tilemap._signed_depths(origins, 100, np.array([[95.0, 30.0]]), 2)
    rect = tilemap._signed_depths(origins, (100, 100), np.array([[95.0, 30.0]]), 2)
    np.testing.assert_allclose(sq[0], rect[0])


# ── memory: prepare streams (user ruling 2026-10-14) ──────────────────────────
MEM_TILE = 512


class Constant(stitching.Segmenter):
    """One small square per tile: polygons, never pixels."""

    def __init__(self):
        sq = cells.Polys(np.array([[10.0, 10.0], [14.0, 10.0], [14.0, 14.0], [10.0, 14.0]]),
                         np.array([0, 4], dtype=np.int64))
        super().__init__(lambda path: sq, "constant")


def test_prepare_memory_does_not_grow_with_the_grid(tmp_path):
    def measure(grid):
        k = len(list(tmp_path.iterdir()))
        src = write_pheno(tmp_path / f"s{k}" / "tonsil", grid=grid, size=MEM_TILE,
                          overlap=0.1, noise=0.0)
        cases = tmp_path / f"c{k}"
        gc.collect()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            # 512 px write-tiles (D2's memory tests' size): the small grid's mosaic is
            # already several of them, so both sides have the writer's blocks in flight
            prepare(cases, src, "stage", seg=Constant(), out_tile=512)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    measure(2)  # warm-up: lazy imports
    small, big = measure(2), measure(6)
    one_tile = MEM_TILE * MEM_TILE * 2
    print(f"prepare peak small={small} big={big} growth={big - small} tile={one_tile}")
    assert big - small < one_tile


# ── the command ───────────────────────────────────────────────────────────────
def test_the_prepare_command(tmp_path, source, monkeypatch, capsys):
    monkeypatch.setattr(
        rawtiles, "make_segmenter", lambda cycle, work: Counting(cycle.pixel_size_um)
    )
    cases = tmp_path / "cases"
    base = ["prepare", "rawtiles", "--dataset", "phenocycler", "--cases", str(cases)]
    assert cli.main([*base, "--front-end", "ashlar_edge", "--image"]) == 0
    assert capsys.readouterr().out.strip() == "ASHLAR"
    assert cli.main([*base, "--source", str(source), "--front-end", "stage"]) == 0
    assert "phenocycler@stage/tonsil__cyc002_reg001" in capsys.readouterr().out
    assert (cases / "phenocycler@stage" / "stitching.csv").is_file()
    assert cli.main([*base, "--source", str(source), "--front-end", "raw"]) == 0
    assert [c.case_id for c in list_cases(cases, "phenocycler@raw")] == [
        "tonsil__cyc002_reg001"
    ]
    with pytest.raises(SystemExit, match="not units"):
        cli.main([*base, "--source", str(source), "--front-end", "stage",
                  "--patients", "nobody"])


# ── fix round 1 ───────────────────────────────────────────────────────────────
def test_a_bare_raw_tile_name_is_a_dataset_to_score_only(capsys):
    for args in (["runs", "--dataset", "phenocycler"],
                 ["list", "--cases", "x", "--dataset", "codex", "--ids"]):
        with pytest.raises(SystemExit, match="name a case set <name>@<fe>"):
            cli.main(args)
    assert runs_cli(capsys, "--dataset", "phenocycler@stage")  # a case set is fine


def test_score_gets_each_case_sets_own_primaries(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(score, "score_rawtiles", lambda *a, **k: seen.update(k))
    assert cli.main(["score", "--cases", str(tmp_path), "--out", str(tmp_path),
                     "--dataset", "phenocycler"]) == 0
    primary = seen["primary"]
    assert set(primary) == {f"phenocycler@{fe}" for fe in RAW_FRONT_ENDS}
    assert primary["phenocycler@raw"] == ["ashlar_clean"]
    for fe in MOSAIC_FES:
        assert "ashlar" in primary[f"phenocycler@{fe}"]
        assert "ashlar_clean" not in primary[f"phenocycler@{fe}"]
    assert set(seen["expected"]) == set(primary)


def _row(ds, method, case="u__c2", ok=True):
    return {"dataset": ds, "method": method, "case_id": case, "cause": "",
            "imputed_initial": not ok}


def test_the_common_block_is_per_case_set():
    rows = [_row("p@stage", m) for m in ("initial", "ashlar", "ashlar_clean", "stare")]
    rows += [_row("p@raw", m) for m in ("initial", "ashlar_clean")]
    rows.append(_row("p@stage", "ashlar_clean", case="u__c3", ok=False))
    rows += [_row("p@stage", m, case="u__c3") for m in ("initial", "ashlar", "stare")]
    primary = {"p@stage": ["stare", "ashlar"], "p@raw": ["ashlar_clean"]}
    block = score.common_block(rows, primary)
    stage = {r["method"]: r for r in block if r["dataset"] == "p@stage"}
    assert set(stage) == {"stare", "ashlar"}  # ashlar_clean is not a primary there
    assert stage["ashlar"]["n_cases_common"] == 2  # its failure on u__c3 does not count
    raw = [r for r in block if r["dataset"] == "p@raw"]
    assert [(r["method"], r["why_not"]) for r in raw] == [("ashlar_clean", "")]
    # the old union would have listed every other primary on @raw as having no row
    union = score.common_block(rows, ["stare", "ashlar", "ashlar_clean"])
    assert {r["method"] for r in union if r["dataset"] == "p@raw"} == {
        "stare", "ashlar", "ashlar_clean"}


def test_deeperhistreg_primary_on_raw_tile_mosaic_sets_is_its_fluo_twin():
    twin = variants.fluo_twin(variants.PRIMARY_DEEPERHISTREG)
    assert twin == "deeperhistreg_hr_fluo_gpu_asdrawn"
    for fe in MOSAIC_FES:
        prim = variants.primary_for(f"codex@{fe}")
        assert twin in prim and variants.PRIMARY_DEEPERHISTREG not in prim
    assert variants.primary_for("codex@raw") == ["ashlar_clean"]
    assert variants.PRIMARY_DEEPERHISTREG in variants.primary_for(None)  # cohort unchanged
    rows = {r[0]: r for r in variants.runs_table(dataset="codex@stage")}
    assert rows["deeperhistreg_hr_fluo_gpu"][10] == twin
