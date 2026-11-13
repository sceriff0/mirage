"""IBEX stitched cycles (Imaris ``.ims`` and gzip ``.nrrd``) as multiplex-style cases.

Every file here is a stand-in written by the test in the real layout (nothing is
downloaded): an HDF5 file with PADDED ``Data`` arrays, byte-char attributes and a
``NumberOfChannels`` larger than the channel groups present; a gzip NRRD with the real
header keys (``sizes`` XYZC, ``space directions``, ``imaris_channels_information:=``).
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import tracemalloc

import h5py
import numpy as np
import pytest
import tifffile

from benchmarks.regbench import EXTERNAL_DATASETS, regimes, variants
from benchmarks.regbench.datasets import ibex

TILE = 16  # the writer's tile in these tests (a multiple of 16)


def _chars(text):
    """A string as the byte-char array Imaris stores attributes in."""
    return np.array([bytes([b]) for b in str(text).encode()], dtype="S1")


def write_ims(path, planes, names, *, X, Y, ext=None, pad=(8, 8), n_attr=None):
    """``planes``: ``[(Z, Y, X) array]``; stored PADDED by ``pad`` (rows, cols)."""
    ext = ext or (0.0, X * 0.5, 0.0, Y * 0.5)
    with h5py.File(path, "w") as f:
        info = f.create_group("DataSetInfo")
        img = info.create_group("Image")
        img.attrs["X"], img.attrs["Y"] = _chars(X), _chars(Y)
        img.attrs["Z"] = _chars(planes[0].shape[0])
        img.attrs["ExtMin0"], img.attrs["ExtMax0"] = _chars(ext[0]), _chars(ext[1])
        img.attrs["ExtMin1"], img.attrs["ExtMax1"] = _chars(ext[2]), _chars(ext[3])
        tp = info.create_group("TimeInfo")
        tp.attrs["DatasetTimePoints"] = _chars(1)
        base = f.create_group("DataSet/ResolutionLevel 0/TimePoint 0")
        for i, (p, n) in enumerate(zip(planes, names)):
            info.create_group(f"Channel {i}").attrs["Name"] = _chars(n)
            z, y, x = p.shape
            padded = np.zeros((z, y + pad[0], x + pad[1]), p.dtype)
            padded[:, :y, :x] = p
            base.create_group(f"Channel {i}").create_dataset(
                "Data", data=padded, chunks=(1, 32, 32), compression="gzip"
            )
        # the file claims more channels than it holds, as the real ones do
        info.create_group("Channel_count").attrs["NumberOfChannels"] = _chars(
            n_attr or len(planes) + 2
        )
        f["DataSetInfo/Image"].attrs["NumberOfChannels"] = _chars(n_attr or len(planes) + 2)


def write_nrrd(path, vol, names, *, px=0.284, extra=""):
    """``vol``: ``(C, Z, Y, X)`` stored XYZC (X fastest), gzip."""
    c, z, y, x = vol.shape
    xml = "<Channels>" + "".join(f'<Channel Name="{n}"/>' for n in names) + "</Channels>"
    head = (
        "NRRD0004\n"
        "type: uint8\ndimension: 4\nspace: left-posterior-superior\n"
        f"sizes: {x} {y} {z} {c}\n"
        f"space directions: ({px},0,0) (0,{px},0) (0,0,1) none\n"
        "kinds: domain domain domain list\nendian: little\nencoding: gzip\n"
        f"imaris_channels_information:={xml}\n{extra}\n"
    )
    with open(path, "wb") as fh:
        fh.write(head.encode())
        with gzip.GzipFile(fileobj=fh, mode="wb") as gz:
            gz.write(np.ascontiguousarray(vol).tobytes())


def blobs(h, w, seed, n=30):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(n):
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        img += 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    return np.clip(img, 0, 255).astype(np.uint8)


# ── .ims ──────────────────────────────────────────────────────────────────────
def test_ims_crops_the_padding_decodes_names_and_finds_the_nuclear_channel(tmp_path):
    a, b = blobs(40, 56, 1), blobs(40, 56, 2)
    src = tmp_path / "Human_Spleen_Panel4.ims"
    write_ims(src, [a[None], b[None]], ["Panel4 : CD3\n\n", "Panel4 : Hoechst\n\n"], X=56, Y=40)
    info = ibex.read_ims(src)
    assert info.names == ["Panel4 : CD3", "Panel4 : Hoechst"]  # decoded and stripped
    assert info.nuclear == 1
    assert info.hw == (40, 56)
    assert info.pixel_size_um == pytest.approx(0.5)  # (ExtMax - ExtMin) / X
    assert info.n_channels == 2  # the groups present, not NumberOfChannels (4)
    dst = tmp_path / "out.ome.tif"
    ibex.convert_cycle(src, dst, tile=TILE)
    got = tifffile.imread(dst)
    assert got.shape == (2, 40, 56)  # cropped
    assert np.array_equal(got[0], a) and np.array_equal(got[1], b)
    assert ibex.names_of(dst) == info.names
    with tifffile.TiffFile(dst) as t:
        assert "PhysicalSizeX=\"0.5\"" in t.ome_metadata


def test_ims_with_z_planes_is_a_max_projection(tmp_path):
    z = np.stack([blobs(32, 32, s) for s in (3, 4, 5)])
    src = tmp_path / "p.ims"
    write_ims(src, [z], ["JOJO-1"], X=32, Y=32)
    dst = tmp_path / "o.ome.tif"
    ibex.convert_cycle(src, dst, tile=TILE)
    assert np.array_equal(tifffile.imread(dst), z.max(axis=0))  # one channel: imread squeezes C


def test_ims_without_a_nuclear_channel_is_refused_naming_the_channels(tmp_path):
    src = tmp_path / "p.ims"
    write_ims(src, [blobs(32, 32, 1)[None]], ["CD3"], X=32, Y=32)
    with pytest.raises(ValueError, match=r"CD3.*(Hoechst|nuclear)|(Hoechst|nuclear).*CD3"):
        ibex.convert_cycle(src, tmp_path / "o.ome.tif", tile=TILE)
    assert not (tmp_path / "o.ome.tif").exists()


def test_the_nuclear_match_is_case_insensitive_on_hoechst_or_jojo():
    assert ibex.nuclear_index(["x", "Panel1 : hoechst"], "f") == 1
    assert ibex.nuclear_index(["JoJo-1", "x"], "f") == 0
    with pytest.raises(ValueError, match="CD3.*CD4|CD4.*CD3"):
        ibex.nuclear_index(["CD3", "CD4"], "f")


def test_ims_with_non_positive_extent_is_refused(tmp_path):
    src = tmp_path / "p.ims"
    write_ims(src, [blobs(32, 32, 1)[None]], ["Hoechst"], X=32, Y=32, ext=(1.0, 1.0, 0.0, 16.0))
    with pytest.raises(ValueError, match="pixel size"):
        ibex.read_ims(src)


def test_ims_x_and_y_voxels_that_disagree_are_refused(tmp_path):
    src = tmp_path / "p.ims"
    write_ims(src, [blobs(32, 32, 1)[None]], ["Hoechst"], X=32, Y=32, ext=(0, 16.0, 0, 32.0))
    with pytest.raises(ValueError, match="X.*Y|Y.*X"):
        ibex.read_ims(src)


def test_ims_data_smaller_than_the_declared_image_is_refused(tmp_path):
    src = tmp_path / "p.ims"
    write_ims(src, [blobs(32, 32, 1)[None]], ["Hoechst"], X=48, Y=32)
    with pytest.raises(ValueError, match="smaller"):
        ibex.convert_cycle(src, tmp_path / "o.ome.tif", tile=TILE)


# ── .nrrd ─────────────────────────────────────────────────────────────────────
def _vol(c, z, y, x, seed=0):
    return np.stack([np.stack([blobs(y, x, seed + 10 * i + j) for j in range(z)]) for i in range(c)])


def test_nrrd_is_xyzc_to_cyx_with_a_max_projection_and_the_header_pixel_size(tmp_path):
    vol = _vol(3, 4, 40, 56)
    src = tmp_path / "spleen_panel2.nrrd"
    write_nrrd(src, vol, ["Panel2 : CD3", "Panel2 : Hoechst  ", "Panel2 : CD4"])
    info = ibex.read_nrrd(src)
    assert info.hw == (40, 56) and info.n_channels == 3
    assert info.names == ["Panel2 : CD3", "Panel2 : Hoechst", "Panel2 : CD4"]
    assert info.nuclear == 1
    assert info.pixel_size_um == pytest.approx(0.284)
    dst = tmp_path / "o.ome.tif"
    ibex.convert_cycle(src, dst, tile=TILE)
    got = tifffile.imread(dst)
    assert got.shape == (3, 40, 56)
    assert np.array_equal(got, vol.max(axis=1))


def test_nrrd_header_that_disagrees_with_the_record_by_more_than_2_percent_is_refused(tmp_path):
    src = tmp_path / "spleen_panel1.nrrd"
    write_nrrd(src, _vol(1, 1, 32, 32), ["Hoechst"], px=0.30)  # 5.6 % off 0.284
    with pytest.raises(ValueError, match="0.284"):
        ibex.read_nrrd(src)
    ok = tmp_path / "spleen_panel2.nrrd"
    write_nrrd(ok, _vol(1, 1, 32, 32), ["Hoechst"], px=0.284 * 1.019)
    assert ibex.read_nrrd(ok).pixel_size_um == pytest.approx(0.284 * 1.019)


def test_nrrd_refuses_what_it_cannot_read(tmp_path):
    vol = _vol(1, 1, 32, 32)
    bad = tmp_path / "a.nrrd"
    write_nrrd(bad, vol, ["Hoechst"])
    raw = bad.read_bytes().replace(b"encoding: gzip", b"encoding: raw")
    (tmp_path / "b.nrrd").write_bytes(raw)
    with pytest.raises(ValueError, match="gzip"):
        ibex.read_nrrd(tmp_path / "b.nrrd")
    write_nrrd(tmp_path / "c.nrrd", vol, ["Hoechst", "extra"])  # 2 names, 1 channel
    with pytest.raises(ValueError, match="channel"):
        ibex.read_nrrd(tmp_path / "c.nrrd")
    # a truncated stream is a refusal, not a short image
    full = (tmp_path / "a.nrrd").read_bytes()
    (tmp_path / "d.nrrd").write_bytes(full[: len(full) - 20])
    with pytest.raises(ValueError, match="truncated|ended"):
        ibex.convert_cycle(tmp_path / "d.nrrd", tmp_path / "o.ome.tif", tile=TILE)


# ── discovery, cases ──────────────────────────────────────────────────────────
def _tree(root):
    root.mkdir(parents=True, exist_ok=True)
    for tissue, seed in (("Spleen", 1), ("Liver", 5)):
        for n in (1, 2):
            write_ims(
                root / f"Human_{tissue}_Panel{n}.ims",
                [blobs(40, 56, seed + n)[None], blobs(40, 56, seed + 9)[None]],
                [f"Panel{n} : CD{n}", f"Panel{n} : Hoechst"],
                X=56, Y=40,
            )
    for n in (1, 2, 3):
        write_nrrd(
            root / f"spleen_panel{n}.nrrd", _vol(2, 2, 40, 56, n), [f"P{n} : CD", f"P{n} : Hoechst"]
        )
    return root


def test_discovery_groups_by_tissue_and_orders_panels(tmp_path):
    found = ibex.discover(_tree(tmp_path / "src"))
    assert sorted(found) == ["Human_Liver", "Human_Spleen", "Mouse_Spleen"]
    assert list(found["Mouse_Spleen"]) == [1, 2, 3]
    assert found["Human_Spleen"][2].name == "Human_Spleen_Panel2.ims"


def test_rows_are_panel1_against_each_later_panel(tmp_path):
    src = _tree(tmp_path / "src")
    cache = tmp_path / "c" / "_cache" / "ibex"
    rows = ibex.rows(ibex.discover(src), cache, tile=TILE)
    ids = [r["case_id"] for r in rows]
    assert ids == [
        "Human_Liver_P2", "Human_Spleen_P2", "Mouse_Spleen_P2", "Mouse_Spleen_P3"
    ]
    r = rows[2]
    assert r["group"] == "Mouse_Spleen"
    assert r["reference"].endswith("Mouse_Spleen_P1.ome.tif")
    assert r["ref_channel"] == "1" and r["mov_channel"] == "1"  # Hoechst, where it is
    assert float(r["pixel_size_um"]) == pytest.approx(0.284)
    assert r["channels_ref"].split("|")[1] == "P1 : Hoechst"


def _args(cases, source, **kw):
    base = dict(
        cases=cases, source=source, groups=None, diameter=8.0, tile=512, max_cells=0,
        index=None, workers=1, force=False, nuclei_dir=None, nuclei_fallback="",
    )
    return argparse.Namespace(**{**base, **kw})


def test_prepare_writes_ibex_cases_with_the_pixel_size_source(tmp_path):
    from benchmarks.regbench.cases import list_cases

    src = _tree(tmp_path / "src")
    cases = tmp_path / "cases"
    assert ibex.prepare(_args(cases, src, groups=["Mouse_Spleen"])) == 0
    got = list_cases(cases, "ibex")
    assert [c.case_id for c in got] == ["Mouse_Spleen_P2", "Mouse_Spleen_P3"]
    c = got[0]
    assert c.dataset == "ibex" and c.modality == "fluorescence" and c.group == "Mouse_Spleen"
    assert c.pixel_size_um == pytest.approx(0.284)
    assert c.extra["ref_channel"] == 1 and c.extra["channels_mov"][1] == "P2 : Hoechst"
    assert "nrrd" in c.extra["pixel_size_source"]
    assert list_cases(cases, "multiplex") == []  # never pooled with the cohort
    assert json.loads((cases / "ibex" / "Mouse_Spleen_P2" / "case.json").read_text())["dataset"] == "ibex"


def test_human_pixel_size_comes_from_the_header_and_says_so(tmp_path):
    from benchmarks.regbench.cases import list_cases

    src = _tree(tmp_path / "src")
    assert ibex.prepare(_args(tmp_path / "cases", src, groups=["Human_Spleen"])) == 0
    (c,) = list_cases(tmp_path / "cases", "ibex")
    assert c.pixel_size_um == pytest.approx(0.5)
    assert "no independent record" in c.extra["pixel_size_source"]


def test_a_panel_whose_pixel_size_differs_from_its_reference_is_refused(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    write_ims(root / "Human_Spleen_Panel1.ims", [blobs(32, 32, 1)[None]], ["Hoechst"], X=32, Y=32)
    write_ims(
        root / "Human_Spleen_Panel2.ims", [blobs(32, 32, 2)[None]], ["Hoechst"],
        X=32, Y=32, ext=(0, 32.0, 0, 32.0),
    )
    with pytest.raises(ValueError, match="pixel size"):
        ibex.rows(ibex.discover(root), tmp_path / "cache", tile=TILE)


# ── registration of the dataset ───────────────────────────────────────────────
def test_ibex_is_an_external_cyclic_if_dataset_with_the_normal_run_list():
    assert "ibex" in EXTERNAL_DATASETS
    assert regimes.regime_of("ibex") == "cyclic_if"
    runs = variants.runs_for("ibex")
    assert any(r.method == "ashlar" for r in runs)
    cohort = {variants.run_name(r) for r in variants.runs_for(None)}
    assert {variants.run_name(r) for r in runs} == cohort  # a stitched slide set: every run
    assert variants.primary_for("ibex") == variants.primary_for(None) or variants.primary_for("ibex")
    variants.check()


# ── memory: bounded by a plane, not by the volume ────────────────────────────
def _peak(fn):
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_ims_conversion_memory_does_not_grow_with_the_volume(tmp_path):
    """Python/numpy allocations traced by ``tracemalloc`` (h5py's C buffers are not
    counted): the peak growth from a 1-plane to a 9x-larger-volume stand-in (more
    channels and Z planes, the SAME plane) stays under a quarter of the larger volume."""
    h = w = 160

    def make(name, c, z):
        p = tmp_path / name
        write_ims(
            p, [np.stack([blobs(h, w, 10 * i + j) for j in range(z)]) for i in range(c)],
            [f"c{i}" for i in range(c - 1)] + ["Hoechst"], X=w, Y=h,
        )
        return p

    small, big = make("s.ims", 2, 1), make("b.ims", 6, 3)  # 18 / 2 = 9x the voxels
    ibex.convert_cycle(small, tmp_path / "warm.ome.tif", tile=TILE)  # lazy imports
    a = _peak(lambda: ibex.convert_cycle(small, tmp_path / "s.ome.tif", tile=TILE))
    b = _peak(lambda: ibex.convert_cycle(big, tmp_path / "b.ome.tif", tile=TILE))
    volume = 6 * 3 * h * w
    print(f"ims peak small={a} big={b} growth={b - a} volume={volume} plane={h * w}")
    assert b - a < volume / 4
    assert b - a < 3 * h * w  # and bounded by about one output plane, not by C or Z


def test_nrrd_conversion_memory_does_not_grow_with_the_volume(tmp_path):
    """The same for the gzip NRRD, which is streamed: one Z plane of one channel at a
    time and a running max, so the peak is about two planes whatever C and Z are."""
    h, w = 160, 128

    def make(name, c, z):
        p = tmp_path / name
        write_nrrd(p, _vol(c, z, h, w), [f"c{i}" for i in range(c - 1)] + ["Hoechst"])
        return p

    small, big = make("s.nrrd", 2, 2), make("b.nrrd", 6, 6)  # 9x
    ibex.convert_cycle(small, tmp_path / "warm.ome.tif", tile=TILE)
    a = _peak(lambda: ibex.convert_cycle(small, tmp_path / "s.ome.tif", tile=TILE))
    b = _peak(lambda: ibex.convert_cycle(big, tmp_path / "b.ome.tif", tile=TILE))
    volume = 6 * 6 * h * w
    print(f"nrrd peak small={a} big={b} growth={b - a} volume={volume} plane={h * w}")
    assert b - a < volume / 4
    assert b - a < 3 * h * w  # and bounded by about one output plane, not by C or Z


# ── the commands ──────────────────────────────────────────────────────────────
def test_prepare_command_then_score_into_ibex_own_tables(tmp_path, capsys):
    import csv

    from benchmarks.regbench import cli, methods
    from benchmarks.regbench.cases import list_cases

    src = _tree(tmp_path / "src")
    cases, out = tmp_path / "cases", tmp_path / "out"
    assert cli.main([
        "prepare", "ibex", "--cases", str(cases), "--source", str(src),
        "--groups", "Mouse_Spleen", "--diameter", "8", "--tile", "512",
    ]) == 0
    assert "ibex/Mouse_Spleen_P2" in capsys.readouterr().out
    for case in list_cases(cases, "ibex"):
        methods.run_case("initial", case, out)
    assert cli.main([
        "score", "--cases", str(cases), "--out", str(out), "--dataset", "ibex",
        "--only-run", "--workers", "1",
    ]) == 0
    rows = list(csv.DictReader(open(out / "tables" / "ibex" / "summary_cells.csv")))
    assert any(r["method"] == "initial" for r in rows)
    assert not (out / "tables" / "summary.md").exists()  # never pooled with the cohort
    with pytest.raises(SystemExit, match="alone"):
        cli.main(["score", "--cases", str(cases), "--out", str(out), "--dataset",
                  "ibex", "synthetic", "--only-run"])
