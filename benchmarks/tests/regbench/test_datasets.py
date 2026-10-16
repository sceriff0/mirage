"""Multiplex cases land in the shared contract with the right nuclei."""

import argparse
import csv
import json
import math

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import manifest as rounds_manifest
from benchmarks.regbench import methods, pairs, scene
from benchmarks.regbench.cases import list_cases, load_points, polys
from benchmarks.regbench.datasets import multiplex
from benchmarks.regbench.score import score

pytestmark = pytest.mark.usefixtures("local_stages")  # STARE runs in-process


def _two_rounds(tmp_path, names=None):
    spec = {**scene._BASE, "family": "t", "n": 768, "shift": (6.0, -4.0), "seed": 5}
    ref, mov, _ = scene.make_pair(spec)
    paths = []
    for name, plane in (("round1", ref), ("round2", mov)):
        p = tmp_path / f"{name}.ome.tif"
        # nuclear channel second, as in a real panel
        tifffile.imwrite(
            p,
            np.stack([np.zeros_like(plane), plane]),
            ome=True,
            metadata={
                "axes": "CYX",
                **({"Channel": {"Name": list(names)}} if names else {}),
                "PhysicalSizeX": 0.325,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.325,
                "PhysicalSizeYUnit": "µm",
            },
        )
        paths.append(p)
    return paths, scene.forward_map(spec)


def _mx_args(cases_root, manifest, **kw):
    base = dict(
        cases=cases_root,
        manifest=manifest,
        diameter=scene.NUCLEUS_DIAMETER,
        tile=512,
        max_cells=0,
        index=None,
        workers=1,
        force=False,
    )
    return argparse.Namespace(**{**base, **kw})


def test_multiplex_manifest_to_dice_and_displacement(tmp_path):
    (r1, r2), _ = _two_rounds(tmp_path)
    manifest = tmp_path / "rounds.csv"
    manifest.write_text(
        "case_id,reference,moving,group,ref_channel,mov_channel\n"
        f"P1_r2,{r1.name},{r2},P1,1,1\n"
    )  # a relative path resolves beside the CSV
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert case.pixel_size_um == pytest.approx(0.325)  # read from the OME header
    assert case.ref_hw == [768, 768] and case.group == "P1"
    assert tifffile.imread(case.ref_nuclear).squeeze().shape == (
        768,
        768,
    )  # the nuclear channel alone
    pts = load_points(case)
    assert "truth_ref" not in pts
    ref, mov = polys(pts, "ref"), polys(pts, "mov")
    assert ref.n_features > 100 and mov.n_features > 100

    out = tmp_path / "out"
    methods.run_case("stare", case, out, opts={"workers": 1})
    res = score(cases_root, out, ["multiplex"])
    by = {r["method"]: r for r in res["cells"]}
    assert by["initial"]["displacement_px_p50"] == pytest.approx(
        math.hypot(6, 4), abs=0.5
    )
    assert by["stare"]["displacement_px_p50"] < 0.5
    assert by["stare"]["displacement_um_p50"] == pytest.approx(
        by["stare"]["displacement_px_p50"] * 0.325
    )
    assert by["stare"]["dice_matched"] > 0.9 > by["initial"]["dice_matched"]
    assert by["stare"]["within_1um"] > by["initial"]["within_1um"]


def test_multiplex_accepts_your_own_masks_and_can_sample_cells(tmp_path):
    (r1, r2), _ = _two_rounds(tmp_path)
    labels = np.zeros((768, 768), np.int32)
    for k, (y, x) in enumerate([(100, 100), (300, 420), (600, 250)], start=1):
        labels[y - 8 : y + 8, x - 8 : x + 8] = k
    lab = tmp_path / "ref_labels.tif"
    tifffile.imwrite(lab, labels)
    manifest = tmp_path / "rounds.csv"
    manifest.write_text(
        "case_id,reference,moving,ref_channel,mov_channel,ref_cells\n"
        f"a,{r1},{r2},1,1,{lab}\n"
    )
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest, max_cells=50)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    pts = load_points(case)
    _, cent = pairs.cp.feature_area_centroid(polys(pts, "ref"))
    # the provided mask, not the built-in segmenter
    assert sorted(cent[:, 1].round()) == [100, 300, 600]
    assert polys(pts, "mov").n_features == 50


# ── the cohort: original slides and the QC nuclei as GeoJSON ──────────────────
def _square(cx, cy, half=6.0):
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]  # closed, as GeoJSON


def _feature(geometry_type, coordinates):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


def _cell_file(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _grid_cells(path, shift=(0.0, 0.0), n=6):
    centres = [
        (100 + 90 * i + shift[0], 100 + 90 * j + shift[1])
        for i in range(n)
        for j in range(n)
    ]
    return _cell_file(path, [_feature("Polygon", [_square(x, y)]) for x, y in centres])


def _cohort_manifest(tmp_path, rows):
    """Rows ``(case_id, ref_cells, mov_cells)`` on the same two rounds of one patient."""
    (r1, r2), _ = _two_rounds(tmp_path)
    manifest = tmp_path / "rounds.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            "case_id reference moving group ref_channel mov_channel channels_ref "
            "channels_mov pixel_size_um ref_cells mov_cells".split()
        )
        for case_id, ref_cells, mov_cells in rows:
            w.writerow(
                [
                    case_id,
                    r1,
                    r2,
                    "P1",
                    1,
                    1,
                    "CD3|DAPI",
                    "CD8|DAPI",
                    0.325,
                    ref_cells,
                    mov_cells,
                ]
            )
    return manifest, (r1, r2)


def test_empty_cell_file_skips_only_that_case(tmp_path, capsys):
    good = _grid_cells(tmp_path / "good.geojson")
    empty = _cell_file(tmp_path / "empty.geojson", [])
    manifest, _ = _cohort_manifest(tmp_path, [("bad", good, empty), ("ok", good, good)])
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    assert [c.case_id for c in list_cases(cases_root, "multiplex")] == ["ok"]
    said = capsys.readouterr().out
    assert "multiplex/bad: FAILED" in said and str(empty) in said


def test_cells_outside_the_slide_skip_only_that_case(tmp_path, capsys):
    good = _grid_cells(tmp_path / "good.geojson")
    # nuclei of a larger slide: one vertex past the right edge of this 768 px slide
    wrong = _cell_file(
        tmp_path / "wrong.geojson",
        [
            _feature("Polygon", [_square(100, 100)]),
            _feature("Polygon", [_square(765, 300)]),
        ],
    )
    manifest, _ = _cohort_manifest(tmp_path, [("bad", wrong, good), ("ok", good, good)])
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    assert [c.case_id for c in list_cases(cases_root, "multiplex")] == ["ok"]
    said = capsys.readouterr().out
    assert "multiplex/bad: FAILED" in said and str(wrong) in said
    assert "outside" in said


def test_every_case_refused_is_an_error(tmp_path):
    empty = _cell_file(tmp_path / "empty.geojson", [])
    manifest, _ = _cohort_manifest(tmp_path, [("bad", empty, empty)])
    assert multiplex.prepare(_mx_args(tmp_path / "cases", manifest)) == 1


def test_border_nuclei_at_minus_half_are_accepted(tmp_path):
    # the pipeline's writer traces a border nucleus half a pixel outside the raster:
    # -0.5 on the top and left, size - 0.5 on the bottom and right (bin/mask_to_geojson.py)
    border = _cell_file(
        tmp_path / "border.geojson",
        [
            _feature(
                "Polygon", [[[-0.5, -0.5], [8, -0.5], [8, 8], [-0.5, 8], [-0.5, -0.5]]]
            ),
            _feature(
                "Polygon",
                [[[760, 760], [767.5, 760], [767.5, 767.5], [760, 767.5], [760, 760]]],
            ),
        ],
    )
    manifest, _ = _cohort_manifest(tmp_path, [("edge", border, border)])
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    ref = polys(load_points(case), "ref")
    assert ref.n_features == 2 and ref.xy.min() == -0.5 and ref.xy.max() == 767.5


def test_holes_and_multipart_cells_survive_the_round_trip(tmp_path):
    ring = _square(200, 200, 20)
    hole = _square(200, 200, 5)
    cells_file = _cell_file(
        tmp_path / "shapes.geojson",
        [
            _feature("Polygon", [ring, hole]),
            _feature("MultiPolygon", [[_square(400, 400)], [_square(430, 400)]]),
            _feature("Polygon", [_square(600, 600)]),
        ],
    )
    manifest, _ = _cohort_manifest(tmp_path, [("shapes", cells_file, cells_file)])
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    for side in ("ref", "mov"):
        ps = polys(points, side)
        assert ps.n_features == 3
        assert ps.ring_hole.tolist() == [False, True, False, False, False]
        assert ps.ring_feat.tolist() == [0, 0, 1, 1, 2]
        area, _ = pairs.cp.feature_area_centroid(ps)
        # the hole is taken out, both parts of the multi-part cell are counted
        assert area == pytest.approx([40 * 40 - 10 * 10, 2 * 12 * 12, 12 * 12])
    assert case.extra["cells"] == "provided"
    # a second prepare reads the cached nuclei and writes the same case
    assert multiplex.prepare(_mx_args(cases_root, manifest, force=False)) == 0
    assert polys(load_points(case), "ref").ring_hole.any()


def test_sampling_is_refused_for_nuclei_given_as_geojson(tmp_path, capsys):
    good = _grid_cells(tmp_path / "good.geojson")
    manifest, _ = _cohort_manifest(tmp_path, [("a", good, good)])
    assert multiplex.prepare(_mx_args(tmp_path / "cases", manifest, max_cells=5)) == 1
    assert "--max-cells" in capsys.readouterr().out


def test_case_points_at_the_original_slide(tmp_path, capsys):
    """From regbench's own input step (the pipeline's preprocessing, laid out as it
    publishes it), through the manifest and the traced outlines, to the case."""
    (r1, r2), _ = _two_rounds(tmp_path)
    root = tmp_path / "inputs" / "preprocess" / "results"
    slides = root / "P1" / "converted"
    slides.mkdir(parents=True)
    ref, mov = slides / "P1.r1.ome.tif", slides / "P1.r2.ome.tif"
    r1.rename(ref)
    r2.rename(mov)
    checkpoint = root / "csv" / "preprocessed.csv"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        "patient_id,id,preprocessed_image,is_reference,channels,pixel_size\n"
        f"P1,P1.r1,{ref},true,CD3|DAPI,0.325\n"
        f"P1,P1.r2,{mov},false,CD8|DAPI,0.325\n"
    )
    outlines = tmp_path / "inputs" / "outlines"
    manifest = tmp_path / "rounds.csv"
    argv = ["--preprocessed", str(checkpoint), "--outlines", str(outlines)]
    assert rounds_manifest.main([*argv, "--out", str(manifest)]) == 0

    # the outlines are not traced yet: the round is DROPPED with the reason, never
    # prepared on an empty set
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 1
    said = capsys.readouterr().out
    assert (
        f"DROPPED {multiplex.NOT_TRACED}: {outlines / 'P1' / 'P1.r1.geojson'}" in said
    )
    assert list_cases(cases_root, "multiplex") == []
    # what the nuclei step leaves (its own trace of each slide's StarDist cell mask)
    _grid_cells(outlines / "P1" / "P1.r1.geojson")
    _grid_cells(outlines / "P1" / "P1.r2.geojson", shift=(6.0, -4.0))
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert case.case_id == "P1_P1.r2" and case.group == "P1"
    # the ORIGINAL multichannel slides, for a method that reads them itself
    assert case.ref_image == str(ref) and case.mov_image == str(mov)
    assert tifffile.imread(case.ref_image).shape == (2, 768, 768)
    assert case.extra["ref_channel"] == 1 and case.extra["mov_channel"] == 1
    assert case.extra["channels_ref"] == ["CD3", "DAPI"]
    assert case.extra["channels_mov"] == ["CD8", "DAPI"]
    assert case.extra["channel_names"] == ["CD8", "DAPI"]  # of the slide that is warped
    assert case.extra["pixel_size_source"] == "preprocessed.csv"
    assert case.pixel_size_um == pytest.approx(0.325)
    assert case.ref_hw == [768, 768] and case.mov_hw == [768, 768]
    # the nuclear channel alone, for a method that must be given one channel
    assert tifffile.imread(case.ref_nuclear).squeeze().shape == (768, 768)
    assert case.ref_nuclear != case.ref_image
    assert (
        tifffile.imread(case.mov_nuclear).squeeze()
        == tifffile.imread(case.mov_image)[1]
    ).all()

    out = tmp_path / "out"
    methods.run_case("initial", case, out)
    (row,) = score(cases_root, out, ["multiplex"])["cells"]
    assert row["n_ref"] == 36 and row["n_moving"] == 36
    assert row["displacement_px_p50"] == pytest.approx(math.hypot(6, 4), abs=1e-3)


def test_channel_names_come_from_the_slide_when_the_manifest_has_none(tmp_path):
    (r1, r2), _ = _two_rounds(tmp_path, names=("CD3", "DAPI"))
    manifest = tmp_path / "rounds.csv"
    manifest.write_text(
        f"case_id,reference,moving,ref_channel,mov_channel\na,{r1},{r2},1,1\n"
    )
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert case.extra["channels_ref"] == ["CD3", "DAPI"]
    assert case.extra["channel_names"] == ["CD3", "DAPI"]
    assert case.extra["pixel_size_source"] == "OME header"


def test_a_case_prepared_from_other_inputs_is_refused_not_kept(tmp_path, capsys):
    """A cases folder prepared from the arms' manifest, then prepared from regbench's
    own: the old case is neither kept silently nor replaced unasked."""
    good = _grid_cells(tmp_path / "good.geojson")
    manifest, _ = _cohort_manifest(tmp_path, [("a", good, good)])
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest, force=False)) == 0
    # the same row again: the case is kept
    assert multiplex.prepare(_mx_args(cases_root, manifest, force=False)) == 0
    other = _grid_cells(tmp_path / "other" / "good.geojson")
    manifest.write_text(manifest.read_text().replace(str(good), str(other)))
    assert multiplex.prepare(_mx_args(cases_root, manifest, force=False)) == 1
    said = capsys.readouterr().out
    assert "was prepared from ref_cells" in said and "--force" in said
    assert multiplex.prepare(_mx_args(cases_root, manifest, force=True)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert case.extra["ref_cells"] == str(other)
