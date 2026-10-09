"""Multiplex cases land in the shared contract with the right nuclei."""

import argparse
import math

import numpy as np
import pytest
import tifffile

from benchmarks.regbench import methods, pairs
from benchmarks.regbench.cases import list_cases, load_points, polys
from benchmarks.regbench.datasets import multiplex, synthetic
from benchmarks.regbench.score import score


def _two_rounds(tmp_path):
    spec = {**synthetic._BASE, "family": "t", "n": 768, "shift": (6.0, -4.0), "seed": 5}
    ref, mov, _ = synthetic.make_pair(spec)
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
                "PhysicalSizeX": 0.325,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.325,
                "PhysicalSizeYUnit": "µm",
            },
        )
        paths.append(p)
    return paths, synthetic.forward_map(spec)


def _mx_args(cases_root, manifest, **kw):
    base = dict(
        cases=cases_root,
        manifest=manifest,
        diameter=synthetic.NUCLEUS_DIAMETER,
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
    assert "landmarks_ref" not in pts
    ref, mov = polys(pts, "ref"), polys(pts, "mov")
    assert ref.n_features > 100 and mov.n_features > 100

    out = tmp_path / "out"
    methods.run_case("stare", case, out, opts={"workers": 1})
    res = score(cases_root, out, ["multiplex"])
    by = {r["method"]: r for r in res["cells"]}
    assert "truth" not in by  # no known map on real slides
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
