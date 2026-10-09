"""ANHIR and multiplex cases land in the shared contract with the right points."""

import argparse
import csv
import math

import numpy as np
import pytest
import tifffile
from regbench import cells, methods
from regbench.cases import list_cases, load_points, polys
from regbench.datasets import anhir, multiplex, synthetic
from regbench.score import score

COVER_HEADER = [
    "", "Image diagonal [pixels]", "Image size [pixels]", "Source image", "Source landmarks",
    "Target image", "Target landmarks", "status", "Warped target landmarks",
    "Warped source landmarks", "Execution time [minutes]",
]


def _anhir_root(root):
    """A two-case ANHIR layout: one training pair with landmarks, one evaluation pair."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for name, hw in (("HE", (300, 400)), ("IHC", (280, 420))):
        d = root / "images" / "T_1" / "scale-25pc"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(120, 255, (*hw, 3), dtype=np.uint8)).save(d / f"{name}.jpg")
    lm = root / "landmarks" / "T_1" / "scale-25pc"
    lm.mkdir(parents=True)
    src = rng.uniform(20, 250, (12, 2))
    for name, xy in (("HE", src), ("IHC", src + [30.0, 40.0])):
        with open(lm / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["", "X", "Y"])
            w.writerows([i, x, y] for i, (x, y) in enumerate(xy))
    with open(root / "dataset_medium.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COVER_HEADER)
        for cid, status in ((7, "training"), (8, "evaluation")):
            w.writerow([cid, 500.0, "(400, 300)", "T_1/scale-25pc/HE.jpg", "T_1/scale-25pc/HE.csv",
                        "T_1/scale-25pc/IHC.jpg", "T_1/scale-25pc/IHC.csv", status, "", "", ""])
    return src


def test_anhir_training_cases_carry_source_landmarks_and_both_modalities(tmp_path):
    data, cases_root = tmp_path / "anhir", tmp_path / "cases"
    src = _anhir_root(data)
    a = argparse.Namespace(cases=cases_root, data_root=data, proxy="lum", tissue=None, case_id=None,
                           limit=None, index=None, workers=1, force=False)
    assert anhir.prepare(a) == 0
    (case,) = list_cases(cases_root, "anhir")  # the evaluation case has no public target
    assert case.case_id == "7" and case.group == "T" and case.modality == "brightfield"
    assert case.mov_image.endswith("HE.jpg") and case.ref_image.endswith("IHC.jpg")
    assert case.mov_hw == [300, 400] and case.ref_hw == [280, 420]
    assert tifffile.imread(case.ref_nuclear).shape[-2:] == (280, 420)
    pts = load_points(case)
    assert pts["landmarks_mov"] == pytest.approx(src)
    assert pts["landmarks_ref"] == pytest.approx(src + [30.0, 40.0])

    res = score(cases_root, tmp_path / "out", ["anhir"])
    (row,) = res["landmarks"]
    assert row["method"] == "initial"
    assert row["rtre_median"] == pytest.approx(50.0 / 500.0)  # the cover table's diagonal
    assert not res["cells"]  # no nuclei to pair across stains


def _two_rounds(tmp_path):
    spec = {**synthetic._BASE, "family": "t", "n": 768, "shift": (6.0, -4.0), "seed": 5}
    ref, mov, _ = synthetic.make_pair(spec)
    paths = []
    for name, plane in (("round1", ref), ("round2", mov)):
        p = tmp_path / f"{name}.ome.tif"
        # nuclear channel second, as in a real panel
        tifffile.imwrite(p, np.stack([np.zeros_like(plane), plane]), ome=True,
                         metadata={"axes": "CYX", "PhysicalSizeX": 0.325, "PhysicalSizeXUnit": "µm",
                                   "PhysicalSizeY": 0.325, "PhysicalSizeYUnit": "µm"})
        paths.append(p)
    return paths, synthetic.forward_map(spec)


def _mx_args(cases_root, manifest, **kw):
    base = dict(cases=cases_root, manifest=manifest, diameter=synthetic.NUCLEUS_DIAMETER, tile=512,
                max_cells=0, index=None, workers=1, force=False)
    return argparse.Namespace(**{**base, **kw})


def test_multiplex_manifest_to_dice_and_displacement(tmp_path):
    (r1, r2), _ = _two_rounds(tmp_path)
    manifest = tmp_path / "rounds.csv"
    manifest.write_text("case_id,reference,moving,group,ref_channel,mov_channel\n"
                        f"P1_r2,{r1.name},{r2},P1,1,1\n")  # a relative path resolves beside the CSV
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert case.pixel_size_um == pytest.approx(0.325)  # read from the OME header
    assert case.ref_hw == [768, 768] and case.group == "P1"
    assert tifffile.imread(case.ref_nuclear).squeeze().shape == (768, 768)  # the nuclear channel alone
    pts = load_points(case)
    assert "landmarks_ref" not in pts
    ref, mov = polys(pts, "cells_ref"), polys(pts, "cells_mov")
    assert len(ref) > 100 and len(mov) > 100

    out = tmp_path / "out"
    methods.run_case("stare", case, out, workers=1)
    res = score(cases_root, out, ["multiplex"])
    by = {r["method"]: r for r in res["cells"]}
    assert not res["landmarks"]
    assert by["initial"]["displacement_px_p50"] == pytest.approx(math.hypot(6, 4), abs=0.5)
    assert by["stare"]["displacement_px_p50"] < 0.5
    assert by["stare"]["displacement_um_p50"] == pytest.approx(by["stare"]["displacement_px_p50"] * 0.325)
    assert by["stare"]["dice_matched"] > 0.9 > by["initial"]["dice_matched"]


def test_multiplex_accepts_your_own_masks_and_can_sample_cells(tmp_path):
    (r1, r2), _ = _two_rounds(tmp_path)
    labels = np.zeros((768, 768), np.int32)
    for k, (y, x) in enumerate([(100, 100), (300, 420), (600, 250)], start=1):
        labels[y - 8 : y + 8, x - 8 : x + 8] = k
    lab = tmp_path / "ref_labels.tif"
    tifffile.imwrite(lab, labels)
    manifest = tmp_path / "rounds.csv"
    manifest.write_text("case_id,reference,moving,ref_channel,mov_channel,ref_cells\n"
                        f"a,{r1},{r2},1,1,{lab}\n")
    cases_root = tmp_path / "cases"
    assert multiplex.prepare(_mx_args(cases_root, manifest, max_cells=50)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    pts = load_points(case)
    _, cent = cells.area_centroid(polys(pts, "cells_ref"))
    assert sorted(cent[:, 1].round()) == [100, 300, 600]  # the provided mask, not the built-in segmenter
    assert len(polys(pts, "cells_mov")) == 50
