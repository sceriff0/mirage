"""Nuclei and outlines for the cohort: regbench's own StarDist run of each slide, its
cell mask traced to the slide's outlines, the nuclei checked against them; the shrunk
outline is a fallback that is never used unasked, and a slide without nuclei is prepared
and scored on outlines.

StarDist is not installed here: the run is driven with a stand-in ``segment.py`` that
writes the two masks the real one writes, under the names the real one gives them.
"""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import cases as cases_mod
from benchmarks.regbench import cells, nuclei, outlines, pairs, site_config
from benchmarks.regbench.cases import list_cases, load_points, polys
from benchmarks.regbench.datasets import multiplex
from benchmarks.regbench.imageio import write_nuclear

REPO = Path(__file__).resolve().parents[3]
HW = (360, 420)


@pytest.fixture(autouse=True)
def no_site_config(monkeypatch, tmp_path):
    """No test reads the real site overlay (conf/ieo.config exists on a cluster
    checkout): the default is a path that is not there, and the variable is unset."""
    monkeypatch.setattr(site_config, "DEFAULT_SITE_CONFIG", tmp_path / "no.config")
    monkeypatch.delenv(site_config.ENV_SITE_CONFIG, raising=False)


PITCH = 40


def labels_of(shift=(0, 0), drop=0, hw=HW):
    """Nuclei 9 px a side on a 40 px pitch (so the 10 px growth never meets a neighbour)."""
    lab = np.zeros(hw, dtype=np.uint32)
    k = 0
    for y in range(30, hw[0] - 30, PITCH):
        for x in range(30, hw[1] - 30, PITCH):
            k += 1
            if k <= drop:
                continue
            lab[y + shift[1] : y + shift[1] + 9, x + shift[0] : x + shift[0] + 9] = k
    return lab


def geojson_of(labels, path):
    """The outlines of a label image as a pipeline arm published them: one Polygon each."""
    ps = cells.label_polys(labels)
    features = []
    for k in range(len(ps)):
        ring = ps.xy[ps.off[k] : ps.off[k + 1]].tolist()
        features.append(
            {
                "type": "Feature",
                "properties": {"label": k + 1},
                "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.fixture
def cohort(tmp_path):
    """Two rounds of one patient: slides and a manifest row as ``regbench.manifest``
    writes it, its outlines named under ``outlines/`` and not traced yet."""
    lab = labels_of()
    slides = {}
    for name in ("P1.r1", "P1.r2"):
        slides[name] = write_nuclear(
            tmp_path / "slides" / f"{name}.ome.tif", (lab > 0).astype(np.uint16), 0.325
        )
    row = {
        "case_id": "P1_P1.r2",
        "reference": str(slides["P1.r1"]),
        "moving": str(slides["P1.r2"]),
        "group": "P1",
        "ref_channel": 0,
        "mov_channel": 0,
        "pixel_size_um": 0.325,
        "ref_cells": str(tmp_path / "outlines" / "P1" / "P1.r1.geojson"),
        "mov_cells": str(tmp_path / "outlines" / "P1" / "P1.r2.geojson"),
    }
    return tmp_path, lab, row


def published(tmp, lab, row):
    """The row of a manifest written from the arms' results before 2026-10-16: its
    outlines are the arm's PUBLISHED GeoJSON (no ``.traced.json`` beside them)."""
    out = dict(row)
    for side, name in (("ref", "P1.r1"), ("mov", "P1.r2")):
        out[f"{side}_cells"] = str(
            geojson_of(cells.grow(lab), tmp / "arm" / "geojson" / f"{name}.geojson")
        )
    return out


def manifest_of(tmp_path, row, **more):
    row = {**row, **more}
    path = tmp_path / "rounds.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return path


def args(**kw):
    base = dict(
        workers=1,
        diameter=20.0,
        tile=256,
        force=False,
        index=None,
        list=False,
        count=False,
        command=False,
        segment=False,
        settings=False,
    )
    return argparse.Namespace(**{**base, **kw})


def mx_args(cases_root, manifest, **kw):
    base = dict(
        cases=cases_root,
        manifest=manifest,
        diameter=20.0,
        tile=256,
        max_cells=0,
        index=None,
        workers=1,
        force=False,
        nuclei_dir=None,
        nuclei_fallback="",
    )
    return argparse.Namespace(**{**base, **kw})


# ── geometry ──────────────────────────────────────────────────────────────────
def test_a_point_is_inside_a_feature_and_outside_its_hole():
    outer = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]
    hole = [[8, 8], [12, 8], [12, 12], [8, 12], [8, 8]]
    other = [[30, 0], [40, 0], [40, 10], [30, 10], [30, 0]]
    fc = {
        "features": [
            {"geometry": {"type": "Polygon", "coordinates": [outer, hole]}},
            {"geometry": {"type": "Polygon", "coordinates": [other]}},
        ]
    }
    ps = pairs.cp.from_feature_collection(fc)
    pts = np.array([[3.0, 3.0], [10.0, 10.0], [35.0, 5.0], [35.0, 5.0], [50.0, 50.0]])
    feat = np.array([0, 0, 1, 0, 1])
    assert cells.inside(ps, feat, pts).tolist() == [True, False, True, False, False]
    # the same answers whatever the chunking
    assert cells.inside(ps, feat, pts, chunk=1).tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_the_true_nuclei_put_one_centroid_in_every_outline():
    lab = labels_of()
    outlines, nuc = cells.label_polys(cells.grow(lab)), cells.label_polys(lab)
    _, cent = cells.area_centroid(nuc)
    assert nuclei.one_nucleus_fraction(outlines, cent) == 1.0
    # the same nuclei half a pitch away: between the outlines
    assert nuclei.one_nucleus_fraction(outlines, cent + PITCH / 2) < 0.1
    # two nuclei in one outline is not "exactly one"
    doubled = np.concatenate([cent, cent[:10] + 1.0])
    assert nuclei.one_nucleus_fraction(outlines, doubled) == pytest.approx(
        1 - 10 / len(cent)
    )


def test_shrinking_an_outline_takes_the_rim_off_and_says_where_it_came_from():
    lab = labels_of()
    # an outline narrower than two rims vanishes: it has no entry, and no parent
    back, parent = cells.shrink(cells.label_polys(cells.grow(lab, 3)), 10)
    assert len(back) == 0 and len(parent) == 0
    grown = cells.label_polys(cells.grow(lab))
    back, parent = cells.shrink(grown, 10)
    assert len(parent) == len(back) == len(grown)
    assert len(set(parent.tolist())) == len(parent)
    area, cent = cells.area_centroid(back)
    _, cent_out = cells.area_centroid(grown)
    assert np.allclose(cent, cent_out[parent], atol=1.0)
    assert np.median(np.sqrt(area)) == pytest.approx(9.0, abs=2.5)


def _lattice(n_side, pitch=40.0, half=14.0):
    """``n_side ** 2`` square outlines on a lattice, as one polygon set, and centres."""
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    cent = np.stack([gx.ravel(), gy.ravel()], axis=1) * pitch + pitch
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    xy = (cent[:, None, :] + corners[None]).reshape(-1, 2)
    off = np.arange(0, 4 * len(cent) + 1, 4, dtype=np.int64)
    return cells.Polys(xy, off), cent


def test_the_outline_check_scales_with_the_number_of_nuclei():
    """A cohort slide has hundreds of thousands of nuclei: the check must not be
    quadratic. 250 000 outlines in seconds, and four times the nuclei in well under
    sixteen times the time."""
    import time

    took = {}
    for n_side in (250, 500):
        outlines, cent = _lattice(n_side)
        t0 = time.perf_counter()
        assert nuclei.one_nucleus_fraction(outlines, cent + 3.0) == 1.0
        took[n_side] = time.perf_counter() - t0
    assert took[500] < 60.0, took
    assert took[500] < 8.0 * max(took[250], 0.05), took


# ── the re-run ────────────────────────────────────────────────────────────────
ENV = {
    "REGBENCH_STARDIST_MODEL": "site_model",
    "REGBENCH_STARDIST_MODEL_DIR": "/models",
}


def test_the_rerun_is_the_command_the_pipeline_renders(cohort):
    tmp, lab, row = cohort
    (ref, _) = nuclei.slides([row])
    argv = nuclei.command(ref, tmp / "dest", ENV)
    assert argv == [
        "python",
        str(REPO / "bin" / "segment.py"),
        "--image",
        row["reference"],
        "--output-dir",
        str(tmp / "dest"),
        "--dapi-channel",
        "0",
        "--model-dir",
        "/models",
        "--model-name",
        "site_model",
        "--n-tiles",
        "16",
        "16",
        "--expand-distance",
        "10",
        "--pmin",
        "1.0",
        "--pmax",
        "99.8",
        "--use-gpu",
    ]
    more = nuclei.command(
        ref, tmp, {**ENV, "REGBENCH_STARDIST_PROB_THRESH": "0.3", nuclei.ENV_CPU: "1"}
    )
    assert more[-2:] == ["--prob-thresh", "0.3"] and "--use-gpu" not in more
    with pytest.raises(SystemExit, match="REGBENCH_STARDIST_MODEL is not set"):
        nuclei.command(ref, tmp, {})


def test_the_rerun_follows_the_pipelines_own_configuration():
    """A guard: the values are nextflow.config's defaults, every flag is one the
    pipeline's ``ext.args`` renders for StarDist, and ``segment.py`` takes each."""
    config = (REPO / "nextflow.config").read_text()

    def default(name):
        (value,) = re.findall(rf"^\s*{name}\s*=\s*([^\s/]+)", config, re.M)
        return value

    assert (int(default("seg_n_tiles_y")), int(default("seg_n_tiles_x"))) == (
        nuclei.STARDIST["n_tiles"]
    )
    assert int(default("seg_expand_distance")) == nuclei.STARDIST["expand_distance"]
    assert float(default("seg_pmin")) == nuclei.STARDIST["pmin"]
    assert float(default("seg_pmax")) == nuclei.STARDIST["pmax"]
    assert default("seg_prob_thresh") == "null" and default("seg_gpu") == "true"
    slide = {"image": "/s/a.ome.tif", "channel": 0}
    argv = nuclei.command(slide, "/d", {**ENV, "REGBENCH_STARDIST_PROB_THRESH": "0.5"})
    flags = [a for a in argv if a.startswith("--")]
    modules = (REPO / "conf" / "modules.config").read_text()
    stardist = modules[modules.index("stardist  : [") : modules.index("instantseg: [")]
    rendered = set(re.findall(r'"(--[a-z-]+) ', stardist)) | {"--use-gpu"}
    backend = (REPO / "lib" / "SegBackends.groovy").read_text()
    assert "flags     : { ctx -> '--dapi-channel 0' }" in backend
    process = (REPO / "modules" / "local" / "segment.nf").read_text()
    assert "--image ${merged_file}" in process and "--output-dir ." in process
    assert set(flags) == rendered | {"--image", "--output-dir", "--dapi-channel"}
    script = (REPO / "bin" / "segment.py").read_text()
    for flag in flags:
        assert f'"{flag}"' in script, flag
    assert nuclei.PIPELINE_IMAGE in backend


FAKE = """
import argparse, sys
from pathlib import Path
import numpy as np, tifffile
from skimage.segmentation import expand_labels
ap = argparse.ArgumentParser()
ap.add_argument("--image"); ap.add_argument("--output-dir")
ap.add_argument("--expand-distance", type=int)
a, rest = ap.parse_known_args()
Path(a.output_dir, "argv.txt").write_text(" ".join(sys.argv[1:]))
lab = np.load({labels!r})
grown = np.load({cells!r})
stem = Path(a.image).stem
tifffile.imwrite(Path(a.output_dir) / f"{{stem}}_nuclei_mask.tif", lab)
tifffile.imwrite(Path(a.output_dir) / f"{{stem}}_cell_mask.tif",
                 expand_labels(grown, distance=a.expand_distance).astype(np.uint32))
"""


def fake_segment(tmp, labels, monkeypatch, cells_from=None):
    """A stand-in ``segment.py``: ``labels`` are its nuclei and its cell mask is
    ``cells_from`` (default: the same nuclei) grown by ``--expand-distance``. A cell mask
    grown from OTHER labels is what makes the traced outlines disagree with the
    nuclei: the self-consistency check must catch it."""
    np.save(tmp / "labels.npy", labels)
    np.save(tmp / "cells.npy", labels if cells_from is None else cells_from)
    script = tmp / "fake_segment.py"
    script.write_text(
        FAKE.format(labels=str(tmp / "labels.npy"), cells=str(tmp / "cells.npy"))
    )
    monkeypatch.setattr(nuclei, "SEGMENT_PY", script)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_a_slide_without_a_mask_is_segmented_and_checked(cohort, monkeypatch, capsys):
    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch)
    manifest = manifest_of(tmp, row)
    assert not Path(row["ref_cells"]).exists()  # nothing is traced before the run
    a = args(manifest=manifest, out=tmp / "nuc", segment=True)
    assert nuclei.main(a) == 0
    for slide in nuclei.slides(multiplex.read_manifest(manifest)):
        rec = nuclei.load(nuclei.store_of(tmp / "nuc", slide))
        assert rec["source"] == "stardist_rerun" and rec["usable"] and rec["exact"]
        assert rec["outlines_from"] == "stardist_rerun_cell_mask"
        assert rec["tolerance"] == 1.0 and rec["outlines"] == slide["outlines"]
        assert rec["agreement"]["centroid_match_frac"] == 1.0
        assert rec["agreement"]["n_cells_rerun"] == rec["agreement"]["n_outlines"]
        assert rec["agreement"]["n_outlines"] == lab.max()
        stem = Path(slide["image"]).stem  # the name bin/segment.py gives its masks
        rerun = nuclei.store_of(tmp / "nuc", slide) / "rerun"
        assert (rerun / f"{stem}_nuclei_mask.tif").is_file() and stem.endswith(".ome")
        assert "--expand-distance 10" in (rerun / "argv.txt").read_text()
        assert shlex.split(rec["command"])[1] == str(nuclei.SEGMENT_PY)
        # the outlines: the run's own cell mask through the pipeline's own tracer
        made = nuclei.traced_record(slide["outlines"])
        assert made["cell_mask"] == str(rerun / f"{stem}_cell_mask.tif")
        assert made["image_fingerprint"] == rec["image_fingerprint"]
        theirs = tmp / "theirs.geojson"
        outlines.trace(made["cell_mask"], theirs, 1.0)
        assert Path(slide["outlines"]).read_text() == theirs.read_text()
    assert "DISAGREEMENT" not in capsys.readouterr().out


def test_a_rerun_that_disagrees_with_the_outlines_is_loud_and_not_exact(
    cohort, monkeypatch, capsys
):
    tmp, lab, row = cohort
    # nuclei 5 short of the cell mask they were said to grow: 51 of 56 traced outlines
    # hold one (0.911), kept, not exact (under 0.90 it is refused)
    fake_segment(tmp, labels_of(drop=5), monkeypatch, cells_from=lab)
    a = args(manifest=manifest_of(tmp, row), out=tmp / "nuc", segment=True, index=0)
    assert nuclei.main(a) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(a.manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    assert rec["usable"] and not rec["exact"]
    assert not rec["agreement"]["exact"] and rec["agreement"]["n_outlines"] == lab.max()
    said = capsys.readouterr().out
    assert "DISAGREEMENT P1/P1.r1" in said and "a tracing bug" in said


def test_a_rerun_far_from_the_outlines_is_kept_and_flagged(cohort, monkeypatch, capsys):
    """Task 28, B9 (was the 2026-10-11 refusal): under 0.90 of the outlines with
    exactly one nucleus the nuclei are KEPT, flagged ``low_agreement`` and every place
    that lists them says so; nothing is blocked. Against their own trace it can only
    be a tracing bug: made here with a cell mask shifted by 20 px."""
    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch, cells_from=labels_of(shift=(20, 20)))
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, segment=True, index=0)) == 0  # never fails
    (ref, mov) = nuclei.slides(multiplex.read_manifest(manifest))
    store = nuclei.store_of(tmp / "nuc", ref)
    rec = nuclei.load(store)
    frac = rec["agreement"]["one_nucleus_frac"]
    assert frac < nuclei.ACCEPT_ONE_NUCLEUS
    assert rec["usable"] and rec["low_agreement"] and not rec.get("refused")
    assert not rec["exact"] and rec["source"] == "stardist_rerun"
    assert (store / "nuclei.npz").is_file()
    n_out, n_nuc = rec["agreement"]["n_outlines"], rec["agreement"]["n_nuclei"]
    why = rec["low_agreement_reason"]
    assert f"only {frac:.3f} of this slide's {n_out} outlines" in why
    assert f"{n_nuc} nuclei (at least 0.9 is required)" in why
    assert "a tracing bug" in why and "kept and flagged" in why
    said = capsys.readouterr().out
    assert f"LOW_AGREEMENT P1/P1.r1: {why}" in said and "REFUSED" not in said
    # the list: its own status, the share and the reason; and nuclei_checks.csv
    assert nuclei.main(args(**base, list=True)) == 0
    listed = capsys.readouterr()
    line = listed.out.splitlines()[1].split("\t")
    assert line[:3] == ["0", "P1/P1.r1", "low_agreement"] and line[4] == "False"
    assert line[5] == f"{frac:.4f}" and "LOW AGREEMENT" in line[7]
    assert "1 kept and flagged LOW AGREEMENT" in listed.err
    with open(tmp / "nuc" / nuclei.CHECKS) as fh:
        checks = list(csv.DictReader(fh))
    (got,) = [r for r in checks if r["slide"] == ref["slide"]]
    assert got["patient_id"] == "P1" and got["low_agreement"] == "1"
    assert float(got["one_nucleus_fraction"]) == round(frac, 4)
    assert float(got["threshold"]) == nuclei.ACCEPT_ONE_NUCLEUS
    # it is not segmented again unasked, and no command is printed for it
    calls, real = [], nuclei.segment
    monkeypatch.setattr(nuclei, "segment", lambda *a: calls.append(a) or real(*a))
    assert nuclei.main(args(**base, segment=True, index=0)) == 0 and not calls
    capsys.readouterr()
    assert nuclei.main(args(**base, command=True, index=0)) == 0
    assert capsys.readouterr().out == ""
    assert 0 not in nuclei.pending(nuclei.slides(multiplex.read_manifest(manifest)),
                                   tmp / "nuc")  # fmt: skip
    # the case is prepared WITH those nuclei, and its record carries the flag
    fake_segment(tmp, lab, monkeypatch)
    assert nuclei.main(args(**base, segment=True, index=1)) == 0
    cases_root = tmp / "cases"
    a = mx_args(cases_root, manifest, nuclei_dir=str(tmp / "nuc"))
    assert multiplex.prepare(a) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    assert polys(points, "ref", "nuclei") is not None
    info = cases_mod.set_info(case, "ref", "nuclei")
    assert info["low_agreement"] is True and info["low_agreement_reason"] == why
    assert not cases_mod.set_info(case, "mov", "nuclei").get("low_agreement")


def test_a_record_an_earlier_checkout_refused_is_checked_again_and_kept(
    cohort, monkeypatch, capsys
):
    """B9: a slide refused before Task 28 (its nuclei.npz deleted, its masks kept) is
    pending again and its next look keeps the masks, flagged; StarDist is not run."""
    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch, cells_from=labels_of(shift=(20, 20)))
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    all_slides = nuclei.slides(multiplex.read_manifest(manifest))
    store = nuclei.store_of(tmp / "nuc", all_slides[0])
    rec = nuclei.load(store)
    # what an earlier checkout wrote for it
    for k in ("low_agreement", "low_agreement_reason"):
        rec.pop(k)
    rec.update(usable=False, refused=True, source="", reason="the re-run was REFUSED")
    nuclei.write_json(store / "nuclei.json", rec)
    (store / "nuclei.npz").unlink()
    assert 0 in nuclei.pending(all_slides, tmp / "nuc")
    calls = []
    monkeypatch.setattr(nuclei, "segment", lambda *a: calls.append(a) or 0)
    capsys.readouterr()
    assert nuclei.main(args(**base, segment=True, index=0)) == 0 and not calls
    again = nuclei.load(store)
    assert again["usable"] and again["low_agreement"] and not again.get("refused")
    assert (store / "nuclei.npz").is_file()
    assert "LOW_AGREEMENT P1/P1.r1" in capsys.readouterr().out
    assert 0 not in nuclei.pending(all_slides, tmp / "nuc")


def test_a_kept_rerun_is_redone_when_its_settings_no_longer_resolve_the_same(
    cohort, monkeypatch, capsys
):
    """Review item 6a: the record holds what the masks were PRODUCED with; when the
    model or a setting resolves to something else, the kept re-run is not reused."""
    tmp, lab, row = cohort
    fake_segment(tmp, labels_of(drop=5), monkeypatch, cells_from=lab)  # not exact
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(manifest))
    store = nuclei.store_of(tmp / "nuc", ref)
    rec = nuclei.load(store)
    assert rec["usable"] and not rec["exact"]
    assert rec["settings_recorded"] == nuclei.AT_SEGMENTATION
    made = json.loads((store / "rerun" / nuclei.PRODUCED).read_text())
    assert made["settings"] == rec["settings"] and made["command"] == rec["command"]
    assert rec["settings"]["values"]["segmentation_model"]["value"] == "site_model"
    # the same settings: reused, nothing said
    capsys.readouterr()
    assert nuclei.resolve(ref, tmp / "nuc", args()) == rec
    assert "STALE" not in capsys.readouterr().out
    # another model now: said aloud, set aside, and the slide needs its re-run again
    monkeypatch.setenv("REGBENCH_STARDIST_MODEL", "corrected_model")
    again = nuclei.resolve(ref, tmp / "nuc", args())
    said = capsys.readouterr().out
    assert "STALE P1/P1.r1" in said and "'site_model' -> 'corrected_model'" in said
    assert not again["usable"] and again["needs_rerun"]
    assert "made with other settings" in again["reason"]
    assert again["stale_settings"] == [
        "segmentation_model: 'site_model' -> 'corrected_model'"
    ]
    assert not (store / "nuclei.npz").exists()
    assert (store / "rerun_stale").is_dir() and not (store / "rerun").exists()
    # the outlines traced from the run that is set aside go with it
    assert not Path(ref["outlines"]).exists()
    assert Path(ref["outlines"] + ".stale").is_file()
    # the command is printed for it again, and the new run records the new model
    assert nuclei.main(args(**base, command=True, index=0)) == 0
    assert "--model-name corrected_model" in capsys.readouterr().out
    fake_segment(tmp, lab, monkeypatch)  # (this resets the model's name)
    monkeypatch.setenv("REGBENCH_STARDIST_MODEL", "corrected_model")
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    new = nuclei.load(store)
    assert new["usable"] and new["exact"]
    assert new["settings"]["values"]["segmentation_model"]["value"] == "corrected_model"
    assert nuclei.settings_changed(new) == []
    assert Path(ref["outlines"]).is_file()


def test_a_setting_spelled_another_way_is_not_a_change(monkeypatch):
    """Fleet wave 2, F8: the site config writes the model folder with a trailing slash,
    the environment route without; a number may be written 0.5 or 0.50. The same value
    spelled another way is not other settings: no slide is set aside for it. A value
    that IS another one still is."""
    rec = {
        "source": "stardist_rerun",
        "settings": {
            "values": {
                "segmentation_model_dir": {"value": "/hpc/models/"},
                "segmentation_model": {"value": "site_model"},
                "seg_prob_thresh": {"value": "0.50"},
                "seg_n_tiles_y": {"value": "4"},
            }
        },
    }
    now = {
        "values": {
            "segmentation_model_dir": {"value": "/hpc/models"},
            "segmentation_model": {"value": "site_model"},
            "seg_prob_thresh": {"value": 0.5},
            "seg_n_tiles_y": {"value": "4.0"},
        }
    }
    monkeypatch.setattr(nuclei, "_settings_or_none", lambda: now)
    assert nuclei.settings_changed(rec) == []
    now["values"]["segmentation_model_dir"]["value"] = "/hpc/other_models"
    now["values"]["seg_prob_thresh"]["value"] = "0.6"
    assert nuclei.settings_changed(rec) == [
        "seg_prob_thresh: '0.50' -> '0.6'",
        "segmentation_model_dir: '/hpc/models/' -> '/hpc/other_models'",
    ]


def _one_slide(cohort, monkeypatch, labels, cells_from=None):
    """The first slide of the cohort, segmented by the stand-in with ``labels``."""
    tmp, _, row = cohort
    fake_segment(tmp, labels, monkeypatch, cells_from)
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(manifest))
    return base, ref, nuclei.store_of(tmp / "nuc", ref)


def test_force_replaces_the_nuclei_of_a_slide_that_looks_usable(
    cohort, monkeypatch, capsys
):
    """A run that is kept and usable (five nuclei short) is not what --force leaves in
    place: it is set aside, StarDist runs again, and the record, the polygons and the
    outlines are the new run's."""
    tmp, lab, _ = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, labels_of(drop=5), lab)
    first = nuclei.load(store)
    n_all = int(lab.max())
    assert first["usable"] and not first["exact"] and first["n_nuclei"] == n_all - 5
    fake_segment(tmp, lab, monkeypatch)  # what StarDist gives now: every nucleus
    calls, real = [], nuclei.segment
    monkeypatch.setattr(nuclei, "segment", lambda *a: calls.append(a) or real(*a))
    # unforced: what is kept answers, nothing runs
    assert nuclei.main(args(**base, segment=True, index=0)) == 0 and not calls
    assert nuclei.load(store) == first
    capsys.readouterr()
    assert nuclei.main(args(**base, segment=True, index=0, force=True)) == 0
    assert len(calls) == 1 and "FORCED P1/P1.r1" in capsys.readouterr().out
    new = nuclei.load(store)
    assert new["usable"] and new["exact"] and new["n_nuclei"] == n_all
    assert pairs.n_cells(nuclei.read_polys(store)) == n_all
    assert (store / "rerun_stale").is_dir()  # the run it replaces, kept aside
    # the array task's own order: the forced look, then the line, then an unforced look
    assert nuclei.main(args(**base, index=0, force=True)) == 0
    forced = nuclei.load(store)
    assert not forced["usable"] and forced["needs_rerun"]
    assert not (store / "nuclei.npz").exists() and not (store / "rerun").exists()
    capsys.readouterr()
    assert nuclei.main(args(**base, command=True, index=0, force=True)) == 0
    assert str(nuclei.SEGMENT_PY) in capsys.readouterr().out  # the line to run


def test_a_record_is_of_the_image_it_was_computed_from(cohort, monkeypatch, capsys):
    """The slide is replaced by another file under the same name: its kept run is not
    this file's nuclei, nor its outlines. Said aloud, set aside, segmented again."""
    tmp, lab, row = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, lab)
    rec = nuclei.load(store)
    assert rec["image_fingerprint"] == cases_mod.file_fingerprint(ref["image"])
    assert set(rec["image_fingerprint"]) == {"size", "mtime_ns", "sample_sha256"}
    capsys.readouterr()
    assert nuclei.resolve(ref, tmp / "nuc", args()) == rec
    assert "STALE" not in capsys.readouterr().out
    write_nuclear(Path(ref["image"]), (labels_of(drop=7) > 0).astype(np.uint16), 0.325)
    assert cases_mod.file_fingerprint(ref["image"]) != rec["image_fingerprint"]
    # prepare does not take the outlines traced from the file that was there before
    fake_segment(tmp, lab, monkeypatch)
    assert nuclei.main(args(**base, segment=True, index=1)) == 0
    cases_root = tmp / "cases"
    a = mx_args(cases_root, base["manifest"], nuclei_dir=str(tmp / "nuc"))
    assert multiplex.prepare(a) == 1
    said = capsys.readouterr().out
    assert "DROPPED the outlines" in said and "traced from another file" in said
    assert list_cases(cases_root, "multiplex") == []
    # the nuclei step: stale, set aside, and the slide needs its re-run again
    again = nuclei.resolve(ref, tmp / "nuc", args())
    said = capsys.readouterr().out
    assert "STALE P1/P1.r1" in said and "not the file its nuclei were computed" in said
    assert not again["usable"] and again["needs_rerun"]
    assert again["stale_image"] and "set aside" in again["reason"]
    assert "has not run yet" in again["reason"]
    assert (store / "rerun_stale").is_dir() and not (store / "nuclei.npz").exists()
    assert not Path(ref["outlines"]).exists()
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    new = nuclei.load(store)
    assert new["usable"]
    assert new["image_fingerprint"] == cases_mod.file_fingerprint(ref["image"])
    assert multiplex.prepare(a) == 0


def test_a_record_that_names_no_image_is_checked_again_and_not_recomputed(
    cohort, monkeypatch, capsys
):
    """A store filled before records carried a fingerprint: the record is not used as
    it is (prepare refuses it), and the nuclei step checks the kept re-run against the
    slide without StarDist and stamps it. Nothing is set aside."""
    tmp, lab, _ = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, lab)
    rec = nuclei.load(store)
    full = dict(rec)
    rec.pop("image_fingerprint")
    (store / "nuclei.json").write_text(json.dumps(rec))
    assert "does not say which image" in nuclei.other_image(rec, ref)
    capsys.readouterr()
    again = nuclei.resolve(ref, tmp / "nuc", args())
    said = capsys.readouterr().out
    assert "RECHECK P1/P1.r1" in said and "STALE" not in said
    assert again["usable"] and not again["needs_rerun"]
    assert again["image_fingerprint"] == full["image_fingerprint"]
    assert nuclei.load(store) == again and nuclei.other_image(again, ref) == ""
    assert (store / "rerun").is_dir() and not (store / "rerun_stale").exists()
    assert (store / "nuclei.npz").is_file()
    # a kept re-run that is NOT this slide's (another size): a real mismatch, set aside
    (store / "nuclei.json").write_text(json.dumps(rec))
    monkeypatch.setattr(
        nuclei, "_shape", lambda path: (1, 1) if "rerun" in str(path) else (2, 2)
    )
    stale = nuclei.resolve(ref, tmp / "nuc", args())
    assert not stale["usable"] and stale["needs_rerun"]
    assert "does not fit the slide: set aside" in stale["reason"]
    assert (store / "rerun_stale").is_dir() and not (store / "rerun").exists()


def test_setting_aside_never_leaves_a_usable_record_over_missing_polygons(
    cohort, monkeypatch
):
    """The task is ended inside `set_aside`, wherever: the slide then has no record
    (decided again), never a record that says usable without its polygons."""
    tmp, lab, _ = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, lab)
    assert nuclei.load(store)["usable"]
    real = Path.rename

    def ended(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "rename", ended)
    with pytest.raises(KeyboardInterrupt):
        nuclei.set_aside(store, ref["outlines"])
    monkeypatch.setattr(Path, "rename", real)
    assert nuclei.load(store) is None  # the record went first
    assert (store / "nuclei.stale.json").is_file()
    nuclei.set_aside(store, ref["outlines"])
    assert not (store / "nuclei.npz").exists() and (store / "rerun_stale").is_dir()
    assert not Path(ref["outlines"]).exists()


def test_a_trace_ended_halfway_never_leaves_outlines_without_their_record(
    cohort, monkeypatch
):
    """The task is ended inside the trace: the slide has the record of what was being
    traced and NO outlines (so they stay this module's, and the next look traces them),
    never a GeoJSON without its `.traced.json` (another's, never to be rewritten) nor
    the old outlines under the new record."""
    tmp, lab, row = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, lab)
    first = Path(ref["outlines"]).read_text()

    def ended(*a, **k):
        raise KeyboardInterrupt

    real = outlines.trace
    monkeypatch.setattr(outlines, "trace", ended)
    with pytest.raises(KeyboardInterrupt):
        nuclei.trace_outlines(ref, nuclei.expected_masks(ref, store / "rerun")[1])
    monkeypatch.setattr(outlines, "trace", real)
    assert not Path(ref["outlines"]).exists()
    assert nuclei.traced_record(ref["outlines"]) is not None
    assert nuclei.is_ours(ref["outlines"])
    assert nuclei.main(args(**base, index=0)) == 0  # the next look, no StarDist
    assert Path(ref["outlines"]).read_text() == first
    assert nuclei.traced_record(ref["outlines"])["n_outlines"] == lab.max()


def test_outlines_that_are_gone_are_traced_again_without_stardist(
    cohort, monkeypatch, capsys
):
    """A usable record whose outlines are not there (or not where the manifest now
    names them) reads the kept run again and traces it: no GPU."""
    tmp, lab, row = cohort
    base, ref, store = _one_slide(cohort, monkeypatch, lab)
    first = Path(ref["outlines"]).read_text()
    Path(ref["outlines"]).unlink()
    calls, real = [], nuclei.segment
    monkeypatch.setattr(nuclei, "segment", lambda *a: calls.append(a) or real(*a))
    assert nuclei.main(args(**base, segment=True, index=0)) == 0 and not calls
    assert Path(ref["outlines"]).read_text() == first
    assert nuclei.load(store)["exact"]
    # the manifest names another folder for the outlines: traced there
    moved = {**row, "ref_cells": str(tmp / "elsewhere" / "P1" / "P1.r1.geojson")}
    manifest = manifest_of(tmp, moved)
    assert nuclei.main(args(manifest=manifest, out=tmp / "nuc", index=0)) == 0
    assert not calls and (tmp / "elsewhere" / "P1" / "P1.r1.geojson").is_file()
    assert nuclei.load(store)["outlines"] == moved["ref_cells"]


def test_the_array_task_keeps_a_low_agreement_slide_and_can_be_forced(
    cohort, monkeypatch, tmp_path
):
    """Review item 6b, then Task 28 B9: a slide under the agreement threshold is kept
    and flagged, so its task has nothing to segment and exits 0 (it was refused and
    failed); REGBENCH_NUCLEI_FORCE=1 still reaches the re-run."""
    import os

    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch, cells_from=labels_of(shift=(20, 20)))
    manifest = manifest_of(tmp, row)
    assert (
        nuclei.main(args(manifest=manifest, out=tmp / "nuc", segment=True, index=0))
        == 0
    )
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(tmp / "cases"),
        REGBENCH_OUT=str(tmp / "out"),
        REGBENCH_MANIFEST=str(manifest),
        REGBENCH_NUCLEI=str(tmp / "nuc"),
        REGBENCH_SITE_CONFIG="none",
        SLURM_ARRAY_TASK_ID="0",
        **ENV,
    )
    script = REPO / "benchmarks" / "regbench" / "slurm" / "nuclei.sbatch"
    done = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, cwd=REPO
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "nothing to segment" in done.stdout and "+ python" not in done.stdout
    assert "is not run again" not in done.stdout and "LOW_AGREEMENT" in done.stdout
    # forced: the task gets as far as the re-run's command (the stand-in segment.py is
    # not what this process's `python bin/segment.py` runs, so only the line is checked)
    forced = subprocess.run(
        ["bash", str(script)],
        env={**env, "REGBENCH_NUCLEI_FORCE": "1"},
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert "+ python" in forced.stdout and "bin/segment.py" in forced.stdout
    assert "is not run again" not in forced.stdout
    # a slide with usable nuclei: nothing to do, exit 0
    fake_segment(tmp, lab, monkeypatch)
    assert (
        nuclei.main(args(manifest=manifest, out=tmp / "nuc", segment=True, index=1))
        == 0
    )
    ok = subprocess.run(
        ["bash", str(script)],
        env={**env, "SLURM_ARRAY_TASK_ID": "1"},
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert ok.returncode == 0 and "nothing to segment" in ok.stdout


def test_a_stardist_run_that_fails_is_written_down_and_climbs(cohort, tmp_path):
    """Fleet wave 2, F7: under `set -e` a StarDist run the kernel killed for memory
    ended the task with nothing written: `prepare multiplex` then dropped the slide's
    rounds as "outlines not traced yet", and the next --prepare repeated the same
    128 GB. Now the task writes the failure down (the slide's record, beside its
    outlines), its end line asks the watcher for the next rung (wants=climb), and the
    drop reason names it."""
    import os

    tmp, lab, row = cohort
    manifest = manifest_of(tmp, row)
    # a `python` that the kernel kills (signal 9) when it runs segment.py, and is the
    # real interpreter for everything else
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "python").write_text(
        "#!/bin/bash\n"
        'case "$1" in *segment.py) kill -9 $$ ;; esac\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    (fake / "python").chmod(0o755)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("REGBENCH_", "SLURM_"))
    }
    env.update(
        PATH=f"{fake}:{env['PATH']}",
        REGBENCH_REPO=str(REPO),
        REGBENCH_CASES=str(tmp / "cases"),
        REGBENCH_OUT=str(tmp / "out"),
        REGBENCH_MANIFEST=str(manifest),
        REGBENCH_NUCLEI=str(tmp / "nuc"),
        REGBENCH_SITE_CONFIG="none",
        SLURM_JOB_ID="51",
        SLURM_JOB_NAME="rb-nuclei",
        SLURM_ARRAY_JOB_ID="50",
        SLURM_ARRAY_TASK_ID="0",
        SLURM_MEM_PER_NODE="131072",
        **ENV,
    )
    a = args(manifest=manifest, out=tmp / "nuc")
    assert nuclei.pending(nuclei.slides(multiplex.read_manifest(manifest)), a.out) == [0, 1]
    script = REPO / "benchmarks" / "regbench" / "slurm" / "nuclei.sbatch"
    done = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, cwd=REPO
    )
    assert done.returncode == 137, done.stdout + done.stderr
    assert "StarDist FAILED (exit 137 at 128 GB)" in done.stdout
    end = (tmp / "out" / "logs" / "end" / "rb-nuclei-50_0.end").read_text().strip()
    assert end == done.stdout.strip().splitlines()[-1]
    assert "status=failed cause=benchmark_oom whose=benchmark wants=climb exit=137" in end
    (ref, _) = nuclei.slides(multiplex.read_manifest(manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    assert rec["segment_failed"]["exit"] == 137 and rec["segment_failed"]["mem_gb"] == 128
    assert "killed by signal 9 at 128 GB" in rec["reason"]
    assert nuclei.pending([ref], a.out) == [0]  # still to do, at the next rung
    # the drop names the failure, not "not traced yet" alone
    cases_root = tmp / "cases"
    import io
    from contextlib import redirect_stdout

    said = io.StringIO()
    with redirect_stdout(said):
        assert multiplex.prepare(mx_args(cases_root, manifest)) == 1
    assert "the nuclei step's StarDist run of this slide FAILED" in said.getvalue()
    assert "killed by signal 9 at 128 GB (exit 137)" in said.getvalue()
    # what nuclei --list says about it
    rows = nuclei.table([ref], tmp / "nuc")
    assert "killed by signal 9" in rows[0]["reason"]


def test_the_nuclei_array_is_submitted_for_the_slides_with_anything_to_do(
    cohort, monkeypatch
):
    """F9: a slide with usable nuclei and its outlines is left out of the array; a
    forced one, one of other settings, or one whose outlines are gone is not."""
    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch)
    manifest = manifest_of(tmp, row)
    all_slides = nuclei.slides(multiplex.read_manifest(manifest))
    out = tmp / "nuc"
    assert nuclei.main(args(manifest=manifest, out=out, segment=True, index=0)) == 0
    assert nuclei.pending(all_slides, out) == [1]
    assert nuclei.pending(all_slides, out, force=True) == [0, 1]
    assert nuclei.fold([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8" and nuclei.fold([]) == ""
    Path(all_slides[0]["outlines"]).unlink()
    assert nuclei.pending(all_slides, out) == [0, 1]


def test_a_rerun_between_the_thresholds_is_still_kept_and_not_exact(
    cohort, monkeypatch, capsys
):
    tmp, lab, row = cohort
    # 0.946: over 0.90, under 0.99
    fake_segment(tmp, labels_of(drop=3), monkeypatch, cells_from=lab)
    a = args(manifest=manifest_of(tmp, row), out=tmp / "nuc", segment=True, index=0)
    assert nuclei.main(a) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(a.manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    frac = rec["agreement"]["one_nucleus_frac"]
    assert nuclei.ACCEPT_ONE_NUCLEUS <= frac < nuclei.EXACT_ONE_NUCLEUS
    assert rec["usable"] and not rec["exact"] and not rec.get("refused")
    said = capsys.readouterr().out
    assert "DISAGREEMENT P1/P1.r1" in said and "REFUSED" not in said


# ── a manifest written from the arms' results (before 2026-10-16) ─────────────
def test_an_old_manifests_published_outlines_are_read_and_never_overwritten(
    cohort, monkeypatch, capsys
):
    """Backward compatibility: a rounds.csv of the arms' results names the arm's
    published outlines. They are not this module's: never rewritten; the StarDist run
    is compared with them as before, and the round is prepared from them as before."""
    tmp, lab, row = cohort
    old = published(tmp, lab, row)
    before = {s: Path(old[f"{s}_cells"]).read_bytes() for s in ("ref", "mov")}
    fake_segment(tmp, lab, monkeypatch)
    manifest = manifest_of(tmp, old)
    assert nuclei.main(args(manifest=manifest, out=tmp / "nuc", segment=True)) == 0
    for s in ("ref", "mov"):
        assert Path(old[f"{s}_cells"]).read_bytes() == before[s]
        assert not Path(old[f"{s}_cells"] + nuclei.TRACED).exists()
    (ref, _) = nuclei.slides(multiplex.read_manifest(manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    assert rec["outlines_from"] == "manifest" and rec["usable"] and rec["exact"]
    # forced: what is set aside is the run, never the arm's outlines
    assert (
        nuclei.main(args(manifest=manifest, out=tmp / "nuc", index=0, force=True)) == 0
    )
    assert Path(old["ref_cells"]).read_bytes() == before["ref"]
    cases_root = tmp / "cases"
    assert multiplex.prepare(mx_args(cases_root, manifest, nuclei_dir=tmp / "nuc")) == 0
    (case,) = list_cases(cases_root, "multiplex")
    assert cases_mod.set_info(case, "ref", "outlines") == {"source": "arm_geojson"}
    assert polys(load_points(case), "ref").n_features == lab.max()


# ── the model and the settings come from the site config ──────────────────────
SITE = """\
// a site overlay
params {
    // segmentation_model     = 'commented_out'
    segmentation_model     = 'site_net'      // the model the arms ran
    segmentation_model_dir = "/site/models/" /* its folder */
    seg_pmax               = 99.5
    seg_n_tiles_y          = 16
    outdir                 = '/site/out//x'
    /* seg_expand_distance = 4 */
}
"""


def site_file(tmp_path, text=SITE):
    path = tmp_path / "site.config"
    path.write_text(text)
    return path


def test_the_site_config_is_read_comment_blind_and_two_keys_only(tmp_path):
    got = site_config.read(site_file(tmp_path))
    assert got == {
        "segmentation_model": "site_net",
        "segmentation_model_dir": "/site/models/",
        "seg_pmax": 99.5,
        "seg_n_tiles_y": 16,
    }  # no `outdir`, nothing from a comment
    assert site_config.strip_comments("a = 'x//y' // c\n/* b */d") == "a = 'x//y' \n d"
    for raw, want in (("'a'", "a"), ("null", None), ("3", 3), ("0.5", 0.5)):
        assert site_config.literal(raw) == want
    with pytest.raises(ValueError):
        site_config.literal("params.other + 1")


def test_the_model_and_the_settings_resolve_from_the_site_config(tmp_path, cohort):
    tmp, lab, row = cohort
    path = site_file(tmp_path)
    env = {site_config.ENV_SITE_CONFIG: str(path)}
    values = nuclei.settings(env)["values"]
    where = f"site config ({path})"
    assert values["segmentation_model"] == {"value": "site_net", "source": where}
    assert values["segmentation_model_dir"]["value"] == "/site/models/"
    assert values["seg_pmax"] == {
        "value": 99.5,
        "source": where,
        "default": 99.8,
        "differs": True,
    }
    assert values["seg_n_tiles_y"]["source"] == where  # set there, to the default
    assert not values["seg_n_tiles_y"]["differs"]
    assert values["seg_pmin"]["source"] == site_config.FROM_DEFAULT
    assert values["seg_expand_distance"]["value"] == 10  # the commented 4 is not read
    # the rendered command carries the site's values
    (ref, _) = nuclei.slides([row])
    argv = nuclei.command(ref, tmp / "dest", env)
    assert argv[argv.index("--model-name") + 1] == "site_net"
    assert argv[argv.index("--model-dir") + 1] == "/site/models/"
    assert argv[argv.index("--pmax") + 1] == "99.5"
    assert argv[argv.index("--pmin") + 1] == "1.0" and "--prob-thresh" not in argv
    # explicit variables win, value by value
    both = {**env, **ENV, "REGBENCH_STARDIST_PROB_THRESH": "0.3"}
    values = nuclei.settings(both)["values"]
    assert values["segmentation_model"] == {
        "value": "site_model",
        "source": "environment (REGBENCH_STARDIST_MODEL)",
    }
    assert values["segmentation_model_dir"]["value"] == "/models"
    assert values["seg_prob_thresh"]["source"].startswith("environment")
    argv = nuclei.command(ref, tmp / "dest", both)
    assert argv[argv.index("--model-name") + 1] == "site_model"
    assert argv[argv.index("--pmax") + 1] == "99.5"  # still the site's
    assert argv[argv.index("--prob-thresh") + 1] == "0.3"
    # the lines a log prints: name, value, source, and what differs
    lines = site_config.lines(nuclei.settings(env))
    assert f"segmentation_model\tsite_net\t{where}" in lines
    assert any(
        ln.startswith("seg_pmax\t99.5\t")
        and "differs from the repository default 99.8" in ln
        for ln in lines
    )
    assert "seg_prob_thresh\tnull\t" + site_config.FROM_DEFAULT in lines


def test_where_the_site_config_is_looked_for(tmp_path, monkeypatch):
    path = site_file(tmp_path)
    # the checkout's own overlay when nothing is named
    monkeypatch.setattr(site_config, "DEFAULT_SITE_CONFIG", path)
    assert nuclei.settings({})["site_config"] == str(path)
    assert nuclei.settings({})["values"]["segmentation_model"]["value"] == "site_net"
    # `none` reads no file; a named file that is not there is an error, not a fallback
    off = nuclei.settings({site_config.ENV_SITE_CONFIG: "none"})
    assert off["site_config"] == "" and off["values"]["seg_pmax"]["value"] == 99.8
    assert off["values"]["segmentation_model"] == {
        "value": "",
        "source": site_config.NOT_SET,
    }
    with pytest.raises(SystemExit, match="no such file"):
        nuclei.settings({site_config.ENV_SITE_CONFIG: str(tmp_path / "gone.config")})
    # the repository's defaults are read from nextflow.config, not restated
    assert site_config.repository_defaults() == {
        "seg_n_tiles_y": nuclei.STARDIST["n_tiles"][0],
        "seg_n_tiles_x": nuclei.STARDIST["n_tiles"][1],
        "seg_expand_distance": nuclei.STARDIST["expand_distance"],
        "seg_pmin": nuclei.STARDIST["pmin"],
        "seg_pmax": nuclei.STARDIST["pmax"],
        "seg_prob_thresh": None,
    }
    # the shipped template has the two keys in the form that is read
    template = site_config.read(REPO / "conf" / "site.config.template")
    assert set(template) >= set(site_config.MODEL_KEYS)


def test_a_site_config_that_cannot_be_read_plainly_is_refused(tmp_path):
    twice = (
        SITE
        + "profiles {\n  a {\n    params.segmentation_model = 'other_net'\n  }\n}\n"
    )
    with pytest.raises(SystemExit, match="set 2 times.*REGBENCH_STARDIST_MODEL"):
        site_config.read(site_file(tmp_path, twice))
    same = SITE + "params.segmentation_model = 'site_net'\n"
    assert site_config.read(site_file(tmp_path, same))["segmentation_model"] == (
        "site_net"
    )
    computed = SITE.replace("99.5", "params.other * 2")
    with pytest.raises(SystemExit, match="seg_pmax = params.other \\* 2"):
        site_config.read(site_file(tmp_path, computed))


def test_the_settings_are_printed_and_recorded(cohort, monkeypatch, capsys, tmp_path):
    tmp, lab, row = cohort
    path = site_file(tmp_path)
    fake_segment(tmp, lab, monkeypatch)
    for k in ENV:
        monkeypatch.delenv(k)
    monkeypatch.setenv(site_config.ENV_SITE_CONFIG, str(path))
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    # `--settings`: every value and its source, on stdout
    assert nuclei.main(args(**base, settings=True)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"segmentation_model\tsite_net\tsite config ({path})"
    assert len(out) == len(site_config.KEYS)
    # `--command`: the shell line alone on stdout, the sources on stderr
    assert nuclei.main(args(**base, command=True, index=0)) == 0
    said = capsys.readouterr()
    (line,) = said.out.splitlines()
    assert "--model-name site_net" in line and "--pmax 99.5" in line
    assert f"# segmentation_model\tsite_net\tsite config ({path})" in said.err
    # the slide's record keeps what the re-run was rendered with
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    assert rec["usable"] and rec["settings"]["site_config"] == str(path)
    assert rec["settings"]["values"]["seg_pmax"]["differs"] is True
    assert "--pmax 99.5" in rec["command"]
    rerun = nuclei.store_of(tmp / "nuc", ref) / "rerun"
    assert "--pmax 99.5" in (rerun / "argv.txt").read_text()
    # without a model anywhere `--settings` fails, having said what it found
    monkeypatch.setenv(site_config.ENV_SITE_CONFIG, "none")
    with pytest.raises(SystemExit, match="no StarDist model"):
        nuclei.main(args(**base, settings=True))


def test_the_list_says_which_slides_have_nuclei_and_which_need_the_rerun(
    cohort, monkeypatch, capsys
):
    tmp, lab, row = cohort
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, count=True)) == 0
    assert capsys.readouterr().out.strip() == "2"
    fake_segment(tmp, lab, monkeypatch)
    assert nuclei.main(args(**base, segment=True, index=0)) == 0
    assert (
        nuclei.main(args(**base, index=1)) == 0
    )  # resolve: nothing is segmented unasked
    capsys.readouterr()
    assert nuclei.main(args(**base, list=True)) == 0
    said = capsys.readouterr()
    lines = [ln.split("\t") for ln in said.out.splitlines()]
    assert lines[0] == list(nuclei.COLUMNS)
    assert lines[1][:5] == ["0", "P1/P1.r1", "usable", "stardist_rerun", "True"]
    assert lines[2][:3] == ["1", "P1/P1.r2", "needs_rerun"]
    assert "1 with usable nuclei, 1 needing the StarDist re-run" in said.err
    # the command is printed only for the slide that needs it
    assert nuclei.main(args(**base, command=True, index=0)) == 0
    assert capsys.readouterr().out == ""
    assert nuclei.main(args(**base, command=True, index=1)) == 0
    line = capsys.readouterr().out.strip()
    assert shlex.split(line)[:4] == [
        "python",
        str(nuclei.SEGMENT_PY),
        "--image",
        row["moving"],
    ]


# ── prepare multiplex ─────────────────────────────────────────────────────────
def test_prepare_takes_the_nuclei_of_each_slide_and_records_their_source(
    cohort, monkeypatch
):
    tmp, lab, row = cohort
    manifest = manifest_of(tmp, row)
    fake_segment(tmp, lab, monkeypatch)
    assert nuclei.main(args(manifest=manifest, out=tmp / "nuc", segment=True)) == 0
    cases_root = tmp / "cases"
    assert multiplex.prepare(mx_args(cases_root, manifest, nuclei_dir=tmp / "nuc")) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    for side in ("ref", "mov"):
        assert polys(points, side, "nuclei").n_features == lab.max()
        assert polys(points, side).n_features == lab.max()
        assert cases_mod.set_info(case, side, "outlines") == {"source": "arm_geojson"}
        assert f"nuclei_{side}_outline" not in points  # not aligned by construction
        info = cases_mod.set_info(case, side, "nuclei")
        assert info["source"] == "stardist_rerun" and info["exact"] is True
        assert info["outlines_from"] == "stardist_rerun_cell_mask"
        assert info["tolerance"] == 1.0
        assert info["agreement"]["one_nucleus_frac"] == 1.0
    assert cases_mod.nuclei_reason(case, points) == ""


def test_a_slide_with_no_nuclei_is_still_prepared_with_the_reason(cohort):
    """Outlines without nuclei (an old manifest's published outlines, no nuclei step):
    prepared and scored on the outlines, the reason kept."""
    tmp, lab, row = cohort
    cases_root = tmp / "cases"
    old = published(tmp, lab, row)
    assert multiplex.prepare(mx_args(cases_root, manifest_of(tmp, old))) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    assert polys(points, "mov").n_features == lab.max()
    assert polys(points, "mov", "nuclei") is None
    info = cases_mod.set_info(case, "mov", "nuclei")
    assert info["source"] == "" and "no StarDist run of this slide" in info["reason"]
    assert "ref: no StarDist run" in cases_mod.nuclei_reason(case, points)


def test_a_round_whose_outlines_were_not_traced_is_dropped(cohort, capsys):
    tmp, lab, row = cohort
    cases_root = tmp / "cases"
    assert multiplex.prepare(mx_args(cases_root, manifest_of(tmp, row))) == 1
    said = capsys.readouterr().out
    assert (
        f"multiplex/P1_P1.r2: DROPPED {multiplex.NOT_TRACED}: {row['ref_cells']}"
        in said
    )
    assert list_cases(cases_root, "multiplex") == []


def test_the_shrunk_outline_is_used_only_when_asked_and_is_never_exact(cohort):
    tmp, lab, row = cohort
    manifest = manifest_of(tmp, published(tmp, lab, row))
    cases_root = tmp / "cases"
    a = mx_args(cases_root, manifest, nuclei_fallback="shrink")
    assert multiplex.prepare(a) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    for side in ("ref", "mov"):
        info = cases_mod.set_info(case, side, "nuclei")
        assert info["source"] == "shrunk_outlines" and info["exact"] is False
        assert info["shrunk_by_px"] == cells.EXPAND_PX
        nuc = polys(points, side, "nuclei")
        parent = points[f"nuclei_{side}_outline"]
        assert nuc.n_features == len(parent) == lab.max()
        a_nuc, _ = pairs.cp.feature_area_centroid(nuc)
        a_out, _ = pairs.cp.feature_area_centroid(polys(points, side))
        assert (a_nuc < a_out[parent]).all()


def test_the_builtin_segmenter_gives_nuclei_and_grown_outlines(tmp_path):
    lab = labels_of()
    blobs = (lab > 0).astype(np.float32) * 1000.0
    from scipy import ndimage as ndi

    plane = ndi.gaussian_filter(blobs, 1.5).astype(np.uint16)
    slides = [write_nuclear(tmp_path / f"r{i}.ome.tif", plane, 0.325) for i in (1, 2)]
    row = {"case_id": "c", "reference": str(slides[0]), "moving": str(slides[1])}
    cases_root = tmp_path / "cases"
    a = mx_args(cases_root, manifest_of(tmp_path, row), diameter=9.0)
    assert multiplex.prepare(a) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    nuc, outl = polys(points, "ref", "nuclei"), polys(points, "ref")
    assert nuc.n_features == outl.n_features == lab.max()
    assert cases_mod.set_info(case, "ref", "nuclei") == {
        "source": "builtin",
        "exact": True,
    }
    assert cases_mod.set_info(case, "ref", "outlines")["expand_px"] == cells.EXPAND_PX
    assert (points["nuclei_ref_outline"] == np.arange(lab.max())).all()


# ── the command line and the launcher ─────────────────────────────────────────
def test_the_command_line_has_the_nuclei_step(cohort):
    tmp, lab, row = cohort
    manifest = manifest_of(tmp, row)
    run = [sys.executable, "-m", "benchmarks.regbench", "nuclei"]
    done = subprocess.run(
        [*run, "--manifest", str(manifest), "--out", str(tmp / "nuc"), "--count"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    )
    assert done.stdout.strip() == "2"
