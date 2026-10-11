"""Bare nuclei for the cohort: an existing label image is reused when it fits its slide,
StarDist is run again only for a slide that has none, the shrunk outline is a fallback
that is never used unasked, and a slide with nothing is prepared and scored on outlines.

StarDist is not installed here: the re-run is driven with a stand-in ``segment.py`` that
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
import tifffile

from benchmarks.regbench import arms_manifest, cells, nuclei, pairs, site_config
from benchmarks.regbench import cases as cases_mod
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
    """The outlines of a label image as the pipeline publishes them: one Polygon each."""
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


def mask_file(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, labels, compression="zlib", tile=(128, 128))
    return path


@pytest.fixture
def cohort(tmp_path):
    """Two rounds of one patient: slides, published outlines, and a manifest row."""
    lab = labels_of()
    slides, outlines = {}, {}
    for name in ("P1.r1", "P1.r2"):
        slides[name] = write_nuclear(
            tmp_path / "slides" / f"{name}.ome.tif", (lab > 0).astype(np.uint16), 0.325
        )
        outlines[name] = geojson_of(
            cells.grow(lab), tmp_path / "geojson" / f"{name}.geojson"
        )
    row = {
        "case_id": "P1_P1.r2",
        "reference": str(slides["P1.r1"]),
        "moving": str(slides["P1.r2"]),
        "group": "P1",
        "ref_channel": 0,
        "mov_channel": 0,
        "pixel_size_um": 0.325,
        "ref_cells": str(outlines["P1.r1"]),
        "mov_cells": str(outlines["P1.r2"]),
    }
    return tmp_path, lab, row


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


# ── an existing label image ───────────────────────────────────────────────────
def test_a_label_image_of_the_slide_is_used_and_called_exact(cohort):
    tmp, lab, row = cohort
    mask = mask_file(tmp / "work" / "P1.r1.ome_nuclei_mask.tif", lab)
    (ref, mov) = nuclei.slides([{**row, "ref_nuclei": str(mask)}])
    assert (ref["key"], mov["key"]) == ("P1/P1.r1", "P1/P1.r2")
    rec = nuclei.resolve(ref, tmp / "nuc", args())
    assert rec["usable"] and rec["exact"] and not rec["needs_rerun"]
    assert rec["source"] == f"label_image:{mask}"
    assert rec["agreement"]["one_nucleus_frac"] == 1.0
    assert rec["agreement"]["n_nuclei"] == rec["agreement"]["n_outlines"]
    assert nuclei.read_polys(nuclei.store_of(tmp / "nuc", ref)).n_features == lab.max()
    # the other slide has none: it waits for the re-run, with the reason
    rec = nuclei.resolve(mov, tmp / "nuc", args())
    assert not rec["usable"] and rec["needs_rerun"]
    assert rec["reason"] == nuclei.NEEDS_RERUN


def test_a_label_image_of_another_size_is_refused_with_the_reason(cohort):
    tmp, lab, row = cohort
    mask = mask_file(tmp / "work" / "wrong.tif", labels_of(hw=(HW[0] + 8, HW[1])))
    (ref, _) = nuclei.slides([{**row, "ref_nuclei": str(mask)}])
    rec = nuclei.resolve(ref, tmp / "nuc", args())
    assert not rec["usable"] and rec["needs_rerun"]
    assert "not a mask of this native slide" in rec["reason"]
    assert "368" in rec["reason"] and "360" in rec["reason"]


def test_a_label_image_that_does_not_sit_in_the_outlines_is_refused(cohort):
    """The seg-arm route's risk: the right size, another frame."""
    tmp, lab, row = cohort
    mask = mask_file(tmp / "seg" / "P1_nuclei_mask.tif", labels_of(shift=(20, 20)))
    (ref, _) = nuclei.slides(
        [{**row, "ref_nuclei": str(mask), "ref_nuclei_from": "seg_arm:seg_stardist"}]
    )
    rec = nuclei.resolve(ref, tmp / "nuc", args())
    assert not rec["usable"]
    assert rec["candidate_refused"].startswith(str(mask))
    assert "at least 0.9 is required" in rec["reason"]
    assert not (nuclei.store_of(tmp / "nuc", ref) / "nuclei.npz").exists()


def test_a_few_missing_nuclei_are_usable_but_not_exact(cohort):
    tmp, lab, row = cohort
    fewer = labels_of(drop=3)  # 3 of 56 outlines have no nucleus: 0.946
    mask = mask_file(tmp / "work" / "P1.r1.ome_nuclei_mask.tif", fewer)
    (ref, _) = nuclei.slides([{**row, "ref_nuclei": str(mask)}])
    rec = nuclei.resolve(ref, tmp / "nuc", args())
    frac = rec["agreement"]["one_nucleus_frac"]
    assert nuclei.ACCEPT_ONE_NUCLEUS <= frac < nuclei.EXACT_ONE_NUCLEUS
    assert rec["usable"] and not rec["exact"]


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
stem = Path(a.image).stem
tifffile.imwrite(Path(a.output_dir) / f"{{stem}}_nuclei_mask.tif", lab)
tifffile.imwrite(Path(a.output_dir) / f"{{stem}}_cell_mask.tif",
                 expand_labels(lab, distance=a.expand_distance).astype(np.uint32))
"""


def fake_segment(tmp, labels, monkeypatch):
    np.save(tmp / "labels.npy", labels)
    script = tmp / "fake_segment.py"
    script.write_text(FAKE.format(labels=str(tmp / "labels.npy")))
    monkeypatch.setattr(nuclei, "SEGMENT_PY", script)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_a_slide_without_a_mask_is_segmented_and_checked(cohort, monkeypatch, capsys):
    tmp, lab, row = cohort
    fake_segment(tmp, lab, monkeypatch)
    manifest = manifest_of(tmp, row)
    a = args(manifest=manifest, out=tmp / "nuc", segment=True)
    assert nuclei.main(a) == 0
    for slide in nuclei.slides(multiplex.read_manifest(manifest)):
        rec = nuclei.load(nuclei.store_of(tmp / "nuc", slide))
        assert rec["source"] == "stardist_rerun" and rec["usable"] and rec["exact"]
        assert rec["agreement"]["centroid_match_frac"] == 1.0
        assert rec["agreement"]["n_cells_rerun"] == rec["agreement"]["n_outlines"]
        stem = Path(slide["image"]).stem  # the name bin/segment.py gives its masks
        rerun = nuclei.store_of(tmp / "nuc", slide) / "rerun"
        assert (rerun / f"{stem}_nuclei_mask.tif").is_file() and stem.endswith(".ome")
        assert "--expand-distance 10" in (rerun / "argv.txt").read_text()
        assert shlex.split(rec["command"])[1] == str(nuclei.SEGMENT_PY)
    assert "DISAGREEMENT" not in capsys.readouterr().out


def test_a_rerun_that_disagrees_with_the_outlines_is_loud_and_not_exact(
    cohort, monkeypatch, capsys
):
    tmp, lab, row = cohort
    # another model's nuclei, 5 of 56 missing: 0.911, kept (under 0.90 it is refused)
    fake_segment(tmp, labels_of(drop=5), monkeypatch)
    a = args(manifest=manifest_of(tmp, row), out=tmp / "nuc", segment=True, index=0)
    assert nuclei.main(a) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(a.manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    assert rec["usable"] and not rec["exact"]
    assert rec["agreement"]["centroid_match_frac"] < nuclei.EXACT_MATCH
    assert "DISAGREEMENT P1/P1.r1" in capsys.readouterr().out


def test_a_rerun_far_from_the_outlines_is_refused(cohort, monkeypatch, capsys):
    """User ruling 2026-10-11: under 0.90 of the outlines with exactly one nucleus the
    re-run is another segmentation. It is not used, and everything says so."""
    tmp, lab, row = cohort
    fake_segment(tmp, labels_of(shift=(20, 20)), monkeypatch)  # nuclei of elsewhere
    manifest = manifest_of(tmp, row)
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, segment=True, index=0)) == 1  # the task fails
    (ref, mov) = nuclei.slides(multiplex.read_manifest(manifest))
    store = nuclei.store_of(tmp / "nuc", ref)
    rec = nuclei.load(store)
    frac = rec["agreement"]["one_nucleus_frac"]
    assert frac < nuclei.ACCEPT_ONE_NUCLEUS
    assert rec["refused"] and not rec["usable"] and not rec["exact"]
    assert rec["source"] == "" and not rec["needs_rerun"]
    assert not (store / "nuclei.npz").exists()
    n_out, n_nuc = rec["agreement"]["n_outlines"], rec["agreement"]["n_nuclei"]
    assert (
        f"only {frac:.3f} of this slide's {n_out} published outlines" in (rec["reason"])
    )
    assert f"{n_nuc} nuclei (at least 0.9 is required)" in rec["reason"]
    said = capsys.readouterr().out
    assert f"REFUSED P1/P1.r1: {rec['reason']}" in said
    assert "DISAGREEMENT" not in said  # that line is for a re-run that is KEPT
    # the list: its own status, the share and the reason
    assert nuclei.main(args(**base, list=True)) == 0
    listed = capsys.readouterr()
    line = listed.out.splitlines()[1].split("\t")
    assert line[:3] == ["0", "P1/P1.r1", "refused"] and line[4] == "False"
    assert line[5] == f"{frac:.4f}" and "REFUSED" in line[7]
    assert "1 whose re-run was REFUSED" in listed.err
    # it is not segmented again unasked, and no command is printed for it
    calls = []
    monkeypatch.setattr(nuclei, "segment", lambda *a: calls.append(a) or 0)
    assert nuclei.main(args(**base, segment=True, index=0)) == 1 and not calls
    capsys.readouterr()
    assert nuclei.main(args(**base, command=True, index=0)) == 0
    assert capsys.readouterr().out == ""
    assert nuclei.main(args(**base, segment=True, index=0, force=True)) == 1 and calls
    # the case is prepared on its outlines, without nuclei, with that reason
    cases_root = tmp / "cases"
    a = mx_args(cases_root, manifest, nuclei_dir=str(tmp / "nuc"))
    assert multiplex.prepare(a) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    assert polys(points, "ref", "nuclei") is None
    info = cases_mod.set_info(case, "ref", "nuclei")
    assert info["source"] == "" and info["reason"] == rec["reason"]
    assert "REFUSED" in cases_mod.nuclei_reason(case, points)


def test_a_rerun_between_the_thresholds_is_still_kept_and_not_exact(
    cohort, monkeypatch, capsys
):
    tmp, lab, row = cohort
    fake_segment(tmp, labels_of(drop=3), monkeypatch)  # 0.946: over 0.90, under 0.99
    a = args(manifest=manifest_of(tmp, row), out=tmp / "nuc", segment=True, index=0)
    assert nuclei.main(a) == 0
    (ref, _) = nuclei.slides(multiplex.read_manifest(a.manifest))
    rec = nuclei.load(nuclei.store_of(tmp / "nuc", ref))
    frac = rec["agreement"]["one_nucleus_frac"]
    assert nuclei.ACCEPT_ONE_NUCLEUS <= frac < nuclei.EXACT_ONE_NUCLEUS
    assert rec["usable"] and not rec["exact"] and not rec.get("refused")
    said = capsys.readouterr().out
    assert "DISAGREEMENT P1/P1.r1" in said and "REFUSED" not in said


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
    mask = mask_file(tmp / "work" / "P1.r1.ome_nuclei_mask.tif", lab)
    manifest = manifest_of(tmp, row, ref_nuclei=str(mask), mov_nuclei="")
    base = dict(manifest=manifest, out=tmp / "nuc")
    assert nuclei.main(args(**base, count=True)) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert nuclei.main(args(**base)) == 0  # resolve: nothing is segmented unasked
    capsys.readouterr()
    assert nuclei.main(args(**base, list=True)) == 0
    said = capsys.readouterr()
    lines = [ln.split("\t") for ln in said.out.splitlines()]
    assert lines[0] == list(nuclei.COLUMNS)
    assert lines[1][:5] == ["0", "P1/P1.r1", "usable", f"label_image:{mask}", "True"]
    assert lines[2][:3] == ["1", "P1/P1.r2", "needs_rerun"]
    assert "1 with usable nuclei, 1 needing the StarDist re-run" in said.err
    # the command is printed only for the slide that needs it
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
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
    mask = mask_file(tmp / "work" / "P1.r1.ome_nuclei_mask.tif", lab)
    manifest = manifest_of(tmp, row, ref_nuclei=str(mask))
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
    ref, mov = (cases_mod.set_info(case, s, "nuclei") for s in ("ref", "mov"))
    assert ref["source"] == f"label_image:{mask}" and ref["exact"] is True
    assert mov["source"] == "stardist_rerun" and mov["exact"] is True
    assert mov["agreement"]["one_nucleus_frac"] == 1.0
    assert cases_mod.nuclei_reason(case, points) == ""


def test_a_slide_with_no_nuclei_is_still_prepared_with_the_reason(cohort):
    tmp, lab, row = cohort
    cases_root = tmp / "cases"
    assert multiplex.prepare(mx_args(cases_root, manifest_of(tmp, row))) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    assert polys(points, "mov").n_features == lab.max()
    assert polys(points, "mov", "nuclei") is None
    info = cases_mod.set_info(case, "mov", "nuclei")
    assert info["source"] == "" and "no nuclear label image" in info["reason"]
    assert "ref: no nuclear label image" in cases_mod.nuclei_reason(case, points)


def test_a_label_image_in_the_manifest_is_checked_without_the_nuclei_step(cohort):
    tmp, lab, row = cohort
    good = mask_file(tmp / "work" / "a.tif", lab)
    bad = mask_file(tmp / "work" / "b.tif", labels_of(shift=(20, 20)))
    manifest = manifest_of(tmp, row, ref_nuclei=str(good), mov_nuclei=str(bad))
    cases_root = tmp / "cases"
    assert multiplex.prepare(mx_args(cases_root, manifest)) == 0
    (case,) = list_cases(cases_root, "multiplex")
    points = load_points(case)
    assert polys(points, "ref", "nuclei").n_features == lab.max()
    assert polys(points, "mov", "nuclei") is None
    assert (
        "at least 0.9 is required"
        in cases_mod.set_info(case, "mov", "nuclei")["reason"]
    )


def test_the_shrunk_outline_is_used_only_when_asked_and_is_never_exact(cohort):
    tmp, lab, row = cohort
    manifest = manifest_of(tmp, row)
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


# ── the manifest: masks that already exist ────────────────────────────────────
def _arms_tree(root):
    from .test_arms_manifest import P1, _tree

    _tree(root, P1)
    return root


def test_the_manifest_finds_masks_kept_in_a_work_directory(tmp_path):
    root = _arms_tree(tmp_path)
    work = tmp_path / "work" / "ab" / "cdef"
    work.mkdir(parents=True)
    (work / "P1.r2.ome_nuclei_mask.tif").touch()
    (work / "P1.r2.ome_cell_mask.tif").touch()
    older = tmp_path / "work" / "00" / "1111"
    older.mkdir(parents=True)
    (older / "P1.r2.ome_nuclei_mask.tif").touch()
    import os

    os.utime(older / "P1.r2.ome_nuclei_mask.tif", (1, 1))
    pattern = str(tmp_path / "work" / "*" / "*" / "{stem}_nuclei_mask.tif")
    rows, problems = arms_manifest.build(root, nuclei_globs=[pattern])
    assert problems == []
    by = {r["case_id"]: r for r in rows}
    assert by["P1_P1.r2"]["mov_nuclei"] == str(work / "P1.r2.ome_nuclei_mask.tif")
    assert by["P1_P1.r2"]["mov_nuclei_from"] == "glob (newest of 2)"
    assert by["P1_P1.r3"]["mov_nuclei"] == "" and by["P1_P1.r3"]["ref_nuclei"] == ""
    assert set(arms_manifest.NUCLEI_COLUMNS) <= set(arms_manifest.COLUMNS)


def test_the_manifest_takes_a_segmentation_arms_mask_for_the_reference_only(tmp_path):
    root = _arms_tree(tmp_path)
    mask = root / "seg_stardist" / "P1" / "segmentation" / "P1_nuclei_mask.tif"
    mask.parent.mkdir(parents=True)
    mask.touch()
    checkpoint = root / "seg_stardist" / "csv" / "segmented.csv"
    checkpoint.parent.mkdir(parents=True)
    with open(checkpoint, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "id", "is_reference", "nuclei_mask"])
        w.writerow(["P1", "P1.r1", "true", str(mask)])
        w.writerow(["P1", "P1.r2", "false", str(mask)])
    rows, _ = arms_manifest.build(root, seg_arm="seg_stardist")
    assert {r["ref_nuclei"] for r in rows} == {str(mask)}
    assert {r["ref_nuclei_from"] for r in rows} == {"seg_arm:seg_stardist"}
    assert {r["mov_nuclei"] for r in rows} == {""}
    out = tmp_path / "rounds.csv"
    arms_manifest.write(rows, out)
    assert multiplex.read_manifest(out)[0]["ref_nuclei"] == str(mask)
    with pytest.raises(SystemExit, match="segmented.csv"):
        arms_manifest.build(root, seg_arm="no_such_arm")


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
