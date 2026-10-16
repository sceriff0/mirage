"""The cohort's outlines are defined to be what the arm (`valis_high_micro2`, reg_qc=2)
published: SEGMENT's StarDist call grown by seg_expand_distance, traced by
bin/mask_to_geojson.py at simplify_tolerance. Each link is pinned here.

The equivalence is DERIVED from the code, link by link; it was never measured on cohort
slides (no StarDist and no site model where these tests run). What no test here can pin:
the site's model files, the library versions inside the images, and a GPU's
run-to-run differences.
"""

import json
import pathlib
import re

import numpy as np

from benchmarks.regbench import cells, manifest, nuclei, outlines, site_config

ROOT = pathlib.Path(__file__).resolve().parents[3]
NO_SITE = {"REGBENCH_SITE_CONFIG": "none"}


def _default(key):
    m = re.search(
        rf"^\s*{key}\s*=\s*([^\s/]+)", (ROOT / "nextflow.config").read_text(), re.M
    )
    return m.group(1).strip("'\"")


def _bin_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "bin" / "utils"))
    monkeypatch.syspath_prepend(str(ROOT / "bin"))


def test_the_rerun_grows_nuclei_by_the_pipelines_expand_distance():
    slide = {"image": "/x/P1.r1.ome.tif", "channel": 0}
    argv = nuclei.command(
        slide, "/tmp/d", env={"REGBENCH_STARDIST_MODEL": "m", **NO_SITE}
    )
    assert argv[argv.index("--expand-distance") + 1] == _default("seg_expand_distance")
    assert _default("seg_expand_distance") == "10"


def test_the_rerun_passes_every_flag_segment_ext_args_renders_for_stardist():
    text = (ROOT / "conf" / "modules.config").read_text()
    block = text[text.index("stardist  : [") : text.index("instantseg: [")]
    rendered = set(re.findall(r'"(--[a-z-]+)', block)) | set(
        re.findall(r"'(--[a-z-]+)", block)
    )
    assert {"--model-name", "--n-tiles", "--expand-distance"} <= rendered
    argv = nuclei.command(
        {"image": "/x/a.tif", "channel": 0},
        "/tmp/d",
        env={
            "REGBENCH_STARDIST_MODEL": "m",
            "REGBENCH_STARDIST_MODEL_DIR": "/m",
            "REGBENCH_STARDIST_PROB_THRESH": "0.5",
            **NO_SITE,
        },
    )
    assert rendered <= set(a for a in argv if a.startswith("--"))


def test_the_model_and_settings_are_the_params_ext_args_reads():
    """The model is the pipeline's segmentation_model / segmentation_model_dir, and
    every setting is the param of the same name, resolved like the pipeline: the site
    config over nextflow.config."""
    text = (ROOT / "conf" / "modules.config").read_text()
    block = text[text.index("stardist  : [") : text.index("instantseg: [")]
    read = set(re.findall(r"params\.([a-z_]+)", block))
    assert read == set(site_config.KEYS)


def test_the_rerun_runs_in_the_pipelines_stardist_image_on_channel_zero():
    """SegBackends: the pipeline's StarDist reads channel 0, and refuses a slide whose
    channel 0 is not nuclear. The manifest's nuclear index is 0 on such a slide, and a
    slide where it is not 0 is one the pipeline would have refused."""
    backend = (ROOT / "lib" / "SegBackends.groovy").read_text()
    assert "flags     : { ctx -> '--dapi-channel 0' }" in backend
    assert "isNuclear(channels ? channels[0] : null" in backend
    assert f"container : '{nuclei.PIPELINE_IMAGE}'" in backend
    assert manifest.nuclear_index(["DAPI", "CD3", "CD8"]) == 0
    argv = nuclei.command(
        {"image": "/x/a.tif", "channel": 0},
        "/d",
        env={"REGBENCH_STARDIST_MODEL": "m", **NO_SITE},
    )
    assert argv[argv.index("--dapi-channel") + 1] == "0"
    # the pipeline asks for the GPU by default (seg_gpu), and so does the re-run
    assert _default("seg_gpu") == "true" and "--use-gpu" in argv


def test_the_cell_mask_is_skimage_expand_labels_of_the_nuclei(monkeypatch):
    _bin_on_path(monkeypatch)
    from segment_io import expand_labels_tiled

    rng = np.random.default_rng(0)
    lab = np.zeros((300, 300), np.int32)
    for i, (y, x) in enumerate(rng.integers(15, 285, (40, 2)), 1):
        lab[y - 4 : y + 4, x - 4 : x + 4] = i
    np.testing.assert_array_equal(
        expand_labels_tiled(lab, distance=10), cells.grow(lab, 10)
    )
    segment = (ROOT / "bin" / "segment.py").read_text()
    assert "expand_labels_tiled(nuclei_labels, distance=expand_distance)" in segment


def test_outlines_are_traced_by_the_pipelines_own_function_at_its_tolerance(
    tmp_path, monkeypatch
):
    import tifffile

    _bin_on_path(monkeypatch)
    import mask_to_geojson

    lab = np.zeros((64, 64), np.uint16)
    lab[10:30, 10:30] = 1
    lab[35:60, 30:50] = 2
    tifffile.imwrite(tmp_path / "m.tif", lab)
    tol = float(_default("simplify_tolerance"))
    assert outlines.tolerance(env=NO_SITE) == tol
    monkeypatch.setenv("REGBENCH_SITE_CONFIG", "none")
    assert outlines.trace(tmp_path / "m.tif", tmp_path / "ours.geojson") == 2
    mask_to_geojson.write_geojson(tmp_path / "m.tif", tmp_path / "theirs.geojson", tol)
    ours = json.loads((tmp_path / "ours.geojson").read_text())
    assert ours == json.loads((tmp_path / "theirs.geojson").read_text())
    # the module traced is the repository's bin/mask_to_geojson.py itself
    assert pathlib.Path(outlines.MASK_TO_GEOJSON) == ROOT / "bin" / "mask_to_geojson.py"


def test_the_tolerance_follows_the_site_config_like_the_pipeline(tmp_path):
    site = tmp_path / "site.config"
    site.write_text("params {\n    simplify_tolerance = 2.5   // site\n}\n")
    assert outlines.tolerance(env={"REGBENCH_SITE_CONFIG": str(site)}) == 2.5
    site.write_text("params {\n    // simplify_tolerance = 9\n}\n")
    assert outlines.tolerance(env={"REGBENCH_SITE_CONFIG": str(site)}) == 1.0


def test_seg_qc_geojson_still_traces_with_mask_to_geojson_at_simplify_tolerance():
    mod = (ROOT / "modules" / "local" / "seg_qc_geojson.nf").read_text()
    assert "mask_to_geojson.py" in mod
    config = (ROOT / "conf" / "modules.config").read_text()
    assert '"--tolerance ${params.simplify_tolerance}"' in config
    qc = (ROOT / "subworkflows" / "local" / "seg_qc.nf").read_text()
    # the QC segments through SEGMENT itself and traces ITS cell mask
    assert "include { SEGMENT as SEG_QC_SEGMENT }" in qc
    assert "SEG_QC_GEOJSON(SEG_QC_SEGMENT.out.cell_mask)" in qc
