"""The cohort's rounds as manifest rows, from regbench's own preprocessing checkpoint
(``launch.sbatch --inputs``: ``<inputs>/preprocess/results/csv/preprocessed.csv``)."""

import csv

import pytest

from benchmarks.regbench import manifest
from benchmarks.regbench.datasets import multiplex

HEADER = ("patient_id", "id", "preprocessed_image", "is_reference", "channels")


def _outline(root, patient, slide):
    return root / "outlines" / patient / f"{slide}.geojson"


def _tree(root, slides, pixel_size="0.325", header=None):
    """``slides``: ``(patient, file name, is_reference, channels)``, in checkpoint order.
    The slides are published under ``results/<patient>/converted/``, as the pipeline's
    preprocessing step publishes them."""
    header = header or (*HEADER, "pixel_size")
    lines = []
    for patient, name, is_ref, channels in slides:
        image = root / "results" / patient / "converted" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
        row = {
            "patient_id": patient,
            "id": manifest.slide_name(name),
            "preprocessed_image": str(image),
            "is_reference": "true" if is_ref else "false",
            "channels": channels,
            "pixel_size": pixel_size,
        }
        lines.append({k: row[k] for k in header})
    path = root / "results" / "csv" / "preprocessed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(lines)
    return path


def _build(root, **kw):
    return manifest.build(
        root / "results" / "csv" / "preprocessed.csv", root / "outlines", **kw
    )


P1 = [
    ("P1", "P1.r1.ome.tif", True, "DAPI|CD3"),
    ("P1", "P1.r2.ome.tif", False, "CD8|DAPI"),
    ("P1", "P1.r3.ome.tiff", False, "DAPI|PANCK|KI67"),
]


def test_one_row_per_moving_round(tmp_path):
    _tree(
        tmp_path,
        [*P1, ("P2", "a.ome.tif", True, "DAPI"), ("P2", "b.tif", False, "DAPI")],
    )
    rows, problems = _build(tmp_path)
    assert problems == []
    assert [r["case_id"] for r in rows] == ["P1_P1.r2", "P1_P1.r3", "P2_b"]
    assert [r["group"] for r in rows] == ["P1", "P1", "P2"]
    first = rows[0]
    assert first["reference"].endswith("P1/converted/P1.r1.ome.tif")
    assert first["moving"].endswith("P1/converted/P1.r2.ome.tif")
    # where the nuclei step will trace each slide's outlines: not there yet, and that
    # is not a reason to drop the round
    assert first["ref_cells"] == str(_outline(tmp_path, "P1", "P1.r1"))
    assert first["mov_cells"] == str(_outline(tmp_path, "P1", "P1.r2"))
    assert not _outline(tmp_path, "P1", "P1.r1").exists()
    assert first["channels_ref"] == "DAPI|CD3" and first["channels_mov"] == "CD8|DAPI"
    assert first["pixel_size_um"] == pytest.approx(0.325)
    assert first["pixel_size_source"] == "preprocessed.csv"
    assert set(manifest.COLUMNS) == set(first)
    # the bare-nuclei columns stay in the schema, empty
    assert {first[c] for c in manifest.NUCLEI_COLUMNS} == {""}


def test_nuclear_channel_is_found_by_name(tmp_path):
    assert manifest.nuclear_index(["CD3", "DAPI"]) == 1
    assert manifest.nuclear_index(["cd3", " dapi "]) == 1  # case-insensitive
    # the pipeline's rule is a SUBSTRING match of DAPI or CELLTOX (lib/MarkerUtils.groovy)
    assert manifest.nuclear_index(["CD8", "PANCK", "CELLTOX Green"]) == 2
    assert manifest.nuclear_index(["CD8", "DAPI_R2"]) == 1
    assert manifest.nuclear_index(["DAPI", "CELLTOX"], nuclear="CELLTOX") == 1
    with pytest.raises(ValueError, match="no channel"):
        manifest.nuclear_index(["CD3", "CD8"])
    with pytest.raises(ValueError, match="2 channels"):
        manifest.nuclear_index(["DAPI", "CD3", "CELLTOX Green"])

    _tree(tmp_path, P1)
    rows, _ = _build(tmp_path)
    assert [(r["ref_channel"], r["mov_channel"]) for r in rows] == [(0, 1), (0, 0)]


def test_a_slide_with_two_nuclear_channels_is_refused_with_the_reason(tmp_path):
    slides = [*P1, ("P1", "P1.r4.ome.tif", False, "DAPI|CELLTOX Green|CD4")]
    _tree(tmp_path, slides)
    rows, problems = _build(tmp_path)
    assert [r["case_id"] for r in rows] == ["P1_P1.r2", "P1_P1.r3"]
    (p,) = problems
    assert p["patient"] == "P1" and p["slide"] == "P1.r4"
    assert "2 channels" in p["reason"] and "CELLTOX Green" in p["reason"]
    assert p["path"].endswith("P1.r4.ome.tif")


def test_a_slide_without_the_nuclear_name_is_dropped(tmp_path):
    _tree(tmp_path, [*P1, ("P1", "P1.r4.ome.tif", False, "CD4|CD8")])
    rows, problems = _build(tmp_path)
    assert len(rows) == 2
    assert [p["slide"] for p in problems] == ["P1.r4"]
    assert "no channel" in problems[0]["reason"]


def test_slide_names_follow_the_pipeline_rule():
    # subworkflows/local/seg_qc.nf: strip .ome.tif / .ome.tiff, then .tif / .tiff
    assert manifest.slide_name("P1.r2.ome.tif") == "P1.r2"
    assert manifest.slide_name("/x/y/P1.r2.ome.tiff") == "P1.r2"
    assert manifest.slide_name("P1.r2.tif") == "P1.r2"
    assert manifest.slide_name("P1.r2.tiff") == "P1.r2"
    assert manifest.slide_name("P1.ome.r2.tif") == "P1.ome.r2"
    assert manifest.slide_name("P1.r2.OME.TIF") == "P1.r2.OME.TIF"  # case-sensitive
    # the two replacements run one after the other, as in the workflow
    assert manifest.slide_name("P1.r2.ome.tif.ome.tif") == "P1.r2.ome"


def test_patient_without_reference_is_dropped_with_a_reason(tmp_path):
    slides = [
        ("P1", "P1.r1.ome.tif", False, "DAPI"),
        ("P1", "P1.r2.ome.tif", False, "DAPI"),
        ("P2", "a.ome.tif", True, "DAPI"),
        ("P2", "b.ome.tif", False, "DAPI"),
        ("P3", "a.ome.tif", True, "DAPI"),
        ("P3", "b.ome.tif", True, "DAPI"),
    ]
    path = _tree(tmp_path, slides)
    rows, problems = _build(tmp_path)
    assert [r["case_id"] for r in rows] == ["P2_b"]
    by = {p["patient"]: p for p in problems}
    assert "no reference" in by["P1"]["reason"] and by["P1"]["path"] == str(path)
    assert "2 reference" in by["P3"]["reason"]


def test_a_missing_slide_drops_the_round_but_a_converted_copy_is_found(tmp_path):
    path = _tree(tmp_path, P1)
    # rows written before the passthrough fix name <pid>/preprocessed/ for a file that was
    # published under <pid>/converted/ (benchmarks/reg_mosaic.py: published_file)
    path.write_text(path.read_text().replace("/converted/P1.r2", "/preprocessed/P1.r2"))
    (tmp_path / "results" / "P1" / "converted" / "P1.r3.ome.tiff").unlink()
    rows, problems = _build(tmp_path)
    assert [r["case_id"] for r in rows] == ["P1_P1.r2"]
    assert rows[0]["moving"].endswith("/converted/P1.r2.ome.tif")
    (p,) = problems
    assert p["slide"] == "P1.r3" and "slide" in p["reason"]
    assert p["path"].endswith("P1.r3.ome.tiff")


def test_a_missing_reference_drops_every_round_of_the_patient(tmp_path):
    _tree(
        tmp_path,
        [*P1, ("P2", "a.ome.tif", True, "DAPI"), ("P2", "b.tif", False, "DAPI")],
    )
    gone = tmp_path / "results" / "P1" / "converted" / "P1.r1.ome.tif"
    gone.unlink()
    rows, problems = _build(tmp_path)
    assert [r["case_id"] for r in rows] == ["P2_b"]
    assert {p["path"] for p in problems} == {str(gone)}
    assert sorted(p["slide"] for p in problems) == ["P1.r2", "P1.r3"]


def test_pixel_size_falls_back_to_the_cohorts_documented_value(tmp_path):
    _tree(tmp_path, P1, pixel_size="")
    rows, _ = _build(tmp_path)
    assert {r["pixel_size_um"] for r in rows} == {0.325}
    assert {r["pixel_size_source"] for r in rows} == {"fallback"}

    _tree(tmp_path, P1, header=HEADER)  # a checkpoint older than the pixel_size column
    rows, _ = _build(tmp_path)
    assert {r["pixel_size_source"] for r in rows} == {"fallback"}

    _tree(tmp_path, P1, pixel_size="0.5")
    rows, _ = _build(tmp_path)
    assert {r["pixel_size_um"] for r in rows} == {0.5}


def test_quoted_fields_are_read_as_csv(tmp_path):
    path = _tree(
        tmp_path,
        [("P1", "a, b.ome.tif", True, "DAPI"), ("P1", "c.ome.tif", False, "DAPI")],
    )
    assert '"' in path.read_text()  # RFC 4180 quoting, as the checkpoint writer does
    rows, problems = _build(tmp_path)
    assert problems == []
    assert rows[0]["reference"].endswith("a, b.ome.tif")


def test_a_missing_checkpoint_or_column_stops_with_the_path(tmp_path):
    with pytest.raises(SystemExit, match="preprocessed.csv.*launch.sbatch --inputs"):
        _build(tmp_path)
    _tree(tmp_path, P1, header=("patient_id", "id", "preprocessed_image", "channels"))
    with pytest.raises(SystemExit, match="is_reference"):
        _build(tmp_path)


def test_cli_writes_the_good_rows(tmp_path, capsys):
    checkpoint = _tree(tmp_path, P1)
    gone = tmp_path / "results" / "P1" / "converted" / "P1.r2.ome.tif"
    gone.unlink()
    out = tmp_path / "made" / "rounds.csv"
    argv = ["--preprocessed", str(checkpoint), "--outlines", str(tmp_path / "outlines")]
    assert manifest.main([*argv, "--out", str(out)]) == 0
    assert str(gone) in capsys.readouterr().out  # the problems are printed
    rows = multiplex.read_manifest(out)  # what `prepare multiplex` reads
    assert [r["case_id"] for r in rows] == ["P1_P1.r3"]
    assert rows[0]["group"] == "P1" and rows[0]["mov_channel"] == "0"
    assert rows[0]["channels_mov"] == "DAPI|PANCK|KI67"
    assert rows[0]["mov_cells"] == str(_outline(tmp_path, "P1", "P1.r3"))

    (tmp_path / "results" / "P1" / "converted" / "P1.r3.ome.tiff").unlink()
    none = tmp_path / "none.csv"
    assert manifest.main([*argv, "--out", str(none)]) == 1
    assert not none.exists()  # no row is left: nothing is written
    # both inputs are required; the arms' options are gone
    for bad in (["--out", str(none)], ["--root", str(tmp_path), "--out", str(none)]):
        with pytest.raises(SystemExit):
            manifest.main(bad)
