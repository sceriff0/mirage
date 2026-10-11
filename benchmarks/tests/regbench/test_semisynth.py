"""Real images under a known map: the copied semi-synthetic suite (ported from
sceriff0/stare @ f5ce62e, ``benchmarks/regbench/tests/test_extensions.py``)."""

import hashlib
import json
import shutil

import numpy as np
import pytest

from benchmarks.regbench import arms_manifest, score, truth
from benchmarks.regbench.cases import list_cases, load_case, load_points
from benchmarks.regbench.datasets import semisynth, synthetic

from .conftest import semisynth_args


def test_semisynth_cases_carry_the_truth(real_like, tmp_path):
    a = semisynth_args(real_like, tmp_path / "cases")
    jobs, noise = semisynth.plan(a)
    assert noise > 0
    names = [j[0] for j in jobs]
    assert sum(n.startswith("dev_") for n in names) == len(semisynth.FAMILIES)
    assert len(names) == 3 * len(semisynth.FAMILIES)
    dev_origin = {j[2] for j in jobs if j[0].startswith("dev_")}
    assert not dev_origin & {j[2] for j in jobs if not j[0].startswith("dev_")}
    assert len({j[1]["seed"] for j in jobs}) == len(jobs)

    name, spec, origin = next(j for j in jobs if j[0].endswith("w0_multiscale"))
    semisynth.prepare_one(a.cases, name, spec, a, origin, noise, 0.5)
    case = load_case(tmp_path / "cases" / "semisynth" / name)
    assert case.pixel_size_um == 0.5 and case.extra["variant"] == "same"
    assert case.group == name and case.family == "multiscale" and case.pose == ""
    pts = load_points(case)
    assert len(pts["truth_mov"]) > 50
    np.testing.assert_allclose(
        synthetic.forward_map(spec)(pts["truth_mov"]), pts["truth_ref"]
    )
    done = score.score(a.cases, tmp_path / "out", ["semisynth"])
    by = {r["method"]: r for r in done["cells"]}
    assert by["truth"]["dice_matched"] > 0.8 > by["initial"]["dice_matched"]
    assert by["truth"]["displacement_px_p50"] < 1.0
    # a real-image case under a literature-derived map is in the headline tier
    assert {r["tier"] for r in done["cells"]} == {"headline"}
    # and it has a known-truth row: the unregistered pose is tens of pixels off
    known = {r["method"]: r for r in done["truth"]}
    assert known["initial"]["tre_um_p50"] > 10.0 and known["truth"]["tre_um_max"] == 0.0
    assert known["initial"]["truth_spec"] == ""  # not one of the frozen cases


def test_multiscale_amplitude_is_the_measured_residual():
    """RMS displacement of the copied multiscale field is --residual-um, in pixels."""
    n, px = 4096, 0.325
    kw = semisynth.family_spec("multiscale", n, px, 2.0, seed=1)
    assert kw["scales"] == (512.0, 256.0, 128.0)
    spec = {**synthetic._BASE, "n": n, "seed": 1, **kw}
    xs = np.linspace(0, n - 1, 200)
    dx, dy = synthetic.extra_field(spec)(*np.meshgrid(xs, xs))
    assert abs(np.sqrt(np.mean(dx**2 + dy**2)) * px - 2.0) < 0.3
    sd = semisynth.family_spec("grid", n, px, 2.0, seed=1)["grid_sd"]
    assert 3.0 <= sd <= 7.0


def test_semisynth_reference_is_the_untouched_window(real_like, tmp_path):
    import tifffile

    a = semisynth_args(real_like, tmp_path / "cases")
    name, spec, origin = semisynth.plan(a)[0][0]
    semisynth.prepare_one(a.cases, name, spec, a, origin, 0.0, 0.5)
    x, y = origin
    whole = np.squeeze(tifffile.imread(real_like))
    got = np.squeeze(
        tifffile.imread(tmp_path / "cases" / "semisynth" / name / "ref.ome.tif")
    )
    np.testing.assert_array_equal(got, whole[y : y + 512, x : x + 512])


# ── the frozen known-truth cases, from the cohort's own list ──────────────────
def _small_spec():
    """The frozen spec at a 512 px window, hashed as its own: prepared exactly as the
    real one is, in seconds."""
    spec = {**truth.load_spec(), "n_px": 512}
    spec["sha256"] = truth.spec_hash(spec)
    return spec


def _rounds_csv(path, image, patients=("046", "052", "107"), rounds=2):
    rows = [
        {
            **dict.fromkeys(arms_manifest.COLUMNS, ""),
            "case_id": f"{p}_r{k}",
            "reference": str(image),
            "moving": f"/nowhere/{p}_r{k}.ome.tif",
            "group": p,
            "ref_channel": 0,
            "mov_channel": 0,
            "pixel_size_um": 0.5,
        }
        for p in patients
        for k in range(1, rounds + 1)
    ]
    return arms_manifest.write(rows, path)


def _truth_args(cases, **kw):
    """``prepare semisynth --truth`` as the command line gives it: no image unless named."""
    base = dict(truth=True, name=None, manifest=None, patients=None, patients_file=None)
    image = kw.pop("image", None)
    a = semisynth_args("", cases, **{**base, "pixel_size_um": None, **kw})
    a.image = image
    return a


def test_manifest_gives_one_entry_per_patient(real_like, tmp_path):
    rounds = _rounds_csv(tmp_path / "rounds.csv", real_like)
    got = semisynth.read_manifest(rounds)
    assert list(got) == ["046", "052", "107"]  # file order, one per `group`
    assert got["052"] == {
        "image": str(real_like),
        "channel": 0,
        "pixel_size_um": 0.5,
    }
    # a patient whose rounds name two references is refused, by name
    text = rounds.read_text().replace(str(real_like), "/other/ref.ome.tif", 1)
    (tmp_path / "two.csv").write_text(text)
    with pytest.raises(SystemExit, match="046.*more than one reference"):
        semisynth.read_manifest(tmp_path / "two.csv")

    # the arms' raw samplesheet: the reference row and the nuclear channel by name
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\n"
        "P1,/raw/P1_r1.ome.tif,false,CD3|DAPI\n"
        "P1,/raw/P1_ref.ome.tif,true,PANCK|SMA|DAPI\n"
        "P2,/raw/P2_ref.ome.tif,TRUE,DAPI|CD8\n"
    )
    assert semisynth.read_manifest(sheet) == {
        "P1": {"image": "/raw/P1_ref.ome.tif", "channel": 2, "pixel_size_um": None},
        "P2": {"image": "/raw/P2_ref.ome.tif", "channel": 0, "pixel_size_um": None},
    }
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/raw/a.tif,false,DAPI\n"
    )
    with pytest.raises(SystemExit, match="P1.*no reference row"):
        semisynth.read_manifest(sheet)
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/raw/a.tif,true,CD3|CD8\n"
    )
    with pytest.raises(SystemExit, match="P1.*no channel"):
        semisynth.read_manifest(sheet)
    sheet.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="neither"):
        semisynth.read_manifest(sheet)


def test_a_declared_subset_must_be_in_the_manifest(real_like, tmp_path):
    rounds = _rounds_csv(tmp_path / "rounds.csv", real_like)
    spec = _small_spec()
    a = _truth_args(tmp_path / "c", manifest=str(rounds))
    jobs, record = semisynth.plan_truth_manifest(a, spec)
    assert record["patients"] == ["046", "052", "107"]
    assert len(jobs) == 3 * 24 == 3 * semisynth.cases_per_patient(spec)
    assert record["manifest"] == str(rounds.resolve())
    assert record["manifest_sha256"] == hashlib.sha256(rounds.read_bytes()).hexdigest()
    assert record["truth_spec_sha256"] == spec["sha256"]
    assert record["sha256"] == truth.patients_hash(record)
    name, job, origin, pa, noise, px = jobs[24]
    assert name == "052_w0_rigid_shift" and pa.name == "052" and px == 0.5
    assert pa.image == str(real_like) and pa.channel == 0 and noise > 0
    assert job["truth_spec_sha256"] == spec["sha256"]
    assert job["truth_patients_sha256"] == record["sha256"]
    assert job["truth_patient"] == "052"

    # a subset, in the manifest's order whatever order it is declared in
    sub = _truth_args(tmp_path / "c", manifest=str(rounds), patients="107,046")
    jobs_sub, rec_sub = semisynth.plan_truth_manifest(sub, spec)
    assert rec_sub["patients"] == ["046", "107"] and len(jobs_sub) == 48
    assert rec_sub["sha256"] != record["sha256"]
    # the same patient gets the same cases whoever else is in the list
    assert jobs_sub[24][0] == "107_w0_rigid_shift"
    assert {
        k: v for k, v in jobs_sub[24][1].items() if k != "truth_patients_sha256"
    } == {k: v for k, v in jobs[48][1].items() if k != "truth_patients_sha256"}
    listed = tmp_path / "ids.txt"
    listed.write_text("# the pilot\n107\n046, 052\n")
    by_file = _truth_args(
        tmp_path / "c", manifest=str(rounds), patients_file=str(listed)
    )
    assert semisynth.plan_truth_manifest(by_file, spec)[1]["sha256"] == record["sha256"]

    bad = _truth_args(tmp_path / "c", manifest=str(rounds), patients="046,999,12")
    with pytest.raises(SystemExit, match=r"999, 12.*not in"):
        semisynth.plan_truth_manifest(bad, spec)
    both = _truth_args(tmp_path / "c", manifest=str(rounds), patients="046",
        patients_file=str(listed),
    )  # fmt: skip
    with pytest.raises(SystemExit, match="one of --patients and --patients-file"):
        semisynth.plan_truth_manifest(both, spec)
    with pytest.raises(SystemExit, match="--manifest"):
        semisynth.prepare(_truth_args(tmp_path / "c", patients="046"), spec)
    with pytest.raises(SystemExit, match="--truth"):
        semisynth.prepare(
            _truth_args(tmp_path / "c", manifest=str(rounds), truth=False),
            spec,
        )
    # --index plans one patient only and reaches the same case
    one = _truth_args(tmp_path / "c", manifest=str(rounds), index=49)
    (only,), rec_one = semisynth.plan_truth_manifest(one, spec)
    assert only[0] == jobs[49][0] == "107_w0_rigid_rot" and rec_one == record
    past = _truth_args(tmp_path / "c", manifest=str(rounds), index=72)
    assert semisynth.plan_truth_manifest(past, spec)[0] == []


def _prepared(real_like, root, rounds, spec, index, **kw):
    a = _truth_args(root, manifest=str(rounds), index=index, **kw)
    assert semisynth.prepare(a, spec) == 0
    return a


def test_the_patient_list_is_written_beside_the_cases(real_like, tmp_path, capsys):
    rounds = _rounds_csv(tmp_path / "rounds.csv", real_like)
    spec = _small_spec()
    root = tmp_path / "cases"
    _prepared(real_like, root, rounds, spec, 0)
    _prepared(real_like, root, rounds, spec, 25)  # another array task, the same list
    record = json.loads((root / "semisynth" / truth.PATIENTS_FILE).read_text())
    assert record["patients"] == ["046", "052", "107"]
    assert record["manifest"] == str(rounds.resolve())
    assert record["truth_spec_sha256"] == spec["sha256"]
    assert record["sources"]["052"] == {"image": str(real_like), "channel": 0}
    cases = list_cases(root, "semisynth")
    assert [c.case_id for c in cases] == ["046_w0_rigid_shift", "052_w0_rigid_rot"]
    for c in cases:
        assert c.extra["truth_patients_sha256"] == record["sha256"]
        assert c.extra["truth_spec_sha256"] == spec["sha256"]
    assert [c.extra["truth_patient"] for c in cases] == ["046", "052"]

    # the summary says which patients the result rests on
    capsys.readouterr()
    res = score.score(root, tmp_path / "out", ["semisynth"])
    assert res["truth_patients"]["sha256"] == record["sha256"]
    text = (tmp_path / "out" / "tables" / "summary.md").read_text()
    head = text[text.index("## Known-truth error") :]
    assert "3 cohort patient(s): 046, 052, 107" in head
    assert "cases are prepared for 2 of them (046, 052)" in head
    assert str(rounds.resolve()) in head and record["manifest_sha256"][:12] in head
    assert spec["sha256"] in head

    # the same folder cannot take cases of another list
    with pytest.raises(SystemExit, match="another patient list"):
        _prepared(real_like, root, rounds, spec, 0, patients="046,052")
    # nor of a manifest that changed
    rounds.write_text(rounds.read_text() + "\n")
    with pytest.raises(SystemExit, match="another patient list"):
        _prepared(real_like, root, rounds, spec, 1)


def test_score_refuses_cases_of_two_patient_lists(real_like, tmp_path):
    rounds = _rounds_csv(tmp_path / "rounds.csv", real_like)
    spec = _small_spec()
    all_root, sub_root = tmp_path / "all", tmp_path / "sub"
    _prepared(real_like, all_root, rounds, spec, 0)
    _prepared(real_like, sub_root, rounds, spec, 1, patients="046")
    assert score.score(all_root, tmp_path / "out_all", ["semisynth"])["truth"]
    assert score.score(sub_root, tmp_path / "out_sub", ["semisynth"])["truth"]
    # a case of the subset's folder dropped into the full list's folder
    shutil.copytree(
        sub_root / "semisynth" / "046_w0_rigid_rot",
        all_root / "semisynth" / "046_w0_rigid_rot",
    )
    with pytest.raises(SystemExit, match="more than one patient list"):
        score.score(all_root, tmp_path / "out_mix", ["semisynth"])
    # one list in the cases, another in the file beside them
    shutil.rmtree(all_root / "semisynth" / "046_w0_rigid_shift")
    with pytest.raises(SystemExit, match="does not match"):
        score.score(all_root, tmp_path / "out_mix", ["semisynth"])
    # and a list file that is gone
    (sub_root / "semisynth" / truth.PATIENTS_FILE).unlink()
    with pytest.raises(SystemExit, match="truth_patients.json"):
        score.score(sub_root, tmp_path / "out_gone", ["semisynth"])


def test_one_patient_form_declares_a_list_of_one(real_like, tmp_path):
    spec = _small_spec()
    root = tmp_path / "cases"
    a = _truth_args(root, name="046", image=str(real_like), index=0)
    a.pixel_size_um = 0.5
    assert semisynth.prepare(a, spec) == 0
    record = json.loads((root / "semisynth" / truth.PATIENTS_FILE).read_text())
    assert record["patients"] == ["046"] and record["manifest"] == ""
    (case,) = list_cases(root, "semisynth")
    assert case.extra["truth_patients_sha256"] == record["sha256"]
    b = _truth_args(root, name="052", image=str(real_like), index=0)
    b.pixel_size_um = 0.5
    with pytest.raises(SystemExit, match="another patient list"):
        semisynth.prepare(b, spec)


def test_cli_takes_the_manifest_form():
    from benchmarks.regbench.cli import build_parser

    a = build_parser().parse_args(
        "prepare semisynth --truth --cases C --manifest rounds.csv "
        "--patients 046,052".split()
    )
    assert (a.manifest, a.patients, a.patients_file) == ("rounds.csv", "046,052", None)
    assert a.image is None and a.name is None and a.truth
    b = build_parser().parse_args(
        "prepare semisynth --truth --cases C --name 046 --image ref.ome.tif "
        "--channel 2".split()
    )
    assert (b.name, b.image, b.channel, b.manifest) == ("046", "ref.ome.tif", 2, None)
    with pytest.raises(SystemExit, match="--image"):
        semisynth.prepare(
            build_parser().parse_args("prepare semisynth --cases C".split())
        )
