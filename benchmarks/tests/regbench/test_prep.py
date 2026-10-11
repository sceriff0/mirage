"""The ``stare_prep`` row: benchmark-side preprocessing of the image STARE registers on.

What must hold: the candidate list is closed and frozen with a hash; no candidate moves
image content or changes a shape; a candidate said to be the identity on one channel IS
(same transform as the plain row, bit for bit); the written image is the ORIGINAL slide,
never the preprocessed pixels; and a label cannot carry another candidate's result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.regbench import methods, prep, variants
from benchmarks.regbench.cases import result_dir
from benchmarks.regbench.methods import stare as adapter

RNG = np.random.default_rng(3)


def smooth_noise(hw, dtype, top):
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(RNG.random(hw), 4)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * top).astype(dtype)


# ── the list and the frozen file ──────────────────────────────────────────────
def test_the_candidate_list_is_small_closed_and_documented():
    assert list(prep.CANDIDATES) == [
        "grey",
        "haem",
        "grey_clahe",
        "haem_clahe",
        "grey_smooth",
    ]
    assert set(prep.SINGLE_CHANNEL) == set(prep.CANDIDATES)
    for name in prep.CANDIDATES:
        assert f"``{name}``" in prep.__doc__
    with pytest.raises(ValueError, match="expected one of"):
        prep.candidate("otsu")
    # what is the identity where: stated, and the same in code
    assert [n for n in prep.CANDIDATES if prep.is_identity(n, False)] == [
        "grey",
        "haem",
    ]
    assert [n for n in prep.CANDIDATES if prep.is_identity(n, True)] == ["grey"]
    for name in ("grey", "haem"):
        assert prep.SINGLE_CHANNEL[name].startswith("identity")


def test_the_frozen_file_is_hashed_and_names_a_candidate():
    record = prep.load_frozen()
    assert record["selected"] in prep.CANDIDATES
    assert record["candidates"] == list(prep.CANDIDATES)
    assert record["status"] in prep.STATUSES
    assert record["criterion"] == prep.CRITERION
    if record["status"] == "provisional":
        assert not record["evidence"] and "PROVISIONAL" in record["note"]
        assert "PROVISIONAL" in prep.frozen_summary()
    else:
        assert record["evidence"] and record["split_sha256"]
        assert record["evidence"][0]["candidate"] == record["selected"]
    # the run list carries the frozen candidate itself
    run = variants.run_of(prep.ROW)
    assert run.opts == {"prep": record["selected"]} and run.method == "stare"


def test_a_hand_edited_frozen_file_is_refused(tmp_path):
    record = prep.load_frozen()
    other = next(c for c in prep.CANDIDATES if c != record["selected"])
    edited = tmp_path / "prep_frozen.json"
    edited.write_text(json.dumps({**record, "selected": other}))
    with pytest.raises(SystemExit, match="does not match its sha256"):
        prep.load_frozen(edited)
    # a selection made over another candidate list is void, even when its hash holds
    stale = prep.make_record("grey", "selected")
    stale["candidates"] = ["grey", "haem"]
    stale["sha256"] = prep.record_hash(stale)
    edited.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="another list is void"):
        prep.load_frozen(edited)


def test_the_criterion_orders_by_median_rtre_then_robustness_then_the_list():
    def row(name, rtre, robust):
        return {
            "candidate": name,
            "median_of_median_rtre": rtre,
            "mean_robustness": robust,
        }

    ranked = prep.rank(
        [
            row("grey", 0.02, 0.9),
            row("haem", 0.01, 0.5),
            row("grey_clahe", 0.01, 0.8),
            row("haem_clahe", 0.01, 0.8),
        ]
    )
    assert [r["candidate"] for r in ranked] == [
        "grey_clahe",
        "haem_clahe",
        "haem",
        "grey",
    ]
    with pytest.raises(ValueError, match="no finite median"):
        prep.rank([row("grey", float("nan"), 0.0)])


# ── what a candidate does to an image ─────────────────────────────────────────
@pytest.mark.parametrize("brightfield", [True, False])
@pytest.mark.parametrize("name", list(prep.CANDIDATES))
def test_no_candidate_moves_image_content(name, brightfield):
    check = prep.frame_check(name, brightfield)
    assert check["ok"] and check["n"] > 40
    assert (
        abs(check["dx"]) < prep.FRAME_GATE_PX and abs(check["dy"]) < prep.FRAME_GATE_PX
    )


def test_a_preprocessing_that_shifts_by_a_pixel_is_caught():
    def shifted(reader, name):
        return np.roll(prep.to_array(reader, name), 1, axis=1)

    check = prep.frame_check("grey_clahe", False, process=shifted)
    assert not check["ok"] and check["dx"] == pytest.approx(1.0, abs=0.05)

    def cropped(reader, name):
        return prep.to_array(reader, name)[:-2]

    assert not prep.frame_check("grey_smooth", True, process=cropped)["ok"]


def test_the_result_does_not_depend_on_the_block_size():
    fluo = smooth_noise((700, 900), np.uint16, 30000)
    rgb = np.stack([smooth_noise((600, 800), np.uint8, 255) for _ in range(3)], -1)
    for image in (fluo, rgb):
        for name in prep.CANDIDATES:
            whole = prep.to_array(prep.array_reader(image), name)
            blocks = prep.to_array(prep.array_reader(image), name, block=256)
            assert whole.shape == image.shape[:2]
            assert np.array_equal(whole, blocks), name


def test_identity_candidates_return_the_channel_and_grey_is_the_stated_formula():
    fluo = smooth_noise((300, 400), np.uint16, 30000)
    for name in ("grey", "haem"):
        got = prep.to_array(prep.array_reader(fluo), name)
        assert got.dtype == fluo.dtype and np.array_equal(got, fluo)
    assert not np.array_equal(
        prep.to_array(prep.array_reader(fluo), "grey_clahe"), fluo
    )
    smooth = prep.to_array(prep.array_reader(fluo), "grey_smooth")
    assert smooth.dtype == np.uint16 and smooth.std() < fluo.std()
    rgb = RNG.integers(0, 256, (50, 60, 3), dtype=np.uint8)
    want = 255.0 - (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    got = prep.to_array(prep.array_reader(rgb), "grey")
    assert got.dtype == np.uint8 and np.abs(got - want).max() <= 0.5 + 1e-3
    white = np.full((40, 40, 3), 255, np.uint8)
    assert prep.to_array(prep.array_reader(white), "grey").max() == 0  # paper is dark
    assert prep.to_array(prep.array_reader(white), "haem").max() == 0
    # CLAHE is defined on 8 bits: its output is, whatever comes in
    assert prep.to_array(prep.array_reader(fluo), "haem_clahe").dtype == np.uint8


# ── the adapter ───────────────────────────────────────────────────────────────
def test_a_label_and_its_candidate_come_together():
    frozen = prep.selected()
    other = next(c for c in prep.CANDIDATES if c != frozen)
    assert adapter.prep_of({}) is None
    assert adapter.prep_of({"label": prep.ROW, "prep": frozen}) == frozen
    assert adapter.prep_of({"label": f"{prep.ROW}_{other}", "prep": other}) == other
    with pytest.raises(ValueError, match="runs the frozen candidate"):
        adapter.prep_of({"label": prep.ROW, "prep": other})
    with pytest.raises(ValueError, match="another row's name"):
        adapter.prep_of({"prep": frozen})  # it would be stored as `stare`
    with pytest.raises(ValueError, match="another row's name"):
        adapter.prep_of({"label": f"{prep.ROW}_{frozen}", "prep": other})
    with pytest.raises(ValueError, match="needs --opt prep"):
        adapter.prep_of({"label": prep.ROW})
    with pytest.raises(ValueError, match="expected one of"):
        adapter.prep_of({"label": prep.ROW, "prep": "otsu"})


def test_the_tuning_runs_are_one_per_candidate_and_resolve_by_name():
    names = [variants.run_name(r) for r in variants.TUNING_RUNS]
    assert names == [f"{prep.ROW}_{c}" for c in prep.CANDIDATES]
    for run, c in zip(variants.TUNING_RUNS, prep.CANDIDATES):
        assert run.opts == {"prep": c} and run.method == "stare"
        assert variants.run_of(variants.run_name(run)) is run
        assert variants.variants_of(run)[-1] == variants.tuning_final(c)
        assert variants.gated_by(variants.run_name(run)) == variants.run_name(run)
    assert not set(names) & {variants.run_name(r) for r in variants.RUNS}
    # in STARE's family, never primary, and gated by its own selfcheck
    assert variants.family(prep.ROW) == "stare" and prep.ROW not in variants.PRIMARY
    assert variants.primary_of("stare") == "stare"
    assert variants.gated_by(prep.ROW) == prep.ROW


OPTS = {"workers": 2}


@pytest.fixture(scope="module")
def ran(synthetic_cases, tmp_path_factory):
    """`crop` through STARE three times: plain, the identity candidate, and CLAHE."""
    pytest.importorskip("stare")
    case = synthetic_cases["crop"]
    out = tmp_path_factory.mktemp("prep")
    assert methods.run_case("stare", case, out, opts=OPTS) == ["stare_rigid", "stare"]
    for c in ("grey", "grey_clahe"):
        label = f"{prep.ROW}_{c}"
        got = methods.run_case(
            "stare", case, out, opts={**OPTS, "prep": c}, label=label
        )
        assert got == [f"{label}_rigid", label]
    return case, out


def load(out, variant, case):
    d = result_dir(out, variant, case)
    with np.load(d / "warped.npz") as z:
        warped = {k: z[k] for k in z.files}
    return json.loads((d / "run.json").read_text()), warped


def test_the_identity_candidate_gives_the_plain_rows_transform(ran):
    case, out = ran
    _, plain = load(out, "stare", case)
    run, same = load(out, "stare_prep_grey", case)
    assert run["info"]["prep"]["identity_here"] is True
    assert np.array_equal(plain["truth_xy"], same["truth_xy"])
    assert np.array_equal(plain["grid"], same["grid"])
    run, other = load(out, "stare_prep_grey_clahe", case)
    assert run["info"]["prep"]["identity_here"] is False
    assert not np.array_equal(plain["truth_xy"], other["truth_xy"])


def test_stare_registers_the_preprocessed_copy_and_writes_the_original(ran):
    import tifffile

    case, out = ran
    plain, _ = load(out, "stare", case)
    run, _ = load(out, "stare_prep_grey_clahe", case)
    info = run["info"]
    assert info["input"]["moving"].endswith("moving.grey_clahe.ome.tif")
    assert info["input"]["reference"].endswith("reference.grey_clahe.ome.tif")
    assert info["nuclear_index"] == 0 and info["prep"]["candidate"] == "grey_clahe"
    assert info["prep"]["hw"] == {"ref": case.ref_hw, "mov": case.mov_hw}
    assert abs(info["prep"]["frame_check_px"]["dx"]) < prep.FRAME_GATE_PX
    assert info["prep"]["prep_s"] > 0
    assert any(
        "preprocessing `grey_clahe`" in s
        for s in info["implementation"]["benchmark_side"]
    )
    # STARE itself: the same version and the same parameters as the plain row
    for key in ("version", "stride", "max_disp", "model"):
        assert info[key] == plain["info"][key]
    assert (
        info["implementation"]["parameters"]
        == plain["info"]["implementation"]["parameters"]
    )
    # the candidate is in the recorded options: what a reuse compares
    assert run["opts"]["prep"] == "grey_clahe" and "prep" not in plain["opts"]
    # the image on disk is the ORIGINAL slide moved: 16 bit, where CLAHE's output is 8
    (written,) = run["native"]["files"]
    image = tifffile.imread(written)
    original = tifffile.imread(case.mov_image)
    assert image.dtype == original.dtype == np.uint16
    assert list(image.shape[-2:]) == case.ref_hw
    assert image.max() > 255  # an 8-bit CLAHE image could not hold this


def test_a_result_cannot_be_stored_under_another_candidates_name(ran):
    case, out = ran
    again = methods.run_case(
        "stare",
        case,
        out,
        opts={**OPTS, "prep": "grey_smooth"},
        label="stare_prep_haem",
    )
    assert again == []  # refused: nothing registered, nothing written as `haem`
    d = result_dir(out, "stare_prep_haem", case)
    run = json.loads((d / "run.json").read_text())
    assert run["ok"] is False and "another row's name" in run["error"]
    assert not (d / "warped.npz").exists()
