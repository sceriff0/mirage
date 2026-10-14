"""The list of runs: what the cohort launcher submits and what the scorer expects.

`EXPECTED` below is written out by hand on purpose. `variants.expected_variants()` is built
through `methods.variant_names`, so a change in an adapter's `VARIANTS` / `VARIANT_LABELS`
or in the run list moves the computed names; the literal list is what notices.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import METHODS, cli, methods, prep, variants
from benchmarks.regbench.methods import ashlar, palom, valis
from benchmarks.regbench.methods import deeperhistreg as dhr

EXPECTED = [
    "stare_rigid",
    "stare",
    "stare_prep_rigid",
    "stare_prep",
    "valis_mirage_rigid",
    "valis_mirage_nonrigid",
    "valis_mirage",
    "valis120_rigid",
    "valis120",
    "valis120_micro",
    "valis120_hires_rigid",
    "valis120_hires_nonrigid",
    "valis120_hires",
    "deeperhistreg_gpu",
    "deeperhistreg_gpu_asdrawn",
    "deeperhistreg_cpu",
    "deeperhistreg_cpu_asdrawn",
    "deeperhistreg_hr_gpu",
    "deeperhistreg_hr_gpu_asdrawn",
    "deeperhistreg_hr_cpu",
    "deeperhistreg_hr_cpu_asdrawn",
    "deeperhistreg_fluo_gpu",
    "deeperhistreg_fluo_gpu_asdrawn",
    "deeperhistreg_fluo_cpu",
    "deeperhistreg_fluo_cpu_asdrawn",
    "deeperhistreg_hr_fluo_gpu",
    "deeperhistreg_hr_fluo_gpu_asdrawn",
    "deeperhistreg_hr_fluo_cpu",
    "deeperhistreg_hr_fluo_cpu_asdrawn",
    "ashlar",
    "ashlar_clean",
    "affine_ashlar",
    "affine_ashlar_clean",
    "palom_affine",
    "palom",
]
PER_PATIENT = {
    "valis_mirage",
    "valis120",
    "valis120_hires",
    "ashlar",
    "ashlar_clean",
    "affine_ashlar",
    "affine_ashlar_clean",
}
GPU = {
    "deeperhistreg_gpu",
    "deeperhistreg_hr_gpu",
    "deeperhistreg_fluo_gpu",
    "deeperhistreg_hr_fluo_gpu",
}


def by_name():
    return {variants.run_name(r): r for r in variants.RUNS}


def test_runs_produce_exactly_the_designs_variants():
    assert len(EXPECTED) == 35 and len(set(EXPECTED)) == 35
    assert variants.expected_variants() == EXPECTED


def test_expected_variants_go_through_the_methods_own_naming(monkeypatch):
    """A renamed variant in an adapter must reach the expected list (and so fail the
    literal list above), not be papered over by names restated in variants.py."""
    monkeypatch.setattr(palom, "VARIANTS", ("palom_renamed", "palom"))
    got = variants.expected_variants()
    assert "palom_renamed" in got and "palom_affine" not in got
    # and a renamed PRIMARY row is refused outright: no run writes it any more
    monkeypatch.setattr(palom, "VARIANTS", ("palom_affine", "palom_renamed"))
    with pytest.raises(ValueError, match="written by no run"):
        variants.expected_variants()


def test_counts():
    assert variants.counts() == (18, 4, 7)
    assert {variants.run_name(r) for r in variants.RUNS if r.gpu} == GPU
    per_patient = {
        variants.run_name(r)
        for r in variants.RUNS
        if variants.scope(r.method) == "group"
    }
    assert per_patient == PER_PATIENT


def test_run_names_are_unique_and_no_stride_run_is_listed():
    names = [variants.run_name(r) for r in variants.RUNS]
    assert len(names) == len(set(names)) == 18
    assert not [n for n in names if n.startswith("stare_s")]
    # STARE itself has ONE configuration; the second row only changes its input
    assert [r.opts for r in variants.RUNS if r.method == "stare"] == [
        {},
        {"prep": prep.selected()},
    ]


def test_every_run_names_a_known_method_and_a_run_it_waits_for():
    names = [variants.run_name(r) for r in variants.RUNS]
    for i, run in enumerate(variants.RUNS):
        assert run.method in METHODS and run.method != "initial"
        assert run.image == run.image.upper() and run.image.isidentifier()
        assert isinstance(run.gpu, bool)
        if run.after is not None:
            # a run waits for one listed BEFORE it, so the launcher can submit in order
            assert run.after in names[:i]
    waiting = {variants.run_name(r): r.after for r in variants.RUNS if r.after}
    assert waiting == {"affine_ashlar": "stare", "affine_ashlar_clean": "stare"}


def test_scopes():
    assert variants.scope("stare") == "pair"
    assert variants.scope("deeperhistreg") == "pair"
    assert variants.scope("palom") == "pair"
    for method in ("valis", "valis_mirage", "ashlar"):
        assert variants.scope(method) == "group"


def test_options_are_strings_the_command_line_can_carry():
    for run in variants.RUNS:
        for key, value in run.opts.items():
            assert isinstance(key, str) and isinstance(value, str)
            assert not set(",\t\n =") & set(key + value), (run, key, value)
        again = methods.parse_opts(variants.opt_args(run))
        assert again == run.opts


def test_every_label_agrees_with_its_options_as_the_adapter_checks_them():
    """Each adapter refuses a label whose options name another run. The run list must
    pass every one of those checks, and name the runs the adapters know."""
    runs = by_name()
    for name, run in runs.items():
        opts = {**run.opts, "label": name}
        if run.method == "deeperhistreg":
            config, device, fluo = dhr.settings(opts)
            assert (config == dhr.HR_CONFIG) == ("_hr_" in name)
            assert (device != "cpu") == run.gpu == name.endswith("_gpu")
            assert fluo == ("_fluo_" in name)
        elif run.method == "ashlar":
            assert ashlar.settings(opts) == ashlar.LABELS[name]
        elif run.method == "palom":
            assert run.opts == {} and palom.settings(opts) == "auto"
        elif run.method == "valis":
            hires = str(run.opts.get("hires", "0")) == "1"
            assert hires == (name == valis.HIRES_LABEL)
    assert {n for n, r in runs.items() if r.method == "ashlar"} == set(ashlar.LABELS)
    assert [n for n, r in runs.items() if r.method == "palom"] == ["palom"]
    assert variants.HR_CONFIG == dhr.HR_CONFIG


def test_a_wrong_option_is_caught_by_the_adapter_check():
    """The check above can fail: the 4096 px label with the 8192 px set is refused."""
    with pytest.raises(ValueError):
        dhr.settings({"label": "deeperhistreg_gpu", "config": variants.HR_CONFIG})
    with pytest.raises(ValueError):
        palom.settings({"label": "palom", "preread": "maybe"})


def test_primary_rows_are_expected_variants():
    assert len(variants.PRIMARY) == len(set(variants.PRIMARY)) == 6
    assert set(variants.PRIMARY) <= set(variants.expected_variants())
    # one primary row per method shown, and none for the anchored ASHLAR family
    families = [variants.family(v) for v in variants.PRIMARY]
    assert sorted(families) == sorted(set(families))
    assert set(families) == set(variants.FAMILY.values()) - {"affine_ashlar"}
    assert variants.primary_of("affine_ashlar") is None
    for v in variants.PRIMARY:
        assert variants.primary_of(variants.family(v)) == v


def test_the_primary_list_is_the_plans():
    """THE one place the two switchable primaries are written out in the tests (ruling
    R3.12: the high-resolution row is primary for VALIS 1.2.0 and for DeeperHistReg).
    Switching one is its `variants.PRIMARY_...` line and the matching line here."""
    assert variants.PRIMARY_VALIS120 == "valis120_hires"
    assert variants.PRIMARY_DEEPERHISTREG == "deeperhistreg_hr_gpu_asdrawn"
    assert variants.PRIMARY == (
        "stare",
        "valis_mirage",
        "valis120_hires",
        "deeperhistreg_hr_gpu_asdrawn",
        "ashlar",
        "palom",
    )
    assert variants.PRIMARY[2] is variants.PRIMARY_VALIS120
    assert variants.PRIMARY[3] is variants.PRIMARY_DEEPERHISTREG


def test_the_high_resolution_rows_are_primary_and_the_defaults_stay_in_their_family():
    """A family's name need not be its primary's: `valis120` and `valis120_hires` share
    the family `valis120`, whose primary is now `valis120_hires`."""
    assert variants.family("valis120") == "valis120"
    assert variants.family("valis120_hires") == "valis120"
    assert variants.primary_of("valis120") == "valis120_hires"
    assert variants.members("valis120") == ["valis120", "valis120_hires"]
    assert variants.primary_of("deeperhistreg") == "deeperhistreg_hr_gpu_asdrawn"
    assert len(variants.members("deeperhistreg")) == 8
    # the upstream-default rows are still run and expected, just not primary
    expected = variants.expected_variants()
    for name in ("valis120", "valis120_micro", "deeperhistreg_gpu_asdrawn"):
        assert name in expected and name not in variants.PRIMARY
    # the run that writes each primary row is the high-resolution one
    runs = by_name()
    assert runs["valis120_hires"].opts == {"hires": "1"}
    assert runs["deeperhistreg_hr_gpu"].opts["config"] == variants.HR_CONFIG


def test_the_deeperhistreg_primary_scores_the_written_image_on_the_gpu():
    name = variants.PRIMARY_DEEPERHISTREG
    assert name.endswith("_asdrawn")  # the transform of the image written to disk
    run = by_name()[name[: -len("_asdrawn")]]
    assert run.method == "deeperhistreg" and run.gpu
    assert "fluo" not in run.opts  # a shipped parameter set, not the benchmark's edit


def test_every_variant_has_a_family_and_the_longest_prefix_wins():
    for v in variants.expected_variants():
        assert variants.family(v) in set(variants.FAMILY.values()), v
    assert variants.family("valis120_hires_nonrigid") == "valis120"
    assert variants.family("valis_mirage_rigid") == "valis_mirage"
    assert variants.family("deeperhistreg_hr_fluo_cpu_asdrawn") == "deeperhistreg"
    assert variants.family("stare_rigid") == "stare"
    assert variants.family("palom_affine") == "palom"
    assert variants.family("ashlar_clean") == "ashlar"
    # a prefix only counts at a name boundary, and an off-seam row is its variant's
    assert variants.family("stare:non_palom") == "stare"
    assert variants.family("palomino") is None
    assert variants.family("initial") is None


def test_both_anchored_rows_share_a_family_that_is_not_ashlars():
    assert variants.family("affine_ashlar") == "affine_ashlar"
    assert variants.family("affine_ashlar_clean") == "affine_ashlar"
    assert variants.family("ashlar") == "ashlar" != variants.family("affine_ashlar")
    assert "affine_ashlar" not in variants.members("ashlar")
    assert variants.members("ashlar") == ["ashlar", "ashlar_clean"]
    assert variants.members("affine_ashlar") == ["affine_ashlar", "affine_ashlar_clean"]


def test_clean_rows_are_gated_by_their_noisy_rows():
    assert variants.GATED_BY == {
        "ashlar_clean": "ashlar",
        "affine_ashlar_clean": "affine_ashlar",
    }
    names = set(by_name())
    assert set(variants.GATED_BY) <= names and set(variants.GATED_BY.values()) <= names
    assert variants.gated_by("ashlar_clean") == "ashlar"
    assert variants.gated_by("affine_ashlar_clean") == "affine_ashlar"
    # every other run stands behind its own selfcheck
    for name in names - set(variants.GATED_BY):
        assert variants.gated_by(name) == name


def test_selfcheck_masks_come_from_the_adapters():
    assert variants.selfcheck_mask("ashlar") == "overlap" == ashlar.MASKS[0]
    assert variants.selfcheck_mask("palom") == "seams" == palom.MASKS[0]
    for method in ("stare", "valis", "valis_mirage", "deeperhistreg"):
        assert variants.selfcheck_mask(method) is None


def test_images_are_the_six_the_launcher_knows():
    images = {r.image for r in variants.RUNS}
    assert images == {
        "STARE_REGBENCH",
        "VALIS_MIRAGE",
        "VALIS120",
        "DEEPERHISTREG",
        "ASHLAR",
        "PALOM",
    }
    # one image per method, whatever the label
    per_method = {}
    for run in variants.RUNS:
        per_method.setdefault(run.method, set()).add(run.image)
    assert all(len(v) == 1 for v in per_method.values())
    assert variants.BUILT_HERE == {
        "STARE_REGBENCH": "stare-regbench",
        "VALIS120": "valis120",
        "DEEPERHISTREG": "deeperhistreg",
        "PALOM": "palom",
    }
    assert set(variants.BUILT_HERE) | set(variants.PULLED) == images
    # the suffix of REGBENCH_SIF_<image> is the build context's name, in capitals with
    # `-` written `_`: the launcher needs no table to go from one to the other
    for image, directory in variants.BUILT_HERE.items():
        assert image == directory.upper().replace("-", "_")
        assert f"REGBENCH_SIF_{image}" in variants.__doc__
    for image in variants.PULLED:
        assert f"REGBENCH_SIF_{image}" in variants.__doc__


def test_stare_runs_in_the_benchmarks_own_image_and_so_does_the_scorer():
    """R3.9a: the pipeline's STARE image has no psutil, so STARE's memory would be read
    by another instrument than every other method's; and the tissue-level block needs the
    pinned OpenCV, in ONE image for a whole result."""
    from pathlib import Path

    both = [r for r in variants.RUNS if r.method == "stare"]
    assert [variants.run_name(r) for r in both] == ["stare", prep.ROW]
    for stare in both:
        assert stare.image == "STARE_REGBENCH" == variants.SCORER_IMAGE
    assert "STARE" not in variants.BUILT_HERE  # the pipeline's image runs no row here
    assert variants.SCORER_IMAGE in variants.BUILT_HERE
    root = Path(__file__).resolve().parents[3]
    pins = (root / "requirements" / "stare-regbench.txt").read_text()
    lines = [ln.split("#")[0].strip() for ln in pins.splitlines()]
    assert "-r stare.txt" in lines and "-c constraints.txt" in lines
    # STARE's own pin is NOT restated: it comes through `-r stare.txt` alone
    assert not [ln for ln in lines if ln.startswith("stare-registration")]
    constraints = (root / "requirements" / "constraints.txt").read_text()
    (opencv,) = [ln for ln in lines if ln.startswith("opencv-python-headless==")]
    assert opencv in [ln.split("#")[0].strip() for ln in constraints.splitlines()]
    assert [ln for ln in lines if ln.startswith("psutil==")]
    assert [ln for ln in lines if ln.startswith("pytest==")]
    smoke = (root / "containers" / "stare-regbench" / "smoke.sh").read_text()
    assert "psutil" in smoke and "__import__('cv2')" in smoke
    # STARE 1.3.0: the image's stage commands take --prep (the `sprep` run option)
    assert "coarse --help | grep -q -- --prep" in smoke


def test_images_built_here_have_a_build_context():
    """An image the run list names and nothing builds would be a "manifest unknown" on
    the cluster. Every directory named must be in containers/images.json."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "containers"
    listed = {e["dir"] for e in json.loads((root / "images.json").read_text())}
    for image, directory in variants.BUILT_HERE.items():
        assert directory in listed, (image, directory)
        assert (root / directory / "Dockerfile").is_file()
        assert (root / directory / "smoke.sh").is_file()


# ── the `runs` subcommand: what the launcher reads ────────────────────────────
COLUMNS = (
    "name",
    "method",
    "label",
    "opts",
    "image",
    "gpu",
    "scope",
    "after",
    "gated_by",
    "selfcheck_mask",
    "primary",
    "variants",
    "active",
    "gate_variants",
    "realcheck",
    "stages",
)
# The twelve columns of the first version: a launcher written against them reads the
# same positions for ever; new columns only go on the end.
FIRST_TWELVE = COLUMNS[:12]


def rows(capsys, *args):
    assert cli.main(["runs", *args]) == 0
    return [line.split("\t") for line in capsys.readouterr().out.splitlines()]


def test_runs_prints_one_tab_separated_line_per_run_in_the_documented_order(capsys):
    assert variants.RUNS_COLUMNS == COLUMNS
    for column in COLUMNS:  # the order is documented where the launcher's author looks
        assert column in variants.__doc__
    doc = variants.__doc__
    assert [doc.index(f"``{c}``") for c in COLUMNS] == sorted(
        doc.index(f"``{c}``") for c in COLUMNS
    )
    assert COLUMNS[:12] == FIRST_TWELVE == variants.RUNS_COLUMNS[:12]
    assert FIRST_TWELVE[0] == "name" and FIRST_TWELVE[11] == "variants"
    got = rows(capsys)
    assert len(got) == 18
    assert all(len(r) == len(COLUMNS) == 16 for r in got)
    # the two runs whose registration is several scheduler jobs per round
    staged = {r[0]: r[15] for r in got if r[15] != "-"}
    assert staged == {
        "stare": "coarse,tile,solve,stitch",
        "stare_prep": "prep,coarse,tile,solve,stitch",
    }
    assert [r[0] for r in got] == [variants.run_name(r) for r in variants.RUNS]
    # no empty field: bash's `read` with a tab IFS merges consecutive tabs
    assert all(field for r in got for field in r)


def test_runs_lines_carry_what_the_launcher_passes_back(capsys):
    got = {r[0]: dict(zip(COLUMNS, r)) for r in rows(capsys)}
    assert got["stare"] == {
        "name": "stare",
        "method": "stare",
        "label": "-",
        "opts": "-",
        "image": "STARE_REGBENCH",
        "gpu": "0",
        "scope": "pair",
        "after": "-",
        "gated_by": "stare",
        "selfcheck_mask": "-",
        "primary": "stare",
        "variants": "stare_rigid,stare",
        "active": "1",
        "gate_variants": "stare_rigid,stare",
        "realcheck": "-",
        "stages": "coarse,tile,solve,stitch",
    }
    assert got["affine_ashlar_clean"] == {
        "name": "affine_ashlar_clean",
        "method": "ashlar",
        "label": "affine_ashlar_clean",
        "opts": "affine=stare,noise=none",
        "image": "ASHLAR",
        "gpu": "0",
        "scope": "group",
        "after": "stare",
        "gated_by": "affine_ashlar",
        "selfcheck_mask": "overlap",
        "primary": "-",
        "variants": "affine_ashlar_clean",
        "active": "1",
        "gate_variants": "affine_ashlar_clean",
        "realcheck": "-",
        "stages": "-",
    }
    hr = got["deeperhistreg_hr_fluo_gpu"]
    assert hr["opts"] == f"device=cuda:0,config={variants.HR_CONFIG},fluo=1"
    assert hr["gpu"] == "1" and hr["scope"] == "pair" and hr["primary"] == "-"
    assert got["palom"]["opts"] == "-"
    assert got["palom"]["selfcheck_mask"] == "seams"
    assert got["palom"]["gated_by"] == "palom"
    assert got["valis120"]["primary"] == "-"
    assert got["valis120_hires"]["primary"] == "valis120_hires"
    assert got["valis120_hires"]["opts"] == "hires=1"
    assert got["deeperhistreg_hr_gpu"]["primary"] == "deeperhistreg_hr_gpu_asdrawn"
    assert got["deeperhistreg_gpu"]["primary"] == "-"
    assert (
        got["valis120"]["method"] == "valis" and got["valis120"]["image"] == "VALIS120"
    )
    primaries = {r["primary"] for r in got.values()} - {"-"}
    assert primaries == set(variants.PRIMARY)


def test_a_runs_line_round_trips_into_a_run_command(capsys):
    """`--label` and each `--opt` built from a line are accepted by the `run` parser and
    name the same run."""
    parser = cli.build_parser()
    for line in rows(capsys):
        row = dict(zip(COLUMNS, line))
        argv = ["run", "--cases", "c", "--out", "o", "--dataset", "multiplex"]
        argv += ["--method", row["method"], "--index", "0"]
        if row["label"] != "-":
            argv += ["--label", row["label"]]
        if row["opts"] != "-":
            for item in row["opts"].split(","):
                argv += ["--opt", item]
        a = parser.parse_args(argv)
        run = by_name()[row["name"]]
        assert (a.method, a.label) == (run.method, run.label)
        assert methods.parse_opts(a.opt) == run.opts


def test_runs_header_names_the_columns(capsys):
    got = rows(capsys, "--header")
    assert tuple(got[0]) == COLUMNS and len(got) == 19


# ── `score` on the command line expects the whole list ────────────────────────
def test_score_passes_the_expected_variants_and_the_primary_rows(monkeypatch):
    from benchmarks.regbench import score as score_module

    seen = {}

    def fake(cases, out, datasets=None, workers=1, **kw):
        seen.update(kw, cases=cases, out=out, datasets=datasets)
        return {}

    monkeypatch.setattr(score_module, "score", fake)
    assert cli.main(["score", "--cases", "c", "--out", "o"]) == 0
    assert seen["expected"] == EXPECTED
    assert seen["primary"] == list(variants.PRIMARY)

    seen.clear()
    assert cli.main(["score", "--cases", "c", "--out", "o", "--only-run"]) == 0
    assert seen["expected"] is None and seen["primary"] is None


# ── paused runs (R3.11 a, R3.12: the flag is built, nothing is paused today) ──
def pause(monkeypatch, *names):
    """The run list with some runs paused: what `active=False` on their lines gives."""
    assert set(names) <= set(by_name())
    paused = tuple(
        r._replace(active=False) if variants.run_name(r) in names else r
        for r in variants.RUNS
    )
    monkeypatch.setattr(variants, "RUNS", paused)


def test_every_run_is_active_today_and_the_flag_is_the_last_field():
    assert variants.Run._fields[-1] == "active"
    assert variants.Run._field_defaults == {"active": True}
    assert all(r.active is True for r in variants.RUNS)
    assert variants.active_runs() == variants.RUNS
    assert variants.paused_names() == []
    variants.check()


def test_a_paused_run_is_not_expected_not_counted_and_not_printed(monkeypatch, capsys):
    pause(monkeypatch, "valis120", "deeperhistreg_fluo_gpu")
    variants.check()
    assert variants.paused_names() == ["valis120", "deeperhistreg_fluo_gpu"]
    gone = {
        "valis120_rigid",
        "valis120",
        "valis120_micro",
        "deeperhistreg_fluo_gpu",
        "deeperhistreg_fluo_gpu_asdrawn",
    }
    assert variants.expected_variants() == [v for v in EXPECTED if v not in gone]
    assert variants.counts() == (16, 3, 6)
    # the family keeps its primary, and its members are the runs that run
    assert variants.members("valis120") == ["valis120_hires"]
    assert variants.members("valis120", paused=True) == ["valis120", "valis120_hires"]
    assert variants.primary_of("valis120") == "valis120_hires"
    # still implemented: its names resolve, and its family too
    run = by_name()["valis120"]
    assert variants.variants_of(run) == ["valis120_rigid", "valis120", "valis120_micro"]
    assert variants.family("valis120_micro") == "valis120"

    got = rows(capsys)
    assert len(got) == 16
    assert "valis120" not in [r[0] for r in got]
    assert {r[12] for r in got} == {"1"}
    everything = rows(capsys, "--all")
    assert [r[0] for r in everything] == [variants.run_name(r) for r in variants.RUNS]
    flags = {r[0]: r[12] for r in everything}
    assert flags["valis120"] == "0" and flags["deeperhistreg_fluo_gpu"] == "0"
    assert sum(v == "1" for v in flags.values()) == 16
    # the same line, flag apart, as when it ran
    (line,) = [r for r in everything if r[0] == "valis120"]
    assert line[:12] == [
        "valis120",
        "valis",
        "valis120",
        "-",
        "VALIS120",
        "0",
        "group",
        "-",
        "valis120",
        "-",
        "-",
        "valis120_rigid,valis120,valis120_micro",
    ]
    assert rows(capsys, "--all", "--header")[0] == list(COLUMNS)


def test_score_does_not_expect_a_paused_run(monkeypatch):
    from benchmarks.regbench import score as score_module

    pause(monkeypatch, "deeperhistreg_cpu")
    seen = {}
    monkeypatch.setattr(
        score_module, "score", lambda cases, out, *a, **kw: seen.update(kw) or {}
    )
    assert cli.main(["score", "--cases", "c", "--out", "o"]) == 0
    assert "deeperhistreg_cpu" not in seen["expected"] and "palom" in seen["expected"]
    assert len(seen["expected"]) == len(EXPECTED) - 2


def test_reactivating_is_one_field(monkeypatch):
    pause(monkeypatch, "valis120")
    assert len(variants.expected_variants()) == len(EXPECTED) - 3
    monkeypatch.setattr(
        variants, "RUNS", tuple(r._replace(active=True) for r in variants.RUNS)
    )
    assert variants.expected_variants() == EXPECTED
    assert variants.counts() == (18, 4, 7)


@pytest.mark.parametrize(
    "name, why",
    [
        ("valis120_hires", "primary"),  # writes the primary row valis120_hires
        ("deeperhistreg_hr_gpu", "primary"),  # writes deeperhistreg_hr_gpu_asdrawn
        ("ashlar", "primary"),  # (and gates ashlar_clean)
        ("affine_ashlar", "selfcheck gates"),  # gates the active affine_ashlar_clean
        ("stare", "primary"),  # (and the anchored ASHLAR rows wait for it)
    ],
)
def test_a_paused_run_cannot_write_a_primary_row_or_gate_an_active_run(
    monkeypatch, name, why
):
    pause(monkeypatch, name)
    with pytest.raises(ValueError, match=why):
        variants.check()
    # and nothing downstream hands out a list built on the broken state
    with pytest.raises(ValueError):
        variants.expected_variants()
    with pytest.raises(ValueError):
        variants.runs_table()


def test_a_paused_run_cannot_be_what_an_active_run_waits_for(monkeypatch):
    """`stare` is primary, so the `after` rule is shown on a list where it is not."""
    pause(monkeypatch, "stare")
    monkeypatch.setattr(
        variants, "PRIMARY", tuple(v for v in variants.PRIMARY if v != "stare")
    )
    with pytest.raises(ValueError, match="waits for"):
        variants.check()
    # pausing the runs that wait as well is a consistent list
    pause(monkeypatch, "affine_ashlar", "affine_ashlar_clean")
    variants.check()
    assert variants.counts() == (15, 4, 5)


def test_a_gated_run_may_be_paused_while_its_gate_runs(monkeypatch):
    pause(monkeypatch, "ashlar_clean", "affine_ashlar_clean")
    variants.check()
    assert variants.members("ashlar") == ["ashlar"]


# ── which rows a run's selfcheck gate reads (R3.12 b) ─────────────────────────
def test_gate_variants_are_all_of_a_runs_rows_except_for_deeperhistreg():
    for name, run in by_name().items():
        names = tuple(variants.variants_of(run))
        if run.method == "deeperhistreg":
            assert variants.gate_variants(name) == (f"{name}_asdrawn",)
            assert name not in variants.gate_variants(name)  # the estimated field
        else:
            assert variants.gate_variants(name) == names
        assert set(variants.gate_variants(name)) <= set(names)
    assert variants.gate_variants("valis120") == (
        "valis120_rigid",
        "valis120",
        "valis120_micro",
    )
    with pytest.raises(KeyError):
        variants.gate_variants("no_such_run")


def test_the_gate_column_is_what_gate_variants_says(capsys):
    got = {r[0]: dict(zip(COLUMNS, r)) for r in rows(capsys)}
    for name, row in got.items():
        assert row["gate_variants"] == ",".join(variants.gate_variants(name))
    assert (
        got["deeperhistreg_hr_gpu"]["gate_variants"] == "deeperhistreg_hr_gpu_asdrawn"
    )
    assert got["deeperhistreg_hr_gpu"]["variants"] == (
        "deeperhistreg_hr_gpu,deeperhistreg_hr_gpu_asdrawn"
    )
    assert got["palom"]["gate_variants"] == "palom_affine,palom"


def record(**statuses):
    """The part of `selfcheck.json` the gate reads: `variants` = name -> status."""
    return {"variants": dict(statuses)}


def test_the_gate_reads_the_record_selfcheck_really_writes(tmp_path):
    """`selfcheck.run` writes `variants: {name: status}` at the top of selfcheck.json
    (and a dict per variant inside each pair): the gate must read the top one."""
    from benchmarks.regbench import selfcheck

    src = selfcheck.__file__
    text = open(src).read()
    assert '"variants": {' in text and "k: (NO_STAGE_IMAGE if" in text
    name = "deeperhistreg_hr_gpu"
    top = {"variants": {name: "inconsistent", f"{name}_asdrawn": "ok"}}
    assert variants.gate_status(name, top) == "ok"
    # a per-variant dict (the shape inside `pairs`) is read the same way
    nested = {"variants": {k: {"status": v} for k, v in top["variants"].items()}}
    assert variants.gate_status(name, nested) == "ok"
    assert variants.dropped_variants(name, nested) == [name]
    # a stage with no image of its own is reported, never counted, never dropped
    rec = record(stare_rigid="no_stage_image", stare="ok")
    assert variants.gate_status("stare", rec) == "ok"
    assert variants.dropped_variants("stare", rec) == []


def test_gate_status_reads_the_gating_rows_only_and_names_the_rows_to_drop():
    name = "deeperhistreg_hr_gpu"
    field, drawn = name, f"{name}_asdrawn"
    # the estimated-field row fails: the method is NOT blocked, that row is dropped
    rec = record(**{field: "inconsistent", drawn: "ok"})
    assert variants.gate_status(name, rec) == "ok"
    assert variants.dropped_variants(name, rec) == [field]
    # the written image's row fails: the method is blocked, nothing is "dropped"
    rec = record(**{field: "ok", drawn: "inconsistent"})
    assert variants.gate_status(name, rec) == "inconsistent"
    assert variants.dropped_variants(name, rec) == []
    # a gating row the record does not have was not checked
    assert variants.gate_status(name, record(**{field: "ok"})) == "not_checked"
    # a non-gating row that could not be compared at all is dropped too, never passed
    rec = record(**{field: "crashed", drawn: "ok"})
    assert variants.dropped_variants(name, rec) == [field]
    rec = record(**{field: "ok", drawn: "ok"})
    assert variants.gate_status(name, rec) == "ok"
    assert variants.dropped_variants(name, rec) == []


def test_for_every_other_method_one_failed_row_blocks_the_run():
    rec = record(palom_affine="inconsistent", palom="ok")
    assert variants.gate_status("palom", rec) == "inconsistent"
    assert variants.dropped_variants("palom", rec) == []
    rec = record(stare_rigid="ok", stare="ok")
    assert variants.gate_status("stare", rec) == "ok"


# ── `gate`: what the launcher asks before it submits a run ────────────────────
def gate(capsys, tmp_path, name, rec):
    import json

    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(rec))
    code = cli.main(["gate", "--name", name, "--selfcheck", str(path)])
    # the first line is the gate's verdict; the task's REGBENCH_END line follows it
    first, end = capsys.readouterr().out.rstrip("\n").split("\n")
    assert end.startswith("REGBENCH_END kind=gate ")
    return code, first.split("\t")


def test_gate_prints_the_status_and_the_rows_to_drop(capsys, tmp_path):
    name = "deeperhistreg_hr_gpu"
    code, out = gate(
        capsys, tmp_path, name, record(**{name: "ok", f"{name}_asdrawn": "ok"})
    )
    assert (code, out) == (0, ["ok", "-"])
    rec = record(**{name: "inconsistent", f"{name}_asdrawn": "ok"})
    code, out = gate(capsys, tmp_path, name, rec)
    assert (code, out) == (0, ["ok", name])  # the method runs; that row is dropped
    rec = record(**{name: "ok", f"{name}_asdrawn": "inconsistent"})
    code, out = gate(capsys, tmp_path, name, rec)
    assert (code, out) == (
        1,
        ["inconsistent", "-"],
    )  # blocked: selfcheck's own exit code
    code, out = gate(
        capsys, tmp_path, "palom", record(palom_affine="crashed", palom="ok")
    )
    assert (code, out) == (3, ["crashed", "-"])
    code, out = gate(capsys, tmp_path, "palom", {"variants": {}})
    assert (code, out) == (2, ["not_checked", "-"])
    with pytest.raises(SystemExit):
        cli.main(
            ["gate", "--name", "nope", "--selfcheck", str(tmp_path / "palom.json")]
        )


def test_the_realcheck_gates_the_primary_rows_of_the_two_valis_adapters(capsys):
    """Column 15: which row's realcheck.json the launcher reads before it starts the
    rest of a run. Only a run with a primary row has one."""
    gated = {
        variants.run_name(r): variants.realcheck_gate(variants.run_name(r))
        for r in variants.RUNS
    }
    assert {k: v for k, v in gated.items() if v} == {
        "valis_mirage": "valis_mirage",
        "valis120_hires": "valis120_hires",
    }
    assert gated["valis120"] is None  # the same adapter, no primary row
    got = {r[0]: r[14] for r in rows(capsys)}
    assert got == {k: v or "-" for k, v in gated.items()}
    assert set(variants.REALCHECK_METHODS) <= set(METHODS)


def test_runs_names_the_image_the_scorer_runs_in(capsys):
    assert cli.main(["runs", "--scorer-image"]) == 0
    assert capsys.readouterr().out.split() == [variants.SCORER_IMAGE]


# ── an option no method reads is refused ──────────────────────────────────────
def test_every_run_passes_only_options_its_adapter_reads():
    import re
    from pathlib import Path

    runs = [*variants.RUNS, *getattr(variants, "TUNING_RUNS", ())]
    assert runs
    for run in runs:
        items = [f"{k}={v}" for k, v in (run.opts or {}).items()]
        assert methods.parse_opts(items, run.method) == dict(run.opts or {})
    # the constant is what the adapter's own code reads (the driver's keys aside)
    folder = Path(methods.__file__).parent
    internal = {"label", "monitor", "native_dir", *methods.COMMON_OPTIONS}
    shared = {"valis": "_valis_points.py", "valis_mirage": "_valis_points.py"}
    for name in ("stare", "valis", "valis_mirage", "deeperhistreg", "ashlar", "palom"):
        code = (folder / f"{name}.py").read_text()
        code += (folder / shared[name]).read_text() if name in shared else ""
        read = set(re.findall(r"opts(?:\.get\(|\[)\s*[\"'](\w+)[\"']", code)) - internal
        assert read <= set(methods.known_options(name)), (name, read)
    assert methods.known_options("initial") == methods.COMMON_OPTIONS
    assert set(methods.load("stare")._FLAGS) <= set(methods.known_options("stare"))


def test_an_option_the_method_does_not_read_is_refused_before_any_work(tmp_path):
    with pytest.raises(SystemExit) as stop:
        methods.parse_opts(["blok=1024", "preread=1"], "palom")
    assert (
        "--opt blok: not an option of palom, which reads: workers, stage_images, "
        in (str(stop.value))
    )
    assert "preread" in str(stop.value) and "Nothing was run." in str(stop.value)
    assert methods.parse_opts(["workers=2"], "initial") == {"workers": "2"}
    assert methods.parse_opts(["anything=1"]) == {"anything": "1"}  # no method named
    with pytest.raises(SystemExit, match="--opt stirde: not an option of stare"):
        cli.main(
            ["run", "--cases", str(tmp_path), "--out", str(tmp_path / "o"), "--dataset",
             "multiplex", "--method", "stare", "--index", "0", "--opt", "stirde=128"]
        )  # fmt: skip
    assert not (tmp_path / "o").exists()
