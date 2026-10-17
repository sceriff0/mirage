"""The list of runs: what the cohort launcher submits and what the scorer expects.

`EXPECTED` below is written out by hand on purpose. `variants.expected_variants()` is built
through `methods.variant_names`, so a change in an adapter's `VARIANTS` / `VARIANT_LABELS`
or in the run list moves the computed names; the literal list is what notices.
"""

from __future__ import annotations

import pytest

from benchmarks.regbench import METHODS, cli, methods, variants
from benchmarks.regbench.methods import ashlar, palom, valis
from benchmarks.regbench.methods import deeperhistreg as dhr

# the two STARE tier runs beside `stare` (the high tier), written out: ruling 2026-10-16
STARE_TIER_RUNS = ["stare_medium", "stare_low"]


def _stare_names():
    """The STARE runs, from the table: every stride x level of the cohort's regime
    (``regimes.COHORT_REGIME``), by name, then the two tier runs."""
    from benchmarks.regbench import regimes as rg_

    return [
        *(
            variants.stare_cell(s_, k, lv)[0]
            for k, lv in rg_.stare_levels(rg_.COHORT_REGIME)
            for s_ in variants.STARE_STRIDES
        ),
        *STARE_TIER_RUNS,
    ]


def _stare_variants(name):
    """The result names of one STARE run: the label replaces the method's name in both."""
    return [f"{name}_rigid", name]


# what the run list holds apart from STARE and ASHLAR: three VALIS runs, eight
# DeeperHistReg runs and PALOM (written out: nothing is generated for them)
OTHER_FIRST = [
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
]
N_OTHER_RUNS = 12  # 3 VALIS + 8 DeeperHistReg + PALOM
EXPECTED = [
    *[v for n in _stare_names() for v in _stare_variants(n)],
    *OTHER_FIRST,
    *ashlar.ASHLAR_CELLS,
    "palom_affine",
    "palom",
]
PER_PATIENT = {
    "valis_mirage",
    "valis120",
    "valis120_hires",
    *ashlar.ASHLAR_CELLS,
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
    assert len(set(EXPECTED)) == len(EXPECTED)
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


# the run list of the cohort, as `python -m benchmarks.regbench runs` prints it (measured
# 2026-10-15 after the stare_prep row and the brightfield regimes were removed: the same
# 45 as before, the paused row never counted; 47 since STARE's medium and low tiers were
# added beside high on 2026-10-17)
N_RUNS = 47


def test_counts():
    n_runs = len(_stare_names()) + N_OTHER_RUNS + len(ashlar.ASHLAR_CELLS)
    assert variants.counts() == (n_runs, len(GPU), len(PER_PATIENT))
    assert n_runs == len(variants.RUNS) == N_RUNS
    assert {variants.run_name(r) for r in variants.RUNS if r.gpu} == GPU
    per_patient = {
        variants.run_name(r)
        for r in variants.RUNS
        if variants.scope(r.method) == "group"
    }
    assert per_patient == PER_PATIENT


def test_run_names_are_unique_and_the_stride_runs_are_exactly_the_generated_set():
    names = [variants.run_name(r) for r in variants.RUNS]
    assert len(names) == len(set(names)) == len(variants.RUNS)
    stare = [variants.run_name(r) for r in variants.RUNS if r.method == "stare"]
    assert sorted(stare) == sorted(_stare_names())
    stride = {n for n in names if n.startswith("stare_s")}
    # stare_s<stride>_<level>: one per non-default stride and level, no more
    levels = {n.split("_", 2)[2] for n in stride}
    assert len(stride) == (len(variants.STARE_STRIDES) - 1) * len(levels)
    assert [r.opts for r in variants.RUNS if variants.run_name(r) == "stare"] == [{}]


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
    assert waiting == {
        n: "stare" for n, c in ashlar.ASHLAR_CELLS.items() if c[0] == "stare"
    }


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
            affine, frac, shift = ashlar.ASHLAR_CELLS[name]
            assert ashlar.settings(opts) == (float(frac), affine, float(shift))
        elif run.method == "palom":
            assert run.opts == {} and palom.settings(opts) == "auto"
        elif run.method == "valis":
            hires = str(run.opts.get("hires", "0")) == "1"
            assert hires == (name == valis.HIRES_LABEL)
    assert {n for n, r in runs.items() if r.method == "ashlar"} == set(
        ashlar.ASHLAR_CELLS
    )
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


def test_the_anchored_rows_share_a_family_that_is_not_ashlars():
    anchored = [n for n, c in ashlar.ASHLAR_CELLS.items() if c[0] == "stare"]
    plain = [n for n, c in ashlar.ASHLAR_CELLS.items() if c[0] == "none"]
    assert "affine_ashlar" in anchored and "affine_ashlar_clean" in anchored
    for n in anchored:
        assert variants.family(n) == "affine_ashlar"
    for n in plain:
        assert variants.family(n) == "ashlar"
    assert variants.family("ashlar") == "ashlar" != variants.family("affine_ashlar")
    assert variants.members("ashlar") == plain
    assert variants.members("affine_ashlar") == anchored


def test_ashlar_rows_are_checked_by_the_row_of_their_affine_level_and_stare_cells_by_stare():
    checked = variants.CHECKED_BY
    for name, (affine, _, _) in ashlar.ASHLAR_CELLS.items():
        checker = "affine_ashlar" if affine == "stare" else "ashlar"
        assert variants.checked_by(name) == checker
        if name != checker:
            assert checked[name] == checker
    for name in _stare_names():
        assert variants.checked_by(name) == "stare"
    names = set(by_name())
    assert set(checked) <= names and set(checked.values()) <= names
    # every other run is checked by its own selfcheck
    for name in names - set(checked):
        assert variants.checked_by(name) == name


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
    assert {variants.run_name(r) for r in both} == set(_stare_names())
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
    "checked_by",
    "selfcheck_mask",
    "primary",
    "variants",
    "active",
    "checked_rows",
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
    assert len(got) == len(variants.runs_for(None))
    assert all(len(r) == len(COLUMNS) == 16 for r in got)
    # the runs whose registration is several scheduler jobs per round: every STARE run
    staged = {r[0]: r[15] for r in got if r[15] != "-"}
    assert set(staged) == set(_stare_names())
    for run in variants.runs_for(None):
        if run.method != "stare":
            continue
        assert staged[variants.run_name(run)] == "coarse,tile,solve,stitch"
    assert [r[0] for r in got] == [
        variants.run_name(r) for r in variants.runs_for(None)
    ]
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
        "checked_by": "stare",
        "selfcheck_mask": "-",
        "primary": "stare",
        "variants": "stare_rigid,stare",
        "active": "1",
        "checked_rows": "stare_rigid,stare",
        "realcheck": "-",
        "stages": "coarse,tile,solve,stitch",
    }
    assert got["affine_ashlar_clean"] == {
        "name": "affine_ashlar_clean",
        "method": "ashlar",
        "label": "affine_ashlar_clean",
        "opts": "affine=stare,noise_frac=0,max_shift_um=15",
        "image": "ASHLAR",
        "gpu": "0",
        "scope": "group",
        "after": "stare",
        "checked_by": "affine_ashlar",
        "selfcheck_mask": "overlap",
        "primary": "-",
        "variants": "affine_ashlar_clean",
        "active": "1",
        "checked_rows": "affine_ashlar_clean",
        "realcheck": "-",
        "stages": "-",
    }
    hr = got["deeperhistreg_hr_fluo_gpu"]
    assert hr["opts"] == f"device=cuda:0,config={variants.HR_CONFIG},fluo=1"
    assert hr["gpu"] == "1" and hr["scope"] == "pair" and hr["primary"] == "-"
    assert got["palom"]["opts"] == "-"
    assert got["palom"]["selfcheck_mask"] == "seams"
    assert got["palom"]["checked_by"] == "palom"
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
    assert tuple(got[0]) == COLUMNS and len(got) == len(variants.runs_for(None)) + 1


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


def test_no_run_is_paused_today():
    assert variants.Run._fields[-1] == "active"
    assert variants.Run._field_defaults == {"active": True}
    assert [r for r in variants.RUNS if not r.active] == []
    assert variants.paused_names() == []
    assert variants.active_runs() == variants.RUNS
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
    n_runs = len(_stare_names()) + N_OTHER_RUNS + len(ashlar.ASHLAR_CELLS)
    assert variants.counts() == (n_runs - 2, len(GPU) - 1, len(PER_PATIENT) - 1)
    # the family keeps its primary, and its members are the runs that run
    assert variants.members("valis120") == ["valis120_hires"]
    assert variants.members("valis120", paused=True) == ["valis120", "valis120_hires"]
    assert variants.primary_of("valis120") == "valis120_hires"
    # still implemented: its names resolve, and its family too
    run = by_name()["valis120"]
    assert variants.variants_of(run) == ["valis120_rigid", "valis120", "valis120_micro"]
    assert variants.family("valis120_micro") == "valis120"

    got = rows(capsys)
    assert len(got) == len(variants.runs_for(None))
    assert "valis120" not in [r[0] for r in got]
    assert {r[12] for r in got} == {"1"}
    everything = rows(capsys, "--all")
    assert [r[0] for r in everything] == [variants.run_name(r) for r in variants.RUNS]
    flags = {r[0]: r[12] for r in everything}
    assert flags["valis120"] == "0" and flags["deeperhistreg_fluo_gpu"] == "0"
    assert sum(v == "1" for v in flags.values()) == n_runs - 2
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
        variants,
        "RUNS",
        tuple(
            r._replace(active=True) if variants.run_name(r) == "valis120" else r
            for r in variants.RUNS
        ),
    )
    assert variants.expected_variants() == EXPECTED
    assert variants.counts()[0] == len(variants.RUNS)


@pytest.mark.parametrize(
    "name, why",
    [
        ("valis120_hires", "primary"),  # writes the primary row valis120_hires
        ("deeperhistreg_hr_gpu", "primary"),  # writes deeperhistreg_hr_gpu_asdrawn
        ("ashlar", "primary"),  # (and checks ashlar_clean)
        ("affine_ashlar", "selfcheck checks"),  # checks the active affine_ashlar_clean
        ("stare", "primary"),  # (and the anchored ASHLAR rows wait for it)
    ],
)
def test_a_paused_run_cannot_write_a_primary_row_or_check_an_active_run(
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
    # every STARE cell is checked by `stare`, so they are paused with it
    stare_cells = [n for n in _stare_names()]
    pause(monkeypatch, *stare_cells)
    monkeypatch.setattr(
        variants, "PRIMARY", tuple(v for v in variants.PRIMARY if v != "stare")
    )
    with pytest.raises(ValueError, match="waits for"):
        variants.check()
    # pausing the runs that wait as well is a consistent list
    anchored = [n for n, c in ashlar.ASHLAR_CELLS.items() if c[0] == "stare"]
    pause(monkeypatch, *stare_cells, *anchored)
    variants.check()
    n_runs = len(_stare_names()) + N_OTHER_RUNS + len(ashlar.ASHLAR_CELLS)
    assert variants.counts() == (
        n_runs - len(stare_cells) - len(anchored),
        len(GPU),
        len(PER_PATIENT) - len(anchored),
    )


def test_a_checked_run_may_be_paused_while_its_checker_runs(monkeypatch):
    pause(monkeypatch, "ashlar_clean", "affine_ashlar_clean")
    variants.check()
    plain = [n for n, c in ashlar.ASHLAR_CELLS.items() if c[0] == "none"]
    assert variants.members("ashlar") == [n for n in plain if n != "ashlar_clean"]


# ── which rows a run's reported selfcheck status is read from (R3.12 b) ──────────
def test_checked_rows_are_all_of_a_runs_rows_except_for_deeperhistreg():
    for name, run in by_name().items():
        names = tuple(variants.variants_of(run))
        if run.method == "deeperhistreg":
            assert variants.checked_rows(name) == (f"{name}_asdrawn",)
            assert name not in variants.checked_rows(name)  # the estimated field
        else:
            assert variants.checked_rows(name) == names
        assert set(variants.checked_rows(name)) <= set(names)
    assert variants.checked_rows("valis120") == (
        "valis120_rigid",
        "valis120",
        "valis120_micro",
    )
    with pytest.raises(KeyError):
        variants.checked_rows("no_such_run")


def test_the_checked_rows_column_is_what_checked_rows_says(capsys):
    got = {r[0]: dict(zip(COLUMNS, r)) for r in rows(capsys)}
    for name, row in got.items():
        assert row["checked_rows"] == ",".join(variants.checked_rows(name))
        assert row["checked_by"] == variants.checked_by(name)
    assert got["deeperhistreg_hr_gpu"]["checked_rows"] == "deeperhistreg_hr_gpu_asdrawn"
    assert got["deeperhistreg_hr_gpu"]["variants"] == (
        "deeperhistreg_hr_gpu,deeperhistreg_hr_gpu_asdrawn"
    )
    assert got["palom"]["checked_rows"] == "palom_affine,palom"


def record(**statuses):
    """The part of `selfcheck.json` the verdict reads: `variants` = name -> status."""
    return {"variants": dict(statuses)}


def status(name, rec):
    return variants.selfcheck_status(name, rec)[0]


def test_the_verdict_reads_the_record_selfcheck_really_writes(tmp_path):
    """`selfcheck.run` writes `variants: {name: status}` at the top of selfcheck.json
    (and a dict per variant inside each pair): the verdict must read the top one."""
    from benchmarks.regbench import selfcheck

    src = selfcheck.__file__
    text = open(src).read()
    assert '"variants": {' in text and "k: (NO_STAGE_IMAGE if" in text
    name = "deeperhistreg_hr_gpu"
    top = {"variants": {name: "inconsistent", f"{name}_asdrawn": "ok"}}
    assert status(name, top) == "ok"
    # a per-variant dict (the shape inside `pairs`) is read the same way
    nested = {"variants": {k: {"status": v} for k, v in top["variants"].items()}}
    assert status(name, nested) == "ok"
    # a stage with no image of its own is reported, never counted
    rec = record(stare_rigid="no_stage_image", stare="ok")
    assert status("stare", rec) == "ok"
    # the rows it was read from and the reference comparison come with it
    assert variants.selfcheck_status("stare", rec) == (
        "ok",
        ("stare_rigid", "stare"),
        "",
    )
    # no record at all is not checked, and nothing is raised
    assert status("stare", {}) == "not_checked"


def test_selfcheck_status_reads_the_checked_rows_only():
    """No row is ever dropped: the estimated-field row of DeeperHistReg stays in every
    table whatever its own selfcheck says (user ruling, option A)."""
    name = "deeperhistreg_hr_gpu"
    field, drawn = name, f"{name}_asdrawn"
    # the estimated-field row fails: the run's status is the written image's row's
    rec = record(**{field: "inconsistent", drawn: "ok"})
    assert status(name, rec) == "ok"
    # the written image's row fails: the run is inconsistent (reported, it still runs)
    rec = record(**{field: "ok", drawn: "inconsistent"})
    assert status(name, rec) == "inconsistent"
    # a checked row the record does not have was not checked
    assert status(name, record(**{field: "ok"})) == "not_checked"
    rec = record(**{field: "crashed", drawn: "ok"})
    assert status(name, rec) == "ok"
    assert not hasattr(variants, "dropped_variants")
    assert not hasattr(variants, "droppable")


def test_for_every_other_method_one_failed_row_makes_the_run_inconsistent():
    rec = record(palom_affine="inconsistent", palom="ok")
    assert status("palom", rec) == "inconsistent"
    rec = record(stare_rigid="ok", stare="ok")
    assert status("stare", rec) == "ok"


# ── `verdict`: what the scorer asks before it scores ──────────────────────────
def verdict(capsys, tmp_path, name, rec):
    import json

    path = tmp_path / "logs" / "selfcheck" / name / "selfcheck.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    code = cli.main(["verdict", "--out", str(tmp_path), "--runs", name])
    # the selfcheck's line, the realcheck's, then the task's REGBENCH_END line
    first, _, end = capsys.readouterr().out.rstrip("\n").split("\n")
    assert end.startswith("REGBENCH_END kind=verdict ")
    return code, first.split("\t")[2]


def test_verdict_prints_the_status(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("REGBENCH_SUBMISSION", raising=False)
    name = "deeperhistreg_hr_gpu"
    got = verdict(
        capsys, tmp_path, name, record(**{name: "ok", f"{name}_asdrawn": "ok"})
    )
    assert got == (0, "ok")
    rec = record(**{name: "inconsistent", f"{name}_asdrawn": "ok"})
    assert verdict(capsys, tmp_path, name, rec) == (0, "ok")
    rec = record(**{name: "ok", f"{name}_asdrawn": "inconsistent"})
    # reported, never a gate: the exit is 0 whatever the verdict
    assert verdict(capsys, tmp_path, name, rec) == (0, "inconsistent")
    rec = record(palom_affine="crashed", palom="ok")
    assert verdict(capsys, tmp_path, "palom", rec) == (0, "crashed")
    assert verdict(capsys, tmp_path, "palom", {"variants": {}}) == (0, "not_checked")


def test_the_realcheck_reports_on_the_primary_rows_of_the_two_valis_adapters(capsys):
    """Column 15: which row's realcheck.json the scorer reports for a run. Only a run
    with a primary row has one."""
    real = {
        variants.run_name(r): variants.realcheck_row(variants.run_name(r))
        for r in variants.runs_for(None)
    }
    assert {k: v for k, v in real.items() if v} == {
        "valis_mirage": "valis_mirage",
        "valis120_hires": "valis120_hires",
    }
    assert real["valis120"] is None  # the same adapter, no primary row
    got = {r[0]: r[14] for r in rows(capsys)}
    assert got == {k: v or "-" for k, v in real.items()}
    assert set(variants.REALCHECK_METHODS) <= set(METHODS)


def test_runs_names_the_image_the_scorer_runs_in(capsys):
    assert cli.main(["runs", "--scorer-image"]) == 0
    assert capsys.readouterr().out.split() == [variants.SCORER_IMAGE]


# ── an option no method reads is refused ──────────────────────────────────────
def test_every_run_passes_only_options_its_adapter_reads():
    import re
    from pathlib import Path

    assert variants.RUNS
    for run in variants.RUNS:
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


# ── generated STARE and ASHLAR runs (Task C3) ─────────────────────────────────
from benchmarks.regbench import regimes as rg  # noqa: E402
from benchmarks.regbench.methods import ashlar as ashlar_adapter  # noqa: E402


def _generated_stare():
    """The STARE runs generated from the stride x level cross: every STARE run but the
    two tier runs (``STARE_TIER_RUNS``), which are listed beside it."""
    got = [r for r in variants.RUNS if r.method == "stare" and "tier" not in r.opts]
    tiers = [variants.run_name(r) for r in variants.RUNS if r.method == "stare"]
    assert [n for n in tiers if n not in {variants.run_name(r) for r in got}] == (
        STARE_TIER_RUNS
    )
    return got


def test_stare_runs_are_strides_times_levels_of_the_cohort_regime():
    names = {variants.run_name(r) for r in _generated_stare()}
    assert len(names) == len(variants.STARE_STRIDES) * len(
        rg.stare_levels(rg.COHORT_REGIME)
    )
    assert "stare" in names  # the default cell is always present (R4)


def test_the_regime_has_the_baseline_at_every_stride():
    """The defaults are in the cross: STARE unchanged at each stride."""
    have = {
        (r.opts or {}).get("stride", "64")
        for r in _generated_stare()
        if "proxy" not in r.opts and "sprep" not in r.opts
    }
    assert have == {str(s) for s in variants.STARE_STRIDES}


def test_a_baseline_run_carries_no_proxy():
    for r in _generated_stare():
        if "proxy" not in r.opts:
            assert set(r.opts) <= {"stride"}
            assert (
                variants.run_name(r).endswith("none") or variants.run_name(r) == "stare"
            )


def test_a_generated_name_maps_to_exactly_one_opts_dict():
    """Two cells producing the same name must produce the same options."""
    seen = {}
    for key, level in rg.stare_levels(rg.COHORT_REGIME):
        for s in variants.STARE_STRIDES:
            name, opts = variants.stare_cell(s, key, level)
            assert seen.setdefault(name, opts) == opts, name
    assert seen.keys() == {variants.run_name(r) for r in _generated_stare()}


def test_level_key_is_injective_over_non_baseline_levels():
    levels = {
        lv
        for reg in rg.REGIMES
        for key, lv in rg.stare_levels(reg)
        if key != "baseline"
    }
    keys = {rg.level_key(lv) for lv in levels}
    assert len(keys) == len(levels)
    assert (
        "none" not in keys
    )  # "none" names the two baselines, which are not in the set


def test_ashlar_is_the_full_factorial():
    cells = {
        (r.opts["affine"], r.opts["noise_frac"], r.opts["max_shift_um"])
        for r in variants.RUNS
        if r.method == "ashlar"
    }
    want = (
        len(variants.ASHLAR_AFFINES)
        * len(variants.ASHLAR_NOISES)
        * len(variants.ASHLAR_SHIFTS)
    )
    assert len(cells) == want == len([r for r in variants.RUNS if r.method == "ashlar"])
    assert ("none", "0", "15") in cells  # ASHLAR as published


def test_existing_ashlar_names_keep_their_cells():
    by = {
        variants.run_name(r): (
            r.opts["affine"],
            r.opts["noise_frac"],
            r.opts["max_shift_um"],
        )
        for r in variants.RUNS
        if r.method == "ashlar"
    }
    assert by["ashlar"] == ("none", "0.002", "15")
    assert by["ashlar_clean"] == ("none", "0", "15")
    assert by["affine_ashlar"] == ("stare", "0.002", "15")
    assert by["affine_ashlar_clean"] == ("stare", "0", "15")
    assert by["ashlar_anone_n01_s60"] == ("none", "0.01", "60")
    assert by["ashlar_astare_n05_s240"] == ("stare", "0.05", "240")


def test_ashlar_cells_tell_the_adapter_what_a_label_means():
    for r in variants.RUNS:
        if r.method != "ashlar":
            continue
        cell = (r.opts["affine"], r.opts["noise_frac"], r.opts["max_shift_um"])
        assert ashlar_adapter.ASHLAR_CELLS[variants.run_name(r)] == cell
        got = ashlar_adapter.settings({**r.opts, "label": variants.run_name(r)})
        assert got == (float(cell[1]), cell[0], float(cell[2]))


def test_ashlar_checkers_and_families_follow_the_affine_factor():
    for r in variants.RUNS:
        if r.method != "ashlar":
            continue
        name = variants.run_name(r)
        anchored = r.opts["affine"] == "stare"
        assert variants.checked_by(name) in (
            "affine_ashlar" if anchored else "ashlar",
            name,
        )
        assert variants.family(name) == ("affine_ashlar" if anchored else "ashlar")
        assert r.after == ("stare" if anchored else None)


def test_runs_for_takes_the_cohorts_dataset_and_no_other():
    names = {variants.run_name(r) for r in variants.runs_for("multiplex")}
    assert "stare_clahe" in names and "stare_percentile" in names
    with pytest.raises(KeyError):
        variants.runs_for("anhir")


def test_the_cohort_is_a_regime_and_takes_only_channel_stare_runs():
    cohort = variants.runs_for(None)
    stare = [r for r in cohort if r.method == "stare"]
    assert stare and all(r.opts.get("proxy", "channel") == "channel" for r in stare)
    # the cohort shares cyclic_if with every dataset of that regime (derived)
    same = [d for d in rg.DATASET_REGIME if rg.DATASET_REGIME[d] == rg.COHORT_REGIME]
    assert same
    for d in same:
        assert {variants.run_name(r) for r in variants.runs_for(d)} == {
            variants.run_name(r) for r in cohort
        }
    assert [r[0] for r in variants.runs_table()] == [
        variants.run_name(r) for r in cohort
    ]
    assert len(variants.expected_variants()) == sum(
        len(variants.variants_of(r)) for r in cohort
    )
    # --all still lists every run, paused or not
    assert len(variants.runs_table(paused=True)) == len(variants.RUNS) == N_RUNS


def test_the_stare_tiers_are_runs_checked_by_stare():
    for name, tier in (("stare_medium", "medium"), ("stare_low", "low")):
        run = variants.run_of(name)
        assert run.method == "stare" and run.opts == {"tier": tier}
        assert (run.image, run.gpu, run.after, run.active) == (
            "STARE_REGBENCH",
            False,
            None,
            True,
        )
        assert variants.checked_by(name) == "stare"
        assert variants.family(name) == "stare"
        assert variants.variants_of(run) == [f"{name}_rigid", name]
        assert variants.stages_of(run) == ("coarse", "tile", "solve", "stitch")
    # the tiers beside high, and nothing else: high is the `stare` row itself
    assert variants.STARE_TIERS == ("medium", "low")
    assert [variants.run_name(r) for r in variants.RUNS if "tier" in r.opts] == (
        STARE_TIER_RUNS
    )
    # neither is a primary row: `stare` stays STARE's primary
    assert variants.primary_of("stare") == "stare"
