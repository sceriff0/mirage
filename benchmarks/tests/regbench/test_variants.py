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

EXPECTED = [
    "stare_rigid",
    "stare",
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
    "palom_b1024_affine",
    "palom_b1024",
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
    monkeypatch.setattr(palom, "VARIANTS", ("palom_affine", "palom_renamed"))
    got = variants.expected_variants()
    assert "palom_renamed" in got and "palom_b1024_renamed" in got
    assert "palom_b1024" not in got


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
    assert [r.opts for r in variants.RUNS if r.method == "stare"] == [{}]


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
            block, _ = palom.settings(opts)
            assert block == palom.LABELS[name]
        elif run.method == "valis":
            hires = str(run.opts.get("hires", "0")) == "1"
            assert hires == (name == valis.HIRES_LABEL)
    assert {n for n, r in runs.items() if r.method == "ashlar"} == set(ashlar.LABELS)
    assert {n for n, r in runs.items() if r.method == "palom"} == set(palom.LABELS)
    assert variants.HR_CONFIG == dhr.HR_CONFIG


def test_a_wrong_option_is_caught_by_the_adapter_check():
    """The check above can fail: the 4096 px label with the 8192 px set is refused."""
    with pytest.raises(ValueError):
        dhr.settings({"label": "deeperhistreg_gpu", "config": variants.HR_CONFIG})
    with pytest.raises(ValueError):
        palom.settings({"label": "palom", "block": "1024"})


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
    """THE one place the DeeperHistReg primary is written out in the tests. It is an
    open decision (4096 px `deeperhistreg_gpu_asdrawn` or 8192 px
    `deeperhistreg_hr_gpu_asdrawn`): switching is `variants.PRIMARY_DEEPERHISTREG` and
    this line."""
    assert variants.PRIMARY_DEEPERHISTREG == "deeperhistreg_gpu_asdrawn"
    assert variants.PRIMARY == (
        "stare",
        "valis_mirage",
        "valis120",
        variants.PRIMARY_DEEPERHISTREG,
        "ashlar",
        "palom",
    )


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
    assert variants.family("palom_b1024_affine") == "palom"
    assert variants.family("ashlar_clean") == "ashlar"
    # a prefix only counts at a name boundary, and an off-seam row is its variant's
    assert variants.family("stare:non_palom") == "stare"
    assert variants.family("palomino") is None
    assert variants.family("initial") is None


def test_palom_b1024_is_in_paloms_family_and_not_primary():
    assert variants.family("palom_b1024") == "palom"
    assert "palom_b1024" not in variants.PRIMARY
    assert "palom" in variants.PRIMARY


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
    # every other run stands behind its own selfcheck, palom_b1024 included
    for name in names - set(variants.GATED_BY):
        assert variants.gated_by(name) == name
    assert variants.gated_by("palom_b1024") == "palom_b1024"


def test_selfcheck_masks_come_from_the_adapters():
    assert variants.selfcheck_mask("ashlar") == "overlap" == ashlar.MASKS[0]
    assert variants.selfcheck_mask("palom") == "seams" == palom.MASKS[0]
    for method in ("stare", "valis", "valis_mirage", "deeperhistreg"):
        assert variants.selfcheck_mask(method) is None


def test_images_are_the_six_the_launcher_knows():
    images = {r.image for r in variants.RUNS}
    assert images == {
        "STARE",
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
    assert set(variants.BUILT_HERE) == {"STARE", "VALIS120", "DEEPERHISTREG", "PALOM"}
    assert set(variants.BUILT_HERE) | set(variants.PULLED) == images


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
)


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
    got = rows(capsys)
    assert len(got) == 18
    assert all(len(r) == len(COLUMNS) for r in got)
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
        "image": "STARE",
        "gpu": "0",
        "scope": "pair",
        "after": "-",
        "gated_by": "stare",
        "selfcheck_mask": "-",
        "primary": "stare",
        "variants": "stare_rigid,stare",
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
    }
    hr = got["deeperhistreg_hr_fluo_gpu"]
    assert hr["opts"] == f"device=cuda:0,config={variants.HR_CONFIG},fluo=1"
    assert hr["gpu"] == "1" and hr["scope"] == "pair" and hr["primary"] == "-"
    assert got["palom_b1024"]["opts"] == "block=1024"
    assert got["palom_b1024"]["selfcheck_mask"] == "seams"
    assert got["palom_b1024"]["gated_by"] == "palom_b1024"
    assert got["valis120"]["primary"] == "valis120"
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
