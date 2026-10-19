"""Public datasets beside the cohort, never pooled with it; what was removed stays removed.

The cohort (``multiplex``) is still what ``score``, ``tissue`` and the launcher take by
default (``DATASETS``). The public datasets of plan F (``regbench.PUBLIC``) are named only
on request, and each is scored into tables of its own (``tables/<dataset>/``): a table of
the cohort never holds a row of a public dataset, whatever the call named. A run whose
method has no input rule for a public dataset's modality is left out of it, with its
reason (``excluded_runs.csv``).

``REMOVED`` is the module-level tuple that later removals extend: names that came back
with nothing (the synthetic rungs, ANHIR's tuning side, the preprocessing row). The
scan below says so for the CLI, the regimes and every launcher file.
"""

import ast
import csv
import dataclasses
import importlib
import pathlib
import re
import sys

import numpy as np
import pytest

from benchmarks.regbench import DATASETS, PUBLIC, base_name, public_of, score, variants
from benchmarks.tests.regbench import scored_cohort

pytestmark = pytest.mark.usefixtures("public_datasets")  # conftest: nucset, tissueset

ROOT = pathlib.Path(__file__).resolve().parents[3]
RB = ROOT / "benchmarks" / "regbench"
REMOVED = ("synthetic", "semisynth", "anhir_tune")
REMOVED += ("stare_prep",)

sys.path.insert(0, str(ROOT / "tests"))
strip_line_comment = importlib.import_module("ci_actions").strip_line_comment


def _code(path):
    """What a file SAYS in code: the string constants of a ``.py`` file (a docstring is
    one), the comment-stripped lines of anything else."""
    if path.suffix == ".py":
        tree = ast.parse(path.read_text())
        return " ".join(
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )
    return "\n".join(strip_line_comment(x) for x in path.read_text().splitlines())


def test_the_cohort_is_still_the_default():
    assert DATASETS == ("multiplex",)


def test_the_public_lookup_reads_the_base_name():
    assert base_name("mcmicro@raw") == "mcmicro" and base_name("nucset") == "nucset"
    assert public_of("mcmicro@raw") is PUBLIC["mcmicro"]
    assert public_of("multiplex") is None and public_of(None) is None
    assert not set(PUBLIC) & set(DATASETS)
    assert {spec["kind"] for spec in PUBLIC.values()} == {
        "landmarks",
        "tissue",
        "nuclei",
    }


@pytest.mark.parametrize("name", ["anhir", "hyreco", "tissueset"])
def test_brightfield_public_sets_leave_out_ashlar_and_palom(name):
    kept = {variants.run_name(r) for r in variants.runs_for(name)}
    assert not any(n.startswith(("ashlar", "affine_ashlar", "palom")) for n in kept)
    why = dict(variants.excluded_for(name))
    assert "no input rule" in why["palom"] or "PALOM" in why["palom"]


def test_a_fluorescence_public_set_takes_the_cohorts_runs_and_leaves_none_out():
    assert variants.runs_for("nucset") == variants.runs_for(None)
    assert variants.excluded_for("nucset") == [] and variants.excluded_for(None) == []


def _public_case(root, out, dataset="nucset", case_id="I1"):
    """One fluorescence ``nucset`` case beside the cohort (a nuclei set: no landmark
    scorer exists yet), registered by the forward stand-in."""
    from benchmarks.regbench import methods
    from benchmarks.regbench.cases import load_case, write_case

    src = load_case(root / "multiplex" / "P2_r1")
    case = dataclasses.replace(
        src, dataset=dataset, case_id=case_id, group=case_id, dir=""
    )
    with np.load(src.points_path) as points:
        write_case(root, case, **scored_cohort.polysets_of(points))
    case = load_case(root / dataset / case_id)
    methods.run_case(scored_cohort.forward("stare", 0.4), case, out)
    return case


def test_excluded_run_listed_with_reason(tmp_path, monkeypatch):
    """``excluded_runs.csv`` of a public dataset lists every run left out of it, with its
    reason, and the summary names them; the cohort's tables do not move. the nuclei test set takes every
    run, so a rule of ``NOT_ON`` is given to it here."""
    root, out, made = scored_cohort.build(tmp_path)
    score.score(root, out, **scored_cohort.SCORE_KW)
    cohort_before = (out / "tables" / "cells.csv").read_bytes()
    _public_case(root, out)
    why_palom = "the test leaves PALOM out of the nuclei test set"
    monkeypatch.setitem(
        variants.NOT_ON, "nucset", ((lambda r: r.method == "palom", why_palom),)
    )
    got = score.score(root, out, datasets=["nucset"], expected=[], primary=["stare"])
    with open(out / "tables" / "nucset" / "excluded_runs.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [{"run": "palom", "reason": why_palom}]
    assert got["public"]["nucset"]["excluded_runs"] == rows
    said = (out / "tables" / "nucset" / "summary.md").read_text()
    assert "`palom`" in said and why_palom in said
    assert "palom" not in {variants.run_name(r) for r in variants.runs_for("nucset")}
    with open(out / "tables" / "nucset" / "cells.csv") as fh:
        assert {r["dataset"] for r in csv.DictReader(fh)} == {"nucset"}
    assert (out / "tables" / "cells.csv").read_bytes() == cohort_before
    # the cohort's own: a header alone, nothing is left out of it
    with open(out / "tables" / "excluded_runs.csv") as fh:
        assert list(csv.DictReader(fh)) == []


def test_public_rows_never_reach_cohort_tables(tmp_path):
    """Named in ONE call with the cohort, a public dataset still goes to its own tables."""
    root, out, made = scored_cohort.build(tmp_path)
    _public_case(root, out)
    score.score(root, out, datasets=["multiplex", "nucset"], **scored_cohort.SCORE_KW)
    for name in ("cells", "cells_nuclei"):
        with open(out / "tables" / f"{name}.csv") as fh:
            assert {r["dataset"] for r in csv.DictReader(fh)} == {"multiplex"}, name
    for name in ("resources", "summary_cells"):
        with open(out / "tables" / f"{name}.csv") as fh:
            assert {r["dataset"] for r in csv.DictReader(fh)} <= {"multiplex"}, name
    with open(out / "tables" / "nucset" / "cells.csv") as fh:
        assert {r["dataset"] for r in csv.DictReader(fh)} == {"nucset"}


def test_a_tissue_set_goes_to_the_tissue_block(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    _public_case(root, out, dataset="tissueset", case_id="A1")
    got = score.score(root, out, datasets=["tissueset"], expected=[])
    where = out / "tables" / "tissueset"
    assert (where / "tissue_missing.csv").is_file()
    assert not (where / "cells.csv").exists()
    assert set(got["public"]["tissueset"]) == {
        "tissue", "tissue_summary", "tissue_missing", "excluded_runs", "losses"
    }  # fmt: skip
    # the real path: methods.input_rule leaves ASHLAR and PALOM out of a brightfield set
    with open(where / "excluded_runs.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["reason"].strip() for r in rows)
    assert {"ashlar", "palom"} <= {r["run"] for r in rows}
    assert rows == [
        {"run": n, "reason": w} for n, w in variants.excluded_for("tissueset")
    ]
    assert not (out / "tables" / "cells.csv").exists()


def test_the_command_line_refuses_an_unscored_kind_before_scoring(tmp_path, monkeypatch):
    """``score --dataset multiplex anhir`` with no landmark scorer: nothing is written, the
    cohort's tables neither (every kind of PUBLIC has a scorer since Task 21, so the
    refusal is exercised with landmarks taken out of SCORED_KINDS)."""
    from benchmarks.regbench import cli

    monkeypatch.setattr(score, "SCORED_KINDS", ("tissue", "nuclei"))
    root, out, made = scored_cohort.build(tmp_path)
    with pytest.raises(SystemExit, match=r"anhir.*later"):
        cli.main(
            ["score", "--cases", str(root), "--out", str(out), "--workers", "1",
             "--dataset", "multiplex", "anhir"]
        )  # fmt: skip
    assert not (out / "tables").exists()


def test_every_public_kind_has_a_scorer():
    assert {spec["kind"] for spec in PUBLIC.values()} <= set(score.SCORED_KINDS)


@pytest.mark.parametrize("name", ["anhir", "hyreco"])
def test_a_landmark_set_is_scored_into_its_own_tables_never_the_cohorts(tmp_path, name):
    """Task 21: ANHIR and HyReCo are scored, each alone, into ``tables/<name>/``."""
    from benchmarks.regbench import cli

    fixtures = pathlib.Path(__file__).resolve().parent / "fixtures" / name
    root, out, made = scored_cohort.build(tmp_path)
    flag = ["--source", str(fixtures)] if name == "anhir" else [
        "--data-root", str(fixtures), "--allow-incomplete"]
    assert cli.main(["prepare", name, *flag, "--cases", str(root)]) == 0
    score.score(root, out, datasets=["multiplex", name], **scored_cohort.SCORE_KW)
    assert (out / "tables" / name / f"{name}_pairs.csv").is_file()
    for path in (out / "tables").glob("*.csv"):
        assert name not in path.read_text(), path.name


@pytest.mark.parametrize("name", REMOVED)
def test_no_module_of_a_removed_dataset_is_left(name):
    assert not (RB / "datasets" / f"{name}.py").exists()


@pytest.mark.parametrize("name", REMOVED)
def test_no_launcher_or_cli_choice_names_a_removed_dataset(name):
    for path in [RB / "cli.py", RB / "regimes.py", *sorted((RB / "slurm").glob("*"))]:
        assert not re.search(rf"\b{name}\b", _code(path)), (
            f"{path.relative_to(ROOT)} names {name}"
        )


def test_the_known_truth_spec_stays_gone():
    """``truth.py`` is back as the landmark scorers' error statistics (Task 21); the
    known-truth specification it once read is not."""
    assert not (RB / "truth_spec.json").exists()


@pytest.mark.parametrize("module", ["challenge", "landmarks"])
def test_the_landmark_scorers_write_only_their_own_tables(module):
    """Task 21 brought ANHIR's and HyReCo's scorers back: every table they write is named
    for their dataset, so none can be taken for a cohort table."""
    mod = importlib.import_module(f"benchmarks.regbench.{module}")
    prefix = "anhir_" if module == "challenge" else "hyreco_"
    assert mod.TABLES and all(t.startswith(prefix) for t in mod.TABLES)


def test_no_benchmarks_anhir_directory():
    assert not (ROOT / "benchmarks" / "anhir").exists()


def test_no_preprocessing_selection():
    """``prep.py`` is back with the brightfield proxies only (Task 6b): the ``stare_prep``
    candidate list, its selection and its frozen file stay gone."""
    from benchmarks.regbench import prep

    assert not (RB / "prep_frozen.json").exists()
    for gone in ("CANDIDATES", "selected", "load_frozen", "rank", "ROW", "CRITERION"):
        assert not hasattr(prep, gone), gone


def test_the_arms_and_sweep_layers_are_gone():
    gone = (
        "run_arms.sh",
        "run_sweep.sh",
        "configs",
        "analysis",
        "reg_mosaic.py",
        "run_ashlar_arm.sh",
        "params_json.py",
    )
    present = [g for g in gone if (ROOT / "benchmarks" / g).exists()]
    assert not present, f"removed harness files are back: {present}"
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "benchmarks"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    allowed = ("benchmarks/__init__.py", "benchmarks/README.md")
    allowed_dirs = ("benchmarks/ashlar/", "benchmarks/regbench/", "benchmarks/tests/")
    stray = [f for f in tracked if f not in allowed and not f.startswith(allowed_dirs)]
    assert not stray, (
        f"tracked files outside benchmarks/{{__init__.py,README.md,ashlar,regbench,tests}}: {stray}"
    )


# ── the selfcheck and the realcheck run and report, and never gate (user ruling) ──────
GATE_SWITCHES = (
    "REGBENCH_SELFCHECK_ADVISORY",
    "REGBENCH_ALLOW_UNCHECKED",
    "REGBENCH_ALLOW_INCONSISTENT",
)
# the one place that still names them: the notice that they are ignored
IGNORED_LOOP = f"for v in {' '.join(GATE_SWITCHES)}; do"
IGNORED_SAID = 'echo "# ignored: $v (the selfcheck no longer gates)"'


def test_nothing_gates():
    """Comment-blind: what the batch scripts and the launcher RUN, and what the Python
    package's strings say. A comment may tell the story of the gate; code may not
    bring it back."""
    slurm = sorted((RB / "slurm").glob("*"))
    assert slurm and not (RB / "slurm" / "gate.sbatch").exists()
    needles = (
        "afterok:${!gate}",
        "afterok:${!real}",
        "status=blocked",
        "cause=gate_blocked",
        "logs/gates",
        "rb-gate-",
        "rb-realgate-",
        "gate.sbatch",
    )
    for path in slurm:
        lines = [x for x in _code(path).splitlines() if x.strip()]
        code = "\n".join(lines)
        for needle in needles:
            assert needle not in code, f"{path.name} runs {needle}"
        for var in GATE_SWITCHES:
            naming = [x.strip() for x in lines if var in x]
            if path.name == "submit_cohort.sh":
                # the switches are only ever said to be ignored, and read for nothing else
                assert naming == [IGNORED_LOOP], naming
                at = [x.strip() for x in lines].index(IGNORED_LOOP)
                assert IGNORED_SAID in lines[at + 1], lines[at + 1]
            else:
                assert not naming, f"{path.name} reads {var}"
    for path in sorted(RB.rglob("*.py")):
        said = _code(path)
        assert "gate_blocked" not in said, path.relative_to(ROOT)
        assert "dropped_row" not in said, path.relative_to(ROOT)
    from benchmarks.regbench import cli, failure_log, selfcheck, variants

    assert not hasattr(selfcheck, "BLOCKING")
    assert "gate" not in failure_log.STAGES and "gate_blocked" not in failure_log.WHOSE
    for gone in ("gate_status", "gate_variants", "dropped_variants", "droppable"):
        assert not hasattr(variants, gone), gone
    import argparse

    (commands,) = [
        a
        for a in cli.build_parser()._actions
        if isinstance(a, argparse._SubParsersAction)
    ]
    assert "verdict" in commands.choices
    assert not {"gate", "realgate"} & set(commands.choices)


def test_the_paper_tables_of_a_public_set_are_its_own(tmp_path):
    """``paper --dataset nucset`` reads ``tables/nucset/`` and writes ``paper/nucset/``; the
    cohort's paper tables have no the nuclei test set row."""
    from benchmarks.regbench.paper import tables as paper_tables

    root, out, made = scored_cohort.build(tmp_path)
    _public_case(root, out)
    score.score(root, out, datasets=["multiplex", "nucset"], **scored_cohort.SCORE_KW)
    paper_tables.write_all(out, "nucset")
    paper_tables.write_all(out)
    with open(out / "paper" / "nucset" / "per_pair.csv") as fh:
        assert {r["dataset"] for r in csv.DictReader(fh)} == {"nucset"}
    with open(out / "paper" / "per_pair.csv") as fh:
        assert {r["dataset"] for r in csv.DictReader(fh)} == {"multiplex"}
