"""The cohort is the only dataset: what was removed stays removed.

``REMOVED`` is the module-level tuple that later removals extend.
"""

import ast
import importlib
import pathlib
import re
import sys

import pytest

from benchmarks.regbench import DATASETS

ROOT = pathlib.Path(__file__).resolve().parents[3]
RB = ROOT / "benchmarks" / "regbench"
REMOVED = ("synthetic", "semisynth", "anhir", "anhir_tune", "hyreco_a", "hyreco_b")
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


def test_the_cohort_is_the_only_dataset():
    assert DATASETS == ("multiplex",)


@pytest.mark.parametrize("name", REMOVED)
def test_no_module_of_a_removed_dataset_is_left(name):
    assert not (RB / "datasets" / f"{name}.py").exists()


@pytest.mark.parametrize("name", REMOVED)
def test_no_launcher_or_cli_choice_names_a_removed_dataset(name):
    for path in [RB / "cli.py", RB / "regimes.py", *sorted((RB / "slurm").glob("*"))]:
        assert not re.search(rf"\b{name}\b", _code(path)), (
            f"{path.relative_to(ROOT)} names {name}"
        )


def test_the_known_truth_scorer_is_gone():
    assert not (RB / "truth.py").exists()
    assert not (RB / "truth_spec.json").exists()


@pytest.mark.parametrize("module", ["challenge", "landmarks", "notrun"])
def test_the_external_scorers_are_gone(module):
    assert not (RB / f"{module}.py").exists()


def test_no_benchmarks_anhir_directory():
    assert not (ROOT / "benchmarks" / "anhir").exists()


def test_no_prep_module():
    from benchmarks.regbench import regimes

    assert not (RB / "prep.py").exists()
    assert not (RB / "prep_frozen.json").exists()
    assert set(regimes.REGIMES) == {"cyclic_if"}


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
