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
REMOVED = ("synthetic", "semisynth")

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
