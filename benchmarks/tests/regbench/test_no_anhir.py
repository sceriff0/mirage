"""ANHIR is gone from regbench (user ruling 2026-10-19: "remove anhir").

No file under ``benchmarks/regbench``, and not the harness diagram, names it, with two
kinds of exception (the tests may: they assert that it is refused):

* the method-code freeze (``method_code_freeze.json``): a frozen file's bytes may not
  change outside a commit of its own, and its docstrings still name ANHIR as the source
  of a rule ported from it. Those mentions are prose, not a dataset.
* ``QUOTED``: a published fact about another method, quoted from a paper (the conditions
  of HyReCo's published numbers), where removing the word would misquote the source.

The dataset itself is unknown everywhere: not public, no regime, no prepare command, no
launcher dataset, no scorer, no table.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from benchmarks.regbench import PUBLIC, regimes, score, variants

ROOT = Path(__file__).resolve().parents[3]
FREEZE = Path(__file__).with_name("method_code_freeze.json")
SCOPE = ("benchmarks/regbench", "docs/figures/regbench-harness.html")
# (path, the exact line fragment): quotes of a paper, not the dataset
QUOTED = {
    (
        "benchmarks/regbench/landmarks.py",
        '"ANHIR, not tuned on HyReCo; licensed software, no public code",',
    ),
}
WORD = re.compile("anhir", re.IGNORECASE)


def _frozen():
    return {
        f for spec in json.loads(FREEZE.read_text()).values() for f in spec["files"]
    }


def _tracked():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *SCOPE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [p for p in out if (ROOT / p).is_file()]


def test_no_file_names_anhir():
    frozen = _frozen()
    hits = []
    for rel in _tracked():
        if rel in frozen:
            continue
        try:
            text = (ROOT / rel).read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if WORD.search(line) and (rel, line.strip()) not in QUOTED:
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_the_quoted_exceptions_still_exist():
    """An exception that matches nothing is stale and is removed."""
    for rel, fragment in QUOTED:
        assert fragment in [ln.strip() for ln in (ROOT / rel).read_text().splitlines()]


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/regbench/datasets/anhir.py",
        "benchmarks/regbench/download/anhir.sh",
        "benchmarks/regbench/challenge.py",
        "benchmarks/tests/regbench/fixtures/anhir",
        "benchmarks/tests/regbench/test_anhir.py",
        "benchmarks/anhir",
    ],
)
def test_the_anhir_files_are_gone(path):
    assert not (ROOT / path).exists()


def test_anhir_is_unknown_everywhere():
    assert "anhir" not in PUBLIC
    assert "anhir" not in regimes.DATASET_REGIME
    assert not [t for t in score.TABLES if WORD.search(t)]
    assert not hasattr(score, "ANHIR_TABLES")
    with pytest.raises(KeyError):
        regimes.regime_of("anhir")
    assert "anhir" not in variants.NOT_ON
