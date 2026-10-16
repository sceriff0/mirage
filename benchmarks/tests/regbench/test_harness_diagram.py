"""The harness figure (``docs/figures/regbench-harness.html``) names what the code has.

The figure is the user's one picture of the whole harness, so it must not drift from the
code it draws. Its VISIBLE text (script, style and comments out, tags stripped) names:

* every primary row (``variants.PRIMARY``) and every run family;
* every table the scorer writes (``score.TABLES``) and the compute-all tables;
* the job kinds of the launcher, each of which is a job-name prefix ``submit_cohort.sh``
  really submits;
* the words "never gates", because the selfcheck and the realcheck are reported only.

It names none of what was removed, and it loads nothing from outside the file.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from benchmarks.regbench import score, variants

REPO = Path(__file__).resolve().parents[3]
FIGURE = REPO / "docs" / "figures" / "regbench-harness.html"
LAUNCHER = REPO / "benchmarks" / "regbench" / "slurm" / "submit_cohort.sh"

FAMILIES = (
    "stare",
    "ashlar",
    "valis_mirage",
    "valis120",
    "deeperhistreg",
    "palom",
    "initial",
)
COMPUTE_TABLES = ("pipeline_cost", "pipeline_cost_summary", "pipeline_cost_by_process")
JOB_KINDS = (
    "inputs",
    "nuclei",
    "prepare",
    "rb-sc",
    "ladders",
    "settle",
    "score",
    "compute",
    "verdict",
)
# job-name prefixes the figure shows, each of which the launcher submits
PREFIXES = (
    "rb-inputs",
    "rb-nuclei",
    "rb-prepare",
    "rb-sc-",
    "rb-settle",
    "rb-score-case",
    "rb-score",
    "rb-compute",
    "rb-score-compute",
)
# removed with the other harnesses: the figure draws the cohort harness alone
FORBIDDEN = (
    r"synthetic",
    r"semisynth",
    r"hyreco",
    r"anhir",
    r"stare_prep",
    r"gate\.sbatch",
    r"(?<![A-Za-z0-9_])arms(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])sweep(?![A-Za-z0-9_])",
)


def visible_text() -> str:
    raw = FIGURE.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
    raw = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


@pytest.fixture(scope="module")
def text() -> str:
    assert FIGURE.is_file(), f"{FIGURE} is not there"
    return visible_text()


def test_primary_rows_are_named(text):
    missing = [r for r in variants.PRIMARY if r not in text]
    assert not missing, missing


def test_run_families_are_named(text):
    missing = [f for f in FAMILIES if f not in text]
    assert not missing, missing


def test_every_run_of_the_list_is_covered_by_a_family(text):
    """A run the figure draws a family for: every run name starts with a family named."""
    for run in variants.RUNS:
        name = variants.run_name(run)
        assert any(name.startswith(f) for f in (*FAMILIES, "affine_ashlar")), name


def test_every_table_is_named(text):
    missing = [t for t in (*score.TABLES, *COMPUTE_TABLES) if t not in text]
    assert not missing, missing


def test_job_kinds_are_named(text):
    missing = [k for k in JOB_KINDS if k not in text]
    assert not missing, missing


def test_job_prefixes_exist_in_the_launcher(text):
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for prefix in PREFIXES:
        assert prefix in text, f"{prefix} is not in the figure"
        assert f'"{prefix}' in launcher, f"the launcher submits no job named {prefix}*"


def test_selfcheck_and_realcheck_never_gate(text):
    assert "never gates" in text


def test_nothing_removed_is_named(text):
    low = text.lower()
    found = [p for p in FORBIDDEN if re.search(p, low)]
    assert not found, found


def test_the_figure_is_self_contained():
    raw = FIGURE.read_text(encoding="utf-8")
    assert not re.search(
        r"<link\b|<script\b[^>]*\bsrc=|@import|url\(\s*['\"]?https?:", raw
    )
    assert not re.search(r"(src|href)=[\"']https?:", raw)
