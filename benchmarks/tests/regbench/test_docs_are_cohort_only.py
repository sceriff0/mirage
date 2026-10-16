"""The regbench documents describe the cohort harness, and nothing that was removed.

``benchmarks/regbench/README.md``, ``benchmarks/README.md`` and ``docs/benchmarks.md`` do
not name the removed harnesses (the synthetic rung, the external landmark sets, the arms
and the sweep) or the removed gate switches. The one place that may is the
``## Removed harnesses`` section of ``benchmarks/README.md``, the record of the removal
and of the archive tag. ``ANHIR`` alone is also allowed in a sentence that says it was
removed (the record, and the caveat that no landmark set is used).

The text is read comment-free (HTML comments out), and every ``submit_cohort.sh`` flag the
README names must be one the script accepts. The script has no ``--help``: its accepted
flags are the labels of its option ``case``, read from the script with its comments
stripped (``ci_actions.strip_line_comment``).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
README = REPO / "benchmarks" / "regbench" / "README.md"
ENTRY = REPO / "benchmarks" / "README.md"
OVERVIEW = REPO / "docs" / "benchmarks.md"
LAUNCHER = REPO / "benchmarks" / "regbench" / "slurm" / "submit_cohort.sh"
DOCS = (README, ENTRY, OVERVIEW)

FORBIDDEN = (
    "semisynth",
    "HyReCo",
    "run_arms",
    "arms.yaml",
    "sweep.yaml",
    "REGBENCH_SELFCHECK_ADVISORY",
    "REGBENCH_ALLOW_UNCHECKED",
    "gate.sbatch",
    "prep.py",
    "stare_prep",
    "n_not_run_or_blocked",
    "TUNING",
    "NO_BRIGHTFIELD",
    "logs/gates",
)
# only in a sentence that says it was removed
REMOVED_ONLY = ("ANHIR",)
# only in a sentence that records a rename or the back-compatible variable
RENAMED_ONLY = ("arms_manifest",)
RENAME_MARKERS = ("REGBENCH_MANIFEST", "renamed")
REMOVED_SECTION = "## Removed harnesses"

sys.path.insert(0, str(REPO / "tests"))
strip_line_comment = importlib.import_module("ci_actions").strip_line_comment


def _without_comments(text: str) -> str:
    """Markdown with its HTML comments removed: what a reader is shown."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _without_removed_section(path: Path, text: str) -> str:
    if path != ENTRY:
        return text
    start = text.index(REMOVED_SECTION)
    after = text[start + len(REMOVED_SECTION) :]
    nxt = re.search(r"^## ", after, flags=re.M)
    end = start + len(REMOVED_SECTION) + (nxt.start() if nxt else len(after))
    return text[:start] + text[end:]


def _visible(path: Path) -> str:
    return _without_removed_section(path, _without_comments(path.read_text()))


def _sentences(text: str):
    # a sentence ends at a full stop before a space or line break; lines are joined first
    flat = re.sub(r"\s*\n\s*", " ", text)
    return re.split(r"(?<=[.!?])\s+", flat)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
@pytest.mark.parametrize("term", FORBIDDEN)
def test_the_docs_do_not_name_what_was_removed(path, term):
    assert term not in _visible(path), (
        f"{path.relative_to(REPO)} names {term!r}, which was removed; it may only be "
        f"named inside the {REMOVED_SECTION!r} section of benchmarks/README.md"
    )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
@pytest.mark.parametrize("term", REMOVED_ONLY)
def test_a_removed_name_is_only_in_a_sentence_that_says_so(path, term):
    for sentence in _sentences(_visible(path)):
        if term in sentence:
            assert "removed" in sentence.lower(), (
                f"{path.relative_to(REPO)}: {term!r} outside a sentence that says it was "
                f"removed: {sentence[:160]!r}"
            )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
@pytest.mark.parametrize("term", RENAMED_ONLY)
def test_a_renamed_module_is_only_named_in_a_rename_sentence(path, term):
    for sentence in _sentences(_visible(path)):
        if term in sentence:
            assert any(m in sentence for m in RENAME_MARKERS), (
                f"{path.relative_to(REPO)}: {term!r} outside a sentence that names "
                f"REGBENCH_MANIFEST or says renamed: {sentence[:160]!r}"
            )


def test_the_removal_record_and_its_tag_are_kept():
    text = ENTRY.read_text()
    assert REMOVED_SECTION in text
    assert "archive/benchmarks-arms-sweep" in _without_comments(text)


def _accepted_flags() -> set[str]:
    """The flags ``submit_cohort.sh`` accepts: the labels of its option ``case``."""
    code = "\n".join(strip_line_comment(x) for x in LAUNCHER.read_text().splitlines())
    start = code.index('case "$1" in')
    block = code[start : code.index("esac", start)]
    flags: set[str] = set()
    for label in re.findall(r"^\s*((?:--[a-z-]+\|?)+)\)", block, flags=re.M):
        flags.update(label.split("|"))
    return flags


def _named_flags(text: str) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        for m in re.finditer(r"submit_cohort\.sh\b(?P<rest>.*)", line):
            # the flags of the command: up to the end of an inline code span, and before
            # a bare `--` (the site's own sbatch options go after it)
            rest = m.group("rest").split("`")[0]
            rest = re.split(r"\s--\s|\s--$", rest)[0]
            out.update(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest))
    return out


def test_the_launcher_flags_are_read_from_the_script():
    flags = _accepted_flags()
    assert {"--dry-run", "--inputs", "--prepare", "--compute-all"} <= flags, flags


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_every_launcher_flag_a_doc_names_is_accepted(path):
    accepted = _accepted_flags()
    named = _named_flags(_visible(path))
    assert not (named - accepted), (
        f"{path.relative_to(REPO)} names submit_cohort.sh flags the script does not "
        f"accept: {sorted(named - accepted)}"
    )


def test_the_readme_names_launcher_flags_at_all():
    assert _named_flags(_visible(README)), "the check above would pass over nothing"
