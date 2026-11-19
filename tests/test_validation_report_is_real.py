"""The committed format-validation report must be evidence, not a placeholder.

The vendor formats cannot be synthesised, so this report stands in for fixtures.
A template, or a report whose rows all failed, reads as coverage it is not.

It must name the commit it was produced from and carry a row for every vendor
format the pipeline claims. It need not have every row OK: a recorded failure is
a finding.

Pending is an explicit state: the cluster kit needs a cluster and private images
CI does not have. The marker `kit-validated: pending` relaxes only the checks that
need a real run (commit SHA, probe container, one successful row) and adds a check
that the pending page fakes no success. Every vendor format must still have a row,
and the template's placeholder path must never appear.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "validation" / "format_validation.md"

# The formats that CANNOT be synthesised, i.e. the ones this report exists for.
# Everything else is covered on every push by tests/integration/formats/.
VENDOR_FORMATS = (".czi", ".nd2", ".lif", ".ndpi", ".svs")

# The exact string the pending page carries instead of a real measurement.
# tests/cluster/validate_formats.sh's own output never contains it (it writes a
# real `pipeline commit: `<sha>`` header), so its presence means no run has landed.
PENDING_MARKER = "kit-validated: pending"


def _text() -> str:
    assert REPORT.is_file(), (
        f"{REPORT.relative_to(ROOT)} is missing. It is produced by "
        "`tests/cluster/validate_formats.sh` on the cluster and committed as the "
        "evidence RULING R3 accepts in place of vendor fixtures -- or, before that "
        f"run has happened, as an explicit pending page carrying '{PENDING_MARKER}'."
    )
    return REPORT.read_text(encoding="utf-8")


def _is_pending(text: str) -> bool:
    return PENDING_MARKER in text


def test_the_report_declares_a_real_run_or_an_explicit_pending_state():
    """The report is exactly one of the two named states: a real run, or pending."""
    text = _text()
    has_real_commit = re.search(r"pipeline commit:\s*`[0-9a-f]{40}`", text) is not None
    assert _is_pending(text) or has_real_commit, (
        f"the report is neither the explicit pending state (containing "
        f"'{PENDING_MARKER}') nor a real run (a 40-character commit SHA under "
        "'pipeline commit:') -- it must be one or the other, never silently neither"
    )


def test_the_report_names_the_commit_it_was_produced_from():
    """Without a SHA the report cannot be tied to the tree it measured."""
    text = _text()
    if _is_pending(text):
        pytest.skip("pending state: no cluster run has been recorded yet")
    assert re.search(r"pipeline commit:\s*`[0-9a-f]{40}`", text), (
        "no 40-character commit SHA in the report header -- validate_formats.sh "
        "writes one, so this file was hand-edited or truncated"
    )


def test_the_report_names_the_container_the_probe_ran_in():
    """`.svs` needs the Bio-Formats jars baked into the image, so which image
    produced these numbers is part of the measurement."""
    text = _text()
    if _is_pending(text):
        pytest.skip("pending state: no cluster run has been recorded yet")
    assert "probe container:" in text


@pytest.mark.parametrize("suffix", VENDOR_FORMATS)
def test_every_vendor_format_has_a_row(suffix):
    """Holds in both states: every promised format is listed, so a dropped format
    is caught."""
    text = _text()
    assert suffix in text, (
        f"the report has no row for {suffix}. That is one of the five formats the "
        "README claims and CI cannot synthesise -- either it was validated and the "
        "row is missing, or it was not validated and the claim should come out of "
        "README.md and bin/convert_image.py's docstring in the same commit."
    )


def test_the_report_is_not_the_template():
    """The template's placeholder path must never reach the committed report, in
    either state."""
    assert "/absolute/path/to/your/" not in _text(), (
        "the committed report still contains the samplesheet template's placeholder "
        "path, so it records a run that never happened"
    )


def test_the_report_records_at_least_one_successful_conversion():
    """A report of nothing but failures may be committed, but at least one row must
    succeed before it counts as validation."""
    text = _text()
    if _is_pending(text):
        pytest.skip("pending state: no cluster run has been recorded yet")
    assert "| OK |" in text, (
        "no row in the report succeeded. Read the FAILED rows: this is a finding "
        "about the readers, not about the report."
    )


def test_the_pending_report_does_not_fake_a_successful_row():
    """The check the pending state adds: `| OK |` is the string validate_formats.sh
    writes for a real success, so a pending page must not carry it."""
    text = _text()
    if not _is_pending(text):
        pytest.skip("real report: OK rows are expected once the kit has run")
    assert "| OK |" not in text, (
        "the pending report contains '| OK |' -- that claims a successful probe "
        "that never ran; remove it, or replace this file with the kit's real output "
        "and drop the pending marker"
    )
