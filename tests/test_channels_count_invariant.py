"""`channels_count` dual invariant: emitted-file count == distinct-name count.

lib/Meta.groovy's `finish()` states the rule directly (see the comment above
`meta.channels_count = requirePerPatientCount(...)`):

    The dual invariant: channels_count must equal BOTH the number of
    distinct channel names AND the number of emitted files. One consumer
    .unique()s the list and one does not, so any divergence desynchronises
    a streaming groupTuple(size:) and the run hangs or mis-groups.

`channels_count` is computed once, per patient, by
`lib/CsvUtils.groovy::countChannelsPerPatient`, which sums the per-slide
keep-lists `resolveKeptChannelsPerSlide` returns. It feeds
`groupKey(patient_id, channels_count)` at TWO call sites that read the SAME
number for two different purposes:

  1. `subworkflows/local/quantify_markers.nf`'s `ch_grouped_csvs` grouping
     (the per-marker QUANTIFY CSVs) -- this one `.unique { entry -> entry[0] }`s
     on `[patient_id, marker]` immediately before grouping, so it needs the
     grouping to close at the DISTINCT-NAME count.
  2. `subworkflows/local/quantify_markers.nf`'s shared `groupTiffsByPatient()`
     function -- used by both `postprocess.nf`'s pyramid-channel grouping and
     `add_cycle.nf`'s priority-`groupTuple`-fed grouping -- which has NO
     internal `.unique()` of its own (its callers dedup upstream instead), so
     in the general case it needs the grouping to close at the EMITTED-FILE
     count.

`resolveKeptChannelsPerSlide`'s claim rule (each marker name claimed once per patient,
reference first then samplesheet order) makes these the same number: claiming a name
marks it emitted, so "emitted" and "distinct" cannot drift apart.

The logic is Groovy-only, reachable only through `tests/lib_probe.nf`. The probe covers
a duplicate across two slides; this file exercises a duplicate within one slide's own
`channels` cell (e.g. `DAPI|DAPI|KI67`).

These tests shell out to `nextflow`, which the pytest CI job does not
install (see tests/test_cleanup_validation.py's identical rationale), so they
skip cleanly there; CI's `nextflow-stub` job runs tests/lib_probe.nf directly,
and tests/lib_probe.nf.test runs it again through nf-test.
"""

import shutil
import subprocess

import pytest

from tests.nfmodel import REPO_ROOT, nf_files

pytestmark = pytest.mark.skipif(
    shutil.which("nextflow") is None,
    reason="nextflow is not on PATH; the probe also runs in CI's nextflow-stub job "
    "and via tests/lib_probe.nf.test",
)


def _lib_probe_path():
    """Locate tests/lib_probe.nf via tests.nfmodel.nf_files, never a private
    Path.glob -- see tests/test_nfmodel.py::test_no_guard_parses_nextflow_source_privately."""
    matches = [p for p in nf_files(("tests",)) if p.name == "lib_probe.nf"]
    assert matches, "tests/lib_probe.nf not found under tests/ -- has it moved?"
    return matches[0]


LIB_PROBE = _lib_probe_path()


def _run_probe():
    return subprocess.run(
        ["nextflow", "run", str(LIB_PROBE), "-lib", "lib"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_channels_count_dual_invariant_holds_for_a_within_slide_duplicate():
    """tests/lib_probe.nf's 'duplicate marker declared twice on one row'
    block must run and hold: the pipeline's live Groovy claim rule
    (lib/CsvUtils.groovy::resolveKeptChannelsPerSlide) already drops the
    same-row repeat, so channels_count, the emitted-file count, and the
    distinct-name count all agree at 4 for patient P9 (DAPI, KI67, CD20,
    CELLTOX) -- not 6, the naive sum of the two rows' declared cells."""
    result = _run_probe()
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "LIB PROBE: all assertions passed" in out, out


def test_the_probe_actually_contains_the_within_slide_duplicate_case():
    """If the probe's block were deleted, the test above would still pass, so
    anchor on the case's own CSV text."""
    text = LIB_PROBE.read_text()
    assert "DAPI|DAPI|KI67|CD20" in text, (
        "tests/lib_probe.nf no longer declares the within-slide duplicate "
        "channels_count case (P9); test_channels_count_dual_invariant_holds_"
        "for_a_within_slide_duplicate is decorative without it."
    )
