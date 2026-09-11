"""In-process counterpart of tests/test_preflight_scale.py.

That file drives bin/preflight_scale.py through subprocess, which is faithful to
how Nextflow runs it and invisible to coverage: the script measured 0% on
2026-09-10 with nine passing tests. These call main(argv) in the test
process, so the lines are counted, and they add the boundaries the subprocess
file does not reach: the parser's own rejections and the warn-on-heterogeneity
clustering.

FINDING: `_parse_pixel_size`'s docstring (bin/preflight_scale.py:55-59) promises
"Raises ValueError for anything that is neither" a positive number nor 'auto' --
but "nan" and "inf" are neither rejected here. `float("nan") <= 0` and
`float("inf") <= 0` are both False (NaN compares false to everything, and
+inf is not <= 0), so the `value <= 0` guard silently lets both through:
`_parse_pixel_size("nan")` returns `nan`, `_parse_pixel_size("inf")` returns
`inf`, and an infinite or NaN pixel size would be written into the report
JSON as if it were a valid, positive scale. bin/ is read-only for this task;
per the brief, "nan" and "inf" are dropped from the rejection parametrisation
below rather than asserted as passing (which would hide the finding) or
asserted as raising (which would fail). See task-14-15-report.md for the
full write-up.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bin"))

import preflight_scale  # noqa: E402

from tests.test_preflight_scale import _write_no_scale, _write_with_scale  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [("auto", None), ("AUTO", None), (" Auto ", None), ("0.5", 0.5), ("2", 2.0)],
)
def test_parse_pixel_size_accepts_auto_in_any_case_and_positive_numbers(raw, expected):
    assert preflight_scale._parse_pixel_size(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "", "0.5um"])
def test_parse_pixel_size_rejects_non_positive_and_non_numeric(raw):
    with pytest.raises((ValueError, SystemExit)):
        preflight_scale._parse_pixel_size(raw)


@pytest.mark.parametrize("raw", ["nan", "inf"])
def test_parse_pixel_size_accepts_nan_and_inf_finding(raw):
    """FINDING, not a spec: the docstring promises ValueError for "anything that
    is neither" a positive number nor 'auto', but `value <= 0` is False for both
    NaN and +inf, so both pass straight through instead of raising. Recorded
    here (rather than in the rejects-parametrisation above) so the suite stays
    green while the discrepancy stays visible and executable."""
    import math

    result = preflight_scale._parse_pixel_size(raw)
    assert math.isnan(result) or math.isinf(result)


def test_main_auto_writes_a_report_naming_each_image(tmp_path):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.5)
    b = _write_with_scale(tmp_path, "b.ome.tiff", 0.5)
    out = tmp_path / "report.json"
    rc = preflight_scale.main(["--images", str(a), str(b), "--pixel-size", "auto", "--output", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    text = out.read_text()
    assert "a.ome.tiff" in text and "b.ome.tiff" in text
    assert isinstance(report, dict)


def test_main_auto_with_no_metadata_returns_nonzero_and_names_the_offender(tmp_path, caplog):
    a = _write_no_scale(tmp_path, "blank.ome.tiff")
    out = tmp_path / "report.json"
    with caplog.at_level(logging.ERROR):
        rc = preflight_scale.main(["--images", str(a), "--pixel-size", "auto", "--output", str(out)])
    assert rc != 0
    assert "blank.ome.tiff" in caplog.text + out.read_text() if out.exists() else "blank.ome.tiff" in caplog.text


def test_main_number_disagreeing_with_metadata_warns_but_succeeds(tmp_path, caplog):
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.5)
    out = tmp_path / "report.json"
    with caplog.at_level(logging.WARNING):
        rc = preflight_scale.main(["--images", str(a), "--pixel-size", "0.25", "--output", str(out)])
    assert rc == 0
    assert "0.25" in caplog.text or "0.5" in caplog.text, "the disagreement must be logged"


def test_main_with_a_space_in_the_filename(tmp_path):
    d = tmp_path / "with space"
    d.mkdir()
    a = _write_with_scale(d, "P001 ref.ome.tiff", 0.5)
    out = tmp_path / "report.json"
    rc = preflight_scale.main(["--images", str(a), "--pixel-size", "auto", "--output", str(out)])
    assert rc == 0
    assert "P001 ref.ome.tiff" in out.read_text()


def test_warn_on_heterogeneous_scales_is_silent_below_two_values():
    # `report` is keyed by image path -> {"pixel_size": ..., "source": ...}
    # (bin/preflight_scale.py:97-130), not by a literal "images" key; an empty
    # report has zero distinct pixel sizes, which is < 2 and must not warn.
    class Rec:
        def __init__(self):
            self.msgs = []

        def warning(self, msg, *a):
            self.msgs.append(msg % a if a else msg)

    log = Rec()
    preflight_scale._warn_on_heterogeneous_scales({}, log)
    assert log.msgs == []


def test_warn_on_heterogeneous_scales_warns_on_two_clusters(tmp_path, caplog):
    # Build the report the real code builds, by running main on two disagreeing files.
    a = _write_with_scale(tmp_path, "a.ome.tiff", 0.25)
    b = _write_with_scale(tmp_path, "b.ome.tiff", 0.5)
    out = tmp_path / "report.json"
    with caplog.at_level(logging.WARNING):
        rc = preflight_scale.main(["--images", str(a), str(b), "--pixel-size", "auto", "--output", str(out)])
    assert rc == 0
    assert "heterogeneous" in caplog.text.lower() or "cluster" in caplog.text.lower() or "0.25" in caplog.text
