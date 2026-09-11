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
JSON as if it were a valid, positive scale. bin/ is read-only for this task,
so `test_parse_pixel_size_rejects_nan_and_inf` below asserts the CONTRACT --
`pytest.raises(ValueError)` -- under `@pytest.mark.xfail(strict=True)`. That
way the test never pins the defect as correct behaviour (a test asserting the
nan/inf return value would turn RED when the bug is fixed, punishing the fix),
and `strict=True` makes it fail loudly the moment `_parse_pixel_size` starts
rejecting them, so the marker cannot outlive the defect. Same shape as the
`_safe_mean` finding in tests/test_helper_edge_cases.py. See
task-14-15-report.md for the full write-up.
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        'FINDING: bin/preflight_scale.py:57 promises "Raises ValueError for anything '
        "that is neither\" a positive number nor 'auto', but `value <= 0` is False for "
        "both NaN and +inf (NaN compares false to everything; +inf is not <= 0), so "
        "both pass straight through and would be written into the report JSON as a "
        "valid scale. bin/ is read-only for this task. This case asserts the CONTRACT, "
        "so it xfails today and turns green -- loudly, because strict=True makes an "
        "unexpected pass a failure -- the moment the guard is fixed."
    ),
)
@pytest.mark.parametrize("raw", ["nan", "inf"])
def test_parse_pixel_size_rejects_nan_and_inf(raw):
    with pytest.raises(ValueError):
        preflight_scale._parse_pixel_size(raw)


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
    # `assert A if cond else B` parses as `assert (A if cond else B)`, which reads
    # as a precedence bug even when it is not one. Two plain asserts instead.
    if out.exists():
        assert "blank.ome.tiff" in caplog.text + out.read_text()
    else:
        assert "blank.ome.tiff" in caplog.text


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
