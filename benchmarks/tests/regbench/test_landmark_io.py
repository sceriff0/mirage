"""Ground-truth landmarks (``regbench.landmark_io``): a sidecar beside ``case.json`` that
the driver moves through each method's map in the map's OWN direction, and whose sha256
a result is reused for only while it is unchanged."""

import json

import numpy as np
import pytest

from benchmarks.regbench import landmark_io, methods
from benchmarks.regbench.mapping import MOV_TO_REF, REF_TO_MOV, PointMap
from benchmarks.tests.regbench import scored_cohort

REF = np.array([[50.0, 60.0], [120.0, 80.0]])
D = np.array([3.0, -2.0])


def _with_truth(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    case = made[0]
    ref = np.array([[50.0, 60.0], [120.0, 80.0]])
    landmark_io.write_truth(case.dir, ref, ref + [3.0, -2.0])
    return root, case


def _result(out, variant, case):
    d = out / variant / case.dataset / case.case_id
    return np.load(d / "warped.npz"), json.loads((d / "run.json").read_text())


def _fwd(name="fwd"):
    return scored_cohort.fake(
        name,
        lambda c, w, o: iter(
            [(name, PointMap(lambda xy: xy - [3.0, -2.0], MOV_TO_REF), {})]
        ),
    )


def _bwd(name="bwd"):
    return scored_cohort.fake(
        name,
        lambda c, w, o: iter(
            [(name, PointMap(lambda xy: xy + [3.0, -2.0], REF_TO_MOV), {})]
        ),
    )


def test_mov_to_ref_moves_the_moving_landmarks(tmp_path):
    root, case = _with_truth(tmp_path)
    out = tmp_path / "o2"
    m = scored_cohort.fake(
        "fwd",
        lambda c, w, o: iter(
            [("fwd", PointMap(lambda xy: xy - [3.0, -2.0], MOV_TO_REF), {})]
        ),
    )
    methods.run_case(m, case, out)
    w = np.load(out / "fwd" / case.dataset / case.case_id / "warped.npz")
    assert str(w["truth_side"]) == "mov"
    np.testing.assert_allclose(w["truth_xy"], landmark_io.read_truth(case)["ref"])
    rec = json.loads(
        (out / "fwd" / case.dataset / case.case_id / "run.json").read_text()
    )
    assert rec["truth_sha256"] == landmark_io.truth_sha256(case)


def test_ref_to_mov_moves_reference_landmarks(tmp_path):
    root, case = _with_truth(tmp_path)
    out = tmp_path / "o3"
    m = scored_cohort.fake(
        "bwd",
        lambda c, w, o: iter(
            [("bwd", PointMap(lambda xy: xy + [3.0, -2.0], REF_TO_MOV), {})]
        ),
    )
    methods.run_case(m, case, out)
    w = np.load(out / "bwd" / case.dataset / case.case_id / "warped.npz")
    assert str(w["truth_side"]) == "ref"
    np.testing.assert_allclose(w["truth_xy"], landmark_io.read_truth(case)["mov"])


def test_a_case_without_landmarks_writes_none(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    w = np.load(out / "stare" / made[0].dataset / made[0].case_id / "warped.npz")
    assert "truth_xy" not in w.files
    assert "truth_side" not in w.files
    rec = json.loads(
        (out / "stare" / made[0].dataset / made[0].case_id / "run.json").read_text()
    )
    assert rec["truth_sha256"] is None
    assert rec["truth_note"] == landmark_io.NO_TRUTH


def test_a_written_truth_reads_back_and_its_sha_follows_the_bytes(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    case = type("C", (), {"dir": str(d)})()
    assert landmark_io.read_truth(case) is None
    assert landmark_io.truth_sha256(case) is None
    landmark_io.write_truth(d, REF, REF + D)
    got = landmark_io.read_truth(case)
    np.testing.assert_array_equal(got["ref"], REF)
    np.testing.assert_array_equal(got["mov"], REF + D)
    first = landmark_io.truth_sha256(case)
    assert isinstance(first, str) and len(first) == 64
    landmark_io.write_truth(d, REF, REF + 2 * D)
    assert landmark_io.truth_sha256(case) != first
    assert not [p.name for p in d.iterdir() if p.name != landmark_io.FILE]


def test_unequal_rows_and_non_finite_points_are_refused_with_the_path(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    with pytest.raises(ValueError, match="truth.npz"):
        landmark_io.write_truth(d, REF, (REF + D)[:1])
    with pytest.raises(ValueError, match="truth.npz"):
        landmark_io.write_truth(d, REF, np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="truth.npz"):
        landmark_io.write_truth(d, None, np.array([[1.0, 2.0, 3.0]]))
    assert not (d / landmark_io.FILE).exists()
    # a file written by other means is checked on reading too
    np.savez(d / landmark_io.FILE, ref=REF, mov=(REF + D)[:1])
    case = type("C", (), {"dir": str(d)})()
    with pytest.raises(ValueError, match=str(d / landmark_io.FILE)):
        landmark_io.read_truth(case)


def test_one_sided_truth_reads_with_no_reference_side(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    landmark_io.write_truth(d, None, REF + D)
    got = landmark_io.read_truth(type("C", (), {"dir": str(d)})())
    assert got["ref"] is None
    np.testing.assert_array_equal(got["mov"], REF + D)


def test_one_sided_truth_moves_under_mov_to_ref(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    case = made[0]
    landmark_io.write_truth(case.dir, None, REF + D)
    out = tmp_path / "o4"
    methods.run_case(_fwd(), case, out)
    w, rec = _result(out, "fwd", case)
    assert str(w["truth_side"]) == "mov"
    np.testing.assert_allclose(w["truth_xy"], REF)
    assert rec["truth_sha256"] == landmark_io.truth_sha256(case)
    assert rec["truth_note"] == ""


def test_one_sided_truth_under_ref_to_mov_writes_none_and_says_why(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    case = made[0]
    landmark_io.write_truth(case.dir, None, REF + D)
    out = tmp_path / "o5"
    methods.run_case(_bwd(), case, out)
    w, rec = _result(out, "bwd", case)
    assert "truth_xy" not in w.files and "truth_side" not in w.files
    assert rec["ok"] and rec["truth_sha256"] == landmark_io.truth_sha256(case)
    assert rec["truth_note"] == landmark_io.ONE_SIDED_REF_TO_MOV


def test_a_landmark_the_map_sends_nowhere_is_non_finite(tmp_path):
    root, case = _with_truth(tmp_path)
    out = tmp_path / "o6"

    def fn(xy):
        got = np.asarray(xy, dtype=float) - D
        got[1] = np.nan
        return got

    m = scored_cohort.fake(
        "gap", lambda c, w, o: iter([("gap", PointMap(fn, MOV_TO_REF), {})])
    )
    methods.run_case(m, case, out)
    w, rec = _result(out, "gap", case)
    assert rec["ok"]
    assert (
        np.isfinite(w["truth_xy"][0]).all() and not np.isfinite(w["truth_xy"][1]).any()
    )


def _counting(calls):
    def register(case, work, opts):
        calls.append(case.case_id)
        yield (
            "cnt",
            PointMap(lambda xy: np.asarray(xy, dtype=float) - D, MOV_TO_REF),
            {},
        )

    return scored_cohort.fake("cnt", register)


def test_a_changed_truth_is_not_reused(tmp_path):
    root, case = _with_truth(tmp_path)
    out, calls = tmp_path / "o7", []
    method = _counting(calls)
    assert methods.run_case(method, case, out, resume=True) == ["cnt"]
    assert methods.run_case(method, case, out, resume=True) == ["cnt"]
    assert calls == [case.case_id]  # unchanged truth: reused
    landmark_io.write_truth(case.dir, REF, REF + 2 * D)
    rec = _result(out, "cnt", case)[1]
    path = out / "cnt" / case.dataset / case.case_id
    assert "truth" in methods.record_gaps(rec, case, path, rec["reuse_key"])
    assert methods.run_case(method, case, out, resume=True) == ["cnt"]
    assert calls == [case.case_id, case.case_id]  # changed truth: run again
    w, rec = _result(out, "cnt", case)
    np.testing.assert_allclose(w["truth_xy"], REF + D)
    assert rec["truth_sha256"] == landmark_io.truth_sha256(case)
    # truth ADDED to a case that had none also invalidates
    other = scored_cohort.multiplex_case(tmp_path / "cases", "P9_r1", "P9", 9)
    assert methods.run_case(method, other, out, resume=True) == ["cnt"]
    landmark_io.write_truth(other.dir, REF, REF + D)
    assert methods.run_case(method, other, out, resume=True) == ["cnt"]
    assert calls.count("P9_r1") == 2


def test_a_record_from_before_truth_stays_reusable_for_a_case_without_truth(tmp_path):
    root, out, made = scored_cohort.build(tmp_path)
    case = made[1]
    out, calls = tmp_path / "o8", []
    method = _counting(calls)
    assert methods.run_case(method, case, out, resume=True) == ["cnt"]
    path = out / "cnt" / case.dataset / case.case_id / "run.json"
    rec = json.loads(path.read_text())
    for k in ("truth_sha256", "truth_note"):  # as written before this task
        rec.pop(k, None)
    path.write_text(json.dumps(rec))
    assert methods.record_gaps(rec, case, path.parent, rec["reuse_key"]) == []
    assert methods.run_case(method, case, out, resume=True) == ["cnt"]
    assert calls == [case.case_id]  # reused, nothing run


def test_the_sidecar_gets_the_umask_default_mode(tmp_path):
    """``mkstemp`` creates its file 0600 and ``os.replace`` keeps that: the sidecar must
    instead have the mode an ordinary new file gets (``0o666 & ~umask``), so another
    account of the group can read the case."""
    import os
    import stat

    d = tmp_path / "case"
    d.mkdir()
    old = os.umask(0o022)
    try:
        landmark_io.write_truth(d, REF, REF + D)
        assert stat.S_IMODE((d / landmark_io.FILE).stat().st_mode) == 0o644
        os.umask(0o002)
        landmark_io.write_truth(d, REF, REF + 2 * D)
        assert stat.S_IMODE((d / landmark_io.FILE).stat().st_mode) == 0o664
    finally:
        os.umask(old)


def _failing(name="bad"):
    def register(case, work, opts):
        raise RuntimeError("boom")
        yield  # a generator

    return scored_cohort.fake(name, register)


def test_a_failed_run_records_its_truth_note(tmp_path):
    """Every run.json carries ``truth_note``, a failure record too: ``NO_TRUTH`` when
    the case has no sidecar, else ``RUN_FAILED`` (nothing was moved because the method
    failed, not because the truth is missing)."""
    root, out, made = scored_cohort.build(tmp_path)
    with_truth, without = made[0], made[1]
    landmark_io.write_truth(with_truth.dir, REF, REF + D)
    out = tmp_path / "o9"
    for case, note in (
        (with_truth, landmark_io.RUN_FAILED),
        (without, landmark_io.NO_TRUTH),
    ):
        assert methods.run_case(_failing(), case, out) == []
        rec = json.loads(
            (out / "bad" / case.dataset / case.case_id / "run.json").read_text()
        )
        assert rec["ok"] is False and rec["cause"] == "exception"
        assert rec["truth_sha256"] == landmark_io.truth_sha256(case)
        assert rec["truth_note"] == note


def test_the_umask_is_read_without_being_set_where_linux_reports_it(
    tmp_path, monkeypatch
):
    """``os.umask`` is process-wide: setting it to read it races any other thread
    that creates a file. Where ``/proc/self/status`` carries ``Umask:`` it is read
    from there and ``os.umask`` is never called; elsewhere the round-trip is the
    fallback."""
    import os

    status = tmp_path / "status"
    status.write_text("Name:\tpython\nUmask:\t0027\nState:\tR (running)\n")
    monkeypatch.setattr(landmark_io, "_PROC_STATUS", str(status))

    def set_umask(mask):
        raise AssertionError("os.umask was called")

    monkeypatch.setattr(landmark_io.os, "umask", set_umask)
    assert landmark_io._umask_mode() == 0o640
    monkeypatch.undo()
    monkeypatch.setattr(landmark_io, "_PROC_STATUS", str(tmp_path / "absent"))
    old = os.umask(0o022)
    try:
        assert landmark_io._umask_mode() == 0o644
        assert os.umask(0o022) == 0o022  # set back
    finally:
        os.umask(old)
