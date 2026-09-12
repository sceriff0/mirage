import numpy as np
import pytest

from benchmarks.anhir import landmarks as lm


def test_landmark_csv_roundtrip_keeps_the_challenge_format(tmp_path):
    xy = np.array([[1.5, 2.0], [3.0, 4.25]])
    p = lm.write_landmarks(tmp_path / "S1.csv", xy)
    assert p.read_text().splitlines()[0] == ",X,Y"
    np.testing.assert_allclose(lm.read_landmarks(p), xy)


def test_read_landmarks_refuses_a_file_without_xy(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing landmark column"):
        lm.read_landmarks(p)


def test_imagej_point_format_is_parsed(tmp_path):
    p = tmp_path / "warped_source_landmarks.txt"
    p.write_text("point\n2\n10.5 20.25\n30 40\n")
    np.testing.assert_allclose(lm.read_imagej_points(p), [[10.5, 20.25], [30, 40]])
    p.write_text("point\n3\n10 20\n")
    with pytest.raises(ValueError, match="declares 3"):
        lm.read_imagej_points(p)


def test_tre_is_euclidean_over_the_paired_prefix():
    warped = [[0, 0], [3, 4], [10, 10]]
    target = [[0, 0], [0, 0]]  # one fewer landmark: the extra warped point is ignored
    np.testing.assert_allclose(lm.tre(warped, target), [0.0, 5.0])


def test_rtre_divides_by_the_diagonal_and_rejects_a_bad_one():
    np.testing.assert_allclose(lm.rtre([5.0, 10.0], 100.0), [0.05, 0.10])
    with pytest.raises(ValueError):
        lm.rtre([1.0], 0)


def test_robustness_is_the_fraction_of_landmarks_that_improved():
    assert lm.robustness([0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.5, 0.1]) == pytest.approx(
        0.5
    )
    assert np.isnan(lm.robustness([], []))


def test_case_stats_reproduces_the_challenge_numbers_by_hand():
    target = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    source = target + [10.0, 0.0]  # initial TRE 10 px everywhere
    # warped: three landmarks fixed exactly, one made WORSE (moved 20 px away)
    warped = target.copy()
    warped[3] = target[3] + [20.0, 0.0]
    s = lm.case_stats(warped, target, source, diagonal=200.0)
    assert s["n_landmarks"] == 4 and s["complete"]
    assert s["rtre_median"] == pytest.approx(0.0)
    assert s["rtre_mean"] == pytest.approx(20.0 / 4 / 200.0)
    assert s["rtre_max"] == pytest.approx(0.1)
    assert s["tre_median_px"] == pytest.approx(0.0)
    assert s["robustness"] == pytest.approx(0.75)


def test_case_stats_flags_an_incomplete_warp_and_an_empty_one():
    target = np.zeros((3, 2))
    source = np.ones((3, 2))
    s = lm.case_stats(np.zeros((2, 2)), target, source, diagonal=10.0)
    assert s["n_landmarks"] == 2 and not s["complete"]
    e = lm.case_stats(np.zeros((0, 2)), target, source, diagonal=10.0)
    assert e["n_landmarks"] == 0 and np.isnan(e["rtre_median"])
