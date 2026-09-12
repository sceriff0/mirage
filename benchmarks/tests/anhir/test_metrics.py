import numpy as np
import pandas as pd
import pytest

from benchmarks.anhir import metrics


def _row(
    case_id, method, med, status="training", tissue="COAD", robust=1.0, scored=True
):
    return {
        "case_id": case_id,
        "tissue": tissue,
        "scale": "scale-25pc",
        "status": status,
        "source_image": "s",
        "target_image": "t",
        "method": method,
        "n_landmarks": 5,
        "scored": scored,
        "rtre_median": med if scored else np.nan,
        "rtre_mean": med if scored else np.nan,
        "rtre_max": 2 * med if scored else np.nan,
        "tre_median_px": 100 * med if scored else np.nan,
        "robustness": robust if scored else np.nan,
        "rank_median_rtre": np.nan,
        "time_min": 1.0,
    }


@pytest.fixture
def cases():
    rows = [
        _row(0, "a", 0.01),
        _row(0, "b", 0.02),
        _row(1, "a", 0.03),
        _row(1, "b", 0.01, robust=0.2),
        _row(2, "a", 0.05, tissue="kidney"),
        _row(2, "b", 0.05, tissue="kidney"),
        # an evaluation case: present, never scored, never ranked
        _row(3, "a", None, status="evaluation", scored=False),
        _row(3, "b", None, status="evaluation", scored=False),
    ]
    return pd.DataFrame(rows, columns=metrics.CASE_COLUMNS)


def test_ranks_are_per_case_with_ties_averaged_and_unscored_rows_unranked(cases):
    r = metrics.add_ranks(cases).set_index(["case_id", "method"])["rank_median_rtre"]
    assert r[(0, "a")] == 1 and r[(0, "b")] == 2
    assert r[(1, "a")] == 2 and r[(1, "b")] == 1
    assert r[(2, "a")] == 1.5 and r[(2, "b")] == 1.5
    assert np.isnan(r[(3, "a")]) and np.isnan(r[(3, "b")])


def test_aggregate_reduces_the_way_the_challenge_does(cases):
    agg = metrics.aggregate(cases).set_index(["method", "subset"])
    a_all = agg.loc[("a", "all")]
    assert a_all["n_cases"] == 3  # the evaluation row is not a scored case
    assert a_all["avg_median_rtre"] == pytest.approx(np.mean([0.01, 0.03, 0.05]))
    assert a_all["med_median_rtre"] == pytest.approx(0.03)
    assert a_all["avg_max_rtre"] == pytest.approx(np.mean([0.02, 0.06, 0.10]))
    assert a_all["avg_rank_median_rtre"] == pytest.approx(np.mean([1, 2, 1.5]))
    assert agg.loc[("b", "all")]["avg_rank_median_rtre"] == pytest.approx(
        np.mean([2, 1, 1.5])
    )
    # status breakdown: the evaluation subset exists but has no scored cases
    assert agg.loc[("a", "training")]["n_cases"] == 3
    assert agg.loc[("a", "evaluation")]["n_cases"] == 0
    assert np.isnan(agg.loc[("a", "evaluation")]["avg_median_rtre"])
    # robust subset drops b's case 1 (robustness 0.2) but keeps everything of a
    assert agg.loc[("a", "robust")]["n_cases"] == 3
    assert agg.loc[("b", "robust")]["n_cases"] == 2
    # tissue breakdown
    assert agg.loc[("a", "tissue:kidney")]["n_cases"] == 1
    assert agg.loc[("a", "tissue:COAD")]["avg_median_rtre"] == pytest.approx(0.02)
    assert list(agg.reset_index().columns) == metrics.AGGREGATE_COLUMNS


def test_aggregate_of_an_empty_frame_is_an_empty_table():
    empty = pd.DataFrame(columns=metrics.CASE_COLUMNS)
    agg = metrics.aggregate(empty)
    assert agg.empty and list(agg.columns) == metrics.AGGREGATE_COLUMNS
