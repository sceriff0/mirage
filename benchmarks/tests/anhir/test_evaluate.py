"""evaluate.py end to end on the miniature download: score, rank, package."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from benchmarks.anhir import dataset as ds
from benchmarks.anhir import evaluate, prepare, warp
from benchmarks.anhir.metrics import AGGREGATE_COLUMNS, CASE_COLUMNS


@pytest.fixture
def warped(anhir_root, tmp_path):
    """Three warped dirs: the identity, the baseline, and a perfect method."""
    _, pairs = prepare.build_inputs(
        anhir_root["cases"], anhir_root["root"], tmp_path / "w", convert=False
    )
    dirs = {
        "initial": tmp_path / "initial",
        "bunwarpj": tmp_path / "bunwarpj",
        "perfect": tmp_path / "perfect",
    }
    warp.warp_all("initial", pairs, dirs["initial"])
    warp.warp_all(
        "bunwarpj", pairs, dirs["bunwarpj"], baseline_root=anhir_root["baseline"]
    )
    dirs["perfect"].mkdir()
    for c in anhir_root["cases"]:
        src = warp.read_landmarks(
            ds.resolve_landmarks(c, anhir_root["landmarks"], "source")
        )
        from benchmarks.tests.anhir.conftest import OFFSET

        warp.write_landmarks(dirs["perfect"] / f"{c.case_id}.csv", src - OFFSET)
    return pairs, dirs


def test_evaluate_scores_training_cases_and_leaves_evaluation_cases_unscored(
    anhir_root, warped
):
    _, dirs = warped
    cases_df, agg, missing = evaluate.evaluate(
        anhir_root["cases"], dirs, anhir_root["landmarks"]
    )
    assert list(cases_df.columns) == CASE_COLUMNS
    assert list(agg.columns) == AGGREGATE_COLUMNS
    assert missing.empty
    assert len(cases_df) == 9  # 3 cases x 3 methods
    by = cases_df.set_index(["case_id", "method"])
    # the identity has the fixture's known error; the perfect warp has none
    diag = anhir_root["cases"][0].diagonal
    assert by.loc[(0, "initial"), "rtre_median"] == pytest.approx(
        np.hypot(30, 20) / diag
    )
    assert by.loc[(0, "initial"), "robustness"] == pytest.approx(0.0)
    assert by.loc[(0, "perfect"), "rtre_median"] == pytest.approx(0.0)
    assert by.loc[(0, "perfect"), "robustness"] == pytest.approx(1.0)
    assert by.loc[(0, "bunwarpj"), "robustness"] == pytest.approx(1.0)
    assert by.loc[(0, "bunwarpj"), "time_min"] == pytest.approx(2.0)
    # ranking: perfect 1, bunwarpj 2, initial 3 on every scored case
    assert by.loc[(1, "perfect"), "rank_median_rtre"] == 1
    assert by.loc[(1, "bunwarpj"), "rank_median_rtre"] == 2
    assert by.loc[(1, "initial"), "rank_median_rtre"] == 3
    # the evaluation case is present but unscored
    ev = cases_df[cases_df["case_id"] == 2]
    assert (~ev["scored"]).all() and ev["rtre_median"].isna().all()
    assert (ev["n_landmarks"] == 8).all()
    a = agg.set_index(["method", "subset"])
    assert a.loc[("perfect", "all"), "n_cases"] == 2
    assert a.loc[("perfect", "all"), "avg_rank_median_rtre"] == pytest.approx(1.0)
    assert a.loc[("initial", "all"), "avg_rank_median_rtre"] == pytest.approx(3.0)


def test_a_missing_case_is_scored_at_the_initial_pose_and_ranked_last(
    anhir_root, warped
):
    """The challenge's rule: a missing registration counts as the initial rTRE."""
    _, dirs = warped
    (dirs["perfect"] / "1.csv").unlink()  # a training case the method dropped
    (dirs["perfect"] / "2.csv").unlink()  # an evaluation case: nothing to impute from
    cases_df, agg, missing = evaluate.evaluate(
        anhir_root["cases"], dirs, anhir_root["landmarks"]
    )
    assert missing.to_dict("records") == [
        {"case_id": 1, "method": "perfect", "reason": "no warped file"},
        {"case_id": 2, "method": "perfect", "reason": "no warped file"},
    ]
    by = cases_df.set_index(["case_id", "method"])
    imputed = by.loc[(1, "perfect")]
    assert bool(imputed["imputed_initial"]) and bool(imputed["scored"])
    assert imputed["rtre_median"] == pytest.approx(
        by.loc[(1, "initial"), "rtre_median"]
    )
    assert imputed["robustness"] == pytest.approx(0.0)
    assert imputed["rank_median_rtre"] == pytest.approx(2.5)  # tied last with initial
    assert (2, "perfect") not in by.index
    assert not by.loc[(0, "perfect"), "imputed_initial"]
    a = agg.set_index(["method", "subset"])
    assert a.loc[("perfect", "all"), "n_cases"] == 2  # the dropped case still counts
    assert a.loc[("perfect", "all"), "avg_rank_median_rtre"] == pytest.approx(
        (1 + 2.5) / 2
    )


def test_submission_package_is_the_cover_table_plus_landmark_files(
    anhir_root, warped, tmp_path
):
    _, dirs = warped
    cover = ds.load_cover(anhir_root["cover"])
    z = evaluate.write_submission(
        cover, anhir_root["cases"], dirs["bunwarpj"], tmp_path / "sub", "bunwarpj"
    )
    assert z.exists() and z.suffix == ".zip"
    names = set(zipfile.ZipFile(z).namelist())
    assert evaluate.SUBMISSION_COVER in names
    assert {"landmarks/0.csv", "landmarks/1.csv", "landmarks/2.csv"} <= names
    sub = pd.read_csv(
        tmp_path / "sub" / "bunwarpj" / evaluate.SUBMISSION_COVER, index_col=0
    )
    assert list(sub.index) == [0, 1, 2]
    assert sub[ds.COL_WARPED_SOURCE].tolist() == [
        "landmarks/0.csv",
        "landmarks/1.csv",
        "landmarks/2.csv",
    ]
    assert sub[ds.COL_TIME].tolist() == pytest.approx([2.0, 2.0, 2.0])
    # every original cover column survives, so the challenge's evaluator can merge on them
    assert set(ds.COVER_COLUMNS) <= set(sub.columns)


def test_cli_writes_the_three_tables_and_a_submission(anhir_root, warped, tmp_path):
    _, dirs = warped
    out = tmp_path / "tables"
    rc = evaluate.main(
        [
            "--dataset",
            str(anhir_root["cover"]),
            "--landmarks-root",
            str(anhir_root["landmarks"]),
            "--warped",
            f"initial={dirs['initial']}",
            "--warped",
            f"bunwarpj={dirs['bunwarpj']}",
            "--out",
            str(out),
            "--submit",
            "bunwarpj",
        ]
    )
    assert rc == 0
    for name in ("anhir_cases.csv", "anhir_aggregates.csv", "anhir_missing.csv"):
        assert (out / name).exists()
    assert (out / "submission" / "bunwarpj.zip").exists()
    with pytest.raises(SystemExit, match="no --warped"):
        evaluate.main(
            [
                "--dataset",
                str(anhir_root["cover"]),
                "--landmarks-root",
                str(anhir_root["landmarks"]),
                "--warped",
                f"initial={dirs['initial']}",
                "--out",
                str(out),
                "--submit",
                "valis",
            ]
        )


def test_reg_eval_table_is_what_the_analysis_loader_accepts(
    anhir_root, warped, tmp_path
):
    """The hook make_figures --reg-eval has carried since the first harness was
    deleted takes exactly this table; a schema drift on either side fails here."""
    from benchmarks.analysis.lib import load

    _, dirs = warped
    cases_df, _, _ = evaluate.evaluate(
        anhir_root["cases"], dirs, anhir_root["landmarks"]
    )
    table = evaluate.reg_eval_table(cases_df)
    assert len(table) == 6  # 2 scored cases x 3 methods
    assert set(table["mode"]) == {"initial", "bunwarpj", "perfect"}
    csv = tmp_path / "anhir_reg_eval.csv"
    table.to_csv(csv, index=False)
    joined = load.load_reg_eval(csv).set_index("registration_method")
    assert joined.loc["perfect", "gt_true_median_rtre"] == pytest.approx(0.0)
    assert joined.loc["initial", "gt_true_median_px"] == pytest.approx(np.hypot(30, 20))
    assert joined.loc["perfect", "gt_n_pairs"] == 2
