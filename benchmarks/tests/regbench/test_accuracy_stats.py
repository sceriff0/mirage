"""The paper's accuracy statistics (benchmarks.regbench.paper.accuracy_stats) on tiny tables."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from benchmarks.regbench.paper import accuracy_stats as acc

CASES = ["10338", "046", "5456", "15370", "24086", "36766", "1111", "2222"]
PANELS = [f"DAPI_M{i}_N{i}" for i in range(8)] + [
    "DAPI_FSP1_L1CAM"
]  # 8 protocol + 1 extra


def _tables(tmp_path):
    rng = np.random.default_rng(0)
    base = {m: 0.5 + 0.02 * i for i, m in enumerate(acc.ROWS)}
    base["stare"] = 0.9
    rows = []
    for p in CASES:
        for pan in PANELS:
            for m in acc.ROWS:
                d = base[m] + rng.normal(0, 0.01)
                rows.append(
                    dict(
                        method=m,
                        case_id=f"{p}_{pan}",
                        patient_id=p,
                        panel=pan,
                        dice_matched=d,
                        dice_matched_mnn=d + 0.01,
                        dice_matched_exact=d - 0.01,
                        dice_matched_full_lsa=d,
                        displacement_readable=True,
                        displacement_um_p50=2 - d,
                        displacement_um_p90=3 - d,
                        pair_fraction=0.9,
                        pair_fraction_excess=0.5,
                    )
                )
    c = pd.DataFrame(rows)
    c.to_csv(tmp_path / "cells.csv", index=False)
    c.assign(dice_matched=c.dice_matched - 0.05).to_csv(
        tmp_path / "cells_nuclei.csv", index=False
    )
    pd.DataFrame(
        [
            dict(row=r, panel=p, discard_frac=0.6)
            for r in ("ashlar", "affine_ashlar")
            for p in PANELS
        ]
    ).to_csv(tmp_path / "ashlar_discard.csv", index=False)
    return tmp_path


def test_holm_matches_the_step_down_definition():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(prev, 2*0.02), max(prev, 1*0.04)
    np.testing.assert_allclose(acc.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert acc.holm([0.9, 0.8]).max() == 1.0


def test_prints_both_sets_with_the_protocol_extras_removed(tmp_path, capsys):
    acc.main([str(_tables(tmp_path))])
    out = capsys.readouterr().out
    assert "################ ALL 78" in out and "################ PROTOCOL 64" in out
    all_part, proto_part = out.split("################ PROTOCOL 64")
    # 8 cases x 9 panels = 72 rows per method on ALL; the FSP1/L1CAM panel removed leaves 64
    assert "  stare                         72 " in all_part
    assert "  stare                         64 " in proto_part
    # every reported row of Table S2 is printed
    for m in acc.ROWS:
        assert f"  {m:28s} " in proto_part


def test_dice_against_five_methods_centroid_against_four(tmp_path, capsys):
    acc.main([str(_tables(tmp_path))])
    proto = capsys.readouterr().out.split("################ PROTOCOL 64")[1]
    dice = [ln for ln in proto.splitlines() if ln.startswith("  dice_mat ")]
    cent = [ln for ln in proto.splitlines() if ln.startswith("  displace ")]
    assert len(dice) == 5 and any("Unregistered" in ln for ln in dice)
    assert len(cent) == 4 and not any("Unregistered" in ln for ln in cent)
    # STARE is better in every pair and case of the fixture: the smallest exact case-level
    # P with eight cases is 2/256 = 0.0078, x5 (Dice) and x4 (centroid) after Holm
    assert all("cases 8/8" in ln for ln in dice + cent)
    assert all("case Holm 0.039" in ln for ln in dice)
    assert all("case Holm 0.031" in ln for ln in cent)
