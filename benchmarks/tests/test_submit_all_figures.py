"""submit_all_figures.sh + placeholder_card.py: one job for every figure, placeholders opt-in.

What is pinned:
  * the stage plan (DRY_RUN=1): each stage runs only with its inputs, and says SKIPPED otherwise;
  * placeholder mode reaches make_figures and ihc_method, and never the hand-off;
  * the hand-off is ONE pass (arms + sweep + ANHIR) from the one results root;
  * a composite card is drawn only for a slot with no rendered file, is named PLACEHOLDER_*,
    is listed in PLACEHOLDER_COMPOSITES.csv, and `clear` removes every card and nothing else.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
SCRIPT = BENCH / "submit_all_figures.sh"
sys_path_card = BENCH / "placeholder_card.py"


def _dry(tmp_path: Path, **env) -> str:
    e = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PLACEHOLDER_MISSING", "STAGES")
    }
    e.update(
        {"DRY_RUN": "1", "OUT": str(tmp_path / "out"), "SRC_DIR": str(BENCH.parent)}
    )
    e.update(env)
    r = subprocess.run(
        ["bash", str(SCRIPT)], env=e, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _full(tmp_path: Path) -> dict:
    (tmp_path / "ihc").mkdir(exist_ok=True)
    return {**FULL, "IHC": str(tmp_path / "ihc")}


FULL = {
    "ARMS_RESULTS": "/b/arm_results",
    "ARMS_PLAN": "/b/arm_plan.csv",
    "SWEEP_RESULTS": "/b/bench_results",
    "SWEEP_PLAN": "/b/bench_run_plan.csv",
    "INPUT": "/in/input.csv",
    "CONFIG": "figures.yaml",
    "ANHIR_DIR": "/anhir",
    "IHC": "/ihc",
    "IHC_BUILD": "1",
}


def test_nothing_given_skips_every_stage_with_a_reason(tmp_path):
    out = _dry(tmp_path)
    for line in (
        "stats/arms: SKIPPED",
        "stats/sweep: SKIPPED",
        "composites: SKIPPED",
        "anhir: SKIPPED",
        "handoff: SKIPPED",
        "ihc: SKIPPED",
    ):
        assert line in out, out
    assert "make_figures" not in out


def test_default_mode_is_real_and_writes_no_placeholders(tmp_path):
    out = _dry(tmp_path, **_full(tmp_path))
    assert "REAL (placeholders off)" in out
    assert "--placeholder-missing" not in out
    assert "placeholder_card.py fill" not in out
    assert "IHC_PLACEHOLDER_MISSING=0" in out
    assert "/stats/arms" in out and "/stats_preview/" not in out
    # the card sweep runs in BOTH modes, so a real run deletes stale cards
    assert "placeholder_card.py clear" in out


def test_placeholder_mode_reaches_stats_composites_and_ihc_but_not_the_hand_off(
    tmp_path,
):
    out = _dry(tmp_path, PLACEHOLDER_MISSING="1", **_full(tmp_path))
    lines = out.splitlines()
    stats = [ln for ln in lines if "benchmarks.analysis.make_figures" in ln]
    assert len(stats) == 2 and all("--placeholder-missing" in ln for ln in stats)
    assert all("/stats_preview/" in ln for ln in stats)
    assert any("placeholder_card.py fill" in ln for ln in lines)
    assert "IHC_PLACEHOLDER_MISSING=1" in out
    hand = [ln for ln in lines if "pull_to_ihc_method.sh" in ln]
    assert len(hand) == 1
    assert all(
        "--placeholder" not in ln and "PLACEHOLDER_MISSING=1" not in ln for ln in hand
    )


def test_hand_off_is_one_pass_carrying_arms_sweep_and_anhir(tmp_path):
    out = _dry(tmp_path, **_full(tmp_path))
    hand = [ln for ln in out.splitlines() if "pull_to_ihc_method.sh" in ln]
    assert len(hand) == 1, hand
    assert "/b/arm_results" in hand[0] and "--append-arms" not in hand[0]
    assert "--anhir /anhir/tables" in hand[0], "ANHIR rode on the removed second pass"


def test_rejects_a_non_boolean_placeholder_switch(tmp_path):
    e = dict(os.environ, DRY_RUN="1", OUT=str(tmp_path), PLACEHOLDER_MISSING="yes")
    r = subprocess.run(["bash", str(SCRIPT)], env=e, capture_output=True, text=True)
    assert r.returncode == 2


# ---- placeholder_card.py ---------------------------------------------------------------


def _card(*args):
    pytest.importorskip("matplotlib")
    return subprocess.run(
        ["python3", str(sys_path_card), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
    )


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "figure_plan.tsv"
    plan.write_text(
        "mosaic\tarms:all\tmosaic\t--x 1\n"
        "overlay\tarm:valis_high_micro2\toverlay/valis_high_micro2/f500_z60\t\n"
        "zoom\tseg:stardist\tzoom/stardist/f150_both\t\n"
    )
    return plan


def test_fill_cards_only_the_slots_that_did_not_render(tmp_path):
    root = tmp_path / "composites"
    (root / "mosaic").mkdir(parents=True)
    real = root / "mosaic" / "P1_mosaic.png"
    real.write_bytes(b"real")

    _card("fill", "--plan", _plan(tmp_path), "--root", root, "--formats", "png")

    assert real.read_bytes() == b"real"
    assert not list((root / "mosaic").glob("PLACEHOLDER_*"))
    assert (
        root / "overlay/valis_high_micro2/f500_z60/PLACEHOLDER_overlay_row2.png"
    ).is_file()
    assert (root / "zoom/stardist/f150_both/PLACEHOLDER_zoom_row3.png").is_file()
    with open(root / "PLACEHOLDER_COMPOSITES.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["kind"] for r in rows} == {"overlay", "zoom"}
    assert "2 composite slot(s)" in (root / "PLACEHOLDER_DATA.txt").read_text()


def test_a_card_does_not_count_as_a_render(tmp_path):
    root = tmp_path / "composites"
    plan = _plan(tmp_path)
    _card("fill", "--plan", plan, "--root", root, "--formats", "png")
    _card("fill", "--plan", plan, "--root", root, "--formats", "png")
    with open(root / "PLACEHOLDER_COMPOSITES.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_clear_removes_every_card_and_nothing_else(tmp_path):
    root = tmp_path / "composites"
    (root / "mosaic").mkdir(parents=True)
    real = root / "mosaic" / "P1_mosaic.png"
    real.write_bytes(b"real")
    _card("fill", "--plan", _plan(tmp_path), "--root", root, "--formats", "png,pdf")
    assert list(root.rglob("PLACEHOLDER_*"))

    _card("clear", "--root", root)

    assert not list(root.rglob("PLACEHOLDER_*"))
    assert real.read_bytes() == b"real"
