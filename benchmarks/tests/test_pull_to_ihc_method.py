"""pull_to_ihc_method.sh, run end to end against throwaway trees.

Two defects this pins:
  * `--anhir <dir>` was documented in the usage block but never parsed, so the
    documented command exited "unknown option" and the ANHIR tables never arrived.
  * a second arm experiment (the DRAPE arms, run into their own results root)
    REPLACED data/registration_arms/arms.csv, so the first root's arms lost their
    labels. `--append-arms` merges the manifest by arm_dir instead.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "pull_to_ihc_method.sh"

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")


def _arm_root(root: Path, arms: dict[str, str]) -> Path:
    for arm in arms:
        qc = root / arm / "P1" / "qc" / "registration"
        qc.mkdir(parents=True)
        (qc / "P1_seg_qc.json").write_text("{}")
    with open(root / "arms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_dir", "backend", "label"])
        w.writeheader()
        for arm, label in arms.items():
            w.writerow({"arm_dir": arm, "backend": "tiled", "label": label})
    return root


def _ihc(tmp_path: Path) -> Path:
    ihc = tmp_path / "ihc_method"
    ihc.mkdir()
    (ihc / "_workflowr.yml").write_text("")
    return ihc


def _run(*args, handoff: Path):
    return subprocess.run(
        ["bash", str(SCRIPT), *map(str, args), "--handoff", str(handoff)],
        capture_output=True,
        text=True,
        check=False,
    )


def _labels(ihc: Path) -> dict[str, str]:
    with open(ihc / "data" / "registration_arms" / "arms.csv", newline="") as f:
        return {r["arm_dir"]: r["label"] for r in csv.DictReader(f)}


def test_anhir_option_is_parsed_and_copies_the_tables(tmp_path):
    ihc = _ihc(tmp_path)
    root = _arm_root(tmp_path / "arms", {"tiled_high_s128": "DRAPE high"})
    anhir = tmp_path / "tables"
    anhir.mkdir()
    for name in ("anhir_cases.csv", "anhir_aggregates.csv", "anhir_missing.csv"):
        (anhir / name).write_text("x\n1\n")

    r = _run(root, ihc, "--anhir", anhir, handoff=tmp_path / "h")

    assert r.returncode == 0, r.stderr
    for name in ("anhir_cases.csv", "anhir_aggregates.csv", "anhir_missing.csv"):
        assert (ihc / "data" / "benchmark" / name).is_file()


def test_default_hand_off_replaces_the_manifest(tmp_path):
    ihc = _ihc(tmp_path)
    first = _arm_root(tmp_path / "a", {"tiled_high_gate2": "STARE high"})
    second = _arm_root(tmp_path / "b", {"tiled_high_s128": "DRAPE high"})
    assert _run(first, ihc, handoff=tmp_path / "h").returncode == 0
    assert _run(second, ihc, handoff=tmp_path / "h").returncode == 0
    assert _labels(ihc) == {"tiled_high_s128": "DRAPE high"}


def test_append_arms_merges_the_manifest_by_arm_dir(tmp_path):
    ihc = _ihc(tmp_path)
    first = _arm_root(
        tmp_path / "a",
        {"tiled_high_gate2": "STARE high", "valis_high_micro2": "VALIS old"},
    )
    second = _arm_root(
        tmp_path / "b",
        {"tiled_high_s128": "DRAPE high", "valis_high_micro2": "VALIS new"},
    )
    assert _run(first, ihc, handoff=tmp_path / "h").returncode == 0

    r = _run(second, ihc, "--append-arms", handoff=tmp_path / "h")

    assert r.returncode == 0, r.stderr
    assert _labels(ihc) == {
        "tiled_high_gate2": "STARE high",  # first root's arm keeps its label
        "tiled_high_s128": "DRAPE high",  # second root's arm is added
        "valis_high_micro2": "VALIS new",  # a clash: the root being added wins
    }
    arms = ihc / "data" / "registration_arms"
    assert (
        arms / "tiled_high_gate2" / "P1" / "qc" / "registration" / "P1_seg_qc.json"
    ).is_file()
    assert (
        arms / "tiled_high_s128" / "P1" / "qc" / "registration" / "P1_seg_qc.json"
    ).is_file()


def test_append_arms_does_not_overwrite_the_first_roots_arm_tables(tmp_path):
    ihc = _ihc(tmp_path)
    first = _arm_root(tmp_path / "a", {"tiled_high_gate2": "STARE high"})
    second = _arm_root(tmp_path / "b", {"tiled_high_s128": "DRAPE high"})
    h1, h2 = tmp_path / "h1", tmp_path / "h2"
    (h1 / "arms").mkdir(parents=True)
    (h1 / "arms" / "arm_summary.csv").write_text("first\n")
    (h2 / "arms").mkdir(parents=True)
    (h2 / "arms" / "arm_summary.csv").write_text("second\n")
    assert _run(first, ihc, handoff=h1).returncode == 0

    assert _run(second, ihc, "--append-arms", handoff=h2).returncode == 0

    assert (
        ihc / "data" / "registration_arms" / "arm_summary.csv"
    ).read_text() == "first\n"


def test_sweep_run_plan_is_handed_off(tmp_path):
    ihc = _ihc(tmp_path)
    root = _arm_root(tmp_path / "arms", {"tiled_high_s128": "DRAPE high"})
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (tmp_path / "sweep_plan.csv").write_text("run_id\nr1\n")

    r = _run(root, ihc, "--sweep", sweep, handoff=tmp_path / "h")

    assert r.returncode == 0, r.stderr
    assert (ihc / "data" / "benchmark" / "run_plan.csv").read_text() == "run_id\nr1\n"
