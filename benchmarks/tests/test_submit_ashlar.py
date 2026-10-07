"""submit_ashlar.sh: one ASHLAR arm on its own, with the launcher's arguments and marker."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "submit_ashlar.sh"


def _site(tmp_path, rc=0):
    """A results root with the two upstream arms, cached images, and a stand-in arm
    script that records what it was called with."""
    results = tmp_path / "arm_results"
    (results / "preprocess_shared" / "csv").mkdir(parents=True)
    (results / "preprocess_shared" / "csv" / "preprocessed.csv").write_text("x\n")
    (results / "valis_high_micro2").mkdir()
    images = tmp_path / "images"
    images.mkdir()
    for name in (
        "labsyspharm-ashlar-1.20.0.img",
        "bolt3x-mirage-stare-1.0.0.img",
        "bolt3x-mirage-regqc-1.0.0.img",
    ):
        (images / name).write_text("img")
    src = tmp_path / "src" / "benchmarks"
    src.mkdir(parents=True)
    for script in ("run_ashlar_arm.sh", "run_ashlar_original_arm.sh"):
        arm = src / script
        arm.write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$#" "$@" > "{tmp_path}/args"\n'
            f'printf "%s\\n" "$ASHLAR_EXEC" "$QC_EXEC" "$ASHLAR_MAX_DISCARD" "$ASHLAR_REG_QC" '
            f'> "{tmp_path}/env"\n'
            f'printf "%s\\n" "{script}" "$ASHLAR_STAGE_JITTER_UM" "$ASHLAR_NOISE_FRAC" '
            f'> "{tmp_path}/which"\n'
            f"exit {rc}\n"
        )
        arm.chmod(0o755)
    env = {
        **os.environ,
        "RESULTS": str(results),
        "IMAGES": str(images),
        "SRC_DIR": str(tmp_path / "src"),
    }
    for k in ("ARM", "SHIFT", "TILE", "MODE", "ASHLAR_EXEC", "QC_EXEC", "REGQC_EXEC"):
        env.pop(k, None)
    return results, env


def test_the_arm_gets_all_seven_arguments_and_the_finished_marker(tmp_path):
    results, env = _site(tmp_path)
    run = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, "SHIFT": "240"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    n, *args = (tmp_path / "args").read_text().splitlines()
    assert n == "7"  # a missing one is "7: maximum shift (um)" in run_ashlar_arm.sh
    assert args == [
        str(results),
        "ashlar_t1024_s240",
        "valis_high_micro2",
        str(results / "preprocess_shared" / "csv" / "preprocessed.csv"),
        "1024",
        "0.1",
        "240",
    ]
    solve, others, discard, reg_qc = (tmp_path / "env").read_text().splitlines()
    assert solve.endswith("labsyspharm-ashlar-1.20.0.img") and "--bind /beegfs" in solve
    assert others.endswith("bolt3x-mirage-stare-1.0.0.img")
    assert (discard, reg_qc) == ("1", "0")
    assert (results / "ashlar_t1024_s240" / ".external_done").is_file()
    # finished: a second submission does nothing
    (tmp_path / "args").unlink()
    again = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, "SHIFT": "240"},
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0 and "already finished" in again.stdout
    assert not (tmp_path / "args").exists()


def test_a_failed_arm_leaves_no_marker_and_a_missing_upstream_is_refused(tmp_path):
    results, env = _site(tmp_path, rc=3)
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert run.returncode == 1 and "FAILED" in run.stderr
    assert not (results / "ashlar_t1024_s15" / ".external_done").exists()
    (results / "preprocess_shared" / "csv" / "preprocessed.csv").unlink()
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert run.returncode == 1 and "has not finished" in run.stderr


def test_mode_original_runs_the_published_ashlar_arm_under_its_own_name(tmp_path):
    """MODE=original dispatches to run_ashlar_original_arm.sh, names the arm ashlar_orig_*
    (so it never overwrites the layer-only arm) and hands on the synthetic-tile settings."""
    results, env = _site(tmp_path)
    run = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, "MODE": "original", "JITTER_UM": "3"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    script, jitter, noise = (tmp_path / "which").read_text().splitlines()
    assert (script, jitter, noise) == ("run_ashlar_original_arm.sh", "3", "0.01")
    n, *args = (tmp_path / "args").read_text().splitlines()
    assert n == "7" and args[1] == "ashlar_orig_t1024_s15"
    assert (results / "ashlar_orig_t1024_s15" / ".external_done").is_file()
    assert not (results / "ashlar_t1024_s15").exists()
    bad = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, "MODE": "full"},
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1 and "not layer or original" in bad.stderr
