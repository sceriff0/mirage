"""keep_results.sh: collect + draft plots + a verified copy of the records, then slim.

Run end to end on the stand-in trees of test_slim.py, with the `python` on PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.tests.regbench import test_slim as t

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks" / "regbench" / "keep_results.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGBENCH_")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    return env


def _run(*args):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], cwd=REPO, env=_env(),
                          capture_output=True, text=True)


def _trees(tmp_path):
    out, arms = tmp_path / "out", tmp_path / "arms"
    keep_images, images = t.regbench_tree(out)
    arms_keep, arms_gone, arms_images = t.arms_tree(arms)
    t._old(out)
    t._old(arms)
    return out, arms, keep_images, images, arms_keep, arms_gone + arms_images


def test_dry_run_deletes_nothing_and_keeps_tables_figures_and_records(tmp_path):
    out, arms, keep_images, images, arms_keep, arms_gone = _trees(tmp_path)
    keep = tmp_path / "keep"
    r = _run("--to", keep, "--regbench", out, "--arms", arms)
    assert r.returncode == 0, r.stdout + r.stderr
    for p in [*sum(images.values(), []), *arms_gone]:
        assert p.exists(), f"a dry run deleted {p}"
    assert any((keep / "tables").glob("*.csv"))
    assert (keep / "figures").is_dir()
    assert (keep / "raw" / "arms" / "tiled_high_s64" / "P1" / "qc" / "registration"
            / "P1_a_seg_qc.json").is_file()
    assert (keep / "raw" / "out" / "stare" / "multiplex" / "P1_r1" / "tissue"
            / "tissue.json").is_file()
    assert "nothing was deleted" in r.stdout
    assert (keep / "README.txt").is_file() and (keep / "keep_results.log").is_file()


def test_apply_frees_images_keeps_regbench_rescorable_and_arms_records_in_the_copy(tmp_path):
    out, arms, keep_images, images, arms_keep, arms_gone = _trees(tmp_path)
    keep = tmp_path / "keep"
    r = _run("--to", keep, "--regbench", out, "--arms", arms, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    for p in [*sum(images.values(), []), *arms_gone]:
        assert not p.exists(), f"--apply left {p}"
    # regbench at level images: the maps and records that re-scoring needs stay in place
    for p in sum(keep_images.values(), []):
        assert p.exists(), f"--apply removed {p}, which re-scoring needs"
    for p in arms_keep:
        assert p.exists()
    assert (out / "SLIMMED.txt").is_file() and (arms / "SLIMMED.txt").is_file()
    # running it again after --apply still verifies and exits 0
    again = _run("--to", keep, "--regbench", out, "--arms", arms)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "0 missing" in again.stdout


def test_a_keep_folder_inside_a_root_is_refused_before_anything(tmp_path):
    out, *_ = _trees(tmp_path)
    r = _run("--to", out / "keep", "--regbench", out)
    assert r.returncode != 0
    assert "inside" in r.stderr
    assert not (out / "keep" / "tables").exists()


def test_an_incomplete_copy_stops_before_slim(tmp_path):
    """A tar that copies nothing: the check finds the records missing and nothing is deleted."""
    out, _, _, images, *_ = _trees(tmp_path)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "tar").write_text("#!/bin/sh\ncat >/dev/null 2>&1\nexit 0\n")
    (fake / "tar").chmod(0o755)
    env = _env()
    env["PATH"] = f"{fake}{os.pathsep}{env['PATH']}"
    r = subprocess.run(["bash", str(SCRIPT), "--to", str(tmp_path / "keep"), "--regbench",
                        str(out), "--apply"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "incomplete" in r.stderr
    assert "incomplete" in (tmp_path / "keep" / "keep_results.log").read_text()
    for p in sum(images.values(), []):
        assert p.exists(), f"{p} was deleted after a failed copy"
