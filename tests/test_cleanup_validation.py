"""Cleanup interacts with three things that must fail at LAUNCH, not halfway.

Each case is a `--dry_run` launch check with an actionable message.

They shell out to `nextflow` and skip where it is not installed (the pytest CI job).
The static half of the contract is tests/test_cleanup_vocabulary.py.
"""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("nextflow") is None,
    reason="nextflow is not on PATH; the static half runs in tests/test_cleanup_vocabulary.py",
)


def _launch(*args, outdir):
    # `-params-file params/dry_run.json`: Nextflow 26 passes every CLI --param as a
    # String, so a boolean on the command line fails schema validation.
    # --outdir is redirected so the test leaves no tree in the repo root.
    return subprocess.run(
        [
            # No `-q`: Nextflow's quiet mode suppresses log.warn from stdout
            # (it still reaches .nextflow.log), and two of the cases below assert on
            # a WARNING rather than a failure.
            "nextflow",
            "run",
            ".",
            "-profile",
            "test",
            "-stub",
            "-params-file",
            "params/dry_run.json",
            "--outdir",
            str(outdir),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_add_cycle_at_a_cleaning_level_is_refused(tmp_path):
    """mode=add_cycle must PUBLISH registered/ and segmentation/ for the NEXT
    cycle to read; at --cleanup_level=final it would not.

    --prior_outdir points at nothing: the cleanup rule must fire before the
    prior-run checks, so the message names the flag that caused it."""
    r = _launch(
        "--mode",
        "add_cycle",
        "--cleanup_level",
        "final",
        "--prior_outdir",
        "/nonexistent-prior-outdir",
        outdir=tmp_path,
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "cleanup_level" in out, out
    assert "add_cycle" in out, out


def test_start_past_preprocessing_at_a_cleaning_level_says_something(tmp_path):
    """--start segmentation re-enters from registered/. This run reads the PRIOR
    run's tree fine, so it is a warning rather than a refusal -- but it must
    name the flag, because this run's own output cannot be re-entered."""
    r = _launch("--start", "segmentation", "--cleanup_level", "final", outdir=tmp_path)
    assert "cleanup_level" in (r.stdout + r.stderr)


def test_an_unknown_level_names_the_valid_set(tmp_path):
    r = _launch("--cleanup_level", "aggressive", outdir=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "none" in out and "final" in out, out


def test_the_default_path_still_launches(tmp_path):
    """The check that keeps the three above honest: if --dry_run refused
    everything, each of them would pass while testing nothing."""
    r = _launch(outdir=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
