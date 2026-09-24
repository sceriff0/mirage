"""draw_mosaics.sh: redraw ONLY the mosaics of a figures run that already exists.

RENDER_EXEC is replaced by a fake that logs its command and exits 0, so what is under test is
the shell's own arithmetic and argument order: which sizes are drawn, when the drawn cell is
capped, where a concentric size series lands, and that the arm directories reach argparse
before the --label flags.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
ARMS = ("valis_high_micro2", "stare_high", "ashlar_t1024_s15")


def _run(tmp_path, arms=ARMS, rounds=10, **env_over):
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        csv = root / arm / "csv"
        csv.mkdir(parents=True)
        rows = ["patient_id,id,registered_image,is_reference,channels,pixel_size"]
        rows.append("046,046_ref,/dev/null,true,DAPI|PANCK,0.325")
        rows += [f"046,046_r{i},/dev/null,false,DAPI|M{i},0.325" for i in range(rounds)]
        (csv / "registered.csv").write_text("\n".join(rows) + "\n")

    log = tmp_path / "calls.log"
    fake = tmp_path / "fake_render"
    fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> {log}\nexit 0\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    proc = subprocess.run(
        ["bash", str(BENCH / "draw_mosaics.sh")],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "RENDER_EXEC": str(fake),
            **{k: str(v) for k, v in env_over.items()},
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    return proc, (log.read_text() if log.exists() else "")


def _flag(line, name):
    args = shlex.split(line)
    return args[args.index(name) + 1] if name in args else None


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("mosaics"))


# --- the size series ---------------------------------------------------------------
def test_every_patch_size_and_kind_is_drawn(default_run):
    proc, calls = default_run
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in calls.splitlines() if "reg_mosaic" in ln]
    assert len(lines) == 4 * 2  # 200/500/1000/2000 um x overlay/checker
    assert {_flag(ln, "--patch-um") for ln in lines} == {"200", "500", "1000", "2000"}
    assert {_flag(ln, "--kinds") for ln in lines} == {"overlay", "checker"}


def test_a_cell_larger_than_the_cap_is_drawn_smaller_not_at_full_size(default_run):
    """At 0.325 um/px a 2000 um cell is 6154 px, so a 10-round x 4-arm grid at 1:1 would be
    1.5 gigapixels -- which matplotlib does not finish. --cell-in caps what is DRAWN."""
    _proc, calls = default_run
    by_patch = {
        _flag(ln, "--patch-um"): ln for ln in calls.splitlines() if "reg_mosaic" in ln
    }
    assert _flag(by_patch["200"], "--cell-in") is None  # 615 px, under the 1600 cap
    assert _flag(by_patch["500"], "--cell-in") is None  # 1538 px, still under
    assert float(_flag(by_patch["1000"], "--cell-in")) == pytest.approx(
        16.0
    )  # 1600/100
    assert float(_flag(by_patch["2000"], "--cell-in")) == pytest.approx(16.0)


def test_the_cap_is_settable(tmp_path):
    _proc, calls = _run(
        tmp_path, PATCHES="500", KINDS="overlay", MAX_CELL_PX=800, DPI=200
    )
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert float(_flag(line, "--cell-in")) == pytest.approx(4.0)  # 800 / 200


# --- a size series is only a zoom series if it shares a centre ----------------------
def test_roi_center_draws_every_size_concentric(tmp_path):
    """--roi is the TOP-LEFT: holding it fixed slides the view as the patch grows, which is
    a pan, not a zoom."""
    _proc, calls = _run(
        tmp_path, PATCHES="200 500 1000 2000", KINDS="overlay", ROI_CENTER="24871,9163"
    )
    seen = {}
    for ln in calls.splitlines():
        if "reg_mosaic" not in ln:
            continue
        y, x = (int(v) for v in _flag(ln, "--roi").split(","))
        px = round(float(_flag(ln, "--patch-um")) / 0.325)
        seen[px] = (y + px // 2, x + px // 2)  # the centre each one actually covers
    assert len(seen) == 4
    for cy, cx in seen.values():
        assert abs(cy - 24871) <= 1 and abs(cx - 9163) <= 1


def test_a_plain_roi_is_passed_through_unchanged(tmp_path):
    _proc, calls = _run(tmp_path, PATCHES="500", KINDS="overlay", ROI="100,200")
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert _flag(line, "--roi") == "100,200"


def test_no_roi_means_each_size_picks_its_own(default_run):
    _proc, calls = default_run
    assert all("--roi" not in ln for ln in calls.splitlines() if "reg_mosaic" in ln)


# --- the arms ----------------------------------------------------------------------
def test_directories_come_before_the_label_flags(default_run):
    """ARM_DIR is nargs="+": argparse stops collecting positionals at the first flag, so an
    interleaved dir/--label/dir list makes every later directory 'unrecognized' (job
    6874795)."""
    _proc, calls = default_run
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    args = shlex.split(line)
    dirs = [i for i, a in enumerate(args) if a.endswith(ARMS)]
    first_flag = next(i for i, a in enumerate(args) if a.startswith("--"))
    assert len(dirs) == 3 and max(dirs) < first_flag


def test_an_arm_without_registered_slides_is_left_out_not_fatal(tmp_path):
    """At ALLOW_MISSING=0 -- the published-figure setting -- the arm with nothing is dropped
    and the run still succeeds on the arms that did register. Losing the whole mosaic to one
    failed arm is the outcome this guards against; drawing the gap (the default) is the
    other answer to the same problem, covered below."""
    proc, calls = _run(
        tmp_path,
        arms=("valis_high_micro2", "stare_high"),
        KINDS="overlay",
        ALLOW_MISSING=0,
    )
    assert proc.returncode == 0
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert "stare_high" in line and "ashlar_t1024_s15" not in line


def test_no_arm_at_all_is_refused_before_anything_runs(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    proc = subprocess.run(
        ["bash", str(BENCH / "draw_mosaics.sh")],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "RENDER_EXEC": "/bin/true",
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    # and it is refused for the RIGHT reason: at the default ALLOW_MISSING=1 every arm is a
    # column whether or not it has data, so the count of columns can never reach zero. An
    # empty ROOT used to fall past this line and die on "could not count the moving rounds".
    assert proc.returncode != 0 and "no arm under" in proc.stderr
    assert "could not count" not in proc.stderr


# --- rows ---------------------------------------------------------------------------
def test_rows_are_counted_off_the_checkpoint(tmp_path):
    """reg_mosaic defaults --rows to every round, but this script passes it explicitly so it
    works on a checkout from before that default existed."""
    _proc, calls = _run(tmp_path, rounds=7, PATCHES="200", KINDS="overlay")
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert _flag(line, "--rows") == "7"


def test_an_explicit_rows_wins(tmp_path):
    _proc, calls = _run(tmp_path, rounds=7, PATCHES="200", KINDS="overlay", ROWS=3)
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert _flag(line, "--rows") == "3"


def test_a_failed_size_is_named_and_the_others_still_draw(tmp_path):
    root = tmp_path / "root"
    for arm in ARMS:
        (root / arm / "csv").mkdir(parents=True)
        (root / arm / "csv" / "registered.csv").write_text(
            "patient_id,id,registered_image,is_reference,channels,pixel_size\n"
            "046,046_ref,/dev/null,true,DAPI,0.325\n"
            "046,046_r0,/dev/null,false,DAPI|M0,0.325\n"
        )
    log = tmp_path / "calls.log"
    fake = tmp_path / "fake"
    fake.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {log}\n'
        'case "$*" in *"--patch-um 200"*) exit 3 ;; esac\nexit 0\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        ["bash", str(BENCH / "draw_mosaics.sh")],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "RENDER_EXEC": str(fake),
            "PATCHES": "200 500",
            "KINDS": "overlay",
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert "FAILED: patch 200um" in proc.stderr
    assert "--patch-um 500" in log.read_text()
    assert "1 mosaic(s)" in proc.stdout and "(1 failed)" in proc.stdout
    assert proc.returncode != 0


def test_an_arm_with_no_checkpoint_is_passed_through_BY_DEFAULT(tmp_path):
    """Filtering it here would hide the gap; reg_mosaic draws it as a labelled empty column,
    which is what a proof of concept drawn mid-benchmark needs. No env var is needed for it:
    a set is drawn many times while arms are still running and once at the end, so the many
    is the default and the once is the opt-out."""
    _proc, calls = _run(
        tmp_path, arms=("valis_high_micro2",), PATCHES="200", KINDS="overlay"
    )
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert "ashlar_t1024_s15" in line  # the arm with nothing is still a column
    # reg_mosaic's own default matches, so nothing is passed either way round
    assert "allow-missing-arms" not in line

    _proc, calls = _run(
        tmp_path / "strict",
        arms=("valis_high_micro2",),
        PATCHES="200",
        KINDS="overlay",
        ALLOW_MISSING=0,
    )
    line = next(ln for ln in calls.splitlines() if "reg_mosaic" in ln)
    assert "--no-allow-missing-arms" in line
    assert "ashlar_t1024_s15" not in line
