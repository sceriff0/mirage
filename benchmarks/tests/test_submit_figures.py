"""submit_figures.sh: does every planned figure reach the right tool, off the right run?

RENDER_EXEC is replaced by a fake that logs its command and exits 0, phase 1 is skipped
(SKIP_REGISTRATION=1) and the runs are pre-marked .done -- so what is under test is the
shell's own control flow: which tool each plan row invokes, which run directory it reads,
where it writes, and what it is told.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]

FULL = """
arms: [valis_high_micro2, stare_high]
reference_arm: valis_high_micro2
segmentation:
  methods: [stardist]
figures:
  mosaic:
    patch_um: [200]
    variants: 2
  overlay:
    field_um: [500, 2000]
    zoom_um: [60]
    variants: 3
  zoom:
    field_um: [150]
    masks: [both]
    crop: also
  crop:
    crop_px: [1024]
  channels:
    names: [DAPI, CD3]
    colors: [white, "#00e5ff"]
    field_um: [150]
    crop_px: [1024]
    autoscale: [clean]
options:
  roi: ["100,200"]
  patient: [P1]
  outline_color: "#ffd400"
  nuclei_color: "#00e5ff"
  outline_width: 2
  channel_label: DAPI
  autoscale: clean
  sat: 0.35
  bg_k: 3.0
  formats: png
  dpi: 50
"""


def _run(tmp_path, config_text, *, env_over=None):
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    # the runs phase 1 would have produced, and the segmentation phase 2 would
    for name in ("valis_high_micro2", "stare_high", "seg_stardist"):
        run = root / name
        (run / "csv").mkdir(parents=True)
        (run / "csv" / "registered.csv").write_text("patient_id\nP1\n")
        (run / "csv" / "segmented.csv").write_text("patient_id\nP1\n")
        (run / ".done").write_text("pre-marked by the test\n")
    config = tmp_path / "figures.yaml"
    config.write_text(config_text)
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    (tmp_path / "site.config").write_text("// test\n")

    log = tmp_path / "calls.log"
    fake = tmp_path / "fake_render"
    fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> {log}\nexit 0\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    proc = subprocess.run(
        ["bash", str(BENCH / "submit_figures.sh"), str(sheet)],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "CONFIG": str(config),
            "RENDER_EXEC": str(fake),
            "SITE_CONFIG": str(tmp_path / "site.config"),
            "CONDA_ENV": "",
            "SKIP_REGISTRATION": "1",
            **(env_over or {}),
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    return proc, (log.read_text() if log.exists() else ""), root


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("figs"), FULL)


def test_every_kind_of_figure_is_drawn(full):
    proc, calls, _ = full
    assert proc.returncode == 0, proc.stderr
    lines = calls.splitlines()
    assert sum("reg_overlay" in ln for ln in lines) == 4  # 2 arms x 2 fields
    assert sum("reg_zoom" in ln for ln in lines) == 2  # one zoom + one crop row
    assert sum("reg_crop" in ln for ln in lines) == 2  # two channels


def test_each_figure_reads_the_run_it_belongs_to(full):
    _proc, calls, root = full
    for ln in calls.splitlines():
        if "reg_overlay" in ln and "--title stare_high" in ln:
            assert str(root / "stare_high") in ln
        if "reg_zoom" in ln:
            assert str(root / "seg_stardist") in ln
        if (
            "reg_crop" in ln
        ):  # a channel crop reads the REFERENCE arm, not a segmentation
            assert str(root / "valis_high_micro2") in ln


def test_the_outputs_are_laid_out_by_what_varies(full):
    _proc, _calls, root = full
    assert (root / "overlay" / "valis_high_micro2" / "f500_z60").is_dir()
    assert (root / "overlay" / "stare_high" / "f2000_z60").is_dir()
    assert (root / "zoom" / "stardist" / "f150_both").is_dir()
    assert (root / "crops" / "stardist" / "f150_p1024_both").is_dir()
    assert (
        root / "crops" / "channels" / "f150_p1024_clean"
    ).is_dir()  # contrast mode too


def test_the_shared_options_reach_every_tool(full):
    _proc, calls, _ = full
    lines = [ln for ln in calls.splitlines() if "benchmarks." in ln]
    assert all(
        "--roi 100,200" in ln for ln in lines
    )  # one ROI: the sizes are comparable
    assert all("--patient P1" in ln and "--dpi 50" in ln for ln in lines)
    over = [ln for ln in lines if "reg_overlay" in ln]
    assert all("--variants 3" in ln and "--zoom-um 60" in ln for ln in over)
    zoom = [ln for ln in lines if "reg_zoom" in ln]
    assert any("--crop also" in ln for ln in zoom)
    assert any("--crop only --crop-px 1024" in ln for ln in zoom)
    assert all("--mask both" in ln and "--outline-color #ffd400" in ln for ln in zoom)
    chan = [ln for ln in lines if "reg_crop" in ln]
    assert any("--channel DAPI --colors white" in ln for ln in chan)
    assert any("--channel CD3 --colors #00e5ff" in ln for ln in chan)
    assert all("--autoscale clean" in ln and "--bg-k 3.0" in ln for ln in chan)


def test_a_zoom_um_of_zero_draws_no_inset(tmp_path):
    cfg = FULL.replace("zoom_um: [60]", "zoom_um: [0]")
    _proc, calls, _ = _run(tmp_path, cfg)
    over = [ln for ln in calls.splitlines() if "reg_overlay" in ln]
    assert over and not any("--zoom-um" in ln for ln in over)


def test_a_bad_config_fails_before_anything_is_drawn(tmp_path):
    proc, calls, _ = _run(tmp_path, "arms: [valis_low]\n")
    assert proc.returncode != 0 and calls == ""
    assert "unknown" in proc.stderr


def test_a_failing_render_is_named_and_the_rest_still_run(tmp_path):
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    for name in ("valis_high_micro2", "seg_stardist"):
        run = root / name
        (run / "csv").mkdir(parents=True)
        (run / "csv" / "registered.csv").write_text("patient_id\nP1\n")
        (run / "csv" / "segmented.csv").write_text("patient_id\nP1\n")
        (run / ".done").write_text("x\n")
    config = tmp_path / "figures.yaml"
    config.write_text(
        "arms: [valis_high_micro2]\nreference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: [stardist]\n"
        "figures:\n  overlay:\n    field_um: [500, 2000]\n"
    )
    log = tmp_path / "calls.log"
    fake = tmp_path / "fake"
    fake.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {log}\n'
        'case "$*" in *"--field-um 500"*) exit 3 ;; esac\nexit 0\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "site.config").write_text("// test\n")
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    proc = subprocess.run(
        ["bash", str(BENCH / "submit_figures.sh"), str(sheet)],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "CONFIG": str(config),
            "RENDER_EXEC": str(fake),
            "SITE_CONFIG": str(tmp_path / "site.config"),
            "CONDA_ENV": "",
            "SKIP_REGISTRATION": "1",
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert "FAILED: overlay overlay/valis_high_micro2/f500_z0" in proc.stderr
    assert "--field-um 2000" in log.read_text()  # the rest were still drawn
    assert "(1 failed)" in proc.stdout and proc.returncode != 0


def test_a_missing_reference_run_is_named_not_guessed(tmp_path):
    """Segmentation resumes from the reference arm: without it, say so rather than launching
    a pipeline that cannot find its input."""
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    config = tmp_path / "figures.yaml"
    config.write_text(
        "arms: [valis_high_micro2]\nreference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: [stardist]\n"
        "figures:\n  zoom:\n    field_um: [150]\n    masks: [cell]\n"
    )
    (tmp_path / "site.config").write_text("// test\n")
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    proc = subprocess.run(
        ["bash", str(BENCH / "submit_figures.sh"), str(sheet)],
        env={
            **os.environ,
            "ROOT": str(root),
            "SRC_DIR": str(BENCH.parent),
            "CONFIG": str(config),
            "RENDER_EXEC": "/bin/true",
            "SITE_CONFIG": str(tmp_path / "site.config"),
            "CONDA_ENV": "",
            "SKIP_REGISTRATION": "1",
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert proc.returncode != 0
    assert "did phase 1 run?" in proc.stderr


def test_an_existing_arm_is_read_where_it_lives_and_never_rebuilt(tmp_path):
    """{name, dir} points at a run you already have -- an arm of the arms benchmark, say.
    Nothing is re-registered, and the figures read it in place."""
    external = tmp_path / "elsewhere" / "valis_high_micro0"
    (external / "csv").mkdir(parents=True)
    (external / "csv" / "registered.csv").write_text("patient_id\nP1\n")
    cfg = (
        "arms:\n"
        "  - valis_high_micro2\n"
        f"  - {{name: arms_m0, dir: {external}}}\n"
        "reference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: []\n"
        "figures:\n  overlay:\n    field_um: [500]\n"
    )
    proc, calls, root = _run(tmp_path, cfg)
    assert proc.returncode == 0, proc.stderr
    assert "1 reused" in proc.stdout
    lines = [ln for ln in calls.splitlines() if "reg_overlay" in ln]
    assert any(str(external) in ln for ln in lines)  # read in place
    assert not (root / "arms_m0").exists()  # and not copied or rebuilt under the root
