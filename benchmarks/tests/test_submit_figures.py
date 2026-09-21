"""submit_figures.sh: does every planned figure reach the right tool, off the right run?

RENDER_EXEC is replaced by a fake that logs its command and exits 0, phase 1 is skipped
(SKIP_REGISTRATION=1) and the runs are pre-marked .done -- so what is under test is the
shell's own control flow: which tool each plan row invokes, which run directory it reads,
where it writes, and what it is told.
"""

from __future__ import annotations

import json
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
    assert "(1 failed or skipped)" in proc.stdout and proc.returncode != 0


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


def test_a_backend_that_cannot_segment_costs_only_its_own_figures(tmp_path):
    """cellsam wants users.deepcell.org and the compute nodes have no outbound network
    (job 68633*). That must not throw away the overlays, the channel crops, or the figures
    of the methods that DID segment."""
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    for name in ("valis_high_micro2", "seg_stardist"):  # note: no seg_cellsam
        run = root / name
        (run / "csv").mkdir(parents=True)
        (run / "csv" / "registered.csv").write_text("patient_id\nP1\n")
        (run / "csv" / "segmented.csv").write_text("patient_id\nP1\n")
        (run / ".done").write_text("x\n")
    config = tmp_path / "figures.yaml"
    config.write_text(
        "arms: [valis_high_micro2]\nreference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: [stardist, cellsam]\n"
        "figures:\n"
        "  overlay:\n    field_um: [500]\n"
        "  zoom:\n    field_um: [150]\n    masks: [cell]\n"
        "  channels:\n    names: [DAPI]\n    field_um: [150]\n    crop_px: [512]\n"
    )
    (tmp_path / "site.config").write_text("// test\n")
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    log = tmp_path / "calls.log"
    fake = tmp_path / "fake"
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
            "PATH": "/usr/bin:/bin",  # no nextflow: cellsam's run cannot be made
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    calls = log.read_text() if log.exists() else ""
    assert "segmentation FAILED for: cellsam" in proc.stderr
    assert "SKIPPED" in proc.stderr and "cellsam" in proc.stderr
    # everything that did not depend on cellsam was still drawn
    assert "reg_overlay" in calls and "reg_crop" in calls
    assert str(root / "seg_stardist") in calls  # the method that segmented
    assert "seg_cellsam" not in calls
    assert proc.returncode != 0  # but the job still reports the failure


def test_segmentation_params_reach_the_pipeline(tmp_path):
    """segmentation.params is how a backend's assets are pinned -- cellsam_model_path above
    all, since the auto-download cannot work on a compute node."""
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    run = root / "valis_high_micro2"
    (run / "csv").mkdir(parents=True)
    (run / "csv" / "registered.csv").write_text("patient_id\nP1\n")
    (run / ".done").write_text("x\n")
    config = tmp_path / "figures.yaml"
    config.write_text(
        "arms: [valis_high_micro2]\nreference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: [cellsam]\n"
        "  params:\n    cellsam_model_path: /models/cellsam_base.pt\n"
        "figures:\n  zoom:\n    field_um: [150]\n    masks: [cell]\n"
    )
    (tmp_path / "site.config").write_text("// test\n")
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    # a fake `nextflow` that only records that it was asked to run, so the params file stays
    binpath = tmp_path / "bin"
    binpath.mkdir()
    (binpath / "nextflow").write_text("#!/usr/bin/env bash\nexit 7\n")
    (binpath / "nextflow").chmod(0o755)
    subprocess.run(
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
            "PATH": f"{binpath}:/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    params = json.loads((root / ".launch" / "seg_cellsam" / "params.json").read_text())
    assert params["cellsam_model_path"] == "/models/cellsam_base.pt"
    assert params["seg_method"] == "cellsam"


def _token_run(tmp_path, rc_text, env_token=None):
    """Run the launcher with a fake HOME whose .bashrc is `rc_text`, and report what the
    token ended up as. cellsam is requested so the token path is exercised."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(rc_text)
    root = tmp_path / "root"
    (root / ".launch").mkdir(parents=True)
    run = root / "valis_high_micro2"
    (run / "csv").mkdir(parents=True)
    (run / "csv" / "registered.csv").write_text("patient_id\nP1\n")
    (run / ".done").write_text("x\n")
    config = tmp_path / "figures.yaml"
    config.write_text(
        "arms: [valis_high_micro2]\nreference_arm: valis_high_micro2\n"
        "segmentation:\n  methods: [cellsam]\n"
        "figures:\n  zoom:\n    field_um: [150]\n    masks: [cell]\n"
    )
    (tmp_path / "site.config").write_text("// test\n")
    sheet = tmp_path / "input.csv"
    sheet.write_text("patient_id,image\nP1,/dev/null\n")
    # the real PATH (the plan needs pyyaml), with `nextflow` shadowed so segmentation stops
    # right after the token check instead of launching anything
    binpath = tmp_path / "bin"
    binpath.mkdir()
    # records its own environment: that is where an unexported token shows up missing
    (binpath / "nextflow").write_text(
        f"#!/usr/bin/env bash\nprintenv > {tmp_path / 'child_env.txt'}\nexit 7\n"
    )
    (binpath / "nextflow").chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(home),
        "ROOT": str(root),
        "SRC_DIR": str(BENCH.parent),
        "CONFIG": str(config),
        "RENDER_EXEC": "/bin/true",
        "SITE_CONFIG": str(tmp_path / "site.config"),
        "CONDA_ENV": "",
        "SKIP_REGISTRATION": "1",
        "PATH": f"{binpath}:{os.environ['PATH']}",
    }
    env.pop("DEEPCELL_ACCESS_TOKEN", None)
    if env_token:
        env["DEEPCELL_ACCESS_TOKEN"] = env_token
    return subprocess.run(
        ["bash", str(BENCH / "submit_figures.sh"), str(sheet)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )


NON_INTERACTIVE_RC = (
    "case $- in *i*) ;; *) return;; esac\n"  # the guard nearly every distro ships
    "export DEEPCELL_ACCESS_TOKEN=tok-from-rc\n"
)


def test_a_token_in_bashrc_is_found_even_though_the_batch_shell_returns_early(tmp_path):
    """~/.bashrc usually returns immediately in a non-interactive shell, so `source` sets
    nothing and the token silently never reaches the container."""
    proc = _token_run(tmp_path, NON_INTERACTIVE_RC)
    assert "taken from a shell rc file" in proc.stdout
    assert f"is set ({len('tok-from-rc')} chars) and exported" in proc.stdout


def test_a_token_without_export_actually_reaches_the_child_process(tmp_path):
    """A bare assignment is a shell VARIABLE, not an environment one, so nextflow -- and
    therefore the container -- never sees it. Asserted on the CHILD's environment, not on a
    log line: the log would say "set" either way."""
    proc = _token_run(tmp_path, "DEEPCELL_ACCESS_TOKEN=tok-no-export\n")
    assert f"is set ({len('tok-no-export')} chars) and exported" in proc.stdout
    child = (tmp_path / "child_env.txt").read_text()
    assert "DEEPCELL_ACCESS_TOKEN=tok-no-export" in child


def test_an_already_exported_token_is_left_alone(tmp_path):
    proc = _token_run(tmp_path, "# nothing here\n", env_token="tok-from-env")
    assert f"is set ({len('tok-from-env')} chars) and exported" in proc.stdout
    assert "taken from a shell rc file" not in proc.stdout


def test_no_token_anywhere_warns_before_the_run_not_after_it_fails(tmp_path):
    proc = _token_run(tmp_path, "# nothing here\n")
    assert "DEEPCELL_ACCESS_TOKEN is EMPTY" in proc.stderr
    assert "cellsam_model_path" in proc.stderr


def test_phase_one_builds_the_arms_and_draws_no_mosaic_of_its_own(tmp_path):
    """Every mosaic is a plan row with its own patch size, kind, numbers and ROI; letting
    submit_mosaic.sh draw one too would add an extra, unasked-for figure."""
    launcher = (BENCH / "submit_figures.sh").read_text()
    assert "DRAW=0" in launcher
    mosaic = (BENCH / "submit_mosaic.sh").read_text()
    assert (
        'if [[ "$DRAW" != "1" ]]; then' in mosaic
    )  # and the switch exists to honour it


def test_the_launcher_never_parses_a_plan_row_by_field_number(tmp_path):
    """The row format is (kind, run, outdir, args). Reading a field by number is how
    `--patch-um mosaic/p200_overlay_auto` reached reg_mosaic (job 687104*): the args moved
    to field 4 and an awk '{print $3}' kept pointing at what used to be there. The only
    legitimate positional reads are of $1 and $2, which name a kind and a run key."""
    launcher = (BENCH / "submit_figures.sh").read_text()
    import re

    for match in re.finditer(r"awk[^\n]*\$(\d)", launcher):
        # $0 is the whole line (a dedupe), not a field read
        assert match.group(1) in ("0", "1", "2"), match.group(0)
    # and the whole argument list is handed over verbatim, never rebuilt
    assert 'eval "set -- $args"' in launcher


def test_one_failed_arm_does_not_stop_the_other_figures(tmp_path):
    """A VALIS that dies must not cost the STARE overlays, the segmentation, or the channel
    crops -- the same rule as a failed segmentation backend."""
    launcher = (BENCH / "submit_figures.sh").read_text()
    phase1 = launcher[
        launcher.index("# ---- 1. registration") : launcher.index("# ---- 2.")
    ]
    assert "exit 1" not in phase1
    assert "drawing from the ones that finished" in phase1


def test_building_arms_without_drawing_still_reports_a_failed_arm(tmp_path):
    """DRAW=0 must not turn a failed arm into a success just because no picture was asked
    for: it exits with the same status the drawing path would."""
    mosaic = (BENCH / "submit_mosaic.sh").read_text()
    block = mosaic[mosaic.index('if [[ "$DRAW" != "1" ]]') :][:400]
    assert "rc_valis == 0 && rc_stare == 0 && rc_ashlar == 0" in block
    assert "exit $?" in block
