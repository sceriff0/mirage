"""run_ashlar_arm.sh without WARP_SEG_QC: ASHLAR_SEG_QC=0 must still register and write QC.

The container prefixes (ASHLAR_EXEC / QC_EXEC / REGQC_EXEC) are replaced by a fake that logs
the command it was handed and exits 0, so the shell's own control flow is what is tested:
which steps run, and what lands in registered.csv, with and without the scorer's nuclei.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def _arm(tmp_path: Path, seg_qc: str):
    root = tmp_path / "root"
    pre = root / "preprocess_shared" / "csv"
    pre.mkdir(parents=True)
    imgs = root / "preprocess_shared" / "033" / "preprocessed"
    imgs.mkdir(parents=True)
    for name in ("033_a.ome.tif", "033_b.ome.tif"):
        (imgs / name).write_bytes(b"\0" * 64)  # trace_step.py sizes every input
    csv_path = pre / "preprocessed.csv"
    csv_path.write_text(
        "patient_id,id,preprocessed_image,is_reference,channels,pixel_size\n"
        f"033,033_a,{imgs / '033_a.ome.tif'},true,SMA|DAPI,0.325\n"
        f"033,033_b,{imgs / '033_b.ome.tif'},false,CD3|DAPI,0.325\n"
    )
    log = tmp_path / "calls.log"
    fake = tmp_path / "fake"
    # logs its command; touches an --out file, as the real stitch writes the registered slide
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {log}\n'
        'while [[ $# -gt 0 ]]; do [[ "$1" == --out ]] && touch "$2"; shift; done\n'
        "exit 0\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "ASHLAR_EXEC": str(fake),
        "QC_EXEC": str(fake),
        "REGQC_EXEC": str(fake),
        "ASHLAR_SEG_QC": seg_qc,
        "ASHLAR_REG_QC": os.environ.get("_TEST_REG_QC", "1"),
    }
    proc = subprocess.run(
        [
            "bash",
            str(BENCH / "run_ashlar_arm.sh"),
            str(root),
            "ashlar_t1024_s30",
            "valis_high_micro2",
            str(csv_path),
            "1024",
            "0.1",
            "30",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = log.read_text() if log.exists() else ""
    return proc, calls, root / "ashlar_t1024_s30"


def test_seg_qc_off_registers_and_writes_qc_without_any_nuclei(tmp_path):
    proc, calls, out = _arm(tmp_path, "0")
    assert proc.returncode == 0, proc.stderr
    for step in (
        "benchmarks.ashlar.retile",
        "benchmarks.ashlar.solve",
        "tiled_stitch.py",
        "generate_registration_qc.py",
    ):
        assert step in calls, step
    assert "warp_seg_qc.py" not in calls
    # every retile shares one canvas (all the patient's slides) at the run's pixel size
    retiles = [ln for ln in calls.splitlines() if "benchmarks.ashlar.retile" in ln]
    assert len(retiles) == 2
    for ln in retiles:
        assert "--pixel-size-um 0.325" in ln
        assert "--canvas-like" in ln and "033_a.ome.tif" in ln and "033_b.ome.tif" in ln
    rows = (out / "csv" / "registered.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[2].startswith("033,033_b_registered,")


def test_seg_qc_on_still_refuses_to_run_without_the_nuclei(tmp_path):
    proc, calls, _ = _arm(tmp_path, "1")
    assert proc.returncode == 1 and "produced no QC nuclei" in proc.stderr
    assert "benchmarks.ashlar.solve" not in calls


def test_reg_qc_off_skips_the_composite_but_still_stitches(tmp_path, monkeypatch):
    """ASHLAR's QC composite is an 8-bit preview the figures do not read, and it failed inside
    the head job on a real run; ASHLAR_REG_QC=0 skips it, the slide is still written."""
    monkeypatch.setenv("_TEST_REG_QC", "0")
    proc, calls, out = _arm(tmp_path, "0")
    assert proc.returncode == 0, proc.stderr
    assert "generate_registration_qc.py" not in calls
    assert "tiled_stitch.py" in calls and "benchmarks.ashlar.solve" in calls
    assert len((out / "csv" / "registered.csv").read_text().splitlines()) == 3
