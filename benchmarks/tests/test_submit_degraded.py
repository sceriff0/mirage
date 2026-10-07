"""submit_degraded.sh: VALIS, STARE and ASHLAR-as-published on equally degraded inputs.

The launchers it calls (submit_arms.sh, submit_ashlar.sh) and sbatch are stand-ins that
record how they were called; the degradation itself runs for real on a small slide. What
is pinned is the orchestration: a results root of its own, noisy slides under the names
the clean ones have, EXACTLY the two arms launched, and ASHLAR cutting its tiles from the
CLEAN slides.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from benchmarks import plan_exact

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "submit_degraded.sh"
ARMS = "^(valis_high_micro2|tiled_high_s64)$"


def _site(tmp_path):
    bench = tmp_path / "bench"
    pre = bench / "arm_results" / "preprocess_shared"
    (pre / "P1" / "preprocessed").mkdir(parents=True)
    slide = pre / "P1" / "preprocessed" / "P1_DAPI_CD3.ome.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(
        slide,
        rng.integers(100, 4000, size=(2, 300, 400)).astype(np.uint16),
        tile=(64, 64),
        photometric="minisblack",
        metadata={"axes": "CYX"},
        ome=True,
    )
    (pre / "csv").mkdir()
    (pre / "csv" / "preprocessed.csv").write_text(
        "patient_id,id,preprocessed_image,is_reference,channels,pixel_size\n"
        f"P1,P1_DAPI_CD3,{slide},true,DAPI|CD3,0.325\n"
    )
    # a checkout whose launchers are stand-ins and whose python package is the real one
    src = tmp_path / "src"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "ashlar").symlink_to(REPO / "benchmarks" / "ashlar")
    (src / "benchmarks" / "__init__.py").write_text("")
    (src / "bin").symlink_to(REPO / "bin")
    record = tmp_path / "arms"
    (src / "benchmarks" / "submit_arms.sh").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$BENCH_DIR" "$RESULTS" "$EXACT" "$ARMS_RESUME" "$ENABLE_CSE" '
        f'"[$ONLY]" > "{record}"\n'
        'mkdir -p "$RESULTS/valis_high_micro2/P1/qc/registration"\n'
        'echo "{}" > "$RESULTS/valis_high_micro2/P1/qc/registration/P1_x_seg_qc.json"\n'
    )
    (src / "benchmarks" / "submit_ashlar.sh").write_text("#!/bin/bash\n")
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{tmp_path}/sbatch"\n')
    sbatch.chmod(0o755)
    images = tmp_path / "images"
    images.mkdir()
    (images / "bolt3x-mirage-stare-1.0.0.img").write_text("img")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "BENCH_DIR": str(bench),
        "SRC_DIR": str(src),
        "IMAGES": str(images),
        "DEGRADE_EXEC": "env",  # no container: the same python
        "NOISE_FRAC": "0.02",
    }
    for k in (
        "DEGRADED",
        "CLEAN_RESULTS",
        "ARMS",
        "RESULTS",
        "ONLY",
        "EXACT",
        "RUN_ASHLAR",
    ):
        env.pop(k, None)
    return bench, slide, env


def test_one_command_degrades_then_runs_the_two_arms_and_submits_ashlar(tmp_path):
    bench, slide, env = _site(tmp_path)
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + run.stdout
    root = bench / "degraded" / "arm_results"
    noisy = root / "preprocess_shared" / "P1" / "preprocessed" / slide.name
    assert noisy.is_file()
    assert not np.array_equal(tifffile.imread(noisy), tifffile.imread(slide))
    assert (
        str(noisy)
        in (root / "preprocess_shared" / "csv" / "preprocessed.csv").read_text()
    )
    bench_dir, results, exact, resume, cse, only = (
        (tmp_path / "arms").read_text().splitlines()
    )
    assert bench_dir == str(bench / "degraded") and results == str(root)
    # EXACT rows, never ONLY (which adds every arm scored on VALIS's nuclei)
    assert exact == ARMS and only == "[]"
    assert (resume, cse) == ("1", "false")
    export, script = (tmp_path / "sbatch").read_text().splitlines()
    assert script.endswith("benchmarks/submit_ashlar.sh")
    assert "MODE=original" in export and f"RESULTS={root}" in export
    assert "FROM_ARM=valis_high_micro2" in export and "NOISE_FRAC=0.02" in export
    # the tiles are cut from the CLEAN slides: their noise is added per tile
    clean_csv = bench / "arm_results" / "preprocess_shared" / "csv" / "preprocessed.csv"
    assert f"PREPROC_CSV={clean_csv}" in export
    # nothing of the main benchmark was written to
    assert not (bench / "arm_results" / "valis_high_micro2").exists()


def test_it_refuses_a_root_inside_the_main_one_and_missing_preprocessing(tmp_path):
    bench, _, env = _site(tmp_path)
    inside = {**env, "DEGRADED": str(bench / "arm_results" / "x")}
    run = subprocess.run(
        ["bash", str(SCRIPT)], env=inside, capture_output=True, text=True
    )
    assert run.returncode == 1 and "its own directory" in run.stderr
    (bench / "arm_results" / "preprocess_shared" / "csv" / "preprocessed.csv").unlink()
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert run.returncode == 1 and "has not finished" in run.stderr
    assert not (tmp_path / "arms").exists()


def test_exact_rows_are_the_named_rows_alone_unlike_the_dependant_closure(tmp_path):
    """The real planner on the real arms.yaml: --only the two arms selects nearly every
    run (everything scored on VALIS's nuclei); plan_exact keeps the two."""
    sheet = tmp_path / "in.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\n"
        "P1,/x/a.nd2,true,DAPI|SMA\nP1,/x/b.nd2,false,DAPI|CD3\n"
    )

    def plan(*extra):
        out = tmp_path / f"plan{len(extra)}.csv"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "benchmarks" / "build_arm_plan.py"),
                "--arms",
                str(REPO / "benchmarks" / "configs" / "arms.yaml"),
                "--input",
                str(sheet),
                "--out",
                str(out),
                "--results-root",
                str(tmp_path / "root"),
                *extra,
            ],
            check=True,
            capture_output=True,
            cwd=REPO,
        )
        return out

    full = plan()
    with open(plan("--only", ARMS), newline="") as fh:
        assert (
            len(list(csv.DictReader(fh))) > 50
        )  # the closure: why ONLY is the wrong tool
    exact = tmp_path / "exact.csv"
    assert plan_exact.main([str(full), str(exact), ARMS]) == 0
    with open(exact, newline="") as fh:
        kept = list(csv.DictReader(fh))
    assert [r["run_id"] for r in kept] == ["valis_high_micro2", "tiled_high_s64"]
    assert {r["from_arm"] for r in kept} == {"preprocess_shared"}
    assert kept[1]["seg_qc_nuclei_from"] == "valis_high_micro2"
    # a row scored on nuclei the pattern leaves out is refused, not launched to fail
    with pytest.raises(SystemExit, match="valis_high_micro2"):
        plan_exact.main([str(full), str(exact), "^tiled_high_s64$"])
    with pytest.raises(SystemExit, match="matches no run_id"):
        plan_exact.main([str(full), str(exact), "^nothing$"])


def test_the_arms_runner_launches_the_two_exact_rows_from_a_bare_checkpoint(tmp_path):
    """What step 2 relies on, through the real run_arms.sh and a stand-in nextflow: a plan
    of the two rows alone, in a root that holds nothing but a preprocessing checkpoint,
    launches VALIS, then STARE scored on VALIS's nuclei IN THAT ROOT -- and no
    preprocessing, which would overwrite the degraded slides with clean ones."""
    import json

    from benchmarks.tests.test_build_arm_plan import _fake_nextflow

    sheet = tmp_path / "in.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.nd2,true,DAPI|SMA\n"
    )
    full = tmp_path / "full.csv"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / "build_arm_plan.py"),
            "--arms",
            str(REPO / "benchmarks" / "configs" / "arms.yaml"),
            "--input",
            str(sheet),
            "--out",
            str(full),
            "--results-root",
            str(tmp_path / "planroot"),
        ],
        check=True,
        capture_output=True,
        cwd=REPO,
    )
    exact = tmp_path / "exact.csv"
    plan_exact.main([str(full), str(exact), ARMS])
    root = tmp_path / "degraded" / "arm_results"
    chk = root / "preprocess_shared" / "csv" / "preprocessed.csv"
    chk.parent.mkdir(parents=True)
    chk.write_text(
        "patient_id,id,preprocessed_image,is_reference,channels,pixel_size\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    env = dict(
        os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", ARMS_RESUME="1"
    )
    env.pop("ARMS_REPLACE", None)
    run = subprocess.run(
        [
            "bash",
            str(REPO / "benchmarks" / "run_arms.sh"),
            str(exact),
            str(sheet),
            str(root),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    names = [ln.split("|")[1] for ln in log.read_text().splitlines()]
    assert names == ["arms-valis_high_micro2", "arms-tiled_high_s64"]  # VALIS first
    assert chk.read_text().startswith("patient_id,id,preprocessed_image")  # untouched
    assert not (root / ".launch" / "preprocess_shared").exists()
    for arm in ("valis_high_micro2", "tiled_high_s64"):
        hist = (root / ".launch" / arm / ".nextflow" / "history").read_text()
        assert f"--input {chk}" in hist and f"--outdir {root / arm}" in hist
    stare = json.loads(
        (root / ".launch" / "tiled_high_s64" / "params.tiled_high_s64.json").read_text()
    )
    assert stare["seg_qc_nuclei_dir"] == str(root / "valis_high_micro2")
