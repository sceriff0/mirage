"""The ASHLAR arm runs each step in the image of the pipeline process it stands in for.

ASHLAR is an external arm: benchmarks/run_ashlar_arm.sh runs inside the arms head
job, outside Nextflow, so nothing picks a container for its steps unless the
submitter does. Before 2026-09-14 submit_arms.sh set none, and every step would
have run bare on the head node, where neither the ashlar package nor the
pipeline's Python stack is installed. Pinned here:

  * the submitter's defaults name the SAME images the pipeline processes declare
    (TILED_STITCH and the tiled WARP_SEG_QC for retile/stitch/seg QC,
    GENERATE_REGISTRATION_QC for the QC composite) plus ASHLAR's own image for the
    solve, and bind the data filesystems Nextflow's autoMounts would have bound;
  * sif_or_docker prefers Nextflow's cached image over a fresh pull;
  * each step in run_ashlar_arm.sh uses the prefix that matches its image;
  * the repo and packages/stare/src are on PYTHONPATH inside the containers, because
    the bin/utils shims import `stare` and the ASHLAR image does not install it.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

from tests.nfmodel import processes

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
sys.path.insert(0, str(REPO / "tests"))
strip_line_comment = importlib.import_module("ci_actions").strip_line_comment

ASHLAR_IMAGE = "labsyspharm/ashlar:1.20.0"


def _code(path: Path) -> str:
    return "\n".join(strip_line_comment(ln) for ln in path.read_text().splitlines())


def _container(process: str) -> str:
    m = re.search(r"""container\s+['"]([^'"]+)['"]""", processes()[process].raw_body)
    assert m, f"{process} declares no literal container"
    return m.group(1)


def _tiled_warp_container() -> str:
    text = (REPO / "lib" / "WarpBackends.groovy").read_text()
    block = text[text.index("tiled: [") :]
    return re.search(r"container\s*:\s*'([^']+)'", block).group(1)


def _default(code: str, var: str) -> str:
    m = re.search(rf'^export {var}="\$\{{{var}:-(.*)\}}"$', code, re.M)
    assert m, f"submit_arms.sh does not export a default for {var}"
    return m.group(1)


def test_submit_arms_runs_each_ashlar_step_in_the_pipelines_own_image():
    code = _code(BENCH / "submit_arms.sh")
    tiled = _container("TILED_STITCH")
    assert _tiled_warp_container() == tiled, (
        "tiled WARP_SEG_QC and TILED_STITCH diverged"
    )
    assert f"sif_or_docker {tiled})" in _default(code, "QC_EXEC")
    assert f"sif_or_docker {_container('GENERATE_REGISTRATION_QC')})" in _default(
        code, "REGQC_EXEC"
    )
    assert f"sif_or_docker {ASHLAR_IMAGE})" in _default(code, "ASHLAR_EXEC")
    assert ASHLAR_IMAGE in (BENCH / "ashlar" / "solve.py").read_text()
    for var in ("ASHLAR_EXEC", "QC_EXEC", "REGQC_EXEC"):
        assert _default(code, var).startswith("singularity exec $SING_BINDS "), var
    binds = re.search(r'^SING_BINDS="\$\{SING_BINDS:-(.*)\}"$', code, re.M).group(1)
    assert "--bind /beegfs" in binds and "--bind /hpcnfs" in binds


def test_sif_or_docker_prefers_nextflows_cached_image(tmp_path):
    text = (BENCH / "submit_arms.sh").read_text()
    fn = text[
        text.index("sif_or_docker() {") : text.index(
            "\n}\n", text.index("sif_or_docker() {")
        )
        + 3
    ]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "bolt3x-mirage-tiled-1.0.0.img").write_bytes(b"")
    snippet = f"{fn}\nsif_or_docker bolt3x/mirage-tiled:1.0.0; echo; sif_or_docker {ASHLAR_IMAGE}"
    r = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "NXF_SINGULARITY_CACHEDIR": str(cache)},
    )
    assert r.returncode == 0, r.stderr
    cached, pulled = r.stdout.splitlines()
    assert cached == str(cache / "bolt3x-mirage-tiled-1.0.0.img")
    assert pulled == f"docker://{ASHLAR_IMAGE}"


def test_each_step_uses_the_prefix_of_its_image():
    code = _code(BENCH / "run_ashlar_arm.sh").replace("\\\n", " ")
    lines = code.splitlines()

    def prefix_of(needle: str) -> set:
        hits = [ln for ln in lines if needle in ln]
        assert hits, f"no step runs {needle}"
        return {re.search(r"\$(\w+_EXEC) python3", ln).group(1) for ln in hits}

    assert prefix_of("benchmarks.ashlar.retile") == {"QC_EXEC"}
    assert prefix_of("benchmarks.ashlar.solve") == {"ASHLAR_EXEC"}
    assert prefix_of("bin/tiled_stitch.py") == {"QC_EXEC"}
    assert prefix_of("bin/warp_seg_qc.py") == {"QC_EXEC"}
    assert prefix_of("bin/generate_registration_qc.py") == {"REGQC_EXEC"}


def test_the_repo_and_the_stare_package_are_on_the_path_inside_the_containers():
    code = _code(BENCH / "run_ashlar_arm.sh")
    m = re.search(r'^STEP_PYTHONPATH="([^"]+)"$', code, re.M)
    assert m and m.group(1) == "$REPO:$REPO/packages/stare/src", m and m.group(1)
    assert 'export PYTHONPATH="$STEP_PYTHONPATH' in code
    assert 'SINGULARITYENV_PYTHONPATH="$STEP_PYTHONPATH"' in code
    assert 'APPTAINERENV_PYTHONPATH="$STEP_PYTHONPATH"' in code
    # the shim that makes it necessary is still a shim
    assert (
        "from stare import manifest"
        in (REPO / "bin" / "utils" / "tiled_manifest.py").read_text()
    )
