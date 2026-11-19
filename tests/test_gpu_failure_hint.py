"""A run that fails because the container runtime has no GPU says what to do.

`seg_gpu` is on by default, so on a laptop, a Mac or WSL without CUDA the runtime
refuses SEGMENT with its own text. `main.nf`'s `announceGpuHint` matches that text
in `workflow.errorMessage` and points at `params/no_gpu.json`.
"""

import json
import re

from tests.nfmodel import REPO_ROOT, strip_comments

MAIN = REPO_ROOT / "main.nf"

# The runtimes' own words, as they reach workflow.errorMessage.
RUNTIME_ERRORS = [
    # Docker Desktop / docker without the NVIDIA toolkit
    'docker: Error response from daemon: could not select device driver "" '
    "with capabilities: [[gpu]].",
    # docker with the toolkit installed but no driver loaded
    "nvidia-container-cli: initialization error: load library failed: "
    "libnvidia-ml.so.1: cannot open shared object file",
]


def _hint_pattern():
    src = strip_comments(MAIN.read_text())
    m = re.search(r"def announceGpuHint\(.*?err =~ /(.*?)/\)", src, re.S)
    assert m, "main.nf no longer defines announceGpuHint with an `err =~ /.../` match"
    return re.compile(m.group(1))


def test_the_hint_matches_each_runtime_error():
    pat = _hint_pattern()
    missed = [e for e in RUNTIME_ERRORS if not pat.search(e)]
    assert not missed, f"announceGpuHint does not recognise: {missed}"


def test_the_hint_ignores_an_unrelated_failure():
    pat = _hint_pattern()
    assert not pat.search(
        "Process `SEGMENT` terminated with an error exit status (137)"
    )


def test_on_complete_calls_the_hint_and_the_preset_exists():
    src = strip_comments(MAIN.read_text())
    on_complete = re.search(r"workflow\.onComplete\s*\{(.*?)\}", src, re.S)
    assert on_complete and "announceGpuHint(" in on_complete.group(1), (
        "workflow.onComplete no longer calls announceGpuHint"
    )
    assert "seg_gpu" in src, "report_cfg no longer carries seg_gpu into onComplete"
    preset = json.loads((REPO_ROOT / "params" / "no_gpu.json").read_text())
    assert preset.get("seg_gpu") is False, (
        "params/no_gpu.json no longer sets seg_gpu false"
    )
