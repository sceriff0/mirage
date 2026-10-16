"""A resumed run may change these params without re-running any task.

The benchmark launchers that resumed runs (the arms and sweep launchers) are
removed; regbench's two Nextflow launchers never pass -resume. The property
below still protects an operator's own -resume: a stop and resume may change
``cleanup_work`` and the head count / job ceiling (``concurrency`` /
``max_forks`` / ``queue_size``). That is free only because Nextflow hashes the
params a process ``script:`` block REFERENCES: ``params.foo`` is itemised, and
only a script that references the bare ``params`` object hashes the whole map.

Measured on Nextflow 25.04.7, non-stub, 2026-09-14, on a minimal pipeline with
one process reading ``params.a`` and one interpolating ``${params}``: changing
``b`` and ``cleanup_work`` on resume left the first task cached and re-ran the
second; changing ``a`` re-ran both.

So the property these tests pin is: no process script reads a changed-on-resume
param, and no process script references the whole map (as an interpolation,
``$params``, or a value passed to a call). If either ever becomes true, a
stop-and-resume silently re-runs every such task -- REGISTER among them, hours
per patient -- while every run still ends green.

Reads the model in tests/nfmodel: ``Process.script_body`` keeps string contents
(a script command is a string, and ``${params.cleanup_work}`` would live inside
it), and ``strip_comments`` removes comments without blanking strings.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_actions import strip_line_comment  # noqa: E402

from tests.nfmodel import param_refs, processes, strip_comments

REPO = Path(__file__).resolve().parents[1]

# The params an operator changes on resume.
LAUNCHER_CHANGED_PARAMS = ("cleanup_work", "concurrency", "max_forks", "queue_size")

# The whole map referenced as a value: `${params}`, `$params`, `f(params)`,
# `f(x, params)`, `[key: params]`. Not `params.foo`, not `params.subMap(...)`, and
# not a local variable that happens to end in "params" (`seg_params`).
_WHOLE_MAP = re.compile(
    r"\$\{\s*params\s*\}"
    r"|\$params\b(?!\s*\.)"
    r"|(?<![\w.])params(?=\s*[,)\]])"
)


def _script(p) -> str:
    return strip_comments(p.script_body)


def test_the_model_sees_param_reads_inside_a_script_string():
    """The premise: a `${params.x}` inside the triple-quoted command is visible
    to the view these guards read, and a comment mentioning one is not."""
    sample = '    """\n    tool --flag ${params.cleanup_work}\n    """\n    // ${params.max_forks}\n'
    assert param_refs(strip_comments(sample)) == {"cleanup_work"}
    assert _WHOLE_MAP.search('echo "${params}"')
    assert _WHOLE_MAP.search("ParamUtils.foo(params)")
    assert not _WHOLE_MAP.search("${params.a} seg_params params.subMap(['x'])")


def test_the_scan_actually_reads_process_scripts():
    procs = processes()
    assert len(procs) > 20, f"only {len(procs)} processes modelled -- the scan is blind"
    with_refs = [n for n, p in procs.items() if param_refs(_script(p))]
    assert len(with_refs) > 10, (
        f"only {len(with_refs)} process scripts read any param -- the view is blanking strings"
    )


def test_no_process_script_reads_a_param_the_launchers_change_on_resume():
    offenders = {
        name: sorted(param_refs(_script(p)) & set(LAUNCHER_CHANGED_PARAMS))
        for name, p in processes().items()
    }
    offenders = {n: keys for n, keys in offenders.items() if keys}
    assert not offenders, (
        "These process scripts read a param the benchmark launchers change between a stop and "
        "a resume, so every stop-and-resume would re-run them from scratch:\n  "
        + "\n  ".join(f"{n}: {', '.join(k)}" for n, k in sorted(offenders.items()))
        + "\nPass the value in some other way, or stop the launchers changing it on resume."
    )


def test_no_process_script_references_the_whole_params_map():
    offenders = []
    for name, p in sorted(processes().items()):
        for i, line in enumerate(_script(p).splitlines(), 1):
            if _WHOLE_MAP.search(line):
                offenders.append(
                    f"{name} ({p.path.name}), script line {i}: {line.strip()}"
                )
    assert not offenders, (
        "A process script references the WHOLE params map, which puts every parameter into "
        "its cache key: any unrelated change -- including the ones the benchmark launchers "
        "make on resume -- re-runs it. Pass the specific values instead.\n  "
        + "\n  ".join(offenders)
    )


def test_no_benchmark_launcher_resumes_a_pipeline_run():
    """regbench's Nextflow launchers never pass -resume to the pipeline."""
    tracked = subprocess.run(
        ["git", "ls-files", "benchmarks"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    launchers = [f for f in tracked if f.endswith((".sbatch", ".sh"))]
    names = {Path(f).name for f in launchers}
    assert {"inputs.sbatch", "compute_all.sbatch"} <= names, (
        f"the scan found {sorted(names)}: it no longer sees the two Nextflow launchers"
    )
    offenders = []
    for f in launchers:
        for n, line in enumerate((REPO / f).read_text().splitlines(), 1):
            code = strip_line_comment(line)
            if re.search(r"\bnextflow\b", code) and re.search(r"(?<![\w-])-resume\b", code):
                offenders.append(f"{f}:{n}: {code.strip()}")
    assert not offenders, "a benchmark launcher resumes a pipeline run:\n  " + "\n  ".join(offenders)
