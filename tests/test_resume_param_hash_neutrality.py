"""A resumed benchmark run may change these params without re-running any task.

The benchmark launchers (benchmarks/run_arms.sh, run_sweep.sh) change a run's
params between a stop and a resume: they pin ``cleanup_work`` false into the
reused params file, and the operator may lower or raise the head count and the
job ceiling (``concurrency`` / ``max_forks`` / ``queue_size``) for the resumed
launch. That is only free because Nextflow hashes the params a process
``script:`` block REFERENCES: ``params.foo`` is itemised, and only a script that
references the bare ``params`` object hashes the whole map.

Measured on Nextflow 25.04.7, non-stub, 2026-09-14, on a minimal pipeline with
one process reading ``params.a`` and one interpolating ``${params}``: changing
``b`` and ``cleanup_work`` on resume left the first task cached and re-ran the
second; changing ``a`` re-ran both.

So the property these tests pin is: no process script reads a launcher-changed
param, and no process script references the whole map (as an interpolation,
``$params``, or a value passed to a call). If either ever becomes true, a
stop-and-resume of the benchmark silently re-runs every such task -- REGISTER
among them, hours per patient -- while every run still ends green.

Reads the model in tests/nfmodel: ``Process.script_body`` keeps string contents
(a script command is a string, and ``${params.cleanup_work}`` would live inside
it), and ``strip_comments`` removes comments without blanking strings.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import param_refs, processes, strip_comments

REPO = Path(__file__).resolve().parents[1]

# The params the launchers change on resume. Keep in step with the launchers: the
# last test below fails when a launcher pins a key this tuple does not cover.
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


def test_every_key_a_launcher_pins_on_resume_is_covered_here():
    pinned = set()
    for launcher in ("run_arms.sh", "run_sweep.sh"):
        text = (REPO / "benchmarks" / launcher).read_text()
        keys = set(re.findall(r'd\["(\w+)"\]\s*=', text))
        assert keys, (
            f"{launcher}: no pinned key found -- the pin heredoc moved or was renamed"
        )
        pinned |= keys
    uncovered = pinned - set(LAUNCHER_CHANGED_PARAMS)
    assert not uncovered, (
        f"the launchers pin {sorted(uncovered)} into a resumed run's params file, but this "
        "guard does not check that no process script reads them: add them to "
        "LAUNCHER_CHANGED_PARAMS"
    )
