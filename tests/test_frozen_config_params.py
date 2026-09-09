"""Every setting nextflow.config derives from a param as a SCALAR is frozen at the line
it is written on. This file pins where those lines sit and that the launch-time check
covering them names every one of them.

THE DEFECT, measured 2026-09-09 on dev @ 3f482063 with a probe script printing
`workflow.session.config` (Nextflow 25.04.7):

    route                                  params.concurrency  executor.queueSize  process.maxForks  cleanup
    -profile test (pins cleanup_work=false)         5                 20                  5           false
    -profile test -c pin.config (7 / true)          7                 20                  5           false
    -profile test -params-file pin.json (7 / true)  7                 99                  3           true
    -profile test --concurrency 7 (CLI)             7                 99                  3           false*

    * cleanup_work is a boolean and cannot be passed on the CLI at all.

A `-c site.config` pin -- the route every documented command ends with -- changed
`params.*` and reached NOTHING it drives: not `cleanup`, not `trace.enabled`, not
`trace.file`, not `executor.queueSize`, not `process.maxForks`, not the four per-process
`maxForks` caps. Nextflow parses nextflow.config IN FULL before any `-c` file is merged,
so a scalar computed from `params.x` inside it has already been evaluated. The profile
route was additionally dead for `concurrency`/`max_forks`/`queue_size` because their
scalars sat ABOVE `profiles {}`; a profile pinning `params.concurrency = 7` left queueSize
at 20 and every maxForks at 5.

The same mechanism, one row down: the `slurm` profile assigned `process.resourceLimits`
as a PLAIN MAP, freezing it to `[cpus:null, memory:null, time:240.h]` on the documented
`-profile slurm,singularity -c site.config` route -- the retry ramp ran unclamped on the
one executor the clamp exists for. The comment above that line described the bug
accurately and the line was still there.

THE FIX, and what each test below pins:

  1. Every params-derived scalar sits AFTER `profiles {}`, so the profile route works.
  2. The `-c` route CANNOT be made to work (nothing in nextflow.config runs after a `-c`
     file), so `ParamUtils.validateFrozenConfig` compares each frozen scalar with the
     final params at launch and REFUSES the run on a mismatch, naming the route. The set
     of params it checks is discovered here by scanning nextflow.config, not restated, so
     a new `foo = params.bar` scalar fails this file until the validator covers it.
  3. `resourceLimits` is a closure everywhere -- never a plain map -- so it is resolved at
     task-submission time against the merged params.

These are static: they read the sources. The behavioural counterpart is
tests/frozen_config_pin.sh, which runs the `-c` pin and asserts the refusal, then runs
the `-params-file` and CLI routes and asserts the values arrived.

Positive rules over "does this line exist" read the comment-STRIPPED view (a comment
quoting the assignment must not satisfy them); the plain-map rule is a negative rule and
also reads the stripped view, because a comment *describing* the plain map (there was
one) must not trip it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.nfmodel import REPO_ROOT as ROOT
from tests.nfmodel import block_extent, strip_comments, strip_comments_and_strings

NEXTFLOW_CONFIG = ROOT / "nextflow.config"
PARAM_UTILS = ROOT / "lib" / "ParamUtils.groovy"
MIRAGE_NF = ROOT / "workflows" / "mirage.nf"
CONFIGS = [NEXTFLOW_CONFIG, *sorted((ROOT / "conf").glob("*.config"))]


@pytest.fixture(scope="module")
def nf_code() -> str:
    """nextflow.config with comments blanked, strings kept (the assignments under
    assertion include `"${params.trace_dir}/trace.txt"`)."""
    return strip_comments(NEXTFLOW_CONFIG.read_text())


def _top_level_block_span(code: str, name: str) -> tuple[int, int]:
    """(start, end) offsets of the top-level `<name> {` block in comment-and-string
    blanked text. `end` is the index of the closing brace."""
    m = re.search(rf"(?m)^{re.escape(name)}\s*\{{", code)
    assert m, f"nextflow.config has no top-level `{name} {{` block"
    return m.start(), block_extent(code, m.end())


def _frozen_scalars(code: str) -> list[tuple[str, str, int]]:
    """Every `<key> = <expr>` in nextflow.config whose expr reads `params.<x>` and is
    NOT a closure, outside the `params {}` and `profiles {}` blocks.

    Returns (key, param, offset) triples, one per param referenced.
    """
    blanked = strip_comments_and_strings(code)
    skip = [_top_level_block_span(blanked, "params"), _top_level_block_span(blanked, "profiles")]
    out = []
    for m in re.finditer(r"(?m)^\s*([\w.]+)\s*=\s*(.+?)\s*$", code):
        start = m.start()
        if any(a <= start <= b for a, b in skip):
            continue
        key, expr = m.group(1), m.group(2)
        if expr.startswith("{"):
            continue  # a closure is resolved lazily; that is the point of it
        for param in sorted(set(re.findall(r"\bparams\.(\w+)", expr))):
            out.append((key, param, start))
    return out


def _paramutils_list(name: str) -> list[str]:
    src = strip_comments(PARAM_UTILS.read_text())
    m = re.search(rf"static\s+final\s+List<String>\s+{name}\s*=\s*\[(.*?)\]", src, flags=re.S)
    assert m, f"lib/ParamUtils.groovy must declare `static final List<String> {name} = [...]`"
    return re.findall(r"'([^']+)'", m.group(1))


# ---------------------------------------------------------------------------
# 1. position
# ---------------------------------------------------------------------------


def test_every_params_derived_scalar_sits_after_the_profiles_block(nf_code):
    """A scalar above `profiles {}` freezes against the pre-profile params; one below
    sees the merged profile stack. Measured: `executor.queueSize` and `process.maxForks`
    above the block ignored a profile's `params.concurrency = 7`."""
    scalars = _frozen_scalars(nf_code)
    assert scalars, "expected at least one params-derived scalar in nextflow.config"
    _, profiles_end = _top_level_block_span(strip_comments_and_strings(nf_code), "profiles")
    early = sorted({(k, p) for k, p, off in scalars if off < profiles_end})
    assert not early, (
        f"params-derived scalar(s) assigned ABOVE `profiles {{}}` in nextflow.config: "
        f"{early}. They freeze against the pre-profile params, so a profile pin of that "
        "param is silently ignored. Move the assignment below the profiles block."
    )


# ---------------------------------------------------------------------------
# 2. the launch check covers exactly the frozen set
# ---------------------------------------------------------------------------


def test_the_frozen_param_set_is_discovered_not_restated(nf_code):
    """The set of params that nextflow.config freezes into scalars must equal
    ParamUtils.FROZEN_CONFIG_PARAMS -- in both directions. A param added to the config
    without a check is the silent-ignore bug again; one listed but no longer frozen is
    a stale refusal waiting to fire."""
    in_config = sorted({p for _, p, _ in _frozen_scalars(nf_code)})
    declared = sorted(_paramutils_list("FROZEN_CONFIG_PARAMS"))
    assert in_config == declared, (
        f"nextflow.config freezes {in_config} into scalars but "
        f"ParamUtils.FROZEN_CONFIG_PARAMS declares {declared}"
    )


def test_validate_frozen_config_reads_every_frozen_param():
    """The validator body must reference each frozen param by name -- the list constant
    alone proves nothing about what the method compares."""
    src = strip_comments(PARAM_UTILS.read_text())
    m = re.search(r"static\s+void\s+validateFrozenConfig\s*\([^)]*\)\s*\{", src)
    assert m, "lib/ParamUtils.groovy must define `static void validateFrozenConfig(...)`"
    body = src[m.end() : block_extent(src, m.end())]
    # One level of static helper is allowed (derivedMaxForks / derivedQueueSize exist so
    # the validator recomputes exactly what the config line computes); their bodies count.
    for helper in set(re.findall(r"\b(derived\w+)\s*\(", body)):
        h = re.search(rf"static\s+\w+\s+{helper}\s*\([^)]*\)\s*\{{", src)
        assert h, f"validateFrozenConfig calls {helper}() but ParamUtils does not define it"
        body += src[h.end() : block_extent(src, h.end())]
    missing = [p for p in _paramutils_list("FROZEN_CONFIG_PARAMS") if f"params.{p}" not in body]
    assert not missing, f"validateFrozenConfig never reads params.{missing}"


def test_mirage_nf_calls_the_validator_with_the_session_config():
    """Called from the workflow, with the resolved session config -- the frozen values
    live nowhere else. Read comment-stripped: a comment naming the call is not a call."""
    code = strip_comments(MIRAGE_NF.read_text())
    assert re.search(
        r"ParamUtils\.validateFrozenConfig\(\s*params\s*,\s*workflow\.session\.config\s*,\s*workflow\.commandLine\s*\)",
        code,
    ), (
        "workflows/mirage.nf must call ParamUtils.validateFrozenConfig(params, "
        "workflow.session.config, workflow.commandLine) -- the command line is what "
        "exempts Nextflow's own -with-trace/-report/-timeline overrides"
    )


# ---------------------------------------------------------------------------
# 3. resourceLimits is never a plain map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(ROOT)) for p in CONFIGS])
def test_resource_limits_is_a_closure_everywhere(path: Path):
    """`resourceLimits = [ ... ]` is evaluated where it is written. Inside a profile that
    is before any `-c` file and before a later profile's params, so it froze to
    [cpus:null, memory:null] on the documented SLURM route. Only the closure form defers
    to task-submission time."""
    code = strip_comments(path.read_text())
    plain = re.findall(r"(?m)^[ \t]*(?:process\.)?resourceLimits[ \t]*=[ \t]*\[.*$", code)
    assert not plain, f"{path.relative_to(ROOT)} assigns resourceLimits as a plain map: {plain}"
