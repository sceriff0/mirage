"""`params.max_forks` and `params.queue_size`, and the config load order they depend on.

Concurrency is tunable from the command line: `--max_forks` caps how many tasks of any ONE
process run at once, `--queue_size` caps how many run at once across the WHOLE pipeline.
The lower of the two binds.

Both are null-declared and derive from a single `--concurrency` knob (default 5,
`queue_size = concurrency * 4`) unless explicitly overridden. A default computed inside
the params block would be evaluated before the CLI is applied and ignore
`--concurrency`, so the derivation lives in nextflow.config's "Concurrency" block after
`profiles {}` (with the per-process caps). The CLI, a -params-file and a profile reach
it; a `-c` pin is refused at launch (tests/test_frozen_config_params.py).

Wiring a parameter into `maxForks` fails in three ways depending on where you write it:

1. `params.x` read EAGERLY from a file included before `nextflow.config`'s params block does
   not evaluate to `null` -- Nextflow resolves it to an empty `ConfigObject`, i.e. a Map. So
   `params.max_forks as int` throws "Cannot coerce a map to class java.lang.Integer" and the
   whole config fails to parse. Loud, but the message names neither the parameter nor the
   include order.
2. The same reference inside an `executor { }` scope is read as the opening of a NESTED
   SCOPE NAMED `params`, and the setting disappears from the resolved config with no
   error: `queueSize` falls back to Nextflow's own default.
3. `maxForks` cannot be deferred into a closure the way `memory` can, because it is not a
   dynamic directive: Nextflow compares it against 0 in `TaskProcessor`'s constructor and
   throws "Cannot compare ... Closure ... and java.lang.Integer with value '0'".

The working arrangement: includes after the params block, `queueSize` assigned in
`nextflow.config`, per-process `maxForks` written as a `Math.min(n, ...)` over the param.
These static tests pin the parts whose breakage is silent or misleading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NEXTFLOW_CONFIG = ROOT / "nextflow.config"
BASE_CONFIG = ROOT / "conf" / "base.config"
MODULES_CONFIG = ROOT / "conf" / "modules.config"


@pytest.fixture(scope="module")
def nf() -> str:
    return NEXTFLOW_CONFIG.read_text()


@pytest.fixture(scope="module")
def base() -> str:
    return BASE_CONFIG.read_text()


@pytest.fixture(scope="module")
def modules() -> str:
    return MODULES_CONFIG.read_text()


# ---------------------------------------------------------------------------
# the parameters exist, exactly once, with the documented defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "default"),
    [("concurrency", "5"), ("max_forks", "null"), ("queue_size", "null")],
)
def test_the_parameter_is_declared_once_in_nextflow_config(nf, param, default):
    """Declared in nextflow.config and nowhere else -- the repo's one-owner rule for
    defaults (tests/test_no_duplicate_param_defaults.py enforces the other direction).

    `max_forks` and `queue_size` are declared `null`, not a numeric literal: they derive
    from `concurrency` unless explicitly overridden, and that derivation must happen AFTER
    the CLI is applied (see this module's docstring), which rules out a computed default
    inside the params block itself.
    """
    declarations = re.findall(rf"^\s*{param}\s*=\s*(\S+)", nf, flags=re.M)
    assert declarations == [default], (
        f"expected exactly one `{param} = {default}` in nextflow.config, found "
        f"{declarations!r}"
    )


# ---------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------


def test_process_max_forks_reads_the_parameter(nf):
    """`process.maxForks` is derived from BOTH params.max_forks and params.concurrency,
    not a literal and not params.max_forks alone.

    params.max_forks is null unless the caller explicitly overrides it, so a bare
    `maxForks = params.max_forks` would set every
    process's fork cap to null once max_forks is unset -- Nextflow throws comparing that
    against `0` in TaskProcessor's constructor. The assignment must fall back to
    params.concurrency instead.
    """
    match = re.search(r"^\s*maxForks\s*=\s*(.+)$", nf, flags=re.M)
    assert match, "nextflow.config's `process` scope must assign maxForks"
    expr = match.group(1).strip()
    assert not re.fullmatch(r"\d+", expr), (
        "process.maxForks must not be a bare integer literal"
    )
    assert "params.max_forks" in expr and "params.concurrency" in expr, (
        "nextflow.config's `process` scope must derive `maxForks` from BOTH "
        f"params.max_forks and params.concurrency (found: {expr!r}); a literal, or "
        "params.max_forks alone, makes --concurrency a no-op once max_forks is null"
    )


def test_queue_size_is_assigned_in_nextflow_config_not_base_config(nf, base):
    """The silent one (failure mode 2 in this module's docstring).

    `queueSize = params.queue_size` written into `conf/base.config`'s `executor { }` scope
    parses as a nested scope named `params` and the setting VANISHES -- no error, and the
    executor quietly falls back to Nextflow's default. So the assignment must live in
    `nextflow.config`, after the params block, and must NOT reappear in `conf/base.config`.
    """
    assert re.search(r"queueSize\s*=.*params\.queue_size", nf), (
        "nextflow.config must derive `queueSize` from `params.queue_size` (in its own "
        "`executor { }` block, after the params block) -- a null-test fallback to "
        "params.concurrency is fine; a bare literal or a missing reference is not"
    )
    assert not re.search(r"^\s*queueSize\s*=", base, flags=re.M), (
        "conf/base.config must NOT assign queueSize: it is included before the params "
        "block, where a `params.queue_size` reference is parsed as a nested scope and the "
        "setting disappears silently. Assign it in nextflow.config instead."
    )


# The exact shape every per-process maxForks override must take; only the integer
# cap may vary. A substring check cannot tell it from the bare
# `Math.min(10, params.max_forks as int)`, which becomes `Math.min(10, null)` and
# throws only on a real run. A `?:` fallback is wrong too (Groovy's 0 is falsy).
_CANONICAL_MAX_FORKS_RE = re.compile(
    r"^Math\.min\(\s*\d+\s*,\s*"
    r"\(\s*params\.max_forks\s*!=\s*null\s*\?\s*params\.max_forks\s*:\s*params\.concurrency\s*\)"
    r"\s*as\s*int\s*\)$"
)


def test_every_per_process_max_forks_is_bounded_by_the_parameter(nf):
    """Lowering --max_forks (or --concurrency) must throttle EVERY module, not just
    those without an override.

    The per-process caps live in nextflow.config's "Concurrency" block, after
    `profiles {}`, not in conf/modules.config (included before the profiles, so a
    scalar there freezes against the pre-profile params).
    tests/test_frozen_config_params.py pins the position; this test pins the shape.

    Each per-process value must be EXACTLY
    `Math.min(<its own limit>, (params.max_forks != null ? params.max_forks : params.concurrency) as int)`:
    the process keeps its own ceiling (set for its own memory reasons) while the
    parameter can always pull it down. A bare literal, a bare `params.max_forks`
    (null once unset), or an `?:` fallback would all silently ignore --max_forks/
    --concurrency, or worse, throw only when the closure runs -- see the module-level
    comment above for why a substring check cannot tell these apart.
    """
    assignments = re.findall(r"withName:\s*'\w+'\s*\{\s*maxForks\s*=\s*(.+?)\s*\}", nf)
    assert assignments, "expected per-process maxForks overrides in nextflow.config"
    offenders = []
    for raw in assignments:
        normalised = re.sub(r"\s+", " ", raw.strip())
        if not _CANONICAL_MAX_FORKS_RE.match(normalised):
            offenders.append(normalised)
    assert not offenders, (
        f"{len(offenders)} per-process maxForks value(s) are not the exact "
        "`Math.min(<cap>, (params.max_forks != null ? params.max_forks : "
        f"params.concurrency) as int)` shape (only <cap> may vary): {offenders!r}"
    )


def test_no_per_process_max_forks_remains_in_modules_config(modules):
    """A cap written here is included before `profiles {}` and, being
    a withName setting, would win over the correctly-placed scope default -- silently
    re-introducing the frozen value for that one process."""
    stray = re.findall(r"^\s*maxForks\s*=.*$", modules, flags=re.M)
    assert not stray, (
        f"conf/modules.config must not assign maxForks (it froze before profiles): {stray}"
    )


def test_per_process_caps_match_the_paramutils_table(nf):
    """ParamUtils.PER_PROCESS_MAX_FORKS_CAP is what validateFrozenConfig recomputes the
    expected per-process value from, and conf/*.config cannot read lib/ classes, so the
    two copies can only be kept equal by a test. Both directions."""
    in_config = dict(
        re.findall(r"withName:\s*'(\w+)'\s*\{\s*maxForks\s*=\s*Math\.min\(\s*(\d+)", nf)
    )
    src = (ROOT / "lib" / "ParamUtils.groovy").read_text()
    m = re.search(r"PER_PROCESS_MAX_FORKS_CAP\s*=\s*\[(.*?)\]", src, flags=re.S)
    assert m, "lib/ParamUtils.groovy must declare PER_PROCESS_MAX_FORKS_CAP"
    in_lib = dict(re.findall(r"(\w+)\s*:\s*(\d+)", m.group(1)))
    assert in_config == in_lib, (
        f"nextflow.config's per-process maxForks caps {in_config} != "
        f"ParamUtils.PER_PROCESS_MAX_FORKS_CAP {in_lib}"
    )


# ---------------------------------------------------------------------------
# the load order those two depend on
# ---------------------------------------------------------------------------


def test_the_modular_includes_come_after_the_params_block(nf):
    """The include order (failure modes 1 and 3).

    `conf/modules.config` reads `params.max_forks` EAGERLY -- it has to, because `maxForks`
    is not a dynamic directive and cannot be deferred into a closure. So the params block
    must already have been parsed when that file is included. With the includes at the top
    of nextflow.config, `params.max_forks` is an empty ConfigObject and `as int` fails the
    whole config parse.
    """
    params_at = nf.index("\nparams {")
    includes = [m.start() for m in re.finditer(r"^includeConfig ", nf, flags=re.M)]
    assert includes, "expected top-level includeConfig statements in nextflow.config"
    early = [i for i in includes if i < params_at]
    assert not early, (
        f"{len(early)} includeConfig statement(s) appear before the params block. "
        "conf/modules.config reads params.max_forks eagerly (maxForks is not a dynamic "
        "directive), and before the params block a `params.*` reference is an empty "
        "ConfigObject -- `as int` on it fails the entire config parse."
    )


def test_the_includes_still_precede_the_process_and_executor_scopes(nf):
    """nextflow.config's own `process { }` and `executor { }` blocks are parsed after
    the includes, or conf/base.config's process defaults would win over them.
    """
    last_include = max(
        m.start() for m in re.finditer(r"^includeConfig ", nf, flags=re.M)
    )
    for scope in ("\nprocess {", "\nexecutor {"):
        assert nf.index(scope) > last_include, (
            f"nextflow.config's `{scope.strip()}` scope must come after the includeConfig "
            "statements, or conf/base.config's settings would override it instead of the "
            "other way round"
        )
