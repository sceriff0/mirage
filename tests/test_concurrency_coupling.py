# max_forks and queue_size are only meaningful together -- the lower binds.
#
# They shipped as two independent params with a 5/20 default and two in-tree
# comments both claiming 100, so the effective cluster throughput was ~4x lower
# than the documentation said and nobody could see it from either knob alone.
#
# Assertions run against tests.nfmodel's comment/string-stripped views, not raw
# text -- tests/test_nfmodel.py's anti-vacuity guard requires every new guard
# that reads Nextflow/Groovy source to query the shared model rather than
# re-implementing its own private parse, and a naive regex over raw text can be
# fooled by a matching comment the same way the four guards documented there
# were.
import re

from tests.nfmodel import REPO_ROOT, strip_comments, with_name_blocks

CFG = strip_comments((REPO_ROOT / "nextflow.config").read_text())


def test_the_individual_caps_are_null_declared():
    # A numeric default here would be computed BEFORE the CLI is applied, so
    # --concurrency would be silently ignored. Null-declared, then derived in
    # the executor/process scopes, is the only ordering that works.
    for name in ("max_forks", "queue_size"):
        assert re.search(rf"^\s*{name}\s*=\s*null", CFG, re.M), (
            f"{name} must be declared null and derived from concurrency"
        )


def test_concurrency_is_declared_with_the_shipped_default():
    assert re.search(r"^\s*concurrency\s*=\s*5\b", CFG, re.M)


def test_concurrency_drives_both_scopes():
    assert re.search(r"queueSize\s*=.*concurrency", CFG), "queueSize is not derived"
    assert re.search(r"maxForks\s*=.*concurrency", CFG), "maxForks is not derived"


# The exact shape every per-process maxForks override must take. Only the integer
# cap may vary -- everything else, including whitespace collapsed to single spaces,
# is fixed. Pinning the SHAPE, not merely which substrings appear in the assignment,
# is the point: `Math.min(10, params.max_forks as int)` -- the bare pre-concurrency
# form -- contains the substring "params.max_forks" and no "?:", so a substring-only
# check cannot tell it apart from the null-tested form. Once params.max_forks is
# null-declared, that bare form throws inside Math.min ONLY when the closure
# actually runs (Nextflow's per-process maxForks is not a dynamic directive) --
# invisible to -stub, invisible to this whole pytest suite, surfacing only against
# a real cluster run on a real slide.
_CANONICAL_MAX_FORKS_RE = re.compile(
    r"^Math\.min\(\s*\d+\s*,\s*"
    r"\(\s*params\.max_forks\s*!=\s*null\s*\?\s*params\.max_forks\s*:\s*params\.concurrency\s*\)"
    r"\s*as\s*int\s*\)$"
)


def test_no_per_process_cap_reads_the_bare_param():
    # params.max_forks is null unless explicitly set, so a bare read yields null
    # and Math.min throws at closure-run time -- the silent-resolution failure
    # mode this repo keeps rediscovering.
    #
    # The per-process caps live in nextflow.config's post-profiles "Concurrency"
    # block since 2026-09-09 (a scalar in conf/modules.config froze before any
    # profile ran -- tests/test_frozen_config_params.py). with_name_blocks() only
    # models conf/modules.config, so the caps are read off the comment-stripped
    # nextflow.config here, and modules.config is asserted to hold NONE.
    offenders = []
    found_any = False
    for selector, expr in re.findall(
        r"withName:\s*'(\w+)'\s*\{\s*maxForks\s*=\s*(.+?)\s*\}", CFG
    ):
        found_any = True
        normalised = re.sub(r"\s+", " ", expr.strip())
        if not _CANONICAL_MAX_FORKS_RE.match(normalised):
            offenders.append(f"{selector}: {normalised!r}")
    stray = [
        b.selector
        for b in with_name_blocks()
        if re.search(r"^\s*maxForks\s*=", b.body, re.M)
    ]
    assert not stray, (
        f"conf/modules.config must not assign maxForks (froze before profiles): {stray}"
    )
    assert found_any, (
        "expected per-process maxForks overrides in nextflow.config's concurrency block"
    )
    assert not offenders, (
        "per-process maxForks must be exactly `Math.min(<cap>, (params.max_forks != "
        "null ? params.max_forks : params.concurrency) as int)` (only <cap> may vary "
        "-- a bare `params.max_forks as int` or an `?:` fallback both silently break "
        "once max_forks is null). Offenders:\n  " + "\n  ".join(offenders)
    )
