"""Every process has a module-level nf-test, and every process whose rendered
command carries a tunable has at least one nf-test case that RENDERS it in the
blocking gate.

Why two rules. `-stub` never evaluates a `script:` block (CLAUDE.md,
"Verification reality" 1), so a stub-only test of a process that reads
`params.*` in its script or takes `ext.args` from conf/modules.config proves
its channel wiring and nothing about the flags it passes. CI's gate is
`nf-test test --tag stub`, so a rendered case tagged only `real` runs in
nightly.yml on the default branch and nowhere else. The 2026-09-10 audit
found 5 processes with no module test and 11 tunable processes with no gated
rendered case; the 2026-09-10 build-out closed all sixteen, so KNOWN_GAPS was
left empty and the list can only shrink. On `dev` it is not empty: it holds the
dev-only EXTRACT_MASK_SERIES, which add_cycle owns and the build-out -- run on
`main`, which carries neither -- never saw, until
tests/modules/extract_mask_series.nf.test exists.

All parsing goes through tests.nfmodel (test_nfmodel.py forbids a private one).
Tunables are read off `strip_comments(raw)` views. `ext.args` is an identifier,
so the blanked `body` view would find that one too; the `params.` scan is the
one that needs strings kept, because the rendered command -- and every
`params.x` inside it -- lives in the module's quoted script string.
"""

from __future__ import annotations

import re

import pytest

from tests.nfmodel import nf_test_cases, processes, strip_comments, with_name_blocks

# Shrink-only. Each entry is a process the audit found short on 2026-09-10 and
# the reason. Delete the entry in the same commit that closes the gap;
# test_known_gaps_only_shrink fails on an entry that is no longer a gap.
# Empty is the goal state, and the dict is kept rather than deleted so that a
# future entry re-arms test_known_gaps_only_shrink; shrink-only is enforced by
# review (nothing here can tell a new entry from a re-added old one).
KNOWN_GAPS = {
    # dev only. EXTRACT_MASK_SERIES is add_cycle's process, and add_cycle lives on
    # `dev` (CLAUDE.md, "Branch model"), so the 2026-09-10 audit and build-out --
    # which ran on `main` -- never saw it. It has no module-level nf-test of its
    # own; tests/subworkflows/add_cycle.nf.test asserts it RUNS inside ADD_CYCLE
    # (`trace.tasks().findAll { it.name.contains('EXTRACT_MASK_SERIES') }.size() == 1`)
    # but names no `process` directive, which is what this guard counts. Not
    # tunable: its withName: block sets no ext.args and its script reads no
    # params., so only the first rule bites. Listed 2026-09-13 as the honest
    # state, not waived -- closing it means adding tests/modules/extract_mask_series.nf.test.
    "EXTRACT_MASK_SERIES": "no module test (dev-only process, add_cycle)",
}


def _ext_args_processes() -> set:
    names = set()
    for block in with_name_blocks():
        if re.search(r"\bext\.args\b", strip_comments(block.raw_body)):
            names.update(block.names)
    return names


def _is_tunable(proc, ext_args_names: set) -> bool:
    return proc.name in ext_args_names or "params." in strip_comments(proc.script_body)


def _module_cases(name: str):
    return [c for c in nf_test_cases() if c.process == name]


def _gated_rendered_cases(name: str):
    """Cases that run in CI's gate AND actually look at the rendered command.

    `not c.stub_option` alone is not enough: a pure failure case (SEG_QUALITY_EVAL's
    "refuse a meta without pixel_size", TILED_REG_TILE's "refuse a panel with no
    nuclear channel") also runs without `-stub`, asserts `process.failed`, and
    never reads `.command.sh` -- so it proves nothing about the flags."""
    return [
        c
        for c in _module_cases(name)
        if "stub" in c.tags and not c.stub_option and c.reads_command_sh
    ]


def _gap_reason(name: str) -> str | None:
    """None if `name` is fully covered, else the rule it breaks."""
    proc = processes()[name]
    if not _module_cases(name):
        return "no module-level nf-test names it in a `process` directive"
    if _is_tunable(proc, _ext_args_processes()) and not _gated_rendered_cases(name):
        return "rendered command carries a tunable but no stub-tagged case runs without -stub"
    return None


@pytest.mark.parametrize("name", sorted(processes()))
def test_process_is_covered_or_is_listed_debt(name):
    reason = _gap_reason(name)
    if reason is None:
        return
    assert name in KNOWN_GAPS, (
        f"{name}: {reason}. Add a module test (and a rendered case if tunable)."
    )


def test_known_gaps_only_shrink():
    stale = [
        f"{n}: listed as {r!r} but is no longer a gap -- delete the entry"
        for n, r in KNOWN_GAPS.items()
        if _gap_reason(n) is None
    ]
    missing = [n for n in KNOWN_GAPS if n not in processes()]
    assert not stale and not missing, "\n".join(
        stale + [f"{n}: no such process" for n in missing]
    )


def test_the_scan_actually_finds_tunables():
    """A guard that finds nothing passes on everything. Pin three known tunables
    of each kind so a regex slip is loud."""
    ext = _ext_args_processes()
    assert {"SEGMENT", "QUANTIFY", "EXPORT_GEOJSON"} <= ext, ext
    script_params = {
        n for n, p in processes().items() if "params." in strip_comments(p.script_body)
    }
    assert {"REGISTER", "PREFLIGHT_SCALE", "MERGE_AND_PYRAMID"} <= script_params, (
        script_params
    )
