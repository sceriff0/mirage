"""A `withName:` clusterOptions assignment REPLACES the profile's, it does not
append.

nextflow.config's `slurm` profile sets `process.clusterOptions` to the site's
`--account`/`--qos`. A process-level `clusterOptions` (SEGMENT's GPU request)
overrides that outright, so it must restate the site settings.

It cannot compose: inside a clusterOptions closure `task.clusterOptions` is the
closure itself and recurses to a StackOverflowError. And conf/*.config cannot share
a helper, so the account/qos clause is inlined. This file guards the two copies
against drifting apart.
"""

import re

import pytest

from tests.nfmodel import REPO_ROOT, strip_comments, with_name_blocks

# The params the site profile's derivation reads. Not restated as a constant
# that could go stale -- read from nextflow.config below.
_PARAM = re.compile(r"params\.(slurm_\w+)")


def _profile_cluster_options_body():
    """The `slurm` profile's process.clusterOptions closure body.

    Read off the comments-blanked, strings-intact view: the closure interpolates
    params inside quoted flags ("--account=${params.slurm_account}"), which the
    fully-stripped view would blank away.
    """
    src = strip_comments((REPO_ROOT / "nextflow.config").read_text())
    m = re.search(r"process\.clusterOptions\s*=\s*\{", src)
    assert m, "nextflow.config no longer sets process.clusterOptions in a profile"
    start = m.end()
    depth, i = 1, start
    while i < len(src) and depth:
        depth += (src[i] == "{") - (src[i] == "}")
        i += 1
    return src[start : i - 1]


def _process_cluster_options():
    """(selector, line, body) for every withName: clusterOptions closure."""
    out = []
    for block in with_name_blocks():
        commentless = strip_comments(block.raw_body)
        for m in re.finditer(r"clusterOptions\s*=\s*\{", block.body):
            start = m.end()
            depth, i = 1, start
            while i < len(block.body) and depth:
                depth += (block.body[i] == "{") - (block.body[i] == "}")
                i += 1
            out.append((block.selector, block.start_line, commentless[start : i - 1]))
    return out


def test_no_process_level_cluster_options_drops_the_site_settings():
    """Every withName: clusterOptions must reproduce every param the profile's
    derivation reads. Asserting the PARAMS rather than the text lets the two
    closures be written differently -- they only have to agree about what the
    site controls."""
    expected = set(_PARAM.findall(_profile_cluster_options_body()))
    assert expected, (
        "the slurm profile's clusterOptions reads no params -- either it stopped "
        "carrying the site settings, or this scan is reading a blanked view"
    )
    offenders = []
    for selector, line, body in _process_cluster_options():
        missing = expected - set(_PARAM.findall(body))
        if missing:
            offenders.append(
                f"conf/modules.config:{line} withName: '{selector}': "
                f"clusterOptions overrides the slurm profile's and does not "
                f"reproduce {sorted(missing)} -- those processes are submitted "
                f"without the site's account/qos"
            )
    assert not offenders, "\n".join(offenders)


def test_no_cluster_options_closure_reads_its_own_directive():
    """`task.clusterOptions` inside a clusterOptions closure is self-referential
    and recurses to a StackOverflowError -- a crash, not a fallback."""
    offenders = [
        f"conf/modules.config:{line} withName: '{selector}' reads "
        f"task.clusterOptions from inside its own clusterOptions closure -- "
        f"self-referential, recurses, StackOverflowError"
        for selector, line, body in _process_cluster_options()
        if "task.clusterOptions" in body
    ]
    assert not offenders, "\n".join(offenders)


def test_recursion_scan_ignores_a_comment_but_catches_real_code():
    """Regression guard on using ONE view (`strip_comments`: comments blanked,
    strings intact) for both this file's checks, rather than a second,
    strings-blanked view for the recursion check specifically.

    A comment in conf/modules.config names `task.clusterOptions` in prose, so the
    recursion check must not read comments; the site-settings check needs string
    contents. `strip_comments` does both. Demonstrated on synthetic text.
    """
    commented = (
        "withName: 'FOO' {\n"
        "    // reading task.clusterOptions here would recurse\n"
        '    clusterOptions = { "--account=${params.slurm_account}" }\n'
        "}\n"
    )
    live = "withName: 'FOO' {\n    clusterOptions = { task.clusterOptions + ' --extra' }\n}\n"

    assert "task.clusterOptions" in commented  # present in the raw text...
    assert "task.clusterOptions" not in strip_comments(commented)  # ...but blanked
    assert "task.clusterOptions" in strip_comments(live)  # real code survives


def test_the_scan_finds_the_gpu_block():
    """SEGMENT sets clusterOptions. If this finds nothing, the extractor is
    stale and both checks above are vacuous."""
    found = _process_cluster_options()
    assert found, "no withName: clusterOptions closures found in conf/modules.config"
    assert any("gres" in body for _s, _l, body in found), (
        "no withName: clusterOptions asks for a --gres allocation any more -- "
        "the block these guards were written for is gone"
    )


# Each site SLURM knob's consumer inside nextflow.config's `slurm` profile:
# `slurm_qos` feeds the `process.clusterOptions` closure; `slurm_partition` drives
# `process.queue`.
_SLURM_SITE_PARAM_CONSUMERS = {
    "slurm_partition": "queue",
    "slurm_qos": "clusterOptions",
}


@pytest.mark.parametrize("param,consumer", sorted(_SLURM_SITE_PARAM_CONSUMERS.items()))
def test_slurm_site_param_is_named_and_reaches_its_consumer(param, consumer):
    """`slurm_partition` and `slurm_qos` must each have a declared default in
    nextflow.config's top-level params block, be offered as a site knob in
    `conf/site.config.template`, and be referenced by name inside
    nextflow.config's `slurm` profile.

    The template only sets the params (both `null`); the consumers live in
    nextflow.config's `slurm` profile, read via `tests.nfmodel.strip_comments`.
    """
    nextflow_cfg = strip_comments((REPO_ROOT / "nextflow.config").read_text())
    site_template = strip_comments(
        (REPO_ROOT / "conf" / "site.config.template").read_text()
    )

    assert re.search(rf"\b{param}\s*=", nextflow_cfg), (
        f"{param} has no default declaration in nextflow.config's params block"
    )
    assert re.search(rf"\b{param}\s*=", site_template), (
        f"{param} is not offered as a site knob in conf/site.config.template"
    )

    if consumer == "clusterOptions":
        closure_body = _profile_cluster_options_body()
        assert f"params.{param}" in closure_body, (
            f"{param} is declared but never referenced inside nextflow.config's "
            f"`slurm` profile's process.clusterOptions closure"
        )
    else:
        # A closure, so it is read per task after every -c file is merged; a plain
        # `params.x` is frozen while nextflow.config is parsed and a site.config pin is lost.
        assert re.search(
            rf"process\.queue\s*=\s*\{{\s*params\.{param}\s*\}}", nextflow_cfg
        ), (
            f"{param} must reach process.queue through a closure "
            f"(`process.queue = {{ params.{param} }}`) in nextflow.config's `slurm` profile"
        )
