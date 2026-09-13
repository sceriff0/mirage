"""A `withName:` clusterOptions assignment REPLACES the profile's, it does not
append.

nextflow.config's `slurm` profile sets `process.clusterOptions` to the site's
`--account`/`--qos`. conf/modules.config then assigned SEGMENT its own
`clusterOptions = { params.seg_gpu ? "--gres=gpu:..." : '' }`, which overrode
that outright -- so every process on the cluster carried the account except the
single GPU process, which is the one a scheduler is most likely to reject
without it.

WHY THE FIX IS A DUPLICATED DERIVATION AND NOT A COMPOSITION.

The obvious fix reads the profile value and appends to it:

    clusterOptions = { (task.clusterOptions ?: '') + ' --gres=...' }

It does not work. Inside a clusterOptions closure, `task.clusterOptions` IS
this closure -- verified 2026-08-25 on NXF_VER=26.04.6, where it does not fall
back to the profile value but recurses: java.lang.StackOverflowError, the task
fails, the run aborts. Shipping it would have replaced a wrong `--account` with
a crashed GPU path.

And there is nowhere shared to put the derivation: conf/*.config cannot see
lib/*.groovy, and NF26's strict parser forbids a helper shared between blocks.
So the account/qos clause is inlined -- the same forced duplication as the
RegPresets tables -- and what this file guards is the thing duplication
actually costs: the two copies drifting apart, silently, because neither one
mentions the other at runtime.
"""

import re

import pytest

from tests.nfmodel import REPO_ROOT, strip_comments, with_name_blocks

# The params the site profile's derivation reads. Not restated as a constant
# that could go stale -- read from nextflow.config below.
_PARAM = re.compile(r"params\.(slurm_\w+)")


def _profile_cluster_options_body():
    """The `slurm` profile's process.clusterOptions closure body.

    Read off the comments-blanked, strings-INTACT view: the closure interpolates
    params inside quoted flags ("--account=${params.slurm_account}"), which the
    fully-stripped view blanks away -- a scan on that view would find no params
    at all and this whole file would compare two empty sets and pass.
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
    and recurses to a StackOverflowError -- a crash, not a fallback. It reads
    like the obvious composition, so it is worth failing loudly rather than at
    the first GPU submission."""
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

    `_process_cluster_options()` extracts each closure body through
    `strip_comments`, and both `test_no_process_level_cluster_options_drops_the_site_settings`
    (needs string contents, since the composed `--account=${params.slurm_account}`
    text lives inside a quoted literal) and this file's recursion check read
    that same body. That is only safe because SEGMENT's own explanatory
    comment in conf/modules.config literally contains the string
    `task.clusterOptions` in prose (documenting the exact trap this guard
    exists to catch) -- if the recursion check read a view that kept comments
    intact, it would trip on its own documentation. `strip_comments` blanks
    comments (so the prose mention vanishes) while keeping strings (so a real
    reference, including one hidden inside a GString interpolation, survives).
    Demonstrated here on synthetic text mirroring the real shape, so a future
    change to `_process_cluster_options()`'s view choice fails a fast, direct
    test rather than only the slow, indirect one against the real file.
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


# Each site SLURM knob's real consumer inside nextflow.config's `slurm` profile.
# `slurm_qos` (with `slurm_account`, not named by this task) feeds the composed
# `process.clusterOptions` closure this file already parses above. `slurm_partition`
# does NOT: it drives a separate scalar, `process.queue = params.slurm_partition`,
# two lines above that closure -- a real distinction, not an oversight, so this
# test checks each param against what it actually reaches rather than assuming
# both funnel through clusterOptions.
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

    conf/site.config.template does NOT itself define a `clusterOptions`
    closure (or a `process.queue` assignment) -- it only sets the params
    (both `null`, "leave null when not using SLURM"); the closures that
    consume them live in nextflow.config's `slurm` profile, read here via
    `Path.read_text()` and `tests.nfmodel.strip_comments` rather than a
    private parse.

    This is a config reader, not a Nextflow-source parser (nf-test's own
    `.nf.test` files and the `modules/*.nf` process bodies are out of scope
    here), so it is exempted from `tests.nfmodel`'s block/closure extraction
    only for the `process.queue` half of the check -- the `clusterOptions`
    half reuses this file's own `_profile_cluster_options_body()` rather than
    re-parsing the closure a second way.
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
        assert re.search(rf"process\.queue\s*=\s*params\.{param}\b", nextflow_cfg), (
            f"{param} is declared but never wired to process.queue in "
            f"nextflow.config's `slurm` profile"
        )
