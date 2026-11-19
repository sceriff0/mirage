"""A `path` input's arity is not fixed, and conf/*.config closures forgot it.

Nextflow binds a `path` input to a bare `Path` when the group holds exactly one
file and to a `List<Path>` when it holds more. Iterating the input directly
(`split_channels.collect { it.size() }`) on a one-file group walks the Path's name
elements and fails with "No such file or directory: channels".

The rule: inside a conf/modules.config closure, never iterate a name you did not bind
yourself. Bind a local with the `instanceof Collection` normalisation and iterate that.
The closure cannot know how many files it got, so the rule is blunt on purpose.
tests/modules/merge_and_pyramid.nf.test's one-channel case covers the one real site;
this guard covers the next closure written the same way.
"""

import re

from tests.nfmodel import with_name_blocks

# Receiver of a `.collect {` / `.each {` / `.any {` ... iteration.
_ITER = re.compile(
    r"\b([a-z_]\w*)\s*\.\s*(collect|each|find|findAll|any|every|sum|inject)\s*\{"
)
# Names bound inside the closure itself.
_LOCAL = re.compile(r"\bdef\s+([a-z_]\w*)")

# Bound by Nextflow in a process-scope closure and safe to iterate: none of
# these is a staged path whose arity varies.
SAFE_RECEIVERS = {"task", "params", "workflow", "meta", "it"}


def _closure_bodies():
    """Every `= { ... }` directive body in conf/modules.config, on the
    comment/string-stripped view so prose and quoted text cannot match."""
    for block in with_name_blocks():
        for m in re.finditer(r"=\s*\{", block.body):
            start = m.end()
            depth, i = 1, start
            while i < len(block.body) and depth:
                depth += (block.body[i] == "{") - (block.body[i] == "}")
                i += 1
            yield block.selector, block.start_line, block.body[start : i - 1]


def test_no_closure_iterates_an_unbound_name():
    offenders = []
    for selector, line, body in _closure_bodies():
        locals_ = set(_LOCAL.findall(body)) | SAFE_RECEIVERS
        for m in _ITER.finditer(body):
            receiver = m.group(1)
            if receiver not in locals_:
                offenders.append(
                    f"conf/modules.config:{line} withName: '{selector}' "
                    f"iterates `{receiver}.{m.group(2)}{{}}` without binding it "
                    f"first -- if `{receiver}` is a path input, Nextflow hands it "
                    f"a bare Path for a one-file group and the iteration walks "
                    f"the path's name elements instead of the files"
                )
    assert not offenders, "\n".join(offenders) + (
        "\n\nBind a local first and normalise while you are there:\n"
        "    def files = (x instanceof Collection) ? x : [x]"
    )


def test_the_scan_finds_iterations_to_check():
    """MERGE_AND_PYRAMID's memory closure iterates. If this finds nothing, the
    extractor is stale and the guard above is passing vacuously."""
    found = [
        (sel, line, m.group(0))
        for sel, line, body in _closure_bodies()
        for m in _ITER.finditer(body)
    ]
    assert found, "no closure iterations found in conf/modules.config"


def test_the_one_file_normalisation_is_still_present():
    """The positive half. If the `instanceof Collection` line is deleted, the
    guard above still passes -- `files` would simply be bound to something
    else -- so the normalisation itself has to be asserted, not just the shape
    of the code around it."""
    bodies = [b for _s, _l, b in _closure_bodies() if "split_channels" in b]
    assert bodies, "MERGE_AND_PYRAMID's memory closure no longer names split_channels"
    assert any("instanceof Collection" in b for b in bodies), (
        "MERGE_AND_PYRAMID's memory closure no longer normalises split_channels "
        "to a list. A one-channel patient aborts the run without it."
    )
