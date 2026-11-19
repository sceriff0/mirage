"""The registration QC's before/after wiring, guarded end to end.

``GENERATE_REGISTRATION_QC`` takes a third (native) image input. Four things
can silently come apart, and none of them is visible to a stub run:

1. The process could lose the input while the Python still expects it.
2. The process could keep the input and stop passing it to the script --
   ``-stub`` never evaluates a ``script:`` block, so every stub test stays green.
   (``tests/modules/generate_registration_qc.nf.test``'s rendered-command case
   catches this at runtime; this is the cheap static twin of it.)
3. One of the two call sites -- ``registration.nf`` and ``add_cycle.nf`` -- could
   be updated and the other not.
4. ``conf/modules.config``'s memory closure could keep sizing the task on two
   inputs while it stages three -- a silent under-request that surfaces as an
   OOM on real data and never on a fixture.

Every assertion reads ``tests.nfmodel`` rather than a private regex
(``tests/test_nfmodel.py`` forbids new private parses), and reads the
comment-stripped view, so a comment mentioning ``--native`` cannot satisfy it.
"""

from __future__ import annotations

import re

from tests.nfmodel import REPO_ROOT, processes, strip_comments, with_name_blocks

PROCESS = "GENERATE_REGISTRATION_QC"
CALL_SITES = (
    "subworkflows/local/registration.nf",
    "subworkflows/local/add_cycle.nf",
)


def _proc():
    procs = processes()
    assert PROCESS in procs, f"{PROCESS} is no longer a process under modules/"
    return procs[PROCESS]


def _path_inputs(inputs_body: str) -> list[str]:
    """The bound name of every `path(...)` in an `input:` section.

    Reads the comment/string-blanked view, so a `path(foo)` written inside a
    comment is not counted. `stageAs:` is tolerated: `path(x, stageAs: 'd/*')`
    binds `x`.
    """
    return re.findall(r"\bpath\(\s*([A-Za-z_][A-Za-z0-9_]*)", inputs_body)


def test_the_process_takes_three_image_inputs():
    names = _path_inputs(_proc().inputs)
    assert names == ["registered", "native_image", "reference"], (
        f"{PROCESS}'s input tuple must be "
        "[meta, registered, native_image, reference] -- the spec's "
        "[meta, registered, native, reference] with `native` renamed because it "
        f"is a Groovy reserved word. Found: {names}"
    )


def test_the_script_passes_the_native_image_to_the_qc_script():
    # strip_comments, NOT the blanked view: the flag lives inside the script:
    # heredoc, i.e. inside a quoted literal, which the blanked view erases.
    body = strip_comments(_proc().script_body)
    assert "generate_registration_qc.py" in body
    assert "--native ${native_image}" in body, (
        "the rendered command must pass the native image; without it the process "
        "stages a third full-resolution image and ignores it, and no stub test "
        "can see the difference"
    )
    # strip_comments only knows Groovy comments; a `#` inside the script:
    # heredoc becomes a shell comment when rendered, so a commented-out
    # "--native ${native_image}" would still match. Reject any occurrence on a
    # shell-commented line.
    native_lines = [
        line for line in body.splitlines() if "--native ${native_image}" in line
    ]
    assert native_lines and all(
        not line.strip().startswith("#") for line in native_lines
    ), (
        "the --native flag is present in source text but shell-commented out "
        "(a leading '#' on its heredoc line); the rendered command silently "
        "drops it even though the text is still there"
    )


def test_both_qc_call_sites_build_a_four_element_tuple():
    for rel in CALL_SITES:
        text = strip_comments((REPO_ROOT / rel).read_text())
        assert f"{PROCESS}(ch_for_qc)" in text, (
            f"{rel} no longer calls {PROCESS}(ch_for_qc); if the channel was "
            "renamed, rename it here too rather than deleting the guard"
        )
        # The assignment that builds it must mention the native image. Read from
        # the assignment to the call, on the comment-stripped view, so a comment
        # naming `native` cannot satisfy this.
        start = text.index("ch_for_qc =")
        end = text.index(f"{PROCESS}(ch_for_qc)", start)
        assignment = text[start:end]

        # The tuple passed to the process is whatever the terminal `.map {}`
        # produces; an earlier link can still mention the native variable after
        # a later map dropped it. The terminal map is textually last in the
        # assignment, so check that one.
        last_map = assignment.rfind(".map {")
        assert last_map != -1, f"{rel}'s ch_for_qc assignment has no .map {{...}}"
        terminal = assignment[last_map:]
        arrow = terminal.index("->")
        bracket_start = terminal.index("[", arrow)
        bracket_end = terminal.index("]", bracket_start)
        elements = [
            e.strip() for e in terminal[bracket_start + 1 : bracket_end].split(",")
        ]

        assert len(elements) == 4, (
            f"{rel}'s terminal .map for ch_for_qc builds a {len(elements)}-element "
            f"tuple {elements}, not the [meta, registered, native, reference] "
            f"{PROCESS} expects"
        )
        assert any("nat" in e for e in elements), (
            f"{rel}'s ch_for_qc does not carry a native image. Both call sites "
            "must feed the before panel or one patient population silently gets "
            "the old single-panel figure."
        )


def test_the_memory_closure_sizes_every_staged_input():
    """A `path()` input that is staged but not summed is a silent under-request."""
    blocks = [b for b in with_name_blocks() if PROCESS in b.names]
    assert blocks, f"no withName: block selects {PROCESS}"

    memory_exprs = []
    for b in blocks:
        body = strip_comments(b.raw_body)
        for m in re.finditer(r"\bmemory\s*=", body):
            memory_exprs.append(body[m.end() :])
    assert memory_exprs, f"{PROCESS}'s withName: block declares no memory"

    joined = "\n".join(memory_exprs)
    for name in _path_inputs(_proc().inputs):
        assert f"{name}.size()" in joined, (
            f"{PROCESS} stages `{name}` but its memory closure never sizes it. "
            "The tier is keyed on input bytes; an unsized input means the task "
            "reads more than it asked for and OOMs only on real data."
        )
