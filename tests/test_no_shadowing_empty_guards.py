"""`ifEmpty { error ... }` erases the failure that actually happened.

A channel fed by upstream processes goes empty for two very different reasons: the input
genuinely had nothing in it, or **something upstream failed**. `ifEmpty` cannot tell them
apart, so an `error` raised from inside it reports the first while the second is what really
occurred.

With a failing process feeding `.ifEmpty { error "..." }`:

  * default `errorStrategy` (terminate): the real error prints, then the `ifEmpty` message,
    so the false diagnosis is the last line the user reads;
  * `errorStrategy 'ignore'` (used in `conf/modules.config`): the real failure is dropped
    and the output is exit 1 and one wrong sentence.

Emptiness checks belong up front: `CsvUtils.validateInputSemantics` rejects a samplesheet
with no reference row before any process is instantiated.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCOPE = ("workflows", "subworkflows", "modules")

# `ifEmpty {` ... up to the closing brace of that block. Nextflow blocks here are short; scan a
# small window rather than trying to balance braces with a regex.
_IF_EMPTY = re.compile(r"\bifEmpty\s*\{")
_WINDOW_LINES = 6


def _nf_sources():
    files = []
    for d in SCOPE:
        files.extend(sorted((REPO / d).rglob("*.nf")))
    assert files, "found no .nf sources to scan -- the glob is wrong, not the repo"
    return files


def _offending_blocks(text):
    """Yield (line_no, snippet) for every `ifEmpty {` whose block raises `error`."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not _IF_EMPTY.search(line):
            continue
        window = lines[i : i + _WINDOW_LINES]
        # stop at the end of the closure if it closes early
        block = []
        for w in window:
            block.append(w)
            if w.strip().startswith("}") and len(block) > 1:
                break
        joined = "\n".join(block)
        if re.search(r"(^|\W)error\s+[\"']", joined) or re.search(
            r"(^|\W)error\s*\(", joined
        ):
            yield i + 1, joined


def test_no_workflow_raises_error_from_inside_ifempty():
    offenders = []
    for f in _nf_sources():
        for line_no, snippet in _offending_blocks(f.read_text()):
            offenders.append(f"{f.relative_to(REPO)}:{line_no}\n{snippet}")

    assert not offenders, (
        "`ifEmpty { error ... }` fires on abort as well as on genuinely-empty input, and under "
        "conf/modules.config's errorStrategy 'ignore' it is the ONLY thing printed -- the real "
        "failure is erased. Validate up front (see CsvUtils.validateInputSemantics) instead:\n\n"
        + "\n\n".join(offenders)
    )


def test_the_scan_actually_looks_at_the_file_that_carried_this_bug():
    """A guard whose glob misses the interesting file checks nothing."""
    scanned = {f.relative_to(REPO).as_posix() for f in _nf_sources()}

    assert "subworkflows/local/segmentation.nf" in scanned


def test_the_detector_recognises_the_shape_it_is_meant_to_catch():
    """Pin the detector against a known offending snippet, so it cannot rot into a no-op."""
    reconstructed = (
        "    ch_references.ifEmpty {\n"
        '        error "No reference images found (is_reference=true). Cannot run segmentation."\n'
        "    }\n"
    )

    assert list(_offending_blocks(reconstructed))


def test_the_detector_does_not_flag_an_ifempty_that_supplies_a_default():
    """`ifEmpty([])` is the correct, common idiom and must stay legal."""
    benign = (
        "    artifactsOf(ch, 'seg_qc').collect(sort: true).ifEmpty([]),\n"
        "    other.ifEmpty { [] }\n"
    )

    assert not list(_offending_blocks(benign))
