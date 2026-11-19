"""subworkflows/local/checkpoint_writer.nf is the only subworkflow that writes a manifest.

The chain -- filter on Checkpoint.writesAtLevel, map through Checkpoint.row, collectFile
with the step's name, seed, sort and storeDir -- lives there once. A copy that loses the
gate writes a manifest naming unpublished artifacts (which --start and add_cycle then
open); one that loses the sort is non-deterministic; one that loses the seed has no
header and misparses under splitCsv(header: true).

Two properties, asserted separately because they break independently:

1. Only checkpoint_writer.nf may hold a `collectFile(` whose `storeDir:` is
   Layout.checkpointDir. A second one is a second writer.
2. Only checkpoint_writer.nf, among subworkflows/, may read
   `params.cleanup_level`: that is the gate, and a second reader is a second gate
   that can disagree.

`params.outdir` is not part of this rule: the row builders must call
`Layout.publishedPath(params.outdir, <pid>, '<kind>', file)` with literal arguments,
because tests/test_layout.py's static scan matches exactly that shape.

Source is read through tests.nfmodel (tests/test_nfmodel.py requires it), on the
string-preserving comment-stripped view: the needles are code, and a comment naming
one must not count.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import strip_comments

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBWORKFLOWS = REPO_ROOT / "subworkflows"

THE_WRITER = SUBWORKFLOWS / "local" / "checkpoint_writer.nf"

#: A collectFile whose storeDir is the checkpoint directory. Matched across the
#: intervening arguments (name:, newLine:, sort:, seed: may appear in any order), which
#: is why this is one regex over the block rather than two independent substring checks
#: -- a file may legitimately hold a collectFile for something else entirely.
_CHECKPOINT_COLLECTFILE = re.compile(
    r"collectFile\((?:[^()]|\([^()]*\))*storeDir\s*:\s*Layout\.checkpointDir",
    re.S,
)

_CLEANUP_LEVEL = re.compile(r"\bparams\.cleanup_level\b")


def _subworkflow_files():
    return sorted(SUBWORKFLOWS.rglob("*.nf"))


def test_the_scan_actually_reads_subworkflows():
    """Three ways this guard could pass while checking nothing: an empty file list,
    a mis-rooted tree, or a writer that no longer writes."""
    files = _subworkflow_files()
    assert len(files) >= 10, (
        f"only {len(files)} subworkflow file(s) under {SUBWORKFLOWS} -- the scan is "
        "mis-rooted, not the tree"
    )
    assert THE_WRITER in files, f"{THE_WRITER} is not in the scanned set"


def test_the_writer_really_writes():
    """The sole exemption must be real. If checkpoint_writer.nf stopped
    holding the collectFile, every 'no writer here' answer below would be
    vacuously true and the pipeline would write no manifests at all."""
    body = strip_comments(THE_WRITER.read_text())
    assert _CHECKPOINT_COLLECTFILE.search(body), (
        f"{THE_WRITER.relative_to(REPO_ROOT)} holds no collectFile writing into "
        "Layout.checkpointDir. It is the only file allowed to; if the write moved, "
        "this guard is pointing at the wrong owner."
    )
    assert _CLEANUP_LEVEL.search(body), (
        f"{THE_WRITER.relative_to(REPO_ROOT)} no longer reads params.cleanup_level, so "
        "the cleanup gate is gone: a run at a cleaning level would write a manifest "
        "whose every path dangles."
    )


def test_only_checkpoint_writer_collects_a_manifest():
    offenders = []
    for path in _subworkflow_files():
        if path == THE_WRITER:
            continue
        if _CHECKPOINT_COLLECTFILE.search(strip_comments(path.read_text())):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert not offenders, (
        "these subworkflows collectFile a checkpoint manifest themselves: "
        + ", ".join(offenders)
        + ". subworkflows/local/checkpoint_writer.nf is the only writer -- call "
        "CHECKPOINT_WRITER(<step>, <channel of row Maps>) instead. A second copy of "
        "the chain is a second chance to lose the cleanup gate, the sort or the seed, "
        "and every one of those losses is silent."
    )


def test_only_checkpoint_writer_reads_the_cleanup_level():
    offenders = []
    for path in _subworkflow_files():
        if path == THE_WRITER:
            continue
        if _CLEANUP_LEVEL.search(strip_comments(path.read_text())):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert not offenders, (
        "these subworkflows read params.cleanup_level: "
        + ", ".join(offenders)
        + ". The manifest gate lives in checkpoint_writer.nf alone. A second read is "
        "either a gate that can disagree with it, or the same gate applied twice -- "
        "which reads as defence in depth and is actually two places to forget. "
        "(workflows/mirage.nf reads it for a DIFFERENT purpose -- warning about a "
        "--start that re-enters a tree this level does not publish -- and is out of "
        "this rule's scope.)"
    )


def test_the_comment_stripped_view_is_what_is_read():
    """A comment mentioning params.cleanup_level or the checkpoint collectFile must
    not count as code."""
    sample = "\n".join(
        [
            "workflow X {",
            "    // params.cleanup_level is read in checkpoint_writer.nf, not here",
            "    /* collectFile(storeDir: Layout.checkpointDir(params.outdir)) */",
            "    main:",
            "    ch = Channel.empty()",
            "}",
        ]
    )
    body = strip_comments(sample)
    assert not _CLEANUP_LEVEL.search(body)
    assert not _CHECKPOINT_COLLECTFILE.search(body)


def test_every_checkpoint_step_has_exactly_one_commissioning_call_site():
    """The other direction: the rule above forbids a second writer, and this one
    forbids a step quietly losing its only one. A step no file commissions is a
    manifest that is never written, which fails at the NEXT run's launch (--start
    or add_cycle refusing) rather than at this one's."""
    call = re.compile(r"CHECKPOINT_WRITER\s*\(\s*Layout\.(\w+)")
    commissioned = set()
    for path in _subworkflow_files():
        commissioned.update(call.findall(strip_comments(path.read_text())))

    assert commissioned == {
        "PREPROCESSED",
        "REGISTERED",
        "SEGMENTED",
        "POSTPROCESSED",
    }, (
        f"CHECKPOINT_WRITER is commissioned for {sorted(commissioned)}; every one of "
        "the four checkpoint steps must have a call site, and no step outside them."
    )
