"""No publishDir in conf/modules.config sets `overwrite: true`.

Nextflow's default already overwrites the output of a task that re-runs. What
`overwrite: true` adds is re-copying the outputs of cached tasks on every `-resume`,
which rewrites each published file's mtime. Anything that reads a published file as a
task input (e.g. a `--start` run from a checkpoint CSV) hashes path + size + mtime, so
it would lose its cache, and every whole-slide image would be re-copied.
"""

from __future__ import annotations

from tests.nfmodel import strip_comments, with_name_blocks


def test_no_publish_rule_overwrites_cached_outputs():
    hits = [
        f"withName: {block.selector}"
        for block in with_name_blocks()
        # strings and code kept, comments removed: the option is code, and a comment
        # explaining why it is gone must not trip the guard
        if "overwrite: true"
        in strip_comments(block.raw_body).replace("overwrite:true", "overwrite: true")
    ]
    assert not hits, (
        "publishDir overwrite: true re-copies cached outputs on resume:\n"
        + "\n".join(hits)
    )
