"""One string-aware model of this repo's Nextflow source.

Seven guards used to carry seven private regex parses of the same files, each
with a different blind spot -- so "the guard passed" said nothing about
whether it had read the file. See tests/test_nfmodel.py for the specific
constructs that defeated the old parses.

Guards assert against this model, never against raw text.
"""

from ._lex import (
    block_extent,
    skip_non_code,
    strip_comments,
    strip_comments_and_strings,
)
from ._model import (
    REPO_ROOT,
    NfTestCase,
    Process,
    WithNameBlock,
    include_aliases,
    nf_files,
    nf_test_cases,
    nf_test_files,
    param_refs,
    processes,
    script_bodies,
    with_name_blocks,
    workflows,
)

__all__ = [
    "REPO_ROOT",
    "NfTestCase",
    "Process",
    "WithNameBlock",
    "block_extent",
    "include_aliases",
    "nf_files",
    "nf_test_cases",
    "nf_test_files",
    "param_refs",
    "processes",
    "script_bodies",
    "skip_non_code",
    "strip_comments",
    "strip_comments_and_strings",
    "with_name_blocks",
    "workflows",
]
