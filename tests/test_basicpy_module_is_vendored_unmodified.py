"""`modules/nf-core/basicpy/` is upstream's file, byte for byte. This is what says so.

`modules/nf-core/basicpy/MIRAGE-NOTES.md` says `main.nf` and `meta.yml` are
"byte-for-byte upstream; nothing here is patched" and cites this file as the guard.

`tests/test_basicpy_defaults_are_deliberate.py` pins the few properties mirage reasons
about; a digest covers the rest of the file in one line. It fails on any edit, including
an upstream re-sync: re-fetch, re-read the diff, update the digest, and re-check
MIRAGE-NOTES.md.

The digests are of the files as vendored from

    https://raw.githubusercontent.com/nf-core/modules/master/modules/nf-core/basicpy/main.nf
    https://raw.githubusercontent.com/nf-core/modules/master/modules/nf-core/basicpy/meta.yml

Re-verify with:

    curl -sSL <url> | shasum -a 256

This pins "unchanged since vendoring", not "identical to upstream today" (that needs
the network). The invocation-line assertion is the readable half, for triaging a
digest mismatch offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parent.parent / "modules" / "nf-core" / "basicpy"

# sha256 of the file as vendored. See the provenance note above before changing either.
VENDORED_DIGESTS = {
    "main.nf": "4563de4d07aca2c47f0b0a121dd06e3de4c9b94153a854d7f0636f7d494db025",
    "meta.yml": "836af09031cba3c0a3c922975cd658d31b302b8a4150ccfc38035d6054feaf8a",
}


@pytest.mark.parametrize("name,expected", sorted(VENDORED_DIGESTS.items()))
def test_vendored_file_is_unchanged_since_it_was_vendored(name, expected):
    path = MODULE_DIR / name
    assert path.is_file(), f"{path} is missing -- the module is no longer vendored"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"modules/nf-core/basicpy/{name} has been edited since it was vendored "
        f"(sha256 {actual}, expected {expected}). This directory is upstream's file and "
        "MIRAGE-NOTES.md says so. If this is a deliberate upstream re-sync, read the diff, "
        "update the digest in this test, and re-check MIRAGE-NOTES.md -- the notes "
        "describe upstream's behaviour and an upstream change may have invalidated them. "
        "If it is a local patch, do not: mirage's arguments belong in conf/modules.config, "
        "and its resources in a withName: block."
    )


def test_the_invocation_line_is_exactly_upstreams():
    """The readable half of the digest: what the container is actually told to run.

    The profile filenames come from `--output-flatfield` / `--output-darkfield`, and
    `subworkflows/local/preprocess.nf` joins BASICPY's output back to the tile stack's
    sidecar on the prefix those flags set.
    """
    text = (MODULE_DIR / "main.nf").read_text()
    assert (
        "/opt/main.py -i $image -o . "
        "--output-flatfield $prefix --output-darkfield $prefix $args"
    ) in text, (
        "the /opt/main.py invocation in the vendored module is not upstream's. mirage "
        "relies on both --output-* flags taking the SAME prefix (meta.id, the tile "
        "stack's simpleName), because subworkflows/local/preprocess.nf keys its join on "
        "it; and on $args being the only place arguments enter, which is what "
        "tests/test_basicpy_defaults_are_deliberate.py guards."
    )


def test_the_output_tuple_is_darkfield_then_flatfield():
    """Upstream's emit order, which APPLY_PROFILES' input tuple has to match.

    `tuple val(meta), path("*-dfp.ome.tif"), path("*-ffp.ome.tif")` -- dfp FIRST. Getting
    this the wrong way round downstream divides by the darkfield and subtracts the
    flatfield, silently, on every pixel. tests/modules/apply_profiles.nf.test pins the CLI
    binding at the other end of the chain, and bin/apply_basic_profiles.py's flatfield
    positivity check catches a swap semantically; this pins the source of the ordering.
    """
    text = (MODULE_DIR / "main.nf").read_text()
    assert (
        'tuple val(meta), path("*-dfp.ome.tif"), path("*-ffp.ome.tif"), emit: profiles'
        in text
    )
    assert text.index("*-dfp.ome.tif") < text.index("*-ffp.ome.tif")
