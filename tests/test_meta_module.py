"""The meta map is an interface. One module owns it.

Every producer builds meta through Meta.fromSamplesheetRow /
Meta.fromCheckpointRow (lib/Meta.groovy). Two producers with different key sets
make --start postprocessing diverge silently from a full run (wrong channels
split, QUANTIFY run twice, the pyramid's DAPI plane from an arbitrary slide).
"""

import re

from tests.nfmodel import REPO_ROOT, nf_files, strip_comments_and_strings

# A map literal assigned to something called `meta`, with patient_id in it.
_META_LITERAL = re.compile(r"\bmeta\s*=\s*\[[^\]]*patient_id\s*:", re.S)

# lib/Meta.groovy IS the constructor -- its own REQUIRED_KEYS-carrying literal is
# the thing every other producer is required to call instead.
#
# subworkflows/local/adapters/valis_adapter.nf is a permanent exemption: its one
# `meta = [patient_id: patient_id]` is REGISTER's control tuple for a fan-in call,
# carries none of Meta.REQUIRED_KEYS and never reaches an `emit:` (the per-slide
# metas travel separately as `all_metas`). The exemption is per file, so a second
# such literal in that file would not be caught.
ALLOWED = {"lib/Meta.groovy", "subworkflows/local/adapters/valis_adapter.nf"}


def test_only_meta_groovy_constructs_a_meta_map():
    offenders = []
    for f in nf_files():
        rel = f.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED:
            continue
        clean = strip_comments_and_strings(f.read_text())
        for m in _META_LITERAL.finditer(clean):
            line = clean.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{line}")
    assert not offenders, "meta maps built outside lib/Meta.groovy:\n  " + "\n  ".join(
        offenders
    )


def test_meta_declares_its_required_keys():
    src = (REPO_ROOT / "lib" / "Meta.groovy").read_text()
    assert "REQUIRED_KEYS" in src
    for key in (
        "patient_id",
        "id",
        "is_reference",
        "channels",
        "keep_channels",
        "channels_count",
    ):
        assert f"'{key}'" in src, f"{key} is not declared in REQUIRED_KEYS"
