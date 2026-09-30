"""ASHLAR was a benchmark baseline, not a production backend. v1.0.0 ships two:
valis and tiled. It survives on the `benchmarking` branch, which is the only place it
was ever used.

This guard is deliberately NARROW -- it checks the backend's own files and the
registration_method enum, NOT every occurrence of the word. Prose describing mcmicro
applying BaSiC profiles inside ASHLAR, and the provenance note on tile_residual's
correlation kernel, are correct and must survive; a blanket grep would delete them.

The nextflow.config check goes through `tests.nfmodel.strip_comments` rather than
raw text, per this repo's rule that a guard asserts against the model. That is not
ceremony here: the model view separates "the parameter is DECLARED" from "the string
appears in a comment", so the two get their own assertions and their own messages.
Both must be clean -- a leftover comment naming a parameter the schema now rejects is
its own defect -- but only the first is a broken pipeline.

Every check below whose passing answer is "zero" is paired with a positive assertion
on the same input, so an empty or mis-rooted scan fails loudly instead of passing
vacuously.

NARROW IS NOT THE SAME AS BLIND. The file/enum/params checks alone all stayed green
against a MEASURED break -- restoring both `choices=[..., "ashlar"]` and
`if a.method in ("tiled", "ashlar")` in bin/warp_seg_qc.py -- so the two live surfaces
that break reaches, the warp CLI and conf/modules.config's withName selectors, get
their own assertions below.
"""

import json
from pathlib import Path

from tests.nfmodel import strip_comments, with_name_blocks

REPO = Path(__file__).resolve().parent.parent

GONE = [
    "bin/ashlar_retile.py",
    "bin/ashlar_solve.py",
    "modules/local/ashlar_retile.nf",
    "modules/local/ashlar_solve.nf",
    "subworkflows/local/adapters/ashlar_adapter.nf",
    "tests/modules/ashlar_retile.nf.test",
    "tests/modules/ashlar_solve.nf.test",
    "tests/subworkflows/local/adapters/ashlar_adapter.nf.test",
    "tests/test_ashlar_retile.py",
    "tests/test_ashlar_retile_pixel_size.py",
    "tests/test_ashlar_solve.py",
]

# Surviving siblings of the deleted paths, one per directory GONE reaches into. They
# prove REPO points at a real checkout: without them "nothing is present" and "the
# root is wrong" are the same passing result.
STILL_HERE = [
    "bin/tiled_stitch.py",
    "modules/local/tiled_stitch.nf",
    "subworkflows/local/adapters/tiled_adapter.nf",
    "tests/modules/warp_seg_qc.nf.test",
    "tests/subworkflows/local/adapters/tiled_adapter.nf.test",
    "tests/test_slide_io_seam.py",
]


def test_ashlar_backend_files_are_deleted():
    missing_control = [p for p in STILL_HERE if not (REPO / p).exists()]
    assert not missing_control, (
        f"the control paths are gone too -- REPO ({REPO}) is not a mirage checkout, so "
        f"this test's 'nothing present' answer means nothing: {missing_control}"
    )
    still_here = [p for p in GONE if (REPO / p).exists()]
    assert not still_here, f"ashlar backend file(s) still present: {still_here}"


def test_registration_method_enum_matches_the_backend_table():
    """The enum has one owner: lib/RegBackends.groovy's BACKENDS table.

    This used to assert `enum == ["valis", "tiled"]` -- a second, independent
    statement of a fact the table already owns, which would have gone on passing
    while the table gained a third backend and the two silently disagreed. It now
    asks tests/test_reg_backends.py for the same list that file's own schema check
    uses, so retiring or adding a backend is one edit, in the table.

    The ASHLAR-specific property this file exists for is unchanged and still
    asserted above: the backend's own files are gone.
    """
    from tests.test_reg_backends import reg_backend_methods

    methods = reg_backend_methods()
    assert "ashlar" not in methods, (
        f"ashlar is back in lib/RegBackends.groovy's table: {methods}. It was a "
        "benchmark baseline, and it survives on the `benchmarking` branch, which is "
        "the only place it was ever used."
    )

    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    found = []

    def walk(node):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict) and "registration_method" in props:
            found.append(props["registration_method"].get("enum"))
        for v in node.values():
            walk(v)

    walk(schema)
    assert found, "registration_method not found in nextflow_schema.json at all"
    for enum in found:
        assert enum == methods, f"schema enum {enum} != RegBackends table {methods}"


def test_no_reg_ashlar_params_remain():
    raw = (REPO / "nextflow.config").read_text()
    code = strip_comments(raw)
    # Proof the stripped view still holds the params block: a sibling STARE parameter
    # that must survive. Without it, a strip that blanked the file would pass.
    assert "reg_tiled_tile" in code, (
        "reg_tiled_tile is missing from the comment-stripped nextflow.config -- the "
        "model view is empty or wrong, so the absence check below proves nothing"
    )
    assert "reg_ashlar_" not in code, "reg_ashlar_* params still DECLARED"
    assert "reg_ashlar_" not in raw, (
        "reg_ashlar_* no longer declared but still named in a nextflow.config comment; "
        "it documents a parameter the schema now rejects at launch"
    )

    schema_text = (REPO / "nextflow_schema.json").read_text()
    assert "reg_tiled_tile" in schema_text, (
        "nextflow_schema.json read back empty or wrong"
    )
    assert "reg_ashlar_" not in schema_text, "reg_ashlar_* still in the schema"


def test_the_warp_dispatch_surface_names_only_two_backends():
    """`bin/warp_seg_qc.py --method` is the backend's other live entry point.

    The three checks above all stay GREEN if someone restores
    `choices=["valis", "tiled", "ashlar"]` together with
    `if a.method in ("tiled", "ashlar")`. That is measured, not assumed: the pair was
    re-applied to this tree and this module reported `3 passed` before this test
    existed. A partial re-introduction of the CLI surface is the most plausible way the
    backend creeps back, so it gets an assertion of its own.

    Flat rather than model-based, deliberately: this is a Python script, not Nextflow
    source, and the name must be gone from the argparse choices, the help text and the
    dispatch comment alike. Every blind spot a smarter parse could have would make this
    MISS a hit, never accept one.
    """
    text = (REPO / "bin" / "warp_seg_qc.py").read_text()
    assert '"--method"' in text and '"tiled"' in text, (
        "bin/warp_seg_qc.py no longer carries a --method dispatch surface -- this test "
        "is reading the wrong file, so its 'no ashlar' answer proves nothing"
    )
    assert "ashlar" not in text.lower(), (
        "bin/warp_seg_qc.py still names the ashlar backend on its --method surface "
        "(argparse choices, help text or the tiled/valis dispatch)"
    )


def test_no_ashlar_withname_selector_survives_in_conf_modules_config():
    """No `withName:` block may select a process the ashlar backend owned.

    Queried through `tests.nfmodel.with_name_blocks`, which filters out a `withName:`
    that appears only inside a comment -- so this asserts on LIVE selectors, not on the
    word. ASHLAR_STITCH was an include ALIAS of TILED_STITCH declared once, in the
    deleted adapter; a surviving selector for it would configure a process no include
    creates, which Nextflow accepts silently.
    """
    names = sorted({n.strip() for b in with_name_blocks() for n in b.names})
    assert "TILED_STITCH" in names, (
        f"with_name_blocks() did not find TILED_STITCH -- it returned {len(names)} "
        "selectors, so the scan is empty or pointed at the wrong file and the absence "
        "check below proves nothing"
    )
    offenders = [n for n in names if n.upper().startswith("ASHLAR")]
    assert not offenders, (
        f"ashlar withName: selector(s) still in conf/modules.config: {offenders}"
    )
