#!/usr/bin/env python3
"""Single-ownership guards for lib/Layout.groovy — the output-layout seam.

`lib/Layout.groovy` is the single owner of where published files land. What it
constructs is asserted by tests/layout.nf.test; this file asserts that no path
template is restated elsewhere and that every `kind` Layout is asked for is one
conf/modules.config publishes into. Plain pytest, no Nextflow.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.nfmodel import block_extent as _block_extent
from tests.nfmodel import strip_comments as _strip_comments
from tests.nfmodel import strip_comments_and_strings as _strip_comments_and_strings

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = ROOT / "lib" / "Layout.groovy"
MODULES_CONFIG = ROOT / "conf" / "modules.config"


# Every Groovy/Nextflow file that could restate the layout. Layout.groovy itself is
# the one place allowed to, and conf/modules.config is the publish authority Layout
# mirrors — both are excluded from the "must not restate it" sweep.
def _callers() -> list[Path]:
    files = [ROOT / "main.nf"]
    files += sorted((ROOT / "workflows").rglob("*.nf"))
    files += sorted((ROOT / "subworkflows").rglob("*.nf"))
    files += sorted(p for p in (ROOT / "lib").glob("*.groovy") if p != LAYOUT)
    return [f for f in files if f.exists()]


def _sweep_files() -> list[Path]:
    """_callers() plus modules/: a restated path template is as wrong in a process."""
    files = _callers() + sorted((ROOT / "modules").rglob("*.nf"))
    return [f for f in files if f.exists()]


def test_layout_exists_and_is_all_static():
    """G4: lib/*.groovy classes hold only static methods — no instance state."""
    text = LAYOUT.read_text()
    assert "class Layout {" in text

    # Every method declaration inside the class must be static. Matches `<mods> Type
    # name(` at method-declaration indentation, skipping the `class` line itself.
    methods = re.findall(
        r"^    ((?:private |protected |public )?)(\w[\w<>,. \[\]]*?) (\w+)\(",
        text,
        re.M,
    )
    assert methods, "no methods found — did the class layout change?"
    for mods, rtype, name in methods:
        decl = f"{mods}{rtype} {name}("
        assert "static" in rtype or "static" in mods, (
            f"Layout.{name} is not static: {decl}"
        )


def test_layout_never_reads_params():
    """Layout takes outdir as an argument; a static class reaching into `params`
    is untestable and unsafe from onComplete / lib contexts."""
    for i, line in enumerate(LAYOUT.read_text().splitlines(), 1):
        code = line.split("//")[0]
        if code.lstrip().startswith("*"):
            continue
        # Strip string literals: the error messages legitimately NAME params.outdir
        # as the thing the caller should pass in.
        code = re.sub(r"\"[^\"]*\"|'[^']*'", "", code)
        assert "params." not in code, (
            f"lib/Layout.groovy:{i} reads params — pass it in instead"
        )


def test_no_caller_restates_a_per_patient_published_path():
    """`<outdir>/<patient_id>/<kind>` belongs to Layout.patientDir alone."""
    # e.g. "${params.outdir}/${meta.patient_id}/registered"
    # Any `${...outdir}/${<anything>}` is a per-patient template, whatever the spelling.
    template = re.compile(r"\$\{params\.(?:prior_)?outdir\}/\$\{[^}]+\}")
    offenders = [
        f"{f.relative_to(ROOT)}:{i}"
        for f in _sweep_files()
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if template.search(line)
    ]
    assert not offenders, (
        "per-patient published-path template restated outside lib/Layout.groovy: "
        + ", ".join(offenders)
        + " — call Layout.patientDir/publishedPath instead"
    )


def test_no_caller_restates_a_checkpoint_csv_path():
    """`csv/<step>.csv` belongs to Layout.checkpointCsv* alone."""
    literal = re.compile(r"""csv/(preprocessed|registered|postprocessed)\.csv""")
    offenders = [
        f"{f.relative_to(ROOT)}:{i}"
        for f in _sweep_files()
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if literal.search(line) and not line.lstrip().startswith(("*", "//"))
    ]
    assert not offenders, (
        "checkpoint-CSV path restated outside lib/Layout.groovy: "
        + ", ".join(offenders)
        + " — call Layout.checkpointCsv/checkpointCsvRelative instead"
    )


def test_no_caller_reimplements_the_work_hash_heuristic():
    """Recognising a Nextflow task directory happens once, in Layout.isTaskDir.

    A task directory is <work>/<2 hex>/<30 hex>. Both 30- and 32-wide spellings are
    matched, so a copy of either the wrong or the right test is caught.
    """
    # Any hex character class of a work-hash width, in either order, and any 30/32
    # length comparison, across modules/ as well (see _sweep_files).
    hash_test = re.compile(
        r"\[(?:0-9a-f|a-f0-9|0-9A-Fa-f)\]\s*\{\s*3[02]"  # [0-9a-f]{32} and friends
        r"|\.\s*(?:length\(\)|size\(\))\s*==\s*3[02]"  # .length() == 32
        r"|==\s*3[02]\s*&&.*(?:hex|hash)"  # == 32 && ...hash
    )
    offenders = [
        f"{f.relative_to(ROOT)}:{i}"
        for f in _sweep_files()
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if hash_test.search(line)
    ]
    assert not offenders, (
        "Nextflow work-hash heuristic reimplemented outside lib/Layout.groovy: "
        + ", ".join(offenders)
    )


def _layout_constants() -> dict[str, str]:
    return dict(
        re.findall(r"static final String (\w+)\s*=\s*'([^']+)'", LAYOUT.read_text())
    )


def test_checkpoint_step_constants_are_the_csv_basenames():
    consts = _layout_constants()
    assert consts.get("PREPROCESSED") == "preprocessed"
    assert consts.get("REGISTERED") == "registered"
    assert consts.get("POSTPROCESSED") == "postprocessed"
    assert consts.get("CSV_DIR") == "csv"


def test_every_kind_asked_of_layout_is_published_by_modules_config():
    """Layout describes the layout and conf/modules.config enforces it; they are
    kept in agreement by hand, so assert the agreement."""
    consts = _layout_constants()
    published = set(
        re.findall(
            r"\$\{params\.outdir\}/\$\{meta\.patient_id\}/([A-Za-z_][A-Za-z0-9_]*)",
            MODULES_CONFIG.read_text(),
        )
    )
    assert published, "no per-patient publishDir paths found in conf/modules.config"

    call = re.compile(
        r"Layout\.(?:publishedPath|patientDir)\("
        r"\s*params\.\w+\s*,\s*[^,]+,\s*(?:'([^']+)'|Layout\.(\w+))"
    )
    asked = []
    for f in _callers():
        # Comment-stripped, string-preserving view: the `kind` is a quoted literal,
        # and a call shown only in a comment must not count as a call site.
        text = _strip_comments(f.read_text())
        # Scanned over the whole file, not line by line: registration.nf's call is
        # wrapped across two lines and a per-line scan would silently skip it.
        for m in call.finditer(text):
            kind = m.group(1) or consts.get(m.group(2))
            line_no = text[: m.start()].count("\n") + 1
            asked.append((f"{f.relative_to(ROOT)}:{line_no}", kind))

    assert asked, "no Layout published-path calls found — did the call sites move?"
    # The two kinds whose loss would break a restart; pinned by name so a call site
    # silently disappearing is a failure rather than a shrinking list.
    assert {"registered", "preprocessed"} <= {kind for _where, kind in asked}
    unpublished = [(where, kind) for where, kind in asked if kind not in published]
    assert not unpublished, (
        "Layout is asked for a `kind` conf/modules.config never publishes into: "
        + ", ".join(f"{where} -> {kind!r}" for where, kind in unpublished)
    )


def test_the_unregistered_slide_path_has_its_own_owner():
    """A slide that was never registered is published by whoever produced it, not
    into <pid>/registered/. The checkpoint writer must ask Layout which case it is
    in rather than assume."""
    writer = (ROOT / "subworkflows" / "local" / "register_patient.nf").read_text()
    assert "Layout.passthroughPath(" in writer, (
        "the registered-checkpoint row builder no longer routes unregistered slides "
        "through Layout.passthroughPath — see tests/checkpoint_manifest.nf.test"
    )
    assert "is_passthrough" in writer

    # Both producers of an unregistered slide must set the flag the writer keys on:
    # REGISTER_PATIENT's single-slide branch, and TILED_ADAPTER's every-patient
    # reference.
    reg = (ROOT / "subworkflows" / "local" / "register_patient.nf").read_text()
    assert "is_passthrough" in reg, (
        "REGISTER_PATIENT's single-slide passthrough branch no longer marks the "
        "slide — its checkpoint row would name a file nothing publishes"
    )
    tiled = (
        ROOT / "subworkflows" / "local" / "adapters" / "tiled_adapter.nf"
    ).read_text()
    assert "is_passthrough" in tiled, (
        "TILED_ADAPTER's reference passes through unregistered but is not marked "
        "is_passthrough — its registered.csv row would name a file nothing publishes"
    )


def _per_patient_publish_leaves() -> set[str]:
    """Every per-patient publish leaf conf/modules.config declares.

    Matches `path: { "${params.outdir}/${meta.patient_id}/<leaf>" }`. Multi-level leaves
    (qc/registration, cell_properties/nuclei, registered/summary, ...) are QC and producer
    sub-paths beneath a kind, not kinds themselves — only the first segment counts, and the
    QC subtree is excluded entirely because no checkpoint CSV names it.
    """
    text = MODULES_CONFIG.read_text()
    leaves = set()
    for m in re.finditer(
        r"\$\{params\.outdir\}/\$\{meta\.patient_id\}/([A-Za-z0-9_/]+)", text
    ):
        first = m.group(1).split("/")[0]
        if first == "qc":
            continue
        leaves.add(first)
    return leaves


def _declared_kinds() -> set[str]:
    """Layout.PUBLISHED_KINDS, parsed from the Groovy source."""
    text = LAYOUT.read_text()
    block = re.search(r"PUBLISHED_KINDS\s*=\s*\[(.*?)\]\.asImmutable\(\)", text, re.S)
    assert block, "Layout.PUBLISHED_KINDS not found — did the constant move?"
    body = block.group(1)
    kinds = set(re.findall(r"'([^']+)'", body))
    # Bare constant references (PREPROCESSED, REGISTERED) resolve to their own literals.
    for const in re.findall(r"^\s*([A-Z_]+),\s*$", body, re.M):
        lit = re.search(rf"static final String {const}\s*=\s*'([^']+)'", text)
        assert lit, (
            f"Layout.PUBLISHED_KINDS names {const}, which has no String constant"
        )
        kinds.add(lit.group(1))
    return kinds


def test_every_per_patient_publish_leaf_is_a_declared_kind():
    """The reverse direction: a leaf the config publishes into that Layout never names is
    a published path with no owner — exactly what Layout exists to prevent."""
    undeclared = _per_patient_publish_leaves() - _declared_kinds()
    assert not undeclared, (
        f"conf/modules.config publishes per-patient into {sorted(undeclared)}, which "
        "Layout.PUBLISHED_KINDS does not declare. Add them there, or stop publishing there."
    )


def test_no_declared_kind_is_dead():
    """A kind Layout declares that nothing publishes into is a path that cannot exist."""
    dead = _declared_kinds() - _per_patient_publish_leaves()
    assert not dead, (
        f"Layout.PUBLISHED_KINDS declares {sorted(dead)}, which conf/modules.config never "
        "publishes a per-patient artifact into."
    )


def test_no_process_relies_on_the_fall_through_publish_default():
    """Every process in modules/local/ must have a withName: block that sets publishDir.

    The fall-through default publishes to _UNROUTED_PUBLISH/; a real run-level default
    lets one patient's outputs silently overwrite another's.
    """
    config = MODULES_CONFIG.read_text()
    processes = set()
    for nf in sorted((ROOT / "modules" / "local").glob("*.nf")):
        m = re.search(r"^process\s+([A-Z][A-Z0-9_]*)\s*\{", nf.read_text(), re.M)
        if m:
            processes.add(m.group(1))

    # Process names in a withName: selector whose block assigns publishDir, read
    # from the comment/string-stripped body so a comment naming publishDir does not count.
    routed = set()
    for m in re.finditer(r"withName:\s*'([^']+)'\s*\{", config):
        names = m.group(1).split("|")
        start = m.end()
        end = _block_extent(config, start)
        block_clean = _strip_comments_and_strings(config[start:end])
        if re.search(r"^\s*publishDir\s*=", block_clean, re.M):
            routed.update(n.strip() for n in names)

    unrouted = processes - routed
    assert not unrouted, (
        f"{sorted(unrouted)} have no withName: block setting publishDir, so they fall "
        "through to the _UNROUTED_PUBLISH default. Give each an explicit publishDir "
        "(or `publishDir = [ enabled: false ]` if its output is intentionally not "
        "published)."
    )


def test_every_publishdir_path_closure_matches_the_literal_shape_the_parser_reads():
    """Every `path:` closure naming `patient_id` is in the one shape
    _per_patient_publish_leaves() can read
    (`${params.outdir}/${meta.patient_id}/<literal segments>`).

    Any other shape would be invisible to the leaf check, so it fails here instead.
    """
    config = MODULES_CONFIG.read_text()
    literal_patient_path = re.compile(
        r'^\s*path:\s*\{\s*"\$\{params\.outdir\}/\$\{meta\.patient_id\}'
        r'(?:/[A-Za-z0-9_/]*)?"\s*\}\s*,?\s*$'
    )
    offenders = []
    for i, line in enumerate(config.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("path:") or "patient_id" not in stripped:
            continue
        if not literal_patient_path.match(line):
            offenders.append(f"conf/modules.config:{i}: {stripped}")
    assert offenders == [], (
        "publishDir path: closure(s) reach for the patient ID in a shape "
        "_per_patient_publish_leaves() cannot parse — Layout's kind-agreement check "
        "cannot see these paths at all:\n" + "\n".join(offenders)
    )


def test_task_dir_test_is_structural_not_a_length_guess():
    """A Nextflow task directory is `<work>/<2 hex>/<30 hex>` (30, not 32): pin a
    30-or-more-wide hex name AND a two-hex-character parent."""
    text = LAYOUT.read_text()
    body = text[text.index("static boolean isTaskDir(") :]
    body = body[: body.index("\n    }")]
    assert "{30,32}" in body, "isTaskDir must accept a 30-character task-dir name"
    assert "{2}" in body, "isTaskDir must also require the two-hex-character parent"
