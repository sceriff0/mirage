#!/usr/bin/env python3
r"""Guard that the shipped `reg_micro_reg` default is `2`, everywhere it lives.

2026-09-16: the default moved 1 -> 2 (micro-rigid + micro non-rigid) by user ruling, on
main, dev and benchmarking together. This file was test_micro_reg_default_is_one.py; every
home below moved with it. The prose check changed direction: "the default is the maximum"
is TRUE again, so what is now forbidden is prose calling micro-rigid-only (1) the default,
or describing the micro non-rigid pass as off / opt-in at the shipped defaults. The
history below is what the homes are and how each was found, written when the default was 1.


`reg_micro_reg` has TEN homes. NINE were enumerated by the 2026-08-30 spec
(Task 5 of the registration-backend-surface plan); the tenth --
`ParamUtils.microRegLevelOf`'s javadoc -- was found by the final whole-branch
review, and is documented at `PARAM_UTILS` below. The nine: `nextflow.config`
(the single source of
truth for the shipped default -- see `test_no_duplicate_param_defaults.py`),
`nextflow_schema.json` (the schema's own copy of that default, which
`validateParameters()` uses), `docs/parameters.md`'s parameter table, three
`docs/figures/*.html` supplementary schematics, two `params/*.json` presets,
and `bin/register.py`. None of these are generated from another -- each is
a hand-maintained restatement -- so a numeric change to one does not
propagate to the rest; this test is the check that they were all moved
together.

`bin/register.py` alone turned out to carry FOUR of its own restatements,
not one: `valis_registration()`'s docstring prose (the home the plan's brief
counted), its Python function-signature default (`micro_reg: int = ...`),
the `--micro-reg` argparse `default=`, and the `[default]` marker inside
that argument's `--help` text. A first pass of this guard checked only the
docstring and missed the other three -- caught in review, not by this file
-- so all four are asserted below, independently, each anchored to its own
syntactic shape rather than to the docstring's.

`conf/test.config` pins `reg_micro_reg = 0` deliberately (Phase 0c: micro-
registration OOMed the JVM on a 15.6 GiB CI runner) and is NOT one of the
nine homes -- it must keep diverging from the shipped default, so it is
asserted to still read `0`, not `1`.

A second, independent check below closes the prose trap the plan's brief
missed on first pass: the claim "the default is the maximum legal value"
appears in FOUR places, not two, and one of them reverses the word order
(`MAX default` in `docs/figures/pipeline-schematic.html`, vs. `default MAX`
elsewhere) -- a grep for the literal brief phrase would not have found it.
Once `reg_micro_reg`'s default is `1` (not `2`, the max of `{0, 1, 2}`),
every one of those four sentences is false, so this test asserts none of
them survive, in either word order, case-insensitively.

`MAX_DEFAULT_PATTERN` WAS TOO NARROW AND LET TWO LIVE SENTENCES THROUGH.
It required the two words to be ADJACENT (`default\s+max|max\s+default`),
so it matched neither of the two sentences that were actually shipping in
`docs/figures/pipeline-schematic.html` on 2026-08-30:

    :371  "Micro-registration at maximum depth by default."
    :849  "...QC on by default and micro-registration at maximum depth..."

Both are the same false claim; neither puts the words side by side. That is
also the reason the tenth home (`ParamUtils.microRegLevelOf`'s "Default 2
(max: ...", see `PARAM_UTILS` above) had to be caught by a hand-written
second check rather than by this one. The pattern now allows up to 60
non-sentence-ending characters between the two words in either order, which
matches all three phrasings. `\bmax(?:imum|imal)?\b` keeps the word
boundary deliberately: `max_memory`, `max_cpus` and `maxRetries` all fail
it, so the many "`max_*` ... default" co-occurrences in `nextflow.config`
and the schema do not become false positives. Measured against the five
`PROSE_HOMES` before the prose was fixed: two hits, both the sentences
above, zero elsewhere.

`docs/figures/registration-schematic.html:559` carried a THIRD instance
("the shipped default (depth 2 = maximum)"). This docstring used to say the
file was deliberately NOT in `PROSE_HOMES` because spec Phase 6 owns
rewriting that figure. That reasoning is withdrawn, on review: Phase 6 owns
the REWRITE, not a licence to leave a false shipped default in a published
figure, and the exclusion is precisely what let `:559` survive the pass that
widened this pattern. The file is now in `PROSE_HOMES`; adding it was
watched fail on that exact line before the prose was fixed:

    AssertionError: these still claim the reg_micro_reg default is the
    maximum value (it is 1, the minimum non-zero value of {0, 1, 2}):
    ["docs/figures/registration-schematic.html: 'default (depth 2 = maximum'"]

Its step-2b HEADER carried the same claim a second time ("register_micro()
- micro_reg = 2 - default") and does not match this pattern at all -- it has
no "max" in it -- so it was fixed by hand alongside. Two copies in one file,
again; see the ELEVENTH/TWELFTH note below.

Still deferred to Phase 6, and correctly: that figure's description of the
deleted coarse front-end, and its stale `coarse_max_dim` value. That is a
genuine rewrite rather than a number, and it is allow-listed with that reason
stated in `tests/test_no_legacy_frontends.py` -- which is also why this
docstring does not name the front-end: that guard forbids new references to
it, and it is right to.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.nfmodel import block_extent as _block_extent
from tests.nfmodel import strip_comments as _strip_comments

ROOT = Path(__file__).resolve().parent.parent

# The nine homes that must all read the shipped default, `2`.
NEXTFLOW_CONFIG = ROOT / "nextflow.config"
SCHEMA = ROOT / "nextflow_schema.json"
PARAMETERS_MD = ROOT / "docs" / "parameters.md"
PIPELINE_SCHEMATIC = ROOT / "docs" / "figures" / "pipeline-schematic.html"
QC_SCHEMATIC = ROOT / "docs" / "figures" / "qc-schematic.html"
REGISTRATION_SCHEMATIC = ROOT / "docs" / "figures" / "registration-schematic.html"
FULL_PIPELINE_PARAMS = ROOT / "params" / "full_pipeline.json"
REGISTRATION_ONLY_PARAMS = ROOT / "params" / "registration_only.json"
REGISTER_PY = ROOT / "bin" / "register.py"
# The TENTH home, found by the final whole-branch review after this file was
# written: `ParamUtils.microRegLevelOf`'s javadoc said "Default 2 (max:
# micro-rigid + micro non-rigid), matching nextflow.config" -- a sentence THIS
# BRANCH made false. It escaped every check here twice over: it is not in the
# nine-home list, and `MAX_DEFAULT_PATTERN` could not reach it either, because
# the phrasing is "Default 2 (max: ...", which neither "default max" nor "max
# default" matches. A prose home needs a check shaped like its own prose.
PARAM_UTILS = ROOT / "lib" / "ParamUtils.groovy"
# ELEVENTH and TWELFTH, same review, same pattern: two published figures carry the
# value TWICE -- once in a `<td class="k">reg_micro_reg</td>` parameter row (which the
# checks above already pin) and once again in a `<span>micro_reg <b>N</b></span>`
# summary badge on the registration step, which nothing reached. Both badges still read
# `2`. A per-home check that matches only one of a file's two copies is not coverage of
# that file.
PIPELINE_MD = ROOT / "docs" / "pipeline.md"
# `docs/figures/registration-schematic.html` carried TWO more restatements, and this
# comment used to say they were deliberately left to spec Phase 6. That deferral was
# wrong and is withdrawn: Phase 6 owns REWRITING the figure, not a licence to keep a
# false shipped default in it, and the deferral is what let both stand. Both are fixed
# (its step-2b header, and the body sentence "the shipped default (depth 2 = maximum)"),
# the file is now in PROSE_HOMES below, and its parameter-table row stays pinned above.
# What IS still deferred to Phase 6 is that figure's description of the deleted coarse
# front-end and its stale `coarse_max_dim` value -- a genuine rewrite, allow-listed with
# that reason stated in tests/test_no_legacy_frontends.py.
_MICRO_REG_BADGE_RE = re.compile(r"<span>micro_reg <b>(\S+?)</b></span>")

# Deliberately NOT one of the nine homes -- see module docstring.
TEST_CONFIG = ROOT / "conf" / "test.config"

# Prose homes. The claim that is now FALSE is the one the default-1 era wrote: micro-rigid
# depth / reg_micro_reg=1 is what ships, and the micro non-rigid pass is off or opt-in.
PROSE_HOMES = [
    NEXTFLOW_CONFIG,
    SCHEMA,
    PARAMETERS_MD,
    PIPELINE_MD,
    PIPELINE_SCHEMATIC,
    PARAM_UTILS,
    REGISTRATION_SCHEMATIC,
]
# `1(?![.\d])`, not `1\b`: registration-schematic.html states STARE's gate as "default 1.0 px",
# and `\b` sits between the 1 and the dot.
STALE_ONE_IS_DEFAULT = re.compile(
    r"micro-rigid (?:only )?(?:depth )?(?:\(default\)|\[default\]|by default|—\s*default)"
    r"|default 1(?![.\d])|\(default 1\)|reg_micro_reg=1</code>\)?[^.]{0,40}?default"
    r"|micro-rigid depth</b> by default|micro-rigid depth by default"
    r"|micro_reg = 2 · NOT the default|does not run at the shipped\s+defaults|what ships\)"
    r"|`?1`?\s*=\s*micro-rigid[^,;|]{0,60}?(?:\[default\]|—\s*default)"
    r"|at micro-rigid depth|shipped default is 1(?![.\d])",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    text = path.read_text()
    assert text, f"{path} is empty -- guard would pass vacuously"
    return text


def _parse_top_level_params_block(config_text: str) -> dict[str, str]:
    """Return {name: declared_value_text} for a top-level `params { ... }`
    block in a .config file.

    Mirrors `test_no_duplicate_param_defaults.py`'s `parse_declared_params`:
    the block's extent comes from `tests.nfmodel.block_extent`'s comment/
    string-aware brace walk (not a naive brace count, which a `{`/`}` inside
    a comment or quoted default could mis-balance), and comments are
    stripped via `tests.nfmodel.strip_comments` before the line-by-line
    `name = value` parse, so both `nextflow.config` and `conf/test.config`
    -- which share this exact shape -- go through the same model rather than
    a second private regex.
    """
    start = config_text.find("params {")
    assert start != -1, "Could not locate `params { ... }` block"
    brace_start = config_text.find("{", start)
    end = _block_extent(config_text, brace_start + 1)
    block = config_text[brace_start + 1 : end - 1]
    clean_block = _strip_comments(block)
    declared: dict[str, str] = {}
    for raw_line in clean_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m:
            declared[m.group(1)] = m.group(2)
    return declared


def test_nextflow_config_default_is_two():
    declared = _parse_top_level_params_block(_read(NEXTFLOW_CONFIG))
    assert "reg_micro_reg" in declared, (
        "reg_micro_reg not declared in nextflow.config's params {}"
    )
    assert declared["reg_micro_reg"] == "2", (
        f"nextflow.config's reg_micro_reg default is {declared['reg_micro_reg']!r}, expected '2'"
    )


def test_schema_default_is_two():
    schema = json.loads(_read(SCHEMA))

    def _find(node):
        if isinstance(node, dict):
            if "reg_micro_reg" in node and isinstance(node["reg_micro_reg"], dict):
                return node["reg_micro_reg"]
            for value in node.values():
                found = _find(value)
                if found is not None:
                    return found
        return None

    prop = _find(schema)
    assert prop is not None, "reg_micro_reg property not found in nextflow_schema.json"
    assert prop.get("default") == 2, (
        f"nextflow_schema.json's reg_micro_reg default is {prop.get('default')!r}, expected 2"
    )


def test_parameters_md_table_default_is_two():
    text = _read(PARAMETERS_MD)
    match = re.search(r"^\|\s*`reg_micro_reg`\s*\|\s*`(\S+)`\s*\|", text, re.MULTILINE)
    assert match, "reg_micro_reg row not found in docs/parameters.md"
    assert match.group(1) == "2", (
        f"docs/parameters.md's reg_micro_reg table default is {match.group(1)!r}, expected '2'"
    )


def test_pipeline_schematic_value_is_two():
    text = _read(PIPELINE_SCHEMATIC)
    match = re.search(
        r'<td class="k">reg_micro_reg</td><td class="v">(\S+?)</td>', text
    )
    assert match, "reg_micro_reg row not found in docs/figures/pipeline-schematic.html"
    assert match.group(1) == "2", (
        f"pipeline-schematic.html's reg_micro_reg value is {match.group(1)!r}, expected '2'"
    )


def test_pipeline_schematic_badge_is_two():
    """The step-summary badge, not the parameter row -- a SECOND copy in the
    same file, which `test_pipeline_schematic_value_is_one` above does not
    reach. It read `2` for the whole of this branch."""
    text = _read(PIPELINE_SCHEMATIC)
    match = _MICRO_REG_BADGE_RE.search(text)
    assert match, (
        "micro_reg summary badge not found in docs/figures/pipeline-schematic.html "
        "-- this check would pass vacuously"
    )
    assert match.group(1) == "2", (
        f"pipeline-schematic.html's micro_reg BADGE reads {match.group(1)!r}, expected "
        "'2' (its parameter-table row is checked separately -- both are homes)"
    )


def test_pipeline_md_badge_is_two():
    """docs/pipeline.md embeds the same step-summary badge."""
    text = _read(PIPELINE_MD)
    match = _MICRO_REG_BADGE_RE.search(text)
    assert match, (
        "micro_reg summary badge not found in docs/pipeline.md -- this check would "
        "pass vacuously"
    )
    assert match.group(1) == "2", (
        f"docs/pipeline.md's micro_reg badge reads {match.group(1)!r}, expected '2'"
    )


def test_qc_schematic_value_is_two():
    text = _read(QC_SCHEMATIC)
    match = re.search(
        r'<td class="k">reg_micro_reg</td><td class="v">(\S+?)</td>', text
    )
    assert match, "reg_micro_reg row not found in docs/figures/qc-schematic.html"
    assert match.group(1) == "2", (
        f"qc-schematic.html's reg_micro_reg value is {match.group(1)!r}, expected '2'"
    )


def test_registration_schematic_value_is_two():
    text = _read(REGISTRATION_SCHEMATIC)
    match = re.search(
        r'<td class="k">reg_micro_reg</td><td class="val">(\S+?)</td>', text
    )
    assert match, (
        "reg_micro_reg row not found in docs/figures/registration-schematic.html"
    )
    assert match.group(1) == "2", (
        f"registration-schematic.html's reg_micro_reg value is {match.group(1)!r}, expected '2'"
    )


def test_full_pipeline_params_default_is_two():
    params = json.loads(_read(FULL_PIPELINE_PARAMS))
    assert "reg_micro_reg" in params, (
        "reg_micro_reg not present in params/full_pipeline.json"
    )
    assert params["reg_micro_reg"] == 2, (
        f"params/full_pipeline.json's reg_micro_reg is {params['reg_micro_reg']!r}, expected 2"
    )


def test_registration_only_params_default_is_two():
    params = json.loads(_read(REGISTRATION_ONLY_PARAMS))
    assert "reg_micro_reg" in params, (
        "reg_micro_reg not present in params/registration_only.json"
    )
    assert params["reg_micro_reg"] == 2, (
        f"params/registration_only.json's reg_micro_reg is {params['reg_micro_reg']!r}, expected 2"
    )


def test_register_py_docstring_calls_two_the_default():
    text = _read(REGISTER_PY)
    match = re.search(
        r"micro_reg\s*:\s*int, optional\n(?:.*\n)*?    (?:stage_checkpoint_dir|Returns)",
        text,
    )
    assert match, "micro_reg docstring block not found in bin/register.py"
    # Collapse whitespace first: docstring prose re-wraps across physical lines.
    normalized = re.sub(r"\s+", " ", match.group(0))
    assert "matches the pipeline's ``reg_micro_reg`` default" in normalized, (
        "bin/register.py's cross-reference to the pipeline default is missing entirely "
        "-- expected wording to be reworded onto 2, not deleted"
    )
    one_clause = re.search(r"1\s*=[^;]*?\bthe default\b", normalized, re.IGNORECASE)
    assert one_clause is None, (
        "bin/register.py still describes micro_reg=1 as 'the default': "
        f"{one_clause.group(0) if one_clause else None!r}"
    )
    assert re.search(r"2\s*=.*?\bthe default\b", normalized, re.IGNORECASE), (
        "bin/register.py's micro_reg docstring does not call 2 the default"
    )


def test_register_py_function_signature_default_is_two():
    """`valis_registration()`'s own Python default -- a direct import (not
    the pipeline, which always passes --micro-reg explicitly: register.nf)
    silently gets whatever this says."""
    text = _read(REGISTER_PY)
    match = re.search(r"^\s*micro_reg:\s*int\s*=\s*(\S+?),", text, re.MULTILINE)
    assert match, "micro_reg: int = ... signature default not found in bin/register.py"
    assert match.group(1) == "2", (
        f"bin/register.py's valis_registration() signature default is "
        f"{match.group(1)!r}, expected '2'"
    )


def test_register_py_argparse_default_is_two():
    """The `--micro-reg` CLI flag's own default -- a hand invocation without
    the flag silently gets whatever this says, independent of the pipeline
    (which always passes --micro-reg explicitly: register.nf)."""
    text = _read(REGISTER_PY)
    match = re.search(r'"--micro-reg",\s*\n\s*type=int,\s*\n\s*default=(\S+?),', text)
    assert match, "--micro-reg argparse block not found in bin/register.py"
    assert match.group(1) == "2", (
        f"bin/register.py's --micro-reg argparse default is {match.group(1)!r}, expected '2'"
    )


def test_register_py_help_text_marks_two_not_one_as_default():
    """The --micro-reg --help string's own '[default]' marker -- must sit on
    the '1=...' clause, not '2=...', or --help visibly lies to an operator
    about which value is shipped."""
    text = _read(REGISTER_PY)
    match = re.search(
        r'help="Micro-registration depth \(nested\):.*?"\s*\n\s*"[^"]*",',
        text,
        re.DOTALL,
    )
    assert match, "--micro-reg help text not found in bin/register.py"
    normalized = re.sub(r"\s+", " ", match.group(0))
    two_clause_match = re.search(r"2\s*=[^,]*?\[default\]", normalized)
    assert two_clause_match, (
        f"bin/register.py's --micro-reg help text does not mark the '2=...' "
        f"clause as [default]: {normalized!r}"
    )
    one_clause_match = re.search(r"1\s*=[^,]*?\[default\]", normalized)
    assert one_clause_match is None, (
        f"bin/register.py's --micro-reg help text still marks the '1=...' "
        f"clause as [default]: {normalized!r}"
    )


def test_conf_test_config_pin_is_untouched():
    """conf/test.config is a deliberate divergence from the shipped default,
    not one of the nine homes -- Phase 0c pinned it to 0 because micro-
    registration OOMed the JVM on a 15.6 GiB CI runner. It must stay 0."""
    declared = _parse_top_level_params_block(_read(TEST_CONFIG))
    assert "reg_micro_reg" in declared, (
        "reg_micro_reg not declared in conf/test.config's params {}"
    )
    assert declared["reg_micro_reg"] == "0", (
        f"conf/test.config's reg_micro_reg pin is {declared['reg_micro_reg']!r}, expected '0' "
        "(this file is deliberately NOT one of the nine homes -- see module docstring)"
    )


def _doc_comment_above(text: str, signature_fragment: str) -> str:
    """The contiguous comment block immediately above the line containing
    `signature_fragment`, returned as normalised prose.

    Deliberately NOT a hand-rolled `/* ... */` regex. `tests/test_nfmodel.py::
    test_no_guard_parses_nextflow_source_privately` forbids one, and rightly:
    the measured case is `path(reference, stageAs: 'ref/*')` in
    `modules/local/register.nf`, whose `/*` opened a FAKE block comment that
    swallowed 64 lines from a naive DOTALL parse. The first draft of this
    helper WAS that regex and that guard caught it.

    So the location comes from the model instead. `tests.nfmodel.strip_comments`
    blanks comments to spaces while preserving line structure, so a line that
    is whitespace in the blanked view but non-empty in the raw file IS a comment
    line -- that difference is the extraction, and the model owns the hard part
    (knowing where a comment really starts and ends).
    """
    blanked = _strip_comments(text)
    raw_lines = text.splitlines()
    blank_lines = blanked.splitlines()
    assert len(raw_lines) == len(blank_lines), (
        "strip_comments changed the line count; the raw/blanked line pairing below "
        "would be meaningless"
    )
    idx = next(
        (i for i, line in enumerate(blank_lines) if signature_fragment in line), None
    )
    assert idx is not None, (
        f"{signature_fragment!r} not found in the code (comment-blanked) view -- "
        "either it was renamed, or it exists only inside a comment"
    )
    out: list[str] = []
    j = idx - 1
    while j >= 0 and raw_lines[j].strip() and not blank_lines[j].strip():
        out.append(raw_lines[j].strip().lstrip("*/ ").rstrip("*/ "))
        j -= 1
    assert out, f"no comment block sits immediately above {signature_fragment!r}"
    return re.sub(r"\s+", " ", " ".join(reversed(out)))


def test_param_utils_doc_comment_calls_two_the_default():
    """`microRegLevelOf`'s javadoc is a tenth home, and this branch falsified it.

    It read "Default 2 (max: micro-rigid + micro non-rigid), matching
    nextflow.config" while nextflow.config had moved to 1. Two separate reasons
    nothing here caught it: the file was not in the nine-home list, and
    `MAX_DEFAULT_PATTERN` ("default max" / "max default") does not match
    "Default 2 (max: ...".

    The `regMicroReg == null ? 2 : ...` fallback VALUE is deliberately left
    alone -- it is unreachable on the pipeline path, since the schema declares
    `reg_micro_reg` non-nullable with `default: 1` and `validateParameters()`
    fills it in first -- so this asserts on the PROSE only, and requires the
    prose to say which of the two numbers ships.
    """
    # Prose re-wraps whenever the surrounding text is edited, so the helper
    # normalises whitespace before matching, as the bin/register.py checks above do.
    doc = _doc_comment_above(_read(PARAM_UTILS), "static int microRegLevelOf(")

    offender = re.search(r"[Dd]efault\s+(?:is\s+|of\s+)?1\b", doc)
    assert offender is None, (
        "lib/ParamUtils.groovy's microRegLevelOf javadoc still calls 1 the "
        f"default: {offender.group(0)!r} -- nextflow.config ships 2"
    )
    assert re.search(r"[Dd]efault\s+(?:is\s+)?2\b", doc), (
        "lib/ParamUtils.groovy's microRegLevelOf javadoc no longer names the "
        "shipped default at all -- expected it reworded onto 2, not deleted"
    )
    assert "nextflow.config" in doc, (
        "microRegLevelOf's javadoc dropped its cross-reference to the single "
        "source of truth; the whole point of naming the default here is to point "
        "a reader at where it is declared"
    )


def test_no_prose_home_still_calls_micro_rigid_only_the_default():
    """Every sentence the default-1 era wrote about what ships is false at 2. Matched on
    whitespace-collapsed text, because figure prose wraps mid-phrase."""
    offenders = []
    for path in PROSE_HOMES:
        text = re.sub(r"\s+", " ", _read(path))
        for match in STALE_ONE_IS_DEFAULT.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)!r}")
    assert not offenders, (
        "these still describe reg_micro_reg=1 (micro-rigid only) as the shipped default, "
        f"or the micro non-rigid pass as opt-in; the default is 2: {offenders}"
    )
