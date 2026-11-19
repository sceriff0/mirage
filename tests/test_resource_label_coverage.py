#!/usr/bin/env python3
"""Guard against process labels that resolve to no `withLabel` selector.

A Nextflow `label` that matches no `withLabel:` selector is not an error --
the process silently falls through to `conf/base.config`'s defaults. That
makes an unmatched label invisible at runtime: it never crashes, it just
quietly under- (or over-) provisions the process.

This test asserts every resource label referenced in `modules/local/*.nf`
resolves to a `withLabel` selector defined in `conf/modules.config`. A `label`
line can be an expression (e.g. a ternary) rather than a bare literal, so
extraction pulls every quoted string off the line.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from tests.nfmodel import block_extent as _block_extent
from tests.nfmodel import processes as _nfmodel_processes
from tests.nfmodel import strip_comments as _strip_comments
from tests.nfmodel import with_name_blocks as _with_name_blocks

ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = ROOT / "modules" / "local"
MODULES_CONFIG = ROOT / "conf" / "modules.config"

# A `label ...` statement, however its right-hand side is built (bare
# literal, ternary, etc.) -- anchored to line start so we don't match
# `label` appearing elsewhere (e.g. inside a string).
LABEL_LINE_RE = re.compile(r"^\s*label\s+.+$")

# Every single- or double-quoted string literal on a line.
QUOTED_STRING_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")

# A trailing `// ...` line comment, or an inline `/* ... */` block comment.
# Stripped before label extraction: extraction takes every quoted string off a
# `label` line, so a quoted word in a comment (`// was 'process_low'`) would
# otherwise be read as a live label.
COMMENT_RE = re.compile(r"//.*$|/\*.*?\*/")

# `withLabel: 'name' { ... }` selectors in conf/modules.config.
WITH_LABEL_RE = re.compile(r"withLabel:\s*(?:'([^']+)'|\"([^\"]+)\")")

# A top-level `cpus =`, `memory =`, or `time =` assignment line inside a
# withName block body. Anchored to line-start (after indentation) so it
# only matches real assignments, not comments or unrelated identifiers.
RESOURCE_FIELD_RE = re.compile(r"^\s*(cpus|memory|time)\s*=", re.MULTILINE)

RESOURCE_FIELDS = frozenset({"cpus", "memory", "time"})


def extract_labels_from_line(line: str) -> list[str]:
    """Return every quoted string literal on a `label ...` line.

    A bare `label 'process_medium'` yields one label. A ternary like
    `label cond ? 'process_high_memory' : 'process_medium'` yields both
    branches.

    Comments are stripped first, so an annotation such as
    `label 'process_high'  // was 'process_low'` yields only the live label.
    """
    return [
        m.group(1) if m.group(1) is not None else m.group(2)
        for m in QUOTED_STRING_RE.finditer(COMMENT_RE.sub("", line))
    ]


def find_module_label_sites() -> list[tuple[Path, int, str, list[str]]]:
    """Return (file, line_no, line_text, labels) for every `label` line in modules/local/*.nf."""
    sites: list[tuple[Path, int, str, list[str]]] = []
    for path in sorted(MODULES_DIR.glob("*.nf")):
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if LABEL_LINE_RE.match(line):
                sites.append(
                    (path, line_no, line.strip(), extract_labels_from_line(line))
                )
    return sites


def find_defined_labels() -> set[str]:
    """Return the set of labels with a `withLabel:` selector in conf/modules.config."""
    text = MODULES_CONFIG.read_text()
    return {
        m.group(1) if m.group(1) is not None else m.group(2)
        for m in WITH_LABEL_RE.finditer(text)
    }


def test_extract_labels_handles_ternary_expression():
    """Extraction must find both branches of a ternary `label` line, not just one literal.

    A guard on the extractor itself: a single-literal pattern such as
    `label\\s+'([^']+)'` would return one match, or none, for a ternary.
    """
    ternary_line = "    label params.spatialdata_include_image ? 'process_high_memory' : 'process_medium'"
    assert extract_labels_from_line(ternary_line) == [
        "process_high_memory",
        "process_medium",
    ]

    literal_line = "    label 'process_high'"
    assert extract_labels_from_line(literal_line) == ["process_high"]


def test_extract_labels_ignores_commented_out_labels():
    """A quoted label inside a comment must not count as a live label.

    Otherwise a bumped label annotated `// was 'process_low'` would match the
    unchanged `docs/resources.md` row and hide the bump from the doc cross-check.
    """
    assert extract_labels_from_line(
        "    label 'process_high'  // was 'process_low'"
    ) == ["process_high"]
    assert extract_labels_from_line("    label /* 'process_low' */ 'process_high'") == [
        "process_high"
    ]
    assert extract_labels_from_line(
        "    label 'process_medium' // bumped, see #123"
    ) == ["process_medium"]


def test_every_module_label_resolves_to_a_conf_modules_config_selector():
    """Every label referenced in modules/local/*.nf must resolve in conf/modules.config.

    An unmatched label is not a Nextflow error -- the process silently
    inherits conf/base.config's defaults instead of the resources its label
    name implies.
    """
    defined = find_defined_labels()
    sites = find_module_label_sites()

    assert sites, (
        "No `label` lines found under modules/local/*.nf -- scan pattern may be stale."
    )
    assert defined, (
        "No `withLabel:` selectors found in conf/modules.config -- scan pattern may be stale."
    )

    offending = []
    for path, line_no, line_text, labels in sites:
        for label in labels:
            if label not in defined:
                offending.append(
                    f"{path.relative_to(ROOT)}:{line_no}: label '{label}' has no "
                    f"`withLabel: '{label}'` selector in conf/modules.config "
                    f"(line: {line_text})"
                )

    assert not offending, (
        f"{len(offending)} label reference(s) resolve to no `withLabel` selector "
        "in conf/modules.config, so the process silently falls back to "
        "conf/base.config's defaults instead of the resources the label name "
        "implies. Define the selector or re-point the module at an existing "
        "label:\n" + "\n".join(offending)
    )


# ---------------------------------------------------------------------------
# Exactly-one-owner guard.
#
# A `withName:` block in conf/modules.config wins over a `label`'s
# `withLabel:` tier whenever both set the same field -- that is Nextflow's
# config-merge precedence, not a bug. So a module can end up with:
#   - a `label`  that is completely redundant (every one of cpus/memory/time
#     is also set by a matching `withName:` block) -- the label is dead
#     prose, and a maintainer editing it to "fix" the process's resources
#     would see no effect.
#   - NEITHER a `label` NOR full `withName:` coverage of cpus/memory/time --
#     whichever field nothing sets falls through to conf/base.config's small
#     defaults (1 cpu, 6 GB, 4 h -- see conf/base.config), silently
#     under-provisioning the task.
#
# Exactly one of {label, full withName coverage} must be true per process. A
# partial withName override (e.g. TILED_SOLVE's block sets only `memory`) is not
# a violation: the label still owns the fields withName leaves alone.
# ---------------------------------------------------------------------------


def find_module_processes() -> dict[str, Path]:
    """Return {PROCESS_NAME: module_file_path} for every modules/local/*.nf file.

    Sourced from `tests.nfmodel.processes()`, which finds `process NAME {` on
    the comment/string-stripped text of every `.nf` file under `modules/`
    (recursively), filtered here to `modules/local/`. `modules/nf-core/basicpy/`
    (`BASICPY`) is a vendored nf-core module outside the one-owner rule;
    docs/resources.md excludes it from every count by name, so widening this
    filter would desync those counts from the doc.
    """
    return {
        name: p.path
        for name, p in _nfmodel_processes().items()
        if p.path.parent == MODULES_DIR
    }


def find_labelled_processes() -> set[str]:
    """Return the set of process names whose module file has a `label` line.

    LABEL_LINE_RE is compiled without `re.MULTILINE`, so it must be applied
    per line, not via .search() on the whole file text, where `^`/`$` would
    anchor only to the whole string and never match.
    """
    labelled = set()
    for name, path in find_module_processes().items():
        if any(LABEL_LINE_RE.match(line) for line in path.read_text().splitlines()):
            labelled.add(name)
    return labelled


def find_withname_resource_fields() -> dict[str, set[str]]:
    """Return {PROCESS_NAME: {fields it sets}} for every `withName:` block.

    A selector's name group can be a `|`-separated alternation naming
    several processes; every name gets credited with the fields the shared
    block sets. A process named by more than one `withName:` block (e.g. a
    shared errorStrategy alternation plus its own block) has its fields
    unioned across all matching blocks.
    """
    fields_by_process: dict[str, set[str]] = {}
    for block in _with_name_blocks():
        # `block.body` is comment/string-stripped (tests.nfmodel's view A), so
        # a `cpus =`/`memory =`/`time =` mention inside a comment or a string
        # cannot be counted as a real assignment.
        fields = {m.group(1) for m in RESOURCE_FIELD_RE.finditer(block.body)}
        for name in block.names:
            fields_by_process.setdefault(name, set()).update(fields)
    return fields_by_process


# Nested-brace and comment-mentioned-selector coverage for the walk this
# function uses lives in tests/test_nfmodel.py.


def test_exactly_one_resource_owner_per_process():
    """Every process must have exactly one resource owner: a `label`, or full
    `withName:` coverage of cpus/memory/time -- never both, never neither.

    `withName:` always wins when both are present (Nextflow config-merge
    precedence), so a `label` alongside a `withName:` block that sets all
    three fields is dead: removing it changes nothing, and leaving it in
    place misleads a maintainer who edits the label expecting an effect.

    A process with neither a label nor full withName coverage is worse: the
    field(s) nothing sets fall through to conf/base.config's defaults (1
    cpu, 6 GB, 4 h) with no error, no warning, and no visible diff against a
    correctly-configured process -- until it dies on real data.
    """
    processes = find_module_processes()
    labelled = find_labelled_processes()
    withname_fields = find_withname_resource_fields()

    assert processes, (
        "No `process NAME {` declarations found under modules/local/*.nf -- scan pattern may be stale."
    )

    both_owners = []
    no_owner = []
    for name in sorted(processes):
        has_label = name in labelled
        fields = withname_fields.get(name, set())
        full_withname_coverage = fields == RESOURCE_FIELDS

        if has_label and full_withname_coverage:
            both_owners.append(
                f"{name} ({processes[name].relative_to(ROOT)}): has a `label` AND a "
                "`withName:` block in conf/modules.config setting cpus, memory, AND "
                "time -- the label is dead; withName always wins, so remove the label."
            )
        elif not has_label and not full_withname_coverage:
            missing = sorted(RESOURCE_FIELDS - fields)
            no_owner.append(
                f"{name} ({processes[name].relative_to(ROOT)}): has no `label` and its "
                f"`withName:` block(s) do not set {missing} -- that field silently "
                "inherits conf/base.config's defaults. Add a `label` back or set the "
                "missing field(s) explicitly in conf/modules.config."
            )

    assert not both_owners, (
        f"{len(both_owners)} process(es) have a redundant `label` fully shadowed by a "
        "`withName:` block:\n" + "\n".join(both_owners)
    )
    assert not no_owner, (
        f"{len(no_owner)} process(es) have no resource owner for one or more of "
        "cpus/memory/time:\n" + "\n".join(no_owner)
    )


# ---------------------------------------------------------------------------
# docs/resources.md's per-process table vs conf/modules.config.
#
# `docs/resources.md` restates every process's cpus/memory/time by hand. The
# guard checks both directions:
#   * every row in the doc table names a real process, and its cpus / memory /
#     time / owner agree with what conf/modules.config (or the module's label)
#     actually sets;
#   * every process that has a label or a `withName:` resource block has a row.
#
# How the numbers are compared, and why not more loosely:
#   * cpus and time are evaluated to integers on both sides. An expression this
#     guard cannot evaluate FAILS the row with the expression quoted; it does
#     not skip.
#   * memory is compared against computed values, never against "every integer
#     in the expression", which would let a wrong doc number pass when it
#     coincides with a tier constant or a `>> 30` shift operand.
#   * memory is compared pairwise, not as pools of numbers: a tier ladder is an
#     ordered list of (threshold, magnitude) rungs, a size-linear request a
#     (coefficient, addend) pair, so the right numbers in the wrong places fail.
#     memory_cell_mismatch knows exactly three shapes and refuses a fourth.
#
# What is not compared, and why:
#   * `TILED_REG_TILE` / `TILED_STITCH` memory is derived from
#     `reg_tiled_tile`/`reg_tiled_halo` and `reg_tiled_out_tile` at task
#     submission, not a literal. Re-deriving it here would duplicate the very
#     formula this guard exists to stop duplicating, and evaluating Groovy from
#     pytest is not on the table. So for these two rows the guard asserts every
#     DERIVATION SOURCE named in the config expression is named in the doc cell
#     too, and leaves the illustrative "4 GB at defaults" figure unchecked.
#     Changing which parameter the formula reads therefore still fails the build;
#     re-tuning its constants does not. docs/resources.md labels the unchecked
#     figure as such. The reads are allowlisted by name in DERIVED_MEMORY_PARAMS
#     so the exemption cannot spread to any closure that mentions `params.`.
#   * `SEG_QC_SEGMENT` (and, on dev, `SPLIT_PRIOR_PYRAMID`) are ALIASES, not modules. Their
#     `withName:` blocks set no resource field (they override publishDir /
#     ext.prefix only), so neither is required to have a row. SEG_QC_SEGMENT has
#     one anyway -- it is a documented GPU/QC cost -- and ALIASES below maps it
#     back to SEGMENT so its numbers are still checked against a real config
#     block rather than skipped.
#   * The ramp shape of a flat request. `32.GB * (2 ** (task.attempt - 1))`
#     documented as `32 GB x attempt` still passes: only magnitudes are compared.
#     A wrong rung count is caught; the notation between rungs is not.
#   * Everything in docs/resources.md outside the per-process tables. The label
#     tier table, the conf/base.config fallback table, the `maxForks` column, the
#     global-ceiling and profile tables, the retry-policy tables and the
#     container table are all hand-maintained and unguarded. Only rows whose
#     table's first header cell is literally `Process` are read.
#   * Prose. A cell whose numbers are right but whose words are wrong (e.g.
#     "tier on channels + masks" naming the wrong inputs) passes.
# ---------------------------------------------------------------------------

DOCS_RESOURCES = ROOT / "docs" / "resources.md"

# Process names in docs/resources.md that are Nextflow ALIASES of another
# process, so conf/modules.config configures them under the original name.
ALIASES = {"SEG_QC_SEGMENT": "SEGMENT"}

# `withLabel: 'name' { ... }` -- the opening brace is needed to brace-match the
# body via `_closure_body` below. `tests.nfmodel` has no `with_label_blocks()`,
# so this scan stays local to this file (its body extent uses `block_extent`).
WITH_LABEL_OPEN_RE = re.compile(r"withLabel:\s*(?:'([^']+)'|\"([^\"]+)\")\s*\{")

# The start of a `cpus = ` / `memory = ` / `time = ` assignment. Same anchoring
# as RESOURCE_FIELD_RE, but this one is used to locate the right-hand side so
# the expression can be read out, not merely to note that the field is set.
FIELD_ASSIGN_RE = re.compile(r"^[ \t]*(cpus|memory|time)[ \t]*=[ \t]*", re.MULTILINE)

# Any decimal number, in a config expression or a doc table cell.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# A duration written the way both conf/modules.config and the doc write it:
# `2.h`, `12.h`. Restricting the doc side to this shape keeps the CPU count and
# the tier thresholds in neighbouring cells from being read as hours.
HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\.\s*h\b")

# The parameters a memory closure may be DERIVED FROM: the reads in the inlined
# `memory = { ... }` arithmetic of the STARE rows (tile, halo, out tile, coarse
# size), TILE_FOR_BASIC / APPLY_PROFILES (`preproc_tile_size`) and
# MERGE_AND_PYRAMID (`pyramid_resolutions`, `pyramid_scale`). None of these
# values is computable from pytest, so the doc row must name every term instead.
#
# Enumerated rather than `params\.(\w+)`: the exemption skips the numeric
# comparison entirely, so a broad pattern would swallow any closure that merely
# mentions a parameter (`params.big_run ? 128.GB : 32.GB * task.attempt`), which
# should instead get memory_cell_mismatch's "fourth shape" refusal.
# test_derived_memory_exemption_does_not_swallow_a_param_dependent_choice pins
# that; test_derived_memory_param_allowlist_is_not_stale keeps this list honest.
# Add a name only when the new closure's value cannot be computed from pytest.
DERIVED_MEMORY_PARAMS = (
    "reg_tiled_tile",
    "reg_tiled_halo",
    "reg_tiled_out_tile",
    "reg_tiled_coarse_max_dim",
    "preproc_tile_size",
    "pyramid_resolutions",
    "pyramid_scale",
)

# A memory request derived from params at submission time: an allowlisted
# `params.reg_tiled_tile`-style read, or a `helper(params)` call. The capture is
# the derivation source (parameter or helper name). Only the param-read form
# occurs in conf/modules.config; test_derived_memory_detects_both_forms covers
# both, since neither is arithmetic expr_values_per_attempt can evaluate.
DERIVED_HELPER_RE = re.compile(
    r"\b(\w+)\(\s*params\s*\)|\bparams\.(" + "|".join(DERIVED_MEMORY_PARAMS) + r")\b"
)


def derived_memory_sources(expr: str) -> set[str]:
    """Return the derivation sources `expr` reads, or an empty set if it is not
    param-derived at all. A non-empty result grants the exemption: the numeric
    comparison is skipped and the doc cell is required to name every source
    instead."""
    return {m.group(1) or m.group(2) for m in DERIVED_HELPER_RE.finditer(expr)}


# One rung of a size-tier ladder, CONFIG side: `file_gb < 10 ? 32.GB`. Threshold
# and magnitude are captured together, in one match, so a doc cell that pairs
# the right numbers backwards (`f<5 -> 64, f<15 -> 32` against the inverse) fails.
CONFIG_TIER_RE = re.compile(r"<\s*(\d+(?:\.\d+)?)\s*\?\s*(\d+(?:\.\d+)?)\s*\.GB")

# The ladder's final `else` branch: the first `: N.GB` after its last rung.
CONFIG_ELSE_RE = re.compile(r":\s*(\d+(?:\.\d+)?)\s*\.GB")

# A non-ladder size-dependent shape, `... * N.GB * task.attempt + M.GB`, linear in file
# size: coefficient and flat addend captured positionally (not pooled), so a doc cell
# cannot state the right pair of numbers in the wrong roles and pass.
#
# Nothing in conf/modules.config currently matches this shape. It is kept so a
# size-linear closure is checked if one returns, and is unit-tested on a synthetic
# literal in test_memory_cell_mismatch_compares_pairs_not_pooled_numbers.
CONFIG_SIZE_LINEAR_RE = re.compile(
    r"\*\s*(\d+(?:\.\d+)?)\s*\.GB\s*\*\s*task\.attempt\s*\+\s*(\d+(?:\.\d+)?)\s*\.GB"
)

# The same three shapes as docs/resources.md writes them.
#   rung:   `f<10` -> 32       (backtick and `+` sign both optional)
#   else:   else 128 GB / else +64 GB
#   linear: `f x 7 GB x attempt + 8 GB`
DOC_TIER_RE = re.compile(
    r"f\s*<\s*(\d+(?:\.\d+)?)\s*`?\s*(?:\u2192|->)\s*\+?\s*(\d+(?:\.\d+)?)"
)
DOC_ELSE_RE = re.compile(r"else\s*\+?\s*(\d+(?:\.\d+)?)")
DOC_SIZE_LINEAR_RE = re.compile(
    r"f\s*\u00d7\s*(\d+(?:\.\d+)?)\s*GB\s*\u00d7\s*attempt\s*\+\s*(\d+(?:\.\d+)?)\s*GB"
)

# The doubling ramp as the doc renders it: `2^(attempt-1)` (Unicode minus or
# ASCII hyphen). Its `2` and `1` are notation, not memory magnitudes, so the
# whole token is removed before the cell's numbers are read.
DOC_RAMP_RE = re.compile(r"\d+\s*\^\s*\(\s*attempt\s*[\u2212-]\s*\d+\s*\)")

# Nextflow retries up to maxRetries = 3 (conf/base.config), so a task runs on
# attempt 1..4 and a ramp has exactly four rungs. docs/resources.md enumerates
# them for WARP_SEG_QC ("32 / 64 / 128 / 256"); computing the same four values
# from the config expression is what makes those numbers checkable.
ATTEMPTS = (1, 2, 3, 4)


def _closure_body(text: str, open_brace_index: int) -> str:
    """Return the body between `text[open_brace_index] == '{'` and its partner.

    A thin adapter over `tests.nfmodel.block_extent`, which wants `start` just
    past the opening `{` and returns the index just past the matching `}`.
    This takes the index of the opening brace and returns the body excluding
    both braces. The walk is comment/string-aware.
    """
    assert text[open_brace_index] == "{"
    end = _block_extent(text, open_brace_index + 1)
    return text[open_brace_index + 1 : end - 1]


def parse_resource_exprs(body: str) -> dict[str, str]:
    """Return {field: right-hand-side expression} for a withName/withLabel body.

    The right-hand side is either a closure (`memory = { ... }`, read by
    brace-matching, since it nests) or a bare literal to end of line
    (`cpus = 8`). `tests.nfmodel.strip_comments` (view B) then drops any
    `/* */` or `//` comment inside the expression while keeping string
    contents verbatim, so a stale number in a comment (`// was 999.GB`)
    cannot license the same stale number in docs/resources.md.
    """
    exprs: dict[str, str] = {}
    for match in FIELD_ASSIGN_RE.finditer(body):
        pos = match.end()
        if pos < len(body) and body[pos] == "{":
            expr = _closure_body(body, pos)
        else:
            end = body.find("\n", pos)
            expr = body[pos:] if end == -1 else body[pos:end]
        exprs[match.group(1)] = _strip_comments(expr).strip()
    return exprs


def find_label_resource_exprs() -> dict[str, dict[str, str]]:
    """Return {label: {field: expression}} for every `withLabel:` block."""
    text = MODULES_CONFIG.read_text()
    return {
        (m.group(1) if m.group(1) is not None else m.group(2)): parse_resource_exprs(
            _closure_body(text, m.end() - 1)
        )
        for m in WITH_LABEL_OPEN_RE.finditer(text)
    }


def find_withname_resource_exprs() -> dict[str, dict[str, str]]:
    """Return {PROCESS: {field: expression}} unioned over matching withName blocks.

    Sourced from `tests.nfmodel.with_name_blocks()`, which ignores a selector
    that only appears inside a comment.
    """
    exprs: dict[str, dict[str, str]] = {}
    for block in _with_name_blocks():
        found = parse_resource_exprs(block.raw_body[:-1])
        for name in block.names:
            exprs.setdefault(name, {}).update(found)
    return exprs


def find_process_labels() -> dict[str, list[str]]:
    """Return {PROCESS: [labels it can resolve to]} for modules/local/*.nf.

    A list, not a single value: `export_spatialdata.nf`'s label is a ternary and
    can resolve to either branch depending on `--spatialdata_include_image`.
    """
    labels: dict[str, list[str]] = {}
    for name, path in find_module_processes().items():
        found: list[str] = []
        for line in path.read_text().splitlines():
            if LABEL_LINE_RE.match(line):
                found.extend(extract_labels_from_line(line))
        labels[name] = found
    return labels


def _eval_arithmetic(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or return None if it is not one.

    Only numbers and `+ - * / ** ( )` are honoured. Anything else -- a ternary,
    a `.size()` call, a `?:` -- returns None, which is how a size-dependent tier
    is distinguished from a flat request. Uses `ast` rather than `eval` so a
    stray identifier cannot be executed.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = walk(node.operand)
            return (
                None
                if operand is None
                else (operand if isinstance(node.op, ast.UAdd) else -operand)
            )
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
        return None

    return walk(tree)


def expr_values_per_attempt(expr: str) -> list[float] | None:
    """Return the expression's value on attempts 1..4, or None if not a constant.

    `12.GB * task.attempt` -> [12, 24, 36, 48]; `100.GB + 100.GB * task.attempt`
    -> [200, 300, 400, 500]; `32.GB * (2 ** (task.attempt - 1))` ->
    [32, 64, 128, 256]. A tiered closure (`(f < 10 ? 32.GB : ...)`) or a
    param-derived one returns None -- those are compared differently.
    """
    if derived_memory_sources(expr):
        return None
    values = []
    for attempt in ATTEMPTS:
        arithmetic = (
            expr.replace("task.attempt", str(attempt))
            .replace(".GB", "")
            .replace(".h", "")
        )
        value = _eval_arithmetic(arithmetic)
        if value is None:
            return None
        values.append(value)
    return values


def _numbers(text: str) -> set[float]:
    return {float(m.group(0)) for m in NUMBER_RE.finditer(text)}


def _gb_literal_counts(expr: str) -> Counter:
    """Multiset of the `N.GB` magnitudes a config expression names.

    A multiset, not a set: CONVERT_IMAGE's closure names `24.GB` twice (once as
    the per-attempt base, once as a tier constant) and collapsing them would let
    the doc drop one of the two and still pass.
    """
    return Counter(
        float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\.GB", expr)
    )


def _doc_magnitude_counts(cell: str) -> Counter:
    """Multiset of the GB magnitudes a doc memory cell states, for a FLAT row.

    The doubling-ramp notation (`2^(attempt-1)`) is removed first: its `2` and
    `1` are structure, not magnitudes. Tiered and size-linear rows are compared
    as (threshold, magnitude) pairs and positional roles instead.
    """
    stripped = DOC_RAMP_RE.sub(" ", cell)
    return Counter(float(m.group(0)) for m in NUMBER_RE.finditer(stripped))


def _config_ladder(expr: str):
    """Return (rungs, else_gb, leftover_gb) for a size-tier ladder, or None.

    `rungs` is an ORDERED list of (threshold, GB) pairs read straight off the
    ternary chain, so `file_gb < 5 ? 32.GB : file_gb < 15 ? 64.GB : 128.GB`
    gives [(5, 32), (15, 64)] with else_gb 128. `leftover_gb` is the multiset of
    `N.GB` literals outside the ladder (e.g. CONVERT_IMAGE's `+ 24.GB *
    task.attempt` per-attempt base).
    """
    rungs = [
        (float(m.group(1)), float(m.group(2))) for m in CONFIG_TIER_RE.finditer(expr)
    ]
    if not rungs:
        return None
    last = list(CONFIG_TIER_RE.finditer(expr))[-1]
    else_match = CONFIG_ELSE_RE.search(expr, last.end())
    else_gb = float(else_match.group(1)) if else_match else None
    consumed = Counter(gb for _, gb in rungs)
    if else_gb is not None:
        consumed[else_gb] += 1
    return rungs, else_gb, _gb_literal_counts(expr) - consumed


def _doc_ladder(cell: str):
    """Return (rungs, else_gb, leftover_gb) as docs/resources.md states them."""
    rungs = [(float(m.group(1)), float(m.group(2))) for m in DOC_TIER_RE.finditer(cell)]
    else_match = DOC_ELSE_RE.search(cell)
    else_gb = float(else_match.group(1)) if else_match else None
    leftover_text = DOC_TIER_RE.sub(" ", DOC_RAMP_RE.sub(" ", cell))
    if else_match:
        leftover_text = leftover_text.replace(else_match.group(0), " ", 1)
    leftover = Counter(float(m.group(0)) for m in NUMBER_RE.finditer(leftover_text))
    return rungs, else_gb, leftover


def _config_size_linear(expr: str):
    """Return (coefficient, addend, leftover_gb) for `* A.GB * attempt + B.GB`, or None."""
    match = CONFIG_SIZE_LINEAR_RE.search(expr)
    if not match:
        return None
    coefficient, addend = float(match.group(1)), float(match.group(2))
    return (
        coefficient,
        addend,
        _gb_literal_counts(expr) - Counter([coefficient, addend]),
    )


def _doc_size_linear(cell: str):
    """Return (coefficient, addend, leftover_gb) as docs/resources.md states it, or None."""
    match = DOC_SIZE_LINEAR_RE.search(cell)
    if not match:
        return None
    coefficient, addend = float(match.group(1)), float(match.group(2))
    leftover_text = cell.replace(match.group(0), " ", 1)
    leftover = Counter(float(m.group(0)) for m in NUMBER_RE.finditer(leftover_text))
    return coefficient, addend, leftover


# A flat ramp whose first-attempt magnitude a STRING PARAM selects (REGISTER's per
# reg_valis_mode request): `(params.reg_valis_mode == 'hires' ? 128.GB : 64.GB) * (2 ** ...)`.
# Each branch is itself a flat request, so each is evaluated, not pattern-matched.
PARAM_SELECTED_RE = re.compile(
    r"^\(\s*params\.(\w+)\s*==\s*'([^']*)'\s*\?\s*(\d+(?:\.\d+)?)\s*\.GB\s*:"
    r"\s*(\d+(?:\.\d+)?)\s*\.GB\s*\)\s*(\*.*)$",
    re.S,
)


def _param_selected_mismatch(m: re.Match, expr: str, cell: str) -> str | None:
    param, value, when_gb, else_gb, rest = m.groups()
    branches = {}
    for label, gb in ((value, when_gb), ("otherwise", else_gb)):
        values = expr_values_per_attempt(f"{gb}.GB {rest}")
        if values is None:
            return f"`{expr}`'s {label} branch is not a flat request this guard can evaluate"
        branches[label] = values
    if param not in cell or value not in cell:
        return f"does not name what selects the request (`{param}` = `{value}`)"
    documented = _doc_magnitude_counts(cell)
    every = {v for values in branches.values() for v in values}
    extra = sorted(set(documented) - every)
    if extra:
        return (
            f"states {extra} GB, which neither branch of `{expr}` takes on any attempt"
        )
    for label, values in branches.items():
        if values[0] not in documented:
            return f"never states the {label} branch's attempt-1 value {values[0]} GB"
    return None


def memory_cell_mismatch(expr: str, cell: str) -> str | None:
    """Return why `cell` misstates `expr`, or None if it states it correctly.

    Three shapes (plus the param-selected flat ramp handled first). Anything
    else fails loudly rather than falling through to a looser comparison.

    * a **flat** request (`32.GB * task.attempt`, `100.GB + 100.GB *
      task.attempt`, `32.GB * (2 ** (task.attempt - 1))`) is evaluated on
      attempts 1..4; every magnitude in the doc cell must be one of those four
      values, and the attempt-1 value must be among them. Comparing against
      computed values stops a tier constant or a `>> 30` byte shift from
      laundering a wrong number.
    * a **size-tier ladder** (`(file_gb < 10 ? 32.GB : ...) * task.attempt`)
      cannot be evaluated without a file, so its rungs are compared as an
      ORDERED list of (threshold, magnitude) PAIRS, plus the `else` branch, plus
      a multiset of any `N.GB` outside the ladder, so thresholds and magnitudes
      paired backwards fail.
    * a **size-linear** request (`* 7.GB * task.attempt + 8.GB`) has no `<`, so
      it is not a ladder; its two numbers are compared positionally,
      coefficient against coefficient and addend against addend.
    """
    selected = PARAM_SELECTED_RE.match(expr.strip())
    if selected:
        return _param_selected_mismatch(selected, expr, cell)

    values = expr_values_per_attempt(expr)
    if values is not None:
        documented = _doc_magnitude_counts(cell)
        if not documented:
            return f"states no GB value at all against a flat request of {values[0]} GB"
        extra = sorted(set(documented) - set(values))
        if extra:
            return (
                f"states {extra} GB, which is not a value `{expr}` takes on any "
                f"attempt (it gives {values})"
            )
        if values[0] not in documented:
            return f"never states the attempt-1 value {values[0]} GB of `{expr}`"
        return None

    ladder = _config_ladder(expr)
    if ladder is not None:
        config_rungs, config_else, config_extra = ladder
        doc_rungs, doc_else, doc_extra = _doc_ladder(cell)
        if doc_rungs != config_rungs:
            mispaired = [
                f"f<{int(th) if th.is_integer() else th} states "
                f"{int(gb) if gb.is_integer() else gb} GB"
                for th, gb in doc_rungs
                if (th, gb) not in config_rungs
            ]
            return f"tiers as {doc_rungs} but `{expr}` tiers as {config_rungs}" + (
                f" -- mispaired: {mispaired}" if mispaired else ""
            )
        if doc_else != config_else:
            return (
                f"gives the else branch {doc_else} GB but `{expr}` gives {config_else}"
            )
        if doc_extra != config_extra:
            return (
                f"states {sorted(doc_extra.elements())} GB outside the tier ladder "
                f"but `{expr}` has {sorted(config_extra.elements())}"
            )
        return None

    linear = _config_size_linear(expr)
    if linear is not None:
        config_coefficient, config_addend, config_extra = linear
        documented = _doc_size_linear(cell)
        if documented is None:
            return (
                f"does not state `{expr}`'s size-linear shape "
                f"(`f x {config_coefficient} GB x attempt + {config_addend} GB`)"
            )
        doc_coefficient, doc_addend, doc_extra = documented
        if (doc_coefficient, doc_addend) != (config_coefficient, config_addend):
            return (
                f"states coefficient {doc_coefficient} GB/GiB and addend "
                f"{doc_addend} GB, but `{expr}` has coefficient "
                f"{config_coefficient} GB/GiB and addend {config_addend} GB"
            )
        if doc_extra != config_extra:
            return (
                f"states {sorted(doc_extra.elements())} GB outside the size-linear "
                f"term but `{expr}` has {sorted(config_extra.elements())}"
            )
        return None

    return (
        f"cannot be checked: `{expr}` is none of the three shapes this guard "
        "knows -- a flat request it can evaluate, a `< N ? M.GB` tier ladder, or "
        "a `* A.GB * task.attempt + B.GB` size-linear term. Teach "
        "memory_cell_mismatch the new shape or exempt the row explicitly; do not "
        "leave the field silently unchecked"
    )


def parse_doc_process_table() -> dict[str, dict[str, str]]:
    """Return {PROCESS: {column: cell}} for every per-process row in docs/resources.md.

    Only tables whose first header cell is literally `Process` are read, which
    excludes the label table (`Label`), the fallback table (`Resource`), the
    ceiling tables and the container table. Column lookup is by header name, so
    the tiled table's extra `maxForks` column does not shift the others.
    """
    lines = DOCS_RESOURCES.read_text().splitlines()
    rows: dict[str, dict[str, str]] = {}
    headers: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            headers = None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set(stripped) <= set("|- "):  # the `|---|---|` separator row
            continue
        normalised = [
            (c.replace("`", "").replace("*", "").split() or [""])[0].lower()
            for c in cells
        ]
        if normalised and normalised[0] == "process":
            headers = normalised
            continue
        if headers is None:
            continue
        name = cells[0].strip("`")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            continue
        rows[name] = dict(zip(headers, cells))
    return rows


def resource_owner_case(name: str) -> str:
    """Return 'withName', 'label' or 'partial' for a module process.

    Mirrors the three cases docs/resources.md's `<div class="gate">` names, and
    is what the per-row Owner column and the three case counts are checked
    against.
    """
    fields = find_withname_resource_fields().get(name, set())
    if not find_process_labels().get(name):
        return "withName"
    return "label" if not fields else "partial"


def effective_exprs(name: str, field: str) -> list[str]:
    """Return the expression(s) that can supply `field` for `name`.

    One entry normally; two when the module's `label` is a ternary and the field
    is label-owned, since either branch is reachable at runtime.
    """
    base = ALIASES.get(name, name)
    withname = find_withname_resource_exprs()
    expr = withname.get(name, {}).get(field) or withname.get(base, {}).get(field)
    if expr:
        return [expr]
    label_exprs = find_label_resource_exprs()
    return [
        label_exprs[label][field]
        for label in find_process_labels().get(base, [])
        if field in label_exprs.get(label, {})
    ]


def test_strip_comments_removes_numbers_a_maintainer_would_annotate_with():
    """A number inside a comment in a memory closure must not count as config.

    conf/modules.config may have no such comment, so this exercises the strip
    in `tests.nfmodel.strip_comments` (imported here as `_strip_comments`).
    """
    annotated = "32.GB * task.attempt   // was 64.GB before the bounding-box rewrite"
    assert _strip_comments(annotated).strip() == "32.GB * task.attempt"
    assert 64.0 not in _numbers(_strip_comments(annotated))
    assert _strip_comments(
        "1.GB /* 4.GB was 120x the observed peak */ * task.attempt"
    ).split() == [
        "1.GB",
        "*",
        "task.attempt",
    ]


def test_doc_table_parser_finds_the_per_process_rows_only():
    """Regression guard on the doc-table parser itself.

    It must pick up the per-process tables and skip the label table -- whose
    rows (`process_single`, ...) are lower-case and whose first header cell is
    `Label`, not `Process`. A parser that swallowed those would compare a label
    against conf/modules.config as if it were a process and could never fail,
    and one that mis-indexed columns would compare cpus against memory.
    """
    rows = parse_doc_process_table()
    assert "SEGMENT" in rows, (
        "per-process tables not parsed -- scan pattern may be stale"
    )
    assert "process_single" not in rows, (
        "the `Label` table must not be read as processes"
    )
    assert rows["SEGMENT"]["cpus"] == "`8`"
    assert rows["SEGMENT"]["time"] == "`4.h × attempt`"
    assert rows["SEGMENT"]["owner"] == "`withName`"
    # The tiled table carries an extra trailing column; lookup is by header
    # name, so `time` must still be the time and not shift into `maxForks`.
    assert rows["TILED_SOLVE"]["time"] == "`8.h × attempt` *(label)*"
    assert rows["TILED_SOLVE"]["maxforks"] == "—"


def test_expr_values_per_attempt_handles_the_shapes_in_modules_config():
    """Regression guard on the evaluator.

    It must fold the two ramp shapes conf/modules.config uses (linear and
    doubling) and the `A + B * attempt` shape of the medium/high labels, and it
    must REFUSE a tiered or param-derived closure rather than silently returning
    a partial number that would then be compared as if it were the whole rule.
    """
    assert expr_values_per_attempt("12.GB * task.attempt") == [12, 24, 36, 48]
    assert expr_values_per_attempt("100.GB + 100.GB * task.attempt") == [
        200,
        300,
        400,
        500,
    ]
    assert expr_values_per_attempt("32.GB * (2 ** (task.attempt - 1))") == [
        32,
        64,
        128,
        256,
    ]
    assert expr_values_per_attempt("8") == [8, 8, 8, 8]
    assert (
        expr_values_per_attempt(
            "(Math.ceil(tiledRegTileGb(params)) as int).GB * task.attempt"
        )
        is None
    )
    assert (
        expr_values_per_attempt("(file_gb < 10 ? 32.GB : 64.GB) * task.attempt") is None
    )
    assert (
        expr_values_per_attempt(
            "def win = ((params.reg_tiled_out_tile as int) as long)\n"
            "(Math.ceil(Math.max(4d, (0.6d + 0.17d * mpx) * 2d)) as int).GB * task.attempt"
        )
        is None
    )


def test_derived_memory_detects_both_forms():
    """DERIVED_HELPER_RE must capture the derivation SOURCE in either form.

    The inlined `params.*` form is the one conf/modules.config uses (the strict
    config parser forbids a helper function). The helper form is still routed to
    the exemption rather than compared against a number it does not have.
    """

    sources = derived_memory_sources

    assert sources("(Math.ceil(tiledRegTileGb(params)) as int).GB * task.attempt") == {
        "tiledRegTileGb"
    }
    assert sources(
        "def win = (((params.reg_tiled_tile as int) + 2 * (params.reg_tiled_halo as int)) as long)"
    ) == {"reg_tiled_tile", "reg_tiled_halo"}
    # A constant request is NOT derived -- otherwise every row would take the
    # exemption path and the numeric comparison would check nothing.
    assert sources("32.GB * task.attempt") == set()
    assert sources("(file_gb < 10 ? 32.GB : 64.GB) * task.attempt") == set()


def test_derived_memory_exemption_does_not_swallow_a_param_dependent_choice():
    """A memory closure that merely MENTIONS a param must not be exempted.

    The exemption skips the numeric comparison outright and asks only that the
    doc cell contain the source's name, so every expression it swallows is an
    expression nothing checks the magnitudes of. That suits closures whose value
    cannot be computed from pytest, not a param-dependent choice between literal
    reservations, whose magnitudes the doc cell should have to state.

    Both expressions below get the "fourth shape" refusal from
    memory_cell_mismatch when not exempted, which is what this pins.
    """
    counterexamples = [
        "params.big_run ? 128.GB : 32.GB * task.attempt",
        "(params.seg_gpu ? 64 : 32).GB * task.attempt",
    ]
    for expr in counterexamples:
        assert derived_memory_sources(expr) == set(), (
            f"`{expr}` was granted the derived-memory exemption. It is a choice "
            "between literal reservations, not a value derived from a parameter "
            "-- exempting it means no doc row states either magnitude and "
            "nothing notices"
        )
        # Not evaluable, not a ladder, not size-linear: with the exemption
        # withheld it must reach the refusal branch, or the row would go
        # unchecked by a different route.
        assert "cannot be checked" in (
            memory_cell_mismatch(expr, "`32 GB \u00d7 attempt`") or ""
        ), f"`{expr}` is silently accepted by memory_cell_mismatch"


def test_derived_memory_param_allowlist_is_not_stale():
    """Every name in the allowlist must really be read by a memory expression.

    A stale entry looks like coverage and checks nothing. If a closure stops
    reading one of these, drop it here too; the row then goes back through the
    numeric comparison.
    """
    memory_exprs = [
        expr
        for name in parse_doc_process_table()
        for expr in effective_exprs(name, "memory")
    ]
    unread = [
        param
        for param in DERIVED_MEMORY_PARAMS
        if not any(f"params.{param}" in expr for expr in memory_exprs)
    ]
    assert not unread, (
        f"{unread} are allowlisted as derived-memory sources but no `memory` "
        "expression in conf/modules.config reads them. Remove the stale "
        "entries rather than leaving an exemption that covers nothing."
    )


def test_memory_cell_mismatch_compares_pairs_not_pooled_numbers():
    """The tiered and size-linear paths must bind each number to its role.

    A pooled comparison of thresholds and magnitudes would let a doc cell hold
    the right numbers in the wrong places. The last two cases are the other
    direction: duplicate magnitudes and duplicate thresholds are legitimate
    config and must not false-positive.
    """
    ladder = "(file_gb < 5 ? 32.GB : file_gb < 15 ? 64.GB : 128.GB) * task.attempt"
    correct = "tier: `f<5` \u2192 32, `f<15` \u2192 64, else 128 GB, `\u00d7 attempt`"
    swapped = "tier: `f<5` \u2192 64, `f<15` \u2192 32, else 128 GB, `\u00d7 attempt`"
    assert memory_cell_mismatch(ladder, correct) is None
    assert "mispaired" in (memory_cell_mismatch(ladder, swapped) or "")

    linear = "(ome_tiff.size() >> 30 ?: 1) * 7.GB * task.attempt + 8.GB"
    assert memory_cell_mismatch(linear, "`f \u00d7 7 GB \u00d7 attempt + 8 GB`") is None
    assert "coefficient" in (
        memory_cell_mismatch(linear, "`f \u00d7 8 GB \u00d7 attempt + 7 GB`") or ""
    )

    # A param-selected flat ramp (REGISTER per reg_valis_mode): BOTH branches must be stated.
    selected = "(params.reg_valis_mode == 'hires' ? 128.GB : 64.GB) * (2 ** (task.attempt - 1))"
    both = "`64 GB` (64 -> 512); at `reg_valis_mode hires` `128 GB` (128 -> 1024)"
    assert memory_cell_mismatch(selected, both) is None
    assert "attempt-1" in (
        memory_cell_mismatch(selected, "`64 GB`; at `reg_valis_mode hires` from 256 GB")
        or ""
    )
    assert "selects" in (memory_cell_mismatch(selected, "`64 GB`, `128 GB`") or "")
    assert "neither branch" in (
        memory_cell_mismatch(selected, "`64 GB`; at `reg_valis_mode hires` `96 GB`")
        or ""
    )

    # Anything that is none of the known shapes must refuse loudly rather
    # than fall through to a looser comparison.
    assert "cannot be checked" in (
        memory_cell_mismatch("Runtime.getRuntime().maxMemory()", "`64 GB`") or ""
    )

    # Legitimate shapes a pairing comparison could wrongly reject:
    # two tiers sharing one magnitude, and two tiers sharing one threshold.
    same_gb = "(file_gb < 10 ? 32.GB : file_gb < 30 ? 32.GB : 128.GB) * task.attempt"
    assert (
        memory_cell_mismatch(
            same_gb,
            "tier: `f<10` \u2192 32, `f<30` \u2192 32, else 128 GB, `\u00d7 attempt`",
        )
        is None
    )
    same_threshold = "(a < 10 ? 32.GB : b < 10 ? 64.GB : 128.GB) * task.attempt"
    assert (
        memory_cell_mismatch(
            same_threshold,
            "tier: `f<10` \u2192 32, `f<10` \u2192 64, else 128 GB, `\u00d7 attempt`",
        )
        is None
    )


def test_every_documented_process_matches_conf_modules_config():
    """Direction 1: every row in docs/resources.md agrees with the config.

    cpus and time are compared as numbers. Memory is compared as the set of
    magnitudes the config expression can produce (its literals plus its value on
    attempts 1..4), with the attempt-1 value required to appear for a flat
    request and every `N.GB` literal required to appear for a size-tiered one.
    Param-derived rows are matched on their derivation sources instead -- see
    the header of this section.
    """
    doc_rows = parse_doc_process_table()
    processes = find_module_processes()
    assert doc_rows, (
        "No per-process rows parsed from docs/resources.md -- pattern may be stale."
    )

    problems: list[str] = []
    for name in sorted(doc_rows):
        row = doc_rows[name]
        base = ALIASES.get(name, name)
        if base not in processes:
            problems.append(
                f"{name}: documented in docs/resources.md but no `process {base} {{` "
                "exists under modules/local/ -- delete the row or add the alias to "
                "ALIASES in this test."
            )
            continue

        # --- cpus and time -------------------------------------------------
        # Both fail loudly rather than skipping when the expression is something
        # this guard cannot evaluate (e.g. `cpus = { params.seg_gpu ? 16 : 12 }`).
        for field, documented in (
            ("cpus", _numbers(row.get("cpus", ""))),
            (
                "time",
                {float(m.group(1)) for m in HOURS_RE.finditer(row.get("time", ""))},
            ),
        ):
            exprs = effective_exprs(name, field)
            if not exprs:
                problems.append(
                    f"{name}: nothing in conf/modules.config sets `{field}` -- neither a "
                    "`withName:` block nor the module's label. The doc row cannot be "
                    "checked and the process would run on conf/base.config's defaults."
                )
                continue
            expected = set()
            unevaluable = [e for e in exprs if expr_values_per_attempt(e) is None]
            if unevaluable:
                problems.append(
                    f"{name}: conf/modules.config's `{field}` is not a plain number this "
                    f"guard can evaluate: {unevaluable}. The doc row states "
                    f"{sorted(documented)} and NOTHING is checking it. Simplify the "
                    "config expression, or teach expr_values_per_attempt the new shape, "
                    "or exempt the row explicitly the way the derived memory closures "
                    "are -- silently skipping the field is not an option."
                )
                continue
            for expr in exprs:
                expected.add(expr_values_per_attempt(expr)[0])
            if not (documented and documented <= expected):
                unit = "" if field == "cpus" else " h"
                problems.append(
                    f"{name}: doc table says {field} {sorted(documented)}{unit} but "
                    f"conf/modules.config gives {sorted(expected)}{unit}"
                )

        # --- memory --------------------------------------------------------
        exprs = effective_exprs(name, "memory")
        cell = row.get("memory", "")
        helpers = {src for expr in exprs for src in derived_memory_sources(expr)}
        if helpers:
            missing = sorted(h for h in helpers if h not in cell)
            if missing:
                problems.append(
                    f"{name}: memory is derived from {missing} in conf/modules.config, "
                    "but the doc cell does not name the derivation source -- a doc row "
                    "stating a flat number here would be wrong the moment a parameter "
                    f"changes: {cell!r}"
                )
        elif not exprs:
            problems.append(
                f"{name}: nothing in conf/modules.config sets `memory` -- neither a "
                "`withName:` block nor the module's label."
            )
        else:
            # One expression normally; two when the module's label is a ternary,
            # in which case either branch is a legitimate thing to document.
            reasons = [memory_cell_mismatch(expr, cell) for expr in exprs]
            if all(reasons):
                # More than one reason only when the module's label is a ternary
                # and neither branch matches -- then say so for both branches
                # rather than picking one arbitrarily.
                joined = "; also ".join(dict.fromkeys(r for r in reasons if r))
                problems.append(f"{name}: doc table's memory cell {cell!r} {joined}")

        # --- owner ---------------------------------------------------------
        case = resource_owner_case(base)
        owner_cell = row.get("owner", "")
        if case == "withName":
            ok = "withName" in owner_cell
        elif case == "partial":
            ok = "partial" in owner_cell
        else:
            ok = any(label in owner_cell for label in find_process_labels()[base])
        if not ok:
            problems.append(
                f"{name}: doc table's Owner column says {owner_cell!r}, but "
                f"conf/modules.config makes it a '{case}' process"
            )

    assert not problems, (
        f"{len(problems)} row(s) in docs/resources.md disagree with "
        "conf/modules.config. conf/modules.config is the source of truth; fix "
        "the doc:\n" + "\n".join(problems)
    )


def test_every_configured_process_is_documented():
    """Direction 2: every process with a resource owner has a doc-table row.

    A process added to modules/local/ with a label or a `withName:` resource
    block but no row is invisible to anyone sizing a cluster allocation from
    docs/resources.md -- the table silently under-reports what the run will ask
    the scheduler for.
    """
    documented = {ALIASES.get(n, n) for n in parse_doc_process_table()}
    labels = find_process_labels()
    withname_fields = find_withname_resource_fields()

    missing = sorted(
        name
        for name in find_module_processes()
        if (labels.get(name) or withname_fields.get(name)) and name not in documented
    )
    assert not missing, (
        f"{len(missing)} process(es) have a resource owner in conf/modules.config "
        "but no row in docs/resources.md's per-process table -- add one under the "
        f"matching section: {missing}"
    )


def test_one_owner_rule_case_counts_match_conf_modules_config():
    """The three counts in docs/resources.md's `case 1/2/3` panel must be real."""
    counts = {"withName": 0, "label": 0, "partial": 0}
    for name in find_module_processes():
        counts[resource_owner_case(name)] += 1

    text = DOCS_RESOURCES.read_text()
    documented = [
        int(m.group(1))
        for m in re.finditer(r'<div class="d">[^<]*?(\d+) processes\.</div>', text)
    ]
    assert len(documented) == 3, (
        "Expected three `N processes.` counts in the one-owner-rule panel of "
        f"docs/resources.md, found {len(documented)} -- scan pattern may be stale."
    )
    assert documented == [counts["withName"], counts["label"], counts["partial"]], (
        f"docs/resources.md's one-owner-rule panel claims {documented} processes "
        f"in cases 1/2/3, but conf/modules.config + modules/local/*.nf give "
        f"{[counts['withName'], counts['label'], counts['partial']]}."
    )
