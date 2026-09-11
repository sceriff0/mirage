"""Meta-test for the shared Nextflow source model.

Each test here plants a construct that historically defeated a private
regex parse, and asserts the model still sees the code that follows it.
A guard built on this model is only as trustworthy as this file.
"""

import re

from tests.nfmodel import (
    REPO_ROOT,
    block_extent,
    include_aliases,
    nf_files,
    nf_test_cases,
    param_refs,
    processes,
    script_bodies,
    strip_comments,
    strip_comments_and_strings,
    with_name_blocks,
    workflows,
)


def test_stage_as_glob_does_not_open_a_block_comment():
    """`stageAs: 'ref/*'` contains `/*`. A DOTALL regex reads that as an open
    block comment and eats everything to the next `*/` -- 64 lines of
    register.nf, including its whole script: block. The string-aware walk must
    treat it as a string literal."""
    src = (
        "process REGISTER {\n"
        "    input:\n"
        "    tuple val(meta), path(reference, stageAs: 'ref/*'), path(pre)\n"
        "\n"
        "    output:\n"
        "    path 'versions.yml', emit: versions\n"
        "\n"
        "    script:\n"
        "    MARKER_MUST_SURVIVE\n"
        "}\n"
    )
    clean = strip_comments_and_strings(src)
    assert "MARKER_MUST_SURVIVE" in clean
    assert "script:" in clean
    assert "emit: versions" in clean


def test_line_numbers_and_length_are_preserved():
    """Guards report file:line. Blanking must not shift a single character."""
    src = "a\n// comment\n'string'\n/* block */\nb\n"
    clean = strip_comments_and_strings(src)
    assert len(clean) == len(src)
    assert clean.count("\n") == src.count("\n")
    assert clean.splitlines()[0] == "a"
    assert clean.splitlines()[4] == "b"


def test_comment_contents_are_blanked():
    src = "x\n// publishDir here\n/* publishDir there */\ny\n"
    clean = strip_comments_and_strings(src)
    assert "publishDir" not in clean


def test_string_contents_are_blanked():
    src = "path 'qc/*.png'\ndef s = \"groupTuple(sort: true)\"\n"
    clean = strip_comments_and_strings(src)
    assert "groupTuple" not in clean
    assert "qc" not in clean


def test_triple_quoted_script_body_is_blanked():
    """Nextflow script bodies are triple-quoted. A `//` inside one is not a
    comment, and a `'` inside one does not open a string."""
    src = 'script:\n"""\nfoo // not a comment\nbar\n"""\n'
    clean = strip_comments_and_strings(src)
    assert "not a comment" not in clean
    assert "script:" in clean


def test_triple_quote_is_matched_as_a_unit_not_three_single_quotes():
    """A `\"\"\"` delimiter must be matched to its own kind -- the next `\"\"\"`
    -- not discovered by pairing off single `"` characters one at a time.

    A parser whose single-/double-quote branch runs before its triple-quote
    special case treats the opening `\"\"\"` as a zero-width empty string (the
    first two `"` chars) immediately followed by a lone `"` that opens a
    *new* single-quoted string extending to the next literal `"` in the body.
    From there it keeps alternately pairing off whichever `"` characters
    happen to appear inside the body, blanking one span and leaking the next.

    That mispairing can happen to land back in sync by the time it reaches
    the real closing `\"\"\"` -- which is exactly what made the previous
    version of this test blind: its body (`foo // not a comment`) contains
    *zero* embedded `"` characters, an even count, so the mis-ordered walk
    re-synchronises at the close and both of that test's assertions pass by
    coincidence, indistinguishable from a correct parser (or even a
    context-free `//`-to-EOL stripper that never looks at quotes at all).

    An ODD number of embedded `"` characters removes that coincidence: the
    mis-ordered walk is still out of sync when it reaches the close, so it
    either swallows the trailing marker as part of a still-open "string" or
    leaks unblanked body words into the output. Either failure is directly
    observable, which is what makes this test an actual guard rather than a
    property that merely happens to hold today.
    """
    src = 'script:\n"""\necho "one" "two" "three\n"""\nMARKER_AFTER\n'
    clean = strip_comments_and_strings(src)
    assert "script:" in clean
    assert "MARKER_AFTER" in clean  # correct pairing closes the body here
    assert "one" not in clean  # body contents really were blanked
    assert "two" not in clean
    assert "three" not in clean


def test_escaped_quote_does_not_end_a_string():
    src = "def s = 'it\\'s fine'\nMARKER\n"
    clean = strip_comments_and_strings(src)
    assert "MARKER" in clean


def test_block_extent_walks_nested_braces():
    text = "withName: 'A' {\n  memory = { 4.GB * task.attempt }\n  cpus = 1\n}\nTAIL"
    start = text.index("{") + 1
    end = block_extent(text, start)
    assert "cpus = 1" in text[start:end]
    assert "TAIL" not in text[start:end]


def test_block_extent_ignores_braces_inside_strings():
    text = "withName: 'A' {\n  ext.args = '--x }'\n  cpus = 1\n}\nTAIL"
    start = text.index("{") + 1
    end = block_extent(text, start)
    assert "cpus = 1" in text[start:end]
    assert "TAIL" not in text[start:end]


def test_block_extent_handles_a_publishdir_list_containing_a_closure():
    """`publishDir` is routinely a LIST of maps, each holding its own closure
    -- `publishDir = [[path: {...}], [path: {...}]]` -- and a maintainer's
    inline comment near one of those closures can itself carry a stray,
    unbalanced `}` (explaining why the value looks the way it does). A brace
    count that is not comment-aware reads that stray `}` as closing real
    nesting one level early and truncates the block well before its actual
    close -- unlike the interpolated `${...}` inside the path strings here,
    whose brace IS balanced and so does not, by itself, distinguish a naive
    walk from a correct one; the comment's unmatched `}` is what does."""
    text = (
        "withName: 'A' {\n"
        "    publishDir = [\n"
        "        [ path: { \"a/${meta.id}\" }, mode: 'copy' ],\n"
        "        // stray } inside this comment must not count\n"
        "        [ path: { \"b/${meta.id}\" }, mode: 'copy', pattern: '*.png' ],\n"
        "    ]\n"
        "    cpus = 1\n"
        "}\n"
        "TAIL"
    )
    start = text.index("{") + 1
    end = block_extent(text, start)
    assert "cpus = 1" in text[start:end]
    assert "TAIL" not in text[start:end]


def test_block_extent_does_not_bleed_into_the_next_selector_block():
    """Two adjacent `withName:` selector blocks, back to back -- the first
    one's `ext.args` value is a string containing an unmatched `{` (a
    legitimate flag value, e.g. a glob-ish pattern, not real nesting). A
    brace count that is not string-aware reads that `{` as opening real
    nesting with nothing left to close it, so it never reaches depth zero at
    the block's real closing `}` and keeps counting straight through it and
    into the next selector's own body -- bleeding block A's extent into
    block B."""
    text = (
        "withName: 'A' {\n"
        "    ext.args = '--pattern {unmatched'\n"
        "    cpus = 1\n"
        "}\n"
        "withName: 'B' {\n"
        "    cpus = 99\n"
        "}\n"
    )
    start = text.index("{") + 1
    end = block_extent(text, start)
    assert "cpus = 1" in text[start:end]
    assert "cpus = 99" not in text[start:end]
    assert "'B'" not in text[start:end]


def test_nf_files_is_recursive():
    """A non-recursive glob('*.nf') misses modules/local/sub/*.nf. One guard
    used exactly that and could not see files it claimed to cover."""
    files = nf_files()
    assert files, "no .nf files found -- the enumerator is wrong, not the repo"
    rels = {f.relative_to(REPO_ROOT).as_posix() for f in files}
    assert "modules/local/register.nf" in rels
    assert "subworkflows/local/input_check.nf" in rels
    assert "workflows/mirage.nf" in rels


def test_register_script_body_is_visible():
    """The load-bearing case: REGISTER's script: block was invisible to
    test_resume_determinism because of stageAs: 'ref/*'."""
    body = script_bodies()["REGISTER"]
    assert body.strip(), "REGISTER's script: body came back empty"
    assert len(body.splitlines()) > 10


def test_every_module_local_process_is_modelled():
    """One process per file in modules/local/ is a project convention. If the
    model finds fewer processes than files, it is silently skipping some."""
    files = [p for p in (REPO_ROOT / "modules" / "local").rglob("*.nf")]
    assert len(processes()) >= len(files)


def test_workflows_sees_every_subworkflow_file():
    """A subworkflow name is a pipeline name. Anything resolving an
    UPPER_SNAKE_CASE name against `processes()` alone -- e.g. the
    docs/figures guard -- reports SEG_QC and VALIS_ADAPTER as nonexistent
    unless the model can see them, and would then be "fixed" by allow-listing
    real names."""
    files = list((REPO_ROOT / "subworkflows").rglob("*.nf"))
    assert files
    found = workflows()
    assert len(found) >= len(files), (
        f"{len(found)} named workflows found across {len(files)} subworkflow "
        "files -- the model is skipping some"
    )
    for expected in ("SEG_QC", "VALIS_ADAPTER", "REGISTER_PATIENT", "MIRAGE"):
        assert expected in found, f"{expected} is declared but the model missed it"


def test_workflows_ignores_one_named_only_in_a_comment(tmp_path):
    """Same blind spot `processes()` closes, on the workflow keyword."""
    d = tmp_path / "subworkflows"
    d.mkdir()
    (d / "x.nf").write_text(
        "// the workflow GHOST { ... } below was removed\n"
        "/* workflow ALSO_GHOST { */\n"
        "workflow REAL {\n    take:\n    ch\n}\n"
    )
    (tmp_path / "workflows").mkdir()
    assert set(workflows(root=tmp_path)) == {"REAL"}


def test_include_aliases_resolves_a_name_that_is_declared_nowhere():
    """SEG_QC_SEGMENT runs, appears in the trace and is selectable by
    `withName:`, yet no file declares `process SEG_QC_SEGMENT`. A resolver
    without this reports a live task as a typo."""
    aliases = include_aliases()
    assert aliases.get("SEG_QC_SEGMENT") == "SEGMENT"
    assert "SEG_QC_SEGMENT" not in processes()


def test_include_aliases_ignores_an_alias_inside_a_comment(tmp_path):
    d = tmp_path / "subworkflows"
    d.mkdir()
    (d / "x.nf").write_text(
        "// include { SEGMENT as GHOST } from '../m'\n"
        "include { SEGMENT as REAL_ALIAS } from '../m'\n"
    )
    (tmp_path / "modules").mkdir()
    (tmp_path / "workflows").mkdir()
    (tmp_path / "main.nf").write_text("workflow {}\n")
    assert include_aliases(root=tmp_path) == {"REAL_ALIAS": "SEGMENT"}


def test_with_name_blocks_finds_the_qc_selector():
    blocks = with_name_blocks()
    assert blocks, "no withName: blocks found"
    qc = [b for b in blocks if "WARP_SEG_QC" in b.names]
    assert qc, "the QC selector block was not found"
    assert "errorStrategy" in qc[0].body


def test_with_name_blocks_ignores_a_selector_mentioned_only_in_a_comment(tmp_path):
    """`strip_comments_and_strings` blanks a string literal's delimiting
    quotes along with its contents, not just comments -- so the selector
    regex needs real quotes to match and has to run against the RAW text.
    That means a comment that merely *mentions* `withName: 'FAKE' {` (this
    repo's conf/modules.config has several, e.g. "the `withName: 'SEGMENT'`
    ext.args closure below") must be filtered back out afterwards, or every
    such comment would be counted as a real selector block."""
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "modules.config").write_text(
        "// see the `withName: 'FAKE' {` block below\n"
        "withName: 'REAL' {\n"
        "    cpus = 1\n"
        "}\n"
    )
    blocks = with_name_blocks(root=tmp_path)
    names = {n for b in blocks for n in b.names}
    assert names == {"REAL"}


def test_strip_comments_keeps_string_contents_but_blanks_comments():
    """View B (`strip_comments`) is for guards that must read text living
    INSIDE a quoted string literal -- e.g. `collectFile(name: 'foo.yml', sort:
    true)` or `artifactsOf(ch_artifacts, 'real_kind')`. Unlike
    `strip_comments_and_strings` (view A), it must NOT blank string contents,
    or the exact text those guards search for vanishes along with any comment
    that merely mentions it -- see
    tests/test_resume_determinism.py::test_final_qc_collects_are_sorted, whose
    two assertions need this."""
    src = (
        "// artifactsOf(ch_artifacts, 'fake_kind').collect() -- just a comment\n"
        "artifactsOf(ch_artifacts, 'real_kind').collect(sort: true)\n"
    )
    clean = strip_comments(src)
    assert "fake_kind" not in clean  # comment content is gone
    assert "real_kind" in clean  # string literal content SURVIVES
    assert "artifactsOf(ch_artifacts, 'real_kind').collect(sort: true)" in clean


def test_strip_comments_does_not_open_a_fake_block_comment_on_a_glob():
    """View B must not re-derive a naive comment parse: it has to share the
    exact same `skip_non_code` boundary walk as view A, so a `stageAs:
    'ref/*'` glob's `/*` cannot be misread as an opening block comment here
    either -- the one and only place that decision is allowed to live."""
    src = "path(reference, stageAs: 'ref/*'), path(pre)\nscript:\nMARKER_MUST_SURVIVE\n"
    clean = strip_comments(src)
    assert "stageAs: 'ref/*'" in clean
    assert "script:" in clean
    assert "MARKER_MUST_SURVIVE" in clean


def test_param_refs_excludes_map_method_calls():
    assert param_refs("params.foo + params.bar") == {"foo", "bar"}
    assert param_refs("params.subMap(['a'])") == set()
    assert param_refs("// params.commented") == {"commented"}  # caller strips first


# --------------------------------------------------------------------------
# Anti-vacuity: no test file may privately re-implement a Nextflow/Groovy
# source parse tests.nfmodel already provides.
#
# This USED to be a hardcoded GUARDS list of seven files. That list was
# itself an instance of this phase's own recurring defect shape: correct the
# day it was written, silently stale the moment an eighth guard was added
# elsewhere and nobody remembered to list it. A repo-wide count taken while
# fixing that list found >=24 additional tests/test_*.py files privately
# parsing Nextflow/Groovy source, unlisted and unchecked -- the exact decay
# this file exists to prevent, discovered inside its own capstone.
#
# So the targets are DISCOVERED, not enumerated: any tests/test_*.py that
# reads a file (`.read_text()`/`.glob()`/`.rglob()`) while mentioning `.nf`,
# `.groovy` or `modules.config` anywhere in its own source. A ninth guard
# added tomorrow is covered the moment it exists, not only once someone
# remembers to add it to a list.
#
# As of 2026-08-25: 32 files discovered. 8 already import tests.nfmodel (the
# 7 guards Task 1.6 repointed, plus this file). The other 24 are pre-existing
# private parses not yet repointed -- tracked honestly below as debt, not
# silently passed. 2026-08-30 adds a 33rd, tests/test_no_legacy_frontends.py,
# which is neither: see FLAT_TREE_SCANNERS immediately below. 2026-08-31 adds a
# 34th, tests/test_release_workflow_graph.py, which is a third kind again: see
# WORKFLOW_YAML_GUARDS.
# --------------------------------------------------------------------------


def _discover_nf_source_readers() -> list:
    """Every tests/test_*.py file that reads Nextflow/Groovy source at all."""
    found = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        text = path.read_text()
        mentions_source = ".nf" in text or ".groovy" in text or "modules.config" in text
        reads_files = bool(re.search(r"\.(read_text|glob|rglob)\(", text))
        if mentions_source and reads_files:
            found.append(path)
    return found


# A DIFFERENT kind of exemption, kept in its own list rather than smuggled into
# the debt allowlist below, because it is not debt and will never be discharged.
#
# A guard that scans the WHOLE TRACKED TREE as flat text for a forbidden token is
# discovered by the heuristic above (it names a `.nf` path and calls
# `.read_text()`), but it is not a Nextflow PARSE: it strips no comments, matches
# no blocks, and asserts the token appears NOWHERE -- in code, strings and
# comments alike. Every blind spot the model exists to close makes such a guard
# MISS a hit, never accept one, so repointing it at the model would strictly
# weaken it. There is also nothing in the model for it to call: it needs every
# tracked file, not the structure of one.
#
# Uniform, mechanically-checked reason, in the same spirit as the debt list
# below: an entry must actually INVOKE `git ls-files` -- the argv literal, not a
# mention of it in prose. That distinction is not pedantry: the first draft of
# test_flat_tree_scanners_are_really_flat searched for the bare string, was fed a
# file whose subprocess call had been swapped for `git diff --name-only HEAD~1`,
# and PASSED on the docstring alone. The argv form is the property that makes an
# entry a whole-tree token scan rather than a parse of a named construct.
_FLAT_REASON = "whole-tree flat token scan over `git ls-files`, not a Nextflow parse"

FLAT_TREE_SCANNERS = {
    "tests/test_no_legacy_frontends.py": _FLAT_REASON,
    # Scope-FILTERED rather than whole-tree (CHANGELOG.md is excluded on
    # purpose), but the same kind of thing: since 2026-09-03 it reads
    # workflows/*.nf and subworkflows/**/*.nf as flat text for a forbidden CLI
    # form, comments and log strings included -- stripping comments would make
    # it MISS the `log.warn` it was widened to catch. Enumerates via
    # `git ls-files`, filters in `_in_scope`.
    "tests/test_no_cli_boolean_params_in_docs.py": _FLAT_REASON,
}


def test_flat_tree_scanners_are_really_flat():
    """Each FLAT_TREE_SCANNERS entry must exist, still be discovered (otherwise the
    exemption is dead weight and must be removed), enumerate the tracked tree via
    `git ls-files`, and carry no Nextflow-parsing machinery of its own."""
    discovered = {
        p.relative_to(REPO_ROOT).as_posix() for p in _discover_nf_source_readers()
    }
    problems = []
    for rel in FLAT_TREE_SCANNERS:
        path = REPO_ROOT / rel
        if not path.is_file():
            problems.append(
                f"{rel}: no longer exists -- remove it from FLAT_TREE_SCANNERS"
            )
            continue
        if rel not in discovered:
            problems.append(
                f"{rel}: no longer reads Nextflow/Groovy source -- remove it from "
                "FLAT_TREE_SCANNERS"
            )
        text = path.read_text()
        # The argv literal, NOT the bare phrase: a docstring that merely NAMES git
        # ls-files satisfied an earlier version of this check while the actual call had
        # been swapped for a diff against HEAD~1. Watched failing before being trusted.
        if '["git", "ls-files"]' not in text:
            problems.append(
                f'{rel}: does not invoke `["git", "ls-files"]` to enumerate the tracked '
                "tree, so it is not a whole-tree flat scan -- it does not qualify for this "
                "exemption"
            )
        for marker in ("withName", "strip_comments", "raw_body"):
            if marker in text:
                problems.append(
                    f"{rel}: mentions {marker!r} -- that is Nextflow parsing, so this file "
                    "belongs in the model, not in FLAT_TREE_SCANNERS"
                )
    assert not problems, "\n".join(problems)


# A THIRD kind, and like FLAT_TREE_SCANNERS it is not debt and will never be
# discharged.
#
# A guard over this repository's CI YAML -- `.github/workflows/*.yml` and, since
# the composite actions landed, `.github/actions/*/action.yml` -- parses YAML, not
# Nextflow. It is discovered by the heuristic above only because a CI step's
# COMMAND STRING or a cache key mentions a `.nf` path -- `nextflow run
# tests/lib_probe.nf -lib lib`, `tests/modules/segment_deepcell_token.nf.test`,
# `hashFiles('**/*.nf')` -- which is the needle the guard looks FOR inside the
# YAML, never a file it opens. There is nothing in the model for it to call: the
# model parses Nextflow sources, and this reads none.
#
# The mechanically-checked property that separates "names a .nf path in a string"
# from "reads Nextflow source" is whether the file ever CONSTRUCTS a path to one.
# A real reader has to: `REPO_ROOT / "modules" / ...`, `glob("*.nf")`. A workflow
# guard does not, and the check below fails the moment one appears -- at which
# point the file is a real reader and belongs in the model.
_WORKFLOW_YAML_REASON = (
    "parses this repo's CI YAML (.github/workflows/*.yml and "
    ".github/actions/*/action.yml); its `.nf` mentions are needles it looks for "
    "inside that YAML, never source it opens"
)

WORKFLOW_YAML_GUARDS = {
    "tests/test_release_workflow_graph.py": _WORKFLOW_YAML_REASON,
    # Added 2026-09-01 with the CI redesign's Phase 3. It trips the heuristic on
    # `hashFiles('**/*.nf')` -- the cache key it exists to FORBID, quoted in its
    # docstring and in its failure message.
    "tests/test_ci_cache_keys.py": _WORKFLOW_YAML_REASON,
}

# A path CONSTRUCTED to a Nextflow/Groovy source file -- `ROOT / "modules/local/x.nf"`,
# `REPO / f"{name}.groovy"` -- as opposed to a `.nf` path merely quoted in a string.
_CONSTRUCTS_NF_PATH_RE = re.compile(
    r"(?:ROOT|REPO|REPO_ROOT)\s*/[^\n]*\.(?:nf|groovy)\b"
)
_GLOBS_NF_RE = re.compile(r"r?glob\(\s*[\"'][^\"']*\.(?:nf|groovy)")


def test_workflow_yaml_guards_really_only_parse_workflow_yaml():
    """Each WORKFLOW_YAML_GUARDS entry must exist, still be discovered (otherwise
    the exemption is dead weight and must be removed), actually parse YAML, and
    construct no path to a Nextflow or Groovy source file."""
    discovered = {
        p.relative_to(REPO_ROOT).as_posix() for p in _discover_nf_source_readers()
    }
    problems = []
    for rel in WORKFLOW_YAML_GUARDS:
        path = REPO_ROOT / rel
        if not path.is_file():
            problems.append(
                f"{rel}: no longer exists -- remove it from WORKFLOW_YAML_GUARDS"
            )
            continue
        if rel not in discovered:
            problems.append(
                f"{rel}: no longer trips the nf-source-reader heuristic -- remove it from "
                "WORKFLOW_YAML_GUARDS"
            )
        text = path.read_text()
        if "yaml.safe_load" not in text:
            problems.append(
                f"{rel}: never calls `yaml.safe_load`, so it is not a workflow-YAML guard "
                "and does not qualify for this exemption"
            )
        if ".github" not in text:
            problems.append(f"{rel}: does not read anything under .github/")
        for regex, what in (
            (_CONSTRUCTS_NF_PATH_RE, "constructs a path to a .nf/.groovy source file"),
            (_GLOBS_NF_RE, "globs .nf/.groovy source files"),
        ):
            m = regex.search(text)
            if m:
                problems.append(
                    f"{rel}: {what} ({m.group(0).strip()!r}) -- it reads Nextflow source "
                    "after all, so it belongs in the model, not in WORKFLOW_YAML_GUARDS"
                )
        for marker in ("withName", "strip_comments", "raw_body"):
            if marker in text:
                problems.append(
                    f"{rel}: mentions {marker!r} -- that is Nextflow parsing, so this file "
                    "belongs in the model, not in WORKFLOW_YAML_GUARDS"
                )
    assert not problems, "\n".join(problems)


# Every entry here carries the SAME reason, deliberately -- this repo has
# been bitten before by an allowlist entry with a bespoke, invented
# justification that turned out to have counterexamples elsewhere (see
# lib/ParamUtils.groovy's `process_high_memory` prose-only label, or the
# per-file excuses `ENTRY_COLUMN_EXEMPT` in test_step_vocabulary_consistency.py
# explicitly warns against reproducing here). A uniform, literally-true
# reason -- "this file has not been repointed yet" -- cannot drift out of
# sync with itself the way a bespoke one can.
_UNREPOINTED_REASON = "pre-existing private parse, not yet repointed (2026-08-25)"

UNREPOINTED_NF_SOURCE_READERS = {
    "tests/test_apply_basic_profiles.py": _UNREPOINTED_REASON,
    "tests/test_basic_illumination_lazy_read.py": _UNREPOINTED_REASON,
    "tests/test_basicpy_defaults_are_deliberate.py": _UNREPOINTED_REASON,
    "tests/test_basicpy_module_is_vendored_unmodified.py": _UNREPOINTED_REASON,
    "tests/test_compartment_mode_routing.py": _UNREPOINTED_REASON,
    "tests/test_concurrency_params.py": _UNREPOINTED_REASON,
    "tests/test_group_key_unwrapped.py": _UNREPOINTED_REASON,
    "tests/test_merge_pyramid_streaming.py": _UNREPOINTED_REASON,
    "tests/test_module_conventions.py": _UNREPOINTED_REASON,
    "tests/test_nextflow_strict_syntax.py": _UNREPOINTED_REASON,
    "tests/test_no_secret_interpolation.py": _UNREPOINTED_REASON,
    "tests/test_no_shadowing_empty_guards.py": _UNREPOINTED_REASON,
    "tests/test_nuclear_marker_routing.py": _UNREPOINTED_REASON,
    "tests/test_nullable_numeric_params_no_elvis.py": _UNREPOINTED_REASON,
    "tests/test_pixel_size_is_passed.py": _UNREPOINTED_REASON,
    "tests/test_process_alias_config.py": _UNREPOINTED_REASON,
    "tests/test_reg_presets_inlined_in_config.py": _UNREPOINTED_REASON,
    "tests/test_seg_backends.py": _UNREPOINTED_REASON,
    "tests/test_seg_backends_ctx_params.py": _UNREPOINTED_REASON,
    "tests/test_seg_qc_geojson_equivalence.py": _UNREPOINTED_REASON,
    "tests/test_stub_control_json_contract.py": _UNREPOINTED_REASON,
    "tests/test_warp_seg_qc.py": _UNREPOINTED_REASON,
}


def test_allowlist_only_shrinks():
    """Every UNREPOINTED_NF_SOURCE_READERS entry must still be real, undischarged
    debt: the file must still exist, must still be discovered by
    _discover_nf_source_readers(), and must still lack the tests.nfmodel
    import. An entry that no longer applies -- the file was deleted, or was
    repointed by a later task without being removed here -- is a hard
    failure. The list may only ever be edited to shrink it; it cannot be left
    to rot in place the way GUARDS did."""
    discovered = {
        p.relative_to(REPO_ROOT).as_posix() for p in _discover_nf_source_readers()
    }
    stale = []
    for rel in UNREPOINTED_NF_SOURCE_READERS:
        path = REPO_ROOT / rel
        if not path.is_file():
            stale.append(f"{rel}: no longer exists -- remove it from the allowlist")
            continue
        if "from tests.nfmodel import" in path.read_text():
            stale.append(
                f"{rel}: already imports tests.nfmodel -- remove it from the allowlist"
            )
            continue
        if rel not in discovered:
            stale.append(
                f"{rel}: no longer reads Nextflow/Groovy source -- remove it from the allowlist"
            )
    assert not stale, "\n".join(stale)


def test_no_guard_parses_nextflow_source_privately():
    """Every tests/test_*.py file that reads Nextflow/Groovy source either
    queries tests.nfmodel, or is honestly tracked as not-yet-repointed debt in
    UNREPOINTED_NF_SOURCE_READERS. A private regex parse is how four guards
    came to pass while checking nothing -- each had a different blind spot,
    and none was visible from the interface -- and a hardcoded list of which
    files to check is the same failure shape one level up (see the discovery
    rationale above). This scans every DISCOVERED file, not a named list, so
    a new private parse is caught the moment it is written, not only once
    someone remembers to name it here.

    If you genuinely need a parse the model does not expose, ADD IT TO THE
    MODEL (with a case in this file), do not re-implement it here."""
    private = re.compile(r"re\.(sub|search|finditer|findall)\([^)]*/\\\*")
    offenders = []
    for path in _discover_nf_source_readers():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text()
        has_import = "from tests.nfmodel import" in text
        if (
            not has_import
            and rel not in UNREPOINTED_NF_SOURCE_READERS
            and rel not in FLAT_TREE_SCANNERS
            and rel not in WORKFLOW_YAML_GUARDS
        ):
            offenders.append(
                f"{rel}: parses Nextflow source without the model, and is on none of the "
                "UNREPOINTED_NF_SOURCE_READERS debt allowlist, FLAT_TREE_SCANNERS or "
                "WORKFLOW_YAML_GUARDS"
            )
        if private.search(text):
            offenders.append(f"{rel}: hand-rolled block-comment regex")
    assert not offenders, "\n".join(offenders)


def _write_nf_test(tmp_path, name, text):
    d = tmp_path / "tests" / "modules"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def test_nf_test_cases_preamble_tag_applies_to_every_case(tmp_path):
    _write_nf_test(tmp_path, "a.nf.test", '''
nextflow_process {
    process "FOO"
    tag "stub"
    test("one") { options "-stub" }
    test("two") { }
}
''')
    cases = nf_test_cases(root=tmp_path)
    assert [c.name for c in cases] == ["one", "two"]
    assert all("stub" in c.tags for c in cases)
    assert [c.stub_option for c in cases] == [True, False]
    assert {c.process for c in cases} == {"FOO"}


def test_nf_test_cases_ignores_a_tag_that_lives_only_in_a_comment(tmp_path):
    _write_nf_test(tmp_path, "b.nf.test", '''
nextflow_process {
    process "FOO"
    // tag "stub"
    test("one") { /* tag "stub" */ }
}
''')
    (case,) = nf_test_cases(root=tmp_path)
    assert "stub" not in case.tags


def test_nf_test_cases_sees_stub_among_other_options(tmp_path):
    _write_nf_test(tmp_path, "c.nf.test", '''
nextflow_process {
    process "FOO"
    test("one") { tag "stub"
        options "-stub -resume" }
    test("two") { tag "stub"
        options "-resume" }
}
''')
    one, two = nf_test_cases(root=tmp_path)
    assert one.stub_option is True
    assert two.stub_option is False


def test_nf_test_cases_sees_the_stub_run_spelling_and_single_quotes(tmp_path):
    """`-stub-run` is Nextflow's documented primary spelling (`-stub` is the
    alias), and nf-test accepts single-quoted option strings. A case written
    either way used to parse as `stub_option=False`, i.e. as a RENDERED case --
    which is how a stub-only case could satisfy the rendered-coverage rule in
    test_process_nf_test_coverage.py while rendering nothing."""
    _write_nf_test(tmp_path, "e.nf.test", '''
nextflow_process {
    process "FOO"
    test("stub-run") { tag "stub"
        options "-stub-run" }
    test("single-quoted") { tag "stub"
        options '-stub' }
    test("stub-run with another flag") { tag "stub"
        options "-stub-run -resume" }
    test("really rendered") { tag "stub"
        options "-resume" }
}
''')
    cases = {c.name: c for c in nf_test_cases(root=tmp_path)}
    assert cases["stub-run"].stub_option is True
    assert cases["single-quoted"].stub_option is True
    assert cases["stub-run with another flag"].stub_option is True
    assert cases["really rendered"].stub_option is False


def test_nf_test_cases_sees_a_single_quoted_case_name(tmp_path):
    """`test('...')` is legal nf-test; a parser that only matched `test("...")`
    dropped the case silently, so its tags and options were never modelled."""
    _write_nf_test(tmp_path, "f.nf.test", """
nextflow_process {
    process 'FOO'
    test('single-quoted name') { tag 'stub' }
}
""")
    (case,) = nf_test_cases(root=tmp_path)
    assert case.name == "single-quoted name"
    assert case.process == "FOO"
    assert case.tags == frozenset({"stub"})


def test_nf_test_cases_records_whether_the_body_reads_command_sh(tmp_path):
    """A case with no `options "-stub"` is not automatically a rendered-command
    case: a pure failure case asserts `process.failed` and never looks at the
    command. `reads_command_sh` is what separates the two."""
    _write_nf_test(tmp_path, "g.nf.test", '''
nextflow_process {
    process "FOO"
    test("renders") { tag "stub"
        then { def cmd = new File("${workDir}/x/.command.sh").text
               assert cmd.contains("--flag") } }
    test("just fails") { tag "stub"
        then { assert process.failed } }
}
''')
    renders, fails = nf_test_cases(root=tmp_path)
    assert renders.reads_command_sh is True
    assert fails.reads_command_sh is False


def test_nf_test_cases_workflow_file_has_no_process(tmp_path):
    _write_nf_test(tmp_path, "d.nf.test", '''
nextflow_workflow {
    workflow "BAR"
    test("one") { tag "stub" }
}
''')
    (case,) = nf_test_cases(root=tmp_path)
    assert case.process is None
    assert case.tags == frozenset({"stub"})


def test_nf_test_cases_covers_the_real_tree():
    cases = nf_test_cases()
    assert len(cases) >= 261, "the audit counted 261 cases on 2026-09-10; a drop means the parser lost files"
    assert any(c.process == "SEG_QC_GEOJSON" and not c.stub_option and "stub" in c.tags for c in cases), (
        "seg_qc_geojson.nf.test's rendered-command case is the precedent; the parser must see it"
    )
