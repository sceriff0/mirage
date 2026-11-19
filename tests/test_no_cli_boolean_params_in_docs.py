"""No document may show a boolean parameter passed on the command line.

Nextflow 26 hands every CLI `--param` to nf-schema as a String, so validation fails with
`* --dry_run (true): Value is [string] but should be [boolean]`. Both the valued form and
a bare valueless flag fail; only `-params-file` with a JSON boolean, or a profile, works.
The boolean parameter list is derived from the schema.

A document showing the broken form is a copy-paste hazard that no other test catches.
"""

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Enumerated from the tracked tree (`git ls-files`), not a filesystem glob, so
# untracked scratch files are ignored; tests/test_nfmodel.py's FLAT_TREE_SCANNERS
# exemption depends on that enumeration. Each clause below is part of the scope.
_TRACKED = [
    REPO / rel
    for rel in subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    if rel
]


def _in_scope(path):
    rel = path.relative_to(REPO)
    parts = rel.parts
    if rel.as_posix() == "README.md":
        return True
    # docs/*.md (the published pages; docs/_archive is history, excluded) and
    # the supplementary figures a reader copies commands from.
    if parts[0] == "docs":
        return (len(parts) == 2 and rel.suffix == ".md") or (
            len(parts) == 3 and parts[1] == "figures" and rel.suffix == ".html"
        )
    # Recursive on purpose: tests/cluster/README.md is the file an operator
    # copy-pastes from.
    if parts[0] == "tests" and rel.suffix in {".md", ".sh"}:
        return True
    # Config comments are documents too: nextflow.config, conf/*.config and
    # conf/site.config.template (which docs/installation.md says to copy).
    if rel.as_posix() == "nextflow.config":
        return True
    if (
        parts[0] == "conf"
        and len(parts) == 2
        and rel.name.endswith((".config", ".config.template"))
    ):
        return True
    # Workflow/subworkflow log messages are operator-facing prose. Read as flat
    # text, comments included, not as a Nextflow parse (see FLAT_TREE_SCANNERS).
    if parts[0] in {"workflows", "subworkflows"} and rel.suffix == ".nf":
        return True
    # CHANGELOG.md is not scanned: it quotes past commands, broken ones
    # included, and rewriting them would falsify the record.
    return False


SCAN = sorted(p for p in _TRACKED if _in_scope(p))

# Files that quote the broken form ON PURPOSE, with the reason. Each is verified below to still
# contain it -- a stale exemption would silently cover a real regression.
ALLOWED = {
    "tests/run_validation_tests.sh": (
        "the header comment explains why the script uses -params-file, quoting the form it "
        "replaced; the executed command uses -params-file"
    ),
}


def _boolean_params():
    schema = json.loads((REPO / "nextflow_schema.json").read_text())
    names = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            for name, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                types = spec.get("type")
                types = types if isinstance(types, list) else [types]
                if "boolean" in types:
                    names.add(name)
        for v in node.values():
            walk(v)

    walk(schema)
    assert names, (
        "schema walk found no boolean params -- the guard would pass vacuously"
    )
    return names


# A line may show the broken form as an explicit counter-example (docs/usage.md's
# "Boolean parameters" section) if the marker is on the same line, so it cannot drift
# away and start excusing a working example.
_COUNTEREXAMPLE = re.compile(r"#\s*x\s+fails\b", re.I)

# The bare form fails too (NF26 delivers a valueless `--flag` as the String
# "true"). It is detected only inside a command, not in prose, where boolean
# parameters are named constantly ("`--expanded_quantification` requires ...").
_FENCE = re.compile(r"^\s*(```|~~~)")
_TRAILING_COMMENT = re.compile(r"\s+#.*$")


def _command_lines(text, always_in_command):
    """Yield (line_no, raw_line, code) for lines that are part of a `nextflow run`.

    In a Markdown file, only lines inside a fenced block are CANDIDATES; in a
    .sh, .config or .nf file, every line is a candidate. A candidate is a
    command line only if it contains `nextflow run` or continues one (a
    backslash continuation keeps the command open). So the BARE form is
    caught only on an actual `nextflow run` line; a bare `--dry_run` in a
    comment or log message is prose. The `--flag true|false` form is caught on
    every scanned line by `_offending_lines`.
    """
    in_fence = always_in_command
    continuing = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not always_in_command and _FENCE.match(line):
            in_fence = not in_fence
            continuing = False
            continue
        if not in_fence:
            continue
        # Stripped before the bare-flag scan: a shell line ending
        # `--dry_run  # explained below` must still read as bare (no value),
        # not as if the trailing comment were the flag's argument.
        code = _TRAILING_COMMENT.sub("", line)
        is_command = "nextflow run" in code or continuing
        continuing = is_command and code.rstrip().endswith("\\")
        if is_command:
            yield line_no, line, code


def _offending_lines(text, params, always_in_command=False):
    alts = "|".join(sorted(params))
    # [=\s]: Nextflow accepts both `--flag value` and `--flag=value` on the CLI.
    with_value = re.compile(rf"--({alts})[=\s]+(?:true|false)\b")
    # Bare: the flag is the last token, is followed by a `\` continuation, or is
    # followed by another option.
    bare = re.compile(rf"--({alts})(?=\s*(?:\\\s*)?$|\s+-)")

    for i, line in enumerate(text.splitlines(), start=1):
        if _COUNTEREXAMPLE.search(line):
            continue
        if with_value.search(line):
            yield i, line.strip()

    for line_no, raw, code in _command_lines(text, always_in_command):
        if _COUNTEREXAMPLE.search(raw):
            continue
        if with_value.search(code):
            continue  # already reported above
        if bare.search(code):
            yield line_no, raw.strip()


def test_no_document_shows_a_boolean_param_on_the_command_line():
    params = _boolean_params()
    offenders = []

    for f in SCAN:
        rel = f.relative_to(REPO).as_posix()
        if rel in ALLOWED:
            continue
        # .config.template (site.config.template) isn't a fenced Markdown doc
        # either -- Path.suffix only sees the LAST extension (".template"), so
        # check the full name, not just .suffix.
        always = f.suffix in (".sh", ".config") or f.name.endswith(".config.template")
        for line_no, line in _offending_lines(f.read_text(), params, always):
            offenders.append(f"{rel}:{line_no}: {line}")

    assert not offenders, (
        "Nextflow 26 delivers every --param as a String, so these examples fail validation on "
        "the engine CI already runs. Use -params-file with a real JSON boolean, or a profile:\n  "
        + "\n  ".join(offenders)
    )


def test_every_allowlisted_file_still_quotes_the_form_it_is_exempt_for():
    """A stale exemption is worse than no exemption -- it covers the next real regression."""
    params = _boolean_params()

    for rel in ALLOWED:
        text = (REPO / rel).read_text()
        assert list(_offending_lines(text, params)), (
            f"{rel} no longer shows a CLI boolean, so its ALLOWED entry is stale. "
            "Remove the entry rather than leaving it to cover a future regression."
        )


def test_the_scan_covers_the_documents_this_guard_exists_for():
    scanned = {f.relative_to(REPO).as_posix() for f in SCAN}

    for expected in ("README.md", "docs/usage.md", "tests/test_validation.md"):
        assert expected in scanned


def test_the_scan_covers_the_configuration_files_too():
    scanned = {f.relative_to(REPO).as_posix() for f in SCAN}
    assert "nextflow.config" in scanned
    assert "conf/test.config" in scanned
    assert "conf/site.config.template" in scanned


def test_the_detector_recognises_the_bare_form_inside_a_command():
    params = _boolean_params()
    fenced = "```bash\nnextflow run . --input s.csv --dry_run\n```\n"
    assert list(_offending_lines(fenced, params))

    continued = "```bash\nnextflow run . \\\n  --cleanup_work \\\n  --outdir r\n```\n"
    assert list(_offending_lines(continued, params))


def test_the_detector_leaves_prose_naming_a_boolean_param_alone():
    """These documents name boolean parameters in prose constantly; a detector
    that fired on those would be noise nobody reads."""
    params = _boolean_params()
    prose = (
        "Either add `--quantify_compartments`, or drop `--expanded_quantification`.\n"
    )
    assert not list(_offending_lines(prose, params))


def test_the_detector_recognises_the_form_that_actually_fails():
    params = _boolean_params()

    assert list(_offending_lines("  --start preprocessing --dry_run true\n", params))
    assert list(_offending_lines("nextflow run . --skip_preprocessing false\n", params))


def test_the_detector_recognises_the_equals_form_too():
    """Nextflow accepts `--flag=value` as well as `--flag value` on the CLI --
    the value form is broken identically either way."""
    params = _boolean_params()

    assert list(_offending_lines("nextflow run . --dry_run=true\n", params))
    assert list(_offending_lines("nextflow run . --skip_preprocessing=false\n", params))


def test_the_detector_leaves_the_working_forms_alone():
    params = _boolean_params()

    assert not list(
        _offending_lines("nextflow run . -params-file params/dry_run.json\n", params)
    )
    assert not list(_offending_lines("nextflow run . -profile test,docker\n", params))
    # a non-boolean param legitimately takes a CLI value
    assert not list(_offending_lines("nextflow run . --start preprocessing\n", params))


def test_the_counterexample_marker_must_be_on_the_same_line():
    """A marker that could sit on a heading would drift and start excusing working examples."""
    params = _boolean_params()

    marked = "nextflow run . --dry_run true   # x fails on Nextflow 26\n"
    assert not list(_offending_lines(marked, params))

    # the same command one line below a marker is NOT excused
    detached = "# x fails on Nextflow 26\nnextflow run . --dry_run true\n"
    assert list(_offending_lines(detached, params))


def test_no_shell_script_passes_a_boolean_param_on_the_command_line():
    """The docs check above is advisory; a script is EXECUTED, so the same mistake
    there is live rather than misleading.

    On Nextflow 26 `--skip_seg_quality_eval false` sets the param to the string
    "false", which is truthy. Use a profile or -params-file with a JSON boolean.

    Scope is every tracked .sh (`git ls-files`), so each branch checks the
    scripts it actually ships.
    """
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bools = _boolean_params()  # noqa: F821 - defined above in this module

    listed = subprocess.run(
        ["git", "ls-files", "*.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    scripts = sorted(root / rel for rel in listed)
    assert scripts, (
        "found no tracked .sh files -- the enumerator is wrong, not the repo"
    )

    offenders = []
    for sh in scripts:
        for i, line in enumerate(sh.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for m in re.finditer(r"--(\w+)\s+(true|false)\b", line):
                if m.group(1) in bools:
                    offenders.append(f"{sh.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "shell scripts pass a boolean param on the command line:\n  "
        + "\n  ".join(offenders)
    )
