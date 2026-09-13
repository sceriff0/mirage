r"""Every `nextflow run` command in the docs must survive launch validation.

Three parameters have no default and are required by `nextflow_schema.json`:
`outdir`, `max_cpus` and `max_memory`. nextflow.config declares the last two
`null` deliberately -- see the comment above `max_cpus = null`. A command that
supplies none of them is refused before any process is submitted:

    ERROR ~ Validation of pipeline parameters failed!
    * Missing required parameter(s): max_cpus, max_memory

25 of the 32 `nextflow run` commands in README.md and docs/*.md were in exactly
that state on 2026-09-02, including the one the Usage page calls "the minimal
command". Nothing executes a fenced code block, so no test caught it and the
reader's first run failed on a perfectly good pipeline.

Ruling R4 for 1.0.0: a documented command satisfies the sizing pair through
`-c site.config` (copied from conf/site.config.template, which carries them),
through a profile that pins them, or through a `-params-file` preset that
carries them. This guard reads the commands the way a reader does -- out of the
fenced code blocks, following backslash continuations -- so it fails on the
copy-paste the reader would actually make.

WHAT THIS FILE ACTUALLY CHECKS, AND WHAT IT DOES NOT. `_verdict()` below is a
STATIC TOKEN CHECK: it looks for `--outdir`/`-c`/`max_cpus`/`max_memory`
substrings and sizing-pinning profile names in the command text. It never
invokes Nextflow, so it is blind to anything that only surfaces at real launch
validation -- an unrecognised flag, a samplesheet that does not exist, a column
the schema rejects. `_verdict("nextflow run . --nonexistent_param 1 --outdir
results -c site.config")` reports no offense, even though nf-schema refuses
that exact command with "Missing required parameter(s): input" /
"* --nonexistent_param: 1". That gap is real and is not closed here -- closing
it needs an actual `nextflow` invocation per command, which this file's fast
in-process assertions deliberately are not.

The execution counterpart is `tests/documented_commands_launch.sh`. It calls
this file's `_commands()` (via `python3 -m tests.test_documented_commands_are_runnable
--emit`, the ONE extractor -- no second regex) to get the same command list,
substitutes each command's `--input`/`--outdir`/`--prior_outdir`/`-profile
<placeholder>`/`-params-file` tokens for real repo fixtures, and actually runs
`nextflow -stub -params-file params/dry_run.json -c <site.config>` for every
one, asserting real launch validation passes. `--dry_run` is what makes that
affordable: CLAUDE.md's own words are "everything runs before any process is
instantiated, and --dry_run exercises exactly that surface", so each of the
~30 commands validates in a few seconds with no container ever pulled and no
process ever submitted -- confirmed by inspecting the resulting work directory
(0 files) after a passing dry run.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# rglob, not glob: docs/validation/format_validation.md is a navigated page
# (mkdocs.yml) that a top-level glob silently left outside this guard. The
# excluded author-facing trees (docs/superpowers, docs/_archive, docs/perf --
# see mkdocs.yml's exclude_docs) are not published, so they are skipped here
# for the same reason CHANGELOG.md is: they quote history, not instructions.
_UNPUBLISHED = {"superpowers", "_archive", "perf"}
SCAN = [
    REPO / "README.md",
    *sorted(
        p
        for p in (REPO / "docs").rglob("*.md")
        if not (set(p.relative_to(REPO / "docs").parts[:-1]) & _UNPUBLISHED)
    ),
]

# Profiles that pin BOTH params.max_cpus and params.max_memory, and those that
# additionally pin params.outdir. Both sets are verified against the config below,
# so a profile that stops pinning cannot stay on the list.
SIZING_PROFILES = {"test", "test_full", "local", "ieo", "shipped_defaults_test"}
OUTDIR_PROFILES = {"test", "test_full", "shipped_defaults_test"}

_FENCE = re.compile(r"^\s*(```|~~~)")
# A line may show a deliberately broken command if it says so ON THE SAME LINE --
# docs/usage.md's "Boolean parameters" section does exactly that.
_COUNTEREXAMPLE = re.compile(r"#\s*x\s+fails\b", re.I)
_TRAILING_COMMENT = re.compile(r"\s+#.*$")


def _commands(text):
    """Yield (line_no, command) for every `nextflow run` inside a fenced block.

    Backslash continuations are joined, so a command spread over six lines is one
    command -- which is how the reader pastes it.
    """
    in_fence = False
    buf = None
    start = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            if not in_fence and buf is not None:
                yield start, buf
                buf = None
            continue
        if not in_fence:
            continue
        stripped = line.strip().removeprefix("$ ")
        if buf is None:
            if not stripped.startswith("nextflow run"):
                continue
            if _COUNTEREXAMPLE.search(line):
                continue
            buf, start = stripped, line_no
        else:
            buf = buf[:-1] + " " + stripped
        if buf.rstrip().endswith("\\"):
            continue
        yield start, buf
        buf = None
    if buf is not None:
        yield start, buf


# The ONE not-runnable convention this repo defines (there was none before Fix
# Round 1): a `# not-runnable: <reason>` comment on the line directly above a
# command's first line. Only docs/parameters.md's CSE example needs it -- its
# trailing `...` elides more flags on purpose (it is prose showing where a
# reader's own flags go, not a paste-ready command) -- so `_verdict()` above
# still scores it (harmlessly clean: it carries --outdir and -c already) and
# only the execution leg below skips it.
_NOT_RUNNABLE = re.compile(r"^\s*#\s*not-runnable:\s*(.+?)\s*$", re.I)


def _skip_reason(text, line_no):
    """The reason tests/documented_commands_launch.sh should SKIP this command
    rather than run it, or None. Checked only by the execution leg; see
    _NOT_RUNNABLE above."""
    lines = text.splitlines()
    if line_no < 2:
        return None
    match = _NOT_RUNNABLE.match(lines[line_no - 2])
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Execution-leg support. tests/documented_commands_launch.sh gets its command
# list from `_emit()` below (via `python3 -m
# tests.test_documented_commands_are_runnable --emit`) rather than re-parsing
# the docs itself -- `_commands()` above stays the ONE extractor. What follows
# is a second, DIFFERENT concern: turning an extracted command into one that
# can actually launch against this repo's own fixtures.
# ---------------------------------------------------------------------------

# Which fixture samplesheet satisfies the entry step's requiredColumns (the
# STEPS table in lib/ParamUtils), keyed by the step name found after --start,
# or 'preprocessing' (the schema default) when --start is absent.
# mode=add_cycle reads its --input through the same preprocessing-shaped
# columns (path_to_file, not a checkpoint column), so it is keyed separately
# and checked first, ahead of any (disallowed, but harmless to check) --start.
_INPUT_FIXTURES = {
    "add_cycle": "tests/testdata/new_cycle.csv",
    "preprocessing": "tests/testdata/valid_preprocessing.csv",
    "registration": "tests/testdata/valid_checkpoint_registration.csv",
    # Named for what it was produced for, not what it feeds here: this
    # fixture's columns (patient_id,registered_image,is_reference,channels)
    # are exactly `segmentation`'s requiredColumns -- confirmed against the
    # checkpoint-manifest test suite's own --start segmentation case.
    "segmentation": "tests/testdata/valid_checkpoint_postprocessing.csv",
    # Likewise: this one carries cell_mask/nuclei_mask/id, which is
    # `postprocessing`'s requiredColumns -- confirmed against the main test
    # suite's "Should run postprocessing from checkpoint" case.
    "postprocessing": "tests/testdata/valid_checkpoint_segmented.csv",
}

# A completed prior run's checkpoint pair (csv/registered.csv +
# csv/postprocessed.csv) -- the shape --prior_outdir must point at
# (Layout.ADD_CYCLE_CHECKPOINTS). Read-only, so every add_cycle-shaped command
# can safely point at the same fixture directory.
_PRIOR_OUTDIR_FIXTURE = "tests/testdata/prior_run"

_INPUT_FLAG = re.compile(r"--input\s+\S+")
_OUTDIR_FLAG = re.compile(r"--outdir\s+\S+")
_PRIOR_OUTDIR_FLAG = re.compile(r"--prior_outdir\s+\S+")
_PARAMS_FILE_FLAG = re.compile(r"-params-file\s+\S+")
_PLACEHOLDER_PROFILE = re.compile(r"-profile\s+<[^>]+>")
_C_FLAG = re.compile(r"(^|\s)-c\s+\S+")

# Filled in by the shell script per invocation with tmp paths IT controls --
# this module only rewrites text, it never creates a file or a directory.
OUTDIR_SENTINEL = "__OUTDIR__"
SITE_CONFIG_SENTINEL = "__SITE_CONFIG__"


def _executable(command):
    """Rewrite an extracted command so it can actually launch against this
    repo's fixtures: the same flags the doc chose, with real (or sentinel)
    values substituted in.

    A substitution only ever REPLACES A VALUE, never removes or adds a flag
    the doc itself did not choose -- `-profile docker`/`slurm,singularity`/etc.
    are left exactly as written. `--dry_run` exits before any process (and
    therefore any container) is ever touched -- see the module docstring for
    the measured 0-files-in-work-dir proof -- so the reader's own profile
    choice is exactly what gets exercised, at no extra cost.
    """
    command = _TRAILING_COMMENT.sub("", command)

    if "--mode add_cycle" in command:
        step = "add_cycle"
    else:
        match = re.search(r"--start\s+(\S+)", command)
        step = match.group(1) if match else "preprocessing"
    fixture = _INPUT_FIXTURES.get(step, _INPUT_FIXTURES["preprocessing"])
    input_path = (REPO / fixture).resolve()
    if _INPUT_FLAG.search(command):
        command = _INPUT_FLAG.sub(f"--input {input_path}", command)

    if _PRIOR_OUTDIR_FLAG.search(command):
        prior = (REPO / _PRIOR_OUTDIR_FIXTURE).resolve()
        command = _PRIOR_OUTDIR_FLAG.sub(f"--prior_outdir {prior}", command)

    if _OUTDIR_FLAG.search(command):
        command = _OUTDIR_FLAG.sub(f"--outdir {OUTDIR_SENTINEL}", command)
    else:
        command += f" --outdir {OUTDIR_SENTINEL}"

    command = _PLACEHOLDER_PROFILE.sub("-profile test", command)

    if _PARAMS_FILE_FLAG.search(command):
        command = _PARAMS_FILE_FLAG.sub("-params-file params/dry_run.json", command)
    else:
        command += " -params-file params/dry_run.json"

    if _C_FLAG.search(command):
        command = _C_FLAG.sub(rf"\1-c {SITE_CONFIG_SENTINEL}", command)
    else:
        command += f" -c {SITE_CONFIG_SENTINEL}"

    if not re.search(r"(^|\s)-stub(\s|$)", command):
        command += " -stub"

    return command


# The "no reason" sentinel, printed literally rather than left as an empty
# field. Bash's `read` treats tab as IFS WHITESPACE regardless of what IFS is
# set to, which means it COLLAPSES an empty field between two tabs instead of
# preserving it -- measured: `IFS=$'\t' read -r a b c d <<<"x\t1\t\ty"` gives
# c=y, d=(empty), silently shifting every field after the gap. A sentinel that
# is never empty sidesteps the collapse entirely and stays human-readable in
# `--emit`'s own output, unlike switching to a control-character separator.
_NO_REASON = "-"


def _emit():
    """Print one TSV row per extracted command: file, line, skip reason (the
    literal `-` sentinel above if runnable), executable command (empty if
    skipped). Consumed only by tests/documented_commands_launch.sh."""
    for path in SCAN:
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text()
        for line_no, command in _commands(text):
            reason = _skip_reason(text, line_no) or _NO_REASON
            executable = "" if reason != _NO_REASON else _executable(command)
            print(f"{rel}\t{line_no}\t{reason}\t{executable}")


def _profile_body(name):
    """The text a `-profile <name>` actually contributes, includes resolved."""
    text = (REPO / "nextflow.config").read_text()
    match = re.search(rf"^\s+{re.escape(name)}\s*\{{", text, re.M)
    if not match:
        return ""
    depth = 0
    start = text.index("{", match.start())
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = text[start : end + 1] if end else text[start:]
    for included in re.findall(r"includeConfig\s+'([^']+)'", body):
        path = REPO / included
        if path.is_file():
            body += "\n" + path.read_text()
    return body


def _profile_pins(name, param):
    return bool(re.search(rf"^\s*(params\.)?{param}\s*=", _profile_body(name), re.M))


def _preset(path_str):
    path = REPO / path_str
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _verdict(command):
    """The reasons this command would be refused at launch, or []."""
    command = _TRAILING_COMMENT.sub("", command)

    profiles = set()
    for match in re.finditer(r"-profile\s+(\S+)", command):
        profiles.update(match.group(1).split(","))

    preset = {}
    for name in re.findall(r"-params-file\s+(\S+)", command):
        preset.update(_preset(name))

    has_c = re.search(r"(^|\s)-c\s+\S+", command) is not None

    reasons = []
    if not (
        "--outdir" in command
        or (profiles & OUTDIR_PROFILES)
        or "outdir" in preset
        or has_c
    ):
        reasons.append("no --outdir (required, no default)")
    if not (
        (profiles & SIZING_PROFILES)
        or has_c
        or ("max_cpus" in preset and "max_memory" in preset)
        or ("--max_cpus" in command and "--max_memory" in command)
    ):
        reasons.append("no max_cpus/max_memory (required, no default)")
    return reasons


def test_every_documented_command_would_launch():
    offenders = []
    total = 0
    for path in SCAN:
        rel = path.relative_to(REPO).as_posix()
        for line_no, command in _commands(path.read_text()):
            total += 1
            reasons = _verdict(command)
            if reasons:
                offenders.append(
                    f"{rel}:{line_no}: {'; '.join(reasons)}\n      {command}"
                )
    assert total >= 25, (
        f"only {total} `nextflow run` commands were extracted from {len(SCAN)} files -- "
        "the extractor has stopped matching and this scan read almost nothing"
    )
    assert not offenders, (
        "documented commands that nf-schema refuses at launch. Add `-c site.config` "
        "(conf/site.config.template) and/or `--outdir`:\n  " + "\n  ".join(offenders)
    )


def test_the_profiles_this_guard_trusts_really_pin_what_it_claims():
    """A profile silently dropping its `max_*` would make this guard accept a
    command that no longer launches."""
    for name in sorted(SIZING_PROFILES):
        assert _profile_pins(name, "max_cpus"), (
            f"profile `{name}` no longer pins max_cpus"
        )
        assert _profile_pins(name, "max_memory"), (
            f"profile `{name}` no longer pins max_memory"
        )
    for name in sorted(OUTDIR_PROFILES):
        assert _profile_pins(name, "outdir"), f"profile `{name}` no longer pins outdir"


def test_the_extractor_joins_continuations_and_skips_prose():
    text = "\n".join(
        [
            "Some prose naming `nextflow run . --input x` outside a fence.",
            "```bash",
            "nextflow run . \\",
            "  --input samplesheet.csv \\",
            "  --outdir results",
            "```",
        ]
    )
    found = list(_commands(text))
    assert len(found) == 1, found
    line_no, command = found[0]
    assert line_no == 3
    assert command == "nextflow run .  --input samplesheet.csv  --outdir results"


def test_the_detector_recognises_the_command_that_actually_fails():
    assert _verdict(
        "nextflow run . --input s.csv --outdir results -profile docker"
    ) == ["no max_cpus/max_memory (required, no default)"]
    assert _verdict("nextflow run . --input s.csv -profile docker") == [
        "no --outdir (required, no default)",
        "no max_cpus/max_memory (required, no default)",
    ]


def test_the_detector_leaves_the_working_forms_alone():
    assert _verdict("nextflow run . -profile test,docker --outdir results") == []
    assert (
        _verdict(
            "nextflow run . --input s.csv --outdir r -profile docker -c site.config"
        )
        == []
    )
    assert (
        _verdict(
            "nextflow run . --input s.csv --outdir r --max_cpus 8 --max_memory 32.GB"
        )
        == []
    )


if __name__ == "__main__":
    import sys

    if "--emit" in sys.argv[1:]:
        _emit()
    else:
        print(
            "usage: python3 -m tests.test_documented_commands_are_runnable --emit",
            file=sys.stderr,
        )
        sys.exit(2)
