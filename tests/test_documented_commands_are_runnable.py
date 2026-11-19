r"""Every `nextflow run` command in the docs must survive launch validation.

Three parameters have no default and are required by `nextflow_schema.json`:
`outdir`, `max_cpus` and `max_memory`. nextflow.config declares the last two
`null` on purpose. A command that supplies none of them is refused before any
process is submitted ("Missing required parameter(s): max_cpus, max_memory").

A documented command satisfies the sizing pair through `-c site.config` (copied
from conf/site.config.template), a profile that pins them, or a `-params-file`
preset that carries them. Commands are read the way a reader copies them: out of
fenced code blocks, following backslash continuations.

`_verdict()` is a static token check and never invokes Nextflow, so it cannot see
an unrecognised flag or a missing samplesheet. The execution counterpart is
`tests/documented_commands_launch.sh`: it takes the same command list from
`python3 -m tests.test_documented_commands_are_runnable --emit`, substitutes real
fixtures for placeholders, and runs each with `-stub -params-file
params/dry_run.json`, which validates without pulling a container or submitting
a process.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# rglob: navigated pages live in subdirectories too. The trees mkdocs.yml's
# exclude_docs leaves unpublished quote history, not instructions, so they are skipped.
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
SIZING_PROFILES = {"test", "test_full", "local", "shipped_defaults_test"}
OUTDIR_PROFILES = {"test", "test_full", "shipped_defaults_test"}

_FENCE = re.compile(r"^\s*(```|~~~)")
# A line may show a broken command if it says so on the same line (`# x fails`).
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
    are left exactly as written. `--dry_run` exits before any process or
    container is touched, so the reader's own profile choice is exercised at
    no extra cost.
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


# The "no reason" sentinel, never an empty field: bash's `read` treats tab as IFS
# whitespace and collapses an empty field between two tabs, shifting the rest.
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
