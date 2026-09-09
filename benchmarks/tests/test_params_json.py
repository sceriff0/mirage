"""The sweep could not launch on Nextflow 26 at all, and this is why.

run_sweep.sh passed every swept axis as `--<name> <value>`. NF26 delivers every
CLI --param as a String and nextflow_schema.json declares real types, so
validateParameters() rejected the run before a single task:

    * --reg_qc (1): Value is [string] but should be [integer]
    * --pyramid_resolutions (4): Value is [string] but should be [integer]

Reproduced 2026-08-25: 25.04.7 accepts the CLI form, 26.04.6 rejects it. The
sweep passes ~40 such axes, so the entire experiment was unlaunchable on the
engine the pipeline is verified on -- and it would have failed on the cluster,
at launch, after the input matrix had already been generated.

benchmarks/params_json.py builds a -params-file instead, which carries JSON
types and works on both engines.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import params_json

ROOT = Path(__file__).resolve().parents[2]
TYPES = params_json.schema_types()


def test_the_schema_scan_finds_the_real_params():
    """Every coercion below is driven by this map. If it came back empty, each
    one would silently fall through to 'pass it through as a string' -- which is
    exactly the broken behaviour being fixed."""
    assert len(TYPES) > 50, f"only {len(TYPES)} params read from the schema"
    for expected in ("reg_qc", "memory_mode", "cleanup_work", "pyramid_resolutions"):
        assert expected in TYPES, f"{expected} missing from the schema scan"


@pytest.mark.parametrize(
    "name,raw,expected",
    [
        ("reg_qc", "1", 1),  # the case that broke the sweep
        ("pyramid_resolutions", "4", 4),
        ("memory_mode", "high", "high"),
        ("cleanup_work", "false", False),
        ("cleanup_work", "true", True),
        ("cleanup_level", "none", "none"),  # a STRING whose value is the word none
    ],
)
def test_values_take_the_type_the_schema_declares(name, raw, expected):
    got = params_json.coerce(name, raw, TYPES)
    assert got == expected
    assert isinstance(got, type(expected)), (
        f"{name}={raw!r} came back as {type(got).__name__}, not {type(expected).__name__}"
    )


def test_a_string_param_that_looks_numeric_stays_a_string():
    """The reason coercion reads the schema instead of guessing from the text.
    'Looks like a number' would break validation in the other direction."""
    string_params = [n for n, t in TYPES.items() if t == {"string"}]
    assert string_params, "no purely-string param in the schema to test with"
    name = string_params[0]
    assert params_json.coerce(name, "2048", TYPES) == "2048"


def test_a_boolean_is_not_read_as_an_integer():
    """0 and 1 are legal spellings of both, so order matters inside coerce()."""
    assert params_json.coerce("cleanup_work", "0", TYPES) is False
    assert params_json.coerce("cleanup_work", "1", TYPES) is True


def test_a_nullable_param_accepts_the_empty_cell():
    """The run plan writes a blank cell for a null-declared param (reg_tiled_tile,
    max_forks). It must become JSON null, not the empty string."""
    nullable = [n for n, t in TYPES.items() if "null" in t]
    assert nullable, "no nullable param in the schema"
    assert params_json.coerce(nullable[0], "", TYPES) is None


def test_a_value_the_type_cannot_hold_fails_loudly():
    """Silently passing 'banana' through as a string would move the failure to
    nf-schema, on the cluster, with a message about the wrong layer."""
    with pytest.raises(params_json.CoercionError):
        params_json.coerce("reg_qc", "banana", TYPES)


def test_an_undeclared_param_is_reported_not_dropped(capsys):
    assert params_json.coerce("not_a_real_param", "x", TYPES) == "x"
    assert "not declared" in capsys.readouterr().err


def test_the_cli_writes_a_usable_params_file(tmp_path):
    out = tmp_path / "p.json"
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.params_json",
            "--out",
            str(out),
            "reg_qc=1",
            "memory_mode=high",
            "cleanup_work=false",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    assert doc == {"reg_qc": 1, "memory_mode": "high", "cleanup_work": False}
    # Types survive the round trip through JSON, which is the entire point.
    assert isinstance(doc["reg_qc"], int) and not isinstance(doc["reg_qc"], bool)
    assert doc["cleanup_work"] is False


RUNNERS = ("benchmarks/run_sweep.sh", "benchmarks/run_arms.sh")

# Path-like flags that stay on the command line. They are `"type": "string"` in
# the schema, so a String IS the right type for them, and they are per-run
# locations rather than swept knobs.
CLI_ALLOWED = {"input", "outdir", "trace_dir"}


def _code_lines(rel):
    for i, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        yield i, line


def test_no_runner_passes_a_typed_param_on_the_command_line():
    """A single `--<axis> <value>` reintroduced into a runner puts the whole
    sweep back to failing at launch on NF26."""
    import re

    offenders = []
    for rel in RUNNERS:
        for i, line in _code_lines(rel):
            for m in re.finditer(r"--([a-z_][a-z0-9_]*)\s", line):
                name = m.group(1)
                if name in CLI_ALLOWED or name not in TYPES:
                    continue
                if TYPES[name] == {"string"}:
                    continue
                offenders.append(f"{rel}:{i}: --{name}")
    assert not offenders, (
        "these pass a typed pipeline param on the command line; NF26 delivers it "
        "as a String and validateParameters() rejects the run:\n  "
        + "\n  ".join(offenders)
        + "\n\nBuild a -params-file with benchmarks/params_json.py instead."
    )


def test_no_runner_builds_a_param_flag_dynamically():
    """The form the defect ACTUALLY took, and the reason the check above is not
    enough on its own.

    Neither runner wrote `--reg_qc 1` literally. run_sweep.sh built
    `params+=("--${k}" "${vals[$i]}")` from the run-plan header and run_arms.sh
    had `add_param() { ARGS+=("--$1" "$2"); }`. A scan for literal flag names
    sees neither, so it passed against the broken scripts -- which is exactly the
    "guard that was never observed failing" failure this repo keeps finding.
    """
    import re

    offenders = [
        f"{rel}:{i}: {line.strip()}"
        for rel in RUNNERS
        for i, line in _code_lines(rel)
        if re.search(r'"--\$', line) or re.search(r"'--\$", line)
    ]
    assert not offenders, (
        "these build a `--<param>` flag from a variable, so the param name is "
        "invisible to a literal scan and the value reaches Nextflow as a String:\n  "
        + "\n  ".join(offenders)
        + "\n\nCollect name=value pairs and write a -params-file instead."
    )


def test_both_runners_actually_use_the_params_file():
    """The positive half. Without this, deleting the param plumbing entirely
    would satisfy both checks above while silently running every arm at the
    pipeline defaults -- a benchmark that measures one configuration N times."""
    for rel in RUNNERS:
        text = (ROOT / rel).read_text()
        assert "benchmarks.params_json" in text, (
            f"{rel} no longer builds a typed params file"
        )
        assert "-params-file" in text, f"{rel} no longer passes -params-file"
