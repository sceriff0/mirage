"""Every `bin/` CLI must propagate its `main()` return code to the process exit status.

Nextflow decides a task failed by reading the exit status, and `conf/base.config`'s
`errorStrategy` selects on `task.exitStatus`. A `__main__` block that calls `main()`
and throws the result away therefore reports success for a `main()` that returned 1,
and the run goes green on a task that did not do its job.

Two forms are accepted:

    raise SystemExit(main())
    sys.exit(main())

Two are rejected:

  * a bare ``main()`` -- discards the return code outright.
  * ``exit(main())`` -- ``exit`` is injected into builtins by the ``site`` module and
    is absent under ``python -S`` and in frozen/embedded interpreters. It also reads
    as ``sys.exit`` to a hurried reader while being a different object.

Scope is ``bin/**/*.py`` minus ``bin/utils/cse/``, which is vendored (the same
exclusion ``ruff.toml`` applies) and is held to its upstream style.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "bin"
VENDORED = BIN / "utils" / "cse"

ACCEPTED = ("raise SystemExit(main())", "sys.exit(main())")

_GUARD = re.compile(
    r'^if __name__ == ["\']__main__["\']:\s*\n((?:[ \t]+.*\n?)+)', re.MULTILINE
)


def _entrypoints() -> dict[str, str]:
    """Map ``bin``-relative path -> the stripped body of its ``__main__`` block."""
    out: dict[str, str] = {}
    for path in sorted(BIN.rglob("*.py")):
        if VENDORED in path.parents:
            continue
        match = _GUARD.search(path.read_text())
        if match:
            out[path.relative_to(REPO).as_posix()] = match.group(1).strip()
    return out


ENTRYPOINTS = _entrypoints()


def test_the_scan_found_the_scripts_it_is_meant_to_cover():
    """A scope glob that matches nothing passes vacuously; this is the tripwire.

    The floor is well below the real count, so deleting one script does not fail
    this test while a regex or glob that stops matching does.
    """
    assert len(ENTRYPOINTS) >= 25, sorted(ENTRYPOINTS)
    assert {
        "bin/convert_image.py",
        "bin/quantify.py",
        "bin/segment.py",
        "bin/register.py",
        "bin/export_geojson.py",
    } <= set(ENTRYPOINTS), sorted(ENTRYPOINTS)


@pytest.mark.parametrize("script", sorted(ENTRYPOINTS))
def test_entrypoint_uses_an_exit_propagating_idiom(script):
    body = ENTRYPOINTS[script]
    assert body in ACCEPTED, (
        f"{script}'s __main__ block is {body!r}. Use one of {ACCEPTED}: a bare "
        f"main() discards the return code (a failing task reports exit 0 to "
        f"Nextflow), and the builtin exit() is injected by `site` and is absent "
        f"under `python -S`."
    )
