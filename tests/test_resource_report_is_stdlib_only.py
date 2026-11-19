"""The resource report runs on the head node, so it may import only the stdlib.

The pipeline entry-point workflow's ``generateResourceReport`` invokes this
script as ``python3 <path>`` from ``workflow.onComplete``. That runs on the
head node, under whatever interpreter the operator has, outside every
container and with none of the pipeline's pinned dependencies. A single
``import matplotlib`` makes the report silently unavailable on every deployment
that does not happen to have it -- and the failure is quiet, because the handler
downgrades every error to a warning so a report problem can never fail a run.

No existing guard covers this file: ``tests/test_container_harmonisation.py``
scans the scripts a process module names, and this script is named by no
process module precisely because it is not a process.

The stdlib set is taken from the interpreter (``sys.stdlib_module_names``) rather
than hand-listed, so a stdlib module is never reported as a missing dependency.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bin" / "generate_resource_report.py"

_STDLIB = set(sys.stdlib_module_names)


def _top_level_imports(path: Path) -> set[str]:
    """Every distribution the module imports, by top-level name.

    Parsed with ``ast``, never a regex: ``^\\s*(import|from)\\s+(\\w+)`` also
    matches prose inside a docstring ("import the module first") and would report
    a package named ``the``.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_the_resource_report_imports_only_the_standard_library():
    imported = _top_level_imports(SCRIPT)
    # __future__ is stdlib but is not in sys.stdlib_module_names on every version.
    third_party = sorted(n for n in imported if n not in _STDLIB and n != "__future__")
    assert not third_party, (
        f"bin/generate_resource_report.py imports {third_party}, which is not in "
        "the standard library. It runs on the head node outside every container "
        "(design finding F6), so a third-party import makes the report silently "
        "unavailable wherever that package is absent -- and the entry-point "
        "workflow downgrades the failure to a warning, so nobody finds out."
    )


def test_the_guard_would_notice_a_third_party_import(tmp_path):
    """The negative case, so this file cannot pass vacuously. A guard whose
    failing branch has never been executed is a guard that checks nothing."""
    probe = tmp_path / "probe.py"
    probe.write_text("import numpy\nimport csv\nfrom pathlib import Path\n")

    imported = _top_level_imports(probe)

    assert "numpy" in imported
    assert "numpy" not in _STDLIB
    assert {"csv", "pathlib"} <= _STDLIB
