"""Guard test: every bin/utils/*.py module must have a real PRODUCTION importer.

This repo uses two import conventions for `bin/utils/*.py` (because many
scripts do `sys.path.insert(0, .../bin/utils)` before importing):

- flat:      `from image_utils import ensure_dir`
- package:   `from utils.metadata import ...`, `from utils import cell_pairs as cp`

"Real importer" means a module under `bin/` (a script or another `bin/utils/`
module) -- not a test. The haystack is deliberately split into a production
half (`_production_haystack_files()`, everything under `bin/`) and a test
half (`_test_haystack_files()`, everything under `tests/`): only the
production half counts as proof of life for the main guard
(`test_bin_utils_module_has_a_real_importer`). A module imported
*exclusively* by its own test file is exactly the false "alive" this split
exists to catch. Earlier this test merged both halves into one haystack, so
a module wired up only for its own test (never called from any real script)
counted as "used" -- a guard that could never actually fail this way is not
evidence of anything. `tiled_pipeline.py` is the one module that genuinely
has no production importer; it is allowlisted below with its reason rather
than silently passing through the old merged haystack.

A naive "does the bare word 'tiling' appear anywhere" scan produces false
negatives for short names (`qc`, `retry`, `validation` all appear constantly
in prose/comments/unrelated identifiers) and false positives are just as
easy: it would call a module "used" because its own name shows up in a
sentence. This test instead parses every candidate file with `ast` and only
counts genuine `import`/`from ... import ...` statements, resolved against
each module's basename and its dotted path under `utils.`.

bin/utils/__init__.py is deliberately excluded from the set of files this
test treats as "importers", as defense in depth against a barrel re-export
being counted as usage. Historically it barrel re-exported `cli.py`
(`create_base_parser`, `setup_logging_from_args`, `add_input_output_args`),
`constants.py` (`ExitCode`, `DEFAULT_FOV_SIZE`, etc.), and later `logger.py`
(`log_timing`, `timed`) even though nothing downstream ever consumed those
re-exported names (verified by exhaustive grep across bin/, tests/,
modules/, conf/, docs/, and containers/ for every symbol in all three
modules). Counting that barrel import as "usage" would have hidden each
module's deadness behind its own package's re-export statement -- the exact
false-negative this exclusion exists to prevent. All three re-export blocks
have since been deleted (see git history); __init__.py currently re-exports
nothing. Modules used elsewhere via `from logger import ...`, `from
metadata import ...`, etc. are unaffected, since those imports come
directly from consumer scripts, not through the barrel.

Known latent limitation: because __init__.py is excluded from the haystack,
a *future* module consumed *exclusively* via `from utils import
<exported_name>` -- i.e. only through a barrel-exported symbol, never by
its own module path or basename -- would be misclassified as dead. The
matcher recognizes `from utils import <module_name>` as proof of life
(e.g. `from utils import cell_pairs`), but has no way to trace `from utils
import <a_symbol_defined_inside_that_module>` back to the module that
defines it, since the symbol's name need not match the module's basename
(e.g. `get_logger` vs. `logger.py`). Not currently triggered -- verified:
`logger.py` is alive via direct imports (`from utils.logger import ...` /
`from logger import ...`), and no bin/utils module is consumed exclusively
through a barrel-exported symbol (the barrel currently re-exports nothing).
A future module that *is* exposed and consumed only that way would need
either a direct import somewhere or an ALLOWLIST entry to avoid a false
"dead" flag.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

from tests.stare_shims import shim_target, shims

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
UTILS_DIR = BIN_DIR / "utils"
TESTS_DIR = REPO_ROOT / "tests"

# bin/utils/__init__.py's barrel re-import doesn't count as genuine usage --
# see module docstring above.
HAYSTACK_EXCLUDE = {UTILS_DIR / "__init__.py"}

# Modules with zero real importers that are intentionally kept anyway.
#
# Map of basename -> {"status": ..., "reason": ...}. `status` is structural,
# not prose, so a machine (and a future reader skimming, not reading) can
# tell the two fundamentally different kinds of entry apart:
#
#   "permanent"        -- deliberate, standing policy. Keep indefinitely.
#   "pending_removal"  -- temporary debt: the module is dead by this test's
#                         own standard, deletion is just deferred. Without a
#                         structural marker, "temporary" silently becomes
#                         "permanent" because nobody re-reads the prose.
#                         See test_pending_removal_entries_are_flagged_loudly:
#                         any entry with this status is surfaced as a pytest
#                         warning on every run, not just skipped quietly.
_ALLOWED_STATUSES = {"permanent", "pending_removal"}

ALLOWLIST: dict[str, dict[str, str]] = {
    "tiled_pipeline": {
        "status": "permanent",
        "reason": (
            "No production importer: only tests/test_tiled_pipeline.py and "
            "tests/test_reg_benchmark.py import it. It is a deliberate "
            "test/benchmark oracle for the STARE registration path, documented "
            "at CHANGELOG.md:487-488 -- not dead code, just never called from "
            "a production script. (Since the move into packages/stare the bin/utils "
            "file is a shim over stare.pipeline and is exempt via _candidate_modules; "
            "this entry stays so test_allowlisted_tiled_pipeline_has_no_production_"
            "importer keeps the oracle's test-only status checked.)"
        ),
    },
    "reg_benchmark": {
        "status": "permanent",
        "reason": (
            "No production importer since bin/registration_benchmark.py -- its only "
            "one -- was deleted from this branch for release 1.0. The CLI was a "
            "hand-run harness no module or conf/ file ever invoked, and it survives "
            "on the `benchmarking` branch alongside the sweep that drives it. The "
            "library it wrapped stays here because it is a deliberate ACCURACY "
            "ORACLE: tests/test_reg_benchmark.py and tests/test_tiled_fanout.py "
            "both import it, and it keeps skimage's ORB as an oracle independent "
            "of the DISK/LightGlue front-end the pipeline itself uses (see "
            "tests/test_no_legacy_frontends.py's exemption for the same reason). "
            "Not dead code -- a test/benchmark oracle, like tiled_pipeline above."
        ),
    },
}


def _candidate_modules() -> list[Path]:
    """bin/utils/**/*.py, excluding __init__.py and the STARE shims.

    A SHIM (`tests/stare_shims.py`: a docstring, `from stare... import x as _impl`,
    `sys.modules[__name__] = _impl`, nothing else) is not a module with a body that could
    be dead code -- it is a compatibility name over `packages/stare`, kept so the flat
    `from mesh_field import MeshField` convention the tests, bin/warp_seg_qc.py and
    benchmarks/anhir/warp.py use keeps resolving. The production-importer rule below
    would flag six of them (their former production importers moved into the package
    with them) for a reason that does not apply. They get their OWN rule instead,
    `test_every_stare_shim_is_imported_by_something`: a shim nothing under bin/, tests/
    or benchmarks/ imports is a name nobody needs and is deleted, not kept.
    """
    mods = []
    for path in sorted(UTILS_DIR.rglob("*.py")):
        if path.name == "__init__.py" or shim_target(path) is not None:
            continue
        mods.append(path)
    return mods


def _production_haystack_files() -> list[Path]:
    """Every .py file under bin/ (scripts and other bin/utils/ modules).

    The only files whose imports count as proof a module is genuinely used. A test
    importing a module does not belong here; see the module docstring for why.
    """
    files = list(BIN_DIR.rglob("*.py"))
    return [f for f in files if f not in HAYSTACK_EXCLUDE]


def _test_haystack_files() -> list[Path]:
    """Every .py file under tests/.

    Only used for allowlist bookkeeping (e.g. confirming a kept-for-tests module
    really is reached from a test); never as proof of life for the main guard.
    """
    return list(TESTS_DIR.rglob("*.py"))


def _dotted_parts(module_path: Path) -> tuple[str, ...]:
    """Dotted path parts relative to bin/utils, e.g. ('image_utils',)."""
    rel = module_path.relative_to(UTILS_DIR).with_suffix("")
    return rel.parts


def _imported_leaf_names(py_file: Path) -> set[str]:
    """All module leaf-names this file genuinely imports (any real import form)."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    leaves: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import tiling / import utils.tiling
            for alias in node.names:
                leaves.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_parts = node.module.split(".")
                # from tiling import X / from utils.tiling import X
                leaves.add(mod_parts[-1])
                # from utils import cell_pairs: the package itself is named, and
                # the real submodule reference is one of the imported names.
                if mod_parts[-1] == "utils":
                    for alias in node.names:
                        leaves.add(alias.name)
            else:
                # from . import cell_pairs  (level > 0, no module)
                for alias in node.names:
                    leaves.add(alias.name)
    return leaves


def _is_imported_anywhere(module_path: Path, haystack: list[Path]) -> bool:
    basename = module_path.stem
    dotted_leaf = _dotted_parts(module_path)[-1]
    assert basename == dotted_leaf
    for f in haystack:
        if f == module_path:
            continue
        if basename in _imported_leaf_names(f):
            return True
    return False


CANDIDATES = _candidate_modules()


@pytest.mark.parametrize(
    "module_path",
    CANDIDATES,
    ids=[str(p.relative_to(UTILS_DIR)) for p in CANDIDATES],
)
def test_bin_utils_module_has_a_real_importer(module_path: Path) -> None:
    basename = module_path.stem
    if basename in ALLOWLIST:
        entry = ALLOWLIST[basename]
        pytest.skip(f"allowlisted ({entry['status']}): {entry['reason']}")

    haystack = _production_haystack_files()
    assert _is_imported_anywhere(module_path, haystack), (
        f"bin/utils/{module_path.relative_to(UTILS_DIR)} is not imported by any real "
        "import statement (flat or package form) under bin/. A test-only import does "
        "not count -- see the module docstring. If it's genuinely dead, delete it; if "
        "it should be kept, add it to ALLOWLIST in this test with a reason."
    )


def test_every_stare_shim_is_imported_by_something() -> None:
    """The rule for a STARE shim (see `_candidate_modules`): it must be imported by a
    real import statement somewhere under bin/, tests/ or benchmarks/ -- by its FLAT name,
    which is the only reason the shim exists. One with no importer at all is deleted."""
    haystack = _production_haystack_files() + _test_haystack_files()
    haystack += list((REPO_ROOT / "benchmarks").rglob("*.py"))
    # bin/utils only: the bin/tiled_*.py shims are ENTRY POINTS the Nextflow modules
    # invoke by name (tests/test_bin_entrypoint_idiom.py covers them), not import names.
    found = [shim for shim in shims() if UTILS_DIR in shim.parents]
    assert found, (
        "no STARE shims found under bin/utils -- tests/stare_shims.py is broken"
    )
    orphans = [
        str(shim.relative_to(REPO_ROOT))
        for shim in found
        if not _is_imported_anywhere(shim, haystack)
    ]
    assert not orphans, (
        f"STARE shim(s) nothing imports by their flat name: {orphans}. A shim exists only "
        "to keep an existing `from <name> import ...` resolving; with no such importer "
        "it is dead and should be deleted (import from `stare` directly instead)."
    )


def test_a_shim_is_excluded_from_the_dead_module_rule_only_if_it_really_is_one(
    tmp_path: Path,
) -> None:
    """`shim_target` must say None for a module with a body -- otherwise adding one
    `sys.modules` line to a real module would exempt it from the dead-code rule."""
    real = tmp_path / "real_module.py"
    real.write_text(
        '"""doc"""\nimport sys\nfrom stare import mesh_field as _impl\n\n'
        "def helper():\n    return 1\n\nsys.modules[__name__] = _impl\n"
    )
    assert shim_target(real) is None
    shim = tmp_path / "shim.py"
    shim.write_text(
        '"""doc"""\nimport sys\nfrom stare import mesh_field as _impl\n\n'
        "sys.modules[__name__] = _impl\n"
    )
    assert shim_target(shim) is not None and shim_target(shim).name == "mesh_field.py"


def test_known_alive_modules_are_not_false_flagged() -> None:
    """Sanity check the matcher itself against modules known to be genuinely used."""
    haystack = _production_haystack_files()
    # logger.py is alive purely through direct imports (`from utils.logger import
    # ...` / `from logger import ...`) -- bin/utils/__init__.py currently re-exports
    # nothing, so this also confirms the matcher doesn't depend on the (excluded)
    # barrel to find it.
    for name in ("image_utils", "validation", "metadata", "cell_pairs", "logger"):
        candidates = [p for p in CANDIDATES if p.stem == name]
        assert candidates, f"expected to find bin/utils/{name}.py among candidates"
        assert _is_imported_anywhere(candidates[0], haystack), (
            f"matcher false-negatived a module known to be alive: {name}"
        )


def test_test_only_import_is_not_proof_of_life(tmp_path: Path) -> None:
    """A module imported ONLY by a test must NOT count as alive.

    This is the specific hole the production/test haystack split closes. Before the
    split, `_haystack_files()` merged bin/ and tests/ into one list, so a module
    wired up only for its own test (never called from any real script) satisfied
    `_is_imported_anywhere` -- proof of life from a test-only import. Plants a test
    file that imports a module name found nowhere else, and asserts: the test
    haystack DOES see the import (the plant is realistic), but the production
    haystack -- the only one the main guard consults -- does not, which is exactly
    what would make the main guard flag such a module as dead.
    """
    fake_module_name = "planted_dead_module"
    fake_test = tmp_path / "test_planted_dead_module.py"
    fake_test.write_text(f"from {fake_module_name} import unused\n")

    assert fake_module_name in _imported_leaf_names(fake_test), (
        "sanity check failed: the plant itself isn't realistic"
    )

    production_leaves: set[str] = set()
    for f in _production_haystack_files():
        production_leaves |= _imported_leaf_names(f)
    assert fake_module_name not in production_leaves, (
        "a module imported only by a test must not be found in the real "
        "production haystack"
    )


def test_allowlisted_tiled_pipeline_has_no_production_importer() -> None:
    """Confirms the tiled_pipeline ALLOWLIST reason is still true, not stale prose.

    tiled_pipeline.py should have zero production importers (it's kept as a
    test/benchmark oracle, not a live dependency) but at least one test importer
    (otherwise it isn't even that, and should be deleted rather than allowlisted).
    """
    # the shim is not a candidate (see _candidate_modules); check the file directly
    candidates = [UTILS_DIR / "tiled_pipeline.py"]
    assert candidates[0].is_file(), "expected bin/utils/tiled_pipeline.py to exist"
    assert not _is_imported_anywhere(candidates[0], _production_haystack_files()), (
        "tiled_pipeline.py now has a production importer -- the ALLOWLIST entry's "
        "reason (test/benchmark-only oracle) is stale; remove it from ALLOWLIST."
    )
    assert _is_imported_anywhere(candidates[0], _test_haystack_files()), (
        "tiled_pipeline.py has neither a production nor a test importer -- it's "
        "genuinely dead, not a kept oracle; delete it instead of allowlisting."
    )


def test_allowlist_entries_declare_a_valid_status() -> None:
    """Every ALLOWLIST entry must be a structured {status, reason} dict.

    This is what makes "permanent" and "pending_removal" a real distinction
    instead of a convention someone can forget: a bare string reason (the
    old format) or an unrecognized status value fails collection outright,
    forcing whoever defers a module next to explicitly pick a kind.
    """
    for name, entry in ALLOWLIST.items():
        assert isinstance(entry, dict) and set(entry) == {"status", "reason"}, (
            f"ALLOWLIST[{name!r}] must be a dict with exactly 'status' and "
            "'reason' keys, not a bare string -- see the ALLOWLIST comment."
        )
        assert entry["status"] in _ALLOWED_STATUSES, (
            f"ALLOWLIST[{name!r}] has status={entry['status']!r}; must be "
            f"one of {sorted(_ALLOWED_STATUSES)}."
        )


def test_pending_removal_entries_are_flagged_loudly() -> None:
    """`pending_removal` entries must not sit in ALLOWLIST silently.

    A `pending_removal` entry marks temporary debt -- a module this test
    would otherwise flag as dead, with deletion deferred to a later task --
    as opposed to a `permanent` entry, which is deliberate standing policy
    (e.g. a module deliberately kept for a documented external consumer).
    Deferring is a legitimate outcome, so this test doesn't fail merely
    because one exists. But it must not be
    possible to defer something and have it quietly age into permanence:
    every `pending_removal` entry is re-emitted as a pytest warning on
    *every* run, so it shows up in the warnings summary (which CI logs and
    reviewers actually scan) instead of disappearing into ALLOWLIST prose
    nobody re-reads once it's merged.
    """
    pending = {
        name: entry
        for name, entry in ALLOWLIST.items()
        if entry["status"] == "pending_removal"
    }
    for name, entry in pending.items():
        warnings.warn(
            f"ALLOWLIST['{name}'] is status=pending_removal (temporary debt, "
            f"not permanent policy) -- deletion is still owed: {entry['reason']}",
            stacklevel=1,
        )
