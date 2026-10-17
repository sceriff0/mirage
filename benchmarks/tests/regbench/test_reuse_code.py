"""What a method's reuse key (and the method-code freeze) counts as its code:
``methods.code_files``. Pre-freeze review wave 1, findings F10 (ASHLAR's tile cutting),
F14 (the pipeline code ``valis_mirage`` runs) and F23 (the driver's own imports)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from benchmarks.regbench import METHODS, methods

REPO = Path(__file__).resolve().parents[3]
REGBENCH = REPO / "benchmarks" / "regbench"


def rel(method):
    return {str(p.relative_to(REPO)) for p in methods.code_files(method)}


@pytest.mark.parametrize("method", METHODS)
def test_every_method_key_has_the_drivers_result_code(method):
    """F23: the driver moves the landmarks (landmark_io), loads the points and writes the
    files (cases): a change there changes every result, so every key holds them."""
    files = rel(method)
    for name in ("landmark_io.py", "cases.py", "mapping.py", "resources.py"):
        assert f"benchmarks/regbench/{name}" in files, (method, name)
    for name in ("billing.py", "plugins.py", "selfcheck.py", "tissue.py"):
        assert f"benchmarks/regbench/{name}" not in files, name


def test_every_import_of_the_driver_is_in_the_key_or_left_out_for_a_reason():
    """A new import of the driver must be classified: in DRIVER_CODE or left out with
    its reason (DRIVER_CODE_LEFT_OUT)."""
    tree = ast.parse((REGBENCH / "methods" / "__init__.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 2:
            names = [node.module] if node.module else [a.name for a in node.names]
            for name in names:
                if (REGBENCH / f"{name}.py").is_file():
                    imported.add(f"{name}.py")
    assert imported, "the scan found no import of the driver"
    classified = set(methods.DRIVER_CODE) | set(methods.DRIVER_CODE_LEFT_OUT)
    assert imported - classified == set()
    assert set(methods.DRIVER_CODE) & set(methods.DRIVER_CODE_LEFT_OUT) == set()


def test_valis_mirage_key_holds_the_pipeline_code_it_runs():
    """F14: valis_mirage loads bin/register.py by path; that file and every bin/utils
    module it imports on the adapter's path are its code. The checkpoint writer is
    imported only with a stage_checkpoint_dir, which the adapter never passes."""
    files = rel("valis_mirage")
    want = {
        "bin/register.py",
        *(
            f"bin/utils/{m}.py"
            for m in (
                "image_utils",
                "ome_io",
                "jvm_cache",
                "pixel_size",
                "tiled_io",
                "logger",
                "progress",
                "retry",
                "valis_micro_rigid",
                "valis_config",
                "valis_preflight",
            )
        ),
    }
    assert want <= files, sorted(want - files)
    assert "bin/utils/stage_checkpoint.py" not in files
    assert "bin/utils/valis_stage_warp.py" not in files


def test_ashlar_key_holds_the_code_that_cuts_its_tiles():
    """F10 (a): the adapter imports benchmarks.ashlar.original / retile absolutely; they
    and the bin/utils readers they import are its code."""
    files = rel("ashlar")
    assert {
        "benchmarks/ashlar/original.py",
        "benchmarks/ashlar/retile.py",
        "bin/utils/tiled_io.py",
        "bin/utils/tiled_manifest.py",
    } <= files


def test_code_outside_regbench_is_named_from_the_repository_root(monkeypatch):
    """Two files of one basename outside the package must not hash as one name."""
    import hashlib

    paths = [REPO / "bin" / "register.py", REGBENCH / "cases.py"]
    monkeypatch.setattr(methods, "code_files", lambda m: paths)
    h = hashlib.sha256()
    for path in sorted(paths):
        name = "cases.py" if path.name == "cases.py" else str(path.relative_to(REPO))
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    assert methods.code_fingerprint("initial") == h.hexdigest()
