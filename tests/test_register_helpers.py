"""Real CI coverage for bin/register.py's pure, VALIS-free helper logic.

bin/register.py imports valis at module level and VALIS is not installed in
CI, so register.py cannot be imported there. Its pure helpers (filename/marker
matching, reference-file glob discovery) are covered without VALIS in two ways:

1. ``find_reference_image`` and ``validate_input_slides`` are standalone
   top-level functions with no VALIS dependency in their bodies. Their source
   is extracted via ``ast`` directly from the current bin/register.py text and
   exec'd in an isolated namespace, so the real source is what runs.

2. The reference-image glob fallback (the ``*.ome.tif`` / ``*.ome.tiff``
   union-glob inside ``valis_registration()``) is inline in a function that
   builds a VALIS Registrar, so it is hand-replicated here. A canary test greps
   bin/register.py for the exact glob literals, so an edit there without
   updating the replica fails.
"""

from __future__ import annotations

import ast
import glob
import os
from pathlib import Path

import pytest
import tifffile

REGISTER_PY = Path(__file__).parent.parent / "bin" / "register.py"


class _StubLogger:
    """No-op logger stand-in so extracted functions don't need the real one."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _extract_function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"function {name!r} not found in {REGISTER_PY}")


def _load_helper(name: str, namespace: dict):
    """Exec a single top-level function from register.py's real source.

    ``from __future__ import annotations`` is prepended so the function's type
    hints (List[str], Optional[...], etc.) stay lazy strings and never need to
    be resolved -- keeping the exec namespace free of typing imports.
    """
    source = REGISTER_PY.read_text()
    func_src = _extract_function_source(source, name)
    exec(  # noqa: S102 - intentional: isolates one function from a valis-only module
        compile(
            "from __future__ import annotations\n" + func_src, str(REGISTER_PY), "exec"
        ),
        namespace,
    )
    return namespace[name]


find_reference_image = _load_helper(
    "find_reference_image", {"os": os, "logger": _StubLogger()}
)
validate_input_slides = _load_helper(
    "validate_input_slides", {"os": os, "tifffile": tifffile, "logger": _StubLogger()}
)


class TestFindReferenceImage:
    """Marker-matching + extension filtering (register.py's real source)."""

    def test_finds_single_match_ome_tif(self, tmp_path):
        (tmp_path / "sample_DAPI_SMA.ome.tif").write_bytes(b"")
        (tmp_path / "sample_PANCK.ome.tif").write_bytes(b"")
        result = find_reference_image(str(tmp_path), ["DAPI", "SMA"])
        assert result == "sample_DAPI_SMA.ome.tif"

    def test_finds_single_match_ome_tiff(self, tmp_path):
        (tmp_path / "sample_DAPI_SMA.ome.tiff").write_bytes(b"")
        result = find_reference_image(str(tmp_path), ["DAPI", "SMA"])
        assert result == "sample_DAPI_SMA.ome.tiff"

    def test_finds_single_match_plain_tif_and_tiff(self, tmp_path):
        (tmp_path / "sample_DAPI_SMA.tif").write_bytes(b"")
        result = find_reference_image(str(tmp_path), ["DAPI", "SMA"])
        assert result == "sample_DAPI_SMA.tif"

        (tmp_path / "sample_DAPI_SMA.tif").unlink()
        (tmp_path / "sample_DAPI_SMA.tiff").write_bytes(b"")
        result = find_reference_image(str(tmp_path), ["DAPI", "SMA"])
        assert result == "sample_DAPI_SMA.tiff"

    def test_case_insensitive_marker_match(self, tmp_path):
        (tmp_path / "sample_dapi_sma.ome.tif").write_bytes(b"")
        result = find_reference_image(str(tmp_path), ["DAPI", "SMA"])
        assert result == "sample_dapi_sma.ome.tif"

    def test_excludes_non_image_extensions(self, tmp_path):
        (tmp_path / "sample_DAPI_SMA.csv").write_bytes(b"")
        (tmp_path / "sample_DAPI_SMA.txt").write_bytes(b"")
        with pytest.raises(FileNotFoundError):
            find_reference_image(str(tmp_path), ["DAPI", "SMA"])

    def test_raises_filenotfound_when_no_match(self, tmp_path):
        (tmp_path / "sample_PANCK.ome.tif").write_bytes(b"")
        with pytest.raises(FileNotFoundError):
            find_reference_image(str(tmp_path), ["DAPI", "SMA"])

    def test_raises_valueerror_on_multiple_matches(self, tmp_path):
        (tmp_path / "a_DAPI_SMA.ome.tif").write_bytes(b"")
        (tmp_path / "b_DAPI_SMA.ome.tiff").write_bytes(b"")
        with pytest.raises(ValueError):
            find_reference_image(str(tmp_path), ["DAPI", "SMA"])


class TestValidateInputSlides:
    """Extension filtering + basic shape validation (register.py's real source)."""

    def test_accepts_ome_tif_and_ome_tiff(self, tmp_path):
        tifffile.imwrite(tmp_path / "a.ome.tif", _valid_image())
        tifffile.imwrite(tmp_path / "b.ome.tiff", _valid_image())
        valid, invalid = validate_input_slides(str(tmp_path))
        assert {os.path.basename(p) for p in valid} == {"a.ome.tif", "b.ome.tiff"}
        assert invalid == []

    def test_ignores_non_tiff_extensions(self, tmp_path):
        (tmp_path / "notes.txt").write_bytes(b"not an image")
        (tmp_path / "meta.csv").write_bytes(b"col1,col2\n")
        valid, invalid = validate_input_slides(str(tmp_path))
        assert valid == []
        assert invalid == []

    def test_flags_too_small_image(self, tmp_path):
        tifffile.imwrite(tmp_path / "tiny.ome.tif", _valid_image(shape=(5, 5)))
        valid, invalid = validate_input_slides(str(tmp_path))
        assert valid == []
        assert len(invalid) == 1
        assert "Too small" in invalid[0][1]


def _valid_image(shape=(20, 20)):
    import numpy as np

    return np.zeros(shape, dtype=np.uint8)


class TestReferenceGlobFallbackPattern:
    """Hand-replica of the inline fallback glob inside valis_registration()
    (bin/register.py, the union of ``*.ome.tif`` and ``*.ome.tiff`` globs used
    when no marker-based reference match is found). Inline in a VALIS-heavy
    function, so it can't be AST-extracted standalone like the helpers above.
    """

    @staticmethod
    def _glob_reference_candidates(input_dir: str):
        # Verbatim copy of bin/register.py's fallback-glob block.
        return sorted(
            set(glob.glob(os.path.join(input_dir, "*.ome.tif")))
            | set(glob.glob(os.path.join(input_dir, "*.ome.tiff")))
        )

    def test_matches_ome_tif_and_ome_tiff(self, tmp_path):
        (tmp_path / "a.ome.tif").write_bytes(b"")
        (tmp_path / "b.ome.tiff").write_bytes(b"")
        result = self._glob_reference_candidates(str(tmp_path))
        assert {os.path.basename(p) for p in result} == {"a.ome.tif", "b.ome.tiff"}

    def test_excludes_non_ome_files(self, tmp_path):
        (tmp_path / "plain.tif").write_bytes(b"")
        (tmp_path / "plain.tiff").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        result = self._glob_reference_candidates(str(tmp_path))
        assert result == []

    def test_empty_dir_yields_no_matches(self, tmp_path):
        result = self._glob_reference_candidates(str(tmp_path))
        assert result == []

    def test_glob_pattern_matches_register_py_source(self):
        """Canary: fails if bin/register.py's fallback-glob literals change,
        signalling that the hand-replica above must be updated to match.
        """
        source = REGISTER_PY.read_text()
        assert 'glob.glob(os.path.join(input_dir, "*.ome.tif"))' in source
        assert 'glob.glob(os.path.join(input_dir, "*.ome.tiff"))' in source
