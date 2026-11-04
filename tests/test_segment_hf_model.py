"""The shipped StarDist model is a Hugging Face snapshot pinned to one commit.

``nextflow.config``'s ``segmentation_model`` is ``hf://DSUnit/stardist@<commit>``;
``bin/segment.py::resolve_hf_model`` downloads its three files once and hands
StarDist a local model folder. These tests pin the contract without network:

* the shipped default parses, names a 40-hex commit, and agrees with the schema;
* a branch name (``@main``) or a missing revision is refused, because the weights
  behind a moving ref can change under a run;
* a folder already in the cache is used WITHOUT touching the network (the
  air-gapped route documented in docs/parameters.md);
* a download goes through ``HF_ENDPOINT`` (here a ``file://`` tree standing in for
  the Hub), lands every file, and leaves no ``.part`` file behind;
* a failed download says how to pre-download instead of a bare URLError.

segment.py imports stardist/csbdeep at module level; they are stubbed in
``sys.modules`` so the resolver is testable in any environment.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HF_DEFAULT = re.compile(r"segmentation_model\s*=\s*'(hf://[^']+)'")


@pytest.fixture(scope="module")
def segment():
    for d in (ROOT / "bin", ROOT / "bin" / "utils"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    stubs = {
        "stardist": types.ModuleType("stardist"),
        "stardist.models": types.ModuleType("stardist.models"),
        "csbdeep": types.ModuleType("csbdeep"),
        "csbdeep.utils": types.ModuleType("csbdeep.utils"),
    }
    setattr(stubs["stardist.models"], "StarDist2D", object)
    setattr(stubs["csbdeep.utils"], "normalize", lambda *_a, **_k: None)
    saved = {k: sys.modules.get(k) for k in stubs}
    for k, v in stubs.items():
        sys.modules.setdefault(k, v)
    try:
        yield importlib.import_module("segment")
    finally:
        for k, v in saved.items():
            if v is None and sys.modules.get(k) is stubs[k]:
                del sys.modules[k]


def shipped_default() -> str:
    m = HF_DEFAULT.search((ROOT / "nextflow.config").read_text())
    assert m, "nextflow.config no longer ships an hf:// segmentation_model"
    return m.group(1)


def test_shipped_default_is_a_pinned_commit_and_matches_schema(segment):
    spec = shipped_default()
    m = segment.HF_SPEC.match(spec)
    assert m, spec
    assert m.group("repo") == "DSUnit/stardist"
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    props = {}
    for d in schema.get("$defs", schema.get("definitions", {})).values():
        props.update(d.get("properties", {}))
    assert props["segmentation_model"]["default"] == spec
    assert props["seg_method"]["default"] == "stardist"


@pytest.mark.parametrize(
    "bad",
    [
        "hf://DSUnit/stardist",
        "hf://DSUnit/stardist@main",
        "hf://DSUnit/stardist@68b94406",
        "hf://stardist@68b94406104bb069a58b64f4c890375c7f2f15df",
    ],
)
def test_moving_or_partial_refs_are_refused(segment, bad, tmp_path):
    with pytest.raises(ValueError, match="40-hex commit"):
        segment.resolve_hf_model(bad, str(tmp_path))


def test_cached_folder_is_used_offline(segment, tmp_path, monkeypatch):
    spec = shipped_default()
    rev = spec.rsplit("@", 1)[1]
    folder = tmp_path / f"DSUnit__stardist__{rev[:12]}"
    folder.mkdir()
    for f in segment.HF_FILES:
        (folder / f).write_text("x")

    def no_network(*a, **k):
        raise AssertionError("cache hit must not open a URL")

    monkeypatch.setattr(segment.urllib.request, "urlopen", no_network)
    assert segment.resolve_hf_model(spec, str(tmp_path)) == (str(tmp_path), folder.name)


def test_download_goes_through_hf_endpoint(segment, tmp_path, monkeypatch):
    spec = shipped_default()
    rev = spec.rsplit("@", 1)[1]
    hub = tmp_path / "hub"
    src = hub / "DSUnit" / "stardist" / "resolve" / rev
    src.mkdir(parents=True)
    for f in segment.HF_FILES:
        (src / f).write_bytes(f.encode() * 3)
    monkeypatch.setenv("HF_ENDPOINT", hub.as_uri())
    cache = tmp_path / "cache"
    basedir, name = segment.resolve_hf_model(spec, str(cache))
    folder = Path(basedir) / name
    for f in segment.HF_FILES:
        assert (folder / f).read_bytes() == f.encode() * 3
    assert not list(folder.glob("*.part"))


def test_failed_download_explains_the_offline_route(segment, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", (tmp_path / "nowhere").as_uri())
    monkeypatch.setattr(segment.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="pre-download"):
        segment.resolve_hf_model(shipped_default(), str(tmp_path / "cache"))
