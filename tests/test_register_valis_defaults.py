"""REGISTER runs valis-wsi 1.2.0 at its defaults, with ONE argument that is not VALIS's.

(Its other mode, `reg_valis_mode hires`, is VALIS's own high-resolution example; it is pinned
by tests/test_register_valis_modes.py. This file pins the default.)

Ruling 2026-11-05: MIRAGE's VALIS mode is the benchmark's ``valis120`` call --
``Valis(src, dst, img_list=..., reference_img_f=...)``, ``register()``, every slide warped in
the reference frame. The tuned 1.0.0 configuration (tiers, SuperPoint/SuperGlue, keypoint cap,
micro-registration, ``align_to_reference=True``) is archived at ``archive/valis-mirage-tuned``.

The one deviation is the registration CHANNEL. VALIS 1.2.0's default fluorescence processing
is ``{"channel": "dapi", "adaptive_eq": True}`` (registration.py:66), and
``SlideReader.get_channel_index`` (slide_io.py:1074-1099) upper-cases the slide's channel names
and takes ``difflib.get_close_matches("DAPI", names)[0]``; with no close match it warns and
uses index 0, and a close-but-wrong name ("DAPI_X") is taken with only a warning. An ``int``
is returned unchanged. [Read this session from the valis-wsi 1.2.0 wheel on PyPI.] The
pipeline's conversion always writes the nuclear channel FIRST under the samplesheet's name, so
the name match picks the right channel only by accident; REGISTER passes index 0 instead.

bin/register.py imports valis at module scope and CI has no valis, so the structural checks
read its AST. The behavioural check against VALIS's real ``get_channel_index`` runs wherever
valis is importable (the VALIS image) and is skipped elsewhere.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTER_PY = ROOT / "bin" / "register.py"


def _tree() -> ast.Module:
    return ast.parse(REGISTER_PY.read_text())


def _function(name: str) -> ast.FunctionDef:
    return next(
        n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _module_constant(name: str):
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"bin/register.py defines no module-level {name}")


def _register_kwargs() -> dict:
    """Evaluate register_kwargs() from its source, with only the constant it reads bound."""
    fn = _function("register_kwargs")
    namespace = {"REGISTRATION_CHANNEL": _module_constant("REGISTRATION_CHANNEL")}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(REGISTER_PY), "exec"),
        namespace,
    )
    return namespace["register_kwargs"]()


def _calls(fn: ast.FunctionDef, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def test_the_registration_channel_is_the_nuclear_channel_by_index():
    assert _module_constant("REGISTRATION_CHANNEL") == 0
    assert _register_kwargs() == {
        "if_processing_kwargs": {"channel": 0, "adaptive_eq": True}
    }, (
        "register() must be given VALIS's own fluorescence arguments with ONLY the channel "
        "replaced by index 0 -- adaptive_eq=True is VALIS's default and stays."
    )


def test_register_is_called_with_exactly_register_kwargs():
    calls = _calls(_function("valis_registration"), "register")
    assert len(calls) == 1, (
        f"expected one registrar.register(...) call, found {len(calls)}"
    )
    call = calls[0]
    assert not call.args, "registrar.register() takes no positional arguments here"
    assert len(call.keywords) == 1 and call.keywords[0].arg is None, (
        "registrar.register() must be called as register(**register_kwargs()) and nothing else"
    )
    inner = call.keywords[0].value
    assert (
        isinstance(inner, ast.Call)
        and getattr(inner.func, "id", None) == "register_kwargs"
    )


def test_the_registrar_gets_the_slides_and_the_reference_and_nothing_else():
    """At the default mode. The one other thing the constructor can be given is
    constructor_kwargs(valis_mode), which is {} at `default` and the hires example's
    MicroRigidRegistrar at `hires` (tests/test_register_valis_modes.py)."""
    calls = _calls(_function("valis_registration"), "Valis")
    assert len(calls) == 1, (
        f"expected one registration.Valis(...) call, found {len(calls)}"
    )
    kwargs = sorted(str(k.arg) for k in calls[0].keywords if k.arg is not None)
    assert kwargs == ["img_list", "reference_img_f"], (
        f"Valis() is given {kwargs}; every constructor argument except the slides and the "
        "reference must be VALIS's default (no tier sizes, detector, matcher, "
        "align_to_reference); a mode's own argument goes through constructor_kwargs()."
    )
    splats = [ast.unparse(k.value) for k in calls[0].keywords if k.arg is None]
    assert splats == ["constructor_kwargs(valis_mode)"], splats
    assert len(calls[0].args) == 2, "Valis(src_dir, dst_dir, ...) positional arguments"
    default_branch = ast.unparse(_function("constructor_kwargs"))
    assert default_branch.rstrip().endswith("return {}"), (
        "constructor_kwargs() must return {} for the default mode"
    )


def test_the_default_mode_runs_no_micro_registration():
    """register_micro() runs only when register_micro_kwargs() returns kwargs, which it does
    only at `hires`; nothing of the tuned configuration is back."""
    fn = _function("valis_registration")
    assert len(_calls(fn, "register_micro")) == 1
    micro = ast.unparse(_function("register_micro_kwargs"))
    assert "if valis_mode != 'hires':\n        return None" in micro
    for gone in ("SuperGlue", "SuperPoint", "MEMORY_PRESETS"):
        assert gone not in ast.unparse(fn), f"{gone} is back in valis_registration"
    assert "valis_micro_rigid" not in REGISTER_PY.read_text()
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "MicroRigidRegistrar" not in names, (
        "MicroRigidRegistrar must reach Valis() only through constructor_kwargs('hires')"
    )


def test_a_name_match_on_dapi_would_pick_a_decoy_over_the_nuclear_channel():
    """Why the index is passed: what VALIS's default does on a converted slide whose
    nuclear channel is not named DAPI, beside a decoy that is. This is the difflib call
    slide_io.get_channel_index makes (names upper-cased, first close match)."""
    names = [n.upper() for n in ("Hoechst", "CD3", "DAPI_x")]
    best = difflib.get_close_matches("DAPI", names)[0]
    assert names.index(best) == 2, "the name match no longer prefers the decoy"
    assert _register_kwargs()["if_processing_kwargs"]["channel"] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("valis") is None, reason="valis is not installed here"
)
def test_valis_reads_channel_zero_when_given_the_index():
    """VALIS's real get_channel_index, on the decoy slide: the index wins, the name loses."""
    from valis import slide_io

    reader = object.__new__(slide_io.SlideReader)
    reader.src_f = "decoy.ome.tiff"
    reader.metadata = type("M", (), {"channel_names": ["Hoechst", "CD3", "DAPI_x"]})()
    channel = _register_kwargs()["if_processing_kwargs"]["channel"]
    assert reader.get_channel_index(channel) == 0
    assert reader.get_channel_index("dapi") == 2
