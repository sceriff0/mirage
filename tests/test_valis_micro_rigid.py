"""VALIS's micro-rigid pass on a TMA core, and the guard that keeps a failed pair rigid.

The chain (bin/utils/valis_micro_rigid.py has it in full): ``MicroRigidRegistrar.
align_slides`` tiles the tissue ROI, drops every tile with fewer than three
SuperPoint/SuperGlue matches, and stacks the survivors with ``np.vstack``. A TMA core's
ROI is about 79x161 px, so every tile is dropped, ``np.vstack([])`` raises ``need at
least one array to concatenate``, and ``Valis.register()`` turns that into a dead JVM
for the whole patient (2026-09-24, patient 005, all four attempts).

The fake below fails the same way (a real ``np.vstack`` on an empty list) and, like
VALIS, writes to the slide only after everything that can raise.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from utils import valis_micro_rigid
from utils.valis_micro_rigid import FALLBACKS, guard_micro_rigid, is_guarded

ROOT = Path(__file__).resolve().parent.parent
LOW_REZ_M = np.eye(3)
REFINED_M = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.5], [0.0, 0.0, 1.0]])


class _FakeMicroRigidRegistrar:
    """Matches ``align_slides`` in VALIS 1.0.0: signature, failure, where it writes."""

    def __init__(self, matches_per_pair):
        # {moving slide name: list of per-tile match arrays, None = tile dropped}
        self.matches_per_pair = matches_per_pair

    def align_slides(self, moving_slide, fixed_slide, processor_dict, mask=None):
        tiles = self.matches_per_pair[moving_slide.name]
        kept = [xy for xy in tiles if xy is not None]
        np.vstack(kept)  # micro_rigid_registrar.py:304 -- raises on []
        moving_slide.M = REFINED_M  # the only write, after everything that can raise
        return "aligned"

    def register(self, pairs):
        for moving, fixed in pairs:
            self.align_slides(moving, fixed, processor_dict={})


@pytest.fixture
def registrar_cls():
    # A fresh class per test: the guard patches the class, and tests must not share it.
    return type("MicroRigidRegistrar", (_FakeMicroRigidRegistrar,), {})


@pytest.fixture(autouse=True)
def _empty_fallbacks():
    FALLBACKS.clear()
    yield
    FALLBACKS.clear()


def _slide(name):
    return SimpleNamespace(name=name, M=LOW_REZ_M.copy())


TMA_CORE = [None, None, None, None]  # every 64x80 tile dropped
WHOLE_SLIDE = [np.ones((5, 2)), None, np.ones((7, 2))]


def test_the_unguarded_pass_crashes_on_a_core_with_no_surviving_tile(registrar_cls):
    """The fake has teeth: without the guard it fails with VALIS's exact message."""
    reg = registrar_cls({"001": TMA_CORE})
    with pytest.raises(ValueError, match="need at least one array to concatenate"):
        reg.align_slides(_slide("001"), _slide("000"), processor_dict={})


def test_a_failed_pair_keeps_its_low_resolution_transform(registrar_cls):
    assert guard_micro_rigid(registrar_cls) is True
    moving = _slide("001")
    result = registrar_cls({"001": TMA_CORE}).align_slides(
        moving, _slide("000"), processor_dict={}
    )
    assert result is None
    np.testing.assert_array_equal(moving.M, LOW_REZ_M)


def test_a_failed_pair_is_recorded_for_the_task_log(registrar_cls, capsys):
    guard_micro_rigid(registrar_cls)
    registrar_cls({"001": TMA_CORE}).align_slides(
        _slide("001"), _slide("000"), processor_dict={}
    )
    assert FALLBACKS == [
        ("001", "000", "ValueError: need at least one array to concatenate")
    ]
    out = capsys.readouterr().out
    assert "001 -> 000" in out and "Keeping low rez registration parameters" in out


def test_a_pair_that_matches_is_refined_exactly_as_before(registrar_cls):
    """The guard may only change pairs that would have crashed."""
    guard_micro_rigid(registrar_cls)
    moving = _slide("001")
    result = registrar_cls({"001": WHOLE_SLIDE}).align_slides(
        moving, _slide("000"), processor_dict={}, mask="m"
    )
    assert result == "aligned"
    np.testing.assert_array_equal(moving.M, REFINED_M)
    assert FALLBACKS == []


def test_one_failed_pair_does_not_stop_the_other_pairs(registrar_cls):
    """The 2026-09-24 log: pair 001 fell back inside VALIS, pair 002 crashed the patient.
    With the guard each pair is decided on its own."""
    guard_micro_rigid(registrar_cls)
    ref, s1, s2, s3 = _slide("000"), _slide("001"), _slide("002"), _slide("003")
    reg = registrar_cls({"001": WHOLE_SLIDE, "002": TMA_CORE, "003": WHOLE_SLIDE})
    reg.register([(s1, ref), (s2, ref), (s3, ref)])
    np.testing.assert_array_equal(s1.M, REFINED_M)
    np.testing.assert_array_equal(s2.M, LOW_REZ_M)
    np.testing.assert_array_equal(s3.M, REFINED_M)
    assert [f[0] for f in FALLBACKS] == ["002"]


def test_memory_error_still_fails_the_task(registrar_cls):
    """An OOM belongs to the retry ramp; quietly degrading the registration would hide it."""

    def oom(self, moving_slide, fixed_slide, processor_dict, mask=None):
        raise MemoryError("tile stack")

    registrar_cls.align_slides = oom
    guard_micro_rigid(registrar_cls)
    with pytest.raises(MemoryError):
        registrar_cls({}).align_slides(_slide("001"), _slide("000"), processor_dict={})
    assert FALLBACKS == []


@pytest.mark.parametrize(
    "exc", [IndexError("ransac"), RuntimeError("superglue"), ZeroDivisionError()]
)
def test_any_other_failure_inside_the_pass_falls_back(registrar_cls, exc):
    """Fewer than three matches can also die later, in RANSAC/Tukey or the transform
    estimate; all of it happens before the slide is written."""

    def boom(self, moving_slide, fixed_slide, processor_dict, mask=None):
        raise exc

    registrar_cls.align_slides = boom
    guard_micro_rigid(registrar_cls)
    moving = _slide("001")
    assert (
        registrar_cls({}).align_slides(moving, _slide("000"), processor_dict={}) is None
    )
    np.testing.assert_array_equal(moving.M, LOW_REZ_M)


def test_the_guard_is_idempotent(registrar_cls):
    assert guard_micro_rigid(registrar_cls) is True
    wrapped = registrar_cls.align_slides
    assert guard_micro_rigid(registrar_cls) is False
    assert registrar_cls.align_slides is wrapped
    assert is_guarded(registrar_cls)


def test_the_guard_keeps_the_method_name(registrar_cls):
    guard_micro_rigid(registrar_cls)
    assert registrar_cls.align_slides.__name__ == "align_slides"


@pytest.mark.parametrize(
    "body",
    [
        {},  # no align_slides at all
        {"align_slides": lambda self, fixed_slide, moving_slide: None},  # reordered
        {"align_slides": lambda self, pair: None},  # reshaped
    ],
)
def test_a_changed_valis_is_left_alone_and_reported_unguarded(body):
    """A VALIS upgrade that moves the method must not get a guard rewriting its
    arguments; the caller sees False and not-guarded, and logs the NOT-applied warning."""
    cls = type("MicroRigidRegistrar", (), body)
    assert guard_micro_rigid(cls) is False
    assert not is_guarded(cls)


def test_the_module_imports_without_valis():
    """It is installed on the git checkout, not in the image: it must not need valis."""
    tree = ast.parse(Path(valis_micro_rigid.__file__).read_text())
    imported = {
        (n.module or "").split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
    } | {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for a in n.names
    }
    assert "valis" not in imported


def test_register_py_guards_the_registrar_before_it_builds_it():
    src = (ROOT / "bin" / "register.py").read_text()
    guard = src.index("valis_micro_rigid.guard_micro_rigid(")
    build = src.index("registration.Valis(")
    assert guard < build, "the guard is applied AFTER the registrar is built"


def test_register_py_logs_when_the_guard_did_not_reach_valis():
    src = (ROOT / "bin" / "register.py").read_text()
    assert "Micro-rigid guard NOT applied" in src
    assert "for moving, fixed, err in valis_micro_rigid.FALLBACKS" in src


VALIS_SRC = ROOT / "valis_lib" / "micro_rigid_registrar.py"


@pytest.mark.skipif(
    not VALIS_SRC.exists(), reason="valis_lib/ is gitignored; a local reference copy"
)
def test_real_valis_writes_the_slide_only_after_everything_that_can_raise():
    """The safety argument, checked against VALIS's own source: in align_slides every
    ``moving_slide.<attr> = ...`` sits after the last ``np.vstack`` and the last
    ``estimate`` call, so an exception always leaves the slide as the rigid stage left it."""
    tree = ast.parse(VALIS_SRC.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "align_slides"
    )
    assert [a.arg for a in fn.args.args[:3]] == list(valis_micro_rigid.EXPECTED_PARAMS)
    writes = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name)
        and t.value.id == "moving_slide"
    ]
    risky = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr
        in {"vstack", "estimate", "filter_matches_ransac", "filter_matches_tukey"}
    ]
    assert writes and risky
    assert min(writes) > max(risky)
