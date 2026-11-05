"""reg_valis_mode: VALIS's own two configurations, `default` and `hires`, and nothing else.

Ruling 2026-11-05: VALIS in the pipeline is valis-wsi 1.2.0's own default call, or its own
high-resolution example (``docs/examples.rst:122-149`` at v1.2.0, the benchmark's
``valis120_hires`` row in ``benchmarks/regbench/methods/valis.py`` on ``regbench-paper``), AS
SHIPPED, with no fix for its micro-pass defect. ``hires`` differs from ``default`` in exactly
three things:

1. ``micro_rigid_registrar_cls=MicroRigidRegistrar`` in the ``Valis(...)`` constructor;
2. ``register()`` (unchanged, channel 0 by index);
3. ``register_micro(max_non_rigid_registration_dim_px=micro_reg_size)``, with
   ``micro_reg_size = floor(min over slides of max(w, h) * 0.25)`` computed as the example does,
   and the same channel kwarg as ``register()``.

bin/register.py imports valis at module scope and CI has no valis, so its helpers are lifted
out of the real source by AST and run with VALIS stubbed (a stand-in ``MicroRigidRegistrar``
class). The wiring inside ``valis_registration`` is read from its AST, and the Nextflow side
(the param, REGISTER's rendered flag and memory request, WARP_SEG_QC's micro depth) through
tests.nfmodel.
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pytest

from tests.nfmodel import (
    REPO_ROOT,
    block_extent,
    processes,
    strip_comments,
    with_name_blocks,
)

REGISTER_PY = REPO_ROOT / "bin" / "register.py"
HELPERS = ("_check_mode", "micro_reg_size", "constructor_kwargs", "register_kwargs",
           "register_micro_kwargs")


class MicroRigidRegistrar:
    """Stand-in for valis.micro_rigid_registrar.MicroRigidRegistrar (VALIS is not installed)."""


def _tree() -> ast.Module:
    return ast.parse(REGISTER_PY.read_text())


def _module_constant(name: str):
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"bin/register.py defines no module-level {name}")


@pytest.fixture(scope="module")
def reg():
    """The helpers, exec'd from the CURRENT source with VALIS stubbed."""
    fns = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {f.name for f in fns} == set(HELPERS), "a helper is missing from bin/register.py"
    ns = {
        "np": np,
        "Optional": object,
        "MicroRigidRegistrar": MicroRigidRegistrar,
        **{c: _module_constant(c) for c in ("REGISTRATION_CHANNEL", "VALIS_MODES",
                                            "MICRO_REG_FRACTION")},
    }
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(REGISTER_PY), "exec"), ns)
    return type("Reg", (), {k: staticmethod(v) if callable(v) else v for k, v in ns.items()})


def _dims(*wh):
    """One Slide.slide_dimensions_wh per (w, h): level 0 first, then a half-size level."""
    return [np.array([[w, h], [w // 2, h // 2]]) for w, h in wh]


# --------------------------------------------------------------------------- micro_reg_size


def test_the_fraction_is_the_examples_quarter():
    assert _module_constant("MICRO_REG_FRACTION") == 0.25


@pytest.mark.parametrize(
    "slides, want",
    [
        # min over slides of the LONGEST side: 40000 (slide 1) vs 36000 (slide 2) -> 9000
        (((40000, 30000), (20000, 36000)), 9000),
        # floor, not round: 0.25 * 10003 = 2500.75 -> 2500
        (((10003, 9000), (12000, 12000)), 2500),
        # one slide alone
        (((8192, 4096),), 2048),
        # level 0 only: the half-size level must not be the one measured
        (((1000, 1000), (50000, 50000)), 250),
    ],
)
def test_micro_reg_size_is_the_examples_formula(reg, slides, want):
    got = reg.micro_reg_size(_dims(*slides))
    assert got == want and isinstance(got, int)
    # the example's own statements, verbatim, give the same number
    img_dims = np.array([d[0] for d in _dims(*slides)])
    min_max_size = np.min([np.max(d) for d in img_dims])
    assert got == np.floor(min_max_size * 0.25).astype(int)


# --------------------------------------------------------------------------- kwargs per mode


def test_the_modes_are_valis_own_two(reg):
    assert reg.VALIS_MODES == ("default", "hires")


def test_default_builds_no_constructor_kwargs_and_no_micro_pass(reg):
    assert reg.constructor_kwargs("default") == {}
    assert reg.register_micro_kwargs("default", _dims((40000, 30000))) is None


def test_hires_builds_the_examples_constructor_kwargs(reg):
    assert reg.constructor_kwargs("hires") == {"micro_rigid_registrar_cls": MicroRigidRegistrar}


def test_hires_builds_the_examples_register_micro_kwargs(reg):
    kwargs = reg.register_micro_kwargs("hires", _dims((40000, 30000), (20000, 36000)))
    assert kwargs == {
        "max_non_rigid_registration_dim_px": 9000,
        # the same registration channel register() gets, nothing else
        "if_processing_kwargs": {"channel": 0, "adaptive_eq": True},
    }
    assert kwargs["if_processing_kwargs"] == reg.register_kwargs()["if_processing_kwargs"]


def test_register_kwargs_are_the_same_in_both_modes(reg):
    # register() takes no mode: both modes call it with exactly register_kwargs()
    assert reg.register_kwargs() == {"if_processing_kwargs": {"channel": 0, "adaptive_eq": True}}


@pytest.mark.parametrize("bad", ["", "high", "HIRES", None])
def test_an_unknown_mode_is_refused(reg, bad):
    with pytest.raises(ValueError):
        reg.constructor_kwargs(bad)
    with pytest.raises(ValueError):
        reg.register_micro_kwargs(bad, _dims((1000, 1000)))


# --------------------------------------------------------------------------- valis_registration wiring


def _valis_registration() -> ast.FunctionDef:
    return next(
        n for n in _tree().body
        if isinstance(n, ast.FunctionDef) and n.name == "valis_registration"
    )


def _calls(attr: str, fn: ast.FunctionDef | None = None) -> list[ast.Call]:
    return [
        n for n in ast.walk(fn or _valis_registration())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


def test_the_constructor_gets_the_modes_kwargs():
    (call,) = _calls("Valis")
    named = sorted(str(k.arg) for k in call.keywords if k.arg)
    splats = [ast.unparse(k.value) for k in call.keywords if k.arg is None]
    assert named == ["img_list", "reference_img_f"]
    assert splats == ["constructor_kwargs(valis_mode)"]


def test_register_micro_runs_only_with_the_modes_kwargs():
    fn = _valis_registration()
    (call,) = _calls("register_micro", fn)
    assert not call.args
    assert [ast.unparse(k.value) for k in call.keywords] == ["micro_kwargs"]
    assert [k.arg for k in call.keywords] == [None]
    guarded = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "micro_kwargs is not None"
        and any(c is call for c in ast.walk(n))
    ]
    assert guarded, "register_micro() must run only under `if micro_kwargs is not None`"
    src = ast.unparse(_valis_registration())
    assert "micro_kwargs = register_micro_kwargs(valis_mode" in src


def test_the_checkpoint_says_whether_a_micro_pass_follows():
    """Written BEFORE register_micro (the only instant the pre-micro field exists), flagged
    by the mode, and corrected after the fact when register_micro was a no-op."""
    fn = _valis_registration()
    (ckpt,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "write_checkpoint"]
    flag = {k.arg: ast.unparse(k.value) for k in ckpt.keywords}["micro_registration"]
    assert flag == "micro_kwargs is not None"
    (micro,) = _calls("register_micro", fn)
    assert ckpt.lineno < micro.lineno, "the checkpoint must be taken before register_micro()"
    fixes = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "set_micro_registration"]
    assert len(fixes) == 1 and fixes[0].lineno > micro.lineno
    assert ast.unparse(fixes[0].args[1]) == "micro_ran"


def test_the_cli_offers_exactly_the_modes():
    parse = next(n for n in _tree().body
                 if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    (opt,) = [n for n in ast.walk(parse) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", None) == "add_argument"
              and n.args and getattr(n.args[0], "value", None) == "--valis-mode"]
    kw = {k.arg: ast.unparse(k.value) for k in opt.keywords}
    assert kw["choices"] == "VALIS_MODES" and kw["default"] == "'default'"


# --------------------------------------------------------------------------- the Nextflow side


def test_the_param_and_schema_offer_the_same_two_modes():
    config = (REPO_ROOT / "nextflow.config").read_text()
    m = re.search(r"^\s*reg_valis_mode\s*=\s*'(\w+)'", config, re.M)
    assert m and m.group(1) == "default"
    schema = json.loads((REPO_ROOT / "nextflow_schema.json").read_text())
    groups = schema.get("$defs") or schema["definitions"]
    entry = groups["registration_options"]["properties"]["reg_valis_mode"]
    assert entry["enum"] == list(_module_constant("VALIS_MODES"))
    assert entry["default"] == "default"


def test_register_renders_the_flag_only_at_hires_from_an_itemised_param():
    script = strip_comments(processes()["REGISTER"].script_body)
    assert "params.reg_valis_mode == 'hires'" in script
    assert "'--valis-mode hires'" in script
    assert not re.search(r"[^.\w]params[^.\w]", script.replace("params.", "")), (
        "REGISTER's script: must not reference the bare `params` map (it hashes the whole map)"
    )


def _register_memory_closure() -> str:
    (block,) = [b for b in with_name_blocks() if "REGISTER" in b.names]
    m = re.search(r"(?m)^\s*memory\s*=\s*\{", block.body)
    assert m, "REGISTER's withName block has no memory closure"
    end = block_extent(block.body, m.end())
    return strip_comments(block.raw_body[m.end(): end]).strip().removesuffix("}")


def test_register_memory_is_sized_per_mode_on_one_ramp():
    body = " ".join(_register_memory_closure().split())
    assert body == (
        "(params.reg_valis_mode == 'hires' ? 128.GB : 64.GB) * (2 ** (task.attempt - 1))"
    ), body


@pytest.mark.parametrize(
    "mode, attempts",
    [("default", (64, 128, 256, 512)), ("hires", (128, 256, 512, 1024))],
)
def test_register_memory_ramp_values(mode, attempts):
    """Evaluate the closure's arithmetic for each attempt (GB). The hires first rung must
    cover the cohort's measured max peak RSS (106.4 GB); default's must stay 64 (51.4 max)."""
    body = " ".join(_register_memory_closure().split())
    expr = (body.replace("params.reg_valis_mode", repr(mode))
                .replace(".GB", "").replace("2 ** ", "2**"))
    m = re.fullmatch(r"\('(\w+)' == 'hires' \? (\d+) : (\d+)\) \* \(2\*\*\(task\.attempt - 1\)\)",
                     expr)
    assert m, expr
    base = int(m.group(2)) if m.group(1) == "hires" else int(m.group(3))
    assert tuple(base * 2 ** (a - 1) for a in (1, 2, 3, 4)) == attempts
    assert attempts[0] > {"default": 51.4, "hires": 106.4}[mode]


def test_warp_seg_qc_states_the_micro_depth_of_the_mode():
    proc = processes()["WARP_SEG_QC"]
    for section in (proc.script_body, proc.stub_body):
        assert "params.reg_valis_mode == 'hires' ? 2 : 0" in strip_comments(section)
