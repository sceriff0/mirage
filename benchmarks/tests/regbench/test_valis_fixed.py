"""VALIS 1.2.0 with its micro pass's one-line defect fixed (``methods/valis_fixed.py``).

Four things are pinned here:

* the patch: what it reads (the sdist's method, by sha256), the one statement it inserts,
  and that it refuses a source it was not written for;
* the defect and the fix ON THE REAL METHOD BODY: the vendored 1.2.0 source
  (``fixtures/valis120_prep_images.txt``) is executed against stand-ins of the modules it
  calls, shipped and patched, and the field each slide's micro-pass image is warped with
  is recorded. Shipped: the last slide's for every slide. Patched: each slide's own. The
  first non-rigid pass and the tiler path are unchanged by the patch;
* the run: through ``fake_valis`` the rows are named ``valis120_fixed*``, written where
  the driver puts them, and carry ``info.micro_fix``;
* the harness: own family, no primary, out of every headline table, its own paper table.

VALIS itself is not installed here; ``python -m benchmarks.regbench.methods.valis_fixed
--check`` is the check of the patch on the installed VALIS, inside the image.
"""

import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarks.regbench import METHODS, methods, variants
from benchmarks.regbench import selfcheck as sc
from benchmarks.regbench.methods import valis as valis120
from benchmarks.regbench.methods import valis_fixed as vf
from benchmarks.regbench.paper import schema, tables

from . import fake_valis as fv

FIXTURE = Path(__file__).with_name("fixtures") / "valis120_prep_images.txt"
MARKER = "==== method source ====\n"
FIXED = ("valis120_fixed_rigid", "valis120_fixed", "valis120_fixed_micro")
FIXED_HIRES = (
    "valis120_fixed_hires_rigid",
    "valis120_fixed_hires_nonrigid",
    "valis120_fixed_hires",
)


def vendored():
    return FIXTURE.read_text().split(MARKER, 1)[1]


# ── the patch ─────────────────────────────────────────────────────────────────
def test_the_vendored_method_is_the_one_the_patch_was_written_for():
    assert "Copyright (c) 2021-2025 Chandler Gatenbee" in FIXTURE.read_text()
    assert vf.sha256(vendored()) == vf.SOURCE_SHA256


def test_method_source_reads_the_class_method_back_byte_for_byte(tmp_path):
    body = "".join("    " + line if line.strip() else line
                   for line in vendored().splitlines(keepends=True))
    path = tmp_path / "registration.py"
    path.write_text("import os\n\n\nclass Other:\n    pass\n\n\nclass Valis:\n" + body
                    + "\n    def after(self):\n        return 1\n")
    assert vf.method_source(path) == vendored()


def test_the_patch_inserts_one_statement_and_a_record_after_the_anchor():
    src = vendored()
    out = vf.patched_source(src)
    old, new = src.splitlines(), out.splitlines()
    assert len(new) == len(old) + len(vf.FIX)
    at = next(i for i, line in enumerate(old) if line.strip() == vf.ANCHOR)
    assert new[: at + 1] == old[: at + 1] and new[at + 1 + len(vf.FIX):] == old[at + 1:]
    indent = old[at][: len(old[at]) - len(old[at].lstrip())]
    assert new[at + 1 : at + 1 + len(vf.FIX)] == [indent + s for s in vf.FIX]
    # the statement is the first loop's own derivation of dxdy, nothing else
    assert vf.FIX[0].startswith("dxdy = slide_obj.bk_dxdy if updating_non_rigid else None")
    compile(out, "<patched>", "exec")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace(vf.ANCHOR, "slide_obj = self.get_slide(img_f_list[j])"),
        lambda s: s.replace(vf.ANCHOR, vf.ANCHOR + "\n                " + vf.ANCHOR),
        lambda s: s.replace(vf.FIRST_LOOP, "dxdy = slide_obj.fwd_dxdy"),
        lambda s: s.replace("processed_warped_img = warp_tools.warp_img(img=normed_img",
                            "processed_warped_img = warp_tools.other(img=normed_img"),
    ],
    ids=["no-anchor", "two-anchors", "no-first-loop", "not-the-warp"],
)
def test_a_source_the_fix_was_not_written_for_is_refused(mutate):
    with pytest.raises(RuntimeError):
        vf.patched_source(mutate(vendored()))


# ── the real method body, against stand-ins ───────────────────────────────────
class _Vips:
    def __init__(self, h=8, w=8):
        self.height, self.width = h, w

    def invert(self):
        return self


class _Proc:
    def __init__(self, image, **kw):
        self.image = image

    def process_image(self, **kw):
        return np.asarray(self.image, dtype=float)


def _stubs(seen, tiler):
    """The modules the method calls, as stand-ins: ``seen`` gets one entry per warp of a
    NORMALISED image (the micro-pass image of a slide): ``(M id, bk_dxdy)``."""

    def warp_img(img=None, M=None, bk_dxdy=None, **kw):
        if isinstance(img, np.ndarray) and "interp_method" not in kw:
            seen.append((M["slide"], bk_dxdy))
        return np.zeros((8, 8), np.uint8)

    warp_tools = types.SimpleNamespace(
        calc_memory_size_gb=lambda shape, n, dtype: 1.0,
        get_src_img_shape_and_M=lambda **kw: (np.array([8, 8]), kw["M"]),
        resize_img=lambda img, shape, **kw: img,
        numpy2vips=lambda a: _Vips(),
        vips2numpy=lambda v: np.ones((8, 8)),
        warp_img=warp_img,
        crop_img=lambda img, xywh: img,
        mask2xy=None, xy2bbox=None,
    )
    return {
        "np": np,
        "warp_tools": warp_tools,
        "pyvips": types.SimpleNamespace(Image=types.SimpleNamespace(
            black=lambda w, h: _Vips(h, w))),
        "tqdm": types.SimpleNamespace(tqdm=lambda it, **kw: it),
        "exposure": types.SimpleNamespace(
            rescale_intensity=lambda img, out_range: np.asarray(img, dtype=float)),
        "preprocessing": types.SimpleNamespace(
            collect_img_stats=lambda imgs: (None, None),
            norm_img_stats=lambda img, stats: img,
            match_histograms=lambda img, hist: img),
        "serial_non_rigid": types.SimpleNamespace(
            IMG_LIST_KEY="img_list", IMG_F_LIST_KEY="img_f_list",
            MASK_LIST_KEY="mask_list", IMG_NAME_KEY="name_list"),
        "deepcopy": lambda x: dict(x),
        "slide_tools": types.SimpleNamespace(BF_FORMAT_NUMPY_DTYPE={}),
        "valtils": types.SimpleNamespace(print_warning=print),
        "TILER_THRESH_GB": -1.0 if tiler else 1e9,
        "PREP_NON_RIGID_MSG": "",
    }


class _Slide:
    def __init__(self, name, idx):
        self.name, self.src_f, self.stack_idx = name, f"/x/{name}.ome.tif", idx
        self.M = {"slide": name}  # the warp records whose matrix it was given
        self.bk_dxdy = f"field of {name}"  # each slide's own field, told apart
        self.processed_img_shape_rc = (8, 8)
        self.reg_img_shape_rc = (8, 8)
        self.slide_dimensions_wh = np.array([[16, 16], [8, 8]])
        self.rigid_reg_mask = np.ones((8, 8))
        self.series, self.bg_color, self.is_rgb = 0, 0, False
        self.non_rigid_reg_mask = None
        self.reader = types.SimpleNamespace(metadata=types.SimpleNamespace(
            bf_datatype=None, n_channels=1))

    def slide2vips(self, level):
        return _Vips()

    def warp_img(self, img, **kw):
        return np.ones((8, 8))

    def warp_slide(self, level, **kw):
        return _Vips()


class _Valis:
    norm_method = "img_stats"  # registration.DEFAULT_NORM_METHOD

    def __init__(self, names):
        self.slide_dict = {n: _Slide(n, i) for i, n in enumerate(names)}
        self.size = len(names)

    def get_ref_slide(self):
        return next(iter(self.slide_dict.values()))

    def get_slide(self, src_f):
        return next(s for s in self.slide_dict.values() if s.src_f == src_f)

    def get_aligned_slide_shape(self, level):
        return np.array([8, 8])


NAMES = ("reference", "moving", "moving_2")  # slide_dict order: the last is moving_2


def _run(src, updating, tiler=False):
    seen, record = [], []
    scope = {**_stubs(seen, tiler), vf.RECORD: record}
    names = {}
    exec(compile(src, "<method>", "exec"), scope, names)
    reg = _Valis(NAMES)
    procs = {n: (_Proc, {}) for n in NAMES}
    names[vf.METHOD](reg, None, procs, updating_non_rigid=updating, rgb=False)
    return seen, record


def test_shipped_every_slides_micro_image_is_warped_with_the_last_slides_field():
    """THE DEFECT, on 1.2.0's own code: each slide is warped with its own M but the last
    slide's field, the reference included."""
    seen, _ = _run(vendored(), updating=True)
    assert seen == [(n, "field of moving_2") for n in NAMES]


def test_fixed_every_slides_micro_image_is_warped_with_its_own_field():
    seen, record = _run(vf.patched_source(vendored()), updating=True)
    assert seen == [(n, f"field of {n}") for n in NAMES]
    assert record == [(True, n, True) for n in NAMES]


@pytest.mark.parametrize("patched", [False, True])
def test_the_first_non_rigid_pass_has_no_field_either_way(patched):
    """``non_rigid_register`` passes ``updating_non_rigid=False``: no field in either."""
    src = vf.patched_source(vendored()) if patched else vendored()
    seen, _ = _run(src, updating=False)
    assert seen == [(n, None) for n in NAMES]


def test_on_the_tiler_path_the_normalisation_loop_does_not_run_and_nothing_differs():
    for src in (vendored(), vf.patched_source(vendored())):
        seen, record = _run(src, updating=True, tiler=True)
        assert seen == [] and record == []


def test_build_patches_the_class_for_the_block_only_and_leaves_the_module(tmp_path):
    body = "".join("    " + line if line.strip() else line
                   for line in vendored().splitlines(keepends=True))
    path = tmp_path / "registration_under_test.py"
    path.write_text("class Valis:\n" + body)
    import importlib.util

    spec = importlib.util.spec_from_file_location("registration_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    for k, v in _stubs(seen, tiler=False).items():
        setattr(module, k, v)
    shipped = module.Valis.__dict__[vf.METHOD]
    with vf.installed(module) as (record, how):
        assert how["source_is_sdist"] and how["source_sha256"] == vf.SOURCE_SHA256
        reg = _Valis(NAMES)
        getattr(module.Valis, vf.METHOD)(
            reg, None, {n: (_Proc, {}) for n in NAMES}, updating_non_rigid=True)
        assert seen == [(n, f"field of {n}") for n in NAMES]
        assert [r[1] for r in record] == list(NAMES)
    assert module.Valis.__dict__[vf.METHOD] is shipped
    assert not hasattr(module, vf.RECORD)  # the record lives in a copy of the globals


def test_build_refuses_other_bytes_unless_told(tmp_path):
    src = vendored().replace("# Get mask covering tissue", "# a comment moved")
    body = "".join("    " + line if line.strip() else line
                   for line in src.splitlines(keepends=True))
    path = tmp_path / "registration_other.py"
    path.write_text("class Valis:\n" + body)
    import importlib.util

    spec = importlib.util.spec_from_file_location("registration_other", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="not the source of valis-wsi 1.2.0"):
        vf.build(module)
    _, _, how = vf.build(module, any_version=True)
    assert how["source_is_sdist"] is False


# ── the run, through fake_valis ───────────────────────────────────────────────
@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
    fv.install(monkeypatch.setitem)
    yield fv
    fv.configure()


def _fake_installed(monkeypatch, loop_ran=True):
    """The fake registration has no 1.2.0 source to patch: the record the patched method
    would leave, written when register_micro runs."""
    import contextlib

    record = []
    how = {"source": "fake", "source_sha256": vf.SOURCE_SHA256, "source_is_sdist": True}
    real_micro = fv.Valis.register_micro

    def register_micro(self, **kw):
        record.extend((False, n, False) for n in self.slide_dict)
        if loop_ran:
            record.extend((True, n, True) for n in self.slide_dict)
        return real_micro(self, **kw)

    @contextlib.contextmanager
    def installed(registration, any_version=False):
        yield record, how

    monkeypatch.setattr(fv.Valis, "register_micro", register_micro)
    monkeypatch.setattr(vf, "installed", installed)


PATIENT = ((512, 640), None, None)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return sc.make_pair(tmp_path_factory.mktemp("valis_fixed_pair"), (512, 640))


@pytest.mark.parametrize(
    "label, opts, names",
    [("valis120_fixed", None, FIXED), ("valis120_fixed_hires", {"hires": 1}, FIXED_HIRES)],
)
def test_a_fixed_run_is_valis120s_run_under_its_own_names(fake, tmp_path, monkeypatch,
                                                          label, opts, names):
    _fake_installed(monkeypatch)
    rec = sc.run("valis_fixed", tmp_path / "w", {"workers": 1, **(opts or {})},
                 label=label, sizes=[PATIENT], in_process=True, keep_work=True)
    assert rec["status"] == sc.OK and rec["variants"] == dict.fromkeys(names, sc.OK)
    (built,) = fv.calls("Valis")
    assert sorted(built["kwargs"]) == sorted(
        ["img_list", "reference_img_f", *(["micro_rigid_registrar_cls"] if opts else [])])
    for p in rec["pairs"]:
        for name in names:
            native = p["variants"][name]["native"]
            if native:  # written under the fixed row's own result folder
                assert f"/{name}/" in native["files"][0], native["files"][0]


def _run_json(out, variant, case):
    from benchmarks.regbench.cases import result_dir

    return json.loads((result_dir(out, variant, case) / "run.json").read_text())


@pytest.mark.parametrize("loop_ran", [True, False])
def test_the_micro_row_records_what_the_fix_changed(fake, tmp_path, monkeypatch, pair,
                                                   loop_ran):
    _fake_installed(monkeypatch, loop_ran)
    from benchmarks.tests.regbench.test_valis_adapters import patient

    cases = patient(tmp_path, pair)
    out = tmp_path / "out"
    done = methods.run_group("valis_fixed", cases, out, {"workers": 1},
                             label="valis120_fixed")
    assert all(v == list(FIXED) for v in done.values())
    micro = _run_json(out, "valis120_fixed_micro", cases[0])
    fix = micro["info"]["micro_fix"]
    assert fix["applied"] and fix["source_is_sdist"]
    assert fix["affects_this_row"] is loop_ran and fix["normalisation_loop_ran"] is loop_ran
    last = micro["info"]["last_slide_in_slide_dict"]
    assert fix["last_slide"] == last
    assert last not in fix["slides_given_own_field"]
    assert (len(fix["slides_given_own_field"]) == 2) is loop_ran
    assert micro["info"]["shipped_row"] == "valis120"
    impl = micro["info"]["implementation"]
    assert vf.NOTE in impl["benchmark_side"]
    before = _run_json(out, "valis120_fixed", cases[0])["info"]["micro_fix"]
    assert before["affects_this_row"] is False


def test_the_hires_label_and_option_come_together(fake, tmp_path, monkeypatch, pair):
    _fake_installed(monkeypatch)
    from benchmarks.tests.regbench.test_valis_adapters import patient

    cases = patient(tmp_path, pair, rounds=1)
    done = methods.run_group("valis_fixed", cases, tmp_path / "o", {"workers": 1},
                             label="valis120_fixed_hires")
    assert done == {cases[0].case_id: []}
    assert not fv.calls("Valis")


# ── the harness ───────────────────────────────────────────────────────────────
def test_the_method_and_its_two_runs_are_listed():
    assert "valis_fixed" in METHODS
    assert methods.scope_of("valis_fixed") == "group"
    assert methods.variant_names("valis_fixed", "valis120_fixed") == list(FIXED)
    assert methods.variant_names("valis_fixed", "valis120_fixed_hires") == list(FIXED_HIRES)
    runs = {variants.run_name(r): r for r in variants.RUNS}
    for name, shipped in vf.SHIPPED_ROW.items():
        fixed, base = runs[name], runs[shipped]
        assert fixed.method == "valis_fixed" and base.method == "valis"
        # the same configuration, the same image: one line of VALIS is the difference
        assert (fixed.opts, fixed.image, fixed.gpu, fixed.regimes) == (
            base.opts, base.image, base.gpu, base.regimes)
    assert set(vf.OPTIONS) == set(valis120.OPTIONS)


def test_the_fixed_rows_are_their_own_family_with_no_primary():
    for name in (*FIXED, *FIXED_HIRES):
        assert variants.family(name) == "valis120_fixed"
    assert variants.family("valis120_hires") == "valis120"
    assert variants.primary_of("valis120_fixed") is None
    assert not set(variants.PRIMARY) & {*FIXED, *FIXED_HIRES}
    assert "valis120_fixed" in variants.SENSITIVITY_FAMILIES
    assert set(variants.members("valis120_fixed")) == set(vf.SHIPPED_ROW)


def test_the_fixed_rows_are_expected_on_the_cohort_and_not_on_hyreco_or_synthetic():
    assert {*FIXED, *FIXED_HIRES} <= set(variants.expected_variants())
    for dataset in ("hyreco", "synthetic"):
        names = {variants.run_name(r) for r in variants.runs_for(dataset)}
        assert not names & set(vf.SHIPPED_ROW), dataset


def test_the_fixed_freeze_does_not_move_valis():
    frozen = json.loads((Path(__file__).with_name("method_code_freeze.json")).read_text())
    assert "benchmarks/regbench/methods/valis_fixed.py" not in frozen["valis"]["files"]
    assert "benchmarks/regbench/methods/valis.py" in frozen["valis_fixed"]["files"]


def test_the_sensitivity_family_is_in_no_headline_table():
    assert "valis120_fixed" not in tables._families(tables.COHORT)
    assert "valis120" in tables._families(tables.COHORT)


def _hand(row, case, patient, dice):
    return {"dataset": "multiplex", "set": "outlines", "row": row, "case_id": case,
            "patient_id": patient, "dice_matched": dice, "excluded_from_summaries": False,
            "imputed_initial": False, "displacement_readable": False,
            "displacement_um_p50": float("nan"), "tool_version": "1", "n_pairs": 0}


def test_the_fix_table_pairs_each_fixed_row_with_its_shipped_row(monkeypatch):
    monkeypatch.setattr(tables, "terminal_rows", lambda ds: {
        "stare": "stare", "valis120": "valis120_micro", "valis120_fixed":
        "valis120_fixed_micro"})
    pp = [
        _hand("valis120_micro", "c1", "p1", 0.50), _hand("valis120_fixed_micro", "c1", "p1", 0.60),
        _hand("valis120_micro", "c2", "p1", 0.40), _hand("valis120_fixed_micro", "c2", "p1", 0.40),
        _hand("valis120_micro", "c3", "p2", 0.30), _hand("valis120_fixed_micro", "c3", "p2", 0.50),
        _hand("valis120_micro", "c4", "p2", 0.90),  # the fixed run has no c4: not paired
        _hand("stare", "c1", "p1", 0.70), _hand("stare", "c2", "p1", 0.70),
        _hand("stare", "c3", "p2", 0.70),
    ]
    best = [{"family": "stare", "best_row": "stare", "set": "outlines",
             "pooling": tables.HEADLINE_POOLING}]
    rows = tables.valis120_fix(pp, "multiplex", best)
    (case,) = [r for r in rows if r["set"] == "outlines" and r["pooling"] == "case"]
    assert case["row"] == "valis120_fixed_micro" and case["shipped_row"] == "valis120_micro"
    assert case["n_cases_paired"] == 3 and case["n_units"] == 3
    assert case["dice_med_shipped"] == pytest.approx(0.40)
    assert case["dice_med_fixed"] == pytest.approx(0.50)
    assert case["d_dice_fixed_minus_shipped_med"] == pytest.approx(0.10)
    assert case["n_units_shipped_ge"] == 1  # c2: a tie
    assert case["stare_row"] == "stare"
    assert case["d_dice_shipped_vs_stare_med"] == pytest.approx(-0.30)
    assert case["d_dice_fixed_vs_stare_med"] == pytest.approx(-0.20)
    assert set(schema.columns("valis120_fix")) >= set(case)
    nuclei = [r for r in rows if r["set"] == "nuclei"]
    assert nuclei and all(r["n_cases_paired"] == 0 for r in nuclei)
    assert all(math.isnan(r["d_dice_fixed_minus_shipped_med"]) for r in nuclei)


def test_the_fix_table_is_written():
    assert "valis120_fix" in tables.WRITTEN and "valis120_fix" in schema.TABLES
