"""Guards for the REAL-SAMPLE arm plan (benchmarks/build_arm_plan.py + arms.yaml).

The sweep's guards (test_build_run_plan.py) tie sweep.yaml to nextflow.config.
These tie arms.yaml to two things the sweep does not touch: the pipeline's own
param names, and the CONSUMER contract in ihc_method/code/registration_arms.R.
A drift on either side fails silently in the same way — a page that renders
cleanly with the wrong answer — so each is pinned here rather than reviewed.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.build_arm_plan import (
    QC_INSTRUMENTS,
    arms_manifest_rows,
    build_arm_plan,
    read_input_patients,
    schema_enums,
    validate_against_schema,
)

BENCH = Path(__file__).parents[1]
REPO_ROOT = BENCH.parent
ARMS_YAML = BENCH / "configs" / "arms.yaml"


def _param_checker():
    path = REPO_ROOT / "tests" / "check_param_consistency.py"
    # That module imports its sibling `_code_view` flat; loading a file by path does
    # NOT put its directory on sys.path, so the sibling import raises
    # ModuleNotFoundError before a single assertion runs. Same fix as
    # test_build_run_plan.py's loader.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_check_param_consistency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(ARMS_YAML.read_text())


@pytest.fixture(scope="module")
def plan(cfg) -> list[dict]:
    return build_arm_plan(cfg)


# ---------------------------------------------------------------------------
# The pipeline contract
# ---------------------------------------------------------------------------


def test_arms_baseline_params_all_exist_in_the_pipeline(cfg):
    """Every param arms.yaml pins must be a real nextflow.config param.

    The same failure the sweep's baseline guard closes: a renamed or removed
    param leaves an arm pinning a value the pipeline never reads, so the run
    costs full price and measures the default instead of the arm.
    """
    defaults = _param_checker().extract_config_defaults(
        (REPO_ROOT / "nextflow.config").read_text()
    )
    unknown = sorted(set(cfg["baseline"]) - set(defaults))
    assert not unknown, (
        f"arms.yaml baseline pins params absent from nextflow.config: {unknown}"
    )


from benchmarks.tests.test_build_run_plan import stare_preset_modes  # noqa: E402


def test_arms_baseline_values_match_the_pipeline_defaults(cfg):
    """The baseline must BE the shipped config, not merely name real params.

    test_arms_baseline_params_all_exist_in_the_pipeline above checks NAMES only, and that
    is not enough: a value that silently diverges from nextflow.config makes the segmentation
    and compute-profile arms measure the cost and quality of a configuration nobody ships,
    while the arms.csv column still says it was the baseline. This is the same guard
    sweep.yaml has had (test_project_sweep_baseline_matches_pipeline_defaults) and arms.yaml
    did not -- and the hole was live: dev moved reg_micro_reg 2 -> 1 (:wrench: 25a232e) and
    nothing here noticed.

    A DELIBERATE divergence is legitimate, but it has to be declared here with its reason,
    not left to look like drift.
    """
    defaults = _param_checker().extract_config_defaults(
        (REPO_ROOT / "nextflow.config").read_text()
    )
    # param -> why arms.yaml deliberately pins something other than the shipped default.
    DELIBERATE = {}
    drift = []
    for k, v in cfg["baseline"].items():
        if k in DELIBERATE or k not in defaults:
            continue
        if str(defaults[k]) != str(v):
            drift.append(f"{k}: arms.yaml={v!r} but nextflow.config={defaults[k]!r}")
    assert not drift, (
        "arms.yaml baseline has drifted from the shipped config:\n  "
        + "\n  ".join(drift)
        + "\nEither track the pipeline, or add the param to DELIBERATE with its reason."
    )


def test_every_flag_run_arms_passes_is_a_real_pipeline_param():
    """The flags run_arms.sh actually passes must all exist in nextflow.config.

    Derived from the script's own `add_param <name>` lines rather than from the
    plan's columns: the plan also carries bookkeeping columns (arm, backend,
    from_arm) that are never passed as flags, so checking columns would either
    fail on those or need an exclusion list that silently grows. Parsing the
    launcher keeps the assertion pinned to what is really sent to Nextflow.
    """
    defaults = _param_checker().extract_config_defaults(
        (REPO_ROOT / "nextflow.config").read_text()
    )
    script = (BENCH / "run_arms.sh").read_text()
    flags = set(re.findall(r"^\s*add_param\s+(\w+)", script, re.M))
    assert flags, "no add_param calls found — did run_arms.sh change shape?"
    unknown = sorted(flags - set(defaults))
    assert not unknown, (
        f"run_arms.sh passes --flags absent from nextflow.config: {unknown}"
    )


def test_every_flag_run_arms_passes_is_carried_by_the_plan(plan):
    """A flag the launcher reads but the plan never emits is always blank, so the
    arm silently runs at the pipeline default instead of the configuration named."""
    script = (BENCH / "run_arms.sh").read_text()
    flags = set(re.findall(r"^\s*add_param\s+(\w+)", script, re.M))
    columns = {k for r in plan for k in r}
    missing = sorted(flags - columns)
    assert not missing, (
        f"run_arms.sh reads plan columns that build_arm_plan never writes: {missing}"
    )


def _schema_enums() -> dict:
    """Delegates to the shipped implementation -- a second copy here could agree
    with itself while disagreeing with what build_arm_plan actually enforces."""
    return schema_enums(REPO_ROOT / "nextflow_schema.json")


def test_every_arm_value_satisfies_the_schema_enum(plan):
    """Checking param NAMES is not enough -- this is the bug that shipped.

    arms.yaml pinned seg_method: instanseg (one 't'). The name check passed, the
    plan built, the launcher ran, and every one of the 14 runs died at
    validateParameters() with "Expected any of [[stardist, instantseg, cellsam]]"
    -- after the job had been queued and scheduled. The typo is especially easy
    here because `instanseg_model_dir` IS spelled with one 't'.

    An enum is exactly the kind of thing a plan can get wrong statically, so it is
    checked statically.
    """
    enums = _schema_enums()
    bad = []
    for r in plan:
        for k, v in r.items():
            if k in enums and v not in ("", None) and v not in enums[k]:
                bad.append(f"{r['arm']}: {k}={v!r} not in {enums[k]}")
    assert not bad, "arm values rejected by nextflow_schema.json:\n  " + "\n  ".join(
        bad
    )


def test_qc_segmenter_cross_values_are_real_backends(cfg):
    """Same check at the source, so the message points at arms.yaml rather than a
    generated row."""
    enums = _schema_enums()
    allowed = enums["seg_method"]
    for key, path in (
        ("qc_segmenter_cross", ("qc_segmenter_cross", "seg_method")),
        ("segmentation_arms", ("segmentation_arms", "seg_method")),
    ):
        node = cfg
        for step in path:
            node = node.get(step, {}) if isinstance(node, dict) else {}
        for m in node or []:
            assert m in allowed, (
                f"arms.yaml {key}.seg_method has {m!r}; allowed: {allowed}"
            )
    assert cfg["baseline"]["seg_method"] in allowed


def test_registration_arms_all_pin_reg_qc_2(plan):
    """reg_qc=2 is what emits the staged QC the whole arm ranking reads.

    An arm at reg_qc<2 does not fail — it produces zero accuracy rows, and the
    consumer renders an empty page with no error.
    """
    reg = [r for r in plan if r["arm_kind"] == "registration"]
    assert reg, "no registration arms in the plan"
    assert all(r["reg_qc"] == 2 for r in reg)


def test_baseline_reg_qc_must_be_2(cfg):
    bad = dict(cfg, baseline=dict(cfg["baseline"], reg_qc=1))
    with pytest.raises(ValueError, match="reg_qc must be 2"):
        build_arm_plan(bad)


# ---------------------------------------------------------------------------
# The backend contract — VALIS-only params must not appear on a tiled arm
# ---------------------------------------------------------------------------


def test_tiled_arm_carries_no_valis_only_params(plan):
    """memory_mode / reg_micro_reg do not exist on the STARE backend.

    Writing a value there would (a) pass `--memory_mode X` to a run that ignores
    it and (b) tell the consumer the tiled arm sat in a cell of the preset x
    depth grid, which is exactly the "one point of comparison, not a seventh
    cell" distinction registration_arms.R is built around.
    """
    tiled = [r for r in plan if r.get("backend") == "tiled"]
    assert tiled, "the tiled comparator arm is missing from the plan"
    for r in tiled:
        assert r["memory_mode"] == ""
        assert r["reg_micro_reg"] == ""


def test_no_ashlar_arms_reach_the_pipeline_plan(plan):
    """ASHLAR IS A BENCHMARK COMPARATOR, NOT A PIPELINE BACKEND, and this plan drives the
    pipeline.

    Replaces three tests that asserted the opposite -- that ashlar arms EXIST here, differ
    only by tile_size, and carry blank VALIS-only columns. Those encoded the arrangement in
    which ashlar was a third registration_method. :fire: 6a54479 removed that for v1.0.0:
    the schema enum is ['valis', 'tiled'] and tests/test_ashlar_backend_removed.py keeps it
    out. An ashlar row here would emit --registration_method ashlar and be rejected at
    launch, so this is now the assertion with teeth in the same place.

    This does NOT say ashlar is un-benchmarked. It IS a row of this plan -- under
    arm_kind='external', dispatched by run_arms.sh to benchmarks/run_ashlar_arm.sh, never
    to Nextflow. So the property with teeth is the narrower one: no ashlar row may carry a
    value that run_pass() would forward as a --flag.

    Asserting the old, broader "no ashlar row exists" would now fail on a correct plan,
    and weakening it to nothing would let a genuine `--registration_method ashlar`
    regression through. The forwarded-column list is restated here rather than imported,
    on purpose: if someone adds a column to run_pass()'s add_param calls without adding it
    here, that is a gap this test SHOULD be updated for, not one it should silently absorb.
    """
    FORWARDED = (
        "start",
        "stop",
        "seg_method",
        "reg_qc",
        "registration_method",
        "memory_mode",
        "reg_micro_reg",
        "reg_tiled_mode",
    )
    ashlar_rows = [r for r in plan if r.get("backend") == "ashlar"]
    assert ashlar_rows, (
        "no ashlar arm in the plan at all -- external_baseline.ashlar.enabled is the "
        "switch, and losing the only external comparator silently is exactly the "
        "'green while proving nothing' failure this file guards"
    )

    for r in ashlar_rows:
        assert r["arm_kind"] == "external", (
            f"ashlar arm {r['arm']!r} is arm_kind={r['arm_kind']!r}; only 'external' is "
            "dispatched away from Nextflow, so any other kind would be launched as a "
            "pipeline run and rejected by validateParameters()"
        )
        set_params = {k: r[k] for k in FORWARDED if str(r.get(k, "")).strip()}
        assert not set_params, (
            f"ashlar arm {r['arm']!r} carries pipeline params {set_params} -- run_arms.sh "
            "forwards these as --flags and the pipeline would reject the run"
        )

    # And the inverse: no ashlar row may sit in the kinds that ARE launched.
    launched = [
        r["arm"]
        for r in plan
        if r["arm_kind"] != "external"
        and ("ashlar" in r["arm"] or r.get("registration_method") == "ashlar")
    ]
    assert not launched, (
        f"ashlar reached a LAUNCHED arm kind, which cannot run it: {launched}"
    )


def test_the_ashlar_comparator_still_has_a_driver():
    """The other half of the test above, and the reason it is safe.

    Removing ashlar from the pipeline plan is only correct while the comparator is still
    Keeping ashlar OUT of the pipeline launch path is only correct while the comparator is
    still driven SOMEWHERE -- otherwise the two tests together would happily certify a
    benchmark that quietly lost its external baseline, which is exactly the "green while
    proving nothing" failure this repo keeps hitting. So assert the driver exists and still
    emits STARE's manifest shape, which is what makes ashlar comparable at all -- and that
    the arm runner that now scores it exists and reaches the pipeline's own scorer.
    """
    import benchmarks.ashlar.solve as solve

    assert hasattr(solve, "main"), "the ashlar comparator driver lost its entry point"
    src = (Path(__file__).parents[1] / "ashlar" / "solve.py").read_text()
    assert "build_manifest" in src, (
        "benchmarks/ashlar/solve.py no longer builds a STARE-shaped manifest; without it "
        "warp_seg_qc.py --method tiled cannot read ashlar and the comparison silently "
        "changes metric family"
    )

    runner = Path(__file__).parents[1] / "run_ashlar_arm.sh"
    assert runner.exists(), (
        "benchmarks/run_ashlar_arm.sh is gone; the external arm has no runner, so "
        "arm_kind='external' rows would be planned and never scored"
    )
    body = runner.read_text()
    # The three legs. Asserted by NAME because each one silently degrades rather than
    # erroring if dropped: no retile -> solve reads nothing; no solve -> no manifest;
    # no warp_seg_qc -> the arm dir exists, is empty, and the consumer renders a gap.
    for needed in (
        "benchmarks.ashlar.retile",
        "benchmarks.ashlar.solve",
        "warp_seg_qc.py",
        "--method tiled",
    ):
        assert needed in body, f"run_ashlar_arm.sh no longer invokes {needed!r}"


def test_valis_arms_are_preset_x_depth_not_a_grid_of_equals(plan):
    """reg_micro_reg is a DEPTH (0/1/2), so 3 presets x 3 depths = 9 arms.

    Was 6 (two presets). `medium` was added alongside STARE's three tiers so both backends
    span three cost/accuracy presets and neither is the tuned one -- the arms-side answer to
    the same fairness question test_project_stare_resolution_axis_mirrors_the_valis_one asks
    of the sweep. Asserting the PRODUCT, not just the count, is the point: the failure this
    catches is someone reading preset x depth as a 2x2 and quietly dropping a depth.
    """
    valis = [
        r
        for r in plan
        if r["arm_kind"] == "registration"
        and r.get("backend") == "valis"
        and "_seg" not in r["arm"]
    ]
    depths = {r["reg_micro_reg"] for r in valis}
    presets = {r["memory_mode"] for r in valis}
    assert depths == {0, 1, 2}, sorted(depths)
    assert presets == {"low", "medium", "high"}, sorted(presets)
    assert len(valis) == len(presets) * len(depths) == 9, [r["arm"] for r in valis]


# ---------------------------------------------------------------------------
# The QC-segmenter cross
# ---------------------------------------------------------------------------


def test_non_tiled_arms_carry_no_tiled_params(plan):
    """The mirror direction, and the one with teeth for the LAUNCHER.

    run_arms.sh maps every non-empty cell to `--<param> <value>`. A VALIS arm carrying
    reg_tiled_mode would be launched with --reg_tiled_mode, which is a real param and a legal
    enum value, so validateParameters() would NOT reject it -- it would just record a knob
    the run never used, in the very table the arm ranking reads. The same failure
    test_non_ashlar_arms_carry_no_ashlar_params was written for.
    """
    for r in plan:
        if r.get("backend") == "tiled":
            continue
        assert r.get("reg_tiled_mode", "") == "", (
            f"{r['arm']} carries reg_tiled_mode={r.get('reg_tiled_mode')!r}"
        )


def test_qc_segmenter_cross_varies_seg_method_only(cfg, plan):
    """A cross arm must differ from its base arm in seg_method and NOTHING else.

    The claim the axis makes is "identical registration, different measuring
    instrument". If a cross arm also moved memory_mode, the difference would be
    a registration difference wearing a robustness label.
    """
    ref = cfg["qc_segmenter_cross"]["reference_arm"]
    base = next(r for r in plan if r["arm"] == ref)
    crossed = [r for r in plan if r["arm"].startswith(f"{ref}_seg")]
    assert crossed, "qc_segmenter_cross produced no arms"
    for r in crossed:
        assert r["seg_method"] != base["seg_method"]
        for k in ("memory_mode", "reg_micro_reg", "registration_method", "reg_qc"):
            assert r[k] == base[k], f"{r['arm']} moved {k} as well as seg_method"


def test_cross_never_duplicates_the_base_arm(cfg, plan):
    """The baseline segmenter is already the base arm; re-running it measures nothing."""
    names = [r["arm"] for r in plan]
    assert len(names) == len(set(names)), "duplicate arm names in the plan"
    default = cfg["baseline"]["seg_method"]
    assert not any(r["arm"].endswith(f"_seg{default}") for r in plan)


def test_reference_arm_must_name_a_real_arm(cfg):
    bad = dict(
        cfg,
        qc_segmenter_cross=dict(
            cfg["qc_segmenter_cross"], reference_arm="valis_medium_micro9"
        ),
    )
    with pytest.raises(ValueError, match="is not an arm this config produces"):
        build_arm_plan(bad)


def test_cross_all_crosses_every_arm(cfg):
    """`cross: all` on both instruments: every base arm gets (segmenters - 1) + (pairings - 1)
    QC-only twins, and the total is the number quoted in arms.yaml's cost gate."""
    full = build_arm_plan(
        dict(
            cfg,
            qc_segmenter_cross=dict(cfg["qc_segmenter_cross"], cross="all"),
            qc_pairing_cross=dict(cfg["qc_pairing_cross"], cross="all"),
        )
    )
    base = [r for r in full if r["arm_kind"] == "registration"]
    qc = [r for r in full if r["arm_kind"] == "registration_qc"]
    n_seg = len(cfg["qc_segmenter_cross"]["seg_method"])
    n_pair = len(cfg["qc_pairing_cross"]["seg_qc_pairing"])
    # 18 = 9 VALIS (3 tiers x 3 micro-depths) + 9 STARE (3 tiers x 3 gates). History: 7, 9,
    # 7, 9, 12 (VALIS gained medium), and 18 now that STARE mirrors VALIS's shape. Quoted
    # in docs/benchmarks_real.md and arms.yaml's cost gate, which is why it is asserted
    # rather than derived: the point is that the prose and the code agree.
    assert len(base) == 18, len(base)
    assert len(qc) == 18 * ((n_seg - 1) + (n_pair - 1)) == 54, len(qc)
    assert len(base) + len(qc) == 72, (
        "arms.yaml's cost gate quotes 72 registration-step launches"
    )


# ---------------------------------------------------------------------------
# Arm factoring — the segmentation arms must RESUME, not re-register
# ---------------------------------------------------------------------------


def test_segmentation_arms_resume_from_a_registration_arm(cfg, plan):
    seg = [r for r in plan if r["arm_kind"] == "segmentation"]
    assert seg
    reg_names = {r["arm"] for r in plan if r["arm_kind"] == "registration"}
    for r in seg:
        assert r["start"] == "segmentation", (
            "a segmentation arm that does not --start segmentation re-runs "
            "registration, which is the cost this factoring exists to avoid"
        )
        assert r["from_arm"] in reg_names


def test_from_arm_must_name_a_real_arm(cfg):
    bad = dict(
        cfg, segmentation_arms=dict(cfg["segmentation_arms"], from_arm="valis_nope")
    )
    with pytest.raises(ValueError, match="names no registration arm"):
        build_arm_plan(bad)


def test_registration_arms_stop_at_registration(plan):
    """Nothing downstream of registration changes the staged registration QC."""
    for r in plan:
        if r["arm_kind"] == "registration":
            assert (r["start"], r["stop"]) == ("registration", "registration")


def test_preprocessing_runs_exactly_once_and_every_arm_resumes_from_it(plan):
    """No arm axis touches a preproc_* param, so 9 registration arms re-running an
    identical preprocessing step would be pure waste -- and on a real WSI it is the
    expensive part. This is the same factoring as the segmentation arms."""
    pre = [r for r in plan if r["arm_kind"] == "preprocess"]
    assert len(pre) == 1, "preprocessing must be paid for exactly once"
    assert (pre[0]["start"], pre[0]["stop"]) == ("preprocessing", "preprocessing")

    reg = [r for r in plan if r["arm_kind"] == "registration"]
    assert reg
    for r in reg:
        assert r["from_arm"] == pre[0]["arm"], r["arm"]
        assert r["from_csv"] == "preprocessed", r["arm"]


def test_every_resuming_arm_names_the_checkpoint_it_reads(plan):
    """from_csv is the step name minus the "-ing" (Layout.checkpointCsvName()).
    A blank or wrong value makes run_arms.sh look for csv/.csv and skip the arm
    with a message that reads like the upstream run failed."""
    valid = {"preprocessed", "registered", "segmented"}
    for r in plan:
        if r["from_arm"]:
            assert r["from_csv"] in valid, f"{r['arm']} reads csv/{r['from_csv']}.csv"
        else:
            assert r["from_csv"] == "", f"{r['arm']} names a checkpoint but no arm"


def test_run_arms_resolves_the_checkpoint_from_the_plan(plan):
    """The launcher must build the path from from_csv, not hardcode one filename.
    It used to hardcode csv/registered.csv, which silently could not serve the
    registration arms once they began resuming from preprocessed.csv."""
    script = (BENCH / "run_arms.sh").read_text()
    assert "csv/${from_csv}.csv" in script
    # Comments legitimately name the concrete checkpoints while explaining the
    # dependency; only executable lines must not pin one.
    code = "\n".join(
        ln for ln in script.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "csv/registered.csv" not in code, (
        "run_arms.sh still hardcodes a checkpoint filename in executable code"
    )


def test_preprocess_arm_runs_before_the_arms_that_resume_from_it():
    """Pass order in run_arms.sh is a dependency, not a preference."""
    script = (BENCH / "run_arms.sh").read_text()
    m = re.search(r"for kind in ([\w ]+); do", script)
    assert m, "could not find the pass loop"
    order = m.group(1).split()
    assert order.index("preprocess") < order.index("registration")
    assert order.index("registration") < order.index("segmentation")


def test_compute_arm_runs_the_whole_pipeline(plan):
    """The compute profile is the ONLY arm that measures the skipped processes."""
    comp = [r for r in plan if r["arm_kind"] == "compute"]
    assert comp
    for r in comp:
        assert r["start"] == "" and r["stop"] == "", (
            "a gated compute arm cannot price SEGMENT / quantification / export, "
            "which is the only reason this arm exists"
        )


# ---------------------------------------------------------------------------
# The CONSUMER contract (ihc_method/code/registration_arms.R)
# ---------------------------------------------------------------------------


def test_arms_manifest_has_exactly_the_columns_the_consumer_reads(plan):
    """registration_arms.R keeps intersect(c(arm_dir, backend, memory_mode,
    micro_reg, label), names(man)). A renamed column is not an error there — it
    is silently dropped, and the arm falls back to directory-name parsing."""
    rows = arms_manifest_rows(plan)
    assert rows
    assert set(rows[0]) == {"arm_dir", "backend", "memory_mode", "micro_reg", "label"}


def test_arms_manifest_lists_registration_arms_only(plan):
    """A segmentation or compute arm in arms.csv becomes an unlabelled box in
    every panel of a page that ranks REGISTRATION configurations."""
    dirs = {r["arm_dir"] for r in arms_manifest_rows(plan)}
    assert not any(d.startswith(("seg_", "compute_")) for d in dirs)


def test_every_manifest_arm_has_a_label(plan):
    """The label bypasses directory-name parsing. Without it a mislabelled arm
    renders a clean figure with the conclusion inverted."""
    for r in arms_manifest_rows(plan):
        assert r["label"], r


def test_tiled_arm_name_survives_the_consumer_fallback(plan):
    """If arms.csv is lost, registration_arms.R reads a directory containing
    `tiled` or `stare` as the tiled backend. The name must satisfy that too."""
    tiled = [r["arm_dir"] for r in arms_manifest_rows(plan) if r["backend"] == "tiled"]
    assert tiled
    for d in tiled:
        assert "tiled" in d or "stare" in d


# ---------------------------------------------------------------------------
# The BASH contract — run_arms.sh parses arm_plan.csv with a naive IFS split
# ---------------------------------------------------------------------------


def _plan_csv(plan) -> str:
    fields: list[str] = []
    for r in plan:
        for k in r:
            if k not in fields:
                fields.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", restval="")
    w.writeheader()
    w.writerows(plan)
    return buf.getvalue()


def test_no_plan_value_contains_a_comma(plan):
    """run_arms.sh splits on IFS=',' and does NOT honour CSV quoting.

    A comma in any value shifts every later column on that row, so the run
    launches a configuration nobody chose — and it still exits 0. This is why
    the human-readable label lives in arms.csv (read by R's readr, which does
    honour quoting) and never in the plan.
    """
    for r in plan:
        for k, v in r.items():
            assert "," not in str(v), (
                f"{k}={v!r} in arm {r['arm']} would break run_arms.sh"
            )


def test_plan_csv_is_written_unquoted(plan):
    assert '"' not in _plan_csv(plan)


def test_every_plan_row_has_every_column(plan):
    """A short row shifts the columns after it in run_arms.sh exactly as an
    embedded comma does. DictWriter pads with restval='' — this asserts the
    padding is actually reached rather than assumed."""
    text = _plan_csv(plan)
    n = len(text.splitlines()[0].split(","))
    for line in text.splitlines()[1:]:
        assert len(line.split(",")) == n, line


# ---------------------------------------------------------------------------
# Samplesheet reading
# ---------------------------------------------------------------------------


def test_read_input_patients_dedupes_in_first_seen_order(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text(
        "patient_id,path_to_file,is_reference,channels\n"
        "B,/b1.tif,true,DAPI\n"
        "B,/b2.tif,false,DAPI\n"
        "A,/a1.tif,true,DAPI\n"
    )
    assert read_input_patients(p) == ["B", "A"]


def test_read_input_patients_rejects_a_sheet_with_no_patient_id(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("sample,path\nX,/x.tif\n")
    with pytest.raises(ValueError, match="patient_id"):
        read_input_patients(p)


# ---------------------------------------------------------------------------
# The shell scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["run_arms.sh", "pull_to_ihc_method.sh"])
def test_shell_scripts_parse(script):
    """`bash -n` on both. A ${1:?...} error word containing an apostrophe once
    swallowed the rest of pull_to_ihc_method.sh and reported the failure ~30
    lines later, so this is checked rather than eyeballed."""
    r = subprocess.run(
        ["bash", "-n", str(BENCH / script)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", ["run_arms.sh", "pull_to_ihc_method.sh"])
def test_shell_scripts_are_tracked_executable(script):
    """Not cosmetic: a non-executable script fails at launch with exit 126, and
    a local chmod does not reach the cluster's git checkout."""
    r = subprocess.run(
        ["git", "ls-files", "-s", f"benchmarks/{script}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.stdout.startswith("100755"), r.stdout or "file not tracked"


# ---------------------------------------------------------------------------
# The pull script's rsync filters
# ---------------------------------------------------------------------------


def _fake_arm_tree(root: Path) -> None:
    """A minimal results tree with the shape run_arms.sh really produces."""
    for arm in ("valis_high_micro2", "tiled_defaults", "compute_all"):
        for pat in ("046", "24086"):
            base = root / arm / pat
            (base / "qc" / "registration").mkdir(parents=True, exist_ok=True)
            (base / "qc" / "registration" / f"{pat}_seg_qc.json").write_text("{}")
            (base / "registered" / "summary").mkdir(parents=True, exist_ok=True)
            (base / "registered" / "summary" / f"{pat}_summary.csv").write_text(
                "slide\n"
            )
            slides = base / "registered" / "registered_slides"
            slides.mkdir(parents=True, exist_ok=True)
            (slides / "big.ome.tiff").write_bytes(b"\0" * 4096)
        (root / arm / "csv").mkdir(parents=True, exist_ok=True)
        (root / arm / "csv" / "registered.csv").write_text("patient_id\n")
    # Only the compute arm runs the FULL pipeline, so only it carries the
    # postprocessing tables the mirage cell pages read. Step 4 auto-detects it.
    for pat in ("046", "24086"):
        base = root / "compute_all" / pat
        (base / "quantification").mkdir(parents=True, exist_ok=True)
        (base / "quantification" / "merged_quant.csv").write_text("label\n1\n")
        (base / "cell_properties").mkdir(parents=True, exist_ok=True)
        (base / "cell_properties" / "morphology.csv").write_text("label,x,y\n1,0,0\n")
    (root / "arms.csv").write_text("arm_dir,backend,memory_mode,micro_reg,label\n")


@pytest.mark.skipif(not shutil.which("rsync"), reason="rsync not available")
def test_pull_script_copies_the_nested_qc_artifacts(tmp_path):
    """The regression this test exists for: an rsync include containing a '/' is
    anchored to the TRANSFER ROOT, so `--include='qc/**'` matched only a
    top-level qc/ and never <arm>/<patient>/qc/. The script copied almost
    nothing and still exited 0 — invisible without checking the destination."""
    src, ihc = tmp_path / "arm_results", tmp_path / "ihc"
    _fake_arm_tree(src)
    (ihc).mkdir()
    (ihc / "_workflowr.yml").touch()
    r = subprocess.run(
        [str(BENCH / "pull_to_ihc_method.sh"), str(src), str(ihc)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    dest = ihc / "data" / "registration_arms"
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*_seg_qc.json"))
    assert got == [
        "compute_all/046/qc/registration/046_seg_qc.json",
        "compute_all/24086/qc/registration/24086_seg_qc.json",
        "tiled_defaults/046/qc/registration/046_seg_qc.json",
        "tiled_defaults/24086/qc/registration/24086_seg_qc.json",
        "valis_high_micro2/046/qc/registration/046_seg_qc.json",
        "valis_high_micro2/24086/qc/registration/24086_seg_qc.json",
    ], got
    assert list(dest.rglob("*_summary.csv")), "VALIS summaries were not copied"
    assert (dest / "arms.csv").exists(), "the label manifest was not copied"


@pytest.mark.skipif(not shutil.which("rsync"), reason="rsync not available")
def test_pull_script_never_copies_the_images(tmp_path):
    """The registered OME-TIFFs are multi-GB per patient and no ihc_method page
    reads them. The include list is an allowlist for exactly this reason."""
    src, ihc = tmp_path / "arm_results", tmp_path / "ihc"
    _fake_arm_tree(src)
    ihc.mkdir()
    (ihc / "_workflowr.yml").touch()
    subprocess.run(
        [str(BENCH / "pull_to_ihc_method.sh"), str(src), str(ihc)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not list((ihc / "data").rglob("*.ome.tiff"))


@pytest.mark.skipif(not shutil.which("rsync"), reason="rsync not available")
def test_pull_script_copies_the_cell_tables_into_data_mirage(tmp_path):
    """data/mirage/ has TWO consumers with different needs. run_qc.R reads qc/**
    and csv/*.csv; mirage_cells.R reads quantification/merged_quant.csv and
    cell_properties/morphology.csv. The filter used to carry only the first pair,
    so load_mirage_cells() found no patient directories at all -- and said so with
    a single cohort-level warning that reads like an empty dataset rather than a
    truncated copy."""
    src, ihc = tmp_path / "arm_results", tmp_path / "ihc"
    _fake_arm_tree(src)
    ihc.mkdir()
    (ihc / "_workflowr.yml").touch()
    r = subprocess.run(
        [str(BENCH / "pull_to_ihc_method.sh"), str(src), str(ihc)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    mirage = ihc / "data" / "mirage"
    quant = sorted(
        p.relative_to(mirage).as_posix() for p in mirage.rglob("merged_quant.csv")
    )
    morph = sorted(
        p.relative_to(mirage).as_posix() for p in mirage.rglob("morphology.csv")
    )
    assert quant == [
        "046/quantification/merged_quant.csv",
        "24086/quantification/merged_quant.csv",
    ], quant
    assert morph == [
        "046/cell_properties/morphology.csv",
        "24086/cell_properties/morphology.csv",
    ], morph
    # and the qc tree the other consumer needs is still there
    assert list(mirage.rglob("*_seg_qc.json")), "run_qc.R's artifacts were dropped"


@pytest.mark.skipif(not shutil.which("rsync"), reason="rsync not available")
def test_arm_tables_never_land_in_data_benchmark(tmp_path):
    """The arms and the sweep are different experiments that both produce a
    measurements.csv. benchmark_plots.R keys on the SWEEP's axes, so an
    arm-derived measurements.csv in data/benchmark/ renders the scaling pages
    with data answering a different question -- silently, because the file
    exists and carries the right columns.

    They are kept apart by DESTINATION, not by run order: sweep tables go to
    data/benchmark/, arm tables to data/registration_arms/."""
    src, ihc = tmp_path / "arm_results", tmp_path / "ihc"
    _fake_arm_tree(src)
    ihc.mkdir()
    (ihc / "_workflowr.yml").touch()

    handoff = tmp_path / "handoff"
    (handoff / "arms").mkdir(parents=True)
    (handoff / "sweep").mkdir(parents=True)
    (handoff / "arms" / "measurements.csv").write_text("process\nARM\n")
    (handoff / "sweep" / "measurements.csv").write_text("process\nSWEEP\n")

    r = subprocess.run(
        [
            str(BENCH / "pull_to_ihc_method.sh"),
            str(src),
            str(ihc),
            "--handoff",
            str(handoff),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    bench = (ihc / "data" / "benchmark" / "measurements.csv").read_text()
    assert "SWEEP" in bench and "ARM" not in bench, (
        "data/benchmark/measurements.csv must come from the sweep, got:\n" + bench
    )
    arm_side = (ihc / "data" / "registration_arms" / "measurements.csv").read_text()
    assert "ARM" in arm_side, arm_side


def test_pull_script_rejects_an_unknown_option():
    """The script grew options; a typo must not be swallowed as a positional and
    silently become the ihc_method path."""
    r = subprocess.run(
        [str(BENCH / "pull_to_ihc_method.sh"), "--sweeep", "x"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "unknown option" in r.stderr


def test_validate_against_schema_is_what_the_cli_enforces(plan):
    """The CLI raises SystemExit on a non-empty result; this pins that the shipped
    validator (not just the test's own reading of the schema) accepts the shipped
    arms.yaml."""
    assert validate_against_schema(plan, REPO_ROOT / "nextflow_schema.json") == []


# ---------------------------------------------------------------------------
# Tracing is owned by the pipeline, not by the launcher's CLI flags
# ---------------------------------------------------------------------------


def test_run_arms_does_not_pass_with_trace_or_with_report():
    """nextflow.config declares trace/report/timeline observers driven by
    params.trace_dir, and benchmark.config enables them. Passing -with-trace on
    the CLI as well declares a SECOND observer for the same file and points it at
    a directory nothing created -- which is how preprocess_shared failed with a
    stderr log containing only the version banner. run_sweep.sh had it right:
    pass --trace_dir and let the pipeline own it."""
    script = (BENCH / "run_arms.sh").read_text()
    code = "\n".join(
        ln for ln in script.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "-with-trace" not in code
    assert "-with-report" not in code
    assert "--trace_dir" in code, "run_arms.sh must set --trace_dir instead"


def test_run_arms_creates_the_trace_dir_it_names():
    """Nextflow does not create the trace file's parent directory."""
    code = (BENCH / "run_arms.sh").read_text()
    assert '"$outdir/trace"' in code and "mkdir -p" in code


def test_trace_lands_where_the_loader_reads_it():
    """load_runs() reads <root>/<run_id>/trace/trace.txt, and run_id IS the arm
    name, so --trace_dir must be <outdir>/trace with outdir=<root>/<arm>."""
    code = (BENCH / "run_arms.sh").read_text()
    assert '--trace_dir "$outdir/trace"' in code
    assert '--outdir "$outdir"' in code


def test_a_failed_run_reports_stdout_not_only_stderr():
    """Nextflow writes run-level errors to stdout; a message naming only the
    stderr log sends you to a file holding just the version banner."""
    code = (BENCH / "run_arms.sh").read_text()
    assert "nextflow.stdout.log" in code and "tail -n" in code
    assert ".nextflow.log" in code, "the full log path should be named too"


# ---------------------------------------------------------------------------
# Portability: bash 3.2 (the /bin/bash on macOS, and on some older nodes)
# ---------------------------------------------------------------------------

BASH4_ONLY = {
    r"\$\{[A-Za-z_][A-Za-z_0-9]*,,\}": "${VAR,,} lowercase expansion",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*\^\^\}": "${VAR^^} uppercase expansion",
    r"\bdeclare\s+-A\b": "associative arrays",
    r"\bmapfile\b": "mapfile",
    r"\breadarray\b": "readarray",
    r"\bwait\s+-n\b": "wait -n",
}


@pytest.mark.parametrize(
    "script",
    [
        "run_arms.sh",
        "run_sweep.sh",
        "pull_to_ihc_method.sh",
        "submit_arms.sh",
        "submit_sweep.sh",
        "submit_matrix.sh",
    ],
)
def test_scripts_avoid_bash4_only_constructs(script):
    """`bash -n` does NOT catch these -- it parses them and they fail at RUNTIME
    with "bad substitution". submit_matrix.sh shipped with ${SOURCE,,} and passed
    a syntax check on bash 3.2 before failing the moment it ran.

    run_sweep.sh already documents this constraint (it uses indexed arrays and
    FIFO waits rather than declare -A and wait -n); this enforces it.
    """
    code = "\n".join(
        ln
        for ln in (BENCH / script).read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    found = [why for pat, why in BASH4_ONLY.items() if re.search(pat, code)]
    assert not found, f"{script} uses bash 4+ only: {found}"


@pytest.mark.parametrize(
    "script",
    [
        "submit_arms.sh",
        "submit_sweep.sh",
        "submit_matrix.sh",
    ],
)
def test_submitters_parse_and_are_tracked_executable(script):
    r = subprocess.run(
        ["bash", "-n", str(BENCH / script)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        ["git", "ls-files", "-s", f"benchmarks/{script}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.stdout.startswith("100755"), r.stdout or "not tracked"


def test_submit_matrix_resolves_python_and_preflights_deps():
    """The matrix job is hours long; a missing numpy must cost seconds, not the
    whole walltime. And `python` may not exist even with a conda env active."""
    code = (BENCH / "submit_matrix.sh").read_text()
    # ASSERTED ON THE PROPERTY, NOT ON ONE LITERAL SPELLING. This line used to match
    # the exact string `PYTHON="${PYTHON:-$(command -v python3`, and it broke the
    # moment that line's trailing `|| true` was rewritten into the assignment chain
    # tests/test_no_swallowed_failures.py requires -- a guard failing on a change that
    # made the code strictly better is a guard testing the wrong thing. The three
    # properties below are what actually matters and none of them moved:
    #   1. an operator-supplied PYTHON wins (a conda env can put the right one
    #      somewhere neither name finds);
    #   2. python3 is TRIED, because `python` may not exist even with an env active;
    #   3. an unresolved interpreter is a NAMED error here, not a `command not found`
    #      hundreds of lines into a multi-hour job.
    assert "${PYTHON:-}" in code or "${PYTHON:-$" in code, (
        "an operator-supplied PYTHON must still win"
    )
    assert "command -v python3" in code, "python3 must be tried"
    assert "no python3/python on PATH" in code, (
        "an unresolved interpreter must abort with a named error"
    )
    assert "python deps OK" in code and "numpy" in code
    assert "from bioio import BioImage" in code, "ND2 sources need a bioio check"


# ---------------------------------------------------------------------------
# Concurrency must be raised on the CLI, not in benchmark.config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["submit_arms.sh", "submit_sweep.sh"])
def test_submitters_raise_concurrency_on_the_command_line(script):
    """conf/modules.config caps every process at Math.min(own, params.max_forks),
    and maxForks is NOT a dynamic directive -- Nextflow compares it against 0 in
    TaskProcessor's constructor, so it cannot be deferred into a closure and is
    evaluated EAGERLY when that file is parsed. That parse happens inside
    nextflow.config, BEFORE a -c file is merged, so benchmark.config cannot reach a
    single clamp. Only a CLI --max_forks is in scope in time.
    """
    code = "\n".join(
        ln
        for ln in (BENCH / script).read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "--max_forks" in code, f"{script} does not pass --max_forks"
    assert "--queue_size" in code, (
        f"{script} passes --max_forks without --queue_size; the LOWER of the pair "
        "binds, and queue_size defaults far below max_forks, so max_forks alone is "
        "a no-op"
    )


def test_benchmark_config_does_not_set_concurrency_directives():
    """Setting them there is worse than useless: process.maxForks cannot lift a
    per-process withName value that already resolved to min(own, 100), so the file
    reads as if it raises concurrency while doing nothing."""
    code = "\n".join(
        ln
        for ln in (BENCH / "configs" / "benchmark.config").read_text().splitlines()
        if not ln.lstrip().startswith(("//", "*", "/*"))
    )
    assert "queueSize" not in code, "benchmark.config still sets executor.queueSize"
    assert "maxForks = 200" not in code, "benchmark.config still sets a global maxForks"


# ---------------------------------------------------------------------------
# The QC instrument crosses — planned as resumes, run as one chain per base arm
# ---------------------------------------------------------------------------


def test_registration_arms_are_symmetric_across_backends(plan):
    """VALIS tier x depth and STARE tier x gate: the same number of draws each, or the
    best-cell ranking is tuned-vs-untuned in whichever direction has more cells."""
    base = [r for r in plan if r["arm_kind"] == "registration"]
    valis = [r for r in base if r["backend"] == "valis"]
    tiled = [r for r in base if r["backend"] == "tiled"]
    assert len(valis) == len(tiled) == 9, (len(valis), len(tiled))


def test_tiled_arms_are_tier_x_gate(plan):
    """Every shipped tier at every gate value, and the gate really varies."""
    tiled = [
        r for r in plan if r["backend"] == "tiled" and r["arm_kind"] == "registration"
    ]
    shipped = set(stare_preset_modes())
    assert {r["reg_tiled_mode"] for r in tiled} == shipped
    gates = {str(r["reg_tiled_gate_tre"]) for r in tiled}
    assert len(gates) == 3, gates
    cells = {(r["reg_tiled_mode"], str(r["reg_tiled_gate_tre"])) for r in tiled}
    assert len(cells) == len(shipped) * len(gates) == len(tiled)
    # and the gate is in the directory name, so a lost arms.csv cannot merge them
    assert len({r["arm"] for r in tiled}) == len(tiled)
    assert all("gate" in r["arm"] and "." not in r["arm"] for r in tiled)


def test_valis_arms_carry_no_gate(plan):
    for r in plan:
        if r.get("backend") == "valis":
            assert r.get("reg_tiled_gate_tre", "") == "", r["arm"]


def test_qc_cross_arms_resume_their_base_arm(cfg, plan):
    """Each cross arm names a base registration arm, differs from it in exactly ONE QC
    instrument, and is planned as arm_kind=registration_qc (the resumed pass)."""
    by_id = {r["run_id"]: r for r in plan}
    crosses = [r for r in plan if r["arm_kind"] == "registration_qc"]
    assert crosses, "no QC cross arms planned"
    for r in crosses:
        base = by_id.get(r["resume_run"])
        assert base is not None and base["arm_kind"] == "registration", r["arm"]
        differs = [k for k in QC_INSTRUMENTS if str(r[k]) != str(base[k])]
        assert differs and len(differs) == 1, (r["arm"], differs)
        # everything that defines the REGISTRATION is identical to the base
        for k in (
            "backend",
            "memory_mode",
            "reg_micro_reg",
            "reg_tiled_mode",
            "reg_tiled_gate_tre",
            "from_arm",
        ):
            assert str(r[k]) == str(base[k]), (r["arm"], k)
    # base arms never resume anything
    for r in plan:
        if r["arm_kind"] != "registration_qc":
            assert r["resume_run"] == "", r["arm"]


def test_qc_crosses_are_one_instrument_at_a_time(cfg, plan):
    """Segmenters at the baseline pairing, pairings at the baseline segmenter -- no
    factorial cell varies both."""
    b = cfg["baseline"]
    for r in plan:
        if r["arm_kind"] != "registration_qc":
            continue
        assert not (
            r["seg_method"] != b["seg_method"]
            and r["seg_qc_pairing"] != b["seg_qc_pairing"]
        ), r["arm"]
    n_base = sum(1 for r in plan if r["arm_kind"] == "registration")
    n_seg = len(cfg["qc_segmenter_cross"]["seg_method"]) - 1
    n_pair = len(cfg["qc_pairing_cross"]["seg_qc_pairing"]) - 1
    n_cross = sum(1 for r in plan if r["arm_kind"] == "registration_qc")
    assert n_cross == n_base * (n_seg + n_pair), (n_cross, n_base, n_seg, n_pair)


def test_base_arms_carry_the_baseline_instruments(cfg, plan):
    b = cfg["baseline"]
    for r in plan:
        if r["arm_kind"] == "registration":
            assert (
                r["seg_method"] == b["seg_method"]
                and r["seg_qc_pairing"] == b["seg_qc_pairing"]
            ), r["arm"]


def test_qc_pass_runs_after_registration_and_before_the_rest():
    script = (BENCH / "run_arms.sh").read_text()
    m = re.search(r"for kind in ([\w ]+); do", script)
    order = m.group(1).split()
    assert (
        order.index("registration")
        < order.index("registration_qc")
        < order.index("external")
    )
    for flag in ("seg_qc_pairing", "reg_tiled_gate_tre"):
        assert f"add_param {flag} " in script, f"run_arms.sh does not forward --{flag}"


def _fake_nextflow(bindir: Path, log: Path) -> None:
    """A `nextflow` that records its argv, fakes the checkpoint a resumed arm needs,
    writes the history line launch() reads the session id from, and FAILS if another
    fake is already running against the same session (the cache-DB lock)."""
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "nextflow").write_text(
        r"""#!/usr/bin/env bash
set -e
LOG="__LOG__"
args=("$@")
outdir=""; name=""; resume=""; prev=""; workdir=""
for a in "${args[@]}"; do
  case "$prev" in
    --outdir) outdir="$a" ;;
    -name) name="$a" ;;
    -resume) resume="$a" ;;
    -work-dir) workdir="$a" ;;
  esac
  [[ "$prev" == "-resume" && "$a" == -* ]] && resume="latest"
  prev="$a"
done
[[ " ${args[*]} " == *" -resume"* && -z "$resume" ]] && resume="latest"
sid="${resume:-sess-$name}"
mkdir -p .nextflow "${workdir:-work}"
lock=".nextflow/lock-$sid"
# mkdir is the atomic test-and-set: two launches started in the same instant both
# passed a `-e` test before either had touched the file.
if ! mkdir "$lock" 2>/dev/null; then
  echo "LOCKFAIL|$name|$sid" >> "$LOG"
  echo "Unable to acquire lock on session $sid" >&2; exit 1
fi
# Hold the session for long enough that two launches of one base issued within the same
# second MUST overlap; a shorter hold let a sabotaged (unchained) launcher pass by luck.
sleep 2
mkdir -p "$outdir/csv"; echo "patient_id" > "$outdir/csv/preprocessed.csv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' now 1s "$name" OK - "$sid" "nextflow ${args[*]}" >> .nextflow/history
echo "$(pwd)|$name|${resume:-}|$outdir" >> "$LOG"
rmdir "$lock"
""".replace("__LOG__", str(log))
    )
    (bindir / "nextflow").chmod(0o755)


def test_run_arms_chains_the_qc_crosses_of_one_base_and_resumes_its_session(tmp_path):
    """Behavioural: with a stub nextflow, the QC pass launches each cross arm INSIDE its
    base arm's launch dir with `-resume <that base's session>`, cross arms of ONE base
    never overlap (the stub dies on a concurrent resume of one session), and different
    bases do run concurrently."""
    import os
    import subprocess

    cfg = yaml.safe_load((BENCH / "configs" / "arms.yaml").read_text())
    # a small plan: cross only the reference arm's twins + one more base, to keep it quick
    cfg["registration_arms"]["valis"]["memory_mode"] = ["high"]
    cfg["registration_arms"]["valis"]["reg_micro_reg"] = [1, 2]
    cfg["registration_arms"]["tiled"]["enabled"] = False
    cfg["external_baseline"]["ashlar"]["enabled"] = False
    cfg["segmentation_arms"]["seg_method"] = []
    cfg["qc_segmenter_cross"]["cross"] = "all"
    cfg["qc_pairing_cross"]["cross"] = "all"
    plan = build_arm_plan(cfg)
    root = tmp_path / "arm_results"
    root.mkdir()
    plan_csv = tmp_path / "plan.csv"
    plan_csv.write_text(_plan_csv(plan))
    sheet = tmp_path / "input.csv"
    sheet.write_text(
        "patient_id,path_to_file,is_reference,channels\nP1,/x/a.tif,true,DAPI\n"
    )
    log = tmp_path / "launches.log"
    _fake_nextflow(tmp_path / "bin", log)
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        ARMS_CONCURRENCY="4",
    )
    r = subprocess.run(
        ["bash", str(BENCH / "run_arms.sh"), str(plan_csv), str(sheet), str(root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    launches = [ln.split("|") for ln in log.read_text().splitlines()]
    collisions = [ln for ln in launches if ln[0] == "LOCKFAIL"]
    assert not collisions, (
        "two cross arms resumed the same base session concurrently: "
        + ", ".join(ln[1] for ln in collisions)
    )
    by_name = {ln[1]: ln for ln in launches}
    bases = [p["run_id"] for p in plan if p["arm_kind"] == "registration"]
    crosses = [p for p in plan if p["arm_kind"] == "registration_qc"]
    assert len(bases) == 2 and len(crosses) == 2 * 3, (len(bases), len(crosses))
    for c in crosses:
        cwd, _, resume, outdir = by_name[f"arms-{c['run_id']}"]
        assert cwd.endswith(f"/.launch/{c['resume_run']}"), (c["run_id"], cwd)
        assert resume == f"sess-arms-{c['resume_run']}", (c["run_id"], resume)
        assert outdir.endswith(f"/{c['arm']}"), outdir
    for b in bases:
        cwd, _, resume, _ = by_name[f"arms-{b}"]
        assert cwd.endswith(f"/.launch/{b}") and resume == ""
    # order: every base before its crosses (the pass barrier)
    names = [ln[1] for ln in launches]
    for c in crosses:
        assert names.index(f"arms-{c['resume_run']}") < names.index(
            f"arms-{c['run_id']}"
        )
