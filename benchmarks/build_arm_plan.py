"""Expand arms.yaml into a real-sample arm plan (one row per pipeline launch).

The counterpart to build_run_plan.py, for the REAL slides rather than the
synthetic matrix. Two outputs, because the sweep and the consumer want different
things from the same expansion:

    arm_plan.csv   one row per pipeline launch, consumed by run_arms.sh
    arms.csv       the LABEL manifest ihc_method's registration_arms.R reads
                   (arm_dir, backend, memory_mode, micro_reg, label)

arms.csv is written into the results root, not beside the plan, because that is
where the consumer looks: `data/registration_arms/arms.csv`.

WHY THE LABEL MANIFEST IS NOT OPTIONAL HERE. registration_arms.R falls back to
parsing the directory name for `high`/`low` and a micro depth when arms.csv is
absent. A mislabelled arm does not fail — it produces a clean figure with the
conclusion inverted. The QC-segmenter-crossed arms below (`valis_high_micro2_seg
stardist`) are exactly the names that fallback would read wrong, so this module
always emits the manifest rather than leaving it to a flag.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Params that exist only on one backend. A tiled arm must carry NEITHER, and the
# consumer requires them blank so it can tell "not applicable" from "at default".
# Writing memory_mode=high on a STARE arm would invent a value the run never had.
VALIS_ONLY = ("memory_mode", "reg_micro_reg")

# The mirror of VALIS_ONLY for the tiled/STARE backend. reg_tiled_mode selects a row of
# RegPresets.STARE and means nothing on a VALIS arm, so a VALIS arm must carry it BLANK --
# both so the consumer can tell "not applicable" from "at default", and so run_arms.sh's
# add_param blank-guard never emits `--reg_tiled_mode ""`, which schema validation rejects.
TILED_ONLY = ("reg_tiled_mode", "reg_tiled_gate_tre")

# The two QC MEASURING INSTRUMENTS every registration-step arm carries: which segmenter
# finds the nuclei the seg-overlap QC scores on (params.seg_method) and how those nuclei
# are paired across slides (params.seg_qc_pairing). Baseline values on a base arm; the
# QC cross arms below vary exactly one of them.
QC_INSTRUMENTS = ("seg_method", "seg_qc_pairing")

# The external-baseline (ashlar) columns. These are arguments to
# benchmarks/run_ashlar_arm.sh, NOT pipeline params -- the ext_ prefix is what
# guarantees that: run_pass() forwards a fixed list of param columns as --flags, and no
# name in this tuple can ever collide with one. Blank on every pipeline arm so all rows
# share a header (run_arms.sh's col_val indexes by header position).
EXTERNAL_ONLY = (
    "ext_tool",
    "ext_from_arm",
    "ext_tile_size",
    "ext_overlap",
    "ext_max_shift_um",
    "ext_seg_method",
)

# ASHLAR_ONLY = ("reg_ashlar_tile", "reg_ashlar_overlap", "reg_ashlar_max_shift_um") is GONE
# with the arm. Those three are not pipeline params any more -- nextflow.config declares none
# of them -- so emitting the columns at all, even blank, would put three names in arms.csv
# that the schema rejects the moment run_arms.sh stopped skipping empties.


def valis_arm_name(memory_mode: str, micro: int) -> str:
    return f"valis_{memory_mode}_micro{micro}"


def tiled_arm_name(mode: str, gate) -> str:
    # The tier AND the gate are IN the name, not only in arms.csv. registration_arms.R
    # falls back to parsing the directory name when the manifest is missing, and arms that
    # differed only by a column it failed to read would render as one replicated box.
    # Keeps the `tiled` substring the consumer's backend fallback keys on. The gate is
    # written without its decimal point (0.5 -> gate05, 1.0 -> gate1, 2.0 -> gate2): a dot
    # in a directory name reads as an extension to half the tools that will touch it.
    g = f"{float(gate):g}".replace(".", "")
    return f"tiled_{mode}_gate{g}"


def _registration_arms(cfg: dict) -> list[dict]:
    """The registration arms, before the QC-segmenter cross is applied."""
    arms: list[dict] = []
    ra = cfg.get("registration_arms", {})

    valis = ra.get("valis") or {}
    for mm in valis.get("memory_mode", []):
        for micro in valis.get("reg_micro_reg", []):
            arms.append(
                {
                    "arm": valis_arm_name(mm, micro),
                    "backend": "valis",
                    "memory_mode": mm,
                    "reg_micro_reg": micro,
                    "label": f"{mm} / micro {micro}",
                }
            )

    # ASHLAR IS NOT A *PIPELINE* ARM, AND CANNOT BE. run_arms.sh passes each
    # registration row's columns as --flags, and ashlar is no longer a pipeline backend:
    # :fire: 6a54479 removed it for v1.0.0, registration_method's schema enum is
    # ['valis', 'tiled'], and tests/test_ashlar_backend_removed.py keeps it out. A
    # registration arm here would emit --registration_method ashlar and be rejected at
    # launch, plus three reg_ashlar_* flags that name nothing in nextflow.config.
    #
    # It IS an arm of this plan, under arm_kind='external' -- see _external_arms below,
    # which run_arms.sh dispatches to benchmarks/run_ashlar_arm.sh instead of Nextflow.
    # Guarded by test_no_ashlar_arms_reach_the_pipeline_plan below, which now asserts
    # the narrower (and true) property: no ashlar row carries a pipeline param.

    tiled = ra.get("tiled") or {}
    if tiled.get("enabled"):
        # STARE fans out over its TIER, not over individual knobs. reg_tiled_mode is the
        # knob an operator actually picks, and each tier moves all five tier-owned values
        # (tile / halo / out_tile / coarse_max_dim / upsample) coherently -- see
        # RegPresets.STARE. Varying them singly is sweep.yaml's job, on synthetic images
        # where a cell is cheap; here a cell is a real WSI at up to 483 GB, so arms carries
        # the three shipped tiers and nothing finer.
        #
        # WHY IT IS NOT ONE ARM ANY MORE: at one tiled arm the ranking tuned VALIS across
        # six configurations and STARE across none, which is the same tuned-vs-untuned bias
        # test_project_stare_resolution_axis_mirrors_the_valis_one guards in the sweep --
        # and it was unguarded here, in the block that produces the manuscript figure.
        # TIER x GATE, the mirror of the VALIS block (2026-09-10). reg_tiled_gate_tre is
        # STARE's refinement depth -- the rigid-stage TRE above which a tile is non-rigidly
        # refined -- and is ungated, so it is legal under every tier; the tier-owned knobs
        # (tile / halo / out_tile / coarse_max_dim / upsample) are not, and stay with the tier.
        # Three tier-only arms against nine VALIS arms gave VALIS three times the draws.
        for mode in tiled.get("reg_tiled_mode", []):
            for gate in tiled.get("reg_tiled_gate_tre", [None]):
                arms.append(
                    {
                        "arm": tiled_arm_name(mode, gate)
                        if gate is not None
                        else f"tiled_{mode}",
                        "backend": "tiled",
                        # No memory_mode, no reg_micro_reg -- see VALIS_ONLY. The consumer
                        # keys "is this the tiled backend" off the `backend` column when
                        # arms.csv is present, and off the substring `tiled`/`stare` when it
                        # is not; the name satisfies both so the fallback path stays correct.
                        "memory_mode": "",
                        "reg_micro_reg": "",
                        "reg_tiled_mode": mode,
                        "reg_tiled_gate_tre": "" if gate is None else gate,
                        "label": f"tiled (STARE, {mode}"
                        + ("" if gate is None else f", gate {float(gate):g} px")
                        + ")",
                    }
                )
    return arms


def ashlar_arm_name(tile: int, shift) -> str:
    # Both varied values are IN the name. registration_arms.R falls back to parsing the
    # directory name when arms.csv is missing, and four ashlar arms differing only by a
    # column it failed to read would render as one quadruplicated box -- the same failure
    # tiled_arm_name() exists to avoid. Keeps the `ashlar` substring the consumer's
    # backend fallback keys on.
    return f"ashlar_t{int(tile)}_s{int(shift)}"


def _external_arms(cfg: dict) -> list[dict]:
    """The external-tool baseline arms (ashlar), as rows of THIS plan.

    WHY THEY LIVE IN THE SAME PLAN. benchmarks/ashlar/solve.py rewrites ashlar's
    per-tile placements into the same M0 + mesh manifest STARE emits, and
    bin/warp_seg_qc.py --method tiled reads that manifest JVM-free. So ashlar can be
    scored by the pipeline's OWN reg_qc=2 seg-overlap scorer, on the same nuclei, into
    the same `<root>/<arm>/<patient>/qc/registration/*_seg_qc.json` tree the twelve
    registration arms write. Same metric family, same columns, one layout, and
    ihc_method's readers pick it up with no path added.

    That is strictly stronger than scoring it in its own harness against synthetic
    ground truth, which is what the deleted synthetic ground-truth rung did: those
    numbers shared no column with this table and could not be ranked against it.

    THE ROWS CARRY NO PIPELINE PARAMS. registration_method/memory_mode/reg_micro_reg/
    reg_tiled_mode are all blank, because nothing here reaches validateParameters() --
    run_arms.sh dispatches arm_kind='external' to run_ashlar_arm.sh, not to Nextflow.
    The ext_* columns are that script's arguments, deliberately prefixed so they can
    never collide with a param name run_pass() would forward as a --flag.
    """
    arms: list[dict] = []
    eb = cfg.get("external_baseline") or {}
    ash = eb.get("ashlar") or {}
    if not ash.get("enabled"):
        return arms

    from_arm = ash.get("from_arm")
    if not from_arm:
        raise ValueError(
            "external_baseline.ashlar.from_arm is required: the ashlar arm reuses that "
            "registration arm's published QC nuclei "
            "(<root>/<from_arm>/<patient>/qc/registration/geojson/) rather than "
            "re-segmenting. Scoring against DIFFERENT nuclei than the arms it is ranked "
            "against would make the comparison meaningless."
        )

    for tile in ash.get("tile_size", []):
        for shift in ash.get("maximum_shift_um", []):
            name = ashlar_arm_name(tile, shift)
            arms.append(
                {
                    "arm": name,
                    "backend": "ashlar",
                    # Blank for the same reason a tiled arm blanks memory_mode: the consumer
                    # must be able to tell "not applicable" from "at default".
                    "memory_mode": "",
                    "reg_micro_reg": "",
                    "reg_tiled_mode": "",
                    "ext_tool": "ashlar",
                    "ext_from_arm": from_arm,
                    "ext_tile_size": tile,
                    "ext_overlap": ash.get("overlap_fraction", 0.1),
                    "ext_max_shift_um": shift,
                    # Filled in by build_arm_plan from ext_from_arm's own seg_method.
                    "ext_seg_method": "",
                    "label": f"ashlar (tile {int(tile)}, shift {int(shift)}um)",
                }
            )
    return arms


def _apply_qc_cross(
    arms: list[dict], cfg: dict, block: str, param: str, suffix: str, note: str
) -> list[dict]:
    """Cross the registration arms with ONE QC measuring instrument.

    Both instruments -- the QC segmenter (params.seg_method, block `qc_segmenter_cross`)
    and the nucleus-pairing rule (params.seg_qc_pairing, block `qc_pairing_cross`) --
    change what the registration accuracy is MEASURED WITH while leaving the registration
    byte-identical: robustness axes on the headline number, never a quality claim.

    A cross arm carries `resume_run` = its base arm, so run_arms.sh launches it in the
    base arm's launch directory with -resume and only the QC chain runs again. One
    instrument at a time: the cross arms produced here vary `param` and keep every other
    instrument at the base arm's value; the caller applies the two crosses to the SAME
    base list, never to each other's output, so there is no factorial.
    """
    x = cfg.get(block) or {}
    mode = x.get("cross", "none")
    values = list(x.get(param, []))
    default = (cfg.get("baseline") or {}).get(param)

    # Every arm carries the baseline instrument unless it is a cross arm.
    for a in arms:
        a.setdefault(param, default)
        a.setdefault("resume_run", "")

    # A reference_arm that names nothing is wrong config in EVERY mode: under `all` it is
    # unused today and silently stale the day someone switches back to `reference`.
    ref = x.get("reference_arm")
    names = {a["arm"] for a in arms}
    if ref is not None and ref not in names:
        raise ValueError(
            f"{block}.reference_arm={ref!r} is not an arm this config produces. "
            f"Available: {sorted(names)}"
        )

    if mode == "none" or not values:
        return []

    if mode == "reference":
        if ref is None:
            raise ValueError(f"{block}.cross=reference needs a reference_arm")
        targets = [a for a in arms if a["arm"] == ref]
    elif mode == "all":
        targets = list(arms)
    else:
        raise ValueError(
            f"{block}.cross must be 'reference', 'all' or 'none', got {mode!r}"
        )

    extra: list[dict] = []
    for base in targets:
        for v in values:
            if v == base[param]:
                continue  # that IS the base arm; a duplicate run measures nothing
            a = dict(base)
            a["arm"] = f"{base['arm']}_{suffix}{v}"
            a[param] = v
            a["resume_run"] = base["arm"]
            a["label"] = f"{base['label']} [{note}: {v}]"
            extra.append(a)
    return extra


def _apply_qc_crosses(arms: list[dict], cfg: dict) -> list[dict]:
    """Base arms first, then the segmenter cross, then the pairing cross -- each applied
    to the BASE arms only (one instrument varied per cross arm)."""
    # Every base arm carries BOTH baseline instruments before either cross copies it,
    # or a segmenter-cross arm would be copied without the pairing column.
    baseline = cfg.get("baseline") or {}
    for a in arms:
        for param in QC_INSTRUMENTS:
            a.setdefault(param, baseline.get(param))
        a.setdefault("resume_run", "")
    seg = _apply_qc_cross(
        arms, cfg, "qc_segmenter_cross", "seg_method", "seg", "QC seg"
    )
    pair = _apply_qc_cross(
        arms, cfg, "qc_pairing_cross", "seg_qc_pairing", "pair", "QC pairing"
    )
    return arms + seg + pair


# The one shared preprocessing run every registration arm resumes from.
PREPROCESS_ARM = "preprocess_shared"


def _compute_arm_name(patient: str, rep: int) -> str:
    return f"compute_{patient or 'all'}" + (f"_rep{rep}" if rep else "")


_LABELS: dict[str, str] = {}


def build_arm_plan(cfg: dict) -> list[dict]:
    """Expand arms.yaml into a flat list of pipeline launches.

    Three arm kinds, deliberately FACTORED rather than crossed:

      registration  --start preprocessing --stop registration, one per config.
      segmentation  --start segmentation, resuming ONE registration arm's
                    csv/registered.csv, so registration is paid for once.
      compute       the full pipeline, traced, for the per-process cost profile.
    """
    baseline = cfg.get("baseline") or {}
    if baseline.get("reg_qc") != 2:
        # Not a style preference: reg_qc<2 emits no staged seg-overlap QC, so
        # every registration arm would produce zero accuracy rows and the
        # consumer would render an empty page with no error.
        raise ValueError(
            "arms.yaml baseline.reg_qc must be 2 — the registration arm ranking "
            f"reads the reg_qc=2 staged QC and nothing else emits it (got "
            f"{baseline.get('reg_qc')!r})"
        )

    rows: list[dict] = []
    n = 0
    _LABELS.clear()

    # PREPROCESSING IS SHARED. Nothing in registration_arms or qc_segmenter_cross
    # touches a preproc_* param, so all 9 registration arms would otherwise re-run
    # an identical (and, on a real WSI, expensive) preprocessing step. Run it once
    # and have every registration arm resume from its csv/preprocessed.csv -- the
    # same factoring already applied to the segmentation arms.
    rows.append(
        {
            "run_id": PREPROCESS_ARM,
            "arm_kind": "preprocess",
            "start": "preprocessing",
            "stop": "preprocessing",
            "from_arm": "",
            "from_csv": "",
            "rep": 0,
            "arm": PREPROCESS_ARM,
            "backend": "",
            "memory_mode": "",
            "reg_micro_reg": "",
            "seg_method": baseline.get("seg_method", ""),
            "seg_qc_pairing": "",
            "resume_run": "",
            "registration_method": "",
            "reg_qc": "",
            **{k: "" for k in TILED_ONLY},
            **{k: "" for k in EXTERNAL_ONLY},
        }
    )
    n += 1

    arms = _apply_qc_crosses(_registration_arms(cfg), cfg)
    for a in arms:
        _LABELS[a["arm"]] = a["label"]
        rows.append(
            {
                "run_id": a["arm"],
                # A QC cross arm is `registration_qc`: run_arms.sh runs that pass AFTER the
                # registration pass and launches each one in its base arm's launch dir with
                # -resume, so REGISTER is served from the cache and only the QC chain runs.
                "arm_kind": "registration_qc"
                if a.get("resume_run")
                else "registration",
                "resume_run": a.get("resume_run", ""),
                "start": "registration",
                "stop": "registration",
                "from_arm": PREPROCESS_ARM,
                "from_csv": "preprocessed",
                "rep": 0,
                "registration_method": a["backend"],
                **{
                    k: a[k]
                    for k in (
                        "arm",
                        "backend",
                        "memory_mode",
                        "reg_micro_reg",
                        "seg_method",
                        "seg_qc_pairing",
                    )
                },
                # Blank on every non-tiled arm, exactly as memory_mode/reg_micro_reg are
                # blank on non-VALIS arms.
                **{k: a.get(k, "") for k in TILED_ONLY},
                **{k: "" for k in EXTERNAL_ONLY},
                "reg_qc": 2,
            }
        )
        n += 1

    # EXTERNAL BASELINE (ashlar). Emitted AFTER the registration arms and run in its own
    # pass, because it reuses `ext_from_arm`'s published QC nuclei -- the same resume
    # discipline the segmentation arms use, for the same reason: re-segmenting would
    # score ashlar against different nuclei than the arms it is ranked against.
    ext = _external_arms(cfg)
    if ext:
        known = {a["arm"] for a in arms}
        for e in ext:
            if e["ext_from_arm"] not in known:
                raise ValueError(
                    f"external_baseline.ashlar.from_arm={e['ext_from_arm']!r} names no "
                    f"registration arm this plan runs. Available: {sorted(known)}"
                )
    for e in ext:
        _LABELS[e["arm"]] = e["label"]
        rows.append(
            {
                "run_id": e["arm"],
                "arm_kind": "external",
                # No --start/--stop: nothing in this row reaches Nextflow.
                "start": "",
                "stop": "",
                # TWO sources, and they are different arms. The IMAGES come from the shared
                # preprocessing run, so ashlar registers exactly what VALIS and STARE
                # registered (comparing it against native slides would hand it a different
                # input and make the ranking meaningless). The NUCLEI come from
                # `ext_from_arm`'s published qc/registration/geojson/. `from_arm`/`from_csv`
                # name the first because that is what run_arms.sh's resume check reads --
                # which also gets the external arm the "SKIP if upstream did not complete"
                # guard for free.
                "from_arm": PREPROCESS_ARM,
                "from_csv": "preprocessed",
                "rep": 0,
                "arm": e["arm"],
                "backend": e["backend"],
                "memory_mode": "",
                "reg_micro_reg": "",
                # seg_method is BLANK, and ext_seg_method carries the value instead. Not
                # pedantry: `seg_method` is one of the columns run_pass() forwards as a
                # --flag, and this row must stay incapable of contributing one even if the
                # external dispatch branch is ever moved. The value is still recorded, because
                # WHICH segmenter found the nuclei is a property of the score -- it is just
                # inherited from ext_from_arm's geojsons rather than chosen here.
                "seg_method": "",
                "seg_qc_pairing": "",
                "resume_run": "",
                "registration_method": "",
                "reg_qc": "",
                **{k: "" for k in TILED_ONLY},
                **{k: e[k] for k in EXTERNAL_ONLY},
                "ext_seg_method": next(
                    a["seg_method"] for a in arms if a["arm"] == e["ext_from_arm"]
                ),
            }
        )
        n += 1

    sa = cfg.get("segmentation_arms") or {}
    if sa:
        frm = sa.get("from_arm")
        if frm not in {a["arm"] for a in arms}:
            raise ValueError(
                f"segmentation_arms.from_arm={frm!r} names no registration arm. "
                f"It must be an arm this plan runs, because the segmentation arms "
                f"resume from its csv/registered.csv. Available: "
                f"{sorted({a['arm'] for a in arms})}"
            )
        for m in sa.get("seg_method", []):
            rows.append(
                {
                    "run_id": f"seg_{m}",
                    "arm_kind": "segmentation",
                    "start": "segmentation",
                    "stop": "",
                    "from_arm": frm,
                    "from_csv": "registered",
                    "rep": 0,
                    "arm": f"seg_{m}",
                    "backend": "",
                    "memory_mode": "",
                    "reg_micro_reg": "",
                    "seg_method": m,
                    "seg_qc_pairing": "",
                    "resume_run": "",
                    "registration_method": "",
                    "reg_qc": "",
                    **{k: "" for k in TILED_ONLY},
                    **{k: "" for k in EXTERNAL_ONLY},
                }
            )
            n += 1

    cp = cfg.get("compute_profile")
    if cp is not None:
        pats = cp.get("patients") or [""]  # "" = every patient in the input.csv
        for p in pats:
            for rep in range(int(cp.get("repeats", 1) or 1)):
                rows.append(
                    {
                        "run_id": _compute_arm_name(p, rep),
                        "arm_kind": "compute",
                        "start": "",
                        "stop": "",
                        "from_arm": "",
                        "from_csv": "",
                        "rep": rep,
                        "arm": _compute_arm_name(p, rep),
                        "backend": baseline.get("registration_method", "valis"),
                        "memory_mode": baseline.get("memory_mode", ""),
                        "reg_micro_reg": baseline.get("reg_micro_reg", ""),
                        "seg_method": baseline.get("seg_method", ""),
                        "seg_qc_pairing": baseline.get("seg_qc_pairing", ""),
                        "resume_run": "",
                        "registration_method": baseline.get(
                            "registration_method", "valis"
                        ),
                        "reg_qc": baseline.get("reg_qc", 2),
                        "only_patient": p,
                        **{k: "" for k in TILED_ONLY},
                        **{k: "" for k in EXTERNAL_ONLY},
                    }
                )
                n += 1
    return rows


def arms_manifest_rows(plan: list[dict]) -> list[dict]:
    """The consumer's arms.csv: the RANKED arms, one row per arm dir.

    Segmentation and compute arms are deliberately absent — registration_arms.R
    ranks REGISTRATION configurations, and a segmentation arm listed there would
    appear as an unlabelled box in every panel.

    The external (ashlar) arms ARE included, because they are ranked: they write the
    same *_seg_qc.json into the same tree and are read by the same panel. Omitting them
    would not hide them — registration_arms.R discovers arms by walking `<root>`, so an
    absent row falls through to its directory-name parser, which reads `ashlar_t1024_s30`
    as neither VALIS nor tiled and labels the box from the raw string. Listing them is
    what makes the external baseline appear as a baseline rather than as a mystery arm.
    """
    return [
        {
            "arm_dir": r["arm"],
            "backend": r["backend"],
            "memory_mode": r["memory_mode"],
            "micro_reg": r["reg_micro_reg"],
            "label": _LABELS[r["arm"]],
        }
        for r in plan
        if r["arm_kind"] in ("registration", "registration_qc", "external")
    ]


def schema_enums(schema_path: Path) -> dict:
    """param -> allowed values, from nextflow_schema.json.

    The same enum plugin/nf-schema's validateParameters() enforces at run time.
    """
    import json

    out: dict[str, list] = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict) and "enum" in v:
                    out[k] = v["enum"]
                walk(v)
        elif isinstance(node, list):
            for i in node:
                walk(i)

    walk(json.loads(schema_path.read_text()))
    return out


def validate_against_schema(plan: list[dict], schema_path: Path) -> list[str]:
    """Every enum-valued param an arm pins must be a value the pipeline accepts.

    This exists because a one-character typo cost a whole submission: arms.yaml
    pinned `seg_method: instanseg` (one 't'), which is NOT the pipeline's
    `instantseg`, and nothing caught it until all 14 runs had been queued,
    scheduled, and died inside validateParameters(). The name-level checks all
    passed -- and the typo is easy, because `instanseg_model_dir` really is
    spelled with one 't'.

    Checked HERE, in the plan builder, so it fails in the seconds-long local step
    rather than after the cluster has accepted the work.
    """
    enums = schema_enums(schema_path)
    bad = []
    for r in plan:
        for k, v in r.items():
            if k in enums and v not in ("", None) and v not in enums[k]:
                bad.append(f"  {r['arm']}: {k}={v!r} -- allowed: {enums[k]}")
    return bad


def _write_csv(path: Path, rows: list[dict], lead: list[str]) -> None:
    fields = list(lead)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        # lineterminator='\n' (not csv's '\r\n') so run_arms.sh's bash column
        # parsing never sees a trailing '\r' on the last field.
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)


def read_input_patients(input_csv: Path) -> list[str]:
    """patient_id values in the real samplesheet, first-seen order, de-duplicated."""
    with open(input_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "patient_id" not in (rows[0] or {}):
        raise ValueError(f"{input_csv} has no patient_id column")
    seen: list[str] = []
    for r in rows:
        p = (r.get("patient_id") or "").strip()
        if p and p not in seen:
            seen.append(p)
    return seen


def main():
    import yaml

    ap = argparse.ArgumentParser(
        description="Expand arms.yaml into arm_plan.csv (+ the consumer's arms.csv)"
    )
    ap.add_argument("--arms", required=True, type=Path)
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="the REAL samplesheet (patient_id,path_to_file,is_reference,channels)",
    )
    ap.add_argument("--out", required=True, type=Path, help="arm_plan.csv")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="where arms.csv is written (default: alongside --out). "
        "Point it at the results root the runs publish into — that "
        "is where registration_arms.R looks for it.",
    )
    a = ap.parse_args()

    cfg = yaml.safe_load(a.arms.read_text())
    plan = build_arm_plan(cfg)

    # Fail before anything is written, so a stale plan is never left behind for
    # the launcher's non-empty check to accept.
    schema = Path(__file__).resolve().parents[1] / "nextflow_schema.json"
    if schema.exists():
        bad = validate_against_schema(plan, schema)
        if bad:
            raise SystemExit(
                "arms.yaml pins values the pipeline will reject:\n"
                + "\n".join(bad)
                + "\n\nThese are checked against nextflow_schema.json here because "
                "validateParameters() would otherwise only reject them once every "
                "run had been queued and scheduled."
            )

    patients = read_input_patients(a.input)
    cp = cfg.get("compute_profile") or {}
    unknown = [p for p in (cp.get("patients") or []) if p not in patients]
    if unknown:
        raise SystemExit(
            f"compute_profile.patients names patients absent from {a.input}: "
            f"{unknown}\nPresent: {patients}"
        )

    _write_csv(a.out, plan, ["run_id", "arm_kind", "arm"])
    root = a.results_root or a.out.parent
    _write_csv(
        root / "arms.csv",
        arms_manifest_rows(plan),
        ["arm_dir", "backend", "memory_mode", "micro_reg", "label"],
    )

    by_kind: dict[str, int] = {}
    for r in plan:
        by_kind[r["arm_kind"]] = by_kind.get(r["arm_kind"], 0) + 1
    kinds = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
    print(f"Wrote {len(plan)} arm runs ({kinds}) to {a.out}")
    print(f"Wrote label manifest to {root / 'arms.csv'}")
    print(f"{len(patients)} patient(s) in {a.input}: {', '.join(patients)}")
    # Say the multiplier out loud. Every registration/segmentation arm runs the
    # WHOLE cohort, so the launch count is not the run count.
    per_cohort = sum(1 for r in plan if r["arm_kind"] != "compute")
    print(
        f"NOTE: {per_cohort} of these launch the full cohort "
        f"({per_cohort} x {len(patients)} patient-runs), plus "
        f"{by_kind.get('compute', 0)} compute-profile launch(es)."
    )


if __name__ == "__main__":
    main()
