#!/usr/bin/env python3
"""supplementary.py -- EVERY supplementary figure (S2-S11 + the method mosaic) in one run.

One results root holds every method (benchmarks/submit_arms.sh: VALIS, STARE, ASHLAR,
the segmentation arms), and arm_plan.csv says which arm is
which method. This module draws the manuscript's supplementary set FROM that root. It
re-registers nothing and re-segments nothing: every picture is a re-render of slides
already on disk, by the same renderers the figure grid uses (reg_mosaic, reg_overlay,
reg_zoom, reg_crop), and every number comes from the reg_qc=2 scorer's JSONs and the
Nextflow traces the arms already wrote.

WHAT YOU CHOOSE BETWEEN. The comparisons are drawn in every combination, so the choice
is made by looking, not by re-running:

    method set   all   = Before | VALIS | STARE | ASHLAR
    config       high  = each method's shipped high tier (supplementary.yaml `high:`).
                         THE DEFAULT: a legend that names no registration tier means the
                         high one (user ruling 2026-09-30). If the configured arm is not
                         on disk, another arm OF THE HIGH TIER stands in -- never a lower
                         tier; a method with no high-tier arm is left out, loudly.
                 best  = each method's arm with the highest median final-stage matched
                         Dice over the cohort (picks.csv says which, and by how much).
                         Always in picks.csv; DRAWN only when `configs:` lists it
    variant      v1..vN = different tissue, SAME tissue in every set and config: the
                 anchor render picks the ROIs/crops once and every other render reuses
                 them, so two panels differ only by the method that registered them

Outputs (``-o OUT``):

    OUT/picks.csv                   the arm behind every (method, config), with its numbers
    OUT/mosaic/<set>_<config>/v<k>/ reg_mosaic per patient (overlay + checker), Dice in cells
    OUT/S4/<set>_<config>/v<k>/     Before | VALIS | STARE (+ASHLAR), matched insets
    OUT/S5/                         registration cost: S5_cost_high (the two high arms, per
                                    patient and phase) and S5_cost_by_tier_all (every tier)
    OUT/S6/r<k>/                    nuclei | cell masks per backend + the pairwise-Dice matrix
    OUT/S7/<patient>/<set>_<config>/v<k>/   as S4, for every other case
    OUT/S8/<set>_<config>/          Dice and displacement by case and by panel pair
    OUT/S2/                         secondary-only controls at ONE fixed contrast (if given)
    OUT/S3, S9, S10, S11            collected from ihc_method (submit_supplementary.sh)
    OUT/index.html                  every variant of every figure on one page, to choose
    OUT/check.csv                   per figure: READY / PARTIAL / MISSING and what was found
                                    (written on every run; ``--check`` stops there)

Run on the cluster through benchmarks/submit_supplementary.sh; locally::

    python -m benchmarks.supplementary --results arm_results --plan arm_plan.csv \\
        --config benchmarks/configs/supplementary.yaml -o supp --only mosaic,S4
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Registration methods, in the order their columns/rows are drawn.
REG_METHODS = ("valis", "stare", "ashlar")
TITLE = {"valis": "VALIS", "stare": "STARE", "ashlar": "ASHLAR"}
DEFAULT_SETS = {"all": ["valis", "stare", "ashlar"]}
# The arms that ARE a registration of their method. registration_qc rows re-score a base
# arm with another QC instrument -- same registration, different ruler -- so ranking them
# would pick a ruler, not a method configuration.
_RANKED_KINDS = ("registration", "external")
FIGURES = ("mosaic", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "gallery")
# The tier is IN every tiered arm's name (build_arm_plan.py): valis_<tier>_micro<d>,
# tiled_<tier>_s<stride> (STARE). ASHLAR has no tier.
_TIER_RE = re.compile(r"^(?:valis|tiled)_(high|medium|low)_")
DEFAULT_CONFIGS = ["high"]


def tier_of(arm: str) -> str:
    """`high` / `medium` / `low` from an arm name, `""` for an untiered arm (ASHLAR)."""
    m = _TIER_RE.match(str(arm))
    return m.group(1) if m else ""


# ------------------------------------------------------------------------ context --
@dataclass
class Ctx:
    root: Path
    plan: pd.DataFrame
    plan_csv: Path
    out: Path
    cfg: dict
    exec_prefix: list[str] = field(default_factory=list)
    dry_run: bool = False
    log: list[str] = field(default_factory=list)
    input_patients: list[str] = field(default_factory=list)

    def opt(self, *keys, default=None):
        node = self.cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return default if node is None else node

    @property
    def formats(self) -> str:
        return str(self.opt("options", "formats", default="png,pdf"))

    @property
    def dpi(self) -> int:
        return int(self.opt("options", "dpi", default=150))

    @property
    def cohort(self) -> set[str] | None:
        """THE case set of every figure: `patients:` from the config when given, else
        the patient_id column of the arms' samplesheet (--input), else None (all)."""
        ps = self.opt("patients", default=[]) or []
        if not ps:
            return set(self.input_patients) or None
        bad = [p for p in ps if not isinstance(p, str)]
        if bad:
            # YAML reads 046 as the octal int 38 and 10338 as an int: a case would drop
            # out of the cohort silently. Refuse instead.
            raise SystemExit(
                f"supplementary.yaml patients: quote every case id, e.g. ['046', "
                f"'10338'] (unquoted, YAML parsed {bad} as numbers)"
            )
        return set(ps) or None

    def arm_dir(self, arm: str) -> Path:
        return self.root / arm

    def render(self, tool: str, args: list[str]) -> bool:
        """One renderer call, through the render container when one is given."""
        cmd = [
            *self.exec_prefix,
            "python3",
            "-m",
            f"benchmarks.{tool}",
            *map(str, args),
        ]
        self.log.append(shlex.join(cmd))
        if self.dry_run:
            print("[dry-run]", shlex.join(cmd))
            return True
        env = dict(os.environ)
        for k in ("PYTHONPATH", "SINGULARITYENV_PYTHONPATH", "APPTAINERENV_PYTHONPATH"):
            env[k] = str(REPO_ROOT)
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        if r.returncode != 0:
            print(f"[supp] FAILED ({r.returncode}): {shlex.join(cmd)}", file=sys.stderr)
        return r.returncode == 0


def _patients_of(arm_dir: Path) -> list[str]:
    p = arm_dir / "csv" / "registered.csv"
    if not p.is_file():
        return []
    seen: list[str] = []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            pid = (r.get("patient_id") or "").strip()
            if pid and pid not in seen:
                seen.append(pid)
    return seen


def read_input_patients(path: Path) -> list[str]:
    """Distinct patient_id of a pipeline samplesheet, as STRINGS (046 stays 046)."""
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        if "patient_id" not in (r.fieldnames or []):
            raise SystemExit(f"{path} has no patient_id column: not a samplesheet")
        return list(
            dict.fromkeys(x["patient_id"].strip() for x in r if x["patient_id"])
        )


def _registered(arm_dir: Path) -> bool:
    return (arm_dir / "csv" / "registered.csv").is_file()


# Ceiling on the dpi a panel composite is saved at (a 3-panel row is then ~9000 px wide).
PANEL_DPI_CAP = 900


# -------------------------------------------------------------------------- picks --
def accuracy_long(ctx: Ctx) -> pd.DataFrame:
    """Final-stage scorer numbers per (run, patient, moving slide)."""
    from benchmarks.analysis.lib import quality

    long = quality.harvest_registration_qc(ctx.root, ctx.plan_csv)
    if long.empty:
        return long
    long["_rank"] = long["stage"].map(quality._STAGE_RANK).fillna(-1)
    final = (
        long.sort_values("_rank")
        .groupby(["run_id", "patient_id", "moving"], as_index=False)
        .tail(1)
    )
    native = long[long["stage"] == "native"][
        ["run_id", "patient_id", "moving", "dice_matched", "displacement_um_p50"]
    ].rename(
        columns={
            "dice_matched": "native_dice",
            "displacement_um_p50": "native_disp_um",
        }
    )
    return final.merge(native, on=["run_id", "patient_id", "moving"], how="left")


def pick_arms(ctx: Ctx, final: pd.DataFrame) -> pd.DataFrame:
    """One arm per (method, config): `high` from the config, `best` by median Dice."""
    plan = ctx.plan
    if "method" not in plan.columns:
        raise SystemExit(
            f"{ctx.plan_csv} has no `method` column: build it with this checkout's "
            "benchmarks/build_arm_plan.py (submit_arms.sh does), which labels every row "
            "valis/stare/ashlar/seg"
        )
    cand = plan[plan["method"].isin(REG_METHODS) & plan["arm_kind"].isin(_RANKED_KINDS)]
    per_run = pd.DataFrame(columns=["run_id", "dice", "disp_um", "n"])
    if not final.empty:
        per_run = final.groupby("run_id", as_index=False).agg(
            dice=("dice_matched", "median"),
            disp_um=("displacement_um_p50", "median"),
            n=("dice_matched", "size"),
        )
    cand = cand.merge(per_run, on="run_id", how="left")
    cand = cand[[_registered(ctx.arm_dir(a)) for a in cand["arm"]]]
    rows = []
    high_cfg = ctx.opt("high", default={}) or {}
    for m in REG_METHODS:
        c = cand[cand["method"] == m]
        if c.empty:
            continue
        scored = c.dropna(subset=["dice"]).sort_values(
            ["dice", "disp_um"], ascending=[False, True]
        )
        best = scored.iloc[0] if len(scored) else c.iloc[0]
        want = high_cfg.get(m)
        tiers = c["arm"].map(tier_of)
        if want in set(c["arm"]):
            high, high_why = c[c["arm"] == want].iloc[0], "configured high tier"
        elif tiers.eq("").all():
            # An untiered method (ASHLAR): "high" has no meaning, its one config stands.
            high, high_why = best, f"untiered method; configured {want!r} not on disk"
        else:
            # A legend naming no tier means HIGH: another high-tier arm stands in, the
            # best-scored one; a lower tier never does.
            h = scored[scored["arm"].map(tier_of) == "high"]
            h = h if len(h) else c[tiers == "high"]
            if h.empty:
                high, high_why = None, ""
                print(
                    f"[supp] {m}: configured high arm {want!r} is not on disk and no "
                    "other high-tier arm is: left out of every `high` figure",
                    file=sys.stderr,
                )
            else:
                high = h.iloc[0]
                high_why = (
                    f"configured {want!r} not on disk: best-scored other high tier arm"
                )
        for config, r, why in (
            ("high", high, high_why),
            (
                "best",
                best,
                f"highest median final-stage Dice of {len(scored)} scored arms"
                if len(scored)
                else "no score on disk: fell back to the first registered arm",
            ),
        ):
            if r is None:
                continue
            rows.append(
                {
                    "method": m,
                    "config": config,
                    "arm": r["arm"],
                    "median_dice": r.get("dice"),
                    "median_disp_um": r.get("disp_um"),
                    "n_slides": r.get("n"),
                    "tier": tier_of(r["arm"]),
                    "why": why,
                }
            )
    picks = pd.DataFrame(
        rows,
        columns=[
            "method",
            "config",
            "arm",
            "median_dice",
            "median_disp_um",
            "n_slides",
            "tier",
            "why",
        ],
    )
    ctx.out.mkdir(parents=True, exist_ok=True)
    picks.to_csv(ctx.out / "picks.csv", index=False)
    return picks


def arm_for(picks: pd.DataFrame, method: str, config: str) -> str | None:
    hit = picks[(picks["method"] == method) & (picks["config"] == config)]
    return None if hit.empty else str(hit["arm"].iloc[0])


def _anchor_arm(picks: pd.DataFrame) -> str:
    """The arm whose crops every other panel reuses: VALIS high, else the first high."""
    for m in REG_METHODS:
        if a := arm_for(picks, m, "high"):
            return a
    return str(picks["arm"].iloc[0])


def method_sets(
    ctx: Ctx, picks: pd.DataFrame, min_methods: int = 2
) -> dict[str, list[str]]:
    """The configured sets, cut to the methods with an arm on disk.

    A comparison (mosaic, S4) needs two methods; a single-method figure (S7: one method
    before vs after; S8: per-arm scores) is drawn from whatever is there. A set cut down
    to other than its configured methods is renamed after what it holds, so a set is never
    labelled with a method it does not hold, and identical cuts collapse to one."""
    sets = ctx.opt("sets", default=None) or DEFAULT_SETS
    have = set(picks["method"])
    out: dict[str, list[str]] = {}
    for name, methods in sets.items():
        ms = [m for m in methods if m in have]
        if len(ms) < min_methods:
            continue
        key = name if ms == list(methods) else "_".join(ms)
        if ms not in out.values():
            out[key] = ms
    return out


def configs(ctx: Ctx) -> list[str]:
    return list(ctx.opt("configs", default=DEFAULT_CONFIGS))


def _label(method: str, arm: str, config: str) -> str:
    return f"{TITLE[method]} ({config})"


# ------------------------------------------------------------------------- mosaic --
def fig_mosaic(ctx: Ctx, picks: pd.DataFrame, patients: list[str]) -> None:
    """Before | VALIS | STARE | ASHLAR, Dice in every cell -- the priority figure.

    The anchor (set `all`, config `high`) picks the ROIs; every other set and config is
    drawn on exactly those ROIs (--rois-json), so a column differs only by its method.
    `patch_um` may be a list (e.g. [200, 500]): one mosaic per patch size, in
    `<set>_<config>_p<patch>/` when there is more than one."""
    m = ctx.opt("mosaic", default={}) or {}
    patches = m.get("patch_um", 200)
    patches = patches if isinstance(patches, list) else [patches]
    for patch in patches:
        suffix = f"_p{patch}" if len(patches) > 1 else ""
        _mosaic_one(ctx, picks, patients, patch, suffix)


def _mosaic_one(ctx: Ctx, picks, patients: list[str], patch, suffix: str) -> None:
    m = ctx.opt("mosaic", default={}) or {}
    variants = int(m.get("variants", 2))
    common = [
        "--kinds",
        ",".join(m.get("kinds", ["overlay", "checker"])),
        "--numbers",
        m.get("numbers", "scorer"),
        "--patch-um",
        patch,
        "--formats",
        ctx.formats,
    ]
    if m.get("rows"):
        common += ["--rows", m["rows"]]
    for r in m.get("rounds", []) or []:
        common += ["--rounds", r]
    sets = method_sets(ctx, picks)
    anchor_methods = sets.get("all") or next(iter(sets.values()))
    anchor = ctx.out / "mosaic" / f"_anchor{suffix}"
    for pid in patients:
        arms = [a for mm in anchor_methods if (a := arm_for(picks, mm, "high"))]
        ctx.render(
            "reg_mosaic",
            [
                *(ctx.arm_dir(a) for a in arms),
                "--patient",
                pid,
                "--variants",
                variants,
                "-o",
                anchor,
                *common,
            ],
        )
        for set_name, methods in sets.items():
            # The mosaic has its own config list (supplementary.yaml mosaic.configs); the
            # other figures keep the global one.
            for config in m.get("configs") or configs(ctx):
                arms = [(mm, a) for mm in methods if (a := arm_for(picks, mm, config))]
                if len(arms) < 2:
                    continue
                labels = []
                for mm, a in arms:
                    labels += ["--label", f"{a}={_label(mm, a, config)}"]
                for v in range(1, variants + 1):
                    rois = anchor / (
                        f"{pid}_v{v}_rois.json" if variants > 1 else f"{pid}_rois.json"
                    )
                    if not rois.is_file() and not ctx.dry_run:
                        continue
                    ctx.render(
                        "reg_mosaic",
                        [
                            *(ctx.arm_dir(a) for _, a in arms),
                            "--patient",
                            pid,
                            "--rois-json",
                            rois,
                            *labels,
                            "-o",
                            ctx.out
                            / "mosaic"
                            / f"{set_name}_{config}{suffix}"
                            / f"v{v}",
                            *common,
                        ],
                    )


# ---------------------------------------------------------------------- S4 and S7 --
def _overlay_panels(
    ctx: Ctx, picks: pd.DataFrame, pid: str, fig: str, spec: dict
) -> Path:
    """Render every (method, config) on the anchor's crops; return the panels dir."""
    variants = int(spec.get("variants", 3))
    base = [
        "--patient",
        pid,
        "--field-um",
        spec.get("field_um", 500),
        "--numbers",
        spec.get("numbers", "scorer"),
        "--formats",
        "png",
        "--dpi",
        ctx.dpi,
    ]
    if spec.get("zoom_um"):
        base += ["--zoom-um", spec["zoom_um"]]
    if _label_mode(spec) != "burned":
        base += ["--labels", "none"]
    rounds = spec.get("rounds") or []
    if rounds:
        base += ["--rounds", *rounds]
    root = ctx.out / fig / pid if fig == "S7" else ctx.out / fig
    anchor = root / "_anchor"
    ctx.render(
        "reg_overlay",
        [
            ctx.arm_dir(_anchor_arm(picks)),
            "-o",
            anchor,
            "--variants",
            variants,
            *base,
        ],
    )
    manifests = sorted(anchor.glob(f"{pid}_*_overlay.json"))
    for mf in manifests:
        man = json.loads(mf.read_text())
        v = int(man.get("variant", 1))
        # the anchor's OWN size: a field fitted to the tissue is smaller than field_um
        # (`--roi=`: a fitted field may start left of or above the slide, and argparse
        # reads a separate "-120,40" as an option)
        pin = [f"--roi={man['crop']['y']},{man['crop']['x']}"]
        pin += ["--field-px", man["crop"]["size_px"]]
        if man.get("zoom"):
            pin += [f"--zoom-roi={man['zoom']['y']},{man['zoom']['x']}"]
        for method in REG_METHODS:
            for config in configs(ctx):
                arm = arm_for(picks, method, config)
                if arm is None:
                    continue
                ctx.render(
                    "reg_overlay",
                    [
                        ctx.arm_dir(arm),
                        "-o",
                        root / "panels" / f"{method}_{config}" / f"v{v}",
                        "--rounds",
                        man["round"],
                        "--title",
                        _label(method, arm, config),
                        *pin,
                        *[b for b in base if b != "--rounds" and b not in rounds],
                    ],
                )
    return root


OVERLAY_LABELS = ("burned", "editable", "none")


def _label_mode(spec: dict) -> str:
    """`labels:` of an overlay figure's options: how the words inside a panel are set.

    burned   = drawn into the panel's pixels by reg_overlay (the historical figure)
    editable = the panel is rendered bare and the composer sets scale-bar text and channel
               names over it as real text, which a vector editor can change or delete
    none     = bare panels, nothing written inside them
    """
    mode = str(spec.get("labels") or "burned")
    if mode not in OVERLAY_LABELS:
        raise SystemExit(f"labels: {mode!r} is not one of {', '.join(OVERLAY_LABELS)}")
    return mode


def _editable_fonts(plt) -> None:
    """Text in a PDF/SVG stays TEXT: TrueType (42), not matplotlib's default Type 3, which
    a vector editor opens as outlines it cannot retype."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def _panel_manifest(panel: Path, which: str) -> dict:
    """The manifest reg_overlay wrote FOR THIS PANEL (`<stem>_overlay.json` beside
    `<stem>_<which>.png`); {} when there is none."""
    mf = panel.with_name(panel.name[: -len(f"_{which}.png")] + "_overlay.json")
    return json.loads(mf.read_text()) if mf.is_file() else {}


# Inside a panel the composer re-sets only what it does not already write outside it: the
# method is the column title and the numbers are the line underneath.
INNER_ROLES = ("bar", "legend")


def _draw_inner_labels(ax, labels: list[dict], roles=INNER_ROLES, rename=None) -> int:
    """Lay a panel's recorded labels over its image as real text; returns how many.
    ``rename(role, text)`` rewrites what a label says."""
    ax.apply_aspect()  # the axes box as the image actually fills it
    fig = ax.figure
    height_pt = ax.get_position().height * fig.get_figheight() * 72.0
    n = 0
    for lab in labels:
        if lab.get("role") not in roles:
            continue
        ax.text(
            lab["x"],
            lab["y"],
            rename(lab["role"], lab["text"]) if rename else lab["text"],
            transform=ax.transAxes,
            ha=lab.get("ha", "left"),
            va=lab.get("va", "bottom"),
            fontsize=lab["size"] * height_pt,
            color=lab.get("color", "white"),
            fontweight=lab.get("weight", "normal"),
        )
        n += 1
    return n


PAIR_ROLES = ("title", "bar", "legend")


def _plain_title(role: str, text: str) -> str:
    """`After (valis_high_micro2)` -> `After`: the pair figure names no method."""
    return text.split(" (")[0] if role == "title" else text


def _compose_pairs(ctx: Ctx, root: Path, pid: str, labels: str) -> list[Path]:
    """`before_after/<pid>_<round>_v<k>`: the anchor's Before and After of one crop, side
    by side and nothing else -- titled Before / After, with the scale bars and the channel
    names, and no method name and no numbers.

    With `labels: editable` the panels are bare and every word is real text in the PDF;
    with `burned` they carry their own words (method name included) as pixels; `none`
    writes nothing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    _editable_fonts(plt)
    written = []
    for mf in sorted((root / "_anchor").glob(f"{pid}_*_overlay.json")):
        man = json.loads(mf.read_text())
        stem = mf.name[: -len("_overlay.json")]
        panels = [mf.with_name(f"{stem}_{w}.png") for w in ("before", "after")]
        if not all(p.is_file() for p in panels):
            continue
        imgs = [mpimg.imread(p) for p in panels]
        h, w = imgs[0].shape[:2]
        width_in = 7.2
        gap = 0.004  # a hairline between the two, as a fraction of the figure's width
        pw = (1.0 - gap) / 2.0
        fig = plt.figure(figsize=(width_in, width_in * pw * h / w), facecolor="white")
        for k, (img, which) in enumerate(zip(imgs, ("before", "after"))):
            ax = fig.add_axes([k * (pw + gap), 0.0, pw, 1.0])
            ax.imshow(img, interpolation="none", aspect="auto")
            ax.set_axis_off()
            if labels == "editable":
                _draw_inner_labels(
                    ax,
                    (man.get("labels") or {}).get(which, []),
                    roles=PAIR_ROLES,
                    rename=_plain_title,
                )
        out = root / "before_after"
        out.mkdir(parents=True, exist_ok=True)
        # each panel at its own pixels, as _compose_overlays
        dpi = min(PANEL_DPI_CAP, max(ctx.dpi, int(-(-w // (width_in * pw)))))
        for fmt in ctx.formats.split(","):
            path = out / f"{stem}.{fmt}"
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)
    return written


def _compose_overlays(
    ctx: Ctx,
    picks,
    root: Path,
    pid: str,
    final: pd.DataFrame,
    min_methods: int = 2,
    labels: str = "burned",
):
    """Per (set, config, variant): Before | one After per method, numbers underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    _editable_fonts(plt)

    for set_name, methods in method_sets(ctx, picks, min_methods).items():
        for config in configs(ctx):
            panel_dirs = {m: root / "panels" / f"{m}_{config}" for m in methods}
            for vdir in sorted((panel_dirs[methods[0]]).glob("v*")):
                v = vdir.name
                cols, titles, notes, inner = [], [], [], []
                befores = sorted(vdir.glob(f"{pid}_*_before.png"))
                if not befores:
                    continue
                cols.append(befores[0])
                titles.append("Before")
                notes.append(_numbers_note(befores[0], "before"))
                inner.append(_panel_manifest(befores[0], "before"))
                for m in methods:
                    after = sorted((panel_dirs[m] / v).glob(f"{pid}_*_after.png"))
                    if not after:
                        continue
                    cols.append(after[0])
                    titles.append(_label(m, arm_for(picks, m, config), config))
                    notes.append(_numbers_note(after[0], "after"))
                    inner.append(_panel_manifest(after[0], "after"))
                fig, axes = plt.subplots(
                    1, len(cols), figsize=(3.4 * len(cols), 3.9), squeeze=False
                )
                panel_w = 0
                for ax, img, t, n in zip(axes[0], cols, titles, notes):
                    pixels = mpimg.imread(img)
                    panel_w = max(panel_w, pixels.shape[1])
                    ax.imshow(pixels)
                    ax.set_title(t, fontsize=9)
                    ax.set_xlabel(n, fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(
                    f"{pid} — {set_name} set, {config} configuration", fontsize=9
                )
                fig.tight_layout()
                if labels == "editable":
                    for ax, man, which in zip(
                        axes[0], inner, ["before"] + ["after"] * len(inner)
                    ):
                        _draw_inner_labels(ax, (man.get("labels") or {}).get(which, []))
                out = root / f"{set_name}_{config}" / v
                out.mkdir(parents=True, exist_ok=True)
                # Saved at the dpi that gives a panel ITS OWN pixels: at options.dpi a
                # 2600 px panel was squeezed into ~480 px and the single-cell inset into
                # ~180, which is what made the zoom unreadable.
                ax_in = axes[0][0].get_position().width * fig.get_figwidth()
                dpi = min(PANEL_DPI_CAP, max(ctx.dpi, int(-(-panel_w // ax_in))))
                for fmt in ctx.formats.split(","):
                    fig.savefig(out / f"{pid}_{set_name}_{config}_{v}.{fmt}", dpi=dpi)
                plt.close(fig)
                _values_table(picks, final, pid, methods, config).to_csv(
                    out / f"{pid}_{set_name}_{config}_values.csv", index=False
                )


def _numbers_note(panel: Path, which: str) -> str:
    """`Dice = 0.92  Δ = 0.4 µm (ROI)` from the manifest reg_overlay wrote FOR THIS PANEL:
    the scorer's matched Dice for the slide, and the nucleus displacement inside this crop
    when it holds enough nuclei (else the slide-level value, marked *).

    The manifest is the panel's own (`<stem>_overlay.json` beside `<stem>_<which>.png`). A
    directory holds one per moving round, and taking "the first manifest" captioned a panel
    with another round's numbers (S4, 2026-10-05: 0.12 in the image, 0.46 under it).
    """
    man = _panel_manifest(panel, which)
    if man:
        n = (man.get("numbers") or {}).get(which)
        if not isinstance(n, dict):
            return ""
        parts = []
        if n.get("dice_matched") is not None:
            parts.append(f"Dice = {float(n['dice_matched']):.2f}")
        if n.get("roi_displacement_um") is not None:
            parts.append(f"Δ = {float(n['roi_displacement_um']):.1f} µm (ROI)")
        elif n.get("slide_displacement_um") is not None:
            parts.append(f"Δ = {float(n['slide_displacement_um']):.1f} µm*")
        elif n.get("slide_displacement_px") is not None:
            parts.append(f"Δ = {float(n['slide_displacement_px']):.1f} px*")
        return "  ".join(parts) + (f"   [{n.get('stage')}]" if n.get("stage") else "")
    return ""


def _values_table(picks, final, pid, methods, config) -> pd.DataFrame:
    """Per-mode slide-level numbers for the figure legend: the AUTHORS TO SUPPLY values."""
    rows = []
    for m in methods:
        arm = arm_for(picks, m, config)
        sub = (
            final[
                (final["run_id"] == arm) & (final["patient_id"].astype(str) == str(pid))
            ]
            if not final.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "method": TITLE[m],
                "config": config,
                "arm": arm,
                "patient": pid,
                "median_dice_matched": sub["dice_matched"].median()
                if len(sub)
                else None,
                "median_centroid_disp_um": sub["displacement_um_p50"].median()
                if len(sub)
                else None,
                "native_dice": sub["native_dice"].median() if len(sub) else None,
                "n_pairs": int(sub["n_pairs"].sum()) if len(sub) else None,
            }
        )
    return pd.DataFrame(rows)


def fig_s4(ctx, picks, patients, final):
    spec = ctx.opt("S4", default={}) or {}
    pid = str(spec.get("patient") or patients[0])
    root = _overlay_panels(ctx, picks, pid, "S4", spec)
    if not ctx.dry_run:
        _compose_overlays(ctx, picks, root, pid, final, labels=_label_mode(spec))
        _compose_pairs(ctx, root, pid, _label_mode(spec))


def fig_s7(ctx, picks, patients, final):
    spec = ctx.opt("S7", default={}) or {}
    s4_pid = str((ctx.opt("S4", default={}) or {}).get("patient") or patients[0])
    for pid in [
        str(p) for p in (spec.get("patients") or [p for p in patients if p != s4_pid])
    ]:
        root = _overlay_panels(ctx, picks, pid, "S7", spec)
        if not ctx.dry_run:
            # S7 is ONE method before vs after (as Fig 4a): VALIS alone draws it.
            _compose_overlays(
                ctx, picks, root, pid, final, min_methods=1, labels=_label_mode(spec)
            )
            _compose_pairs(ctx, root, pid, _label_mode(spec))


# ----------------------------------------------------------------------------- S5 --
def fig_s5(ctx: Ctx):
    """Registration cost by tier, VALIS against STARE."""
    if ctx.dry_run:
        print("[dry-run] S5: registration_cost_by_tier from the traces")
        return
    import matplotlib

    matplotlib.use("Agg")
    from benchmarks.analysis.lib import load, plotting, quality

    runs = load.load_runs(ctx.root, ctx.plan_csv)
    # An attempt killed for memory and retried is not the method's cost: the numbers are
    # those of the attempts that finished (the failed ones are counted, not summed, in
    # S5_cost_by_patient.csv).
    cost = quality.registration_cost_by_tier(load.only_successful(runs), ctx.root)
    out = ctx.out / "S5"
    out.mkdir(parents=True, exist_ok=True)
    cost.to_csv(out / "registration_cost_by_tier.csv", index=False)
    if cost.empty:
        print("[supp] S5: no registration trace on disk", file=sys.stderr)
        return
    per_slide = cost["n_slides"].notna().any()
    if per_slide:
        # An arm still running has a partial trace and no csv/registered.csv yet: its
        # peak RSS and CPU-h are those of the tasks finished SO FAR, not the arm's. It is
        # in the CSV (n_slides empty) and not in the figure.
        running = cost["n_slides"].isna()
        if running.any():
            print(
                f"[supp] S5: {int(running.sum())} arm(s) not drawn, registration "
                f"unfinished: {sorted(cost.loc[running, 'run_id'])}",
                file=sys.stderr,
            )
    unit = "per slide" if per_slide else "per run"
    # No wall-clock: first task start to last task end of a tile-parallel arm measures
    # how many tiles the scheduler ran at once, not the method. It stays in the raw CSV.
    metrics = [
        "reg_peak_rss_gb",
        "cpu_hours_per_slide" if per_slide else "reg_cpu_hours",
    ]
    labels = ["peak RSS GB\n(largest task)", f"reserved core-h\n{unit}"]
    for name, keep in (("all", ("valis", "stare")),):
        sub = cost[cost["backend"].isin(keep)]
        if per_slide:
            sub = sub[sub["n_slides"].notna()]
        if sub["backend"].nunique() < 1:
            continue
        fig = plotting.cost_by_tier(sub, metrics, labels)
        plotting.save_fig(
            fig, out / f"S5_cost_by_tier_{name}", formats=ctx.formats.split(",")
        )
    (
        cost.groupby(["backend", "tier"], as_index=False)[metrics]
        .median()
        .to_csv(out / "S5_values_median_by_tier.csv", index=False)
    )
    _s5_two_arms(ctx, runs, out)


S5_ARMS = ("valis_high_micro2", "tiled_high_s64")


def _s5_two_arms(ctx: Ctx, runs: pd.DataFrame, out: Path) -> None:
    """The figure: the two high arms side by side, per patient and per phase."""
    from benchmarks.analysis.lib import plotting, quality

    spec = ctx.opt("S5", default={}) or {}
    arms = [str(a) for a in (spec.get("arms") or S5_ARMS)]
    per = quality.registration_cost_by_patient(runs, ctx.root)
    per.to_csv(out / "S5_cost_by_patient.csv", index=False)
    have = set(per["run_id"]) if len(per) else set()
    missing = [a for a in arms if a not in have]
    if missing:
        print(
            f"[supp] S5: no finished registration task in the trace of {missing}; "
            f"S5_cost_high is drawn from {[a for a in arms if a in have]}",
            file=sys.stderr,
        )
    sub = pd.concat([per[per["run_id"] == a] for a in arms if a in have] or [per[:0]])
    sub = sub[sub["n_slides"].notna()]
    if sub.empty:
        return
    labels = {
        a: f"{TITLE.get(str(m), str(m))}\n({a})"
        for a, m in zip(sub["run_id"], sub["backend"])
    }
    fig = plotting.cost_two_arms(sub, quality.PHASE_ORDER, labels)
    plotting.save_fig(fig, out / "S5_cost_high", formats=ctx.formats.split(","))
    tot = (
        sub.groupby(["run_id", "backend", "patient_id"])
        .agg(
            core_h_per_slide=("cpu_hours_per_slide", "sum"),
            core_h=("cpu_hours", "sum"),
            core_h_used=("cpu_hours_used", lambda s: s.sum(min_count=1)),
            peak_rss_gb=("peak_rss_gb", "max"),
            failed=("n_failed_attempts", "sum"),
        )
        .reset_index()
    )
    rows = []
    for (arm, backend), g in tot.groupby(["run_id", "backend"], sort=False):
        phase = sub[sub["run_id"] == arm].groupby("phase")["cpu_hours"].sum()
        rows.append(
            {
                "arm": arm,
                "backend": backend,
                "n_patients": len(g),
                "median_reserved_core_h_per_slide": g["core_h_per_slide"].median(),
                "median_peak_rss_gb": g["peak_rss_gb"].median(),
                # of the cores reserved, the fraction measured busy (NaN: no %cpu field)
                "used_over_reserved": g["core_h_used"].sum(min_count=1)
                / g["core_h"].sum(),
                "failed_attempts_not_counted": int(g["failed"].sum()),
                **{
                    f"share_{p}": float(phase[p] / phase.sum())
                    for p in quality.PHASE_ORDER
                    if p in phase.index
                },
            }
        )
    pd.DataFrame(rows).to_csv(out / "S5_values_high.csv", index=False)


# ----------------------------------------------------------------------------- S6 --
def fig_s6(ctx: Ctx, patients: list[str]):
    """Nuclear and whole-cell masks per backend on the same regions + pairwise Dice."""
    spec = ctx.opt("S6", default={}) or {}
    seg = ctx.plan[ctx.plan["arm_kind"] == "segmentation"]
    methods = [
        (str(r["seg_method"]), str(r["arm"]))
        for _, r in seg.iterrows()
        if (ctx.arm_dir(r["arm"]) / "csv" / "segmented.csv").is_file() or ctx.dry_run
    ]
    if not methods:
        print("[supp] S6: no finished segmentation arm", file=sys.stderr)
        return
    pid = str(spec.get("patient") or patients[0])
    field_um = spec.get("field_um", 150)
    crop_px = spec.get("crop_px", 768)
    out = ctx.out / "S6"
    n_regions = int(spec.get("regions", 4))
    rois = [str(r) for r in spec.get("rois", []) or []]
    if len(rois) < n_regions:
        # Auto regions: picked ONCE on the reference (every backend segments the same
        # reference slide, so the canvas is shared) and reused by every backend and mask
        # below -- the panels of one region differ only by the segmenter.
        anchor = out / "_anchor"
        ctx.render(
            "reg_zoom",
            [
                ctx.arm_dir(methods[0][1]),
                "--patient",
                pid,
                "--field-um",
                field_um,
                "--pick-rois",
                n_regions,
                "-o",
                anchor,
            ],
        )
        rj = anchor / f"{pid}_rois.json"
        if rj.is_file():
            rois += [f"{r['y']},{r['x']}" for r in json.loads(rj.read_text())["rois"]]
        elif ctx.dry_run:
            rois += [f"<roi {k}>" for k in range(len(rois) + 1, n_regions + 1)]
        rois = list(dict.fromkeys(rois))[:n_regions]
    if not ctx.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "region": k,
                    "patient": pid,
                    "y_px": r.split(",")[0],
                    "x_px": r.split(",")[1],
                    "field_um": field_um,
                }
                for k, r in enumerate(rois, 1)
            ]
        ).to_csv(out / "S6_regions.csv", index=False)
    for k, roi in enumerate(rois, 1):
        for method, arm in methods:
            for mask in ("nuclei", "cell", "both"):
                ctx.render(
                    "reg_zoom",
                    [
                        ctx.arm_dir(arm),
                        "--patient",
                        pid,
                        "--roi",
                        roi,
                        "--field-um",
                        field_um,
                        "--mask",
                        mask,
                        "--crop",
                        "only",
                        "--crop-px",
                        crop_px,
                        "--title",
                        method,
                        "--formats",
                        "png",
                        "-o",
                        out / f"r{k}" / f"{method}_{mask}",
                    ],
                )
    if ctx.dry_run:
        return
    _compose_s6(ctx, out, pid, [m for m, _ in methods], len(rois))


def agreement_matrix(agree, methods=None, value="foreground_dice", patient=None):
    """The backend x backend matrix of ``value``: the median over patients, or one
    patient's own values with ``patient``. Diagonal 1; a pair with no row is NaN.
    Returns (matrix, number of patients behind it)."""
    import numpy as np

    need = {"method_a", "method_b", value}
    if agree is None or agree.empty or not need <= set(agree.columns):
        methods = list(methods or [])
        mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
        return mat, 0
    if patient is not None and "patient_id" in agree.columns:
        agree = agree[agree["patient_id"].astype(str) == str(patient)]
    methods = list(methods or sorted(set(agree["method_a"]) | set(agree["method_b"])))
    mat = pd.DataFrame(np.nan, index=methods, columns=methods)
    for m in methods:
        mat.loc[m, m] = 1.0
    vals = pd.to_numeric(agree[value], errors="coerce")
    for (a, b), v in (
        vals.groupby([agree["method_a"], agree["method_b"]]).median().items()
    ):
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = mat.loc[b, a] = v
    n = agree["patient_id"].nunique() if "patient_id" in agree.columns else len(agree)
    return mat, int(n)


def _draw_agreement(ax, mat, title, fontsize=7):
    import numpy as np

    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), vmin=0, vmax=1, cmap="viridis")
    names = list(mat.index)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(range(len(names)), names, fontsize=fontsize)
    for i in range(len(names)):
        for j in range(len(names)):
            v = arr[i, j]
            if np.isfinite(v):
                # white on the dark end of viridis, black on the yellow end
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="k" if v > 0.7 else "w",
                    fontsize=fontsize,
                )
    ax.set_title(title, fontsize=fontsize + 1)
    return im


def agreement_heatmap(
    agree,
    stem,
    formats="png,pdf",
    dpi=300,
    methods=None,
    value="foreground_dice",
    patient=None,
):
    """S6's pairwise-agreement matrix as a figure of its own: <stem>.<fmt>. ``agree`` is
    the S6_pairwise_agreement.csv frame (or its path). Nothing is written when the
    table has no usable row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not isinstance(agree, pd.DataFrame):
        agree = pd.read_csv(agree, dtype={"patient_id": str})
    mat, n = agreement_matrix(agree, methods, value, patient)
    if not n or mat.empty:
        return []
    label = {"foreground_dice": "pairwise Dice"}.get(value, value.replace("_", " "))
    scope = f"patient {patient}" if patient is not None else f"median of {n} section(s)"
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = _draw_agreement(ax, mat, f"{label}, whole section\n({scope})", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    written = []
    for fmt in str(formats).split(","):
        path = Path(f"{stem}.{fmt.strip()}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def _compose_s6(ctx, out: Path, pid: str, methods: list[str], n_regions: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from benchmarks.analysis.lib import quality

    agree = quality.segmentation_agreement(ctx.root, ctx.plan_csv)
    agree.to_csv(out / "S6_pairwise_agreement.csv", index=False)
    mat, n_sections = agreement_matrix(agree, methods)
    # The matrix alone, for a figure laid out by hand: over the cohort, and for the one
    # section the crops are drawn from (what a legend saying "this section" needs).
    agreement_heatmap(
        agree, out / "S6_pairwise_agreement", ctx.formats, ctx.dpi, methods
    )
    agreement_heatmap(
        agree,
        out / f"S6_pairwise_agreement_{pid}",
        ctx.formats,
        ctx.dpi,
        methods,
        patient=pid,
    )
    for layout in ("nuclei_cell", "both"):
        masks = ("nuclei", "cell") if layout == "nuclei_cell" else ("both",)
        ncol = len(methods) * len(masks) + 1
        fig = plt.figure(figsize=(2.6 * ncol, 2.7 * n_regions))
        gs = fig.add_gridspec(n_regions, ncol)
        for k in range(1, n_regions + 1):
            c = 0
            for m in methods:
                for mask in masks:
                    ax = fig.add_subplot(gs[k - 1, c])
                    c += 1
                    crops = sorted(
                        (out / f"r{k}" / f"{m}_{mask}").glob(f"{pid}_crop.png")
                    )
                    if crops:
                        ax.imshow(mpimg.imread(crops[0]))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if k == 1:
                        ax.set_title(f"{m}\n{mask}", fontsize=8)
                    if c == 1:
                        ax.set_ylabel(f"region {k}", fontsize=8)
        ax = fig.add_subplot(gs[:, -1])
        scope = f"median of {n_sections}" if n_sections else "whole section"
        im = _draw_agreement(ax, mat, f"pairwise Dice\n({scope})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        for fmt in ctx.formats.split(","):
            fig.savefig(out / f"S6_{layout}.{fmt}", dpi=ctx.dpi)
        plt.close(fig)


# ----------------------------------------------------------------------------- S8 --
def _pair_labels(ctx: Ctx, picks: pd.DataFrame) -> dict[str, str]:
    """moving-slide name -> its panel (channel set), from every picked arm's checkpoint.
    VALIS names a moving slide by file stem, the manifest backends by channel set."""
    labels: dict[str, str] = {}
    for arm in set(picks["arm"]):
        p = ctx.arm_dir(arm) / "csv" / "registered.csv"
        if not p.is_file():
            continue
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r.get("channels") or "").replace("|", "_")
                for name in (r.get("id"), key, f"{r.get('patient_id')}_{key}"):
                    if name:
                        labels[name] = key
    return labels


def fig_s8(ctx: Ctx, picks: pd.DataFrame, final: pd.DataFrame):
    """Dice and displacement per arm, grouped by case and by panel pair."""
    if ctx.dry_run:
        print("[dry-run] S8: Dice/displacement by case and by panel pair")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ctx.out / "S8"
    out.mkdir(parents=True, exist_ok=True)
    if final.empty:
        print("[supp] S8: no reg_qc=2 scorer JSON on disk", file=sys.stderr)
        return
    pairs = _pair_labels(ctx, picks)
    df = final.copy()
    df["panel_pair"] = df["moving"].map(lambda m: pairs.get(str(m), str(m)))
    df.to_csv(out / "S8_values_per_slide.csv", index=False)
    for set_name, methods in method_sets(ctx, picks, min_methods=1).items():
        for config in configs(ctx):
            arms = {
                a: _label(m, "", config)
                for m in methods
                if (a := arm_for(picks, m, config))
            }
            sub = df[df["run_id"].isin(arms)].assign(
                method=lambda d: d["run_id"].map(arms)
            )
            if sub.empty:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
            for row, (metric, ylab) in enumerate(
                (
                    ("dice_matched", "matched-pair Dice"),
                    ("displacement_um_p50", "centroid displacement (µm, median)"),
                )
            ):
                for col, (group, xlab) in enumerate(
                    (
                        ("patient_id", "case"),
                        ("panel_pair", "panel pair (moving vs reference)"),
                    )
                ):
                    ax = axes[row, col]
                    groups = sorted(sub[group].astype(str).unique())
                    ms = list(arms.values())
                    w = 0.8 / max(len(ms), 1)
                    for i, mname in enumerate(ms):
                        data = [
                            sub[
                                (sub[group].astype(str) == g) & (sub["method"] == mname)
                            ][metric]
                            .dropna()
                            .to_numpy()
                            for g in groups
                        ]
                        pos = [
                            j + (i - (len(ms) - 1) / 2) * w for j in range(len(groups))
                        ]
                        bp = ax.boxplot(
                            [d if len(d) else [float("nan")] for d in data],
                            positions=pos,
                            widths=w * 0.9,
                            patch_artist=True,
                            showfliers=False,
                        )
                        for b in bp["boxes"]:
                            b.set_facecolor(f"C{i}")
                            b.set_alpha(0.6)
                        for j, d in enumerate(data):
                            ax.plot([pos[j]] * len(d), d, ".", color=f"C{i}", ms=3)
                        ax.plot([], [], "s", color=f"C{i}", label=mname)
                    ax.set_xticks(
                        range(len(groups)), groups, rotation=45, ha="right", fontsize=7
                    )
                    ax.set_ylabel(ylab, fontsize=8)
                    if row == 1:
                        ax.set_xlabel(xlab, fontsize=8)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=7)
            fig.suptitle(
                f"(a) by case   (b) by panel pair — {set_name} set, {config}",
                fontsize=9,
            )
            fig.tight_layout()
            d = out / f"{set_name}_{config}"
            d.mkdir(exist_ok=True)
            for fmt in ctx.formats.split(","):
                fig.savefig(d / f"S8_{set_name}_{config}.{fmt}", dpi=ctx.dpi)
            plt.close(fig)
            (
                sub.groupby(["method", "patient_id"])[
                    ["dice_matched", "displacement_um_p50"]
                ]
                .median()
                .to_csv(d / "S8_values_by_case.csv")
            )
            (
                sub.groupby(["method", "panel_pair"])[
                    ["dice_matched", "displacement_um_p50"]
                ]
                .median()
                .to_csv(d / "S8_values_by_panel_pair.csv")
            )


# ----------------------------------------------------------------------------- S3 --
def fig_s3a(ctx: Ctx):
    """Per-round DAPI retention: the median over cells of `nuclear_retention_raw`, per case.

    `nuclear_retention_raw` (bin/cell_qc.py) is a cell's DAPI in that round over its DAPI
    in the reference round -- NOT the `QC: Nuclear retention` key, which is re-centred on
    each round's median and so reads 1.0 in every round by construction and could never
    show a decline. The normalisation is therefore "per cell, relative to the reference
    round", which is what the legend should state."""
    if ctx.dry_run:
        print("[dry-run] S3a: per-round retention from */quantification/*_round_qc.csv")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = ctx.opt("S3", default={}) or {}
    seg = ctx.plan[ctx.plan["arm_kind"].isin(["segmentation", "compute"])]
    arms = [spec["from_arm"]] if spec.get("from_arm") else list(seg["arm"])
    files = []
    for arm in arms:
        files = sorted(ctx.arm_dir(arm).glob("*/quantification/*_round_qc.csv"))
        if files:
            break
    out = ctx.out / "S3"
    out.mkdir(parents=True, exist_ok=True)
    if not files:
        print(
            "[supp] S3a: no *_round_qc.csv under any full-pipeline arm", file=sys.stderr
        )
        return
    rows = []
    for f in files:
        d = pd.read_csv(f)
        if "nuclear_retention_raw" not in d.columns:
            continue
        pid = f.name[: -len("_round_qc.csv")]
        if ctx.cohort and pid not in ctx.cohort:
            continue
        g = d.groupby("round_id", sort=False)["nuclear_retention_raw"]
        for rid, v in g:
            rows.append(
                {
                    "patient_id": pid,
                    "round_id": str(rid),
                    "median": v.median(),
                    "q25": v.quantile(0.25),
                    "q75": v.quantile(0.75),
                    "n_cells": int(v.notna().sum()),
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "S3a_values.csv", index=False)
    if tab.empty:
        return
    order = [str(r) for r in spec.get("round_order") or []] or list(
        dict.fromkeys(tab["round_id"])
    )
    x = {r: i for i, r in enumerate(order)}
    tab = tab[tab["round_id"].isin(x)]
    for style in ("lines", "box"):
        fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(order) + 2), 3.2))
        if style == "lines":
            for i, (pid, g) in enumerate(tab.groupby("patient_id")):
                g = g.assign(_x=g["round_id"].map(x)).sort_values("_x")
                ax.plot(
                    g["_x"],
                    g["median"],
                    "-o",
                    ms=3,
                    lw=1,
                    color=f"C{i % 10}",
                    label=pid,
                )
            ax.legend(fontsize=6, ncol=2, title="case", title_fontsize=6)
        else:
            data = [tab[tab["round_id"] == r]["median"].to_numpy() for r in order]
            ax.boxplot(data, positions=range(len(order)), showfliers=False)
            for i, d in enumerate(data):
                ax.plot([i] * len(d), d, ".", color="k", ms=3)
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("DAPI retention\n(cell / reference round, median)", fontsize=8)
        ax.set_xlabel("round", fontsize=8)
        ax.set_title(f"n = {tab['patient_id'].nunique()} cases", fontsize=8)
        fig.tight_layout()
        for fmt in ctx.formats.split(","):
            fig.savefig(out / f"S3a_retention_{style}.{fmt}", dpi=ctx.dpi)
        plt.close(fig)


# ----------------------------------------------------------------------------- S2 --
def fig_s2(ctx: Ctx):
    """Secondary-only controls, one crop per (acquisition, channel), at ONE contrast.

    The claim is ABSENCE of signal, so every panel of a channel shares a pinned black and
    white point (`vmin`/`vmax`, or the first stripped acquisition's clean autoscale) --
    re-stretching each crop would amplify background into apparent carry-over. The
    acquisitions are listed in a checkpoint-shaped CSV (S2.csv: patient_id, id,
    registered_image, channels, is_reference), because they are extra acquisitions the
    arms never registered."""
    spec = ctx.opt("S2", default={}) or {}
    if not spec.get("csv"):
        print(
            "[supp] S2: SKIPPED (no S2.csv of secondary-only acquisitions in the config)"
        )
        return
    out = ctx.out / "S2"
    channels = spec.get("channels") or ["DAPI"]
    for auto in spec.get("autoscale", ["pinned", "clean"]):
        for ch in channels:
            args = [
                Path(spec["csv"]).parent,
                "--csv",
                spec["csv"],
                "--channel",
                ch,
                "--field-um",
                spec.get("field_um", 300),
                "--crop-px",
                spec.get("crop_px", 768),
                "--formats",
                "png",
                "-o",
                out / auto / ch,
            ]
            if spec.get("roi"):
                args += ["--roi", spec["roi"]]
            if auto == "pinned":
                lim = (spec.get("limits") or {}).get(ch)
                if not lim:
                    continue  # no pinned limits for this channel: only the autoscale row
                args += ["--vmin", lim[0], "--vmax", lim[1]]
            else:
                args += ["--autoscale", auto]
            for pid in spec.get("patients") or [None]:
                ctx.render("reg_crop", args + (["--patient", pid] if pid else []))


# ------------------------------------------------------------------------ gallery --
def _slide_channels(arm_dir: Path, pid: str) -> list[str]:
    """Every channel of one patient's registered slides, reference first, deduplicated."""
    p = arm_dir / "csv" / "registered.csv"
    if not p.is_file():
        return []
    rows = [r for r in csv.DictReader(p.open()) if str(r.get("patient_id")) == pid]
    rows.sort(key=lambda r: str(r.get("is_reference", "")).lower() != "true")
    out: list[str] = []
    for r in rows:
        for ch in (r.get("channels") or "").split("|"):
            if ch.strip() and ch.strip() not in out:
                out.append(ch.strip())
    return out


def _pick_regions(ctx: Ctx, seg_arm: str, pid: str, field, n: int, out: Path) -> list:
    """N separated regions of `field` um, picked ONCE per patient on the reference."""
    ctx.render(
        "reg_zoom",
        [
            ctx.arm_dir(seg_arm),
            "--patient",
            pid,
            "--field-um",
            field,
            "--pick-rois",
            n,
            "-o",
            out,
        ],
    )
    rj = out / f"{pid}_rois.json"
    if rj.is_file():
        return [f"{r['y']},{r['x']}" for r in json.loads(rj.read_text())["rois"]]
    return [f"<roi {k}>" for k in range(1, n + 1)] if ctx.dry_run else []


def fig_gallery(ctx: Ctx, picks: pd.DataFrame, patients: list[str]) -> None:
    """The image gallery, for EVERY patient, drawn from the arms already on disk:

    overlay/<arm>/f<field>_z<zoom>/v<k>/       Before | After | locator, per moving round,
                                               every registration method's high arm on the
                                               SAME crop (one anchor pick per round x variant)
    zoom/<backend>/f<field>_<mask>/r<k>/       overview + outlined zoom, per segmentation
    crops/<backend>/f<field>_p<px>_<mask>/r<k>/  backend, the outlined crop alone; every
                                               backend at the SAME regions
    crops/channels/f<field>_p<px>_<scale>/r<k>/  every channel of the patient, at the same
                                               regions as the segmentation crops
    (the mosaic takes several patch sizes itself: mosaic.patch_um)

    Nothing is registered or segmented; each file is one renderer call."""
    g = ctx.opt("gallery", default={}) or {}
    ov = g.get("overlay") or {}
    zm = g.get("zoom") or {}
    ch = g.get("channels") or {}
    methods = [
        m for m in (g.get("methods") or REG_METHODS) if arm_for(picks, m, "high")
    ]
    reg = [(m, arm_for(picks, m, "high")) for m in methods]
    seg = ctx.plan[ctx.plan["arm_kind"] == "segmentation"]
    backends = [
        (str(r["seg_method"]), str(r["arm"]))
        for _, r in seg.iterrows()
        if (ctx.arm_dir(r["arm"]) / "csv" / "segmented.csv").is_file() or ctx.dry_run
    ]
    out = ctx.out / "gallery"
    fields_ov = ov.get("field_um", [500, 2000])
    zoom_um = ov.get("zoom_um", 60)
    variants = int(ov.get("variants", 2))
    numbers = ov.get("numbers", "auto")
    fields_z = zm.get("field_um", [150, 300])
    masks = zm.get("masks", ["both", "cell", "nuclei"])
    n_regions = int(zm.get("regions", 3))
    zoom_crop_px = zm.get("crop_px", 1024)
    ch_field = ch.get("field_um", 150)
    ch_px = ch.get("crop_px", 1024)
    scales = ch.get("autoscale", ["clean", "percentile"])
    ref_arm = arm_for(picks, "valis", "high") or (reg[0][1] if reg else None)
    for pid in patients:
        # ---- registration overlays: one anchor pick, every arm on the same crop ----
        if reg:
            for field in fields_ov:
                tag = f"f{field}_z{zoom_um}"
                anchor = out / "_anchor" / "overlay" / tag
                ctx.render(
                    "reg_overlay",
                    [
                        ctx.arm_dir(reg[0][1]),
                        "--patient",
                        pid,
                        "--field-um",
                        field,
                        "--zoom-um",
                        zoom_um,
                        "--variants",
                        variants,
                        "--numbers",
                        numbers,
                        "--formats",
                        "png",
                        "-o",
                        anchor,
                    ],
                )
                mans = [
                    json.loads(mf.read_text())
                    for mf in sorted(anchor.glob(f"{pid}_*_overlay.json"))
                ]
                mans = [mm for mm in mans if str(mm.get("patient", pid)) == pid]
                for man in mans:
                    v = int(man.get("variant", 1))
                    pin = [f"--roi={man['crop']['y']},{man['crop']['x']}"]
                    pin += ["--field-px", man["crop"]["size_px"]]
                    if man.get("zoom"):
                        pin += [f"--zoom-roi={man['zoom']['y']},{man['zoom']['x']}"]
                    for m, arm in reg:
                        ctx.render(
                            "reg_overlay",
                            [
                                ctx.arm_dir(arm),
                                "--patient",
                                pid,
                                "--rounds",
                                man["round"],
                                "--field-um",
                                field,
                                "--zoom-um",
                                zoom_um,
                                *pin,
                                "--numbers",
                                numbers,
                                "--title",
                                f"{TITLE[m]} (high)",
                                "--formats",
                                ctx.formats,
                                "--dpi",
                                ctx.dpi,
                                "-o",
                                out / "overlay" / arm / tag / f"v{v}",
                            ],
                        )
        # ---- segmentation: N regions per field, every backend x mask there ----
        regions_by_field: dict = {}
        if backends:
            for field in fields_z:
                rois = _pick_regions(
                    ctx,
                    backends[0][1],
                    pid,
                    field,
                    n_regions,
                    out / "_anchor" / "zoom" / f"f{field}",
                )
                regions_by_field[field] = rois
                for k, roi in enumerate(rois, 1):
                    for method, arm in backends:
                        for mask in masks:
                            base = [
                                ctx.arm_dir(arm),
                                "--patient",
                                pid,
                                "--roi",
                                roi,
                                "--field-um",
                                field,
                                "--mask",
                                mask,
                                "--title",
                                method,
                                "--formats",
                                ctx.formats,
                            ]
                            ctx.render(
                                "reg_zoom",
                                [
                                    *base,
                                    "--crop",
                                    "also",
                                    "-o",
                                    out
                                    / "zoom"
                                    / method
                                    / f"f{field}_{mask}"
                                    / f"r{k}",
                                ],
                            )
                            ctx.render(
                                "reg_zoom",
                                [
                                    *base,
                                    "--crop",
                                    "only",
                                    "--crop-px",
                                    zoom_crop_px,
                                    "-o",
                                    out
                                    / "crops"
                                    / method
                                    / f"f{field}_p{zoom_crop_px}_{mask}"
                                    / f"r{k}",
                                ],
                            )
        # ---- every channel, at the segmentation regions (else one auto region) ----
        ch_arm = ch.get("arm") or ref_arm  # whose registered slides are cropped
        if ch_arm:
            chans = ch.get("names") or _slide_channels(ctx.arm_dir(ch_arm), pid)
            rois = regions_by_field.get(ch_field) or [None]
            for k, roi in enumerate(rois, 1):
                for scale in scales:
                    for name in chans:
                        args = [
                            ctx.arm_dir(ch_arm),
                            "--patient",
                            pid,
                            "--channel",
                            name,
                            "--field-um",
                            ch_field,
                            "--crop-px",
                            ch_px,
                            "--autoscale",
                            scale,
                            "--colors",
                            "white",
                            "--title",
                            "",
                            "--formats",
                            ctx.formats,
                            "-o",
                            out
                            / "crops"
                            / "channels"
                            / f"f{ch_field}_p{ch_px}_{scale}"
                            / f"r{k}",
                        ]
                        if roi:
                            args += ["--roi", roi]
                        ctx.render("reg_crop", args)
                        if roi is None and not ctx.dry_run:
                            # pin the rest to the first channel's auto pick: same tissue
                            cj = (
                                out
                                / "crops"
                                / "channels"
                                / f"f{ch_field}_p{ch_px}_{scale}"
                                / f"r{k}"
                                / f"{pid}_{name}_crop.json"
                            )
                            if cj.is_file():
                                c = json.loads(cj.read_text())["crop"]
                                roi = f"{c['y']},{c['x']}"
                                rois[k - 1] = roi


# -------------------------------------------------------------------------- check --
IHC_CHECK_CSV = Path("output") / "figures" / "supplementary" / "check_ihc.csv"


def check(ctx: Ctx, picks: pd.DataFrame, final: pd.DataFrame, ihc: Path | None):
    """What each figure would be drawn FROM, without drawing it: READY / PARTIAL /
    MISSING per figure, plus the legend items still left to the authors (TODO rows).
    Written to OUT/check.csv on every run."""
    rows: list[dict] = []

    def add(fig, status, detail):
        rows.append({"figure": fig, "status": status, "detail": detail})

    cfgs = configs(ctx)
    high = {m: arm_for(picks, m, "high") for m in REG_METHODS}
    have = {m: a for m, a in high.items() if a}
    scored = (
        final.groupby("run_id")["patient_id"].nunique().to_dict()
        if not final.empty
        else {}
    )
    tier_note = (
        "configs "
        + "+".join(cfgs)
        + "; high = "
        + ", ".join(f"{m}:{a}" for m, a in have.items())
    )
    missing = [m for m in REG_METHODS if m not in have]
    reg_status = "READY" if len(have) >= 2 else "MISSING"
    if reg_status == "READY" and missing:
        reg_status = "PARTIAL"
    reg_detail = tier_note + (f"; no high arm for {missing}" if missing else "")
    add("mosaic", reg_status, reg_detail)
    patients = _patients_of(ctx.arm_dir(_anchor_arm(picks))) if len(picks) else []
    if ctx.cohort:
        add(
            "cohort",
            "READY" if ctx.cohort <= set(patients) else "PARTIAL",
            f"{len(ctx.cohort)} case(s) from "
            f"{'the config' if ctx.opt('patients') else 'the --input samplesheet'}; "
            "on disk "
            f"{sorted(ctx.cohort & set(patients))}; absent "
            f"{sorted(ctx.cohort - set(patients))}; left out "
            f"{sorted(set(patients) - ctx.cohort)}",
        )
        patients = [p for p in patients if p in ctx.cohort]
    else:
        add("cohort", "ALL", f"patients: [] -> every case on disk: {patients}")
    s4 = ctx.opt("S4", default={}) or {}
    s4_pid = str(s4.get("patient") or (patients[0] if patients else ""))
    # S4 compares VALIS with STARE; S7 and S8 need one method.
    s4_ok = "valis" in have and "stare" in have
    add(
        "S4",
        ("READY" if len(have) == len(REG_METHODS) else "PARTIAL")
        if s4_ok and s4_pid
        else "MISSING",
        f"case {s4_pid or '?'}; needs VALIS + STARE; {reg_detail}",
    )
    s7 = [p for p in patients if p != s4_pid]
    add(
        "S7",
        ("READY" if len(have) == 1 or not missing else "PARTIAL")
        if s7 and have
        else "MISSING",
        f"{len(s7)} other case(s): {s7}; methods {list(have)}",
    )
    n_scored = {m: scored.get(a, 0) for m, a in have.items()}
    s8 = "READY" if any(n_scored.values()) else "MISSING"
    if s8 == "READY" and not all(n_scored.values()):
        s8 = "PARTIAL"
    add("S8", s8, f"reg_qc=2 scorer cases per high arm {n_scored} ({'+'.join(cfgs)})")

    # S5: every tier by design (the legend names the three cost tiers).
    try:
        from benchmarks.analysis.lib import load, quality

        cost = quality.registration_cost_by_tier(
            load.load_runs(ctx.root, ctx.plan_csv), ctx.root
        )
        tiers = cost.groupby("backend")["tier"].nunique().to_dict() if len(cost) else {}
        s5 = (
            "MISSING"
            if cost.empty
            else ("READY" if all(v >= 3 for v in tiers.values()) else "PARTIAL")
        )
        add("S5", s5, f"traced tiers per backend {tiers} (all three tiers, by design)")
    except Exception as exc:  # the check must not die on one unreadable input
        add("S5", "MISSING", f"could not read the traces: {exc}")

    seg = ctx.plan[ctx.plan["arm_kind"] == "segmentation"]
    done = [
        f"{r['seg_method']}<-{r.get('from_arm', '')}"
        for _, r in seg.iterrows()
        if (ctx.arm_dir(r["arm"]) / "csv" / "segmented.csv").is_file()
    ]
    add(
        "S6",
        "READY" if len(done) >= 3 else ("PARTIAL" if len(done) >= 2 else "MISSING"),
        f"{len(done)}/{len(seg)} segmentation arms finished: {done}",
    )

    s3 = ctx.opt("S3", default={}) or {}
    s3_arms = (
        [s3["from_arm"]]
        if s3.get("from_arm")
        else list(
            ctx.plan[ctx.plan["arm_kind"].isin(["segmentation", "compute"])]["arm"]
        )
    )
    s3_files = []
    for arm in s3_arms:
        fs = sorted(ctx.arm_dir(arm).glob("*/quantification/*_round_qc.csv"))
        fs = [f for f in fs if "nuclear_retention_raw" in f.open().readline()]
        if fs:
            s3_files = fs
            add(
                "S3a", "READY", f"{len(fs)} case(s) with nuclear_retention_raw in {arm}"
            )
            break
    if not s3_files:
        add(
            "S3a",
            "MISSING",
            "no *_round_qc.csv with nuclear_retention_raw "
            "(needs a run on fdf042c0 or later)",
        )

    g = ctx.opt("gallery", default={}) or {}
    n_seg = len(done)
    n_fields_ov = len((g.get("overlay") or {}).get("field_um", [500, 2000]))
    n_var = int((g.get("overlay") or {}).get("variants", 2))
    n_fields_z = len((g.get("zoom") or {}).get("field_um", [150, 300]))
    n_masks = len((g.get("zoom") or {}).get("masks", ["both", "cell", "nuclei"]))
    n_reg_z = int((g.get("zoom") or {}).get("regions", 3))
    n_scales = len((g.get("channels") or {}).get("autoscale", ["clean", "percentile"]))
    per_pid = []
    for pid in patients:
        rounds = max(len(_slide_channels(ctx.arm_dir(_anchor_arm(picks)), pid)) - 1, 1)
        n = n_fields_ov * n_var * rounds * max(len(have), 1)
        n += n_fields_z * n_reg_z * n_seg * n_masks * 2
        n += (
            n_reg_z
            * n_scales
            * len(_slide_channels(ctx.arm_dir(_anchor_arm(picks)), pid))
        )
        per_pid.append(n)
    add(
        "gallery",
        "READY" if have and patients else "MISSING",
        f"{len(patients)} case(s) x {list(have)} arms x {n_seg} seg backend(s): "
        f"~{sum(per_pid)} renders (overlay per round, zoom/crops per backend x mask, "
        "crops per channel)",
    )

    s2 = ctx.opt("S2", default={}) or {}
    add(
        "S2",
        "READY" if s2.get("csv") and Path(s2["csv"]).is_file() else "MISSING",
        f"S2.csv = {s2.get('csv')}",
    )

    ihc_rows = {}
    if ihc is not None and (ihc / IHC_CHECK_CSV).is_file():
        for r in csv.DictReader((ihc / IHC_CHECK_CSV).open()):
            ihc_rows[r["figure"]] = r
    for fig, need in (
        ("S3b", None),
        ("S9", None),
        ("S10", Path("data") / "clinical_data.xlsx"),
        ("S11", Path("output") / "paired_deconv.rds"),
    ):
        if fig in ihc_rows:
            add(
                fig,
                ihc_rows[fig]["status"],
                "supplementary.R --check: " + ihc_rows[fig]["detail"],
            )
        elif ihc is None:
            add(fig, "NOT CHECKED", "pass --ihc <ihc_method checkout>")
        elif not (ihc / "figures" / "_common.R").is_file():
            add(fig, "MISSING", f"{ihc} is not an ihc_method checkout")
        elif need is not None and not (ihc / need).is_file():
            add(fig, "MISSING", f"{ihc / need} absent")
        else:
            add(
                fig,
                "PARTIAL",
                "files present; arm cells unverified "
                "(run `Rscript benchmarks/ihc/supplementary.R --check` in the checkout)",
            )

    # The legends' AUTHORS TO SUPPLY items that live in the config, still unset.
    for key, val, what in (
        ("S4.patient", s4.get("patient"), "the case of Figure 2b"),
        ("S4.rounds", s4.get("rounds"), "the two panels of Figure 2b"),
        (
            "S3.round_order",
            s3.get("round_order"),
            "the acquisition order of the rounds",
        ),
        ("S6.patient", (ctx.opt("S6", default={}) or {}).get("patient"), "S6's case"),
    ):
        if not val:
            add("TODO", "AUTHORS", f"{key} unset in the config: {what}")

    tab = pd.DataFrame(rows, columns=["figure", "status", "detail"])
    ctx.out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(ctx.out / "check.csv", index=False)
    with pd.option_context("display.max_colwidth", 110, "display.width", 200):
        print(tab.to_string(index=False))
    return tab


# -------------------------------------------------------------------------- index --
def write_index(out: Path) -> Path:
    """One page listing every PNG under OUT, grouped by figure, to pick variants."""
    parts = [
        "<!doctype html><meta charset=utf-8><title>Supplementary figures</title>",
        "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
        "figure{display:inline-block;margin:6px;vertical-align:top;max-width:420px}"
        "img{max-width:420px;border:1px solid #ccc}figcaption{font-size:11px;"
        "word-break:break-all}h2{border-top:2px solid #333;padding-top:8px}</style>",
        "<h1>Supplementary figures — every variant</h1>",
    ]
    picks = out / "picks.csv"
    if picks.is_file():
        parts.append(
            "<h2>Arm picks</h2><pre>" + html.escape(picks.read_text()) + "</pre>"
        )
    for fig in (
        "mosaic",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
        "gallery",
    ):
        d = out / fig
        if not d.is_dir():
            continue
        pngs = sorted(
            p
            for p in d.rglob("*.png")
            if "_anchor" not in p.parts
            and "panels" not in p.parts
            and "_patches" not in str(p)
        )
        parts.append(f"<h2>{fig} ({len(pngs)} images)</h2>")
        for p in pngs:
            rel = p.relative_to(out).as_posix()
            parts.append(
                f'<figure><a href="{html.escape(rel)}"><img loading=lazy src="{html.escape(rel)}">'
                f"</a><figcaption>{html.escape(rel)}</figcaption></figure>"
            )
    idx = out / "index.html"
    idx.write_text("\n".join(parts))
    return idx


# --------------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True, help="the arm results root")
    ap.add_argument("--plan", type=Path, required=True, help="the FULL arm_plan.csv")
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "benchmarks/configs/supplementary.yaml",
    )
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--only", default="", help=f"comma list of {list(FIGURES)}")
    ap.add_argument(
        "--exec", default="", help="renderer command prefix (the container)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report what each figure would be drawn from (OUT/check.csv), draw nothing",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="the arms' samplesheet: its patient_id set is the cohort unless the "
        "config's `patients:` overrides it",
    )
    ap.add_argument(
        "--ihc", type=Path, default=None, help="ihc_method checkout, for S3b/S9-S11"
    )
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text()) or {}
    ctx = Ctx(
        root=a.results.resolve(),
        plan=pd.read_csv(a.plan, dtype=str).fillna(""),
        plan_csv=a.plan.resolve(),
        out=a.out.resolve(),
        cfg=cfg,
        exec_prefix=shlex.split(a.exec),
        dry_run=a.dry_run,
        input_patients=read_input_patients(a.input) if a.input else [],
    )
    only = [f.strip() for f in a.only.split(",") if f.strip()] or list(FIGURES)
    bad = sorted(set(only) - set(FIGURES))
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; known: {list(FIGURES)}")

    final = accuracy_long(ctx)
    if ctx.cohort:
        # The cohort cuts EVERY number, not only the drawn cases: the arm picks, S8's
        # boxes, S4/S7's values and S3a all see the same case set.
        on_disk = set(final["patient_id"].astype(str)) if not final.empty else set()
        absent = sorted(ctx.cohort - on_disk)
        if absent and not final.empty:
            print(f"[supp] cohort cases with no scorer JSON: {absent}", file=sys.stderr)
        if not final.empty:
            final = final[final["patient_id"].astype(str).isin(ctx.cohort)]
    picks = pick_arms(ctx, final)
    print(picks.to_string(index=False))
    if picks.empty:
        raise SystemExit("no registered arm of any method under --results")
    on_disk = _patients_of(ctx.arm_dir(_anchor_arm(picks)))
    patients = (
        [p for p in on_disk if p in ctx.cohort] + sorted(ctx.cohort - set(on_disk))
        if ctx.cohort
        else on_disk
    )
    check(ctx, picks, final, a.ihc.resolve() if a.ihc else None)
    if a.check:
        return 0
    status = {}
    for name, fn in (
        ("mosaic", lambda: fig_mosaic(ctx, picks, patients)),
        ("S4", lambda: fig_s4(ctx, picks, patients, final)),
        ("S7", lambda: fig_s7(ctx, picks, patients, final)),
        ("S5", lambda: fig_s5(ctx)),
        ("S6", lambda: fig_s6(ctx, patients)),
        ("S8", lambda: fig_s8(ctx, picks, final)),
        ("S2", lambda: fig_s2(ctx)),
        ("S3", lambda: fig_s3a(ctx)),
        ("gallery", lambda: fig_gallery(ctx, picks, patients)),
    ):
        if name not in only:
            continue
        try:
            fn()
            status[name] = "OK"
        except Exception as exc:  # one figure failing must not cost the others
            status[name] = f"FAILED: {exc}"
            print(f"[supp] {name} FAILED: {exc!r}", file=sys.stderr)
    (ctx.out / "commands.txt").write_text("\n".join(ctx.log) + "\n")
    idx = write_index(ctx.out)
    for k, v in status.items():
        print(f"  {k:7s} {v}")
    print(f"index: {idx}")
    return 0 if all(v == "OK" for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
