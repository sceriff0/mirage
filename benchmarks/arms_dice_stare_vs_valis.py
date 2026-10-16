#!/usr/bin/env python3
"""arms_dice_stare_vs_valis.py: paired matched-nucleus Dice, STARE vs VALIS, from an arm results root.

    python3 arms_dice_stare_vs_valis.py [arm_results] [--valis ARM] [--stare ARM] [--csv OUT]

    default root  /beegfs/scratch/ieo7660/ihc_method/benchmark/arm_results
    default arms  valis_high_micro2 vs tiled_high_s64 (supplementary.yaml's `high` picks)

Reads <root>/<arm>/<patient>/qc/registration/*_seg_qc.json (the reg_qc=2 scorer). Per moving
slide the headline is the `full_transform` record (the final transform paired in its own frame);
a JSON written before that record existed falls back to its most-registered stage, the same rule
benchmarks/analysis/lib/quality.py applies. `native` (unregistered) is kept as the floor.

A PANEL PAIR is (patient, moving panel vs that patient's reference). VALIS names a moving slide by
file stem and STARE by channel set, so both are mapped to the channel set through each arm's
csv/registered.csv. Only panel pairs BOTH arms scored are compared; the rest are listed.

Reports: median Dice STARE, median Dice VALIS, median paired difference (STARE - VALIS), the IQR
of each, n panel pairs, and how many pairs STARE wins. Stdlib only.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

ROOT = "/beegfs/scratch/ieo7660/ihc_method/benchmark/arm_results"
STAGE_RANK = {"native": 0, "rigid": 1, "non_rigid": 2, "refined": 2, "micro": 3}
EXTS = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".ome.zarr", ".zarr")


def num(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def stem(name):
    name = os.path.basename(str(name or ""))
    for e in EXTS:
        if name.lower().endswith(e):
            return name[: -len(e)]
    return name


def panel_names(arm_dir):
    """Every name a moving slide may carry in this arm -> its channel set (`A_B_C`)."""
    names = {}
    p = os.path.join(arm_dir, "csv", "registered.csv")
    if not os.path.isfile(p):
        return names
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r.get("channels") or "").replace("|", "_")
            if not key:
                continue
            pid = (r.get("patient_id") or "").strip()
            cands = {r.get("id"), key, f"{pid}_{key}"}
            cands |= {stem(v) for v in r.values() if v and ("/" in v or "." in v)}
            for c in cands:
                if c:
                    names[(pid, stem(c))] = key
    return names


def harvest(root, arm):
    """(patient, panel) -> {"dice", "native", "stage", "moving"} for one arm."""
    arm_dir = os.path.join(root, arm)
    if not os.path.isdir(arm_dir):
        sys.exit(
            f"no arm {arm!r} under {root}; arms there: "
            + ", ".join(sorted(os.listdir(root)))
        )
    names = panel_names(arm_dir)
    out, bad = {}, 0
    for j in sorted(
        glob.glob(os.path.join(arm_dir, "*", "qc", "registration", "*_seg_qc.json"))
    ):
        try:
            d = json.load(open(j))
        except (OSError, ValueError):
            bad += 1
            continue
        pid = str(d.get("patient_id") or j.split(os.sep)[-4])
        stages = d.get("stages") or {}
        full = d.get("full_transform") or {}
        if num(full.get("dice_matched")) is not None:
            stage, dice = "full_transform", num(full["dice_matched"])
        else:
            order = [
                s
                for s in (d.get("stage_order") or list(stages))
                if num((stages.get(s) or {}).get("dice_matched")) is not None
            ]
            if not order:
                bad += 1
                continue
            stage = max(order, key=lambda s: STAGE_RANK.get(s, -1))
            dice = num(stages[stage]["dice_matched"])
        mv = stem(d.get("moving"))
        panel = (
            names.get((pid, mv)) or names.get((pid, mv.removeprefix(f"{pid}_"))) or mv
        )
        if (pid, panel) in out:
            print(
                f"[{arm}] {pid}:{panel} scored twice ({out[(pid, panel)]['moving']}, {mv}); "
                "keeping the first",
                file=sys.stderr,
            )
            continue
        out[(pid, panel)] = {
            "dice": dice,
            "native": num((stages.get("native") or {}).get("dice_matched")),
            "stage": stage,
            "moving": mv,
        }
    if bad:
        print(f"[{arm}] {bad} unreadable or unscored JSON(s) skipped", file=sys.stderr)
    return out


def iqr(v):
    if len(v) < 2:
        return float("nan"), float("nan")
    q1, _, q3 = statistics.quantiles(v, n=4, method="inclusive")
    return q1, q3


def line(label, v, sign=""):
    q1, q3 = iqr(v)
    f = f"{sign}.3f"
    return (
        f"  {label:30s} median {statistics.median(v):{f}}  "
        f"IQR [{q1:{f}}, {q3:{f}}]  (width {q3 - q1:.3f})"
    )


ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("root", nargs="?", default=ROOT)
ap.add_argument("--valis", default="valis_high_micro2")
ap.add_argument("--stare", default="tiled_high_s64")
ap.add_argument("--csv", help="write the per-panel-pair table here")
a = ap.parse_args()

V, S = harvest(a.root, a.valis), harvest(a.root, a.stare)
both = sorted(set(V) & set(S))
for name, only in ((a.valis, set(V) - set(S)), (a.stare, set(S) - set(V))):
    if only:
        print(
            f"[{name}] {len(only)} panel pair(s) the other arm did not score: "
            + ", ".join(f"{p}:{k}" for p, k in sorted(only)),
            file=sys.stderr,
        )
if not both:
    sys.exit(
        f"no panel pair scored by both {a.valis} ({len(V)}) and {a.stare} ({len(S)})"
    )

print(
    f"{'patient':10s} {'panel':34s} {'VALIS':>7s} {'STARE':>7s} {'S-V':>7s} {'native':>7s}"
)
rows = []
for pid, panel in both:
    v, s = V[(pid, panel)], S[(pid, panel)]
    diff = s["dice"] - v["dice"]
    nat = v["native"] if v["native"] is not None else s["native"]
    rows.append(
        {
            "patient_id": pid,
            "panel": panel,
            "valis_dice": v["dice"],
            "stare_dice": s["dice"],
            "diff_stare_minus_valis": diff,
            "native_dice": nat,
            "valis_stage": v["stage"],
            "stare_stage": s["stage"],
        }
    )
    print(
        f"{pid:10s} {panel[:34]:34s} {v['dice']:7.3f} {s['dice']:7.3f} {diff:+7.3f} "
        + (f"{nat:7.3f}" if nat is not None else "      -")
    )

dv = [r["valis_dice"] for r in rows]
ds = [r["stare_dice"] for r in rows]
dd = [r["diff_stare_minus_valis"] for r in rows]
wins = sum(d > 0 for d in dd)
ties = sum(d == 0 for d in dd)
stages = sorted({r["valis_stage"] for r in rows} | {r["stare_stage"] for r in rows})
patients = len({r["patient_id"] for r in rows})

print(
    f"\n{a.stare} (STARE) vs {a.valis} (VALIS): n = {len(rows)} panel pairs, "
    f"{patients} patient(s); stage scored: {', '.join(stages)}"
)
print(line("STARE Dice", ds))
print(line("VALIS Dice", dv))
print(line("paired difference STARE-VALIS", dd, sign="+"))
print(
    f"  STARE higher in {wins}/{len(rows)} pairs" + (f" ({ties} tied)" if ties else "")
)

q1, q3 = iqr(dd)
print(
    f"\nMedian matched-nucleus Dice was {statistics.median(ds):.3f} for STARE and "
    f"{statistics.median(dv):.3f} for VALIS (median paired difference "
    f"{statistics.median(dd):+.3f}, IQR {q1:+.3f} to {q3:+.3f}; n = {len(rows)} panel pairs)."
)

if a.csv:
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"per-pair table -> {a.csv}", file=sys.stderr)
