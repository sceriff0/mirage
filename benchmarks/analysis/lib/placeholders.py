"""Opt-in SYNTHETIC placeholders for benchmark points that have not landed yet.

The arms and the sweep run for days on the cluster; until they finish, many expected
(run x patient x metric) or (run x process x metric) points are simply absent. This
module lets make_figures draw the figures anyway, so their layout can be judged early,
while making it impossible to mistake a synthetic point for a measured one.

WHAT IT IS NOT: an imputation method. The numbers it produces are noise around
neighbouring measurements, chosen to fill a plot, and carry NO information about the
run they stand in for. Nothing numeric may be read off a placeholder.

THE GUARD, in five parts -- every one of them is load-bearing:

  1. OFF by default. Only `--placeholder-missing` (or env PLACEHOLDER_MISSING=1) turns
     it on; `resolve_enabled` is the one place either is read.
  2. Real points are never touched. `Ledger.fill` only APPENDS synthetic rows, or fills
     a NaN cell of a real row; a measured value is never rewritten.
  3. Every synthetic value is marked on its row (`is_placeholder`, `placeholder_rule`)
     and listed in `<outdir>/placeholders.csv`, and the directory carries
     `PLACEHOLDER_DATA.txt` (the NO_GROUND_TRUTH.txt convention).
  4. Every figure with >= 1 synthetic point goes through `Ledger.finish_figure`, which
     stamps a large diagonal "PLACEHOLDER -- N synthetic points" watermark on it.
  5. The resource fits and modules.optimized.config are computed from real rows only
     (make_figures fits BEFORE filling), and benchmarks/pull_to_ihc_method.sh refuses a
     directory carrying PLACEHOLDER_DATA.txt -- the consumer's R figures cannot show the
     marking, so placeholders must never reach it.

SYNTHESIS, per missing (key, metric) point, first rule that has data wins:

  (a) same_run(...)   mean +- sd noise of the SAME run's completed points for that metric
                      (e.g. the other patients of that arm)
  (b) neighbours(...) the pooled points of runs sharing a family column (backend /
                      registration_method, then memory_mode tier, ...)
  (c) global(...)     every completed point of that metric (within the caller's scope,
                      e.g. per process for a per-process figure)
  (d) prior           the fixed, documented PRIORS table below

The caller names the grouping columns of (a)-(c); the rule label records them, e.g.
`neighbours(registration_method)`. A group with one point uses sd = 10% of |mean|
(a lone point has no spread to borrow). Every value is clipped to `valid_range(metric)`.

DETERMINISM: each point draws from its own generator seeded by sha256(seed, key,
metric), so a point's value depends only on the seed, its identity and the pooled
data -- not on how many OTHER points happen to be missing this time.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENV_VAR = "PLACEHOLDER_MISSING"
DEFAULT_SEED = 0
MARKER = "PLACEHOLDER_DATA.txt"
SIDECAR = "placeholders.csv"
WATERMARK = "PLACEHOLDER — {n} synthetic points"

# (d): the fixed prior, (mean, sd), per metric. Deliberately round, deliberately
# uninformative: a figure drawn from priors alone shows layout, nothing else.
PRIORS: dict[str, tuple[float, float]] = {
    "peak_rss_gb": (8.0, 4.0),
    "input_gb": (1.0, 0.5),
    "realtime_s": (1800.0, 900.0),
    "cpu_hours": (2.0, 1.0),
    "gpu_hours": (0.5, 0.25),
    "total_realtime_s": (7200.0, 3600.0),
    "dice_matched": (0.6, 0.1),
    "displacement_um_p50": (5.0, 2.0),
    "displacement_um_p90": (10.0, 4.0),
}
FALLBACK_PRIOR = (1.0, 0.5)
# a single-point group has no spread; borrow this fraction of |mean| as sd
LONE_POINT_REL_SD = 0.1


def valid_range(metric: str) -> tuple[float, float]:
    """The range a synthetic value is clipped to. Deltas are signed; ratios in [0,1];
    everything else (errors, sizes, costs, counts) non-negative."""
    m = metric.lower()
    if m.startswith("delta_") or "_delta_" in m:
        return (-math.inf, math.inf)
    if any(
        t in m for t in ("dice", "iou", "fraction", "frac_", "_f1", "bottleneck_frac")
    ):
        return (0.0, 1.0)
    return (0.0, math.inf)


def resolve_enabled(flag: bool, env: dict | None = None) -> bool:
    """The ONE place the opt-in is read: the CLI flag, or PLACEHOLDER_MISSING=1."""
    env = os.environ if env is None else env
    return bool(flag) or str(env.get(ENV_VAR, "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )


def clear_stale(outdir) -> list[Path]:
    """Remove a PREVIOUS placeholder run's sidecar, marker and watermarked figures.

    Called at the start of every make_figures run, placeholder mode or not: a default
    run into a directory that once held a preview must not leave a watermarked figure
    (or a marker claiming synthetic data) sitting beside its real output.
    """
    outdir = Path(outdir)
    removed = []
    sidecar = outdir / SIDECAR
    if sidecar.is_file():
        try:
            figs = (
                pd.read_csv(sidecar)
                .get("figure", pd.Series(dtype=str))
                .dropna()
                .unique()
            )
        except Exception:
            figs = []
        for stem in figs:
            stem = outdir / str(stem)
            for p in stem.parent.glob(stem.name + ".*"):
                p.unlink()
                removed.append(p)
    for name in (SIDECAR, MARKER):
        p = outdir / name
        if p.is_file():
            p.unlink()
            removed.append(p)
    return removed


def _point_rng(seed: int, key: tuple, metric: str) -> np.random.Generator:
    h = hashlib.sha256(
        repr((int(seed), tuple(map(str, key)), metric)).encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _draw(values: pd.Series, rng) -> float:
    v = values.dropna().to_numpy(dtype=float)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else LONE_POINT_REL_SD * abs(mean)
    return mean + sd * float(rng.standard_normal())


@dataclass
class Ledger:
    """Accumulates every synthetic point of one output directory.

    With `enabled=False` every method is a no-op that returns its input unchanged, so
    call sites need no branching and the default path cannot produce a placeholder.
    """

    enabled: bool = False
    seed: int = DEFAULT_SEED
    points: list[dict] = field(default_factory=list)
    figures: dict[str, int] = field(default_factory=dict)

    # ── data ────────────────────────────────────────────────────────────────
    def fill(
        self,
        observed: pd.DataFrame,
        expected: pd.DataFrame,
        *,
        keys: list[str],
        metrics: list[str],
        levels: list[tuple[str, list[str]]],
        expect_by: list[str] | None = None,
        always_expected: bool = False,
        figure: str,
    ) -> pd.DataFrame:
        """Return `observed` plus a synthetic row for every expected-but-absent point.

        observed   real rows; may hold several rows per key (e.g. one per task).
        expected   one row per EXPECTED key, carrying `keys` plus every column the
                   `levels`/`expect_by` groupings name (run-plan columns, typically).
        levels     ordered [(rule_name, group_cols)] for rules (a)-(c); rule_name is
                   one of same_run / neighbours / global. (d) prior is implicit.
        expect_by  a (key, metric) point is expected only if some observed row sharing
                   these columns has that metric -- "the completed runs of that kind
                   have it". None = every observed row counts. `always_expected`
                   skips the test (a run that ran at all has a cost).
        figure     path stem (relative to outdir) the points will be drawn in; it goes
                   to the sidecar and is how clear_stale finds the file later.

        A key with no observed row gets a new row (is_placeholder=True). An observed
        key whose metric is NaN where its kind has the metric gets that ONE cell
        filled, and the row is flagged -- the row's real cells are left exactly as they
        were. Output columns: observed's + is_placeholder + placeholder_rule.
        """
        if not self.enabled:
            return observed
        obs = observed.copy()
        obs["is_placeholder"] = False
        obs["placeholder_rule"] = ""
        if expected is None or expected.empty:
            return obs
        exp = expected.drop_duplicates(keys).reset_index(drop=True)
        present = (
            set(map(tuple, obs[keys].astype(str).to_numpy()))
            if not obs.empty
            else set()
        )
        new_rows = []
        for _, erow in exp.iterrows():
            key = tuple(str(erow[k]) for k in keys)
            is_new = key not in present
            if is_new:
                target_idx = None
                row = {c: erow[c] for c in exp.columns}
            else:
                mask = (obs[keys].astype(str) == pd.Series(key, index=keys)).all(axis=1)
                target_idx = obs.index[mask]
            rules = []
            for m in metrics:
                if not is_new:
                    cur = (
                        obs.loc[target_idx, m]
                        if m in obs.columns
                        else pd.Series([np.nan])
                    )
                    if cur.notna().any():
                        continue  # measured: never touched
                if not always_expected and not self._expected(obs, erow, m, expect_by):
                    continue
                value, rule = self._synthesize(obs, erow, m, levels, key)
                rules.append(f"{m}={rule}")
                self.points.append(
                    {
                        "figure": figure,
                        **{k: erow[k] for k in keys},
                        "metric": m,
                        "value": value,
                        "placeholder_rule": rule,
                        "new_row": is_new,
                    }
                )
                if is_new:
                    row[m] = value
                else:
                    if m not in obs.columns:
                        obs[m] = np.nan
                    obs.loc[target_idx, m] = value
            if not rules:
                continue
            if is_new:
                row["is_placeholder"] = True
                row["placeholder_rule"] = ";".join(rules)
                new_rows.append(row)
            else:
                obs.loc[target_idx, "is_placeholder"] = True
                obs.loc[target_idx, "placeholder_rule"] = ";".join(rules)
        if new_rows:
            obs = pd.concat([obs, pd.DataFrame(new_rows)], ignore_index=True)
        obs["is_placeholder"] = obs["is_placeholder"].astype(bool)
        return obs

    @staticmethod
    def _expected(obs, erow, metric, expect_by) -> bool:
        if obs.empty or metric not in obs.columns:
            return False
        pool = obs[~obs["is_placeholder"]]
        for c in expect_by or []:
            if c not in pool.columns or c not in erow.index:
                continue
            pool = pool[pool[c].astype(str) == str(erow[c])]
        return bool(pool[metric].notna().any())

    def _synthesize(self, obs, erow, metric, levels, key) -> tuple[float, str]:
        rng = _point_rng(self.seed, key, metric)
        real = obs[~obs["is_placeholder"]] if not obs.empty else obs
        value, rule = None, None
        if metric in real.columns:
            for name, cols in levels:
                pool = real
                usable = True
                for c in cols:
                    if c not in pool.columns or c not in erow.index or pd.isna(erow[c]):
                        usable = False
                        break
                    pool = pool[pool[c].astype(str) == str(erow[c])]
                if not usable or pool[metric].notna().sum() == 0:
                    continue
                value = _draw(pool[metric], rng)
                rule = f"{name}({'+'.join(cols)})" if cols else name
                break
        if value is None:
            mean, sd = PRIORS.get(metric, FALLBACK_PRIOR)
            value, rule = mean + sd * float(rng.standard_normal()), "prior"
        lo, hi = valid_range(metric)
        return float(min(max(value, lo), hi)), rule

    # ── figures ─────────────────────────────────────────────────────────────
    def finish_figure(self, fig, figure: str, n: int):
        """Stamp the watermark on `fig` when it carries >= 1 synthetic point.

        Every make_figures figure goes through here, placeholder mode or not; with
        n == 0 (always, when disabled) the figure is returned untouched.
        """
        if not self.enabled or n <= 0:
            return fig
        text = WATERMARK.format(n=n)
        fig.text(
            0.5,
            0.5,
            text,
            rotation=30,
            fontsize=26,
            color="#c0392b",
            alpha=0.35,
            ha="center",
            va="center",
            weight="bold",
            zorder=1000,
        )
        fig.suptitle(text, color="#c0392b", fontsize=9, y=1.02)
        self.figures[figure] = n
        return fig

    # ── outputs ─────────────────────────────────────────────────────────────
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.points)

    def write(self, outdir) -> int:
        """Write placeholders.csv + PLACEHOLDER_DATA.txt; return the point count.

        Written whenever the mode is on, even with zero points, so a preview directory
        always says that it is one.
        """
        if not self.enabled:
            return 0
        outdir = Path(outdir)
        df = self.frame()
        if df.empty:
            df = pd.DataFrame(columns=["figure", "metric", "value", "placeholder_rule"])
        df.to_csv(outdir / SIDECAR, index=False)
        rules = df["placeholder_rule"].value_counts().to_dict() if len(df) else {}
        lines = [
            "THIS DIRECTORY CONTAINS SYNTHETIC PLACEHOLDER DATA.",
            "",
            f"It was built with --placeholder-missing (or {ENV_VAR}=1), seed {self.seed}.",
            f"{len(df)} expected-but-missing point(s) were filled with SYNTHETIC values so",
            "the figures could be drawn before every run finished. They are noise around",
            "neighbouring measurements and carry NO information about the runs they stand",
            "in for. Do not read, quote or publish any number from a watermarked figure.",
            "",
            f"Every synthetic point is listed in {SIDECAR} (figure, key, metric, value,",
            "placeholder_rule). The CSV tables in this directory, the resource fits and",
            "modules.optimized.config are REAL data only: placeholders reach the figures",
            "and the sidecar, nothing else. pull_to_ihc_method.sh refuses this directory.",
            "",
            "Rules used (benchmarks/analysis/lib/placeholders.py):",
            *[f"  {r}: {n}" for r, n in sorted(rules.items())],
            "",
            "Watermarked figures:",
            *[
                f"  {f}  ({n} synthetic points)"
                for f, n in sorted(self.figures.items())
            ],
            "",
            "Re-run make_figures WITHOUT the flag to replace this preview with real output.",
        ]
        (outdir / MARKER).write_text("\n".join(lines) + "\n")
        return len(df)


def placeholder_mask(df: pd.DataFrame) -> pd.Series:
    if "is_placeholder" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["is_placeholder"].fillna(False).astype(bool)
