"""Per-process resource regression.

Formalises notebooks/resource_regression.ipynb: fit peak_rss_gb ~ input_gb with
scikit-learn, report r2 and residual sigma, and size with an additive-sigma
retry buffer (attempt N => base + N*sigma) matching the repo's task.attempt scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def fit_memory_model(x, y) -> dict:
    """Fit y ~ x. <3 points => flat fallback (slope 0, intercept = mean y)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n < 3:
        return {
            "slope": 0.0,
            "intercept": float(np.mean(y)) if n else 0.0,
            "r2": float("nan"),
            "sigma": float(np.std(y)) if n else 0.0,
            "n": n,
        }
    reg = LinearRegression().fit(x, y)
    pred = reg.predict(x)
    resid = y - pred
    return {
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "r2": float(r2_score(y, pred)),
        "sigma": float(np.std(resid, ddof=0)),
        "n": n,
    }


def fit_per_process(
    df: pd.DataFrame, predictor: str = "input_gb", target: str = "peak_rss_gb"
) -> dict:
    models = {}
    for proc, g in df.groupby("process"):
        sub = g[[predictor, target]].dropna()
        models[proc] = fit_memory_model(
            sub[predictor].to_numpy(), sub[target].to_numpy()
        )
    return models


def models_to_frame(
    models: dict, predictor: str = "input_gb", target: str = "peak_rss_gb"
) -> pd.DataFrame:
    """Flatten {process: model} into one tidy row per process for CSV export.

    Columns: process, predictor, target, slope, intercept, r2, sigma, n —
    everything needed to reproduce the memory closure (and the fit quality) in R.
    """
    rows = [
        {
            "process": proc,
            "predictor": predictor,
            "target": target,
            "slope": m["slope"],
            "intercept": m["intercept"],
            "r2": m["r2"],
            "sigma": m["sigma"],
            "n": m["n"],
        }
        for proc, m in sorted(models.items())
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "process",
            "predictor",
            "target",
            "slope",
            "intercept",
            "r2",
            "sigma",
            "n",
        ],
    )
