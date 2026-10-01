import numpy as np
import pandas as pd
import pytest

from benchmarks.analysis.lib import regress


def test_fit_memory_model_recovers_known_line():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 3.0 * x + 2.0  # slope 3, intercept 2, no noise
    m = regress.fit_memory_model(x, y)
    assert m["slope"] == pytest.approx(3.0)
    assert m["intercept"] == pytest.approx(2.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["sigma"] == pytest.approx(0.0, abs=1e-9)
    assert m["n"] == 5


def test_fit_memory_model_too_few_points_is_flat_fallback():
    m = regress.fit_memory_model(np.array([2.0]), np.array([10.0]))
    assert m["slope"] == 0.0
    assert m["intercept"] == pytest.approx(10.0)
    assert m["n"] == 1


def test_fit_per_process_groups_by_process():
    df = pd.DataFrame(
        {
            "process": ["A", "A", "A", "B", "B", "B"],
            "input_gb": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "peak_rss_gb": [3.0, 5.0, 7.0, 11.0, 21.0, 31.0],  # A: 2x+1 ; B: 10x+1
        }
    )
    models = regress.fit_per_process(df, predictor="input_gb", target="peak_rss_gb")
    assert models["A"]["slope"] == pytest.approx(2.0)
    assert models["B"]["slope"] == pytest.approx(10.0)
