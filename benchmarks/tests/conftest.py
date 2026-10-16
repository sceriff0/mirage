"""Shared by every test under ``benchmarks/tests``: the ``slow`` marker.

``slow`` marks a test that takes tens of seconds or more on a laptop: the one that runs
STARE on a 4096 x 5120 px pair, and the ones that prepare or score a small toy-cases
fixture several times. It is a label, not a switch: CI runs them all
(``.github/workflows/benchmarks.yml``; ``test_ci_time_budget.py`` holds it to that).
Locally, ``-m "not slow"`` leaves them out.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: takes tens of seconds or more (STARE at 4096 px or larger, or a "
        "toy-cases fixture scored several times); never deselected in CI",
    )
