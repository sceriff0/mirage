"""ARCHIVED from benchmarks/regbench/score.py (not imported, run, linted or tested; see
benchmarks/archive/README.md). The landmark scorer of the public landmark set (HyReCo),
and its branch of ``score.score``'s dispatch, verbatim."""


def _score_landmarks(tables, cases_root, out, dataset, workers=1, expected=None,
                     primary=None):  # fmt: skip
    """The landmark scorer of a public ``landmarks`` dataset (HyReCo's,
    ``regbench.landmarks``), into ``tables``."""
    from . import landmarks

    return landmarks.score(
        cases_root, out, dataset, workers, expected, primary, tables=tables
    )


# in score.score, after the ``tissue`` branch:
#
#         elif kind_of(ds) == "landmarks":
#             done["public"][ds] = _score_landmarks(
#                 where, cases_root, out, ds, workers, expected, primary
#             )
