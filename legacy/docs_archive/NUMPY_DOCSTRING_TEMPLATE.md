NumPy docstring + typing template

Use this short template when writing function docstrings and type hints.
Keep descriptions concise and include Parameters, Returns, and Notes when helpful.

Template
--------

def function_name(arg1: int, arg2: "np.ndarray") -> "np.ndarray":
    """One-line summary.

    Extended description (optional).

    Parameters
    ----------
    arg1 : int
        Short description of arg1.
    arg2 : ndarray
        Short description of arg2. Mention expected shape if relevant.

    Returns
    -------
    ndarray
        Description of return value and shape.

    Notes
    -----
    Any implementation notes or references.
    """
    ...

Guidelines
----------
- Always use type hints for public functions.
- Prefer ``numpy.typing.NDArray`` for ndarray annotations when available.
- Keep one-line summary under 80 characters when possible.
- Avoid implementation details in the summary; use the Notes section instead.
- Use ``from __future__ import annotations`` at module top for forward references.
