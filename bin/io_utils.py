"""Input/output utilities for serialization and deserialization.

This module provides utilities for loading and saving Python objects
using pickle format, with proper error handling and logging.

Examples
--------
>>> from lib.io_utils import save_pickle, load_pickle
>>> data = {'accuracy': 0.95, 'loss': 0.05}
>>> save_pickle(data, "results/metrics.pkl")
>>> loaded_data = load_pickle("results/metrics.pkl")

Notes
-----
Pickle files should only be loaded from trusted sources as they can
execute arbitrary code during deserialization.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from lib.logger import get_logger

__all__ = [
    "load_pickle",
    "save_pickle",
]

logger = get_logger(__name__)


def load_pickle(pickle_path: str | Path) -> Any:
    """Load Python object from pickle file.

    Parameters
    ----------
    pickle_path : str or Path
        Path to pickle file.

    Returns
    -------
    Any
        Deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If pickle file does not exist.
    pickle.UnpicklingError
        If file is corrupted or incompatible.

    Warnings
    --------
    Only unpickle files from trusted sources. Pickle files can execute
    arbitrary code during deserialization.

    Examples
    --------
    >>> data = load_pickle("model_weights.pkl")
    >>> print(type(data))
    <class 'dict'>

    See Also
    --------
    save_pickle : Save object to pickle file
    """
    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    logger.debug(f"Loading pickle: {pickle_path.name}")

    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)

    logger.debug(f"  Loaded {type(obj).__name__}")

    return obj


def save_pickle(
    obj: Any,
    output_path: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> Path:
    """Save Python object to pickle file.

    Parameters
    ----------
    obj : Any
        Python object to serialize.
    output_path : str or Path
        Path where pickle file will be saved.
    protocol : int, default=pickle.HIGHEST_PROTOCOL
        Pickle protocol version. Higher versions are more efficient
        but may not be compatible with older Python versions.

    Returns
    -------
    Path
        Path object for the saved file.

    Notes
    -----
    Automatically creates parent directories if needed.
    Uses highest protocol by default for efficiency.

    Examples
    --------
    >>> results = {'accuracy': 0.95, 'loss': 0.05}
    >>> save_pickle(results, "output/metrics.pkl")

    See Also
    --------
    load_pickle : Load object from pickle file
    """
    from lib.image_utils import ensure_dir

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    logger.debug(f"Saving pickle: {output_path.name} ({type(obj).__name__})")

    with open(output_path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

    logger.debug(f"  Saved successfully")

    return output_path
