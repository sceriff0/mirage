"""Pipeline constants and configuration defaults.

This module centralizes magic numbers and configuration values used
throughout the ATEIA pipeline for consistency and maintainability.

Examples
--------
>>> from utils.constants import DEFAULT_FOV_SIZE, ExitCode
>>> fov_size = DEFAULT_FOV_SIZE  # 1950
>>> sys.exit(ExitCode.SUCCESS)
"""

from __future__ import annotations

__all__ = [
    # Image processing
    "DEFAULT_FOV_SIZE",
    "DEFAULT_PIXEL_SIZE",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_PYRAMID_SCALE",
    "DEFAULT_PYRAMID_LEVELS",
    # Exit codes
    "ExitCode",
    # File patterns
    "OME_TIFF_EXTENSIONS",
    "TIFF_EXTENSIONS",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Logging
    "LOG_SEPARATOR",
    "LOG_SEPARATOR_THIN",
]


# =============================================================================
# Image Processing Defaults
# =============================================================================

# BaSiC illumination correction
DEFAULT_FOV_SIZE: int = 1950  # Field of view tile size in pixels

# Physical dimensions
DEFAULT_PIXEL_SIZE: float = 0.325  # Microns per pixel

# Pyramid generation
DEFAULT_TILE_SIZE: int = 512  # Tile dimensions for pyramidal TIFF
DEFAULT_PYRAMID_SCALE: int = 2  # Scale factor between pyramid levels
DEFAULT_PYRAMID_LEVELS: int = 3  # Number of pyramid resolution levels

# Registration
DEFAULT_FEATURE_COUNT: int = 5000  # Number of features for detection
DEFAULT_MAX_PROCESSED_DIM: int = 512  # Max dimension for rigid registration
DEFAULT_MAX_NON_RIGID_DIM: int = 2048  # Max dimension for non-rigid registration


# =============================================================================
# Exit Codes
# =============================================================================

class ExitCode:
    """Standard exit codes for CLI scripts.

    Using consistent exit codes allows calling processes to understand
    the nature of failures without parsing error messages.

    Attributes
    ----------
    SUCCESS : int
        Successful completion (0).
    GENERAL_ERROR : int
        Unspecified error (1).
    INPUT_ERROR : int
        Input validation or file not found (2).
    OUTPUT_ERROR : int
        Output writing or permission error (3).
    RESOURCE_ERROR : int
        Memory, GPU, or compute resource error (4).
    VALIDATION_ERROR : int
        Data validation failure (5).

    Examples
    --------
    >>> import sys
    >>> from utils.constants import ExitCode
    >>> if not input_path.exists():
    ...     sys.exit(ExitCode.INPUT_ERROR)
    >>> sys.exit(ExitCode.SUCCESS)
    """

    SUCCESS: int = 0
    GENERAL_ERROR: int = 1
    INPUT_ERROR: int = 2
    OUTPUT_ERROR: int = 3
    RESOURCE_ERROR: int = 4
    VALIDATION_ERROR: int = 5


# =============================================================================
# File Patterns
# =============================================================================

# OME-TIFF specific extensions
OME_TIFF_EXTENSIONS: tuple[str, ...] = (".ome.tif", ".ome.tiff")

# All TIFF extensions (including OME-TIFF)
TIFF_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".ome.tif", ".ome.tiff")

# All supported image formats
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".nd2",
    ".czi",
    ".lif",
)


# =============================================================================
# Logging Formatting
# =============================================================================

# Standard separator for log sections (70 characters)
LOG_SEPARATOR: str = "=" * 70

# Thin separator for subsections
LOG_SEPARATOR_THIN: str = "-" * 50
