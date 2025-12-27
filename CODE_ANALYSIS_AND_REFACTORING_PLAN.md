# Code Analysis & Refactoring Plan

## Executive Summary

After analyzing all 23 Python scripts in `/bin`, I've identified **significant violations** of KISS, DRY, and software design principles. The codebase would benefit greatly from consolidation into shared utility libraries.

---

## Critical Issues Found

### 1. ❌ **MASSIVE DRY Violations - Code Duplication**

#### Duplicated Functions Across Multiple Files:

| Function | Duplicated In | Lines Wasted |
|----------|---------------|--------------|
| `get_channel_names()` | register.py, register_cpu.py, register_gpu.py, generate_registration_qc.py, merge_registered.py | ~150 lines |
| `autoscale()` | register.py, register_cpu.py, register_gpu.py, generate_registration_qc.py | ~80 lines |
| `load_image()` | quantify.py, segment.py | ~50 lines |
| `ensure_dir()` | Duplicated in `_common.py` but also reimplemented in multiple scripts | ~30 lines |
| `log_progress()`/`log()` | register.py, merge_registered.py, compute_features.py | ~45 lines |
| `create_qc_rgb_composite()` | register_cpu.py, register_gpu.py (nearly identical implementations) | ~140 lines |
| `extract_crop_coords()` | register_cpu.py, register_gpu.py (identical) | ~30 lines |
| `calculate_bounds()` | register_cpu.py, register_gpu.py (identical) | ~30 lines |
| `apply_affine_cv2()` | register_cpu.py, register_gpu.py (identical) | ~10 lines |
| `compute_affine_mapping_cv2()` | register_cpu.py, register_gpu.py (identical) | ~100 lines |
| `create_memmaps_for_merge()` | register_cpu.py, register_gpu.py (identical) | ~30 lines |
| `cleanup_memmaps()` | register_cpu.py, register_gpu.py (identical) | ~15 lines |
| `extract_channel_names()` | register_cpu.py, register_gpu.py (identical) | ~40 lines |
| `create_ome_xml()` | register_cpu.py, register_gpu.py (identical) | ~25 lines |

**Total estimated duplicated lines: ~800+ lines** 🚨

### 2. ❌ **Logger Anti-Pattern - No Singleton**

Every script creates its own logger and duplicates logging configuration:

```python
# Repeated in EVERY script:
logger = logging.getLogger(__name__)

# And in main():
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Problem**: No centralized logging configuration, inconsistent formats, can't easily change log level globally.

### 3. ❌ **Inconsistent Image I/O**

Multiple image loading approaches across scripts:

- `quantify.py`: Custom `load_image()` with aicsimageio fallback
- `segment.py`: Memory-mapped loading with tifffile
- `preprocess.py`: Direct tifffile.imread
- `convert_nd2.py`: nd2 library
- `merge_registered.py`: Complex memmap logic

**Problem**: No standardized error handling, metadata extraction inconsistent, can't easily switch backends.

### 4. ❌ **Poor Separation of Concerns**

Examples:
- QC generation code embedded in registration scripts (should be separate module)
- Channel name parsing duplicated across 5+ files
- OME-XML creation logic duplicated
- Crop extraction/reconstruction logic not centralized

### 5. ❌ **Type Hints Inconsistency**

- `_common.py` has good type hints
- Newer scripts (segment.py, quantify.py) have partial type hints
- Older scripts (register.py, merge_registered.py) have no type hints
- Return types often missing

---

## ✅ Good Practices Found

1. **`_common.py` is a step in the right direction** - but underutilized
2. **Use of `__all__`** for explicit exports (in some modules)
3. **Good NumPy-style docstrings** in some modules (segment.py, quantify.py)
4. **`from __future__ import annotations`** in newer code
5. **Proper use of context managers** for file I/O

---

## Refactoring Strategy

### Phase 1: Create Centralized Library (`lib/`)

#### 1.1 `lib/logger.py` - Singleton Logger
```python
"""Centralized logging with singleton pattern."""
- get_logger(__name__) → Logger
- configure_logging(level, log_file, format_string)
```

#### 1.2 `lib/image_io.py` - Unified Image I/O
```python
"""Standardized image loading/saving with consistent error handling."""
- load_image(path) → (ndarray, metadata)
- save_tiff(image, path, **kwargs)
- save_ome_tiff(image, path, channel_names, **kwargs)
- load_image_channel(path, channel_idx) → ndarray
- get_image_shape_memmap(path) → tuple
```

#### 1.3 `lib/metadata.py` - Metadata Utilities
```python
"""OME-TIFF and channel metadata utilities."""
- get_channel_names(filename) → List[str]
- extract_channel_names_from_ome(path) → List[str]
- create_ome_xml(channel_names, dtype, width, height) → str
- get_ome_metadata(path) → dict
```

#### 1.4 `lib/registration_utils.py` - Registration Helpers
```python
"""Shared registration utilities."""
- autoscale(img, low_p, high_p) → ndarray
- extract_crop_coords(shape, crop_size, overlap) → List[Dict]
- calculate_bounds(start, crop_dim, total_dim, overlap) → tuple
- create_memmaps_for_merge(shape, dtype, prefix) → tuple
- cleanup_memmaps(tmp_dir)
```

#### 1.5 `lib/affine.py` - Affine Registration
```python
"""Affine transformation utilities."""
- compute_affine_mapping_cv2(ref, moving, n_features) → matrix
- apply_affine_cv2(image, matrix) → ndarray
```

#### 1.6 `lib/qc.py` - QC Generation
```python
"""Quality control image generation."""
- create_qc_rgb_composite(ref_path, reg_path, output_path, scale_factor)
- autoscale_for_display(img) → ndarray
```

###Phase 2: Refactor Scripts to Use Library

Update all scripts to import from `lib/` instead of duplicating code:

```python
# Before:
logger = logging.getLogger(__name__)
def get_channel_names(filename):
    # ... 20 lines of code ...

# After:
from lib.logger import get_logger
from lib.metadata import get_channel_names

logger = get_logger(__name__)
```

### Phase 3: Apply NumPy Docstring Standards

All functions should follow NumPy docstring format:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    param1 : Type1
        Description of param1.
    param2 : Type2
        Description of param2.

    Returns
    -------
    ReturnType
        Description of return value.

    Raises
    ------
    ErrorType
        When this error occurs.

    Examples
    --------
    >>> function_name(val1, val2)
    expected_output

    Notes
    -----
    Additional implementation notes.
    """
```

### Phase 4: Improve Type Hints

Add comprehensive type hints everywhere:

```python
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from numpy.typing import NDArray

def process_image(
    image_path: str | Path,
    output_dir: str | Path,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[NDArray, Dict[str, Any]]:
    """Process image with type-safe parameters."""
    ...
```

---

## Implementation Priority

### Must Do (P0):
1. ✅ Create `lib/logger.py` (singleton pattern)
2. ✅ Create `lib/metadata.py` (channel names, OME-XML)
3. ✅ Create `lib/registration_utils.py` (shared registration code)
4. ✅ Refactor `register_cpu.py` and `register_gpu.py` to use shared code
5. ✅ Create `lib/qc.py` and refactor QC generation

### Should Do (P1):
6. Create `lib/image_io.py` (unified image loading)
7. Refactor `quantify.py` to use `lib/image_io.py`
8. Apply NumPy docstrings to all library functions

### Nice to Have (P2):
9. Add comprehensive type hints to all functions
10. Create `lib/validation.py` for input validation
11. Add unit tests for all library functions

---

## Metrics

### Before Refactoring:
- **Lines of code**: ~11,000 (estimated)
- **Duplicated code**: ~800+ lines
- **Scripts without proper logging**: 15+
- **Scripts with type hints**: 5/23 (22%)
- **Code reuse score**: 3/10

### After Refactoring (Projected):
- **Lines of code**: ~9,500 (est.)
- **Duplicated code**: <50 lines
- **Scripts without proper logging**: 0
- **Scripts with type hints**: 23/23 (100%)
- **Code reuse score**: 9/10
- **Maintainability**: Significantly improved

---

## Breaking Changes

⚠️ **Minimal breaking changes** expected:
- Scripts will need to import from `lib/` instead of defining functions locally
- Logging configuration should be called once at script entry point
- All existing CLI interfaces will remain unchanged

---

## Recommendations

1. **Logger as Singleton**: YES - This is a standard pattern and prevents configuration duplication
2. **Consolidated Utils**: YES - Critical for maintainability
3. **NumPy Docstrings**: YES - Already partially implemented, just needs consistency
4. **Type Hints**: YES - Improves IDE support and catches bugs early
5. **Separate QC Module**: YES - QC generation is currently embedded in registration scripts

---

## Next Steps

1. Get approval for refactoring approach
2. Create `lib/` directory structure
3. Implement core utility modules (logger, metadata, registration_utils)
4. Refactor 2-3 scripts as proof of concept
5. Roll out to remaining scripts
6. Add comprehensive tests

**Estimated effort**: 2-3 days for complete refactoring
**Risk**: Low (backwards compatible at CLI level)
**Benefit**: High (much easier to maintain, debug, and extend)
