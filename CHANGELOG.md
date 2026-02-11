# Changelog

All notable changes to the MIRAGE pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized constants module (`bin/utils/constants.py`) for pipeline-wide configuration
- CLI utilities module (`bin/utils/cli.py`) for standardized argument parsing
- Timing utilities in logger (`log_timing` context manager, `timed` decorator)
- Package exports in `bin/utils/__init__.py`
- EditorConfig file for consistent code formatting across editors

### Changed
- Simplified container declarations in all 17 Nextflow modules
- Added standardized process header comments to 14 Nextflow modules
- Added `when:` directive to 4 modules for conditional execution support
- Migrated `register.py` to unified logger pattern with proper log levels
- Migrated `compute_features.py` to unified logger pattern
- Improved logging in `preprocess.py`:
  - Changed per-channel logs to DEBUG level
  - Combined verbose metadata logs
  - Standardized status symbols (`[OK]`, `[SKIP]`, `[WARN]`)
- Improved logging in `quantify.py`:
  - Added entry/exit logging to `compute_morphology()`
  - Added error logging before exceptions
  - Changed detailed output logs to DEBUG level
- Standardized status symbols across all scripts:
  - `[OK]` for success
  - `[SKIP]` for skipped items
  - `[WARN]` for warnings
  - `[FAIL]` for errors

### Fixed
- Fixed parameter formatting in `preprocess.py` line 249 (`skip_dapi : bool` -> `skip_dapi: bool`)
- Removed redundant `import tifffile` in `preprocess.py`

## [0.1.0] - 2024-01-01

### Added
- Initial release of MIRAGE pipeline
- **Preprocessing**: BaSiC illumination correction with FOV tiling
  - ND2 to OME-TIFF conversion
  - Multi-channel parallel processing
  - Configurable FOV tile size and overlap
- **Registration**: Multi-method image registration
  - VALIS whole-slide registration
  - VALIS pairwise registration
  - GPU-accelerated diffeomorphic registration
  - CPU multi-threaded diffeomorphic registration
  - Feature-based error estimation (SuperPoint, DISK, DeDoDe, BRISK)
- **Segmentation**: StarDist cell segmentation
  - GPU-accelerated inference
  - Configurable tile sizes for memory management
  - Nuclei and whole-cell mask output
- **Quantification**: Marker intensity quantification
  - Per-cell morphological features
  - Per-channel intensity measurements
  - GPU-accelerated option
- **Phenotyping**: Cell phenotype classification
  - Z-score based thresholding
  - QuPath-compatible GeoJSON output
  - Configurable marker cutoffs
- **Output**: Pyramidal OME-TIFF generation
  - QuPath visualization compatibility
  - Segmentation and phenotype overlays
  - Configurable pyramid levels and tile sizes

### Infrastructure
- DSL2 Nextflow pipeline architecture
- Checkpoint-based restart capability (preprocessing, registration, postprocessing)
- SLURM/Singularity execution profiles
- Docker container support
- Nextflow Tower integration
- Comprehensive parameter validation with JSON schema
- Groovy utility libraries for CSV parsing and validation
