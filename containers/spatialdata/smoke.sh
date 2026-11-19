#!/usr/bin/env bash
# Import smoke test for bolt3x/mirage-spatialdata.
#
# Run as a build step by the Dockerfile and again by containers.yml against the built
# image, so a stack that resolves in pip but fails on import fails the build.
#
# Asserts the third-party modules this image's scripts import: bin/export_spatialdata.py and bin/join_flowpath.py,
# whose spatialdata/anndata/geopandas imports are all lazy (inside functions).
# Do not add a module these scripts never import; do not drop one they do.
set -euo pipefail

# Some bases have only `python3`.
PY="$(command -v python || command -v python3)"

"$PY" -c "\
import spatialdata, anndata, geopandas, shapely, zarr, dask, tifffile, scipy, pandas, numpy; \
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel; \
from spatialdata.transformations import Identity, Scale; \
print('spatialdata image OK:', \
      'spatialdata', spatialdata.__version__, \
      '| anndata', anndata.__version__, \
      '| geopandas', geopandas.__version__, \
      '| zarr', zarr.__version__, \
      '| dask', dask.__version__)"

# procps supplies `ps`, which Nextflow's task-metrics wrapper needs.
ps -e -o pid= -o ppid= > /dev/null && echo "procps OK: nextflow task-metrics wrapper can run"
