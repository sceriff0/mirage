#!/bin/bash
# Build script for CREATE_PYRAMID Docker image

set -e

IMAGE_NAME="ateia/pyramid"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

docker build \
    -f Dockerfile.pyramid \
    -t ${IMAGE_NAME}:${TAG} \
    .

echo "✓ Docker image built successfully: ${IMAGE_NAME}:${TAG}"

# Test the image
echo ""
echo "Testing libvips installation..."
docker run --rm ${IMAGE_NAME}:${TAG} vips --version

echo ""
echo "Testing Python packages..."
docker run --rm ${IMAGE_NAME}:${TAG} python -c "
import pyvips
import tifffile
import numpy as np
print('✓ All packages imported successfully')
print(f'  - pyvips: {pyvips.__version__}')
print(f'  - tifffile: {tifffile.__version__}')
print(f'  - numpy: {np.__version__}')
"

echo ""
echo "Image ready to use!"
echo "To push to registry:"
echo "  docker tag ${IMAGE_NAME}:${TAG} your-registry/${IMAGE_NAME}:${TAG}"
echo "  docker push your-registry/${IMAGE_NAME}:${TAG}"
