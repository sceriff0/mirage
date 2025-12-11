# CREATE_PYRAMID Docker Image

Docker image for creating pyramidal OME-TIFF files with libvips for the ATEIA pipeline.

## Base Image

**Base:** `python:3.11-slim`

## Included Tools & Libraries

### System Tools
- **libvips-tools** - Command-line vips utilities
- **libvips-dev** - libvips development files
- Image format support: JPEG, PNG, TIFF, WebP, HEIF

### Python Packages
- **pyvips** (2.2.3) - Python bindings for libvips
- **tifffile** (2024.8.30) - TIFF file I/O
- **numpy** (1.26.4) - Array processing
- **pandas** (2.2.2) - Data manipulation

## Why This Image?

### Speed & Memory Efficiency
- **libvips** streams data and processes in tiles
- Constant memory usage regardless of image size
- Faster than Java-based tools (bfconvert) or pure Python solutions

### Comparison

| Tool | Speed | Memory | Method |
|------|-------|--------|--------|
| vips CLI | **Fastest** | **Minimal** | Streaming, C implementation |
| pyvips | Fast | Low | Python bindings to libvips |
| bfconvert | Slower | High | Java-based, loads into heap |
| Python/tifffile | Slowest | Very High | Loads entire arrays |

## Building the Image

```bash
cd docker
./build_pyramid.sh
```

Or manually:

```bash
docker build -f docker/Dockerfile.pyramid -t ateia/pyramid:latest docker/
```

## Using the Image

### With vips CLI (recommended for performance)

```bash
docker run --rm -v /data:/data ateia/pyramid:latest \
  vips bandjoin "input1.tif input2.tif input3.tif" combined.tif

docker run --rm -v /data:/data ateia/pyramid:latest \
  vips tiffsave combined.tif pyramid.ome.tiff \
    --compression lzw \
    --tile \
    --tile-width 256 \
    --tile-height 256 \
    --pyramid \
    --bigtiff
```

### With Python/pyvips

```bash
docker run --rm -v /data:/data ateia/pyramid:latest \
  python /path/to/create_pyramid.py \
    --merged-image merged.ome.tiff \
    --seg-mask segmentation.tiff \
    --phenotype-mask phenotype.tiff \
    --output pyramid.ome.tiff
```

## Performance Characteristics

### Memory Usage
- **Constant**: ~200-500 MB regardless of input image size
- Processes images in tiles (default: 256×256 pixels)
- Streams data without loading entire image

### Speed
For a 60,000 × 60,000 pixel, 20-channel image (~288 GB uncompressed):
- **vips CLI**: ~5-10 minutes
- **bfconvert**: ~30-60 minutes
- **Python/numpy**: 60+ minutes (if enough RAM available)

### Output
Creates pyramidal TIFF with:
- Multiple resolution levels (auto-determined)
- 2× downsampling between levels (default)
- Tiled structure (256×256 tiles)
- LZW compression
- BigTIFF format support

## Nextflow Integration

In `nextflow.config`:

```groovy
process {
    withName: 'CREATE_PYRAMID' {
        container = 'ateia/pyramid:latest'
    }
}
```

## References

Based on research from:
- [libvips official documentation](https://www.libvips.org/)
- [pyvips PyPI](https://pypi.org/project/pyvips/)
- [libvips Docker images discussion](https://github.com/libvips/libvips/discussions/3504)
- [Community Docker images](https://hub.docker.com/r/marcbachmann/libvips)
- [large-image project](https://pypi.org/project/large-image/)

## License

This Dockerfile is part of the ATEIA pipeline project.
