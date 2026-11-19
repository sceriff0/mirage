"""The one raster pixel-addressing rule.

MIRAGE publishes artifacts in two conventions:

* **Centre-of-pixel** (skimage, ``regionprops`` centroids): integer ``(row, col)`` is the
  pixel's centre, so it spans ``[r - 0.5, r + 0.5] x [c - 0.5, c + 0.5]``.
* **Corner-of-pixel** (QuPath, ImageJ): integer ``(x, y)`` is the top-left corner, so the
  pixel spans ``[x, x + 1] x [y, y + 1]``.

``corner = centre + CORNER_OFFSET``. Mixing the two in one artifact shifts every point
half a pixel up-left of its polygon, which skews neighbourhood analyses and can fail a
point-in-polygon check on small cells.

The first three rows convert; the last three must know which side they are on.

===========================================  ============  ==========================
site                                         convention    consumer
===========================================  ============  ==========================
``extract_cell_properties.extract_contours``  corner        QuPath (via GeoJSON) and
                                                            the SpatialData shapes
``export_geojson`` centroids (x3)             corner        QuPath ``cells.geojson``
``export_spatialdata`` ``obsm["spatial"]``    corner        SpatialData ``.zarr``
``mask_to_geojson``                           **centre**    VALIS ``warp_geojson``
``join_reg_residuals`` (both sides)           **centre**    an internal QC join
``join_flowpath.join_by_centroid``            corner        an internal cohort join
===========================================  ============  ==========================

``bin/mask_to_geojson.py`` must stay centre: its contours go to VALIS's
``Slide.warp_geojson``, which addresses the raster like skimage. It also feeds
``join_reg_residuals``, which pairs those centroids with the quantification CSV's raw
(centre) x/y; adding an offset to one side would bias that join by 0.707 px.

``bin/join_flowpath.py`` matches FlowPath's µm centroids against ``obsm["spatial"]``;
both are corner, so the inverse is a plain ``/ pixel_size``. Before writing ``- 0.5``,
check which convention the other side is in.

Imported flat (``from pixel_convention import ...``) after ``bin/utils`` is on sys.path.
"""

from __future__ import annotations

#: Distance from a pixel's centre to its top-left corner, in pixels, per axis.
CORNER_OFFSET = 0.5


def centre_to_corner(coords):
    """Translate centre-of-pixel coordinates into corner-of-pixel ones.

    Parameters
    ----------
    coords
        Anything that supports ``+ float``: a Python scalar, a numpy array of
        any shape, or a pandas object. The offset is per-axis and identical on
        every axis, so an ``(N, 2)`` array of ``(x, y)`` (or ``(y, x)``) pairs
        can be passed whole -- there is nothing to get the axis order wrong.

    Returns
    -------
    The same kind of object, translated by :data:`CORNER_OFFSET`. Never in
    place: an input array is left untouched.
    """
    return coords + CORNER_OFFSET
