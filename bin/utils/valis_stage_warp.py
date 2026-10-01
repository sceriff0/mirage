"""Stage-resolved point warping through a VALIS registrar.

``Slide.warp_geojson`` gives you exactly one answer: the *final* transform. That is a problem
for QC, because VALIS's final state is a composition that has destroyed its own intermediates —
``MicroRigidRegistrar`` overwrites ``slide.M`` (micro_rigid_registrar.py) and ``register_micro``
does ``fwd_dxdy = fwd_dxdy + micro_residual`` in place (registration.py). The pickle therefore
cannot tell you what the alignment looked like before micro-registration.

This module does two things about that:

1. :func:`slide_warp_params` extracts the parameters ``Slide.warp_geojson`` derives, so the
   same warp can be driven with a **substituted** displacement field — the pre-micro field
   saved by ``bin/register.py``'s checkpoint — instead of only the slide's current one.

2. :func:`warp_points` applies them to a flat ``(N, 2)`` vertex array in **one** call. VALIS
   warps a GeoJSON with a per-feature Python loop over shapely geometries; on a whole-slide
   collection that is millions of iterations per stage, and the staged QC needs three or four
   stages. Vertices are position-independent under this transform, so batching is exact.

**Deliberate divergence from ``warp_geojson``:** the vertex clip to the aligned frame
(``warp_tools.clip_xy``) is off by default. VALIS clips so that warped annotations land inside
the image it is about to write. For a QC metric, clipping flattens every cell that straddles
the crop boundary onto that boundary — in *both* slides, identically — which inflates their
IoU toward 1 for reasons that have nothing to do with registration. Pass ``clip=True`` to
reproduce VALIS's output exactly.
"""

from __future__ import annotations

import numpy as np

# The transform states a staged QC can distinguish, in the order VALIS applies them.
STAGE_NATIVE = "native"
STAGE_RIGID = "rigid"
STAGE_NON_RIGID = "non_rigid"
STAGE_MICRO = "micro"


def slide_warp_params(slide, crop="reference") -> dict:
    """Parameters ``Slide.warp_geojson(pt_level=0, slide_level=0, crop=crop)`` would use.

    Mirrors registration.py's derivation term for term; ``shift_xy`` is the crop origin that
    ``_warp_shapely`` subtracts after warping.
    """
    pt_dim_rc = np.asarray(slide.slide_dimensions_wh[0])[::-1]
    aligned_shape_rc = np.asarray(slide.aligned_slide_shape_rc)

    shift_xy = None
    crop_method = slide.get_crop_method(crop)
    if crop_method is not False:
        # CROP_REF measures against the reference slide's own level-0 shape; CROP_OVERLAP
        # against the common aligned shape. Both are what warp_geojson passes.
        if crop_method == "reference":
            ref_slide = slide.val_obj.get_ref_slide()
            scaled_shape_rc = np.asarray(ref_slide.slide_dimensions_wh[0])[::-1]
        else:
            scaled_shape_rc = aligned_shape_rc
        crop_bbox_xywh, _ = slide.get_crop_xywh(crop_method, scaled_shape_rc)
        shift_xy = np.asarray(crop_bbox_xywh[0:2], dtype=float)

    return {
        "M": slide.M,
        "transformation_src_shape_rc": slide.processed_img_shape_rc,
        "transformation_dst_shape_rc": slide.reg_img_shape_rc,
        "src_shape_rc": pt_dim_rc,
        "dst_shape_rc": aligned_shape_rc,
        "shift_xy": shift_xy,
    }


def warp_points(xy, params, fwd_dxdy=None, clip=False) -> np.ndarray:
    """Warp ``(N, 2)`` vertices with ``params`` and an explicit displacement field.

    ``fwd_dxdy=None`` gives the rigid-only warp — the same thing ``non_rigid=False`` gives in
    ``warp_geojson``, because that is precisely what that flag does.
    """
    from valis import warp_tools

    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return xy.copy()

    warped = warp_tools.warp_xy(
        xy,
        M=params["M"],
        transformation_src_shape_rc=params["transformation_src_shape_rc"],
        transformation_dst_shape_rc=params["transformation_dst_shape_rc"],
        src_shape_rc=params["src_shape_rc"],
        dst_shape_rc=params["dst_shape_rc"],
        fwd_dxdy=fwd_dxdy,
    )
    warped = np.asarray(warped, dtype=float)
    if params.get("shift_xy") is not None:
        warped = warped - params["shift_xy"]
    if clip:
        warped = warp_tools.clip_xy(warped, params["dst_shape_rc"])
    return warped


def _slide_fwd_dxdy(slide):
    """The slide's *current* forward displacement field, or None if it has none.

    Reference slides carry no field (``register_micro`` skips them), and a run whose non-rigid
    stage failed leaves the attribute unset — both mean "rigid only" and both return None.
    """
    return getattr(slide, "fwd_dxdy", None)


def to_numpy_field(field):
    """Normalise a VALIS displacement field to a ``(2, H, W)`` float32 numpy array.

    VALIS stores fields either as ``np.array([dx, dy])`` or, when ``stored_dxdy`` is set, as a
    2-band pyvips image loaded from disk. Checkpointing has to handle both.
    """
    if field is None:
        return None
    if isinstance(field, np.ndarray):
        arr = field
    else:  # pyvips.Image -> (H, W, 2)
        from valis import warp_tools

        arr = warp_tools.vips2numpy(field)
        arr = np.transpose(arr, (2, 0, 1))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(
            f"expected a (2, H, W) displacement field, got shape {arr.shape}"
        )
    return arr


def make_warper(registrar, crop="reference", clip=False, checkpoint=None):
    """Build ``warp(slide_name, xy, stage) -> xy`` over a loaded registrar.

    ``checkpoint`` is the :class:`~stage_checkpoint.StageCheckpoint` written by REGISTER before
    micro-registration; it supplies the ``non_rigid`` stage's field. Without it that stage is
    not reachable and asking for it raises rather than silently returning the micro answer.
    """
    params_cache: dict = {}
    field_cache: dict = {}

    def _params(name):
        if name not in params_cache:
            params_cache[name] = slide_warp_params(
                registrar.slide_dict[name], crop=crop
            )
        return params_cache[name]

    def _field(name, stage):
        key = (name, stage)
        if key in field_cache:
            return field_cache[key]
        if stage == STAGE_RIGID:
            field = None
        elif stage == STAGE_MICRO:
            field = _slide_fwd_dxdy(registrar.slide_dict[name])
        elif stage == STAGE_NON_RIGID:
            if checkpoint is None:
                raise ValueError(
                    "the 'non_rigid' stage needs the REGISTER pre-micro checkpoint; without it "
                    "the pickle's field already has the micro residual composed into it"
                )
            field = checkpoint.fwd_dxdy(name)
        else:
            raise ValueError(f"unknown stage {stage!r}")
        field_cache[key] = field
        return field

    def warp(slide_name, xy, stage):
        if stage == STAGE_NATIVE:
            return np.asarray(xy, dtype=float)
        return warp_points(
            xy, _params(slide_name), fwd_dxdy=_field(slide_name, stage), clip=clip
        )

    return warp


def slide_pixel_size_um(registrar, slide_name):
    """Physical pixel size in µm, or None when the slide metadata does not carry one."""
    slide = registrar.slide_dict.get(slide_name)
    res = getattr(slide, "resolution", None)
    units = getattr(slide, "units", None)
    if res is None or not np.isfinite(res) or res <= 0:
        return None
    if units and str(units).strip().lower() not in (
        "um",
        "µm",
        "µm",
        "micron",
        "microns",
    ):
        return None
    return float(res)
