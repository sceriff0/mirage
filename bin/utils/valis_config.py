"""Single source of truth for the VALIS registrar configuration.

`bin/register.py` builds its `Valis` object from `build_registrar_kwargs(...)` here, so the
registrar's feature detector, matcher, MicroRigidRegistrar, image-dim caps and affine optimizer
have a single definition. `init_jvm(...)` is the shared BioFormats JVM-heap sizer, also used by
`bin/warp_seg_qc.py` for the reg_qc>=2 segmentation-overlap QC.
"""

import os

from valis import feature_detectors, feature_matcher
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis.non_rigid_registrars import OpticalFlowWarper

# Memory mode presets — bundle feature detector, matcher, and dimension settings.
# (Kept identical to the historical register.py MEMORY_PRESETS.)
#
# NOT ALL KEYS ARE LIVE. build_registrar_kwargs() below passes `feature_detector_cls`, `matcher`,
# `max_processed_image_dim_px` and `max_non_rigid_registration_dim_px` to Valis(...). It does NOT
# pass `num_features`, nor the 'low' row's `tile_wh` / `tile_buffer` — those are dead keys that
# reach nothing. They are left in place rather than deleted because removing them is a behavioural
# question (num_features would have to be threaded through the detector constructor), but there is
# deliberately no pipeline param for them: a knob that changes nothing is worse than no knob.
MEMORY_PRESETS = {
    "high": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "max_processed_image_dim_px": 2048,
        "max_non_rigid_registration_dim_px": 2048,
        "num_features": 5000,
    },
    "medium": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "max_processed_image_dim_px": 1024,
        "max_non_rigid_registration_dim_px": 1024,
        "num_features": 5000,
    },
    "low": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "num_features": 5000,
        "max_processed_image_dim_px": 512,
        "max_non_rigid_registration_dim_px": 512,
        "tile_wh": 512,
        "tile_buffer": 100,
    },
}


# The tier vocabulary, shared with the STARE backend and with nextflow_schema.json's
# `memory_mode` enum. 'custom' is not a row in MEMORY_PRESETS: it means "start from 'high' and
# apply whichever per-knob overrides the caller passed", which is what resolve_memory_mode encodes.
# Mirrored by lib/RegPresets.groovy (MODES / DEFAULT_MODE) for the tiled backend; the two are
# pinned together by tests/test_reg_presets_inlined_in_config.py.
MEMORY_MODES = ["high", "medium", "low", "custom"]
DEFAULT_MEMORY_MODE = "high"


def resolve_memory_mode(memory_mode):
    """Map a tier name onto the MEMORY_PRESETS row it draws its base values from.

    'custom' resolves to 'high' so that any knob the user did NOT override keeps its high value.
    An unknown mode raises: this is called before the JVM starts and long before any expensive
    work, so failing loudly here is strictly better than silently registering at a tier the user
    did not ask for.
    """
    mode = memory_mode or DEFAULT_MEMORY_MODE
    if mode == "custom":
        return DEFAULT_MEMORY_MODE
    if mode not in MEMORY_PRESETS:
        raise ValueError(
            f"Unknown memory_mode {memory_mode!r}. Expected one of {MEMORY_MODES}."
        )
    return mode


def build_registrar_kwargs(
    reference_img_f,
    memory_mode="high",
    micro_reg=2,
    max_image_dim_px=4000,
    max_processed_dim=None,
    max_non_rigid_dim=None,
):
    """Return the exact kwargs dict passed to `registration.Valis(...)` by classic register.py.

    ``micro_reg`` is the ordinal micro-registration depth (0/1/2). It controls only the *first*
    micro pass here — ``MicroRigidRegistrar``, which runs inside ``Valis.register()`` and refines
    ``slide.M`` — via the ``micro_rigid_registrar_cls`` constructor kwarg: enabled at level >= 1.
    The *second* pass (``register_micro``, the non-rigid micro step) is a separate method call
    gated at level >= 2 by ``register.py``; it is not configured here.

    NOTE: a fresh `SuperGlueMatcher()` instance is created per call (mirrors register.py, which
    instantiates the matcher from the preset). The matcher carries no cross-run RNG state that
    affects determinism for our purposes (SuperPoint/SuperGlue inference is deterministic).
    """
    preset = MEMORY_PRESETS[resolve_memory_mode(memory_mode)]

    # Explicit `is not None`, never `or` / `?:`: those are falsy-coalescing, so a deliberate 0
    # would be silently rewritten to the preset value. Same rule the Nextflow side follows
    # (tests/test_nullable_numeric_params_no_elvis.py).
    processed_dim = (
        max_processed_dim
        if max_processed_dim is not None
        else preset["max_processed_image_dim_px"]
    )
    non_rigid_dim = (
        max_non_rigid_dim
        if max_non_rigid_dim is not None
        else preset["max_non_rigid_registration_dim_px"]
    )

    return {
        "reference_img_f": reference_img_f,
        "align_to_reference": True,
        "crop": "reference",
        "max_processed_image_dim_px": processed_dim,
        "max_non_rigid_registration_dim_px": non_rigid_dim,
        "max_image_dim_px": preset.get("max_image_dim_px", max_image_dim_px),
        "feature_detector_cls": preset["feature_detector_cls"],
        "matcher": preset["matcher"],
        "non_rigid_registrar_cls": OpticalFlowWarper,
        "affine_optimizer_cls": None,
        "micro_rigid_registrar_cls": MicroRigidRegistrar if micro_reg >= 1 else None,
        "create_masks": True,
    }


def _system_memory_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) // (1024**3)
    except Exception:
        return None


def init_jvm(input_dir, override_gb=None):
    """Size and start the BioFormats JVM heap to the inputs, mirroring bin/register.py:333-335.

    Used by bin/warp_seg_qc.py, which constructs a real ``Valis`` and reads slides via BioFormats.
    Heap formula: total input size * 3 + 8, min 8, capped at 75% of system memory."""
    # Point scyjava's jgo/Maven cache off a read-only $HOME (HPC nodes) BEFORE the JVM starts.
    # scyjava<1.11 derives the cache path from Path.home() and ignores JGO_CACHE_DIR/M2_REPO, so
    # the Dockerfile ENV knobs are inert; this uses scyjava.config.set_cache_dir instead. Without
    # it, jgo's os.makedirs($HOME/.jgo) dies with EROFS on /hpcnfs. See jvm_cache.py.
    from jvm_cache import point_jvm_cache_off_readonly_home
    from valis import registration

    point_jvm_cache_off_readonly_home()

    if override_gb is not None and override_gb > 0:
        mem_gb = int(override_gb)
    else:
        total_gb = 0.0
        for f in os.listdir(input_dir):
            if f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
                total_gb += os.path.getsize(os.path.join(input_dir, f)) / (1024**3)
        sys_mem = _system_memory_gb()
        max_heap = int(sys_mem * 0.75) if sys_mem else 64
        mem_gb = max(8, min(max_heap, int(total_gb * 3 + 8)))
    registration.init_jvm(mem_gb=mem_gb)
    return mem_gb
