"""Single source of truth for the VALIS registrar configuration.

`bin/register.py` builds its `Valis` object from `build_registrar_kwargs(...)` here, so the
registrar's feature detector, matcher, MicroRigidRegistrar, image-dim caps and affine optimizer
have a single definition. `init_jvm(...)` is the shared BioFormats JVM-heap sizer, also used by
`bin/warp_seg_qc.py` for the reg_qc>=2 segmentation-overlap QC.
"""

import functools
import multiprocessing
import os
import types

import torch
from valis import feature_detectors, feature_matcher
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis.non_rigid_registrars import OpticalFlowWarper

# Keypoints kept per image, for BOTH SuperPoint (detection) and SuperGlue (matching).
#
# VALIS 1.0.0-1.2.0 hard-codes `feature_detectors.MAX_FEATURES = 20000` and copies it into
# every SuperPoint / SuperGlue config at construction time (feature_detectors.py:429,
# feature_matcher.py:1007/1220). SuperGlue's attention map and Sinkhorn matrix are QUADRATIC
# in that count -- ~1.6 GB each per image pair at 20000 -- and serial_rigid.py matches every
# pair concurrently (see bound_cpu_count below). That, not pixel count, is REGISTER's peak:
# measured 2026-08-12 on whole slides at 483 GB, and again 2026-09-10 on five ~2800 px TMA
# cores, which were OOM-killed at "Matching images 0/10" on 128 GB.
#
# The value is VALIS's own 20000 -- what every run so far has used -- by user ruling
# (2026-09-10): the larger memory term turned out to be autograd (see _inference_only
# below), and keeping 20000 keeps new registrations comparable with old ones. The
# mechanism stays so the number has ONE home that reaches both SuperPoint and SuperGlue;
# lowering it (5000 is plenty for a rigid fit at 2048 px and cuts the per-pair footprint
# sixteenfold) is a results-changing decision, not a memory tweak. The presets' old
# `num_features: 5000` never reached VALIS. Guarded by tests/test_valis_matching_bounds.py.
MAX_KEYPOINTS = 20000


def _cap_valis_keypoints(n_keep):
    """Lower VALIS's keypoint ceiling to ``n_keep`` before any detector or matcher exists.

    Must run BEFORE ``MEMORY_PRESETS`` is built: ``SuperGlueMatcher()`` reads
    ``feature_detectors.MAX_FEATURES`` in its constructor, so a cap applied later would
    leave the matcher at 20000 while the detector honoured the cap.

    Parameters
    ----------
    n_keep : int
        Maximum keypoints per image.
    """
    feature_detectors.MAX_FEATURES = n_keep
    # filter_features(kp, desc, n_keep=MAX_FEATURES) bound the OLD value as its default at
    # definition time; the OpenCV-detector paths call it with no argument. Rebind it too.
    filt = getattr(feature_detectors, "filter_features", None)
    if isinstance(filt, types.FunctionType) and filt.__defaults__:
        filt.__defaults__ = (n_keep,)


_cap_valis_keypoints(MAX_KEYPOINTS)


def apply_keypoint_cap(n_keep=None):
    """Set the keypoint ceiling at RUN time, reaching the already-built preset matchers.

    ``_cap_valis_keypoints`` ran at import with MAX_KEYPOINTS; the pipeline's
    ``--reg_valis_max_keypoints`` (register.py ``--max-keypoints``) arrives later, after
    ``MEMORY_PRESETS`` has constructed its ``SuperGlueMatcher()`` instances, each of which
    snapshotted the constant into ``config['superpoint']['max_keypoints']``. So the module
    constant AND every preset matcher's config are rewritten here. SuperPoint detectors
    are constructed later, inside ``Valis``, and read the module constant.

    Parameters
    ----------
    n_keep : int or None
        Keypoints per image. None keeps MAX_KEYPOINTS (VALIS's own 20000).

    Returns
    -------
    int
        The value now in force.
    """
    n = MAX_KEYPOINTS if n_keep is None else int(n_keep)
    _cap_valis_keypoints(n)
    for row in MEMORY_PRESETS.values():
        cfg = getattr(row["matcher"], "config", None)
        if isinstance(cfg, dict) and isinstance(cfg.get("superpoint"), dict):
            cfg["superpoint"]["max_keypoints"] = n
        row["num_features"] = n
    return n


def _inference_only(cls, method_name):
    """Run ``cls.method_name`` under ``torch.no_grad()``.

    VALIS 1.0.0 never disables autograd: ``SuperGlueMatcher._match_images`` builds a fresh
    ``SuperGlue`` and calls it, and ``prep_data`` runs SuperPoint's conv stack on each 2048 px
    image, all with gradient tracking ON -- so every intermediate (18 attention layers of
    ``4 x N x M`` scores and probabilities per pair, plus the conv activations) is RETAINED
    for a backward pass nobody will run. Measured 2026-09-10: four pairs in flight at 5000
    keypoints exceeded a 64 GB cgroup. Grad mode is THREAD-LOCAL in PyTorch, and VALIS
    matches on joblib threads, so ``torch.set_grad_enabled(False)`` in the main thread
    would reach none of them; wrapping the method runs the guard on the calling thread.

    Parameters
    ----------
    cls : type
        The VALIS class to patch.
    method_name : str
        The method to wrap. Skipped when absent or not a plain function (test doubles).
    """
    fn = getattr(cls, method_name, None)
    if not isinstance(fn, types.FunctionType):
        return

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with torch.no_grad():
            return fn(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)


for _cls, _method in (
    (feature_matcher.SuperGlueMatcher, "match_images"),
    (getattr(feature_matcher, "SuperPointAndGlue", None), "match_images"),
    (feature_detectors.SuperPointFD, "detect_and_compute"),
    (feature_detectors.SuperPointFD, "detect_and_compute_sg"),
    (feature_detectors.SuperPointFD, "compute"),
):
    if _cls is not None:
        _inference_only(_cls, _method)


def bound_cpu_count(n_cpus=None):
    """Make VALIS's parallel sections see the CPUs this task was GIVEN, not the node's.

    Six places in VALIS 1.0.0 size their thread pools as
    ``multiprocessing.cpu_count() - 1`` (serial_rigid.py:578/643 -- feature matching --
    micro_rigid_registrar.py:294, non_rigid_registrars.py:1362, slide_io.py:903,
    warp_tools.py:2882). ``cpu_count()`` reports the NODE's cores, so under SLURM a task
    granted 8 CPUs on a 128-core node matches all of its image pairs at once, and the
    per-pair SuperGlue footprint (see MAX_KEYPOINTS) is multiplied by the pair count
    rather than by the CPU budget. Rebinding ``multiprocessing.cpu_count`` is the only
    seam VALIS exposes; it is process-wide, which is what we want -- every VALIS pool
    should respect the allocation.

    Parameters
    ----------
    n_cpus : int or None
        The task's CPU allocation (Nextflow's ``task.cpus``). ``None`` falls back to the
        scheduler affinity mask when the platform has one, else ``os.cpu_count()``.

    Returns
    -------
    int
        The value ``multiprocessing.cpu_count()`` now returns. Never below 2, so that
        VALIS's ``cpu_count() - 1`` stays a legal ``n_jobs``.
    """
    if n_cpus is None:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_cpus = os.cpu_count() or 1
    bounded = max(2, int(n_cpus))
    multiprocessing.cpu_count = lambda: bounded
    # torch's intra-op pool defaults to the node's core count too; that is CPU
    # oversubscription rather than memory, but the allocation is the right ceiling.
    torch.set_num_threads(bounded)
    return bounded


# Memory mode presets — bundle feature detector, matcher, and dimension settings.
# (Kept identical to the historical register.py MEMORY_PRESETS.)
#
# NOT ALL KEYS ARE LIVE. build_registrar_kwargs() below passes `feature_detector_cls`, `matcher`,
# `max_processed_image_dim_px` and `max_non_rigid_registration_dim_px` to Valis(...). It does NOT
# pass the 'low' row's `tile_wh` / `tile_buffer` — those are dead keys that reach nothing, left in
# place because removing them is a behavioural question; there is deliberately no pipeline param
# for them: a knob that changes nothing is worse than no knob. `num_features` reports the ONE
# MAX_KEYPOINTS value applied above, before the matchers below are constructed (it used to say
# 5000 while VALIS ran at its own 20000); the rows restate it only so register.py's settings log
# prints the number that is actually in force.
MEMORY_PRESETS = {
    "high": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "max_processed_image_dim_px": 2048,
        "max_non_rigid_registration_dim_px": 2048,
        "num_features": MAX_KEYPOINTS,
    },
    "medium": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "max_processed_image_dim_px": 1024,
        "max_non_rigid_registration_dim_px": 1024,
        "num_features": MAX_KEYPOINTS,
    },
    "low": {
        "feature_detector_cls": feature_detectors.SuperPointFD,
        "matcher": feature_matcher.SuperGlueMatcher(),
        "num_features": MAX_KEYPOINTS,
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
