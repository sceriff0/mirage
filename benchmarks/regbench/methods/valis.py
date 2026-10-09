"""VALIS (MathOnco/valis, ``pip install valis-wsi``), upstream defaults.

One registration yields three variants:

``valis_rigid``   the rigid stage only
``valis``         rigid + non-rigid, ``Valis.register()`` with its defaults
``valis_micro``   after ``Valis.register_micro()``, the higher-resolution non-rigid pass

The reference is fixed (``reference_img_f``, ``align_to_reference=True``), and points go
through ``Slide.warp_xy_from_to`` into the reference slide's own unwarped pixel frame, so no
crop or canvas offset is involved.

Fluorescence cases are given the one-channel nuclear OME-TIFF; brightfield cases the
original RGB image.

Options (``--opt key=value``); anything not given is VALIS's own default:

``max_image_dim_px`` ``max_processed_image_dim_px`` ``max_non_rigid_registration_dim_px``
``micro_dim``     ``max_non_rigid_registration_dim_px`` of ``register_micro``
``micro_fraction`` size the micro pass as this fraction of the slide's long side, never below
                  VALIS's default (4096 px): ``micro_fraction=0.25`` is the sizing of VALIS's
                  own examples (docs/examples.rst, ``micro_reg_fraction``). When the slide is
                  no larger than the first non-rigid pass there is nothing finer to register,
                  so the pass is skipped and ``valis_micro`` is the ``valis`` result.
``micro=0``       skip the micro pass (no ``valis_micro``)
``micro_rigid=1`` also run ``MicroRigidRegistrar`` inside ``register()`` (off upstream; the
                  mirage pipeline turns it on). Use ``--label`` to keep its results apart.
``valis_src``     a folder holding a ``valis`` package to import instead of the installed
                  one (e.g. a vendored fork)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VARIANTS = ("valis_rigid", "valis", "valis_micro")
REF_NAME, MOV_NAME = "reference", "moving"
_INT_KWARGS = ("max_image_dim_px", "max_processed_image_dim_px",
               "max_non_rigid_registration_dim_px")


def _truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _link(src, dst_dir, name):
    """``dst_dir/<name><all suffixes of src>`` -> ``src``, so the two slides have known names."""
    src = Path(src).resolve()
    dst = Path(dst_dir) / (name + "".join(src.suffixes[-2:] if src.name.lower().endswith(
        (".ome.tif", ".ome.tiff")) else src.suffixes[-1:]))
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)
    return dst


def register(case, work, opts):
    # A read-only $HOME on a cluster breaks numba's on-disk cache during the valis import.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(work) / "numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(work) / "mplconfig"))
    if opts.get("valis_src"):
        sys.path.insert(0, str(Path(opts["valis_src"]).expanduser()))
    import valis
    from valis import registration

    work = Path(work)
    src_dir, dst_dir = work / "in", work / "out"
    src_dir.mkdir(parents=True, exist_ok=True)
    ref_f = _link(case.ref_image, src_dir, REF_NAME)
    mov_f = _link(case.mov_image, src_dir, MOV_NAME)

    kwargs = {k: int(opts[k]) for k in _INT_KWARGS if k in opts}
    if _truthy(opts.get("micro_rigid", 0)):
        from valis.micro_rigid_registrar import MicroRigidRegistrar

        kwargs["micro_rigid_registrar_cls"] = MicroRigidRegistrar
    info = {"version": getattr(valis, "__version__", "?"),
            "source": str(Path(valis.__file__).parent), "kwargs": {k: str(v) for k, v in kwargs.items()},
            "implementation": {
                "package": "valis-wsi", "source": str(Path(valis.__file__).parent),
                "registration": "valis.registration.Valis(...).register(), then .register_micro()",
                "parameters": "upstream defaults",
                "changed_from_defaults": {"reference_img_f": "the benchmark's reference",
                                          "align_to_reference": True,
                                          **{k: str(v) for k, v in kwargs.items()}},
                "point_warp": "valis.registration.Slide.warp_xy_from_to",
                "benchmark_side": ["images passed as files, read by VALIS's own readers"]}}
    try:
        reg = registration.Valis(str(src_dir), str(dst_dir), img_list=[str(ref_f), str(mov_f)],
                                 reference_img_f=str(ref_f), align_to_reference=True, **kwargs)
        reg.register()
        ref, mov = reg.get_slide(str(ref_f)), reg.get_slide(str(mov_f))

        def warper(non_rigid):
            return lambda xy: mov.warp_xy_from_to(xy, ref, non_rigid=non_rigid)

        yield "valis_rigid", warper(False), info
        yield "valis", warper(True), info
        if _truthy(opts.get("micro", 1)):
            micro = {"reference_img_f": str(ref_f), "align_to_reference": True}
            long_side = max(max(case.ref_hw or [0]), max(case.mov_hw or [0]))
            first = kwargs.get("max_non_rigid_registration_dim_px",
                               registration.DEFAULT_MAX_NON_RIGID_REG_SIZE)
            skipped = "micro_fraction" in opts and 0 < long_side <= first
            if "micro_fraction" in opts:
                micro["max_non_rigid_registration_dim_px"] = max(
                    registration.DEFAULT_MAX_MICRO_REG_SIZE,
                    int(round(float(opts["micro_fraction"]) * long_side)))
            if "micro_dim" in opts:
                micro["max_non_rigid_registration_dim_px"] = int(opts["micro_dim"])
            if not skipped:
                reg.register_micro(**micro)
            yield "valis_micro", warper(True), {**info, "micro_skipped": skipped,
                                                "micro": {k: str(v) for k, v in micro.items()}}
    finally:
        try:  # a live BioFormats JVM keeps the interpreter from exiting
            registration.kill_jvm()
        except Exception:
            pass
