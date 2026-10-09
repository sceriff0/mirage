"""DeeperHistReg (MWod/DeeperHistReg, ``pip install deeperhistreg``), default initial + non-rigid.

The best-ranked ANHIR leaderboard entry that ships installable code ("DeeperHistReg - NR
1024", the successor of the challenge-winning AGH entry). One variant: ``deeperhistreg``.

**Roles are swapped on purpose.** DeeperHistReg's displacement field is a backward map: it
lives on its *target* grid and says where in its *source* each target pixel is sampled from
(``warped(x) = source(x + u(x))``, ``dhr_utils/utils.py:np_df_to_pyvips_df``). The benchmark
needs moving -> reference, so the benchmark's reference is handed over as DeeperHistReg's
source and the moving slide as its target; the field then *is* the point map, read off at
the moving points, with no inversion. This is the maintainer's own recipe for landmarks
(issue #12).

**What it is given.** An 8-bit RGB image of each slide, at most ``max_dim`` px on its long
side (block-averaged by an integer factor; points are scaled to match). Brightfield cases
are the original image. A fluorescence nuclear channel is percentile-normalised and
inverted to dark-nuclei-on-white, the appearance its preprocessing assumes (pad value 255,
``flip_intensity``).

**From the field to full-resolution points** follows the package's own
``apply_deformation_pyvips``: both images are centre-padded to a common shape, the field is
stretched to that shape and its values scaled by the same ratio.

Options (``--opt key=value``):

``config``    a ``deeperhistreg.configs`` factory name (default ``default_initial_nonrigid``)
``max_dim``   long side of the images handed over (default 8192)
``device``    default ``cuda:0`` when available, else ``cpu``
``direction`` ``swapped`` (default, above) or ``native``: the moving slide as DeeperHistReg's
              source and the reference as its target, the way the package is meant to be
              called. Its field then maps reference -> moving, and the point map is that
              field inverted numerically (:func:`inverse_warper`). Use it with ``--label`` to
              measure what the swap costs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VARIANTS = ("deeperhistreg",)
DEFAULT_MAX_DIM = 8192
BAND = 1024


def block_mean(img, f):
    """Mean over ``f x f`` blocks of an ``(H, W[, C])`` array; the ragged edge is dropped."""
    if f == 1:
        return np.asarray(img, dtype=np.float32)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    v = np.asarray(img[:h, :w], dtype=np.float32)
    return v.reshape(h // f, f, w // f, f, *v.shape[2:]).mean(axis=(1, 3))


def to_rgb8(case, which, out, max_dim):
    """Write the 8-bit RGB image DeeperHistReg reads; returns ``(path, factor, (h, w))``."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # ANHIR's medium images exceed PIL's bomb guard
    path = str(getattr(case, f"{which}_image"))
    if case.modality == "brightfield" and path.lower().endswith((".tif", ".tiff")):
        from ..imageio import (
            read_rgb_reduced,  # a whole-slide TIFF: never read at full size
        )

        rgb, f = read_rgb_reduced(path, max_dim)
    elif case.modality == "brightfield":
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        f = max(1, math.ceil(max(rgb.shape[:2]) / max_dim))
        rgb = block_mean(rgb, f).round().clip(0, 255).astype(np.uint8)
    else:
        import tifffile

        # tifffile alone, so this environment needs none of STARE's pinned I/O stack
        plane = np.squeeze(tifffile.imread(getattr(case, f"{which}_nuclear")))
        if plane.ndim != 2:
            raise ValueError(f"expected one nuclear channel, got shape {plane.shape}")
        h, w = plane.shape
        f = max(1, math.ceil(max(h, w) / max_dim))
        rows = BAND * f  # reduce in bands: the float copy is one band, not the slide
        g = np.concatenate([block_mean(plane[y0 : y0 + rows], f)
                            for y0 in range(0, (h // f) * f, rows)])
        del plane
        lo, hi = np.percentile(g[::4, ::4], (1.0, 99.9))
        g = 255.0 - ((g - lo) / max(hi - lo, 1e-6) * 255.0).clip(0, 255)
        rgb = np.repeat(g.astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(rgb).save(out, compress_level=1)
    return Path(out), f, rgb.shape[:2]


def centre_pads(size_a, size_b):
    """``(top, left)`` padding of each image when both are centre-padded to a common shape
    (``dhr_utils/utils.py:calculate_pad_value``: the leading pad is ``floor(diff / 2)``)."""
    pads = []
    for mine, other in ((size_a, size_b), (size_b, size_a)):
        pads.append(tuple(max(o - m, 0) // 2 for m, o in zip(mine, other)))
    return pads


def _linear_field_warp(landmarks, field):
    """``dhr_utils.warping.warp_landmarks`` with linear interpolation; the stand-in used only
    when the package is not installed (the adapter's own unit tests)."""
    from scipy.ndimage import map_coordinates

    x, y = landmarks[:, 0], landmarks[:, 1]
    ux = map_coordinates(field[0], [y, x], order=1, mode="nearest")
    uy = map_coordinates(field[1], [y, x], order=1, mode="nearest")
    return np.stack([x + ux, y + uy], axis=1)


def field_warper(field, ref_hw, mov_hw, f_ref, f_mov, warp_landmarks=_linear_field_warp):
    """``warp(xy moving px) -> xy reference px`` from a backward field on the padded moving grid.

    ``field`` is ``(2, h, w)``: X then Y displacement, in field pixels. ``ref_hw`` / ``mov_hw``
    are the shapes of the images that were registered, ``f_*`` the factor each was reduced by.
    ``warp_landmarks(xy, field)`` moves points given in FIELD pixels; the adapter passes
    DeeperHistReg's own function, so the only arithmetic here is the change of coordinates.
    """
    field = np.asarray(field, dtype=np.float32)
    _, fh, fw = field.shape
    (pr_y, pr_x), (pm_y, pm_x) = centre_pads(ref_hw, mov_hw)
    hp, wp = max(ref_hw[0], mov_hw[0]), max(ref_hw[1], mov_hw[1])
    sx, sy = wp / fw, hp / fh

    def warp(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        # block averaging by f puts full-resolution pixel p at (p + 0.5) / f - 0.5
        px = (xy[:, 0] + 0.5) / f_mov - 0.5 + pm_x
        py = (xy[:, 1] + 0.5) / f_mov - 0.5 + pm_y
        moved = warp_landmarks(np.stack([(px + 0.5) / sx - 0.5, (py + 0.5) / sy - 0.5], axis=1), field)
        rx = (moved[:, 0] + 0.5) * sx - 0.5 - pr_x
        ry = (moved[:, 1] + 0.5) * sy - 0.5 - pr_y
        return np.stack([(rx + 0.5) * f_ref - 0.5, (ry + 0.5) * f_ref - 0.5], axis=1)

    return warp


def inverse_warper(forward, hw, iters=30, tol=0.05):
    """The inverse of a point map ``forward`` (defined on an ``hw`` image), by Newton steps
    with the Jacobian of ``forward``'s best affine fit. A point whose residual stays above
    1 px comes back NaN, which the scorer counts at its initial position."""
    h, w = hw
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 24), np.linspace(0, h - 1, 24))
    src = np.stack([gx.ravel(), gy.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], forward(src), rcond=None)
    lin_inv = np.linalg.inv(coef[:2].T)

    def warp(xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        x = (xy - coef[2]) @ lin_inv.T
        for _ in range(iters):
            r = xy - forward(x)
            if not len(r) or np.abs(r).max() < tol:
                break
            x = x + r @ lin_inv.T
        x[np.hypot(*(xy - forward(x)).T) > 1.0] = np.nan
        return x

    return warp


def _dist_version(name):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "?"


def _set_device(params, device):
    """The configs bake ``cuda:0`` into nested blocks at creation; rewrite every one of them."""
    for k, v in params.items():
        if isinstance(v, dict):
            _set_device(v, device)
        elif k == "device":
            params[k] = device
        elif k == "cuda":
            params[k] = device != "cpu"


def register(case, work, opts):
    import deeperhistreg
    import SimpleITK as sitk
    import torch

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    max_dim = int(opts.get("max_dim", DEFAULT_MAX_DIM))
    ref_p, f_ref, ref_hw = to_rgb8(case, "ref", work / "reference.png", max_dim)
    mov_p, f_mov, mov_hw = to_rgb8(case, "mov", work / "moving.png", max_dim)

    config = opts.get("config", "default_initial_nonrigid")
    params = getattr(deeperhistreg.configs, config)()
    device = opts.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_device(params, device)
    # Everything that differs from the package's `config` defaults is in `changed`, and is
    # recorded with the run. Nothing in the registration blocks themselves is touched.
    changed = {"loading_params.loader": "pil", "loading_params.source_resample_ratio": 1.0,
               "loading_params.target_resample_ratio": 1.0, "save_final_images": False,
               "device": device}
    params["loading_params"].update(loader="pil", source_resample_ratio=1.0,
                                    target_resample_ratio=1.0)
    params["save_final_images"] = False  # only the field is needed
    out = work / "out"
    direction = opts.get("direction", "swapped")
    if direction not in ("swapped", "native"):
        raise ValueError(f"direction must be swapped or native, got {direction!r}")
    src_p, tgt_p = (ref_p, mov_p) if direction == "swapped" else (mov_p, ref_p)
    deeperhistreg.run_registration(
        source_path=str(src_p), target_path=str(tgt_p), output_path=str(out),
        registration_parameters=params, case_name="case", save_displacement_field=True,
        copy_target=False, delete_temporary_results=False, temporary_path=str(work / "tmp"))
    field_p = out / "displacement_field.mha"
    if not field_p.exists():
        # run_registration prints and swallows the pipeline's exception; surface it here
        log = work / "tmp" / "logs.txt"
        tail = log.read_text()[-600:] if log.exists() else "no log"
        raise RuntimeError(f"DeeperHistReg wrote no displacement field ({tail})")
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(field_p)))
    if field.ndim != 3 or field.shape[0] != 2:
        raise RuntimeError(f"unexpected displacement field shape {field.shape}")
    from dhr_utils import (
        warping,  # the package's own module (it extends sys.path on import)
    )

    info = {"version": _dist_version("deeperhistreg"), "torch": torch.__version__,
            "device": device, "config": config, "direction": direction, "factor_ref": f_ref, "factor_mov": f_mov,
            "field_shape": list(field.shape),
            "implementation": {
                "package": "deeperhistreg", "source": str(Path(deeperhistreg.__file__).parent),
                "registration": "deeperhistreg.run_registration",
                "parameters": f"deeperhistreg.configs.{config}()",
                "changed_from_defaults": changed,
                "point_warp": "dhr_utils.warping.warp_landmarks",
                "benchmark_side": [
                    "roles swapped: reference given as source, moving as target, so the "
                    "backward field is the moving -> reference point map (maintainer's recipe, issue #12)"
                    if direction == "swapped" else
                    "native roles (moving as source); the reference -> moving field is inverted "
                    "numerically for the point map",
                    f"inputs converted to 8-bit RGB, block-averaged x{f_ref} / x{f_mov} to <= {max_dim} px",
                    "fluorescence inverted to dark-on-white" if case.modality != "brightfield" else "brightfield passed as is",
                    "field -> full-resolution coordinates as apply_deformation_pyvips does (centre pad, stretch)",
                ]}}
    pp = out / "postprocessing_params.json"
    if pp.exists():
        info["postprocessing_params"] = json.loads(pp.read_text())
    if direction == "swapped":
        warp = field_warper(field, ref_hw, mov_hw, f_ref, f_mov, warping.warp_landmarks)
    else:
        to_moving = field_warper(field, mov_hw, ref_hw, f_mov, f_ref, warping.warp_landmarks)
        warp = inverse_warper(to_moving, case.ref_hw or [v * f_ref for v in ref_hw])
    yield "deeperhistreg", warp, info
