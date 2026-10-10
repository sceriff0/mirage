"""A stand-in for the ``deeperhistreg`` package, for the adapter's tests where it is not
installed (and neither are ``SimpleITK`` and ``pyvips``; ``torch`` is replaced too, so no
test depends on a GPU or on torch being there).

It does three things and nothing else:

* **records** every call the adapter makes into it (``CALLS``): ``run_registration`` with
  its keywords and the parameters as received, every writer call with its arguments;
* keeps the package's **bookkeeping**, transcribed from MWod/DeeperHistReg main @
  ``42e7c9d`` ``[PARTIAL: the functions named here]``: the two parameter factories the
  benchmark runs (``registration_params.py:484-565, 652-733``), ``calculate_pad_value``
  (``dhr_utils/utils.py:152-180``), ``calculate_resampling_ratio`` (``:530-540``),
  ``warp_landmarks`` (``dhr_utils/warping.py:212-221``), what ``run_registration`` does
  around its swallowed ``try`` (``run.py:19-68``), the device fallback
  (``full_resolution.py:58``), the contents of ``postprocessing_params.json``
  (``full_resolution.py:82-83, 104-106``) and the log lines;
* **draws** a warped image the way ``apply_deformation_pyvips`` is written to (``draw``):
  both slides centre-padded at full resolution, ``grid + u`` enlarged to the canvas the
  way libvips' ``resize(kernel='linear')`` does it (output pixel ``X`` reads the input at
  ``X / S - 0.5``, edge copied; libvips v8.15.2 ``resize.c:233-277``, ``affine.c:445-480,
  320-349``), multiplied by the ratio, the padded moving slide sampled there linearly
  with a zero background (``mapim.c:589-593``), cropped back by the reference's pads. That
  is an emulation written from the source, never the real writer. It writes the plane as
  the first page of a TIFF with a reduced second page, the layout of a libvips pyramid.

It does NOT register anything: ``run_registration`` invents a small similarity and a
smooth field. The selfcheck needs a writer and a point path, not a good registration.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import types
from pathlib import Path

import numpy as np

CALLS = []
# What a test may set before ``run_registration`` runs.
CONFIG = {
    "version": "1.0.1",
    "cuda": False,  # what torch.cuda.is_available() answers
    "torch_threads": 4,
    "initial_resolution": None,  # the working short side; None = the parameters' own
    "rotation_deg": 0.6,
    "shift_px": (3.0, -2.0),  # field pixels
    "amp_px": 1.5,  # field pixels
    "wavelength_px": 60.0,  # field pixels
    "fail": None,  # the text of an exception raised inside the swallowed try
    "writer_fails": None,
}
DEFAULTS = dict(CONFIG)


def configure(**config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown fake settings {sorted(unknown)}")
    CONFIG.clear()
    CONFIG.update(DEFAULTS)
    CONFIG.update(config)
    del CALLS[:]


def record(name, **kw):
    CALLS.append({"call": name, **kw})


def calls(name):
    return [c for c in CALLS if c["call"] == name]


# ── the package's bookkeeping, transcribed ────────────────────────────────────
def calculate_pad_value(size_1, size_2):
    """``dhr_utils/utils.py:152-180``: ``[(top, bottom), (left, right)]`` of each image."""
    y_size_1, x_size_1 = size_1
    y_size_2, x_size_2 = size_2
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad_2[0] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad_1[0] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad_2[1] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad_1[1] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
    return pad_1, pad_2


def calculate_resampling_ratio(x_sizes, y_sizes, min_resolution):
    """``dhr_utils/utils.py:530-540``."""
    min_size = min(max(x_sizes), max(y_sizes))
    return 1 if min_resolution > min_size else min_size / min_resolution


def warp_landmarks(landmarks, displacement_field):
    """``dhr_utils/warping.py:212-221``."""
    import scipy.ndimage as nd

    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(
        displacement_field[0, :, :], [landmarks_y, landmarks_x], mode="nearest"
    )
    uy = nd.map_coordinates(
        displacement_field[1, :, :], [landmarks_y, landmarks_x], mode="nearest"
    )
    return np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)


def _factory(resolution, sizes, angle_step, levels, iterations, rates, alphas):
    params = {
        "device": "cuda:0",
        "echo": False,
        "save_final_images": True,
        "save_final_displacement_field": True,
        "case_name": "Default_Nonrigid",
        "logging_path": None,
        "loading_params": {
            "final_level": 0,
            "pad_value": 255.0,
            "loader": "tiff",
            "source_resample_ratio": 0.2,
            "target_resample_ratio": 0.2,
        },
        "saving_params": {
            "saver": "pil",
            "final_saver": "tiff",
            "save_params": "pil",
            "extension": ".jpg",
            "final_extension": ".tiff",
        },
        "preprocessing_params": {
            "preprocessing_function": "basic_preprocessing",
            "initial_resampling": True,
            "initial_resolution": resolution,
            "normalization": True,
            "convert_to_gray": True,
            "clahe": True,
            "save_results": True,
            "flip_intensity": True,
        },
        "run_initial_registration": True,
        "initial_registration_params": {
            "save_results": True,
            "initial_registration_function": "multi_feature",
            "registration_size": 620,
            "registration_sizes": list(sizes),
            "transform_type": "rigid",
            "keypoint_threshold": 0.005,
            "match_threshold": 0.3,
            "sinkhorn_iterations": 50,
            "show": False,
            "angle_step": angle_step,
            "cuda": True,
            "device": "cuda:0",
            "echo": False,
            "run_sift_ransac": True,
            "run_superpoint_superglue": True,
            "run_superpoint_ransac": False,
        },
        "run_nonrigid_registration": True,
        "nonrigid_registration_params": {
            "save_results": True,
            "nonrigid_registration_function": (
                "instance_optimization_nonrigid_registration"
            ),
            "device": "cuda:0",
            "echo": False,
            "cost_function": "ncc_local",
            "cost_function_params": {"win_size": 7},
            "regularization_function": "diffusion_relative",
            "regularization_function_params": {},
            "registration_size": resolution,
            "num_levels": levels,
            "used_levels": levels,
            "iterations": list(iterations),
            "learning_rates": list(rates),
            "alphas": list(alphas),
        },
    }
    return params


def default_initial_nonrigid():
    """``registration_params.py:484-565``."""
    return _factory(
        4096,
        [150, 200, 250, 300, 350, 400, 450, 500],
        90,
        8,
        [100] * 7 + [200],
        [0.005] + [0.0025] * 6 + [0.0015],
        [1.5] * 7 + [1.8],
    )


def default_initial_nonrigid_high_resolution():
    """``registration_params.py:652-733``."""
    return _factory(
        8192,
        [150, 200, 250, 300, 350, 400, 450, 500, 600],
        60,
        9,
        [100] * 7 + [200, 200],
        [0.005] + [0.0025] * 6 + [0.0015, 0.001],
        [1.5] * 7 + [1.8, 2.1],
    )


# ── files ─────────────────────────────────────────────────────────────────────
def _pages(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        return len(tif.pages)


def _read(path, level=0):
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        return tif.pages[int(level)].asarray()


def _shape(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        return tuple(int(v) for v in tif.pages[0].shape[:2])


def write_field(path, field):
    """The stand-in for ``sitk.WriteImage``: the array, as ``.npy`` bytes, at ``path``."""
    with open(path, "wb") as fh:
        np.save(fh, np.asarray(field, dtype=np.float32))


class _SitkImage:
    def __init__(self, array):
        self.array = array


def _sitk_read(path):
    with open(str(path), "rb") as fh:
        return _SitkImage(np.load(fh))


# ── the invented registration ─────────────────────────────────────────────────
def invent_field(h, w):
    """``(2, h, w)``: X then Y displacement in field pixels, ``source = x + u(x)``."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    th = math.radians(CONFIG["rotation_deg"])
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    ux = math.cos(th) * dx - math.sin(th) * dy + cx - xx + CONFIG["shift_px"][0]
    uy = math.sin(th) * dx + math.cos(th) * dy + cy - yy + CONFIG["shift_px"][1]
    lam, amp = float(CONFIG["wavelength_px"]), float(CONFIG["amp_px"])
    ux = ux + amp * np.sin(2 * math.pi * yy / lam) * np.cos(
        2 * math.pi * xx / (1.7 * lam)
    )
    uy = uy + amp * np.cos(2 * math.pi * xx / lam) * np.sin(
        2 * math.pi * yy / (1.3 * lam)
    )
    return np.stack([ux, uy]).astype(np.float32)


def geometry(source_hw, target_hw, params):
    """The pads, the resample ratio and the field's size, as the pipeline gets them:
    libvips' ``resize`` rounds the loaded size (``reduceh.cpp:411``), the tensors are
    padded (``pair_full_loader.py:75-78``), then resampled by ``1 / k`` with the size
    floored (``utils.py:60-64``, torch's ``interpolate`` with a scale factor)."""
    loading = params["loading_params"]
    r_s, r_t = loading["source_resample_ratio"], loading["target_resample_ratio"]
    src = tuple(int(math.floor(v * r_s + 0.5)) for v in source_hw)
    tgt = tuple(int(math.floor(v * r_t + 0.5)) for v in target_hw)
    pad_1, pad_2 = calculate_pad_value(src, tgt)
    hp, wp = max(src[0], tgt[0]), max(src[1], tgt[1])
    pre = params["preprocessing_params"]
    post = {
        "pad_1": pad_1,
        "pad_2": pad_2,
        "source_resample_ratio": r_s,
        "target_resample_ratio": r_t,
        "initial_resampling": pre["initial_resampling"],
    }
    k = 1
    if pre["initial_resampling"]:
        resolution = CONFIG["initial_resolution"] or pre["initial_resolution"]
        k = calculate_resampling_ratio((wp, wp), (hp, hp), resolution)
        post["initial_resample_ratio"] = k
    field_hw = (int(math.floor(hp * (1 / k))), int(math.floor(wp * (1 / k))))
    return post, field_hw


class _Pipeline:
    """``DeeperHistReg_FullResolution``, reduced to what leaves a trace."""

    def __init__(self, params):
        torch = sys.modules["torch"]
        self.params = params
        # full_resolution.py:58: ONLY the top-level device is rewritten
        params["device"] = params["device"] if torch.cuda.is_available() else "cpu"

    def run_registration(self, source_path, target_path, save_path):
        params = self.params
        nested = (
            params["initial_registration_params"]["device"],
            params["nonrigid_registration_params"]["device"],
        )
        torch = sys.modules["torch"]
        if not torch.cuda.is_available() and any(d != "cpu" for d in nested):
            raise RuntimeError(
                "Attempting to deserialize object on a CUDA device but "
                "torch.cuda.is_available() is False"
            )
        if CONFIG["fail"]:
            raise RuntimeError(CONFIG["fail"])
        for path in (source_path, target_path):
            if _pages(path) != 1:
                raise RuntimeError(
                    f"{path}: {_pages(path)} pages read as pyramid levels"
                )
        post, (h, w) = geometry(_shape(source_path), _shape(target_path), params)
        final = Path(save_path) / params["case_name"] / "Results_Final"
        final.mkdir(parents=True, exist_ok=True)
        write_field(final / "displacement_field.mha", invent_field(h, w))
        (final / "postprocessing_params.json").write_text(json.dumps(post))
        with open(params["logging_path"], "a") as log:
            log.write("Image loading finished.\nLoading time: 1.5 seconds.\n")
            log.write("Preprocessing finished.\nPreprocessing time: 0.25 seconds.\n")
            log.write("Initial registration time: 12.0 seconds.\n")
            log.write("Nonrigid registration time: 30.5 seconds.\n")
            log.write("Total registration time: 44.25 seconds.\n")
            log.write("Final displacement field saved.\n")


def run_registration(**config):
    """``run.py:19-68``, step for step."""
    record(
        "run_registration",
        **{k: v for k, v in config.items() if k != "registration_parameters"},
        parameters=copy.deepcopy(config["registration_parameters"]),
    )
    registration_parameters = config["registration_parameters"]
    source_path = config["source_path"]
    target_path = config["target_path"]
    output_path = Path(config["output_path"])
    experiment_name = config["case_name"]
    save_path = Path(config["temporary_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    try:
        registration_parameters["logging_path"] = save_path / "logs.txt"
        registration_parameters["case_name"] = experiment_name
        pipeline = _Pipeline(registration_parameters)
        pipeline.run_registration(source_path, target_path, save_path)
    except Exception as e:
        print(f"Exception: {e}")
    final = save_path / experiment_name / "Results_Final"
    if registration_parameters["save_final_images"]:
        raise AssertionError("the adapter turns final saving off")
    if config["save_displacement_field"]:
        shutil.copy(
            final / "displacement_field.mha", output_path / "displacement_field.mha"
        )
    try:
        shutil.copy(
            final / "postprocessing_params.json",
            output_path / "postprocessing_params.json",
        )
    except Exception:
        pass
    if config["delete_temporary_results"]:
        shutil.rmtree(save_path)


# ── the writer, emulated ──────────────────────────────────────────────────────
def enlarge(image, out_hw):
    """libvips ``resize`` with ``kernel='linear'`` to a larger size: output pixel ``X``
    reads the input at ``X / S - 0.5`` (per axis), linearly, the edge copied."""
    from scipy.ndimage import map_coordinates

    h, w = image.shape
    sy, sx = out_hw[0] / h, out_hw[1] / w
    yy = np.clip(np.arange(out_hw[0]) / sy - 0.5, 0, h - 1)
    xx = np.clip(np.arange(out_hw[1]) / sx - 0.5, 0, w - 1)
    at = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(image, at, order=1, mode="nearest", prefilter=False)


def draw(source, target_hw, field, pad_value):
    """The plane ``apply_deformation_pyvips`` writes for one source plane."""
    from scipy.ndimage import map_coordinates

    source = np.asarray(source)
    pad_1, pad_2 = calculate_pad_value(source.shape, target_hw)
    padded = np.pad(source, pad_1, mode="constant", constant_values=pad_value)
    hp, wp = padded.shape
    if (hp, wp) != (
        max(source.shape[0], target_hw[0]),
        max(source.shape[1], target_hw[1]),
    ):
        raise AssertionError("the two padded slides do not share a canvas")
    _, fh, fw = field.shape
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    # np_df_to_pyvips_df (utils.py:704-713), warp_pyvips (warping.py:224-239)
    index_x = enlarge(field[0] + xx, (hp, wp)) * (wp / fw)
    index_y = enlarge(field[1] + yy, (hp, wp)) * (hp / fh)
    out = map_coordinates(
        padded.astype(np.float64),
        [index_y, index_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    out = out[:hp, :wp]  # crop_to_template against the padded target
    (top, bottom), (left, right) = pad_2  # unpad with the TARGET's pads
    out = out[top : hp - bottom, left : wp - right]
    if np.issubdtype(source.dtype, np.integer):
        lim = np.iinfo(source.dtype)
        out = np.clip(np.rint(out), lim.min, lim.max)
    return out.astype(source.dtype)


def apply_deformation_pyvips(
    source_image_path,
    target_image_path,
    warped_image_path,
    displacement_field_path,
    loader=None,
    saver=None,
    save_params=None,
    level=0,
    pad_value=255.0,
    save_source_only=True,
    to_template_shape=True,
    to_save_target_path=None,
):
    import tifffile

    record(
        "apply_deformation_pyvips",
        source=str(source_image_path),
        target=str(target_image_path),
        warped=str(warped_image_path),
        field=str(displacement_field_path),
        loader=getattr(loader, "__name__", repr(loader)),
        saver=getattr(saver, "__name__", repr(saver)),
        save_params=save_params,
        level=level,
        pad_value=pad_value,
        save_source_only=save_source_only,
        to_template_shape=to_template_shape,
        to_save_target_path=to_save_target_path,
    )
    if CONFIG["writer_fails"]:
        raise RuntimeError(CONFIG["writer_fails"])
    field = _sitk_read(displacement_field_path).array
    plane = draw(
        _read(source_image_path, level), _shape(target_image_path), field, pad_value
    )
    with tifffile.TiffWriter(str(warped_image_path), bigtiff=True) as tif:
        tif.write(plane, photometric="minisblack", metadata=None)
        tif.write(
            plane[::2, ::2], photometric="minisblack", metadata=None, subfiletype=1
        )


class TIFFLoader:
    pass


class TIFFSaver:
    pass


class PILSaver:
    pass


# ── installing ────────────────────────────────────────────────────────────────
def modules():
    """The fake modules, by the names the adapter imports."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.3.1+fake"
    torch.cuda = types.SimpleNamespace(is_available=lambda: bool(CONFIG["cuda"]))
    torch.get_num_threads = lambda: int(CONFIG["torch_threads"])

    sitk = types.ModuleType("SimpleITK")
    sitk.ReadImage = _sitk_read
    sitk.GetArrayFromImage = lambda image: np.array(image.array)

    pipeline = types.SimpleNamespace(
        loader_mapper={"tiff": TIFFLoader},
        saver_mapper={"tiff": TIFFSaver, "pil": PILSaver},
        adf=types.SimpleNamespace(apply_deformation_pyvips=apply_deformation_pyvips),
        w=types.SimpleNamespace(warp_landmarks=warp_landmarks),
    )
    package = types.ModuleType("deeperhistreg")
    package.__version__ = CONFIG["version"]
    package.__file__ = str(Path(__file__).with_name("deeperhistreg") / "__init__.py")
    package.configs = types.SimpleNamespace(
        default_initial_nonrigid=default_initial_nonrigid,
        default_initial_nonrigid_high_resolution=(
            default_initial_nonrigid_high_resolution
        ),
    )
    package.direct_registration = pipeline
    package.run_registration = run_registration
    return {"deeperhistreg": package, "torch": torch, "SimpleITK": sitk}


def install(setitem=None, **config):
    """Configure the fake and put it in ``sys.modules`` (through ``monkeypatch.setitem``
    when given, so a test leaves nothing behind)."""
    configure(**config)
    put = setitem or (lambda mapping, key, value: mapping.__setitem__(key, value))
    for name, mod in modules().items():
        put(sys.modules, name, mod)
