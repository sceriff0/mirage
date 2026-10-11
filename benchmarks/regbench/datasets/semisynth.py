"""Real nuclear images under a KNOWN map: the synthetic suite's geometry on real texture.

A window of a real nuclear channel is the reference, untouched. The moving slide is the same
image seen through the synthetic suite's forward map ``g`` (``datasets/synthetic.py``), so
the truth for any moving point is one evaluation of ``g``, exactly as there::

    mov(p) = image(origin + g(p)) + noise

The pixels ``g`` reaches outside the window are read from the slide around it, so the moving
slide has real tissue up to its edges.

Two variants, by where the moving pixels come from:

``same``   (default) the reference image itself. The moving slide gets fresh noise, at
           ``--noise`` times the image's own background noise (robust SD of the pixels below
           the tissue threshold); its real noise and texture are otherwise the reference's,
           resampled.
``cross``  ``--mov-image``: another staining round of the same section, ALREADY registered to
           the reference. Noise, bleaching and lost cells are then real and independent. The
           truth is ``g`` composed with that upstream registration, so its residual is a floor
           under every method's error here.

**Deformations.** Two families have their parameters from published work and make the
headline tier:

``grid``        a 9 x 9 grid of i.i.d. Gaussian control-point moves, SD drawn from 3-7 px,
                spline-interpolated (Casamitjana et al., arXiv:2104.14873)
``multiscale``  random fields at 1/8, 1/16 and 1/32 of the window (the resolutions SynthMorph
                sums, arXiv:2004.10282), scaled so the displacement's RMS is ``--residual-um``
                (2 um): the non-rigid residual measured between two scans of one section is a
                median of 1.6 um (Lotz et al., arXiv:2106.13150) to 1.9 um (Muhlich et al.
                2022). The equal split of energy between the three scales is this file's
                choice; nobody has measured the spectrum of real inter-cycle deformation.

``wave``, ``bumps`` and ``seams`` have parameters without a source and are scored in the
secondary tier, never pooled with the two above.

``--windows K`` test windows are taken where there is most tissue, each paired with every
deformation family; one more window makes the ``dev_*`` cases, the only ones method options
may be chosen on. Test and development windows never overlap.

**The frozen known-truth cases** (``--truth``, ``regbench.truth``). ``truth_spec.json``
fixes the windows per patient, the six families with their amplitudes, and the two poses;
it names no patient. The patients are the cohort's own::

    prepare semisynth --truth --manifest rounds.csv [--patients 046,052 | --patients-file F]

``--manifest`` is the ``rounds.csv`` that ``regbench.arms_manifest`` writes (each patient's
``reference`` and ``ref_channel``, one entry per ``group``) or the arms' raw samplesheet
(``patient_id, path_to_file, is_reference, channels``: the reference row, and the channel
whose name holds a nuclear marker; the files must be TIFFs this reader opens). Every
patient of the manifest is used unless a subset is declared; an id that is not in the
manifest is an error. The list used is written beside the cases
(``truth_patients.json``) and its hash into every case. ``--name --image --channel`` is the
same for ONE patient given by hand (tests); it declares a list of that one patient.

A patient here is (window, family): the untouched window is its reference, kept once, and
each pose is one round. Variant ``same`` only. The window side, the noise and every
amplitude come from the spec, not from the command line. Each cohort patient gives
``windows_per_patient`` x families x poses cases (2 x 6 x 2 = 24) of ``n_px`` (8192) px.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zlib
from pathlib import Path

import numpy as np

from .. import arms_manifest, cells, truth
from ..cases import Case, case_dir, write_case
from ..imageio import pixel_size_um, write_nuclear_tiles
from . import synthetic

_POSE = dict(theta=2.0, shift=(30.0, -20.0))
HEADLINE = ("grid", "multiscale")
FAMILIES = ("grid", "multiscale", "wave", "bumps", "seams")


def family_spec(family, n, px_um, residual_um, seed):
    """The deformation parameters of one case; ``px_um`` turns micrometres into pixels."""
    if family == "grid":
        return dict(
            field="grid", grid_sd=float(np.random.default_rng(seed).uniform(3.0, 7.0))
        )
    if family == "multiscale":
        # three scales, two components each, equal energy: RMS of |w| is residual_um
        rms = residual_um / px_um / math.sqrt(6.0)
        return dict(
            field="multiscale", scales=(n / 8.0, n / 16.0, n / 32.0), rms=(rms,) * 3
        )
    return {
        "wave": dict(amp=10.0, wavelength=1500.0),
        "bumps": dict(field="bumps"),
        "seams": dict(field="seams"),
    }[family]


PROBE = 256  # side of the patches a candidate window is sampled with
_MARGIN = 16


class Source:
    """One channel of a slide, read by window; pixels outside the slide are zero."""

    def __init__(self, path, channel=0):
        from stare.slide_io import open_lazy

        self.arr, self.dtype, self.close = open_lazy(path)
        self.channel = int(channel)
        c, self.h, self.w = self.arr.shape
        if not 0 <= self.channel < c:
            raise SystemExit(f"{path}: channel {channel} out of range for C={c}")

    def read(self, x0, y0, x1, y1):
        out = np.zeros((y1 - y0, x1 - x0), self.dtype)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if cx1 > cx0 and cy1 > cy0:
            out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.asarray(
                self.arr[self.channel, slice(cy0, cy1), slice(cx0, cx1)]
            )
        return out


def pick_windows(src, n, count, threshold):
    """Origins ``(x, y)`` of the ``count`` non-overlapping ``n`` px windows with most tissue,
    best first, and the robust SD of the background pixels met while looking."""
    scored, background = [], []
    for y0 in range(0, max(src.h - n, 0) + 1, n):
        for x0 in range(0, max(src.w - n, 0) + 1, n):
            at = np.linspace(0, n - PROBE, 4).astype(int)
            patches = [
                src.read(x0 + dx, y0 + dy, x0 + dx + PROBE, y0 + dy + PROBE)
                for dy in at
                for dx in at
            ]
            v = np.concatenate([p.ravel() for p in patches]).astype(np.float32)
            scored.append((float((v > threshold).mean()), x0, y0))
            background.append(v[v <= threshold][::16])
    if not scored:
        raise SystemExit(f"the image is smaller than one {n} px window")
    scored.sort(key=lambda t: (-t[0], t[2], t[1]))
    bg = np.concatenate(background) if background else np.zeros(1)
    sd = 1.4826 * float(np.median(np.abs(bg - np.median(bg)))) if len(bg) else 0.0
    return [(x, y, f) for f, x, y in scored[:count]], sd


class RealScene:
    """The reference window and its moving counterpart, a tile at a time."""

    def __init__(self, spec, ref, mov, origin, noise_sd):
        self.spec, self.ref, self.mov = spec, ref, mov
        self.ox, self.oy = origin
        self.noise_sd = float(noise_sd)
        self.n = spec["n"]
        self.g = synthetic.forward_map(spec)

    def tile(self, which, x0, y0, x1, y1):
        from scipy.ndimage import map_coordinates

        if which == "ref":
            return self.ref.read(self.ox + x0, self.oy + y0, self.ox + x1, self.oy + y1)
        yy = np.arange(y0, y1, dtype=float)[:, None]
        xx = np.arange(x0, x1, dtype=float)[None, :]
        q = self.g(np.stack(np.broadcast_arrays(xx, yy), axis=-1))
        bx0, by0 = (int(math.floor(q[..., i].min())) - _MARGIN for i in (0, 1))
        bx1, by1 = (int(math.ceil(q[..., i].max())) + _MARGIN + 1 for i in (0, 1))
        base = self.mov.read(self.ox + bx0, self.oy + by0, self.ox + bx1, self.oy + by1)
        img = map_coordinates(
            base.astype(np.float32),
            [q[..., 1] - by0, q[..., 0] - bx0],
            order=3,
            mode="nearest",
        )
        if self.noise_sd:
            rng = np.random.default_rng([self.spec["seed"], y0, x0])
            img = img + rng.normal(0, self.noise_sd, img.shape).astype(np.float32)
        info = (
            np.iinfo(self.ref.dtype)
            if np.issubdtype(self.ref.dtype, np.integer)
            else None
        )
        if info is not None:
            img = np.rint(img).clip(info.min, info.max)
        return img.astype(self.ref.dtype)


def _write_slide(scene, which, path, dtype, px):
    n = scene.n
    write_nuclear_tiles(
        path,
        (n, n),
        dtype,
        (scene.tile(which, *b) for b in synthetic.tile_boxes((n, n))),
        synthetic.TILE,
        px,
    )


def prepare_one(cases_root, name, spec, a, origin, noise_sd, px, force=False):
    d = case_dir(cases_root, "semisynth", name)
    if (d / "case.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    # a known-truth case whose true map folds is made with the next seed that does not
    spec, reseed = synthetic.settle(spec, px)
    n = spec["n"]
    patient = spec.get("patient")
    ref_src = Source(a.image, a.channel)
    mov_src = Source(a.mov_image, a.mov_channel) if a.mov_image else ref_src
    try:
        scene = RealScene(spec, ref_src, mov_src, origin, noise_sd)
        ref_p, mov_p = d / "ref.ome.tif", d / "mov.ome.tif"
        dtype = ref_src.dtype
        if patient:  # one reference file for every round of the patient
            ref_p = synthetic.write_once(
                synthetic.shared_reference(cases_root, "semisynth", patient),
                lambda tmp: _write_slide(scene, "ref", tmp, dtype, px),
            )
        else:
            _write_slide(scene, "ref", ref_p, dtype, px)
        _write_slide(scene, "mov", mov_p, dtype, px)
    finally:
        ref_src.close()
        if mov_src is not ref_src:
            mov_src.close()
    seg = dict(diameter=a.diameter, workers=a.workers)
    # the segmented objects are the bare nuclei; the outlines are the same labels grown
    (mov_cells, mov_outlines), (ref_cells, ref_outlines) = (
        cells.segment_slide_sets(mov_p, **seg),
        cells.segment_slide_sets(ref_p, **seg),
    )
    if not len(ref_cells) or not len(mov_cells):
        raise RuntimeError(
            f"{name}: no nuclei segmented; check --channel and --diameter"
        )
    g = synthetic.forward_map(spec)
    _, centroids = cells.area_centroid(mov_cells)
    probes = truth.probe_arrays(centroids, g, (n, n), spec["seed"])
    case = Case(
        dataset="semisynth",
        case_id=name,
        group=patient or name,
        family=spec["family"],
        pose=spec.get("pose", ""),
        ref_nuclear=str(ref_p),
        mov_nuclear=str(mov_p),
        ref_image=str(ref_p),
        mov_image=str(mov_p),
        modality="fluorescence",
        diagonal=math.hypot(n, n),
        pixel_size_um=px,
        ref_hw=[n, n],
        mov_hw=[n, n],
        extra={
            "spec": dict(spec),
            "image": str(a.image),
            "mov_image": str(a.mov_image or a.image),
            "variant": "cross" if a.mov_image else "same",
            "origin_xy": list(origin),
            "noise_sd": noise_sd,
            "in_stare_model": spec.get("in_stare_model", ""),
            "truth_spec_sha256": spec.get("truth_spec_sha256", ""),
            "truth_patient": spec.get("truth_patient", ""),
            "truth_patients_sha256": spec.get("truth_patients_sha256", ""),
            **synthetic.known_truth_field(spec, px),
            **({"reseed": reseed} if reseed else {}),
            "cell_sets": synthetic.BUILTIN_SETS,
        },
    )
    return write_case(
        cases_root,
        case,
        probes["truth_mov"],
        probes["truth_ref"],
        truth_cell=probes["truth_cell"],
        **synthetic.two_sets(mov_cells, mov_outlines, ref_cells, ref_outlines, g),
    )


def _pixel_size(a):
    px = a.pixel_size_um or pixel_size_um(a.image)
    if not px:
        raise SystemExit(
            f"{a.image}: no pixel size in the header; give --pixel-size-um "
            "(the multiscale deformation is defined in micrometres)"
        )
    return px


def cases_per_patient(spec):
    """Known-truth cases one cohort patient gives: windows x families x poses."""
    return (
        int(spec["semisynth"]["windows_per_patient"])
        * len(truth.family_names(spec))
        * len(truth.pose_names(spec))
    )


def plan_truth(a, spec=None, n=None):
    """``[(name, spec, origin)]`` and the noise SD of the frozen known-truth cases of ONE
    patient (``a.name``, ``a.image``, ``a.channel``): its windows with most tissue, each
    with every family and both poses.

    ``n`` makes a reduced copy for a test; its cases carry no spec hash.
    """
    spec = spec or truth.load_spec()
    conf = spec["semisynth"]
    patient = str(a.name or "")
    if not patient or not a.image:
        raise SystemExit(
            "--truth: give --manifest (the cohort's list), or --name and --image for "
            "one patient"
        )
    if a.mov_image:
        raise SystemExit("--truth: the frozen cases are the `same` variant only")
    frozen = n is None
    n = int(n or spec["n_px"])
    px = _pixel_size(a)
    src = Source(a.image, a.channel)
    try:
        threshold = cells.slide_threshold(a.image, a.channel)
        windows, bg_sd = pick_windows(src, n, conf["windows_per_patient"], threshold)
    finally:
        src.close()
    if len(windows) < conf["windows_per_patient"]:
        raise SystemExit(
            f"{a.image}: {len(windows)} window(s) of {n} px; the frozen spec asks for "
            f"{conf['windows_per_patient']} per patient"
        )
    key = zlib.crc32(patient.encode()) % 100_000
    jobs = []
    for i, (x, y, _) in enumerate(windows):
        for family in truth.family_names(spec):
            j = truth.family_index(spec, family)  # its own index, not its position
            group = f"{patient}_w{i}_{family}"
            for k, pose in enumerate(truth.pose_names(spec)):
                job = {
                    **synthetic._BASE,
                    "n": n,
                    "family": family,
                    "pose": pose,
                    "patient": group,
                    "seed": conf["seed"] + 1000 * key + 100 * i + 10 * j + k,
                    **truth.pose_kwargs(spec, pose),
                    **truth.family_kwargs(spec, family, px, None if frozen else n),
                    **truth.case_extra(spec, family, frozen),
                }
                jobs.append((f"{group}_{pose}", job, (x, y)))
    return jobs, conf["noise_background_sd"] * bg_sd


# ── the cohort's own list ─────────────────────────────────────────────────────
def _rows(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"--manifest {path}: not found")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [c.strip() for c in reader.fieldnames or []], [
            {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            for row in reader
        ]


def read_manifest(path):
    """``{patient: {"image", "channel", "pixel_size_um"}}`` in file order: each patient's
    reference round and the index of its nuclear channel.

    Two layouts are read. ``rounds.csv`` of ``regbench.arms_manifest`` (``group``,
    ``reference``, ``ref_channel``, ``pixel_size_um``): one row per moving round, so a
    patient appears several times with the same reference. The arms' raw samplesheet
    (``patient_id``, ``path_to_file``, ``is_reference``, ``channels``): the row marked as
    the reference, and the one channel whose name holds a nuclear marker
    (``arms_manifest.nuclear_index``); it has no pixel size.
    """
    columns, rows = _rows(path)
    out = {}
    if {"group", "reference", "ref_channel"} <= set(columns):
        for row in rows:
            patient = row["group"]
            if not patient:
                continue
            px = arms_manifest._pixel_size(row.get("pixel_size_um"))
            entry = {
                "image": row["reference"],
                "channel": int(row["ref_channel"] or 0),
                "pixel_size_um": px,
            }
            have = out.setdefault(patient, entry)
            if (have["image"], have["channel"]) != (entry["image"], entry["channel"]):
                raise SystemExit(
                    f"{path}: patient {patient} has more than one reference "
                    f"({have['image']} channel {have['channel']}, {entry['image']} "
                    f"channel {entry['channel']})"
                )
        return out
    if {"patient_id", "path_to_file", "is_reference", "channels"} <= set(columns):
        per = {}
        for row in rows:
            if row["patient_id"]:
                per.setdefault(row["patient_id"], []).append(row)
        for patient, slides in per.items():
            refs = [r for r in slides if arms_manifest._truthy(r["is_reference"])]
            if len(refs) != 1:
                n = "no" if not refs else str(len(refs))
                raise SystemExit(
                    f"{path}: patient {patient} has {n} reference row"
                    f"{'s' * (len(refs) != 1)}"
                )
            names = [c.strip() for c in refs[0]["channels"].split("|") if c.strip()]
            try:
                channel = arms_manifest.nuclear_index(names)
            except ValueError as exc:
                raise SystemExit(f"{path}: patient {patient}: {exc}") from None
            out[patient] = {
                "image": refs[0]["path_to_file"],
                "channel": channel,
                "pixel_size_um": None,
            }
        return out
    raise SystemExit(
        f"{path}: neither a rounds.csv of regbench.arms_manifest (group, reference, "
        "ref_channel) nor the arms' samplesheet (patient_id, path_to_file, "
        f"is_reference, channels); its columns are {columns}"
    )


def declared_patients(a):
    """The subset given by ``--patients`` or ``--patients-file``; None for every patient."""
    listed, from_file = getattr(a, "patients", None), getattr(a, "patients_file", None)
    if listed and from_file:
        raise SystemExit("give one of --patients and --patients-file, not both")
    if from_file:
        path = Path(from_file)
        if not path.is_file():
            raise SystemExit(f"--patients-file {path}: not found")
        listed = ",".join(
            line.split("#", 1)[0] for line in path.read_text().splitlines()
        )
    if listed is None:
        return None
    ids = [i for i in re.split(r"[,\s]+", listed) if i]
    if not ids:
        raise SystemExit("the declared subset of patients is empty")
    return ids


def select_patients(manifest, wanted, path=""):
    """The patients to build, in the manifest's order; an unknown id is an error."""
    if wanted is None:
        return list(manifest)
    unknown = list(dict.fromkeys(i for i in wanted if i not in manifest))
    if unknown:
        raise SystemExit(
            f"patient(s) {', '.join(unknown)} not in the manifest {path} "
            f"(it has: {', '.join(manifest)})"
        )
    return [p for p in manifest if p in set(wanted)]


def _patient_args(a, patient, entry):
    """``a`` as the one-patient form would have it for a patient of the manifest."""
    return argparse.Namespace(
        **{
            **vars(a),
            "name": patient,
            "image": entry["image"],
            "channel": entry["channel"],
            "mov_image": None,
            # the command line wins, then the manifest, then the file's own header
            "pixel_size_um": getattr(a, "pixel_size_um", None)
            or entry["pixel_size_um"],
        }
    )


def _tag(jobs, record, patient):
    for _, job, _ in jobs:
        if job.get("truth_spec_sha256"):
            job["truth_patient"] = patient
            job["truth_patients_sha256"] = record[truth.HASH_KEY]


def plan_truth_manifest(a, spec=None, n=None):
    """``([(name, spec, origin, args, noise SD, pixel size)], patient list)`` of the frozen
    known-truth cases of every patient of ``a.manifest``, or of the declared subset.

    With ``a.index`` only the patient that case belongs to is opened: an array task does
    not read every slide of the cohort to make one case.
    """
    spec = spec or truth.load_spec()
    manifest = read_manifest(a.manifest)
    if not manifest:
        raise SystemExit(f"--manifest {a.manifest}: no patient in it")
    ids = select_patients(manifest, declared_patients(a), a.manifest)
    sources = {
        p: {"image": manifest[p]["image"], "channel": manifest[p]["channel"]}
        for p in ids
    }
    record = truth.patient_list(ids, spec, a.manifest, sources)
    per = cases_per_patient(spec)
    todo, pick = ids, None
    if a.index is not None:
        if not 0 <= a.index < per * len(ids):
            return [], record
        todo, pick = [ids[a.index // per]], a.index % per
    out = []
    for patient in todo:
        pa = _patient_args(a, patient, manifest[patient])
        jobs, noise = plan_truth(pa, spec, n)
        _tag(jobs, record, patient)
        px = _pixel_size(pa)
        if pick is not None:
            jobs = jobs[pick : pick + 1]
        out += [(*job, pa, noise, px) for job in jobs]
    return out, record


def _plan_truth_one(a, spec=None):
    """The one-patient form as the manifest form returns it: a list of that patient."""
    spec = spec or truth.load_spec()
    jobs, noise = plan_truth(a, spec)
    patient = str(a.name)
    record = truth.patient_list(
        [patient], spec, None, {patient: {"image": str(a.image), "channel": a.channel}}
    )
    _tag(jobs, record, patient)
    px = _pixel_size(a)
    if a.index is not None:
        jobs = jobs[a.index : a.index + 1]
    return [(*job, a, noise, px) for job in jobs], record


def plan(a):
    """``[(name, spec, origin)]`` and the noise SD: the dev window's cases, then the test ones."""
    src = Source(a.image, a.channel)
    try:
        threshold = cells.slide_threshold(a.image, a.channel)
        windows, bg_sd = pick_windows(src, a.n, a.windows + 1, threshold)
    finally:
        src.close()
    px = _pixel_size(a)
    stem = a.name or Path(a.image).name.split(".")[0]
    key = zlib.crc32(stem.encode()) % 100_000
    # the window with the least tissue of those picked is the development one
    n_test = min(a.windows, len(windows) - 1) if len(windows) > 1 else 1
    order = [(len(windows) - 1, "dev")] if len(windows) > 1 else []
    order += [(i, "test") for i in range(n_test)]
    jobs = []
    for i, role in order:
        x, y, _ = windows[i]
        for j, family in enumerate(FAMILIES):
            seed = 1_000_000 + key * 100 + i * 10 + j
            spec = {
                **synthetic._BASE,
                **_POSE,
                "n": a.n,
                "family": family,
                "seed": seed,
                **family_spec(family, a.n, px, a.residual_um, seed),
            }
            name = f"{stem}_w{i}_{family}"
            jobs.append((f"dev_{name}" if role == "dev" else name, spec, (x, y)))
    noise = 0.0 if a.mov_image else a.noise * bg_sd
    return jobs, noise


def prepare(a, spec=None):
    """``spec`` stands in for ``truth_spec.json`` (a test's smaller window)."""
    manifest = getattr(a, "manifest", None)
    if getattr(a, "truth", False):
        if manifest and (a.image or a.name):
            raise SystemExit(
                "--manifest names the patients: leave out --name / --image"
            )
        if not manifest and (
            getattr(a, "patients", None) or getattr(a, "patients_file", None)
        ):
            raise SystemExit("--patients / --patients-file choose among a --manifest's")
        jobs, record = (
            plan_truth_manifest(a, spec) if manifest else _plan_truth_one(a, spec)
        )
        # before any case: a folder holding another list's cases is refused untouched
        where = truth.write_patients(a.cases, record)
        print(
            f"known-truth patients ({len(record['patients'])}): "
            f"{', '.join(record['patients'])} -> {where}"
        )
        for name, job, origin, pa, noise, px in jobs:
            d = prepare_one(a.cases, name, job, pa, origin, noise, px, force=a.force)
            print(f"semisynth/{name} -> {d}")
        return 0
    if manifest or getattr(a, "patients", None) or getattr(a, "patients_file", None):
        raise SystemExit("--manifest / --patients belong to --truth")
    if not a.image:
        raise SystemExit(
            "prepare semisynth: --image is required (or --truth --manifest)"
        )
    jobs, noise = plan(a)
    px = _pixel_size(a)
    if a.index is not None:
        jobs = jobs[a.index : a.index + 1]
    for name, spec, origin in jobs:
        d = prepare_one(a.cases, name, spec, a, origin, noise, px, force=a.force)
        print(f"semisynth/{name} -> {d}")
    return 0
