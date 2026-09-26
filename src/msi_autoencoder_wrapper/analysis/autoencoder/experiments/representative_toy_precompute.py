"""Precomputation for the representative (toy) model campaign.

The toy campaigns train several objectives on one small image with a latent width
``L`` of 4 or 3. Because the bottleneck ``LayerNorm`` places the canonical latent
``u = (z - beta) / gamma`` on ``S^{L-2}(sqrt L)`` inside ``1^perp``, an ``L = 4`` model
has its complete latent geometry on the two-sphere and an ``L = 3`` model on the
circle, which the presentation shows directly in three or two dimensions.

Every quantity requiring a trained model is computed here; notebooks only read the
written tables. Registered analyses:

- ``dataset`` -- pixel coordinates, split, labels and decoded spectra of the image;
- ``training`` -- per-epoch training and validation losses of every variant;
- ``latent`` -- canonical latent, sphere coordinates, Masserstein error, head scores;
- ``sensitivity`` -- latent angle under Fisher--Rao geodesic input perturbations;
- ``contrastive_views`` -- latent angle between a spectrum and its InfoNCE views.

Command line::

    python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.representative_toy_precompute \\
        --settings <analysis_settings.yaml> [--analysis latent]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from ....metrics.strategies.masserstein import SpectrumMasserstein
from ....models.model_loader import ModelLoader
from ....utils.logger import get_custom_logger
from ..latent import sphere_geometry as geometry
from .pretraining_campaign import build_axis_wrapper, decode_source_spectra, split_assignments

logger = get_custom_logger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
ANALYSES: dict[str, Callable[["ToyCampaign"], dict[str, Any]]] = {}
TARGET_FIELD = "molecule"


def register(name: str) -> Callable:
    """Register one analysis under its settings key."""

    def decorator(function: Callable[["ToyCampaign"], dict[str, Any]]) -> Callable:
        ANALYSES[name] = function
        return function

    return decorator


# --------------------------------------------------
# Section: settings and result I/O
# --------------------------------------------------

def load_settings(path: Path | str) -> dict:
    """Read the analysis settings shared by notebooks and this runner.

    :param path: Settings YAML file.
    :type path: pathlib.Path | str
    :return: Settings with ``settings_path`` and ``repository_root`` added.
    :rtype: dict
    """
    settings_path = Path(path).resolve()
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings["settings_path"] = str(settings_path)
    settings["repository_root"] = str(REPOSITORY_ROOT)
    return settings


def results_directory(settings: dict, analysis: str) -> Path:
    """Return the result directory of one registered analysis.

    :raises ValueError: If the analysis has no configured directory.
    """
    configured = settings.get("analyses", {}).get(analysis)
    if configured is None:
        raise ValueError(f"Analysis '{analysis}' is not configured in {settings['settings_path']}.")
    return Path(settings["settings_path"]).parent / configured["result_directory"]


def run_command(settings: dict, analysis: Optional[str] = None, *, background: bool = True) -> str:
    """Return the canonical command that produces the tables of one analysis."""
    command = (
        f"{shlex.quote(str(REPOSITORY_ROOT / '.venv/bin/python'))} -m {__name__} "
        f"--settings {shlex.quote(settings['settings_path'])}"
    )
    if analysis is not None:
        command += f" --analysis {shlex.quote(analysis)}"
    if not background:
        return command
    log = Path(settings["settings_path"]).parent / "precompute.log"
    return f"nohup {command} > {shlex.quote(str(log))} 2>&1 &"


def load_table(settings: dict, analysis: str, table: str) -> pd.DataFrame:
    """Load one result table, or raise with the command that produces it.

    :raises FileNotFoundError: When the table has not been computed.
    """
    path = results_directory(settings, analysis) / f"{table}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {run_command(settings, analysis)}")
    return pd.read_csv(path)


def load_metadata(settings: dict, analysis: str) -> dict:
    """Load the metadata/provenance record of one analysis."""
    path = results_directory(settings, analysis) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing '{path}'. Produce it with:\n  {run_command(settings, analysis)}")
    return json.loads(path.read_text())


def _provenance(settings: dict, device: torch.device) -> dict:
    """Commit, device, seeds and source digests of the producing code."""
    record = {
        "settings_path": settings["settings_path"],
        "device": str(device),
        "seed": settings["seed"],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "source_sha256": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None
    return record


def _write(directory: Path, tables: dict[str, Any], metadata: dict) -> None:
    """Write tables as CSV and metadata as JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    metadata = {**metadata, "tables": sorted(tables)}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def assign_regions(pixels: pd.DataFrame, classes: pd.DataFrame, rules: list[dict]) -> pd.Series:
    """Assign every pixel the first display region whose marker rule it satisfies.

    A rule may require that ``all`` listed ions are present, that ``any`` of them is
    present, and that ``none`` of them is present; ions are identified by their m/z
    rounded to three decimals. A rule without conditions matches every pixel.

    :param pixels: Pixel table with ``label::<class name>`` columns.
    :type pixels: pandas.DataFrame
    :param classes: Class table with ``class_name`` and ``mz``.
    :type classes: pandas.DataFrame
    :param rules: Ordered region rules from the settings file.
    :type rules: list[dict]
    :return: Region name per pixel.
    :rtype: pandas.Series
    :raises KeyError: If a rule references an ion that is not a class.
    """
    columns = {f"{mz:.3f}": f"label::{name}" for name, mz in zip(classes.class_name, classes.mz.round(3))}

    def present(ions: list[str]) -> np.ndarray:
        return pixels[[columns[f"{float(ion):.3f}"] for ion in ions]].to_numpy() > 0  # (N, K)

    region = pd.Series(pd.NA, index=pixels.index, dtype="object")
    for rule in rules:
        match = np.ones(len(pixels), dtype=bool)
        if rule.get("all"):
            match &= present(rule["all"]).all(axis=1)
        if rule.get("any"):
            match &= present(rule["any"]).any(axis=1)
        if rule.get("none"):
            match &= ~present(rule["none"]).any(axis=1)
        region[region.isna() & match] = rule["name"]
    return region


# --------------------------------------------------
# Section: sphere coordinates
# --------------------------------------------------

def orthonormal_complement_basis(dimension: int) -> np.ndarray:
    """Return a fixed orthonormal basis of ``1^perp`` in ``R^D`` (Helmert rows).

    :param dimension: Latent width ``D``.
    :type dimension: int
    :return: Basis with shape ``(D, D - 1)``; columns are orthonormal and orthogonal to ``1``.
    :rtype: numpy.ndarray
    """
    basis = np.zeros((dimension, dimension - 1))
    for column in range(1, dimension):
        basis[:column, column - 1] = 1.0
        basis[column, column - 1] = -float(column)
        basis[:, column - 1] /= np.linalg.norm(basis[:, column - 1])
    return basis  # (D, D - 1)


def sphere_coordinates(u: np.ndarray) -> np.ndarray:
    """Map canonical latents to unit-sphere coordinates in ``1^perp``.

    ``u`` satisfies ``1^T u = 0`` and ``||u|| = sqrt(D)`` up to the LayerNorm epsilon,
    so ``Q^T u / ||u||`` with a fixed orthonormal basis ``Q`` of ``1^perp`` is an
    isometric chart onto ``S^{D-2}``.

    :param u: Canonical latents, shape ``(N, D)``.
    :type u: numpy.ndarray
    :return: Unit vectors, shape ``(N, D - 1)``.
    :rtype: numpy.ndarray
    """
    basis = orthonormal_complement_basis(u.shape[1])  # (D, D - 1)
    projected = u @ basis  # (N, D - 1)
    return projected / np.linalg.norm(projected, axis=1, keepdims=True)  # (N, D - 1)


def procrustes_rotation(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Orthogonal matrix ``R`` minimizing ``||source R - reference||_F``.

    Orthogonal maps are isometries of the sphere, so aligning variants to a common
    reference changes only the viewing orientation, not any angle between points.

    :param source: Points to rotate, shape ``(N, K)``.
    :param reference: Target points, shape ``(N, K)``.
    :return: Orthogonal matrix, shape ``(K, K)``.
    """
    left, _, right = np.linalg.svd(source.T @ reference)  # (K, K), (K,), (K, K)
    return left @ right  # (K, K)


def coordinate_columns(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    """Name sphere coordinates ``sphere_a``, ``sphere_b``, ... (``S^{L-2}`` has ``L - 1``)."""
    return {f"sphere_{'abcdefgh'[index]}": coordinates[:, index] for index in range(coordinates.shape[1])}


def angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Row-wise angle between vectors, in degrees."""
    cosine = np.sum(first * second, axis=-1) / (
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def fisher_rao_geodesic(spectra: torch.Tensor, angle: float, generator: torch.Generator) -> torch.Tensor:
    """Move TIC spectra along random Fisher--Rao geodesics of a fixed length.

    With ``s = sqrt(x)`` on the unit sphere, a random unit tangent ``t`` at ``s`` and
    ``s' = cos(angle) s + sin(angle) t``, the perturbed spectrum ``x' = s'^2`` is again
    on the simplex and its Fisher--Rao distance to ``x`` equals ``2 * angle``.

    :param spectra: TIC-normalized spectra, shape ``(B, M)``.
    :type spectra: torch.Tensor
    :param angle: Geodesic half-length in radians.
    :type angle: float
    :param generator: Random generator on the spectra device.
    :type generator: torch.Generator
    :return: Perturbed TIC-normalized spectra, shape ``(B, M)``.
    :rtype: torch.Tensor
    """
    root = spectra.clamp_min(0.0).sqrt()  # (B, M)
    direction = torch.randn(root.shape, generator=generator, device=root.device, dtype=root.dtype)  # (B, M)
    tangent = direction - (direction * root).sum(dim=1, keepdim=True) * root  # (B, M)
    tangent = tangent / tangent.norm(dim=1, keepdim=True)  # (B, M)
    moved = np.cos(angle) * root + np.sin(angle) * tangent  # (B, M)
    return moved.square()  # (B, M)


# --------------------------------------------------
# Section: campaign access
# --------------------------------------------------

@dataclass
class ToyModel:
    """One trained variant of the toy campaign."""

    name: str
    order: int
    task_id: str
    model: torch.nn.Module
    history: list
    grid_parameters: dict


class ToyCampaign:
    """Lazy access to the toy campaign: dataset, spectra and trained variants.

    :param settings: Resolved settings.
    :type settings: dict
    :param device: Compute device.
    :type device: torch.device
    """

    def __init__(self, settings: dict, device: torch.device) -> None:
        self.settings = settings
        self.device = device

    # Campaign records
    @cached_property
    def execution_directory(self) -> Path:
        """Execution directory of the configured campaign fingerprint."""
        campaign = self.settings["campaign"]
        directory = REPOSITORY_ROOT / campaign["execution_root"] / f"{campaign['experiment_name']}__cfg_{campaign['config_fingerprint']}"
        if not directory.is_dir():
            raise FileNotFoundError(f"Campaign directory '{directory}' does not exist.")
        return directory

    @cached_property
    def records(self) -> list[dict]:
        """Task records in configured variant order (any status)."""
        records = []
        for path in sorted((self.execution_directory / "status").glob("task_*.yaml")):
            if path.name.endswith("-progress.yaml"):
                continue
            records.extend(yaml.safe_load(path.read_text(encoding="utf-8"))["records"].values())
        order = self.settings["campaign"]["variants"]
        records = [record for record in records if record["task"]["grid_parameters"]["objectives"]["name"] in order]
        records.sort(key=lambda record: order.index(record["task"]["grid_parameters"]["objectives"]["name"]))
        return records

    @cached_property
    def parameters(self) -> dict:
        """Resolved task parameters shared by all variants (same data and split)."""
        return self.records[0]["task"]["parameters"]

    # Dataset
    @cached_property
    def wrapper(self) -> Any:
        """Training wrapper (reader, binner, dataset) of the first task."""
        return build_axis_wrapper(self.parameters)

    @cached_property
    def dataset(self) -> Any:
        return self.wrapper.active_dataset

    @cached_property
    def class_names(self) -> list[str]:
        mapping = self.dataset.get_class_mappings()[TARGET_FIELD]
        return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]

    @cached_property
    def dataset_provenance(self) -> dict:
        """Selection record written by the dataset builder."""
        factory = self.parameters["factory_parameters"]
        path = REPOSITORY_ROOT / factory["project_path"] / Path(factory["image_path"]).parent / "provenance.json"
        return json.loads(path.read_text())

    @cached_property
    def classes(self) -> pd.DataFrame:
        """Head columns with their ion m/z."""
        classes = pd.DataFrame({"class_index": range(len(self.class_names)), "class_name": self.class_names})
        classes["mz"] = [next(ion["mz"] for ion in self.dataset_provenance["ions"]
                              if f"{ion['formula']}|{ion['adduct']}" == name) for name in self.class_names]
        return classes

    @cached_property
    def display_source_ids(self) -> np.ndarray:
        """Configured display pixels, or one per annotated region nearest its spatial centroid."""
        configured = self.settings["sensitivity"].get("display_source_ids") or []
        if configured:
            return np.asarray(configured, dtype=np.int64)
        pixels = self.pixels.assign(region=assign_regions(self.pixels, self.classes, self.settings["regions"]))
        selected = []
        for rule in self.settings["regions"]:
            members = pixels[pixels.region == rule["name"]]
            if members.empty or not (rule.get("all") or rule.get("any")):
                continue
            centre = members[["x", "y"]].mean().to_numpy()
            distance = np.linalg.norm(members[["x", "y"]].to_numpy() - centre, axis=1)
            selected.append(int(members.source_id.iloc[int(np.argmin(distance))]))
        return np.asarray(selected, dtype=np.int64)

    @cached_property
    def mass_axis(self) -> np.ndarray:
        return np.asarray(self.dataset.active_context.binner.GetXAxis(), dtype=np.float64)  # (M,)

    @cached_property
    def pixels(self) -> pd.DataFrame:
        """One row per source spectrum: coordinates, split and binary labels."""
        reader = self.dataset.active_context.reader
        count = int(reader.GetNumberOfSpectra())
        assignments = split_assignments(self.parameters)
        split = np.full(count, "excluded", dtype=object)
        for name, identifiers in assignments.items():
            split[identifiers] = name
        positions = np.asarray([reader.GetSpectrumPosition(index)[:2] for index in range(count)])  # (N, 2)
        frame = pd.DataFrame({"source_id": np.arange(count), "x": positions[:, 0] - 1,
                              "y": positions[:, 1] - 1, "split": split})
        ## Source section of every pixel and the display subsample used by latent-space plots
        sources = self.dataset_provenance.get("sources", [])
        frame["image"] = ""
        for source in sources:
            inside = (frame.x >= source["x_offset"]) & (frame.x < source["x_offset"] + source["width"])
            frame.loc[inside, "image"] = source["dataset_id"]
        stride = int(self.settings.get("display_stride", 1))
        local_x = frame.x - frame.image.map({s["dataset_id"]: s["x_offset"] for s in sources}).fillna(0).astype(int)
        frame["display"] = (local_x % stride == 0) & (frame.y % stride == 0)
        targets = self.decoded["targets"]  # (N, C)
        for column, name in enumerate(self.class_names):
            frame[f"label::{name}"] = targets[:, column].astype(int)
        return frame

    @cached_property
    def decoded(self) -> dict[str, np.ndarray]:
        """Binned TIC-normalized spectra and targets of every source spectrum."""
        count = int(self.dataset.active_context.reader.GetNumberOfSpectra())
        return decode_source_spectra(self.dataset, np.arange(count), target_field=TARGET_FIELD, batch_size=512)

    @cached_property
    def spectra(self) -> torch.Tensor:
        return torch.as_tensor(self.decoded["spectra"], device=self.device)  # (N, M)

    # Models
    @cached_property
    def models(self) -> list[ToyModel]:
        """Trained variants in evaluation mode on the compute device."""
        loaded = []
        for order, record in enumerate(self.records):
            if record["status"] != "completed":
                raise RuntimeError(f"Task {record['task']['task_id']} is {record['status']}, not completed.")
            model_path = record["result"]["model_path"]
            model, _, directory = ModelLoader.load_artifact(model_path, strict=True)
            history_path = Path(directory) / "config" / "history.json"
            history = json.loads(history_path.read_text()) if history_path.is_file() else []
            loaded.append(ToyModel(
                name=record["task"]["grid_parameters"]["objectives"]["name"],
                order=order,
                task_id=record["task"]["task_id"],
                model=model.to(self.device).eval(),
                history=history,
                grid_parameters=record["task"]["grid_parameters"],
            ))
        logger.info("Loaded %s toy variants.", len(loaded))
        return loaded

    def canonical(self, variant: ToyModel, spectra: torch.Tensor) -> np.ndarray:
        """Canonical latent ``u`` of a batch, shape ``(B, D)``."""
        gamma, beta = geometry.encoder_layer_norm_parameters(variant.model)
        with torch.no_grad():
            latent = variant.model.encoder(spectra)  # (B, D)
        return geometry.canonicalize(latent.cpu().numpy(), gamma, beta)  # (B, D)

    @cached_property
    def rotations(self) -> dict[str, np.ndarray]:
        """Display rotation of each variant, aligned to the previous variant.

        The first variant keeps the fixed Helmert chart; every next variant is
        Procrustes-aligned to the already-aligned previous one, so consecutive
        slides differ by the learned geometry, not by an arbitrary orientation.
        """
        rotations: dict[str, np.ndarray] = {}
        previous: Optional[np.ndarray] = None
        for variant in self.models:
            coordinates = sphere_coordinates(self.canonical(variant, self.spectra))  # (N, L - 1)
            rotation = np.eye(coordinates.shape[1]) if previous is None else procrustes_rotation(coordinates, previous)
            rotations[variant.name] = rotation
            previous = coordinates @ rotation  # (N, L - 1)
        return rotations

    def display_coordinates(self, variant: ToyModel, spectra: torch.Tensor) -> np.ndarray:
        """Aligned unit-sphere coordinates of a batch, shape ``(B, L - 1)``."""
        return sphere_coordinates(self.canonical(variant, spectra)) @ self.rotations[variant.name]


# --------------------------------------------------
# Section: registered analyses
# --------------------------------------------------

@register("dataset")
def dataset_tables(campaign: ToyCampaign) -> dict[str, Any]:
    """Pixel table, class table and the decoded spectra in long sparse form."""
    spectra = campaign.decoded["spectra"]  # (N, M)
    rows, columns = np.nonzero(spectra > 0)
    long_spectra = pd.DataFrame({
        "source_id": rows,
        "bin": columns,
        "mz": campaign.mass_axis[columns],
        "intensity": spectra[rows, columns],
    })
    return {"tables": {"pixels": campaign.pixels, "classes": campaign.classes, "spectra": long_spectra},
            "metadata": {"bin_count": int(spectra.shape[1]), "mz_range": [float(campaign.mass_axis[0]),
                                                                        float(campaign.mass_axis[-1])],
                         "dataset_provenance": campaign.dataset_provenance}}


@register("training")
def training_tables(campaign: ToyCampaign) -> dict[str, Any]:
    """Per-epoch loss components of every variant."""
    rows = []
    for variant in campaign.models:
        for entry in variant.history:
            metrics = entry.get("metrics", entry)
            if metrics.get("epoch") is None:
                continue
            flat = {"variant": variant.name, "order": variant.order}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat[key] = value
            rows.append(flat)
    return {"tables": {"history": pd.DataFrame(rows)}, "metadata": {}}


@register("latent")
def latent_tables(campaign: ToyCampaign) -> dict[str, Any]:
    """Canonical latents, sphere coordinates, reconstruction error and head scores."""
    metric = SpectrumMasserstein(reduction="none")
    metric.set_mass_axis(torch.as_tensor(campaign.mass_axis, dtype=torch.float32))
    metric = metric.to(campaign.device)
    frames, checks = [], []
    for variant in campaign.models:
        # Forward pass over the whole image
        with torch.no_grad():
            outputs = variant.model(campaign.spectra)
        u = campaign.canonical(variant, campaign.spectra)  # (N, D)
        display = campaign.display_coordinates(variant, campaign.spectra)  # (N, L - 1)
        with torch.no_grad():
            masserstein = metric(outputs["reconstruction"], campaign.spectra,
                                 inputs_tic_normalized=True).cpu().numpy()  # (N,)
        frame = campaign.pixels[["source_id", "x", "y", "split", "image", "display"]].copy()
        frame.insert(0, "variant", variant.name)
        frame.insert(1, "order", variant.order)
        for index in range(u.shape[1]):
            frame[f"u{index}"] = u[:, index]
        for column, values in coordinate_columns(display).items():
            frame[column] = values
        frame["masserstein"] = masserstein
        ## Head scores (untrained for the reconstruction-only variant)
        head_key = next((key for key in outputs if key.startswith("head_")), None)
        if head_key is not None:
            probabilities = torch.sigmoid(outputs[head_key]).cpu().numpy()  # (N, C)
            for column, name in enumerate(campaign.class_names):
                frame[f"probability::{name}"] = probabilities[:, column]
        frames.append(frame)
        check = geometry.verify_canonicalization(u)
        checks.append({"variant": variant.name, **check})
        logger.info("Latent '%s': mean Masserstein %.4f.", variant.name, float(masserstein.mean()))
    return {"tables": {"latent": pd.concat(frames, ignore_index=True),
                       "canonicalization": pd.DataFrame(checks)},
            "metadata": {"rotations": {name: rotation.tolist() for name, rotation in campaign.rotations.items()}}}


@register("sensitivity")
def sensitivity_tables(campaign: ToyCampaign) -> dict[str, Any]:
    """Latent angle under random Fisher--Rao geodesic perturbations of the input."""
    configuration = campaign.settings["sensitivity"]
    generator_seed = int(campaign.settings["seed"])
    angles, display_rows = [], []
    display_ids = campaign.display_source_ids
    for variant in campaign.models:
        # Identical perturbations for every variant: the generator is re-seeded per variant
        generator = torch.Generator(device=campaign.device).manual_seed(generator_seed)
        reference = campaign.canonical(variant, campaign.spectra)  # (N, D)
        for geodesic_angle in configuration["geodesic_angles"]:
            for repeat in range(int(configuration["directions"])):
                perturbed = fisher_rao_geodesic(campaign.spectra, float(geodesic_angle), generator)  # (N, M)
                moved = campaign.canonical(variant, perturbed)  # (N, D)
                angles.append(pd.DataFrame({
                    "variant": variant.name, "order": variant.order,
                    "source_id": campaign.pixels["source_id"], "fisher_rao_distance": 2.0 * float(geodesic_angle),
                    "repeat": repeat, "latent_angle_deg": angle_degrees(reference, moved),
                }))
        ## Display cloud: many directions at one angle for a few pixels
        generator = torch.Generator(device=campaign.device).manual_seed(generator_seed + 1)
        base = campaign.spectra[torch.as_tensor(display_ids, device=campaign.device)]  # (P, M)
        repeated = base.repeat(int(configuration["display_directions"]), 1)  # (P * K, M)
        perturbed = fisher_rao_geodesic(repeated, float(configuration["display_geodesic_angle"]), generator)
        coordinates = campaign.display_coordinates(variant, perturbed)  # (P * K, 3)
        origin = campaign.display_coordinates(variant, base)  # (P, 3)
        source = np.tile(display_ids, int(configuration["display_directions"]))
        display_rows.append(pd.DataFrame({
            "variant": variant.name, "order": variant.order, "source_id": source, "kind": "perturbed",
            **coordinate_columns(coordinates)}))
        display_rows.append(pd.DataFrame({
            "variant": variant.name, "order": variant.order, "source_id": display_ids, "kind": "original",
            **coordinate_columns(origin)}))
    return {"tables": {"angles": pd.concat(angles, ignore_index=True),
                       "display_cloud": pd.concat(display_rows, ignore_index=True)},
            "metadata": {"configuration": configuration, "display_source_ids": display_ids.tolist()}}


@register("contrastive_views")
def contrastive_tables(campaign: ToyCampaign) -> dict[str, Any]:
    """Latent angle between spectra and the peak-permutation views used by InfoNCE.

    The views are produced by the training criterion itself (``InfoNCELoss``) with the
    parameters of the contrastive variant, once per selection method, and reused for
    every variant so that all variants are compared on identical inputs.
    """
    from ....training.criterions.autoencoder.contrastive.infoNCE_loss import MSIInfoNCELoss

    configuration = campaign.settings["contrastive_views"]
    contrastive = next(variant for variant in campaign.models
                       if variant.grid_parameters["objectives"]["contrastive"])
    parameters = deepcopy(contrastive.grid_parameters["objectives"]["contrastive"]["peak_permutation"]["params"])
    identifiers = torch.as_tensor(campaign.pixels["source_id"].to_numpy().copy(), device=campaign.device)
    display_ids = campaign.display_source_ids
    display_positions = torch.as_tensor(
        np.searchsorted(campaign.pixels["source_id"].to_numpy(), display_ids), device=campaign.device
    )
    rows, view_rows, cloud_rows = [], [], []
    for method in configuration["selection_methods"]:
        # View generation by the training criterion
        criterion = MSIInfoNCELoss(**{**parameters, "peak_selection_method": method})
        cache: dict[str, Any] = {}
        criterion.on_phase_start(contrastive.model, campaign.dataset, cache)
        torch.manual_seed(int(campaign.settings["seed"]))
        views = []
        for _ in range(int(configuration["views_per_spectrum"])):
            combined = criterion.on_batch_start((identifiers, campaign.spectra.clone()), cache)[1]  # (2N, M)
            views.append(combined[len(identifiers):])  # (N, M)
        ## Display cloud: many views of a few pixels, mapped by every variant
        display_views = []
        for _ in range(int(configuration["display_views"])):
            combined = criterion.on_batch_start(
                (identifiers[display_positions], campaign.spectra[display_positions].clone()), cache
            )[1]  # (2P, M)
            display_views.append(combined[len(display_ids):])  # (P, M)
        stacked = torch.cat(display_views, dim=0)  # (P * K, M)
        for variant in campaign.models:
            for kind, batch, source in (
                ("original", campaign.spectra[display_positions], display_ids),
                ("view", stacked, np.tile(display_ids, int(configuration["display_views"]))),
            ):
                coordinates = campaign.display_coordinates(variant, batch)  # (B, 3)
                cloud_rows.append(pd.DataFrame({
                    "variant": variant.name, "order": variant.order, "method": method, "source_id": source,
                    "kind": kind, **coordinate_columns(coordinates)}))
        for view_index, view in enumerate(views):
            changed = (view - campaign.spectra).abs().sum(dim=1).cpu().numpy()  # (N,)
            view_rows.append(pd.DataFrame({"method": method, "view": view_index,
                                           "source_id": campaign.pixels["source_id"], "l1_change": changed}))
            for variant in campaign.models:
                reference = campaign.canonical(variant, campaign.spectra)  # (N, D)
                moved = campaign.canonical(variant, view)  # (N, D)
                rows.append(pd.DataFrame({
                    "variant": variant.name, "order": variant.order, "method": method, "view": view_index,
                    "source_id": campaign.pixels["source_id"], "latent_angle_deg": angle_degrees(reference, moved),
                }))
    return {"tables": {"angles": pd.concat(rows, ignore_index=True),
                       "views": pd.concat(view_rows, ignore_index=True),
                       "display_cloud": pd.concat(cloud_rows, ignore_index=True)},
            "metadata": {"criterion_parameters": parameters, "configuration": configuration}}


# --------------------------------------------------
# Section: runner
# --------------------------------------------------

def resolve_device(settings: dict, *, allow_cpu: bool = False) -> torch.device:
    """Return the configured device, refusing a silent CPU fallback.

    :raises RuntimeError: If CUDA is configured but unavailable and ``allow_cpu`` is false.
    """
    requested = str(settings.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is configured but unavailable; pass --allow-cpu to run on CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def precompute(settings: dict, analyses: Optional[list[str]] = None, *, allow_cpu: bool = False) -> list[Path]:
    """Run registered analyses and write their tables.

    :param settings: Resolved settings.
    :type settings: dict
    :param analyses: Analysis keys; ``None`` runs every configured analysis.
    :type analyses: list[str] | None
    :param allow_cpu: Permit CPU when CUDA is configured but unavailable.
    :type allow_cpu: bool
    :return: Written result directories.
    :rtype: list[pathlib.Path]
    """
    device = resolve_device(settings, allow_cpu=allow_cpu)
    campaign = ToyCampaign(settings, device)
    selected = analyses or list(settings["analyses"])
    written = []
    for name in selected:
        logger.info("Running toy analysis '%s' on %s.", name, device)
        result = ANALYSES[name](campaign)
        directory = results_directory(settings, name)
        _write(directory, result["tables"], {"analysis": name, "variants": [m.name for m in campaign.models]
                                             if name != "dataset" else [],
                                             "provenance": _provenance(settings, device), **result["metadata"]})
        written.append(directory)
    return written


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Precompute toy-campaign analysis tables.")
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--analysis", action="append", choices=sorted(ANALYSES))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    for directory in precompute(load_settings(args.settings), args.analysis, allow_cpu=args.allow_cpu):
        print(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
