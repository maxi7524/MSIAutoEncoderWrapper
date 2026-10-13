"""Read external MSI references without fitting preprocessing or inventing features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference import PixelMask
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class EvaluationImage:
    """Canonical source image with an explicit input representation contract.

    :param dataset_id: External collection identity.
    :param image_id: Acquisition/sample identity.
    :param pixel_ids: Original serialized-object row indices, shape ``(N,)``.
    :param values: Features in source scale, shape ``(N, M)``.
    :param mass_axis: Increasing source m/z coordinates, shape ``(M,)``.
    :param xy: Source positions, shape ``(N, 2)``, with units in metadata.
    :param annotation: Optional identity-aligned source mask.
    :param metadata: Source hashes, representation, units and usage role.
    """

    dataset_id: str
    image_id: str
    pixel_ids: np.ndarray
    values: np.ndarray
    mass_axis: np.ndarray
    xy: np.ndarray
    annotation: PixelMask | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n, m = self.values.shape if self.values.ndim == 2 else (0, 0)
        if n == 0 or m == 0 or self.pixel_ids.shape != (n,) or self.xy.shape != (n, 2) or self.mass_axis.shape != (m,):
            raise ValueError("Evaluation image arrays are inconsistent.")
        if self.pixel_ids.dtype.kind not in "iu" or np.any(self.pixel_ids < 0) or len(np.unique(self.pixel_ids)) != n:
            raise ValueError("Evaluation image requires unique source pixel IDs.")
        if not all(np.isfinite(a).all() for a in [self.values, self.mass_axis, self.xy]) or np.any(np.diff(self.mass_axis) <= 0):
            raise ValueError("Evaluation arrays must be finite and m/z strictly increasing.")
        if self.metadata.get("usage_role") != "external_test_only":
            raise ValueError("External loaders require external_test_only metadata.")
        if self.annotation is not None:
            if self.annotation.metadata["dataset_id"] != self.dataset_id or self.annotation.metadata["image_id"] != self.image_id or self.annotation.metadata["coordinate_space"] != "msi_source":
                raise ValueError("Annotation belongs to a different image or coordinate space.")
            self.annotation.align_to(self.pixel_ids)

    def assert_use(self, purpose: str) -> None:
        """Allow evaluation only; reject every training/fit/selection purpose.

        :param purpose: ``inference``, ``fixed_clustering`` or ``reporting``.
        :raises ValueError: For an unauthorized purpose.
        """
        if purpose not in {"inference", "fixed_clustering", "reporting"}:
            raise ValueError("External references cannot be used for training, fitting or model selection.")

    def assert_model_input(self, representation: str) -> None:
        """Reject an incompatible representation rather than zero-filling missing peaks.

        :param representation: Required representation identifier.
        :raises ValueError: When source and required representations differ.
        """
        if representation != self.metadata["representation"]:
            raise ValueError(f"Source representation {self.metadata['representation']} does not satisfy {representation}.")


def _rdata():
    """Import the optional evaluation dependency with an actionable error."""
    try:
        import rdata
    except ImportError as error:
        raise ImportError("Install the evaluation extra: uv sync --extra cu118 --extra evaluation.") from error
    return rdata


def _scalar(value: Any) -> Any:
    """Require one source metadata scalar."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar source metadata field.")
    return array.ravel()[0].item()


def poseidon_from_frames(coords: Any, assay: Any, *, assay_name: str, row_alignment: str = "keys") -> EvaluationImage:
    """Validate and align source Poseidon DataFrames without intensity transforms.

    :param coords: Source coordinate DataFrame with X/Y columns.
    :param assay: Source feature DataFrame with numeric m/z column names.
    :param assay_name: Lipids, Glycans or Peptides.
    :param row_alignment: keys, or explicitly documented author_positional order.
    :return: Canonical selected-feature image, not a raw-spectrum model input.
    :raises ValueError: For inconsistent source ordering or feature axes.
    """
    if list(coords.columns) != ["X", "Y"] or len(coords) != len(assay):
        raise ValueError("Poseidon coordinate and assay arrays disagree.")
    if row_alignment not in {"keys", "author_positional"}:
        raise ValueError("Unknown source row alignment policy.")
    if row_alignment == "keys" and not coords.index.equals(assay.index):
        raise ValueError("Poseidon coordinate and assay row keys must match exactly.")
    axis = np.asarray(assay.columns, dtype=float)  # (M,)
    order = np.argsort(axis)  # (M,)
    return EvaluationImage("poseidon_ccrcc", "ccrcc_section_1", np.arange(len(coords)),
                           assay.to_numpy()[:, order], axis[order], coords.to_numpy(), None,
                           {"usage_role": "external_test_only", "representation": "selected_transformed_features",
                            "assay": assay_name, "coordinate_units": "source_grid", "spacing_um": None,
                            "source_feature_order": order.tolist(), "registration_status": "pending",
                            "row_alignment": row_alignment, "source_coordinate_row_names": [str(i) for i in coords.index],
                            "source_assay_row_names": [str(i) for i in assay.index]})


def load_poseidon(directory: Path | str, *, assay: str = "Lipids") -> EvaluationImage:
    """Read one Poseidon assay and its original coordinates.

    :param directory: Dataset directory containing raw/FinalMatrix files.
    :param assay: Lipids, Glycans or Peptides.
    :return: Validated selected-feature image.
    :raises ValueError: If assay or source arrays are invalid.
    """
    if assay not in {"Lipids", "Glycans", "Peptides"}:
        raise ValueError("Unknown Poseidon assay.")
    root, r = Path(directory) / "raw", _rdata()
    # REMARK: The pinned author script 040_Extract_results_Import_data.R removes
    # assay row names and pairs all matrices by position. Coordinate row names
    # preserve older filtered indices, while assays have been renumbered.
    logger.info("Reading Poseidon %s source features.", assay)
    return poseidon_from_frames(r.read_rds(root / "FinalMatrix_Coor.RDS"),
                                r.read_rds(root / f"FinalMatrix_{assay}.RDS"), assay_name=assay, row_alignment="author_positional")


def blocksam_from_objects(objects: dict[str, Any], sample_name: str) -> EvaluationImage:
    """Extract one BlockSAM sample with source row IDs and diagnostic labels.

    :param objects: Converted RData contents.
    :param sample_name: Exact pixelInfoAll.name value; no inferred aliases.
    :return: Source-aligned selected-feature image and Normal/RCC annotation.
    :raises ValueError: For unknown samples or inconsistent objects.
    """
    infos = objects["pixelInfoAll"]
    axis, matrix = np.asarray(objects["filtered_mz"]), np.asarray(objects["xall"])  # (M,), (N_all, M)
    if matrix.shape != (len(infos), len(axis)):
        raise ValueError("BlockSAM spectra, m/z and pixel metadata disagree.")
    names = np.asarray([str(_scalar(info.name)) for info in infos])  # (N_all,)
    rows = np.flatnonzero(names == sample_name)  # (N,)
    if not len(rows):
        raise ValueError(f"Unknown BlockSAM sample: {sample_name}.")
    records = [infos[i] for i in rows]
    vocabulary = {"Normal": 0, "RCC": 1}
    labels = np.asarray([vocabulary[str(_scalar(info.className))] for info in records], dtype=np.int32)  # (N,)
    xy = np.asarray([[_scalar(info.x), _scalar(info.y)] for info in records], dtype=float)  # (N, 2)
    mask = PixelMask(rows, labels, {"dataset_id": "blocksam_kidney", "image_id": sample_name,
                     "coordinate_space": "msi_source", "class_names": {"0": "normal", "1": "rcc"},
                     "usage_role": "external_test_only", "annotation_status": "source_diagnostic_labels",
                     "provenance": {"field": "pixelInfoAll.className", "not_anatomical_subregions": True}})
    order = np.argsort(axis)  # (M,)
    return EvaluationImage("blocksam_kidney", sample_name, rows, matrix[rows][:, order], axis[order], xy, mask,
                           {"usage_role": "external_test_only", "representation": "source_filtered_tic_features",
                            "coordinate_units": "unconfirmed_source_units", "spacing_um": None,
                            "patient_ids": sorted({int(_scalar(info.patNum)) for info in records}),
                            "source_feature_order": order.tolist()})


def load_blocksam(directory: Path | str, *, sample_name: str) -> EvaluationImage:
    """Decode BlockSAM once; use canonical caching for repeated model evaluations.

    :param directory: Dataset directory containing raw/kidney_desi_msi.RData.
    :param sample_name: Exact source sample name.
    :return: One canonical source sample.
    """
    logger.info("Decoding BlockSAM RData for sample %s.", sample_name)
    objects = _rdata().read_rda(Path(directory) / "raw" / "kidney_desi_msi.RData")
    return blocksam_from_objects(objects, sample_name)



def _stream_hash(handle: Any) -> str:
    """Hash file contents with bounded memory on supported Python versions."""
    import hashlib
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def save_evaluation_image(image: EvaluationImage, directory: Path | str) -> None:
    """Cache decoded arrays once, preserving float precision and source identities.

    :param image: Canonical image.
    :param directory: Cache destination outside the versioned tree.
    """
    import json
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ["pixel_ids", "values", "mass_axis", "xy"]:
        path = root / f"{name}.npy"
        np.save(path, getattr(image, name), allow_pickle=False)
        with path.open("rb") as handle:
            hashes[path.name] = _stream_hash(handle)
    if image.annotation:
        from .reference import save_mask
        save_mask(image.annotation, root / "annotation")
    manifest = {"schema_id": "msi_evaluation_image", "schema_version": 1, "dataset_id": image.dataset_id,
                "image_id": image.image_id, "metadata": image.metadata, "hashes": hashes,
                "annotation": image.annotation is not None}
    (root / "image_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def load_evaluation_image(directory: Path | str) -> EvaluationImage:
    """Verify cache hashes and load numeric arrays using read-only memory maps.

    :param directory: Cache produced by save_evaluation_image.
    :return: Source image; no preprocessing fit occurs.
    :raises ValueError: For incompatible schemas or hash mismatch.
    """
    import json
    from .reference import load_mask
    root = Path(directory)
    manifest = json.loads((root / "image_manifest.json").read_text())
    if manifest.get("schema_id") != "msi_evaluation_image" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation image schema.")
    names = ["pixel_ids", "values", "mass_axis", "xy"]
    if set(manifest["hashes"]) != {f"{name}.npy" for name in names}:
        raise ValueError("Incomplete cache array inventory.")
    for name in names:
        with (root / f"{name}.npy").open("rb") as handle:
            if _stream_hash(handle) != manifest["hashes"][f"{name}.npy"]:
                raise ValueError(f"Evaluation checksum mismatch: {name}.")
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
    return EvaluationImage(manifest["dataset_id"], manifest["image_id"], **arrays,
                           annotation=load_mask(root / "annotation") if manifest["annotation"] else None,
                           metadata=manifest["metadata"])


def cardinal_from_arrays(pixel_data: Any, mass_axis: np.ndarray, spectra: np.ndarray, sample_name: str) -> EvaluationImage:
    """Extract a Cardinal sample from its documented feature-by-pixel matrix.

    :param pixel_data: Source DataFrame with x, y, sample and diagnosis.
    :param mass_axis: Source m/z vector, shape ``(M,)``.
    :param spectra: Source intensity matrix, shape ``(M, N_all)``.
    :param sample_name: Exact Cardinal acquisition identity.
    :return: Image with available cancer/normal diagnostic labels only.
    :raises ValueError: For inconsistent objects or unknown labels/samples.
    """
    axis = np.asarray(mass_axis)  # (M,)
    if spectra.shape != (len(axis), len(pixel_data)) or not {"x", "y", "sample", "diagnosis"}.issubset(pixel_data.columns):
        raise ValueError("Cardinal feature-by-pixel matrix and source metadata disagree.")
    rows = np.flatnonzero(pixel_data["sample"].astype(str).to_numpy() == sample_name)  # (N,)
    if not len(rows):
        raise ValueError(f"Unknown Cardinal sample: {sample_name}.")
    selected = pixel_data.iloc[rows]
    vocabulary = {"normal": 0, "cancer": 1}
    labels = np.full(len(rows), -1, dtype=np.int32)  # (N,)
    for j, value in enumerate(selected["diagnosis"]):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if str(value) not in vocabulary:
            raise ValueError(f"Unknown Cardinal diagnosis: {value}.")
        labels[j] = vocabulary[str(value)]
    mask = PixelMask(rows, labels, {"dataset_id": "cardinal_rcc", "image_id": sample_name,
                     "coordinate_space": "msi_source", "class_names": {"0": "normal", "1": "cancer"},
                     "usage_role": "external_test_only", "annotation_status": "source_diagnostic_labels",
                     "provenance": {"field": "rcc.pixelData.data.diagnosis", "not_anatomical_subregions": True}})
    order = np.argsort(axis)  # (M,)
    values = spectra[:, rows].T[:, order]  # (N, M)
    return EvaluationImage("cardinal_rcc", sample_name, rows, values, axis[order],
                           selected[["x", "y"]].to_numpy(), mask,
                           {"usage_role": "external_test_only", "representation": "source_binned_intensity",
                            "coordinate_units": "source_grid", "spacing_um": None,
                            "source_feature_order": order.tolist()})


def _r_slots(obj: Any, r: Any) -> dict[str, Any]:
    """Read named attributes from the public parsed RObject representation."""
    slots = {}
    node = obj.attributes
    while node is not None and node.info.type == r.parser.RObjectType.LIST:
        name = str(r.conversion.convert(node.tag))
        if name in slots:
            raise ValueError("Duplicate serialized R slot.")
        slots[name] = node.value[0]
        node = node.value[1]
    return slots


def load_cardinal(directory: Path | str, *, sample_name: str = "MH0204_33") -> EvaluationImage:
    """Read required Cardinal slots without converting unrelated legacy classes.

    REMARK: Full rdata conversion of this release fails on named dimensions in
    an unused positionArray. Selective conversion uses public parser objects and
    preserves the source iData matrix and pixelData; no library patch is applied.

    :param directory: Dataset directory containing raw/rcc.rda.
    :param sample_name: Exact acquisition name.
    :return: Source image and diagnosis labels, with unknown pixels preserved.
    :raises ValueError: For an unsupported release/object layout.
    """
    r = _rdata()
    logger.info("Parsing Cardinal source object for sample %s.", sample_name)
    parsed = r.parser.parse_file(Path(directory) / "raw" / "rcc.rda")
    node, root = parsed.object, None
    while node.info.type == r.parser.RObjectType.LIST:
        if str(r.conversion.convert(node.tag)) == "rcc":
            root = node.value[0]
            break
        node = node.value[1]
    if root is None or root.info.type != r.parser.RObjectType.S4:
        raise ValueError("Expected the source Cardinal rcc S4 object.")
    slots = _r_slots(root, r)
    image_slots = _r_slots(slots["imageData"], r)
    pixel_slots = _r_slots(slots["pixelData"], r)
    feature_slots = _r_slots(slots["featureData"], r)
    environment = r.conversion.convert(image_slots["data"])
    pixel_data = r.conversion.convert(pixel_slots["data"])
    feature_data = r.conversion.convert(feature_slots["data"])
    image = cardinal_from_arrays(pixel_data, feature_data["mz"].to_numpy(), environment["iData"], sample_name)
    return image
