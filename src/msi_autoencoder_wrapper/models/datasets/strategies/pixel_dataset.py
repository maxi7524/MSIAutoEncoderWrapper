"""
Concrete dataset strategy executing single-pixel spectra mapping sequences driven by an active context.
"""

from typing import Tuple, Any, Optional, Literal, Mapping, Sequence, Dict
import torch
import numpy as np

from ..dataset_manager import DatasetManager
from ..base_dataset import MSIBaseDataset
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error
from ....core.mixins.active_context.active_context_mixin import ActiveContextProxy
from ..class_assignment import build_class_mapping, metadata_values, molecule_key

# Logger initialization
logger = get_custom_logger(__name__)


@DatasetManager.register_dataset("PixelDataset")
class PixelDataset(MSIBaseDataset):
    """
    Concrete single-pixel sampling strategy that pulls raw arrays and maps them onto uniform grids.
    """

    def __init__(
        self,
        active_context: Optional[ActiveContextProxy] = None,
        source: Literal["image", "latent"] = "image",
        normalization: Optional[Literal["none", "tic", "max", "l2"]] = None,
        normalization_epsilon: float = 1e-12,
        target_fields: Optional[Sequence[str]] = None,
        class_mappings: Optional[Mapping[str, Mapping[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructs the independent pixel sampling dataset pipeline layer.

        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[ActiveContextProxy]
        :param source: Image or latent data source.
        :type source: Literal["image", "latent"]
        :param normalization: Spectrum scaling. Defaults to ``tic`` for images
            and ``none`` for latent data.
        :type normalization: Optional[Literal["none", "tic", "max", "l2"]]
        :param normalization_epsilon: Positive denominator safety threshold.
        :type normalization_epsilon: float
        :param target_fields: Metadata fields and/or ``molecule`` to return as
            targets. When omitted, the historical two-item sample is returned.
        :type target_fields: Sequence[str] | None
        :param class_mappings: Optional explicit semantic value-to-class maps.
        :type class_mappings: Mapping[str, Mapping[str, int]] | None
        :raises ValidationError: If normalization settings are invalid.
        """
        super().__init__(active_context=active_context, **kwargs)
        resolved_normalization = normalization or (
            "none" if source == "latent" else "tic"
        )
        if resolved_normalization not in {"none", "tic", "max", "l2"}:
            raise_validation_error(
                context_name="PixelDataset",
                message=(
                    "normalization must be 'none', 'tic', 'max', or 'l2'."
                ),
            )
        if normalization_epsilon <= 0:
            raise_validation_error(
                context_name="PixelDataset",
                message="normalization_epsilon must be greater than zero.",
            )
        self.source = source
        self.normalization = resolved_normalization
        self.normalization_epsilon = float(normalization_epsilon)
        self.target_fields = tuple(target_fields or ())
        self.class_mappings = {
            str(field): {str(value): int(index) for value, index in mapping.items()}
            for field, mapping in dict(class_mappings or {}).items()
        }
        self._resolved_class_mappings: Optional[Dict[str, Dict[str, int]]] = None
        self._config = {
            "source": source,
            "normalization": resolved_normalization,
            "normalization_epsilon": self.normalization_epsilon,
            "target_fields": list(self.target_fields),
            "class_mappings": self.class_mappings,
        }

    def __len__(self) -> int:
        """
        Retrieves total pixel coordinates count exposed by the underlying storage loader.

        :return: Absolute size of the total spatial image layout.
        :rtype: int
        :raises ValueError: If the attached active context reader session is unassigned.
        """
        # Session state verification
        if not self.active_context:
            raise_validation_error(
                context_name="PixelDataset",
                message="The active image context has no reader instance.",
            )
            
        return self.active_context.get_data_reader(self.source).GetNumberOfSpectra()

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Extracts and resolves a singular experimental spectrum onto the active target grid.

        :param idx: Flat position tracking coordinate index targeting an explicit single tissue pixel.
        :type idx: int
        :return: Aligned tuple holding the unique flat spatial key index token and its intensity tensor.
        :rtype: Tuple[int, torch.Tensor]
        """
        # Context extraction layer
        reader = self.active_context.get_data_reader(self.source)

        # Read raw variables arrays from data drivers
        xs, ys = reader.GetSpectrum(idx)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            raise_validation_error(
                context_name="PixelDataset",
                message=f"Spectrum {idx} contains non-finite input values.",
            )

        if self.source == "latent":
            values = np.array(ys, dtype=np.float32, copy=True)
            normalized_values = self._normalize(values)
            return self._sample(idx, torch.from_numpy(normalized_values))

        binner = self.active_context.binner

        # Transformation execution pipeline block
        ## Map and normalize one spectrum without hiding invalid reader output
        mapped_values = np.asarray(binner(xs=xs, ys=ys), dtype=np.float32)
        if not np.all(np.isfinite(mapped_values)):
            raise_validation_error(
                context_name="PixelDataset",
                message=f"Binned spectrum {idx} contains non-finite values.",
            )
        return self._sample(idx, torch.from_numpy(self._normalize(mapped_values)))

    def get_class_mappings(self) -> Dict[str, Dict[str, int]]:
        """Return deterministic target mappings for the active annotation store.

        :return: Mapping from target field to semantic value-to-index mapping.
        :rtype: Dict[str, Dict[str, int]]
        :raises ValidationError: If targets are requested without an annotation
            reader in the active context.
        """
        if self._resolved_class_mappings is not None:
            return self._resolved_class_mappings
        annotation_reader = getattr(self.active_context, "annotation_reader", None)
        if annotation_reader is None:
            raise_validation_error(
                "PixelDataset",
                "target_fields require an annotation reader in the active context.",
            )
        metadata = annotation_reader.get_dataset_metadata()
        all_annotations = annotation_reader.get_annotations()
        mappings: Dict[str, Dict[str, int]] = {}
        for field in self.target_fields:
            values = (
                [molecule_key(annotation) for annotation in all_annotations]
                if field == "molecule"
                else metadata_values(metadata, field)
            )
            mappings[field] = build_class_mapping(values, self.class_mappings.get(field))
        self._resolved_class_mappings = mappings
        return mappings

    def _sample(self, spectrum_id: int, spectrum: torch.Tensor) -> Tuple[Any, ...]:
        """Attach configured metadata and multi-label molecule targets."""
        if not self.target_fields:
            return spectrum_id, spectrum
        annotation_reader = self.active_context.annotation_reader
        mappings = self.get_class_mappings()
        metadata = annotation_reader.get_spectrum_metadata(spectrum_id)
        targets: Dict[str, torch.Tensor] = {}
        for field in self.target_fields:
            mapping = mappings[field]
            if field == "molecule":
                target = torch.zeros(len(mapping), dtype=torch.float32)
                for annotation in annotation_reader.get_spectrum_annotations(spectrum_id):
                    class_index = mapping.get(molecule_key(annotation))
                    if class_index is not None:
                        target[class_index] = 1.0
                targets[field] = target
                continue
            values = metadata_values(metadata, field)
            if len(values) != 1:
                raise_validation_error(
                    "PixelDataset",
                    f"Target '{field}' is ambiguous for spectrum {spectrum_id}.",
                )
            class_index = mapping.get(values[0])
            if class_index is None:
                raise_validation_error(
                    "PixelDataset",
                    f"Target '{field}' value '{values[0]}' has no class mapping.",
                )
            targets[field] = torch.tensor(class_index, dtype=torch.long)
        return spectrum_id, spectrum, targets

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Return one spectrum using the configured stable scale.

        :param values: Binned image intensities or latent components.
        :type values: numpy.ndarray
        :return: Float32 spectrum with the configured normalization.
        :rtype: numpy.ndarray
        """
        if self.normalization == "none":
            return values
        if self.normalization == "tic":
            denominator = float(np.sum(np.abs(values), dtype=np.float64))
        elif self.normalization == "max":
            denominator = float(np.max(np.abs(values), initial=0.0))
        else:
            denominator = float(np.linalg.norm(values))
        if denominator <= self.normalization_epsilon:
            return np.zeros_like(values)
        return values / denominator
