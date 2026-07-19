"""
Concrete dataset strategy executing single-pixel spectra mapping sequences driven by an active context.
"""

from typing import Tuple, Any, Optional, Literal
import torch
import numpy as np

from ..dataset_manager import DatasetManager
from ..base_dataset import MSIBaseDataset
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error
from ....core.mixins.active_context.active_context_mixin import ActiveContextProxy

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
        self._config = {
            "source": source,
            "normalization": resolved_normalization,
            "normalization_epsilon": self.normalization_epsilon,
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

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
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
            return idx, torch.from_numpy(normalized_values)

        binner = self.active_context.binner

        # Transformation execution pipeline block
        ## Map and normalize one spectrum without hiding invalid reader output
        mapped_values = np.asarray(binner(xs=xs, ys=ys), dtype=np.float32)
        if not np.all(np.isfinite(mapped_values)):
            raise_validation_error(
                context_name="PixelDataset",
                message=f"Binned spectrum {idx} contains non-finite values.",
            )
        return idx, torch.from_numpy(self._normalize(mapped_values))

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
