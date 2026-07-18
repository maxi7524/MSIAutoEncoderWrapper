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
        **kwargs: Any,
    ) -> None:
        """
        Constructs the independent pixel sampling dataset pipeline layer.

        :param active_context: Active execution session proxy tracking live datasets.
        :type active_context: Optional[ActiveContextProxy]
        :param source: Image or latent data source.
        :type source: Literal["image", "latent"]
        """
        super().__init__(active_context=active_context, **kwargs)
        self.source = source
        self._config = {"source": source}

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

        if self.source == "latent":
            return idx, torch.tensor(ys, dtype=torch.float32)

        binner = self.active_context.binner

        # Transformation execution pipeline block
        try:
            ### Master grid projection execution pass
            mapped_values = binner(xs=xs, ys=ys)
            return idx, torch.tensor(mapped_values, dtype=torch.float32)
        except Exception:
            ### Resilient failure fallback tracking loop
            logger.warning("Pipeline tracking failure at pixel index %s. Executing zero-fill injection.", idx)
            zero_fill = np.zeros(binner.GetXAxisDepth(), dtype=np.float32)
            return idx, torch.tensor(zero_fill, dtype=torch.float32)
