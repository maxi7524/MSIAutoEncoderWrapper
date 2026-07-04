from typing import Tuple, Any
import torch
import numpy as np

# Purely relative imports linking back to interface and registry contexts
from ..manager import DatasetManager
from .base_dataset import MSIBaseDataset
from ....loader.strategies.base_loader import MSIBaseLoader
from ....binners.binners_strategies.base_binner import MSIBaseBinner


@DatasetManager.register_dataset("PixelDataset")
class PixelDataset(MSIBaseDataset):
    """
    Concrete dataset processing strategy executing pixel-level mass spectra extraction.

    This component implements standard single-pixel mapping loops. It loads individual
    irregular spectra from the underlying storage layer and casts them on-the-fly into
    homogeneous intensity vectors aligned directly to the verified master grid-x-axis.
    """

    def __init__(self, loader: MSIBaseLoader, binner: MSIBaseBinner) -> None:
        """
        Constructs the independent pixel sampling dataset pipeline layer.

        :param loader: Active file reader subclass handle managing native binary matrix tracking.
        :type loader: MSIBaseLoader
        :param binner: Active processing manager specifying target grid-x-axis ticks boundaries.
        :type binner: MSIBaseBinner
        """
        # Execute ancestor structural binding setup sequence
        super().__init__(loader=loader, binner=binner)

    def __len__(self) -> int:
        """
        Retrieves total pixel coordinates count limits exposed by the underlying storage loader.

        :return: Absolute density size of the total spatial image layout.
        :rtype: int
        """
        return len(self._loader)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        """
        Extracts and resolves a singular experimental spectrum onto the active target grid-x-axis.

        :param idx: Flat position tracking coordinate index targeting an explicit single tissue pixel.
        :type idx: int
        :return: Aligned tuple holding the unique flat spatial key index token and its 32-bit float intensity tensor.
        :rtype: tuple(int, torch.Tensor)
        """
        # Read low-level experimental variables array entries from data drivers
        xs, ys = self._loader.get_raw_spectrum(idx)

        # Transformation execution pipeline block
        try:
            ### Master grid projection execution pass
            # Accumulate raw signals into fixed discrete mass-to-charge (m/z) bins channels frame
            mapped_values = self._binner(xs=xs, ys=ys)
            
            return idx, torch.tensor(mapped_values, dtype=torch.float32)
        except Exception:
            ### Resilient failure fallback tracking loop
            # Provide zero-filled structures to insulate pipelines against incomplete storage buffers
            zero_fill = np.zeros(self.GetGridXAxisDepth(), dtype=np.float32)
            return idx, torch.tensor(zero_fill, dtype=torch.float32)