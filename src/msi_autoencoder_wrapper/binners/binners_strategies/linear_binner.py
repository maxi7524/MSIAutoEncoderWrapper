import numpy as np
from scipy.stats import binned_statistic
from ..base_binner import MSIBaseBinner
from ..binners_manager import BinnerManager


@BinnerManager.register_binner("LinearBinning")
class LinearBinning(MSIBaseBinner):
    """
    Concrete processing strategy executing fast linear quantization via equidistant mass-to-charge bins.
    """

    def __init__(self, x_min: float, x_max: float, bin_step: float) -> None:
        super().__init__()
        self._config = {"x_min": x_min, "x_max": x_max, "bin_step": bin_step}
        
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.bin_step = float(bin_step)
        
        # Build strict boundary coordinates configuration matrices
        self.bin_edges = np.arange(self.x_min, self.x_max + self.bin_step, self.bin_step, dtype=np.float64)
        self.grid = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if xs.size == 0 or ys.size == 0:
            return np.zeros(self.GetXAxisDepth(), dtype=np.float32)
            
        intensities, _, _ = binned_statistic(
            xs, ys, statistic="sum", bins=self.bin_edges
        )
        return intensities.astype(np.float32)

    def GetXMin(self) -> float:
        return self.x_min

    def GetXMax(self) -> float:
        return self.x_max

    def GetXAxis(self) -> np.ndarray:
        return self.grid

    def GetXAxisDepth(self) -> int:
        return len(self.grid)