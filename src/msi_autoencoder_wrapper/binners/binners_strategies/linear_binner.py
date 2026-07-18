import numpy as np
from scipy.stats import binned_statistic
from ..base_binner import MSIBaseBinner
from ..binners_manager import BinnerManager
from typing import Any, Optional
from ...utils.exceptions import raise_validation_error


@BinnerManager.register_binner("LinearBinning")
class LinearBinning(MSIBaseBinner):
    """
    Concrete processing strategy executing fast linear quantization via equidistant mass-to-charge bins.
    """

    def __init__(self, bin_step: float, x_min: Optional[float] = None, x_max: Optional[float] = None, active_context: Optional[Any] = None) -> None:
        """
        Initializes the linear binning generator, falling back to active_context for missing boundaries.
        """
        super().__init__(active_context=active_context)
        
        # Dynamic parameter resolution from context
        resolved_x_min = x_min
        resolved_x_max = x_max

        if active_context and getattr(active_context, "reader", None) is not None:
            if resolved_x_min is None:
                resolved_x_min = active_context.reader.GetXMin()
            if resolved_x_max is None:
                resolved_x_max = active_context.reader.GetXMax()

        if resolved_x_min is None or resolved_x_max is None:
            raise_validation_error(
                context_name="LinearBinning",
                message=(
                    "Explicit 'x_min' and 'x_max' values or an active context reader "
                    "session are required."
                ),
            )

        self.x_min = float(resolved_x_min)
        self.x_max = float(resolved_x_max)
        self.bin_step = float(bin_step)
        
        self._config = {"x_min": self.x_min, "x_max": self.x_max, "bin_step": self.bin_step}
        
        # Build strict boundary coordinates configuration matrices
        self.bin_edges = np.arange(self.x_min, self.x_max + self.bin_step, self.bin_step, dtype=np.float64)
        self.grid = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if xs.size == 0 or ys.size == 0:
            return np.zeros(self.GetXAxisDepth(), dtype=np.float32)
            
        intensities, _, _ = binned_statistic(
            xs, ys, statistic="sum", bins=self.bin_edges
        )
        return intensities

    def GetXMin(self) -> float:
        return self.x_min

    def GetXMax(self) -> float:
        return self.x_max

    def GetXAxis(self) -> np.ndarray:
        return self.grid

    def GetXAxisDepth(self) -> int:
        return len(self.grid)
