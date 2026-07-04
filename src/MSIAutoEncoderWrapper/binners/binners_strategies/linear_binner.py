import numpy as np
from scipy.stats import binned_statistic
from .base_binner import MSIBaseBinner
from ..manager import BinnerManager


@BinnerManager.register_binner("LinearBinning")
class LinearBinning(MSIBaseBinner):
    """
    Concrete processing strategy executing fast linear interpolation via equidistant bins.
    
    This strategy relies on optimized scipy histogram algorithms to accumulate mass spectra
    intensities onto a predictable and uniform input vector sequence.
    """

    def __init__(self, x_min: float, x_max: float, bin_step: float) -> None:
        """
        Constructs a structural linear grid over explicit geometric boundaries.

        :param x_min: Boundary defining the minimum lower range limit for spectrum inclusion.
        :type x_min: float
        :param x_max: Boundary defining the maximum upper range limit for spectrum inclusion.
        :type x_max: float
        :param bin_step: Constant coordinate interval delta value separating discrete bin units.
        :type bin_step: float
        """
        # Base initializer call
        super().__init__()
        
        # Configuration serialization routing
        ## Populate local configuration storage with parameter definitions for reproduction steps
        self._config = {"x_min": x_min, "x_max": x_max, "bin_step": bin_step}
        
        # Attribute state persistence
        self.x_min = x_min
        self.x_max = x_max
        self.bin_step = bin_step
        
        # Coordinate grid compilation pipeline
        ## Generate regular sampling array matrix over defined boundary span
        self.grid = np.arange(x_min, x_max + bin_step, bin_step)
        ## Calculate shift values for bin edge optimization
        ### Edges array must equal length of grid array plus one element increment boundary
        self.bin_edges = np.append(self.grid - bin_step / 2, self.grid[-1] + bin_step / 2)

    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Projects arbitrary coordinate points onto the structured internal linear grid boundaries.

        :param xs: Raw irregular experimental m/z vectors.
        :type xs: np.ndarray
        :param ys: Empirical signal magnitude vectors.
        :type ys: np.ndarray
        :return: Fixed-length spectrum representation containing cumulative interval intensities.
        :rtype: np.ndarray
        """
        # Boundary conditional optimization block
        ## Evaluate input data length to intercept and safely handle null matrices
        if len(xs) == 0:
            #### Zero tensor deployment
            # Prevent failures from unacquired signals by matching expected feature length
            return np.zeros(self.GetXAxisDepth(), dtype=np.float32)
            
        # Mathematical reduction calculation
        ## Execute accelerated histogram integration leveraging underlying native scipy functions
        intensities, _, _ = binned_statistic(
            xs, ys, statistic="sum", bins=self.bin_edges
        )
        
        # Format conversion enforcement
        ## Explicitly cast precision to 32-bit floating points to prevent PyTorch type mismatches
        return intensities.astype(np.float32)

    def GetXMin(self) -> float:
        """
        Retrieves structural coordinate floor parameter value.

        :return: Absolute lower boundary coordinate.
        :rtype: float
        """
        return self.x_min

    def GetXMax(self) -> float:
        """
        Retrieves structural coordinate ceiling parameter value.

        :return: Absolute upper boundary coordinate.
        :rtype: float
        """
        return self.x_max

    def GetXAxis(self) -> np.ndarray:
        """
        Retrieves reference array vector containing synthesized central bin positions.

        :return: Grid coordinate array matrix.
        :rtype: np.ndarray
        """
        return self.grid

    def GetXAxisDepth(self) -> int:
        """
        Computes total array length for configuration verification steps.

        :return: Quantitative size measurement of the compiled coordinate grid array.
        :rtype: int
        """
        return len(self.grid)