import numpy as np
from .base_inverse import MSIBaseInverseBinner
from ..binners_strategies.base_binner import MSIBaseBinner
from ..manager import BinnerManager


@BinnerManager.register_inverse_binner("TopPeaksInverseBinner")
class TopPeaksInverseBinner(MSIBaseInverseBinner):
    """
    Resolution reduction algorithm selecting dominant peaks using localized context windows.
    
    This inverse strategy scans input arrays to filter out background noise signals, tracking
    peak heights over non-overlapping windows up to an explicit cap threshold value.
    """

    def __init__(self, binner: MSIBaseBinner, max_bins: int = 500, window_size: int = 3) -> None:
        """
        Configures algorithmic threshold boundaries for inverse filtering execution loops.

        :param binner: Reference tracking target grid dimensions.
        :type binner: msi_lib.binners.binners_strategies.base_binner.MSIBaseBinner
        :param max_bins: Upper numerical ceiling limit determining maximum retained peak positions.
        :type max_bins: int
        :param window_size: Geometric radius determining neighborhood span masks around selected maxima.
        :type window_size: int
        """
        # Initialize ancestor dependencies
        super().__init__(binner)
        
        # Serialize parameter configuration mapping properties
        self._config = {"max_bins": max_bins, "window_size": window_size}
        
        # Internal configuration storage
        self._max_bins = max_bins
        self._window_size = window_size

    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs structural peak extraction loops over dense uniform intensity projections.

        :param grid_ys: Dense spatial coordinate channel intensity evaluations.
        :type grid_ys: np.ndarray
        :return: Parsed sparse matrix outputs consisting of matched (m/z, intensity) pairs.
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        # Optimization sorting block
        ## Sort input signal array indices in descending order based on total magnitude intensity
        sorted_indices = np.argsort(grid_ys)[::-1]
        ## Construct a boolean lookup mask to trace structural coverage states across indices
        keep_mask = np.zeros(len(grid_ys), dtype=bool)
        
        # State tracking initialization
        count = 0
        depth = len(grid_ys)

        # Main peak filtering selection engine
        for idx in sorted_indices:
            ### Intercept empty bounds or capacity overload events
            if grid_ys[idx] == 0 or count >= self._max_bins:
                #### Loop termination cascade
                # Exit evaluation routine immediately when encountering sub-threshold points
                break
            
            ### Context mask isolation sequence
            if not keep_mask[idx]:
                #### Boundary limitation calculation
                # Enforce safe clipping array boundaries preventing index overflow violations
                start = max(0, idx - self._window_size)
                end = min(depth, idx + self._window_size + 1)
                
                #### Evaluation step for tracking window mutations
                new_points = ~keep_mask[start:end]
                added_count = np.sum(new_points)
                
                #### Capacity validation threshold gate
                if count + added_count <= self._max_bins:
                    ##### Commit mask update
                    # Authorize permanent inclusion of targeted region window block segment
                    keep_mask[start:end] = True
                    count += added_count
                else:
                    ##### Hard threshold containment bypass
                    # Abort tracking loop to protect strict constraint ceilings from overflow
                    break

        # Coordinate matching phase
        ## Resolve master coordinate positioning arrays from linked forward binner object
        grid_xs = self._Binner.GetXAxis()
        
        # Sparse slice vector delivery
        ## Apply the boolean selection filter to both components simultaneously to yield aligned vectors
        return grid_xs[keep_mask], grid_ys[keep_mask]