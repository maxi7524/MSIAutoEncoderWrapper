import numpy as np
from .base_inverse import MSIBaseInverseBinner
from ..binners_strategies.base_binner import MSIBaseBinner
from ..manager import BinningManager


@BinningManager.register_inverse_binner("TopPeaksInverseBinner")
class TopPeaksInverseBinner(MSIBaseInverseBinner):
    """
    Resolution reduction algorithm tracking top peak heights across non-overlapping contextual coordinate masks.
    """

    def __init__(self, binner: MSIBaseBinner, max_bins: int = 500, window_size: int = 3) -> None:
        super().__init__(binner)
        self._config = {"max_bins": max_bins, "window_size": window_size}
        self._max_bins = int(max_bins)
        self._window_size = int(window_size)

    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth = len(grid_ys)
        keep_mask = np.zeros(depth, dtype=bool)
        
        # Fast sorting optimization pulling structural max indices targets
        sorted_indices = np.argsort(grid_ys)[::-1]
        count = 0
        
        for idx in sorted_indices:
            if count >= self._max_bins:
                break
                
            if not keep_mask[idx]:
                start = max(0, idx - self._window_size)
                end = min(depth, idx + self._window_size + 1)
                
                added_points = ~keep_mask[start:end]
                added_count = np.sum(added_points)
                
                if count + added_count <= self._max_bins:
                    keep_mask[start:end] = True
                    count += added_count
                else:
                    break
                    
        grid_xs = self._Binner.GetXAxis()
        return grid_xs[keep_mask], grid_ys[keep_mask]