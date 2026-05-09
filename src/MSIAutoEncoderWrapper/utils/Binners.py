
# python
from abc import ABC, abstractmethod

# computational
import numpy as np

# LinearBinning
from scipy.stats import binned_statistic


# ----------------------------------------
# Binning
# ----------------------------------------

class MSIPyTorchBinner(ABC):
    '''
    Class which implements binning for `MSIPyTorchDataset`. 

    Subclasses need to have implemented:
    - `__call__` that maps (xs, ys) to defined grid and returns ys representation. 
    - `get_config`
    propozycja
    - `GetXMin`
    - `GetXMax`
    - `GetXAxis` - returns grid m/z axis (same name )
    - `GetXAxisDepth` - returns length of grid 
    - `GetSpacing` - returns spacing of grid (if it is regular) 
    '''

    def __init__(self):
        self._config = {}


    # ---------------------
    # main functionality
    # ---------------------
    
    @abstractmethod
    def __call__(self, xs: np.array, ys: np.array):
        '''Maps x-axis (xs) and y-axis (ys) to grid-x-axis.

        :param xs: x-axis (m/z index) array
        :param ys: y-axis (intensities) array

        :return: np.arrays (grid-x-axis, Map_function(y-axis))
        '''
        pass

    # ---------------------
    # model configuration functionalities
    # ---------------------

    def GetConfig(self) -> dict:
        '''Return dictionary with initial parameters for `MSIPyTorchBinner` object reconstruction.'''
        return self._config

    # ---------------------
    # getters & setters
    # ---------------------

    @abstractmethod
    def GetXAxis(self):
        '''Get the grid-x-axis values (i.e. m/z values on the grid-x axis).

        :return: np.array
        '''
        pass

    @abstractmethod
    def GetXAxisDepth(self):
        '''Get the size of the grid-x axis. For processed imzML files, 
        this value is the number of bins used to represent the grid-x-axis.

        :return: Number of grid-x values.'''
        pass

    @abstractmethod
    def GetXMin(self):
        '''Get the minimum grid-x-axis value
        
        :return Minimum grid-x value
        '''
        pass

    @abstractmethod
    def GetXMax(self):
        '''Get the maximum grid-x axis value
        
        :return Maximum grid-x value
        '''
        pass



# --------------------
# External 
# --------------------


class M2aiaBinning(MSIPyTorchBinner):
    def __init__(self):
        '''
        Use binning from m2aia library by using `GetXAxis` method
        '''
        #TODO 

    pass

# --------------------
# Functional transformations
# --------------------

class LinearBinning(MSIPyTorchBinner):
    def __init__(
            self, 
            xs_min, 
            xs_max, 
            xs_res, 
            xs_bin_window = None, 
            ys_agg_method='mean'
        ):
        '''TODO: NAPISAĆ DOKUMENTACJE

        The method creates bins centered at each grid point with a width equal to xs_bin_window (default: half of resolution). 
        
        '''
        # initialization of grid-xs:
        ## Grid arrange {x in [min_mz, max_mz] | x = min_mz + i * mz_resolution, i \in N}
        self._grid_xs = np.arange(
            xs_min,
            xs_max + xs_res,    # we include xs_max value
            xs_res
        )
        # __call__ params:
        ## bin window: [- bin_win/2 + bin, bin + bin_win/2)
        if xs_bin_window is None:
            ### be default take separated sets 
            self._xs_bin_window = xs_res 
        else:
            self._xs_bin_window = xs_bin_window
        ## aggregation method
        self._ys_agg_method = ys_agg_method

        # saving initialization params for class reconstruction
        self._config = {
            "xs_min": xs_min,
            "xs_max": xs_max,
            "xs_res": xs_res,
            "xs_bin_window": xs_bin_window,
            "ys_agg_method": ys_agg_method
        }

    # --- functionality ---

    def __call__(self, xs: np.array, ys: np.array):
        
        # local variables
        bin_win = self._xs_bin_window

        ## define bins (from, to) (for scipy method)
        bins = np.concatenate([
            self._grid_xs - bin_win, # [mz_min - grid_dist, ...., mz_max - grid_dit]
            # adding last part,
            [self._grid_xs[-1] + bin_win]
        ])

        ## resampling method
        ### (binned values , bin info,  )
        rtn_val, _, _ = binned_statistic(
            xs, 
            ys, 
            statistic=self._ys_agg_method, 
            bins=bins
        )

        ## change nan to zeros
        return np.nan_to_num(rtn_val, nan=0.0)

    # --- getters ---

    def GetXAxis(self):
        return self._grid_xs
    
    def GetXAxisDepth(self):
        return len(self._grid_xs)
    
    def GetXMin(self):
        return self._grid_xs[0]
    
    def GetXMax(self):
        return self._grid_xs[-1]


class LogarithmicBinning(MSIPyTorchBinner):
    '''No tutaj by była klas a któa ma logarytmiczne biny (zwiększające się z osią)'''
    pass




# ----------------------------------------
# Binning inverters
# ----------------------------------------


class MSIPyTorchInverseBinner(ABC):
    '''
    
    '''

    def __init__(self):
        self._config = {}

    @abstractmethod
    def __call__(self, grid_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ''' Maps grid-ys to (xs, ys)'''
        pass



    # ---------------------
    # model configuration functionalities
    # ---------------------

    def GetConfig(self) -> dict:
        '''Return dictionary with initial parameters for `MSIPyTorchInverseBinner` object reconstruction.'''
        return self._config

    # --- getters ---

   

# --------------------
# Functional transformations
# --------------------

class NotEmptyInverseBinner(MSIPyTorchInverseBinner):
    ''' 
    REMARK: numerically not stable
    #TODO 
    - threshold value should be given 
    '''
    def __init__(self, Binner: MSIPyTorchBinner):
        self._Binner = Binner 

        self._config = {
            'Binner': Binner.GetConfig(),

        }

    def __call__(self, grid_ys):
        

        # mask for positive values
        mask = grid_ys > 0

        # obtain xs axis
        grid_xs = self.Binner.GetXAxis()

        return grid_xs[mask], grid_ys[mask]
    

    @property
    def Binner(self):
        return self._Binner
    

class TopPeaksInverseBinner(MSIPyTorchInverseBinner):
    def __init__(self, Binner: MSIPyTorchBinner, max_bins: int, window_size: int = 5, threshold: float = 1e-2):
        """
        Filters grid intensities by selecting top peaks and their neighborhood.
        
        Args:
            Binner: The forward binner object to access the m/z axis.
            max_bins: Maximum number of bins (points) to keep per pixel.
            window_size: How many points to the left and right of a peak to include.
            threshold: Absolute intensity below which signal is ignored.
        """
        #TODO - to validate (promt) 
        self._Binner = Binner
        self._max_bins = max_bins
        self._window_size = window_size
        self._threshold = threshold
        
        self._config = {
            'Binner': Binner.GetConfig(),
            'max_bins': max_bins,
            'window_size': window_size,
            'threshold': threshold
        }

    def __call__(self, grid_ys: np.ndarray):
        # Flatten and thresholding
        grid_ys = np.asarray(grid_ys).flatten()
        grid_ys[grid_ys < self._threshold] = 0
        
        if np.sum(grid_ys) == 0:
            return np.array([]), np.array([])

        # Get indices of intensities sorted descending
        sorted_indices = np.argsort(grid_ys)[::-1]
        
        # Tracking which bins to keep
        keep_mask = np.zeros_like(grid_ys, dtype=bool)
        count = 0
        depth = len(grid_ys)

        for idx in sorted_indices:
            if grid_ys[idx] == 0 or count >= self._max_bins:
                break
            
            if not keep_mask[idx]:
                # Define window around the peak
                start = max(0, idx - self._window_size)
                end = min(depth, idx + self._window_size + 1)
                
                # Count only new points added to the mask
                new_points = ~keep_mask[start:end]
                added_count = np.sum(new_points)
                
                if count + added_count <= self._max_bins:
                    keep_mask[start:end] = True
                    count += added_count
                else:
                    # If the whole window doesn't fit, just stop or add partially
                    break

        grid_xs = self._Binner.GetXAxis()
        return grid_xs[keep_mask], grid_ys[keep_mask]


# ----------------------------------------
# Helpers
# ----------------------------------------

# 


MSIPyTorchBinnerRegistry = {
    # MSIPyTorchInverseBinner
    "NotEmptyInverseBinner": NotEmptyInverseBinner,
    "TopPeaksInverseBinner": TopPeaksInverseBinner
}

MSIPyTorchInverseBinnerRegistry = {
    # MSIPyTorchBinner
    # "M2aiaBinning": M2aiaBinning, TODO
    "LinearBinning": LinearBinning,
    # "LogarithmicBinning": LogarithmicBinning,TODO
}

BINNER_REGISTRY = {
    **MSIPyTorchBinnerRegistry, 
    **MSIPyTorchInverseBinnerRegistry
}