"""
ims_contrastive_model/dataloader.py
----------------------------------
Contains code for dataloader for m2aia ims_contrastive_model
"""

# base
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
# IMS
import m2aia as m2
# resampling method
from scipy.stats import binned_statistic


# TODO - we could use certain *mask* (so we do not batch noise pixel ~ match faster learning)


class IMSPyTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for M2AIA ImzMLReader.
    
    This class handles the transformation of raw Mass Spectrometry Imaging (MSI) 
    spectra into a fixed-size tensor format suitable for Deep Learning. It performs 
    on-the-fly resampling / binning of irregular m/z axes to a common reference grid.

    Key Features:
    - Extracts tissue pixels (valid spectra) for training.
    - Standardizes m/z axes across the entire dataset.
    - Handles missing data or corrupted spectra gracefully.
    
    Note:
        It is assumed that the input m2aia_img is already preprocessed (TIC 
        normalization, noise reduction) before being passed to this dataset.
    """
    def __init__(self, 
                # obligatory 
                m2aia_img: m2.ImzMLReader,
                data_path: str | Path, # give path to .imzML file
                # optional
                resampling_method = 'mean', 
                mz_min: float = None, 
                mz_max: float = None, 
                mz_resolution: float = None,
            ):
        """
        Initializes the IMS dataset with a fixed m/z grid.

        Args:
            m2aia_img (m2.ImzMLReader): An opened and loaded M2AIA reader object.
            resampling_method (str): Statistical method for binning ('mean', 'max', 'sum').
                Defaults to 'mean'.
            mz_min (float, optional): Lower bound of the m/z grid. 
                If None, extracted from metadata.
            mz_max (float, optional): Upper bound of the m/z grid. 
                If None, extracted from metadata.
            mz_resolution (float, optional): The step size between m/z bins. 
                If None, extracted from 'pixel size z' metadata.

        Warning:
            The 'mz_resolution' is critical for memory and performance. Too small 
            resolution results in very large tensors, potentially causing OOM errors.
        """
        ## image
        self._img = m2aia_img
        self.data_path = Path(data_path)

        ## grid stats
        # WARNING: here i use hardcoded values from GetMetaData dict 
        ### min
        self._mz_min = mz_min if mz_min is not None else float(m2aia_img.GetMetaData()['m2aia.xs.min'])
        ### max
        self._mz_max = mz_max if mz_max is not None else float(m2aia_img.GetMetaData()['m2aia.xs.max'])
        ### reslution 
        self._mz_resolution = mz_resolution if mz_resolution is not None else float(m2aia_img.GetMetaData()['pixel size z'])
        
        ## resampling 
        ### 
        self._resampling_method = resampling_method
        ### Grid arrange {x in [min_mz, max_mz] | x = min_mz + i * mz_resolution, i \in N}
        self._grid = np.arange(
            self.mz_min, 
            self.mz_max + self._mz_resolution, # to consider max_mz
            self._mz_resolution
        )

    
    # ---------------------
    # dataset essentials 
    # ---------------------

    def __len__(self):
        """Returns the total number of spectra available in the m2aia object."""
        return self.img.GetNumberOfSpectra()

    def __getitem__(self, idx):
        """
        Retrieves, resamples, and converts a spectrum to a PyTorch tensor.

        Args:
            idx (int): Index of the spectrum to retrieve.

        Returns:
            torch.Tensor: A 1D tensor of shape (len(grid),) containing 
                resampled intensities as float32.
        
        Note:
            If a spectrum cannot be loaded or is empty, a zero-filled tensor 
            of the correct shape is returned to maintain batch consistency.
        """
        # load data
        try: 
            mzs, intensities = self.img.GetSpectrum(idx)
            ## map to common axis
            mapped_val = self._resample(mzs=mzs, intensities=intensities)
            return torch.tensor(mapped_val, dtype=torch.float32)
        except:
            ## if GetSpectrum(idx) is None
            return torch.zeros(len(self.grid), dtype=torch.float32)
    
    # ---------------------
    # helpers
    # ---------------------
    def _resample(self, mzs, intensities):
        """
        Projects raw m/z intensities onto the fixed self.grid.

        The method creates bins centered at each grid point with a width 
        equal to self.mz_resolution. 

        Args:
            mzs (np.ndarray): Raw m/z values from m2aia.
            intensities (np.ndarray): Raw intensity values from m2aia.

        Returns:
            np.ndarray: Binned intensities corresponding to self.grid.
        """
        # resampling
        ## active width (how far from grid point)
        # --------------
        # START TODO - decide range / use in header function 
        grid_dis = self.mz_resolution / 2 
        # END TOD 
        # --------------

        ## define bins (from, to) (for scipy method)
        bins = np.concatenate([
            self.grid - grid_dis, # [mz_min - grid_dist, ...., mz_max - grid_dit]
            # adding last part,
            [self.grid[-1] + grid_dis]
        ])

        ## resampling method
        ### (binned values , bin info,  )
        rtn_val, _, _ = binned_statistic(
            mzs, 
            intensities, 
            statistic=self.resampling_method, 
            bins=bins
        )

        ## change nan to zeros
        return np.nan_to_num(rtn_val, nan=0.0)

    # ---------------------
    # getters and setters
    # ---------------------
    
    @property
    def img(self):
        return self._img

    @property
    def mz_min(self):
        return self._mz_min

    @property
    def mz_max(self):
        return self._mz_max
    
    @property
    def mz_resolution(self):
        return self._mz_resolution

    @property
    def resampling_method(self):
        return self._resampling_method
    
    @property
    def grid(self):
        return self._grid








        

    
    
