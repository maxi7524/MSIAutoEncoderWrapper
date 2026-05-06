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
# binning module
from .utils.Binners import IMSPyTorchBinner


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
                Binner: IMSPyTorchBinner
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
        self._Binner = Binner
        # self.data_path = Path(data_path)
    
    # ---------------------
    # dataset essentials 
    # ---------------------

    def __len__(self):
        """Returns the total number of spectra available in the m2aia object."""
        return self.img.GetNumberOfSpectra()

    def __getitem__(self, spatial_idx):
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
            ## Sample is single spectrum 
            xs, ys = self.img.GetSpectrum(spatial_idx)
            ## map to common axis
            mapped_val = self.Binner(xs=xs, ys=ys)
            return spatial_idx, torch.tensor(mapped_val, dtype=torch.float32)
        except:
            ## if GetSpectrum(idx) is None
            return spatial_idx, torch.zeros(len(self.GetGridXAxis), dtype=torch.float32)
    
    # ---------------------
    # helpers
    # ---------------------


    # ---------------------
    # getters and setters (functions for m2aia convenience)
    # ---------------------
    
    @property
    def img(self):
        return self._img

    @property
    def Binner(self):
        return self._Binner

    # --- image part ---

    def GetXMin(self):
        return self.img.GetXAxis[0]
    
    def GetXMax(self):
        return self.img.GetXAxis[-1]
    
    def GetXAxis(self):
        return self.img.GetXAxis()
    
    def GetXAxisDepth(self):
        return self.img.GetXAxisDepth()

    # --- binner part ---

    def GetGridXMin(self):
        return self.Binner.GetXMin()

    def GetGridXMax(self):
        return self.Binner.GetXMax()
    
    def GetGridXAxis(self):
        return self.Binner.GetXAxis()
    
    def GetGridXAxisDepth(self):
        return self.Binner.GetXAxisDepth()
    
    # @property
    # def mz_resolution(self):
    #     return self._mz_resolution

    # @property
    #TODO - to powinno zwracać nazwę tej używanej metody 
    # def resampling_method(self):
    #     return self._resampling_method
    
    


    





        

    
    
